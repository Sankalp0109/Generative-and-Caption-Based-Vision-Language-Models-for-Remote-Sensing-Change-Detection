import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

# Mock PIL Image first to avoid any actual disk operations during dataset loading
from PIL import Image
def get_dummy_image(*args, **kwargs):
    return Image.new("RGB", (256, 256))

# Apply mock to PIL.Image.open before importing other modules
patcher = patch("PIL.Image.open", side_effect=get_dummy_image)
patcher.start()

import torch
import torch.nn as nn

# Setup project path so imports from src work
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import configs and models
from src.config import DataConfig, ModelConfig, RemoteCLIPConfig, TrainConfig
from src.dataset import (
    build_remoteclip_transforms,
    build_vocabulary_from_multiple_annotations,
    get_levircc_loaders,
    load_levircc_annotations,
    split_samples_by_split,
)
from src.models.final_model import (
    RemoteCLIPCrossAttentionModel,
    phase5_total_loss,
)
from src.models.phase6 import (
    Phase6Config,
    TileBasedChangeCaptioningModel,
    phase6_total_loss,
)
from src.training import (
    build_criterion,
    build_optimizer_and_scheduler,
    save_checkpoint,
)
from src.utils import get_device, set_seed

class FakeRemoteCLIP(nn.Module):
    """A fake RemoteCLIP backbone to bypass downloading the heavy weights checkpoint."""
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.projection = nn.Linear(3, output_dim, bias=False)
        self.embed_dim = output_dim
        class Visual:
            def __init__(self, output_dim):
                self.output_dim = output_dim
        self.visual = Visual(output_dim)

    def encode_image(self, images):
        # Pool the spatial dimensions (H, W) to (1, 1), mapping (B, 3, H, W) -> (B, output_dim)
        pooled = images.mean(dim=(-2, -1))
        return self.projection(pooled)

def run_dry_run():
    print("======================================================================")
    print("Starting Dry-Run Pipeline Compatibility Check...")
    print("======================================================================")
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Create a temporary directory and dummy annotation file
    temp_dir = Path("temp_dryrun_data")
    temp_dir.mkdir(exist_ok=True)
    dummy_json_path = temp_dir / "dummy_annotations.json"
    
    dummy_annotations = {
        "images": [
            {
                "filename": "sample1.png",
                "filepath": "train",
                "split": "train",
                "changeflag": 1,
                "sentences": [{"raw": "A building was constructed"}, {"raw": "New construction is visible"}],
            },
            {
                "filename": "sample2.png",
                "filepath": "val",
                "split": "val",
                "changeflag": 0,
                "sentences": [{"raw": "No change is visible"}, {"raw": "The area remains unchanged"}],
            },
            {
                "filename": "sample3.png",
                "filepath": "test",
                "split": "test",
                "changeflag": 1,
                "sentences": [{"raw": "Road was built"}, {"raw": "A new road is constructed"}],
            }
        ]
    }
    
    with open(dummy_json_path, "w") as f:
        json.dump(dummy_annotations, f)

    # Initialize configurations
    data_cfg = DataConfig(batch_size=2, val_batch_size=2)
    model_cfg = ModelConfig()
    remoteclip_cfg = RemoteCLIPConfig()
    train_cfg = TrainConfig()
    phase6_cfg = Phase6Config()

    print("Building vocabulary from dummy annotations...")
    shared_vocab = build_vocabulary_from_multiple_annotations(
        [dummy_annotations],
        min_freq=1,  # Keep all words in this tiny vocabulary
    )
    print(f"Vocabulary size: {len(shared_vocab.word2idx)}")

    print("Building RemoteCLIP transforms...")
    remoteclip_transform = build_remoteclip_transforms(
        img_size=(256, 256),
        mean=remoteclip_cfg.image_mean,
        std=remoteclip_cfg.image_std,
    )

    print("Loading DataLoaders (using mock PIL.Image.open)...")
    train_loader, val_loader, test_loader, vocab = get_levircc_loaders(
        caption_json=dummy_json_path,
        image_root=temp_dir,
        vocab=shared_vocab,
        batch_size=2,
        val_batch_size=2,
        device=str(device),
        min_word_freq=1,
        caption_index=0,
        num_workers=0,
        vocab_path=None,
        transforms_fn=remoteclip_transform,
    )
    print(f"DataLoaders loaded. Train batches: {len(train_loader)}")

    print("\n-------------------------------------------------------------")
    print("Test 1: Phase Final (RemoteCLIP Cross-Attention Model) Check")
    print("-------------------------------------------------------------")
    fake_backbone = FakeRemoteCLIP(output_dim=512)
    model_p5 = RemoteCLIPCrossAttentionModel(
        vocab_size=len(vocab.word2idx),
        encoder_dim=512,
        embed_dim=128,
        num_heads=4,
        num_decoder_layers=1,
        max_caption_len=15,
        dropout=0.1,
        pad_idx=vocab.pad_idx,
        backbone=fake_backbone,
        backbone_dim=512,
        contrastive_dim=128,
    ).to(device)

    trainable_p5 = sum(p.numel() for p in model_p5.parameters() if p.requires_grad)
    print(f"Model initialized successfully. Trainable params: {trainable_p5:,}")

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["images"].to(device)
    caption_tokens = batch["caption_tokens"].to(device)
    input_tokens = caption_tokens[:, :-1]
    target_tokens = caption_tokens[:, 1:]

    print("Running forward pass...")
    outputs = model_p5(images, input_tokens, return_aux=True)
    print("Forward pass successful. Outputs keys:", list(outputs.keys()))
    print("Logits shape:", outputs["logits"].shape)
    
    print("Computing loss...")
    criterion = build_criterion(vocab)
    total_loss, cap_loss, cont_loss = phase5_total_loss(
        logits=outputs["logits"],
        target_tokens=target_tokens,
        criterion=criterion,
        image_embeddings=outputs["image_embeddings"],
        text_embeddings=outputs["text_embeddings"],
        contrastive_weight=0.1,
        temperature=0.07,
        pad_idx=vocab.pad_idx,
    )
    print(f"Loss computed successfully: Total={total_loss.item():.4f}, Caption={cap_loss.item():.4f}, Contrastive={cont_loss.item():.4f}")

    print("Running backward pass...")
    optimizer, scheduler = build_optimizer_and_scheduler(model_p5, train_cfg)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print("Backward pass and optimizer step completed successfully!")

    print("\n-------------------------------------------------------------")
    print("Test 2: Phase 6 (Hierarchical Tile-Based Model) Check")
    print("-------------------------------------------------------------")
    fake_backbone_p6 = FakeRemoteCLIP(output_dim=512)
    # Adjust Config parameters to fit our lightweight testing
    phase6_cfg.remoteclip_model_name = "ViT-B-32"
    phase6_cfg.fusion_dim = 128
    phase6_cfg.global_dim = 128
    phase6_cfg.embed_dim = 128
    phase6_cfg.contrastive_dim = 128
    phase6_cfg.num_fusion_layers = 1
    phase6_cfg.num_decoder_layers = 1

    model_p6 = TileBasedChangeCaptioningModel(
        vocab_size=len(vocab.word2idx),
        config=phase6_cfg,
        pad_idx=vocab.pad_idx,
        backbone=fake_backbone_p6,
    ).to(device)

    trainable_p6 = sum(p.numel() for p in model_p6.parameters() if p.requires_grad)
    print(f"Model initialized successfully. Trainable params: {trainable_p6:,}")

    print("Running forward pass...")
    outputs_p6 = model_p6(images, input_tokens, return_aux=True)
    print("Forward pass successful. Outputs keys:", list(outputs_p6.keys()))
    print("Logits shape:", outputs_p6["logits"].shape)

    print("Computing loss...")
    total_loss_p6, cap_loss_p6, cont_loss_p6 = phase6_total_loss(
        logits=outputs_p6["logits"],
        target_tokens=target_tokens,
        criterion=criterion,
        image_embeddings=outputs_p6["image_embeddings"],
        text_embeddings=outputs_p6["text_embeddings"],
        contrastive_weight=phase6_cfg.contrastive_weight,
        temperature=phase6_cfg.temperature,
        pad_idx=vocab.pad_idx,
    )
    print(f"Loss computed successfully: Total={total_loss_p6.item():.4f}, Caption={cap_loss_p6.item():.4f}, Contrastive={cont_loss_p6.item():.4f}")

    print("Running backward pass...")
    optimizer_p6, scheduler_p6 = build_optimizer_and_scheduler(model_p6, train_cfg)
    optimizer_p6.zero_grad()
    total_loss_p6.backward()
    optimizer_p6.step()
    print("Backward pass and optimizer step completed successfully!")

    print("Testing save_checkpoint functionality...")
    saved_path = save_checkpoint(
        model=model_p6,
        optimizer=optimizer_p6,
        epoch=1,
        loss=total_loss_p6.item(),
        vocab=vocab,
        checkpoint_dir=Path(temp_dir),
        filename="dry_run_test.pt",
    )
    print(f"save_checkpoint test successful! File created at: {saved_path}")

    # Cleanup temporary directories and files
    patcher.stop()
    shutil.rmtree(temp_dir)
    print("\n======================================================================")
    print("Dry-run pipeline test successful! No compatibility issues found.")
    print("======================================================================")

if __name__ == "__main__":
    run_dry_run()
