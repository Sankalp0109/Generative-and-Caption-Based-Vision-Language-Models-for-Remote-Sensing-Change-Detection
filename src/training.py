"""Shared training, validation, checkpoint, and inference utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


def prepare_teacher_forcing_inputs(caption_tokens, pad_idx: int = 0) -> Tuple:
    """
    Split padded caption tokens into decoder input and target.

    Example:
        caption_tokens = [<START>, w1, w2, <END>, <PAD>]
        input_tokens   = [<START>, w1, w2, <END>]
        target_tokens  = [w1, w2, <END>, <PAD>]
    """
    input_tokens = caption_tokens[:, :-1]
    target_tokens = caption_tokens[:, 1:]
    return input_tokens, target_tokens


def captioning_loss(logits, target_tokens, criterion, pad_idx: int = 0):
    """Compute token-level cross entropy for caption generation."""
    batch_size, seq_len, vocab_size = logits.shape
    loss = criterion(
        logits.reshape(batch_size * seq_len, vocab_size),
        target_tokens.reshape(batch_size * seq_len),
    )
    return loss


def caption_token_accuracy(logits, target_tokens, pad_idx: int = 0) -> float:
    """Compute token accuracy while ignoring padded target positions."""
    predictions = logits.argmax(dim=-1)
    valid_mask = target_tokens != pad_idx
    total_tokens = valid_mask.sum().item()
    if total_tokens == 0:
        return 0.0

    correct_tokens = (predictions == target_tokens) & valid_mask
    return correct_tokens.sum().item() / total_tokens


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    pad_idx: int = 0,
    grad_clip: float = 1.0,
    log_every: int = 10,
) -> float:
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        images = batch["images"].to(device, non_blocking=True)
        caption_tokens = batch["caption_tokens"].to(device, non_blocking=True)
        input_tokens, target_tokens = prepare_teacher_forcing_inputs(
            caption_tokens,
            pad_idx=pad_idx,
        )

        logits = model(images, input_tokens)
        loss = captioning_loss(logits, target_tokens, criterion, pad_idx=pad_idx)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if log_every and (batch_idx + 1) % log_every == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(train_loader)}: loss = {total_loss / num_batches:.4f}",
                flush=True,
            )

    return total_loss / max(num_batches, 1)


def validate(
    model,
    data_loader,
    criterion,
    device,
    pad_idx: int = 0,
    return_accuracy: bool = False,
):
    """Validate model and return average loss, optionally with token accuracy."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_accuracy = 0.0

    with torch.no_grad():
        for batch in data_loader:
            images = batch["images"].to(device, non_blocking=True)
            caption_tokens = batch["caption_tokens"].to(device, non_blocking=True)
            input_tokens, target_tokens = prepare_teacher_forcing_inputs(
                caption_tokens,
                pad_idx=pad_idx,
            )

            logits = model(images, input_tokens)
            loss = captioning_loss(logits, target_tokens, criterion, pad_idx=pad_idx)
            accuracy = caption_token_accuracy(logits, target_tokens, pad_idx=pad_idx)

            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_accuracy = total_accuracy / max(num_batches, 1)
    if return_accuracy:
        return avg_loss, avg_accuracy
    return avg_loss


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    vocab,
    checkpoint_dir: Path,
    name: str = "model",
    filename: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Path:
    """Save model checkpoint with vocabulary for reproducible inference."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "vocab": vocab,
        "model_name": getattr(model, "model_name", model.__class__.__name__),
    }
    if extra:
        payload["extra"] = extra

    if filename is None:
        filename = f"{name}_epoch{epoch}.pt"

    path = checkpoint_dir / filename
    torch.save(payload, path)
    print(f"Checkpoint saved: {path}", flush=True)
    return path


def load_checkpoint(model, optimizer, checkpoint_path: Path, device):
    """Load model checkpoint and return the saved training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    vocab = checkpoint.get("vocab")
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss")
    print(f"Checkpoint loaded from {checkpoint_path}", flush=True)
    if loss is not None:
        print(f"  Epoch: {epoch}, Loss: {loss:.4f}", flush=True)
    else:
        print(f"  Epoch: {epoch}", flush=True)
    return model, optimizer, epoch, vocab, loss


def get_checkpoint_epoch(checkpoint_path: Path, device):
    """Read a checkpoint file and return the saved epoch."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint epoch: {epoch}", flush=True)
    return epoch


@torch.no_grad()
def generate_caption(
    model,
    before_image,
    after_image,
    vocab,
    device,
    max_len: int = 100,
) -> str:
    """Greedy caption generation for one image pair."""
    model.eval()

    images = torch.stack([before_image, after_image], dim=0).unsqueeze(0).to(device)
    if hasattr(model, "encode_images"):
        encoder_features = model.encode_images(images)[0]
    elif hasattr(model, "encoder"):
        encoder_features = model.encoder(images)
    else:
        raise AttributeError(
            f"'{type(model).__name__}' object has no attribute 'encoder' or 'encode_images'"
        )

    caption_tokens = [vocab.start_idx]
    for _ in range(max_len):
        cap_tensor = torch.tensor([caption_tokens], dtype=torch.long, device=device)
        logits = model.decoder(encoder_features, cap_tensor)
        next_token = logits[0, -1, :].argmax(-1).item()
        caption_tokens.append(next_token)
        if next_token == vocab.end_idx:
            break

    return vocab.decode(caption_tokens)


def visualize_predictions(
    model,
    data_loader,
    vocab,
    device,
    num_samples: int = None,
    output_pdf: str = "predictions.pdf",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Save before/after images with reference and predicted captions to a PDF.

    Args:
        model: Trained captioning model.
        data_loader: DataLoader.
        vocab: Vocabulary object.
        device: torch device.
        num_samples: Number of samples to save. If None, saves the entire dataset.
        output_pdf: Output PDF filename.
        mean: ImageNet mean.
        std: ImageNet std.
    """
    import math
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from src.utils import denormalize_image

    model.eval()

    dataset_size = len(data_loader.dataset)
    if num_samples is None:
        num_samples = dataset_size
    else:
        num_samples = min(num_samples, dataset_size)

    sample_count = 0

    with PdfPages(output_pdf) as pdf:
        with torch.no_grad():
            for batch in data_loader:
                batch_size = batch["before_images"].shape[0]

                for idx in range(batch_size):
                    if sample_count >= num_samples:
                        break

                    before_img = batch["before_images"][idx]
                    after_img = batch["after_images"][idx]
                    ref_caption = batch["captions"][idx]

                    pred_caption = generate_caption(
                        model,
                        before_img,
                        after_img,
                        vocab,
                        device,
                    )

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(
                        denormalize_image(
                            before_img,
                            mean=mean,
                            std=std,
                        )
                    )
                    axes[0].set_title("Before")
                    axes[0].axis("off")

                    axes[1].imshow(
                        denormalize_image(
                            after_img,
                            mean=mean,
                            std=std,
                        )
                    )
                    axes[1].set_title("After")
                    axes[1].axis("off")

                    axes[2].axis("off")
                    axes[2].text(
                        0,
                        1,
                        f"Sample: {sample_count + 1}\n\n"
                        f"Reference:\n{ref_caption}\n\n"
                        f"Prediction:\n{pred_caption}",
                        fontsize=10,
                        wrap=True,
                        verticalalignment="top",
                    )

                    plt.tight_layout()

                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                    sample_count += 1

                if sample_count >= num_samples:
                    break

    print(f"Saved {sample_count} predictions to '{output_pdf}'.")


def build_optimizer_and_scheduler(model, train_config):
    """Create Adam optimizer and cosine scheduler from TrainConfig."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config.scheduler_t_max,
    )
    return optimizer, scheduler


def build_criterion(vocab, label_smoothing: float = 0.0) -> Callable:
    """Shared cross-entropy loss with padding ignored and optional label smoothing."""
    return nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=label_smoothing)
