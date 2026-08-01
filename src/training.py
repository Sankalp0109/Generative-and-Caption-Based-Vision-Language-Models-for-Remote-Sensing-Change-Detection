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
    optimizer,
    device,
    grad_clip: float = 1.0,
    log_every: int = 10,
    max_caption_len: int = 128,
) -> float:
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        before_images = batch["before_images"].to(device, non_blocking=True)
        after_images = batch["after_images"].to(device, non_blocking=True)
        captions = batch["captions"]

        # Tokenize captions with the model's BPE tokenizer
        tokenized = model.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_caption_len,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # Labels: mask padding tokens with -100 so they are ignored in loss
        labels = input_ids.clone()
        labels[labels == model.tokenizer.pad_token_id] = -100

        outputs = model(
            before_image=before_images,
            after_image=after_images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
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
    device,
    max_caption_len: int = 128,
) -> float:
    """Validate model and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            before_images = batch["before_images"].to(device, non_blocking=True)
            after_images = batch["after_images"].to(device, non_blocking=True)
            captions = batch["captions"]

            tokenized = model.tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_caption_len,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            labels = input_ids.clone()
            labels[labels == model.tokenizer.pad_token_id] = -100

            outputs = model(
                before_image=before_images,
                after_image=after_images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


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

    import io
    import os

    path = checkpoint_dir / filename
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    
    # Save atomically by writing to a temporary file first and renaming it
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "wb") as f:
            f.write(buffer.getvalue())
        os.replace(tmp_path, path)
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise e
        
    print(f"Checkpoint saved: {path}", flush=True)
    return path


def load_checkpoint(model, optimizer, checkpoint_path: Path, device):
    """Load model checkpoint and return the saved training state."""
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"\n[ERROR] Failed to load checkpoint from {checkpoint_path}!", flush=True)
        print(f"The file may be corrupted, empty, or truncated. Details: {e}", flush=True)
        
        if checkpoint_path.exists():
            corrupted_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".corrupted")
            try:
                checkpoint_path.rename(corrupted_path)
                print(f"[INFO] Renamed corrupted checkpoint to {corrupted_path}", flush=True)
            except Exception as rename_err:
                print(f"[ERROR] Could not rename corrupted checkpoint: {rename_err}", flush=True)
        
        raise RuntimeError(
            f"Checkpoint file {checkpoint_path} is corrupted. "
            f"It has been renamed to {checkpoint_path.name}.corrupted to avoid blocking future runs. "
            f"Please re-run the process to start fresh or fall back to the best checkpoint."
        ) from e

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
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"\n[ERROR] Failed to load checkpoint from {checkpoint_path} to get epoch!", flush=True)
        print(f"The file may be corrupted, empty, or truncated. Details: {e}", flush=True)
        
        if checkpoint_path.exists():
            corrupted_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".corrupted")
            try:
                checkpoint_path.rename(corrupted_path)
                print(f"[INFO] Renamed corrupted checkpoint to {corrupted_path}", flush=True)
            except Exception as rename_err:
                print(f"[ERROR] Could not rename corrupted checkpoint: {rename_err}", flush=True)
        
        raise RuntimeError(
            f"Checkpoint file {checkpoint_path} is corrupted. "
            f"It has been renamed to {checkpoint_path.name}.corrupted to avoid blocking future runs."
        ) from e

    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint epoch: {epoch}", flush=True)
    return epoch


@torch.no_grad()
def generate_caption(
    model,
    before_image,
    after_image,
    device,
    max_new_tokens: int = 64,
) -> str:
    """Generate a caption for one before/after image pair."""
    model.eval()
    if before_image.dim() == 3:
        before_image = before_image.unsqueeze(0)
    if after_image.dim() == 3:
        after_image = after_image.unsqueeze(0)

    before_image = before_image.to(device)
    after_image = after_image.to(device)

    captions = model.generate_caption(
        before_image, after_image, max_new_tokens=max_new_tokens
    )
    return captions[0]


def visualize_predictions(
    model,
    data_loader,
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
        device: torch device.
        num_samples: Number of samples to save. If None, saves the entire dataset.
        output_pdf: Output PDF filename.
        mean: ImageNet mean.
        std: ImageNet std.
    """
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
                before_images = batch["before_images"]
                after_images = batch["after_images"]
                captions = batch.get("captions", [None] * before_images.size(0))
                batch_size = before_images.size(0)

                # Generate captions for the batch
                preds = model.generate_caption(
                    before_images.to(device),
                    after_images.to(device),
                )

                for idx in range(batch_size):
                    if sample_count >= num_samples:
                        break

                    before_img = before_images[idx]
                    after_img = after_images[idx]
                    pred_caption = preds[idx]
                    ref_caption = captions[idx] if captions[idx] is not None else ""

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(
                        denormalize_image(before_img, mean=mean, std=std)
                    )
                    axes[0].set_title("Before")
                    axes[0].axis("off")

                    axes[1].imshow(
                        denormalize_image(after_img, mean=mean, std=std)
                    )
                    axes[1].set_title("After")
                    axes[1].axis("off")

                    axes[2].axis("off")
                    text_content = f"Sample: {sample_count + 1}\n\n"
                    if ref_caption:
                        text_content += f"Reference:\n{ref_caption}\n\n"
                    text_content += f"Prediction:\n{pred_caption}"
                    axes[2].text(
                        0,
                        1,
                        text_content,
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


def build_criterion(vocab) -> Callable:
    """Shared cross-entropy loss with padding ignored."""
    return nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
