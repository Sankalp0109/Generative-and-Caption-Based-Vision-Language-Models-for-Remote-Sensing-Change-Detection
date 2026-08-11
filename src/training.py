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
    """Train one epoch and return average loss. Uses GradScaler for float16 stability."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

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

        optimizer.zero_grad()

        # Use autocast for mixed precision if on CUDA
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    before_image=before_images,
                    after_image=after_images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            if grad_clip > 0:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                before_image=before_images,
                after_image=after_images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
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

            # Use autocast for float16 consistency (no gradients needed, no scaler)
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(
                        before_image=before_images,
                        after_image=after_images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
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
    best_epoch: Optional[int] = None,
    best_loss: Optional[float] = None,
    is_best: bool = False,
    save_best: bool = True,
    save_current: bool = True,
) -> tuple:
    """Save model checkpoint(s) with vocabulary for reproducible inference.

    Saves up to two checkpoint files:
    - current: Current epoch model + optimizer (always updated)
    - best: Best model weights + metrics (only when is_best=True)

    Args:
        is_best: If True, also save as best checkpoint.
        save_best: Whether to save best checkpoint.
        save_current: Whether to save current checkpoint.

    Returns:
        tuple: (current_checkpoint_path, best_checkpoint_path or None)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    import io
    import os

    if filename is None:
        filename = f"{name}_epoch{epoch}.pt"

    current_path = None
    best_path = None

    # SAVE CURRENT CHECKPOINT: Always save current epoch's state
    if save_current:
        current_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "vocab": vocab,
            "model_name": getattr(model, "model_name", model.__class__.__name__),
            "best_epoch": best_epoch if best_epoch is not None else epoch,
            "best_loss": best_loss if best_loss is not None else loss,
        }
        if extra:
            current_payload["extra"] = extra

        current_path = checkpoint_dir / f"{name}_current.pt"
        buffer = io.BytesIO()
        torch.save(current_payload, buffer)

        tmp_path = current_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "wb") as f:
                f.write(buffer.getvalue())
            os.replace(tmp_path, current_path)
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise e

        print(f"  Current checkpoint: {current_path}", flush=True)

    # SAVE BEST CHECKPOINT: Only update when is_best=True
    if save_best and is_best:
        best_payload = {
            "epoch": best_epoch if best_epoch is not None else epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": best_loss if best_loss is not None else loss,
            "vocab": vocab,
            "model_name": getattr(model, "model_name", model.__class__.__name__),
            "best_epoch": best_epoch if best_epoch is not None else epoch,
            "best_loss": best_loss if best_loss is not None else loss,
        }
        if extra:
            best_payload["extra"] = extra

        best_path = checkpoint_dir / f"{name}_best.pt"
        buffer = io.BytesIO()
        torch.save(best_payload, buffer)

        tmp_path = best_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "wb") as f:
                f.write(buffer.getvalue())
            os.replace(tmp_path, best_path)
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise e

        print(f"  Best checkpoint: {best_path}", flush=True)

    return current_path, best_path


def load_checkpoint(model, optimizer, checkpoint_path: Path, device):
    """Load model checkpoint and return the saved training state.

    Returns: (model, optimizer, epoch, vocab, loss, best_epoch, best_loss)
    """
    from src.dataset import Vocabulary
    torch.serialization.add_safe_globals([Vocabulary])

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
    best_epoch = checkpoint.get("best_epoch", epoch)
    best_loss = checkpoint.get("best_loss", loss)
    print(f"Checkpoint loaded from {checkpoint_path}", flush=True)
    print(f"  Current Epoch: {epoch}, Best Epoch: {best_epoch}", flush=True)
    if loss is not None:
        print(f"  Current Loss: {loss:.4f}, Best Loss: {best_loss:.4f}", flush=True)
    return model, optimizer, epoch, vocab, loss, best_epoch, best_loss


def load_best_metrics(checkpoint_dir: Path, checkpoint_name: str, device):
    """Load only the best metrics from best checkpoint without loading model.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        checkpoint_name: Base name of checkpoint (e.g., "phase8_stage1a").
        device: Device to load to.

    Returns:
        tuple: (best_epoch, best_loss) or (None, None) if checkpoint not found.
    """
    from src.dataset import Vocabulary
    torch.serialization.add_safe_globals([Vocabulary])

    best_path = Path(checkpoint_dir) / f"{checkpoint_name}_best.pt"
    if not best_path.exists():
        return None, None

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        best_epoch = checkpoint.get("best_epoch", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"Best metrics loaded: epoch {best_epoch}, loss {best_loss:.4f}", flush=True)
        return best_epoch, best_loss
    except Exception as e:
        print(f"[WARNING] Could not load best metrics: {e}", flush=True)
        return None, None


def load_best_model(model, checkpoint_dir: Path, checkpoint_name: str, device):
    """Load the best model for evaluation/inference (no optimizer needed).

    Args:
        model: Model to load weights into.
        checkpoint_dir: Directory containing checkpoints.
        checkpoint_name: Base name of checkpoint (e.g., "phase8_stage1a").
        device: Device to load to.

    Returns:
        tuple: (model, best_epoch, best_loss, vocab) or (model, None, None, None) if checkpoint not found.
    """
    from src.dataset import Vocabulary
    torch.serialization.add_safe_globals([Vocabulary])

    best_path = Path(checkpoint_dir) / f"{checkpoint_name}_best.pt"
    if not best_path.exists():
        print(f"[WARNING] Best checkpoint not found: {best_path}", flush=True)
        return model, None, None, None

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        best_epoch = checkpoint.get("best_epoch", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        vocab = checkpoint.get("vocab")

        print(f"Best model loaded for evaluation", flush=True)
        print(f"  Best Epoch: {best_epoch}, Best Loss: {best_loss:.4f}", flush=True)
        return model, best_epoch, best_loss, vocab
    except Exception as e:
        print(f"[ERROR] Failed to load best model: {e}", flush=True)
        raise RuntimeError(
            f"Could not load best checkpoint from {best_path}. "
            f"Training may not have completed successfully."
        ) from e


def get_checkpoint_epoch(checkpoint_path: Path, device):
    """Read a checkpoint file and return the saved epoch."""
    from src.dataset import Vocabulary
    torch.serialization.add_safe_globals([Vocabulary])

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
