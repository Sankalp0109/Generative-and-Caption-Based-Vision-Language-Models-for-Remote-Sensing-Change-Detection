"""Training entry point for Phase 8 stage 1a: validate the difference module.

Only DifferenceModule and LightweightCaptionDecoder parameters are trained;
RemoteCLIPEncoder stays frozen. Reuses the existing generic training
helpers (loss, accuracy, checkpointing) from src/training.py unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.training import (
    build_criterion,
    captioning_loss,
    caption_token_accuracy,
    load_best_model,
    load_best_metrics,
    load_checkpoint,
    prepare_teacher_forcing_inputs,
    save_checkpoint,
)


def _trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def run_epoch(
    model,
    data_loader,
    vocab,
    device,
    criterion,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 1.0,
    log_every: int = 10,
) -> tuple:
    """Run one epoch. Trains if `optimizer` is given, otherwise evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(data_loader):
            before_images = batch["before_images"].to(device, non_blocking=True)
            after_images = batch["after_images"].to(device, non_blocking=True)
            caption_tokens = batch["caption_tokens"].to(device, non_blocking=True)

            input_tokens, target_tokens = prepare_teacher_forcing_inputs(
                caption_tokens, pad_idx=vocab.pad_idx
            )

            logits = model(before_images, after_images, input_tokens)
            loss = captioning_loss(logits, target_tokens, criterion, pad_idx=vocab.pad_idx)
            acc = caption_token_accuracy(logits, target_tokens, pad_idx=vocab.pad_idx)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(_trainable_parameters(model), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

            if is_train and log_every and (batch_idx + 1) % log_every == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(data_loader)}: "
                    f"loss={total_loss / num_batches:.4f} acc={total_acc / num_batches:.4f}",
                    flush=True,
                )

    denom = max(num_batches, 1)
    return total_loss / denom, total_acc / denom


def train_stage1a(
    model,
    train_loader,
    val_loader,
    vocab,
    device,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    log_every: int = 10,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_name: str = "phase8_stage1a",
):
    """Train DifferenceModule + LightweightCaptionDecoder end-to-end.

    RemoteCLIPEncoder is frozen by construction (see RemoteCLIPEncoder),
    so the optimizer only ever sees diff-module and decoder parameters.

    Two checkpoints are saved:
    - {checkpoint_name}_current.pt: Latest epoch (for resume)
    - {checkpoint_name}_best.pt: Best model (for evaluation)

    On resume, loads current checkpoint to continue from latest epoch with
    best metrics restored.

    After training, use load_best_model() to load the best model for evaluation:
        model, best_epoch, best_loss, vocab = load_best_model(
            model, checkpoint_dir, checkpoint_name, device
        )
    """
    model.to(device)
    criterion = build_criterion(vocab)
    optimizer = torch.optim.Adam(
        _trainable_parameters(model), lr=lr, weight_decay=weight_decay
    )

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_dir_path = Path(checkpoint_dir) if checkpoint_dir else None

    # Load current checkpoint (resume from current epoch with current weights)
    current_ckpt_path = checkpoint_dir_path / f"{checkpoint_name}_current.pt" if checkpoint_dir_path else None
    if current_ckpt_path is not None and current_ckpt_path.is_file():
        _, optimizer, start_epoch, _, loaded_loss, loaded_best_epoch, loaded_best_loss = load_checkpoint(
            model, optimizer, current_ckpt_path, device
        )
        if loaded_best_epoch is not None:
            best_epoch = loaded_best_epoch
        if loaded_best_loss is not None:
            best_val_loss = loaded_best_loss
        print(
            f"[Phase8 Stage1a] Resuming from epoch {start_epoch} "
            f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})",
            flush=True,
        )
    else:
        # If no current checkpoint, try loading best metrics
        if checkpoint_dir_path:
            loaded_best_epoch, loaded_best_loss = load_best_metrics(
                checkpoint_dir_path, checkpoint_name, device
            )
            if loaded_best_epoch is not None:
                best_epoch = loaded_best_epoch
            if loaded_best_loss is not None:
                best_val_loss = loaded_best_loss

    history = []
    if start_epoch >= epochs:
        print(
            f"[Phase8 Stage1a] Checkpoint epoch {start_epoch} already reached "
            f"target epochs={epochs}; nothing to train.",
            flush=True,
        )
        return history

    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, vocab, device, criterion,
            optimizer=optimizer, grad_clip=grad_clip, log_every=log_every,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, vocab, device, criterion, optimizer=None,
        )
        print(
            f"Epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
             "val_loss": val_loss, "val_acc": val_acc}
        )

        if checkpoint_dir is not None:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, val_loss, vocab,
                checkpoint_dir, name=checkpoint_name,
                best_epoch=best_epoch,
                best_loss=best_val_loss,
                is_best=is_best,
                save_current=True,
                save_best=True,
            )

    return history


def train_stage1b(
    model,
    train_loader,
    val_loader,
    vocab,
    device,
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    log_every: int = 10,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_name: str = "phase8_stage1b",
    approach: str = "A/B",
):
    """Fine-tune stage 1a model with frozen LLM.

    Two approaches:
    - A/B: DifferenceModule + Adaptive Bridge → Frozen Qwen
    - C: Q-Former → Frozen Qwen

    Two checkpoints are saved:
    - {checkpoint_name}_current.pt: Latest epoch (for resume)
    - {checkpoint_name}_best.pt: Best model (for evaluation)

    On resume, loads current checkpoint to continue from latest epoch with
    best metrics restored.

    After training, use load_best_model() to load the best model for evaluation:
        model, best_epoch, best_loss, vocab = load_best_model(
            model, checkpoint_dir, checkpoint_name, device
        )
    """
    model.to(device)
    criterion = build_criterion(vocab)
    optimizer = torch.optim.Adam(
        _trainable_parameters(model), lr=lr, weight_decay=weight_decay
    )

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_dir_path = Path(checkpoint_dir) if checkpoint_dir else None

    # Load current checkpoint (resume from current epoch with current weights)
    current_ckpt_path = checkpoint_dir_path / f"{checkpoint_name}_current.pt" if checkpoint_dir_path else None
    if current_ckpt_path is not None and current_ckpt_path.is_file():
        _, optimizer, start_epoch, _, loaded_loss, loaded_best_epoch, loaded_best_loss = load_checkpoint(
            model, optimizer, current_ckpt_path, device
        )
        if loaded_best_epoch is not None:
            best_epoch = loaded_best_epoch
        if loaded_best_loss is not None:
            best_val_loss = loaded_best_loss
        print(
            f"[Phase8 Stage1b] Resuming from epoch {start_epoch} "
            f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})",
            flush=True,
        )
    else:
        # If no current checkpoint, try loading best metrics
        if checkpoint_dir_path:
            loaded_best_epoch, loaded_best_loss = load_best_metrics(
                checkpoint_dir_path, checkpoint_name, device
            )
            if loaded_best_epoch is not None:
                best_epoch = loaded_best_epoch
            if loaded_best_loss is not None:
                best_val_loss = loaded_best_loss

    history = []
    if start_epoch >= epochs:
        print(
            f"[Phase8 Stage1b] Checkpoint epoch {start_epoch} already reached "
            f"target epochs={epochs}; nothing to train.",
            flush=True,
        )
        return history

    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, vocab, device, criterion,
            optimizer=optimizer, grad_clip=grad_clip, log_every=log_every,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, vocab, device, criterion, optimizer=None,
        )
        print(
            f"Epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"[{approach}]",
            flush=True,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
             "val_loss": val_loss, "val_acc": val_acc}
        )

        if checkpoint_dir is not None:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, val_loss, vocab,
                checkpoint_dir, name=checkpoint_name,
                best_epoch=best_epoch,
                best_loss=best_val_loss,
                is_best=is_best,
                save_current=True,
                save_best=True,
            )

    return history
