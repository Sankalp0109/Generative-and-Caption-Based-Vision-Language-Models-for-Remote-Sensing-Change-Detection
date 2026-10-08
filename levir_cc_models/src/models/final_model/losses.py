"""Loss helpers for the phase 5 RemoteCLIP final model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def captioning_loss(logits, target_tokens, criterion, pad_idx: int = 0):
    """Compute token-level caption loss with the provided criterion."""
    batch_size, seq_len, vocab_size = logits.shape
    return criterion(
        logits.reshape(batch_size * seq_len, vocab_size),
        target_tokens.reshape(batch_size * seq_len),
    )


def contrastive_caption_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
):
    """Symmetric InfoNCE loss for image-caption alignment.

    The batch order defines the positive pairs: image i matches caption i.
    """
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("contrastive_caption_loss expects 2D embedding tensors.")
    if image_embeddings.size(0) != text_embeddings.size(0):
        raise ValueError("Image and text embeddings must have the same batch size.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    logits = torch.matmul(image_embeddings, text_embeddings.t())
    logits = logits / temperature
    labels = torch.arange(image_embeddings.size(0), device=image_embeddings.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)


def phase5_total_loss(
    logits,
    target_tokens,
    criterion,
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    contrastive_weight: float = 0.1,
    temperature: float = 0.07,
    pad_idx: int = 0,
):
    """Combine caption loss and contrastive alignment loss.

    Returns a tuple of:
        total_loss, caption_loss, contrastive_loss
    """
    if contrastive_weight < 0:
        raise ValueError("contrastive_weight must be non-negative.")

    caption_loss_value = captioning_loss(logits, target_tokens, criterion, pad_idx=pad_idx)
    contrastive_loss_value = contrastive_caption_loss(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        temperature=temperature,
    )
    total_loss = caption_loss_value + contrastive_weight * contrastive_loss_value
    return total_loss, caption_loss_value, contrastive_loss_value