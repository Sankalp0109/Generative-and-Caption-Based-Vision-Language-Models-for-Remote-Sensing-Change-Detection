"""Loss helper functions for Phase 7 tile-based change captioning.

Provides the combined objective for training: cross-entropy caption loss plus
an optional InfoNCE contrastive caption alignment loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def captioning_loss(logits: torch.Tensor, target_tokens: torch.Tensor, criterion: torch.nn.Module, pad_idx: int = 0) -> torch.Tensor:
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
) -> torch.Tensor:
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


def phase7_total_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    criterion: torch.nn.Module,
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    contrastive_weight: float = 0.1,
    temperature: float = 0.07,
    pad_idx: int = 0,
):
    """Combine caption loss and contrastive alignment loss.

    Returns:
        total_loss, caption_loss, contrastive_loss
    """
    if contrastive_weight < 0:
        raise ValueError("contrastive_weight must be non-negative.")

    caption_loss_value = captioning_loss(
        logits, target_tokens, criterion, pad_idx=pad_idx
    )

    if contrastive_weight > 0:
        contrastive_loss_value = contrastive_caption_loss(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            temperature=temperature,
        )
        total_loss = (
            caption_loss_value + contrastive_weight * contrastive_loss_value
        )
    else:
        # Avoid computing contrastive loss if weight is 0.
        contrastive_loss_value = torch.tensor(0.0, device=logits.device)
        total_loss = caption_loss_value

    return total_loss, caption_loss_value, contrastive_loss_value
