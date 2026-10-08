"""Loss helper functions for Phase 6 tile-based change captioning.

Provides the combined objective for training: cross-entropy caption loss plus
an optional InfoNCE contrastive caption alignment loss.
"""

from __future__ import annotations

import torch
from ..final_model.losses import (
    captioning_loss,
    contrastive_caption_loss,
)


def phase6_total_loss(
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
