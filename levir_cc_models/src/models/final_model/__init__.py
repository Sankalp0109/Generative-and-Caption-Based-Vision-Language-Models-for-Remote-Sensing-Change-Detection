"""Phase 5 final model package.

This package contains the RemoteCLIP + cross-attention fusion model and the
contrastive caption-alignment losses used in the final proposed framework.
"""

from .losses import captioning_loss, contrastive_caption_loss, phase5_total_loss
from .remoteclip_cross_attention import (
    Phase5RemoteCLIPModel,
    RemoteCLIPCrossAttentionModel,
)

__all__ = [
    "RemoteCLIPCrossAttentionModel",
    "Phase5RemoteCLIPModel",
    "captioning_loss",
    "contrastive_caption_loss",
    "phase5_total_loss",
]