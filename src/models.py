"""Backward-compatible re-exports. Prefer importing from src.models package."""

from src.models.baseline.baseline import RSICCformerBaseline
from src.models.baseline.decoder import SimpleDecoder
from src.models.baseline.encoder import SimpleEncoder
from src.models.interface import ChangeCaptioningModel
from src.models.final_model import (
    Phase5RemoteCLIPModel,
    RemoteCLIPCrossAttentionModel,
    captioning_loss,
    contrastive_caption_loss,
    phase5_total_loss,
)
from src.models.phase6 import (
    Phase6Config,
    TileBasedChangeCaptioningModel,
    phase6_total_loss,
)

__all__ = [
    "ChangeCaptioningModel",
    "SimpleEncoder",
    "SimpleDecoder",
    "RSICCformerBaseline",
    "RemoteCLIPCrossAttentionModel",
    "Phase5RemoteCLIPModel",
    "captioning_loss",
    "contrastive_caption_loss",
    "phase5_total_loss",
    "Phase6Config",
    "TileBasedChangeCaptioningModel",
    "phase6_total_loss",
]

