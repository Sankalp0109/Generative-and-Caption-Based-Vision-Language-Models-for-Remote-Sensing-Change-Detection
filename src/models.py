"""Backward-compatible re-exports. Prefer importing from src.models package."""

from src.models.baseline.baseline import RSICCformerBaseline
from src.models.baseline.decoder import SimpleDecoder
from src.models.baseline.encoder import SimpleEncoder
from src.models.interface import ChangeCaptioningModel

__all__ = [
    "ChangeCaptioningModel",
    "SimpleEncoder",
    "SimpleDecoder",
    "RSICCformerBaseline",
]
