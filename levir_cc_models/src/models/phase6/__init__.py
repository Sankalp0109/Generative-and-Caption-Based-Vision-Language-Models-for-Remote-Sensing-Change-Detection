"""Phase 6: Hierarchical RemoteCLIP Tile-Based Change Captioning.

Architecture:
    Image Pair → Tile Extraction → RemoteCLIP Encoding →
    Per-Tile Bidirectional Cross-Attention → Difference Embedding →
    Tile Fusion Transformer (CLS) → Caption Decoder
"""

from .config import Phase6Config
from .losses import phase6_total_loss
from .model import TileBasedChangeCaptioningModel

__all__ = [
    "Phase6Config",
    "TileBasedChangeCaptioningModel",
    "phase6_total_loss",
]
