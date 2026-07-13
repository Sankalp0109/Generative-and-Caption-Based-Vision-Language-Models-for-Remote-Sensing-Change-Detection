"""Phase 7: Hierarchical RemoteCLIP Tile-Based Change Captioning.

Architecture:
    Image Pair → Tile Extraction → RemoteCLIP Encoding →
    Per-Tile Bidirectional Cross-Attention → Difference Embedding →
    Tile Fusion Transformer (CLS) → Caption Decoder
"""

from .config import Phase7Config
from .losses import phase7_total_loss
from .model import TileBasedChangeCaptioningModel

__all__ = [
    "Phase7Config",
    "TileBasedChangeCaptioningModel",
    "phase7_total_loss",
]
