"""Configuration for Phase 8: curriculum-trained difference-embedding RSICC.

Stage 1a of the curriculum: frozen RemoteCLIP -> trainable cross-attention
difference module -> trainable lightweight decoder. See
src/models/phase8/model.py for how these pieces compose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class Phase8Config:
    """Hyperparameters for the Phase 8 single-embedding difference model."""

    # ── Input & RemoteCLIP backbone ──────────────────────────────────────────
    img_size: Tuple[int, int] = (256, 256)
    """Fixed input resolution. No patching in stage 1 (whole image = one tile)."""

    remoteclip_model_name: str = "ViT-B-32"
    """OpenCLIP architecture string matching the RemoteCLIP checkpoint."""

    remoteclip_checkpoint_path: Path = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")
    """Local path to the real RemoteCLIP weights (512-dim pooled output)."""

    backbone_dim: int = 512
    """Pooled embedding dimension produced by RemoteCLIP ViT-B-32."""

    remoteclip_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    remoteclip_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    # ── Difference module (trainable) ────────────────────────────────────────
    fusion_dim: int = 512
    """Output dimension of the difference embedding."""

    fusion_heads: int = 4
    """Number of attention heads in the forward cross-attention."""

    diff_mlp_dropout: float = 0.1
    """Dropout rate inside the difference MLP."""

    # ── Lightweight caption decoder (trainable, stage 1a only) ──────────────
    embed_dim: int = 256
    """Token embedding dimension inside the lightweight decoder."""

    num_decoder_heads: int = 4
    """Number of attention heads in the lightweight decoder."""

    num_decoder_layers: int = 2
    """Number of Transformer decoder layers."""

    max_caption_len: int = 100
    """Maximum decoded caption length."""

    decoder_dropout: float = 0.1
    """Dropout rate inside the lightweight decoder."""

    def __post_init__(self):
        self.remoteclip_checkpoint_path = Path(self.remoteclip_checkpoint_path)
