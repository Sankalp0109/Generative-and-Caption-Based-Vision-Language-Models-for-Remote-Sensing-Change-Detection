"""Phase 6 configuration dataclass.

All hyperparameters specific to the hierarchical tile-based model live here.
Shared hyperparameters (data paths, base model dims) continue to be read from
src/config.py -- this file only extends those defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Phase6Config:
    """Hyperparameters for the Phase 6 tile-based change captioning model.

    These values were chosen to work within ADA cluster constraints:
        - 1 GPU, 16 GB RAM
        - batch_size=8  (DataConfig)
        - num_workers=0 (required on ADA -- no DataLoader multiprocessing)
    """

    # ── Tile extraction ──────────────────────────────────────────────────────
    grid_size: int = 2
    """Side length of the tile grid. grid_size=2 → 2×2 = 4 tiles per image."""

    tile_size: Tuple[int, int] = (224, 224)
    """Each extracted tile is resized to this resolution before RemoteCLIP encoding."""

    # ── Tile difference module ────────────────────────────────────────────────
    fusion_dim: int = 512
    """Dimension of per-tile difference embeddings (must equal RemoteCLIP output dim)."""

    fusion_heads: int = 4
    """Number of attention heads in the per-tile bidirectional cross-attention."""

    diff_mlp_dropout: float = 0.1
    """Dropout rate inside the per-tile difference MLP."""

    # ── Tile fusion transformer ───────────────────────────────────────────────
    num_fusion_layers: int = 2
    """Number of Transformer encoder layers in the tile fusion transformer."""

    fusion_nhead: int = 4
    """Number of attention heads in the tile fusion transformer."""

    fusion_ffn_dim: int = 2048
    """Feed-forward hidden dimension inside the fusion transformer."""

    fusion_dropout: float = 0.1
    """Dropout rate in the tile fusion transformer."""

    # ── Global projection ─────────────────────────────────────────────────────
    global_dim: int = 512
    """Projected global embedding dimension passed to the caption decoder."""

    # ── Caption decoder (mirrors Phase 5 / ModelConfig defaults) ─────────────
    embed_dim: int = 256
    """Token embedding dimension inside the caption decoder."""

    num_decoder_heads: int = 4
    """Number of attention heads in the caption decoder."""

    num_decoder_layers: int = 2
    """Number of Transformer decoder layers."""

    max_caption_len: int = 100
    """Maximum decoded caption length."""

    decoder_dropout: float = 0.1
    """Dropout rate inside the caption decoder."""

    # ── Contrastive alignment (optional) ────────────────────────────────────
    contrastive_dim: int = 256
    """Projection dim for contrastive image/caption embeddings."""

    contrastive_weight: float = 0.1
    """Weight λ of the contrastive loss term in the combined objective."""

    temperature: float = 0.07
    """InfoNCE temperature τ."""

    # ── RemoteCLIP backbone ───────────────────────────────────────────────────
    remoteclip_model_name: str = "ViT-B-32"
    """Open-CLIP model architecture string for RemoteCLIP."""

    remoteclip_checkpoint_path: Path = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")
    """Local path to the RemoteCLIP weight file."""

    freeze_remoteclip: bool = True
    """If True, backbone gradients are disabled and backbone stays in eval mode."""

    download_if_missing: bool = False
    """Download checkpoint from HuggingFace Hub when not found locally."""

    remoteclip_repo_id: str = "chendelong/RemoteCLIP"
    """HuggingFace repository for RemoteCLIP weights."""

    # RemoteCLIP ViT-B-32 normalisation constants (OpenCLIP convention)
    remoteclip_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    remoteclip_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    def __post_init__(self):
        self.remoteclip_checkpoint_path = Path(self.remoteclip_checkpoint_path)

    @property
    def num_tiles(self) -> int:
        """Total number of tiles per image."""
        return self.grid_size * self.grid_size
