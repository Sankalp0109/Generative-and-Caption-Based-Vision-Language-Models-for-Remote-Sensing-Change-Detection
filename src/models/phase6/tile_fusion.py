"""Stage 4–5: Tile Fusion Transformer + Global Representation Extraction.

Takes the sequence of per-tile difference embeddings and fuses them into a
single global change representation using a standard Transformer encoder with
a learnable CLS token.

Pipeline:
    diff_embeddings (B, N, fusion_dim)
      → prepend CLS token → add positional embeddings
      → Transformer encoder
      → CLS' (updated) → project to global_dim
      → global_representation (B, global_dim)

Design notes
------------
- A learnable CLS token acts as a query that aggregates information from all
  tiles, allowing the model to attend to any spatial combination.
- Learnable positional embeddings distinguish tile spatial positions.
- The CLS token position embedding is included at index 0.
- After the transformer, the CLS output is extracted and projected to the
  decoder's expected encoder_dim via a small projection head.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class TileFusionTransformer(nn.Module):
    """Fuse per-tile difference embeddings into a single global representation.

    Args:
        fusion_dim: Dimensionality of each tile embedding (from TileDifference).
        num_tiles: Number of tiles per image (N = grid_size^2).
        global_dim: Output dimension of the projected global representation.
        num_layers: Number of Transformer encoder layers.
        num_heads: Attention heads in each encoder layer.
        ffn_dim: Hidden dimension of the feedforward sublayer.
        dropout: Dropout rate throughout.
    """

    def __init__(
        self,
        fusion_dim: int = 512,
        num_tiles: int = 4,
        global_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        if fusion_dim % num_heads != 0:
            raise ValueError(
                f"fusion_dim ({fusion_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.fusion_dim = fusion_dim
        self.num_tiles = num_tiles
        self.global_dim = global_dim
        self.seq_len = num_tiles + 1  # +1 for CLS token

        # ── CLS token and positional embeddings ────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding covers [CLS, tile_0, tile_1, ..., tile_{N-1}]
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.seq_len, fusion_dim)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # ── Transformer encoder ────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm (more stable for small batches)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(fusion_dim),
        )

        # ── Global projection head ─────────────────────────────────────────
        self.global_projection = nn.Sequential(
            nn.Linear(fusion_dim, global_dim),
            nn.LayerNorm(global_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, diff_embeddings: torch.Tensor) -> torch.Tensor:
        """Fuse tile embeddings into a global change representation.

        Args:
            diff_embeddings: (B, N, fusion_dim)

        Returns:
            global_repr: (B, global_dim) — projected CLS token output.
        """
        if diff_embeddings.ndim != 3:
            raise ValueError(
                "TileFusionTransformer expects diff_embeddings shaped "
                f"(B, N, fusion_dim). Got shape {tuple(diff_embeddings.shape)}."
            )

        B, N, D = diff_embeddings.shape

        # ── Prepend CLS token ──────────────────────────────────────────────
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        sequence = torch.cat([cls, diff_embeddings], dim=1)  # (B, N+1, D)

        # ── Add positional embeddings ──────────────────────────────────────
        sequence = sequence + self.pos_embedding[:, : N + 1, :]

        # ── Transformer encoding ───────────────────────────────────────────
        encoded = self.transformer(sequence)             # (B, N+1, D)

        # ── Extract CLS output (position 0) and project ───────────────────
        cls_output = encoded[:, 0, :]                   # (B, D)
        global_repr = self.global_projection(cls_output)  # (B, global_dim)
        return global_repr
