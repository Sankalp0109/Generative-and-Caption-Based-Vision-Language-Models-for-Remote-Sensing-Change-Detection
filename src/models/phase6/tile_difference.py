"""Stage 3: Per-Tile Difference Embedding.

For each spatial tile i, directional change information is extracted simply
by taking the difference between before (A) and after (B) features:

  diff_ab = A - B
  diff_ba = B - A

These are concatenated with the raw features to
form a compact change embedding via an MLP:

  Di = MLP([A, B, A-B, B-A])   Di ∈ R^fusion_dim

Input:
    before_features  (B, N, backbone_dim)
    after_features   (B, N, backbone_dim)

Output:
    diff_embeddings  (B, N, fusion_dim)

Design notes
------------
- All N tiles share the same MLP weights (parameter
  efficient, consistent with the spec's "shared encoder" intent).
- The MLP input dimension is 4 * backbone_dim (A, B, A-B, B-A).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TileDifference(nn.Module):
    """Per-tile MLP difference embedding.

    Args:
        backbone_dim: Feature dimension from RemoteCLIP (e.g. 512).
        fusion_dim: Output dimension of each per-tile difference embedding.
        num_heads: Number of attention heads (unused, kept for compatibility).
        dropout: Dropout applied inside the MLP.
    """

    def __init__(
        self,
        backbone_dim: int = 512,
        fusion_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if backbone_dim % num_heads != 0:
            raise ValueError(
                f"backbone_dim ({backbone_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.backbone_dim = backbone_dim
        self.fusion_dim = fusion_dim

        # ── MLP: [A, B, A-B, B-A] → fusion_dim ──────────────────────
        mlp_in = 4 * backbone_dim
        self.diff_mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_in // 2),
            nn.LayerNorm(mlp_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_in // 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        before_features: torch.Tensor,
        after_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-tile difference embeddings.

        Args:
            before_features: (B, N, backbone_dim)
            after_features:  (B, N, backbone_dim)

        Returns:
            diff_embeddings: (B, N, fusion_dim)
        """
        if before_features.shape != after_features.shape:
            raise ValueError(
                "before_features and after_features must have the same shape. "
                f"Got {tuple(before_features.shape)} and "
                f"{tuple(after_features.shape)}."
            )

        # ── Compute directional differences ───────────────────────────────
        diff_ab = before_features - after_features
        diff_ba = after_features - before_features

        # ── Concatenate per-patch signals in spatial sequence order ──────
        combined = torch.cat(
            [before_features, after_features, diff_ab, diff_ba],
            dim=-1,
        )  # (B, N, 4 * backbone_dim)

        diff_embeddings = self.diff_mlp(combined)  # (B, N, fusion_dim)
        return diff_embeddings
