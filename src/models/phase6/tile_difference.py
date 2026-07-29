"""Stage 3: Per-Tile Forward Cross-Attention + Difference Embedding.

For each spatial tile i, a forward cross-attention operation extracts directional
change information:

  Forward  (before→after): Q=Fi, K=Gi, V=Gi   → "what appeared"

These are concatenated with the raw features and the absolute difference to
form a compact change embedding via an MLP:

  Di = MLP([Fi, Gi, forward_i, |Fi - Gi|])   Di ∈ R^fusion_dim

Input:
    before_features  (B, N, backbone_dim)
    after_features   (B, N, backbone_dim)

Output:
    diff_embeddings  (B, N, fusion_dim)

Design notes
------------
- All N tiles share the same cross-attention and MLP weights (parameter
  efficient, consistent with the spec's "shared encoder" intent).
- Cross-attention is implemented as standard nn.MultiheadAttention applied
  element-wise across the tile dimension (tiles are processed as a sequence
  of length N with batch dimension B).
- The MLP input dimension is 4 * backbone_dim (Fi, Gi, fwd, |diff|).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TileDifference(nn.Module):
    """Per-tile forward cross-attention + MLP difference embedding.

    Args:
        backbone_dim: Feature dimension from RemoteCLIP (e.g. 512).
        fusion_dim: Output dimension of each per-tile difference embedding.
        num_heads: Number of attention heads.  Must divide backbone_dim.
        dropout: Dropout applied inside attention and the MLP.
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

        # ── Forward cross-attention ───────────────────────────────────────
        self.forward_attn = nn.MultiheadAttention(
            embed_dim=backbone_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_fwd = nn.LayerNorm(backbone_dim)

        # ── MLP: [Fi, Gi, fwd, |diff|] → fusion_dim ──────────────────────
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

        # ── Compute per-patch difference matching spatial index i ─────────
        # Ensures patch_i (before) is strictly paired with patch_i (after)
        fwd_context, _ = self.forward_attn(
            query=before_features,
            key=after_features,
            value=after_features,
        )
        fwd_context = self.norm_fwd(before_features + fwd_context)

        # ── Element-wise absolute feature difference ──────────────────────
        abs_diff = torch.abs(after_features - before_features)

        # ── Concatenate per-patch signals in spatial sequence order ──────
        combined = torch.cat(
            [before_features, after_features, fwd_context, abs_diff],
            dim=-1,
        )  # (B, N, 4 * backbone_dim)

        diff_embeddings = self.diff_mlp(combined)  # (B, N, fusion_dim)
        return diff_embeddings


TileBidirectionalDifference = TileDifference
