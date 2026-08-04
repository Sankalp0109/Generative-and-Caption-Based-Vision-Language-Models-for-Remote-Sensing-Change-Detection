"""Trainable cross-attention difference module for Phase 8.

This is the module the entire curriculum exists to validate: given frozen
before/after RemoteCLIP embeddings, learn a difference embedding that a
decoder can turn into a change caption. It is deliberately the *only*
component trained in stage 1a, paired with a cheap lightweight decoder, so
gradient signal reaches it directly instead of being diluted by backprop
through a large frozen LLM.

Operates on (B, N, backbone_dim): N=1 for the single whole-image stage 1
curriculum, N>1 for a future per-patch stage 2 -- the same shared weights
apply to each position independently.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Phase8Config


class DifferenceModule(nn.Module):
    """Forward cross-attention + MLP difference embedding.

    Di = MLP([before_i, after_i, cross_attn_i, |before_i - after_i|])

    Args:
        backbone_dim: Feature dimension from RemoteCLIP (e.g. 512).
        fusion_dim: Output dimension of the difference embedding.
        num_heads: Number of cross-attention heads. Must divide backbone_dim.
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
                f"backbone_dim ({backbone_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.backbone_dim = backbone_dim
        self.fusion_dim = fusion_dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=backbone_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(backbone_dim)

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

    @classmethod
    def from_config(cls, config: Phase8Config) -> "DifferenceModule":
        return cls(
            backbone_dim=config.backbone_dim,
            fusion_dim=config.fusion_dim,
            num_heads=config.fusion_heads,
            dropout=config.diff_mlp_dropout,
        )

    def forward(
        self,
        before_features: torch.Tensor,
        after_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute difference embeddings.

        Args:
            before_features: (B, N, backbone_dim)
            after_features:  (B, N, backbone_dim)
        Returns:
            diff_embeddings: (B, N, fusion_dim)
        """
        if before_features.shape != after_features.shape:
            raise ValueError(
                "before_features and after_features must have the same shape. "
                f"Got {tuple(before_features.shape)} and {tuple(after_features.shape)}."
            )

        B, N, D = before_features.shape
        q = before_features.reshape(B * N, 1, D)
        k = after_features.reshape(B * N, 1, D)
        v = after_features.reshape(B * N, 1, D)

        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        attn_out = attn_out.reshape(B, N, D)
        attn_out = self.norm_attn(before_features + attn_out)

        abs_diff = torch.abs(before_features - after_features)

        combined = torch.cat(
            [before_features, after_features, attn_out, abs_diff], dim=-1
        )  # (B, N, 4 * backbone_dim)

        return self.diff_mlp(combined)  # (B, N, fusion_dim)
