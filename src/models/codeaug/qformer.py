"""Q-Former Token Compressor for CodeAug RSICC.

Compresses 324 visual patch tokens (18x18 grid) down to 64 learnable latent query tokens
(a 61.3% sequence reduction: 424 -> 164 total tokens), preventing KV-cache bloat and
fitting comfortably inside the ADA cluster 11 GB VRAM envelope.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CodeAugConfig


class QFormerLayer(nn.Module):
    """Single transformer cross-attention + self-attention + FFN layer for Q-Former."""

    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        # Self-attention among query tokens
        q_self, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q_self)

        # Cross-attention: queries attend to 324 visual patch tokens
        q_cross, _ = self.cross_attn(queries, visual_tokens, visual_tokens)
        queries = self.norm2(queries + q_cross)

        # Feed-forward network
        ffn_out = self.ffn(queries)
        queries = self.norm3(queries + ffn_out)

        return queries


class QFormerTokenCompressor(nn.Module):
    """Compresses N visual tokens (e.g. 324) down to `num_queries` (64) tokens and projects to LLM embedding dim."""

    def __init__(self, config: CodeAugConfig, llm_hidden_size: int = 1536):
        super().__init__()
        self.num_queries = config.num_queries
        self.fusion_dim = config.fusion_dim
        self.llm_hidden_size = llm_hidden_size

        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, self.fusion_dim) * 0.02)

        self.layers = nn.ModuleList(
            [
                QFormerLayer(
                    embed_dim=self.fusion_dim,
                    num_heads=config.qformer_heads,
                    dropout=0.1,
                )
                for _ in range(config.qformer_layers)
            ]
        )

        self.norm = nn.LayerNorm(self.fusion_dim)
        self.to_llm = nn.Linear(self.fusion_dim, self.llm_hidden_size)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Compress visual tokens from (B, N, fusion_dim) -> (B, num_queries, llm_hidden_size).
        
        Args:
            visual_tokens: Tensor of shape (B, N, fusion_dim) e.g. (B, 324, 1024)
        Returns:
            projected_queries: Tensor of shape (B, 64, llm_hidden_size) ready for LLM input
        """
        B = visual_tokens.size(0)
        queries = self.query_tokens.expand(B, -1, -1)  # (B, 64, fusion_dim)

        for layer in self.layers:
            queries = layer(queries, visual_tokens)

        queries = self.norm(queries)
        return self.to_llm(queries)  # (B, 64, llm_hidden_size)
