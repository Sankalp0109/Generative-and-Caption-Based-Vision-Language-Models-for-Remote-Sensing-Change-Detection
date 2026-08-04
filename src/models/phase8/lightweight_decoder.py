"""Lightweight caption decoder used only to validate the difference module.

Stage 1a swaps in this small transformer decoder instead of the eventual
frozen Qwen2-VL-2B bridge (stage 1b), so the loss gradient reaching
DifferenceModule is strong and cheap to iterate on. It uses the project's
existing Vocabulary (src/dataset.py) rather than a BPE tokenizer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .config import Phase8Config


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class LightweightCaptionDecoder(nn.Module):
    """Small Transformer decoder that cross-attends to a difference embedding.

    forward(memory, input_tokens) -> logits (B, L, vocab_size), matching the
    (memory, caption_tokens) -> logits convention already used elsewhere in
    this repo (see src/training.py).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        config: Optional[Phase8Config] = None,
    ):
        super().__init__()
        self.config = config or Phase8Config()
        embed_dim = self.config.embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=self.config.max_caption_len)
        self.memory_proj = nn.Linear(self.config.fusion_dim, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=self.config.num_decoder_heads,
            dim_feedforward=embed_dim * 4,
            dropout=self.config.decoder_dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.config.num_decoder_layers
        )
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.pad_idx = pad_idx

    def forward(self, memory: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory: (B, N, fusion_dim) difference embedding(s) from DifferenceModule.
            input_tokens: (B, L) teacher-forcing input token ids.
        Returns:
            logits: (B, L, vocab_size)
        """
        memory = self.memory_proj(memory)  # (B, N, embed_dim)

        tokens = self.token_embed(input_tokens)
        tokens = self.pos_encoding(tokens)

        L = input_tokens.size(1)
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        padding_mask = input_tokens == self.pad_idx

        hidden = self.decoder(
            tgt=tokens,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        return self.output_proj(hidden)
