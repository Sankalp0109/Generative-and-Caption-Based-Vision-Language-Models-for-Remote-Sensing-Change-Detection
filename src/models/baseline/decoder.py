"""Decoder architectures for caption generation."""

import torch
import torch.nn as nn


class SimpleDecoder(nn.Module):
    """
    Transformer decoder for caption generation.

    Input:
        encoder_features: (batch, encoder_dim)
        caption_tokens:   (batch, seq_len)
    Output:
        logits: (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 100,
        encoder_dim: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.encoder_projection = nn.Linear(encoder_dim, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features, caption_tokens):
        batch_size, seq_len = caption_tokens.shape
        device = caption_tokens.device

        embedded = self.dropout(self.embedding(caption_tokens))
        embedded = embedded + self.positional_encoding[:, :seq_len, :].to(device)

        if encoder_features.ndim == 2:
            encoder_memory = self.encoder_projection(encoder_features).unsqueeze(1)
        else:
            encoder_memory = self.encoder_projection(encoder_features)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        decoded = self.transformer_decoder(
            tgt=embedded,
            memory=encoder_memory,
            tgt_mask=tgt_mask,
        )
        return self.output_projection(decoded)
