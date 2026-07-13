"""Stage 7: Tile-Based Change Captioning Model.

Integrates:
  1. TileExtractor
  2. TileEncoder (shared RemoteCLIP backbone)
  3. TileBidirectionalDifference
  4. TileFusionTransformer
  5. SimpleDecoder (defined locally)
  6. CaptionContrastiveEncoder (defined locally)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..interface import ChangeCaptioningModel

from .tile_extractor import TileExtractor
from .tile_encoder import TileEncoder, REMOTECLIP_REPO_ID
from .tile_difference import TileDifference
from .tile_fusion import TileFusionTransformer
from .config import Phase7Config


class SimpleDecoder(nn.Module):
    """Transformer decoder for caption generation.

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

        encoder_memory = self.encoder_projection(encoder_features).unsqueeze(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        decoded = self.transformer_decoder(
            tgt=embedded,
            memory=encoder_memory,
            tgt_mask=tgt_mask,
        )
        return self.output_projection(decoded)


class CaptionContrastiveEncoder(nn.Module):
    """Encode caption tokens into the shared contrastive space."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        projection_dim: int,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, caption_tokens: torch.Tensor) -> torch.Tensor:
        if caption_tokens.ndim != 2:
            raise ValueError("CaptionContrastiveEncoder expects caption tokens shaped (batch, seq_len).")

        embedded = self.embedding(caption_tokens)
        mask = (caption_tokens != self.pad_idx).unsqueeze(-1).float()
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = (embedded * mask).sum(dim=1) / lengths
        return self.projection(pooled)


class TileBasedChangeCaptioningModel(ChangeCaptioningModel):
    """Phase 7 Model: Hierarchical Tile-Based Change Captioning.

    This architecture extracts tile grids, encodes them with a shared RemoteCLIP
    backbone, computes per-tile difference embeddings using forward cross-attention,
    fuses them via a Transformer with a CLS token, and generates captions with a standard
    decoder.
    """

    def __init__(
        self,
        vocab_size: int,
        config: Optional[Phase7Config] = None,
        # Or individual overrides:
        pad_idx: int = 0,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = config or Phase7Config()
        self.pad_idx = pad_idx

        # ── 1. Tile Extraction ────────────────────────────────────────────
        self.tile_extractor = TileExtractor(
            grid_size=self.cfg.grid_size,
            tile_size=self.cfg.tile_size,
        )

        # ── 2. Tile Encoder (RemoteCLIP Backbone) ─────────────────────────
        self.tile_encoder = TileEncoder(
            model_name=self.cfg.remoteclip_model_name,
            checkpoint_path=self.cfg.remoteclip_checkpoint_path,
            freeze_backbone=self.cfg.freeze_remoteclip,
            download_if_missing=self.cfg.download_if_missing,
            repo_id=self.cfg.remoteclip_repo_id,
            backbone=backbone,
        )

        # ── 3. Difference Module ──────────────────────────────────────────
        self.tile_diff = TileDifference(
            backbone_dim=self.tile_encoder.backbone_dim,
            fusion_dim=self.cfg.fusion_dim,
            num_heads=self.cfg.fusion_heads,
            dropout=self.cfg.diff_mlp_dropout,
        )

        # ── 4. Tile Fusion Transformer ────────────────────────────────────
        self.tile_fusion = TileFusionTransformer(
            fusion_dim=self.cfg.fusion_dim,
            num_tiles=self.cfg.num_tiles,
            global_dim=self.cfg.global_dim,
            num_layers=self.cfg.num_fusion_layers,
            num_heads=self.cfg.fusion_nhead,
            ffn_dim=self.cfg.fusion_ffn_dim,
            dropout=self.cfg.fusion_dropout,
        )

        # ── 5. Caption Decoder ────────────────────────────────────────────
        self.decoder = SimpleDecoder(
            vocab_size=vocab_size,
            embed_dim=self.cfg.embed_dim,
            num_heads=self.cfg.num_decoder_heads,
            num_layers=self.cfg.num_decoder_layers,
            max_len=self.cfg.max_caption_len,
            encoder_dim=self.cfg.global_dim,
            dropout=self.cfg.decoder_dropout,
            pad_idx=pad_idx,
        )

        # ── 6. Contrastive Projection and Text Encoder ───────────────────
        self.image_projection = nn.Sequential(
            nn.Linear(self.cfg.global_dim, self.cfg.contrastive_dim),
            nn.LayerNorm(self.cfg.contrastive_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.decoder_dropout),
            nn.Linear(self.cfg.contrastive_dim, self.cfg.contrastive_dim),
        )

        self.caption_encoder = CaptionContrastiveEncoder(
            vocab_size=vocab_size,
            embed_dim=self.cfg.embed_dim,
            projection_dim=self.cfg.contrastive_dim,
            pad_idx=pad_idx,
            dropout=self.cfg.decoder_dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Ensure frozen backbone stays in eval() mode.
        self.tile_encoder.train(mode)
        self.tile_extractor.train(mode)
        self.tile_diff.train(mode)
        self.tile_fusion.train(mode)
        return self

    def encode_images(self, images: torch.Tensor):
        """Encode images into global change representations and contrastive embeddings.

        Args:
            images: (B, 2, 3, H, W)

        Returns:
            global_features: (B, global_dim)
            image_embeddings: (B, contrastive_dim)
            diff_embeddings: (B, N, fusion_dim)
        """
        # Stage 1: Extract tiles
        tiles = self.tile_extractor(images)  # (B, N, 2, 3, th, tw)

        # Stage 2: Encode tiles
        before_feats, after_feats = self.tile_encoder(tiles)  # (B, N, D)

        # Stage 3: Forward difference
        diff_embeddings = self.tile_diff(before_feats, after_feats)  # (B, N, fusion_dim)

        # Stage 4: Fusion Transformer
        global_features = self.tile_fusion(diff_embeddings)  # (B, global_dim)

        # Contrastive space projection
        image_embeddings = self.image_projection(global_features)

        return global_features, image_embeddings, diff_embeddings

    def encode_captions(self, caption_tokens: torch.Tensor) -> torch.Tensor:
        """Encode caption tokens to contrastive text embeddings."""
        return self.caption_encoder(caption_tokens)

    def forward(
        self,
        images: torch.Tensor,
        caption_tokens: torch.Tensor,
        return_aux: bool = False,
    ):
        """Forward pass generating caption logits.

        Args:
            images: (B, 2, 3, H, W)
            caption_tokens: (B, L)
            return_aux: If True, returns dict with auxiliary tensors for contrastive loss.
        """
        global_features, image_embeddings, diff_embeddings = self.encode_images(images)
        logits = self.decoder(global_features, caption_tokens)

        if not return_aux:
            return logits

        text_embeddings = self.encode_captions(caption_tokens)
        return {
            "logits": logits,
            "change_features": global_features,
            "fused_tokens": diff_embeddings,
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }
