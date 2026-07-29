"""Stage 6: Tile-Based Change Captioning Model.

Integrates:
  1. TileExtractor
  2. TileEncoder (shared RemoteCLIP backbone)
  3. TileBidirectionalDifference
  4. TileFusionTransformer
  5. SimpleDecoder
  6. CaptionContrastiveEncoder (for alignment)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..baseline.decoder import SimpleDecoder
from ..interface import ChangeCaptioningModel
from ..final_model.remoteclip_cross_attention import (
    CaptionContrastiveEncoder,
    REMOTECLIP_REPO_ID,
)

from .tile_extractor import TileExtractor
from .tile_encoder import TileEncoder
from .tile_difference import TileBidirectionalDifference
from .tile_fusion import TileFusionTransformer
from .config import Phase6Config


class TileBasedChangeCaptioningModel(ChangeCaptioningModel):
    """Phase 6 Model: Hierarchical Tile-Based Change Captioning.

    This architecture extracts tile grids, encodes them with a shared RemoteCLIP
    backbone, computes bidirectional per-tile difference embeddings, fuses them
    via a Transformer with a CLS token, and generates captions with a standard
    decoder.
    """

    def __init__(
        self,
        vocab_size: int,
        config: Optional[Phase6Config] = None,
        # Or individual overrides:
        pad_idx: int = 0,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = config or Phase6Config()
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

        # ── 3. Bidirectional Difference Module ────────────────────────────
        self.tile_diff = TileBidirectionalDifference(
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

        # Stage 3: Bidirectional difference
        diff_embeddings = self.tile_diff(before_feats, after_feats)  # (B, N, fusion_dim)

        # Stage 4: Fusion Transformer
        global_features, fused_tokens = self.tile_fusion(diff_embeddings, return_all_tokens=True)  # (B, global_dim), (B, N+1, global_dim)

        # Contrastive space projection
        image_embeddings = self.image_projection(global_features)

        return global_features, image_embeddings, fused_tokens

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
        global_features, image_embeddings, fused_tokens = self.encode_images(images)
        logits = self.decoder(fused_tokens, caption_tokens)

        if not return_aux:
            return logits

        text_embeddings = self.encode_captions(caption_tokens)
        return {
            "logits": logits,
            "change_features": global_features,
            "fused_tokens": fused_tokens,
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }
