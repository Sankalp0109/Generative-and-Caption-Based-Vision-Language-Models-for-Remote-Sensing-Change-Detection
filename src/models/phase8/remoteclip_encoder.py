"""Frozen RemoteCLIP ViT-B-32 encoder for Phase 8.

Loads the real RemoteCLIP checkpoint (satellite-domain-pretrained CLIP
weights, not generic LAION weights) and exposes a single pooled embedding
per image. Entirely frozen -- this backbone is never trained in the
Phase 8 curriculum, only the difference module downstream of it.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import resolve_remoteclip_checkpoint

from .config import Phase8Config


def _interpolate_pos_embed(pos_embed: torch.Tensor, new_side: int) -> torch.Tensor:
    """Bicubic-interpolate a ViT's (1 + old_side**2, C) positional embedding
    (CLS token excluded from the spatial reshape) to a new square grid, so a
    checkpoint pretrained at one resolution can serve a different fixed
    input resolution (e.g. RemoteCLIP's native 224px -> this module's 256px)."""
    cls_embed, spatial_embed = pos_embed[:1], pos_embed[1:]
    old_n, C = spatial_embed.shape
    old_side = int(math.sqrt(old_n))
    if old_side * old_side != old_n:
        raise ValueError(f"Spatial pos_embed token count {old_n} is not a square number.")
    if old_side == new_side:
        return pos_embed

    grid = spatial_embed.reshape(1, old_side, old_side, C).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new_side, new_side), mode="bicubic", align_corners=False)
    spatial_embed = grid.permute(0, 2, 3, 1).reshape(new_side * new_side, C)
    return torch.cat([cls_embed, spatial_embed], dim=0)


def _unwrap_state_dict(checkpoint) -> dict:
    """RemoteCLIP checkpoints are typically a raw state_dict, but tolerate
    common wrapper keys in case a differently-packaged checkpoint is used."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


class RemoteCLIPEncoder(nn.Module):
    """Frozen RemoteCLIP ViT-B-32 image encoder producing a pooled embedding.

    Accepts either a single image batch (B, 3, H, W) or a pre-patched batch
    (B, N, 3, H, W); the patch dimension (if present) is flattened through
    the encoder and restored on output, so the same module serves both the
    single-image stage 1 curriculum and a future per-patch stage 2.
    """

    def __init__(self, config: Optional[Phase8Config] = None):
        super().__init__()
        self.config = config or Phase8Config()
        self.model = self._load_backbone(self.config)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _load_backbone(self, config: Phase8Config) -> nn.Module:
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            config.remoteclip_model_name, pretrained=None
        )
        checkpoint_path = resolve_remoteclip_checkpoint(
            config.remoteclip_checkpoint_path,
            config.remoteclip_model_name,
            download_if_missing=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _unwrap_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            warnings.warn(
                f"[Phase8] RemoteCLIP checkpoint loaded with {len(missing)} missing "
                f"and {len(unexpected)} unexpected keys."
            )

        # The checkpoint's positional embedding is sized for RemoteCLIP's native
        # resolution (224px / 32px patches -> 7x7 grid); interpolate it to match
        # this module's fixed config.img_size so pretrained knowledge transfers.
        visual = model.visual
        patch_size = visual.conv1.kernel_size[0]
        new_side = config.img_size[0] // patch_size
        visual.positional_embedding = nn.Parameter(
            _interpolate_pos_embed(visual.positional_embedding.data, new_side)
        )
        return model

    def train(self, mode: bool = True):
        # Backbone is always frozen/eval, regardless of the parent module's mode.
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into pooled RemoteCLIP embeddings.

        Args:
            images: (B, 3, H, W) or (B, N, 3, H, W).
        Returns:
            (B, backbone_dim) or (B, N, backbone_dim).
        """
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            flat = images.reshape(B * N, C, H, W)
            pooled = self.model.encode_image(flat)
            return pooled.reshape(B, N, -1)
        if images.dim() == 4:
            return self.model.encode_image(images)
        raise ValueError(
            f"RemoteCLIPEncoder expects (B,3,H,W) or (B,N,3,H,W), got {tuple(images.shape)}."
        )
