"""2D Bicubic Positional Embedding Interpolation for RemoteCLIP ViT-L-14.

Reshapes the pretrained 256-token spatial grid (16x16 for 224px / 14px patches) to an 18x18 grid (324 tokens)
via 2D bicubic interpolation while excluding the CLS token at index 0 from the spatial reshape.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def bicubic_interpolate_pos_embed(
    pos_embed_param: nn.Parameter,
    new_grid_size: Tuple[int, int] = (18, 18),
) -> nn.Parameter:
    """Interpolate ViT spatial positional embedding from (old_h, old_w) to (new_h, new_w).
    
    Args:
        pos_embed_param: Parameter tensor of shape (1, 1 + old_N, C) or (1 + old_N, C)
        new_grid_size: Tuple (new_h, new_w) e.g. (18, 18) for 252px / 14px patches
    Returns:
        new_pos_embed: nn.Parameter of shape (1, 1 + new_h*new_w, C) or (1 + new_h*new_w, C)
    """
    is_3d = pos_embed_param.dim() == 3
    pos_embed = pos_embed_param.data if is_3d else pos_embed_param.data.unsqueeze(0)
    B, total_tokens, C = pos_embed.shape

    cls_embed = pos_embed[:, :1, :]
    spatial_embed = pos_embed[:, 1:, :]  # (1, old_N, C)

    old_n = spatial_embed.size(1)
    old_side = int(math.sqrt(old_n))
    if old_side * old_side != old_n:
        raise ValueError(f"Spatial pos_embed token count {old_n} is not a square number.")

    new_h, new_w = new_grid_size
    if old_side == new_h and old_side == new_w:
        # No interpolation needed
        return pos_embed_param

    # Reshape spatial_embed to (1, C, old_side, old_side) for 2D interpolation
    spatial_embed_2d = spatial_embed.reshape(1, old_side, old_side, C).permute(0, 3, 1, 2)
    interpolated_2d = F.interpolate(
        spatial_embed_2d,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    )
    # Permute back to (1, new_h*new_w, C)
    interpolated_spatial = interpolated_2d.permute(0, 2, 3, 1).reshape(1, new_h * new_w, C)

    new_pos_embed = torch.cat([cls_embed, interpolated_spatial], dim=1)
    if not is_3d:
        new_pos_embed = new_pos_embed.squeeze(0)

    print(
        f"[CodeAug] Bicubic-interpolated ViT pos_embed from grid {old_side}x{old_side} ({old_n} tokens) "
        f"to {new_h}x{new_w} ({new_h * new_w} tokens). Total tokens with CLS: {new_pos_embed.size(0 if not is_3d else 1)}"
    )
    return nn.Parameter(new_pos_embed)


def update_vit_pos_embed(vit_model: nn.Module, new_grid_size: Tuple[int, int] = (18, 18)):
    """Update positional embedding attribute on an OpenCLIP or standard ViT model."""
    if hasattr(vit_model, "positional_embedding"):
        vit_model.positional_embedding = bicubic_interpolate_pos_embed(
            vit_model.positional_embedding, new_grid_size
        )
    elif hasattr(vit_model, "pos_embed"):
        vit_model.pos_embed = bicubic_interpolate_pos_embed(
            vit_model.pos_embed, new_grid_size
        )
    elif hasattr(vit_model, "embeddings") and hasattr(vit_model.embeddings, "position_embeddings"):
        vit_model.embeddings.position_embeddings = bicubic_interpolate_pos_embed(
            vit_model.embeddings.position_embeddings, new_grid_size
        )
    else:
        raise AttributeError("Could not locate positional embedding parameter on ViT model.")
