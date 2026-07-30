"""Stage 1: Tile Extraction.

Splits a batch of before/after image pairs into a regular grid of tiles.
Each tile corresponds to the same geographical region in both images.

Input shape:  (B, 2, 3, H, W)   -- stacked image pairs from the dataloader
Output shape: (B, N, 2, 3, tile_h, tile_w)

    where N = grid_size * grid_size (number of tiles per image)

Design notes
------------
- Extraction is performed *inside* the model forward pass so the dataloader
  and dataset remain entirely unchanged.
- Tiles are extracted with `torch.nn.functional.unfold` style slicing rather
  than requiring H and W to be perfect multiples of tile_size.
- The module is stateless and contains no learnable parameters.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TileExtractor(nn.Module):
    """Extract a regular grid of non-overlapping 256x256 patches from stacked before/after image pairs.

    If image dimensions are not divisible by patch_size, right/bottom borders are padded with 0.
    Patch indices strictly follow row-major spatial ordering.

    Args:
        patch_size: Side length of each square patch (default 256).
    """

    def __init__(
        self,
        patch_size: int = 256,
        grid_size: Optional[int] = None,
        tile_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.tile_size = tile_size or (patch_size, patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract ordered patches from a batch of image pairs.

        Args:
            images: Tensor of shape (B, N, 2, 3, ph, pw) or (B, 2, 3, H, W).

        Returns:
            tiles: Tensor of shape (B, N, 2, 3, patch_size, patch_size).
        """
        # If images are already pre-patched: (B, N, 2, 3, ph, pw)
        if images.ndim == 6:
            return images

        if images.ndim != 5 or images.size(1) != 2:
            raise ValueError(
                "TileExtractor expects images shaped (B, 2, 3, H, W) or (B, N, 2, 3, ph, pw). "
                f"Got shape {tuple(images.shape)}."
            )

        import math

        B, _, C, H, W = images.shape
        ph, pw = self.patch_size, self.patch_size

        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw

        if pad_h > 0 or pad_w > 0:
            # F.pad format: (left, right, top, bottom)
            images = F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=0)

        padded_H, padded_W = images.shape[-2], images.shape[-1]

        tiles: list[torch.Tensor] = []
        for r in range(0, padded_H, ph):
            for c in range(0, padded_W, pw):
                patch = images[:, :, :, r : r + ph, c : c + pw]  # (B, 2, C, ph, pw)
                tiles.append(patch)

        return torch.stack(tiles, dim=1)  # (B, N, 2, C, ph, pw)
