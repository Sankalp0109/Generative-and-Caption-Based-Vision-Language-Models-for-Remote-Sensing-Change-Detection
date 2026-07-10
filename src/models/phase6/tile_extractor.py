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
  than requiring H and W to be perfect multiples of tile_size; each slice is
  independently resized to tile_size using bilinear interpolation.
- The module is stateless and contains no learnable parameters.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TileExtractor(nn.Module):
    """Extract a regular grid of tiles from stacked before/after image pairs.

    Args:
        grid_size: Side length of the tile grid.  grid_size=2 → 2×2 = 4 tiles.
        tile_size: (height, width) to which every extracted patch is resized.
    """

    def __init__(
        self,
        grid_size: int = 2,
        tile_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.num_tiles = grid_size * grid_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract tiles from a batch of image pairs.

        Args:
            images: Tensor of shape (B, 2, 3, H, W).
                    Channel 0 = before image, Channel 1 = after image.

        Returns:
            tiles: Tensor of shape (B, N, 2, 3, tile_h, tile_w).
                   N = grid_size^2 tiles per image pair.

        Raises:
            ValueError: If ``images`` does not have 5 dimensions or the second
                        dimension is not 2.
        """
        if images.ndim != 5 or images.size(1) != 2:
            raise ValueError(
                "TileExtractor expects images shaped (B, 2, 3, H, W). "
                f"Got shape {tuple(images.shape)}."
            )

        B, _, C, H, W = images.shape
        g = self.grid_size
        th, tw = self.tile_size

        # Compute tile boundaries along H and W.
        row_edges = self._split_edges(H, g)
        col_edges = self._split_edges(W, g)

        tiles: list[torch.Tensor] = []
        for r_start, r_end in row_edges:
            for c_start, c_end in col_edges:
                # (B, 2, C, patch_h, patch_w)
                patch = images[:, :, :, r_start:r_end, c_start:c_end]

                # Resize each temporal channel independently to tile_size.
                # Merge B and temporal dims so we can call interpolate once.
                B2, two, C2, ph, pw = patch.shape
                patch_flat = patch.view(B2 * two, C2, ph, pw)
                patch_resized = F.interpolate(
                    patch_flat,
                    size=self.tile_size,
                    mode="bilinear",
                    align_corners=False,
                )
                # (B, 2, C, th, tw)
                patch_resized = patch_resized.view(B2, two, C2, th, tw)
                tiles.append(patch_resized)

        # Stack into (B, N, 2, C, th, tw)
        return torch.stack(tiles, dim=1)

    @staticmethod
    def _split_edges(length: int, num_parts: int):
        """Divide [0, length) into num_parts roughly equal intervals."""
        edges = []
        prev = 0
        for i in range(1, num_parts + 1):
            end = int(round(length * i / num_parts))
            edges.append((prev, end))
            prev = end
        return edges
