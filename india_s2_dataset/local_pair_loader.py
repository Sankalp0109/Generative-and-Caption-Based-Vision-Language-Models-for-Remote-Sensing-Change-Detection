"""Load pre-aligned local before/after image pairs (no georeferencing required).

Unlike ``pair_alignment.load_aligned_pair`` (which reprojects two georeferenced
rasters onto one shared AOI grid), this is for image pairs that are already on
the same pixel grid -- e.g. a change-detection dataset exported as plain PNGs
with matching filenames in a `before/` and an `after/` directory. There is no
CRS or bbox to validate here, so the only safety check is that every pair
actually matches: same filename present in both directories, identical pixel
dimensions. A filename found in only one directory is reported rather than
silently skipped, so a partially-copied dataset can't quietly mislabel pairs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def list_pairs(before_dir: Path, after_dir: Path, pattern: str = "*.png") -> list[str]:
    """Return sorted filenames present in both directories; raise on any mismatch."""
    before_files = {p.name for p in before_dir.glob(pattern)}
    after_files = {p.name for p in after_dir.glob(pattern)}
    only_before = sorted(before_files - after_files)
    only_after = sorted(after_files - before_files)
    if only_before or only_after:
        raise ValueError(
            f"before/after directories do not match one-to-one; "
            f"only in before: {only_before}; only in after: {only_after}"
        )
    return sorted(before_files, key=lambda name: (len(name), name))


def load_local_pair(before_path: str | Path, after_path: str | Path,
                    max_dimension: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return two float32 RGB arrays on one identical pixel grid.

    Raises if the two images are not already the same size -- this loader does
    not resample one image onto the other, since without georeferencing there
    is no way to know that would still compare the same ground area.
    """
    before_image, after_image = Image.open(before_path).convert("RGB"), Image.open(after_path).convert("RGB")
    if before_image.size != after_image.size:
        raise ValueError(
            f"before/after size mismatch for {Path(before_path).name}: "
            f"{before_image.size} vs {after_image.size}"
        )
    if max_dimension is not None:
        largest = max(before_image.size)
        if largest > max_dimension:
            scale = max_dimension / largest
            target = (max(1, round(before_image.width * scale)), max(1, round(before_image.height * scale)))
            before_image = before_image.resize(target, Image.BILINEAR)
            after_image = after_image.resize(target, Image.BILINEAR)
    return (np.asarray(before_image, dtype=np.float32), np.asarray(after_image, dtype=np.float32))
