"""Strictly align two georeferenced rasters to the same requested ground area."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio

# Some Earth Search COGs resolve to an S3 path that GDAL otherwise tries to sign,
# failing with InvalidCredentials on a machine with no AWS credentials configured.
# These are public buckets, so requests never need to be signed.
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds


def _contains(outer, inner, tolerance=1e-7):
    return (outer.left <= inner[0] + tolerance and outer.bottom <= inner[1] + tolerance
            and outer.right >= inner[2] - tolerance and outer.top >= inner[3] - tolerance)


def _native_resolution_in(crs, source) -> tuple[float, float]:
    """Pixel size of ``source`` expressed in ``crs`` units, without resampling it."""
    if source.crs == crs:
        return abs(source.transform.a), abs(source.transform.e)
    transform, _width, _height = calculate_default_transform(
        source.crs, crs, source.width, source.height, *source.bounds)
    return abs(transform.a), abs(transform.e)


def _finer_resolution(before, after) -> tuple[float, float]:
    """Pick the finer (smaller) pixel size of the two sources so neither image loses detail."""
    before_resolution = _native_resolution_in(before.crs, before)
    after_resolution = _native_resolution_in(before.crs, after)
    return (min(before_resolution[0], after_resolution[0]),
            min(before_resolution[1], after_resolution[1]))


def _snap_to_grid(source, bounds, x_resolution, y_resolution):
    """Expand ``bounds`` outward onto ``source``'s own pixel grid when the output uses its pixel size.

    Without this the output grid sits a fraction of a pixel off the source grid, so bilinear
    resampling blurs every pixel of both images. Snapped, a same-tile Sentinel-2 pair is
    copied pixel-for-pixel. Falls back to the unsnapped bounds for rotated or coarser grids,
    or when snapping would reach past the source's edge.
    """
    t = source.transform
    if t.b or t.d or not (math.isclose(abs(t.a), x_resolution) and math.isclose(abs(t.e), y_resolution)):
        return bounds
    eps = 1e-6  # in pixels: absorbs float error so an already-aligned edge is not pushed out
    left = t.c + math.floor((bounds[0] - t.c) / x_resolution + eps) * x_resolution
    right = t.c + math.ceil((bounds[2] - t.c) / x_resolution - eps) * x_resolution
    top = t.f - math.floor((t.f - bounds[3]) / y_resolution + eps) * y_resolution
    bottom = t.f - math.ceil((t.f - bounds[1]) / y_resolution - eps) * y_resolution
    snapped = (left, bottom, right, top)
    return snapped if _contains(source.bounds, snapped) else bounds


def load_aligned_pair(before_path: str | Path, after_path: str | Path,
                      bbox_wgs84: Sequence[float], max_dimension: int = 1024):
    """Return two arrays on one grid covering exactly ``bbox_wgs84``.

    Both sources must fully cover the requested AOI. This deliberately rejects
    partial overlaps and ungeoreferenced imagery rather than comparing different places.
    """
    if len(bbox_wgs84) != 4 or bbox_wgs84[0] >= bbox_wgs84[2] or bbox_wgs84[1] >= bbox_wgs84[3]:
        raise ValueError("invalid WGS84 bbox")
    with rasterio.open(before_path) as before, rasterio.open(after_path) as after:
        if before.crs is None or after.crs is None:
            raise ValueError("both images must contain a CRS")
        target_bounds = transform_bounds("EPSG:4326", before.crs, *bbox_wgs84, densify_pts=21)
        if not _contains(before.bounds, target_bounds):
            raise ValueError("before image does not fully cover the selected AOI")
        after_bounds_in_target = transform_bounds(after.crs, before.crs, *after.bounds, densify_pts=21)
        from rasterio.coords import BoundingBox
        if not _contains(BoundingBox(*after_bounds_in_target), target_bounds):
            raise ValueError("after image does not fully cover the selected AOI")

        x_resolution, y_resolution = _finer_resolution(before, after)
        target_bounds = _snap_to_grid(before, target_bounds, x_resolution, y_resolution)
        width = max(1, int(round((target_bounds[2] - target_bounds[0]) / x_resolution)))
        height = max(1, int(round((target_bounds[3] - target_bounds[1]) / y_resolution)))
        scale = max(width / max_dimension, height / max_dimension, 1)
        width, height = max(1, int(np.ceil(width / scale))), max(1, int(np.ceil(height / scale)))
        target_transform = from_bounds(*target_bounds, width, height)

        def warp(source):
            # Assumes band order R,G,B (true for the Sentinel-2 "visual"/TCI asset this
            # pipeline downloads); a 4th alpha/mask band, if present, is dropped here.
            # All bands go through one reproject call so each remote block is fetched once.
            band_count = min(source.count, 3)
            destination = np.zeros((band_count, height, width), dtype=np.float32)
            reproject(
                rasterio.band(source, list(range(1, band_count + 1))), destination,
                src_transform=source.transform, src_crs=source.crs,
                dst_transform=target_transform, dst_crs=before.crs,
                resampling=Resampling.bilinear,
            )
            return np.moveaxis(destination, 0, -1)

        before_array, after_array = warp(before), warp(after)
        metadata = {
            "crs": str(before.crs), "transform": tuple(target_transform),
            "shape": (height, width), "bbox_wgs84": tuple(bbox_wgs84),
        }
        return before_array, after_array, metadata

