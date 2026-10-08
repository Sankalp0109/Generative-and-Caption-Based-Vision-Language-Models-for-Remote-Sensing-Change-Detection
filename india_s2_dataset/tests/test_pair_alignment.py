from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from pair_alignment import load_aligned_pair


def write_raster(path: Path, bounds, value):
    data = np.full((3, 20, 20), value, dtype=np.uint16)
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=3,
                       dtype=data.dtype, crs="EPSG:4326", transform=from_bounds(*bounds, 20, 20)) as dst:
        dst.write(data)


def test_pair_is_forced_onto_identical_aoi_grid(tmp_path):
    before, after = tmp_path / "before.tif", tmp_path / "after.tif"
    write_raster(before, (77, 8, 79, 10), 10)
    write_raster(after, (76, 7, 80, 11), 20)
    first, second, metadata = load_aligned_pair(before, after, (77.5, 8.5, 78.5, 9.5))
    assert first.shape == second.shape == (10, 10, 3)
    assert metadata["bbox_wgs84"] == (77.5, 8.5, 78.5, 9.5)


def test_pair_uses_finer_of_the_two_resolutions(tmp_path):
    before, after = tmp_path / "before.tif", tmp_path / "after.tif"
    # before: 2 degrees over 20px -> 0.1 deg/px (coarse).
    write_raster(before, (77, 8, 79, 10), 10)
    # after: 4 degrees over 80px -> 0.05 deg/px (finer, same units/CRS as before).
    data = np.full((3, 80, 80), 20, dtype=np.uint16)
    with rasterio.open(after, "w", driver="GTiff", width=80, height=80, count=3,
                       dtype=data.dtype, crs="EPSG:4326",
                       transform=from_bounds(76, 7, 80, 11, 80, 80)) as dst:
        dst.write(data)
    first, second, _metadata = load_aligned_pair(before, after, (77.5, 8.5, 78.5, 9.5))
    # A coarse-before/fine-after pair should keep the after image's finer pixel size
    # (1deg AOI / 0.05 deg/px = 20px), not silently downsample it to match before's grid.
    assert first.shape == second.shape == (20, 20, 3)


def test_pair_rejects_different_place(tmp_path):
    before, after = tmp_path / "before.tif", tmp_path / "after.tif"
    write_raster(before, (77, 8, 79, 10), 10)
    write_raster(after, (80, 11, 82, 13), 20)
    with pytest.raises(ValueError, match="after image does not fully cover"):
        load_aligned_pair(before, after, (77.5, 8.5, 78.5, 9.5))


def test_unaligned_aoi_is_snapped_to_the_source_grid_and_copied_exactly(tmp_path):
    # A gradient makes any resampling visible: an off-grid bilinear read would average neighbours.
    before, after = tmp_path / "before.tif", tmp_path / "after.tif"
    data = (np.arange(400, dtype=np.uint16).reshape(20, 20))[None].repeat(3, axis=0)
    for path in (before, after):
        with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=3, dtype="uint16",
                           crs="EPSG:4326", transform=from_bounds(77, 8, 79, 10, 20, 20)) as dst:
            dst.write(data)
    # 0.03 deg off the 0.1 deg grid on every side
    first, second, meta = load_aligned_pair(before, after, (77.53, 8.53, 78.47, 9.47))
    assert first.shape == (10, 10, 3)                       # snapped outward: 77.5..78.5
    assert meta["transform"][2] == pytest.approx(77.5) and meta["transform"][5] == pytest.approx(9.5)
    np.testing.assert_array_equal(first[..., 0], data[0, 5:15, 5:15])  # exact source pixels
    np.testing.assert_array_equal(first, second)


def test_joint_stretch_keeps_colour_balance_and_brightness_differences():
    from change_visualization import joint_stretch
    forest = np.zeros((4, 4, 3), dtype=np.uint8)
    forest[...] = (17, 29, 11)                      # dark green, as Sentinel-2 true colour stores it
    brighter = forest.copy(); brighter[0, 0] = (60, 60, 60)
    before, after = joint_stretch(forest, brighter, 0, 100)
    r, g, b = before[1, 1]
    assert g > r > b                                 # still green, not rebalanced to grey
    assert after[0, 0].mean() > before[0, 0].mean()  # the brighter pixel stays brighter
