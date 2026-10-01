from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from local_pair_loader import list_pairs, load_local_pair


def write_png(path: Path, size, value):
    Image.fromarray(np.full((size[1], size[0], 3), value, dtype=np.uint8)).save(path)


def test_list_pairs_matches_common_filenames(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir(); after.mkdir()
    write_png(before / "a.png", (4, 4), 1)
    write_png(after / "a.png", (4, 4), 2)
    assert list_pairs(before, after) == ["a.png"]


def test_list_pairs_rejects_unmatched_filenames(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir(); after.mkdir()
    write_png(before / "a.png", (4, 4), 1)
    write_png(before / "b.png", (4, 4), 1)
    write_png(after / "a.png", (4, 4), 2)
    with pytest.raises(ValueError, match="only in before"):
        list_pairs(before, after)


def test_load_local_pair_rejects_size_mismatch(tmp_path):
    before_path, after_path = tmp_path / "before.png", tmp_path / "after.png"
    write_png(before_path, (10, 8), 1)
    write_png(after_path, (12, 8), 2)
    with pytest.raises(ValueError, match="size mismatch"):
        load_local_pair(before_path, after_path)


def test_load_local_pair_downscales_to_max_dimension(tmp_path):
    before_path, after_path = tmp_path / "before.png", tmp_path / "after.png"
    write_png(before_path, (100, 50), 1)
    write_png(after_path, (100, 50), 2)
    before, after = load_local_pair(before_path, after_path, max_dimension=20)
    assert before.shape == after.shape == (10, 20, 3)
