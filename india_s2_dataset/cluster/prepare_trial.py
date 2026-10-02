#!/usr/bin/env python3
"""Cut 256 px patch-pairs for the Ada haze/cloud trial and write a manifest.

    python cluster/prepare_trial.py            # -> data/trial/patches/*.png + data/trial/manifest.jsonl

Trial set (200 patch-pairs, fixed seed):
* 100 evaluation pairs: 40 from the winter Delhi sample blocks that are visibly hazy (known
  positives) + 60 random patches from the main dataset. These get reference labels by eye.
* 100 more random patches from the main dataset, for the keep/discard run.

Patches are cut from the stored 1024 px blocks (4 x 4 grid, no overlap) and brightened with the
colour-preserving stretch computed on the whole block, so every patch of a block shares one
exposure. The stored GeoTIFFs are untouched.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from change_visualization import joint_stretch

PATCH = 256
OUT = Path("data/trial")
HAZY_WINTER = [f"data/samples/delhi/pairs/delhi_sample_{i}" for i in ("04", "05", "07", "09", "10", "11", "12")]


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.moveaxis(src.read(), 0, -1)


def patches_of(site_dir: Path) -> list[tuple[Path, int, int]]:
    return [(site_dir / "pair_1", r, c) for r in range(4) for c in range(4)]


def main() -> None:
    rng = random.Random(42)
    hazy = [p for d in HAZY_WINTER for p in patches_of(Path(d))]
    main_sites = sorted(p.parent.parent for p in Path("data/sentinel2").glob("*/pair_1/pair.json"))
    main_pool = [p for d in main_sites for p in patches_of(d)]
    eval_set = rng.sample(hazy, 40) + rng.sample(main_pool, 60)
    rng.shuffle(eval_set)
    used = set(eval_set)
    extra = rng.sample([p for p in main_pool if p not in used], 100)

    (OUT / "patches").mkdir(parents=True, exist_ok=True)
    stretched: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for split, items in (("eval", eval_set), ("extra", extra)):
        for pair_dir, r, c in items:
            if pair_dir not in stretched:
                b, a = joint_stretch(read(pair_dir / "before.tif"), read(pair_dir / "after.tif"), 1, 99.5)
                stretched[pair_dir] = ((b * 255).astype("uint8"), (a * 255).astype("uint8"))
            b8, a8 = stretched[pair_dir]
            site = pair_dir.parent.name
            pid = f"{site}_r{r}c{c}"
            window = np.s_[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
            for role, img in (("before", b8), ("after", a8)):
                Image.fromarray(img[window]).save(OUT / "patches" / f"{pid}_{role}.png")
            meta = json.loads((pair_dir / "pair.json").read_text())
            rows.append({"id": pid, "split": split, "site": site, "row": r, "col": c,
                         "source": "delhi_winter_sample" if "samples" in str(pair_dir) else "dataset",
                         "state": (meta.get("site") or {}).get("state") or "Delhi",
                         "before_date": meta["before"]["datetime"][:10], "after_date": meta["after"]["datetime"][:10],
                         "before": f"patches/{pid}_before.png", "after": f"patches/{pid}_after.png"})
    with open(OUT / "manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} patch-pairs ({sum(r['split'] == 'eval' for r in rows)} eval) -> {OUT}/manifest.jsonl")


if __name__ == "__main__":
    main()
