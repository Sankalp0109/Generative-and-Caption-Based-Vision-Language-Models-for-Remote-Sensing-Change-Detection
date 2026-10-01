#!/usr/bin/env python3
"""Package all 15,610 accepted pairs as a Kaggle dataset folder: kaggle/india-s2-change-pairs/.

    python kaggle/build_full.py            # all pairs
    python kaggle/build_full.py --limit 1000   # quick trial build

Splits are by site. Sites that appear in the labelled dataset keep that dataset's split, so the two
releases never put one site in different splits; the remaining sites are split 80/10/10.
Images: one zip per split (images_<split>.zip -> <patch_id>/{before,after}.{png,tif}); Kaggle
extracts each into a folder named after the zip.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle/india-s2-change-pairs"
FINAL = ROOT / "data/FINAL"
LABELLED = ROOT / "kaggle/india-s2-change-labelled/labels.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    keep = {f: (OUT / f).read_text() for f in ("README.md", "dataset-metadata.json") if (OUT / f).exists()}
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for f, text in keep.items():                      # hand-written card and metadata survive rebuilds
        (OUT / f).write_text(text)

    rows_m = [json.loads(l) for l in open(FINAL / "manifest.jsonl")]
    acc = sorted((r for r in rows_m if r["status"] == "accepted"), key=lambda r: r["patch_id"])
    lab = {r["patch_id"]: r for r in csv.DictReader(open(LABELLED))}
    site_split = {r["site"]: r["split"] for r in lab.values()}
    others = sorted({r["site"] for r in acc} - set(site_split))
    random.Random(42).shuffle(others)
    n = len(others)
    for i, s in enumerate(others):
        site_split[s] = "train" if i < 0.8 * n else "val" if i < 0.9 * n else "test"
    if args.limit:
        acc = random.Random(1).sample(acc, args.limit)

    rows = []
    for r in acc:
        lon0, lat0, lon1, lat1 = r["bounds_wgs84"]
        rat = (r.get("model") or {}).get("ratings") or {}
        haze = max((rat.get(d, {}).get("haze", 0) for d in ("before", "after")), default="")
        L = lab.get(r["patch_id"], {})
        split = site_split[r["site"]]
        rows.append({
            "patch_id": r["patch_id"], "site": r["site"], "split": split,
            "state": r["state"], "region": r["region"],
            "lat": round((lat0 + lat1) / 2, 5), "lon": round((lon0 + lon1) / 2, 5),
            "bbox_wgs84": json.dumps([round(v, 6) for v in r["bounds_wgs84"]]),
            "before_date": r["before"]["date"], "after_date": r["after"]["date"],
            "before_scene": r["before"]["item_id"], "after_scene": r["after"]["item_id"],
            "crs": r["crs"], "tile": r["tile"],
            "pixel_changed_fraction": r["change"]["changed_fraction"],
            "cloud_shadow_fraction": r["screen"]["cloud_shadow"], "water_fraction": r["screen"]["water"],
            "snow_fraction": r["screen"]["snow"], "alignment_shift_px": r["alignment"]["shift_px"],
            "haze_rating": haze,
            "label": L.get("label", ""), "label_source": L.get("label_source", ""),
            "image_dir": f"images_{split}/{r['patch_id']}",
        })
    rows.sort(key=lambda x: (x["split"], x["patch_id"]))
    with open(OUT / "pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    folder = {r["patch_id"]: r["folder"] for r in rows_m}
    for split in ("train", "val", "test"):
        with zipfile.ZipFile(OUT / f"images_{split}.zip", "w", zipfile.ZIP_STORED) as z:
            for r in rows:
                if r["split"] == split:
                    for name in ("before.png", "after.png", "before.tif", "after.tif"):
                        z.write(FINAL / folder[r["patch_id"]] / name, f"{r['patch_id']}/{name}")

    stats = {"pairs": len(rows), "sites": len({r["site"] for r in rows}), "states": len({r["state"] for r in rows}),
             "by_split": Counter(r["split"] for r in rows), "labelled": sum(bool(r["label"]) for r in rows),
             "zip_mb": {p.name: round(p.stat().st_size / 1e6) for p in sorted(OUT.glob("images_*.zip"))}}
    sites_by = {}
    for r in rows:
        sites_by.setdefault(r["site"], set()).add(r["split"])
    stats["sites_in_two_splits"] = sum(len(v) > 1 for v in sites_by.values())
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
