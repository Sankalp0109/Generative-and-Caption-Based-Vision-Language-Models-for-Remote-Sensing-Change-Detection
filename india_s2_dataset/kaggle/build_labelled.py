#!/usr/bin/env python3
"""Package the ~1,000 labelled pairs as a Kaggle dataset folder: kaggle/india-s2-change-labelled/.

    python kaggle/build_labelled.py

labels.csv: one row per pair. label = human label where reviewed, otherwise the automatic label;
label_source says which. Captions exist only for some automatic change labels.
Images: images.zip with images/<patch_id>/{before,after}.png and {before,after}.tif (georeferenced).
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle/india-s2-change-labelled"
FINAL = ROOT / "data/FINAL"
H = ROOT / "validation_1k"
AUTO = {"land_use": "change", "seasonal": "seasonal", "none": "no_change", "atmospheric": "cant_tell", "unusable": "cant_tell"}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    man = {json.loads(l)["patch_id"]: json.loads(l) for l in open(FINAL / "manifest.jsonl")}
    ref = [json.loads(l) for l in open(H / "reference_labels.jsonl")]
    human = json.load(open(H / "check/check_labels.json"))
    m32 = {json.loads(l)["patch_id"]: json.loads(l)["mark"] for l in open(H / "results_32b.jsonl")}
    m235 = {json.loads(l)["patch_id"]: json.loads(l)["mark"] for l in open(H / "results_235b.jsonl")}
    to_label = lambda m: AUTO.get(m, "cant_tell")

    sites = sorted({man[r["patch_id"]]["site"] for r in ref})       # new split by site: no site in two splits
    random.Random(42).shuffle(sites)
    n = len(sites)
    split_of = {s: "train" if i < 0.7 * n else "val" if i < 0.85 * n else "test" for i, s in enumerate(sites)}
    rows = []
    for r in ref:
        pid, m = r["patch_id"], man[r["patch_id"]]
        hum = human.get(pid, {}).get("label")
        label = hum or AUTO[r["label"]]
        lon0, lat0, lon1, lat1 = m["bounds_wgs84"]
        rows.append({
            "patch_id": pid, "site": m["site"], "split": split_of[m["site"]], "source_split": m.get("split") or "",
            "label": label,
            "label_source": "human" if hum else "automatic",
            "caption": (r.get("caption") or "") if not hum and label == "change" else "",
            "state": m["state"], "region": m["region"],
            "lat": round((lat0 + lat1) / 2, 5), "lon": round((lon0 + lon1) / 2, 5),
            "bbox_wgs84": json.dumps([round(v, 6) for v in m["bounds_wgs84"]]),
            "before_date": m["before"]["date"], "after_date": m["after"]["date"],
            "before_scene": m["before"]["item_id"], "after_scene": m["after"]["item_id"],
            "crs": m["crs"], "tile": m["tile"],
            "qwen3vl_32b_says": to_label(m32.get(pid)), "qwen3vl_235b_says": to_label(m235.get(pid)),
            "image_dir": f"images/{pid}",
        })
    rows.sort(key=lambda x: (x["split"], x["patch_id"]))
    with open(OUT / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with zipfile.ZipFile(OUT / "images.zip", "w", zipfile.ZIP_STORED) as z:     # PNG/TIF already compressed
        for r in rows:
            src = FINAL / man[r["patch_id"]]["folder"]
            for name in ("before.png", "after.png", "before.tif", "after.tif"):
                z.write(src / name, f"{r['patch_id']}/{name}")   # Kaggle extracts images.zip into images/

    stats = {"pairs": len(rows), "by_label": Counter(r["label"] for r in rows),
             "by_source": Counter(r["label_source"] for r in rows), "by_split": Counter(r["split"] for r in rows), "sites": len(sites),
             "human_by_label": Counter(r["label"] for r in rows if r["label_source"] == "human"),
             "states": len({r["state"] for r in rows})}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), f"\nimages.zip {(OUT / 'images.zip').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
