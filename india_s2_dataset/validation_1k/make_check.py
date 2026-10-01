#!/usr/bin/env python3
"""Build the human check set: every pair where a model disagrees with Claude on land-use change,
plus 50 random pairs where both models and Claude agree (to measure shared mistakes).

    python validation_1k/make_check.py

Writes check/images/, check/pairs.js (shuffled, no labels or model answers shown) and
work/check_manifest.json (which group each pair is in; used only by score.py).
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "validation_1k"
FINAL = ROOT / "data/FINAL"
N_AGREE = 50


def load(name: str) -> dict:
    return {json.loads(l)["patch_id"]: json.loads(l) for l in open(HERE / name)}


def main() -> None:
    ref = load("reference_labels.jsonl")
    m32, m235 = load("results_32b.jsonl"), load("results_235b.jsonl")
    manifest = {json.loads(l)["patch_id"]: json.loads(l) for l in open(FINAL / "manifest.jsonl")}
    lu = lambda x: x == "land_use"
    disagree, agree = [], []
    for p, r in ref.items():
        if p not in m32 or p not in m235:
            continue
        mine = lu(r["label"])
        (disagree if lu(m32[p]["mark"]) != mine or lu(m235[p]["mark"]) != mine else agree).append(p)
    rng = random.Random(11)
    sample = rng.sample(agree, min(N_AGREE, len(agree)))
    picked = [(p, "disagree") for p in disagree] + [(p, "agree_sample") for p in sample]
    rng.shuffle(picked)

    img = HERE / "check/images"
    if img.exists():
        shutil.rmtree(img)
    img.mkdir(parents=True)
    pairs = []
    for i, (p, _) in enumerate(picked):
        row = manifest[p]
        for n in ("before", "after"):
            shutil.copy(FINAL / row["folder"] / f"{n}.png", img / f"{p}_{n}.png")
        pairs.append({"idx": i, "patch_id": p, "before_date": row["before"]["date"], "after_date": row["after"]["date"],
                      "before": f"images/{p}_before.png", "after": f"images/{p}_after.png"})
    (HERE / "check/pairs.js").write_text("const PAIRS = " + json.dumps(pairs) + ";\n")
    (HERE / "work/check_manifest.json").write_text(json.dumps({p: g for p, g in picked}, indent=1))
    print(f"{len(disagree)} disagreements + {len(sample)} agreement samples = {len(picked)} pairs to check")


if __name__ == "__main__":
    main()
