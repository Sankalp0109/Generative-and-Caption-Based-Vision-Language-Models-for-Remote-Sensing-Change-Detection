#!/usr/bin/env python3
"""Build a labelling set as a zip: labeler/images/ + labeler/pairs.js (+ pairs.json).

    python labeler/make_trial.py --name set1_dev --n 150            # -> labeler_set1_dev_150.zip
    python labeler/make_trial.py --name set2_locked --n 250

One patch per site. Sites already issued in an earlier set (labeler/issued.json) or used in earlier
experiments are skipped, so every set is site-disjoint from the others. About 2/3 of each set is a
random draw spread by region; the rest are high change-gate scores, so real changes are present.
The stratum is recorded in pairs.json but never shown on the page.
"""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "labeler"
FINAL = ROOT / "data/FINAL"
ISSUED = HERE / "issued.json"
USED = [ROOT / "data/RSICC/gold100/pairs.jsonl", ROOT / "data/RSICC/gold100/visibility_labels.jsonl",
        ROOT / "data/RSICC/triage100/results.jsonl"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="set name, e.g. set1_dev")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    issued = json.loads(ISSUED.read_text()) if ISSUED.exists() else {}
    if args.name in issued:
        raise SystemExit(f"set {args.name} was already issued")
    rows = [json.loads(line) for line in open(FINAL / "manifest.jsonl")]
    site_of = {r["patch_id"]: r["site"] for r in rows}
    used = {site_of[json.loads(line)["patch_id"]] for p in USED if p.exists() for line in open(p)}
    used |= {s for info in issued.values() for s in info["sites"]}
    gate = {json.loads(l)["patch_id"]: json.loads(l)["score"] for l in open(ROOT / "data/RSICC/gate/gate_scores.jsonl")}

    rng = random.Random(args.seed)
    by_site = defaultdict(list)
    for r in rows:
        if r["status"] == "accepted" and r["site"] not in used:
            by_site[r["site"]].append(r)
    one = [rng.choice(v) for v in by_site.values()]              # one random patch per free site
    n_hard = args.n // 3
    # random stratum: spread over regions in proportion to their share of sites
    by_region = defaultdict(list)
    for r in one:
        by_region[r["region"]].append(r)
    n_rand, picked = args.n - n_hard, []
    for reg, pool in sorted(by_region.items()):
        rng.shuffle(pool)
        picked += [(r, "random") for r in pool[:round(n_rand * len(pool) / len(one))]]
    picked = picked[:n_rand]
    taken = {r["site"] for r, _ in picked}
    # hard stratum: the highest-scoring patch of each remaining site, top scores first
    best = sorted(((max(v, key=lambda r: gate.get(r["patch_id"], 0)), s) for s, v in by_site.items() if s not in taken),
                  key=lambda t: -gate.get(t[0]["patch_id"], 0))
    picked += [(r, "high_gate") for r, _ in best[:args.n - len(picked)]]
    rng.shuffle(picked)

    zpath = ROOT / f"labeler_{args.name}_{len(picked)}.zip"
    pairs = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:     # PNGs are already compressed
        for i, (r, stratum) in enumerate(picked):
            for name in ("before", "after"):
                z.write(FINAL / r["folder"] / f"{name}.png", f"labeler/images/{r['patch_id']}_{name}.png")
            lon0, lat0, lon1, lat1 = r["bounds_wgs84"]
            pairs.append({"idx": i, "patch_id": r["patch_id"], "set": args.name, "stratum": stratum,
                          "state": r["state"], "before_date": r["before"]["date"], "after_date": r["after"]["date"],
                          "before": f"images/{r['patch_id']}_before.png", "after": f"images/{r['patch_id']}_after.png",
                          "lat": round((lat0 + lat1) / 2, 5), "lon": round((lon0 + lon1) / 2, 5)})
        public = [{k: v for k, v in p.items() if k != "stratum"} for p in pairs]   # page never sees the stratum
        z.writestr("labeler/pairs.js", "const PAIRS = " + json.dumps(public) + ";\n")
        z.writestr("labeler/pairs.json", json.dumps(public, indent=1))
    (HERE / f"pairs_{args.name}.json").write_text(json.dumps(pairs, indent=1))   # private copy with strata
    issued[args.name] = {"n": len(pairs), "zip": zpath.name, "sites": sorted({r["site"] for r, _ in picked})}
    ISSUED.write_text(json.dumps(issued, indent=1))
    strata = {s: sum(1 for _, t in picked if t == s) for s in ("random", "high_gate")}
    print(f"{len(pairs)} pairs {strata} -> {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
