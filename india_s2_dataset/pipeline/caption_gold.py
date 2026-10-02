#!/usr/bin/env python3
"""Caption the 100-pair gold set (test split) through the API, with live cost tracking and a cap.

    python -m pipeline.caption_gold --limit 5          # cost test: first 5 pairs, then stop
    python -m pipeline.caption_gold                    # the rest (resumes; skips done pairs)

The 100 pairs are drawn once from the held-out TEST split of data/FINAL (65 changed / 35 unchanged,
spread across regions) so they can serve as a test set for the RSICC models without leakage.
Each pair: region hints from change_visualization, then the two-stage captioner asking for 3
captions. Every call's tokens and cost are logged; the run stops if the total reaches --cap.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from change_visualization import prepare_visual_evidence
from vlm_captioning import _resolve_api_key, caption_pair_two_stage

FINAL = ROOT / "data" / "FINAL"
OUT = ROOT / "data" / "RSICC" / "gold100"
MODEL = "qwen/qwen3-vl-32b-instruct"
PRICE = {"qwen/qwen3-vl-32b-instruct": (0.104, 0.416)}   # $ per 1M tokens in/out, checked 29 Sep 2026


def choose_pairs(n: int = 100, seed: int = 7) -> list[dict]:
    path = OUT / "pairs.jsonl"
    if path.exists():
        return [json.loads(line) for line in open(path)]
    rows = [json.loads(line) for line in open(FINAL / "manifest.jsonl")]
    test = [r for r in rows if r["status"] == "accepted" and r.get("selected") and r.get("split") == "test"]
    rng = random.Random(seed)
    picked = []
    for changed, quota in ((True, round(n * 0.65)), (False, n - round(n * 0.65))):
        pool = defaultdict(list)
        for r in test:
            if r["change"]["changed"] == changed:
                pool[r["region"]].append(r)
        for region in pool:
            rng.shuffle(pool[region])
        while quota and any(pool.values()):          # round-robin over regions: even spread
            for region in sorted(pool):
                if quota and pool[region]:
                    picked.append(pool[region].pop())
                    quota -= 1
    rng.shuffle(picked)
    keep = [{"patch_id": r["patch_id"], "folder": r["folder"], "state": r["state"], "region": r["region"],
             "changed": r["change"]["changed"], "changed_fraction": r["change"]["changed_fraction"]} for r in picked]
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(k) + "\n" for k in keep))
    return keep


def load_view(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for name in ("before.tif", "after.tif"):
        with rasterio.open(folder / name) as src:
            img = np.moveaxis(src.read(), 0, -1)
        arrays.append(np.asarray(Image.fromarray(img).resize((512, 512), Image.BICUBIC), dtype="float32"))
    return arrays[0], arrays[1]


def credit(key: str) -> float | None:
    try:
        data = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"},
                            timeout=30).json()["data"]
        return data.get("limit_remaining")
    except (requests.RequestException, KeyError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="caption at most this many new pairs")
    parser.add_argument("--cap", type=float, default=0.30, help="stop when total cost reaches this ($)")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    pairs = choose_pairs()
    out = OUT / "captions_raw.jsonl"
    done = {json.loads(line)["patch_id"] for line in open(out)} if out.exists() else set()
    spent = sum(json.loads(line)["cost_usd"] for line in open(out)) if out.exists() else 0.0
    todo = [p for p in pairs if p["patch_id"] not in done][:args.limit]
    key = _resolve_api_key(None)
    price_in, price_out = (x / 1e6 for x in PRICE[args.model])
    print(f"{len(pairs)} gold pairs ({sum(p['changed'] for p in pairs)} changed); {len(done)} done; "
          f"captioning {len(todo)} with {args.model}; credit before ${credit(key)}", flush=True)
    session = requests.Session()
    with open(out, "a") as sink:
        for i, pair in enumerate(todo, 1):
            if spent >= args.cap:
                print(f"cost cap ${args.cap} reached; stopping", flush=True)
                break
            before, after, *_, regions = prepare_visual_evidence(*load_view(FINAL / pair["folder"]), max_regions=4)
            t0 = time.time()
            result = caption_pair_two_stage(before, after, regions, model=args.model, n_captions=3,
                                            max_output_tokens=700, session=session)
            usage = [result.raw_response[s].get("usage", {}) for s in ("stage1", "stage2")]
            tin = sum(u.get("prompt_tokens", 0) for u in usage)
            tout = sum(u.get("completion_tokens", 0) for u in usage)
            cost = tin * price_in + tout * price_out
            spent += cost
            sink.write(json.dumps({**pair, "model": args.model, "overall_caption": result.overall_caption,
                                   "captions": result.captions, "change_type": result.change_type,
                                   "likely_artifact": result.likely_artifact, "regions": result.regions,
                                   "tokens_in": tin, "tokens_out": tout, "cost_usd": round(cost, 6),
                                   "seconds": round(time.time() - t0, 1)}) + "\n")
            sink.flush()
            print(f"[{i}/{len(todo)}] {pair['patch_id']}: {tin} in / {tout} out = ${cost:.5f}  "
                  f"({'changed' if pair['changed'] else 'unchanged'}) -> {result.overall_caption[:90]}", flush=True)
    rows = [json.loads(line) for line in open(out)]
    per = sum(r["cost_usd"] for r in rows) / len(rows)
    print(f"\n{len(rows)} captioned, total ${sum(r['cost_usd'] for r in rows):.4f}, "
          f"avg {sum(r['tokens_in'] for r in rows) / len(rows):.0f} in / {sum(r['tokens_out'] for r in rows) / len(rows):.0f} out "
          f"= ${per:.5f}/pair; credit after ${credit(key)}")
    for label, n in (("100 gold pairs", 100), ("15,610 accepted", 15610), ("~15,000 balanced", 15000), ("~25,000 after extra collection", 25000)):
        print(f"  projected {label}: ${per * n:.2f}")


if __name__ == "__main__":
    main()
