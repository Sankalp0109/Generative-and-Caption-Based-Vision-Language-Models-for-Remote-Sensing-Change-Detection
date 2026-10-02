#!/usr/bin/env python3
"""Can a vision-language model see land-use change in 10 m Sentinel-2 patches?

    python -m pipeline.perception_test --models qwen/qwen3-vl-32b-instruct ... --cap 0.30

Each model classifies every pair of data/RSICC/gold100/visibility_labels.jsonl (labelled by eye)
into land_use / seasonal / none / atmospheric, once from the two images alone ("plain") and once
with a third image outlining the most-changed pixels ("overlay"). Scores accuracy and land-use
recall/precision per model and variant; logs every call's tokens and cost.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from change_visualization import joint_stretch
from pipeline.patches import apply_match, brightness_match
from vlm_captioning import OPENROUTER_URL, _resolve_api_key

LABELS = ROOT / "data/RSICC/gold100/visibility_labels.jsonl"
OUT = ROOT / "data/RSICC/perception"
CLASSES = ("land_use", "seasonal", "none", "atmospheric")
PRICES = {"qwen/qwen3-vl-8b-instruct": (0.117, 0.455), "google/gemma-3-27b-it": (0.08, 0.45),
          "qwen/qwen3-vl-32b-instruct": (0.104, 0.416), "qwen/qwen3-vl-235b-a22b-instruct": (0.21, 1.9)}

PROMPT = """You compare two Sentinel-2 true-colour satellite images (10 m pixels) of the same 2.56 km x 2.56 km area in India, taken about 6 years apart in the same season. Image 1 is BEFORE, image 2 is AFTER.{overlay}

Classify the pair into exactly one category:
- land_use: a clearly visible, lasting change in land use or structures: new buildings or settlement growth, new roads, railways or interchanges, construction sites, quarries or mines, new ponds, reservoirs, aquaculture or canals, solar farms, forest clearing, or new farmland on previously bare land.
- seasonal: only differences expected between two dates: crop fields changing colour (sown, green, harvested, ploughed), vegetation greenness, river or lake water level, sandbars.
- none: no meaningful difference.
- atmospheric: one image is washed out, hazy, cloudy or shadowed enough to hide the ground.
At 10 m a house is only 1-2 pixels, so report land_use only when a structure-like change is clearly visible. If unsure between land_use and seasonal, choose seasonal.

Reply with JSON only:
{{"category": "<land_use|seasonal|none|atmospheric>", "confidence": <0-1>, "evidence": "<one sentence: what changed and where, e.g. north-east quarter>"}}"""
OVERLAY_TEXT = (" Image 3 is the AFTER image with red outlines around the pixels whose colour changed most "
                "(after correcting overall brightness). Outlined areas are only candidates: they also include "
                "crop and water changes, so judge them yourself.")


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.moveaxis(src.read(), 0, -1)


def views(folder: Path) -> tuple[Image.Image, Image.Image, Image.Image]:
    before, after = read(folder / "before.tif"), read(folder / "after.tif")
    b8, a8 = [Image.fromarray((v * 255).astype("uint8")).resize((512, 512), Image.BICUBIC)
              for v in joint_stretch(before, after, 1, 99.5)]
    valid = np.all(before > 0, axis=2) & np.all(after > 0, axis=2)
    diff = np.abs(apply_match(after, brightness_match(before, after, valid)) - before.astype("float32")).mean(axis=2)
    mask = Image.fromarray(((diff > 40) * 255).astype("uint8")).filter(ImageFilter.MaxFilter(3)).resize((512, 512))
    edge = np.asarray(mask.filter(ImageFilter.FIND_EDGES)) > 0
    over = np.asarray(a8).copy()
    over[edge] = (255, 30, 30)
    return b8, a8, Image.fromarray(over)


def url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(model: str, variant: str, item: dict, imgs: dict, key: str, session) -> dict:
    b, a, o = imgs[item["patch_id"]]
    content = [{"type": "text", "text": "Image 1 (BEFORE):"}, {"type": "image_url", "image_url": {"url": b}},
               {"type": "text", "text": "Image 2 (AFTER):"}, {"type": "image_url", "image_url": {"url": a}}]
    if variant == "overlay":
        content += [{"type": "text", "text": "Image 3 (AFTER with changed pixels outlined in red):"},
                    {"type": "image_url", "image_url": {"url": o}}]
    content.append({"type": "text", "text": PROMPT.format(overlay=OVERLAY_TEXT if variant == "overlay" else "")})
    body, t0 = None, time.time()
    for attempt in range(4):
        try:
            r = session.post(OPENROUTER_URL, timeout=120, headers={"Authorization": f"Bearer {key}"},
                             json={"model": model, "temperature": 0, "max_tokens": 200,
                                   "messages": [{"role": "user", "content": content}]})
            r.raise_for_status()
            body = r.json()
            break
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    text = (body or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
    usage = (body or {}).get("usage", {})
    pin, pout = PRICES[model]
    category = None
    try:
        category = json.loads(text[text.find("{"):text.rfind("}") + 1]).get("category")
    except ValueError:
        pass
    return {"model": model, "variant": variant, "patch_id": item["patch_id"], "idx": item["idx"],
            "label": item["label"], "category": category if category in CLASSES else None, "raw": text,
            "tokens_in": usage.get("prompt_tokens", 0), "tokens_out": usage.get("completion_tokens", 0),
            "cost_usd": (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6,
            "seconds": round(time.time() - t0, 1)}


def score(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["category"]]
    lu_true = [r for r in ok if r["label"] == "land_use"]
    lu_pred = [r for r in ok if r["category"] == "land_use"]
    tp = sum(r["label"] == "land_use" for r in lu_pred)
    return {"n": len(rows), "unparsed": len(rows) - len(ok),
            "accuracy": sum(r["category"] == r["label"] for r in ok) / max(len(ok), 1),
            "lu_recall": tp / max(len(lu_true), 1), "lu_precision": tp / max(len(lu_pred), 1),
            "lu_found": f"{tp}/{len(lu_true)}", "lu_false": len(lu_pred) - tp,
            "atm_recall": sum(r["category"] == "atmospheric" for r in ok if r["label"] == "atmospheric")
                          / max(sum(r["label"] == "atmospheric" for r in ok), 1),
            "cost": sum(r["cost_usd"] for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(PRICES))
    parser.add_argument("--variants", nargs="+", default=["plain", "overlay"])
    parser.add_argument("--cap", type=float, default=0.30)
    args = parser.parse_args()
    items = [json.loads(line) for line in open(LABELS)]
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = {}
    for it in items:
        b, a, o = views(ROOT / "data/FINAL" / it["folder"])
        imgs[it["patch_id"]] = (url(b), url(a), url(o))
        if it["idx"] in (8, 24, 36, 44):
            o.save(OUT / f"overlay_{it['patch_id']}.png")
    key, session = _resolve_api_key(None), requests.Session()
    out = OUT / "results.jsonl"
    done = {(r["model"], r["variant"], r["patch_id"]) for r in map(json.loads, open(out))} if out.exists() else set()
    spent = sum(r["cost_usd"] for r in map(json.loads, open(out))) if out.exists() else 0.0
    for model in args.models:
        for variant in args.variants:
            todo = [it for it in items if (model, variant, it["patch_id"]) not in done]
            if spent >= args.cap:
                print(f"cap ${args.cap} reached; skipping {model} {variant}")
                continue
            with ThreadPoolExecutor(6) as pool, open(out, "a") as sink:
                for r in pool.map(lambda it: ask(model, variant, it, imgs, key, session), todo):
                    sink.write(json.dumps(r) + "\n")
                    spent += r["cost_usd"]
            print(f"done {model} {variant}; spent so far ${spent:.4f}", flush=True)
    rows = [json.loads(line) for line in open(out)]
    print(f"\n{'model':34s} {'variant':8s} {'acc':>5s} {'LU found':>9s} {'LU prec':>8s} {'LU false':>9s} {'haze rec':>9s} {'cost':>8s}")
    for model in args.models:
        for variant in args.variants:
            sub = [r for r in rows if r["model"] == model and r["variant"] == variant]
            if not sub:
                continue
            s = score(sub)
            print(f"{model:34s} {variant:8s} {s['accuracy']:5.0%} {s['lu_found']:>9s} {s['lu_precision']:8.0%} "
                  f"{s['lu_false']:9d} {s['atm_recall']:9.0%} ${s['cost']:.4f}" + (f"  unparsed {s['unparsed']}" if s['unparsed'] else ""))
    print(f"total spent ${sum(r['cost_usd'] for r in rows):.4f}")


if __name__ == "__main__":
    main()
