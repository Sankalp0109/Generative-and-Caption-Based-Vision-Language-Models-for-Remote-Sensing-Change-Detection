#!/usr/bin/env python3
"""Do change-region boxes push the model into reporting change? Four prompt variants, same pairs.

    python -m pipeline.prompt_bias_test --cap 0.40

Variants (Qwen3-VL-8B via the API):
  A plain    before + after only, neutral 4-class prompt (none is the default answer)
  B boxes    same images with the top-3 pixel-change regions boxed, neutral per-box prompt
  C crops    B plus zoomed before/after crops of each box
  D forcing  boxes + the old style: "change was detected here, describe it" (known-bad reference)
  E cnn      as B, but boxes from the ResNet-50 feature-change map (tolerant of 1-2 px drift)
  F cnn+crop as C, with the CNN boxes
Pairs: Claude-labelled land_use / seasonal / none pairs, plus identical-image controls (after =
before, but still boxed where the real pair changed), so any change reported there is invented.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from change_visualization import prepare_visual_evidence
from pipeline.change_gate import CNN, read, stretch_pair
from pipeline.vlm_triage import CLASSIFY, url
from vlm_captioning import OPENROUTER_URL, _resolve_api_key

FINAL = ROOT / "data/FINAL"
LABELS = ROOT / "data/RSICC/claude_labels/labels.jsonl"
OUT = ROOT / "data/RSICC/prompt_bias"
MODEL, PRICE = "qwen/qwen3-vl-8b-instruct", (0.117, 0.455)
SIDE, N_BOX = 512, 3
COLOURS = [(255, 40, 40), (40, 160, 255), (255, 200, 0)]

HEAD = ("These are two Sentinel-2 true-colour satellite images (10 m pixels) of the same 2.56 km x 2.56 km area in India, "
        "taken about 6 years apart in the same season. Image 1 is BEFORE, image 2 is AFTER.\n")
CATS = """Categories:
- none: no meaningful difference.
- seasonal: only crop fields (sown, green, harvested, ploughed), vegetation greenness, or water level differ.
- land_use: a clearly visible, lasting change: new or expanded buildings or settlement, roads, construction sites, quarries or mines, new ponds or reservoirs, cleared forest, solar farms.
- haze: one image is too hazy or cloudy to compare.
At 10 m a house is only 1-2 pixels, so choose land_use only if a structure-like change is clearly visible."""

NEUTRAL = HEAD + """Numbered coloured boxes (1-{n}) mark places where the pixel colours differ most. Most such differences are only crops, water level or lighting, so a box is NOT evidence of change. {crops}
For each box say what is there BEFORE and AFTER, then choose its category ("none" is the most common answer). Then give the overall category for the whole pair, which can also come from a clear change outside the boxes.
""" + CATS + """
Reply with JSON only:
{{"boxes": [{{"box": <n>, "before": "<few words>", "after": "<few words>", "category": "<none|seasonal|land_use|haze>"}}],
 "overall": "<none|seasonal|land_use|haze>"}}"""

FORCING = HEAD + """Automated change detection has found changes inside the numbered coloured boxes (1-{n}). Trust these detections.
Describe the change in each box, then give the overall category of change for the pair.
""" + CATS + """
Reply with JSON only:
{{"boxes": [{{"box": <n>, "change": "<what changed>", "category": "<none|seasonal|land_use|haze>"}}],
 "overall": "<none|seasonal|land_use|haze>"}}"""


def latest_labels() -> dict[str, str]:
    out = {}
    for line in open(LABELS):
        r = json.loads(line)
        out[r["patch_id"]] = r["label"]
    return out


def choose(seed: int, n_each: dict) -> list[dict]:
    labels = latest_labels()
    folder = {json.loads(l)["patch_id"]: json.loads(l)["folder"] for l in open(FINAL / "manifest.jsonl")}
    rng = random.Random(seed)
    items = []
    for lab, n in n_each.items():
        ids = sorted(p for p, v in labels.items() if v == lab and p in folder)
        items += [{"patch_id": p, "folder": folder[p], "label": lab, "control": False} for p in rng.sample(ids, min(n, len(ids)))]
    donors = [it for it in items if it["label"] in ("seasonal", "none")]
    for it in rng.sample(donors, 15):                # identical images, boxed where the real pair changed
        items.append({**it, "label": "control", "control": True})
    return items


def cnn_regions(cnn: CNN, before, after, cell: int = 16) -> list[dict]:
    """Top boxes of the ResNet-50 layer3 change map: 16x16 cells of 16 px, connected cells merged."""
    import torch
    b8, a8 = stretch_pair(before, after)
    with torch.inference_mode():
        fb, fa = cnn.body(cnn._prep([b8])), cnn.body(cnn._prep([a8]))
        d = (1 - torch.nn.functional.cosine_similarity(fb, fa, dim=1))[0].numpy()
    mask = d >= max(0.2, float(np.percentile(d, 85)))
    seen, found = np.zeros_like(mask), []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack, cells = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < d.shape[0] and 0 <= nx < d.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        ys, xs = zip(*cells)
        found.append({"x": min(xs) * cell, "y": min(ys) * cell, "width": (max(xs) - min(xs) + 1) * cell,
                      "height": (max(ys) - min(ys) + 1) * cell, "score": float(sum(d[y, x] for y, x in cells))})
    return sorted(found, key=lambda r: -r["score"])[:N_BOX]


def views(item: dict, cnn: CNN):
    before, after = read(FINAL / item["folder"] / "before.tif"), read(FINAL / item["folder"] / "after.tif")
    bv, av, *_, regions = prepare_visual_evidence(before, after, max_regions=N_BOX)
    b = Image.fromarray((bv * 255).astype("uint8"))
    a = b.copy() if item["control"] else Image.fromarray((av * 255).astype("uint8"))
    return b, a, {"pixel": regions[:N_BOX], "cnn": cnn_regions(cnn, before, after)}   # controls keep the real pair's boxes


def boxed(img: Image.Image, regions) -> Image.Image:
    s = SIDE / img.width
    out = img.resize((SIDE, SIDE), Image.BICUBIC)
    d = ImageDraw.Draw(out)
    for k, r in enumerate(regions):
        x0, y0 = r["x"] * s, r["y"] * s
        d.rectangle([x0, y0, (r["x"] + r["width"]) * s, (r["y"] + r["height"]) * s], outline=COLOURS[k], width=3)
        d.rectangle([x0, y0, x0 + 16, y0 + 18], fill=COLOURS[k])
        d.text((x0 + 4, y0 + 3), str(k + 1), fill=(0, 0, 0))
    return out


def crop(img: Image.Image, r: dict, pad: int = 12) -> Image.Image:
    box = (max(0, r["x"] - pad), max(0, r["y"] - pad), min(img.width, r["x"] + r["width"] + pad),
           min(img.height, r["y"] + r["height"] + pad))
    c = img.crop(box)
    k = 256 / max(c.size)
    return c.resize((max(1, round(c.width * k)), max(1, round(c.height * k))), Image.BICUBIC)


def content(variant: str, b, a, regions) -> list[dict]:
    parts = []
    def add(text, img):
        parts.extend([{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url(img)}}])
    if variant == "A":
        add("Image 1 (BEFORE):", b.resize((SIDE, SIDE), Image.BICUBIC))
        add("Image 2 (AFTER):", a.resize((SIDE, SIDE), Image.BICUBIC))
        parts.append({"type": "text", "text": CLASSIFY})
        return parts
    add("Image 1 (BEFORE):", boxed(b, regions))
    add("Image 2 (AFTER):", boxed(a, regions))
    crops = ""
    if variant in "CF":
        for k, r in enumerate(regions):
            add(f"Box {k + 1}, BEFORE (zoomed):", crop(b, r))
            add(f"Box {k + 1}, AFTER (zoomed):", crop(a, r))
        crops = "Zoomed before/after crops of each box follow the two full images."
    prompt = FORCING if variant == "D" else NEUTRAL
    parts.append({"type": "text", "text": prompt.format(n=len(regions), crops=crops)})
    return parts


def ask(parts, key, session) -> tuple[str, dict, float]:
    body = {}
    for attempt in range(4):
        try:
            r = session.post(OPENROUTER_URL, timeout=120, headers={"Authorization": f"Bearer {key}"},
                             json={"model": MODEL, "temperature": 0, "max_tokens": 400,
                                   "messages": [{"role": "user", "content": parts}]})
            r.raise_for_status()
            body = r.json()
            break
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    u = body.get("usage", {})
    cost = (u.get("prompt_tokens", 0) * PRICE[0] + u.get("completion_tokens", 0) * PRICE[1]) / 1e6
    try:
        parsed = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except ValueError:
        parsed = {}
    return text, parsed, cost


def overall(variant: str, parsed: dict) -> str | None:
    cat = parsed.get("category") if variant == "A" else parsed.get("overall")
    return cat if cat in ("none", "seasonal", "land_use", "haze") else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--variants", default="ABCD")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "results.jsonl"
    done = {(json.loads(l)["patch_id"], json.loads(l)["control"], json.loads(l)["variant"]) for l in open(out)} if out.exists() else set()
    spent = sum(json.loads(l)["cost"] for l in open(out)) if out.exists() else 0.0
    items = choose(args.seed, {"land_use": 50, "seasonal": 35, "none": 35})
    key, session = _resolve_api_key(None), requests.Session()
    jobs = [(it, v) for it in items for v in args.variants if (it["patch_id"], it["control"], v) not in done]
    print(f"{len(items)} pairs, {len(jobs)} calls to make, ${spent:.3f} spent so far")
    cnn = CNN()
    cache = {(it["patch_id"], it["control"]): views(it, cnn) for it in items}   # CNN on CPU, before threading

    def run(job):
        it, v = job
        b, a, boxes = cache[(it["patch_id"], it["control"])]
        regions = boxes["cnn" if v in "EF" else "pixel"]
        text, parsed, cost = ask(content(v, b, a, regions), key, session)
        return {"patch_id": it["patch_id"], "label": it["label"], "control": it["control"], "variant": v,
                "answer": overall(v, parsed), "parsed": parsed, "raw": text, "n_boxes": len(regions), "cost": cost}

    with open(out, "a") as sink, ThreadPoolExecutor(8) as pool:
        for rec in pool.map(run, jobs):
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            spent += rec["cost"]
            if spent >= args.cap:
                print(f"cost cap ${args.cap} reached")
                break
    report(out)


def report(path: Path) -> None:
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)
    print(f"\nTotal cost ${sum(r['cost'] for r in rows):.3f}")
    print(f"{'variant':<12}{'land-use found':>16}{'false LU (seas+none)':>22}{'false LU controls':>19}{'unparsed':>10}")
    names = {"A": "A plain", "B": "B boxes", "C": "C crops", "D": "D forcing", "E": "E cnn", "F": "F cnn+crop"}
    for v in "ABCDEF":
        rs = by.get(v, [])
        if not rs:
            continue
        lu = [r for r in rs if r["label"] == "land_use"]
        neg = [r for r in rs if r["label"] in ("seasonal", "none")]
        ctl = [r for r in rs if r["control"]]
        f = lambda xs: f"{sum(r['answer'] == 'land_use' for r in xs)}/{len(xs)}"
        print(f"{names[v]:<12}{f(lu):>16}{f(neg):>22}{f(ctl):>19}{sum(r['answer'] is None for r in rs):>10}")


if __name__ == "__main__":
    main()
