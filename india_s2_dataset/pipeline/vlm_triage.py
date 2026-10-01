#!/usr/bin/env python3
"""VLM step for pairs that passed the change gate: neutral classification, then captions for confirmed change.

    python -m pipeline.vlm_triage --n 100 --cap 0.30

Why the prompts look like this (the first captions invented change):
* The old pipeline always proposed changed regions and asked the model to describe "the change",
  so it always found one. Here the model gets only the two images, no hints, and "none" is the
  first and default answer.
* "Seasonal" (crops, greenness, water level) is its own answer, so it is not reported as land use.
* The model must name WHERE the change is, from a fixed list of 9 zones. That zone must actually
  contain changed pixels (brightness-matched colour change plus new edges); otherwise the claim is
  treated as invented.
* A pair is land-use change only if Qwen3-VL-32B (confidence >= 0.9) and Qwen3-VL-8B both say so
  and the location check passes (validated on 60 eye-labelled pairs: 11/12 found, 1/48 false).
* Captions are written only for confirmed pairs, with a prompt that states what was confirmed
  and forbids inventing details; everything else gets a fixed caption.
* Controls: 10 pairs where the before and after images are identical. Any change reported there
  is invented, which measures the false-alarm rate directly.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from change_visualization import joint_stretch
from pipeline.change_gate import read
from pipeline.patches import apply_match, brightness_match
from vlm_captioning import OPENROUTER_URL, _resolve_api_key

FINAL = ROOT / "data/FINAL"
OUT = ROOT / "data/RSICC/triage100"
M32, M8 = "qwen/qwen3-vl-32b-instruct", "qwen/qwen3-vl-8b-instruct"
PRICE = {M32: (0.104, 0.416), M8: (0.117, 0.455)}     # $ per 1M tokens in/out, checked 29 Sep 2026
ZONES = ["north-west", "north", "north-east", "west", "centre", "east", "south-west", "south", "south-east"]

CLASSIFY = """These are two Sentinel-2 true-colour satellite images (10 m pixels) of the same 2.56 km x 2.56 km area in India, taken about 6 years apart in the same season. Image 1 is BEFORE, image 2 is AFTER.

Choose the ONE category that best describes the difference between them:
- none: no meaningful difference. This is the most common answer.
- seasonal: only differences expected between two dates: crop fields sown, green, harvested or ploughed; vegetation greenness; river or lake water level; sandbars.
- land_use: a clearly visible, lasting change: new or expanded buildings or settlement, new roads, railways or interchanges, construction sites, quarries or mines, new ponds, reservoirs or canals, solar farms, cleared forest, new farmland on bare land.
- haze: one image is too hazy, cloudy or washed out to compare.

At 10 m a house is only 1-2 pixels, so choose land_use only if a structure-like change is clearly visible. If unsure, do not choose land_use.
If you choose land_use, give the zone where it is: one of north-west, north, north-east, west, centre, east, south-west, south, south-east. Otherwise the zone is "none".

Reply with JSON only:
{"category": "<none|seasonal|land_use|haze>", "confidence": <0.0-1.0>, "zone": "<zone or none>", "evidence": "<one short sentence>"}"""

CAPTION = """These two Sentinel-2 true-colour images (10 m pixels) show the same 2.56 km x 2.56 km area in India, about 6 years apart. Image 1 is BEFORE, image 2 is AFTER.
A land-use change has been confirmed in the {zone} part of the image ({evidence}).

Write 3 different one-sentence captions (8-25 words each) describing that land-use change: what changed and where. Describe only what is visible; ignore crop colour and water level; do not invent small details (a house is 1-2 pixels).
Reply with JSON only: {{"captions": ["<caption 1>", "<caption 2>", "<caption 3>"]}}"""

FIXED = {"none": ["there is no change", "no significant change is visible between the two images",
                  "the scene remains the same"],
         "seasonal": ["there is no land-use change, only seasonal differences in the fields",
                      "the fields change colour with the season but nothing new is built",
                      "only crop and vegetation differences are visible, with no lasting change"],
         "haze": ["one of the images is too hazy to compare"]}


def views(folder: Path):
    before, after = read(folder / "before.tif"), read(folder / "after.tif")
    b8, a8 = [Image.fromarray((v * 255).astype("uint8")).resize((512, 512), Image.BICUBIC)
              for v in joint_stretch(before, after, 1, 99.5)]
    valid = np.all(before > 0, axis=2) & np.all(after > 0, axis=2)
    diff = np.abs(apply_match(after, brightness_match(before, after, valid)) - before).mean(axis=2)
    zone_change = {}
    for k, z in enumerate(ZONES):                    # share of clearly changed pixels per 3x3 zone
        r, c = divmod(k, 3)
        cell = diff[r * 256 // 3:(r + 1) * 256 // 3, c * 256 // 3:(c + 1) * 256 // 3]
        zone_change[z] = float((cell > 40).mean())
    return b8, a8, zone_change


def url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def call(model: str, b: str, a: str, prompt: str, key: str, session, max_tokens: int = 200) -> dict:
    content = [{"type": "text", "text": "Image 1 (BEFORE):"}, {"type": "image_url", "image_url": {"url": b}},
               {"type": "text", "text": "Image 2 (AFTER):"}, {"type": "image_url", "image_url": {"url": a}},
               {"type": "text", "text": prompt}]
    body = {}
    for attempt in range(4):
        try:
            r = session.post(OPENROUTER_URL, timeout=120, headers={"Authorization": f"Bearer {key}"},
                             json={"model": model, "temperature": 0, "max_tokens": max_tokens,
                                   "messages": [{"role": "user", "content": content}]})
            r.raise_for_status()
            body = r.json()
            break
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    usage = body.get("usage", {})
    pin, pout = PRICE[model]
    try:
        parsed = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except ValueError:
        parsed = {}
    return {"parsed": parsed, "raw": text,
            "cost": (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6}


def choose(n: int) -> list[dict]:
    gate = [json.loads(l) for l in open(ROOT / "data/RSICC/gate/gate_scores.jsonl")]
    labels = {}
    for line in open(ROOT / "data/RSICC/claude_labels/labels.jsonl"):
        r = json.loads(line)
        labels[r["patch_id"]] = r["label"]
    passed = [g for g in gate if g["pass"]]
    rng = random.Random(30)
    lab = [g for g in passed if g["patch_id"] in labels]
    unl = [g for g in passed if g["patch_id"] not in labels]
    n_real = n - 10
    picks = rng.sample(lab, min(len(lab), n_real * 2 // 3))
    picks += rng.sample(unl, min(len(unl), n_real - len(picks)))
    folders = {json.loads(l)["patch_id"]: json.loads(l)["folder"] for l in open(FINAL / "manifest.jsonl")}
    items = [{"patch_id": g["patch_id"], "folder": folders[g["patch_id"]], "gate_score": g["score"],
              "my_label": labels.get(g["patch_id"]), "control": False} for g in picks]
    for g in rng.sample(passed, 10):                 # identical-image controls
        items.append({"patch_id": g["patch_id"], "folder": folders[g["patch_id"]], "gate_score": g["score"],
                      "my_label": "none", "control": True})
    rng.shuffle(items)
    return items


def run(item: dict, key: str, session) -> dict:
    b8, a8, zones = views(FINAL / item["folder"])
    if item["control"]:
        a8 = b8                                       # same image twice: nothing can have changed
        zones = {z: 0.0 for z in zones}
    bu, au = url(b8), url(a8)
    r32 = call(M32, bu, au, CLASSIFY, key, session)
    r8 = call(M8, bu, au, CLASSIFY, key, session)
    p32, p8 = r32["parsed"], r8["parsed"]
    zone = str(p32.get("zone", "none")).lower()
    zone_ok = zones.get(zone, 0.0) >= 0.02
    confirmed = (p32.get("category") == "land_use" and float(p32.get("confidence") or 0) >= 0.9
                 and p8.get("category") == "land_use" and zone_ok)
    if confirmed:
        cap = call(M32, bu, au, CAPTION.format(zone=zone, evidence=p32.get("evidence", "")), key, session, 300)
        captions, final = cap["parsed"].get("captions") or [], "land_use"
        cost_cap = cap["cost"]
    else:
        cat = p32.get("category") if p32.get("category") in FIXED else "none"
        if p32.get("category") == "land_use":      # claimed but not confirmed: treat as no land-use change
            cat = "seasonal" if p8.get("category") == "seasonal" else "none"
        captions, final, cost_cap = FIXED[cat], cat, 0.0
    return {**item, "m32": p32, "m8": p8, "zone_change": zones, "zone_ok": zone_ok, "final": final,
            "captions": captions, "cost": r32["cost"] + r8["cost"] + cost_cap}


def page(rows: list[dict]) -> None:
    name = {"land_use": "Land-use change", "seasonal": "Seasonal", "none": "No change", "haze": "Haze",
            "atmospheric": "Haze", None: "–"}
    cards = []
    for r in rows:
        f = "../../FINAL/" + r["folder"]
        after = f + ("/before.png" if r["control"] else "/after.png")
        agree = "" if r["my_label"] is None else ("✓" if (r["final"] == "land_use") == (r["my_label"] == "land_use") else "✗")
        cards.append(f"""<section data-final="{r['final']}" data-control="{int(r['control'])}">
<h2>{html.escape(r['patch_id'])} {'<span class="ctl">CONTROL: identical images</span>' if r['control'] else ''}</h2>
<div class="pair"><figure><img loading="lazy" src="{f}/before.png" alt="before"><figcaption>Before</figcaption></figure>
<figure><img loading="lazy" src="{after}" alt="after"><figcaption>After</figcaption></figure></div>
<p><b>Final: {name.get(r['final'])}</b> · my label: {name.get(r['my_label'])} {agree} ·
32B: {html.escape(str(r['m32'].get('category')))} ({r['m32'].get('confidence')}, zone {html.escape(str(r['m32'].get('zone')))}) ·
8B: {html.escape(str(r['m8'].get('category')))} · location check: {'passed' if r['zone_ok'] else 'failed'}</p>
<p class="muted">32B evidence: {html.escape(str(r['m32'].get('evidence', '')))}</p>
<ol>{''.join(f'<li>{html.escape(str(c))}</li>' for c in r['captions'])}</ol></section>""")
    total = sum(r["cost"] for r in rows)
    ctl = [r for r in rows if r["control"]]
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>API Triage 100</title>
<style>:root{{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e2e0da}}@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#141416;--panel:#1e1e21;--ink:#ececef;--muted:#9a9aa2;--line:#2e2e33}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}} main{{max-width:1050px;margin:auto;padding:16px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}} h2{{font-size:14px;margin:0 0 8px}}
.pair{{display:flex;gap:8px;flex-wrap:wrap}} figure{{margin:0;flex:1 1 300px}} img{{width:100%;border-radius:4px}} figcaption,.muted{{color:var(--muted);font-size:13px}}
.ctl{{color:#b5651d}} select{{font:inherit;padding:4px 8px}}</style></head><body><main>
<h1 style="font-size:20px">API run: 100 pairs that passed the free change check</h1>
<p class="muted">Neutral 4-class prompt (Qwen3-VL-32B + 8B), location check, captions only for confirmed change. Cost ${total:.4f}.
Controls (identical images) reported as land-use: {sum(r['final'] == 'land_use' for r in ctl)} of {len(ctl)}.</p>
<p>Show: <select id="s"><option value="">All</option><option value="land_use">Land-use change</option><option value="seasonal">Seasonal</option><option value="none">No change</option><option value="haze">Haze</option><option value="c">Controls</option></select></p>
{''.join(cards)}<script>const S=[...document.querySelectorAll("section")];document.getElementById("s").onchange=e=>{{const v=e.target.value;S.forEach(x=>x.style.display=(!v||(v==="c"?x.dataset.control==="1":x.dataset.final===v))?"":"none")}};</script></main></body></html>"""
    (OUT / "results.html").write_text(doc, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--cap", type=float, default=0.30)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = choose(args.n)
    key, session = _resolve_api_key(None), requests.Session()
    out = OUT / "results.jsonl"
    rows, spent = [], 0.0
    with ThreadPoolExecutor(6) as pool, open(out, "w") as sink:
        for r in pool.map(lambda it: run(it, key, session), items):
            rows.append(r)
            spent += r["cost"]
            sink.write(json.dumps(r) + "\n")
            if spent >= args.cap:
                print(f"cost cap ${args.cap} reached")
                break
    page(rows)
    real = [r for r in rows if not r["control"] and r["my_label"]]
    tp = sum(r["final"] == "land_use" and r["my_label"] == "land_use" for r in real)
    fn = sum(r["final"] != "land_use" and r["my_label"] == "land_use" for r in real)
    fp = sum(r["final"] == "land_use" and r["my_label"] != "land_use" for r in real)
    ctl = [r for r in rows if r["control"]]
    from collections import Counter
    print(f"{len(rows)} pairs, cost ${spent:.4f}; final: {dict(Counter(r['final'] for r in rows))}")
    print(f"vs my labels ({len(real)} labelled): land-use found {tp}/{tp + fn}, false land-use {fp}")
    print(f"identical-image controls reported as land-use: {sum(r['final'] == 'land_use' for r in ctl)}/{len(ctl)}; "
          f"32B said anything but none: {sum(r['m32'].get('category') != 'none' for r in ctl)}/{len(ctl)}")


if __name__ == "__main__":
    main()
