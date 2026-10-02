#!/usr/bin/env python3
"""Caption pairs already triaged as land-use change: what changed, where, how much (3 variants).

Single call (no two-stage): the triage step has already established that a real land-use change
is present, so the model is asked to describe it, and to ignore seasonal crop/water differences.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.perception_test import PRICES, url, views
from vlm_captioning import OPENROUTER_URL, _resolve_api_key

PROMPT = """These are Sentinel-2 true-colour satellite images (10 m pixels) of the same 2.56 km x 2.56 km area in India, about 6 years apart in the same season. Image 1 is BEFORE, image 2 is AFTER. A land-use change has been detected in this pair.

Describe ONLY the lasting land-use change: new or expanded buildings/settlement, roads, railways, interchanges, construction sites, quarries or mines, ponds, reservoirs, aquaculture, canals, solar farms, cleared forest, or new farmland on bare land. Ignore crop colour, vegetation greenness and water-level differences. At 10 m a single house is 1-2 pixels: describe areas, not individual buildings, and do not invent details you cannot see.

Reply with JSON only:
{"change_type": "<urban_growth|road|construction|mining|water_body|solar|deforestation|farmland_expansion|other>",
 "location": "<where in the image, e.g. north-east quarter, along the left edge>",
 "extent": "<small|medium|large>",
 "captions": ["<caption 1>", "<caption 2>", "<caption 3>"]}
Each caption is one plain sentence (8-25 words) stating what changed and where, worded differently from the others."""


def caption(model: str, folder: Path, key: str, session) -> dict:
    b, a, _ = views(folder)
    content = [{"type": "text", "text": "Image 1 (BEFORE):"}, {"type": "image_url", "image_url": {"url": url(b)}},
               {"type": "text", "text": "Image 2 (AFTER):"}, {"type": "image_url", "image_url": {"url": url(a)}},
               {"type": "text", "text": PROMPT}]
    for attempt in range(4):
        try:
            r = session.post(OPENROUTER_URL, timeout=120, headers={"Authorization": f"Bearer {key}"},
                             json={"model": model, "temperature": 0.3, "max_tokens": 400,
                                   "messages": [{"role": "user", "content": content}]})
            r.raise_for_status()
            body = r.json()
            break
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    text = body["choices"][0]["message"]["content"] or ""
    usage = body.get("usage", {})
    pin, pout = PRICES[model]
    try:
        parsed = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except ValueError:
        parsed = {}
    return {"model": model, **parsed, "raw": text,
            "cost_usd": (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6}


if __name__ == "__main__":
    items = [json.loads(line) for line in open(ROOT / "data/RSICC/gold100/visibility_labels.jsonl")]
    lu = [it for it in items if it["label"] == "land_use"]
    key, session = _resolve_api_key(None), requests.Session()
    out = ROOT / "data/RSICC/perception/landuse_captions.jsonl"
    with open(out, "w") as sink:
        for model in ("qwen/qwen3-vl-32b-instruct", "qwen/qwen3-vl-235b-a22b-instruct"):
            for it in lu:
                rec = {"idx": it["idx"], "patch_id": it["patch_id"], **caption(model, ROOT / "data/FINAL" / it["folder"], key, session)}
                sink.write(json.dumps(rec) + "\n")
    rows = [json.loads(line) for line in open(out)]
    print(f"{len(rows)} captions, cost ${sum(r['cost_usd'] for r in rows):.4f}")
    for it in lu:
        print(f"\n#{it['idx']} {it['patch_id']}")
        for r in rows:
            if r["idx"] == it["idx"]:
                print(f"  {r['model'].split('/')[1][:12]:12s} [{r.get('change_type')}, {r.get('location')}, {r.get('extent')}] {(r.get('captions') or ['?'])[0]}")
