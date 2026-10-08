#!/usr/bin/env python3
"""Ask Qwen3-VL-32B and -235B the plain question about every reference pair, then mark each answer.

    python validation_1k/run_api.py --cap 1.5

Vision call: before + after (512 px, joint stretch) and one plain question, no format, no hints.
Marking call (text only, 32B): what does the answer conclude? land_use / seasonal / none / unclear.
Resumable: pairs already in results_<model>.jsonl are skipped; stops at the cost cap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.vlm_triage import views, url
from vlm_captioning import OPENROUTER_URL, _resolve_api_key

HERE = ROOT / "validation_1k"
FINAL = ROOT / "data/FINAL"
MODELS = {"32b": ("qwen/qwen3-vl-32b-instruct", (0.104, 0.416)),
          "235b": ("qwen/qwen3-vl-235b-a22b-instruct", (0.21, 1.9))}
QUESTION = ("Before and after images of the same place, 6 years apart. Has anything lasting changed "
            "(like buildings, roads, ponds, cleared land), or are the differences only seasonal?")
MARK = ("Below is someone's answer about two satellite images. Classify what the answer concludes. "
        "Reply with one word only: land_use (it says something lasting changed), seasonal (only seasonal differences), "
        "none (nothing changed), or unclear.\n\nAnswer:\n")


def call(session, key, model, price, content, max_tokens):
    body = {}
    for attempt in range(5):
        try:
            r = session.post(OPENROUTER_URL, timeout=240, headers={"Authorization": f"Bearer {key}"},
                             json={"model": model, "temperature": 0, "max_tokens": max_tokens,
                                   "messages": [{"role": "user", "content": content}]})
            r.raise_for_status()
            body = r.json()
            if body.get("choices"):
                break
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (attempt + 1))
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    u = body.get("usage", {})
    return text, (u.get("prompt_tokens", 0) * price[0] + u.get("completion_tokens", 0) * price[1]) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=1.5)
    ap.add_argument("--models", default="32b,235b")
    args = ap.parse_args()
    ref = [json.loads(l) for l in open(HERE / "reference_labels.jsonl")]
    folder = {json.loads(l)["patch_id"]: json.loads(l)["folder"] for l in open(FINAL / "manifest.jsonl")}
    key, session = _resolve_api_key(None), requests.Session()
    spent = sum(json.loads(l)["cost"] for m in MODELS for p in [HERE / f"results_{m}.jsonl"] if p.exists() for l in open(p))
    for m in args.models.split(","):
        name, price = MODELS[m]
        out = HERE / f"results_{m}.jsonl"
        done = {json.loads(l)["patch_id"] for l in open(out)} if out.exists() else set()
        todo = [r for r in ref if r["patch_id"] not in done]
        print(f"{m}: {len(todo)} to do, ${spent:.3f} spent so far", flush=True)

        def run(r):
            b, a, _ = views(FINAL / folder[r["patch_id"]])
            ans, c1 = call(session, key, name, price,
                           [{"type": "text", "text": "Image 1 (BEFORE):"}, {"type": "image_url", "image_url": {"url": url(b)}},
                            {"type": "text", "text": "Image 2 (AFTER):"}, {"type": "image_url", "image_url": {"url": url(a)}},
                            {"type": "text", "text": QUESTION}], 600)
            mark, c2 = call(session, key, MODELS["32b"][0], MODELS["32b"][1], MARK + ans, 5) if ans else ("", 0.0)
            m_ = mark.strip().lower().strip(".").strip()
            return {"patch_id": r["patch_id"], "model": name, "answer": ans,
                    "mark": m_ if m_ in ("land_use", "seasonal", "none", "unclear") else "unparsed", "cost": c1 + c2}

        with open(out, "a") as sink, ThreadPoolExecutor(8) as pool:
            for n, rec in enumerate(pool.map(run, todo), 1):
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
                spent += rec["cost"]
                if n % 100 == 0:
                    print(f"  {m} {n}/{len(todo)}  ${spent:.3f}", flush=True)
                if spent >= args.cap:
                    print(f"cost cap ${args.cap} reached", flush=True)
                    pool.shutdown(cancel_futures=True)
                    return


if __name__ == "__main__":
    main()
