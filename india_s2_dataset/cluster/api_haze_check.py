#!/usr/bin/env python3
"""Same haze/cloud question as the Ada job, asked of the hosted full-precision model (reference).

    python cluster/api_haze_check.py --data data/trial --out data/trial/results_api.jsonl --split eval

Uses the identical prompt, image size and keep/reject rule (``haze_prompt``), so any difference
from the Ada results comes from the model precision/serving, not from how it was asked.
Resumable: pairs already in the output are skipped. Token usage and cost are recorded per pair.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from haze_prompt import MAX_NEW_TOKENS, MODEL_SIDE_PX, PROMPT, PROMPT_VERSION, REJECT_AT, parse, verdict
from vlm_captioning import OPENROUTER_URL, _resolve_api_key

MODEL = "qwen/qwen3-vl-8b-instruct"
PRICES = {  # $ per 1M tokens (in, out), OpenRouter, checked 28 Sep 2026
    "qwen/qwen3-vl-8b-instruct": (0.117, 0.455),
    "qwen/qwen3-vl-32b-instruct": (0.104, 0.416),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.21, 1.9),
    "qwen/qwen3-vl-8b-thinking": (0.18, 2.1),
}


def data_url(path: Path) -> str:
    image = Image.open(path).convert("RGB").resize((MODEL_SIDE_PX, MODEL_SIDE_PX), Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def ask(row: dict, data: Path, key: str, session: requests.Session, model: str = MODEL) -> dict:
    content = [{"type": "text", "text": "Image 1 (BEFORE):"},
               {"type": "image_url", "image_url": {"url": data_url(data / row["before"])}},
               {"type": "text", "text": "Image 2 (AFTER):"},
               {"type": "image_url", "image_url": {"url": data_url(data / row["after"])}},
               {"type": "text", "text": PROMPT}]
    t0 = time.time()
    for attempt in range(4):
        try:
            response = session.post(OPENROUTER_URL, timeout=120, headers={"Authorization": f"Bearer {key}"},
                                    json={"model": model, "temperature": 0,
                                          "max_tokens": 4000 if "thinking" in model else MAX_NEW_TOKENS,
                                          "messages": [{"role": "user", "content": content}]})
            response.raise_for_status()
            body = response.json()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    text = body["choices"][0]["message"]["content"] or ""
    usage = body.get("usage", {})
    ratings = parse(text)
    price_in, price_out = (p / 1e6 for p in PRICES.get(model, (0, 0)))
    return {"id": row["id"], "split": row["split"], "quant": "api", "model": model, "prompt": PROMPT_VERSION, "ratings": ratings,
            "verdict": verdict(ratings, REJECT_AT["235b"] if "235b" in model else REJECT_AT["8b"]),
            "raw": text, "tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens"),
            "cost_usd": usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out,
            "provider": body.get("provider"), "seconds": round(time.time() - t0, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model", default=MODEL, choices=sorted(PRICES))
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.data / "manifest.jsonl")]
    rows = [r for r in rows if not args.split or r["split"] == args.split]
    done = {json.loads(line)["id"] for line in open(args.out)} if args.out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    key, session = _resolve_api_key(None), requests.Session()
    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "a") as sink:
        for n, record in enumerate(pool.map(lambda r: ask(r, args.data, key, session, args.model), todo), 1):
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}]", flush=True)
    records = [json.loads(line) for line in open(args.out)]
    print(f"{len(records)} rated, total cost ${sum(r.get('cost_usd') or 0 for r in records):.4f}, "
          f"avg tokens in {sum(r['tokens_in'] or 0 for r in records) / len(records):.0f}")


if __name__ == "__main__":
    main()
