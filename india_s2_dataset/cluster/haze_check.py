#!/usr/bin/env python3
"""Rate haze/cloud in every patch-pair of a manifest with Qwen3-VL-8B on one GPU (runs on Ada).

    python haze_check.py --data trial --out trial/results_8bit.jsonl --quant 8bit

Writes one JSON line per pair as it goes (ratings, keep/reject, raw answer, tokens, seconds), and
skips pairs already in the output, so a job that hits its time limit is simply resubmitted.
2080 Ti (Turing) has no bfloat16, so compute is float16; 8-bit weights fit its 11 GB.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from haze_prompt import MAX_NEW_TOKENS, MODEL_SIDE_PX, PROMPT, PROMPT_VERSION, REJECT_AT, parse, verdict

MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def load(quant: str):
    kwargs = {"device_map": "cuda:0", "dtype": torch.float16}
    if quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    processor.tokenizer.padding_side = "left"   # batched generation needs left padding
    return model, processor


def image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((MODEL_SIDE_PX, MODEL_SIDE_PX), Image.BICUBIC)


def _messages(before: Image.Image, after: Image.Image) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "text", "text": "Image 1 (BEFORE):"}, {"type": "image", "image": before},
        {"type": "text", "text": "Image 2 (AFTER):"}, {"type": "image", "image": after},
        {"type": "text", "text": PROMPT}]}]


@torch.inference_mode()
def rate_batch(model, processor, pairs: list[tuple[Image.Image, Image.Image]]) -> list[tuple[str, int, int]]:
    """Rate several pairs in one generate call (left-padded); returns (text, tokens_in, tokens_out) each."""
    texts = [processor.apply_chat_template(_messages(b, a), tokenize=False, add_generation_prompt=True)
             for b, a in pairs]
    images = [img for pair in pairs for img in pair]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    pad = processor.tokenizer.pad_token_id
    decoded = processor.batch_decode(new, skip_special_tokens=True)
    n_in = inputs["attention_mask"].sum(dim=1).tolist()
    n_out = (new != pad).sum(dim=1).tolist()
    return list(zip(decoded, map(int, n_in), map(int, n_out)))


def rate_safe(model, processor, pairs):
    """rate_batch, halving the batch on GPU out-of-memory until it fits."""
    try:
        return rate_batch(model, processor, pairs)
    except torch.cuda.OutOfMemoryError:
        import gc
        gc.collect()
        torch.cuda.empty_cache()   # free the failed attempt's tensors before retrying smaller
        if len(pairs) == 1:
            raise
        half = len(pairs) // 2
        return rate_safe(model, processor, pairs[:half]) + rate_safe(model, processor, pairs[half:])


def rate_dataset(model, processor, data: Path, out: Path, quant: str, batch_size: int,
                 split: str | None = None) -> int:
    """Rate every not-yet-rated pair of ``data/manifest.jsonl`` into ``out`` (resumable). Returns count."""
    rows = [json.loads(line) for line in open(data / "manifest.jsonl")]
    if split:
        rows = [r for r in rows if r["split"] == split]
    done = {json.loads(line)["id"] for line in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{data.name}: {len(rows)} pairs, {len(done)} already rated, {len(todo)} to do ({quant})", flush=True)
    done_count = 0
    with open(out, "a") as sink:
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            t0 = time.time()
            answers = rate_safe(model, processor, [(image(data / r["before"]), image(data / r["after"]))
                                                   for r in chunk])
            per_pair = (time.time() - t0) / len(chunk)
            for row, (text, n_in, n_out) in zip(chunk, answers):
                ratings = parse(text)
                record = {"id": row["id"], "split": row.get("split"), "quant": quant, "prompt": PROMPT_VERSION,
                          "ratings": ratings,
                          "verdict": verdict(ratings, REJECT_AT["8b_4bit" if quant == "4bit" else "8b"]), "raw": text,
                          "tokens_in": n_in, "tokens_out": n_out, "seconds": round(per_pair, 2)}
                sink.write(json.dumps(record) + "\n")
            sink.flush()
            done_count += len(chunk)
            if done_count % 200 < batch_size or done_count == len(todo):
                print(f"[{done_count}/{len(todo)}] {per_pair:.2f}s/pair (batch {len(chunk)})", flush=True)
    return len(todo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="folder with manifest.jsonl and patches/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quant", choices=["8bit", "4bit", "fp16"], default="8bit")
    parser.add_argument("--split", help="only rows of this split (e.g. eval)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    started = time.time()
    model, processor = load(args.quant)
    print(f"model loaded in {time.time() - started:.0f}s; GPU memory "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on {torch.cuda.get_device_name(0)}", flush=True)
    rate_dataset(model, processor, args.data, args.out, args.quant, args.batch_size, args.split)
    print(f"done in {time.time() - started:.0f}s; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
