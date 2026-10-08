"""
Stage 2: caption every before/after pair in the collected dataset using an
open-weight vision-language model (Qwen2-VL-7B-Instruct, 4-bit NF4).

Reads pairs from data_collection/india_dataset/dataset/ (pN_before.png /
pN_after.png, produced by collect_dataset.py) and writes one caption per
pair to captions.jsonl.

RESUMABLE, same pattern as collect_dataset.py:
- Every completed pair is appended to captions.jsonl immediately (flushed +
  fsynced), so a crash/preemption/Ctrl+C only loses whatever was mid-flight.
- On startup, already-captioned pair numbers are read back and skipped.
- This is a single-process, single-GPU script -- Ada's QOS caps this
  account at 1 GPU total across all running jobs (confirmed via sacctmgr),
  so there is no concurrent-job speedup to design for here; use --max-pairs
  for a small timed calibration run before committing to the full dataset.

Usage:
    python caption_dataset.py --dataset-dir ../data_collection/india_dataset/dataset \
        --out captions.jsonl --max-pairs 20   # calibration run first
    python caption_dataset.py --dataset-dir ../data_collection/india_dataset/dataset \
        --out captions.jsonl                  # full run, same command resumes if interrupted
"""
import argparse
import json
import os
import re
import threading
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

PROMPT = (
    "These are two satellite images of the same location in India, taken years apart. "
    "The first image is BEFORE, the second is AFTER. "
    "In one or two precise sentences, describe what changed between them "
    "(e.g. new buildings, roads, deforestation, agricultural change, coastal change). "
    "If nothing meaningful changed, say so plainly."
)

_progress_lock = threading.Lock()


def append_progress(out_path, record):
    line = json.dumps(record) + "\n"
    with _progress_lock:
        with open(out_path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def load_completed(out_path):
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["pair_num"])
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a truncated last line from a hard kill
    return done


def discover_pairs(dataset_dir):
    """Find every (pN_before.png, pN_after.png) pair actually present on disk."""
    pattern = re.compile(r"^p(\d+)_before\.png$")
    pairs = []
    for f in sorted(dataset_dir.iterdir()):
        m = pattern.match(f.name)
        if not m:
            continue
        n = int(m.group(1))
        before_path = f
        after_path = dataset_dir / f"p{n}_after.png"
        if after_path.exists():
            pairs.append((n, before_path, after_path))
    return sorted(pairs, key=lambda t: t[0])


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    return model, processor


def caption_pair(model, processor, before_path, after_path):
    before_img = Image.open(before_path).convert("RGB")
    after_img = Image.open(after_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": before_img},
                {"type": "image", "image": after_img},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[before_img, after_img], padding=True, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    caption = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return caption.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="folder with pN_before.png / pN_after.png")
    ap.add_argument("--out", default="captions.jsonl")
    ap.add_argument("--max-pairs", type=int, default=None, help="cap for a calibration run, e.g. 20")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_path = Path(args.out)

    pairs = discover_pairs(dataset_dir)
    completed = load_completed(out_path)
    remaining = [p for p in pairs if p[0] not in completed]
    if args.max_pairs:
        remaining = remaining[: args.max_pairs]

    print(f"pairs found: {len(pairs)} | already captioned: {len(completed)} | doing now: {len(remaining)}")
    if not remaining:
        print("nothing left to do.")
        return

    print(f"loading {MODEL_ID} in 4-bit NF4 ...")
    t0 = time.time()
    model, processor = load_model()
    print(f"model loaded in {time.time()-t0:.1f}s, peak VRAM so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    for i, (n, before_path, after_path) in enumerate(remaining, 1):
        t0 = time.time()
        try:
            caption = caption_pair(model, processor, before_path, after_path)
            record = {"pair_num": n, "caption": caption, "elapsed_s": round(time.time() - t0, 2), "ts": time.time()}
        except Exception as e:  # noqa: BLE001 -- log and move on, never let one pair kill the run
            record = {"pair_num": n, "error": f"{type(e).__name__}: {e}", "ts": time.time()}
        append_progress(out_path, record)
        if i % 10 == 0 or i == len(remaining):
            print(f"[{i}/{len(remaining)}] p{n}: {record.get('caption', record.get('error'))[:80]}")

    print(f"done. Captions in {out_path}. Re-run the same command to caption any newly added pairs or retry failures.")


if __name__ == "__main__":
    main()
