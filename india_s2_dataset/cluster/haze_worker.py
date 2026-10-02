#!/usr/bin/env python3
"""Long-running Ada worker: load the model once, then screen batches as they arrive.

    python haze_worker.py --root batches --quant 4bit --batch-size 8

A batch folder is picked up when it has a READY marker and no DONE marker; its results go to
<batch>/results.jsonl (resumable) and DONE is written when every pair is rated. The worker exits
when <root>/STOP exists and nothing is left, or after --idle-exit minutes with no work.
One job for all batches avoids a queue wait and a model download per batch.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from haze_check import load, rate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--quant", default="4bit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--idle-exit", type=float, default=120, help="minutes without work before exiting")
    args = parser.parse_args()
    started = time.time()
    model, processor = load(args.quant)
    print(f"worker: model loaded in {time.time() - started:.0f}s on {torch.cuda.get_device_name(0)}", flush=True)
    idle_since = time.time()
    while True:
        ready = sorted(d for d in args.root.iterdir()
                       if d.is_dir() and (d / "READY").exists() and not (d / "DONE").exists())
        if ready:
            batch = ready[0]
            t0 = time.time()
            rate_dataset(model, processor, batch, batch / "results.jsonl", args.quant, args.batch_size)
            (batch / "DONE").write_text(f"{time.time() - t0:.0f}s\n")
            print(f"worker: {batch.name} DONE in {time.time() - t0:.0f}s", flush=True)
            idle_since = time.time()
            continue
        if (args.root / "STOP").exists() or time.time() - idle_since > args.idle_exit * 60:
            print("worker: no work left, exiting", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
