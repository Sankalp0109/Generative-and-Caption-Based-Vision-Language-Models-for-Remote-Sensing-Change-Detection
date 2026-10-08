#!/usr/bin/env python3
"""Visual labelling of patch pairs by Claude (reading contact sheets), saved as they are made.

    python -m pipeline.claude_label make 5            # next 5 sheets of 12 unlabelled pairs from the queue
    python -m pipeline.claude_label add s0007 "0:S 1:N 2:L:new road along east edge 3:H ..."
    python -m pipeline.claude_label stats

Codes: L land-use change (+ caption), S seasonal only (crops, vegetation, water level), N no meaningful
change, H haze/cloud/shadow hides the ground in one date, X unusable (misaligned, no-data, artefact).
Queue: the 100 gold test pairs first, then the rest of the test split, then val, then train.
Labels append to data/RSICC/claude_labels/labels.jsonl (one line per pair; the latest line wins).
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data/RSICC/claude_labels"
FINAL = ROOT / "data/FINAL"
CODES = {"L": "land_use", "S": "seasonal", "N": "none", "H": "atmospheric", "X": "unusable"}
PER_SHEET, COLS, PX = 12, 3, 256


def queue() -> list[str]:
    path = DIR / "queue.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = [json.loads(line) for line in open(FINAL / "manifest.jsonl")]
    sel = [r for r in rows if r.get("selected")]
    gold = [json.loads(line)["patch_id"] for line in open(ROOT / "data/RSICC/gold100/pairs.jsonl")]
    rng = random.Random(29)
    order = list(gold)
    for split in ("test", "val", "train"):
        ids = sorted(r["patch_id"] for r in sel if r["split"] == split and r["patch_id"] not in gold)
        rng.shuffle(ids)
        order += ids
    path.write_text(json.dumps(order))
    return order


def labelled() -> dict[str, dict]:
    path = DIR / "labels.jsonl"
    out = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            out[r["patch_id"]] = r
    return out


def folders() -> dict[str, str]:
    return {json.loads(l)["patch_id"]: json.loads(l)["folder"] for l in open(FINAL / "manifest.jsonl")}


def make(n_sheets: int) -> None:
    done, pending = labelled(), set()
    for f in (DIR / "sheets").glob("s*.json"):
        sheet = json.loads(f.read_text())
        if not all(p in done for p in sheet["ids"]):
            pending.update(sheet["ids"])
    todo = [p for p in queue() if p not in done and p not in pending]
    folder = folders()
    n_existing = len(list((DIR / "sheets").glob("s*.json")))
    font = ImageFont.load_default(size=22) if hasattr(ImageFont, "load_default") else None
    for k in range(n_sheets):
        ids = todo[k * PER_SHEET:(k + 1) * PER_SHEET]
        if not ids:
            break
        sid = f"s{n_existing + k:04d}"
        rows = (len(ids) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * (2 * PX + 22) + 8, rows * (PX + 30) + 6), (18, 18, 18))
        draw = ImageDraw.Draw(sheet)
        for i, pid in enumerate(ids):
            x, y = (i % COLS) * (2 * PX + 22) + 8, (i // COLS) * (PX + 30) + 4
            draw.text((x, y), f"#{i}", fill=(255, 225, 0), font=font)
            for j, name in enumerate(("before", "after")):
                sheet.paste(Image.open(FINAL / folder[pid] / f"{name}.png"), (x + j * (PX + 4), y + 26))
        sheet.save(DIR / "sheets" / f"{sid}.jpg", quality=93)
        (DIR / "sheets" / f"{sid}.json").write_text(json.dumps({"sheet": sid, "ids": ids}))
        print(sid, len(ids))


def add(sid: str, codes: str) -> None:
    sheet = json.loads((DIR / "sheets" / f"{sid}.json").read_text())
    ids = sheet["ids"]
    entries = {}
    for token in [t.strip() for t in codes.split("|")]:
        if not token:
            continue
        idx, rest = token.split(":", 1)
        code, _, caption = rest.partition(":")
        entries[int(idx)] = (code.strip().upper(), caption.strip())
    missing = [i for i in range(len(ids)) if i not in entries]
    if missing:
        raise SystemExit(f"{sid}: missing labels for {missing}")
    with open(DIR / "labels.jsonl", "a") as sink:
        for i, pid in enumerate(ids):
            code, caption = entries[i]
            if code not in CODES:
                raise SystemExit(f"{sid} #{i}: bad code {code}")
            sink.write(json.dumps({"patch_id": pid, "label": CODES[code], "caption": caption or None,
                                   "sheet": sid, "labeler": "claude-visual", "ts": round(time.time())}) + "\n")
    print(f"{sid}: saved {len(ids)}")


def stats() -> None:
    from collections import Counter
    done = labelled()
    rows = {json.loads(l)["patch_id"]: json.loads(l) for l in open(FINAL / "manifest.jsonl")}
    c = Counter(r["label"] for r in done.values())
    by_split = Counter(rows[p].get("split") or "other" for p in done)
    print(f"{len(done)} labelled: {dict(c)} | by split {dict(by_split)}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make(int(sys.argv[2]))
    elif cmd == "add":
        add(sys.argv[2], sys.argv[3])
    else:
        stats()
