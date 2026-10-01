#!/usr/bin/env python3
"""Sort trial patch-pairs into kept/ and rejected/ by the model's haze/cloud verdict.

    python cluster/apply_trial.py data/trial/results_8bit.jsonl

Copies (never moves) each pair's before/after PNGs into data/trial/kept/ or data/trial/rejected/,
writes data/trial/verdicts.csv, and a contact sheet of every rejected pair
(data/trial/rejected_sheet.jpg) so the discards can be checked by eye. Unparsed answers go to
rejected/ too: an unreadable answer is never silently kept.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

TRIAL = Path("data/trial")


def main() -> None:
    results = {r["id"]: r for r in (json.loads(line) for line in open(sys.argv[1]))}
    manifest = [json.loads(line) for line in open(TRIAL / "manifest.jsonl")]
    for folder in ("kept", "rejected"):
        shutil.rmtree(TRIAL / folder, ignore_errors=True)
        (TRIAL / folder).mkdir()
    rows, rejected = [], []
    for row in manifest:
        res = results.get(row["id"])
        if res is None:
            continue
        folder = "kept" if res["verdict"] == "keep" else "rejected"
        for role in ("before", "after"):
            shutil.copy(TRIAL / row[role], TRIAL / folder / Path(row[role]).name)
        rows.append({"id": row["id"], "split": row["split"], "state": row["state"], "verdict": res["verdict"],
                     **{f"{role}_{k}": (res["ratings"] or {}).get(role, {}).get(k) for role in ("before", "after")
                        for k in ("haze", "cloud")}})
        if folder == "rejected":
            rejected.append((row, res))
    with open(TRIAL / "verdicts.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    W = 200
    sheet = Image.new("RGB", (4 * (2 * W + 14) + 10, max(1, (len(rejected) + 3) // 4) * (W + 22) + 6), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (row, res) in enumerate(rejected):
        x, y = (i % 4) * (2 * W + 14) + 8, (i // 4) * (W + 22) + 4
        r = res["ratings"] or {}
        draw.text((x, y), f"{row['id'][:30]}  B{r.get('before')} A{r.get('after')}".replace("'", ""), fill=(255, 220, 0))
        for j, role in enumerate(("before", "after")):
            sheet.paste(Image.open(TRIAL / row[role]).resize((W, W)), (x + j * (W + 3), y + 14))
    sheet.save(TRIAL / "rejected_sheet.jpg", quality=88)
    kept = sum(r["verdict"] == "keep" for r in rows)
    print(f"{len(rows)} pairs: kept {kept}, rejected {len(rows) - kept} "
          f"({sum(r['verdict'] == 'unparsed' for r in rows)} unparsed) -> {TRIAL}/kept, {TRIAL}/rejected, verdicts.csv")


if __name__ == "__main__":
    main()
