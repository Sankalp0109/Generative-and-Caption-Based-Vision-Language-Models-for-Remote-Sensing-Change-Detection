#!/usr/bin/env python3
"""Score haze/cloud model results against the eye-made reference labels.

    python cluster/evaluate_trial.py data/trial/results_8bit.jsonl [data/trial/results_api.jsonl ...]
    REFERENCE=data/trial_dev/reference_labels.jsonl python cluster/evaluate_trial.py <dev results>

"Positive" = obstructed (reference: haze/cloud clearly visible in either date; model: verdict
"reject"). Reports accuracy, recall (hazy pairs caught), false-reject rate (clean pairs thrown
away), with and without the pairs marked uncertain, plus throughput and every mistake.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REFERENCE = Path(os.environ.get("REFERENCE", "data/trial/reference_labels.jsonl"))


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(line) for line in open(path))}


def score(results: dict, reference: dict, skip_uncertain: bool) -> dict:
    tp = fp = tn = fn = unparsed = 0
    mistakes = []
    for pid, ref in reference.items():
        if pid not in results or ref["snow"] or (skip_uncertain and ref["uncertain"]):
            continue
        res = results[pid]
        if res["verdict"] == "unparsed":
            unparsed += 1
            continue
        truth = ref["before_obstructed"] or ref["after_obstructed"]
        predicted = res["verdict"] == "reject"
        tp += truth and predicted; fn += truth and not predicted
        fp += predicted and not truth; tn += not truth and not predicted
        if truth != predicted:
            mistakes.append((ref["idx"], pid, "missed haze" if truth else "false reject", res["ratings"]))
    n = tp + fp + tn + fn
    return {"n": n, "accuracy": (tp + tn) / n if n else 0, "recall": tp / (tp + fn) if tp + fn else 0,
            "precision": tp / (tp + fp) if tp + fp else 0, "false_reject_rate": fp / (fp + tn) if fp + tn else 0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "unparsed": unparsed, "mistakes": mistakes}


def main() -> None:
    reference = load(REFERENCE)
    for path in map(Path, sys.argv[1:]):
        results = {k: v for k, v in load(path).items() if k in reference}
        seconds = [r["seconds"] for r in results.values()]
        print(f"\n=== {path.name}: {len(results)} pairs rated, "
              f"{sum(seconds) / max(len(seconds), 1):.1f} s/pair, "
              f"~{sum(r['tokens_in'] or 0 for r in results.values()) / max(len(results), 1):.0f} tokens in ===")
        for label, skip in (("all labelled pairs", False), ("excluding uncertain", True)):
            s = score(results, reference, skip)
            print(f"{label:22s} n={s['n']:3d}  accuracy {s['accuracy']:.0%}  recall (haze caught) {s['recall']:.0%}  "
                  f"precision {s['precision']:.0%}  false-reject {s['false_reject_rate']:.0%}  "
                  f"[TP {s['tp']} FP {s['fp']} TN {s['tn']} FN {s['fn']}; unparsed {s['unparsed']}]")
        for idx, pid, kind, ratings in score(results, reference, False)["mistakes"]:
            print(f"   #{idx:<3d} {kind:13s} {pid:34s} {ratings}")


if __name__ == "__main__":
    main()
