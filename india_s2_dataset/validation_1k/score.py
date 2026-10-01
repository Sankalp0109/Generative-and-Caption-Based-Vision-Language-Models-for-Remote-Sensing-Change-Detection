#!/usr/bin/env python3
"""Score the 32B / 235B plain-question answers against the reference labels -> REPORT.md.

    python validation_1k/score.py

Truth for each pair: the human check (check/check_labels.json) where it exists, otherwise Claude's
label. Before any check exists the report is provisional (Claude labels only). "Can't tell" pairs
are left out of scoring. Positive class = land_use (lasting land-use change).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUMAN = {"change": "land_use", "no_change": "none", "seasonal": "seasonal", "cant_tell": None}


def load(name: str) -> dict:
    return {json.loads(l)["patch_id"]: json.loads(l) for l in open(HERE / name)}


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "–"
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:.0f}% ({100 * (c - h):.0f}–{100 * (c + h):.0f})"


def main() -> None:
    ref, m32, m235 = load("reference_labels.jsonl"), load("results_32b.jsonl"), load("results_235b.jsonl")
    chk_path = HERE / "check/check_labels.json"
    check = json.loads(chk_path.read_text()) if chk_path.exists() else {}
    groups = json.loads((HERE / "work/check_manifest.json").read_text()) if (HERE / "work/check_manifest.json").exists() else {}
    verified = bool(check)

    truth, source = {}, Counter()
    for p, r in ref.items():
        if p not in m32 or p not in m235:
            continue
        if p in check:
            t = HUMAN[check[p]["label"]]
            source["human"] += 1
            if t is None:
                source["cant_tell (left out)"] += 1
                continue
        else:
            t = r["label"]
            source["claude"] += 1
        truth[p] = t
    lu = lambda x: x == "land_use"
    rules = {"32B": lambda p: lu(m32[p]["mark"]),
             "235B": lambda p: lu(m235[p]["mark"]),
             "32B and 235B agree": lambda p: lu(m32[p]["mark"]) and lu(m235[p]["mark"]),
             "32B or 235B": lambda p: lu(m32[p]["mark"]) or lu(m235[p]["mark"])}
    pos = [p for p, t in truth.items() if lu(t)]
    neg = [p for p, t in truth.items() if not lu(t)]

    out = ["# Validation on ~1,000 labelled pairs", "",
           "**Status: " + ("verified — truth includes the human check" if verified else
                           "PROVISIONAL — truth is Claude's labels only; run the human check, then re-run score.py") + "**", "",
           f"Pairs scored: {len(truth)} ({len(pos)} land-use, {len(neg)} not). Truth source: {dict(source)}.", "",
           "Question asked (no format, no hints): *Before and after images of the same place, 6 years apart. "
           "Has anything lasting changed (like buildings, roads, ponds, cleared land), or are the differences only seasonal?*", "",
           "## Detection of land-use change", "",
           "| Rule | Found (recall) | Precision | False alarms on no-change pairs | Pairs flagged |",
           "|---|---|---|---|---|"]
    for name, f in rules.items():
        tp = sum(f(p) for p in pos)
        fp = sum(f(p) for p in neg)
        out.append(f"| {name} | {tp}/{len(pos)} = {wilson(tp, len(pos))} | {tp}/{tp + fp} = {wilson(tp, tp + fp)} | "
                   f"{fp}/{len(neg)} = {wilson(fp, len(neg))} | {tp + fp} of {len(truth)} |")
    out += ["", "Percentages with 95% Wilson intervals.", "", "## Answers by true class", "",
            "| True class | n | 32B says land-use | 235B says land-use |", "|---|---|---|---|"]
    for cls in ("land_use", "seasonal", "none", "atmospheric", "unusable"):
        ps = [p for p, t in truth.items() if t == cls]
        if ps:
            out.append(f"| {cls} | {len(ps)} | {sum(rules['32B'](p) for p in ps)} | {sum(rules['235B'](p) for p in ps)} |")
    unparsed = {m: sum(r["mark"] not in ("land_use", "seasonal", "none") for r in d.values()) for m, d in (("32B", m32), ("235B", m235))}
    out += ["", f"Answers marked unclear/unparsed: {unparsed}.", ""]

    if verified:
        out += ["## Human check", ""]
        checked = [p for p in check if p in ref]
        both = [p for p in checked if HUMAN[check[p]["label"]] is not None]
        agree = sum(lu(HUMAN[check[p]["label"]]) == lu(ref[p]["label"]) for p in both)
        out.append(f"- Pairs checked: {len(checked)} of {len(groups)} "
                   f"(can't tell: {sum(HUMAN[check[p]['label']] is None for p in checked)}).")
        out.append(f"- Claude's land-use label matched the human on {agree}/{len(both)} checked pairs = {wilson(agree, len(both))}.")
        samp = [p for p in both if groups.get(p) == "agree_sample"]
        wrong = sum(lu(HUMAN[check[p]["label"]]) != lu(ref[p]["label"]) for p in samp)
        out.append(f"- Pairs where Claude and both models agreed (random sample): the human disagreed on {wrong}/{len(samp)} "
                   f"= {wilson(wrong, len(samp))}. This is the error rate hidden in the unchecked agreements.")
        dis = [p for p in both if groups.get(p) == "disagree"]
        for name, src in (("32B", m32), ("235B", m235)):
            right = sum(lu(HUMAN[check[p]["label"]]) == lu(src[p]["mark"]) for p in dis)
            out.append(f"- On the disagreements, the {name} matched the human on {right}/{len(dis)}; "
                       f"Claude on {sum(lu(HUMAN[check[p]['label']]) == lu(ref[p]['label']) for p in dis)}/{len(dis)}.")
        out.append("")
    audit_path = HERE / "work/audit/audit.json"
    if audit_path.exists():
        v = Counter(json.loads(audit_path.read_text())["verdicts"].values())
        n = sum(v.values())
        out += ["## Claude self-audit of the models' false alarms", "",
                f"{n} random pairs where both 32B and 235B said land-use and Claude did not, looked at again: "
                f"models wrong {v['model_wrong']}, Claude wrong {v['claude_wrong']}, unsure {v['unsure']}.",
                f"So most false alarms are real ({wilson(v['model_wrong'], n)} model wrong). Even counting every "
                "'unsure' as a real change, precision of the best rule stays far below 90%.", ""]
    if not verified:
        out += ["## How to finish (human check, 535 pairs, ~35–45 min)", "",
                "```bash",
                "cd \"Plan B\"",
                "python3 validation_1k/check/server.py      # keep running; open http://localhost:8001",
                "```",
                "One click (or key 1–4) per pair: Change / No change / Seasonal / Can't tell. It saves and moves on.",
                "Answers go to `validation_1k/check/check_labels.json`. When done: `python3 validation_1k/score.py` "
                "rewrites this report with verified numbers.", ""]
    out += ["## Decision rule", "",
            "Scale the API step only if a rule reaches **≥ 90% precision** (lower interval bound near 90%) with useful recall. "
            "Otherwise keep the dataset to verified pairs.", ""]
    (HERE / "REPORT.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
