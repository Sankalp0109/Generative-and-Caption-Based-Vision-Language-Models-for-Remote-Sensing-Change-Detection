#!/usr/bin/env python3
"""Build the report figures from the actual experiment outputs -> submissions/figures/.

    python submissions/make_figures.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "submissions/figures"
FINAL = ROOT / "data/FINAL"
OUT.mkdir(parents=True, exist_ok=True)
FOLDER = {json.loads(l)["patch_id"]: json.loads(l)["folder"] for l in open(FINAL / "manifest.jsonl")}
HUMAN = json.load(open(ROOT / "validation_1k/check/check_labels.json"))
M32 = {json.loads(l)["patch_id"]: json.loads(l) for l in open(ROOT / "validation_1k/results_32b.jsonl")}
M235 = {json.loads(l)["patch_id"]: json.loads(l) for l in open(ROOT / "validation_1k/results_235b.jsonl")}
FONT = ImageFont.load_default(size=18)
SMALL = ImageFont.load_default(size=15)
S = 360


def img(pid: str, name: str, side: int = S) -> Image.Image:
    return Image.open(FINAL / FOLDER[pid] / f"{name}.png").convert("RGB").resize((side, side), Image.BICUBIC)


def panel(images: list[tuple[Image.Image, str]], title: str | None = None, side: int = S) -> Image.Image:
    top = 30 if title else 0
    out = Image.new("RGB", (len(images) * (side + 8) + 8, side + 34 + top), "white")
    d = ImageDraw.Draw(out)
    if title:
        d.text((8, 6), title, fill=(20, 20, 20), font=FONT)
    for k, (im, cap) in enumerate(images):
        out.paste(im, (8 + k * (side + 8), top + 4))
        d.text((8 + k * (side + 8), top + side + 9), cap, fill=(60, 60, 60), font=SMALL)
    return out


def stack(parts: list[Image.Image]) -> Image.Image:
    w = max(p.width for p in parts)
    out = Image.new("RGB", (w, sum(p.height for p in parts)), "white")
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height
    return out


def conclusion(text: str, n: int = 230) -> str:
    """The sentence(s) in a free answer that state its conclusion, shortened."""
    t = re.sub(r"[*#✅]", "", text).replace("\n", " ")
    m = re.search(r"(Conclusion|Final Answer|In summary|Overall)[^:]*:\s*(.*)", t, re.I)
    s = (m.group(2) if m else t.split(". ")[-2] if ". " in t else t).strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def first(cond):
    return next(p for p in sorted(HUMAN) if cond(p))


lu = lambda d, p: d[p]["mark"] == "land_use"

# Fig 1: coverage map
shutil.copy(ROOT / "locations/coverage_map.png", OUT / "fig1_coverage_map.png")

# Fig 2: one human-labelled example per class
ex = {c: first(lambda p, c=c: HUMAN[p]["label"] == c) for c in ("change", "seasonal", "no_change")}
names = {"change": "Change", "seasonal": "Seasonal", "no_change": "No change"}
stack([panel([(img(p, "before"), "Before"), (img(p, "after"), "After")], f"Human label: {names[c]}  ({p})")
       for c, p in ex.items()]).save(OUT / "fig2_classes.png")

# Fig 3: region boxes (pixel and CNN) on a human-confirmed change the plain prompt found but the boxed prompts missed
from pipeline.change_gate import CNN, read
from pipeline.prompt_bias_test import boxed, cnn_regions
from change_visualization import prepare_visual_evidence
pb = [json.loads(l) for l in open(ROOT / "data/RSICC/prompt_bias/results.jsonl")]
ans = {}
for r in pb:
    if not r["control"]:
        ans.setdefault(r["patch_id"], {})[r["variant"]] = r["answer"]
cands = [p for p, a in ans.items() if HUMAN.get(p, {}).get("label") == "change"
         and a.get("A") == "land_use" and a.get("B") != "land_use" and a.get("E") != "land_use"]
pid = sorted(cands)[0]
before, after = read(FINAL / FOLDER[pid] / "before.tif"), read(FINAL / FOLDER[pid] / "after.tif")
bv, av, *_, regions = prepare_visual_evidence(before, after, max_regions=3)
b, a = [Image.fromarray((v * 255).astype("uint8")) for v in (bv, av)]
creg = cnn_regions(CNN(), before, after)
fig3 = stack([
    panel([(b.resize((S, S)), "Before"), (a.resize((S, S)), "After")],
          f"Plain prompt: change found (correct). Human label: Change  ({pid})"),
    panel([(boxed(b, regions[:3]).resize((S, S)), "Before, pixel-difference boxes"),
           (boxed(a, regions[:3]).resize((S, S)), "After, pixel-difference boxes")],
          f"Pixel boxes: answer '{ans[pid]['B']}' (missed)"),
    panel([(boxed(b, creg).resize((S, S)), "Before, CNN boxes"), (boxed(a, creg).resize((S, S)), "After, CNN boxes")],
          f"CNN boxes: answer '{ans[pid]['E']}' (missed)")])
fig3.save(OUT / "fig3_region_boxes.png")
FIG3_PID = pid

# Fig 4: rule-based change map on a pair the human labelled no change / seasonal
cm = [json.loads(l) for l in open(ROOT / "changemap_test/results_api_32b.jsonl")]
cmp = next(r for r in cm if HUMAN.get(r["patch_id"], {}).get("label") in ("no_change", "seasonal")
           and (ROOT / "changemap_test/maps" / r["patch_id"] / "comparison.png").exists())
comp = Image.open(ROOT / "changemap_test/maps" / cmp["patch_id"] / "comparison.png").convert("RGB")
comp = comp.resize((comp.width * 2, comp.height * 2), Image.NEAREST)
raw = cmp["raw"]
short = json.loads(raw[raw.find("{"):raw.rfind("}") + 1]).get("short_caption", "")
head = Image.new("RGB", (max(comp.width, 9 * len(short) + 260), 64), "white")
d = ImageDraw.Draw(head)
d.text((8, 6), f"Human label: {names[HUMAN[cmp['patch_id']]['label']]}  ({cmp['patch_id']}).  Before | After | change map", fill=(20, 20, 20), font=FONT)
d.text((8, 34), f"32B caption with map: \"{short}\"", fill=(180, 30, 30), font=SMALL)
stack([head, comp]).save(OUT / "fig4_change_map.png")
FIG4_PID = cmp["patch_id"]

# Fig 5: 1,000-pair validation examples (human truth): correct, false alarm, missed.
# Pairs chosen by eye from the qualifying cases so the change (or its absence) is visible at print size.
FIG5 = [("gujarat_048_r2c1", "(a) Correct: human Change; 32B and 235B say change"),
        ("andhra_pradesh_001_r3c3", "(b) False alarm: human Seasonal; 32B and 235B say change"),
        ("gujarat_012_r0c3", "(c) Missed: human Change; 32B and 235B say seasonal")]
assert HUMAN["gujarat_048_r2c1"]["label"] == "change" and lu(M32, "gujarat_048_r2c1") and lu(M235, "gujarat_048_r2c1")
assert HUMAN["andhra_pradesh_001_r3c3"]["label"] == "seasonal" and lu(M32, "andhra_pradesh_001_r3c3") and lu(M235, "andhra_pradesh_001_r3c3")
assert HUMAN["gujarat_012_r0c3"]["label"] == "change" and not lu(M32, "gujarat_012_r0c3") and not lu(M235, "gujarat_012_r0c3")
stack([panel([(img(p, "before", 300), "Before"), (img(p, "after", 300), "After")], t, 300) for p, t in FIG5]
      ).save(OUT / "fig5_validation_examples.png")
cases = [(p, None, conclusion(M235[p]["answer"])) for p, _ in FIG5]

# Fig 6: recall vs precision per rule (two panels: human-labelled set, all pairs)
rules = ["32B", "235B", "32B and 235B\nagree", "32B or 235B"]
human = {"recall": [89, 45, 42, 93], "precision": [11, 17, 18, 11]}
allp = {"recall": [96, 79, 78, 97], "precision": [26, 48, 51, 26]}
blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (title, d) in zip(axes, (("Human-labelled pairs (n = 495)", human), ("All pairs, human label where available (n = 962)", allp))):
    y = range(len(rules))
    h = 0.36
    ax.barh([i - h / 2 - 0.02 for i in y], d["recall"], h, color=blue, label="Changes found (recall)")
    ax.barh([i + h / 2 + 0.02 for i in y], d["precision"], h, color=orange, label="Precision of 'change' calls")
    for i in y:
        ax.text(d["recall"][i] + 1, i - h / 2 - 0.02, f"{d['recall'][i]}%", va="center", fontsize=9, color=ink)
        ax.text(d["precision"][i] + 1, i + h / 2 + 0.02, f"{d['precision'][i]}%", va="center", fontsize=9, color=ink)
    ax.axvline(90, color=muted, lw=1, ls="--")
    ax.text(89, -0.62, "90% precision target", fontsize=8, color=muted, ha="right")
    ax.set_xlim(0, 108)
    ax.set_title(title, fontsize=10, color=ink, loc="left")
    ax.set_yticks(list(y), rules, fontsize=9)
    ax.tick_params(colors=muted, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.set_xlabel("%", fontsize=8, color=muted)
axes[0].set_ylim(len(rules) - 0.4, -0.85)        # first rule on top (shared y: set once)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "fig6_recall_precision.png", dpi=200)

json.dump({"fig3": FIG3_PID, "fig4": FIG4_PID, "fig5": [c[0] for c in cases], "fig2": ex,
           "fig5_quotes": [c[2] for c in cases]}, open(OUT / "figure_sources.json", "w"), indent=1)
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
