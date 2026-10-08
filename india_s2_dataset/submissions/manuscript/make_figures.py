#!/usr/bin/env python3
"""Manuscript and appendix figures (legible fonts, multi-panel figures labelled a, b, c).

    python submissions/manuscript/make_figures.py   ->  submissions/manuscript/figures/
"""
import csv
import json
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUB = ROOT / "submissions"
EV = SUB / "final/figures"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED = "#111111", "#444444"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10, "font.family": "DejaVu Sans"})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel(ax, letter):
    ax.text(-0.08, 1.06, f"({letter})", transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")


def bars(ax, labels, series, ymax, fmt="{:.1f}"):
    n = len(series)
    w = 0.8 / n
    for k, (name, vals) in enumerate(series.items()):
        xs = [i + (k - (n - 1) / 2) * w for i in range(len(labels))]
        ax.bar(xs, vals, w * 0.92, color=C[k], label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + ymax * 0.012, fmt.format(v), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, ymax)
    style(ax)


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Figure 1 — study area: full India, all sites, accepted sites highlighted (no trimming at the top)
states = gpd.read_file(ROOT / "locations/layers/india_adm1.geojson")
man = [json.loads(l) for l in open(ROOT / "data/FINAL/manifest.jsonl")]
acc_sites = {r["site"] for r in man if r["status"] == "accepted"}
centre = {}
for r in man:
    lon0, lat0, lon1, lat1 = r["bounds_wgs84"]
    centre.setdefault(r["site"], ((lon0 + lon1) / 2, (lat0 + lat1) / 2))
reg = list(csv.DictReader(open(ROOT / "locations/india_aois.csv")))
fig, ax = plt.subplots(figsize=(7.2, 8.2))
states.plot(ax=ax, color="#f4f1ea", edgecolor="#777777", linewidth=0.5)
rej = [(float(r["lon"]), float(r["lat"])) for r in reg if r["key"] not in acc_sites]
acc = [centre[s] for s in acc_sites]
ax.scatter([p[0] for p in rej], [p[1] for p in rej], s=7, color="#bbbbbb", label=f"Site without accepted pairs ({len(rej)})", zorder=2)
ax.scatter([p[0] for p in acc], [p[1] for p in acc], s=9, color=C[0], label=f"Site with accepted pairs ({len(acc)})", zorder=3)
xmin, ymin, xmax, ymax = states.total_bounds
ax.set_xlim(xmin - 1.0, xmax + 1.0)
ax.set_ylim(ymin - 1.0, ymax + 1.5)            # full extent incl. Ladakh; margin at the top
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
ax.legend(loc="lower left", frameon=True)
ax.annotate("N", xy=(xmax - 0.2, ymax + 0.9), xytext=(xmax - 0.2, ymax - 1.0), ha="center", fontsize=12,
            fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
ax.set_aspect("equal"); ax.grid(color="#e5e5e5", lw=0.5)
save(fig, "fig1_study_area.png")

# Figure 2 — workflow
fig, ax = plt.subplots(figsize=(12, 5.4)); ax.axis("off"); ax.set_xlim(0, 10.6); ax.set_ylim(0, 5.4)
def box(x, y, w, h, text, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=col, ec="#555555", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2)
def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
rows = [("Part A\nLEVIR-CC / SECOND-CC", ["Captioning models\n(CNN, RemoteCLIP,\ncross-attention, tiles)", "Evaluation\nBLEU, METEOR,\nROUGE-L, cosine", "Transfer to\nSECOND-CC\n(fine-tuning)", "Language-model\ndecoders\n(Qwen bridge)"], "#dbe8f8"),
        ("Part B\nIndian pilot", ["Sentinel-2 pilot\ncollection\n(1,072 pairs)", "Off-the-shelf\ncaptioners\n(TeoChat, Qwen2.5-VL)", "Qualitative\nassessment\n(30 pairs)"], "#fde3d6"),
        ("Part C\nIndia-wide dataset", ["1,200 sites;\nscene selection,\nseason matching", "Download, patching,\nquality and haze\nscreening", "15,610 pairs;\nreference labels\n(535 human)", "Vision-language\nlabelling tests;\n1,000-pair validation"], "#d6f1e6")]
for i, (name, steps, col) in enumerate(rows):
    y = 4.1 - i * 1.75
    box(0.05, y, 1.75, 1.15, name, "#eeeeee")
    for j, s in enumerate(steps):
        x = 2.2 + j * 2.1
        box(x, y, 1.85, 1.15, s, col)
        if j:
            arrow(x - 0.25, y + 0.575, x, y + 0.575)
    arrow(1.8, y + 0.575, 2.2, y + 0.575)
    if i < 2:
        arrow(0.92, y, 0.92, y - 0.6)
save(fig, "fig2_workflow.png")

# Figure 3 — Part A results: (a) LEVIR-CC scores, (b) Phase 2 learning curve, (c) forgetting
fig = plt.figure(figsize=(11, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.45, wspace=0.25)
ax = fig.add_subplot(gs[0, :])
bars(ax, ["Phase 1\nCNN baseline", "Phase 2\nRemoteCLIP diff.", "Final\ncross-attention", "Phase 6\ntile-based", "Phase 8 (1a)\nattention diff."],
     {"BLEU-4": [30.3, 32.8, 33.7, 33.6, 32.5], "METEOR": [59.7, 63.0, 63.9, 63.6, 61.6], "ROUGE-L": [62.1, 65.5, 66.0, 66.2, 64.8],
      "Cosine": [64.8, 70.6, 71.2, 71.5, 70.1]}, 85)
ax.set_ylabel("LEVIR-CC test score (×100)"); ax.legend(ncol=4, loc="upper left", frameon=False); panel(ax, "a")
ax = fig.add_subplot(gs[1, 0])
tr = [1.8641, 1.2202, 1.0649, 0.9813, 0.9009, 0.8454, 0.7957, 0.7527, 0.7246, 0.7164]
va = [1.3050, 1.1653, 1.1078, 1.0512, 1.0682, 1.0451, 1.0554, 1.0536, 1.0515, 1.0545]
ax.plot(range(1, 11), tr, color=C[0], lw=2, marker="o", ms=4, label="Training loss")
ax.plot(range(1, 11), va, color=C[1], lw=2, marker="s", ms=4, label="Validation loss")
ax.axvline(6, color=MUTED, ls="--", lw=1); ax.text(6.15, 1.7, "best validation\n(epoch 6)", fontsize=9, color=MUTED)
ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.legend(frameon=False); style(ax); panel(ax, "b")
ax = fig.add_subplot(gs[1, 1])
bars(ax, ["Phase 6\nbefore", "Phase 6\nafter", "Final\nbefore", "Final\nafter"],
     {"BLEU-4": [33.6, 12.1, 33.7, 14.9], "ROUGE-L": [66.2, 37.4, 66.0, 43.7]}, 95)
ax.set_ylabel("LEVIR-CC test score (×100)"); ax.legend(frameon=False, loc="upper center", ncol=2); panel(ax, "c")
save(fig, "fig3_partA_results.png")

# Figure 4 — prediction behaviour on 300 saved LEVIR-CC test samples
st = {r["file"]: r for r in json.load(open(SUB / "final/evidence_src/pdf_stats.json"))}
files = ["predictions_phase7.pdf", "predictions_phase8.pdf", "predictions_phase_final_finetuneS.pdf"]
names = ["Phase 6", "Phase 8 (1a)", "Final after\nSECOND-CC FT"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
v = [100 * st[f]["change_called_nochange"] / st[f]["change_refs"] for f in files]
axes[0].bar(names, v, color=C[1], width=0.55)
for i, x in enumerate(v):
    axes[0].text(i, x + 2, f"{x:.0f}%", ha="center", fontsize=10)
axes[0].set_ylim(0, 112); axes[0].set_ylabel("Changed pairs captioned\n\"there is no difference\" (%)"); style(axes[0]); panel(axes[0], "a")
v = [st[f]["distinct_preds"] for f in files]
axes[1].bar(names, v, color=C[0], width=0.55)
for i, x in enumerate(v):
    axes[1].text(i, x + 3, str(x), ha="center", fontsize=10)
axes[1].axhline(163, color=MUTED, ls="--", lw=1); axes[1].text(2.35, 150, "163 distinct references", ha="right", fontsize=9.5, color=MUTED)
axes[1].set_ylim(0, 185); axes[1].set_ylabel("Distinct captions produced"); style(axes[1]); panel(axes[1], "b")
fig.tight_layout(); save(fig, "fig4_prediction_behaviour.png")

# Figure 5 — dataset screening: (a) rejection reasons, (b) haze-screen trial
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.05, 1]})
reasons = [("Haze / cloud\n(model screen)", 10994), ("Misaligned", 592), ("Water", 408), ("Snow", 133), ("No-data", 77), ("Cloud / shadow", 55), ("Whiteout", 19)]
ax = axes[0]
ax.barh([r for r, _ in reasons][::-1], [n for _, n in reasons][::-1], color=C[0])
for i, (_, n) in enumerate(reasons[::-1]):
    ax.text(n + 120, i, f"{n:,}", va="center", fontsize=9.5)
ax.set_xlim(0, 13000); ax.set_xlabel("Rejected patch pairs"); style(ax); panel(ax, "a")
ax = axes[1]
bars(ax, ["8B\n8-bit", "8B\n4-bit (used)", "8B full\nprecision", "235B"],
     {"Accuracy": [78, 77, 73, 92], "Hazy caught": [70, 81, 56, 91], "Clean rejected": [12, 27, 8, 8]}, 112, fmt="{:.0f}")
ax.set_ylabel("% of trial pairs (n = 84)"); ax.legend(ncol=3, frameon=False, loc="upper left", fontsize=9); panel(ax, "b")
fig.tight_layout(); save(fig, "fig5_screening.png")

# Figure 6 — human-labelled class examples (a) change (b) seasonal (c) no change
font = ImageFont.load_default(size=30)
ex = [("a", "Change", "andhra_pradesh_t1_024_r1c1"), ("b", "Seasonal", "andhra_pradesh_001_r0c2"), ("c", "No change", "andhra_pradesh_001_r1c2")]
folder = {r["patch_id"]: r["folder"] for r in man}
S = 420
img = Image.new("RGB", (2 * S + 30, 3 * (S + 64)), "white"); d = ImageDraw.Draw(img)
for i, (letter, lab, pid) in enumerate(ex):
    y = i * (S + 64)
    d.text((6, y + 8), f"({letter}) {lab}", fill=INK, font=font)
    for j, n in enumerate(("before", "after")):
        img.paste(Image.open(ROOT / "data/FINAL" / folder[pid] / f"{n}.png").convert("RGB").resize((S, S)), (j * (S + 30), y + 52))
img.save(OUT / "fig6_class_examples.png")

# Figure 7 — vision-language labelling experiments: (a) model size, (b) hints
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.45]})
bars(axes[0], ["Qwen3-VL\n8B", "Gemma 3\n27B", "Qwen3-VL\n32B", "Qwen3-VL\n235B"],
     {"Changes found (of 12)": [10, 8, 11, 4], "False changes (of 45)": [8, 16, 16, 1]}, 21, fmt="{:.0f}")
axes[0].set_ylabel("Pairs"); axes[0].legend(frameon=False, loc="upper left", fontsize=9); panel(axes[0], "a")
bars(axes[1], ["No\nhints", "Pixel\nboxes", "Pixel boxes\n+ crops", "CNN\nboxes", "CNN boxes\n+ crops", "Forcing\nprompt"],
     {"Changes found (of 33)": [24, 11, 14, 10, 16, 29], "False changes (of 82)": [12, 3, 7, 6, 4, 51], "False on identical pairs (of 15)": [0, 0, 0, 0, 0, 12]}, 60, fmt="{:.0f}")
axes[1].set_ylabel("Pairs"); axes[1].legend(frameon=False, loc="upper left", fontsize=9); panel(axes[1], "b")
fig.tight_layout(); save(fig, "fig7_vlm_tests.png")

# Figure 8 — 1,000-pair validation: (a) human-labelled set, (b) all labelled pairs
rules = ["32B", "235B", "Both\nagree", "Either"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
for ax, (title, rec, prec), letter in zip(axes, (("Human-labelled pairs (n = 495)", [89, 45, 42, 93], [11, 17, 18, 11]),
                                                ("All labelled pairs (n = 962)", [96, 79, 78, 97], [26, 48, 51, 26])), "ab"):
    bars(ax, rules, {"Changes found (recall)": rec, "Precision of change calls": prec}, 112, fmt="{:.0f}")
    ax.axhline(90, color=MUTED, ls="--", lw=1); ax.text(1.5, 92, "90% precision target", ha="center", fontsize=9, color=MUTED)
    ax.set_title(title, loc="left"); panel(ax, letter)
axes[0].set_ylabel("%")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, ncol=2, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(); save(fig, "fig8_validation.png")

# Figure 9 — validation examples (already panelled a, b, c)
shutil.copy(EV / "C11_validation_examples.png", OUT / "fig9_validation_examples.png")

# Appendix figures (reused evidence figures)
for src, dst in [("A2_training_curves.png", "appA1_training_curves.png"), ("A5_same_samples.png", "appA2_levir_samples.png"),
                 ("A7_secondcc_samples.png", "appA3_secondcc_samples.png"), ("A8_patchwise.png", "appA4_patchwise.png"),
                 ("B1_pilot_sample.png", "appA5_pilot_sample.png"), ("B2_offtheshelf.png", "appA6_offtheshelf.png"),
                 ("C7_region_boxes.png", "appA7_region_boxes.png"), ("C8_change_map.png", "appA8_change_map.png"),
                 ("C9_change_gate.png", "appA9_change_gate.png"), ("A4_secondcc_domain.png", "appA10_secondcc_domain.png")]:
    shutil.copy(EV / src, OUT / dst)
print(sorted(p.name for p in OUT.iterdir()))
