#!/usr/bin/env python3
"""Evidence figures for the final reports -> submissions/final/figures/.   python make_evidence.py

Numbers come from notebook outputs (resrc/REPORT_SUMMARY.md), the prediction PDFs in resrc/,
and the Indian-dataset experiments (validation_1k, prompt tests, change gate)."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
SRC = HERE / "evidence_src"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED = "#0b0b0b", "#52514e"
F = lambda s: ImageFont.load_default(size=s)


def style(ax, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")


def grouped(ax, labels, series, ymax=None, fmt="{:.1f}"):
    n = len(series)
    w = 0.8 / n
    for k, (name, vals) in enumerate(series.items()):
        xs = [i + (k - (n - 1) / 2) * w for i in range(len(labels))]
        ax.bar(xs, vals, w * 0.9, color=C[k], label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + (ymax or max(vals)) * 0.01, fmt.format(v), ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    if ymax:
        ax.set_ylim(0, ymax)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


# ---------------- Part A: LEVIR-CC models ----------------
fig, ax = plt.subplots(figsize=(10, 4))
grouped(ax, ["Phase 1\nCNN baseline", "Phase 2\nRemoteCLIP diff.", "Final\ncross-attention", "Phase 6\ntile-based", "Phase 8 (1a)\nattention diff."],
        {"BLEU-4": [30.3, 32.8, 33.7, 33.6, 32.5], "METEOR": [59.7, 63.0, 63.9, 63.6, 61.6],
         "ROUGE-L": [62.1, 65.5, 66.0, 66.2, 64.8], "Semantic cosine": [64.8, 70.6, 71.2, 71.5, 70.1]}, ymax=85)
ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left"); ax.set_ylabel("score ×100", fontsize=8, color=MUTED)
style(ax); save(fig, "A1_levir_phases.png")

curves = {
    "Phase 2 RemoteCLIP difference (LEVIR-CC)": ([1.8641, 1.2202, 1.0649, 0.9813, 0.9009, 0.8454, 0.7957, 0.7527, 0.7246, 0.7164],
                                                 [1.3050, 1.1653, 1.1078, 1.0512, 1.0682, 1.0451, 1.0554, 1.0536, 1.0515, 1.0545]),
    "Phase 8 Stage 1a (LEVIR-CC)": ([1.7495, 1.1564, 1.0187, 0.9390, 0.8692, 0.8226, 0.7781, 0.7305, 0.6939, 0.6512, 0.6154, 0.5816, 0.5477, 0.5141, 0.4817],
                                    [1.2376, 1.1495, 1.0492, 1.0502, 1.0260, 1.0442, 1.0171, 1.0199, 1.0348, 1.0377, 1.0411, 1.0598, 1.1374, 1.1392, 1.1362]),
    "Phase 8 Stage 1b, Qwen2-0.5B bridge (LEVIR-CC)": ([2.1771, 1.4699, 1.2941, 1.1957, 1.1193, 1.0472, 0.9907, 0.9371, 0.8897, 0.8385],
                                                       [1.5565, 1.4450, 1.4758, 1.4165, 1.4919, 1.5535, 1.5336, 1.4976, 1.5548, 1.5982]),
    "Final model fine-tuned on SECOND-CC": ([2.5434, 2.0132, 1.8174, 1.6847, 1.5754, 1.4795, 1.4067, 1.3461, 1.3072, 1.2836, 1.2798, 1.2775, 1.2811, 1.2748],
                                            [1.7732, 1.6418, 1.5764, 1.5501, 1.5282, 1.5224, 1.5274, 1.5258, 1.5323, 1.5296, 1.5296, 1.5312, 1.5372, 1.5509]),
}
fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
for ax, (title, (tr, va)) in zip(axes.flat, curves.items()):
    ep = range(1, len(tr) + 1)
    ax.plot(ep, tr, color=C[0], lw=2, label="train loss")
    ax.plot(ep, va, color=C[1], lw=2, label="validation loss")
    b = min(range(len(va)), key=va.__getitem__)
    ax.scatter([b + 1], [va[b]], color=C[1], s=40, zorder=3)
    ax.annotate(f"best val, epoch {b + 1}", (b + 1, va[b]), textcoords="offset points", xytext=(4, -14), fontsize=7.5, color=MUTED)
    ax.set_xlabel("epoch", fontsize=8, color=MUTED); style(ax, title)
axes[0, 0].legend(fontsize=8, frameon=False)
save(fig, "A2_training_curves.png")

shutil.copy(SUB / "phase1/figures/fig2_secondcc_forgetting.png", OUT / "A3_forgetting.png")

fig, ax = plt.subplots(figsize=(8, 3.6))
grouped(ax, ["Phase 8 (1a)\nzero-shot", "Phase 6\nfine-tuned", "Final\nfine-tuned"],
        {"BLEU-4": [8.1, 12.3, 13.0], "ROUGE-L": [30.0, 41.6, 43.3], "Semantic cosine": [37.8, 53.0, 55.0]}, ymax=70)
ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left"); ax.set_ylabel("SECOND-CC test score ×100", fontsize=8, color=MUTED)
style(ax); save(fig, "A4_secondcc_domain.png")

# same LEVIR-CC test samples across three runs (prediction PDFs)
caps = json.load(open(SRC / "captions.json"))
runs = [("Phase 6 run", "predictions_phase7.pdf"), ("Phase 8 run", "predictions_phase8.pdf"),
        ("Final, after SECOND-CC fine-tuning", "predictions_phase_final_finetuneS.pdf")]
samples = [4, 17, 19, 42, 2]
parts = []
for s in samples:
    page = Image.open(SRC / f"lv_{s}-{s:03d}.png").convert("RGB")
    imgs = page.crop((0, 0, int(page.width * 0.625), page.height))
    imgs = imgs.resize((int(imgs.width * 0.8), int(imgs.height * 0.8)))
    ref = caps["predictions_phase7.pdf"][s - 1]["ref"]
    lines = [("Reference", ref)] + [(n, caps[f][s - 1]["pred"]) for n, f in runs]
    canvas = Image.new("RGB", (imgs.width + 640, imgs.height + 10), "white")
    canvas.paste(imgs, (0, 5))
    d = ImageDraw.Draw(canvas)
    d.text((imgs.width + 12, 8), f"LEVIR-CC test sample {s}", fill=INK, font=F(16))
    y = 34
    for n, t in lines:
        d.text((imgs.width + 12, y), n + ":", fill=MUTED, font=F(13))
        t = t or ""
        for chunk in [t[i:i + 70] for i in range(0, len(t), 70)]:
            y += 17
            d.text((imgs.width + 24, y), chunk, fill=(150, 30, 30) if n != "Reference" else INK, font=F(14))
        y += 22
    parts.append(canvas)
W = max(p.width for p in parts)
grid = Image.new("RGB", (W, sum(p.height for p in parts)), "white")
y = 0
for p in parts:
    grid.paste(p, (0, y)); y += p.height
grid.save(OUT / "A5_same_samples.png")

stats = {r["file"]: r for r in json.load(open(SRC / "pdf_stats.json"))}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
names = ["Phase 6 run", "Phase 8 run", "Final after\nSECOND-CC FT"]
files = ["predictions_phase7.pdf", "predictions_phase8.pdf", "predictions_phase_final_finetuneS.pdf"]
v1 = [100 * stats[f]["change_called_nochange"] / stats[f]["change_refs"] for f in files]
v2 = [stats[f]["distinct_preds"] for f in files]
axes[0].bar(names, v1, color=C[1], width=0.55)
for i, v in enumerate(v1):
    axes[0].text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
axes[0].set_ylim(0, 110); style(axes[0], "Changed pairs captioned \"there is no difference\" (of 164)")
axes[1].bar(names, v2, color=C[0], width=0.55)
for i, v in enumerate(v2):
    axes[1].text(i, v + 1, str(v), ha="center", fontsize=8)
axes[1].axhline(163, color=MUTED, ls="--", lw=1); axes[1].text(2.3, 158, "163 distinct references", fontsize=7.5, color=MUTED, ha="right")
axes[1].set_ylim(0, 180); style(axes[1], "Distinct captions produced (300 LEVIR-CC samples)")
for ax in axes:
    ax.tick_params(axis="x", labelsize=8)
save(fig, "A6_prediction_behaviour.png")

def strip(prefix, nums, title, ref):
    parts = []
    for s in nums:
        page = Image.open(next(SRC.glob(f"{prefix}_{s}-*.png"))).convert("RGB")
        parts.append(page.resize((int(page.width * 0.72), int(page.height * 0.72))))
    W = max(p.width for p in parts)
    out = Image.new("RGB", (W, sum(p.height for p in parts) + 30), "white")
    ImageDraw.Draw(out).text((6, 6), title, fill=INK, font=F(16))
    y = 30
    for p in parts:
        out.paste(p, (0, y)); y += p.height
    return out
strip("sc", [1, 5, 9], "SECOND-CC test samples (Phase 6 run)", True).save(OUT / "A7_secondcc_samples.png")
a, b = strip("s1", [1, 2, 3], "Whole-image inference", False), strip("pw", [1, 2, 3], "Patch-wise inference", False)
both = Image.new("RGB", (a.width + b.width + 10, max(a.height, b.height)), "white")
both.paste(a, (0, 0)); both.paste(b, (a.width + 10, 0)); both.save(OUT / "A8_patchwise.png")

# ---------------- Part B ----------------
shutil.copy(SUB / "phase1/figures/india_pilot_sample.png", OUT / "B1_pilot_sample.png")
shutil.copy(SUB / "phase1/figures/fig4_offtheshelf_example.png", OUT / "B2_offtheshelf.png")

# ---------------- Part C ----------------
shutil.copy(SUB / "figures/fig1_coverage_map.png", OUT / "C1_coverage.png")
fig, ax = plt.subplots(figsize=(8, 3.2))
reasons = [("Haze/cloud (model screen)", 10994), ("Misaligned", 592), ("Water", 408), ("Snow", 133), ("No-data", 77), ("Cloud/shadow", 55), ("Whiteout", 19)]
ax.barh([r for r, _ in reasons][::-1], [v for _, v in reasons][::-1], color=C[0])
for i, (_, v) in enumerate(reasons[::-1]):
    ax.text(v + 80, i, f"{v:,}", va="center", fontsize=8)
style(ax, "Rejected patch pairs by reason (12,278 rejected, 15,610 accepted of 27,888)"); ax.set_xlim(0, 12500)
save(fig, "C2_rejections.png")
fig, ax = plt.subplots(figsize=(8, 3.4))
grouped(ax, ["8B, 8-bit", "8B, 4-bit (used)", "8B, full precision", "235B (API)"],
        {"Accuracy": [78, 77, 73, 92], "Hazy pairs caught": [70, 81, 56, 91], "Clean pairs rejected": [12, 27, 8, 8]}, ymax=110, fmt="{:.0f}")
ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left"); ax.set_ylabel("% (100-pair trial)", fontsize=8, color=MUTED)
style(ax); save(fig, "C3_haze_screen.png")
shutil.copy(SUB / "figures/fig2_classes.png", OUT / "C4_classes.png")
fig, ax = plt.subplots(figsize=(8, 3.4))
grouped(ax, ["Qwen3-VL-8B", "Gemma-3-27B", "Qwen3-VL-32B", "Qwen3-VL-235B"],
        {"Changes found (of 12)": [10, 8, 11, 4], "False changes (of 45)": [8, 16, 16, 1]}, ymax=20, fmt="{:.0f}")
ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper left"); style(ax, "4-class prompt with tie-break rule, 57 decidable pairs")
save(fig, "C5_model_comparison.png")
fig, ax = plt.subplots(figsize=(10, 3.8))
grouped(ax, ["No hints", "Pixel boxes", "Pixel boxes\n+ crops", "CNN boxes", "CNN boxes\n+ crops", "Forcing\nprompt"],
        {"Changes found (of 33)": [24, 11, 14, 10, 16, 29], "False changes (of 82)": [12, 3, 7, 6, 4, 51],
         "False on identical images (of 15)": [0, 0, 0, 0, 0, 12]}, ymax=60, fmt="{:.0f}")
ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left"); style(ax, "Qwen3-VL-8B with and without region hints")
save(fig, "C6_prompt_assistance.png")
shutil.copy(SUB / "figures/fig3_region_boxes.png", OUT / "C7_region_boxes.png")
shutil.copy(SUB / "figures/fig4_change_map.png", OUT / "C8_change_map.png")
fig, ax = plt.subplots(figsize=(8, 3.2))
grouped(ax, ["Colour only", "Pixel structure", "CNN only", "All features"],
        {"Logistic": [0.198, 0.333, 0.346, 0.432], "Boosted trees": [0.162, 0.358, 0.332, 0.449]}, ymax=0.6, fmt="{:.2f}")
ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper left"); style(ax, "Change gate: PR-AUC for land-use change (site-grouped cross-validation)")
save(fig, "C9_change_gate.png")
shutil.copy(SUB / "figures/fig6_recall_precision.png", OUT / "C10_recall_precision.png")
shutil.copy(SUB / "figures/fig5_validation_examples.png", OUT / "C11_validation_examples.png")
print(sorted(p.name for p in OUT.glob("*.png")))
