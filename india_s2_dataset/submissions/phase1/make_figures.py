#!/usr/bin/env python3
"""Figures for the Phase 1 report, from numbers printed in the repository notebooks.   python make_figures.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "figures"
blue, orange, aqua, ink, muted = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")
    ax.tick_params(colors=muted, labelsize=8)

# Fig 1: LEVIR-CC test metrics per phase (phase1.ipynb, phase2.ipynb, phase6_output.ipynb, phase_final_output_finetuneS.ipynb)
models = ["Phase 1\nCNN baseline", "Phase 2\nRemoteCLIP\ndifference", "Phase 6\ntile-based", "Phase Final\ncross-attention"]
metrics = {"BLEU-4": [30.3, 32.8, 33.6, 33.7], "METEOR": [59.7, 63.0, 63.6, 63.9], "ROUGE-L": [62.1, 65.5, 66.2, 66.0]}
fig, ax = plt.subplots(figsize=(9, 3.8))
w = 0.26
for k, (name, vals) in enumerate(metrics.items()):
    xs = [i + (k - 1) * w for i in range(len(models))]
    ax.bar(xs, vals, w - 0.03, color=(blue, orange, aqua)[k], label=name)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1, f"{v:.1f}", ha="center", fontsize=7.5, color=ink)
ax.set_xticks(range(len(models)), models, fontsize=8.5)
ax.set_ylim(0, 80); ax.set_ylabel("score (×100)", fontsize=8, color=muted)
ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
style(ax); fig.tight_layout(); fig.savefig(OUT / "fig1_levircc_phases.png", dpi=200)

# Fig 2: forgetting after sequential SECOND-CC fine-tuning (BLEU-4 and ROUGE-L on the LEVIR-CC test set)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
data = {"Phase 6 tile-based": ([33.6, 12.1], [66.2, 37.4]), "Phase Final cross-attention": ([33.7, 14.9], [66.0, 43.7])}
for ax, (name, (b4, rl)) in zip(axes, data.items()):
    xs = [0, 1]
    ax.bar([x - 0.18 for x in xs], b4, 0.34, color=blue, label="BLEU-4")
    ax.bar([x + 0.18 for x in xs], rl, 0.34, color=orange, label="ROUGE-L")
    for x, v in zip(xs, b4): ax.text(x - 0.18, v + 1, f"{v:.1f}", ha="center", fontsize=8, color=ink)
    for x, v in zip(xs, rl): ax.text(x + 0.18, v + 1, f"{v:.1f}", ha="center", fontsize=8, color=ink)
    ax.set_xticks(xs, ["LEVIR-CC only", "after SECOND-CC\nfine-tuning"], fontsize=8.5)
    ax.set_title(name, fontsize=9.5, color=ink, loc="left"); ax.set_ylim(0, 80); style(ax)
axes[0].set_ylabel("LEVIR-CC test score (×100)", fontsize=8, color=muted)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, ncol=2, fontsize=8, frameon=False, loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "fig2_secondcc_forgetting.png", dpi=200)

# Fig 4: one Indian pair from the off-the-shelf captioning check, with the captions produced
R = Path(__file__).resolve().parents[2] / "Rouge/sample_30"
S = 360; f = ImageFont.load_default(size=16); f2 = ImageFont.load_default(size=14)
lines = ["Qwen2.5-VL: \"...increased urban density and expansion of buildings, particularly along the central axis,",
         "indicating rapid development. There's also visible growth in road infrastructure...\"",
         "TeoChat: \"Nothing meaningful changed.\""]
out = Image.new("RGB", (2 * S + 24, S + 40 + 22 * len(lines)), "white"); d = ImageDraw.Draw(out)
for j, n in enumerate(("before", "after")):
    out.paste(Image.open(R / f"p58_{n}.png").convert("RGB").resize((S, S)), (8 + j * (S + 8), 8))
    d.text((8 + j * (S + 8), S + 12), n.capitalize(), fill=muted, font=f)
for k, t in enumerate(lines):
    d.text((8, S + 38 + 22 * k), t, fill=(120, 30, 30), font=f2)
out.save(OUT / "fig4_offtheshelf_example.png")
print(sorted(p.name for p in OUT.glob("*.png")))
