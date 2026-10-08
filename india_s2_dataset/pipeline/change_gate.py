#!/usr/bin/env python3
"""VLM-free check for land-use change in a patch pair (the gate before any vision-language model).

    python -m pipeline.change_gate features      # features for every accepted patch -> data/RSICC/gate/features.jsonl
    python -m pipeline.change_gate evaluate      # site-grouped CV against the eye labels, report + misses page
    python -m pipeline.change_gate apply         # score all accepted patches with the chosen threshold

Features (per 256 px patch, from the stored GeoTIFFs; nothing new downloaded):
  colour   colour difference after per-band brightness matching (the old "changed" signal)
  edges    NEW geometry: share of strong edges in the after image with no edge nearby in the before
           image (and the reverse). Crops change colour inside unchanged field boundaries; roads and
           buildings add new edges.
  bright   pixels that became bright and grey (low greenness): concrete, roofs, cleared sites
  blobs    size/compactness of the largest changed region (construction is compact or linear)
  cnn      feature distance from a pretrained ResNet-50 (layer 3, 1/16 resolution): pooling over
           ~16 px makes it tolerant of 1-2 px misregistration, unlike pixel differences
Labels: data/RSICC/claude_labels/labels.jsonl (visual labels; target = land_use).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.patches import apply_match, brightness_match

FINAL = ROOT / "data/FINAL"
OUT = ROOT / "data/RSICC/gate"
LABELS = ROOT / "data/RSICC/claude_labels/labels.jsonl"
TARGET_RECALL = 0.90


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.moveaxis(src.read(), 0, -1).astype("float32")


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=r)


def pixel_features(before: np.ndarray, after: np.ndarray) -> dict:
    from scipy.ndimage import label, sobel
    valid = np.all(before > 0, axis=2) & np.all(after > 0, axis=2)
    matched = apply_match(after, brightness_match(before, after, valid))
    diff = np.abs(matched - before).mean(axis=2)
    f = {"colour_mean": float(diff.mean()), "colour_frac40": float((diff > 40).mean()),
         "colour_frac25": float((diff > 25).mean())}

    def edges(img):
        lum = img.mean(axis=2)
        return np.hypot(sobel(lum, 0), sobel(lum, 1))
    eb, ea = edges(before), edges(matched)
    t = np.percentile(np.concatenate([eb.ravel(), ea.ravel()]), 85)
    sb, sa = eb > t, ea > t
    new = sa & ~_dilate(sb, 2)
    lost = sb & ~_dilate(sa, 2)
    f["edge_new"] = float(new.mean())
    f["edge_lost"] = float(lost.mean())
    f["edge_corr"] = float(np.corrcoef(eb.ravel(), ea.ravel())[0, 1]) if eb.std() > 0 and ea.std() > 0 else 1.0

    def bright_grey(img):
        lum = img.mean(axis=2)
        green = img[..., 1] - (img[..., 0] + img[..., 2]) / 2
        return (lum > np.percentile(lum, 80)) & (green < 2) & (img.max(axis=2) - img.min(axis=2) < 40)
    became = bright_grey(matched) & ~bright_grey(before)
    f["bright_new"] = float(became.mean())

    changed = (diff > 40) & (new | became | _dilate(new, 1))
    lab, n = label(changed)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        big = int(np.argmax(sizes)) + 1
        ys, xs = np.nonzero(lab == big)
        box = (np.ptp(ys) + 1) * (np.ptp(xs) + 1)
        f.update(blob_max=float(sizes.max() / changed.size), blob_n=float(n),
                 blob_fill=float(sizes.max() / box), blob_struct=float(changed.mean()))
    else:
        f.update(blob_max=0.0, blob_n=0.0, blob_fill=0.0, blob_struct=0.0)
    return f


class CNN:
    """ImageNet ResNet-50 up to layer3; per-cell cosine distance between before and after features."""

    def __init__(self):
        import torch
        import torchvision
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        net = torchvision.models.resnet50(weights=weights).eval()
        self.body = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
        self.torch = torch
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        torch.set_num_threads(12)

    def _prep(self, imgs):
        t = self.torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
        return (t - self.mean) / self.std

    def features(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
        torch = self.torch
        with torch.inference_mode():
            fb = self.body(self._prep([p[0] for p in pairs]))
            fa = self.body(self._prep([p[1] for p in pairs]))
            dist = 1 - torch.nn.functional.cosine_similarity(fb, fa, dim=1)   # (N, 16, 16)
        out = []
        for d in dist.numpy():
            v = d.ravel()
            out.append({"cnn_mean": float(v.mean()), "cnn_p90": float(np.percentile(v, 90)),
                        "cnn_max": float(v.max()), "cnn_frac": float((v > 0.35).mean())})
        return out


def stretch_pair(before, after):
    from change_visualization import joint_stretch
    b, a = joint_stretch(before, after, 1, 99.5)
    return (b * 255).astype("uint8"), (a * 255).astype("uint8")


def compute_features(limit: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(FINAL / "manifest.jsonl")]
    acc = [r for r in rows if r["status"] == "accepted"]
    labelled = {json.loads(l)["patch_id"] for l in open(LABELS)}
    acc.sort(key=lambda r: r["patch_id"] not in labelled)   # labelled first, so evaluation can start early
    out = OUT / "features.jsonl"
    done = {json.loads(l)["patch_id"] for l in open(out)} if out.exists() else set()
    todo = [r for r in acc if r["patch_id"] not in done][:limit]
    cnn = CNN()
    print(f"{len(acc)} accepted, {len(done)} done, {len(todo)} to do", flush=True)
    with open(out, "a") as sink:
        for s in range(0, len(todo), 32):
            chunk = todo[s:s + 32]
            pix, pairs = [], []
            for r in chunk:
                b, a = read(FINAL / r["folder"] / "before.tif"), read(FINAL / r["folder"] / "after.tif")
                pix.append(pixel_features(b, a))
                pairs.append(stretch_pair(b, a))
            for r, p, c in zip(chunk, pix, cnn.features(pairs)):
                sink.write(json.dumps({"patch_id": r["patch_id"], "site": r["site"], "split": r.get("split"),
                                       "selected": r.get("selected"), **p, **c}) + "\n")
            sink.flush()
            if (s // 32) % 20 == 0:
                print(f"[{s + len(chunk)}/{len(todo)}]", flush=True)


FEATURE_SETS = {
    "colour only (old signal)": ["colour_mean", "colour_frac40", "colour_frac25"],
    "pixel structure": ["colour_mean", "colour_frac40", "edge_new", "edge_lost", "edge_corr", "bright_new",
                        "blob_max", "blob_n", "blob_fill", "blob_struct"],
    "cnn only": ["cnn_mean", "cnn_p90", "cnn_max", "cnn_frac"],
}
FEATURE_SETS["all"] = FEATURE_SETS["pixel structure"] + FEATURE_SETS["cnn only"]


def load_xy():
    labels = {}
    for line in open(LABELS):
        r = json.loads(line)
        labels[r["patch_id"]] = r["label"]
    feats = {json.loads(l)["patch_id"]: json.loads(l) for l in open(OUT / "features.jsonl")}
    ids = [p for p in labels if p in feats]
    return ids, feats, labels


def evaluate() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ids, feats, labels = load_xy()
    y = np.array([labels[i] == "land_use" for i in ids], dtype=int)
    groups = np.array([feats[i]["site"] for i in ids])
    print(f"{len(ids)} labelled pairs with features: {int(y.sum())} land-use, {dict(Counter(labels[i] for i in ids))}")
    report, best = [], None
    for name, cols in FEATURE_SETS.items():
        X = np.array([[feats[i][c] for c in cols] for i in ids])
        for mname, model in (("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                             ("boosted trees", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                              max_iter=200, class_weight="balanced"))):
            score = np.zeros(len(ids))
            for tr, te in GroupKFold(5).split(X, y, groups):
                model.fit(X[tr], y[tr])
                score[te] = model.predict_proba(X[te])[:, 1]
            thr = np.sort(score[y == 1])[int(np.floor((1 - TARGET_RECALL) * y.sum()))]
            keep = score >= thr
            row = {"features": name, "model": mname, "pr_auc": average_precision_score(y, score),
                   "recall": keep[y == 1].mean(), "kept": keep.mean(),
                   "seasonal_kept": keep[[labels[i] == "seasonal" for i in ids]].mean(),
                   "none_kept": keep[[labels[i] == "none" for i in ids]].mean(),
                   "haze_kept": keep[[labels[i] == "atmospheric" for i in ids]].mean()}
            report.append(row)
            if name == "all" and (best is None or row["pr_auc"] > best[0]["pr_auc"]):
                best = (row, mname, score, thr)
    print(f"\n{'features':26s} {'model':14s} {'PR-AUC':>6s} {'recall':>7s} {'pairs kept':>10s} {'seasonal kept':>13s} {'no-change kept':>14s} {'haze kept':>9s}")
    for r in report:
        print(f"{r['features']:26s} {r['model']:14s} {r['pr_auc']:6.2f} {r['recall']:7.0%} {r['kept']:10.0%} "
              f"{r['seasonal_kept']:13.0%} {r['none_kept']:14.0%} {r['haze_kept']:9.0%}")
    row, mname, score, thr = best
    json.dump({"model": mname, "report": report}, open(OUT / "evaluation.json", "w"), indent=1)
    misses = [(i, float(s)) for i, s, t in zip(ids, score, y) if t == 1 and s < thr]
    json.dump({"missed_landuse": misses, "cv_scores": dict(zip(ids, map(float, score)))}, open(OUT / "cv_scores.json", "w"))
    print(f"\nchosen: all features, {mname}; missed land-use pairs: {len(misses)}")


def apply() -> None:
    """Fit the chosen model on all labels, score every accepted patch, keep those above the CV threshold."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ev = json.load(open(OUT / "evaluation.json"))
    cols = FEATURE_SETS["all"]
    ids, feats, labels = load_xy()
    y = np.array([labels[i] == "land_use" for i in ids], dtype=int)
    X = np.array([[feats[i][c] for c in cols] for i in ids])
    model = (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced")
             if ev["model"] == "boosted trees"
             else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")))
    model.fit(X, y)
    cv = json.load(open(OUT / "cv_scores.json"))["cv_scores"]
    cv_scores = np.array([cv[i] for i in ids])
    thr = float(np.sort(cv_scores[y == 1])[int(np.floor((1 - TARGET_RECALL) * y.sum()))])
    all_feats = [json.loads(l) for l in open(OUT / "features.jsonl")]
    scores = model.predict_proba(np.array([[f[c] for c in cols] for f in all_feats]))[:, 1]
    with open(OUT / "gate_scores.jsonl", "w") as sink:
        for f, s in zip(all_feats, scores):
            sink.write(json.dumps({"patch_id": f["patch_id"], "site": f["site"], "split": f["split"],
                                   "score": round(float(s), 4), "pass": bool(s >= thr)}) + "\n")
    passed = int((scores >= thr).sum())
    print(f"threshold {thr:.3f}: {passed} of {len(all_feats)} accepted patches pass the gate ({passed / len(all_feats):.0%})")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        compute_features(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "apply":
        apply()
