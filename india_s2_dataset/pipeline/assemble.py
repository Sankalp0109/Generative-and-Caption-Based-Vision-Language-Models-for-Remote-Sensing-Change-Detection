#!/usr/bin/env python3
"""Final assembly of data/FINAL: consistency checks, 65/35 selection, split by site, manifest, report, gallery.

    python -m pipeline.assemble

Reads every patch's meta.json (accepted/ and rejected/*/), so the manifest always describes the
folders exactly. Then:
* checks every accepted patch: before/after GeoTIFFs open, 256 x 256 x 3, identical CRS and transform;
* selects 50% changed / 50% unchanged from the accepted patches ("selected" flag; nothing removed);
* splits the selected patches train/val/test 80/10/10 BY SITE, stratified by state;
* writes manifest.jsonl + manifest.csv, REPORT.md and review.html.
"""
from __future__ import annotations

import csv
import html
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import rasterio

ROOT = Path(__file__).resolve().parent.parent
FINAL = Path(os.environ.get("FINAL_DIR", ROOT / "data" / "FINAL"))
TARGET = 15000
CHANGED_SHARE = 0.50   # 50/50 like LEVIR-CC; 65/35 taught a 'there is always change' prior
SEED = 2026


def load_records() -> list[dict]:
    records = []
    for meta in sorted(list((FINAL / "accepted").glob("*/meta.json")) + list((FINAL / "rejected").glob("*/*/meta.json"))):
        record = json.loads(meta.read_text())
        record["folder"] = str(meta.parent.relative_to(FINAL))
        records.append(record)
    return records


def check_grid(record: dict) -> str | None:
    folder = FINAL / record["folder"]
    try:
        with rasterio.open(folder / "before.tif") as b, rasterio.open(folder / "after.tif") as a:
            if (b.width, b.height, b.count) != (256, 256, 3) or (a.width, a.height, a.count) != (256, 256, 3):
                return "shape"
            if b.crs != a.crs or b.transform != a.transform:
                return "grid"
            if list(b.transform)[:6] != [round(v, 6) if isinstance(v, float) else v for v in record["transform"]] and \
                    any(abs(x - y) > 1e-6 for x, y in zip(list(b.transform)[:6], record["transform"])):
                return "transform_label"
    except rasterio.errors.RasterioIOError:
        return "unreadable"
    for name in ("before.png", "after.png"):
        if not (folder / name).exists():
            return f"missing_{name}"
    return None


def previous_state() -> tuple[dict[str, str], set[str]]:
    """Site splits and selected patch ids from the last assembly, so a re-run keeps them stable."""
    splits_file = FINAL / "site_splits.json"
    splits = json.loads(splits_file.read_text()) if splits_file.exists() else {}
    selected = set()
    if (FINAL / "manifest.jsonl").exists():
        for line in open(FINAL / "manifest.jsonl"):
            r = json.loads(line)
            if r.get("selected"):
                selected.add(r["patch_id"])
                if r.get("split") and r["site"] not in splits:
                    splits[r["site"]] = r["split"]
    return splits, selected


def select_and_split(accepted: list[dict]) -> None:
    """Balanced selection (CHANGED_SHARE changed) and a train/val/test split by site.

    Stable across re-runs: previously selected patches stay selected (more are added as needed)
    and sites keep their split; only sites never assigned before are given one, per state, so
    that each state stays near 80/10/10.
    """
    rng = random.Random(SEED)
    old_splits, old_selected = previous_state()
    changed = [r for r in accepted if r["change"]["changed"] and r.get("category") != "holdout"]
    unchanged = [r for r in accepted if not r["change"]["changed"] and r.get("category") != "holdout"]
    n_unchanged = min(len(unchanged), round(len(changed) * (1 - CHANGED_SHARE) / CHANGED_SHARE))
    keep = [r for r in unchanged if r["patch_id"] in old_selected][:n_unchanged]
    rest = [r for r in unchanged if r["patch_id"] not in old_selected]
    extra = rng.sample(rest, max(0, n_unchanged - len(keep)))
    chosen = {r["patch_id"] for r in changed} | {r["patch_id"] for r in keep + extra}

    by_state = defaultdict(set)
    for r in accepted:
        if r["patch_id"] in chosen:
            by_state[r["state"]].add(r["site"])
    split_of_site = dict(old_splits)
    for state, sites in by_state.items():
        new_sites = sorted(s for s in sites if s not in split_of_site)
        rng.shuffle(new_sites)
        counts = Counter(split_of_site[s] for s in sites if s in split_of_site)
        total = len(sites)
        for site in new_sites:
            want_test = max(1, round(total * 0.1)) if total >= 3 else 0
            want_val = max(1, round(total * 0.1)) if total >= 3 else 0
            split = "test" if counts["test"] < want_test else "val" if counts["val"] < want_val else "train"
            split_of_site[site] = split
            counts[split] += 1
    for r in accepted:
        r["selected"] = r["patch_id"] in chosen
        r["split"] = split_of_site.get(r["site"]) if r["selected"] else None
    (FINAL / "site_splits.json").write_text(json.dumps(split_of_site, indent=1, sort_keys=True))


def write_meta(record: dict) -> None:
    path = FINAL / record["folder"] / "meta.json"
    data = {k: v for k, v in record.items() if k != "folder"}
    path.write_text(json.dumps(data, indent=1))


def flat(record: dict) -> dict:
    model = record.get("model") or {}
    ratings = model.get("ratings") or {}
    return {
        "patch_id": record["patch_id"], "status": record["status"], "stage": record.get("stage"),
        "reason": record.get("reason"), "selected": record.get("selected", False), "split": record.get("split"),
        "site": record["site"], "state": record["state"], "region": record["region"], "category": record.get("category"),
        "row": record["row"], "col": record["col"], "tile": record["tile"], "block": record["block"], "crs": record["crs"],
        "west": record["bounds_wgs84"][0], "south": record["bounds_wgs84"][1], "east": record["bounds_wgs84"][2],
        "north": record["bounds_wgs84"][3], "before_date": record["before"]["date"], "after_date": record["after"]["date"],
        "before_item": record["before"]["item_id"], "after_item": record["after"]["item_id"],
        "shift_px": record["alignment"]["shift_px"], **{f"scl_{k}": v for k, v in record["screen"].items()},
        "changed": record["change"]["changed"], "changed_fraction": record["change"]["changed_fraction"],
        "mean_abs_diff": record["change"]["mean_abs_diff"],
        "model_verdict": model.get("verdict"),
        **{f"model_{role}_{k}": (ratings.get(role) or {}).get(k) for role in ("before", "after") for k in ("haze", "cloud")},
        "folder": record["folder"],
    }


def gpu_stats() -> tuple[float, int]:
    seconds, pairs = 0.0, 0
    for results in (FINAL / "_ada").glob("*/results.jsonl"):
        for line in open(results):
            r = json.loads(line)
            seconds += r.get("seconds") or 0
            pairs += 1
    return seconds, pairs


def gallery(records: list[dict]) -> None:
    rng = random.Random(SEED)
    acc = [r for r in records if r["status"] == "accepted"]
    rej = [r for r in records if r["status"] == "rejected" and (FINAL / r["folder"] / "before.png").exists()]
    sample = rng.sample(acc, min(150, len(acc))) + rng.sample(rej, min(150, len(rej)))
    items = [{"id": r["patch_id"], "status": r["status"], "reason": r.get("reason") or "", "state": r["state"],
              "split": r.get("split") or "", "selected": r.get("selected", False),
              "dates": f"{r['before']['date']} → {r['after']['date']}",
              "changed": r["change"]["changed"], "ratings": (r.get("model") or {}).get("ratings"),
              "before": f"{r['folder']}/before.png", "after": f"{r['folder']}/after.png"} for r in sample]
    data = json.dumps(items).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final Dataset Review</title><style>
:root{{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e2e0da;--ok:#2a9d8f;--bad:#d1495b}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#141416;--panel:#1e1e21;--ink:#ececef;--muted:#9a9aa2;--line:#2e2e33}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
h1{{font-size:17px;margin:0 8px 0 0}} button,select{{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}}
button.on{{box-shadow:inset 0 -2px 0 #2a6f97;font-weight:600}} main{{padding:14px 16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
.card .imgs{{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:var(--line)}} .card img{{width:100%;aspect-ratio:1;display:block;image-rendering:pixelated}}
.meta{{padding:7px 9px;font-size:12px;color:var(--muted)}} .meta b{{color:var(--ink)}}
.tag{{font-size:11px;padding:1px 7px;border-radius:10px;color:#fff;display:inline-block;margin:2px 3px 0 0}} .ok{{background:var(--ok)}} .bad{{background:var(--bad)}} .n{{background:#7a7f88}}
</style></head><body><header><h1>Final Dataset Review</h1>
<button data-f="accepted" class="on">Accepted sample</button><button data-f="rejected">Rejected sample</button>
<select id="st"><option value="">All states</option></select><span class="meta">Random sample: 150 accepted + 150 rejected. Before | after, 2.56 km each.</span></header>
<main><div class="grid" id="g"></div></main><script>
const P={data}; let f="accepted"; const g=document.getElementById("g"), st=document.getElementById("st");
[...new Set(P.map(p=>p.state))].sort().forEach(s=>{{const o=document.createElement("option");o.value=o.textContent=s;st.append(o)}});
const rt=r=>r?`B haze ${{r.before.haze}}/cloud ${{r.before.cloud}} · A haze ${{r.after.haze}}/cloud ${{r.after.cloud}}`:"";
function draw(){{g.textContent="";P.filter(p=>p.status===f&&(!st.value||p.state===st.value)).forEach(p=>{{const c=document.createElement("div");c.className="card";
c.innerHTML=`<div class="imgs"><img loading="lazy" src="${{p.before}}" alt="before"><img loading="lazy" src="${{p.after}}" alt="after"></div><div class="meta"><b></b><br><span></span><br>
<span class="tag ${{p.status==="accepted"?"ok":"bad"}}">${{p.status}}${{p.reason?": "+p.reason:""}}</span>${{p.changed?'<span class="tag n">changed</span>':""}}${{p.split?`<span class="tag n">${{p.split}}</span>`:""}}</div>`;
c.querySelector("b").textContent=p.state; c.querySelector(".meta span").textContent=`${{p.dates}} · ${{rt(p.ratings)}}`; g.append(c)}})}}
document.querySelectorAll("button").forEach(b=>b.onclick=()=>{{f=b.dataset.f;document.querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));draw()}});
st.onchange=draw; draw();</script></body></html>"""
    (FINAL / "review.html").write_text(page, encoding="utf-8")


def main() -> None:
    pending = list((FINAL / "_pending").glob("*/meta.json")) if (FINAL / "_pending").exists() else []
    records = load_records()
    accepted = [r for r in records if r["status"] == "accepted"]
    bad_grid = {}
    for r in accepted:
        problem = check_grid(r)
        if problem:
            bad_grid[r["patch_id"]] = problem
    select_and_split(accepted)
    for r in accepted:
        write_meta(r)
    rows = [flat(r) for r in records]
    with open(FINAL / "manifest.jsonl", "w") as handle:
        for r in records:
            handle.write(json.dumps({k: v for k, v in r.items()}) + "\n")
    with open(FINAL / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gallery(records)

    rejected = [r for r in records if r["status"] == "rejected"]
    selected = [r for r in accepted if r.get("selected")]
    by_reason = Counter((r["stage"], r["reason"].split(":")[0]) for r in rejected)
    by_state = defaultdict(lambda: [0, 0, 0])
    for r in records:
        by_state[r["state"]][0 if r["status"] == "accepted" else 1] += 1
        by_state[r["state"]][2] += bool(r.get("selected"))
    by_region = Counter((r["region"], r["status"]) for r in records)
    seconds, pairs = gpu_stats()
    splits = Counter(r["split"] for r in selected)
    changed_sel = sum(r["change"]["changed"] for r in selected)
    lines = [
        "# Final dataset report",
        "",
        f"Location: `data/FINAL/`. Accepted patches are in `accepted/<patch_id>/`; rejected ones are in "
        f"`rejected/<reason>/<patch_id>/`. Each patch folder has `before.tif`, `after.tif` (raw 8-bit, "
        f"georeferenced, identical grid), `before.png`, `after.png` (display copies), and `meta.json` "
        f"(every label). `manifest.jsonl` and `manifest.csv` list every patch.",
        "",
        "## Totals",
        "",
        "| | Patches |",
        "|---|---|",
        f"| All patches | {len(records)} |",
        f"| **Accepted** | **{len(accepted)}** (target {TARGET}: "
        f"{'met' if len(accepted) >= TARGET else 'SHORT by ' + str(TARGET - len(accepted))}) |",
        f"| Rejected | {len(rejected)} |",
        f"| Selected for the dataset (from accepted) | {len(selected)}: {changed_sel} changed "
        f"({changed_sel / max(len(selected), 1):.0%}), {len(selected) - changed_sel} unchanged |",
        f"| Split (by site, stratified by state) | train {splits.get('train', 0)} · val {splits.get('val', 0)} · "
        f"test {splits.get('test', 0)} |",
        f"| Awaiting model (should be 0) | {len(pending)} |",
        f"| Accepted patches failing the grid/file check (should be 0) | {len(bad_grid)} |",
        "",
        "## Rejections by stage and reason",
        "",
        "| Stage | Reason | Patches |",
        "|---|---|---|",
        *[f"| {stage} | {reason} | {n} |" for (stage, reason), n in by_reason.most_common()],
        "",
        "## By region",
        "",
        "| Region | Accepted | Rejected |",
        "|---|---|---|",
        *[f"| {reg} | {by_region[(reg, 'accepted')]} | {by_region[(reg, 'rejected')]} |"
          for reg in sorted({r['region'] for r in records})],
        "",
        "## By state",
        "",
        "| State | Accepted | Rejected | Selected |",
        "|---|---|---|---|",
        *[f"| {s} | {a} | {rj} | {sel} |" for s, (a, rj, sel) in sorted(by_state.items(), key=lambda kv: -kv[1][0])],
        "",
        "## Ada haze screen",
        "",
        f"- Model: Qwen3-VL-8B-Instruct, 8-bit, 1 × RTX 2080 Ti per job, batches of 8 pairs.",
        f"- {pairs} pairs screened in {seconds / 3600:.1f} GPU-hours ({seconds / max(pairs, 1):.2f} s per pair).",
        "- **Known limitation, from the 100-pair trial:** the 8B model catches about 70% of hazy pairs and "
        "wrongly rejects about 12% of clean ones. Expect some thin haze among the accepted patches.",
        "",
        "## How patches were checked",
        "",
        "1. Download checks: footprint coverage, same tile, cloud ≤ 1% (≤ 3% in cloudy regions), snow ≤ 20%, "
        "water ≤ 70%, season match, whole-image haze and whiteout.",
        "2. Site checks: files intact, identical grid, before/after shift ≤ 0.5 px, no ground overlap with another site.",
        "3. Patch checks (per 256 px patch, from the cloud mask): no-data ≤ 0.1%, cloud/shadow ≤ 1%, snow ≤ 20%, "
        "water ≤ 70%, whiteout ≤ 30%.",
        "4. The Ada vision-model haze/cloud screen (reject at haze ≥ 1 or cloud ≥ 2).",
    ]
    (FINAL / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"assembled: {len(records)} patches, {len(accepted)} accepted, {len(selected)} selected, "
          f"{len(bad_grid)} grid problems, {len(pending)} pending")


if __name__ == "__main__":
    main()
