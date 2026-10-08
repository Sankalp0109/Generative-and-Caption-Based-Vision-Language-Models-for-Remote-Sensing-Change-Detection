#!/usr/bin/env python3
"""Browser gallery for eyeballing downloaded pairs (Stage 5), plus label import into the DB.

    python review_gallery.py                  # render new pairs, rebuild data/review/index.html
    xdg-open data/review/index.html           # open it in the browser
    python review_gallery.py --import-labels ~/Downloads/review_labels.csv

The page shows every pair as before/after thumbnails, filterable by state and label. Click
one for a large viewer: side-by-side, swipe (drag a divider across), blink (flip before and
after in place: the quickest way to see change), and the difference heatmap. Keys 1-4 label
a pair (real change / no change / cloud-haze / bad), arrows move between pairs. Labels are
kept in the browser as you go; "Export CSV" downloads them, and ``--import-labels`` stores
them in the ``review`` table of the download database next to each site.

Rendering is incremental: pairs already rendered are skipped, so rerunning during a long
download only processes the new ones. GeoTIFFs are untouched; the JPEGs are display copies.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sqlite3
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

from aoi_download import quality_flags
from change_visualization import joint_stretch
from stac_select import SCL_BAD

DATA_ROOT = Path("data/sentinel2")
REVIEW_DIR = Path("data/review")
DB_PATH = Path("data/download_state.sqlite")
THUMB_PX = 256
LABELS = [("change", "Real change"), ("nochange", "No change"), ("cloud", "Cloud / haze"), ("bad", "Bad pair")]
RECORD_VERSION = 4          # bump when records gain fields, so cached pairs are re-rendered


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.moveaxis(src.read(), 0, -1) if src.count > 1 else src.read(1)


def _display_pair(before: np.ndarray, after: np.ndarray):
    """True-colour display copies: the colour-preserving shared stretch the model also sees."""
    return [(view * 255).astype("uint8") for view in joint_stretch(before, after, low=1, high=99.5)]


def render_pair(pair_dir: Path, out_dir: Path) -> dict:
    """Write before/after/diff JPEGs (+ thumbnails) for one pair; return its gallery record."""
    meta = json.loads((pair_dir / "pair.json").read_text())
    raw_before, raw_after = _read(pair_dir / "before.tif"), _read(pair_dir / "after.tif")
    quality = quality_flags(raw_before, raw_after)
    before, after = _display_pair(raw_before, raw_after)
    difference = np.abs(after.astype("float32") - before.astype("float32")).mean(axis=2) / 255
    heat = (plt.get_cmap("magma")(np.clip(difference * 2.5, 0, 1))[..., :3] * 255).astype("uint8")
    cloud = np.isin(_read(pair_dir / "scl_before.tif"), SCL_BAD) | np.isin(_read(pair_dir / "scl_after.tif"), SCL_BAD)
    heat[cloud] = (heat[cloud] * 0.4 + np.array([60, 140, 255]) * 0.6).astype("uint8")  # cloud tinted blue

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, array in (("before", before), ("after", after), ("diff", heat)):
        image = Image.fromarray(array)
        image.save(out_dir / f"{name}.jpg", quality=88, optimize=True)
        image.resize((THUMB_PX, THUMB_PX), Image.LANCZOS).save(out_dir / f"{name}_t.jpg", quality=82)

    site, block = meta.get("site", {}), meta.get("block", {})
    record = {
        "id": f"{pair_dir.parent.name}/{pair_dir.name}", "key": pair_dir.parent.name,
        "label": site.get("label") or pair_dir.parent.name, "state": site.get("state") or "?",
        "lon": site.get("lon"), "lat": site.get("lat"), "block": block.get("id", ""),
        "before_date": (meta["before"].get("datetime") or "")[:10],
        "after_date": (meta["after"].get("datetime") or "")[:10],
        "before_cloud": meta["before"].get("aoi_cloud"), "after_cloud": meta["after"].get("aoi_cloud"),
        "gap_days": meta.get("gap_days"), "mean_diff": round(float(difference.mean()), 4),
        "changed_pct": round(float((difference > 0.15).mean() * 100), 1),
        "img": f"pairs/{pair_dir.parent.name}/{pair_dir.name}",
        "version": RECORD_VERSION, **quality,
    }
    (out_dir / "record.json").write_text(json.dumps(record))
    return record


def draw_status_map(db_path: Path, destination: Path) -> bool:
    """India map of every site, coloured by download status."""
    if not db_path.exists():
        return False
    import geopandas as gpd
    from build_aoi_registry import BOUNDARY_PATH
    rows = sqlite3.connect(db_path).execute("SELECT lon, lat, status FROM aoi").fetchall()
    colours = {"pending": "#cfcfcf", "running": "#264653", "done": "#2a9d8f", "selected": "#8ab17d",
               "no_pairs": "#e9c46a", "failed": "#d1495b"}
    figure, axis = plt.subplots(figsize=(8, 9))
    gpd.read_file(BOUNDARY_PATH).boundary.plot(ax=axis, color="#888", linewidth=0.4)
    for status, colour in colours.items():
        points = [(lon, lat) for lon, lat, s in rows if s == status]
        if points:
            axis.scatter(*zip(*points), s=5 if status == "pending" else 12, c=colour,
                         label=f"{status} ({len(points)})", zorder=2 if status == "pending" else 3)
    axis.set_title("Download sites by status")
    axis.legend(loc="lower left", fontsize=8)
    axis.set_aspect("equal")
    axis.set_xticks([]); axis.set_yticks([])
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return True


def build_gallery(data_root: Path = DATA_ROOT, review_dir: Path = REVIEW_DIR, db_path: Path = DB_PATH,
                  force: bool = False) -> dict:
    pairs = sorted(p for p in data_root.glob("*/pair_*") if (p / "pair.json").exists() and p.is_dir()
                   and not p.name.endswith(".tmp"))
    records, rendered = [], 0
    for pair in pairs:
        out_dir = review_dir / "pairs" / pair.parent.name / pair.name
        cached = out_dir / "record.json"
        if cached.exists() and not force and cached.stat().st_mtime >= (pair / "pair.json").stat().st_mtime:
            record = json.loads(cached.read_text())
            if record.get("version") == RECORD_VERSION:
                records.append(record)
                continue
        records.append(render_pair(pair, out_dir))
        rendered += 1
    has_map = draw_status_map(db_path, review_dir / "status_map.png")
    (review_dir / "index.html").write_text(page_html(records, has_map), encoding="utf-8")
    return {"pairs": len(records), "rendered": rendered, "page": review_dir / "index.html"}


def import_labels(csv_path: Path, db_path: Path = DB_PATH) -> int:
    """Store exported review labels in the download DB (``review`` table, one row per pair)."""
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS review (pair_id TEXT PRIMARY KEY, aoi_key TEXT, "
                 "label TEXT, note TEXT, reviewed_at REAL)")
    count = 0
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("label") not in {name for name, _ in LABELS}:
                continue
            conn.execute("INSERT OR REPLACE INTO review VALUES (?, ?, ?, ?, ?)",
                         (row["pair_id"], row["pair_id"].split("/")[0], row["label"],
                          row.get("note", ""), time.time()))
            count += 1
    conn.commit()
    return count


def page_html(records: list[dict], has_map: bool) -> str:
    data = json.dumps(records).replace("</", "<\\/")
    labels = json.dumps(LABELS)
    generated = time.strftime("%Y-%m-%d %H:%M")
    map_block = ('<details class="map"><summary>Site status map</summary>'
                 '<img src="status_map.png" alt="Map of India with every download site coloured by status"></details>'
                 if has_map else "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pair Review</title>
<style>
:root {{ --bg:#f6f5f2; --panel:#ffffff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e2e0da; --accent:#2a6f97;
  --change:#2a9d8f; --nochange:#8a8f98; --cloud:#5b8def; --bad:#d1495b; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141416; --panel:#1e1e21;
  --ink:#ececef; --muted:#9a9aa2; --line:#2e2e33; --accent:#7fb8d8; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line);
  padding:12px 16px; display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; }}
h1 {{ font-size:17px; margin:0 8px 0 0; }}
.stats {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
select, button, input {{ font:inherit; color:inherit; background:var(--panel); border:1px solid var(--line);
  border-radius:6px; padding:5px 9px; }}
button {{ cursor:pointer; }} button:hover {{ border-color:var(--accent); }}
.spacer {{ flex:1; }}
main {{ padding:14px 16px 40px; }}
details.map {{ margin-bottom:14px; }} details.map img {{ max-width:min(560px,100%); display:block; margin-top:8px; border-radius:6px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:12px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; cursor:pointer;
  border-left:4px solid transparent; }}
.card:hover {{ border-color:var(--accent); }}
.card .imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:2px; background:var(--line); }}
.card img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; }}
.card .meta {{ padding:7px 9px; display:flex; justify-content:space-between; gap:6px; font-size:12.5px; }}
.card .meta b {{ font-weight:600; }} .card .meta span {{ color:var(--muted); }}
.tag {{ font-size:11px; padding:1px 7px; border-radius:10px; color:#fff; white-space:nowrap; }}
.l-change {{ border-left-color:var(--change); }} .t-change {{ background:var(--change); }}
.l-nochange {{ border-left-color:var(--nochange); }} .t-nochange {{ background:var(--nochange); }}
.l-cloud {{ border-left-color:var(--cloud); }} .t-cloud {{ background:var(--cloud); }}
.l-bad {{ border-left-color:var(--bad); }} .t-bad {{ background:var(--bad); }}
.t-flag {{ background:#b5651d; display:inline-block; margin-top:2px; }}
#viewer {{ position:fixed; inset:0; background:rgba(10,10,12,.94); color:#eee; z-index:10; display:none;
  flex-direction:column; }}
#viewer.open {{ display:flex; }}
.vbar {{ display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; padding:10px 16px; border-bottom:1px solid #333; }}
.vbar select, .vbar button, .vbar input {{ background:#222; border-color:#444; color:#eee; }}
.vbar button.on {{ border-color:#7fb8d8; background:#23343f; }}
.vtitle {{ font-weight:600; }} .vsub {{ color:#aaa; font-size:13px; }}
.stage {{ flex:1; min-height:0; display:flex; align-items:center; justify-content:center; gap:10px; padding:10px; }}
.stage figure {{ margin:0; height:100%; display:flex; flex-direction:column; align-items:center; min-width:0; }}
.stage figure img {{ max-height:calc(100% - 22px); max-width:100%; object-fit:contain; image-rendering:auto; }}
.stage figcaption {{ font-size:12px; color:#bbb; height:22px; line-height:22px; }}
.swipe {{ position:relative; height:100%; aspect-ratio:1; max-width:100%; user-select:none; touch-action:none; }}
.swipe img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }}
.swipe .top {{ clip-path:inset(0 0 0 var(--cut,50%)); }}
.swipe .handle {{ position:absolute; top:0; bottom:0; left:var(--cut,50%); width:2px; background:#fff; box-shadow:0 0 4px #000; }}
.swipe .lab {{ position:absolute; top:6px; font-size:12px; background:rgba(0,0,0,.55); padding:1px 6px; border-radius:4px; }}
.labels button {{ min-width:0; }}
.labels button.sel-change {{ background:var(--change); border-color:var(--change); }}
.labels button.sel-nochange {{ background:var(--nochange); border-color:var(--nochange); }}
.labels button.sel-cloud {{ background:var(--cloud); border-color:var(--cloud); }}
.labels button.sel-bad {{ background:var(--bad); border-color:var(--bad); }}
.help {{ color:#999; font-size:12px; }}
@media (max-width:700px) {{ .stage {{ flex-direction:column; }} .stage figure {{ height:auto; width:100%; }} .help {{ display:none; }} }}
</style></head>
<body>
<header>
  <h1>Pair Review</h1>
  <span class="stats" id="stats"></span>
  <select id="fState" aria-label="Filter by state"><option value="">All states</option></select>
  <select id="fLabel" aria-label="Filter by label"><option value="">All labels</option><option value="none">Unlabelled</option><option value="flagged">⚠ Auto-flagged</option></select>
  <select id="sort" aria-label="Sort"><option value="order">Download order</option><option value="state">State</option>
    <option value="diff">Most different first</option><option value="diffasc">Least different first</option></select>
  <span class="spacer"></span>
  <button id="export">Export CSV</button>
  <span class="stats">built {html.escape(generated)}</span>
</header>
<main>
  {map_block}
  <div class="grid" id="grid"></div>
</main>
<div id="viewer" role="dialog" aria-modal="true" aria-label="Pair viewer">
  <div class="vbar">
    <span class="vtitle" id="vTitle"></span><span class="vsub" id="vSub"></span>
    <span class="spacer"></span>
    <span class="modes"><button data-mode="side">Side by side</button><button data-mode="swipe">Swipe</button>
      <button data-mode="blink">Blink</button><button data-mode="diff">Difference</button></span>
    <button id="close" aria-label="Close viewer">Close ✕</button>
  </div>
  <div class="stage" id="stage"></div>
  <div class="vbar">
    <span class="labels" id="labelBtns"></span>
    <input id="note" placeholder="note (optional)" size="34" aria-label="Note">
    <span class="spacer"></span>
    <button id="prev">← Prev</button><span class="vsub" id="pos"></span><button id="next">Next →</button>
    <span class="help">Keys: 1-4 label · ←/→ move · S/W/B/D modes · Esc close</span>
  </div>
</div>
<script>
const PAIRS = {data};
const LABELS = {labels};
const STORE = "pair-review-labels-v1";
let labels = {{}};
try {{ labels = JSON.parse(localStorage.getItem(STORE) || "{{}}"); }} catch (e) {{ labels = {{}}; }}
function save() {{ try {{ localStorage.setItem(STORE, JSON.stringify(labels)); }} catch (e) {{}} }}
PAIRS.forEach((p, i) => p.order = i);
const $ = id => document.getElementById(id);
let view = [], cur = -1, mode = "blink", blinkTimer = null, blinkAfter = true;

const states = [...new Set(PAIRS.map(p => p.state))].sort();
states.forEach(s => {{ const o = document.createElement("option"); o.value = s; o.textContent = s; $("fState").append(o); }});
LABELS.forEach(([k, name]) => {{ const o = document.createElement("option"); o.value = k; o.textContent = name; $("fLabel").append(o); }});

function pct(v) {{ return v == null ? "?" : (v * 100).toFixed(1) + "%"; }}
function labelName(k) {{ return (LABELS.find(l => l[0] === k) || [k, k])[1]; }}

function refresh() {{
  const fs = $("fState").value, fl = $("fLabel").value, sort = $("sort").value;
  view = PAIRS.filter(p => (!fs || p.state === fs) &&
    (!fl || (fl === "none" ? !labels[p.id] : fl === "flagged" ? (p.flags || []).length : (labels[p.id] || {{}}).label === fl)));
  if (sort === "state") view.sort((a, b) => a.state.localeCompare(b.state) || a.order - b.order);
  else if (sort === "diff") view.sort((a, b) => b.mean_diff - a.mean_diff);
  else if (sort === "diffasc") view.sort((a, b) => a.mean_diff - b.mean_diff);
  else view.sort((a, b) => a.order - b.order);
  const grid = $("grid"); grid.textContent = "";
  view.forEach((p, i) => {{
    const lab = (labels[p.id] || {{}}).label;
    const card = document.createElement("div");
    card.className = "card" + (lab ? " l-" + lab : "");
    card.tabIndex = 0;
    card.innerHTML = `<div class="imgs"><img loading="lazy" src="${{p.img}}/before_t.jpg" alt="before">
      <img loading="lazy" src="${{p.img}}/after_t.jpg" alt="after"></div>
      <div class="meta"><div><b></b><br><span></span></div><div>${{lab ? `<span class="tag t-${{lab}}">${{labelName(lab)}}</span>` : ""}}
      ${{(p.flags || []).map(f => `<span class="tag t-flag">⚠ ${{f}}</span>`).join(" ")}}</div></div>`;
    card.querySelector("b").textContent = p.label;
    card.querySelector(".meta span").textContent = `${{p.before_date.slice(0, 7)}} → ${{p.after_date.slice(0, 7)}} · Δ ${{p.changed_pct}}%`;
    card.onclick = () => openViewer(i);
    card.onkeydown = e => {{ if (e.key === "Enter") openViewer(i); }};
    grid.append(card);
  }});
  const done = PAIRS.filter(p => labels[p.id]).length;
  const flagged = PAIRS.filter(p => (p.flags || []).length).length;
  const counts = LABELS.map(([k, n]) => `${{n}} ${{PAIRS.filter(p => (labels[p.id] || {{}}).label === k).length}}`).join(" · ");
  $("stats").textContent = `${{PAIRS.length}} pairs · ⚠ ${{flagged}} auto-flagged · ${{done}} reviewed (${{counts}}) · showing ${{view.length}}`;
}}

function img(src, alt) {{ const i = new Image(); i.src = src; i.alt = alt; return i; }}
function render() {{
  const p = view[cur]; if (!p) return;
  clearInterval(blinkTimer); blinkTimer = null;
  document.querySelectorAll(".modes button").forEach(b => b.classList.toggle("on", b.dataset.mode === mode));
  $("vTitle").textContent = `${{p.label}} · ${{p.state}}`;
  $("vSub").textContent = `${{p.block}} · before ${{p.before_date}} (cloud ${{pct(p.before_cloud)}}) · after ${{p.after_date}} (cloud ${{pct(p.after_cloud)}}) · season gap ${{p.gap_days}} d · ${{p.lat?.toFixed(3)}}, ${{p.lon?.toFixed(3)}}`;
  $("pos").textContent = `${{cur + 1}} / ${{view.length}}`;
  if ((p.flags || []).length) $("vSub").textContent += ` · ⚠ auto-flag: ${{p.flags.join(", ")}} (dark floor ${{p.before_dark_blue}} vs ${{p.after_dark_blue}}, white ${{pct(p.before_white)}} / ${{pct(p.after_white)}})`;
  const stage = $("stage"); stage.textContent = "";
  if (mode === "side") {{
    [["before", "Before " + p.before_date], ["after", "After " + p.after_date]].forEach(([n, cap]) => {{
      const f = document.createElement("figure"); f.append(img(`${{p.img}}/${{n}}.jpg`, cap));
      const c = document.createElement("figcaption"); c.textContent = cap; f.append(c); stage.append(f); }});
  }} else if (mode === "swipe") {{
    const s = document.createElement("div"); s.className = "swipe";
    const a = img(`${{p.img}}/before.jpg`, "before"), b = img(`${{p.img}}/after.jpg`, "after"); b.className = "top";
    const h = document.createElement("div"); h.className = "handle";
    const l1 = document.createElement("span"); l1.className = "lab"; l1.style.left = "6px"; l1.textContent = "Before " + p.before_date;
    const l2 = document.createElement("span"); l2.className = "lab"; l2.style.right = "6px"; l2.textContent = "After " + p.after_date;
    s.append(a, b, h, l1, l2); stage.append(s);
    const move = e => {{ const r = s.getBoundingClientRect(); const x = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
      s.style.setProperty("--cut", (x * 100) + "%"); }};
    s.onpointerdown = e => {{ s.setPointerCapture(e.pointerId); move(e); }};
    s.onpointermove = e => {{ if (e.buttons) move(e); }};
  }} else if (mode === "blink") {{
    const f = document.createElement("figure");
    const a = img(`${{p.img}}/before.jpg`, "before"), b = img(`${{p.img}}/after.jpg`, "after");
    const c = document.createElement("figcaption");
    const show = () => {{ f.replaceChildren(blinkAfter ? b : a, c);
      c.textContent = (blinkAfter ? "After " + p.after_date : "Before " + p.before_date) + "  (blinking; B to pause)"; }};
    blinkAfter = false; show(); stage.append(f);
    blinkTimer = setInterval(() => {{ blinkAfter = !blinkAfter; show(); }}, 800);
  }} else {{
    const f = document.createElement("figure"); f.append(img(`${{p.img}}/diff.jpg`, "difference"));
    const c = document.createElement("figcaption"); c.textContent = "Difference (brighter = more change; blue tint = cloud/shadow in SCL)";
    f.append(c); stage.append(f);
  }}
  const lab = labels[p.id] || {{}};
  document.querySelectorAll("#labelBtns button").forEach(b => b.className = lab.label === b.dataset.k ? "sel-" + b.dataset.k : "");
  $("note").value = lab.note || "";
  [view[cur + 1], view[cur - 1]].forEach(q => {{ if (q) ["before", "after"].forEach(n => img(`${{q.img}}/${{n}}.jpg`, "")); }});
}}
function openViewer(i) {{ cur = i; $("viewer").classList.add("open"); render(); }}
function closeViewer() {{ clearInterval(blinkTimer); $("viewer").classList.remove("open"); refresh(); }}
function go(d) {{ if (view[cur + d]) {{ cur += d; render(); }} }}
function setLabel(k) {{
  const p = view[cur]; const prev = labels[p.id] || {{}};
  if (prev.label === k) delete labels[p.id]; else labels[p.id] = {{label: k, note: $("note").value, at: Date.now()}};
  save(); render();
}}
LABELS.forEach(([k, n], i) => {{ const b = document.createElement("button"); b.dataset.k = k; b.textContent = `${{i + 1}} ${{n}}`;
  b.onclick = () => setLabel(k); $("labelBtns").append(b); }});
$("note").oninput = () => {{ const p = view[cur]; if (labels[p.id]) {{ labels[p.id].note = $("note").value; save(); }} }};
document.querySelectorAll(".modes button").forEach(b => b.onclick = () => {{ mode = b.dataset.mode; render(); }});
$("close").onclick = closeViewer; $("prev").onclick = () => go(-1); $("next").onclick = () => go(1);
["fState", "fLabel", "sort"].forEach(id => $(id).onchange = refresh);
document.addEventListener("keydown", e => {{
  if (!$("viewer").classList.contains("open") || e.target === $("note")) return;
  const k = e.key.toLowerCase();
  if (k === "escape") closeViewer(); else if (k === "arrowright") go(1); else if (k === "arrowleft") go(-1);
  else if (k >= "1" && k <= String(LABELS.length)) setLabel(LABELS[+k - 1][0]);
  else if (k === "s") {{ mode = "side"; render(); }} else if (k === "w") {{ mode = "swipe"; render(); }}
  else if (k === "d") {{ mode = "diff"; render(); }}
  else if (k === "b") {{ if (mode === "blink" && blinkTimer) {{ clearInterval(blinkTimer); blinkTimer = null; }} else {{ mode = "blink"; render(); }} }}
}});
$("export").onclick = () => {{
  const rows = [["pair_id", "state", "label", "note", "before_date", "after_date", "changed_pct"]];
  PAIRS.forEach(p => {{ const l = labels[p.id]; if (l) rows.push([p.id, p.state, l.label, l.note || "", p.before_date, p.after_date, p.changed_pct]); }});
  const csv = rows.map(r => r.map(v => `"${{String(v).replace(/"/g, '""')}}"`).join(",")).join("\\n");
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], {{type: "text/csv"}}));
  a.download = "review_labels.csv"; a.click();
}};
refresh();
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=REVIEW_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--force", action="store_true", help="re-render every pair")
    parser.add_argument("--import-labels", type=Path, metavar="CSV", help="store exported labels in the DB")
    args = parser.parse_args()
    if args.import_labels:
        print(f"imported {import_labels(args.import_labels, args.db)} labels into {args.db} (table: review)")
        return
    result = build_gallery(args.data_root, args.out, args.db, args.force)
    print(f"{result['pairs']} pairs ({result['rendered']} newly rendered) -> {result['page']}")


if __name__ == "__main__":
    main()
