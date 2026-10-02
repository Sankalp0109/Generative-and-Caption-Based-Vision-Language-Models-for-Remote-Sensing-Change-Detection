#!/usr/bin/env python3
"""Browser page of the trial's filtered patch-pairs: data/trial/index.html.

    python cluster/trial_gallery.py [results.jsonl]      # default: data/trial/results_ada_8bit.jsonl

Tabs for kept / rejected / all. Each card shows before | after, the model's haze and cloud
ratings, and, for the 100 test pairs, whether the verdict matches the adjudicated reference.
Click a card for a large viewer (side by side or blink); arrow keys move, Esc closes.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

TRIAL = Path("data/trial")


def main() -> None:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else TRIAL / "results_ada_8bit.jsonl"
    results = {r["id"]: r for r in map(json.loads, open(results_path))}
    reference = {r["id"]: r for r in map(json.loads, open(TRIAL / "reference_labels_adjudicated.jsonl"))}
    pairs = []
    for row in map(json.loads, open(TRIAL / "manifest.jsonl")):
        res = results.get(row["id"])
        if not res:
            continue
        ref = reference.get(row["id"])
        truth = None
        if ref and not ref["snow"] and not ref["uncertain"]:
            truth = "obstructed" if ref["before_obstructed"] or ref["after_obstructed"] else "clear"
        agrees = None if truth is None else ((res["verdict"] == "reject") == (truth == "obstructed"))
        observation = ""
        try:
            observation = json.loads(res["raw"][res["raw"].find("{"):res["raw"].rfind("}") + 1]).get("observation", "")
        except ValueError:
            pass
        pairs.append({"id": row["id"], "state": row["state"], "split": row["split"], "verdict": res["verdict"],
                      "ratings": res["ratings"], "truth": truth, "agrees": agrees, "observation": observation,
                      "before": row["before"], "after": row["after"],
                      "dates": f"{row['before_date']} → {row['after_date']}"})
    kept = sum(p["verdict"] == "keep" for p in pairs)
    data = json.dumps(pairs).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Haze Filter Trial</title>
<style>
:root {{ --bg:#f6f5f2; --panel:#fff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e2e0da; --accent:#2a6f97;
  --keep:#2a9d8f; --reject:#d1495b; --warn:#b5651d; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141416; --panel:#1e1e21; --ink:#ececef;
  --muted:#9a9aa2; --line:#2e2e33; --accent:#7fb8d8; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 16px;
  display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; }}
h1 {{ font-size:17px; margin:0 6px 0 0; }}
.tabs button, select {{ font:inherit; color:inherit; background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:5px 11px; cursor:pointer; }}
.tabs button.on {{ border-color:var(--accent); box-shadow:inset 0 -2px 0 var(--accent); font-weight:600; }}
.muted {{ color:var(--muted); }}
main {{ padding:14px 16px 40px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:12px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; cursor:pointer; }}
.card:hover {{ border-color:var(--accent); }}
.card .imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:2px; background:var(--line); }}
.card img {{ width:100%; aspect-ratio:1; display:block; object-fit:cover; }}
.meta {{ padding:7px 9px; font-size:12.5px; display:flex; flex-direction:column; gap:5px; }}
.meta .tag + .tag, .meta br + .tag {{ margin-left:4px; }}
.tag {{ font-size:11px; padding:1px 7px; border-radius:10px; color:#fff; white-space:nowrap; display:inline-block; }}
.t-keep {{ background:var(--keep); }} .t-reject {{ background:var(--reject); }} .t-miss {{ background:var(--warn); }}
.t-ok {{ background:#6b8f71; }}
#viewer {{ position:fixed; inset:0; background:rgba(10,10,12,.95); color:#eee; z-index:10; display:none; flex-direction:column; }}
#viewer.open {{ display:flex; }}
.vbar {{ display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; padding:10px 16px; border-bottom:1px solid #333; }}
.vbar button {{ font:inherit; background:#222; color:#eee; border:1px solid #444; border-radius:6px; padding:5px 11px; cursor:pointer; }}
.vbar button.on {{ border-color:#7fb8d8; background:#23343f; }}
.stage {{ flex:1; min-height:0; display:flex; gap:10px; align-items:center; justify-content:center; padding:10px; }}
.stage figure {{ margin:0; height:100%; display:flex; flex-direction:column; align-items:center; min-width:0; }}
.stage img {{ max-height:calc(100% - 22px); max-width:100%; object-fit:contain; image-rendering:pixelated; }}
.stage figcaption {{ font-size:12px; color:#bbb; height:22px; line-height:22px; }}
.obs {{ color:#bbb; font-size:13px; padding:0 16px 10px; }}
.spacer {{ flex:1; }}
@media (max-width:700px) {{ .stage {{ flex-direction:column; }} .stage figure {{ height:auto; width:100%; }} }}
</style></head><body>
<header>
  <h1>Haze Filter Trial</h1>
  <span class="tabs"><button data-tab="keep" class="on">Kept ({kept})</button><button data-tab="reject">Rejected ({len(pairs) - kept})</button><button data-tab="all">All ({len(pairs)})</button></span>
  <select id="fState" aria-label="Filter by state"><option value="">All states</option></select>
  <select id="fCheck" aria-label="Filter by reference"><option value="">All pairs</option>
    <option value="wrong">Disagrees with reference</option><option value="right">Agrees with reference</option></select>
  <span class="spacer"></span>
  <span class="muted">Qwen3-VL-8B, 8-bit, on Ada · {html.escape(results_path.name)}</span>
</header>
<main><p class="muted" id="count"></p><div class="grid" id="grid"></div></main>
<div id="viewer" role="dialog" aria-modal="true" aria-label="Pair viewer">
  <div class="vbar"><b id="vTitle"></b><span id="vSub" class="muted"></span><span class="spacer"></span>
    <button data-mode="side" class="on">Side by side</button><button data-mode="blink">Blink</button>
    <button id="prev">← Prev</button><span id="pos"></span><button id="next">Next →</button><button id="close">Close ✕</button></div>
  <div class="obs" id="obs"></div>
  <div class="stage" id="stage"></div>
</div>
<script>
const PAIRS = {data};
let tab = "keep", view = [], cur = -1, mode = "side", timer = null;
const $ = id => document.getElementById(id);
[...new Set(PAIRS.map(p => p.state))].sort().forEach(s => {{ const o = document.createElement("option"); o.value = o.textContent = s; $("fState").append(o); }});
const rate = r => r ? `B haze ${{r.before.haze}} cloud ${{r.before.cloud}} · A haze ${{r.after.haze}} cloud ${{r.after.cloud}}` : "unreadable";
function refTag(p) {{
  if (p.agrees === null) return "";
  return p.agrees ? `<span class="tag t-ok">matches reference (${{p.truth}})</span>`
                  : `<span class="tag t-miss">${{p.verdict === "keep" ? "missed haze" : "clean, wrongly rejected"}}</span>`;
}}
function refresh() {{
  const fs = $("fState").value, fc = $("fCheck").value;
  view = PAIRS.filter(p => (tab === "all" || p.verdict === tab) && (!fs || p.state === fs) &&
    (!fc || (fc === "wrong" ? p.agrees === false : p.agrees === true)));
  const grid = $("grid"); grid.textContent = "";
  view.forEach((p, i) => {{
    const c = document.createElement("div"); c.className = "card"; c.tabIndex = 0;
    c.innerHTML = `<div class="imgs"><img loading="lazy" src="${{p.before}}" alt="before"><img loading="lazy" src="${{p.after}}" alt="after"></div>
      <div class="meta"><div><b></b><br><span class="muted"></span></div>
      <div><span class="tag t-${{p.verdict === "keep" ? "keep" : "reject"}}">${{p.verdict}}</span> ${{refTag(p)}}</div></div>`;
    c.querySelector("b").textContent = p.state;
    c.querySelector(".meta .muted").textContent = `${{p.dates}} · ${{rate(p.ratings)}}`;
    c.onclick = () => open(i); c.onkeydown = e => {{ if (e.key === "Enter") open(i); }};
    grid.append(c);
  }});
  $("count").textContent = `Showing ${{view.length}} pairs. Each image is a 256 × 256 px patch (2.56 km), before on the left, after on the right.`;
}}
function img(src, alt) {{ const i = new Image(); i.src = src; i.alt = alt; return i; }}
function render() {{
  const p = view[cur]; clearInterval(timer);
  document.querySelectorAll(".vbar [data-mode]").forEach(b => b.classList.toggle("on", b.dataset.mode === mode));
  $("vTitle").textContent = `${{p.state}} · ${{p.verdict.toUpperCase()}}`;
  $("vSub").textContent = `${{p.id}} · ${{p.dates}} · ${{rate(p.ratings)}}`;
  $("obs").textContent = p.observation ? `Model: "${{p.observation}}"` : "";
  $("pos").textContent = `${{cur + 1}} / ${{view.length}}`;
  const st = $("stage"); st.textContent = "";
  if (mode === "side") {{
    [["before", "Before"], ["after", "After"]].forEach(([k, cap]) => {{ const f = document.createElement("figure");
      f.append(img(p[k], cap)); const c = document.createElement("figcaption"); c.textContent = cap; f.append(c); st.append(f); }});
  }} else {{
    const f = document.createElement("figure"), c = document.createElement("figcaption"); let after = false;
    const a = img(p.before, "before"), b = img(p.after, "after");
    const show = () => {{ f.replaceChildren(after ? b : a, c); c.textContent = after ? "After" : "Before"; }};
    show(); st.append(f); timer = setInterval(() => {{ after = !after; show(); }}, 800);
  }}
}}
function open(i) {{ cur = i; $("viewer").classList.add("open"); render(); }}
function close() {{ clearInterval(timer); $("viewer").classList.remove("open"); }}
function go(d) {{ if (view[cur + d]) {{ cur += d; render(); }} }}
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {{ tab = b.dataset.tab;
  document.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("on", x === b)); refresh(); }});
document.querySelectorAll(".vbar [data-mode]").forEach(b => b.onclick = () => {{ mode = b.dataset.mode; render(); }});
["fState", "fCheck"].forEach(id => $(id).onchange = refresh);
$("prev").onclick = () => go(-1); $("next").onclick = () => go(1); $("close").onclick = close;
document.addEventListener("keydown", e => {{ if (!$("viewer").classList.contains("open")) return;
  if (e.key === "Escape") close(); else if (e.key === "ArrowRight") go(1); else if (e.key === "ArrowLeft") go(-1);
  else if (e.key === "b") {{ mode = "blink"; render(); }} else if (e.key === "s") {{ mode = "side"; render(); }} }});
refresh();
</script></body></html>"""
    (TRIAL / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(pairs)} pairs ({kept} kept) -> {TRIAL / 'index.html'}")


if __name__ == "__main__":
    main()
