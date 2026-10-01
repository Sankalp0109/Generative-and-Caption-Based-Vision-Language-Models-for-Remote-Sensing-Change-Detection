#!/usr/bin/env python3
"""Browse-everything gallery of data/FINAL: samples from every state, plus rejected examples.

    python -m pipeline.gallery_all            # -> data/FINAL/gallery.html

* Selected (the dataset): up to 4 changed + 4 unchanged pairs from every state.
* Rejected: up to 24 examples of each rejection reason, spread across states.
Each card: before | after (the 256 px patch at native pixels), state, dates, split, change share,
model haze ratings, and a Google Maps satellite link to the same spot for a high-resolution
comparison. Click a card for a large viewer: side by side or blink; arrows move, Esc closes.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "FINAL"
PER_STATE = 4
PER_REASON = 24
SEED = 11


def main() -> None:
    rows = [json.loads(line) for line in open(FINAL / "manifest.jsonl")]
    rng = random.Random(SEED)
    picked = []
    by_state = defaultdict(lambda: {True: [], False: []})
    for r in rows:
        if r.get("selected"):
            by_state[r["state"]][r["change"]["changed"]].append(r)
    for state in sorted(by_state):
        for changed in (True, False):
            pool = by_state[state][changed]
            picked += [(r, "changed" if changed else "unchanged") for r in rng.sample(pool, min(PER_STATE, len(pool)))]
    by_reason = defaultdict(list)
    for r in rows:
        if r["status"] == "rejected" and (FINAL / r["folder"] / "before.png").exists():
            by_reason[r["reason"].split(":")[0]].append(r)
    for reason, pool in sorted(by_reason.items()):
        rng.shuffle(pool)
        seen, sample = Counter(), []
        for r in pool:                     # spread across states: at most 2 per state per reason
            if seen[r["state"]] < 2:
                sample.append(r)
                seen[r["state"]] += 1
            if len(sample) == PER_REASON:
                break
        picked += [(r, f"rejected: {reason}") for r in sample]

    items = []
    for r, group in picked:
        w, s, e, n = r["bounds_wgs84"]
        lat, lon = (s + n) / 2, (w + e) / 2
        model = r.get("model") or {}
        items.append({
            "id": r["patch_id"], "group": group, "state": r["state"], "region": r["region"],
            "split": r.get("split") or "", "category": r.get("category") or "",
            "dates": f"{r['before']['date']} → {r['after']['date']}",
            "changed_pct": round(100 * r["change"]["changed_fraction"], 1),
            "ratings": model.get("ratings"), "observation": model.get("observation") or "",
            "reason": r.get("reason") or "", "lat": round(lat, 5), "lon": round(lon, 5),
            "maps": f"https://www.google.com/maps/@{lat:.5f},{lon:.5f},15z/data=!3m1!1e3",
            "before": f"{r['folder']}/before.png", "after": f"{r['folder']}/after.png"})
    selected = sum(1 for r in rows if r.get("selected"))
    counts = Counter(g for _, g in picked)
    data = json.dumps(items).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset Gallery</title><style>
:root{{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e2e0da;--accent:#2a6f97;--ch:#2a9d8f;--un:#7a7f88;--rej:#d1495b}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#141416;--panel:#1e1e21;--ink:#ececef;--muted:#9a9aa2;--line:#2e2e33;--accent:#7fb8d8}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center}}
h1{{font-size:17px;margin:0 6px 0 0}} select,button,label{{font:inherit;color:inherit}}
select,button{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 9px;cursor:pointer}}
.muted{{color:var(--muted)}} main{{padding:12px 16px 40px}}
.note{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:12px;max-width:1000px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer;border-top:4px solid var(--line)}}
.card:hover{{border-color:var(--accent)}} .g-changed{{border-top-color:var(--ch)}} .g-unchanged{{border-top-color:var(--un)}} .g-rej{{border-top-color:var(--rej)}}
.imgs{{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:var(--line)}} .imgs img{{width:100%;aspect-ratio:1;display:block}}
.meta{{padding:7px 9px;font-size:12.5px}} .meta b{{font-size:13px}} .tag{{font-size:11px;padding:1px 7px;border-radius:10px;color:#fff;display:inline-block;margin:3px 3px 0 0}}
.t-changed{{background:var(--ch)}} .t-unchanged{{background:var(--un)}} .t-rej{{background:var(--rej)}} .t-split{{background:#5a6b7d}}
a{{color:var(--accent)}}
#viewer{{position:fixed;inset:0;background:rgba(10,10,12,.95);color:#eee;z-index:10;display:none;flex-direction:column}} #viewer.open{{display:flex}}
.vbar{{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;padding:10px 16px;border-bottom:1px solid #333}}
.vbar button{{background:#222;color:#eee;border:1px solid #444}} .vbar button.on{{border-color:#7fb8d8;background:#23343f}} .vbar a{{color:#9cd0ee}}
.stage{{flex:1;min-height:0;display:flex;gap:10px;align-items:center;justify-content:center;padding:10px}}
.stage figure{{margin:0;height:100%;display:flex;flex-direction:column;align-items:center;min-width:0}}
.stage img{{max-height:calc(100% - 22px);max-width:100%;object-fit:contain}} .stage.px img{{image-rendering:pixelated}}
.stage figcaption{{font-size:12px;color:#bbb;height:22px;line-height:22px}} .obs{{padding:0 16px 8px;color:#bbb;font-size:13px}}
.spacer{{flex:1}} @media (max-width:700px){{.stage{{flex-direction:column}} .stage figure{{height:auto;width:100%}}}}
</style></head><body>
<header><h1>Dataset Gallery</h1>
<select id="fGroup" aria-label="Group"><option value="">All groups</option></select>
<select id="fRegion" aria-label="Region"><option value="">All regions</option></select>
<select id="fState" aria-label="State"><option value="">All states</option></select>
<span class="spacer"></span><span class="muted" id="count"></span></header>
<main><div class="note">Samples from <b>data/FINAL</b>: up to {PER_STATE} changed + {PER_STATE} unchanged pairs from every state
(dataset: {selected:,} selected pairs, 50% changed) and up to {PER_REASON} examples per rejection reason. Each image is a
<b>2.56 × 2.56 km</b> patch at Sentinel-2's <b>10 m</b> pixels (256 × 256). Use <b>Google Maps</b> on any card to see the
same spot in high resolution. Click a card for a large view; <b>Blink</b> flips before/after in place.</div>
<div class="grid" id="grid"></div></main>
<div id="viewer" role="dialog" aria-modal="true" aria-label="Pair viewer"><div class="vbar"><b id="vt"></b><span id="vs" class="muted"></span><span class="spacer"></span>
<a id="vmap" target="_blank" rel="noopener">Google Maps ↗</a><button data-m="side" class="on">Side by side</button><button data-m="blink">Blink</button>
<button id="px">Pixels: sharp</button><button id="prev">← Prev</button><span id="pos"></span><button id="next">Next →</button><button id="close">Close ✕</button></div>
<div class="obs" id="obs"></div><div class="stage px" id="stage"></div></div>
<script>
const P={data}; let view=[],cur=-1,mode="side",timer=null,sharp=true; const $=id=>document.getElementById(id);
const opts=(id,vals)=>vals.forEach(v=>{{const o=document.createElement("option");o.value=o.textContent=v;$(id).append(o)}});
opts("fGroup",[...new Set(P.map(p=>p.group))].sort()); opts("fRegion",[...new Set(P.map(p=>p.region))].sort()); opts("fState",[...new Set(P.map(p=>p.state))].sort());
const gcls=g=>g==="changed"?"changed":g==="unchanged"?"unchanged":"rej";
const rt=r=>r?`haze B${{r.before.haze}}/A${{r.after.haze}}, cloud B${{r.before.cloud}}/A${{r.after.cloud}}`:"not screened by model";
function refresh(){{const g=$("fGroup").value,rg=$("fRegion").value,s=$("fState").value;
 view=P.filter(p=>(!g||p.group===g)&&(!rg||p.region===rg)&&(!s||p.state===s)); const grid=$("grid"); grid.textContent="";
 view.forEach((p,i)=>{{const c=document.createElement("div"); c.className="card g-"+gcls(p.group); c.tabIndex=0;
  c.innerHTML=`<div class="imgs"><img loading="lazy" src="${{p.before}}" alt="before"><img loading="lazy" src="${{p.after}}" alt="after"></div>
  <div class="meta"><b></b> <span class="muted"></span><br><span class="muted d"></span><br>
  <span class="tag t-${{gcls(p.group)}}"></span>${{p.split?`<span class="tag t-split">${{p.split}}</span>`:""}}
  <br><a href="${{p.maps}}" target="_blank" rel="noopener">Google Maps ↗</a></div>`;
  c.querySelector("b").textContent=p.state; c.querySelector(".meta .muted").textContent=p.region;
  c.querySelector(".d").textContent=`${{p.dates}} · ${{p.changed_pct}}% px changed`; c.querySelector(".tag").textContent=p.group;
  c.querySelector("a").onclick=e=>e.stopPropagation(); c.onclick=()=>open(i); c.onkeydown=e=>{{if(e.key==="Enter")open(i)}}; grid.append(c)}});
 $("count").textContent=`${{view.length}} pairs shown`}}
const img=(s,a)=>{{const i=new Image();i.src=s;i.alt=a;return i}};
function render(){{const p=view[cur]; clearInterval(timer); document.querySelectorAll("[data-m]").forEach(b=>b.classList.toggle("on",b.dataset.m===mode));
 $("vt").textContent=`${{p.state}} · ${{p.group}}`; $("vs").textContent=`${{p.id}} · ${{p.dates}} · ${{p.changed_pct}}% px changed · ${{rt(p.ratings)}} · ${{p.lat}}, ${{p.lon}}`;
 $("vmap").href=p.maps; $("obs").textContent=p.observation?`Haze model: "${{p.observation}}"`:""; $("pos").textContent=`${{cur+1}} / ${{view.length}}`;
 const st=$("stage"); st.textContent="";
 if(mode==="side"){{[["before","Before"],["after","After"]].forEach(([k,t])=>{{const f=document.createElement("figure");f.append(img(p[k],t));const c=document.createElement("figcaption");c.textContent=t;f.append(c);st.append(f)}})}}
 else{{const f=document.createElement("figure"),c=document.createElement("figcaption");let a=false;const b1=img(p.before,"before"),b2=img(p.after,"after");
  const show=()=>{{f.replaceChildren(a?b2:b1,c);c.textContent=a?"After":"Before"}};show();st.append(f);timer=setInterval(()=>{{a=!a;show()}},800)}}}}
function open(i){{cur=i;$("viewer").classList.add("open");render()}} function close(){{clearInterval(timer);$("viewer").classList.remove("open")}}
function go(d){{if(view[cur+d]){{cur+=d;render()}}}}
document.querySelectorAll("[data-m]").forEach(b=>b.onclick=()=>{{mode=b.dataset.m;render()}});
$("px").onclick=()=>{{sharp=!sharp;$("stage").classList.toggle("px",sharp);$("px").textContent="Pixels: "+(sharp?"sharp":"smooth")}};
$("prev").onclick=()=>go(-1);$("next").onclick=()=>go(1);$("close").onclick=close;["fGroup","fRegion","fState"].forEach(id=>$(id).onchange=refresh);
document.addEventListener("keydown",e=>{{if(!$("viewer").classList.contains("open"))return;if(e.key==="Escape")close();else if(e.key==="ArrowRight")go(1);else if(e.key==="ArrowLeft")go(-1);else if(e.key==="b"){{mode="blink";render()}}else if(e.key==="s"){{mode="side";render()}}}});
refresh();
</script></body></html>"""
    (FINAL / "gallery.html").write_text(page, encoding="utf-8")
    print(f"{len(items)} pairs in the gallery: " + ", ".join(f"{g} {n}" for g, n in sorted(counts.items())))


if __name__ == "__main__":
    main()
