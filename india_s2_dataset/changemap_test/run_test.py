#!/usr/bin/env python3
"""Test the teammate's change-map + prompt (change_caption.py) on 50 labelled pairs.

    python changemap_test/run_test.py maps          # 1. change maps + prompts, local, free
    python changemap_test/run_test.py api --cap 0.10 # 2. Qwen3-VL-32B via API (independent of Ada)
    python changemap_test/run_test.py score          # scores maps, API and (if present) Ada results

Pairs: 25 Claude-labelled land_use + 25 others (13 seasonal, 12 none), from validation_1k.
Road width 2 px (~20 m at 10 m/px), as the script's own help says (road width / resolution).
A pair counts as "change" when the answer (short + detailed caption) describes a lasting
land-use change, judged by the same text-only marker used in validation_1k.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "changemap_test"
FINAL = ROOT / "data/FINAL"
sys.path.insert(0, str(ROOT))
from vlm_captioning import OPENROUTER_URL, _resolve_api_key

PY = sys.executable
M32, P32 = "qwen/qwen3-vl-32b-instruct", (0.104, 0.416)
MARK = ("Below is a caption written about two satellite images. Classify what it concludes. "
        "Reply with one word only: land_use (it reports something lasting changed, e.g. new buildings, roads, "
        "construction, ponds, clearing), seasonal (only seasonal/vegetation/water differences), "
        "none (no change), or unclear.\n\nCaption:\n")


def pairs() -> list[dict]:
    path = HERE / "pairs.json"
    if path.exists():
        return json.loads(path.read_text())
    ref = [json.loads(l) for l in open(ROOT / "validation_1k/reference_labels.jsonl")]
    folder = {json.loads(l)["patch_id"]: json.loads(l)["folder"] for l in open(FINAL / "manifest.jsonl")}
    rng = random.Random(50)
    pick = lambda lab, n: rng.sample(sorted(r["patch_id"] for r in ref if r["label"] == lab), n)
    chosen = [(p, "land_use") for p in pick("land_use", 25)] + [(p, "seasonal") for p in pick("seasonal", 13)] + \
             [(p, "none") for p in pick("none", 12)]
    out = [{"patch_id": p, "label": lab, "folder": folder[p]} for p, lab in chosen]
    path.write_text(json.dumps(out, indent=1))
    return out


def maps() -> None:
    for i, p in enumerate(pairs()):
        out = HERE / "maps" / p["patch_id"]
        if (out / "prompt.txt").exists():
            continue
        f = FINAL / p["folder"]
        subprocess.run([PY, str(HERE / "change_caption.py"), str(f / "before.png"), str(f / "after.png"),
                        "--out", str(out), "--road-width", "2"], check=True, capture_output=True)
        print(i + 1, p["patch_id"], json.loads((out / "stats.json").read_text())["caption"][:90])


def b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def call(session, key, content, max_tokens):
    body = {}
    for attempt in range(5):
        try:
            r = session.post(OPENROUTER_URL, timeout=180, headers={"Authorization": f"Bearer {key}"},
                             json={"model": M32, "temperature": 0, "max_tokens": max_tokens,
                                   "messages": [{"role": "user", "content": content}]})
            r.raise_for_status()
            body = r.json()
            if body.get("choices"):
                break
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (attempt + 1))
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    u = body.get("usage", {})
    return text, (u.get("prompt_tokens", 0) * P32[0] + u.get("completion_tokens", 0) * P32[1]) / 1e6


def caption_text(raw: str) -> str:
    try:
        j = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        return f"{j.get('short_caption', '')} {j.get('detailed_caption', '')}".strip()
    except ValueError:
        return raw


def mark(session, key, text):
    m, c = call(session, key, [{"type": "text", "text": MARK + text}], 5)
    m = m.strip().lower().strip(".").strip()
    return (m if m in ("land_use", "seasonal", "none", "unclear") else "unparsed"), c


def api(cap: float) -> None:
    out = HERE / "results_api_32b.jsonl"
    done = {json.loads(l)["patch_id"] for l in open(out)} if out.exists() else set()
    todo = [p for p in pairs() if p["patch_id"] not in done]
    key, session = _resolve_api_key(None), requests.Session()

    def run(p):
        d = HERE / "maps" / p["patch_id"]
        f = FINAL / p["folder"]
        raw, c1 = call(session, key, [
            {"type": "text", "text": "Image 1 (BEFORE):"}, {"type": "image_url", "image_url": {"url": b64(f / "before.png")}},
            {"type": "text", "text": "Image 2 (AFTER):"}, {"type": "image_url", "image_url": {"url": b64(f / "after.png")}},
            {"type": "text", "text": "Image 3 (CHANGE MAP):"}, {"type": "image_url", "image_url": {"url": b64(d / "change_map.png")}},
            {"type": "text", "text": (d / "prompt.txt").read_text()}], 500)
        m, c2 = mark(session, key, caption_text(raw))
        return {"patch_id": p["patch_id"], "label": p["label"], "raw": raw, "mark": m, "cost": c1 + c2}

    spent = 0.0
    with open(out, "a") as sink, ThreadPoolExecutor(8) as pool:
        for rec in pool.map(run, todo):
            sink.write(json.dumps(rec) + "\n")
            spent += rec["cost"]
            if spent >= cap:
                print("cost cap reached")
                break
    print(f"api: {len(todo)} pairs, ${spent:.4f}")


def score() -> None:
    ps = pairs()
    lab = {p["patch_id"]: p["label"] for p in ps}
    pos = [p for p, l in lab.items() if l == "land_use"]
    neg = [p for p, l in lab.items() if l != "land_use"]
    lines = ["| Method | Land-use found (of %d) | False change on seasonal/none (of %d) |" % (len(pos), len(neg)), "|---|---|---|"]

    def row(name, flag):
        lines.append(f"| {name} | {sum(flag(p) for p in pos)} | {sum(flag(p) for p in neg)} |")

    st = {p: json.loads((HERE / "maps" / p / "stats.json").read_text()) for p in lab if (HERE / "maps" / p / "stats.json").exists()}
    built = lambda t: t["from"] != t["to"] and ("building" in (t["from"], t["to"]) or "road" in (t["from"], t["to"]))
    row("Change map alone: any building/road appeared or vanished >= 0.3%",
        lambda p: any(built(t) and t["area_pct"] >= 0.3 for t in st[p]["transitions"]))
    row("Change map alone: any change at all (draft caption not 'no change')",
        lambda p: not st[p]["caption"].startswith("There is no significant change"))
    for name, path in (("Map + prompt, Qwen3-VL-32B (API)", "results_api_32b.jsonl"),
                       ("Map + prompt, Qwen3-VL-8B 4-bit (Ada)", "results_ada_8b.jsonl")):
        if (HERE / path).exists():
            r = {json.loads(l)["patch_id"]: json.loads(l) for l in open(HERE / path)}
            row(name, lambda p, r=r: r.get(p, {}).get("mark") == "land_use")
    v1k32 = {json.loads(l)["patch_id"]: json.loads(l) for l in open(ROOT / "validation_1k/results_32b.jsonl")}
    row("Reference: plain question, 32B, no map (validation_1k)", lambda p: v1k32[p]["mark"] == "land_use")
    pct = [sum(t["area_pct"] for t in s["transitions"]) for s in st.values()]
    lines += ["", f"Changed area marked by the map: median {sorted(pct)[len(pct) // 2]:.0f}% of the image, "
              f"max {max(pct):.0f}%. Truth = Claude's labels (provisional)."]
    old = (HERE / "RESULTS.md").read_text() if (HERE / "RESULTS.md").exists() else ""
    tail = old[old.find("\n## Verdict"):] if "\n## Verdict" in old else ""   # keep the written verdict
    (HERE / "RESULTS.md").write_text("# Change-map test (50 pairs)\n\n" + "\n".join(lines) + "\n" + tail)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["maps", "api", "score"])
    ap.add_argument("--cap", type=float, default=0.10)
    a = ap.parse_args()
    {"maps": maps, "api": lambda: api(a.cap), "score": score}[a.step]()
