#!/usr/bin/env python3
"""Human review with AI suggestions: serve the page and save every answer to review2/human_labels.json.

    python3 validation_1k/review2/server.py       # open http://localhost:8002 (keep this running)

Images are read straight from data/FINAL, so nothing is copied.
human_labels.json: {patch_id: {"category", "first_pick", "ai_label", "agrees_with_ai",
                               "caption", "ai_caption", "caption_edited", "ts"}}
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LABELS = HERE / "human_labels.json"
LOCK = threading.Lock()
VALID = {"change", "no_change", "seasonal", "cant_tell"}
FOLDER = {json.loads(l)["patch_id"]: json.loads(l)["folder"] for l in open(ROOT / "data/FINAL/manifest.jsonl")}


def load() -> dict:
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, code: int, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/labels":
            with LOCK:
                return self._json(200, load())
        if self.path.startswith("/img/"):                     # /img/<patch_id>/<before|after>.png
            _, _, pid, name = self.path.split("/", 3)
            if pid in FOLDER and name in ("before.png", "after.png"):
                data = (ROOT / "data/FINAL" / FOLDER[pid] / name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            return self._json(404, {"error": "not found"})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/save":
            return self._json(404, {"error": "not found"})
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            pid, cat, first = b["patch_id"], b["category"], b["first_pick"]
            caption, ai_caption = (b.get("caption") or "").strip(), (b.get("ai_caption") or "").strip()
        except (ValueError, KeyError, AttributeError):
            return self._json(400, {"error": "bad request"})
        if cat not in VALID or first not in VALID or pid not in FOLDER:
            return self._json(400, {"error": "invalid category or pair"})
        if cat in ("change", "seasonal") and not caption:
            return self._json(400, {"error": "a description is required for Change and Seasonal"})
        rec = {"category": cat, "first_pick": first, "first_pick_source": b.get("first_pick_source", "this_page"),
               "ai_label": b.get("ai_label"), "agrees_with_ai": first == b.get("ai_label"),
               "caption": caption or None, "ai_caption": ai_caption or None,
               "caption_edited": bool(caption) and caption != ai_caption, "ts": round(time.time())}
        with LOCK:
            labels = load()
            labels[pid] = rec
            tmp = LABELS.with_suffix(".tmp")
            tmp.write_text(json.dumps(labels, indent=1))
            os.replace(tmp, LABELS)
        self._json(200, {"ok": True})

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(HERE)))
    print(f"Human review: http://localhost:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
