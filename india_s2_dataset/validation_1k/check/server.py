#!/usr/bin/env python3
"""Serve the labelling page and save every answer to validation_1k/check/check_labels.json.

    python validation_1k/check/server.py            # http://localhost:8001
    python validation_1k/check/server.py --host 0.0.0.0 --port 8001   # reachable from other machines on the network

check_labels.json: {patch_id: {"label": ..., "caption": ..., "ts": ...}}
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
LABELS = HERE / "check_labels.json"
LOCK = threading.Lock()
VALID = {"change", "no_change", "seasonal", "cant_tell"}


def load() -> dict:
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")      # page may be opened as a file or from another server
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

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
        return super().do_GET()

    def do_POST(self):
        if self.path != "/save":
            return self._json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            pid, label = body["patch_id"], body["label"]
            caption = (body.get("caption") or "").strip()
        except (ValueError, KeyError, AttributeError):
            return self._json(400, {"error": "bad request"})
        if label not in VALID:
            return self._json(400, {"error": "a valid label is required"})
        with LOCK:
            labels = load()
            labels[pid] = {"label": label, "caption": caption or None, "ts": round(time.time())}
            tmp = LABELS.with_suffix(".tmp")
            tmp.write_text(json.dumps(labels, indent=1))
            os.replace(tmp, LABELS)                  # atomic: a crash never leaves a half-written file
        self._json(200, {"ok": True})

    def log_message(self, fmt, *args):
        if "POST" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(HERE)))
    print(f"Change check: http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
