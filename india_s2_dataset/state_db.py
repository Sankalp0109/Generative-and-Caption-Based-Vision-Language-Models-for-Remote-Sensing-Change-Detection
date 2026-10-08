"""SQLite record of the download: one row per AOI, one per saved pair, plus an event log.

The database is the source of truth for "what is left to do", so a run that is killed,
crashes or loses its connection resumes from the next unfinished AOI instead of starting
over. WAL mode lets many worker threads write status while another process reads it.

AOI status lifecycle::

    pending -> running -> done        (>=1 pair saved)
                       -> no_pairs    (searched fine, nothing usable: cloud, coverage, gap)
                       -> failed      (error; see ``error``, retry with --retry failed)
    pending -> selected               (--select-only: pairs found, nothing downloaded)

While ``running``, ``stage`` says where it is: searching, selecting, downloading, saving.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

STATUSES = ("pending", "selected", "running", "done", "no_pairs", "failed")

SCHEMA = """
CREATE TABLE IF NOT EXISTS aoi (
    key TEXT PRIMARY KEY,
    label TEXT, state TEXT, district TEXT, category TEXT,
    lon REAL, lat REAL, size_km REAL, bbox TEXT, season_months TEXT, max_aoi_cloud REAL,
    landcover TEXT, terrain TEXT, change_type TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    stage TEXT, stage_note TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    n_pairs INTEGER NOT NULL DEFAULT 0,
    tile TEXT, error TEXT, diag TEXT,
    started_at REAL, finished_at REAL, updated_at REAL
);
CREATE INDEX IF NOT EXISTS aoi_status ON aoi(status);
CREATE TABLE IF NOT EXISTS pair (
    aoi_key TEXT NOT NULL, idx INTEGER NOT NULL,
    before_item TEXT, after_item TEXT, before_date TEXT, after_date TEXT,
    tile TEXT, gap_days INTEGER,
    before_aoi_cloud REAL, after_aoi_cloud REAL, before_tile_cloud REAL, after_tile_cloud REAL,
    shape TEXT, bytes INTEGER, path TEXT, created_at REAL,
    PRIMARY KEY (aoi_key, idx)
);
CREATE TABLE IF NOT EXISTS event (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, aoi_key TEXT, level TEXT, message TEXT
);
CREATE TABLE IF NOT EXISTS run (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started_at REAL, ended_at REAL, workers INTEGER,
    args TEXT, processed INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS block_claim (block_id TEXT PRIMARY KEY, aoi_key TEXT NOT NULL);
"""

AOI_FIELDS = ("label", "state", "district", "category", "lon", "lat", "size_km", "season_months",
              "max_aoi_cloud", "landcover", "terrain", "change_type")


def _num(value):
    return None if value in (None, "") else float(value)


class StateDB:
    """Thread-safe: each thread gets its own connection to the same file."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=60000")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ---- registry -> rows -------------------------------------------------------------
    def seed(self, registry: dict[str, dict]) -> dict:
        """Insert new AOIs as pending; refresh descriptive columns of existing ones.

        Never touches ``status`` of an existing row, so re-seeding an edited registry cannot
        throw away finished work.
        """
        conn, added, existing = self._conn(), 0, 0
        conn.execute("BEGIN")
        try:
            for key, row in registry.items():
                bbox = json.dumps(list(row["bbox"]))
                values = {f: (_num(row.get(f)) if f in ("lon", "lat", "size_km", "max_aoi_cloud")
                              else row.get(f) or None) for f in AOI_FIELDS}
                found = conn.execute("SELECT 1 FROM aoi WHERE key=?", (key,)).fetchone()
                if found:
                    sets = ", ".join(f"{f}=?" for f in AOI_FIELDS)
                    conn.execute(f"UPDATE aoi SET {sets}, bbox=? WHERE key=?",
                                 (*values.values(), bbox, key))
                    existing += 1
                else:
                    cols = ", ".join(("key", *AOI_FIELDS, "bbox", "updated_at"))
                    marks = ", ".join("?" * (len(AOI_FIELDS) + 3))
                    conn.execute(f"INSERT INTO aoi ({cols}) VALUES ({marks})",
                                 (key, *values.values(), bbox, time.time()))
                    added += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return {"added": added, "existing": existing}

    def recover_interrupted(self) -> int:
        """A run that died leaves rows ``running``; put them back in the queue."""
        conn = self._conn()
        keys = [r["key"] for r in conn.execute("SELECT key FROM aoi WHERE status='running'")]
        for key in keys:
            self.log(key, "warn", "was 'running' when a previous run ended; re-queued")
        conn.execute("UPDATE aoi SET status='pending', stage=NULL, stage_note=NULL "
                     "WHERE status='running'")
        return len(keys)

    def queue(self, statuses=("pending", "selected"), category: str | None = None,
              state: str | None = None, limit: int | None = None,
              keys: list[str] | None = None) -> list[dict]:
        where, args = [f"status IN ({','.join('?' * len(statuses))})"], list(statuses)
        if keys:
            where.append(f"key IN ({','.join('?' * len(keys))})")
            args.extend(keys)
        for column, value in (("category", category), ("state", state)):
            if value:
                where.append(f"{column}=?")
                args.append(value)
        sql = f"SELECT * FROM aoi WHERE {' AND '.join(where)} ORDER BY rowid"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in self._conn().execute(sql, args)]

    # ---- per-AOI progress (called from worker threads) --------------------------------
    def start(self, key: str) -> None:
        now = time.time()
        self._conn().execute(
            "UPDATE aoi SET status='running', stage='searching', stage_note=NULL, error=NULL, "
            "attempts=attempts+1, started_at=?, updated_at=? WHERE key=?", (now, now, key))

    def stage(self, key: str, stage: str, note: str | None = None) -> None:
        self._conn().execute("UPDATE aoi SET stage=?, stage_note=?, updated_at=? WHERE key=?",
                             (stage, note, time.time(), key))

    def add_pair(self, key: str, idx: int, record: dict) -> None:
        cols = ("before_item", "after_item", "before_date", "after_date", "tile", "gap_days",
                "before_aoi_cloud", "after_aoi_cloud", "before_tile_cloud", "after_tile_cloud",
                "shape", "bytes", "path")
        self._conn().execute(
            f"INSERT OR REPLACE INTO pair (aoi_key, idx, {', '.join(cols)}, created_at) "
            f"VALUES (?, ?, {', '.join('?' * len(cols))}, ?)",
            (key, idx, *(record.get(c) for c in cols), time.time()))

    def finish(self, key: str, status: str, *, n_pairs: int = 0, tile: str | None = None,
               error: str | None = None, diag: dict | None = None) -> None:
        assert status in STATUSES
        now = time.time()
        self._conn().execute(
            "UPDATE aoi SET status=?, stage=NULL, stage_note=NULL, n_pairs=?, tile=?, error=?, "
            "diag=?, finished_at=?, updated_at=? WHERE key=?",
            (status, n_pairs, tile, error, json.dumps(diag) if diag else None, now, now, key))

    def claim_block(self, block_id: str, key: str) -> str:
        """Reserve a Sentinel-2 block for one site; returns whichever site owns it now."""
        conn = self._conn()
        conn.execute("INSERT OR IGNORE INTO block_claim (block_id, aoi_key) VALUES (?, ?)", (block_id, key))
        return conn.execute("SELECT aoi_key FROM block_claim WHERE block_id=?", (block_id,)).fetchone()[0]

    def release_block(self, key: str) -> None:
        self._conn().execute("DELETE FROM block_claim WHERE aoi_key=?", (key,))

    def pairs_saved(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM pair").fetchone()[0]

    def reset_for_retry(self, key: str) -> None:
        """Forget the pairs of an AOI that is about to be downloaded again."""
        self._conn().execute("DELETE FROM pair WHERE aoi_key=?", (key,))

    def log(self, key: str | None, level: str, message: str) -> None:
        self._conn().execute("INSERT INTO event (ts, aoi_key, level, message) VALUES (?,?,?,?)",
                             (time.time(), key, level, message))

    # ---- runs and metadata ------------------------------------------------------------
    def begin_run(self, workers: int, args: dict) -> int:
        cur = self._conn().execute("INSERT INTO run (started_at, workers, args) VALUES (?,?,?)",
                                   (time.time(), workers, json.dumps(args, default=str)))
        return cur.lastrowid

    def end_run(self, run_id: int, processed: int) -> None:
        self._conn().execute("UPDATE run SET ended_at=?, processed=? WHERE id=?",
                             (time.time(), processed, run_id))

    def set_meta(self, key: str, value) -> None:
        self._conn().execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                             (key, json.dumps(value)))

    def get_meta(self, key: str, default=None):
        row = self._conn().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # ---- reporting --------------------------------------------------------------------
    def summary(self, recent_minutes: float = 30) -> dict:
        conn = self._conn()
        counts = {s: 0 for s in STATUSES}
        for row in conn.execute("SELECT status, COUNT(*) n FROM aoi GROUP BY status"):
            counts[row["status"]] = row["n"]
        pairs = conn.execute("SELECT COUNT(*) n, COALESCE(SUM(bytes),0) b FROM pair").fetchone()
        since = time.time() - recent_minutes * 60
        recent = conn.execute("SELECT COUNT(*) n FROM aoi WHERE finished_at>? AND status IN "
                              "('done','no_pairs','failed','selected')", (since,)).fetchone()["n"]
        running = [dict(r) for r in conn.execute(
            "SELECT key, stage, stage_note, started_at FROM aoi WHERE status='running' "
            "ORDER BY started_at")]
        by_state = [dict(r) for r in conn.execute(
            "SELECT COALESCE(state,'?') state, COUNT(*) aois, "
            "SUM(status='done') done, SUM(n_pairs) pairs FROM aoi GROUP BY state ORDER BY aois DESC")]
        by_category = [dict(r) for r in conn.execute(
            "SELECT COALESCE(category,'?') category, COUNT(*) aois, "
            "SUM(status='done') done, SUM(n_pairs) pairs FROM aoi GROUP BY category ORDER BY aois DESC")]
        return {"counts": counts, "total": sum(counts.values()), "pairs": pairs["n"],
                "bytes": pairs["b"], "recent_finished": recent, "recent_minutes": recent_minutes,
                "running": running, "by_state": by_state, "by_category": by_category}

    def failures(self, limit: int = 10) -> list[dict]:
        return [dict(r) for r in self._conn().execute(
            "SELECT key, status, attempts, COALESCE(error, diag) reason FROM aoi "
            "WHERE status IN ('failed','no_pairs') ORDER BY updated_at DESC LIMIT ?", (limit,))]

    def error_breakdown(self) -> list[dict]:
        return [dict(r) for r in self._conn().execute(
            "SELECT status, substr(COALESCE(error,''),1,90) reason, COUNT(*) n FROM aoi "
            "WHERE status IN ('failed') GROUP BY status, reason ORDER BY n DESC LIMIT 10")]
