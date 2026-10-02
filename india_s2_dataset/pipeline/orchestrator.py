#!/usr/bin/env python3
"""Streaming pipeline: download -> preprocess -> Ada haze screen -> sort, batch by batch.

    python -m pipeline.orchestrator setup      # one-time: plains to Feb-Apr, build batches
    python -m pipeline.orchestrator run        # the loop (run detached; safe to restart)
    python -m pipeline.orchestrator status     # compact progress

Batches of ~100 sites move through stages
    queued -> downloading -> downloaded -> preprocessed -> submitted -> sorted
and several batches are in different stages at once: while Ada screens batch k, batch k+1
downloads and preprocesses here. All state is in data/FINAL/pipeline.sqlite, so killing and
restarting the loop continues where it stopped. When every batch is sorted it tops up sites if
fewer than 15,000 patches were accepted, then assembles the final dataset (see assemble.py).
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = Path(os.environ.get("FINAL_DIR", ROOT / "data" / "FINAL"))
DB = FINAL / "pipeline.sqlite"
SITE_DB = ROOT / "data" / "download_state.sqlite"
SENTINEL = ROOT / "data" / "sentinel2"
PY = sys.executable
BATCH_SITES = 100
MAX_ADA_IN_FLIGHT = 3                      # batches uploaded ahead for the Ada worker
MAX_ADA_ATTEMPTS = 4
ADA = ["ssh", "-i", str(Path.home() / ".ssh/ada_ed25519"), "-o", "BatchMode=yes", "-o", "LogLevel=ERROR",
       "-o", "ConnectTimeout=30", "sankalp0109@ada.iiit.ac.in"]
RSYNC_SSH = f"ssh -i {Path.home() / '.ssh/ada_ed25519'} -o BatchMode=yes -o LogLevel=ERROR"
REMOTE = "sankalp0109@ada.iiit.ac.in"
PLAINS = ("Punjab", "Haryana", "Delhi", "Chandigarh", "Uttar Pradesh", "Bihar", "West Bengal")
TARGET_ACCEPTED = 15000

SCHEMA = """
CREATE TABLE IF NOT EXISTS batch (
    id TEXT PRIMARY KEY, keys TEXT NOT NULL, needs_download INTEGER NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued', ada_job TEXT, ada_attempts INTEGER DEFAULT 0,
    n_patches INTEGER, n_for_model INTEGER, n_accepted INTEGER, note TEXT, updated REAL,
    dl_attempts INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with open(FINAL / "logs" / "orchestrator.log", "a") as handle:
        handle.write(line + "\n")


def db() -> sqlite3.Connection:
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "logs").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB, timeout=60, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def kv(conn, key, value=None, default=None):
    if value is None:
        row = conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(row["v"]) if row else default
    conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))


def set_stage(conn, batch_id, stage, **fields):
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE batch SET stage=?, updated=?{', ' + sets if sets else ''} WHERE id=?",
                 (stage, time.time(), *fields.values(), batch_id))


def ada(cmd: str, timeout: int = 120) -> str:
    return subprocess.run(ADA + [cmd], capture_output=True, text=True, timeout=timeout, check=True).stdout


# ------------------------------------------------------------------ setup -----------------
def setup() -> None:
    """Plains -> Feb-Apr and re-queue them; split all work into batches. Safe to re-run."""
    import csv
    import shutil
    sys.path.insert(0, str(ROOT))
    from state_db import StateDB
    conn = db()
    if kv(conn, "setup_done"):
        log("setup already done")
        return
    registry = ROOT / "locations" / "india_aois.csv"
    rows = list(csv.DictReader(open(registry, newline="", encoding="utf-8")))
    for row in rows:
        if row["state"] in PLAINS:
            row["season_months"] = "2,3,4"     # measured on Delhi: avoids the winter smog
    with open(registry, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sites = StateDB(SITE_DB)
    sconn = sites._conn()
    plains = [r["key"] for r in sconn.execute(
        f"SELECT key FROM aoi WHERE state IN ({','.join('?' * len(PLAINS))}) ORDER BY rowid", PLAINS)]
    backup = ROOT / "data" / "sentinel2_winter_backup"
    for key in plains:                      # keep the old winter pair (nothing is deleted)
        if (SENTINEL / key).exists() and not (backup / key).exists():
            backup.mkdir(exist_ok=True)
            shutil.move(str(SENTINEL / key), str(backup / key))
        sites.release_block(key)
        sites.reset_for_retry(key)
        sconn.execute("UPDATE aoi SET status='pending', n_pairs=0, tile=NULL, diag=NULL, error=NULL WHERE key=?", (key,))
    others = [r["key"] for r in sconn.execute(
        f"SELECT key FROM aoi WHERE status='done' AND state NOT IN ({','.join('?' * len(PLAINS))}) ORDER BY rowid", PLAINS)]
    n = 0
    for keys, needs in ((others, 0), (plains, 1)):
        for i in range(0, len(keys), BATCH_SITES):
            n += 1
            conn.execute("INSERT OR IGNORE INTO batch (id, keys, needs_download, stage, updated) VALUES (?,?,?,?,?)",
                         (f"b{n:03d}", json.dumps(keys[i:i + BATCH_SITES]), needs,
                          "queued" if needs else "downloaded", time.time()))
    kv(conn, "setup_done", True)
    log(f"setup: {len(plains)} plains sites re-queued for Feb-Apr (winter pairs kept in {backup.name}); "
        f"{len(others)} existing sites; {n} batches")


# ------------------------------------------------------------------ stages ----------------
def site_status(keys: list[str]) -> dict[str, str]:
    conn = sqlite3.connect(SITE_DB)
    return dict(conn.execute(f"SELECT key, status FROM aoi WHERE key IN ({','.join('?' * len(keys))})", keys).fetchall())


def overlap_rejects() -> dict[str, str]:
    """Sites whose block overlaps (>10%) an earlier site's block: keep the first, reject the rest."""
    import numpy as np
    boxes = []
    for pair in sorted(SENTINEL.glob("*/pair_1/pair.json")):
        meta = json.loads(pair.read_text())
        boxes.append((pair.parent.parent.name, meta["block"]["bbox_wgs84"]))
    if not boxes:
        return {}
    b = np.array([x[1] for x in boxes])
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
    frac = iw * ih / area[None, :]
    np.fill_diagonal(frac, 0)
    rejects = {}
    for j in range(len(boxes)):
        earlier = np.where(frac[:j, j] > 0.10)[0]
        earlier = [i for i in earlier if boxes[i][0] not in rejects]
        if earlier:
            rejects[boxes[j][0]] = f"overlap_duplicate:{boxes[earlier[0]][0]}"
    return rejects


def preprocess(batch_id: str) -> None:
    """Cut and screen every downloaded site of a batch; write the Ada upload list."""
    sys.path.insert(0, str(ROOT))
    from pipeline.patches import process_site
    conn = db()
    keys = json.loads(conn.execute("SELECT keys FROM batch WHERE id=?", (batch_id,)).fetchone()["keys"])
    registry = {}
    import csv
    for row in csv.DictReader(open(ROOT / "locations" / "india_aois.csv", newline="", encoding="utf-8")):
        registry[row["key"]] = row
    overlaps = overlap_rejects()
    status = site_status(keys)
    rows = []
    for key in keys:
        pair = SENTINEL / key / "pair_1"
        if status.get(key) != "done" or not pair.exists():
            continue
        site_reject = "overlap_duplicate" if key in overlaps else None
        rows += process_site(key, pair, FINAL, registry.get(key, {}), site_reject)
    pending = [r for r in rows if r["status"] == "pending"]
    upload = FINAL / "_ada" / batch_id
    upload.mkdir(parents=True, exist_ok=True)
    with open(upload / "manifest.jsonl", "w") as handle:
        for r in pending:
            handle.write(json.dumps({"id": r["patch_id"], "split": batch_id,
                                     "before": f"_pending/{r['patch_id']}/before.png",
                                     "after": f"_pending/{r['patch_id']}/after.png"}) + "\n")
    (upload / "files.txt").write_text("".join(f"_pending/{r['patch_id']}/{n}.png\n" for r in pending
                                              for n in ("before", "after")))
    set_stage(conn, batch_id, "preprocessed", n_patches=len(rows), n_for_model=len(pending))
    log(f"{batch_id}: preprocessed {len(rows)} patches, {len(pending)} to Ada, "
        f"{len(rows) - len(pending)} rejected by checks")


def submit(conn, batch: sqlite3.Row) -> None:
    upload = FINAL / "_ada" / batch["id"]
    remote = f"haze/batches/{batch['id']}"
    if batch["n_for_model"] == 0:
        set_stage(conn, batch["id"], "sorted", n_accepted=0, note="nothing for the model")
        return
    ada(f"mkdir -p {remote}")
    subprocess.run(["rsync", "-a", "-e", RSYNC_SSH, "--files-from", str(upload / "files.txt"), f"{FINAL}/",
                    f"{REMOTE}:{remote}/"], check=True, timeout=3600)
    subprocess.run(["rsync", "-a", "-e", RSYNC_SSH, str(upload / "manifest.jsonl"), f"{REMOTE}:{remote}/"],
                   check=True, timeout=300)
    ada(f"rm -f {remote}/DONE && touch {remote}/READY")      # the worker picks it up from here
    set_stage(conn, batch["id"], "submitted", ada_job="worker", ada_attempts=(batch["ada_attempts"] or 0) + 1)
    log(f"{batch['id']}: uploaded {batch['n_for_model']} pairs, queued for the Ada worker")


def ensure_worker(conn) -> None:
    """Keep exactly one long-running Ada worker job alive; replace it if it died (e.g. broken node)."""
    job = kv(conn, "worker_job")
    state = "NONE"
    if job:
        out = ada(f"sacct -n -X -P -j {job} -o State").strip().splitlines()
        state = out[0].split()[0] if out else "UNKNOWN"
    if state in ("PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "UNKNOWN"):
        return
    if job:
        try:
            tail = ada(f"tail -3 ~/haze/haze-worker-{job}.log")
            if "CUDA not usable on" in tail:
                node = tail.split("CUDA not usable on ")[1].split(";")[0].strip()
                bad = kv(conn, "bad_nodes", default=["gnode076"])
                if node not in bad:
                    kv(conn, "bad_nodes", bad + [node])
                log(f"worker: node {node} has broken CUDA; excluded")
        except subprocess.SubprocessError:
            pass
    exclude = ",".join(kv(conn, "bad_nodes", default=["gnode076"]))
    quant, size = kv(conn, "ada_quant", default="4bit"), kv(conn, "ada_batch", default=8)
    new = ada(f"cd ~/haze && rm -f batches/STOP && sbatch --parsable --export=ALL,BATCH={size} "
              f"--exclude={exclude} haze_worker.sbatch batches {quant}").strip()
    kv(conn, "worker_job", new)
    log(f"worker: started Ada job {new} (previous {job or '-'} {state})")


def collect(conn, batch: sqlite3.Row, state: str) -> None:
    """Pull an Ada job's results; sort the batch if complete, else resubmit."""
    sys.path.insert(0, str(ROOT))
    from pipeline.patches import finalize
    upload = FINAL / "_ada" / batch["id"]
    remote = f"haze/batches/{batch['id']}"
    subprocess.run(["rsync", "-a", "-e", RSYNC_SSH, f"{REMOTE}:{remote}/results.jsonl", str(upload / "results.jsonl")],
                   timeout=600)
    results = {}
    if (upload / "results.jsonl").exists():
        for line in open(upload / "results.jsonl"):
            r = json.loads(line)
            results[r["id"]] = r
    wanted = [json.loads(line)["id"] for line in open(upload / "manifest.jsonl")]
    missing = [w for w in wanted if w not in results]
    if missing:
        log_text = ""
        try:
            log_text = ada(f"tail -5 ~/haze/haze-{batch['id']}-{batch['ada_job']}.log")
        except subprocess.SubprocessError:
            pass
        if "CUDA not usable on" in log_text:
            node = log_text.split("CUDA not usable on ")[1].split(";")[0].strip()
            bad = kv(conn, "bad_nodes", default=["gnode076"])
            if node not in bad:
                kv(conn, "bad_nodes", bad + [node])
            log(f"{batch['id']}: node {node} has broken CUDA; excluded")
        if (batch["ada_attempts"] or 0) >= MAX_ADA_ATTEMPTS:
            set_stage(conn, batch["id"], "failed", note=f"Ada {state}, {len(missing)} missing after retries")
            log(f"{batch['id']}: FAILED after {batch['ada_attempts']} Ada attempts ({len(missing)} missing)")
            return
        log(f"{batch['id']}: Ada {batch['ada_job']} {state}, {len(missing)}/{len(wanted)} missing; resubmitting")
        set_stage(conn, batch["id"], "preprocessed")   # submit() re-uploads (idempotent) and re-marks READY
        return
    accepted = 0
    for pid in wanted:
        if (FINAL / "_pending" / pid).exists():
            r = results[pid]
            observation = ""
            try:
                observation = json.loads(r["raw"][r["raw"].find("{"):r["raw"].rfind("}") + 1]).get("observation", "")
            except ValueError:
                pass
            model = {"name": "Qwen/Qwen3-VL-8B-Instruct", "quant": r.get("quant"), "prompt": r.get("prompt"),
                     "ratings": r["ratings"], "verdict": r["verdict"], "observation": observation}
            accepted += finalize(FINAL, pid, model, r["verdict"])["status"] == "accepted"
    try:
        ada(f"rm -rf {remote}")
    except subprocess.SubprocessError:
        pass
    set_stage(conn, batch["id"], "sorted", n_accepted=accepted)
    log(f"{batch['id']}: sorted, {accepted}/{len(wanted)} accepted by the model")


# ------------------------------------------------------------------ top-up ----------------
def plan_topup(conn, early: bool = False) -> bool:
    """If accepted patches will fall short of the target, add evenly spread sites as new batches.

    ``early``: decide from a projection (accepted so far + the not-yet-screened patches at the
    acceptance rate seen so far) once half the batches are sorted, so the top-up downloads
    overlap with the remaining Ada screening instead of starting after it.
    """
    rounds = kv(conn, "topup_rounds", default=0)
    if rounds >= 2:
        return False
    rows = conn.execute("SELECT stage, keys, n_for_model, n_accepted FROM batch").fetchall()
    sorted_rows = [r for r in rows if r["stage"] == "sorted"]
    accepted = sum(r["n_accepted"] or 0 for r in sorted_rows)
    screened = sum(r["n_for_model"] or 0 for r in sorted_rows)
    rate = accepted / screened if screened else 0.6
    waiting = sum(r["n_for_model"] or 0 for r in rows if r["stage"] in ("preprocessed", "submitted"))
    unprocessed_sites = sum(len(json.loads(r["keys"])) for r in rows if r["stage"] in ("queued", "downloading", "downloaded"))
    per_site = accepted / max(1, sum(len(json.loads(r["keys"])) for r in sorted_rows))
    projected = accepted + waiting * rate + unprocessed_sites * per_site
    if early and (len(sorted_rows) < len(rows) / 2 or unprocessed_sites):
        return False
    if projected >= TARGET_ACCEPTED:
        return False
    need = int((TARGET_ACCEPTED - projected) / max(per_site, 1) * 1.25) + 10
    log(f"top-up check: {accepted} accepted, projected {projected:.0f} (acceptance {rate:.0%}, "
        f"{per_site:.1f} accepted per site)")
    sys.path.insert(0, str(ROOT))
    from pipeline.topup import add_sites
    keys = add_sites(need, round_no=rounds + 1)
    n = conn.execute("SELECT COUNT(*) FROM batch").fetchone()[0]
    for i in range(0, len(keys), BATCH_SITES):
        n += 1
        conn.execute("INSERT INTO batch (id, keys, needs_download, stage, updated) VALUES (?,?,1,'queued',?)",
                     (f"b{n:03d}", json.dumps(keys[i:i + BATCH_SITES]), time.time()))
    kv(conn, "topup_rounds", rounds + 1)
    log(f"top-up round {rounds + 1}: {accepted} accepted < {TARGET_ACCEPTED}; added {len(keys)} sites "
        f"(~{per_site:.1f} accepted per site)")
    return True


# ------------------------------------------------------------------ loop ------------------
def run() -> None:
    conn = db()
    if not kv(conn, "setup_done"):
        setup()
    download_proc, pre_proc = None, None
    download_batch_id = pre_batch_id = None
    # After a crash or reboot: a download that was cut off restarts (the downloader resumes per site).
    cut = [r["id"] for r in conn.execute("SELECT id FROM batch WHERE stage='downloading'")]
    for batch_id in cut:
        set_stage(conn, batch_id, "queued")
    log(f"orchestrator started{' (re-queued interrupted downloads: ' + ', '.join(cut) + ')' if cut else ''}")
    while True:
        try:
            batches = conn.execute("SELECT * FROM batch ORDER BY id").fetchall()
            # 1. download: one batch at a time, 48 workers
            if download_proc and download_proc.poll() is not None:
                row = conn.execute("SELECT keys, dl_attempts FROM batch WHERE id=?", (download_batch_id,)).fetchone()
                states = site_status(json.loads(row["keys"]))
                unfinished = sum(v in ("pending", "running", "failed", "selected") for v in states.values())
                if unfinished and (row["dl_attempts"] or 0) < 3:
                    set_stage(conn, download_batch_id, "queued")   # retry the unfinished sites
                    log(f"{download_batch_id}: download ended with {unfinished} unfinished sites; retrying")
                else:
                    set_stage(conn, download_batch_id, "downloaded")
                    log(f"{download_batch_id}: download finished ({unfinished} unfinished)")
                download_proc = None
            if download_proc is None:
                nxt = next((b for b in batches if b["stage"] == "queued"), None)
                if nxt:
                    keys = ",".join(json.loads(nxt["keys"]))
                    download_proc = subprocess.Popen(
                        [PY, "download_batch.py", "run", "--keys", keys, "--workers", "48", "--report", "600",
                         "--retry", "failed"], cwd=ROOT, stdout=open(FINAL / "logs" / f"download_{nxt['id']}.log", "a"),
                        stderr=subprocess.STDOUT)
                    download_batch_id = nxt["id"]
                    set_stage(conn, nxt["id"], "downloading", dl_attempts=(nxt["dl_attempts"] or 0) + 1)
                    log(f"{nxt['id']}: downloading {len(json.loads(nxt['keys']))} sites")
            # 2. preprocess: one batch at a time, in a child process
            if pre_proc and pre_proc.poll() is not None:
                if pre_proc.returncode != 0:
                    log(f"{pre_batch_id}: preprocess exited {pre_proc.returncode}; will retry")
                pre_proc = None
            batches = conn.execute("SELECT * FROM batch ORDER BY id").fetchall()
            if pre_proc is None:
                nxt = next((b for b in batches if b["stage"] == "downloaded"), None)
                if nxt:
                    pre_proc = subprocess.Popen([PY, "-m", "pipeline.orchestrator", "preprocess", nxt["id"]], cwd=ROOT,
                                                stdout=open(FINAL / "logs" / "preprocess.log", "a"), stderr=subprocess.STDOUT)
                    pre_batch_id = nxt["id"]
            # 3. Ada: submit up to MAX_ADA_IN_FLIGHT, poll the rest
            in_flight = [b for b in batches if b["stage"] == "submitted"]
            legacy = [b for b in in_flight if b["ada_job"] != "worker"]
            if legacy:
                ids = ",".join(b["ada_job"] for b in legacy)
                states = dict(line.split("|")[:2] for line in ada(f"sacct -n -X -P -j {ids} -o JobID,State").splitlines()
                              if "|" in line)
                for b in legacy:
                    state = states.get(b["ada_job"], "UNKNOWN").split()[0]
                    if state not in ("PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "UNKNOWN"):
                        collect(conn, b, state)
            worker_batches = [b for b in in_flight if b["ada_job"] == "worker"]
            if worker_batches:
                done_marks = set(ada("ls -d haze/batches/*/DONE 2>/dev/null | cut -d/ -f3 || true").split())
                for b in worker_batches:
                    if b["id"] in done_marks:
                        collect(conn, b, "DONE")
                ensure_worker(conn)
            batches = conn.execute("SELECT * FROM batch ORDER BY id").fetchall()
            in_flight = sum(b["stage"] == "submitted" for b in batches)
            for b in batches:
                if in_flight >= MAX_ADA_IN_FLIGHT or not kv(conn, "ada_ready"):
                    break   # Ada settings (precision, batch size) are confirmed before any submission
                if b["stage"] == "preprocessed":
                    submit(conn, b)
                    in_flight += 1
            # 4. early top-up: start extra downloads while Ada is still screening
            if not kv(conn, "topup_rounds") and not download_proc:
                plan_topup(conn, early=True)
            # 5. finish: top-up if short, else assemble
            batches = conn.execute("SELECT * FROM batch ORDER BY id").fetchall()
            if all(b["stage"] in ("sorted", "failed") for b in batches) and not download_proc and not pre_proc:
                if plan_topup(conn):
                    continue
                try:
                    ada("touch haze/batches/STOP")                 # the worker exits once idle
                except subprocess.SubprocessError:
                    pass
                if not kv(conn, "assembled"):
                    log("all batches sorted: assembling the final dataset")
                    subprocess.run([PY, "-m", "pipeline.assemble"], cwd=ROOT, check=True,
                                   stdout=open(FINAL / "logs" / "assemble.log", "a"), stderr=subprocess.STDOUT)
                    kv(conn, "assembled", True)
                    log("DONE: final dataset assembled")
                return
        except Exception:  # never let one bad step kill a multi-hour run
            log("error in loop:\n" + traceback.format_exc())
        time.sleep(60)


def status() -> None:
    conn = db()
    rows = conn.execute("SELECT stage, COUNT(*) n FROM batch GROUP BY stage").fetchall()
    acc = sum(1 for _ in (FINAL / "accepted").glob("*/meta.json")) if (FINAL / "accepted").exists() else 0
    rej = sum(1 for _ in (FINAL / "rejected").glob("*/*/meta.json")) if (FINAL / "rejected").exists() else 0
    pend = sum(1 for _ in (FINAL / "_pending").glob("*/meta.json")) if (FINAL / "_pending").exists() else 0
    print("batches:", {r["stage"]: r["n"] for r in rows}, f"| patches accepted {acc} rejected {rej} awaiting model {pend}")
    for b in conn.execute("SELECT id, stage, ada_job, n_for_model, n_accepted FROM batch WHERE stage NOT IN ('sorted','queued')"):
        print(f"  {b['id']} {b['stage']} job={b['ada_job']} for_model={b['n_for_model']} accepted={b['n_accepted']}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"setup": setup, "run": run, "status": status}.get(command, lambda: None)()
    if command == "preprocess":
        preprocess(sys.argv[2])
