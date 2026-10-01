#!/usr/bin/env python3
"""Concurrent, resumable download of the all-India before/after dataset.

    python download_batch.py seed                       # load locations/india_aois.csv into the DB
    python download_batch.py run --workers 8            # download everything still unfinished
    python download_batch.py run --limit 10 --select-only   # dry run: estimate yield, no downloads
    python download_batch.py run --retry failed,no_pairs    # also re-attempt those
    python download_batch.py status [--watch 10]        # progress, rate, ETA, failures
    python download_batch.py workers 24                 # change concurrency of a live run

State lives in ``data/download_state.sqlite``. Stopping (Ctrl-C) or crashing at any point is
safe: rerun the same command and it continues with whatever is not ``done``.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import locations
from aoi_download import Config, process_aoi
from stac_select import make_session
from state_db import STATUSES, StateDB

DEFAULT_DB = Path("data/download_state.sqlite")
DEFAULT_OUT = Path("data/sentinel2")
PATCH = 256
MAX_WORKERS = 64  # ceiling for live changes via ``download_batch.py workers N``


def _patches(shape: str | None) -> int:
    if not shape:
        return 0
    height, width = (int(v) for v in shape.split("x"))
    return (height // PATCH) * (width // PATCH)


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


# ------------------------------------------------------------------ status --------------
def format_status(db: StateDB, top: int = 8) -> str:
    s = db.summary()
    c, total = s["counts"], max(s["total"], 1)
    finished = c["done"] + c["no_pairs"] + c["failed"]
    patches = sum(_patches(r["shape"]) for r in db._conn().execute("SELECT shape FROM pair"))
    target = db.get_meta("target_patches", 15000)
    lines = [
        "== Download status ==",
        f"Workers   {db.get_meta('workers', '?')} (change live: download_batch.py workers N)",
        f"AOIs      {s['total']:>6}  |  " + "  ".join(f"{k} {c[k]}" for k in STATUSES),
        f"Progress  {finished}/{s['total']} AOIs finished ({finished / total:.1%})",
        f"Pairs     {s['pairs']} saved = {patches} patch-pairs (256px) of target {target} "
        f"({patches / max(target, 1):.1%})",
        f"Disk      {_human_bytes(s['bytes'])}",
    ]
    if finished:
        yield_per_aoi = patches / finished
        lines.append(f"Yield     {yield_per_aoi:.1f} patch-pairs per finished AOI "
                     f"-> projected {yield_per_aoi * s['total']:.0f} over all {s['total']} AOIs"
                     + ("" if yield_per_aoi * s["total"] >= target else "  ** below target: add AOIs **"))
    selected = db._conn().execute(
        "SELECT COUNT(*) n, COALESCE(SUM(n_pairs), 0) pairs, AVG(size_km) km FROM aoi "
        "WHERE status='selected'").fetchone()
    if selected["n"]:
        per_pair = (int(round((selected["km"] or 5.12) * 100)) // PATCH) ** 2
        estimate = selected["pairs"] * per_pair
        searched = selected["n"] + c["no_pairs"]
        hit_rate = selected["n"] / max(searched, 1)
        lines.append(f"Selected  {selected['n']} AOIs have pairs ({hit_rate:.0%} of those searched) -> "
                     f"~{estimate} patch-pairs if downloaded; at this hit rate all {s['total']} AOIs "
                     f"-> ~{estimate / max(selected['n'], 1) * hit_rate * s['total']:.0f}")
    if s["recent_finished"]:
        rate = s["recent_finished"] / s["recent_minutes"] * 60
        remaining = c["pending"] + c["selected"] + c["running"]
        lines.append(f"Rate      {rate:.0f} AOIs/hour over last {s['recent_minutes']:.0f} min"
                     f" -> ETA {_duration(remaining / rate * 3600) if rate else '?'} for {remaining} remaining")
    if s["running"]:
        lines.append(f"Running now ({len(s['running'])}):")
        for r in s["running"][:top]:
            lines.append(f"  {r['key'][:44]:<44} {r['stage'] or '':<12} {r['stage_note'] or ''} "
                         f"({_duration(time.time() - (r['started_at'] or time.time()))})")
    for title, rows, label in (("By state", s["by_state"], "state"), ("By category", s["by_category"], "category")):
        lines.append(f"{title}:")
        for r in rows[:top]:
            lines.append(f"  {str(r[label])[:28]:<28} aois {r['aois']:>4}  done {r['done'] or 0:>4}  pairs {r['pairs'] or 0:>4}")
        if len(rows) > top:
            lines.append(f"  ... {len(rows) - top} more")
    bad = db.failures(5)
    if bad:
        lines.append("Recent failures / empty:")
        for r in bad:
            lines.append(f"  [{r['status']}] {r['key'][:40]}: {str(r['reason'])[:110]}")
    return "\n".join(lines)


def cmd_workers(args) -> None:
    if not 1 <= args.n <= MAX_WORKERS:
        raise SystemExit(f"workers must be between 1 and {MAX_WORKERS}")
    db = StateDB(args.db)
    db.set_meta("workers", args.n)
    db.log(None, "info", f"workers set to {args.n}")
    print(f"workers set to {args.n}; a running download picks this up within a few seconds "
          "(extra sites start immediately, fewer takes effect as sites finish)")


def cmd_status(args) -> None:
    db = StateDB(args.db)
    while True:
        text = format_status(db)
        print(("\033[2J\033[H" if args.watch else "") + text, flush=True)
        if not args.watch:
            return
        time.sleep(args.watch)


# --------------------------------------------------------------------- seed -------------
def cmd_seed(args) -> None:
    db = StateDB(args.db)
    registry = locations.load_registry(Path(args.registry))
    if not registry:
        raise SystemExit(f"registry {args.registry} is missing or empty")
    result = db.seed(registry)
    if args.target:
        db.set_meta("target_patches", args.target)
    print(f"seeded {len(registry)} AOIs: {result['added']} new, {result['existing']} refreshed")


# ---------------------------------------------------------------------- run -------------
def cmd_run(args) -> None:
    db = StateDB(args.db)
    lock = open(str(args.db) + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another download run is already using this database")

    if not args.no_seed:
        registry = locations.load_registry(Path(args.registry))
        if registry:
            print("registry:", db.seed(registry))
    if args.target:
        db.set_meta("target_patches", args.target)
    recovered = db.recover_interrupted()
    if recovered:
        print(f"re-queued {recovered} AOIs left 'running' by an interrupted run")

    statuses = ["pending", "selected", *[s for s in (args.retry or "").split(",") if s]]
    keys = [k.strip() for k in args.keys.split(",") if k.strip()] if args.keys else None
    rows = db.queue(statuses, category=args.category, state=args.state, limit=args.limit, keys=keys)
    if not rows:
        print("nothing to do: no AOIs match the queue")
        print(format_status(db))
        return

    cfg = Config(before_season=args.before_season, after_season=args.after_season,
                 max_aoi_cloud=args.max_aoi_cloud, n_pairs=args.pairs_per_aoi)
    stop = threading.Event()
    sessions = threading.local()
    run_id = db.begin_run(args.workers, vars(args))
    db.set_meta("workers", args.workers)
    processed = 0
    outcomes: dict[str, int] = {}
    started = time.time()

    def on_sigint(_signum, _frame):
        if stop.is_set():
            print("\nforcing exit; unfinished AOIs will be re-queued on the next run", flush=True)
            os._exit(130)
        stop.set()
        print("\nstopping after the AOIs in flight (Ctrl-C again to force)...", flush=True)

    previous_handler = signal.signal(signal.SIGINT, on_sigint)

    def work(row: dict) -> str:
        if not hasattr(sessions, "session"):
            sessions.session = make_session()
        return process_aoi(row, args.out, cfg, db, sessions.session, stop=stop,
                           select_only=args.select_only)

    if args.stop_after_pairs:
        print(f"will stop starting new sites once {args.stop_after_pairs} pairs are saved")
    print(f"queue: {len(rows)} AOIs, {args.workers} workers"
          + (" (select-only: nothing is downloaded)" if args.select_only else ""))
    last_report = 0.0
    pending_iter = iter(rows)
    in_flight: dict = {}
    # The pool is sized for the ceiling; how many sites run at once is read from the DB each
    # loop, so ``download_batch.py workers N`` retunes a live run without restarting it.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        while True:
            stop_submitting = bool(args.stop_after_pairs) and db.pairs_saved() >= args.stop_after_pairs
            target = min(MAX_WORKERS, max(1, int(db.get_meta("workers", args.workers))))
            while not stop.is_set() and not stop_submitting and len(in_flight) < target:
                row = next(pending_iter, None)
                if row is None:
                    break
                in_flight[pool.submit(work, row)] = row["key"]
            if not in_flight:
                break
            done, _ = wait(in_flight, timeout=2, return_when=FIRST_COMPLETED)
            for future in done:
                key = in_flight.pop(future)
                try:
                    outcome = future.result()
                except Exception as error:  # process_aoi records its own failures; this is a bug guard
                    outcome = "failed"
                    db.log(key, "error", f"unhandled: {error!r}")
                    db.finish(key, "failed", error=f"unhandled {type(error).__name__}: {error}"[:500])
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                processed += 1
                print(f"[{processed}/{len(rows)}] {key}: {outcome}", flush=True)
            if time.time() - last_report > args.report:
                last_report = time.time()
                s = db.summary()
                print(f"  -- {', '.join(f'{k} {v}' for k, v in s['counts'].items() if v)} | "
                      f"pairs {s['pairs']} | {_human_bytes(s['bytes'])} | "
                      f"elapsed {_duration(time.time() - started)}", flush=True)

    signal.signal(signal.SIGINT, previous_handler)
    db.end_run(run_id, processed)
    print(f"\nrun finished in {_duration(time.time() - started)}: "
          + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))
    print(format_status(db))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="load the AOI registry CSV into the database")
    seed.add_argument("--registry", default=str(locations.REGISTRY_PATH))
    seed.add_argument("--target", type=int, help="goal in 256px patch-pairs (default 15000)")
    seed.set_defaults(func=cmd_seed)

    run = sub.add_parser("run", help="download every unfinished AOI")
    run.add_argument("--registry", default=str(locations.REGISTRY_PATH))
    run.add_argument("--no-seed", action="store_true", help="do not re-read the registry CSV first")
    run.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run.add_argument("--workers", type=int, default=48, help=f"sites at once (1-{MAX_WORKERS}; changeable live)")
    run.add_argument("--stop-after-pairs", type=int, metavar="N",
                     help="stop starting new sites once N pairs are saved in total (pilot runs)")
    run.add_argument("--limit", type=int, help="process at most this many AOIs")
    run.add_argument("--category")
    run.add_argument("--state")
    run.add_argument("--keys", help="comma-separated AOI keys to restrict to")
    run.add_argument("--retry", help="also re-attempt AOIs in these statuses, e.g. failed,no_pairs")
    run.add_argument("--select-only", action="store_true", help="find pairs but download nothing")
    run.add_argument("--before-season", type=int, default=2019, help="year the before season starts")
    run.add_argument("--after-season", type=int, default=2025, help="year the after season starts")
    run.add_argument("--max-aoi-cloud", type=float, default=0.01)
    run.add_argument("--pairs-per-aoi", type=int, default=1)
    run.add_argument("--target", type=int)
    run.add_argument("--report", type=float, default=60, help="seconds between progress lines")
    run.set_defaults(func=cmd_run)

    workers = sub.add_parser("workers", help="change how many sites a running download works on at once")
    workers.add_argument("n", type=int)
    workers.set_defaults(func=cmd_workers)

    status = sub.add_parser("status", help="show progress")
    status.add_argument("--watch", type=float, metavar="SECONDS", help="refresh continuously")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
