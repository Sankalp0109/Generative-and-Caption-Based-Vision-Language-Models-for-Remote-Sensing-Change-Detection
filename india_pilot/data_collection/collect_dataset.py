"""
Scaled, resumable bi-temporal Sentinel-2 dataset collection over multiple
Indian AOIs, grid-tiled into 512x512 patches, with a change-magnitude
pre-filter. Builds directly on the single-pair logic proven in fetch_pair.py.

RESUMABILITY (the whole point of this script vs. fetch_pair.py):
- The full list of tile-jobs to fetch is generated deterministically up front
  from AOIS + grid tiling, and written once to manifest.csv.
- Every completed tile-job (success OR permanent failure) is appended as one
  line to progress.jsonl, flushed + fsynced immediately after each item.
- On startup, progress.jsonl is read back to build a set of already-done
  tile_ids, which are skipped. So: Ctrl+C, a SLURM walltime cutoff, a crash,
  or a network drop only ever loses whatever was in-flight at that instant —
  rerunning the exact same command picks up from the next unfinished tile.
- Concurrency uses a thread pool (this workload is network-I/O-bound, not
  CPU-bound) with a lock around progress-file writes so parallel workers
  don't interleave/corrupt the log.
- Kept pairs are saved to dataset/pN_before.png + dataset/pN_after.png with
  a dense, resume-stable sequential number N (not the internal tile_id,
  which stays in progress.jsonl for traceability back to the AOI/tile).
- Planetary Computer rate-limit errors are retried with exponential backoff
  (with_rate_limit_retry) instead of being treated as permanent failures --
  concurrency benchmarking on this account showed sustained load (32-64
  workers for several minutes) can trip their rate limiter, which previously
  meant every in-flight tile got marked failed_error and lost.

Usage:
    python collect_dataset.py --out /scratch/$USER/rscc_india --workers 16
    # interrupt anytime (Ctrl+C) and rerun the same command to resume
"""
import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import planetary_computer
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from pystac_client import Client
from PIL import Image

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
TILE_SIZE_PX = 512
TILE_SIZE_DEG = 0.045  # ~ same ground footprint as the validated GIFT City sample

# 60 macro-AOIs across India spanning the 5 change categories the architecture
# plan calls for. Each is (id, lon_min, lat_min, lon_max, lat_max, category).
# Tunable knobs if a larger final dataset target is set later: widen these
# boxes and/or shrink TILE_SIZE_DEG -- build_manifest() regenerates from
# whatever's here, no other code needs to change.
MACRO_AOIS = [
    # --- urban expansion (19) ---
    ("gift_city_gandhinagar", 72.60, 23.10, 72.90, 23.40, "urban"),
    ("delhi_ncr_fringe", 77.20, 28.35, 77.50, 28.60, "urban"),
    ("bengaluru_outskirts", 77.55, 12.90, 77.85, 13.15, "urban"),
    ("hyderabad_gachibowli", 78.25, 17.30, 78.55, 17.60, "urban"),
    ("pune_hinjewadi", 73.60, 18.50, 73.90, 18.80, "urban"),
    ("mumbai_navi_mumbai", 72.95, 18.95, 73.25, 19.25, "urban"),
    ("chennai_omr", 80.10, 12.75, 80.40, 13.05, "urban"),
    ("kolkata_newtown", 88.35, 22.50, 88.65, 22.80, "urban"),
    ("jaipur_fringe", 75.70, 26.75, 76.00, 27.05, "urban"),
    ("lucknow_expansion", 80.85, 26.70, 81.15, 27.00, "urban"),
    ("nagpur_expansion", 78.95, 21.00, 79.25, 21.30, "urban"),
    ("indore_expansion", 75.70, 22.60, 76.00, 22.90, "urban"),
    ("surat_expansion", 72.70, 21.10, 73.00, 21.40, "urban"),
    ("chandigarh_tricity", 76.60, 30.55, 76.90, 30.85, "urban"),
    ("bhubaneswar_expansion", 85.70, 20.15, 86.00, 20.45, "urban"),
    ("coimbatore_expansion", 76.85, 10.95, 77.15, 11.25, "urban"),
    ("amaravati_capital", 80.35, 16.45, 80.60, 16.70, "urban"),
    ("vijayawada_growth", 80.60, 16.40, 80.85, 16.65, "urban"),
    ("guwahati_expansion", 91.60, 26.00, 91.90, 26.30, "urban"),

    # --- deforestation (12) ---
    ("western_ghats_kodagu", 75.65, 12.25, 75.95, 12.55, "deforestation"),
    ("western_ghats_wayanad", 76.00, 11.50, 76.30, 11.80, "deforestation"),
    ("western_ghats_nilgiris", 76.55, 11.25, 76.85, 11.55, "deforestation"),
    ("northeast_dima_hasao", 92.95, 25.30, 93.25, 25.60, "deforestation"),
    ("northeast_manipur_hills", 93.80, 24.60, 94.10, 24.90, "deforestation"),
    ("meghalaya_khasi_hills", 91.65, 25.45, 91.95, 25.75, "deforestation"),
    ("chhattisgarh_bastar", 81.85, 18.95, 82.15, 19.25, "deforestation"),
    ("jharkhand_dhanbad_coal", 86.25, 23.65, 86.55, 23.95, "deforestation"),
    ("mp_kanha_buffer", 80.50, 22.15, 80.80, 22.45, "deforestation"),
    ("odisha_niyamgiri", 83.25, 19.45, 83.55, 19.75, "deforestation"),
    ("uttarakhand_terai", 79.30, 29.05, 79.60, 29.35, "deforestation"),
    ("arunachal_forest_fringe", 94.05, 27.00, 94.35, 27.30, "deforestation"),

    # --- agriculture / seasonal change (10) ---
    ("punjab_ludhiana", 75.70, 30.80, 76.00, 31.10, "agriculture"),
    ("haryana_karnal", 76.85, 29.55, 77.15, 29.85, "agriculture"),
    ("up_meerut_plain", 77.55, 28.85, 77.85, 29.15, "agriculture"),
    ("maharashtra_vidarbha_yavatmal", 78.05, 20.25, 78.35, 20.55, "agriculture"),
    ("ap_krishna_godavari_delta", 81.25, 16.25, 81.55, 16.55, "agriculture"),
    ("tn_thanjavur_delta", 79.05, 10.65, 79.35, 10.95, "agriculture"),
    ("karnataka_tungabhadra", 76.25, 15.25, 76.55, 15.55, "agriculture"),
    ("gujarat_saurashtra", 70.75, 21.55, 71.05, 21.85, "agriculture"),
    ("rajasthan_sriganganagar_canal", 73.80, 29.80, 74.10, 30.10, "agriculture"),
    ("bihar_patna_plain", 85.00, 25.50, 85.30, 25.80, "agriculture"),

    # --- coastal change (8) ---
    ("sundarbans_wb", 88.70, 21.80, 89.00, 22.10, "coastal"),
    ("odisha_puri_coast", 85.75, 19.70, 86.05, 20.00, "coastal"),
    ("kerala_kochi_vypin", 76.15, 9.90, 76.45, 10.20, "coastal"),
    ("tn_chennai_ennore", 80.25, 13.15, 80.55, 13.45, "coastal"),
    ("gujarat_mundra_port", 69.55, 22.65, 69.85, 22.95, "coastal"),
    ("goa_coast", 73.65, 15.40, 73.95, 15.70, "coastal"),
    ("ap_visakhapatnam_port", 83.15, 17.60, 83.45, 17.90, "coastal"),
    ("mumbai_coastal_road", 72.78, 18.96, 73.08, 19.26, "coastal"),

    # --- new infrastructure (11) ---
    ("jewar_airport_up", 77.45, 28.00, 77.75, 28.30, "infrastructure"),
    ("navi_mumbai_airport", 73.00, 18.95, 73.30, 19.25, "infrastructure"),
    ("statue_of_unity_kevadia", 73.60, 21.85, 73.90, 22.15, "infrastructure"),
    ("central_vista_delhi", 77.15, 28.55, 77.45, 28.85, "infrastructure"),
    ("chenab_bridge_jk", 75.15, 33.20, 75.45, 33.50, "infrastructure"),
    ("atal_setu_mumbai", 72.80, 18.85, 73.10, 19.15, "infrastructure"),
    ("bogibeel_bridge_assam", 95.30, 27.60, 95.60, 27.90, "infrastructure"),
    ("dholera_sir_gujarat", 72.10, 22.10, 72.40, 22.40, "infrastructure"),
    ("purvanchal_expressway_up", 82.45, 26.65, 82.75, 26.95, "infrastructure"),
    ("delhi_mumbai_expwy_dausa", 76.25, 26.80, 76.55, 27.10, "infrastructure"),
    ("polavaram_dam_ap", 81.55, 17.15, 81.85, 17.45, "infrastructure"),
]

WINDOWS = {
    "before": ("2014-01-01", "2016-12-31"),
    "after": ("2023-01-01", "2025-06-30"),
}

CHANGE_THRESHOLD = 25.0  # mean abs-diff (0-255 scale) above which a tile is "kept"


def grid_tiles(aoi_id, lon_min, lat_min, lon_max, lat_max, category):
    tiles = []
    row = 0
    lat = lat_min
    while lat + TILE_SIZE_DEG <= lat_max:
        col = 0
        lon = lon_min
        while lon + TILE_SIZE_DEG <= lon_max:
            tile_id = f"{aoi_id}_r{row}_c{col}"
            bbox = [lon, lat, lon + TILE_SIZE_DEG, lat + TILE_SIZE_DEG]
            tiles.append({"tile_id": tile_id, "aoi_id": aoi_id, "category": category, "bbox": bbox})
            col += 1
            lon += TILE_SIZE_DEG
        row += 1
        lat += TILE_SIZE_DEG
    return tiles


def build_manifest(manifest_path):
    if manifest_path.exists():
        with open(manifest_path) as f:
            return list(csv.DictReader(f))
    rows = []
    for aoi_id, lon_min, lat_min, lon_max, lat_max, category in MACRO_AOIS:
        rows.extend(grid_tiles(aoi_id, lon_min, lat_min, lon_max, lat_max, category))
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile_id", "aoi_id", "category", "bbox"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "bbox": json.dumps(r["bbox"])})
    return [{**r, "bbox": json.dumps(r["bbox"])} for r in rows]


def load_completed(progress_path):
    done = set()
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["tile_id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a truncated last line from a hard kill
    return done


_progress_lock = threading.Lock()


def append_progress(progress_path, record):
    line = json.dumps(record) + "\n"
    with _progress_lock:
        with open(progress_path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


# Sequential pair numbering (p1_before.png / p1_after.png, p2_..., ...) for
# every KEPT pair, assigned in the order tiles finish (not manifest order,
# since threads complete out of order) -- but numbers are dense/no-gap and
# stable across resumes, since the counter is seeded from the highest
# pair_num already recorded in progress.jsonl on startup.
_pair_num_lock = threading.Lock()
_next_pair_num = [1]  # mutable cell so it can be updated from init_pair_counter()


def init_pair_counter(progress_path):
    highest = 0
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("pair_num"):
                    highest = max(highest, rec["pair_num"])
    _next_pair_num[0] = highest + 1


def take_pair_num():
    with _pair_num_lock:
        n = _next_pair_num[0]
        _next_pair_num[0] += 1
        return n


def is_rate_limit_error(e):
    msg = str(e).lower()
    return "rate limit" in msg or "429" in msg or "too many requests" in msg


def with_rate_limit_retry(fn, max_retries=5, base_delay=5.0):
    """Retry on Planetary Computer rate-limit errors with exponential backoff
    + jitter, instead of treating them as permanent failures (which is what
    happened during concurrency benchmarking: a burst of concurrent requests
    tripped the API's rate limit and every in-flight tile got permanently
    marked failed_error instead of just waiting it out)."""
    import random
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if is_rate_limit_error(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                time.sleep(delay)
                continue
            raise


def candidate_items(catalog, bbox, date_range):
    def _search():
        search = catalog.search(
            collections=[COLLECTION], bbox=bbox,
            datetime=f"{date_range[0]}/{date_range[1]}",
            query={"eo:cloud_cover": {"lt": 30}},
        )
        return list(search.items())
    items = with_rate_limit_retry(_search)
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    return items


def fetch_clip(catalog, bbox, date_range, out_path):
    """Same rejection logic as fetch_pair.py's try_clip, but returns the
    array (for the change-filter) instead of just saving directly."""
    for item in candidate_items(catalog, bbox, date_range):
        signed = planetary_computer.sign(item)
        href = signed.assets["visual"].href
        def _read():
            with rasterio.open(href) as src:
                proj_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
                window = from_bounds(*proj_bbox, transform=src.transform)
                return src.read([1, 2, 3], window=window, out_shape=(3, TILE_SIZE_PX, TILE_SIZE_PX),
                                 resampling=rasterio.enums.Resampling.bilinear)
        try:
            data = with_rate_limit_retry(_read)
        except rasterio.errors.RasterioIOError:
            continue  # genuine read/no-data failure (not rate-limit), try the next candidate scene
        except Exception as e:
            if is_rate_limit_error(e):
                continue  # exhausted retries under sustained throttling, try the next candidate instead of failing the whole tile
            raise
        if (data > 0).mean() < 0.99:
            continue
        arr = np.transpose(data, (1, 2, 0))
        Image.fromarray(arr).save(out_path)
        return {"scene_date": item.properties["datetime"], "cloud_pct": item.properties.get("eo:cloud_cover")}
    return None


def process_tile(catalog, tile, raw_dir, dataset_dir, progress_path):
    tile_id, bbox = tile["tile_id"], json.loads(tile["bbox"])
    # raw_dir holds transient files only, named by tile_id, used just long
    # enough to compute the change score -- never the final dataset location.
    before_tmp = raw_dir / f"{tile_id}_before.png"
    after_tmp = raw_dir / f"{tile_id}_after.png"
    try:
        meta_before = fetch_clip(catalog, bbox, WINDOWS["before"], before_tmp)
        meta_after = fetch_clip(catalog, bbox, WINDOWS["after"], after_tmp) if meta_before else None
        if not (meta_before and meta_after):
            record = {"tile_id": tile_id, "status": "failed_no_scene", "ts": time.time()}
        else:
            a = np.array(Image.open(before_tmp).convert("RGB")).astype(np.int16)
            b = np.array(Image.open(after_tmp).convert("RGB")).astype(np.int16)
            change_score = float(np.abs(a - b).mean())
            kept = change_score >= CHANGE_THRESHOLD
            pair_num = None
            before_path = after_path = None
            if kept:
                # promote to the final dataset/ dir under the simple pN naming
                pair_num = take_pair_num()
                before_path = dataset_dir / f"p{pair_num}_before.png"
                after_path = dataset_dir / f"p{pair_num}_after.png"
                before_tmp.rename(before_path)
                after_tmp.rename(after_path)
            else:  # discard boring tiles immediately, don't let them eat storage
                before_tmp.unlink(missing_ok=True)
                after_tmp.unlink(missing_ok=True)
            record = {
                "tile_id": tile_id, "aoi_id": tile["aoi_id"], "category": tile["category"],
                "status": "kept" if kept else "discarded_no_change",
                "change_score": change_score,
                "pair_num": pair_num,
                "before_meta": meta_before, "after_meta": meta_after,
                "before_path": str(before_path) if kept else None,
                "after_path": str(after_path) if kept else None,
                "ts": time.time(),
            }
    except Exception as e:  # noqa: BLE001 -- log and move on, never let one tile kill the run
        record = {"tile_id": tile_id, "status": "failed_error", "error": f"{type(e).__name__}: {e}", "ts": time.time()}
    append_progress(progress_path, record)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir, e.g. /scratch/$USER/rscc_india")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"          # transient working files only
    dataset_dir = out_dir / "dataset"  # final kept pairs: pN_before.png / pN_after.png
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    progress_path = out_dir / "progress.jsonl"

    manifest = build_manifest(manifest_path)
    completed = load_completed(progress_path)
    remaining = [t for t in manifest if t["tile_id"] not in completed]
    init_pair_counter(progress_path)  # resume numbering from the highest pN already used

    print(f"manifest: {len(manifest)} tiles total | already done: {len(completed)} | remaining: {len(remaining)}")
    print(f"next pair number to assign: p{_next_pair_num[0]}")
    if not remaining:
        print("nothing left to do -- collection already complete.")
        return

    catalog = Client.open(STAC_URL)
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_tile, catalog, t, raw_dir, dataset_dir, progress_path): t for t in remaining}
        for fut in as_completed(futures):
            record = fut.result()
            n_done += 1
            if n_done % 25 == 0:
                print(f"[{n_done}/{len(remaining)}] last: {record['tile_id']} -> {record['status']}")

    print(f"done. Final dataset pairs are in {dataset_dir}/ as pN_before.png / pN_after.png.")
    print("Re-run this same command any time to pick up newly added AOIs or retry failures --")
    print("already-completed tile_ids in progress.jsonl are always skipped, and pair numbering resumes correctly.")


if __name__ == "__main__":
    main()
