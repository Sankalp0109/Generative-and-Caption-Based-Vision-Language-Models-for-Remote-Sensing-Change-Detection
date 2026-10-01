"""Download the before/after pairs for one site, reporting progress into the state DB.

A site is a centre point; the pair covers the 1024 x 1024 px Sentinel-2 block (10.24 km) that
holds it, on whichever tile holds it most centrally. Before and after come from the same tile,
so they share one pixel grid and the block is read directly from both, with no reprojection or
resampling: the saved pixels are the source pixels.

``process_aoi`` is the unit of work the batch runner schedules. It is safe to rerun on a site
that was interrupted: a pair is built in a ``.tmp`` folder and renamed into place only once
complete, and earlier partial output for the site is discarded first.

Output layout (one folder per site, so regions can never mix)::

    <out_root>/<site_key>/pair_<n>/before.tif      uint8 RGB, 1024 x 1024 (Sentinel-2 true colour)
                                  /after.tif       same grid as before.tif
                                  /scl_before.tif, scl_after.tif   scene classification masks
                                  /pair.json       site, block, scene ids, dates, cloud, sun angle
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window, from_bounds

from state_db import StateDB
from stac_select import (BLOCK_PX, DEFAULT_CATALOG, DEFAULT_COLLECTION, GDAL_OPTIONS, PIXEL_M, SCL_BAD,
                         candidate_blocks, is_diverse, item_date, rank_pairs, search, season_range)

DEFAULT_SEASON = "11,12,1,2,3"
SEARCH_RADIUS_DEG = 0.001  # the catalogue search only needs to find tiles holding the centre


@dataclass(frozen=True)
class Config:
    catalog: str = DEFAULT_CATALOG
    collection: str = DEFAULT_COLLECTION
    before_season: int = 2019      # July-June cycle: 2019 -> Jul 2019 .. Jun 2020
    after_season: int = 2025
    tile_cloud: float = 40.0       # catalogue prefilter only; the real gate is max_aoi_cloud
    max_aoi_cloud: float = 0.01    # fraction of block pixels that may be cloud/shadow/nodata
    max_nodata: float = 0.001      # fraction of all-zero pixels tolerated in a saved image
    max_snow_diff: float = 0.05    # snow-cover difference tolerated between before and after
    n_pairs: int = 1               # pairs kept per site
    top_k: int = 8                 # lowest tile-cloud scenes per side given the block-cloud check
    max_gap_days: int = 30         # before/after position within their seasons
    min_separation_days: int = 15  # between pairs of one site
    max_tiles: int = 4             # overlapping tiles tried per site (up to 4 meet at a corner)
    max_snow: float = 0.20         # scene snow-cover share above which it is not used
    max_water: float = 0.70        # scene water share above which it is not used (open sea)
    max_downloads: int = 4         # candidate pairs downloaded per site before giving up
    network_retries: int = 3       # whole-site retries on transient network errors (DNS, resets)
    retry_wait_s: float = 20.0     # first retry wait; doubles each time


class Stopped(Exception):
    """Raised inside a worker when a shutdown was requested between steps."""


TRANSIENT_MARKERS = ("could not resolve host", "connection reset", "connection refused", "timed out",
                     "timeout", "temporary failure", "ssl", "curl error", "connection aborted",
                     "remote end closed", "max retries exceeded", "503", "502", "504",
                     # after a network failure GDAL caches the URL as missing, then reports:
                     "does not exist in the file system", "read failed")


def clear_gdal_http_cache() -> None:
    """Forget GDAL's cached HTTP results, including URLs it wrongly cached as missing.

    After a DNS failure GDAL remembers the URL as "does not exist", so a retry in the same
    process fails again without touching the network unless this cache is cleared.
    """
    global _gdal_lib
    try:
        if _gdal_lib is None:
            import ctypes
            import glob
            libs = glob.glob(os.path.join(os.path.dirname(rasterio.__file__), os.pardir,
                                          "rasterio.libs", "libgdal*.so*"))
            _gdal_lib = ctypes.CDLL(libs[0]) if libs else ctypes.CDLL("libgdal.so")
        _gdal_lib.VSICurlClearCache()
    except (OSError, AttributeError, IndexError):
        pass  # no way to clear it: the retry may still work once the negative cache expires


_gdal_lib = None


def is_transient(error: BaseException) -> bool:
    """Network hiccups worth retrying the whole site for, as opposed to real data problems."""
    import requests
    if isinstance(error, (requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError)):
        return True
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def save_geotiff(path: Path, array: np.ndarray, grid: dict) -> None:
    """Write an ``HxW`` or ``HxWxC`` uint8 array as a compressed, tiled GeoTIFF."""
    array = array[..., None] if array.ndim == 2 else array
    temporary = path.with_suffix(path.suffix + ".part")
    with rasterio.open(temporary, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
                       count=array.shape[2], dtype="uint8", crs=grid["crs"],
                       transform=Affine(*grid["transform"][:6]), compress="deflate",
                       predictor=2, tiled=True, blockxsize=256, blockysize=256) as out:
        out.write(np.moveaxis(array, -1, 0))
    os.replace(temporary, path)


def read_block(href: str, block: dict, bands: int | None = None) -> np.ndarray:
    """The block's pixels from ``href``, exactly as stored (``HxWxC`` uint8).

    Refuses a source whose grid does not line up with the block rather than resampling it,
    so a saved image is always the original pixels.
    """
    with rasterio.open(href) as src:
        if src.crs is None or src.crs.to_epsg() != block["epsg"]:
            raise ValueError(f"{href}: CRS {src.crs} does not match block EPSG:{block['epsg']}")
        window = from_bounds(*block["bounds"], transform=src.transform)
        offsets = (window.col_off, window.row_off, window.width, window.height)
        if any(abs(v - round(v)) > 1e-6 for v in offsets):
            raise ValueError(f"{href}: block is not aligned to this image's pixel grid")
        window = Window(*(int(round(v)) for v in offsets))
        count = min(src.count, bands or src.count)
        data = src.read(list(range(1, count + 1)), window=window)
    return np.moveaxis(data, 0, -1)


HAZE_DARK_FLOOR_DIFF = 25              # 0-255 gap in darkest-blue level between the dates
WHITE_LEVEL, WHITE_FRACTION = 245, 0.30  # a date this white (snow, cloud, clipped ground) has no detail


class QualityRejected(ValueError):
    """A downloaded pair failed an image check; ``scene`` is the date to avoid next time, if known."""

    def __init__(self, message: str, scene: str | None = None):
        super().__init__(message)
        self.scene = scene


def quality_flags(before: np.ndarray, after: np.ndarray) -> dict:
    """Checks the SCL cloud mask misses, computed from the raw true-colour pixels of both dates.

    * haze: haze and fog are not cloud in the SCL, but they lift the darkest pixels. Before and
      after show the same place in the same season, so their dark floors should match; a big
      gap means haze in one date. Bright ground (desert) is bright in both, so it passes.
    * white: a date that is mostly saturated white (snow, cloud, or bright ground clipped by the
      8-bit product) carries no detail to describe.
    """
    stats = {}
    for role, image in (("before", before), ("after", after)):
        valid = image.max(axis=2) > 0
        stats[f"{role}_dark_blue"] = float(np.percentile(image[..., 2][valid], 1)) if valid.any() else 0.0
        stats[f"{role}_white"] = float((image.min(axis=2) >= WHITE_LEVEL).mean())
    flags = []
    if abs(stats["before_dark_blue"] - stats["after_dark_blue"]) > HAZE_DARK_FLOOR_DIFF:
        flags.append("haze")
    if max(stats["before_white"], stats["after_white"]) > WHITE_FRACTION:
        flags.append("white")
    return {"flags": flags, **{k: round(v, 3) for k, v in stats.items()}}


def _zero_fraction(image: np.ndarray) -> float:
    return float(np.all(image == 0, axis=-1).mean())


def _scene_meta(item: dict, block_cloud: float) -> dict:
    p = item["properties"]
    return {"item_id": item["id"], "datetime": p.get("datetime"), "tile": p.get("grid:code"),
            "tile_cloud": p.get("eo:cloud_cover"), "aoi_cloud": block_cloud,
            "sun_elevation": p.get("view:sun_elevation"), "sun_azimuth": p.get("view:sun_azimuth"),
            "processing_baseline": p.get("s2:processing_baseline"),
            "visual": item["assets"]["visual"]["href"]}


def save_pair(candidate: dict, site: dict, destination: Path, cfg: Config) -> dict:
    """Read, validate and write one pair. Raises ValueError/OSError on any unusable pair."""
    before_item, after_item, block = candidate["before"], candidate["after"], candidate["block"]
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        with rasterio.Env(**GDAL_OPTIONS):
            images = {role: read_block(item["assets"]["visual"]["href"], block, bands=3)
                      for role, item in (("before", before_item), ("after", after_item))}
            for role, image in images.items():
                if image.shape[:2] != (BLOCK_PX, BLOCK_PX):
                    raise ValueError(f"{role} block is {image.shape[:2]}, expected {BLOCK_PX}px square")
                if _zero_fraction(image) > cfg.max_nodata:
                    raise ValueError(f"{role} image has {_zero_fraction(image):.2%} no-data pixels")
            quality = quality_flags(images["before"], images["after"])
            if "haze" in quality["flags"]:
                hazy = "before" if quality["before_dark_blue"] > quality["after_dark_blue"] else "after"
                raise QualityRejected(f"haze in {hazy} (dark floor {quality['before_dark_blue']:.0f} vs "
                                      f"{quality['after_dark_blue']:.0f})", candidate[hazy]["id"])
            if "white" in quality["flags"]:
                whiter = "before" if quality["before_white"] > quality["after_white"] else "after"
                both = min(quality["before_white"], quality["after_white"]) > WHITE_FRACTION
                raise QualityRejected(f"mostly white ({quality['before_white']:.0%} / {quality['after_white']:.0%})",
                                      None if both else candidate[whiter]["id"])
            # SCL is 20 m on the same tile grid: one 512 px read, doubled to the 10 m grid.
            masks = {role: read_block(item["assets"]["scl"]["href"], block, bands=1)[..., 0]
                     .repeat(2, axis=0).repeat(2, axis=1)
                     for role, item in (("before", before_item), ("after", after_item))}
        cloud = {role: float(np.isin(mask, SCL_BAD).mean()) for role, mask in masks.items()}
        grid = {"crs": f"EPSG:{block['epsg']}",
                "transform": (PIXEL_M, 0.0, block["bounds"][0], 0.0, -PIXEL_M, block["bounds"][3])}
        for role in ("before", "after"):
            save_geotiff(temporary / f"{role}.tif", images[role], grid)
            save_geotiff(temporary / f"scl_{role}.tif", masks[role], grid)
        record = {
            "site": {k: site.get(k) for k in ("key", "label", "state", "category", "lon", "lat")},
            "block": {"id": block["id"], "tile": block["tile"], "epsg": block["epsg"],
                      "col": block["col"], "row": block["row"], "bounds_utm": list(block["bounds"]),
                      "bbox_wgs84": list(block["bbox_wgs84"])},
            "crs": grid["crs"], "transform": list(grid["transform"]), "shape": [BLOCK_PX, BLOCK_PX],
            "gap_days": candidate["gap_days"], "quality": quality,
            "before": _scene_meta(before_item, cloud["before"]),
            "after": _scene_meta(after_item, cloud["after"]),
        }
        (temporary / "pair.json").write_text(json.dumps(record, indent=2) + "\n")
        size = sum(f.stat().st_size for f in temporary.iterdir())
        shutil.rmtree(destination, ignore_errors=True)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"before_item": before_item["id"], "after_item": after_item["id"],
            "before_date": item_date(before_item).isoformat(),
            "after_date": item_date(after_item).isoformat(), "tile": block["id"],
            "gap_days": candidate["gap_days"], "before_aoi_cloud": cloud["before"],
            "after_aoi_cloud": cloud["after"],
            "before_tile_cloud": before_item["properties"].get("eo:cloud_cover"),
            "after_tile_cloud": after_item["properties"].get("eo:cloud_cover"),
            "shape": f"{BLOCK_PX}x{BLOCK_PX}", "bytes": size, "path": str(destination)}


def select_pairs(site: dict, cfg: Config, session, stage=lambda *_: None,
                 check_stop=lambda: None) -> tuple[list[dict], dict]:
    """Search both seasons and return (ranked pairs for the best usable block, diagnostics)."""
    lon, lat = float(site["lon"]), float(site["lat"])
    months = site.get("season_months") or DEFAULT_SEASON
    max_aoi_cloud = cfg.max_aoi_cloud if site.get("max_aoi_cloud") is None else float(site["max_aoi_cloud"])
    box = (lon - SEARCH_RADIUS_DEG, lat - SEARCH_RADIUS_DEG, lon + SEARCH_RADIUS_DEG, lat + SEARCH_RADIUS_DEG)
    windows = {"before": season_range(cfg.before_season, months),
               "after": season_range(cfg.after_season, months)}
    found = {}
    for role, (start, end) in windows.items():
        check_stop()
        stage("searching", f"{role} season {start}..{end}")
        found[role] = search(session, cfg.catalog, cfg.collection, box, start, end, cfg.tile_cloud)

    blocks = candidate_blocks(found["before"], found["after"], lon, lat)[:cfg.max_tiles]
    diag = {"before_found": len(found["before"]), "after_found": len(found["after"]),
            "seasons": {r: [str(s), str(e)] for r, (s, e) in windows.items()},
            "tiles_tried": []}
    for block in blocks:
        check_stop()
        stage("selecting", f"{block['id']}: {len(found['before'])} before / {len(found['after'])} after scenes")
        pairs, tile_diag = rank_pairs(found["before"], found["after"], block,
                                      windows["before"][0], windows["after"][0],
                                      max_aoi_cloud=max_aoi_cloud, top_k=cfg.top_k,
                                      max_gap_days=cfg.max_gap_days, max_nodata=cfg.max_nodata,
                                      max_snow_diff=cfg.max_snow_diff, max_snow=cfg.max_snow,
                                      max_water=cfg.max_water)
        diag["tiles_tried"].append(tile_diag)
        if pairs:
            return pairs, diag
    return [], diag


def process_aoi(aoi: dict, out_root: Path, cfg: Config, db: StateDB, session,
                stop: threading.Event | None = None, select_only: bool = False) -> str:
    """Find and (unless ``select_only``) save the pairs for one site. Returns the final status.

    Transient network errors (DNS failures, resets, timeouts) retry the whole site a few times
    with backoff before it is recorded as ``failed``; anything else fails immediately.
    """
    key = aoi["key"]

    def check_stop():
        if stop is not None and stop.is_set():
            raise Stopped

    db.start(key)
    wait = cfg.retry_wait_s
    for attempt in range(cfg.network_retries + 1):
        try:
            return _process_once(aoi, out_root, cfg, db, session, check_stop, select_only)
        except Stopped:
            shutil.rmtree(out_root / key, ignore_errors=True)
            db.release_block(key)
            db.finish(key, "pending")
            return "pending"
        except Exception as error:  # network, catalogue or disk problems: record and move on
            shutil.rmtree(out_root / key, ignore_errors=True)
            db.release_block(key)
            message = f"{type(error).__name__}: {error}"
            if attempt < cfg.network_retries and is_transient(error):
                db.log(key, "warn", f"transient error, retry {attempt + 1}/{cfg.network_retries} "
                       f"in {wait:.0f}s: {message[:200]}")
                db.stage(key, "retrying", f"network error, retry {attempt + 1} in {wait:.0f}s")
                clear_gdal_http_cache()
                if stop is not None and stop.wait(wait):
                    db.finish(key, "pending")
                    return "pending"
                if stop is None:
                    time.sleep(wait)
                wait *= 2
                continue
            db.log(key, "error", message)
            db.finish(key, "failed", error=message[:500])
            return "failed"
    return "failed"  # unreachable: the last attempt always returns


def _process_once(aoi: dict, out_root: Path, cfg: Config, db: StateDB, session,
                  check_stop, select_only: bool) -> str:
    key = aoi["key"]
    pairs, diag = select_pairs(aoi, cfg, session, stage=lambda s, n: db.stage(key, s, n),
                               check_stop=check_stop)
    if not pairs:
        db.log(key, "info", f"no usable pair: {diag}")
        db.finish(key, "no_pairs", diag=diag)
        return "no_pairs"
    block = pairs[0]["block"]

    chosen: list[dict] = []
    for candidate in pairs:
        if len(chosen) < cfg.n_pairs and is_diverse(candidate, chosen, cfg.min_separation_days):
            chosen.append(candidate)
    if select_only:
        db.finish(key, "selected", n_pairs=len(chosen), tile=block["id"], diag=diag)
        return "selected"

    owner = db.claim_block(block["id"], key)
    if owner != key:
        db.finish(key, "no_pairs", tile=block["id"],
                  diag={**diag, "note": f"block {block['id']} already used by site {owner}"})
        return "no_pairs"

    target = out_root / key
    shutil.rmtree(target, ignore_errors=True)
    db.reset_for_retry(key)
    accepted: list[dict] = []
    avoid: set[str] = set()        # scenes an image check found hazy / white: skip their other pairs
    rejections: list[str] = []
    for candidate in pairs:
        if len(accepted) >= cfg.n_pairs or len(rejections) >= cfg.max_downloads:
            break
        if candidate["before"]["id"] in avoid or candidate["after"]["id"] in avoid:
            continue
        if not is_diverse(candidate, accepted, cfg.min_separation_days):
            continue
        check_stop()
        index = len(accepted) + 1
        db.stage(key, "downloading", f"pair {index}/{cfg.n_pairs} {block['id']} "
                 f"({item_date(candidate['before']).isoformat()} -> {item_date(candidate['after']).isoformat()})")
        try:
            record = save_pair(candidate, aoi, target / f"pair_{index}", cfg)
        except (ValueError, OSError, rasterio.errors.RasterioError) as error:
            if is_transient(error):
                raise  # a network blip is not a bad candidate: retry the whole site
            db.log(key, "warn", f"candidate rejected ({candidate['before']['id']} / "
                   f"{candidate['after']['id']}): {error}")
            rejections.append(str(error))
            if isinstance(error, QualityRejected):
                if error.scene is None:
                    break          # both dates white: permanent snow / glacier, no pair will do
                avoid.add(error.scene)
            continue
        db.add_pair(key, index, record)
        accepted.append(candidate)

    if not accepted:
        db.release_block(key)
        db.finish(key, "no_pairs", diag={**diag, "note": "every downloaded candidate failed its checks",
                                         "rejections": rejections})
        return "no_pairs"
    db.finish(key, "done", n_pairs=len(accepted), tile=block["id"], diag=diag)
    return "done"
