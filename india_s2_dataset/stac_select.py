"""Scene search and selection for the Sentinel-2 change dataset.

A site is one 1024 x 1024 px block of a Sentinel-2 tile (10.24 km at 10 m). True-colour COGs
store exactly these blocks, and a read always fetches whole blocks, so a site aligned to one
block costs one ~1.8 MB block per image instead of the up-to-four a free-floating AOI hits.

For a site's centre point this module finds the tiles that contain it, the block around it,
the scenes whose data footprint covers that block, the cloud over the block itself (from the
SCL mask, not the whole 110 km tile), and before/after pairs from the same tile at a similar
point in the season so crop stage and sun angle are comparable.
"""

from __future__ import annotations

import math
import os
from datetime import date, datetime
from typing import Callable, Iterable, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Public buckets: never sign requests. Retry flaky range reads instead of failing a pair.
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
GDAL_OPTIONS = {
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": "300",
    "GDAL_HTTP_CONNECTTIMEOUT": "60",
    "CPL_VSIL_CURL_CACHE_SIZE": "134217728",
    "GDAL_CACHEMAX": 512,
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
}

import numpy as np
import rasterio
from rasterio.warp import transform as transform_points
from rasterio.windows import from_bounds

DEFAULT_CATALOG = "https://earth-search.aws.element84.com/v1"
# Collection 1 is the reprocessed archive: one processing baseline across all years, so
# the 2022 radiometric offset in the older collection cannot masquerade as change. Its
# coverage is dense only from about May 2019, which is why seasons run July-June.
DEFAULT_COLLECTION = "sentinel-2-c1-l2a"
REQUIRED_ASSETS = ("visual", "scl")

BLOCK_PX = 1024            # true-colour COG block size
PIXEL_M = 10.0             # true-colour resolution
BLOCK_M = BLOCK_PX * PIXEL_M
FULL_BLOCKS = 10           # a 10980 px tile holds 10 full blocks per axis (+ a 740 px sliver)
# Coverage is tested this far inside the block edge. Blocks on a tile's top row / left column
# share their edge with the scene footprint, and footprints are lon/lat polygons with few
# vertices whose straight chords sag inside the true (UTM-straight) tile edge: up to 178 m over
# India's zones. Testing the exact edge made every such block look uncovered (~20% of blocks).
# Real no-data inside this margin is still caught by the no-data check after the read.
COVER_INSET_M = 250.0

# Scene Classification Layer classes. 0 no data, 1 saturated/defective, 3 cloud shadow,
# 8/9 cloud (medium/high probability), 10 thin cirrus. Snow (11) is tracked separately
# because it is legitimate ground cover in the Himalaya rather than an obstruction.
SCL_BAD = (0, 1, 3, 8, 9, 10)
SCL_SNOW = 11
SCL_WATER = 6


def make_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, allowed_methods=None,
                  status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
    return session


# ---------------------------------------------------------------- seasons ---------------
def parse_months(months: str | Sequence[int]) -> list[int]:
    """'11,12,1,2,3' -> [11, 12, 1, 2, 3]; order is kept because it defines season start."""
    values = ([int(v) for v in months.split(",") if v.strip()]
              if isinstance(months, str) else [int(v) for v in months])
    if not values or not set(values) <= set(range(1, 13)):
        raise ValueError(f"months must be 1-12, got {months!r}")
    if any((b - a) % 12 != 1 for a, b in zip(values, values[1:])):
        # season_range() spans first..last month, so a gap would silently widen the window
        raise ValueError(f"season months must be consecutive (e.g. 11,12,1,2,3), got {months!r}")
    return values


def season_range(cycle_year: int, months: str | Sequence[int]) -> tuple[date, date]:
    """First day and exclusive end of a season inside the July-June cycle starting ``cycle_year``.

    Every season of cycle 2019 lies in Jul 2019 .. Jun 2020, whichever months it uses: Nov-Mar
    is Nov 2019 .. Mar 2020, Jan-Apr is Jan-Apr 2020, Oct-Dec is Oct-Dec 2019. So "2019 vs 2025"
    means the same gap for every region, and never reaches into sparse early-2019 coverage.
    """
    months = parse_months(months)
    first, last = months[0], months[-1]
    start_year = cycle_year if first >= 7 else cycle_year + 1
    start = date(start_year, first, 1)
    end_year = start_year if last >= first else start_year + 1
    end = date(end_year + (last == 12), 1 if last == 12 else last + 1, 1)
    return start, end


# ---------------------------------------------------------------- catalogue -------------
def search(session, catalog: str, collection: str, bbox: Sequence[float], start: date, end: date,
           max_cloud: float, limit: int = 200, max_pages: int = 20) -> list[dict]:
    """All matching items, following ``next`` links so a busy window is never silently cut off."""
    body = {
        "collections": [collection], "bbox": list(bbox), "limit": limit,
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T00:00:00Z",
        "query": {"eo:cloud_cover": {"lte": max_cloud}},
    }
    url, features = f"{catalog.rstrip('/')}/search", []
    for _ in range(max_pages):
        response = session.post(url, json=body, timeout=60)
        response.raise_for_status()
        page = response.json()
        batch = page.get("features", [])
        features.extend(batch)
        matched = page.get("numberMatched", (page.get("context") or {}).get("matched"))
        link = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if not batch or link is None or (matched is not None and len(features) >= matched):
            break
        url = link.get("href", url)
        body = {**body, **link["body"]} if link.get("merge") else link.get("body", body)
    return features


def item_tile(item: dict) -> str:
    """MGRS tile, e.g. ``MGRS-43QBB``. (``s2:tile_id`` is a per-scene id, not the tile.)"""
    props = item.get("properties", {})
    if props.get("grid:code"):
        return str(props["grid:code"])
    parts = (props.get("mgrs:utm_zone"), props.get("mgrs:latitude_band"), props.get("mgrs:grid_square"))
    return "MGRS-" + "".join(str(p) for p in parts) if all(parts) else item.get("id", "")


def item_date(item: dict) -> date:
    return datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")).date()


def item_epsg(item: dict) -> int:
    props = item.get("properties", {})
    if props.get("proj:epsg"):
        return int(props["proj:epsg"])
    code = props.get("proj:code") or item["assets"]["visual"].get("proj:code", "")
    return int(str(code).split(":")[-1])


# ---------------------------------------------------------------- geometry --------------
def _in_ring(ring, x: float, y: float) -> bool:
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def footprint_covers(geometry: dict | None, points: Sequence[tuple[float, float]]) -> bool:
    """True when one polygon of the item's data footprint contains every lon/lat point.

    Scene *bounds* overlap a site far more often than the scene has data there (orbit swath
    edges are diagonal no-data wedges), which is what made whole months fail late.
    """
    if not geometry:
        return False
    kind, coords = geometry.get("type"), geometry.get("coordinates")
    polygons = [coords] if kind == "Polygon" else coords if kind == "MultiPolygon" else []
    for polygon in polygons:
        ring = [tuple(p[:2]) for p in polygon[0]]
        if all(_in_ring(ring, x, y) for x, y in points):
            return True
    return False


def block_for(item: dict, lon: float, lat: float) -> dict:
    """The full 1024 px block of ``item``'s tile nearest to (lon, lat), plus how central it is.

    ``margin_m`` is how far the point sits inside the tile's full-block area (negative when the
    point is on the 740 px sliver or outside the tile). Where tiles overlap, the tile holding the
    point most comfortably is tried first.
    """
    transform = item["assets"]["visual"].get("proj:transform") or item["properties"].get("proj:transform")
    if not transform:
        raise ValueError(f"{item['id']}: no proj:transform, cannot place the block")
    origin_x, origin_y = float(transform[2]), float(transform[5])
    epsg = item_epsg(item)
    (x,), (y,) = transform_points("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
    col = min(max(int(math.floor((x - origin_x) / BLOCK_M)), 0), FULL_BLOCKS - 1)
    row = min(max(int(math.floor((origin_y - y) / BLOCK_M)), 0), FULL_BLOCKS - 1)
    span = FULL_BLOCKS * BLOCK_M
    margin = min(x - origin_x, origin_x + span - x, origin_y - y, y - (origin_y - span))
    left, top = origin_x + col * BLOCK_M, origin_y - row * BLOCK_M
    bounds = (left, top - BLOCK_M, left + BLOCK_M, top)
    xs = (bounds[0] + COVER_INSET_M, (bounds[0] + bounds[2]) / 2, bounds[2] - COVER_INSET_M)
    ys = (bounds[1] + COVER_INSET_M, (bounds[1] + bounds[3]) / 2, bounds[3] - COVER_INSET_M)
    lons, lats = transform_points(f"EPSG:{epsg}", "EPSG:4326",
                                  [px for px in xs for _ in ys], [py for _ in xs for py in ys])
    tile = item_tile(item)
    return {"tile": tile, "epsg": epsg, "col": col, "row": row, "bounds": bounds,
            "id": f"{tile}:c{col}r{row}", "margin_m": margin, "points": list(zip(lons, lats)),
            "bbox_wgs84": (min(lons), min(lats), max(lons), max(lats))}


def candidate_blocks(before_items: Sequence[dict], after_items: Sequence[dict],
                     lon: float, lat: float) -> list[dict]:
    """One block per tile found in both years, the tile holding the point most centrally first."""
    after_tiles = {item_tile(i) for i in after_items}
    by_tile: dict[str, dict] = {}
    for item in before_items:
        tile = item_tile(item)
        if tile in after_tiles and tile not in by_tile and "visual" in item.get("assets", {}):
            by_tile[tile] = block_for(item, lon, lat)
    return sorted(by_tile.values(), key=lambda b: (-b["margin_m"], b["id"]))


# ---------------------------------------------------------------- cloud -----------------
def scl_stats(href: str, block: dict) -> dict:
    """Cloud/shadow/nodata/snow fractions over the block only (one 512 px SCL block read)."""
    with rasterio.Env(**GDAL_OPTIONS), rasterio.open(href) as src:
        window = from_bounds(*block["bounds"], transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=window, boundless=True, fill_value=0)
    total = max(data.size, 1)
    return {"bad": float(np.isin(data, SCL_BAD).sum()) / total,
            "nodata": float((data == 0).sum()) / total,
            "snow": float((data == SCL_SNOW).sum()) / total,
            "water": float((data == SCL_WATER).sum()) / total}


# ---------------------------------------------------------------- pairing ---------------
def eligible(items: Iterable[dict], block: dict) -> list[dict]:
    """Scenes of the block's tile with the needed assets and data over the whole block."""
    keep = [it for it in items
            if item_tile(it) == block["tile"]
            and all(a in it.get("assets", {}) for a in REQUIRED_ASSETS)
            and footprint_covers(it.get("geometry"), block["points"])]
    return sorted(keep, key=lambda it: (float(it["properties"].get("eo:cloud_cover", 100)),
                                        it["properties"]["datetime"], it["id"]))


def rank_pairs(before_items: Sequence[dict], after_items: Sequence[dict], block: dict,
               before_start: date, after_start: date, *, stats: Callable[[dict], dict] | None = None,
               max_aoi_cloud: float = 0.01, top_k: int = 5, max_gap_days: int = 30,
               max_nodata: float = 0.001, max_snow_diff: float = 0.05, max_snow: float = 0.20,
               max_water: float = 0.70) -> tuple[list[dict], dict]:
    """Every acceptable before/after pairing for one block, best first, plus shortfall counts.

    Only the ``top_k`` lowest tile-cloud scenes per side get the (network) block-cloud check.
    Pairs must sit within ``max_gap_days`` of the same point in their seasons and differ by at
    most ``max_snow_diff`` in snow cover, so fresh snowfall is not mistaken for change. A scene
    more than ``max_snow`` snow (no ground detail under it) or ``max_water`` water (open sea or a
    reservoir: little land to describe) is not used at all.
    """
    stats = stats or (lambda item: scl_stats(item["assets"]["scl"]["href"], block))
    cache: dict[str, dict] = {}

    def measured(item: dict) -> dict:
        if item["id"] not in cache:
            cache[item["id"]] = stats(item)
        return cache[item["id"]]

    before_ok, after_ok = eligible(before_items, block), eligible(after_items, block)
    diag = {"block": block["id"], "before_found": len(before_items), "after_found": len(after_items),
            "before_covering": len(before_ok), "after_covering": len(after_ok),
            "before_clear": 0, "after_clear": 0, "snow_covered": 0, "mostly_water": 0,
            "rejected_gap": 0, "rejected_snow": 0, "pairs": 0}

    def clear(items: list[dict]) -> list[dict]:
        out = []
        for item in items[:top_k]:
            s = measured(item)
            if s["bad"] > max_aoi_cloud or s["nodata"] > max_nodata:
                continue
            if s.get("snow", 0.0) > max_snow:
                diag["snow_covered"] += 1
            elif s.get("water", 0.0) > max_water:
                diag["mostly_water"] += 1
            else:
                out.append(item)
        return out

    before_clear, after_clear = clear(before_ok), clear(after_ok)
    diag["before_clear"], diag["after_clear"] = len(before_clear), len(after_clear)

    pairs = []
    for b in before_clear:
        for a in after_clear:
            gap = abs((item_date(b) - before_start).days - (item_date(a) - after_start).days)
            if gap > max_gap_days:
                diag["rejected_gap"] += 1
                continue
            if abs(measured(b)["snow"] - measured(a)["snow"]) > max_snow_diff:
                diag["rejected_snow"] += 1
                continue
            pairs.append({"before": b, "after": a, "gap_days": gap, "tile": block["tile"],
                          "block": block, "before_stats": measured(b), "after_stats": measured(a),
                          "score": measured(b)["bad"] + measured(a)["bad"] + gap / 10000})
    pairs.sort(key=lambda p: (p["score"], p["before"]["id"], p["after"]["id"]))
    diag["pairs"] = len(pairs)
    return pairs, diag


def is_diverse(pair: dict, accepted: Sequence[dict], min_separation_days: int = 15) -> bool:
    """A candidate is usable if it reuses no accepted scene and is not a near-repeat.

    Consecutive Sentinel-2 revisits are ~5 days apart; two pairs from adjacent passes are
    near-duplicates and would just teach the model the same scene twice.
    """
    for other in accepted:
        if pair["before"]["id"] == other["before"]["id"] or pair["after"]["id"] == other["after"]["id"]:
            return False
        if abs((item_date(pair["before"]) - item_date(other["before"])).days) < min_separation_days:
            return False
    return True
