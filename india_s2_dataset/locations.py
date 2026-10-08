"""AOI registry: named areas of interest loaded from ``locations/india_aois.csv``.

Every script that downloads a location imports this registry rather than hardcoding a
bbox, so a typo'd coordinate cannot mislabel where a set of images came from. A row holds
a centre point and a size; the bbox is derived from those so hundreds of sites can be
generated without typing coordinates.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent / "locations" / "india_aois.csv"
DEFAULT_SIZE_KM = 10.24  # one 1024 px Sentinel-2 block at 10 m = a 4x4 grid of 256 px patches
DEFAULT_SEASON = "11,12,1,2,3"


def bbox_from_center(lon: float, lat: float, size_km: float = DEFAULT_SIZE_KM):
    """WGS84 (min_lon, min_lat, max_lon, max_lat) of a square ``size_km`` wide around a point."""
    half = size_km / 2
    dlat = half / 110.574
    dlon = half / (111.320 * math.cos(math.radians(lat)))
    return (round(lon - dlon, 6), round(lat - dlat, 6), round(lon + dlon, 6), round(lat + dlat, 6))


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict]:
    """Read the registry CSV into ``{key: row}``; each row gains a derived ``bbox``."""
    if not Path(path).exists():
        return {}
    rows: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row["key"].strip()
            if key in rows:
                raise ValueError(f"duplicate location key {key!r} in {path}")
            row["lon"], row["lat"] = float(row["lon"]), float(row["lat"])
            row["size_km"] = float(row.get("size_km") or DEFAULT_SIZE_KM)
            row["season_months"] = row.get("season_months") or DEFAULT_SEASON
            row["bbox"] = bbox_from_center(row["lon"], row["lat"], row["size_km"])
            rows[key] = row
    return rows


LOCATIONS: dict[str, dict] = load_registry()


def resolve_bbox(location: str, bbox: tuple[float, float, float, float] | None):
    """Return an explicit bbox if given, else the registered bbox for ``location``.

    Raises if ``location`` is not registered and no explicit bbox was given, so an
    unrecognized location name can never silently fall back to the wrong place.
    """
    if bbox is not None:
        return tuple(bbox)
    if location not in LOCATIONS:
        raise ValueError(f"unknown location {location!r}; pass an explicit bbox or add it to {REGISTRY_PATH}")
    return LOCATIONS[location]["bbox"]
