"""Add evenly spread extra sites when accepted patches fall short of the target.

New sites are placed by the same farthest-point method as the original registry, with every
existing site already counted as occupied, so they fill the gaps instead of clustering. They
are appended to locations/india_aois.csv (category "topup") and downloaded like any other site.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from build_aoi_registry import DEFAULT_SEASON, EQUAL_AREA, SEASONS, candidates, farthest_points, load_states, slug

PLAINS = {"Punjab", "Haryana", "Delhi", "Chandigarh", "Uttar Pradesh", "Bihar", "West Bengal"}
REGISTRY = ROOT / "locations" / "india_aois.csv"


def add_sites(n: int, round_no: int) -> list[str]:
    from pyproj import Transformer
    rows = list(csv.DictReader(open(REGISTRY, newline="", encoding="utf-8")))
    fields = list(rows[0])
    to_m = Transformer.from_crs("EPSG:4326", EQUAL_AREA, always_xy=True)
    to_ll = Transformer.from_crs(EQUAL_AREA, "EPSG:4326", always_xy=True)
    ex, ey = to_m.transform([float(r["lon"]) for r in rows], [float(r["lat"]) for r in rows])
    occupied = np.column_stack([ex, ey])
    states = load_states()
    states = states[states["area_km2"] > 2000]            # tiny UTs are already at capacity
    share = states["area_km2"] / states["area_km2"].sum()
    new_keys = []
    for (_, state), frac in zip(states.iterrows(), share):
        k = int(round(n * frac))
        if k == 0:
            continue
        picks = farthest_points(candidates(state.geometry, k), k, occupied)
        occupied = np.vstack([occupied, picks])
        months, cloud = SEASONS.get(state["name"], (DEFAULT_SEASON, None))
        if state["name"] in PLAINS:
            months = "2,3,4"
        lons, lats = to_ll.transform(picks[:, 0], picks[:, 1])
        for i, (lon, lat) in enumerate(zip(lons, lats), 1):
            key = f"{slug(state['name'])}_t{round_no}_{i:03d}"
            rows.append({f: "" for f in fields} | {
                "key": key, "label": f"{state['name']} top-up {round_no}.{i}", "state": state["name"],
                "category": "topup", "lon": round(float(lon), 5), "lat": round(float(lat), 5), "size_km": 10.24,
                "season_months": months, "max_aoi_cloud": "" if cloud is None else cloud})
            new_keys.append(key)
    with open(REGISTRY, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return new_keys
