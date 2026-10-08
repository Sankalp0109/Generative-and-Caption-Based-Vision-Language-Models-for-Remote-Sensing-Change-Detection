#!/usr/bin/env python3
"""Generate ``locations/india_aois.csv``: download sites spread evenly over every state and UT.

    python build_aoi_registry.py --sites 1200

How sites are placed:

* **Every state and union territory gets sites.** Each gets ``--min-per-state`` (12) unless it
  is too small to hold that many separate 10 km blocks (Chandigarh, Lakshadweep, ...). The
  remaining sites are shared in proportion to area, so coverage density is even across India.
* **Even spread inside each state.** Candidates on a fine grid inside the state; sites are then
  picked by farthest-point sampling, each new site as far as possible from every site chosen so
  far, including those already placed in neighbouring states, so borders stay spread too.
* **Rows are interleaved state by state** (every state's 1st site, then every state's 2nd, ...),
  so any ``--limit N`` or pilot run of the downloader is already spread across all of India.
* **Season and cloud limit per region**, e.g. autumn in the Himalaya (winter snow would look
  like change), Jan-Apr on the Tamil Nadu coast (NE monsoon in Oct-Dec), a relaxed cloud limit
  in the always-cloudy Northeast and islands.

State boundaries: geoBoundaries gbOpen IND ADM1 (CC BY 2.5 IN), cached in ``locations/layers``.
Also writes ``locations/coverage_map.png`` to eyeball the spread.
"""

from __future__ import annotations

import argparse
import csv
import math
import unicodedata
from pathlib import Path

import numpy as np
import requests

BOUNDARY_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM1/"
                "geoBoundaries-IND-ADM1_simplified.geojson")
BOUNDARY_PATH = Path("locations/layers/india_adm1.geojson")
REGISTRY_PATH = Path("locations/india_aois.csv")
MAP_PATH = Path("locations/coverage_map.png")
EQUAL_AREA = "EPSG:7755"          # WGS 84 / India NSF LCC: metres, low distortion over India
BLOCK_AREA_KM2 = 10.24 ** 2
DEFAULT_SEASON = "11,12,1,2,3"    # post-monsoon dry season: clearest skies for most of India

# (months, max block cloud fraction or None for the downloader default of 1%)
SEASONS = {
    "Ladakh": ("7,8,9,10", None),                       # cold desert, dry; snow-free late summer
    "Jammu and Kashmir": ("10,11,12", None),            # post-monsoon, before heavy snow
    "Himachal Pradesh": ("10,11,12", None),
    "Uttarakhand": ("10,11,12", None),
    "Sikkim": ("10,11,12", 0.03),
    "Arunachal Pradesh": ("10,11,12", 0.03),
    "Tamil Nadu": ("1,2,3,4", None),                    # NE monsoon rains Oct-Dec
    "Puducherry": ("1,2,3,4", None),
    "Kerala": ("12,1,2,3", 0.03),
    "Lakshadweep": ("1,2,3,4", 0.03),
    "Andaman and Nicobar Islands": ("1,2,3,4", 0.03),
    "Assam": (DEFAULT_SEASON, 0.03),                    # persistent cloud in the Northeast
    "Meghalaya": (DEFAULT_SEASON, 0.03),
    "Nagaland": (DEFAULT_SEASON, 0.03),
    "Manipur": (DEFAULT_SEASON, 0.03),
    "Mizoram": (DEFAULT_SEASON, 0.03),
    "Tripura": (DEFAULT_SEASON, 0.03),
}

CURATED = [  # well-known change sites kept after the even grid (not part of the interleaved order)
    ("polavaram_project_andhra_pradesh", "Polavaram project", "Andhra Pradesh", 81.6400, 17.2925),
    ("navi_mumbai_airport_maharashtra", "Navi Mumbai International Airport", "Maharashtra", 73.0500, 18.9700),
    ("statue_of_unity_gujarat", "Statue of Unity / Kevadia", "Gujarat", 73.7200, 21.8400),
]


def ascii_name(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in ascii_name(name).lower()).strip("_").replace("__", "_")


def load_states(path: Path = BOUNDARY_PATH):
    import geopandas as gpd
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(BOUNDARY_URL, timeout=600)
        response.raise_for_status()
        path.write_bytes(response.content)
    states = gpd.read_file(path).to_crs(EQUAL_AREA)
    states["name"] = states["shapeName"].map(ascii_name)
    states["area_km2"] = states.area / 1e6
    return states.sort_values("area_km2").reset_index(drop=True)


def allocate(areas: dict[str, float], total: int, minimum: int) -> dict[str, int]:
    """Sites per state: a floor for every state (capped by how many blocks fit), rest by area."""
    capacity = {s: max(1, int(a // (BLOCK_AREA_KM2 * 2))) for s, a in areas.items()}
    counts = {s: min(minimum, capacity[s]) for s in areas}
    remaining = total - sum(counts.values())
    if remaining < 0:
        raise ValueError(f"--sites {total} is below the per-state floor total {sum(counts.values())}")
    open_states = {s for s in areas if counts[s] < capacity[s]}
    while remaining > 0 and open_states:
        area_open = sum(areas[s] for s in open_states)
        shares = {s: remaining * areas[s] / area_open for s in open_states}
        # largest-remainder rounding, never above capacity
        grant = {s: min(int(shares[s]), capacity[s] - counts[s]) for s in open_states}
        leftover = remaining - sum(grant.values())
        for s in sorted(open_states, key=lambda s: shares[s] - int(shares[s]), reverse=True):
            if leftover <= 0:
                break
            if counts[s] + grant[s] < capacity[s]:
                grant[s] += 1
                leftover -= 1
        if not any(grant.values()):
            break
        for s, g in grant.items():
            counts[s] += g
            remaining -= g
        open_states = {s for s in open_states if counts[s] < capacity[s]}
    return counts


def candidates(geometry, n_target: int) -> np.ndarray:
    """Grid points inside ``geometry`` (metres), fine enough to place ``n_target`` sites evenly."""
    import shapely
    minx, miny, maxx, maxy = geometry.bounds
    spacing = max(250.0, math.sqrt(geometry.area / max(n_target * 150, 1)))
    xs = np.arange(minx + spacing / 2, maxx, spacing)
    ys = np.arange(miny + spacing / 2, maxy, spacing)
    gx, gy = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(geometry, gx.ravel(), gy.ravel())
    points = np.column_stack([gx.ravel()[inside], gy.ravel()[inside]])
    if len(points) == 0:  # a sliver smaller than the grid: use its representative point
        p = geometry.representative_point()
        points = np.array([[p.x, p.y]])
    return points


def farthest_points(points: np.ndarray, n: int, occupied: np.ndarray) -> np.ndarray:
    """Pick ``n`` of ``points`` greedily, each as far as possible from everything chosen so far."""
    if len(occupied):
        nearest = np.concatenate([  # chunked: points x occupied would be hundreds of MB at once
            np.min(np.linalg.norm(chunk[:, None, :] - occupied[None, :, :], axis=2), axis=1)
            for chunk in np.array_split(points, max(1, len(points) // 2000))])
    else:  # first state: start from the point nearest its middle
        centre = points.mean(axis=0)
        nearest = np.full(len(points), np.inf)
        nearest[np.argmin(np.linalg.norm(points - centre, axis=1))] = 1e18
    chosen = []
    for _ in range(min(n, len(points))):
        i = int(np.argmax(nearest))
        chosen.append(i)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[i], axis=1))
        nearest[i] = -1
    return points[chosen]


def build(total: int, minimum: int) -> list[dict]:
    from pyproj import Transformer
    states = load_states()
    counts = allocate(dict(zip(states["name"], states["area_km2"])), total, minimum)
    to_lonlat = Transformer.from_crs(EQUAL_AREA, "EPSG:4326", always_xy=True)
    occupied = np.empty((0, 2))
    per_state: dict[str, list[dict]] = {}
    for _, state in states.iterrows():  # smallest first: tiny UTs keep their spot, big states flow around
        n = counts[state["name"]]
        picks = farthest_points(candidates(state.geometry, n), n, occupied)
        occupied = np.vstack([occupied, picks])
        months, cloud = SEASONS.get(state["name"], (DEFAULT_SEASON, None))
        lons, lats = to_lonlat.transform(picks[:, 0], picks[:, 1])
        per_state[state["name"]] = [
            {"key": f"{slug(state['name'])}_{rank:03d}", "label": f"{state['name']} #{rank}",
             "state": state["name"], "district": "", "category": "even_grid",
             "lon": round(float(lon), 5), "lat": round(float(lat), 5), "size_km": 10.24,
             "season_months": months, "max_aoi_cloud": "" if cloud is None else cloud,
             "rank": rank}
            for rank, (lon, lat) in enumerate(zip(lons, lats), start=1)]

    rows = []
    for rank in range(1, max(len(v) for v in per_state.values()) + 1):
        for name in sorted(per_state):
            if rank <= len(per_state[name]):
                rows.append(per_state[name][rank - 1])
    for key, label, state, lon, lat in CURATED:
        months, cloud = SEASONS.get(state, (DEFAULT_SEASON, None))
        rows.append({"key": key, "label": label, "state": state, "district": "", "category": "curated",
                     "lon": lon, "lat": lat, "size_km": 10.24, "season_months": months,
                     "max_aoi_cloud": "" if cloud is None else cloud, "rank": ""})
    return rows


def write_registry(rows: list[dict], path: Path = REGISTRY_PATH) -> None:
    fields = ["key", "label", "state", "district", "category", "lon", "lat", "size_km",
              "season_months", "max_aoi_cloud", "rank"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def draw_map(rows: list[dict], highlight: int, path: Path = MAP_PATH) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd
    states = gpd.read_file(BOUNDARY_PATH)
    figure, axis = plt.subplots(figsize=(10, 11))
    states.boundary.plot(ax=axis, color="#888", linewidth=0.5)
    grid = [r for r in rows if r["category"] == "even_grid"]
    axis.scatter([r["lon"] for r in grid[highlight:]], [r["lat"] for r in grid[highlight:]],
                 s=6, c="#9bb7d4", label=f"later sites ({len(grid) - highlight})")
    axis.scatter([r["lon"] for r in grid[:highlight]], [r["lat"] for r in grid[:highlight]],
                 s=10, c="#d1495b", label=f"first {highlight} rows (pilot order)")
    axis.set_title(f"Download sites: {len(grid)} evenly spread over {len({r['state'] for r in grid})} states/UTs")
    axis.legend(loc="lower left")
    axis.set_aspect("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sites", type=int, default=1200,
                        help="total even-grid sites (default 1200: ~940 pairs x 16 patches = 15k at ~80%% success)")
    parser.add_argument("--min-per-state", type=int, default=12)
    parser.add_argument("--map-highlight", type=int, default=250, help="rows highlighted on the map")
    args = parser.parse_args()
    rows = build(args.sites, args.min_per_state)
    write_registry(rows)
    draw_map(rows, args.map_highlight)
    grid = [r for r in rows if r["category"] == "even_grid"]
    by_state: dict[str, int] = {}
    for r in grid:
        by_state[r["state"]] = by_state.get(r["state"], 0) + 1
    print(f"wrote {REGISTRY_PATH}: {len(grid)} even-grid sites in {len(by_state)} states/UTs "
          f"+ {len(rows) - len(grid)} curated; map -> {MAP_PATH}")
    for name, n in sorted(by_state.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<42} {n:>4}")


if __name__ == "__main__":
    main()
