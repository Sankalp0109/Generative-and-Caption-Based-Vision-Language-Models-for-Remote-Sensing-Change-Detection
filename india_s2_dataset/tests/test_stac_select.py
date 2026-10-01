from datetime import date

import pytest
import rasterio

from stac_select import (BLOCK_M, GDAL_OPTIONS, block_for, candidate_blocks, footprint_covers,
                         is_diverse, item_tile, parse_months, rank_pairs, search, season_range)

ORIGIN = (300000.0, 2100000.0)            # UL corner of a fake UTM 43N tile
LON, LAT = 73.2, 18.95                    # a point inside it (checked below)
COVER = {"type": "Polygon", "coordinates": [[[72.9, 18.8], [74.2, 18.8], [74.2, 19.1], [72.9, 19.1], [72.9, 18.8]]]}
PARTIAL = {"type": "Polygon", "coordinates": [[[72.9, 18.8], [73.1, 18.8], [73.1, 19.1], [72.9, 19.1], [72.9, 18.8]]]}


def item(item_id, when, cloud=1.0, tile="MGRS-43QBB", geometry=COVER, origin=ORIGIN):
    return {"id": item_id, "geometry": geometry,
            "properties": {"datetime": f"{when}T05:00:00Z", "eo:cloud_cover": cloud, "grid:code": tile,
                           "proj:epsg": 32643},
            "assets": {"visual": {"href": "v", "proj:transform": [10, 0, origin[0], 0, -10, origin[1]]},
                       "scl": {"href": "s"}}}


def clear(_item):
    return {"bad": 0.0, "nodata": 0.0, "snow": 0.0}


def block(tile="MGRS-43QBB"):
    return block_for(item("x", "2019-12-01", tile=tile), LON, LAT)


# ---- seasons -----------------------------------------------------------------------------
def test_seasons_fall_inside_the_july_june_cycle():
    assert season_range(2019, "11,12,1,2,3") == (date(2019, 11, 1), date(2020, 4, 1))
    assert season_range(2019, "1,2,3,4") == (date(2020, 1, 1), date(2020, 5, 1))   # not sparse early 2019
    assert season_range(2019, "10,11,12") == (date(2019, 10, 1), date(2020, 1, 1))
    assert season_range(2025, "7,8,9,10") == (date(2025, 7, 1), date(2025, 11, 1))


def test_months_must_be_valid_and_consecutive():
    assert parse_months("11,12,1,2,3") == [11, 12, 1, 2, 3]
    with pytest.raises(ValueError):
        parse_months("0,13")
    with pytest.raises(ValueError, match="consecutive"):
        parse_months("1,2,3,11,12")  # would otherwise become Jan..Dec


# ---- geometry ----------------------------------------------------------------------------
def test_block_is_the_1024px_tile_block_holding_the_point():
    b = block()
    left, bottom, right, top = b["bounds"]
    assert right - left == top - bottom == BLOCK_M
    assert (left - ORIGIN[0]) % BLOCK_M == 0 and (ORIGIN[1] - top) % BLOCK_M == 0   # on the block grid
    (x,), (y,) = rasterio.warp.transform("EPSG:4326", "EPSG:32643", [LON], [LAT])
    assert left <= x < right and bottom < y <= top
    assert b["id"] == f"MGRS-43QBB:c{b['col']}r{b['row']}"


def test_tile_holding_the_point_most_centrally_is_tried_first():
    edge = item("b2", "2019-12-01", tile="MGRS-EDGE", origin=(ORIGIN[0] + 60000, ORIGIN[1]))
    blocks = candidate_blocks([item("b1", "2019-12-01"), edge],
                              [item("a1", "2025-12-01"), item("a2", "2025-12-01", tile="MGRS-EDGE")], LON, LAT)
    assert [b["tile"] for b in blocks] == ["MGRS-43QBB", "MGRS-EDGE"]
    assert blocks[1]["margin_m"] < 0      # point is west of that tile's origin: clamped block


def test_footprint_must_cover_the_whole_block_not_just_overlap_it():
    points = block()["points"]
    assert footprint_covers(COVER, points)
    assert not footprint_covers(PARTIAL, points)
    assert not footprint_covers(None, points)


def test_tile_is_grid_code_not_the_per_scene_id():
    props = {"grid:code": "MGRS-43QBB", "s2:tile_id": "S2A_OPER_MSI_L2A_TL_unique"}
    assert item_tile({"id": "x", "properties": props}) == "MGRS-43QBB"
    assert item_tile({"id": "x", "properties": {"mgrs:utm_zone": 43, "mgrs:latitude_band": "Q",
                                                "mgrs:grid_square": "BB"}}) == "MGRS-43QBB"


# ---- pairing -----------------------------------------------------------------------------
def rank(before, after, **kw):
    kw.setdefault("stats", clear)
    return rank_pairs(before, after, block(), date(2019, 11, 1), date(2025, 11, 1), **kw)


def test_pairs_need_the_blocks_tile_and_a_similar_point_in_season():
    before = [item("b1", "2019-12-10"), item("b2", "2019-12-15", tile="MGRS-43QBA")]
    after = [item("a1", "2025-12-12"), item("a2", "2026-03-20")]  # a2 is ~4 months later in season
    pairs, diag = rank(before, after)
    assert [(p["before"]["id"], p["after"]["id"]) for p in pairs] == [("b1", "a1")]
    assert diag["before_covering"] == 1 and diag["block"] == block()["id"]


def test_cloudy_over_the_block_is_rejected_even_if_tile_cloud_is_low():
    stats = lambda i: {"bad": 0.4 if i["id"] == "b1" else 0.0, "nodata": 0.0, "snow": 0.0}
    pairs, diag = rank([item("b1", "2019-12-10", cloud=2), item("b2", "2019-12-12", cloud=30)],
                       [item("a1", "2025-12-11")], stats=stats)
    assert [p["before"]["id"] for p in pairs] == ["b2"] and diag["before_clear"] == 1


def test_snow_difference_is_not_accepted_as_change():
    stats = lambda i: {"bad": 0.0, "nodata": 0.0, "snow": 0.15 if i["id"] == "a1" else 0.0}
    pairs, _ = rank([item("b1", "2019-12-10")], [item("a1", "2025-12-11"), item("a2", "2025-12-12")],
                    stats=stats)
    assert [p["after"]["id"] for p in pairs] == ["a2"]


def test_scenes_that_do_not_contain_the_block_are_counted_and_dropped():
    pairs, diag = rank([item("b1", "2019-12-10", geometry=PARTIAL)], [item("a1", "2025-12-12")])
    assert pairs == [] and diag["before_found"] == 1 and diag["before_covering"] == 0


def test_only_top_k_scenes_get_the_network_cloud_check():
    measured = []
    stats = lambda i: measured.append(i["id"]) or {"bad": 0.0, "nodata": 0.0, "snow": 0.0}
    rank([item(f"b{n}", f"2019-12-{10 + n}", cloud=n) for n in range(4)], [item("a1", "2025-12-11")],
         stats=stats, top_k=2)
    assert sorted(measured) == ["a1", "b0", "b1"]


def test_second_pair_must_not_be_a_near_repeat_of_the_first():
    pairs, _ = rank([item("b1", "2019-12-10"), item("b2", "2019-12-15"), item("b3", "2020-01-20")],
                    [item("a1", "2025-12-10"), item("a2", "2025-12-15"), item("a3", "2026-01-20")])
    first = next(p for p in pairs if p["before"]["id"] == "b1" and p["after"]["id"] == "a1")
    usable = {(p["before"]["id"], p["after"]["id"]) for p in pairs if is_diverse(p, [first], 15)}
    assert ("b2", "a2") not in usable      # 5 days after the first pair: a near-duplicate
    assert ("b1", "a3") not in usable      # reuses the first pair's before scene
    assert ("b3", "a3") in usable          # different scenes, 41 days apart


# ---- catalogue ---------------------------------------------------------------------------
def test_search_follows_next_pages_until_everything_matched_is_collected():
    class Resp:
        def __init__(self, payload): self.payload = payload
        def raise_for_status(self): pass
        def json(self): return self.payload

    class Session:
        def __init__(self): self.bodies = []
        def post(self, url, json, timeout):
            self.bodies.append(json)
            if "next" not in json:
                return Resp({"features": [{"id": "a"}, {"id": "b"}], "numberMatched": 3,
                             "links": [{"rel": "next", "body": {"next": "tok"}, "merge": True}]})
            return Resp({"features": [{"id": "c"}], "numberMatched": 3,
                         "links": [{"rel": "next", "body": {"next": "tok2"}, "merge": True}]})

    session = Session()
    found = search(session, "https://x", "c", (73, 18, 74, 19), date(2019, 11, 1), date(2020, 4, 1), 40, limit=2)
    assert [f["id"] for f in found] == ["a", "b", "c"]
    assert len(session.bodies) == 2 and session.bodies[1]["next"] == "tok"
    assert session.bodies[1]["collections"] == ["c"]  # merged, not replaced


def test_gdal_options_are_accepted_by_rasterio():
    with rasterio.Env(**GDAL_OPTIONS):
        pass


def test_top_row_block_sharing_the_tile_edge_with_the_footprint_counts_as_covered():
    # Point in the tile's first block row: the block's top edge IS the tile's (and footprint's) top edge.
    (x0,), (y0,) = rasterio.warp.transform("EPSG:32643", "EPSG:4326", [ORIGIN[0] + 15000], [ORIGIN[1] - 3000])
    b = block_for(item("x", "2019-12-01"), x0, y0)
    assert b["row"] == 0
    xs, ys = [ORIGIN[0], ORIGIN[0] + 109800], [ORIGIN[1] - 109800, ORIGIN[1]]
    lons, lats = rasterio.warp.transform("EPSG:32643", "EPSG:4326",
                                         [xs[0], xs[1], xs[1], xs[0], xs[0]], [ys[1], ys[1], ys[0], ys[0], ys[1]])
    tile_footprint = {"type": "Polygon", "coordinates": [list(map(list, zip(lons, lats)))]}
    assert footprint_covers(tile_footprint, b["points"])


def test_rejections_are_explained_in_the_diagnostics():
    # 15% snow: allowed on its own (< 20%), but 15 points more than the before date
    stats = lambda i: {"bad": 0.0, "nodata": 0.0, "snow": 0.15 if i["id"] == "a1" else 0.0}
    _, diag = rank([item("b1", "2019-12-10")], [item("a1", "2025-12-11"), item("a2", "2026-03-01")],
                   stats=stats)
    assert diag["rejected_snow"] == 1 and diag["rejected_gap"] == 1 and diag["pairs"] == 0


def test_snow_covered_and_mostly_water_scenes_are_not_used():
    stats = {"b1": {"snow": 0.5}, "b2": {"water": 0.9}, "b3": {}}
    measured = lambda i: {"bad": 0.0, "nodata": 0.0, "snow": 0.0, "water": 0.0, **stats.get(i["id"], {})}
    pairs, diag = rank([item("b1", "2019-12-10"), item("b2", "2019-12-11"), item("b3", "2019-12-12")],
                       [item("a1", "2025-12-12")], stats=measured)
    assert [p["before"]["id"] for p in pairs] == ["b3"]
    assert diag["snow_covered"] == 1 and diag["mostly_water"] == 1
