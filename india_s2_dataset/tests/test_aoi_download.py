import json
import threading
from datetime import date

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as transform_points

import aoi_download
import stac_select
from aoi_download import Config, process_aoi
from state_db import StateDB

ORIGIN = (300000.0, 2100000.0)            # fake tile in UTM 43N; block (0,0) spans the first 10.24 km
TILE_PX = 1100                            # just over one 1024 px block keeps the fixtures small
(CENTRE_LON,), (CENTRE_LAT,) = transform_points("EPSG:32643", "EPSG:4326", [ORIGIN[0] + 5120], [ORIGIN[1] - 5120])
FOOTPRINT = {"type": "Polygon", "coordinates": [[[72.5, 18.5], [73.5, 18.5], [73.5, 19.5], [72.5, 19.5], [72.5, 18.5]]]}


def write_tile(path, bands, value, pixel):
    size = int(TILE_PX * 10 / pixel)
    data = np.full((bands, size, size), value, dtype=np.uint8)
    data[:, 0, 0] = value + 1 if bands == 3 else value      # a marker pixel at the block's corner
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=bands, dtype="uint8",
                       crs="EPSG:32643", transform=from_origin(*ORIGIN, pixel, pixel)) as dst:
        dst.write(data)
    return str(path)


def make_item(tmp_path, item_id, when, tci=100, scl=4):
    return {"id": item_id, "geometry": FOOTPRINT,
            "properties": {"datetime": f"{when}T05:30:00Z", "eo:cloud_cover": 1.0,
                           "grid:code": "MGRS-43QBB", "proj:epsg": 32643, "view:sun_elevation": 45.0,
                           "s2:processing_baseline": "05.11"},
            "assets": {"visual": {"href": write_tile(tmp_path / f"{item_id}_tci.tif", 3, tci, 10),
                                  "proj:transform": [10, 0, ORIGIN[0], 0, -10, ORIGIN[1]]},
                       "scl": {"href": write_tile(tmp_path / f"{item_id}_scl.tif", 1, scl, 20)}}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    scenes = {"before": make_item(tmp_path, "b1", "2019-12-10", tci=100),
              "after": make_item(tmp_path, "a1", "2025-12-12", tci=110)}

    def fake_search(_session, _catalog, _collection, _bbox, start, end, _max_cloud):
        return [s for s in scenes.values()
                if start <= date.fromisoformat(s["properties"]["datetime"][:10]) < end]

    monkeypatch.setattr(aoi_download, "search", fake_search)
    db = StateDB(tmp_path / "state.sqlite")
    db.seed({"site": {"label": "Site", "state": "X", "category": "test", "lon": CENTRE_LON,
                      "lat": CENTRE_LAT, "size_km": 10.24, "bbox": (0, 0, 1, 1),
                      "season_months": "11,12,1,2,3"}})
    return {"db": db, "out": tmp_path / "out", "scenes": scenes, "tmp": tmp_path}


def run(env, key="site", **kw):
    aoi = env["db"].queue(("pending", "selected", "failed", "no_pairs"), keys=[key])[0]
    return process_aoi(aoi, env["out"], Config(retry_wait_s=0), env["db"], session=None, **kw)


def test_saves_the_exact_block_pixels_with_metadata_and_records_it(env):
    assert run(env) == "done"
    folder = env["out"] / "site" / "pair_1"
    assert sorted(p.name for p in folder.iterdir()) == ["after.tif", "before.tif", "pair.json",
                                                        "scl_after.tif", "scl_before.tif"]
    with rasterio.open(folder / "before.tif") as before, rasterio.open(folder / "after.tif") as after:
        assert before.dtypes == ("uint8",) * 3 and before.shape == after.shape == (1024, 1024)
        assert before.transform == after.transform == from_origin(*ORIGIN, 10, 10)
        pixels = before.read()
        assert pixels[0, 0, 0] == 101 and int(np.median(pixels)) == 100   # copied, not resampled
        assert int(np.median(after.read())) == 110
    with rasterio.open(folder / "scl_before.tif") as scl:
        assert scl.shape == (1024, 1024) and scl.transform == from_origin(*ORIGIN, 10, 10)
    meta = json.loads((folder / "pair.json").read_text())
    assert meta["block"]["id"] == "MGRS-43QBB:c0r0" and meta["block"]["bounds_utm"][0] == ORIGIN[0]
    assert meta["before"]["item_id"] == "b1" and meta["before"]["sun_elevation"] == 45.0
    assert meta["site"]["state"] == "X"
    row = env["db"]._conn().execute("SELECT * FROM pair").fetchone()
    assert row["before_date"] == "2019-12-10" and row["shape"] == "1024x1024" and row["bytes"] > 0
    assert env["db"].summary()["counts"]["done"] == 1


def test_no_output_when_the_block_is_cloudy_even_though_tile_cloud_is_low(env, tmp_path):
    env["scenes"]["before"] = make_item(tmp_path, "b1", "2019-12-10", scl=9)
    assert run(env) == "no_pairs"
    assert not (env["out"] / "site").exists()
    diag = json.loads(env["db"]._conn().execute("SELECT diag FROM aoi").fetchone()["diag"])
    tried = diag["tiles_tried"][0]
    assert tried["before_covering"] == 1 and tried["before_clear"] == 0


def test_nodata_pair_is_rejected_and_leaves_no_partial_files(env, tmp_path):
    env["scenes"]["after"] = make_item(tmp_path, "a1", "2025-12-12", tci=0)  # all-zero image
    assert run(env) == "no_pairs"
    assert not list(env["out"].rglob("*.tmp")) and not (env["out"] / "site" / "pair_1").exists()


def test_second_site_on_the_same_block_is_skipped_as_a_duplicate(env):
    env["db"].seed({"twin": {"label": "Twin", "state": "X", "lon": CENTRE_LON + 0.01, "lat": CENTRE_LAT,
                             "bbox": (0, 0, 1, 1), "season_months": "11,12,1,2,3"}})
    assert run(env) == "done"
    assert run(env, key="twin") == "no_pairs"
    note = json.loads(env["db"]._conn().execute("SELECT diag FROM aoi WHERE key='twin'").fetchone()["diag"])["note"]
    assert "already used by site site" in note


def test_stop_request_requeues_instead_of_failing(env):
    stop = threading.Event()
    stop.set()
    assert run(env, stop=stop) == "pending"
    assert [r["key"] for r in env["db"].queue()] == ["site"]


def test_select_only_downloads_nothing(env):
    assert run(env, select_only=True) == "selected"
    assert not env["out"].exists()
    assert env["db"].summary()["counts"]["selected"] == 1


def test_network_error_is_recorded_and_the_site_can_be_retried(env, monkeypatch):
    good = aoi_download.search

    def broken(*_a, **_k):
        raise ConnectionError("connection reset")

    monkeypatch.setattr(aoi_download, "search", broken)
    assert run(env) == "failed"
    row = env["db"]._conn().execute("SELECT status, error, attempts FROM aoi").fetchone()
    assert row["status"] == "failed" and "connection reset" in row["error"]
    monkeypatch.setattr(aoi_download, "search", good)
    assert run(env) == "done"
    assert env["db"]._conn().execute("SELECT attempts FROM aoi").fetchone()["attempts"] == 2


def test_per_site_cloud_threshold_of_zero_is_respected_not_replaced_by_default(env, monkeypatch):
    # 0.5% cloud passes the 1% default, so only an honoured per-site 0.0 can reject it
    monkeypatch.setattr(stac_select, "scl_stats", lambda href, block: {"bad": 0.005, "nodata": 0.0, "snow": 0.0})
    env["db"]._conn().execute("UPDATE aoi SET max_aoi_cloud=0.0")
    assert run(env) == "no_pairs"
    env["db"]._conn().execute("UPDATE aoi SET max_aoi_cloud=NULL")
    assert run(env) == "done"


def test_transient_network_error_is_retried_and_the_site_still_completes(env, monkeypatch):
    good, calls = aoi_download.search, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise rasterio.errors.RasterioIOError("CURL error: Could not resolve host: e84-earth-search")
        return good(*a, **k)

    monkeypatch.setattr(aoi_download, "search", flaky)
    assert run(env) == "done"
    events = [r["message"] for r in env["db"]._conn().execute("SELECT message FROM event")]
    assert any("transient error, retry 1/3" in m for m in events)


def test_real_errors_are_not_retried(env, monkeypatch):
    calls = []

    def broken(*_a, **_k):
        calls.append(1)
        raise KeyError("visual")

    monkeypatch.setattr(aoi_download, "search", broken)
    assert run(env) == "failed" and len(calls) == 1


def test_network_error_while_saving_retries_the_site_instead_of_rejecting_the_candidate(env, monkeypatch):
    real, calls = aoi_download.save_pair, []

    def flaky_save(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise rasterio.errors.RasterioIOError("CURL error: Could not resolve host: e84-earth-search")
        return real(*a, **k)

    cleared = []
    monkeypatch.setattr(aoi_download, "save_pair", flaky_save)
    monkeypatch.setattr(aoi_download, "clear_gdal_http_cache", lambda: cleared.append(1))
    assert run(env) == "done"
    assert cleared == [1]
    events = [r["message"] for r in env["db"]._conn().execute("SELECT message FROM event")]
    assert not any("candidate rejected" in m for m in events)


def test_gdal_negative_cache_message_counts_as_transient():
    err = rasterio.errors.RasterioIOError("'/vsicurl/https://x/SCL.tif' does not exist in the file system, "
                                          "and is not recognized as a supported dataset name.")
    assert aoi_download.is_transient(err)
    assert not aoi_download.is_transient(KeyError("visual"))


def test_clearing_the_gdal_cache_does_not_raise():
    aoi_download.clear_gdal_http_cache()


def test_hazy_date_is_rejected_and_its_other_pairs_skipped(env, tmp_path):
    # b1 is the better-timed before scene but hazy (dark floor 200 vs 100): its pair is downloaded,
    # rejected, and the next pair must not reuse b1.
    env["scenes"]["before"] = make_item(tmp_path, "b1", "2019-12-11", tci=200)
    env["scenes"]["before2"] = make_item(tmp_path, "b2", "2019-12-01", tci=100)
    env["scenes"]["after2"] = make_item(tmp_path, "a2", "2025-12-14", tci=110)
    assert run(env) == "done"
    meta = json.loads((env["out"] / "site" / "pair_1" / "pair.json").read_text())
    assert meta["before"]["item_id"] == "b2" and meta["quality"]["flags"] == []
    events = [r["message"] for r in env["db"]._conn().execute("SELECT message FROM event")]
    assert sum("haze in before" in m for m in events) == 1   # b1 tried once, then avoided


def test_both_dates_white_gives_up_after_one_download(env, tmp_path):
    env["scenes"]["before"] = make_item(tmp_path, "b1", "2019-12-10", tci=250)
    env["scenes"]["after"] = make_item(tmp_path, "a1", "2025-12-12", tci=250)
    env["scenes"]["after2"] = make_item(tmp_path, "a2", "2025-12-14", tci=250)
    assert run(env) == "no_pairs"
    diag = json.loads(env["db"]._conn().execute("SELECT diag FROM aoi").fetchone()["diag"])
    assert len(diag["rejections"]) == 1 and "mostly white" in diag["rejections"][0]


def test_quality_flags_on_raw_pixels():
    clear = np.full((8, 8, 3), 20, dtype=np.uint8)
    hazy = np.full((8, 8, 3), 90, dtype=np.uint8)
    snow = np.full((8, 8, 3), 255, dtype=np.uint8)
    assert aoi_download.quality_flags(clear, clear)["flags"] == []
    assert aoi_download.quality_flags(hazy, clear)["flags"] == ["haze"]
    assert "white" in aoi_download.quality_flags(snow, clear)["flags"]
