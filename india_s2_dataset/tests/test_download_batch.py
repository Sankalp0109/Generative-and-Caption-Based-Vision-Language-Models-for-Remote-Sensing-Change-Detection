import download_batch
from state_db import StateDB

HEADER = "key,label,state,district,category,lon,lat,size_km,season_months,max_aoi_cloud\n"


def write_registry(path, n):
    rows = "".join(f"site{i},Site {i},State{i % 2},D,cat,73.{i:02d},19.0,5.12,,\n" for i in range(n))
    path.write_text(HEADER + rows)
    return str(path)


def test_run_finishes_everything_then_has_nothing_left(tmp_path, monkeypatch, capsys):
    registry = write_registry(tmp_path / "r.csv", 6)
    seen = []

    def fake(row, out, cfg, db, session, stop=None, select_only=False):
        seen.append(row["key"])
        db.start(row["key"])
        db.finish(row["key"], "done", n_pairs=1)
        return "done"

    monkeypatch.setattr(download_batch, "process_aoi", fake)
    db_path = tmp_path / "s.sqlite"
    download_batch.main(["--db", str(db_path), "run", "--registry", registry, "--workers", "3",
                         "--out", str(tmp_path / "o")])
    assert sorted(seen) == [f"site{i}" for i in range(6)]
    assert StateDB(db_path).summary()["counts"]["done"] == 6
    seen.clear()
    download_batch.main(["--db", str(db_path), "run", "--registry", registry, "--out", str(tmp_path / "o")])
    assert seen == [] and "nothing to do" in capsys.readouterr().out


def test_rerun_resumes_from_unfinished_and_retries_only_when_asked(tmp_path, monkeypatch):
    registry = write_registry(tmp_path / "r.csv", 4)
    outcome = {"site1": "failed", "site2": "no_pairs"}
    seen = []

    def fake(row, out, cfg, db, session, stop=None, select_only=False):
        seen.append(row["key"])
        db.start(row["key"])
        status = outcome.get(row["key"], "done")
        db.finish(row["key"], status, n_pairs=int(status == "done"), error="boom" if status == "failed" else None)
        return status

    monkeypatch.setattr(download_batch, "process_aoi", fake)
    db_path = tmp_path / "s.sqlite"
    base = ["--db", str(db_path), "run", "--registry", registry, "--out", str(tmp_path / "o")]
    download_batch.main([*base, "--limit", "2"])           # only the first two
    assert seen == ["site0", "site1"]
    seen.clear()
    download_batch.main(base)                              # resumes: site2, site3 only
    assert seen == ["site2", "site3"]
    seen.clear()
    outcome.clear()
    download_batch.main([*base, "--retry", "failed,no_pairs"])
    assert sorted(seen) == ["site1", "site2"]
    assert StateDB(db_path).summary()["counts"]["done"] == 4


def test_interrupted_running_rows_are_picked_up_again(tmp_path, monkeypatch):
    registry = write_registry(tmp_path / "r.csv", 2)
    db_path = tmp_path / "s.sqlite"
    download_batch.main(["--db", str(db_path), "seed", "--registry", registry])
    db = StateDB(db_path)
    db.start("site0")                                      # a run died mid-AOI
    seen = []

    def fake(row, out, cfg, db, session, stop=None, select_only=False):
        seen.append(row["key"]); db.start(row["key"]); db.finish(row["key"], "done", n_pairs=1)
        return "done"

    monkeypatch.setattr(download_batch, "process_aoi", fake)
    download_batch.main(["--db", str(db_path), "run", "--no-seed", "--out", str(tmp_path / "o")])
    assert sorted(seen) == ["site0", "site1"]


def test_status_report_mentions_progress_and_failures(tmp_path):
    db = StateDB(tmp_path / "s.sqlite")
    db.seed({"a": {"label": "a", "state": "S", "lon": 1, "lat": 1, "bbox": (0, 0, 1, 1)}})
    db.start("a")
    db.finish("a", "failed", error="ConnectionError: reset")
    text = download_batch.format_status(db)
    assert "failed 1" in text and "ConnectionError" in text and "target 15000" in text


def test_keys_are_applied_before_limit(tmp_path, monkeypatch):
    registry = write_registry(tmp_path / "r.csv", 5)
    seen = []

    def fake(row, out, cfg, db, session, stop=None, select_only=False):
        seen.append(row["key"]); db.start(row["key"]); db.finish(row["key"], "done", n_pairs=1)
        return "done"

    monkeypatch.setattr(download_batch, "process_aoi", fake)
    download_batch.main(["--db", str(tmp_path / "s.sqlite"), "run", "--registry", registry,
                         "--keys", "site3", "--limit", "1", "--out", str(tmp_path / "o")])
    assert seen == ["site3"]


def test_status_estimates_yield_from_a_select_only_run(tmp_path):
    db = StateDB(tmp_path / "s.sqlite")
    rows = {k: {"label": k, "state": "S", "lon": 1, "lat": 1, "size_km": 5.12, "bbox": (0, 0, 1, 1)}
            for k in ("a", "b", "c", "d")}
    db.seed(rows)
    for key in ("a", "b", "c"):
        db.start(key); db.finish(key, "selected", n_pairs=2)
    db.start("d"); db.finish("d", "no_pairs")
    text = download_batch.format_status(db)
    # 3 AOIs x 2 pairs x 4 patches = 24 now; 75% hit rate x 8 per AOI x 4 AOIs = 24 projected
    assert "3 AOIs have pairs (75% of those searched) -> ~24 patch-pairs" in text


def test_pilot_stops_starting_sites_once_enough_pairs_are_saved(tmp_path, monkeypatch):
    registry = write_registry(tmp_path / "r.csv", 10)
    seen = []

    def fake(row, out, cfg, db, session, stop=None, select_only=False):
        seen.append(row["key"]); db.start(row["key"])
        db.add_pair(row["key"], 1, {"shape": "1024x1024", "bytes": 1, "path": "x"})
        db.finish(row["key"], "done", n_pairs=1)
        return "done"

    monkeypatch.setattr(download_batch, "process_aoi", fake)
    download_batch.main(["--db", str(tmp_path / "s.sqlite"), "run", "--registry", registry, "--workers", "1",
                         "--stop-after-pairs", "3", "--out", str(tmp_path / "o")])
    assert len(seen) == 3
    assert StateDB(tmp_path / "s.sqlite").summary()["counts"]["pending"] == 7


def test_worker_count_can_be_changed_while_a_run_is_going(tmp_path, monkeypatch):
    import threading
    registry = write_registry(tmp_path / "r.csv", 12)
    db_path = tmp_path / "s.sqlite"
    concurrent, peak, lock = [0], [0], threading.Lock()
    release = threading.Event()

    def fake(row, out, cfg, db, session, stop=None, select_only=False):
        with lock:
            concurrent[0] += 1
            peak[0] = max(peak[0], concurrent[0])
            if row["key"] == "site1":          # while the run is at 2 workers, raise it to 5
                download_batch.main(["--db", str(db_path), "workers", "5"])
        release.wait(0.3)
        db.start(row["key"]); db.finish(row["key"], "done", n_pairs=1)
        with lock:
            concurrent[0] -= 1
        return "done"

    monkeypatch.setattr(download_batch, "process_aoi", fake)
    download_batch.main(["--db", str(db_path), "run", "--registry", registry, "--workers", "2",
                         "--out", str(tmp_path / "o")])
    assert peak[0] == 5
    assert StateDB(db_path).summary()["counts"]["done"] == 12


def test_worker_count_is_bounded(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        download_batch.main(["--db", str(tmp_path / "s.sqlite"), "workers", "0"])
