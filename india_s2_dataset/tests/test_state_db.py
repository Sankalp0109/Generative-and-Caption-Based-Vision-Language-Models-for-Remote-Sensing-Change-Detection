import pytest

from state_db import StateDB


def registry(*keys):
    return {k: {"label": k, "state": "S", "district": "D", "category": "c", "lon": 73.0, "lat": 19.0,
                "size_km": 5.12, "bbox": (72.9, 18.9, 73.0, 19.0), "season_months": "11,12,1,2,3"}
            for k in keys}


def test_seed_is_idempotent_and_keeps_progress(tmp_path):
    db = StateDB(tmp_path / "s.sqlite")
    assert db.seed(registry("a", "b")) == {"added": 2, "existing": 0}
    db.start("a"); db.finish("a", "done", n_pairs=2)
    assert db.seed(registry("a", "b", "c")) == {"added": 1, "existing": 2}
    assert db.summary()["counts"]["done"] == 1  # re-seeding did not reset finished work


def test_interrupted_running_rows_are_requeued(tmp_path):
    db = StateDB(tmp_path / "s.sqlite")
    db.seed(registry("a", "b"))
    db.start("a")
    assert db.recover_interrupted() == 1
    assert {r["key"] for r in db.queue()} == {"a", "b"}


def test_queue_filters_by_status_and_limit(tmp_path):
    db = StateDB(tmp_path / "s.sqlite")
    db.seed(registry("a", "b", "c"))
    db.start("a"); db.finish("a", "failed", error="boom")
    db.start("b"); db.finish("b", "done", n_pairs=1)
    assert [r["key"] for r in db.queue()] == ["c"]
    assert {r["key"] for r in db.queue(("pending", "failed"))} == {"a", "c"}
    assert len(db.queue(("pending", "failed"), limit=1)) == 1


def test_stage_pairs_and_summary(tmp_path):
    db = StateDB(tmp_path / "s.sqlite")
    db.seed(registry("a"))
    db.start("a")
    db.stage("a", "downloading", "pair 1/2")
    assert db.summary()["running"][0]["stage"] == "downloading"
    db.add_pair("a", 1, {"shape": "512x512", "bytes": 1000, "path": "x"})
    db.finish("a", "done", n_pairs=1)
    summary = db.summary()
    assert summary["pairs"] == 1 and summary["bytes"] == 1000 and summary["running"] == []
    assert db.get_meta("missing", 5) == 5
    db.set_meta("target", 15000)
    assert db.get_meta("target") == 15000


def test_finish_rejects_unknown_status(tmp_path):
    db = StateDB(tmp_path / "s.sqlite")
    db.seed(registry("a"))
    with pytest.raises(AssertionError):
        db.finish("a", "bogus")
