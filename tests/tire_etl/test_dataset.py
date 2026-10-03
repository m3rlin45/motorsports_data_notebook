"""Tests for dataset upsert + MANIFEST management."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from motorsports_data_notebook.tire_etl.dataset import (
    load_manifest,
    upsert_session,
)
from motorsports_data_notebook.tire_etl.extract import ExtractResult


def _session_row(session_id: str, path: str, date_str: str) -> pa.Table:
    y, m, d = (int(x) for x in date_str.split("-"))
    return pa.table(
        {
            "session_id": [session_id],
            "xrk_path": [path],
            "xrk_mtime_ns": pa.array([1234567890], type=pa.int64()),
            "file_size": pa.array([1024], type=pa.int64()),
            "date": pa.array([date(y, m, d)], type=pa.date32()),
            "session_start_utc": pa.array(
                [datetime(y, m, d, tzinfo=timezone.utc)], type=pa.timestamp("us", tz="UTC")
            ),
            "driver": ["CMD"],
            "car": ["Inferno 86"],
            "track": ["Tsukuba"],
            "track_canonical": ["tsukuba_2000"],
            "session_type": ["Generic testing"],
            "run_num": pa.array([1], type=pa.int32()),
            "logger_id": ["abc"],
            "profile_name": ["default"],
            "extractor_version": ["0.1.0"],
            "extracted_at": pa.array(
                [datetime(y, m, d, tzinfo=timezone.utc)], type=pa.timestamp("us", tz="UTC")
            ),
            "status": ["ok"],
            "error_msg": pa.array([None], type=pa.string()),
            "n_laps": pa.array([5], type=pa.int32()),
            "n_tire_usable_laps": pa.array([3], type=pa.int32()),
            "has_tpms": [True],
            "has_surface_temp": [False],
            "has_ambient_temp": [False],
            "has_track_temp": [False],
            "ts_rate_hz": pa.array([1.0], type=pa.float32()),
        }
    )


def _simple_extract_result(session_id: str, path: str, date_str: str) -> ExtractResult:
    session = _session_row(session_id, path, date_str)
    laps = pa.table(
        {
            "session_id": pa.array([session_id], type=pa.string()),
            "lap_num": pa.array([1], type=pa.int16()),
            "stint_id": pa.array([1], type=pa.int16()),
            "lap_time_s": pa.array([60.0], type=pa.float32()),
        }
    )
    ts = pa.table(
        {
            "session_id": pa.array([session_id] * 3, type=pa.string()),
            "lap_num": pa.array([1, 1, 1], type=pa.int16()),
            "sample_idx": pa.array([0, 1, 2], type=pa.int32()),
            "t_lap_s": pa.array([0.0, 1.0, 2.0], type=pa.float32()),
        }
    )
    return ExtractResult(
        session_row=session,
        laps_rows=laps,
        timeseries=ts,
        status="ok",
        error_msg=None,
    )


def test_upsert_writes_all_three_artifacts(tmp_path: Path) -> None:
    r = _simple_extract_result("s1", "/fake/path/a.xrk", "2026-04-04")
    upsert_session(tmp_path, r)

    assert (tmp_path / "sessions" / "2026-04.parquet").exists()
    assert (tmp_path / "laps" / "2026-04.parquet").exists()
    assert (tmp_path / "timeseries" / "2026-04" / "s1.parquet").exists()
    manifest = load_manifest(tmp_path)
    assert "/fake/path/a.xrk" in manifest
    assert manifest["/fake/path/a.xrk"]["n_samples"] == 3


def test_upsert_replaces_existing_session(tmp_path: Path) -> None:
    r1 = _simple_extract_result("s1", "/fake/a.xrk", "2026-04-04")
    r2 = _simple_extract_result("s2", "/fake/b.xrk", "2026-04-04")
    upsert_session(tmp_path, r1)
    upsert_session(tmp_path, r2)

    sessions = pq.read_table(tmp_path / "sessions" / "2026-04.parquet")
    assert sorted(sessions.column("session_id").to_pylist()) == ["s1", "s2"]

    # Re-upsert s1 (e.g. after a re-extract) — count stays at 2.
    upsert_session(tmp_path, r1)
    sessions = pq.read_table(tmp_path / "sessions" / "2026-04.parquet")
    assert sorted(sessions.column("session_id").to_pylist()) == ["s1", "s2"]


def test_manifest_sorted_by_date_then_session(tmp_path: Path) -> None:
    r_mar = _simple_extract_result("z", "/fake/z.xrk", "2026-03-04")
    r_apr_a = _simple_extract_result("a", "/fake/a.xrk", "2026-04-04")
    r_apr_b = _simple_extract_result("b", "/fake/b.xrk", "2026-04-04")
    upsert_session(tmp_path, r_apr_b)
    upsert_session(tmp_path, r_apr_a)
    upsert_session(tmp_path, r_mar)

    lines = (tmp_path / "MANIFEST.jsonl").read_text().splitlines()
    assert len(lines) == 3
    import json

    dates = [json.loads(line)["date"] for line in lines]
    sids = [json.loads(line)["session_id"] for line in lines]
    assert dates == ["2026-03-04", "2026-04-04", "2026-04-04"]
    assert sids == ["z", "a", "b"]


def test_prune_orphan_sessions_drops_rows_not_in_manifest(tmp_path) -> None:
    """A regrouped restart pair leaves the old grouping's session_id behind in
    sessions/laps/timeseries; the manifest (keyed by file) is the truth."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from motorsports_data_notebook.tire_etl.dataset import prune_orphan_sessions

    root = tmp_path
    (root / "sessions").mkdir()
    (root / "laps").mkdir()
    (root / "timeseries" / "2026-01").mkdir(parents=True)
    pq.write_table(
        pa.table({"session_id": ["live", "orphan"], "date": ["2026-01-01", "2026-01-01"]}),
        root / "sessions" / "2026-01.parquet",
    )
    pq.write_table(
        pa.table({"session_id": ["live", "live", "orphan"], "lap_num": [1, 2, 1]}),
        root / "laps" / "2026-01.parquet",
    )
    pq.write_table(pa.table({"x": [1]}), root / "timeseries" / "2026-01" / "live.parquet")
    pq.write_table(pa.table({"x": [1]}), root / "timeseries" / "2026-01" / "orphan.parquet")
    (root / "MANIFEST.jsonl").write_text(
        json.dumps({"session_id": "live", "xrk_path": "/a.xrk", "date": "2026-01-01"}) + "\n"
    )

    counts = prune_orphan_sessions(root)
    assert counts == {"session_rows": 1, "lap_rows": 1, "timeseries_files": 1}
    assert pq.read_table(root / "sessions" / "2026-01.parquet").column(
        "session_id"
    ).to_pylist() == ["live"]
    assert len(pq.read_table(root / "laps" / "2026-01.parquet")) == 2
    assert not (root / "timeseries" / "2026-01" / "orphan.parquet").exists()
    assert (root / "timeseries" / "2026-01" / "live.parquet").exists()
    # Idempotent.
    assert prune_orphan_sessions(root) == {"session_rows": 0, "lap_rows": 0, "timeseries_files": 0}


def test_anti_join_keeps_incoming_column_order() -> None:
    """Partitions must share one column order or non-promoting readers fail."""
    import pyarrow as pa

    from motorsports_data_notebook.tire_etl.dataset import _anti_join_and_concat

    existing = pa.table({"session_id": ["a"], "n_laps": [1], "track": ["x"]})
    new = pa.table({"session_id": ["b"], "track": ["y"], "layout_id": ["l"], "n_laps": [2]})
    out = _anti_join_and_concat(existing, new)
    assert out.schema.names == ["session_id", "track", "layout_id", "n_laps"]
    assert out.column("layout_id").to_pylist() == [None, "l"]


def test_compare_manifests_by_file_group() -> None:
    from motorsports_data_notebook.tire_etl.dataset import compare_manifests

    base = [
        {"session_id": "s1", "xrk_path": "/d/a.xrk", "status": "ok", "n_laps": 10},
        {"session_id": "s2", "xrk_path": "/d/b.xrk", "status": "ok", "n_laps": 5},
        {"session_id": "s3", "xrk_path": "/d/c.xrk", "status": "ok", "n_laps": 6},
        {"session_id": "s4", "xrk_path": "/d/gone.xrk", "status": "error", "n_laps": 0},
    ]
    current = [
        {"session_id": "t1", "xrk_path": "/d/a.xrk", "status": "ok", "n_laps": 9},  # changed
        {"session_id": "t2", "xrk_path": "/d/b.xrk", "status": "ok", "n_laps": 10},  # merged
        {"session_id": "t2", "xrk_path": "/d/c.xrk", "status": "ok", "n_laps": 10},
        {"session_id": "t5", "xrk_path": "/d/new.xrk", "status": "ok", "n_laps": 3},  # added
    ]
    d = compare_manifests(base, current)
    assert [r["paths"] for r in d["changed"]] == [["/d/a.xrk"]]
    assert d["changed"][0]["n_laps"] == (10, 9)
    assert [r["paths"] for r in d["regrouped"]] == [["/d/b.xrk", "/d/c.xrk"]]
    assert [r["paths"] for r in d["added"]] == [["/d/new.xrk"]]
    assert [r["paths"] for r in d["removed"]] == [["/d/gone.xrk"]]
