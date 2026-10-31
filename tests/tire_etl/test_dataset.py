"""Tests for dataset upsert + MANIFEST management."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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


def test_upsert_with_no_laps_drops_the_sessions_earlier_lap_rows(tmp_path: Path) -> None:
    """A session that re-extracts as a zero-lap error must not keep the lap
    rows an earlier run wrote for it (the writer used to skip the laps
    partition entirely when the new extraction had no rows)."""
    root = tmp_path
    first = _simple_extract_result("sess_a", "/aim/2026-01-01/a.xrk", "2026-01-01")
    upsert_session(root, first)
    laps_path = root / "laps" / "2026-01.parquet"
    assert pq.read_table(laps_path).num_rows == 1

    errored = ExtractResult(
        session_row=first.session_row,
        laps_rows=pa.table({}),
        timeseries=pa.table({}),
        status="error",
        error_msg="no laps",
    )
    upsert_session(root, errored)
    assert pq.read_table(laps_path).num_rows == 0
    # ... nor the timeseries file the earlier run wrote for it.
    assert not (root / "timeseries" / "2026-01" / "sess_a.parquet").exists()

    # Other sessions' rows in the same partition survive.
    other = _simple_extract_result("sess_b", "/aim/2026-01-02/b.xrk", "2026-01-02")
    upsert_session(root, other)
    upsert_session(root, errored)
    assert pq.read_table(laps_path).column("session_id").to_pylist() == ["sess_b"]


def test_remove_manifest_entries_drops_only_named_paths(tmp_path) -> None:
    from motorsports_data_notebook.tire_etl.dataset import (
        _write_manifest,
        load_manifest,
        remove_manifest_entries,
    )

    entries = [
        {
            "xrk_path": "/aim/2026-10-24/CMD_KK-F_X_a_0370.xrz",
            "session_id": "a",
            "date": "2026-10-24",
        },
        {
            "xrk_path": "/aim/2026-10-24/CMD_KK-F_X_a_0376.xrk",
            "session_id": "b",
            "date": "2026-10-24",
        },
    ]
    _write_manifest(tmp_path, entries)
    assert (
        remove_manifest_entries(tmp_path, ["/aim/2026-10-24/CMD_KK-F_X_a_0370.xrz", "/nope"]) == 1
    )
    assert list(load_manifest(tmp_path)) == ["/aim/2026-10-24/CMD_KK-F_X_a_0376.xrk"]
    assert remove_manifest_entries(tmp_path, ["/nope"]) == 0


class TestTimebaseLayout:
    """Schema 7: one row group per raw timebase, loaders merge at query time."""

    def _table(self):
        import numpy as np

        from motorsports_data_notebook.tire_etl.dataset import build_timeseries_table

        def keys(t):
            return {
                "lap_num": pa.array(np.ones(len(t), dtype=np.int16)),
                "stint_id": pa.array(np.ones(len(t), dtype=np.int16)),
                "sample_idx": pa.array(np.arange(len(t), dtype=np.int32)),
                "t_session_s": pa.array(t, type=pa.float64()),
                "t_lap_s": pa.array(np.asarray(t, dtype=np.float32)),
            }

        t_g = np.arange(0.0, 4.0, 0.04)
        t_p = np.arange(0.0, 4.0, 0.8)
        gps = {**keys(t_g), "speed_ms": pa.array(np.full(len(t_g), 30.0, dtype=np.float32))}
        tp = {
            **keys(t_p),
            "tpms_press_fl_bar": pa.array(np.linspace(2.0, 2.1, len(t_p)).astype(np.float32)),
        }
        return build_timeseries_table(
            "s1", [("gps", "gps", pa.table(gps)), ("tpms/fl", "tpms", pa.table(tp))]
        )

    def test_build_puts_keys_first_and_nulls_outside_a_timebase(self) -> None:
        from motorsports_data_notebook.tire_etl.dataset import TIMESERIES_KEY_COLUMNS

        tbl = self._table()
        assert tbl.schema.names[: len(TIMESERIES_KEY_COLUMNS)] == TIMESERIES_KEY_COLUMNS
        assert tbl.column("speed_ms").null_count == 5
        assert tbl.column("tpms_press_fl_bar").null_count == 100

    def test_write_read_round_trip_with_row_groups_and_metadata(self, tmp_path: Path) -> None:
        from motorsports_data_notebook.tire_etl.dataset import (
            channel_series,
            load_group,
            read_timeseries,
            timeseries_metadata,
            timeseries_summary,
            write_timeseries,
        )

        tbl = self._table()
        summary = timeseries_summary(tbl)
        assert [e["timebase"] for e in summary] == ["gps", "tpms/fl"]
        assert summary[0]["rate_hz"] == pytest.approx(25.0) and summary[1]["n"] == 5
        path = tmp_path / "timeseries" / "2026-01" / "s1.parquet"
        write_timeseries(tbl, path, summary=summary, kinematics={"accel_source": "imu"})
        md = pq.read_metadata(path)
        assert md.num_row_groups == 2
        meta = timeseries_metadata(path)
        assert meta["schema"] == 9 and meta["kinematics"] == {"accel_source": "imu"}
        assert [e["timebase"] for e in meta["timebases"]] == ["gps", "tpms/fl"]
        tp = read_timeseries(path, columns=["tpms_press_fl_bar"], families=["tpms"])
        assert len(tp) == 5 and "speed_ms" not in tp.schema.names
        grp = load_group(tmp_path, "s1", "gps")
        assert (
            grp is not None
            and len(grp) == 100
            and set(grp.schema.names) >= {"speed_ms", "t_session_s"}
        )
        t, v = channel_series(grp, "speed_ms", "t_session_s")
        assert len(t) == 100 and v[0] == 30.0
        assert load_group(tmp_path, "s1", "surface") is None
        assert load_group(tmp_path, "nope", "gps") is None


def test_read_timeseries_refuses_a_file_of_another_schema(tmp_path) -> None:
    """A file without the schema-7 metadata (an older resampled layout, or a
    hand-written parquet) must not be read as if it were native-rate."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest

    from motorsports_data_notebook.tire_etl.dataset import read_timeseries

    path = tmp_path / "old.parquet"
    pq.write_table(pa.table({"t_lap_s": [0.0, 1.0], "speed_ms": [1.0, 2.0]}), path)
    with pytest.raises(ValueError, match="schema missing"):
        read_timeseries(path)
    old = pa.table({"t_lap_s": [0.0]}).replace_schema_metadata({b"tire_etl.schema": b"6"})
    pq.write_table(old, path)
    with pytest.raises(ValueError, match="schema 6"):
        read_timeseries(path)


def test_narrowest_float_picks_the_exact_width() -> None:
    import numpy as np
    import pyarrow as pa

    from motorsports_data_notebook.tire_etl.dataset import narrowest_float

    # A float16 source widened to float32 by libxrk narrows back exactly.
    f16 = np.array([0.2308, -0.98486, 7.1289, 0.0, np.nan], dtype=np.float16).astype(np.float32)
    arr = narrowest_float(f16)
    assert arr.type == pa.float16()
    assert np.array_equal(
        arr.to_numpy(zero_copy_only=False).astype(np.float32), f16, equal_nan=True
    )
    # A genuine float32 value (TPMS 1.28 bar, an IR 48.7 °C) stays float32.
    assert narrowest_float(np.array([1.28, 1.25], dtype=np.float32)).type == pa.float32()
    assert narrowest_float(np.array([48.7, 48.4], dtype=np.float32)).type == pa.float32()
    # Beyond float16's range stays float32 even when the mantissa would fit.
    assert narrowest_float(np.array([70000.0, 0.0], dtype=np.float32)).type == pa.float32()
    # Positions are always float64.
    assert narrowest_float(np.array([35.39534]), float64=True).type == pa.float64()
    # All-NaN or empty: float32.
    assert narrowest_float(np.array([np.nan, np.nan], dtype=np.float32)).type == pa.float32()


def test_write_timeseries_encodings(tmp_path) -> None:
    """Floats are byte-stream-split (float16 PLAIN, DuckDB's limit), no
    dictionary on numbers, delta-packed sample_idx, and float16 columns
    read back as float16."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from motorsports_data_notebook.tire_etl.dataset import (
        build_timeseries_table,
        narrowest_float,
        read_timeseries,
        write_timeseries,
    )

    n = 2000
    keys = {
        "lap_num": pa.array(np.ones(n, dtype=np.int16)),
        "stint_id": pa.array(np.ones(n, dtype=np.int16)),
        "sample_idx": pa.array(np.arange(n, dtype=np.int32)),
        "t_session_s": pa.array(np.arange(n) / 50.0, type=pa.float64()),
        "t_lap_s": pa.array((np.arange(n) / 50.0).astype(np.float32)),
    }
    acc = dict(keys)
    acc["lat_g"] = narrowest_float(np.sin(np.arange(n) / 7.0).astype(np.float16).astype(np.float32))
    gps = dict(keys)
    gps["speed_ms"] = narrowest_float(np.linspace(0.0, 55.123, n))
    gps["gps_lat"] = narrowest_float(35.0 + np.arange(n) * 1e-6, float64=True)
    table = build_timeseries_table(
        "s", [("accel", "accel", pa.table(acc)), ("gps", "gps", pa.table(gps))]
    )
    assert table.schema.field("lat_g").type == pa.float16()
    assert table.schema.field("speed_ms").type == pa.float32()
    path = tmp_path / "s.parquet"
    write_timeseries(table, path)
    md = pq.ParquetFile(path).metadata
    enc = {}
    for rg in range(md.num_row_groups):
        g = md.row_group(rg)
        for i in range(g.num_columns):
            c = g.column(i)
            enc.setdefault(c.path_in_schema, set()).update(str(e) for e in c.encodings)
    assert "BYTE_STREAM_SPLIT" in enc["speed_ms"] and "BYTE_STREAM_SPLIT" in enc["gps_lat"]
    assert "BYTE_STREAM_SPLIT" in enc["t_session_s"]
    assert "RLE_DICTIONARY" not in enc["lat_g"] and "BYTE_STREAM_SPLIT" not in enc["lat_g"]
    assert "DELTA_BINARY_PACKED" in enc["sample_idx"]
    assert "RLE_DICTIONARY" in enc["timebase"]
    back = read_timeseries(path, columns=["lat_g", "speed_ms"], families=["accel"])
    assert back.schema.field("lat_g").type == pa.float16()
    lat = back.column("lat_g").to_numpy(zero_copy_only=False).astype(np.float32)
    assert np.array_equal(lat, acc["lat_g"].to_numpy(zero_copy_only=False).astype(np.float32))


def test_timeseries_summary_rate_is_the_mean_rate_not_the_median_interval() -> None:
    """The KK-SII dash logs its accelerometer as pairs 2 ms apart on 50 Hz
    frames: 100 samples per second whose median interval is 2 ms."""
    import numpy as np
    import pyarrow as pa

    from motorsports_data_notebook.tire_etl.dataset import (
        build_timeseries_table,
        timeseries_summary,
    )

    frames = np.arange(0, 20_000, 20)  # 1000 frames over 20 s
    t_ms = np.sort(np.concatenate([frames, frames + 2]))
    n = t_ms.size
    tbl = pa.table(
        {
            "lap_num": pa.array(np.ones(n, dtype=np.int16)),
            "stint_id": pa.array(np.ones(n, dtype=np.int16)),
            "sample_idx": pa.array(np.arange(n, dtype=np.int32)),
            "t_session_s": pa.array(t_ms / 1000.0, type=pa.float64()),
            "t_lap_s": pa.array((t_ms / 1000.0).astype(np.float32)),
            "lat_g": pa.array(np.zeros(n, dtype=np.float32)),
        }
    )
    summary = timeseries_summary(build_timeseries_table("s", [("accel", "accel", tbl)]))
    assert summary[0]["n"] == n and abs(summary[0]["rate_hz"] - 100.0) < 0.5
    assert summary[0]["gap_s"] == 0.0 and summary[0]["n_gaps"] == 0
    # A stream that stops for three minutes mid-session (the KK-SII IMU in
    # spring 2026) still has a 50 Hz rate; the dropout is reported as gap time.
    t_ms = np.concatenate([np.arange(0, 20_000, 20), np.arange(200_000, 220_000, 20)])
    n = t_ms.size
    gappy = pa.table(
        {
            "lap_num": pa.array(np.ones(n, dtype=np.int16)),
            "stint_id": pa.array(np.ones(n, dtype=np.int16)),
            "sample_idx": pa.array(np.arange(n, dtype=np.int32)),
            "t_session_s": pa.array(t_ms / 1000.0, type=pa.float64()),
            "t_lap_s": pa.array((t_ms / 1000.0).astype(np.float32)),
            "lat_g": pa.array(np.zeros(n, dtype=np.float32)),
        }
    )
    summary = timeseries_summary(build_timeseries_table("s", [("accel", "accel", gappy)]))
    assert abs(summary[0]["rate_hz"] - 50.0) < 0.5
    assert summary[0]["n_gaps"] == 1 and abs(summary[0]["gap_s"] - 180.02) < 0.1
