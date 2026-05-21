"""Tests for lap-level filter logic."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from motorsports_data_notebook.tire_etl.filters import FilterConfig, apply_filters


def _laps(rows: list[dict]) -> pa.Table:
    # Column-orient; align on keys.
    if not rows:
        return pa.table({})
    cols: dict[str, list] = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k in cols:
            cols[k].append(r.get(k, None))
    return pa.table(cols)


def _base_lap(**overrides) -> dict:
    d = {
        "lap_num": 1,
        "stint_id": 1,
        "lap_time_s": 60.0,
        "is_outlap": False,
        "is_inlap": False,
        "speed_kmh_mean": 120.0,
        "on_track_s": 60.0,
        # Healthy warmup span of 0.10 bar (100 kPa equivalent) — far above the
        # 0.003 bar stuck threshold.
        "tpms_press_fl_min": 2.00,
        "tpms_press_fl_max": 2.10,
        "tpms_press_fr_min": 2.00,
        "tpms_press_fr_max": 2.10,
        "tpms_press_rl_min": 2.00,
        "tpms_press_rl_max": 2.10,
        "tpms_press_rr_min": 2.00,
        "tpms_press_rr_max": 2.10,
    }
    d.update(overrides)
    return d


def test_outlap_excluded() -> None:
    t = _laps([_base_lap(is_outlap=True)])
    out = apply_filters(t, FilterConfig())
    assert out.column("tire_usable").to_pylist() == [False]
    assert out.column("exclude_reason").to_pylist() == ["outlap"]


def test_short_lap_excluded() -> None:
    t = _laps([_base_lap(lap_time_s=10.0)])
    out = apply_filters(t, FilterConfig())
    assert out.column("exclude_reason").to_pylist() == ["lap_too_short"]


def test_stuck_tpms_excluded() -> None:
    # All four corners stuck within tolerance
    base = _base_lap()
    for c in ("fl", "fr", "rl", "rr"):
        base[f"tpms_press_{c}_min"] = 2.00
        base[f"tpms_press_{c}_max"] = 2.001  # 0.001 bar range < 0.003 tolerance
    t = _laps([base])
    out = apply_filters(t, FilterConfig())
    assert out.column("exclude_reason").to_pylist() == ["tpms_stuck"]


def test_missing_tpms_excluded() -> None:
    base = _base_lap()
    for c in ("fl", "fr", "rl", "rr"):
        base[f"tpms_press_{c}_min"] = float("nan")
        base[f"tpms_press_{c}_max"] = float("nan")
    t = _laps([base])
    out = apply_filters(t, FilterConfig())
    assert out.column("exclude_reason").to_pylist() == ["no_tpms"]


def test_normal_lap_passes() -> None:
    t = _laps([_base_lap()])
    out = apply_filters(t, FilterConfig())
    assert out.column("tire_usable").to_pylist() == [True]
    assert out.column("exclude_reason").to_pylist() == [None]


def test_slow_relative_to_best_excluded() -> None:
    # Best lap 60s, one lap at 120s -> 2x best -> exceeds 1.4x default
    t = _laps(
        [
            _base_lap(lap_num=1, lap_time_s=60.0),
            _base_lap(lap_num=2, lap_time_s=120.0),
        ]
    )
    out = apply_filters(t, FilterConfig())
    reasons = out.column("exclude_reason").to_pylist()
    assert reasons[0] is None
    assert reasons[1] == "lap_too_slow"


def test_empty_table() -> None:
    t = pa.table(
        {
            "lap_num": pa.array([], type=pa.int16()),
            "stint_id": pa.array([], type=pa.int16()),
            "lap_time_s": pa.array([], type=pa.float32()),
            "is_outlap": pa.array([], type=pa.bool_()),
            "is_inlap": pa.array([], type=pa.bool_()),
            "speed_kmh_mean": pa.array([], type=pa.float32()),
            "on_track_s": pa.array([], type=pa.float32()),
        }
    )
    out = apply_filters(t, FilterConfig())
    assert len(out) == 0
    assert "tire_usable" in out.schema.names
