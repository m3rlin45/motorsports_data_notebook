"""Tests for the training-side cleaning: stale-TPMS masking, rebuilt
per-lap TPMS aggregates and the lap usability
filters. The dataset on disk is raw; these rules run when the model loads it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from motorsports_data_notebook.tire_etl.aggregates import compute_corner_aggregates
from tests._timeseries_fixture import write_session_timeseries
from motorsports_data_notebook.tire_model.cleaning import (
    FilterConfig,
    _stale_prefix_len,
    apply_filters,
    clean_laps,
    mask_stale_session_prefix,
)


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


def test_outlap_is_usable_when_it_rolled_long_enough() -> None:
    """Out-laps are tire-model data (pit-exit reading = stint start). They are
    judged on rolling time: lap time and mean speed include the grid wait."""
    t = _laps(
        [
            # 4 min on the grid then a 70 s roll to the line: usable
            _base_lap(is_outlap=True, lap_time_s=310.0, speed_kmh_mean=25.0, moving_s=70.0),
            # rolled only 20 s (recording started just before the line): not usable
            _base_lap(lap_num=2, is_outlap=True, lap_time_s=20.0, moving_s=20.0),
            _base_lap(lap_num=3, moving_s=60.0),
        ]
    )
    out = apply_filters(t, FilterConfig())
    assert out.column("tire_usable").to_pylist() == [True, False, True]
    assert out.column("exclude_reason").to_pylist() == [None, "outlap_too_short", None]
    # The out-lap's long lap time must not count as the stint's best lap nor
    # make the flying lap look slow.
    legacy = apply_filters(t, FilterConfig(exclude_outlap=True))
    assert legacy.column("exclude_reason").to_pylist()[0] == "outlap"


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


def test_stale_prefix_len_detects_constant_run() -> None:
    # 4 bit-exact samples then a real change → prefix is 4.
    assert _stale_prefix_len(np.array([2.40, 2.40, 2.40, 2.40, 2.45, 2.50])) == 4


def test_stale_prefix_len_zero_when_data_starts_moving() -> None:
    # Sample 1 already differs from sample 0 → no stale prefix, return 0.
    assert _stale_prefix_len(np.array([2.40, 2.45, 2.50, 2.55])) == 0


def test_stale_prefix_len_returns_full_size_when_all_constant() -> None:
    arr = np.array([2.40, 2.40, 2.40, 2.40])
    assert _stale_prefix_len(arr) == arr.size


def test_mask_stale_session_prefix_masks_each_tpms_channel_independently() -> None:
    # FL sleeps for 3 samples then wakes; FR is alive from the start.
    ts = pa.table(
        {
            "t_lap_s": pa.array([0.0, 0.1, 0.2, 0.3, 0.4], type=pa.float32()),
            "tpms_press_fl_bar": pa.array([2.40, 2.40, 2.40, 2.45, 2.45], type=pa.float32()),
            "tpms_press_fr_bar": pa.array([2.30, 2.31, 2.32, 2.33, 2.34], type=pa.float32()),
            "tpms_temp_fl_c": pa.array([30.0, 30.0, 30.0, 32.0, 32.0], type=pa.float32()),
        }
    )
    out = mask_stale_session_prefix(ts)
    fl = out.column("tpms_press_fl_bar").to_pylist()
    assert all(v is None or v != v for v in fl[:3])  # NaN
    assert fl[3] == pytest.approx(2.45)
    fr = out.column("tpms_press_fr_bar").to_pylist()
    # FR was alive from sample 0 — no mask applied anywhere.
    assert fr[0] == pytest.approx(2.30)
    # FL temp also masked at the same 3-sample prefix length (per-channel).
    temp = out.column("tpms_temp_fl_c").to_pylist()
    assert all(v is None or v != v for v in temp[:3])
    assert temp[3] == pytest.approx(32.0)


def test_stale_mask_is_per_corner_first_change_on_either_stream() -> None:
    """One module reports a corner's pressure and temperature. As soon as
    either stream changes the module is awake, so the other stream's
    constant reading from that sample on is real and must not be masked."""
    import numpy as np
    import pyarrow as pa
    from motorsports_data_notebook.tire_model.cleaning import mask_stale_session_prefix

    n = 12
    # FL: pressure changes at sample 3, temperature only at sample 8.
    press_fl = np.array([2.0] * 3 + [2.03] * 9)
    temp_fl = np.array([20.0] * 8 + [21.0] * 4)
    # RR: both change at sample 5.
    press_rr = np.array([2.0] * 5 + [2.03] * 7)
    temp_rr = np.array([20.0] * 5 + [21.0] * 7)
    ts = pa.table(
        {
            "tpms_press_fl_bar": press_fl,
            "tpms_temp_fl_c": temp_fl,
            "tpms_press_rr_bar": press_rr,
            "tpms_temp_rr_c": temp_rr,
        }
    )
    out = mask_stale_session_prefix(ts)
    fl_t = out.column("tpms_temp_fl_c").to_numpy(zero_copy_only=False)
    fl_p = out.column("tpms_press_fl_bar").to_numpy(zero_copy_only=False)
    # FL temperature is valid from sample 3 (pressure proved the module awake),
    # still reading its constant 20 °C until it ticks up at sample 8.
    assert np.isnan(fl_t[:3]).all() and np.isnan(fl_p[:3]).all()
    assert fl_t[3] == 20.0 and fl_p[3] == 2.03
    rr_t = out.column("tpms_temp_rr_c").to_numpy(zero_copy_only=False)
    assert np.isnan(rr_t[:5]).all() and rr_t[5] == 21.0


def test_clean_laps_rebuilds_tpms_summary_from_masked_timeseries(tmp_path) -> None:
    """A raw dataset whose out-lap starts with a stale hot reading (the
    module asleep from an earlier run): the committed lap row carries the
    stale start, clean_laps replaces it with the first live reading and
    flags the laps."""
    root = tmp_path
    (root / "laps").mkdir()
    (root / "timeseries" / "2026-10").mkdir(parents=True)
    sid = "abc123"
    n0, n1 = 8, 6
    press = [2.19, 2.19, 2.19, 1.60, 1.60, 1.60, 1.63, 1.63] + [1.66] * n1
    temp = [36.0, 36.0, 36.0, 27.0, 27.0, 27.0, 27.0, 28.0] + [31.0] * n1
    ts = pa.table(
        {
            "session_id": [sid] * (n0 + n1),
            "lap_num": pa.array([0] * n0 + [1] * n1, type=pa.int16()),
            "t_lap_s": pa.array(
                list(np.arange(n0, dtype=np.float32)) + list(np.arange(n1, dtype=np.float32)),
                type=pa.float32(),
            ),
            "tpms_press_rl_bar": pa.array(press, type=pa.float32()),
            "tpms_temp_rl_c": pa.array(temp, type=pa.float32()),
        }
    )
    write_session_timeseries(root, sid, ts, partition="2026-10")
    laps = pd.DataFrame(
        [
            {
                "session_id": sid,
                "lap_num": 0,
                "stint_id": 1,
                "lap_time_s": 120.0,
                "is_outlap": True,
                "is_inlap": False,
                "on_track_s": 120.0,
                "moving_s": 100.0,
                "speed_kmh_mean": 60.0,
                "tpms_press_rl_start": 2.19,
                "tpms_temp_rl_start": 36.0,
                "tpms_press_rl_min": 1.60,
                "tpms_press_rl_max": 2.19,
            },
            {
                "session_id": sid,
                "lap_num": 1,
                "stint_id": 1,
                "lap_time_s": 80.0,
                "is_outlap": False,
                "is_inlap": False,
                "on_track_s": 80.0,
                "moving_s": 80.0,
                "speed_kmh_mean": 100.0,
                "tpms_press_rl_start": 1.66,
                "tpms_temp_rl_start": 31.0,
                "tpms_press_rl_min": 1.66,
                "tpms_press_rl_max": 1.66,
            },
        ]
    )
    pq.write_table(
        pa.Table.from_pandas(laps, preserve_index=False), root / "laps" / "2026-10.parquet"
    )

    out = clean_laps(root, laps)
    lap0 = out[out["lap_num"] == 0].iloc[0]
    assert lap0["tpms_press_rl_start"] == pytest.approx(1.60)
    assert lap0["tpms_temp_rl_start"] == pytest.approx(27.0)
    assert lap0["tpms_press_rl_first_valid_s"] == pytest.approx(3.0)
    assert lap0["tpms_press_rl_max"] == pytest.approx(1.63)
    assert bool(lap0["tire_usable"])
    lap1 = out[out["lap_num"] == 1].iloc[0]
    # Flat pressure over the lap: the filters flag it as stuck.
    assert lap1["exclude_reason"] == "tpms_stuck"
    assert not bool(lap1["tire_usable"])


def test_clean_laps_passes_through_synthetic_frames_with_usability() -> None:
    laps = pd.DataFrame({"session_id": ["s"], "lap_num": [1], "tire_usable": [True]})
    out = clean_laps(Path("/nonexistent"), laps)
    assert out.equals(laps)


def test_apply_filters_best_lap_is_per_session_and_stint() -> None:
    """Two sessions whose stints both carry id 1: a 100 s lap is slow next to
    an 80 s best only within its own session."""
    rows = []
    for sid, times in (("a", [80.0, 100.0]), ("b", [100.0, 101.0])):
        for i, t in enumerate(times):
            rows.append(
                {
                    "session_id": sid,
                    "lap_num": i + 1,
                    "stint_id": 1,
                    "lap_time_s": t,
                    "is_outlap": False,
                    "is_inlap": False,
                    "on_track_s": t,
                    "speed_kmh_mean": 90.0,
                    "tpms_press_fl_min": 1.6,
                    "tpms_press_fl_max": 1.8,
                }
            )
    out = apply_filters(pd.DataFrame(rows), FilterConfig(max_lap_time_vs_best_pct=1.2))
    assert isinstance(out, pd.DataFrame)
    assert out["exclude_reason"].tolist() == [None, "lap_too_slow", None, None]
