"""Tests for per-lap tire aggregate stats."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from motorsports_data_notebook.tire_etl.aggregates import (
    compute_corner_aggregates,
    compute_lap_dynamics,
)


def _ts(cols: dict[str, list]) -> pa.Table:
    return pa.table({k: pa.array(v) for k, v in cols.items()})


def test_corner_aggregates_basic() -> None:
    ts = _ts(
        {
            "t_lap_s": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "tpms_press_fl_bar": [2.00, 2.05, 2.10, 2.15, 2.20, 2.25, 2.30],
            "tpms_temp_fl_c": [30.0, 40.0, 50.0, 55.0, 58.0, 60.0, 62.0],
        }
    )
    a = compute_corner_aggregates(ts, "fl")
    assert a.press_start == pytest.approx(2.00)
    assert a.press_end == pytest.approx(2.30)
    assert a.press_min == pytest.approx(2.00)
    assert a.press_max == pytest.approx(2.30)
    assert a.press_mean == pytest.approx(2.15)
    # Rise rate: (2.30-2.00)/60s = 0.005 bar/s = 0.3 bar/min
    assert a.press_rise_bar_per_min == pytest.approx(0.3)
    assert a.temp_start == 30.0
    assert a.temp_max == 62.0


def test_corner_aggregates_missing_channel() -> None:
    ts = _ts({"t_lap_s": [0.0, 1.0, 2.0]})  # no tire columns
    a = compute_corner_aggregates(ts, "fl")
    assert np.isnan(a.press_start)
    assert np.isnan(a.press_end)
    assert np.isnan(a.press_rise_bar_per_min)


def test_corner_aggregates_too_short_for_rise_rate() -> None:
    ts = _ts(
        {
            "t_lap_s": [0.0, 0.5],
            "tpms_press_fl_bar": [2.00, 2.01],
        }
    )
    a = compute_corner_aggregates(ts, "fl")
    # Span < 1s -> rise_rate is NaN
    assert np.isnan(a.press_rise_bar_per_min)
    # But other stats still present
    assert a.press_mean == pytest.approx(2.005)


def test_lap_dynamics_heat_proxy_increases_with_lat_g() -> None:
    # Flat 0.5g lateral for 60s
    ts_low = _ts(
        {
            "t_lap_s": [float(i) for i in range(61)],
            "lat_g": [0.5] * 61,
            "long_g": [0.0] * 61,
            "speed_kmh": [100.0] * 61,
            "brake_bar": [0.0] * 61,
            "throttle_pct": [80.0] * 61,
        }
    )
    d_low = compute_lap_dynamics(ts_low)
    # Double the lat_g -> heat proxy 4x (squared)
    ts_hi = _ts(
        {
            "t_lap_s": [float(i) for i in range(61)],
            "lat_g": [1.0] * 61,
            "long_g": [0.0] * 61,
            "speed_kmh": [100.0] * 61,
            "brake_bar": [0.0] * 61,
            "throttle_pct": [80.0] * 61,
        }
    )
    d_hi = compute_lap_dynamics(ts_hi)
    assert d_hi["heat_proxy"] > d_low["heat_proxy"] * 3.5
    assert d_low["speed_kmh_mean"] == 100.0
    assert d_low["throttle_mean"] == 80.0


def test_lap_dynamics_moving_time_excludes_standstill() -> None:
    """An out-lap that waits on the grid before rolling: on_track_s spans the
    whole lap, moving_s only the rolling part, speed_kmh_first is the pit speed."""
    import numpy as np
    import pyarrow as pa
    from motorsports_data_notebook.tire_etl.aggregates import compute_lap_dynamics

    t = np.arange(0, 100, 1.0)
    speed = np.where(t < 40, 0.0, 80.0)
    speed[40:45] = 15.0  # rolling out of the pit lane
    lap = pa.table(
        {"t_lap_s": t, "speed_kmh": speed, "lat_g": np.zeros_like(t), "long_g": np.zeros_like(t)}
    )
    dyn = compute_lap_dynamics(lap)
    assert dyn["on_track_s"] == 99.0
    assert 59.0 <= dyn["moving_s"] <= 61.0
    assert dyn["speed_kmh_first"] == 0.0
