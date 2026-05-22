"""Unit tests for helpers inside :mod:`tire_etl.extract`.

Full end-to-end extraction requires libxrk + a real AIM file; those are
exercised by an opt-in integration test. This file covers the pure
PyArrow transforms that are straightforward to unit-test.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from motorsports_data_notebook.tire_etl.extract import _rename_to_canonical


def test_rename_converts_aim_names_to_canonical() -> None:
    profile = {
        "tpms_press_fl": "TPMS_Press_LF",
        "tpms_temp_fl": "TPMS_Temp_LF",
        "throttle": "PPS",
        "brake": "BrakePress",
        "gps_speed": "GPS Speed",
        "lateral_g": "LateralAcc",
        "inline_g": "InlineAcc",
    }
    ts = pa.table(
        {
            "TPMS_Press_LF": pa.array([1.8, 1.85, 1.9]),
            "TPMS_Temp_LF": pa.array([30.0, 40.0, 50.0]),
            "PPS": pa.array([0.0, 50.0, 100.0]),
            "BrakePress": pa.array([0.0, 10.0, 5.0]),
            "GPS Speed": pa.array([10.0, 20.0, 30.0]),
            "LateralAcc": pa.array([0.1, 0.2, 0.3]),
            "InlineAcc": pa.array([-0.5, -0.4, 0.1]),
        }
    )
    out = _rename_to_canonical(ts, profile_names=profile)
    names = set(out.schema.names)
    assert "tpms_press_fl_bar" in names
    assert "tpms_temp_fl_c" in names
    assert "throttle_pct" in names
    assert "brake_bar" in names
    assert "speed_ms" in names
    assert "lat_g" in names
    assert "long_g" in names
    # speed_kmh is derived from speed_ms
    assert "speed_kmh" in names
    speed_kmh = out.column("speed_kmh").to_pylist()
    assert speed_kmh == pytest.approx([36.0, 72.0, 108.0])


def test_tpms_pressure_in_kpa_is_converted_to_bar() -> None:
    """Catch the case where a logger reports TPMS pressure in kPa (values > 10)."""
    profile = {"tpms_press_fl": "TPMS_Press_LF"}
    ts = pa.table({"TPMS_Press_LF": pa.array([180.0, 200.0, 220.0])})
    out = _rename_to_canonical(ts, profile_names=profile)
    vals = out.column("tpms_press_fl_bar").to_pylist()
    assert vals == pytest.approx([1.80, 2.00, 2.20])


def test_tpms_pressure_in_bar_is_left_as_is() -> None:
    """Values already in the bar magnitude range (<10) should not be scaled."""
    profile = {"tpms_press_fl": "TPMS_Press_LF"}
    ts = pa.table({"TPMS_Press_LF": pa.array([1.80, 2.00, 2.20])})
    out = _rename_to_canonical(ts, profile_names=profile)
    vals = out.column("tpms_press_fl_bar").to_pylist()
    assert vals == pytest.approx([1.80, 2.00, 2.20])


def test_surface_temp_collapses_to_mean() -> None:
    """8 per-sensor IR channels should produce a corner mean column."""
    profile = {f"tire_temp_fl_{i}": f"FL_Ch{i}" for i in range(1, 9)}
    cols = {f"FL_Ch{i}": pa.array([10.0 * i, 10.0 * i + 5.0]) for i in range(1, 9)}
    ts = pa.table(cols)
    out = _rename_to_canonical(ts, profile_names=profile)
    assert "surf_temp_fl_mean_c" in out.schema.names
    mean = out.column("surf_temp_fl_mean_c").to_numpy()
    # (10+20+...+80)/8 = 45 ; second row is +5
    np.testing.assert_allclose(mean, [45.0, 50.0])


def test_rename_noop_when_profile_has_no_matches() -> None:
    """Channel names that don't appear in the table should be silently skipped."""
    profile = {"tpms_press_fl": "Missing_Channel"}
    ts = pa.table({"something_else": pa.array([1.0, 2.0])})
    out = _rename_to_canonical(ts, profile_names=profile)
    assert out.schema.names == ["something_else"]
