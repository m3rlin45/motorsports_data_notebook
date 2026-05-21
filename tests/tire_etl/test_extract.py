"""Unit tests for helpers inside :mod:`tire_etl.extract`.

Full end-to-end extraction requires libxrk + a real AIM file; those are
exercised by an opt-in integration test. This file covers the pure
PyArrow transforms that are straightforward to unit-test.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from datetime import datetime, timezone

from motorsports_data_notebook.tire_etl.extract import (
    _build_empty_session_row,
    _extract_session_datetime_utc,
    _rename_to_canonical,
)
from motorsports_data_notebook.tire_etl.discovery import SessionCandidate

from datetime import date as _date
from pathlib import Path


class _FakeLog:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata


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


def test_session_time_from_aim_log_date_and_log_time() -> None:
    """AIM's `Log Date` (MM/DD/YYYY) + `Log Time` must be parsed and converted
    from track-local tz to UTC. Real AIM fleet data uses these keys; the old
    fallback (midnight local) was dropping within-day resolution and causing
    weather joins to land on the wrong hour (e.g. missing rain)."""
    log = _FakeLog({"Log Date": "04/04/2026", "Log Time": "12:57:49"})
    dt = _extract_session_datetime_utc(log, "2026-04-04", "tsukuba_2000")
    # 12:57:49 JST = 03:57:49 UTC
    assert dt == datetime(2026, 4, 4, 3, 57, 49, tzinfo=timezone.utc)


def test_session_time_from_aim_log_date_unambiguous_month_day() -> None:
    """MM/DD vs DD/MM was resolved empirically against a 04/17 file — 04/17 is
    only valid as MM/DD, so we must parse that format."""
    log = _FakeLog({"Log Date": "04/17/2026", "Log Time": "12:23:55"})
    dt = _extract_session_datetime_utc(log, "2026-04-17", "tsukuba_2000")
    assert dt == datetime(2026, 4, 17, 3, 23, 55, tzinfo=timezone.utc)


def test_session_time_falls_back_to_midnight_local_when_metadata_missing() -> None:
    log = _FakeLog({})
    dt = _extract_session_datetime_utc(log, "2026-04-04", "tsukuba_2000")
    # Midnight JST = 15:00 UTC previous day
    assert dt == datetime(2026, 4, 3, 15, 0, 0, tzinfo=timezone.utc)


def test_session_row_error_msg_is_always_string_typed() -> None:
    """Regression: error_msg must be pa.string() even when value is None.

    Otherwise PyArrow infers pa.null() for all-None monthly partitions, and
    cross-month concat/DuckDB reads fail with a schema-mismatch error.
    """
    cand = SessionCandidate(
        path=Path("/tmp/foo.xrk"),
        date=_date(2026, 4, 1),
        driver="CMD",
        car="Inferno 86",
        track_raw="Tsukuba",
        track_canonical="tsukuba_2000",
        session_type="testing",
        run_num=1,
    )
    row = _build_empty_session_row(
        session_id="abc123",
        path=Path("/tmp/foo.xrk"),
        mtime_ns=0,
        file_size=0,
        cand=cand,
        extractor_version="0.3.0",
    )
    assert row.schema.field("error_msg").type == pa.string()
