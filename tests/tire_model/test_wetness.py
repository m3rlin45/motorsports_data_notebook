"""Surface water balance: the expected track wetness at roll-out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model import wetness as wx


def _hours(start: str, n: int, **cols) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    base = {
        "ts_utc": [t.strftime("%Y-%m-%dT%H:%M") for t in ts],
        "precipitation": np.zeros(n),
        "temperature_2m": np.full(n, 20.0),
        "relative_humidity_2m": np.full(n, 70.0),
        "wind_speed_10m": np.full(n, 5.0),
        "cloud_cover": np.full(n, 50.0),
    }
    for k, v in cols.items():
        base[k] = np.asarray(v, dtype=float)
    return pd.DataFrame(base)


def test_evaporation_rises_with_heat_sun_and_wind_and_falls_with_humidity() -> None:
    cool_cloudy = wx.evaporation_mm_per_h(10.0, 85.0, 5.0, 100.0)
    hot_sunny = wx.evaporation_mm_per_h(38.0, 36.0, 10.0, 0.0)
    assert hot_sunny > 10 * cool_cloudy
    assert wx.evaporation_mm_per_h(20.0, 60.0, 20.0, 50.0) > wx.evaporation_mm_per_h(
        20.0, 60.0, 0.0, 50.0
    )
    assert wx.evaporation_mm_per_h(20.0, 100.0, 0.0, 100.0) == pytest.approx(0.0)
    # A summer afternoon dries a 0.6 mm shower well inside an hour.
    assert hot_sunny > 0.6


def test_surface_water_caps_at_storage_and_forgets_rain_once_dry() -> None:
    w = _hours(
        "2025-07-01T00:00",
        8,
        precipitation=[20.0, 20.0, 0, 0, 0, 0, 0, 0],
        temperature_2m=[12.0] * 8,
        relative_humidity_2m=[95.0] * 8,
        cloud_cover=[100.0] * 8,
    )
    d = wx.surface_water_series(w)
    assert d.iloc[0] == pytest.approx(wx.SURFACE_STORAGE_MM)  # 20 mm does not pile up
    assert d.iloc[1] == pytest.approx(wx.SURFACE_STORAGE_MM)
    assert d.is_monotonic_decreasing
    assert d.iloc[-1] < d.iloc[1]
    # Dry it out completely, then a later state does not depend on the deluge.
    w2 = _hours(
        "2025-07-01T00:00",
        8,
        precipitation=[20.0, 0, 0, 0, 0, 0, 0, 0.3],
        temperature_2m=[35.0] * 8,
        relative_humidity_2m=[30.0] * 8,
        cloud_cover=[0.0] * 8,
    )
    d2 = wx.surface_water_series(w2)
    assert d2.iloc[3] == pytest.approx(0.0)
    assert d2.iloc[-1] == pytest.approx(0.0)  # 0.3 mm evaporates within the hour at 35 °C


def test_tsukuba_2025_07_26_scenario_is_dry() -> None:
    """0.6 mm in the hour BEFORE a 14:08 JST start on a 38 °C, clear day: the
    old start-hour rule said damp; the surface is dry by roll-out."""
    w = _hours(
        "2025-07-26T00:00",
        10,
        precipitation=[0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0],
        temperature_2m=[32.5, 34.5, 36.0, 37.1, 38.3, 35.9, 35.7, 35.5, 34.2, 33.0],
        relative_humidity_2m=[58, 50, 44, 39, 36, 43, 42, 43, 46, 50],
        wind_speed_10m=[10.0] * 10,
        cloud_cover=[0, 0, 6, 12, 13, 72, 54, 34, 21, 24],
    )
    d = wx.surface_water_series(w)
    f = wx.session_wetness(d, pd.Timestamp("2025-07-26T05:08:14Z"), duration_s=55 * 60)
    assert f["depth_run_max_mm"] == pytest.approx(0.0, abs=1e-9)
    assert wx.classify_wetness(f["depth_run_max_mm"]) == "dry"


def test_cool_steady_rain_is_wet_and_drizzle_is_damp() -> None:
    rain = _hours(
        "2026-02-25T00:00",
        6,
        precipitation=[0, 0, 1.5, 1.5, 1.5, 0],
        temperature_2m=[8.0] * 6,
        relative_humidity_2m=[95.0] * 6,
        cloud_cover=[100.0] * 6,
    )
    d = wx.surface_water_series(rain)
    f = wx.session_wetness(d, pd.Timestamp("2026-02-25T03:30:00Z"), duration_s=40 * 60)
    assert wx.classify_wetness(f["depth_run_max_mm"]) == "wet"
    drizzle = _hours(
        "2026-02-25T00:00",
        6,
        precipitation=[0, 0, 0.15, 0.15, 0, 0],
        temperature_2m=[12.0] * 6,
        relative_humidity_2m=[90.0] * 6,
        cloud_cover=[100.0] * 6,
    )
    d2 = wx.surface_water_series(drizzle)
    f2 = wx.session_wetness(d2, pd.Timestamp("2026-02-25T03:30:00Z"), duration_s=40 * 60)
    assert wx.classify_wetness(f2["depth_run_max_mm"]) == "damp"


def test_session_outside_series_is_unknown() -> None:
    d = wx.surface_water_series(_hours("2026-01-01T00:00", 3))
    f = wx.session_wetness(d, pd.Timestamp("2026-03-01T00:00:00Z"), duration_s=1800)
    assert np.isnan(f["depth_run_max_mm"])
    assert wx.classify_wetness(f["depth_run_max_mm"]) == "unknown"
    assert wx.classify_wetness(None) == "unknown"
