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


# ------------------------------------------------- the owner's word (#395)


def _drizzle_day() -> pd.DataFrame:
    """The drizzle of the test above: a damp film at 03:30 UTC."""
    return _hours(
        "2026-02-25T00:00",
        6,
        precipitation=[0, 0, 0.15, 0.15, 0, 0],
        temperature_2m=[12.0] * 6,
        relative_humidity_2m=[90.0] * 6,
        cloud_cover=[100.0] * 6,
    )


def _write_overrides(root, body: str) -> None:
    (root / wx.CONDITION_OVERRIDES_FILE).write_text(body, encoding="utf-8")


def test_overrides_missing_file_is_empty(tmp_path) -> None:
    assert wx.load_condition_overrides(tmp_path) == {}


def test_overrides_are_keyed_by_track_and_day(tmp_path) -> None:
    _write_overrides(
        tmp_path,
        "days:\n"
        "  - track: fuji\n    date: 2026-11-25\n    condition: Dry\n    evidence: the owner\n"
        "  - track: tsukuba_2000\n    date: '2026-02-25'\n    condition: wet\n    evidence: x\n",
    )
    assert wx.load_condition_overrides(tmp_path) == {
        ("fuji", "2026-11-25"): "dry",
        ("tsukuba_2000", "2026-02-25"): "wet",
    }


@pytest.mark.parametrize(
    "entry",
    [
        "  - track: fuji\n    date: 2026-11-25\n    condition: drizzly\n    evidence: x\n",
        "  - track: fuji\n    date: 2026-11-25\n    condition: dry\n",
        "  - date: 2026-11-25\n    condition: dry\n    evidence: x\n",
        "  - track: fuji\n    date: someday\n    condition: dry\n    evidence: x\n",
        "  - track: fuji\n    condition: dry\n    evidence: x\n",
        "  - track: fuji\n    date: 2026-11-25\n    condition: dry\n    evidence: x\n" * 2,
    ],
)
def test_a_malformed_override_raises(tmp_path, entry: str) -> None:
    _write_overrides(tmp_path, "days:\n" + entry)
    with pytest.raises(ValueError):
        wx.load_condition_overrides(tmp_path)


def test_apply_overrides_changes_only_the_listed_days() -> None:
    import datetime as dt

    frame = pd.DataFrame(
        {
            "session_id": ["a", "b", "c", "d"],
            "track_canonical": ["fuji", "fuji", "suzuka", "fuji"],
            "date": [dt.date(2026, 11, 25), dt.date(2026, 11, 25), dt.date(2026, 11, 25), None],
            "condition": ["damp", "dry", "damp", "unknown"],
        }
    )
    out = wx.apply_condition_overrides(frame, {("fuji", "2026-11-25"): "dry"})
    assert out["condition"].tolist() == ["dry", "dry", "damp", "unknown"]
    assert frame["condition"].tolist() == ["damp", "dry", "damp", "unknown"]
    assert wx.apply_condition_overrides(frame, {}) is frame
    assert wx.apply_condition_overrides(frame, None) is frame


def test_the_committed_overrides_load() -> None:
    from motorsports_data_notebook.tire_etl.paths import default_dataset_root

    overrides = wx.load_condition_overrides(default_dataset_root())
    assert overrides.get(("fuji", "2026-11-25")) == "dry"
    assert set(overrides.values()) <= set(wx.CONDITION_LABELS)


def _one_session_laps(track: str = "fuji") -> pd.DataFrame:
    import datetime as dt

    return pd.DataFrame(
        {
            "session_id": ["s1", "s1"],
            "lap_num": [1, 2],
            "track_canonical": [track, track],
            "session_start_utc": [pd.Timestamp("2026-02-25T03:30:00Z")] * 2,
            "date": [dt.date(2026, 2, 25)] * 2,
            "on_track_s": [1200.0, 1200.0],
        }
    )


def _weather_for(track: str = "fuji") -> pd.DataFrame:
    w = _drizzle_day()
    w["track_canonical"] = track
    return w


def test_fit_label_takes_the_override() -> None:
    from motorsports_data_notebook.tire_model.warmup_table import _attach_weather

    laps, weather = _one_session_laps(), _weather_for()
    assert set(_attach_weather(laps.copy(), weather, None)["condition"]) == {"damp"}
    out = _attach_weather(laps.copy(), weather, {("fuji", "2026-02-25"): "dry"})
    assert set(out["condition"]) == {"dry"}
    # The film depths stay as computed.
    assert (out["track_wetness_run_max_mm"] >= wx.DAMP_FILM_MM).all()
    # Another track's override leaves it alone.
    other = _attach_weather(laps.copy(), weather, {("suzuka", "2026-02-25"): "dry"})
    assert set(other["condition"]) == {"damp"}


def test_lapsim_label_takes_the_override(tmp_path) -> None:
    from motorsports_data_notebook.tire_etl.paths import weather_dir
    from motorsports_data_notebook.tire_model.lapsim import geometry as geo

    wdir = weather_dir(tmp_path) / "fuji"
    wdir.mkdir(parents=True)
    _drizzle_day().to_parquet(wdir / "2026.parquet")
    laps = _one_session_laps()
    sessions = laps.drop_duplicates("session_id")[
        ["session_id", "track_canonical", "session_start_utc", "date"]
    ]
    assert geo.session_conditions(tmp_path, sessions, laps).to_dict() == {"s1": "damp"}
    sig = geo._index_signature(tmp_path)
    _write_overrides(
        tmp_path,
        "days:\n  - track: fuji\n    date: 2026-02-25\n    condition: dry\n    evidence: x\n",
    )
    assert geo.session_conditions(tmp_path, sessions, laps).to_dict() == {"s1": "dry"}
    # The cached lap index (and the tables keyed on it) rekey on the file.
    assert geo._index_signature(tmp_path) != sig
