"""Tests for Open-Meteo client + caching."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest
import pyarrow.parquet as pq
import respx

from motorsports_data_notebook.tire_etl import weather


def _canned_payload() -> dict:
    return {
        "hourly": {
            "time": ["2026-04-04T00:00", "2026-04-04T01:00", "2026-04-04T02:00"],
            "temperature_2m": [10.0, 11.0, 12.0],
            "relative_humidity_2m": [60.0, 62.0, 65.0],
            "surface_pressure": [1013.0, 1013.5, 1014.0],
            "wind_speed_10m": [5.0, 6.0, 7.0],
            "wind_direction_10m": [90.0, 95.0, 100.0],
            "precipitation": [0.0, 0.0, 0.1],
            "cloud_cover": [20.0, 30.0, 40.0],
        }
    }


@respx.mock
def test_fetch_for_track_writes_cache(tmp_path: Path) -> None:
    route = respx.get(weather._API_URL).mock(
        return_value=httpx.Response(200, json=_canned_payload())
    )
    weather.fetch_for_track("tsukuba_2000", [date(2026, 4, 4)], dataset_root=tmp_path)
    assert route.call_count == 1

    out = tmp_path / "weather_hourly" / "tsukuba_2000" / "2026.parquet"
    assert out.exists()
    tbl = pq.read_table(out)
    assert len(tbl) == 3
    assert set(tbl.schema.names) >= {"ts_utc", "temperature_2m", "precipitation"}


@respx.mock
def test_fetch_skips_already_covered_dates(tmp_path: Path) -> None:
    route = respx.get(weather._API_URL).mock(
        return_value=httpx.Response(200, json=_canned_payload())
    )
    weather.fetch_for_track("tsukuba_2000", [date(2026, 4, 4)], dataset_root=tmp_path)
    weather.fetch_for_track("tsukuba_2000", [date(2026, 4, 4)], dataset_root=tmp_path)
    # Second call should hit cache, not the API.
    assert route.call_count == 1


@respx.mock
def test_unknown_track_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    respx.get(weather._API_URL).mock(return_value=httpx.Response(200, json=_canned_payload()))
    weather.fetch_for_track("nonexistent", [date(2026, 4, 4)], dataset_root=tmp_path)
    assert not (tmp_path / "weather_hourly" / "nonexistent").exists()
