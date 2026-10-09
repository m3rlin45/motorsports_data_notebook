from datetime import datetime, timezone

import pytest

from motorsports_data_notebook.tire_model import solar


def test_tokyo_summer_noon_elevation() -> None:
    # 2025-06-21 12:00 JST at Tokyo: the sun is ≈ 77.7° up (NOAA calculator).
    e = solar.solar_elevation_deg(datetime(2025, 6, 21, 3, 0, tzinfo=timezone.utc), 35.68, 139.69)
    assert e == pytest.approx(77.7, abs=0.6)


def test_winter_morning_is_low_and_night_is_below_horizon() -> None:
    morning = solar.solar_elevation_deg(
        datetime(2025, 12, 21, 23, 0, tzinfo=timezone.utc), 36.15, 139.92
    )
    assert 5.0 < morning < 20.0  # 08:00 JST, December, Tsukuba
    night = solar.solar_elevation_deg(
        datetime(2025, 12, 21, 14, 0, tzinfo=timezone.utc), 36.15, 139.92
    )
    assert night < 0
    assert (
        solar.sun_index(datetime(2025, 12, 21, 14, 0, tzinfo=timezone.utc), 36.15, 139.92, 0.0)
        == 0.0
    )


def test_cloud_transmission_shape() -> None:
    assert solar.cloud_transmission(0) == 1.0
    assert solar.cloud_transmission(100) == pytest.approx(0.25)
    assert solar.cloud_transmission(50) == pytest.approx(1 - 0.75 * 0.5**3.4)
    assert solar.cloud_transmission(None) == 1.0
