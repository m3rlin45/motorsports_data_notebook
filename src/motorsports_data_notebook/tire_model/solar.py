"""Solar geometry for the cooling sink: how much sun the asphalt sees.

The dataset's track temperature is a proxy (air + a flat clear-sky offset,
see :func:`energy_balance.t_road_proxy_c`); it knows nothing about the
sun's height. This module gives the clear-sky irradiance factor
``sin(elevation)`` for a session's start time and the track's coordinates,
attenuated by the hourly cloud cover with the Kasten–Czeplak law
``1 − 0.75·N^3.4`` (``N`` the cloud fraction), so a fitted coefficient
turns it into the sink's solar offset in kelvin.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np


def solar_elevation_deg(ts_utc: datetime, lat_deg: float, lon_deg: float) -> float:
    """Sun elevation above the horizon (degrees) at ``ts_utc`` for a place
    (NOAA's low-precision algorithm, good to ≈ 0.1°)."""
    t = ts_utc.astimezone(timezone.utc) if ts_utc.tzinfo else ts_utc.replace(tzinfo=timezone.utc)
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60.0 + t.second / 3600.0
    g = 2.0 * math.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(g)
        - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g)
        - 0.040849 * math.sin(2 * g)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(g)
        + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g)
        + 0.00148 * math.sin(3 * g)
    )
    tst = hour * 60.0 + eqtime + 4.0 * lon_deg  # true solar time, minutes
    ha = math.radians(tst / 4.0 - 180.0)
    lat = math.radians(lat_deg)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, cos_zen))))


def cloud_transmission(cloud_cover_pct: float | None) -> float:
    """Fraction of clear-sky global irradiance reaching the ground under
    ``cloud_cover_pct`` (Kasten & Czeplak 1980: ``1 − 0.75·N^3.4``);
    1 when the cover is unknown."""
    if cloud_cover_pct is None or not np.isfinite(cloud_cover_pct):
        return 1.0
    n = min(max(cloud_cover_pct / 100.0, 0.0), 1.0)
    return float(1.0 - 0.75 * n**3.4)


def sun_index(
    ts_utc: datetime, lat_deg: float, lon_deg: float, cloud_cover_pct: float | None
) -> float:
    """``sin(elevation)⁺ · transmission``: the irradiance on the asphalt as a
    fraction of the overhead clear-sky value (0 at night or fully overcast
    ≈ 0.25)."""
    elev = solar_elevation_deg(ts_utc, lat_deg, lon_deg)
    return max(math.sin(math.radians(elev)), 0.0) * cloud_transmission(cloud_cover_pct)
