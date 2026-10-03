"""Expected track wetness from the hourly weather history.

The track condition used to be read off the Open-Meteo precipitation of the
hour containing the session start. That is wrong twice over: Open-Meteo's
hourly precipitation is the sum of the *preceding* hour, so a shower that
ended before roll-out was attributed to the run, and a trace of rain in the
grid cell says nothing about whether the asphalt is wet when you drive on it
(0.6 mm an hour before a 38 °C start is gone in twenty minutes; the same
0.6 mm at 10 °C under cloud lingers all afternoon).

What the model wants is the water on the surface at roll-out. We integrate a
surface water balance over the hourly series::

    d(t + 1h) = clamp(d(t) + P − E(T_air, RH, wind, cloud), 0, SURFACE_STORAGE_MM)

``P`` is the hour's precipitation (mm), ``E`` a bulk evaporation rate (mm/h)
driven by the vapour-pressure deficit between the sun-warmed surface and the
air, with a wind term, and the clamp encodes two facts: a track holds about a
millimetre of water before the rest runs off, and once dry it has forgotten
everything that fell before. From ``d(t)`` a session gets its film depth at
roll-out and its maximum over the run, and the three categories the rest of
the model understands:

    dry   : no film during the run
    damp  : partial film (≥ DAMP_FILM_MM)
    wet   : film at or near saturation (≥ WET_FILM_MM)

The evaporation coefficient and the storage depth are physical priors, not
fits; the sensitivity to the coefficient is documented in docs/tire_model.md.
All functions here are pure.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SURFACE_STORAGE_MM = 1.0  # max water film the asphalt holds; excess runs off
EVAP_COEFF_MM_PER_H_KPA = 0.2  # bulk evaporation per kPa of vapour-pressure deficit, no wind
WIND_FACTOR_PER_M_S = 0.54  # Penman-style wind function: E ∝ (1 + 0.54·u[m/s])
SURFACE_SUN_EXCESS_C = 10.0  # surface runs this much above air in full sun (same as T_road proxy)
DAMP_FILM_MM = 0.05
WET_FILM_MM = 0.5
RUN_MARGIN_S = 600.0  # hours overlapping [start, end + margin] count as "during the run"


def saturation_vapour_pressure_kpa(t_c: float) -> float:
    """Tetens formula over water."""
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


def evaporation_mm_per_h(
    t_air_c: float,
    rh_pct: float,
    wind_kmh: float,
    cloud_cover_pct: float | None,
    *,
    k_e: float = EVAP_COEFF_MM_PER_H_KPA,
    sun_excess_c: float = SURFACE_SUN_EXCESS_C,
) -> float:
    """Bulk evaporation from a wet track surface, mm per hour.

    Surface temperature is the air temperature plus a sun term that cloud
    cover attenuates (the same proxy the thermal model uses for T_road);
    the driving force is the deficit between saturation at the surface and
    the actual vapour pressure of the air.
    """
    cloud = (
        100.0 if cloud_cover_pct is None or not np.isfinite(cloud_cover_pct) else cloud_cover_pct
    )
    cloud = min(100.0, max(0.0, cloud))
    rh = min(100.0, max(0.0, rh_pct if np.isfinite(rh_pct) else 70.0))
    wind_ms = max(0.0, wind_kmh if np.isfinite(wind_kmh) else 0.0) / 3.6
    t_surface = t_air_c + sun_excess_c * (1.0 - cloud / 100.0)
    vpd = max(
        saturation_vapour_pressure_kpa(t_surface)
        - rh / 100.0 * saturation_vapour_pressure_kpa(t_air_c),
        0.0,
    )
    return k_e * vpd * (1.0 + WIND_FACTOR_PER_M_S * wind_ms)


def surface_water_series(
    weather: pd.DataFrame,
    *,
    storage_mm: float = SURFACE_STORAGE_MM,
    k_e: float = EVAP_COEFF_MM_PER_H_KPA,
) -> pd.Series:
    """Water film depth (mm) at the END of each hourly step, indexed by the
    step's timestamp (UTC). ``weather`` needs ``ts_utc`` plus
    ``precipitation``, ``temperature_2m``, ``relative_humidity_2m``,
    ``wind_speed_10m``, ``cloud_cover`` — the hourly table the ETL caches.
    Precipitation at ``ts`` is the preceding hour's sum (Open-Meteo
    convention), so the value at ``ts`` is the state just after that hour.
    Missing hours carry the state forward unchanged.
    """
    if weather.empty:
        return pd.Series(dtype=float)
    w = weather.copy()
    w["ts"] = pd.to_datetime(w["ts_utc"], format="%Y-%m-%dT%H:%M", utc=True)
    w = w.sort_values("ts").drop_duplicates("ts")
    ts = list(w["ts"])
    precip = w["precipitation"].to_numpy(dtype=float)
    t_air = w["temperature_2m"].to_numpy(dtype=float)
    rh = w["relative_humidity_2m"].to_numpy(dtype=float)
    wind = w["wind_speed_10m"].to_numpy(dtype=float)
    cloud = w["cloud_cover"].to_numpy(dtype=float)
    depth = 0.0
    depths = np.empty(len(ts))
    for i in range(len(ts)):
        p = float(precip[i]) if np.isfinite(precip[i]) else 0.0
        e = evaporation_mm_per_h(
            float(t_air[i]),
            float(rh[i]),
            float(wind[i]),
            float(cloud[i]) if np.isfinite(cloud[i]) else None,
            k_e=k_e,
        )
        depth = min(max(depth + p - e, 0.0), storage_mm)
        depths[i] = depth
    return pd.Series(depths, index=pd.DatetimeIndex(ts), dtype=float)


def session_wetness(
    depth_series: pd.Series,
    start_utc: pd.Timestamp,
    duration_s: float,
    *,
    margin_s: float = RUN_MARGIN_S,
) -> dict[str, float]:
    """Film depth at roll-out (linear within the hour) and its maximum over
    the hours that overlap the run. NaN when the series does not cover the
    session."""
    if depth_series.empty:
        return {"depth_start_mm": float("nan"), "depth_run_max_mm": float("nan")}
    start = pd.Timestamp(start_utc)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    end = start + pd.Timedelta(seconds=float(duration_s) + margin_s)
    h0 = start.floor("h")
    h1 = h0 + pd.Timedelta(hours=1)
    if h0 not in depth_series.index or h1 not in depth_series.index:
        return {"depth_start_mm": float("nan"), "depth_run_max_mm": float("nan")}
    frac = (start - h0).total_seconds() / 3600.0
    d0, d1 = float(depth_series[h0]), float(depth_series[h1])
    depth_start = d0 + (d1 - d0) * frac
    run_hours = depth_series.loc[h1 : end.ceil("h")]
    depth_run_max = max(depth_start, float(run_hours.max()) if len(run_hours) else depth_start)
    return {"depth_start_mm": depth_start, "depth_run_max_mm": depth_run_max}


def classify_wetness(depth_run_max_mm: float | None) -> str:
    """Map the run's maximum film depth to the model's condition category."""
    if depth_run_max_mm is None:
        return "unknown"
    try:
        if not math.isfinite(depth_run_max_mm):
            return "unknown"
    except TypeError:
        return "unknown"
    if depth_run_max_mm >= WET_FILM_MM:
        return "wet"
    if depth_run_max_mm >= DAMP_FILM_MM:
        return "damp"
    return "dry"
