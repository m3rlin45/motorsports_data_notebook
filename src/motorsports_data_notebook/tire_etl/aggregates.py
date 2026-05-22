"""Per-lap tire aggregates derived from a single-lap timeseries slice.

The timeseries pa.Table is the source of truth; this module computes cheap
summary statistics for the ``laps/`` fast-query layer. Statistics gracefully
degrade when channels are missing (NaN passthrough).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

CORNERS = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class CornerTireAggregates:
    press_start: float
    press_end: float
    press_min: float
    press_max: float
    press_mean: float
    press_rise_bar_per_min: float
    temp_start: float
    temp_end: float
    temp_min: float
    temp_max: float
    temp_mean: float
    surf_mean: float
    surf_min: float
    surf_max: float


def _safe_stats(arr: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return (start, end, min, max, mean) ignoring NaNs; all-NaN -> all-NaN."""
    if arr.size == 0 or np.all(np.isnan(arr)):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    # start = first finite; end = last finite
    first = float(finite[0])
    last = float(finite[-1])
    return (first, last, float(np.min(finite)), float(np.max(finite)), float(np.mean(finite)))


def _rise_rate_per_min(t_s: np.ndarray, values: np.ndarray) -> float:
    """Linear-fit slope of ``values`` over ``t_s`` seconds; returns units/min."""
    mask = ~np.isnan(values) & ~np.isnan(t_s)
    if mask.sum() < 2:
        return float("nan")
    t = t_s[mask]
    v = values[mask]
    if t.max() - t.min() < 1.0:  # too short to estimate
        return float("nan")
    slope_per_s = np.polyfit(t, v, 1)[0]
    return float(slope_per_s * 60.0)


def _col_or_nan(ts: pa.Table, name: str, n: int) -> np.ndarray:
    if name in ts.schema.names:
        arr = ts.column(name).to_numpy(zero_copy_only=False).astype(np.float64)
        return np.asarray(arr, dtype=np.float64)
    return np.full(n, np.nan, dtype=np.float64)


def compute_corner_aggregates(lap_ts: pa.Table, corner: str) -> CornerTireAggregates:
    """Compute per-corner tire aggregates from a single lap's timeseries rows.

    ``lap_ts`` is a wide timeseries table with columns like
    ``tpms_press_fl_kpa``, ``tpms_temp_fl_c``, ``surf_temp_fl_mean_c``, and
    ``t_lap_s``. Stale TPMS samples are masked at the session level (see
    :func:`mask_stale_session_prefix` in ``extract.py``) before this runs,
    so ``_safe_stats`` already ignores them via NaN.
    """
    n = len(lap_ts)
    t = _col_or_nan(lap_ts, "t_lap_s", n)
    press = _col_or_nan(lap_ts, f"tpms_press_{corner}_bar", n)
    # Note: rise rate is in bar/min (not kPa/min)
    temp = _col_or_nan(lap_ts, f"tpms_temp_{corner}_c", n)
    surf = _col_or_nan(lap_ts, f"surf_temp_{corner}_mean_c", n)

    p_start, p_end, p_min, p_max, p_mean = _safe_stats(press)
    tt_start, tt_end, tt_min, tt_max, tt_mean = _safe_stats(temp)
    s_start, s_end, s_min, s_max, s_mean = _safe_stats(surf)  # noqa: F841
    rise = _rise_rate_per_min(t, press)

    return CornerTireAggregates(
        press_start=p_start,
        press_end=p_end,
        press_min=p_min,
        press_max=p_max,
        press_mean=p_mean,
        press_rise_bar_per_min=rise,
        temp_start=tt_start,
        temp_end=tt_end,
        temp_min=tt_min,
        temp_max=tt_max,
        temp_mean=tt_mean,
        surf_mean=s_mean,
        surf_min=s_min,
        surf_max=s_max,
    )


def compute_lap_dynamics(lap_ts: pa.Table) -> dict[str, float]:
    """Compute lap-level dynamics summary (speed, G, heat proxy, etc.)."""
    n = len(lap_ts)
    t = _col_or_nan(lap_ts, "t_lap_s", n)
    speed = _col_or_nan(lap_ts, "speed_kmh", n)
    brake = _col_or_nan(lap_ts, "brake_bar", n)
    throttle = _col_or_nan(lap_ts, "throttle_pct", n)
    lat_g = _col_or_nan(lap_ts, "lat_g", n)
    long_g = _col_or_nan(lap_ts, "long_g", n)

    speed_valid = speed[~np.isnan(speed)]
    brake_valid = brake[~np.isnan(brake)]
    thr_valid = throttle[~np.isnan(throttle)]

    heat_proxy = float("nan")
    heat_proxy_corner: dict[str, float] = {c: float("nan") for c in CORNERS}
    if n >= 2 and not np.all(np.isnan(lat_g)) and not np.all(np.isnan(long_g)):
        lg = np.nan_to_num(lat_g, nan=0.0)
        lng = np.nan_to_num(long_g, nan=0.0)
        dt = np.diff(t, prepend=t[0] if t.size else 0.0)
        dt = np.where(np.isnan(dt), 0.0, dt)
        heat_proxy = float(np.sum((lg * lg + lng * lng) * dt))
        # Per-corner decomposition: positive lat_g ⇒ right turn ⇒ load on
        # LEFT corners; positive long_g ⇒ accel ⇒ rear; negative long_g ⇒
        # brake ⇒ front. Each corner sums only the G components that load
        # it, so a hard-braking lap inflates FL/FR but not RL/RR.
        lat_pos = np.where(lg > 0, lg, 0.0)
        lat_neg = np.where(lg < 0, -lg, 0.0)
        long_pos = np.where(lng > 0, lng, 0.0)
        long_neg = np.where(lng < 0, -lng, 0.0)
        per_corner = {
            "fl": lat_pos * lat_pos + long_neg * long_neg,  # right turn + brake
            "fr": lat_neg * lat_neg + long_neg * long_neg,  # left turn + brake
            "rl": lat_pos * lat_pos + long_pos * long_pos,  # right turn + accel
            "rr": lat_neg * lat_neg + long_pos * long_pos,  # left turn + accel
        }
        heat_proxy_corner = {c: float(np.sum(v * dt)) for c, v in per_corner.items()}

    long_g_min = (
        float(np.nanmin(long_g)) if long_g.size and not np.all(np.isnan(long_g)) else float("nan")
    )
    lat_g_abs_max = (
        float(np.nanmax(np.abs(lat_g)))
        if lat_g.size and not np.all(np.isnan(lat_g))
        else float("nan")
    )
    on_track_s = float("nan")
    if t.size:
        finite_t = t[~np.isnan(t)]
        if finite_t.size >= 2:
            on_track_s = float(finite_t.max() - finite_t.min())
    distance_m = _col_or_nan(lap_ts, "distance_m", n)
    dist_val = (
        float(np.nanmax(distance_m))
        if distance_m.size and not np.all(np.isnan(distance_m))
        else float("nan")
    )

    return {
        "speed_kmh_mean": float(np.mean(speed_valid)) if speed_valid.size else float("nan"),
        "speed_kmh_max": float(np.max(speed_valid)) if speed_valid.size else float("nan"),
        "brake_mean": float(np.mean(brake_valid)) if brake_valid.size else float("nan"),
        "brake_max": float(np.max(brake_valid)) if brake_valid.size else float("nan"),
        "throttle_mean": float(np.mean(thr_valid)) if thr_valid.size else float("nan"),
        "lat_g_peak": lat_g_abs_max,
        "long_g_peak_brake": long_g_min,  # negative value; more negative = harder braking
        "heat_proxy": heat_proxy,
        "heat_proxy_fl": heat_proxy_corner["fl"],
        "heat_proxy_fr": heat_proxy_corner["fr"],
        "heat_proxy_rl": heat_proxy_corner["rl"],
        "heat_proxy_rr": heat_proxy_corner["rr"],
        "on_track_s": on_track_s,
        "distance_m": dist_val,
    }
