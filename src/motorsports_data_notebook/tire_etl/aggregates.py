"""Per-lap tire aggregates derived from a single-lap timeseries slice.

The timeseries pa.Table is the source of truth; this module computes cheap
summary statistics for the ``laps/`` fast-query layer. Statistics gracefully
degrade when channels are missing (NaN passthrough).

Since dataset schema 7 a lap slice holds several raw timebases: every channel
column is an Arrow null outside the rows of its own timebase, so each
statistic reads a channel together with the ``t_lap_s`` of exactly its own
samples (:func:`channel_series`). A single-clock table where every column is
valid on every row (the pre-schema-7 layout, synthetic fixtures) is the
special case where all series share one clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

CORNERS = ("fl", "fr", "rl", "rr")

_EMPTY = np.zeros(0, dtype=np.float64)


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
    # Lap-relative time (s) of the first valid temperature / pressure sample.
    # On an out-lap this says when the TPMS woke up relative to the lap
    # start, so the pit-exit reading can be placed on the stint clock.
    temp_first_valid_s: float = float("nan")
    press_first_valid_s: float = float("nan")


def channel_series(
    ts: pa.Table, name: str, t_col: str = "t_lap_s"
) -> tuple[np.ndarray, np.ndarray]:
    """``(t, values)`` of ``name``'s own samples, as float64 arrays.

    The channel's own samples are the rows where the column is *valid*
    (not an Arrow null); NaN values inside its timebase are kept, so a
    sensor dropout stays a NaN sample rather than vanishing. Missing
    column or time column: two empty arrays.
    """
    if name not in ts.schema.names or t_col not in ts.schema.names:
        return _EMPTY, _EMPTY
    col = ts.column(name)
    if col.null_count:
        sub = ts.select([t_col, name]).filter(pc.is_valid(col))
        t = sub.column(t_col).to_numpy(zero_copy_only=False)
        v = sub.column(name).to_numpy(zero_copy_only=False)
    else:
        t = ts.column(t_col).to_numpy(zero_copy_only=False)
        v = col.to_numpy(zero_copy_only=False)
    return np.asarray(t, dtype=np.float64), np.asarray(v, dtype=np.float64)


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


def _first_valid_t(t: np.ndarray, values: np.ndarray) -> float:
    ok = ~np.isnan(values) & ~np.isnan(t)
    return float(t[ok][0]) if ok.any() else float("nan")


def _interp_onto(t_out: np.ndarray, t_in: np.ndarray, v_in: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``v_in(t_in)`` at ``t_out``; NaN outside the
    finite support and when fewer than two finite samples exist."""
    ok = np.isfinite(v_in) & np.isfinite(t_in)
    if ok.sum() < 2:
        return np.full(t_out.shape, np.nan)
    ti, vi = t_in[ok], v_in[ok]
    order = np.argsort(ti, kind="stable")
    ti, vi = ti[order], vi[order]
    out = np.asarray(np.interp(t_out, ti, vi), dtype=np.float64)
    out[(t_out < ti[0]) | (t_out > ti[-1])] = np.nan
    return out


def surface_zone_mean(ts: pa.Table, corner: str) -> tuple[np.ndarray, np.ndarray]:
    """``(t, mean)`` of the corner's IR zones across the zones.

    The eight zone channels of a corner may sit on two raw timebases
    (zones 1-4 and 5-8 on the AIM IR controller); the mean is taken on the
    clock of the zone with the most samples, the others interpolated onto
    it.
    """
    zones = [channel_series(ts, f"surf_temp_{corner}_ch{i}_c") for i in range(1, 9)]
    zones = [(t, v) for t, v in zones if t.size]
    if not zones:
        return _EMPTY, _EMPTY
    base_t, base_v = max(zones, key=lambda z: z[0].size)
    stacked = [base_v]
    for t, v in zones:
        if t is base_t:
            continue
        if t.shape == base_t.shape and np.array_equal(t, base_t):
            stacked.append(v)
        else:
            stacked.append(_interp_onto(base_t, t, v))
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
            mean = np.nanmean(np.vstack(stacked), axis=0)
    return base_t, mean


def compute_corner_aggregates(lap_ts: pa.Table, corner: str) -> CornerTireAggregates:
    """Compute per-corner tire aggregates from a single lap's timeseries rows.

    ``lap_ts`` is a wide timeseries table with columns like
    ``tpms_press_fl_bar``, ``tpms_temp_fl_c``, ``surf_temp_fl_ch1_c`` and
    ``t_lap_s``. The extractor runs this on the raw timeseries; the model
    runs it again on the masked one (``tire_model.cleaning``), so the stale
    TPMS wake-up samples are judged at training time, not here.
    """
    t_p, press = channel_series(lap_ts, f"tpms_press_{corner}_bar")
    # Note: rise rate is in bar/min (not kPa/min)
    t_t, temp = channel_series(lap_ts, f"tpms_temp_{corner}_c")
    _, surf = surface_zone_mean(lap_ts, corner)

    p_start, p_end, p_min, p_max, p_mean = _safe_stats(press)
    tt_start, tt_end, tt_min, tt_max, tt_mean = _safe_stats(temp)
    s_start, s_end, s_min, s_max, s_mean = _safe_stats(surf)  # noqa: F841
    rise = _rise_rate_per_min(t_p, press)

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
        temp_first_valid_s=_first_valid_t(t_t, temp),
        press_first_valid_s=_first_valid_t(t_p, press),
    )


def _dt_of(t: np.ndarray) -> np.ndarray:
    """Per-sample step of a time series (first step 0); NaN or negative steps
    (a recording gap written out of order) count as 0."""
    if t.size == 0:
        return _EMPTY
    dt = np.diff(t, prepend=t[0])
    return np.where(np.isnan(dt) | (dt < 0), 0.0, dt)


def _speed_kmh_series(lap_ts: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    """GPS speed in km/h on its own samples (stored as ``speed_ms``)."""
    t, v = channel_series(lap_ts, "speed_ms")
    return t, v * 3.6


def compute_lap_dynamics(lap_ts: pa.Table, kin: pa.Table | None = None) -> dict[str, float]:
    """Compute lap-level dynamics summary (speed, G, heat proxy, etc.).

    Speed, brake and throttle come from ``lap_ts``; the accelerations from
    ``kin`` when given (the session's validated kinematics on the chosen
    sensor's clock, see :mod:`.kinematics`), else from the ``lat_g`` /
    ``long_g`` columns of ``lap_ts``. Lateral and longitudinal g may sit on
    different clocks: the heat proxies are sums of per-axis integrals, so
    each axis is integrated on its own samples.
    """
    t_all = (
        np.asarray(lap_ts.column("t_lap_s").to_numpy(zero_copy_only=False), dtype=np.float64)
        if "t_lap_s" in lap_ts.schema.names
        else _EMPTY
    )
    t_speed, speed = _speed_kmh_series(lap_ts)
    _, brake = channel_series(lap_ts, "brake_bar")
    _, throttle = channel_series(lap_ts, "throttle_pct")
    g_src = kin if kin is not None else lap_ts
    t_lat, lat_g = channel_series(g_src, "lat_g")
    t_long, long_g = channel_series(g_src, "long_g")

    speed_valid = speed[~np.isnan(speed)]
    brake_valid = brake[~np.isnan(brake)]
    thr_valid = throttle[~np.isnan(throttle)]

    heat_proxy = float("nan")
    heat_proxy_corner: dict[str, float] = {c: float("nan") for c in CORNERS}
    has_lat = lat_g.size >= 2 and not np.all(np.isnan(lat_g))
    has_long = long_g.size >= 2 and not np.all(np.isnan(long_g))
    if has_lat and has_long:
        lg = np.nan_to_num(lat_g, nan=0.0)
        lng = np.nan_to_num(long_g, nan=0.0)
        dt_lat = _dt_of(t_lat)
        dt_long = _dt_of(t_long)
        heat_proxy = float(np.sum(lg * lg * dt_lat) + np.sum(lng * lng * dt_long))
        # Per-corner decomposition: positive lat_g ⇒ right turn ⇒ load on
        # LEFT corners; positive long_g ⇒ accel ⇒ rear; negative long_g ⇒
        # brake ⇒ front. Each corner sums only the G components that load
        # it, so a hard-braking lap inflates FL/FR but not RL/RR.
        lat_pos = float(np.sum(np.where(lg > 0, lg, 0.0) ** 2 * dt_lat))
        lat_neg = float(np.sum(np.where(lg < 0, -lg, 0.0) ** 2 * dt_lat))
        long_pos = float(np.sum(np.where(lng > 0, lng, 0.0) ** 2 * dt_long))
        long_neg = float(np.sum(np.where(lng < 0, -lng, 0.0) ** 2 * dt_long))
        heat_proxy_corner = {
            "fl": lat_pos + long_neg,  # right turn + brake
            "fr": lat_neg + long_neg,  # left turn + brake
            "rl": lat_pos + long_pos,  # right turn + accel
            "rr": lat_neg + long_pos,  # left turn + accel
        }

    long_g_min = float(np.nanmin(long_g)) if has_long else float("nan")
    lat_g_abs_max = float(np.nanmax(np.abs(lat_g))) if has_lat else float("nan")
    on_track_s = float("nan")
    if t_all.size:
        finite_t = t_all[~np.isnan(t_all)]
        if finite_t.size >= 2:
            on_track_s = float(finite_t.max() - finite_t.min())
    # Time actually rolling (> 5 km/h). Equals on_track_s for a flying lap;
    # for an out-lap it excludes the grid / pit-lane wait before the car
    # moves, which must not count as warm-up time. speed_kmh_first tells
    # whether the lap started from standstill (a pit exit) or mid-track.
    moving_s = float("nan")
    speed_kmh_first = float("nan")
    if t_speed.size >= 2 and not np.all(np.isnan(speed)):
        dt_m = _dt_of(t_speed)
        moving_s = float(np.sum(np.where(np.nan_to_num(speed, nan=0.0) > 5.0, dt_m, 0.0)))
        speed_kmh_first = float(speed_valid[0]) if speed_valid.size else float("nan")
    _, distance_m = channel_series(lap_ts, "distance_m")
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
        "moving_s": moving_s,
        "speed_kmh_first": speed_kmh_first,
        "distance_m": dist_val,
    }
