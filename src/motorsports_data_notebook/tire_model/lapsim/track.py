"""The track profile (lap simulation segment B2, issue #204).

A track is the driven line's curvature against distance, ``κ(s)``, built
from the build laps of a layout (:func:`~.geometry.clean_lap_index`):

1. The fastest build lap, resampled every ``ds_m``, is the first reference
   line.
2. Every lap is projected onto it. A sample is matched only to the piece of
   the line within :data:`PROJECT_WINDOW_M` of where the lap's own distance
   says it should be, so the two branches of a crossover (Suzuka) are never
   confused.
3. The per-bin median of the laps' lateral offsets, smoothed, moves the
   reference onto the median line, which is closed, and the laps are
   projected again; then once more on the closed line.
4. Per bin, the **median of the laps' Doppler course**; ``κ = dψ/ds`` over
   :data:`KAPPA_WINDOW_M` on the line's own arc length. Positive is a right
   turn, the convention of the logged lateral g and of
   ``heat_input.share_heat_rate``.
5. The stored line is the integral of ``(sin ψ, cos ψ)``, its closing gap
   spread over the lap, laid on the median XY line. Nothing is
   differentiated from it; it serves :func:`project_lap` and plots.
6. Altitude: each session's offset removed, per-bin median, grade over
   :data:`GRADE_WINDOW_M`.

The median of the course is at the accelerometer's own day-to-day floor
and keeps the lap's total turning exactly; a median of XY switches between
laps and is rougher than any one of them, and a median of per-lap κ loses
1 % of the turning (measured in the design review of #204).

Beside the profile the build keeps, per lap and bin, the course, speed,
offset, altitude and time, and the accelerometer's lateral g where the lap
is IMU-sourced (:class:`TrackBuild`): the gates and the holdouts work on
those.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ...tire_etl.kinematics import G_MS2, load_kinematics
from . import _kernel
from . import geometry as geo

logger = logging.getLogger(__name__)

DS_M = 2.0
PROJECT_WINDOW_M = 80.0
#: Whether the search for the nearest vertex runs compiled (numba); the
#: numpy search it replaces gives the same vertex and runs without it.
USE_COMPILED = _kernel.HAVE_NUMBA
OFFSET_SMOOTH_M = 20.0
KAPPA_WINDOW_M = 10.0
GRADE_WINDOW_M = 50.0
ALT_SMOOTH_M = 20.0
#: Accelerometer lateral g is smoothed over this before it is put on bins.
INERTIAL_SMOOTH_S = 0.2
KAPPA_METHOD = "doppler_course_median"
#: A bin needs this many laps for its median to be used.
MIN_BIN_LAPS = 3
#: Corner table: a corner is a stretch turning one way with |κ| above the
#: first value, kept when its apex reaches the second and it turns by at
#: least the third (degrees).
CORNER_EDGE_KAPPA = 0.002
CORNER_APEX_KAPPA = 0.004
CORNER_MIN_ANGLE_DEG = 10.0
IMU_SOURCES = ("imu", "imu_swapped")
#: Build populations: basis name -> the lap-index column that selects it.
BASES = {"dry": "in_build", "wet": "in_build_wet"}
ACCEL_DELAY_RANGE_S = (-0.1, 0.3)
#: Fixed-place GPS errors: a stretch this long whose lateral g differs from
#: the inertial profile by more than the threshold is filled from it.
FILL_THRESHOLD_G = 0.25
FILL_STRETCH_M = 20.0


# ------------------------------------------------------------------ the laps


@dataclass
class LapRecord:
    """One lap with what a build needs."""

    session_id: str
    lap_num: int
    car: str
    date: str
    lap_time_s: float
    trace: geo.LapTrace
    #: Accelerometer lateral (g, right-positive, divided by the session's
    #: audit slope so it is in the ground frame's scale) on ``trace.t``;
    #: None unless the whole lap is IMU-sourced.
    lat_g: np.ndarray | None = None
    #: The session's accelerometer delay behind GPS position (s), removed
    #: from ``lat_g``.
    accel_delay_s: float = float("nan")

    @property
    def key(self) -> tuple[str, int]:
        return (self.session_id, self.lap_num)


def _session_slopes(sessions: pd.DataFrame) -> dict[str, float]:
    """``session_id -> lateral slope of the chosen accelerometer`` against
    the extractor's position reference (the audit's number)."""
    out: dict[str, float] = {}
    for row in sessions.itertuples(index=False):
        src = getattr(row, "accel_source", None)
        col = {"imu": "kin_imu_lat_slope", "imu_swapped": "kin_imu_swapped_lat_slope"}.get(str(src))
        val = getattr(row, col, None) if col else None
        if val is not None and np.isfinite(val) and val > 0:
            out[str(row.session_id)] = float(val)
    return out


def _session_inertial(root: Path, session_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t, lat_g smoothed, is_imu)`` of the session's validated kinematics."""
    kin = load_kinematics(root, session_id)
    if kin is None or "lat_source" not in kin.schema.names:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
    t = np.asarray(kin.column("t_session_s").to_numpy(zero_copy_only=False), float)
    lat = np.asarray(kin.column("lat_g").to_numpy(zero_copy_only=False), float)
    src = np.asarray(kin.column("lat_source").to_numpy(zero_copy_only=False), dtype=object)
    order = np.argsort(t, kind="stable")
    t, lat, src = t[order], lat[order], src[order]
    smooth = geo.nan_window_mean(lat, t, INERTIAL_SMOOTH_S / 2.0)
    return t, smooth, np.isin(src, IMU_SOURCES)


def accel_delay(
    geom: geo.SessionGeometry, t_acc: np.ndarray, lat_acc: np.ndarray, is_imu: np.ndarray
) -> float:
    """How far the accelerometer's lateral g trails GPS position (s).

    The logger's inertial channels reach the file later than the fix of
    the same instant (about 100 ms on the SW4, 30–45 ms on the Inferno
    86's loggers, 10 ms on the SFJ's). The delay is the shift of the
    accelerometer that best correlates it with the Doppler lateral g on
    the position clock, over the IMU-sourced samples at speed, searched
    over :data:`ACCEL_DELAY_RANGE_S` in 10 ms steps and refined on the
    parabola through the best three. ``gps_lat_g`` is speed times a
    backward difference of the course, so it is itself half a GPS period
    late; that half period is added. NaN without enough samples.
    """
    if t_acc.size < 200:
        return float("nan")
    dop = geo.nan_window_mean(geom.lat_g_doppler, geom.t, 0.1)
    use = np.isfinite(dop) & np.isfinite(geom.speed) & (geom.speed >= geo.LAG_MIN_SPEED_MS)
    use &= np.interp(geom.t, t_acc, is_imu.astype(float)) > 0.999
    use &= (geom.t > t_acc[0] + 1.0) & (geom.t < t_acc[-1] - 1.0)
    if use.sum() < 500:
        return float("nan")
    tt, ref = geom.t[use], dop[use]
    lags = np.arange(ACCEL_DELAY_RANGE_S[0], ACCEL_DELAY_RANGE_S[1] + 1e-9, 0.01)
    corr = np.full(lags.size, np.nan)
    for i, lag in enumerate(lags):
        x = np.interp(tt + lag, t_acc, lat_acc)
        ok = np.isfinite(x)
        if ok.sum() >= 500 and np.std(x[ok]) > 0:
            corr[i] = np.corrcoef(x[ok], ref[ok])[0, 1]
    if not np.isfinite(corr).any():
        return float("nan")
    j = int(np.nanargmax(corr))
    best = float(lags[j])
    if 0 < j < lags.size - 1 and np.isfinite(corr[j - 1]) and np.isfinite(corr[j + 1]):
        curve = corr[j - 1] - 2.0 * corr[j] + corr[j + 1]
        if curve < 0:
            best += 0.01 * 0.5 * (corr[j - 1] - corr[j + 1]) / curve
    return best + 0.5 * float(np.median(np.diff(geom.t)))


def load_layout_laps(
    root: Path,
    layout_key: str,
    rows: pd.DataFrame,
    *,
    inertial: bool = True,
    sessions: pd.DataFrame | None = None,
) -> list[LapRecord]:
    """The lap records of ``rows`` (lap-index rows of one layout: at least
    ``session_id, lap_num, car, date, lap_time_s``), session by session.

    ``inertial`` also loads each session's accelerometer and attaches its
    lateral g to the laps it covers from start to end.
    """
    rows = rows[rows["layout_id"] == layout_key] if "layout_id" in rows.columns else rows
    slopes: dict[str, float] = {}
    if inertial:
        slopes = _session_slopes(sessions if sessions is not None else geo.load_sessions(root))
    out: list[LapRecord] = []
    for sid, grp in rows.groupby("session_id", sort=True):
        geom = geo.load_session_geometry(root, str(sid), layout_key)
        if geom is None:
            continue
        empty = np.zeros(0)
        t_in, lat_in, imu_in = empty, empty, np.zeros(0, dtype=bool)
        delay = float("nan")
        if inertial and str(sid) in slopes:
            t_in, lat_in, imu_in = _session_inertial(root, str(sid))
            delay = accel_delay(geom, t_in, lat_in, imu_in)
        usable = bool(np.isfinite(delay)) and t_in.size > 1
        ordered = grp.sort_values("lap_num")
        for lap_num, car, date, lap_time in zip(
            ordered["lap_num"], ordered["car"], ordered["date"], ordered["lap_time_s"]
        ):
            tr = geo.lap_trace(geom, int(lap_num))
            if tr is None or len(tr.t) < 50:
                continue
            lat_g = None
            if usable:
                i0, i1 = np.searchsorted(t_in, [tr.t[0], tr.t[-1]])
                covered = (
                    i1 - i0 > 50
                    and bool(np.all(imu_in[i0:i1]))
                    and float(np.max(np.diff(t_in[i0:i1]))) <= 0.5
                    and t_in[i0] - tr.t[0] <= 0.5
                    and tr.t[-1] - t_in[i1 - 1] <= 0.5
                )
                if covered:
                    lat_g = np.interp(tr.t + delay, t_in, lat_in) / slopes[str(sid)]
            out.append(
                LapRecord(
                    session_id=str(sid),
                    lap_num=int(lap_num),
                    car=str(car),
                    date=str(date),
                    lap_time_s=float(lap_time),
                    trace=tr,
                    lat_g=lat_g,
                    accel_delay_s=delay,
                )
            )
    return out


# -------------------------------------------------------------- projection


def resample_line(x: np.ndarray, y: np.ndarray, ds_m: float) -> tuple[np.ndarray, np.ndarray]:
    """The polyline ``(x, y)`` at ``n + 1`` points evenly spaced along it,
    first and last kept, with ``n = round(length / ds_m)``."""
    step = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(step)])
    keep = np.concatenate([[True], step > 0])
    n = max(int(round(s[-1] / ds_m)), 2)
    grid = np.linspace(0.0, s[-1], n + 1)
    return np.interp(grid, s[keep], x[keep]), np.interp(grid, s[keep], y[keep])


def project_onto(
    xr: np.ndarray,
    yr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s_guess: np.ndarray,
    *,
    window_m: float = PROJECT_WINDOW_M,
    closed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance along, and signed offset from, the reference line.

    ``(xr, yr)`` are the line's vertices, evenly spaced; ``s_guess`` is
    where along the line each point ``(x, y)`` is expected. Only the
    vertices within ``window_m`` of the guess are candidates; the point is
    then dropped onto the two segments meeting at the nearest one. The
    offset is positive to the right of the direction of travel.

    ``closed`` says the last vertex is the first again: the line is then
    continued by a window's length past both ends, so a point just before
    the start gets a small negative distance and one just past the end a
    distance over the lap length. On an open line such points are clamped
    to the ends.
    """
    seg = np.hypot(np.diff(xr), np.diff(yr))
    n = len(xr) - 1
    ds = float(seg.sum()) / n
    half = max(int(math.ceil(window_m / ds)), 1)
    pad = 0
    if closed:
        pad = min(half, n - 1)
        xr = np.concatenate([xr[-pad - 1 : -1], xr, xr[1 : pad + 1]])
        yr = np.concatenate([yr[-pad - 1 : -1], yr, yr[1 : pad + 1]])
        seg = np.hypot(np.diff(xr), np.diff(yr))
        n = len(xr) - 1
    sr = np.concatenate([[0.0], np.cumsum(seg)])
    sr = sr - sr[pad]
    centre = np.clip(np.rint(np.asarray(s_guess, float) / ds).astype(np.int64) + pad, 0, n)
    if USE_COMPILED:
        near = _kernel.nearest_vertex(
            np.ascontiguousarray(xr, dtype=float),
            np.ascontiguousarray(yr, dtype=float),
            np.ascontiguousarray(x, dtype=float),
            np.ascontiguousarray(y, dtype=float),
            centre,
            half,
            n,
        )
    else:  # the same search as one array of points × candidates
        cand = np.clip(centre[:, None] + np.arange(-half, half + 1)[None, :], 0, n)
        d2 = (x[:, None] - xr[cand]) ** 2 + (y[:, None] - yr[cand]) ** 2
        near = cand[np.arange(len(x)), np.argmin(d2, axis=1)]
    best_s = np.full(len(x), np.nan)
    best_off = np.full(len(x), np.nan)
    best_d = np.full(len(x), np.inf)
    for first in (np.clip(near - 1, 0, n - 1), np.clip(near, 0, n - 1)):
        ax, ay = xr[first], yr[first]
        tx, ty = xr[first + 1] - ax, yr[first + 1] - ay
        length = np.maximum(seg[first], 1e-9)
        tx, ty = tx / length, ty / length
        along = np.clip((x - ax) * tx + (y - ay) * ty, 0.0, length)
        px, py = ax + along * tx, ay + along * ty
        dist = np.hypot(x - px, y - py)
        # right of travel: the tangent turned clockwise is (ty, -tx)
        off = (x - px) * ty - (y - py) * tx
        better = dist < best_d
        best_d = np.where(better, dist, best_d)
        best_s = np.where(better, sr[first] + along, best_s)
        best_off = np.where(better, off, best_off)
    return best_s, best_off


def _bin_series(
    s: np.ndarray,
    value: np.ndarray,
    grid: np.ndarray,
    *,
    lap_m: float | None = None,
    wrap_value: float = 0.0,
) -> np.ndarray:
    """``value`` (a function of the lap's projected distance ``s``) at the
    grid distances; NaN outside the lap's range and where the value is.

    A lap runs from one gate crossing to the next, which on the line is
    from a little after (or before) its start to a little before (or after)
    its end. With ``lap_m`` the grid points the lap itself does not reach
    are read from its other end, one lap length away, where the car is at
    the same place a lap later (``wrap_value`` is what the value gains over
    a lap: the lap's whole turns for the course).
    """
    ok = np.isfinite(s) & np.isfinite(value)
    if ok.sum() < 2:
        return np.full(grid.shape, np.nan)
    s, value = s[ok], value[ok]
    mono = np.maximum.accumulate(s)
    keep = np.concatenate([[True], np.diff(mono) > 0])
    s, value = mono[keep], value[keep]
    out = np.interp(grid, s, value)
    before, after = grid < s[0] - 1e-6, grid > s[-1] + 1e-6
    out[before | after] = np.nan
    if lap_m is not None:
        for mask, shift, gain in ((before, -lap_m, -wrap_value), (after, lap_m, wrap_value)):
            if not mask.any():
                continue
            g = grid[mask]
            val = np.interp(g, s + shift, value + gain)
            val[(g < s[0] + shift - 1e-6) | (g > s[-1] + shift + 1e-6)] = np.nan
            out[mask] = val
    return np.asarray(out)


@dataclass
class Projection:
    """Every lap's quantities on the reference line's grid (laps × bins)."""

    grid: np.ndarray  # distance of each bin along the reference (m), n + 1 points
    course: np.ndarray
    speed: np.ndarray
    offset: np.ndarray
    alt: np.ndarray
    time: np.ndarray  # session time at which the lap passes the bin
    lat_g: np.ndarray  # NaN rows for laps without IMU cover


def lap_turns(trace: geo.LapTrace) -> int:
    """Whole turns of the course over the lap."""
    c = trace.course[np.isfinite(trace.course)]
    return int(np.rint((c[-1] - c[0]) / (2.0 * math.pi))) if c.size >= 2 else 0


def project_laps(
    laps: Sequence[LapRecord], xr: np.ndarray, yr: np.ndarray, *, closed: bool = True
) -> Projection:
    """Project ``laps`` onto the line ``(xr, yr)`` and bin their series.

    ``closed`` (the line's last vertex is its first) lets a lap's ends
    reach past the line's and fills the bins a lap does not reach from its
    other end (:func:`_bin_series`); the first reference, one lap's own
    trace, is open.
    """
    seg = np.hypot(np.diff(xr), np.diff(yr))
    grid = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(grid[-1])
    shape = (len(laps), len(grid))
    out = {k: np.full(shape, np.nan) for k in ("course", "speed", "offset", "alt", "time", "lat_g")}
    lap_m = length if closed else None
    for i, lap in enumerate(laps):
        tr = lap.trace
        guess = tr.s * (length / tr.s[-1]) if tr.s[-1] > 0 else tr.s
        s, off = project_onto(xr, yr, tr.x, tr.y, guess, closed=closed)
        whole = 2.0 * math.pi * lap_turns(tr)
        out["course"][i] = _bin_series(s, tr.course, grid, lap_m=lap_m, wrap_value=whole)
        out["speed"][i] = _bin_series(s, tr.speed, grid, lap_m=lap_m)
        out["offset"][i] = _bin_series(s, off, grid, lap_m=lap_m)
        out["alt"][i] = _bin_series(s, tr.alt, grid, lap_m=lap_m)
        out["time"][i] = _bin_series(s, tr.t, grid)
        if lap.lat_g is not None:
            out["lat_g"][i] = _bin_series(s, lap.lat_g, grid, lap_m=lap_m)
    return Projection(grid=grid, **out)


# ----------------------------------------------------------------- profile


def _periodic_mean(v: np.ndarray, half: int) -> np.ndarray:
    """Moving mean over ``2·half + 1`` points of a closed series."""
    if half < 1:
        return np.asarray(v, float)
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(np.concatenate([v[-half:], v, v[:half]]), kernel, mode="valid")


def _fill_nan_periodic(v: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v)
    if ok.all():
        return np.asarray(v, float)
    if not ok.any():
        return np.zeros_like(v, dtype=float)
    idx = np.arange(len(v))
    return np.asarray(np.interp(idx, idx[ok], v[ok], period=len(v)))


def median_course(course: np.ndarray, min_laps: int = MIN_BIN_LAPS) -> tuple[np.ndarray, int]:
    """Per-bin median of the laps' courses (laps × bins, each unwrapped
    along its lap), each lap's constant removed; and the lap's whole turns.

    The first lap with full cover is the carrier: every lap's difference
    from it is small and smooth, so the median is taken on the difference.
    """
    full = np.flatnonzero(np.isfinite(course).all(axis=1))
    carrier_i = int(full[0]) if full.size else int(np.argmax(np.isfinite(course).sum(axis=1)))
    carrier = course[carrier_i].copy()
    ok = np.isfinite(carrier)
    idx = np.arange(carrier.size)
    carrier = np.interp(idx, idx[ok], carrier[ok])
    delta = course - carrier[None, :]
    with np.errstate(invalid="ignore"):
        const = geo.nan_median(delta, 1)
        delta = delta - const[:, None]
        count = np.isfinite(delta).sum(axis=0)
        med = geo.nan_median(delta, 0)
    med[count < min_laps] = np.nan
    med = np.interp(idx, idx[np.isfinite(med)], med[np.isfinite(med)])
    psi = carrier + med
    turns = int(np.rint((psi[-1] - psi[0]) / (2.0 * math.pi)))
    return psi, turns


def close_course(psi: np.ndarray, turns: int) -> tuple[np.ndarray, float]:
    """``psi`` (n + 1 points, first and last the same place) with its end
    made exactly ``turns`` whole turns from its start, the difference
    spread evenly; and that difference in turns (the profile's closure)."""
    want = psi[0] + 2.0 * math.pi * turns
    miss = psi[-1] - want
    ramp = np.linspace(0.0, 1.0, len(psi))
    return psi - miss * ramp, float(miss / (2.0 * math.pi))


def curvature(psi: np.ndarray, ds: float, turns: int, window_m: float) -> np.ndarray:
    """``dψ/ds`` of a closed course (n + 1 points spaced ``ds``, the last
    equal to the first plus ``turns`` turns) as the difference over
    ``window_m`` centred on each of the first n points."""
    n = len(psi) - 1
    total = 2.0 * math.pi * turns
    ext_s = np.arange(-n, 2 * n + 1) * ds
    body = psi[:-1]
    ext = np.concatenate([body - total, body, body + total, [psi[0] + 2 * total]])
    s = np.arange(n) * ds
    half = window_m / 2.0
    return np.asarray(
        (np.interp(s + half, ext_s, ext) - np.interp(s - half, ext_s, ext)) / window_m
    )


def integrate_line(psi: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray, float]:
    """The line whose compass course is ``psi`` (n + 1 points): ``x, y`` at
    the first n points with the closing gap spread over the lap, and the
    size of that gap (m)."""
    mid = 0.5 * (psi[1:] + psi[:-1])
    x = np.concatenate([[0.0], np.cumsum(np.sin(mid) * ds)])
    y = np.concatenate([[0.0], np.cumsum(np.cos(mid) * ds)])
    gap = float(math.hypot(x[-1], y[-1]))
    ramp = np.linspace(0.0, 1.0, len(psi))
    return (x - x[-1] * ramp)[:-1], (y - y[-1] * ramp)[:-1], gap


def corner_table(kappa: np.ndarray, ds: float) -> list[dict]:
    """Corners of a closed profile: stretches turning one way with ``|κ|``
    above :data:`CORNER_EDGE_KAPPA` whose apex reaches
    :data:`CORNER_APEX_KAPPA` and which turn by
    :data:`CORNER_MIN_ANGLE_DEG` or more. ``s_in, s_apex, s_out`` (m),
    ``sign`` (+1 right), ``turn_angle`` (degrees, signed) and
    ``kappa_apex`` (1/m, signed). A corner spanning the start/finish has
    ``s_out < s_in``."""
    n = len(kappa)
    state = np.where(np.abs(kappa) > CORNER_EDGE_KAPPA, np.sign(kappa), 0).astype(int)
    if np.all(state == state[0]):
        return []
    # start the scan at a change of state so no stretch is cut by the lap's end
    shift = int(np.flatnonzero(state != np.roll(state, 1))[0])
    st = np.roll(state, -shift)
    kp = np.roll(kappa, -shift)
    edges = np.flatnonzero(np.concatenate([[True], st[1:] != st[:-1], [True]]))
    out: list[dict] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if st[a] == 0:
            continue
        piece = kp[a:b]
        apex = int(np.argmax(np.abs(piece)))
        angle = float(np.degrees(np.sum(piece) * ds))
        if abs(piece[apex]) < CORNER_APEX_KAPPA or abs(angle) < CORNER_MIN_ANGLE_DEG:
            continue
        out.append(
            {
                "s_in": float(((a + shift) % n) * ds),
                "s_apex": float(((a + apex + shift) % n) * ds),
                "s_out": float(((b + shift) % n) * ds),
                "sign": int(st[a]),
                "turn_angle": angle,
                "kappa_apex": float(piece[apex]),
            }
        )
    out.sort(key=lambda c: c["s_apex"])
    return out


# ------------------------------------------------------------------- model


@dataclass
class TrackModel:
    """A layout's curvature profile and line on a closed grid of ``n`` bins
    (bin ``i`` is at ``s = i · ds_m``; ``n · ds_m = length_m``)."""

    layout_key: str
    track_canonical: str
    length_m: float
    ds_m: float
    kappa_heat: np.ndarray  # 1/m, positive in a right turn
    x_m: np.ndarray  # east of the start/finish gate
    y_m: np.ndarray  # north of it
    alt_m: np.ndarray
    grade: np.ndarray  # d alt / ds
    n_laps: np.ndarray
    kappa_se: np.ndarray
    corners: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    gate: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    #: Vertical curvature d(grade)/ds (1/m): positive in a compression,
    #: negative over a crest. The normal load goes with 1 + V²·kappa_v / g.
    kappa_v: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n(self) -> int:
        return int(len(self.kappa_heat))

    @property
    def s_m(self) -> np.ndarray:
        return np.arange(self.n) * self.ds_m

    @property
    def turns(self) -> int:
        return int(self.meta.get("turns", 1))

    def course(self) -> np.ndarray:
        """Compass course (rad) at the ``n`` bins, rebuilt from the line."""
        x = np.concatenate([self.x_m, self.x_m[:1]])
        y = np.concatenate([self.y_m, self.y_m[:1]])
        return np.unwrap(np.arctan2(np.diff(x), np.diff(y)))

    def to_json(self) -> dict:
        def _list(a: np.ndarray, nd: int) -> list:
            return [None if not np.isfinite(v) else round(float(v), nd) for v in a]

        return {
            "layout_key": self.layout_key,
            "track_canonical": self.track_canonical,
            "sf_gate": self.gate,
            "length_m": round(self.length_m, 3),
            "ds_m": self.ds_m,
            "n": self.n,
            "kappa_heat_1_per_m": _list(self.kappa_heat, 7),
            "x_m": _list(self.x_m, 3),
            "y_m": _list(self.y_m, 3),
            "alt_m": _list(self.alt_m, 3),
            "grade": _list(self.grade, 5),
            "kappa_v_1_per_m": _list(self.kappa_v, 7),
            "n_laps": [int(v) for v in self.n_laps],
            "kappa_se": _list(self.kappa_se, 7),
            "corners": self.corners,
            "fills": self.fills,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, data: dict) -> "TrackModel":
        def _arr(key: str) -> np.ndarray:
            return np.array([np.nan if v is None else v for v in data[key]], dtype=float)

        return cls(
            layout_key=data["layout_key"],
            track_canonical=data["track_canonical"],
            length_m=float(data["length_m"]),
            ds_m=float(data["ds_m"]),
            kappa_heat=_arr("kappa_heat_1_per_m"),
            x_m=_arr("x_m"),
            y_m=_arr("y_m"),
            alt_m=_arr("alt_m"),
            grade=_arr("grade"),
            n_laps=np.asarray(data["n_laps"], dtype=int),
            kappa_se=_arr("kappa_se"),
            corners=list(data.get("corners", [])),
            fills=list(data.get("fills", [])),
            gate=dict(data.get("sf_gate", {})),
            meta=dict(data.get("meta", {})),
            kappa_v=_arr("kappa_v_1_per_m") if "kappa_v_1_per_m" in data else np.zeros(0),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json()))
        return path

    @classmethod
    def load(cls, path: Path) -> "TrackModel":
        return cls.from_json(json.loads(Path(path).read_text()))


@dataclass
class TrackBuild:
    """A built track with what it was built from, bin by bin."""

    model: TrackModel
    laps: list[LapRecord]
    proj: Projection  # on the final reference line; bins 0..n (the last is the first again)
    psi: np.ndarray  # the closed median course at bins 0..n
    kappa_doppler: np.ndarray  # κ before any fill from the inertial profile
    ref_x: np.ndarray  # the reference line the laps are binned on (n + 1 vertices)
    ref_y: np.ndarray
    closure_turns: float  # what the median course missed a whole turn by
    line_gap_m: float  # closing gap of the integrated line before it is spread
    reproject_rms_g: float | None = None  # how far one more re-projection moves V²κ

    @property
    def speed_median(self) -> np.ndarray:
        """Per-bin median speed of the build laps at the model's n bins."""
        with np.errstate(invalid="ignore"):
            return _fill_nan_periodic(geo.nan_median(self.proj.speed[:, :-1], 0))


def _reference_from_fastest(
    laps: Sequence[LapRecord], ds_m: float
) -> tuple[np.ndarray, np.ndarray]:
    fastest = min(laps, key=lambda lap: lap.lap_time_s)
    return resample_line(fastest.trace.x, fastest.trace.y, ds_m)


def _rereference(
    xr: np.ndarray, yr: np.ndarray, offset: np.ndarray, ds_m: float, *, closed: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Move the line by the smoothed per-bin median offset of the laps and
    resample it; the result is closed (its last point is its first).

    ``closed=False`` is for the first reference, one lap's own trace: that
    lap ends where it crossed the line a lap later, which can be metres to
    the side of where it started (9.7 m on Suzuka's straight), so its two
    ends are different baselines and nothing is smoothed across them. Both
    ends move onto the median line and are then made one point.
    """
    with np.errstate(invalid="ignore"):
        med = geo.nan_median(offset, 0)
    seg = np.hypot(np.diff(xr), np.diff(yr))
    half = int(round(OFFSET_SMOOTH_M / 2.0 / float(seg.mean())))
    if closed:
        body = _periodic_mean(_fill_nan_periodic(med[:-1]), half)
        x0, y0 = xr[:-1], yr[:-1]
    else:
        idx = np.arange(len(med))
        ok = np.isfinite(med)
        body = np.interp(idx, idx[ok], med[ok])
        if half >= 1:
            kernel = np.ones(2 * half + 1) / (2 * half + 1)
            body = np.convolve(np.pad(body, half, mode="edge"), kernel, mode="valid")
        x0, y0 = xr, yr
    tx, ty = np.gradient(x0), np.gradient(y0)
    if closed:  # the tangent at the seam uses the points either side of it
        tx[0], ty[0] = x0[1] - x0[-1], y0[1] - y0[-1]
        tx[-1], ty[-1] = x0[0] - x0[-2], y0[0] - y0[-2]
    norm = np.maximum(np.hypot(tx, ty), 1e-9)
    x = x0 + body * ty / norm
    y = y0 - body * tx / norm
    if closed:
        x, y = np.concatenate([x, x[:1]]), np.concatenate([y, y[:1]])
    else:
        x[0] = x[-1] = 0.5 * (x[0] + x[-1])
        y[0] = y[-1] = 0.5 * (y[0] + y[-1])
    return resample_line(x, y, ds_m)


def _altitude(
    proj: Projection, laps: Sequence[LapRecord], ds: float
) -> tuple[np.ndarray, np.ndarray]:
    alt = proj.alt[:, :-1].copy()
    sessions = np.array([lap.session_id for lap in laps])
    with np.errstate(invalid="ignore"):
        profile = geo.nan_median(alt, 0)
        for sid in np.unique(sessions):
            m = sessions == sid
            off = np.nanmedian(alt[m] - profile[None, :])
            if np.isfinite(off):
                alt[m] -= off
        profile = geo.nan_median(alt, 0)
    profile = _fill_nan_periodic(profile)
    profile = _periodic_mean(profile, int(round(ALT_SMOOTH_M / 2.0 / ds)))
    half = max(int(round(GRADE_WINDOW_M / 2.0 / ds)), 1)
    grade = (np.roll(profile, -half) - np.roll(profile, half)) / (2 * half * ds)
    return profile, grade


def vertical_curvature(grade: np.ndarray, ds: float) -> np.ndarray:
    """``d(grade)/ds`` of a closed profile over :data:`GRADE_WINDOW_M`
    (the grade is itself a difference over that window, so this is the
    altitude's second difference over twice it)."""
    half = max(int(round(GRADE_WINDOW_M / 2.0 / ds)), 1)
    return np.asarray((np.roll(grade, -half) - np.roll(grade, half)) / (2 * half * ds))


def _kappa_from_course(course: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray, int, float]:
    psi, turns = median_course(course)
    psi, closure = close_course(psi, turns)
    return curvature(psi, ds, turns, KAPPA_WINDOW_M), psi, turns, closure


def bootstrap_kappa_se(
    course: np.ndarray, days: Sequence[str], ds: float, *, n_boot: int = 40, seed: int = 0
) -> np.ndarray:
    """Per-bin standard error of κ from resampling whole days."""
    day_arr = np.asarray(days)
    uniq = np.unique(day_arr)
    if uniq.size < 2:
        return np.full(course.shape[1] - 1, np.nan)
    rng = np.random.default_rng(seed)
    rows = {d: np.flatnonzero(day_arr == d) for d in uniq}
    reps = []
    for _ in range(n_boot):
        pick = np.concatenate([rows[d] for d in rng.choice(uniq, size=uniq.size)])
        try:
            reps.append(_kappa_from_course(course[pick], ds)[0])
        except (ValueError, IndexError):  # a draw without cover somewhere
            continue
    if len(reps) < 2:
        return np.full(course.shape[1] - 1, np.nan)
    return np.asarray(np.std(np.vstack(reps), axis=0, ddof=1))


def build_from_laps(
    laps: Sequence[LapRecord],
    layout_key: str,
    *,
    ds_m: float = DS_M,
    n_boot: int = 40,
    measure_convergence: bool = False,
) -> TrackBuild:
    """Build a track from lap records (see the module docstring).

    ``measure_convergence`` projects once more onto a line re-referenced a
    second time and records how far ``V²κ`` moves (g rms on the median
    speed trace), the check that one re-referencing is enough.
    """
    laps = list(laps)
    if len(laps) < MIN_BIN_LAPS:
        raise ValueError(f"{layout_key}: {len(laps)} laps, need {MIN_BIN_LAPS} to build a track")
    venue, layout = geo.LAYOUTS[layout_key]
    gate = geo.layout_gate(layout_key)
    xr, yr = _reference_from_fastest(laps, ds_m)
    # The fastest lap as an open line, then the median line (closed), then
    # the median line once more now that the laps are binned around the seam.
    first = project_laps(laps, xr, yr, closed=False)
    xr, yr = _rereference(xr, yr, first.offset, ds_m, closed=False)
    second = project_laps(laps, xr, yr)
    xr, yr = _rereference(xr, yr, second.offset, ds_m, closed=True)
    proj = project_laps(laps, xr, yr)
    n = len(xr) - 1
    ds = float(proj.grid[-1] / n)
    kappa, psi, turns, closure = _kappa_from_course(proj.course, ds)

    reproject_rms = None
    if measure_convergence:
        xr2, yr2 = _rereference(xr, yr, proj.offset, ds_m, closed=True)
        proj2 = project_laps(laps, xr2, yr2)
        ds2 = float(proj2.grid[-1] / (len(xr2) - 1))
        kappa2 = _kappa_from_course(proj2.course, ds2)[0]
        with np.errstate(invalid="ignore"):
            v = _fill_nan_periodic(geo.nan_median(proj.speed[:, :-1], 0))
        k2 = np.interp(np.arange(n) / n, np.arange(len(kappa2)) / len(kappa2), kappa2, period=1.0)
        reproject_rms = float(np.sqrt(np.mean((v * v * (kappa - k2) / G_MS2) ** 2)))

    fills: list[dict] = []
    kappa_doppler = kappa
    if np.isfinite(proj.lat_g).any():
        with np.errstate(invalid="ignore"):
            v_med = _fill_nan_periodic(geo.nan_median(proj.speed[:, :-1], 0))
        k_in = _inertial_kappa(proj)[0]
        kappa, fills = fill_from_inertial(kappa, k_in, v_med, ds)

    xl, yl, gap = integrate_line(psi, ds)
    # lay the integrated line on the median XY line: same centroid
    xl = xl + (float(np.mean(xr[:-1])) - float(np.mean(xl)))
    yl = yl + (float(np.mean(yr[:-1])) - float(np.mean(yl)))
    alt, grade = _altitude(proj, laps, ds)
    days = [lap.date for lap in laps]
    model = TrackModel(
        layout_key=layout_key,
        track_canonical=layout.track_canonical,
        length_m=float(proj.grid[-1]),
        ds_m=ds,
        kappa_heat=kappa,
        x_m=xl,
        y_m=yl,
        alt_m=alt,
        grade=grade,
        n_laps=np.isfinite(proj.course[:, :-1]).sum(axis=0),
        kappa_se=(
            bootstrap_kappa_se(proj.course, days, ds, n_boot=n_boot)
            if n_boot
            else np.full(n, np.nan)
        ),
        corners=corner_table(kappa, ds),
        fills=fills,
        kappa_v=vertical_curvature(grade, ds),
        gate={"lat": gate.lat, "lon": gate.lon},
        meta={
            "venue_id": venue.venue_id,
            "turns": turns,
            "kappa_method": KAPPA_METHOD,
            "built_from": {
                "n_laps": len(laps),
                "n_sessions": len({lap.session_id for lap in laps}),
                "n_days": len(set(days)),
                "cars": sorted({lap.car for lap in laps}),
            },
            "fit_at_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return TrackBuild(
        model=model,
        laps=laps,
        proj=proj,
        psi=psi,
        kappa_doppler=kappa_doppler,
        ref_x=xr,
        ref_y=yr,
        closure_turns=closure,
        line_gap_m=gap,
        reproject_rms_g=reproject_rms,
    )


def build_track_model(
    root: Path,
    layout_key: str,
    *,
    cars: Iterable[str] | None = None,
    ds_m: float = DS_M,
    exclude_session_ids: Iterable[str] = (),
    cache_dir: Path | None = None,
    inertial: bool = True,
    n_boot: int = 40,
    measure_convergence: bool = False,
    basis: str = "dry",
) -> TrackBuild:
    """Build the track of ``layout_key`` from its build laps: every car's,
    or only ``cars``', without the sessions in ``exclude_session_ids``.

    ``basis`` chooses the population (:data:`BASES`): ``"dry"``, the laps
    at the car's pace, or ``"wet"``, the fast laps of wet sessions judged
    against each other. The road is one; the line driven on it is not, so
    a layout has a track per basis where it has the laps.
    """
    index = geo.clean_lap_index(root, cache_dir=cache_dir)
    rows = index[index[BASES[basis]] & (index["layout_id"] == layout_key)]
    if cars is not None:
        rows = rows[rows["car"].isin(list(cars))]
    rows = rows[~rows["session_id"].isin(list(exclude_session_ids))]
    laps = load_layout_laps(root, layout_key, rows, inertial=inertial)
    build = build_from_laps(
        laps, layout_key, ds_m=ds_m, n_boot=n_boot, measure_convergence=measure_convergence
    )
    build.model.meta["basis"] = basis
    return build


def project_lap(
    track: TrackModel, x: np.ndarray, y: np.ndarray, *, s_prev: np.ndarray | None = None
) -> np.ndarray:
    """Distance along ``track`` of the points ``(x, y)`` (metres east and
    north of the layout's gate, in driving order).

    ``s_prev`` is where each point is expected (for example the previous
    estimate); without it the points' own path length, scaled to the
    track's, is used, which needs the points to start near ``s = 0``. The
    match is confined to :data:`PROJECT_WINDOW_M` around the expectation.
    """
    xr = np.concatenate([track.x_m, track.x_m[:1]])
    yr = np.concatenate([track.y_m, track.y_m[:1]])
    if s_prev is None:
        own = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        s_prev = own
    s, _ = project_onto(
        xr, yr, np.asarray(x, float), np.asarray(y, float), np.asarray(s_prev), closed=True
    )
    return s


def inertial_kappa(
    build: TrackBuild, rows: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(κ_inertial, n_laps)`` per bin: the median over the IMU-sourced
    laps (all of the build's, or the lap rows ``rows``) of accelerometer
    lateral over V², in 1/m. The accelerometer is in the chassis frame (it
    reads cornering less the roll and the banking), scaled lap by lap by
    its session's audit slope and moved onto the position clock."""
    return _inertial_kappa(build.proj, rows)


def _inertial_kappa(
    proj: Projection, rows: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    lat = proj.lat_g[:, :-1]
    v = proj.speed[:, :-1]
    if rows is not None:
        lat, v = lat[rows], v[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(v > 5.0, lat * G_MS2 / (v * v), np.nan)
        count = np.isfinite(k).sum(axis=0)
        med = geo.nan_median(k, 0) if len(k) else np.full(lat.shape[1], np.nan)
    med = np.where(count >= MIN_BIN_LAPS, med, np.nan)
    return np.asarray(med), count


def lateral_difference_g(
    kappa: np.ndarray, kappa_ref: np.ndarray, speed: np.ndarray, ds: float, stretch_m: float
) -> np.ndarray:
    """``V²(κ − κ_ref)`` in g as its mean over ``stretch_m`` around each bin
    of a closed profile; bins without a reference count as no difference."""
    with np.errstate(invalid="ignore"):
        d = speed * speed * (kappa - kappa_ref) / G_MS2
    return _periodic_mean(np.nan_to_num(d, nan=0.0), int(round(stretch_m / 2.0 / ds)))


def find_zones(over: np.ndarray, ds: float, widen_m: float = 0.0) -> list[tuple[int, int]]:
    """Runs of True in a closed boolean profile as ``(first, last)`` bins,
    after every True has been spread by ``widen_m`` either side, so runs
    closer than twice that merge (a run across the seam has
    ``first > last``)."""
    n = len(over)
    pad = int(round(widen_m / ds))
    wide = over.copy()
    for k in range(1, pad + 1):
        wide |= np.roll(over, k) | np.roll(over, -k)
    if not wide.any():
        return []
    if wide.all():
        return [(0, n - 1)]
    shift = int(np.flatnonzero(wide & ~np.roll(wide, 1))[0])
    rolled = np.roll(wide, -shift)
    edges = np.flatnonzero(np.concatenate([[True], rolled[1:] != rolled[:-1], [True]]))
    return [
        (int((a + shift) % n), int((b - 1 + shift) % n))
        for a, b in zip(edges[:-1], edges[1:])
        if rolled[a]
    ]


def fill_from_inertial(
    kappa: np.ndarray,
    kappa_inertial: np.ndarray,
    speed: np.ndarray,
    ds: float,
    *,
    threshold_g: float = FILL_THRESHOLD_G,
    stretch_m: float = FILL_STRETCH_M,
) -> tuple[np.ndarray, list[dict]]:
    """Replace the stretches where the Doppler profile and the inertial
    one disagree (a GPS error that sits at one place every lap: an
    underpass) by the inertial profile.

    A zone is where the ``stretch_m`` mean of ``V²(κ − κ_inertial)``
    exceeds ``threshold_g``, widened by a stretch either side so that the
    two lobes of one excursion are one zone. The course is absolute, so
    past the error it is right again: inside the zone κ becomes the
    inertial κ, scaled so that the course change across the zone stays the
    Doppler one (shifted instead where the inertial profile turns by under
    a degree there). Returns the filled profile and one record per
    zone: ``s_from, s_to`` (m) and the mean lateral g ``before`` and
    ``after`` on the median speed trace.
    """
    out = np.asarray(kappa, float).copy()
    diff = lateral_difference_g(kappa, kappa_inertial, speed, ds, stretch_m)
    fills: list[dict] = []
    n = len(out)
    for first, last in find_zones(np.abs(diff) > threshold_g, ds, stretch_m):
        idx = np.arange(first, last + 1) if first <= last else np.arange(first, last + 1 + n) % n
        ref = kappa_inertial[idx]
        if not np.isfinite(ref).all():
            continue
        turn, turn_ref = float(np.sum(out[idx]) * ds), float(np.sum(ref) * ds)
        if abs(turn_ref) >= math.radians(1.0) and turn / turn_ref > 0:
            new = ref * (turn / turn_ref)
        else:
            new = ref + (turn - turn_ref) / (len(idx) * ds)
        v2 = speed[idx] ** 2 / G_MS2
        fills.append(
            {
                "s_from": float(first * ds),
                "s_to": float(last * ds),
                "before": float(np.mean(v2 * out[idx])),
                "after": float(np.mean(v2 * new)),
            }
        )
        out[idx] = new
    return out, fills


def model_summary(model: TrackModel) -> dict:
    """The few numbers that identify a build (for logs)."""
    return {
        "layout": model.layout_key,
        "length_m": round(model.length_m, 1),
        "n": model.n,
        "turns": model.turns,
        "basis": model.meta.get("basis", "dry"),
        "corners": len(model.corners),
        "built_from": model.meta.get("built_from"),
    }


__all__ = [
    "LapRecord",
    "Projection",
    "TrackBuild",
    "TrackModel",
    "accel_delay",
    "bootstrap_kappa_se",
    "build_from_laps",
    "build_track_model",
    "close_course",
    "corner_table",
    "curvature",
    "fill_from_inertial",
    "find_zones",
    "inertial_kappa",
    "lateral_difference_g",
    "integrate_line",
    "lap_turns",
    "load_layout_laps",
    "median_course",
    "project_lap",
    "project_laps",
    "project_onto",
    "resample_line",
    "vertical_curvature",
]
