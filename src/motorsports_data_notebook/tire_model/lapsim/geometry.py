"""Lap geometry on one clock (lap simulation segment B1, issue #204).

What a track is built from, per session and at query time (nothing here is
stored in the dataset):

**Position** is the logged GPS fix in a local metric frame whose origin is
the layout's start/finish gate, so every session of a venue shares it.
Implausible fixes are dropped the way the extractor's reference drops them
(:func:`~...tire_etl.kinematics._plausible_fixes`), the trace is cut into
*segments* at every gap over :data:`GAP_MAX_S` and nothing is interpolated
across a cut.

**Course over ground** comes from the stored ``gps_yaw_rate_dps``. That
column is not a logger channel: libxrk writes the one-sample difference of
the Doppler course ``atan2(vE, vN)`` over the raw timecode step, so
``Σ gps_yaw_rate_dps · Δt`` gives the course back exactly, up to one
constant per segment (:func:`doppler_course`). The constant is set from the
heading of the position chords, which makes the course absolute. Curvature
from it needs one derivative of a precise quantity where position needs two
of a noisy one.

**Lag.** The receiver's velocity solution trails its position solution by a
latency that depends on the logger (65–430 ms). ``course_Doppler −
heading_chord`` is therefore ``−τ·ω``; the slope of that difference on the
yaw rate is the session's lag ``τ`` (:func:`estimate_lag`). Speed, course
and the logger's ``gps_lat_g`` are handed out **on the position clock**
(their value at ``t + τ``); everything downstream takes them from here.

**Laps keep the dataset's ``lap_num``.** The crossings of the layout's
start/finish gate only place ``s = 0``: a lap is the crossing interval that
coincides with the dataset lap of that number (:func:`lap_table`).

**Position slews.** A receiver losing satellites moves its position
solution by metres within a second; curvature from position then reads
several g. A lap is *flagged* when ``|V²κ_position − gps_lat_g|`` exceeds
:data:`SPIKE_FLAG_G` (both smoothed over 0.5 s, on the position clock, the
position curvature from the extractor's own ``position_kinematics``). A
flagged lap is left out of a track build and counted; nothing is repaired.

**Gyro scale** per lap: the regression of the gyro's integrated yaw on the
Doppler course over the lap, with a free zero offset (:func:`gyro_scale`).

:func:`clean_lap_index` puts the per-lap numbers of every session into one
table with the clean-lap verdict (#202) and the build population (#204).

Conventions: course is a compass heading in radians (clockwise from north,
so it *increases* in a right turn, like the logged lateral g and
``heat_input.share_heat_rate``'s ``lat_pos``), unwrapped within a segment;
``x`` is east and ``y`` north, in metres.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ...tire_etl import dataset
from ...tire_etl.aggregates import channel_series
from ...tire_etl.kinematics import (
    G_MS2,
    LONG_SMOOTH_S,
    MOVING_MS,
    _plausible_fixes,
    position_kinematics,
)
from ...tire_etl.layouts import VENUES, Gate, GpsTrace, Layout, Venue, gate_crossings
from ...tire_etl.paths import laps_dir, sessions_dir, timeseries_dir

logger = logging.getLogger(__name__)

GPS_COLUMNS = ["gps_lat", "gps_lon", "gps_alt_m", "speed_ms", "gps_yaw_rate_dps", "gps_lat_g"]
#: A step between consecutive fixes longer than this cuts the trace (three
#: 40 ms GPS periods, the clean-lap definition's own bound).
GAP_MAX_S = 0.12
_GAP_EPS_S = 1e-4  # timecodes are whole milliseconds
#: Position chords span ± this many samples around the sample they are for.
CHORD_HALF_SPAN = 2
#: Samples slower than this take no part in the lag fit or the course anchor:
#: the chord is too short for its heading to mean anything.
LAG_MIN_SPEED_MS = 8.0
LAG_MAX_YAW_RADS = 1.5  # a spin, not a corner
LAG_MIN_SAMPLES = 200
_LAG_CLIP_FLOOR_RAD = math.radians(0.05)
ANCHOR_MIN_SAMPLES = 25
#: A lap is flagged when position and Doppler lateral g differ by more.
SPIKE_FLAG_G = 1.5
SMOOTH_S = LONG_SMOOTH_S
#: Share of a lap and of a crossing interval that must coincide.
LAP_OVERLAP_MIN = 0.98
#: Half-window around a crossing over which course − chord heading is averaged.
CLOSURE_WINDOW_S = 0.5
#: A recording that stops (or starts) within this distance of the gate line
#: gets its crossing by extrapolation: the logger closes a file on its own
#: beacon, 3–5 m before the registered gate point, so the last lap of a file
#: has no fix beyond the line.
EDGE_CROSSING_MAX_M = 10.0
EDGE_GAP_S = 1.0
EDGE_HALF_WIDTH_M = 15.0
EDGE_MIN_SPEED_MS = 5.0
GYRO_MIN_SAMPLES = 200
GYRO_COVERAGE_TOL_S = 0.1

# Clean lap (#202) and build population (#204).
CLEAN_PACE = 1.12
CLEAN_LENGTH_TOL = 0.02
BUILD_LAP_PACE = 1.02
BUILD_SESSION_PACE = 1.05
REFERENCE_PACE_RANK = 3  # the third-fastest session best
#: A session whose Doppler course and position heading still scatter by more
#: than this once the lag is removed is not a sound receiver for a build.
LAG_SCATTER_MAX_DEG = 0.5
EXCLUDED_LAYOUTS = ("minami",)
INDEX_VERSION = 7

LAYOUTS: dict[str, tuple[Venue, Layout]] = {
    lay.layout_id: (venue, lay) for venue in VENUES.values() for lay in venue.layouts
}


def layout_gate(layout_id: str) -> Gate:
    """The start/finish gate of ``layout_id``; ``KeyError`` when unknown."""
    venue, lay = LAYOUTS[layout_id]
    return venue.gates[lay.sf_gate]


_WGS84_A_M = 6_378_137.0
_WGS84_E2 = 6.69437999014e-3


def local_xy(lat: np.ndarray, lon: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
    """East / north metres from ``(ref_lat, ref_lon)`` on the WGS84
    ellipsoid; shape ``(n, 2)``.

    The plane is tangent at the reference with the ellipsoid's two radii
    of curvature there (meridional for north, prime-vertical for east), so
    a metre is a metre in both directions to parts in 10⁵ over a circuit.
    The extractor's :func:`~...tire_etl.layouts._local_xy` uses one sphere
    radius for both: at 36° N that reads north-south distances 0.21 % long
    and east-west ones 0.23 % short, an anisotropy a curvature profile
    cannot carry (it shows as path length against the integral of speed).
    Height above the ellipsoid is ignored (under 0.01 % at Fuji).
    """
    k = math.pi / 180.0
    sin_phi = math.sin(ref_lat * k)
    w2 = 1.0 - _WGS84_E2 * sin_phi * sin_phi
    r_east = _WGS84_A_M / math.sqrt(w2)  # prime vertical
    r_north = _WGS84_A_M * (1.0 - _WGS84_E2) / (w2 * math.sqrt(w2))  # meridional
    x = (np.asarray(lon, float) - ref_lon) * k * r_east * math.cos(ref_lat * k)
    y = (np.asarray(lat, float) - ref_lat) * k * r_north
    return np.column_stack([x, y])


def local_lat_lon(
    x: np.ndarray, y: np.ndarray, ref_lat: float, ref_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of :func:`local_xy`: latitude and longitude of points
    ``x`` east and ``y`` north of the reference."""
    unit = local_xy(np.array([ref_lat + 1.0]), np.array([ref_lon + 1.0]), ref_lat, ref_lon)
    return ref_lat + np.asarray(y, float) / unit[0, 1], ref_lon + np.asarray(x, float) / unit[0, 0]


def wrap_angle(a: np.ndarray | float) -> np.ndarray:
    """``a`` wrapped into ``[−π, π)``."""
    return np.asarray((np.asarray(a, float) + math.pi) % (2.0 * math.pi) - math.pi)


def _circular_mean(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.angle(np.mean(np.exp(1j * a))))


def _robust_sd(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def nan_window_mean(x: np.ndarray, key: np.ndarray, half_width: float) -> np.ndarray:
    """Mean of the finite ``x`` whose ``key`` (sorted) lies within
    ``± half_width`` of each sample's own key; NaN where there is none."""
    x = np.asarray(x, float)
    fin = np.isfinite(x)
    lo = np.searchsorted(key, key - half_width, side="left")
    hi = np.searchsorted(key, key + half_width, side="right")
    cs = np.concatenate([[0.0], np.cumsum(np.where(fin, x, 0.0))])
    cn = np.concatenate([[0], np.cumsum(fin)])
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan))


# ---------------------------------------------------------------- the course


def doppler_course(t_s: np.ndarray, yaw_rate_dps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(course_rad, segment)`` from the stored GPS yaw rate.

    ``gps_yaw_rate_dps[i]`` is ``(heading[i] − heading[i−1]) / (t[i] −
    t[i−1])``, so the running sum of ``yaw · Δt`` is the heading itself, up
    to a constant. The sum is cut, and ``segment`` incremented, wherever
    the step cannot be trusted: a non-finite yaw rate (libxrk writes NaN on
    a record without a fix and on the next), a non-positive time step, or a
    gap over :data:`GAP_MAX_S` (the heading difference is only known modulo
    a turn). The course carries an unknown constant per segment.
    """
    t = np.asarray(t_s, float)
    w = np.radians(np.asarray(yaw_rate_dps, float))
    if t.size == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    dt = np.diff(t, prepend=t[0])
    ok = np.isfinite(w) & (dt > 0.0) & (dt <= GAP_MAX_S + _GAP_EPS_S)
    inc = np.where(ok, w * dt, 0.0)
    cut = ~ok
    cut[0] = False
    return np.cumsum(inc), np.cumsum(cut).astype(np.int64)


def chord_heading(
    x: np.ndarray, y: np.ndarray, seg: np.ndarray, half_span: int = CHORD_HALF_SPAN
) -> np.ndarray:
    """Compass heading (rad) of the chord from sample ``i − half_span`` to
    ``i + half_span``; NaN where the chord leaves the segment or has no
    length. To second order it is the path's heading at sample ``i``."""
    n = len(x)
    out = np.full(n, np.nan)
    k = half_span
    if n <= 2 * k:
        return out
    dx = x[2 * k :] - x[: -2 * k]
    dy = y[2 * k :] - y[: -2 * k]
    h = np.arctan2(dx, dy)
    h[(seg[2 * k :] != seg[: -2 * k]) | (np.hypot(dx, dy) < 1e-3)] = np.nan
    out[k:-k] = h
    return out


def _central_rate(t: np.ndarray, x: np.ndarray, seg: np.ndarray, half_span: int) -> np.ndarray:
    """``dx/dt`` over the same stencil as :func:`chord_heading`."""
    n = len(x)
    out = np.full(n, np.nan)
    k = half_span
    if n <= 2 * k:
        return out
    dt = t[2 * k :] - t[: -2 * k]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (x[2 * k :] - x[: -2 * k]) / dt
    r[(seg[2 * k :] != seg[: -2 * k]) | ~(dt > 0)] = np.nan
    out[k:-k] = r
    return out


def at_offset(t: np.ndarray, x: np.ndarray, seg: np.ndarray, tau_s: float) -> np.ndarray:
    """``x`` at ``t + tau_s``, linear between the samples either side and NaN
    where that bracket leaves the sample's own segment. ``t`` strictly
    increasing."""
    n = len(t)
    if n < 2:
        return np.full(n, np.nan)
    tq = t + tau_s
    hi = np.clip(np.searchsorted(t, tq, side="left"), 1, n - 1)
    lo = hi - 1
    inside = (tq >= t[lo]) & (tq <= t[hi]) & (seg[lo] == seg[hi]) & (seg[lo] == seg)
    span = t[hi] - t[lo]
    f = np.where(span > 0, (tq - t[lo]) / np.where(span > 0, span, 1.0), 0.0)
    out = np.asarray(x[lo] + f * (x[hi] - x[lo]), dtype=float)
    out[~inside] = np.nan
    return out


@dataclass(frozen=True)
class LagFit:
    """The session's GPS velocity lag behind GPS position."""

    tau_s: float = float("nan")
    #: Robust scatter (deg) of Doppler course − chord heading as logged.
    scatter_raw_deg: float = float("nan")
    #: The same after the course is moved onto the position clock.
    scatter_deg: float = float("nan")
    n: int = 0


def estimate_lag(
    t: np.ndarray,
    course: np.ndarray,
    chord: np.ndarray,
    speed: np.ndarray,
    seg: np.ndarray,
    half_span: int = CHORD_HALF_SPAN,
) -> LagFit:
    """Lag of the Doppler course behind the position heading.

    With the velocity solution ``τ`` late, ``course(t) = heading(t − τ)``
    and ``course − chord ≈ −τ·ω`` plus the segment's constant. The slope is
    fitted with one intercept per segment and four rounds of 4-MAD
    clipping, on samples at :data:`LAG_MIN_SPEED_MS` or more.
    """
    omega = _central_rate(t, course, seg, half_span)
    use = (
        np.isfinite(chord)
        & np.isfinite(omega)
        & np.isfinite(speed)
        & (speed >= LAG_MIN_SPEED_MS)
        & (np.abs(omega) <= LAG_MAX_YAW_RADS)
    )
    if use.sum() < LAG_MIN_SAMPLES:
        return LagFit(n=int(use.sum()))
    idx = np.flatnonzero(use)
    _, group = np.unique(seg[idx], return_inverse=True)
    counts = np.bincount(group)
    big = counts[group] >= ANCHOR_MIN_SAMPLES
    idx, group = idx[big], group[big]
    if idx.size < LAG_MIN_SAMPLES:
        return LagFit(n=int(idx.size))
    d = wrap_angle(course[idx] - chord[idx])
    # Unwrap each segment's difference around its circular mean.
    n_groups = int(group.max()) + 1
    centre = np.array([_circular_mean(d[group == g]) for g in range(n_groups)])
    d = wrap_angle(d - centre[group])
    w = omega[idx]

    def _demean(v: np.ndarray, keep: np.ndarray) -> np.ndarray:
        cnt = np.bincount(group, weights=keep.astype(float), minlength=n_groups)
        tot = np.bincount(group, weights=np.where(keep, v, 0.0), minlength=n_groups)
        return np.asarray(v - (tot / np.maximum(cnt, 1.0))[group])

    keep = np.ones(idx.size, dtype=bool)
    slope = 0.0
    resid = _demean(d, keep)
    raw_sd = _robust_sd(resid)
    fitted = False
    for _ in range(4):
        dc, wc = _demean(d, keep), _demean(w, keep)
        den = float(np.sum(wc[keep] ** 2))
        if den <= 0:
            break
        slope = float(np.sum(wc[keep] * dc[keep]) / den)
        resid = dc - slope * wc
        fitted = True
        # The clip never goes under the heading noise of a good receiver:
        # on a near-noiseless trace the MAD is set by the straights alone
        # and would clip every corner.
        keep = np.abs(resid) < 4.0 * max(_robust_sd(resid[keep]), _LAG_CLIP_FLOOR_RAD)
    if not fitted:
        return LagFit(n=int(idx.size))
    return LagFit(
        tau_s=-slope,
        scatter_raw_deg=math.degrees(raw_sd),
        scatter_deg=math.degrees(_robust_sd(resid)),
        n=int(idx.size),
    )


# --------------------------------------------------------- session geometry


@dataclass
class SessionGeometry:
    """One session's GPS on the position clock (plausible fixes only)."""

    session_id: str
    layout_id: str
    t: np.ndarray  # session time (s), strictly increasing
    x: np.ndarray  # east of the start/finish gate (m)
    y: np.ndarray  # north of it (m)
    alt: np.ndarray  # GPS altitude (m), as logged
    lap_num: np.ndarray
    seg: np.ndarray  # segment id; nothing is continuous across a change
    speed: np.ndarray  # m/s, on the position clock
    course: np.ndarray  # rad, compass, unwrapped within a segment; NaN unanchored
    lat_g_doppler: np.ndarray  # the logger's gps_lat_g on the position clock
    lat_g_position: np.ndarray  # V²κ from position_kinematics, V on the position clock
    discard: np.ndarray  # the extractor's discard mark
    lag: LagFit
    crossings: np.ndarray  # session times (s) of the start/finish crossings
    #: True where the crossing was extrapolated from the last (first) fix of a
    #: recording that stops (starts) just short of (past) the line.
    crossing_edge: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    #: ``lap_num -> (first, last)`` session time of the lap's GPS rows, every row.
    lap_spans: dict[int, tuple[float, float]] = field(default_factory=dict)
    #: Session times of the fixes dropped as implausible.
    t_implausible: np.ndarray = field(default_factory=lambda: np.zeros(0))
    t_discard: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return int(len(self.t))


def _column(tbl: pa.Table, name: str, n: int) -> np.ndarray:
    if name not in tbl.schema.names:
        return np.full(n, np.nan)
    return np.asarray(tbl.column(name).to_numpy(zero_copy_only=False), dtype=float)


def edge_crossings(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    chord: np.ndarray,
    crossings: np.ndarray,
) -> np.ndarray:
    """Crossing times extrapolated at the ends of a recording.

    A crossing is found between two fixes either side of the gate line, so
    a recording that stops on the logger's beacon, a few metres before the
    line, leaves its last lap without one. Where the last fix before a gap
    of :data:`EDGE_GAP_S` (or the end of the data) is within
    :data:`EDGE_CROSSING_MAX_M` behind the line, moving, the crossing is
    put at ``t + distance / speed``; likewise before the first fix after a
    gap when it is just past the line. The gate's direction is the mean
    heading at the crossings already found (none found: no extrapolation).
    ``x, y`` have the gate at the origin.
    """
    if crossings.size == 0 or t.size < 2:
        return np.zeros(0)
    fin = np.flatnonzero(np.isfinite(chord))
    if fin.size == 0:
        return np.zeros(0)
    near = fin[np.clip(np.searchsorted(t[fin], crossings), 0, fin.size - 1)]
    h = _circular_mean(chord[near])
    along = x * math.sin(h) + y * math.cos(h)
    across = x * math.cos(h) - y * math.sin(h)
    gap_after = np.concatenate([np.diff(t) > EDGE_GAP_S, [True]])
    gap_before = np.concatenate([[True], np.diff(t) > EDGE_GAP_S])
    ok = np.isfinite(speed) & (speed >= EDGE_MIN_SPEED_MS) & (np.abs(across) <= EDGE_HALF_WIDTH_M)
    out: list[float] = []
    for i in np.flatnonzero(gap_after & ok & (along < 0.0) & (along >= -EDGE_CROSSING_MAX_M)):
        out.append(float(t[i] - along[i] / speed[i]))
    for i in np.flatnonzero(gap_before & ok & (along >= 0.0) & (along <= EDGE_CROSSING_MAX_M)):
        out.append(float(t[i] - along[i] / speed[i]))
    return np.asarray(sorted(out), dtype=float)


def session_geometry(gps: pa.Table, session_id: str, layout_id: str) -> SessionGeometry | None:
    """Geometry of one session from its ``gps`` family rows
    (:func:`~...tire_etl.dataset.load_group`); None when the session has no
    usable GPS or too little driving for a lag fit."""
    gate = layout_gate(layout_id)
    names = set(gps.schema.names)
    if not {"t_session_s", "gps_lat", "gps_lon", "speed_ms", "gps_yaw_rate_dps"} <= names:
        return None
    n = len(gps)
    if n < LAG_MIN_SAMPLES:
        return None
    t = _column(gps, "t_session_s", n)
    order = np.argsort(t, kind="stable")
    t = t[order]
    lat = _column(gps, "gps_lat", n)[order]
    lon = _column(gps, "gps_lon", n)[order]
    alt = _column(gps, "gps_alt_m", n)[order]
    v = _column(gps, "speed_ms", n)[order]
    yaw = _column(gps, "gps_yaw_rate_dps", n)[order]
    lat_g = _column(gps, "gps_lat_g", n)[order]
    lap = (
        np.asarray(gps.column("lap_num").to_numpy(zero_copy_only=False))[order].astype(np.int64)
        if "lap_num" in names
        else np.zeros(n, dtype=np.int64)
    )
    if "discard" in names:
        disc = np.asarray(gps.column("discard").fill_null(False).to_numpy(zero_copy_only=False))
        disc = disc[order].astype(bool)
    else:
        disc = np.zeros(n, dtype=bool)

    lap_spans: dict[int, tuple[float, float]] = {}
    if n:
        edges = np.flatnonzero(np.concatenate([[True], lap[1:] != lap[:-1], [True]]))
        for a, b in zip(edges[:-1], edges[1:]):
            num = int(lap[a])
            lo, hi = float(t[a]), float(t[b - 1])
            if num in lap_spans:
                lo, hi = min(lo, lap_spans[num][0]), max(hi, lap_spans[num][1])
            lap_spans[num] = (lo, hi)

    course_raw, raw_seg = doppler_course(t, yaw)
    ok = _plausible_fixes(t, lat, lon) & np.isfinite(t)
    ok &= np.concatenate([[True], np.diff(t) > 0])  # strictly increasing time
    t_bad = t[~ok]
    if ok.sum() < LAG_MIN_SAMPLES:
        return None
    t, lat, lon, alt, v, lat_g = t[ok], lat[ok], lon[ok], alt[ok], v[ok], lat_g[ok]
    lap, disc_k, course_raw, raw_seg = lap[ok], disc[ok], course_raw[ok], raw_seg[ok]
    cut = (np.diff(raw_seg) != 0) | (np.diff(t) > GAP_MAX_S + _GAP_EPS_S)
    seg = np.concatenate([[0], np.cumsum(cut)]).astype(np.int64)

    xy = local_xy(lat, lon, gate.lat, gate.lon)
    x, y = xy[:, 0], xy[:, 1]
    chord = chord_heading(x, y, seg)
    lag = estimate_lag(t, course_raw, chord, v, seg)
    if not np.isfinite(lag.tau_s):
        return None
    speed = at_offset(t, v, seg, lag.tau_s)
    course = at_offset(t, course_raw, seg, lag.tau_s)
    lat_g_dop = at_offset(t, lat_g, seg, lag.tau_s)

    # Anchor each segment's constant to the position heading.
    use = np.isfinite(chord) & np.isfinite(course) & (speed >= LAG_MIN_SPEED_MS)
    diff = wrap_angle(course - chord)
    seg_ids, group = np.unique(seg, return_inverse=True)
    anchored = np.zeros(len(t), dtype=bool)
    for g in np.unique(group[use]):
        m = group == g
        mu = use & m
        if mu.sum() < ANCHOR_MIN_SAMPLES:
            continue
        course[m] -= _circular_mean(diff[mu])
        anchored |= m
    course[~anchored] = np.nan

    a_pos, _, _ = position_kinematics(t, lat, lon, np.nan_to_num(speed, nan=0.0))

    trace = GpsTrace(t_ms=np.round(t * 1000.0).astype(np.int64), lat=lat, lon=lon)
    keep = np.concatenate([[True], np.diff(trace.t_ms) > 0])
    trace = GpsTrace(t_ms=trace.t_ms[keep], lat=lat[keep], lon=lon[keep])
    crossings = np.asarray(gate_crossings(trace, gate) / 1000.0, dtype=float)
    edge = edge_crossings(t, x, y, v, chord, crossings)
    # an extrapolated crossing never duplicates a measured one
    edge = np.asarray([e for e in edge if not np.any(np.abs(crossings - e) < EDGE_GAP_S)])
    all_cross = np.concatenate([crossings, edge])
    is_edge = np.concatenate([np.zeros(crossings.size, bool), np.ones(edge.size, bool)])
    order_c = np.argsort(all_cross, kind="stable")

    return SessionGeometry(
        session_id=session_id,
        layout_id=layout_id,
        t=t,
        x=x,
        y=y,
        alt=alt,
        lap_num=lap,
        seg=seg,
        speed=speed,
        course=course,
        lat_g_doppler=lat_g_dop,
        lat_g_position=np.asarray(a_pos, dtype=float),
        discard=disc_k,
        lag=lag,
        crossings=all_cross[order_c],
        crossing_edge=is_edge[order_c],
        lap_spans=lap_spans,
        t_implausible=t_bad,
        t_discard=np.asarray(gps.column("t_session_s").to_numpy(zero_copy_only=False), float)[
            order
        ][disc],
    )


def load_session_geometry(root: Path, session_id: str, layout_id: str) -> SessionGeometry | None:
    """:func:`session_geometry` of a dataset session; None without a
    timeseries file, a GPS family or a known layout."""
    if layout_id not in LAYOUTS:
        return None
    gps = dataset.load_group(root, session_id, "gps", GPS_COLUMNS)
    if gps is None:
        return None
    return session_geometry(gps, session_id, layout_id)


# ----------------------------------------------------------------- the laps


def match_laps(geom: SessionGeometry) -> dict[int, tuple[float, float, float]]:
    """``lap_num -> (t_start, t_end, overlap)``: the crossing interval that
    coincides with each dataset lap. ``overlap`` is the interval's and the
    lap's common time over the longer of the two; a lap appears only when
    it reaches :data:`LAP_OVERLAP_MIN`."""
    tc = geom.crossings
    out: dict[int, tuple[float, float, float]] = {}
    if tc.size < 2:
        return out
    for num, (lo, hi) in geom.lap_spans.items():
        if hi <= lo:
            continue
        k = int(np.searchsorted(tc, 0.5 * (lo + hi))) - 1
        if k < 0 or k + 1 >= tc.size:
            continue
        a, b = float(tc[k]), float(tc[k + 1])
        common = min(hi, b) - max(lo, a)
        overlap = common / max(hi - lo, b - a)
        if overlap >= LAP_OVERLAP_MIN:
            out[num] = (a, b, overlap)
    return out


def _interp_in(t: np.ndarray, x: np.ndarray, at: float) -> float:
    return float(np.interp(at, t, x))


def _value_at(geom: SessionGeometry, arr: np.ndarray, at: float) -> float:
    """``arr`` at session time ``at``: linear between the fixes either side
    when they belong to one segment, else the nearest finite sample within
    :data:`CLOSURE_WINDOW_S` (an extrapolated crossing lies past the last
    fix of its recording, and the lag shift leaves the last samples of a
    segment without speed or course)."""
    t = geom.t
    j = int(np.searchsorted(t, at))
    if 0 < j < len(t) and geom.seg[j - 1] == geom.seg[j]:
        span = t[j] - t[j - 1]
        f = (at - t[j - 1]) / span if span > 0 else 0.0
        val = float(arr[j - 1] + f * (arr[j] - arr[j - 1]))
        if np.isfinite(val):
            return val
    i0, i1 = np.searchsorted(t, [at - CLOSURE_WINDOW_S, at + CLOSURE_WINDOW_S])
    idx = np.arange(i0, i1)
    idx = idx[np.isfinite(arr[idx])]
    if idx.size == 0:
        return float("nan")
    return float(arr[idx[np.argmin(np.abs(t[idx] - at))]])


def _offset_at(geom: SessionGeometry, chord: np.ndarray, at: float) -> float:
    """Circular mean of course − chord heading within
    :data:`CLOSURE_WINDOW_S` of ``at``."""
    i0, i1 = np.searchsorted(geom.t, [at - CLOSURE_WINDOW_S, at + CLOSURE_WINDOW_S])
    return _circular_mean(wrap_angle(geom.course[i0:i1] - chord[i0:i1]))


LAP_TABLE_COLUMNS = [
    "lap_num",
    "matched",
    "edge",
    "overlap",
    "t_start_s",
    "t_end_s",
    "length_m",
    "v_integral_m",
    "turns",
    "closure",
    "one_segment",
    "n_implausible",
    "max_gap_s",
    "has_discard",
    "min_speed_ms",
    "spike_g",
    "spike_s_m",
    "peak_position_g",
    "peak_doppler_g",
]


def lap_table(geom: SessionGeometry) -> pd.DataFrame:
    """One row per dataset lap of the session (:data:`LAP_TABLE_COLUMNS`).

    ``length_m`` is the XY path between the two crossings and
    ``v_integral_m`` the integral of speed over the same time. ``turns`` is
    the course change in whole turns and ``closure`` what is left of it
    once the course at both ends is referred to the position heading there
    (the car does not cross the line at the same angle every lap, so the
    raw course change differs from a whole number of turns by a degree or
    two; the Doppler sum itself loses nothing when ``closure`` is zero).
    ``edge`` marks a lap one of whose crossings was extrapolated
    (:func:`edge_crossings`): its trace stops at the last fix, a few metres
    short of the line. ``spike_g`` is the largest ``|V²κ_position −
    gps_lat_g|`` of the lap and ``spike_s_m`` where along the lap it is,
    the two peaks the largest ``|lateral g|`` of each, all smoothed over
    :data:`SMOOTH_S` on samples at :data:`~...tire_etl.kinematics.MOVING_MS`
    or more.
    """
    t = geom.t
    matched = match_laps(geom)
    step = np.hypot(np.diff(geom.x), np.diff(geom.y))
    path = np.concatenate([[0.0], np.cumsum(step)])
    v = np.nan_to_num(geom.speed, nan=0.0)
    v_int = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    chord = chord_heading(geom.x, geom.y, geom.seg)
    half = SMOOTH_S / 2.0
    pos_s = nan_window_mean(geom.lat_g_position, t, half)
    dop_s = nan_window_mean(geom.lat_g_doppler, t, half)
    moving = np.isfinite(geom.speed) & (geom.speed >= MOVING_MS)
    rows: list[dict] = []
    for num in sorted(geom.lap_spans):
        row: dict = {c: np.nan for c in LAP_TABLE_COLUMNS}
        row.update(lap_num=int(num), matched=False, one_segment=False, has_discard=False)
        row["edge"] = False
        row["n_implausible"] = 0
        hit = matched.get(num)
        if hit is None:
            rows.append(row)
            continue
        a, b, overlap = hit
        i0, i1 = (int(i) for i in np.searchsorted(t, [a, b]))
        lo, hi = max(i0 - 1, 0), min(i1, len(t) - 1)
        row.update(matched=True, overlap=overlap, t_start_s=a, t_end_s=b)
        row["edge"] = bool(
            np.any(geom.crossing_edge[np.isin(geom.crossings, [a, b])])
            if geom.crossing_edge.size
            else False
        )
        row["length_m"] = _interp_in(t, path, b) - _interp_in(t, path, a)
        row["v_integral_m"] = _interp_in(t, v_int, b) - _interp_in(t, v_int, a)
        row["one_segment"] = bool(geom.seg[lo] == geom.seg[hi])
        row["n_implausible"] = int(np.sum((geom.t_implausible >= a) & (geom.t_implausible <= b)))
        row["max_gap_s"] = float(np.max(np.diff(t[lo : hi + 1]))) if hi > lo else np.nan
        row["has_discard"] = bool(np.any((geom.t_discard >= a) & (geom.t_discard <= b)))
        sl = slice(i0, i1)
        if i1 > i0:
            row["min_speed_ms"] = (
                float(np.nanmin(geom.speed[sl])) if np.isfinite(geom.speed[sl]).any() else np.nan
            )
            m = moving[sl]
            if m.any():
                with np.errstate(invalid="ignore"):
                    diff = np.abs(pos_s[sl] - dop_s[sl])[m]
                    if np.isfinite(diff).any():
                        at = i0 + int(np.flatnonzero(m)[int(np.nanargmax(diff))])
                        row["spike_g"] = float(np.nanmax(diff))
                        row["spike_s_m"] = float(path[at] - _interp_in(t, path, a))
                    row["peak_position_g"] = float(np.nanmax(np.abs(pos_s[sl][m])))
                    row["peak_doppler_g"] = (
                        float(np.nanmax(np.abs(dop_s[sl][m])))
                        if np.isfinite(dop_s[sl][m]).any()
                        else np.nan
                    )
        if row["one_segment"]:
            c0, c1 = _value_at(geom, geom.course, a), _value_at(geom, geom.course, b)
            off0, off1 = _offset_at(geom, chord, a), _offset_at(geom, chord, b)
            if np.isfinite([c0, c1, off0, off1]).all():
                row["turns"] = float(np.round((c1 - c0) / (2.0 * math.pi)))
                row["closure"] = float(wrap_angle(off1 - off0) / (2.0 * math.pi))
        rows.append(row)
    return pd.DataFrame(rows, columns=LAP_TABLE_COLUMNS)


@dataclass
class LapTrace:
    """One lap from start/finish crossing to the next, on the position clock."""

    lap_num: int
    t: np.ndarray  # session time (s); first and last are the crossings
    s: np.ndarray  # XY path from the crossing (m)
    x: np.ndarray
    y: np.ndarray
    alt: np.ndarray
    speed: np.ndarray
    course: np.ndarray

    @property
    def length_m(self) -> float:
        return float(self.s[-1])

    @property
    def lap_time_s(self) -> float:
        return float(self.t[-1] - self.t[0])


def lap_trace(geom: SessionGeometry, lap_num: int) -> LapTrace | None:
    """The samples of ``lap_num`` between its two crossings, with the
    crossing instants themselves interpolated in as the first and last
    points; None when the lap matches no crossing interval."""
    hit = match_laps(geom).get(int(lap_num))
    if hit is None:
        return None
    a, b, _ = hit
    t = geom.t
    i0, i1 = (int(i) for i in np.searchsorted(t, [a, b]))
    inner = slice(i0, i1)
    tt = np.concatenate([[a], t[inner], [b]])
    keep = np.concatenate([[True], np.diff(tt) > 0])

    def _with_ends(arr: np.ndarray) -> np.ndarray:
        full = np.concatenate([[_value_at(geom, arr, a)], arr[inner], [_value_at(geom, arr, b)]])
        return np.asarray(full[keep])

    x, y = _with_ends(geom.x), _with_ends(geom.y)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return LapTrace(
        lap_num=int(lap_num),
        t=tt[keep],
        s=s,
        x=x,
        y=y,
        alt=_with_ends(geom.alt),
        speed=_with_ends(geom.speed),
        course=_with_ends(geom.course),
    )


def gyro_scale(
    geom: SessionGeometry,
    t_gyro: np.ndarray,
    yaw_gyro_dps: np.ndarray,
    t_start: float,
    t_end: float,
) -> float:
    """Scale of the gyro against the Doppler course over ``[t_start, t_end]``.

    The gyro's yaw rate is integrated on its own clock and regressed on the
    course with a free zero offset: ``∫ω_gyro dt = a·course + b·t + c``.
    ``a`` is the scale (a gyro reading 0.88× returns 0.88). The lap is not
    treated as closed, so the gyro's zero offset, which a lap's closure
    would turn into a scale error, goes to ``b``. NaN without enough
    samples or where the gyro has a gap in the lap.
    """
    tg = np.asarray(t_gyro, float)
    wg = np.asarray(yaw_gyro_dps, float)
    okg = np.isfinite(tg) & np.isfinite(wg)
    tg, wg = tg[okg], wg[okg]
    if tg.size < GYRO_MIN_SAMPLES:
        return float("nan")
    order = np.argsort(tg, kind="stable")
    tg, wg = tg[order], np.radians(wg[order])
    i0, i1 = np.searchsorted(tg, [t_start, t_end])
    if i1 - i0 < GYRO_MIN_SAMPLES or tg[i0] - t_start > 0.5 or t_end - tg[i1 - 1] > 0.5:
        return float("nan")
    tg, wg = tg[i0:i1], wg[i0:i1]
    if np.max(np.diff(tg)) > GYRO_COVERAGE_TOL_S + _GAP_EPS_S:
        return float("nan")
    theta = np.concatenate([[0.0], np.cumsum(0.5 * (wg[1:] + wg[:-1]) * np.diff(tg))])
    j0, j1 = np.searchsorted(geom.t, [tg[0], tg[-1]])
    tp, cp = geom.t[j0:j1], geom.course[j0:j1]
    ok = np.isfinite(cp)
    if ok.sum() < GYRO_MIN_SAMPLES or len(np.unique(geom.seg[j0:j1])) != 1:
        return float("nan")
    tp, cp = tp[ok], cp[ok]
    th = np.interp(tp, tg, theta)
    tc = tp - tp.mean()
    design = np.column_stack([cp - cp.mean(), tc, np.ones(tp.size)])
    coef, *_ = np.linalg.lstsq(design, th, rcond=None)
    return float(coef[0])


# ------------------------------------------------------------ the lap index


def load_sessions(root: Path) -> pd.DataFrame:
    """The dataset's session rows, one frame."""
    files = sorted(sessions_dir(root).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no session partitions under {root}")
    tables = [pq.read_table(f) for f in files]
    out: pd.DataFrame = pa.concat_tables(tables, promote_options="default").to_pandas()
    return out


def load_laps(root: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """The dataset's lap rows (``columns`` of them), one frame."""
    files = sorted(laps_dir(root).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no lap partitions under {root}")
    tables = [pq.read_table(f, columns=columns) for f in files]
    out: pd.DataFrame = pa.concat_tables(tables, promote_options="default").to_pandas()
    return out


def session_conditions(root: Path, sessions: pd.DataFrame, laps: pd.DataFrame) -> pd.Series:
    """``session_id -> dry / damp / wet / unknown`` from the production
    wetness model (the surface water balance over the hourly weather)."""
    from ..warmup_table import _load_weather
    from ..wetness import classify_wetness, session_wetness, surface_water_series

    weather = _load_weather(root)
    out: dict[str, str] = {}
    depth = (
        {str(tr): surface_water_series(grp) for tr, grp in weather.groupby("track_canonical")}
        if not weather.empty
        else {}
    )
    duration = (
        laps.groupby("session_id")["on_track_s"].sum() if len(laps) else pd.Series(dtype=float)
    )
    starts = pd.to_datetime(sessions["session_start_utc"], utc=True)
    for sid, track, start in zip(sessions["session_id"], sessions["track_canonical"], starts):
        series = depth.get(str(track))
        if series is None or pd.isna(start):
            out[str(sid)] = "unknown"
            continue
        feats = session_wetness(series, start, float(duration.get(sid, 0.0)))
        out[str(sid)] = classify_wetness(feats["depth_run_max_mm"])
    return pd.Series(out, name="condition")


def _reference_pace(bests: np.ndarray) -> float:
    """The third-fastest of a car's session bests at a layout; the fastest
    when it has fewer than three sessions there."""
    b = np.sort(bests[np.isfinite(bests)])
    if b.size == 0:
        return float("nan")
    return float(b[REFERENCE_PACE_RANK - 1] if b.size >= REFERENCE_PACE_RANK else b[0])


def classify_laps(index: pd.DataFrame) -> pd.DataFrame:
    """Add the clean-lap verdict and the build population to a lap index.

    Needs per lap ``session_id, car, layout_id, lap_type, lap_time_s,
    matched, one_segment, n_implausible, max_gap_s, has_discard,
    min_speed_ms, length_m, spike_g, lag_scatter_deg``. Adds:

    - ``clean`` and ``clean_reason`` (the first test the lap fails, in the
      order of #202's definition: ``not_full``, ``layout`` (an excluded
      layout), ``no_interval``, ``implausible_fix``, ``gap``, ``discard``,
      ``slow`` (a sample under 3 m/s), ``length`` (outside 2 % of the
      layout median), ``pace`` (over 1.12 × the session best));
    - ``session_best_s``: the fastest lap of the session that passes every
      test but the pace one;
    - ``flagged``: the position-slew flag (:data:`SPIKE_FLAG_G`);
    - ``reference_pace_s``: the car's third-fastest session best at the
      layout;
    - ``in_build`` and ``build_reason`` (``not_clean``, ``flagged``,
      ``receiver`` (the session's lag scatter is over
      :data:`LAG_SCATTER_MAX_DEG`), ``slow_session``, ``slow_lap``).

    The build does not look at ``condition``. The road is the same wet or
    dry; a session that was wet is slow and falls to the pace rule, and
    the weather model's label (unknown on many sessions, and "wet" on
    sessions that had dried) would only remove laps driven at pace.
    """
    df = index.copy()
    reason = pd.Series("", index=df.index, dtype=object)

    def _fail(mask: pd.Series, why: str) -> None:
        reason.loc[mask.fillna(False).astype(bool) & (reason == "")] = why

    _fail(df["lap_type"] != "full", "not_full")
    _fail(df["layout_id"].isin(EXCLUDED_LAYOUTS) | df["layout_id"].isna(), "layout")
    _fail(~df["matched"].astype(bool), "no_interval")
    _fail(df["n_implausible"] > 0, "implausible_fix")
    _fail(~df["one_segment"].astype(bool) | ~(df["max_gap_s"] <= GAP_MAX_S + _GAP_EPS_S), "gap")
    _fail(df["has_discard"].astype(bool), "discard")
    _fail(~(df["min_speed_ms"] >= MOVING_MS), "slow")
    cand = reason == ""
    median_len = df[cand].groupby("layout_id")["length_m"].median()
    df["layout_length_m"] = df["layout_id"].map(median_len)
    _fail(~((df["length_m"] / df["layout_length_m"] - 1.0).abs() <= CLEAN_LENGTH_TOL), "length")
    cand = reason == ""
    best = df[cand].groupby("session_id")["lap_time_s"].min()
    df["session_best_s"] = df["session_id"].map(best)
    _fail(~(df["lap_time_s"] <= CLEAN_PACE * df["session_best_s"]), "pace")
    df["clean_reason"] = reason
    df["clean"] = reason == ""
    df["flagged"] = df["spike_g"] > SPIKE_FLAG_G

    per_session = (
        df[df["clean"]]
        .groupby(["car", "layout_id", "session_id"], as_index=False)["session_best_s"]
        .first()
    )
    ref = per_session.groupby(["car", "layout_id"])["session_best_s"].apply(
        lambda s: _reference_pace(s.to_numpy(dtype=float))
    )
    keys = pd.MultiIndex.from_frame(df[["car", "layout_id"]])
    df["reference_pace_s"] = ref.reindex(keys).to_numpy() if len(ref) else np.nan

    build = pd.Series("", index=df.index, dtype=object)

    def _out(mask: pd.Series, why: str) -> None:
        build.loc[mask.fillna(False).astype(bool) & (build == "")] = why

    _out(~df["clean"], "not_clean")
    _out(df["flagged"], "flagged")
    _out(~(df["lag_scatter_deg"] <= LAG_SCATTER_MAX_DEG), "receiver")
    _out(~(df["session_best_s"] <= BUILD_SESSION_PACE * df["reference_pace_s"]), "slow_session")
    _out(~(df["lap_time_s"] <= BUILD_LAP_PACE * df["session_best_s"]), "slow_lap")
    df["build_reason"] = build
    df["in_build"] = build == ""
    return df


def session_lap_rows(root: Path, session_id: str, layout_id: str) -> pd.DataFrame | None:
    """:func:`lap_table` of one dataset session plus ``lag_s``, the lag
    scatters and ``gyro_scale`` per lap (the raw gyro, whatever the
    session's ``gyro_source``); None without geometry."""
    geom = load_session_geometry(root, session_id, layout_id)
    if geom is None:
        return None
    rows = lap_table(geom)
    rows["lag_s"] = geom.lag.tau_s
    rows["lag_scatter_raw_deg"] = geom.lag.scatter_raw_deg
    rows["lag_scatter_deg"] = geom.lag.scatter_deg
    rows["n_crossings"] = int(geom.crossings.size)
    rows["gyro_scale"] = np.nan
    gyro = dataset.load_group(root, session_id, "gyro", ["yaw_rate_dps"])
    if gyro is not None:
        tg, wg = channel_series(gyro, "yaw_rate_dps", "t_session_s")
        if tg.size >= GYRO_MIN_SAMPLES:
            order = np.argsort(tg, kind="stable")
            tg, wg = tg[order], wg[order]
            for i, row in rows.iterrows():
                if row["matched"]:
                    rows.at[i, "gyro_scale"] = gyro_scale(
                        geom, tg, wg, float(row["t_start_s"]), float(row["t_end_s"])
                    )
    return rows


def _index_signature(root: Path) -> str:
    h = hashlib.sha256(f"v{INDEX_VERSION}".encode())
    files = (
        sorted(sessions_dir(root).glob("*.parquet"))
        + sorted(laps_dir(root).glob("*.parquet"))
        + sorted(timeseries_dir(root).glob("*/*.parquet"))
    )
    for f in files:
        st = f.stat()
        h.update(f"{f.name}:{st.st_mtime_ns}:{st.st_size};".encode())
    return h.hexdigest()


_INDEX_CACHE: dict[str, pd.DataFrame] = {}
_LAP_COLUMNS = ["session_id", "lap_num", "stint_id", "lap_time_s", "lap_type", "on_track_s"]
_SESSION_COLUMNS = [
    "session_id",
    "date",
    "car",
    "driver",
    "logger_id",
    "layout_id",
    "track_canonical",
    "lap_length_m",
    "lap_source",
    "accel_source",
    "gyro_source",
]


def clean_lap_index(root: Path, *, cache_dir: Path | None = None) -> pd.DataFrame:
    """Every lap of every session with a known layout: the per-lap geometry
    numbers (:func:`session_lap_rows`), the session's car, layout, date,
    logger and condition, and the verdicts of :func:`classify_laps`
    (``clean``, ``flagged``, ``in_build`` and their reasons).

    Reading every session's GPS takes minutes, so the per-lap numbers are
    cached: in the process, and in ``cache_dir`` (a parquet file keyed by
    the dataset's file list) when one is given. The verdicts are recomputed
    on every call.
    """
    root = Path(root)
    sig = _index_signature(root)
    key = f"{root.resolve()}:{sig}"
    raw = _INDEX_CACHE.get(key)
    cache_file = Path(cache_dir) / "lap_index.parquet" if cache_dir is not None else None
    if raw is None and cache_file is not None and cache_file.exists():
        meta = pq.read_schema(cache_file).metadata or {}
        if meta.get(b"lapsim.signature", b"").decode() == sig:
            raw = pq.read_table(cache_file).to_pandas()
    if raw is None:
        raw = _build_index(root)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tbl = pa.Table.from_pandas(raw, preserve_index=False)
            tbl = tbl.replace_schema_metadata(
                {**(tbl.schema.metadata or {}), b"lapsim.signature": sig.encode()}
            )
            pq.write_table(tbl, cache_file)
    _INDEX_CACHE[key] = raw
    return classify_laps(raw)


def _build_index(root: Path) -> pd.DataFrame:
    sessions = load_sessions(root)
    laps = load_laps(root, _LAP_COLUMNS)
    sessions = sessions[sessions["status"] != "error"]
    cond = session_conditions(root, sessions, laps)
    parts: list[pd.DataFrame] = []
    for sess in sessions.sort_values(["date", "session_id"]).itertuples(index=False):
        sid = str(sess.session_id)
        layout_id = sess.layout_id if isinstance(sess.layout_id, str) else None
        own = laps[laps["session_id"] == sid][["lap_num", "stint_id", "lap_time_s", "lap_type"]]
        if own.empty:
            continue
        geo: pd.DataFrame | None = None
        if layout_id in LAYOUTS:
            try:
                geo = session_lap_rows(root, sid, str(layout_id))
            except Exception:  # noqa: BLE001 - one unreadable session must not stop the index
                logger.warning("lap index: geometry failed for %s", sid, exc_info=True)
        if geo is None:
            geo = pd.DataFrame({"lap_num": own["lap_num"].astype(int)})
        part = own.merge(geo, on="lap_num", how="left")
        part.insert(0, "session_id", sid)
        for col in _SESSION_COLUMNS[1:]:
            part[col] = getattr(sess, col, None)
        part["condition"] = cond.get(sid, "unknown")
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    for col in LAP_TABLE_COLUMNS + ["lag_s", "lag_scatter_raw_deg", "lag_scatter_deg"]:
        if col not in out.columns:
            out[col] = np.nan
    for col in ("matched", "edge", "one_segment", "has_discard"):
        out[col] = out[col].fillna(False).astype(bool)
    out["n_implausible"] = out["n_implausible"].fillna(0).astype(int)
    if "n_crossings" not in out.columns:
        out["n_crossings"] = 0
    out["n_crossings"] = out["n_crossings"].fillna(0).astype(int)
    if "gyro_scale" not in out.columns:
        out["gyro_scale"] = np.nan
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    logger.info("lap index: %d laps in %d sessions", len(out), out["session_id"].nunique())
    return out


def index_summary_json(index: pd.DataFrame) -> str:
    """Counts of the index's verdicts (for logs and reports)."""
    return json.dumps(
        {
            "laps": int(len(index)),
            "clean": int(index["clean"].sum()),
            "flagged_clean": int((index["clean"] & index["flagged"]).sum()),
            "in_build": int(index["in_build"].sum()),
            "clean_reasons": index["clean_reason"].value_counts().to_dict(),
            "build_reasons": index["build_reason"].value_counts().to_dict(),
        }
    )


__all__ = [
    "LAYOUTS",
    "LagFit",
    "LapTrace",
    "SessionGeometry",
    "at_offset",
    "chord_heading",
    "classify_laps",
    "clean_lap_index",
    "doppler_course",
    "estimate_lag",
    "gyro_scale",
    "lap_table",
    "lap_trace",
    "layout_gate",
    "load_laps",
    "load_session_geometry",
    "load_sessions",
    "local_lat_lon",
    "local_xy",
    "match_laps",
    "session_conditions",
    "session_geometry",
    "session_lap_rows",
    "wrap_angle",
]
