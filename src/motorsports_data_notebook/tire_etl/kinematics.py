"""IMU validation against GPS and the per-session sensor verdict.

The logger's inertial channels are not always trustworthy: a miscalibrated
IMU reads lateral g at half the true value, a swapped harness flips or
twists an axis, a dead sensor is flat. The extractor therefore compares each
candidate series with a **reference derived from GPS position and speed** on
the session's fastest full lap and records **one verdict per session** for
the accelerometer (``accel_source``: both axes together) and one for the
gyro (``gyro_source``), with every regression number, on the session row and
in the timeseries file's metadata. Fallback order: IMU → the logger's own
GPS-derived channels (validated the same way) → series derived here from
position and speed. The raw channels are stored untouched; the verdict is
applied by :func:`load_kinematics`, which puts speed and the chosen
lateral / longitudinal / yaw series on one clock for the consumers.

Reference (right-positive lateral, forward-positive longitudinal, right-
positive yaw rate, matching the logged ``LateralAcc`` convention)::

    κ      = signed curvature of the ~10 m smoothed local-XY path
    a_lat  = V² κ            a_long = dV/dt (0.5 s centred)      ω = V κ

Checks per candidate axis (moving samples, V ≥ 3 m/s, of the reference lap;
the candidate smoothed over the same 0.5 s window as the reference before
the comparison): has samples (coverage ≥ 90 %), not flat (std ≥ 0.1 g /
5 °/s), in family (OLS slope ∈ [0.85, 1.15] for the accelerometer,
[0.9, 1.1] for the gyro; corr ≥ 0.9 lateral and yaw / ≥ 0.85 longitudinal;
|offset| ≤ 0.4 g / 10 °/s) and not twisted (correlation with the *other*
axis' reference ≤ 0.5). The audit also reports the longitudinal regression
on braking samples only (reference < −0.2 g), the A2 gate's definition.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .aggregates import channel_series
from .layouts import _local_xy

logger = logging.getLogger(__name__)

G_MS2 = 9.80665
XY_SMOOTH_M = 10.0  # total window of the position smoothing (± half)
PATH_GRID_M = 1.0  # the path is resampled onto this arc-length grid for the curvature
LONG_SMOOTH_S = 0.5  # centred window for dV/dt
# Fix plausibility: a receiver acquiring its fix reports positions anywhere
# on the globe (lat 45, lon ±179 at zero speed for the first seconds of a
# file); without a filter the arc-length grid below would span the planet.
MAX_FIX_RADIUS_M = 20_000.0  # from the session's median position; a circuit is < 10 km across
MAX_FIX_JUMP_M = 100.0  # a sample both of whose neighbouring steps exceed this is a spike
MAX_PATH_POINTS = 5_000_000  # hard cap on the curvature grid (5000 km at 1 m)
MOVING_MS = 3.0
COVERAGE_MIN = 0.9
COVERAGE_TOL_S = 0.5
STD_MIN = {"g": 0.1, "dps": 5.0}
SLOPE_BAND = (0.85, 1.15)
# The gyro gets the A2 gate's own band: a 0.87× yaw scale (the Inferno 86
# since 2025-06-09, the SFJ) is a calibration fault the position-derived
# yaw does not have, where a 0.9× lateral g is still the better sensor.
SLOPE_BAND_YAW = (0.9, 1.1)
CORR_MIN = 0.9
# dV/dt from 25 Hz GPS speed is the noisier reference: a healthy IMU's
# longitudinal axis correlates 0.87-0.98 with it where the lateral axis
# reaches 0.96-0.99, so the longitudinal check uses the A2 gate's own bound.
CORR_MIN_LONG = 0.85
OFFSET_MAX = {"g": 0.4, "dps": 10.0}
CROSS_CORR_MAX = 0.5
BRAKE_REF_G = -0.2
MIN_BRAKE_SAMPLES = 50
MIN_FIT_SAMPLES = 20

ACCEL_SOURCES = ("imu", "gps_channel", "gps_position", "none")
GYRO_SOURCES = ("imu", "gps_position", "none")
AXES = ("lat", "long", "yaw")
CANDIDATES = {"imu", "gps"}
CANDIDATE_COLUMNS = {
    ("imu", "lat"): "lat_g",
    ("imu", "long"): "long_g",
    ("imu", "yaw"): "yaw_rate_dps",
    ("gps", "lat"): "gps_lat_g",
    ("gps", "long"): "gps_long_g",
    ("gps", "yaw"): "gps_yaw_rate_dps",
}
METRICS = ("slope", "corr", "offset", "coverage", "std", "cross")
# Session-row columns the verdict contributes (all nullable).
KIN_ROW_COLUMNS: list[tuple[str, pa.DataType]] = (
    [
        ("accel_source", pa.string()),
        ("gyro_source", pa.string()),
        ("kin_ref_lap", pa.int32()),
        ("kin_note", pa.string()),
    ]
    + [
        (f"kin_{cand}_{axis}_{m}", pa.float32())
        for cand in ("imu", "gps")
        for axis in AXES
        for m in METRICS
    ]
    + [(f"kin_{cand}_{axis}_ok", pa.bool_()) for cand in ("imu", "gps") for axis in AXES]
)


@dataclass
class AxisCheck:
    """Regression of one candidate axis on its reference."""

    candidate: str
    axis: str
    n: int = 0
    coverage: float = float("nan")
    std: float = float("nan")
    slope: float = float("nan")
    corr: float = float("nan")
    offset: float = float("nan")
    cross: float = float("nan")
    passed: bool = False
    reason: str = "no_samples"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class KinematicsVerdict:
    accel_source: str = "none"
    gyro_source: str = "none"
    ref_lap: int | None = None
    note: str = ""
    checks: dict[str, AxisCheck] = field(default_factory=dict)

    # -- serialisation ---------------------------------------------------
    def to_json(self) -> dict:
        return {
            "accel_source": self.accel_source,
            "gyro_source": self.gyro_source,
            "ref_lap": self.ref_lap,
            "note": self.note,
            "checks": {k: v.as_dict() for k, v in self.checks.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> "KinematicsVerdict":
        checks = {k: AxisCheck(**v) for k, v in (data.get("checks") or {}).items()}
        return cls(
            accel_source=str(data.get("accel_source", "none")),
            gyro_source=str(data.get("gyro_source", "none")),
            ref_lap=data.get("ref_lap"),
            note=str(data.get("note", "")),
            checks=checks,
        )

    def to_row(self) -> dict:
        """The session-row columns (:data:`KIN_ROW_COLUMNS`)."""
        row: dict = {
            "accel_source": self.accel_source,
            "gyro_source": self.gyro_source,
            "kin_ref_lap": self.ref_lap,
            "kin_note": self.note or None,
        }
        for cand in ("imu", "gps"):
            for axis in AXES:
                chk = self.checks.get(f"{cand}_{axis}")
                for m in METRICS:
                    v = getattr(chk, m) if chk is not None else float("nan")
                    row[f"kin_{cand}_{axis}_{m}"] = None if v is None or np.isnan(v) else float(v)
                row[f"kin_{cand}_{axis}_ok"] = bool(chk.passed) if chk is not None else None
        return row

    def source_for(self, axis: str) -> str:
        return self.gyro_source if axis == "yaw" else self.accel_source


def empty_verdict_row() -> dict:
    """Session-row columns of a session without a verdict (all null)."""
    return {name: None for name, _ in KIN_ROW_COLUMNS}


# ------------------------------------------------------------------ reference


def _window_mean(x: np.ndarray, key: np.ndarray, half_width: float) -> np.ndarray:
    """Mean of ``x`` over the samples whose ``key`` (sorted) lies within
    ``± half_width`` of each sample's own key."""
    lo = np.searchsorted(key, key - half_width, side="left")
    hi = np.searchsorted(key, key + half_width, side="right")
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return np.asarray((cs[hi] - cs[lo]) / np.maximum(hi - lo, 1))


def _plausible_fixes(t_s: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Mask of the GPS samples worth projecting: finite, not the (0, 0)
    no-fix value, within :data:`MAX_FIX_RADIUS_M` of the session's median
    position (a receiver still acquiring reports positions anywhere on the
    globe) and not an isolated spike (both neighbouring steps above
    :data:`MAX_FIX_JUMP_M`)."""
    ok = np.asarray(np.isfinite(lat) & np.isfinite(lon) & (lat != 0.0) & (lon != 0.0), bool)
    if ok.sum() < 5:
        return ok
    ref_lat = float(np.median(lat[ok]))
    ref_lon = float(np.median(lon[ok]))
    xy = _local_xy(lat, lon, ref_lat, ref_lon)
    with np.errstate(invalid="ignore"):
        ok &= np.hypot(xy[:, 0], xy[:, 1]) <= MAX_FIX_RADIUS_M
    idx = np.flatnonzero(ok)
    if idx.size >= 3:
        d = np.hypot(np.diff(xy[idx, 0]), np.diff(xy[idx, 1]))
        spike = np.zeros(idx.size, dtype=bool)
        spike[1:-1] = (d[:-1] > MAX_FIX_JUMP_M) & (d[1:] > MAX_FIX_JUMP_M)
        ok[idx[spike]] = False
    return np.asarray(ok, dtype=bool)


def position_kinematics(
    t_s: np.ndarray, lat: np.ndarray, lon: np.ndarray, v_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(a_lat_g, a_long_g, yaw_dps)`` from GPS position and speed.

    Right-positive lateral acceleration ``V²κ``, forward-positive
    longitudinal ``dV/dt`` and right-positive yaw rate ``Vκ`` on the input
    samples (sorted by time). Curvature comes from the local-XY path
    smoothed over :data:`XY_SMOOTH_M` and is zero below 1 m/s, where the
    fix jitters; ``dV/dt`` uses a :data:`LONG_SMOOTH_S` centred window.
    Implausible fixes (:func:`_plausible_fixes`) are interpolated across.
    """
    n = len(t_s)
    nan = np.full(n, np.nan)
    if n < 5:
        return nan, nan.copy(), nan.copy()
    t = np.asarray(t_s, dtype=float)
    v = np.nan_to_num(np.asarray(v_ms, dtype=float), nan=0.0)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = _plausible_fixes(t, lat, lon)
    if ok.sum() < 5:
        return nan, nan.copy(), nan.copy()
    xy = _local_xy(lat, lon, float(np.median(lat[ok])), float(np.median(lon[ok])))
    x = np.interp(t, t[ok], xy[ok, 0])
    y = np.interp(t, t[ok], xy[ok, 1])
    # Curvature on a uniform arc-length grid: a held or jittering fix at low
    # speed would otherwise put near-zero steps under the derivatives.
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    keep = np.concatenate([[True], np.diff(s) > 0])
    kappa_right = np.zeros(n)
    if s[-1] > MAX_PATH_POINTS * PATH_GRID_M:
        # Should be unreachable after _plausible_fixes; never allocate a
        # planet-sized grid whatever the input.
        logger.warning(
            "position_kinematics: path length %.0f km exceeds the grid cap; no reference",
            s[-1] / 1000.0,
        )
        return nan, nan.copy(), nan.copy()
    if s[-1] > 4 * XY_SMOOTH_M:
        s_grid = np.arange(0.0, s[-1], PATH_GRID_M)
        xg = np.interp(s_grid, s[keep], x[keep])
        yg = np.interp(s_grid, s[keep], y[keep])
        w = max(int(round(XY_SMOOTH_M / 2.0 / PATH_GRID_M)), 1)
        kernel = np.ones(2 * w + 1) / (2 * w + 1)
        # Edge-pad before smoothing so the path's ends do not bend towards 0.
        xs = np.convolve(np.pad(xg, w, mode="edge"), kernel, mode="valid")
        ys = np.convolve(np.pad(yg, w, mode="edge"), kernel, mode="valid")
        dx = np.gradient(xs, PATH_GRID_M)
        dy = np.gradient(ys, PATH_GRID_M)
        ddx = np.gradient(dx, PATH_GRID_M)
        ddy = np.gradient(dy, PATH_GRID_M)
        kappa_ccw = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
        kappa_right = -np.interp(s, s_grid, kappa_ccw)
    kappa_right[v < 1.0] = 0.0
    # Strictly increasing time for the dV/dt window and gradient.
    tt = t + np.arange(n) * 1e-9
    v_s = _window_mean(v, tt, LONG_SMOOTH_S / 2.0)
    with np.errstate(all="ignore"):
        a_long = np.gradient(v_s, tt) / G_MS2
    a_lat = v * v * kappa_right / G_MS2
    yaw = v * kappa_right * (180.0 / math.pi)
    return a_lat, np.nan_to_num(a_long, nan=0.0), yaw


# --------------------------------------------------------------------- checks


def _sorted_series(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if t.size and np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        t, x = t[order], x[order]
    return t, x


def _nearest_distance(t_c: np.ndarray, t_r: np.ndarray) -> np.ndarray:
    """Distance from each reference time to the nearest candidate sample."""
    if t_c.size == 0:
        return np.full(t_r.shape, np.inf)
    idx = np.searchsorted(t_c, t_r)
    hi = np.minimum(idx, t_c.size - 1)
    lo = np.maximum(idx - 1, 0)
    return np.asarray(np.minimum(np.abs(t_c[hi] - t_r), np.abs(t_c[lo] - t_r)))


def check_axis(
    t_c: np.ndarray,
    x_c: np.ndarray,
    t_r: np.ndarray,
    ref: np.ndarray,
    moving: np.ndarray,
    *,
    candidate: str,
    axis: str,
    other_ref: np.ndarray | None = None,
    braking_only: bool = False,
) -> AxisCheck:
    """Regress a candidate series on its reference over the moving samples.

    The candidate is interpolated onto the reference clock for the
    comparison only. ``braking_only`` restricts the regression to samples
    where the reference is below :data:`BRAKE_REF_G` (the A2 gate's
    longitudinal score), falling back to every moving sample when fewer
    than :data:`MIN_BRAKE_SAMPLES` exist.
    """
    kind = "dps" if axis == "yaw" else "g"
    band = SLOPE_BAND_YAW if axis == "yaw" else SLOPE_BAND
    chk = AxisCheck(candidate=candidate, axis=axis)
    t_c, x_c = _sorted_series(np.asarray(t_c, float), np.asarray(x_c, float))
    t_r = np.asarray(t_r, float)
    ref = np.asarray(ref, float)
    moving = np.asarray(moving, bool) & np.isfinite(ref) & np.isfinite(t_r)
    if t_c.size < 2 or moving.sum() < MIN_FIT_SAMPLES:
        return chk
    covered = _nearest_distance(t_c, t_r) <= COVERAGE_TOL_S
    chk.coverage = float(covered[moving].mean())
    sel = moving & covered
    if sel.sum() < MIN_FIT_SAMPLES:
        chk.reason = "coverage"
        return chk
    # Compare at the reference's bandwidth: the reference is a 0.5 s window
    # over 25 Hz GPS, the IMU a raw 50-500 Hz stream whose vibration alone
    # pulls a healthy sensor's longitudinal correlation under the threshold.
    x_c = _window_mean(x_c, t_c + np.arange(t_c.size) * 1e-9, LONG_SMOOTH_S / 2.0)
    x = np.interp(t_r[sel], t_c, x_c)
    r = ref[sel]
    chk.n = int(sel.sum())
    chk.std = float(np.std(x))
    fit = np.ones(x.shape, dtype=bool)
    if braking_only:
        braking = r < BRAKE_REF_G
        if braking.sum() >= MIN_BRAKE_SAMPLES:
            fit = braking
    if fit.sum() >= MIN_FIT_SAMPLES and np.std(r[fit]) > 0 and np.std(x[fit]) > 0:
        slope, offset = np.polyfit(r[fit], x[fit], 1)
        chk.slope = float(slope)
        chk.offset = float(offset)
        chk.corr = float(np.corrcoef(r[fit], x[fit])[0, 1])
    if other_ref is not None:
        o = np.asarray(other_ref, float)[sel]
        if np.std(o) > 0 and np.std(x) > 0:
            chk.cross = float(abs(np.corrcoef(x, o)[0, 1]))
    # Twisted (swapped or rotated axes: the candidate follows the *other*
    # axis' reference) is tested before the slope so a 90° mount reads
    # "twisted", not "flipped" — its slope on its own reference is a small
    # number of either sign. A truly flipped axis has |slope| ≈ 1 and a
    # small cross-correlation.
    if chk.coverage < COVERAGE_MIN:
        chk.reason = "coverage"
    elif not chk.std >= STD_MIN[kind]:
        chk.reason = "flat"
    elif other_ref is not None and chk.cross > CROSS_CORR_MAX:
        chk.reason = "twisted"
    elif not (band[0] <= chk.slope <= band[1]):
        chk.reason = "slope" if not chk.slope < 0 else "flipped"
    elif not chk.corr >= (CORR_MIN_LONG if axis == "long" else CORR_MIN):
        chk.reason = "corr"
    elif not abs(chk.offset) <= OFFSET_MAX[kind]:
        chk.reason = "offset"
    else:
        chk.reason = "ok"
        chk.passed = True
    return chk


# -------------------------------------------------------------------- verdict


def choose_reference_lap(
    lap_nums: list[int], lap_times_s: list[float], lap_types: list[str] | None = None
) -> int | None:
    """The fastest full lap (any lap when none is typed full), or None."""
    if not lap_nums:
        return None
    types = lap_types or ["full"] * len(lap_nums)
    cands = [(t, n) for n, t, ty in zip(lap_nums, lap_times_s, types) if ty == "full" and t > 0]
    if not cands:
        cands = [(t, n) for n, t in zip(lap_nums, lap_times_s) if t > 0]
    if not cands:
        return None
    return int(min(cands)[1])


def _lap_rows(ts: pa.Table, lap_num: int) -> pa.Table:
    return ts.filter(pc.equal(ts.column("lap_num"), lap_num))


def _gps_reference(ts: pa.Table) -> dict | None:
    """GPS position + speed of ``ts`` (one lap or the session) as reference
    arrays on the speed clock; None without usable GPS."""
    t_v, v = channel_series(ts, "speed_ms", "t_session_s")
    t_la, la = channel_series(ts, "gps_lat", "t_session_s")
    t_lo, lo = channel_series(ts, "gps_lon", "t_session_s")
    if t_v.size < MIN_FIT_SAMPLES or t_la.size < MIN_FIT_SAMPLES or t_lo.size < MIN_FIT_SAMPLES:
        return None
    t_v, v = _sorted_series(t_v, v)
    la_i = np.interp(t_v, *_sorted_series(t_la, la))
    lo_i = np.interp(t_v, *_sorted_series(t_lo, lo))
    a_lat, a_long, yaw = position_kinematics(t_v, la_i, lo_i, v)
    return {
        "t": t_v,
        "v": v,
        "lat": a_lat,
        "long": a_long,
        "yaw": yaw,
        "moving": v >= MOVING_MS,
    }


def validate_session(ts: pa.Table, ref_lap: int | None) -> KinematicsVerdict:
    """Run every candidate axis against the GPS reference of ``ref_lap`` and
    decide the session's accelerometer and gyro sources."""
    names = set(ts.schema.names)
    verdict = KinematicsVerdict(ref_lap=ref_lap)
    has_imu_acc = {"lat_g", "long_g"} <= names
    has_imu_yaw = "yaw_rate_dps" in names
    has_gps_acc = {"gps_lat_g", "gps_long_g"} <= names
    ref = None
    if ref_lap is not None and "lap_num" in names:
        ref = _gps_reference(_lap_rows(ts, ref_lap))
    if ref is None:
        verdict.note = "no_gps_reference"
        verdict.accel_source = "imu" if has_imu_acc else "none"
        verdict.gyro_source = "imu" if has_imu_yaw else "none"
        return verdict
    lap = _lap_rows(ts, ref_lap)  # type: ignore[arg-type]
    other = {"lat": ref["long"], "long": ref["lat"], "yaw": None}
    for (cand, axis), col in CANDIDATE_COLUMNS.items():
        if col not in names:
            continue
        t_c, x_c = channel_series(lap, col, "t_session_s")
        verdict.checks[f"{cand}_{axis}"] = check_axis(
            t_c,
            x_c,
            ref["t"],
            ref[axis],
            ref["moving"],
            candidate=cand,
            axis=axis,
            other_ref=other[axis],
        )

    def _ok(key: str) -> bool:
        c = verdict.checks.get(key)
        return bool(c is not None and c.passed)

    if _ok("imu_lat") and _ok("imu_long"):
        verdict.accel_source = "imu"
    elif _ok("gps_lat") and _ok("gps_long"):
        verdict.accel_source = "gps_channel"
    else:
        verdict.accel_source = "gps_position"
    verdict.gyro_source = "imu" if _ok("imu_yaw") else "gps_position"
    if not has_imu_acc and not has_gps_acc:
        verdict.note = "no_accelerometer"
    return verdict


# --------------------------------------------------------------------- loader


def _interp_series(t_out: np.ndarray, t_in: np.ndarray, x_in: np.ndarray) -> np.ndarray:
    t_in, x_in = _sorted_series(np.asarray(t_in, float), np.asarray(x_in, float))
    if t_in.size < 2:
        return np.full(t_out.shape, np.nan)
    out = np.asarray(np.interp(t_out, t_in, x_in), dtype=float)
    out[(t_out < t_in[0]) | (t_out > t_in[-1])] = np.nan
    return out


def session_kinematics(ts: pa.Table, verdict: KinematicsVerdict) -> pa.Table | None:
    """GPS speed plus the verdict's lateral / longitudinal / yaw series on one
    clock: the IMU's when ``accel_source == "imu"`` (longitudinal and yaw
    interpolated onto the lateral channel's samples when they sit on
    another timebase), the GPS clock on fallback. Columns ``t_session_s,
    lap_num, stint_id, t_lap_s, speed_ms, lat_g, long_g, yaw_rate_dps``
    and the constant ``lat_source / long_source / yaw_source``.
    """
    names = set(ts.schema.names)
    if "t_session_s" not in names:
        return None
    gps = _gps_reference(ts) if {"speed_ms", "gps_lat", "gps_lon"} <= names else None
    t_v, v = channel_series(ts, "speed_ms", "t_session_s")
    t_v, v = _sorted_series(t_v, v)

    src = verdict.accel_source
    if src == "imu" and "lat_g" in names:
        base = ts.filter(pc.is_valid(ts.column("lat_g")))
        if len(base) == 0:
            return None
        t_b = np.asarray(base.column("t_session_s").to_numpy(zero_copy_only=False), float)
        lat = np.asarray(base.column("lat_g").to_numpy(zero_copy_only=False), float)
        if "long_g" in names:
            tl, xl = channel_series(ts, "long_g", "t_session_s")
            long = (
                xl
                if (tl.shape == t_b.shape and np.array_equal(tl, t_b))
                else _interp_series(t_b, tl, xl)
            )
        else:
            long = np.full(t_b.shape, np.nan)
    elif src in ("gps_channel", "gps_position") and t_v.size:
        base = ts.filter(pc.is_valid(ts.column("speed_ms")))
        t_b = np.asarray(base.column("t_session_s").to_numpy(zero_copy_only=False), float)
        if src == "gps_channel":
            lat = _interp_series(t_b, *channel_series(ts, "gps_lat_g", "t_session_s"))
            long = _interp_series(t_b, *channel_series(ts, "gps_long_g", "t_session_s"))
        else:
            if gps is None:
                return None
            lat = _interp_series(t_b, gps["t"], gps["lat"])
            long = _interp_series(t_b, gps["t"], gps["long"])
    else:
        return None

    speed = _interp_series(t_b, t_v, v) if t_v.size else np.full(t_b.shape, np.nan)
    if verdict.gyro_source == "imu" and "yaw_rate_dps" in names:
        yaw = _interp_series(t_b, *channel_series(ts, "yaw_rate_dps", "t_session_s"))
    elif verdict.gyro_source == "gps_position" and gps is not None:
        yaw = _interp_series(t_b, gps["t"], gps["yaw"])
    else:
        yaw = np.full(t_b.shape, np.nan)

    n = len(t_b)
    cols: dict = {"t_session_s": pa.array(t_b, type=pa.float64())}
    for key, typ in (("lap_num", pa.int16()), ("stint_id", pa.int16()), ("t_lap_s", pa.float32())):
        if key in base.schema.names:
            cols[key] = base.column(key).cast(typ)
    cols["speed_ms"] = pa.array(speed.astype(np.float32))
    cols["lat_g"] = pa.array(np.asarray(lat, np.float32))
    cols["long_g"] = pa.array(np.asarray(long, np.float32))
    cols["yaw_rate_dps"] = pa.array(np.asarray(yaw, np.float32))
    for axis in ("lat", "long", "yaw"):
        cols[f"{axis}_source"] = pa.array([verdict.source_for(axis)] * n, type=pa.string())
    return pa.table(cols)


KINEMATICS_COLUMNS = [
    "speed_ms",
    "gps_lat",
    "gps_lon",
    "lat_g",
    "long_g",
    "yaw_rate_dps",
    "gps_lat_g",
    "gps_long_g",
    "gps_yaw_rate_dps",
]


def load_kinematics(dataset_root: Path, session_id: str) -> pa.Table | None:
    """The session's validated kinematics (see :func:`session_kinematics`)
    from its timeseries file and the verdict stored in it; None when the
    session has no file, no verdict (pre-schema-7 file) or no usable series.
    """
    from .dataset import read_timeseries, timeseries_metadata, timeseries_path

    path = timeseries_path(dataset_root, session_id)
    if path is None:
        return None
    meta = timeseries_metadata(path)
    if not meta["kinematics"]:
        return None
    verdict = KinematicsVerdict.from_json(meta["kinematics"])
    ts = read_timeseries(path, columns=KINEMATICS_COLUMNS, families=["gps", "accel", "gyro"])
    if len(ts) == 0:
        return None
    return session_kinematics(ts, verdict)


def score_chosen(ts: pa.Table, verdict: KinematicsVerdict) -> dict[str, AxisCheck]:
    """Re-regress the *chosen* series on the reference lap (the A2 gate,
    longitudinal on braking samples): ``{"lat": ..., "long": ..., "yaw":
    ...}``; empty without a reference."""
    out: dict[str, AxisCheck] = {}
    if verdict.ref_lap is None or "lap_num" not in ts.schema.names:
        return out
    kin = session_kinematics(ts, verdict)
    if kin is None:
        return out
    lap = _lap_rows(ts, verdict.ref_lap)
    ref = _gps_reference(lap)
    if ref is None:
        return out
    kin_lap = _lap_rows(kin, verdict.ref_lap) if "lap_num" in kin.schema.names else kin
    t_k = np.asarray(kin_lap.column("t_session_s").to_numpy(zero_copy_only=False), float)
    other = {"lat": ref["long"], "long": ref["lat"], "yaw": None}
    for axis, col in (("lat", "lat_g"), ("long", "long_g"), ("yaw", "yaw_rate_dps")):
        x = np.asarray(kin_lap.column(col).to_numpy(zero_copy_only=False), float)
        out[axis] = check_axis(
            t_k,
            x,
            ref["t"],
            ref[axis],
            ref["moving"],
            candidate=verdict.source_for(axis),
            axis=axis,
            other_ref=other[axis],
            braking_only=(axis == "long"),
        )
    return out


GATE_LAT_SLOPE = (0.9, 1.1)
GATE_LONG_SLOPE = (0.85, 1.15)
GATE_YAW_SLOPE = (0.9, 1.1)
GATE_LAT_CORR = 0.9
GATE_LONG_CORR = 0.85


def _gate_flags(scores: dict[str, AxisCheck]) -> str:
    """The A2 gate on the chosen series: ``"ok"`` or the failing checks."""
    bad: list[str] = []
    lat, long, yaw = scores.get("lat"), scores.get("long"), scores.get("yaw")
    if lat is None or not (GATE_LAT_SLOPE[0] <= lat.slope <= GATE_LAT_SLOPE[1]):
        bad.append("lat_slope")
    if lat is None or not lat.corr >= GATE_LAT_CORR:
        bad.append("lat_corr")
    if long is None or not (GATE_LONG_SLOPE[0] <= long.slope <= GATE_LONG_SLOPE[1]):
        bad.append("long_slope")
    if long is None or not long.corr >= GATE_LONG_CORR:
        bad.append("long_corr")
    if yaw is None or not (GATE_YAW_SLOPE[0] <= yaw.slope <= GATE_YAW_SLOPE[1]):
        bad.append("yaw_slope")
    return "ok" if not bad else "+".join(bad)


def kinematics_audit(
    dataset_root: Path,
    *,
    since=None,
    car: str | None = None,
    recheck: bool = False,
):
    """One row per session with a timeseries file: the stored verdict, the
    IMU regression numbers, the chosen series re-scored on the reference lap
    and the gate verdict (see the ``kinematics-audit`` CLI)."""
    import pandas as pd
    import pyarrow.parquet as pq

    from .dataset import read_timeseries, timeseries_metadata, timeseries_path
    from .paths import sessions_dir

    parts = [
        pq.read_table(f).to_pandas() for f in sorted(sessions_dir(dataset_root).glob("*.parquet"))
    ]
    if not parts:
        return pd.DataFrame()
    sessions = pd.concat(parts, ignore_index=True)
    if since is not None:
        sessions = sessions[pd.to_datetime(sessions["date"]) >= pd.Timestamp(since)]
    if car:
        sessions = sessions[sessions["car"].astype(str).str.contains(car, case=False, na=False)]
    rows: list[dict] = []
    for sess in sessions.sort_values(["date", "session_id"]).itertuples(index=False):
        sid = str(sess.session_id)
        path = timeseries_path(dataset_root, sid)
        if path is None:
            continue
        meta = timeseries_metadata(path)
        stored = KinematicsVerdict.from_json(meta["kinematics"]) if meta["kinematics"] else None
        ts = read_timeseries(path, columns=KINEMATICS_COLUMNS, families=["gps", "accel", "gyro"])
        verdict = stored
        changed = False
        if recheck or stored is None:
            ref = stored.ref_lap if stored is not None else None
            if ref is None and "lap_num" in ts.schema.names and "t_lap_s" in ts.schema.names:
                df = ts.select(["lap_num", "t_lap_s"]).to_pandas()
                spans = df.groupby("lap_num")["t_lap_s"].max()
                spans = spans[spans.index > 0] if (spans.index > 0).any() else spans
                ref = int(spans.idxmin()) if len(spans) else None
            verdict = validate_session(ts, ref)
            changed = stored is not None and (
                verdict.accel_source != stored.accel_source
                or verdict.gyro_source != stored.gyro_source
            )
        if verdict is None:
            continue
        scores = score_chosen(ts, verdict)
        row = {
            "date": sess.date,
            "car": sess.car,
            "profile_name": getattr(sess, "profile_name", None),
            "session_id": sid,
            "n_laps": int(getattr(sess, "n_laps", 0) or 0),
            "accel_source": verdict.accel_source,
            "gyro_source": verdict.gyro_source,
            "ref_lap": verdict.ref_lap,
        }
        for key in ("imu_lat", "imu_long", "imu_yaw"):
            chk = verdict.checks.get(key)
            row[f"{key}_slope"] = round(chk.slope, 3) if chk else float("nan")
            row[f"{key}_corr"] = round(chk.corr, 3) if chk else float("nan")
            row[f"{key}_why"] = chk.reason if chk else "absent"
        for axis in ("lat", "long", "yaw"):
            sc = scores.get(axis)
            row[f"chosen_{axis}_slope"] = round(sc.slope, 3) if sc else float("nan")
            row[f"chosen_{axis}_corr"] = round(sc.corr, 3) if sc else float("nan")
        row["gate"] = _gate_flags(scores) if scores else "no_reference"
        row["note"] = verdict.note
        if recheck:
            row["changed"] = changed
        rows.append(row)
    return pd.DataFrame(rows)


def verdict_from_metadata(meta: dict | None) -> KinematicsVerdict | None:
    return KinematicsVerdict.from_json(meta) if meta else None


__all__ = [
    "AxisCheck",
    "KinematicsVerdict",
    "KIN_ROW_COLUMNS",
    "check_axis",
    "choose_reference_lap",
    "empty_verdict_row",
    "load_kinematics",
    "position_kinematics",
    "score_chosen",
    "session_kinematics",
    "validate_session",
    "kinematics_audit",
]
