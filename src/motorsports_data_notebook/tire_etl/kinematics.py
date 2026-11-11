"""IMU validation against GPS and the per-session sensor verdict.

The logger's inertial channels are not always trustworthy: a miscalibrated
IMU reads lateral g at half the true value, a swapped harness flips or
twists an axis, a dead sensor is flat. The extractor therefore compares each
candidate series with a **reference derived from GPS position and speed** on
the session's fastest full lap and records **one verdict per session** for
the accelerometer (``accel_source``: both axes together) and one for the
gyro (``gyro_source``), with every regression number, on the session row and
in the timeseries file's metadata. Fallback order: IMU → IMU with its two
horizontal channels exchanged (``imu_swapped``, a dash whose LateralAcc is
the car's longitudinal acceleration and vice versa) → the logger's own
GPS-derived channels (validated the same way) → series derived here from
position and speed. The raw channels are stored untouched; the verdict is
applied by :func:`load_kinematics`, which puts speed and the chosen
lateral / longitudinal / yaw series on one clock for the consumers.

The session verdict says which sensor is sound; a **per-window source** says
where it can be used. The session is cut into fixed 20 s windows of session
time and each gets one of three sources: ``imu`` (the session's IMU is sound
and has samples in the window), ``gps`` (it does not, and the GPS fallback —
the logger's GPS channels when they validated, else the position-derived
series — has no spike in the window) or ``discard`` (neither). An IMU stream
that stops part-way through a run is used while it lasts and the GPS takes
over; a stretch where the IMU is frozen while the car is driving goes to
the GPS; a GPS spike with no IMU to fall back on is discarded and the
consumers leave a hole there (:func:`session_kinematics` emits NaN rows
with source ``discard``). Windows where a live IMU does not follow the GPS
reference are counted (``n_mismatch``): up to three are taken as the
reference's fault (a receiver still acquiring, a spin), more than that and
the IMU is not trusted for the session.

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

# Per-window source. Windows are fixed bins of session time.
WINDOW_S = 20.0
WINDOW_COVERAGE_MIN = 0.9  # IMU live in a window: this share of its GPS samples has an IMU sample
WINDOW_MIN_MOVING_S = 5.0  # a window is compared with the reference only with this much moving time
# Reference excitation (std, g) a window needs for its correlation to mean
# anything, and the correlation under which a live IMU is a mismatch. On
# 22,000 windows of trusted sessions the 1st percentile of the lateral
# correlation is 0.90 at this excitation.
WINDOW_REF_STD_MIN = {"lat": 0.3, "long": 0.2}
WINDOW_CORR_MIN = 0.5
# More mismatched windows than this and the IMU is not trusted for the
# session. Healthy sessions have 0-2 (the GPS reference at fault: a receiver
# still acquiring at the start of a file, a position spike, a spin).
MISMATCH_CAP = 3
# A frozen IMU: flat (STD_MIN) while the reference shows cornering at speed.
# The speed floor keeps a parked car under a wandering GPS fix out of it.
WINDOW_STUCK_SPEED_MS = 15.0
# A GPS-derived acceleration above this is a spike, not the car (the fastest
# car here peaks near 2 g; the 99th percentile of a window's maximum is 2.3).
GPS_SANE_MAX_G = 3.0
WINDOW_IMU, WINDOW_GPS, WINDOW_DISCARD, WINDOW_EMPTY = "I", "G", "D", "-"
DISCARD = "discard"  # the lat_source / long_source of a discarded window's rows

# ``imu_swapped``: the IMU's two horizontal channels are exchanged in the file
# (the KK-SII dash since 2026-03-18 logs the car's longitudinal acceleration
# as LateralAcc and vice versa, both signs intact); the loader hands out the
# columns exchanged, the stored channels stay as logged.
ACCEL_SOURCES = ("imu", "imu_swapped", "gps_channel", "gps_position", "none")
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
    + [(f"kin_imu_swapped_{axis}_{m}", pa.float32()) for axis in ("lat", "long") for m in METRICS]
    + [(f"kin_imu_swapped_{axis}_ok", pa.bool_()) for axis in ("lat", "long")]
    + [
        ("kin_gps_fallback", pa.string()),
        ("kin_imu_ref_lap", pa.int32()),
        ("kin_win_imu", pa.int32()),
        ("kin_win_gps", pa.int32()),
        ("kin_win_discard", pa.int32()),
        ("kin_win_mismatch", pa.int32()),
    ]
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
    # The GPS series used where the IMU is not: ``gps_channel`` or ``gps_position``.
    gps_fallback: str = "gps_position"
    # The lap the IMU was scored on when it does not cover ``ref_lap``.
    imu_ref_lap: int | None = None
    # Per-window source: window ``i`` of ``windows`` spans
    # ``[(window_first + i) · window_s, (window_first + i + 1) · window_s)`` of
    # session time and is ``I`` (imu), ``G`` (gps fallback), ``D`` (discard)
    # or ``-`` (no GPS sample). Empty: no windows were scored and the whole
    # session uses ``accel_source``.
    window_s: float = WINDOW_S
    window_first: int = 0
    windows: str = ""
    n_mismatch: int = 0

    # -- serialisation ---------------------------------------------------
    def to_json(self) -> dict:
        return {
            "accel_source": self.accel_source,
            "gyro_source": self.gyro_source,
            "ref_lap": self.ref_lap,
            "note": self.note,
            "checks": {k: v.as_dict() for k, v in self.checks.items()},
            "gps_fallback": self.gps_fallback,
            "imu_ref_lap": self.imu_ref_lap,
            "window_s": self.window_s,
            "window_first": self.window_first,
            "windows": self.windows,
            "n_mismatch": self.n_mismatch,
        }

    @classmethod
    def from_json(cls, data: dict) -> "KinematicsVerdict":
        known = set(AxisCheck.__dataclass_fields__)
        checks = {
            k: AxisCheck(**{f: x for f, x in v.items() if f in known})
            for k, v in (data.get("checks") or {}).items()
        }
        return cls(
            accel_source=str(data.get("accel_source", "none")),
            gyro_source=str(data.get("gyro_source", "none")),
            ref_lap=data.get("ref_lap"),
            note=str(data.get("note", "")),
            checks=checks,
            gps_fallback=str(data.get("gps_fallback", "gps_position")),
            imu_ref_lap=data.get("imu_ref_lap"),
            window_s=float(data.get("window_s", WINDOW_S)),
            window_first=int(data.get("window_first", 0)),
            windows=str(data.get("windows", "")),
            n_mismatch=int(data.get("n_mismatch", 0)),
        )

    def window_codes(self, t_session_s: np.ndarray) -> np.ndarray:
        """The window code (``b"I"`` / ``b"G"`` / ``b"D"`` / ``b"-"``) of each
        session time; ``b"-"`` outside the scored span."""
        t = np.asarray(t_session_s, float)
        out = np.full(t.shape, WINDOW_EMPTY.encode(), dtype="S1")
        if not self.windows:
            return out
        codes = np.frombuffer(self.windows.encode("ascii"), dtype="S1")
        fin = np.isfinite(t)
        idx = np.floor(np.where(fin, t, 0.0) / self.window_s).astype(np.int64) - self.window_first
        ok = fin & (idx >= 0) & (idx < codes.size)
        out[ok] = codes[idx[ok]]
        return out

    def window_counts(self) -> dict[str, int]:
        return {
            "imu": self.windows.count(WINDOW_IMU),
            "gps": self.windows.count(WINDOW_GPS),
            "discard": self.windows.count(WINDOW_DISCARD),
        }

    def to_row(self) -> dict:
        """The session-row columns (:data:`KIN_ROW_COLUMNS`)."""
        row: dict = {
            "accel_source": self.accel_source,
            "gyro_source": self.gyro_source,
            "kin_ref_lap": self.ref_lap,
            "kin_note": self.note or None,
        }
        for cand, axes in (("imu", AXES), ("gps", AXES), ("imu_swapped", ("lat", "long"))):
            for axis in axes:
                chk = self.checks.get(f"{cand}_{axis}")
                for m in METRICS:
                    v = getattr(chk, m) if chk is not None else float("nan")
                    row[f"kin_{cand}_{axis}_{m}"] = None if v is None or np.isnan(v) else float(v)
                row[f"kin_{cand}_{axis}_ok"] = bool(chk.passed) if chk is not None else None
        counts = self.window_counts()
        row["kin_gps_fallback"] = self.gps_fallback
        row["kin_imu_ref_lap"] = self.imu_ref_lap
        row["kin_win_imu"] = counts["imu"] if self.windows else None
        row["kin_win_gps"] = counts["gps"] if self.windows else None
        row["kin_win_discard"] = counts["discard"] if self.windows else None
        row["kin_win_mismatch"] = self.n_mismatch if self.windows else None
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
    # over 25 Hz GPS, the IMU a raw 50-100 Hz stream whose vibration alone
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


def reference_lap_order(
    lap_nums: list[int], lap_times_s: list[float], lap_types: list[str] | None = None
) -> list[int]:
    """The full laps, fastest first (every lap when none is typed full)."""
    if not lap_nums:
        return []
    types = lap_types or ["full"] * len(lap_nums)
    cands = [(t, n) for n, t, ty in zip(lap_nums, lap_times_s, types) if ty == "full" and t > 0]
    if not cands:
        cands = [(t, n) for n, t in zip(lap_nums, lap_times_s) if t > 0]
    return [int(n) for _, n in sorted(cands)]


def choose_reference_lap(
    lap_nums: list[int], lap_times_s: list[float], lap_types: list[str] | None = None
) -> int | None:
    """The fastest full lap (any lap when none is typed full), or None."""
    order = reference_lap_order(lap_nums, lap_times_s, lap_types)
    return order[0] if order else None


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


def _imu_reference_lap(ts: pa.Table, ref_lap: int, ref_laps: list[int] | None) -> int:
    """The first of ``ref_laps`` (fastest first) the IMU accelerometer covers;
    ``ref_lap`` when it covers that one, when there is no alternative, or
    when it covers none of them."""
    names = set(ts.schema.names)
    if not ref_laps or not {"lat_g", "speed_ms", "lap_num"} <= names:
        return ref_lap
    t_c, _ = _sorted_series(*channel_series(ts, "lat_g", "t_session_s"))
    if t_c.size < 2:
        return ref_lap
    sp = ts.filter(pc.is_valid(ts.column("speed_ms")))
    t = np.asarray(sp.column("t_session_s").to_numpy(zero_copy_only=False), float)
    v = np.asarray(sp.column("speed_ms").to_numpy(zero_copy_only=False), float)
    lap = np.asarray(sp.column("lap_num").to_numpy(zero_copy_only=False))
    covered = _nearest_distance(t_c, t) <= COVERAGE_TOL_S
    moving = v >= MOVING_MS
    for cand in [ref_lap] + [n for n in ref_laps if n != ref_lap]:
        m = (lap == cand) & moving
        if m.sum() >= MIN_FIT_SAMPLES and covered[m].mean() >= COVERAGE_MIN:
            return int(cand)
    return ref_lap


def _on_clock(
    ts: pa.Table, column: str, t_out: np.ndarray, *, smooth: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """``column`` interpolated onto ``t_out`` and whether each of those times
    has one of its samples within :data:`COVERAGE_TOL_S`. ``smooth`` applies
    the reference's 0.5 s window first (see :func:`check_axis`)."""
    t_c, x_c = _sorted_series(*channel_series(ts, column, "t_session_s"))
    if t_c.size < 2:
        return np.full(t_out.shape, np.nan), np.zeros(t_out.shape, dtype=bool)
    if smooth:
        x_c = _window_mean(x_c, t_c + np.arange(t_c.size) * 1e-9, LONG_SMOOTH_S / 2.0)
    return np.interp(t_out, t_c, x_c), _nearest_distance(t_c, t_out) <= COVERAGE_TOL_S


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < MIN_FIT_SAMPLES or not np.std(a) > 0 or not np.std(b) > 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_windows(ts: pa.Table, imu: str | None, gps_fallback: str) -> tuple[int, str, int]:
    """Source of every :data:`WINDOW_S` window of the session.

    ``imu`` is the session's sound accelerometer (``"imu"`` /
    ``"imu_swapped"``) or None; ``gps_fallback`` the GPS series used where
    the IMU is not. Returns ``(window_first, codes, n_mismatch)``: one code
    per consecutive window from ``window_first`` (see
    :class:`KinematicsVerdict`) and the number of windows where a live IMU
    does not follow a sane, excited position reference.

    A window is ``I`` when the IMU has samples for it and is not frozen,
    else ``G`` when the GPS fallback covers it without a spike over
    :data:`GPS_SANE_MAX_G` (smoothed over 0.5 s, car moving), else ``D``.
    """
    names = set(ts.schema.names)
    ref = _gps_reference(ts) if {"speed_ms", "gps_lat", "gps_lon"} <= names else None
    if ref is None:
        return 0, "", 0
    t, v, mv = ref["t"], ref["v"], ref["moving"]
    dt = np.clip(np.diff(t, append=t[-1]), 0.0, 1.0)
    b = np.floor(t / WINDOW_S).astype(np.int64)  # non-decreasing: t is sorted
    first, last = int(b[0]), int(b[-1])
    # Spikes are judged at the reference's bandwidth (0.5 s) and while the
    # car is moving: one noisy 25 Hz sample of the logger's GPS-derived g is
    # not a spike, half a second of 5 g is.
    if gps_fallback == "gps_channel":
        f_lat, c_lat = _on_clock(ts, "gps_lat_g", t, smooth=True)
        f_long, c_long = _on_clock(ts, "gps_long_g", t, smooth=True)
        f_cov = c_lat & c_long
    else:
        f_lat, f_long = ref["lat"], ref["long"]
        f_cov = np.isfinite(f_lat) & np.isfinite(f_long)
    f_spike = mv & ((np.abs(f_lat) > GPS_SANE_MAX_G) | (np.abs(f_long) > GPS_SANE_MAX_G))
    ref_spike = mv & (
        (np.abs(ref["lat"]) > GPS_SANE_MAX_G) | (np.abs(ref["long"]) > GPS_SANE_MAX_G)
    )
    x: dict[str, np.ndarray] = {}
    live = np.zeros(t.shape, dtype=bool)
    if imu in ("imu", "imu_swapped") and {"lat_g", "long_g"} <= names:
        lat_col, long_col = ("lat_g", "long_g") if imu == "imu" else ("long_g", "lat_g")
        x["lat"], k_lat = _on_clock(ts, lat_col, t, smooth=True)
        x["long"], k_long = _on_clock(ts, long_col, t, smooth=True)
        live = k_lat & k_long
    codes: list[str] = []
    n_mismatch = 0
    edges = np.searchsorted(b, np.arange(first, last + 2))
    for i0, i1 in zip(edges[:-1], edges[1:]):
        if i1 <= i0:
            codes.append(WINDOW_EMPTY)
            continue
        w = slice(int(i0), int(i1))
        gps_sane = bool(f_cov[w].mean() >= WINDOW_COVERAGE_MIN and not f_spike[w].any())
        usable = bool(x) and bool(live[w].mean() >= WINDOW_COVERAGE_MIN)
        m = mv[w] & live[w]
        if usable and dt[w][mv[w]].sum() >= WINDOW_MIN_MOVING_S and not ref_spike[w].any():
            mismatch = False
            for axis in ("lat", "long"):
                r_ref = ref[axis][w][m]
                if r_ref.size >= MIN_FIT_SAMPLES and np.std(r_ref) >= WINDOW_REF_STD_MIN[axis]:
                    r = _corr(r_ref, x[axis][w][m])
                    mismatch |= not r >= WINDOW_CORR_MIN
            lat_ref = ref["lat"][w][m]
            frozen = bool(
                lat_ref.size >= MIN_FIT_SAMPLES
                and np.std(lat_ref) >= WINDOW_REF_STD_MIN["lat"]
                and np.std(x["lat"][w][m]) < STD_MIN["g"]
                and float(np.mean(v[w][m])) >= WINDOW_STUCK_SPEED_MS
            )
            n_mismatch += int(mismatch or frozen)
            usable = not frozen
        codes.append(WINDOW_IMU if usable else WINDOW_GPS if gps_sane else WINDOW_DISCARD)
    return first, "".join(codes), n_mismatch


def validate_session(
    ts: pa.Table, ref_lap: int | None, ref_laps: list[int] | None = None
) -> KinematicsVerdict:
    """Run every candidate axis against the GPS reference of ``ref_lap``,
    decide the session's accelerometer and gyro sources, and score the
    source of every window of the session (:func:`score_windows`).

    ``ref_laps`` (the full laps, fastest first) lets the IMU be scored on
    the fastest lap it has samples for when its stream stopped before
    ``ref_lap``; the windows then use it for the part of the run it covers.
    """
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
    # The IMU's own reference lap: the fastest full lap it covers.
    imu_lap_num = _imu_reference_lap(ts, ref_lap, ref_laps)  # type: ignore[arg-type]
    imu_lap, imu_ref = lap, ref
    if imu_lap_num != ref_lap:
        alt = _gps_reference(_lap_rows(ts, imu_lap_num))
        if alt is not None:
            imu_lap, imu_ref = _lap_rows(ts, imu_lap_num), alt
            verdict.imu_ref_lap = int(imu_lap_num)
    for (cand, axis), col in CANDIDATE_COLUMNS.items():
        if col not in names:
            continue
        c_lap, c_ref = (imu_lap, imu_ref) if cand == "imu" else (lap, ref)
        other = {"lat": c_ref["long"], "long": c_ref["lat"], "yaw": None}
        t_c, x_c = channel_series(c_lap, col, "t_session_s")
        verdict.checks[f"{cand}_{axis}"] = check_axis(
            t_c,
            x_c,
            c_ref["t"],
            c_ref[axis],
            c_ref["moving"],
            candidate=cand,
            axis=axis,
            other_ref=other[axis],
        )

    def _ok(key: str) -> bool:
        c = verdict.checks.get(key)
        return bool(c is not None and c.passed)

    if has_imu_acc and not (_ok("imu_lat") and _ok("imu_long")):
        # Exchanged horizontal channels: score the longitudinal channel as
        # the lateral candidate and vice versa. A rotated mount fails this
        # too (its cross-correlation survives the exchange); only a clean
        # swap passes both axes.
        t_lo, x_lo = channel_series(imu_lap, "long_g", "t_session_s")
        t_la, x_la = channel_series(imu_lap, "lat_g", "t_session_s")
        verdict.checks["imu_swapped_lat"] = check_axis(
            t_lo,
            x_lo,
            imu_ref["t"],
            imu_ref["lat"],
            imu_ref["moving"],
            candidate="imu_swapped",
            axis="lat",
            other_ref=imu_ref["long"],
        )
        verdict.checks["imu_swapped_long"] = check_axis(
            t_la,
            x_la,
            imu_ref["t"],
            imu_ref["long"],
            imu_ref["moving"],
            candidate="imu_swapped",
            axis="long",
            other_ref=imu_ref["lat"],
        )

    imu: str | None = None
    if _ok("imu_lat") and _ok("imu_long"):
        imu = "imu"
    elif _ok("imu_swapped_lat") and _ok("imu_swapped_long"):
        imu = "imu_swapped"
    verdict.gps_fallback = "gps_channel" if _ok("gps_lat") and _ok("gps_long") else "gps_position"
    first, codes, n_mismatch = score_windows(ts, imu, verdict.gps_fallback)
    if imu is not None and n_mismatch > MISMATCH_CAP:
        # The IMU passes on its reference lap but fails to follow GPS in too
        # many windows of the run to be the reference's fault.
        verdict.note = "imu_window_mismatch"
        imu = None
        first, codes, _ = score_windows(ts, None, verdict.gps_fallback)
    verdict.accel_source = imu if imu is not None else verdict.gps_fallback
    verdict.window_first, verdict.windows, verdict.n_mismatch = first, codes, n_mismatch
    verdict.gyro_source = "imu" if _ok("imu_yaw") else "gps_position"
    if not has_imu_acc and not has_gps_acc:
        verdict.note = "no_accelerometer"
    return verdict


def mark_discarded(ts: pa.Table, verdict: KinematicsVerdict) -> pa.Table:
    """``ts`` with the sparse ``discard`` column: true on every row (of any
    timebase) whose session time falls in a window the verdict discarded,
    null elsewhere. The samples themselves are untouched; the column makes
    the verdict's mark something a query can filter on. Placed after the
    other key columns."""
    if "t_session_s" not in ts.schema.names:
        return ts
    t = np.asarray(ts.column("t_session_s").to_numpy(zero_copy_only=False), float)
    hit = verdict.window_codes(t) == WINDOW_DISCARD.encode()
    column = pa.array(np.ones(len(t), dtype=bool), mask=~hit, type=pa.bool_())
    if "discard" in ts.schema.names:
        ts = ts.drop_columns(["discard"])
    keys = {"session_id", "timebase", "family", "lap_num", "stint_id", "sample_idx"}
    keys |= {"t_session_s", "t_lap_s"}
    at = 1 + max(i for i, n in enumerate(ts.schema.names) if n in keys)
    return ts.add_column(at, "discard", column)


# --------------------------------------------------------------------- loader


def _interp_series(t_out: np.ndarray, t_in: np.ndarray, x_in: np.ndarray) -> np.ndarray:
    t_in, x_in = _sorted_series(np.asarray(t_in, float), np.asarray(x_in, float))
    if t_in.size < 2:
        return np.full(t_out.shape, np.nan)
    out = np.asarray(np.interp(t_out, t_in, x_in), dtype=float)
    out[(t_out < t_in[0]) | (t_out > t_in[-1])] = np.nan
    return out


def _kin_part(
    base: pa.Table,
    t_b: np.ndarray,
    lat: np.ndarray,
    long: np.ndarray,
    source: str,
    speed: tuple[np.ndarray, np.ndarray],
    yaw_imu: tuple[np.ndarray, np.ndarray] | None,
    gps: dict | None,
) -> pa.Table:
    """Rows of the kinematics table on ``base``'s clock from one source."""
    n = len(t_b)
    t_v, v = speed
    spd = _interp_series(t_b, t_v, v) if t_v.size else np.full(n, np.nan)
    yaw = _interp_series(t_b, gps["t"], gps["yaw"]) if gps is not None else np.full(n, np.nan)
    yaw_src = np.full(n, "gps_position" if gps is not None else "none", dtype=object)
    if yaw_imu is not None and yaw_imu[0].size >= 2:
        # the gyro where it has samples, the position-derived yaw rate elsewhere
        have = _nearest_distance(yaw_imu[0], t_b) <= COVERAGE_TOL_S
        yaw = np.where(have, np.interp(t_b, *yaw_imu), yaw)
        yaw_src[have] = "imu"
    cols: dict = {"t_session_s": pa.array(t_b, type=pa.float64())}
    for key, typ in (("lap_num", pa.int16()), ("stint_id", pa.int16()), ("t_lap_s", pa.float32())):
        if key in base.schema.names:
            cols[key] = base.column(key).cast(typ)
    cols["speed_ms"] = pa.array(spd.astype(np.float32))
    cols["lat_g"] = pa.array(np.asarray(lat, np.float32))
    cols["long_g"] = pa.array(np.asarray(long, np.float32))
    cols["yaw_rate_dps"] = pa.array(np.asarray(yaw, np.float32))
    cols["lat_source"] = pa.array([source] * n, type=pa.string())
    cols["long_source"] = pa.array([source] * n, type=pa.string())
    cols["yaw_source"] = pa.array(yaw_src.tolist(), type=pa.string())
    return pa.table(cols)


def session_kinematics(ts: pa.Table, verdict: KinematicsVerdict) -> pa.Table | None:
    """GPS speed plus the verdict's lateral / longitudinal / yaw series.

    Each window of the session contributes rows from the source the verdict
    gave it (:func:`score_windows`): the IMU's own samples where it is
    ``imu`` (the stored channels exchanged for ``imu_swapped``; longitudinal
    interpolated onto the lateral channel's samples when they sit on another
    timebase), the GPS clock with the fallback series where it is ``gps``,
    and the GPS clock with NaN accelerations where it is ``discard`` — the
    rows are kept so that a consumer integrating over time sees the hole
    instead of stepping across it. A verdict without windows uses
    ``accel_source`` for the whole session.

    Columns ``t_session_s, lap_num, stint_id, t_lap_s, speed_ms, lat_g,
    long_g, yaw_rate_dps`` and per row ``lat_source / long_source``
    (``imu``, ``imu_swapped``, ``gps_channel``, ``gps_position`` or
    ``discard``) and ``yaw_source``.
    """
    names = set(ts.schema.names)
    if "t_session_s" not in names:
        return None
    gps = _gps_reference(ts) if {"speed_ms", "gps_lat", "gps_lon"} <= names else None
    speed = _sorted_series(*channel_series(ts, "speed_ms", "t_session_s"))
    yaw_imu = None
    if verdict.gyro_source == "imu" and "yaw_rate_dps" in names:
        yaw_imu = _sorted_series(*channel_series(ts, "yaw_rate_dps", "t_session_s"))

    def _times(tbl: pa.Table) -> np.ndarray:
        return np.asarray(tbl.column("t_session_s").to_numpy(zero_copy_only=False), float)

    def _gps_series(source: str, t_out: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if source == "gps_channel" and {"gps_lat_g", "gps_long_g"} <= names:
            return (
                _interp_series(t_out, *channel_series(ts, "gps_lat_g", "t_session_s")),
                _interp_series(t_out, *channel_series(ts, "gps_long_g", "t_session_s")),
            )
        if gps is None:
            return None
        return (
            _interp_series(t_out, gps["t"], gps["lat"]),
            _interp_series(t_out, gps["t"], gps["long"]),
        )

    src = verdict.accel_source
    windowed = bool(verdict.windows)
    parts: list[pa.Table] = []
    if src in ("imu", "imu_swapped") and {"lat_g", "long_g"} <= names:
        # The stored channels as logged; ``imu_swapped`` reads them exchanged.
        lat_col, long_col = ("lat_g", "long_g") if src == "imu" else ("long_g", "lat_g")
        base = ts.filter(pc.is_valid(ts.column(lat_col)))
        if windowed and len(base):
            base = base.filter(pa.array(verdict.window_codes(_times(base)) == b"I"))
        if len(base):
            t_b = _times(base)
            lat = np.asarray(base.column(lat_col).to_numpy(zero_copy_only=False), float)
            tl, xl = channel_series(ts, long_col, "t_session_s")
            if windowed:
                keep = verdict.window_codes(tl) == b"I"
                tl, xl = tl[keep], xl[keep]
            same = tl.shape == t_b.shape and np.array_equal(tl, t_b)
            long = xl if same else _interp_series(t_b, tl, xl)
            parts.append(_kin_part(base, t_b, lat, long, src, speed, yaw_imu, gps))
        elif not windowed:
            return None
    if "speed_ms" in names and speed[0].size:
        base = ts.filter(pc.is_valid(ts.column("speed_ms")))
        codes = verdict.window_codes(_times(base)) if windowed else None
        fallback = verdict.gps_fallback if src in ("imu", "imu_swapped") else src
        wanted: list[tuple[bytes | None, str]] = []
        if windowed:
            wanted = [(b"G", fallback), (b"D", DISCARD)]
        elif src in ("gps_channel", "gps_position"):
            wanted = [(None, src)]
        for code, source in wanted:
            sub = base if code is None else base.filter(pa.array(codes == code))
            if len(sub) == 0:
                continue
            t_s = _times(sub)
            if source == DISCARD:
                lat = long = np.full(t_s.shape, np.nan)
            else:
                series = _gps_series(source, t_s)
                if series is None:
                    if not windowed:
                        return None
                    lat = long = np.full(t_s.shape, np.nan)
                    source = DISCARD
                else:
                    lat, long = series
            parts.append(_kin_part(sub, t_s, lat, long, source, speed, yaw_imu, gps))
    if not parts:
        return None
    out = parts[0] if len(parts) == 1 else pa.concat_tables(parts)
    return out.sort_by("t_session_s") if len(parts) > 1 else out


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
            imu_ref = stored.imu_ref_lap if stored is not None else None
            verdict = validate_session(ts, ref, [imu_ref] if imu_ref is not None else None)
            changed = stored is not None and (
                verdict.accel_source != stored.accel_source
                or verdict.gyro_source != stored.gyro_source
                or verdict.windows != stored.windows
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
        counts = verdict.window_counts()
        row["win_imu"], row["win_gps"] = counts["imu"], counts["gps"]
        row["win_discard"], row["win_mismatch"] = counts["discard"], verdict.n_mismatch
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
    "mark_discarded",
    "position_kinematics",
    "reference_lap_order",
    "score_chosen",
    "score_windows",
    "session_kinematics",
    "validate_session",
    "kinematics_audit",
]
