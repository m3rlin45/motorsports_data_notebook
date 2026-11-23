"""The car's sample tables (lap simulation segment C1, issue #205).

What each car did on its build laps, as tables a person can read; the car
model (C2) takes its numbers from them. Everything is in the ground frame
on the position clock (:mod:`.frame`), and every sample knows where on the
layout's track it is, so the track's grade can be taken out. Since
version 5 (6 with an apex's cross readings, 7 with the track's line) the laterals are also kept in the tyres' frame, beside the
ground-frame ones (``*_tyre_g``: what the tyres hold on the road's bank,
:func:`~.bank.to_tyres_frame`; the same number on a track without a
bank), and an apex has the ``load`` on the tyres there (#312).

- **Full throttle** (``throttle``): the fixes where the throttle has been
  wide open for :data:`FULL_HELD_S` and stays so for the rest of the
  derivative's window, with no gear change within :data:`SHIFT_CLEAR_S`.
  The value is ``dV/dt + g·grade``, what the engine gives less drag. A
  throttle lift for a shift ends a run by itself; a flat shift (the KK-F's
  paddle) shows only in the stored ``gear``. Samples under
  :data:`STRAIGHT_LAT_G` of lateral are the *straight table*; the rest
  are kept for the cost of cornering at full throttle.
- **Coast** (``coast``): throttle closed and brakes off, in a straight
  line. The lift-off deceleration: drag, rolling and engine braking.
- **Braking zones** (``brake``): one row per brake application, from the
  brake pressure where it is logged and from ``dV/dt`` under
  :data:`DECEL_ON_G` where it is not: where it began, its peak and
  sustained deceleration with the grade taken out, and the lateral g
  alongside.
- **Apexes** (``apex``): per lap and per corner of the track's corner
  table: the speed at the apex, ``V²κ`` of the track there, the lap's own
  peak ground-frame lateral and the accelerometer's.
- **Speed traces** (``speed``): every lap's speed at the track's nodes,
  what a simulated lap is compared with.

The throttle is read relative to the session's own closed and wide-open
readings (its 1st and 99th percentiles while moving), so one rule serves a
pedal in per cent, the KK-F's throttle angle (5° closed, 84° open) and the
SFJ's channel that tops out at 53 %. A brake is on above
:data:`BRAKE_ON_SHARE` of the session's hardest application, so the rule
does not care that one logger stores bar and another ECU counts. Channels
logged on change (the KK-F's brake pressure) hold their last value.

Populations: the dry build laps and the wet build laps of
:func:`~.geometry.clean_lap_index`, told apart by ``basis``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ...tire_etl import dataset
from ...tire_etl.aggregates import channel_series
from ...tire_etl.kinematics import G_MS2
from . import bank as bankmod
from . import frame as frm
from . import geometry as geo
from . import track as trk
from .gates import GateItem, GateReport, md_table

logger = logging.getLogger(__name__)

TABLES_VERSION = 7
TABLE_NAMES = ("sessions", "throttle", "coast", "brake", "apex")
#: Cars the programme fits a model for; the others get tables where their
#: channels allow and are listed.
MODEL_CARS = ("Inferno 86", "KK-SII", "KK-F")

#: Throttle position, 0 closed to 1 wide open, at or above which it is full.
FULL_THROTTLE = 0.95
CLOSED_THROTTLE = 0.05
#: The closed-to-open span of a throttle channel must be this share of its
#: wide-open reading, or the channel is not a throttle that moved.
THROTTLE_MIN_SPAN = 0.3
FULL_HELD_S = 0.8
#: A selection must also hold this long after the sample: half the window
#: of the ground-frame derivative.
WINDOW_AFTER_S = frm.GROUND_SMOOTH_S / 2.0
COAST_HELD_S = 0.3
SHIFT_CLEAR_S = 0.3
STRAIGHT_LAT_G = 0.15
MIN_SPEED_MS = 8.0
#: A channel logged on change is held this long at most (its heartbeat is 2 s).
HOLD_MAX_S = 2.5
BRAKE_ON_SHARE = 0.05
#: Under this (in the channel's own unit) at its hardest, a brake channel is dead.
BRAKE_MIN_RANGE = 3.0
DECEL_ON_G = 0.3
ZONE_MERGE_S = 0.2
ZONE_MIN_S = 0.3
#: A stop is from full throttle when the throttle was full this shortly
#: before the brake application.
LIFT_MAX_S = 1.0
#: The brake has built when its pressure is this share of the zone's highest.
BRAKE_BUILD_SHARE = 0.8
SPEED_BIN_MS = 5.0
#: A speed bin of a straight table counts with this many samples.
BIN_MIN_SAMPLES = 50
GRIP_LIMITED_SHARE = 0.85
MIN_VOTE_DAYS = 3
APEX_COVER_MIN = 0.99
#: The road at an apex (its bank, the load on the tyres) is the median
#: over this far either side of it.
APEX_ROAD_M = 10.0
#: A stop's held part needs this many fixes.
HELD_MIN_FIXES = 3


# ------------------------------------------------------------------ helpers


def hold_last(
    t_query: np.ndarray, t: np.ndarray, x: np.ndarray, max_age_s: float = HOLD_MAX_S
) -> np.ndarray:
    """The last sample of ``x`` (on the sorted clock ``t``) at or before
    each query time; NaN before the first sample and where the last one is
    older than ``max_age_s``."""
    out = np.full(np.shape(t_query), np.nan)
    if len(t) == 0:
        return out
    i = np.searchsorted(t, t_query, side="right") - 1
    ok = i >= 0
    j = np.clip(i, 0, len(t) - 1)
    ok &= (t_query - t[j]) <= max_age_s
    out[ok] = x[j][ok]
    return out


def held_for(t: np.ndarray, cond: np.ndarray) -> np.ndarray:
    """Seconds for which ``cond`` has held without a break up to each
    sample (0 where it does not hold). The series' first sample counts as a
    break: what came before is not known."""
    cond = np.asarray(cond, bool)
    if len(t) == 0:
        return np.zeros(0)
    marks = np.where(~cond, t, -np.inf)
    marks[0] = t[0]
    return np.asarray(np.where(cond, t - np.maximum.accumulate(marks), 0.0))


def held_after(t: np.ndarray, cond: np.ndarray) -> np.ndarray:
    """Seconds for which ``cond`` goes on holding after each sample."""
    cond = np.asarray(cond, bool)
    if len(t) == 0:
        return np.zeros(0)
    marks = np.where(~cond, t, np.inf)
    marks[-1] = t[-1]
    nxt = np.minimum.accumulate(marks[::-1])[::-1]
    return np.asarray(np.where(cond, nxt - t, 0.0))


def runs(cond: np.ndarray) -> list[tuple[int, int]]:
    """``(first, last)`` index of every run of True."""
    cond = np.asarray(cond, bool)
    if not cond.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[False], cond, [False]]).astype(np.int8)))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def _channel(
    root: Path, session_id: str, family: str, names: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``name -> (t, values)`` of the channels of ``family`` the session has,
    each on its own clock, sorted, non-finite samples dropped."""
    tbl = dataset.load_group(root, session_id, family, names)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if tbl is None:
        return out
    for name in names:
        t, x = channel_series(tbl, name, "t_session_s")
        ok = np.isfinite(t) & np.isfinite(x)
        if ok.sum() < 10:
            continue
        order = np.argsort(t[ok], kind="stable")
        out[name] = (t[ok][order], x[ok][order])
    return out


# ------------------------------------------------------------- one session


@dataclass
class Signals:
    """One session on the position clock, with what the tables select on."""

    geom: geo.SessionGeometry
    lat_g: np.ndarray  # ground frame
    long_g: np.ndarray
    throttle: np.ndarray  # 0 closed .. 1 wide open; NaN without a channel
    brake: np.ndarray  # as logged; NaN without a channel
    brake_on: np.ndarray  # bool
    gear: np.ndarray
    rpm: np.ndarray
    lat_acc: np.ndarray  # accelerometer lateral over 0.5 s on this clock; NaN unless IMU
    info: dict  # the session's row of the ``sessions`` table
    #: The accelerometer's longitudinal, as ``lat_acc``; None: not read.
    long_acc: np.ndarray | None = None
    #: ``lat_g`` in the tyres' frame, once the session is laid on a track
    #: (:func:`in_tyres_frame`); None: as ``lat_g`` (a flat road).
    lat_tyre: np.ndarray | None = None

    @property
    def lat_tyre_g(self) -> np.ndarray:
        return self.lat_g if self.lat_tyre is None else self.lat_tyre


def session_signals(
    root: Path, session_id: str, layout_id: str, clean_laps: pd.DataFrame | None = None
) -> Signals | None:
    """Load one session's geometry, ground-frame series and controls.

    ``clean_laps`` (``t_start_s, t_end_s``) are the laps the session's
    frame record is measured on; without them only the accelerometer's
    delay is measured.
    """
    geom = geo.load_session_geometry(root, session_id, layout_id)
    if geom is None:
        return None
    t = geom.t
    n = len(t)
    lat_g, long_g = frm.ground_series(geom)
    moving = np.isfinite(geom.speed) & (geom.speed >= MIN_SPEED_MS)
    info: dict = {"session_id": session_id, "lag_s": geom.lag.tau_s}

    controls = _channel(root, session_id, "controls", ["throttle_deg", "throttle_pct", "brake_bar"])
    drive = _channel(root, session_id, "drivetrain", ["gear", "engine_rpm"])

    throttle = np.full(n, np.nan)
    info.update(throttle_channel="", throttle_closed=np.nan, throttle_open=np.nan)
    for name in ("throttle_deg", "throttle_pct"):  # the angle is on every KK-F session
        if name not in controls:
            continue
        raw = hold_last(t, *controls[name])
        use = moving & np.isfinite(raw)
        if use.sum() < 100:
            continue
        closed, wide = (float(v) for v in np.percentile(raw[use], [1.0, 99.0]))
        if wide <= 0 or wide - closed < THROTTLE_MIN_SPAN * abs(wide):
            continue
        throttle = (raw - closed) / (wide - closed)
        info.update(throttle_channel=name, throttle_closed=closed, throttle_open=wide)
        break

    brake = np.full(n, np.nan)
    info.update(brake_channel="", brake_max=np.nan, brake_on_at=np.nan)
    if "brake_bar" in controls:
        raw = hold_last(t, *controls["brake_bar"])
        use = moving & np.isfinite(raw)
        hardest = float(np.percentile(raw[use], 99.5)) if use.sum() >= 100 else float("nan")
        if np.isfinite(hardest) and hardest >= BRAKE_MIN_RANGE:
            brake = raw
            info.update(
                brake_channel="brake_bar", brake_max=hardest, brake_on_at=BRAKE_ON_SHARE * hardest
            )
    if info["brake_channel"]:
        brake_on = np.nan_to_num(brake, nan=0.0) >= info["brake_on_at"]
    else:
        with np.errstate(invalid="ignore"):
            brake_on = np.nan_to_num(long_g, nan=0.0) <= -DECEL_ON_G

    gear = hold_last(t, *drive["gear"]) if "gear" in drive else np.full(n, np.nan)
    rpm = hold_last(t, *drive["engine_rpm"]) if "engine_rpm" in drive else np.full(n, np.nan)
    info.update(has_gear=bool(np.isfinite(gear).any()), has_rpm=bool(np.isfinite(rpm).any()))

    lat_acc = np.full(n, np.nan)
    long_acc = np.full(n, np.nan)
    record = frm.FrameRecord(session_id=session_id)
    inertial = frm.load_inertial(root, session_id)
    if inertial is not None and inertial.is_imu.any():
        laps = (
            clean_laps if clean_laps is not None else pd.DataFrame(columns=["t_start_s", "t_end_s"])
        )
        record = frm.session_frame(geom, inertial, laps)
        if np.isfinite(record.accel_delay_s):
            acc, lng, is_imu = frm._accel_on_position_clock(geom, inertial, record.accel_delay_s)
            lat_acc = np.where(is_imu, acc, np.nan)
            long_acc = np.where(is_imu, lng, np.nan)
    info.update(
        has_accel=bool(np.isfinite(lat_acc).any()),
        accel_delay_s=record.accel_delay_s,
        frame_lat_slope=record.lat_slope if record.fitted else np.nan,
        frame_lat_zero_g=record.lat_zero_g if record.fitted else np.nan,
        frame_long_zero_g=record.long_zero_g if record.fitted else np.nan,
    )
    return Signals(
        geom=geom,
        lat_g=lat_g,
        long_g=long_g,
        throttle=throttle,
        brake=brake,
        brake_on=np.asarray(brake_on, bool),
        gear=gear,
        rpm=rpm,
        lat_acc=lat_acc,
        info=info,
        long_acc=long_acc,
    )


def in_tyres_frame(sig: Signals, track: trk.TrackModel, s: np.ndarray) -> np.ndarray:
    """The session's ground-frame lateral as what the tyres hold on the
    road's bank at each fix's place on ``track`` (``s`` from
    :func:`track_position`); ``lat_g`` itself where the track has no bank
    or the fix is on no lap."""
    left = bankmod.road_arrays(track)[0]
    at = np.zeros(len(s))
    ok = np.isfinite(s)
    at[ok] = np.interp(s[ok], track.s_m, left, period=track.length_m)
    return bankmod.to_tyres_frame(sig.lat_g, at)


def track_position(
    sig: Signals, track: trk.TrackModel, laps: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(s, grade, lap_num)`` per fix: distance along ``track`` (m, within
    ``[0, length)``) and the track's grade there, for the fixes inside the
    laps of ``laps`` (``lap_num, t_start_s, t_end_s``); NaN and −1 elsewhere."""
    geom = sig.geom
    n = len(geom)
    s = np.full(n, np.nan)
    lap_of = np.full(n, -1, dtype=np.int64)
    length = track.length_m
    xr = np.concatenate([track.x_m, track.x_m[:1]])
    yr = np.concatenate([track.y_m, track.y_m[:1]])
    for lap_num, a, b in zip(laps["lap_num"], laps["t_start_s"], laps["t_end_s"]):
        i0, i1 = (int(i) for i in np.searchsorted(geom.t, [a, b]))
        if i1 - i0 < 10:
            continue
        x, y = geom.x[i0:i1], geom.y[i0:i1]
        own = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        guess = own * (length / own[-1]) if own[-1] > 0 else own
        along, _ = trk.project_onto(xr, yr, x, y, guess, closed=True)
        s[i0:i1] = np.mod(along, length)
        lap_of[i0:i1] = int(lap_num)
    grade = np.full(n, np.nan)
    ok = np.isfinite(s)
    grade[ok] = np.interp(s[ok], track.s_m, track.grade, period=length)
    return s, grade, lap_of


def _f32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def throttle_samples(
    sig: Signals, s: np.ndarray, grade: np.ndarray, lap_of: np.ndarray
) -> pd.DataFrame:
    """The full-throttle fixes of the session's laps (module docstring).
    ``a_g`` is ``dV/dt`` plus the grade, in g; ``straight`` marks the ones
    under :data:`STRAIGHT_LAT_G` of lateral."""
    t = sig.geom.t
    full = np.nan_to_num(sig.throttle, nan=0.0) >= FULL_THROTTLE
    pick = (held_for(t, full) >= FULL_HELD_S) & (held_after(t, full) >= WINDOW_AFTER_S)
    pick &= _base_mask(sig, s, grade, lap_of)
    if np.isfinite(sig.gear).any():
        same = np.concatenate([[True], np.diff(np.nan_to_num(sig.gear, nan=-1.0)) == 0])
        pick &= (held_for(t, same) >= SHIFT_CLEAR_S) & (held_after(t, same) >= SHIFT_CLEAR_S)
    return _sample_frame(sig, s, grade, lap_of, pick, straight=True)


def coast_samples(
    sig: Signals, s: np.ndarray, grade: np.ndarray, lap_of: np.ndarray
) -> pd.DataFrame:
    """The fixes with the throttle closed and the brakes off, in a straight
    line. Empty for a session without both channels: a deceleration cannot
    say by itself that the brakes are off."""
    t = sig.geom.t
    if not sig.info["throttle_channel"] or not sig.info["brake_channel"]:
        return _sample_frame(sig, s, grade, lap_of, np.zeros(len(t), bool), straight=False)
    off = (np.nan_to_num(sig.throttle, nan=1.0) <= CLOSED_THROTTLE) & ~sig.brake_on
    pick = (held_for(t, off) >= COAST_HELD_S) & (held_after(t, off) >= WINDOW_AFTER_S)
    pick &= _base_mask(sig, s, grade, lap_of)
    with np.errstate(invalid="ignore"):
        pick &= np.abs(sig.lat_g) < STRAIGHT_LAT_G
    return _sample_frame(sig, s, grade, lap_of, pick, straight=False)


def _base_mask(sig: Signals, s: np.ndarray, grade: np.ndarray, lap_of: np.ndarray) -> np.ndarray:
    v = sig.geom.speed
    ok = (lap_of >= 0) & np.isfinite(s) & np.isfinite(grade) & np.isfinite(v)
    ok &= np.isfinite(sig.lat_g) & np.isfinite(sig.long_g)
    with np.errstate(invalid="ignore"):
        ok &= v >= MIN_SPEED_MS
    return np.asarray(ok)


def _sample_frame(
    sig: Signals,
    s: np.ndarray,
    grade: np.ndarray,
    lap_of: np.ndarray,
    pick: np.ndarray,
    *,
    straight: bool,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "lap_num": lap_of[pick].astype(np.int32),
            "t_s": sig.geom.t[pick],
            "s_m": _f32(s[pick]),
            "v_ms": _f32(sig.geom.speed[pick]),
            "a_g": _f32(sig.long_g[pick] + grade[pick]),
            "lat_g": _f32(sig.lat_g[pick]),
            "grade": _f32(grade[pick]),
            "gear": _f32(sig.gear[pick]),
            "rpm": _f32(sig.rpm[pick]),
        }
    )
    if straight:
        out["straight"] = np.abs(out["lat_g"].to_numpy()) < STRAIGHT_LAT_G
    out["lat_tyre_g"] = _f32(sig.lat_tyre_g[pick])
    return out


BRAKE_COLUMNS = [
    "lap_num",
    "source",
    "t_on_s",
    "duration_s",
    "s_on_m",
    "s_off_m",
    "v_on_ms",
    "v_min_ms",
    "decel_p95_g",
    "decel_median_g",
    "v_peak_ms",
    "lat_peak_g",
    "lat_mean_g",
    "grade_mean",
    "brake_max",
    "lift_s",
    "build_s",
    "lat_peak_tyre_g",
    "lat_mean_tyre_g",
    "held_s",
    "decel_held_g",
    "decel_held_accel_g",
    "v_held_ms",
    "lat_held_tyre_g",
]


def brake_zones(sig: Signals, s: np.ndarray, grade: np.ndarray, lap_of: np.ndarray) -> pd.DataFrame:
    """One row per brake application that begins inside one of the laps
    (:data:`BRAKE_COLUMNS`).

    Applications closer than :data:`ZONE_MERGE_S` are one, and one shorter
    than :data:`ZONE_MIN_S` is dropped. ``decel_*`` are of ``−(dV/dt +
    g·grade)`` over the zone, in g: what the tyres and the air take out,
    the slope removed. ``lat_peak_g`` is ``|lateral|`` where the
    deceleration peaks and ``lat_mean_g`` its mean over the zone.
    ``source`` is ``pressure`` or ``decel``. A zone's peak comes from a
    0.5 s difference of speed, so a stab shorter than that reads low.
    ``lift_s`` is the time from the last sample at full throttle to the
    application (NaN when that is over :data:`LIFT_MAX_S` before: not a
    stop from full throttle) and ``build_s`` the time from the application
    to :data:`BRAKE_BUILD_SHARE` of the zone's highest pressure.

    ``lat_peak_tyre_g`` and ``lat_mean_tyre_g`` are the two laterals in
    the tyres' frame. The ``*held*`` columns are over the *held part* of
    the stop, the fixes where the pressure is at
    :data:`BRAKE_BUILD_SHARE` or more of the zone's highest (NaN without a
    pressure channel or with fewer than :data:`HELD_MIN_FIXES` of them):
    ``held_s`` its time, ``decel_held_g`` the mean deceleration there (a
    level the car held, not a peak), ``decel_held_accel_g`` the same from
    the accelerometer's longitudinal as the sensor reads it (it reads the
    slope with the deceleration, as ``decel_held_g`` has it taken out;
    NaN unless every fix is IMU-sourced), ``v_held_ms`` the mean speed
    and ``lat_held_tyre_g`` the mean lateral in the tyres' frame.
    """
    t = sig.geom.t
    v = sig.geom.speed
    on = sig.brake_on & np.isfinite(v)
    with np.errstate(invalid="ignore"):
        on &= v >= MIN_SPEED_MS
    # merge short releases
    merged = on.copy()
    spans = runs(on)
    for (_, last), (first, _) in zip(spans[:-1], spans[1:]):
        if t[first] - t[last] <= ZONE_MERGE_S and sig.geom.seg[first] == sig.geom.seg[last]:
            merged[last:first] = True
    source = "pressure" if sig.info["brake_channel"] else "decel"
    full = np.nan_to_num(sig.throttle, nan=0.0) >= FULL_THROTTLE
    with np.errstate(invalid="ignore"):
        decel = -(sig.long_g + grade)
        lat_abs = np.abs(sig.lat_g)
        tyre_abs = np.abs(sig.lat_tyre_g)
    long_acc = sig.long_acc if sig.long_acc is not None else np.full(len(t), np.nan)
    period = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    rows: list[dict] = []
    for first, last in runs(merged):
        if lap_of[first] < 0 or t[last] - t[first] < ZONE_MIN_S:
            continue
        sl = slice(first, last + 1)
        d = decel[sl]
        if np.isfinite(d).sum() < 3:
            continue
        peak = first + int(np.nanargmax(d))
        # how the stop began: from the last sample at full throttle to the
        # application, and from the application to most of its pressure
        lift, build = np.nan, np.nan
        back = int(np.searchsorted(t, t[first] - LIFT_MAX_S))
        was_full = np.flatnonzero(full[back:first])
        if was_full.size:
            # the throttle left full somewhere in the sample after its last
            # full one and the brake came on somewhere in the sample before
            # its first: between the middles of the two
            last_full = back + int(was_full[-1])
            lift = float(0.5 * (t[first] + t[first - 1]) - 0.5 * (t[last_full] + t[last_full + 1]))
        if source == "pressure":
            pressure = sig.brake[sl]
            most = np.flatnonzero(pressure >= BRAKE_BUILD_SHARE * np.nanmax(pressure))
            if most.size:
                build = float(t[first + most[0]] - t[first])
        held = {
            "held_s": np.nan,
            "decel_held_g": np.nan,
            "decel_held_accel_g": np.nan,
            "v_held_ms": np.nan,
            "lat_held_tyre_g": np.nan,
        }
        if source == "pressure":
            part = first + np.flatnonzero(
                (sig.brake[sl] >= BRAKE_BUILD_SHARE * np.nanmax(sig.brake[sl])) & np.isfinite(d)
            )
            if part.size >= HELD_MIN_FIXES:
                acc = long_acc[part]
                held = {
                    "held_s": float(part.size * period),
                    "decel_held_g": float(np.mean(decel[part])),
                    "decel_held_accel_g": (
                        float(-np.mean(acc)) if np.isfinite(acc).all() else np.nan
                    ),
                    "v_held_ms": float(np.mean(v[part])),
                    "lat_held_tyre_g": float(np.nanmean(tyre_abs[part])),
                }
        rows.append(
            {
                "lap_num": int(lap_of[first]),
                "source": source,
                "t_on_s": float(t[first]),
                "duration_s": float(t[last] - t[first]),
                "s_on_m": float(s[first]),
                "s_off_m": float(s[last]),
                "v_on_ms": float(v[first]),
                "v_min_ms": float(np.nanmin(v[sl])),
                "decel_p95_g": float(np.nanpercentile(d, 95.0)),
                "decel_median_g": float(np.nanmedian(d)),
                "v_peak_ms": float(v[peak]),
                "lat_peak_g": float(lat_abs[peak]),
                "lat_mean_g": float(np.nanmean(lat_abs[sl])),
                "grade_mean": float(np.nanmean(grade[sl])),
                "brake_max": float(np.nanmax(sig.brake[sl])) if source == "pressure" else np.nan,
                "lift_s": lift,
                "build_s": build,
                "lat_peak_tyre_g": float(tyre_abs[peak]),
                "lat_mean_tyre_g": float(np.nanmean(tyre_abs[sl])),
                **held,
            }
        )
    return pd.DataFrame(rows, columns=BRAKE_COLUMNS)


APEX_COLUMNS = [
    "lap_num",
    "corner",
    "sign",
    "s_apex_m",
    "v_apex_ms",
    "v_min_ms",
    "lat_track_g",
    "lat_doppler_g",
    "lat_accel_g",
    "lat_tyre_g",
    "load",
    "lat_tyre_cross_g",
    "lat_accel_cross_g",
    "lat_track_tyre_g",
]


def apex_rows(
    sig: Signals, track: trk.TrackModel, s: np.ndarray, lap_of: np.ndarray
) -> pd.DataFrame:
    """One row per lap and corner of the track's corner table
    (:data:`APEX_COLUMNS`), all lateral g positive into the corner.

    ``v_apex_ms`` is the speed at the fix nearest the track's apex,
    ``lat_track_g`` is ``V²|κ|`` of the track there, ``lat_doppler_g`` the
    lap's own largest ground-frame lateral inside the corner and
    ``lat_accel_g`` the accelerometer's (as the sensor reads it, over
    0.5 s on the position clock; NaN on a lap that is not IMU-sourced
    throughout the corner). A corner that spans the start/finish line is
    read from both ends of the lap.

    ``lat_tyre_g`` is ``lat_doppler_g`` in the tyres' frame: what the
    tyres hold of it on the road's bank at the apex (the bank's median
    over ± :data:`APEX_ROAD_M`). ``load`` is the load on the tyres there
    over the car's weight at the apex speed, ``cos θ·cos b + V²(κ_v +
    |κ|·sin b) / g`` (the same median): the lateral over it is the share
    of the grip the pass used.

    The largest value of a noisy series is raised by its own noise, so
    each sensor is also read where the *other* one peaks, a choice its
    own noise has no part in: ``lat_tyre_cross_g`` is the Doppler's
    lateral (tyres' frame) at the fix of the accelerometer's largest
    value, and ``lat_accel_cross_g`` the accelerometer's at the fix of
    the Doppler's (both NaN without the accelerometer). They read low
    where the two peaks are apart on a corner that is not flat.
    ``lat_track_tyre_g`` is ``lat_track_g`` in the tyres' frame: the pass
    as the solver would have it, on the track's line and not the lap's
    own path.
    """
    length = track.length_m
    v = sig.geom.speed
    left, cos, kv = bankmod.road_arrays(track)
    reach = max(int(round(APEX_ROAD_M / track.ds_m)), 0)
    road: list[tuple[float, np.ndarray, np.ndarray]] = []
    for corner in track.corners:
        near = (int(round(corner["s_apex"] / track.ds_m)) + np.arange(-reach, reach + 1)) % track.n
        road.append((float(np.median(left[near])), cos[near], kv[near]))
    rows: list[dict] = []
    for lap_num in np.unique(lap_of[lap_of >= 0]):
        in_lap = lap_of == lap_num
        for k, corner in enumerate(track.corners):
            s_in, s_out, s_apex = corner["s_in"], corner["s_out"], corner["s_apex"]
            with np.errstate(invalid="ignore"):
                inside = (s >= s_in) & (s <= s_out) if s_in <= s_out else (s >= s_in) | (s <= s_out)
            idx = np.flatnonzero(in_lap & inside & np.isfinite(v) & np.isfinite(sig.lat_g))
            if idx.size < 3:
                continue
            sign = float(corner["sign"])
            off = np.abs(np.mod(s[idx] - s_apex + length / 2.0, length) - length / 2.0)
            at = idx[int(np.argmin(off))]
            acc = sign * sig.lat_acc[idx]
            own = sign * sig.lat_g[idx]
            doppler = float(np.max(own))
            bank_left, cos_near, kv_near = road[k]
            both = bool(np.isfinite(acc).all())
            tyre_cross = (
                float(bankmod.to_tyres_frame(own[int(np.argmax(acc))], sign * bank_left))
                if both
                else np.nan
            )
            rows.append(
                {
                    "lap_num": int(lap_num),
                    "corner": k,
                    "sign": int(corner["sign"]),
                    "s_apex_m": float(s_apex),
                    "v_apex_ms": float(v[at]),
                    "v_min_ms": float(np.min(v[idx])),
                    "lat_track_g": float(v[at] ** 2 * abs(corner["kappa_apex"]) / G_MS2),
                    "lat_doppler_g": doppler,
                    "lat_accel_g": float(np.max(acc)) if both else np.nan,
                    # positive into the corner, so the lean into it
                    "lat_tyre_g": float(bankmod.to_tyres_frame(doppler, sign * bank_left)),
                    "load": float(np.median(cos_near + v[at] ** 2 * kv_near / G_MS2)),
                    "lat_tyre_cross_g": tyre_cross,
                    "lat_accel_cross_g": float(acc[int(np.argmax(own))]) if both else np.nan,
                    # the pass on the track's line: V²|κ| of the track at its apex
                    "lat_track_tyre_g": float(
                        bankmod.to_tyres_frame(
                            v[at] ** 2 * abs(corner["kappa_apex"]) / G_MS2, sign * bank_left
                        )
                    ),
                }
            )
    return pd.DataFrame(rows, columns=APEX_COLUMNS)


def speed_on_track(
    sig: Signals, track: trk.TrackModel, s: np.ndarray, lap_of: np.ndarray
) -> tuple[list[int], np.ndarray]:
    """``(lap_nums, v)``: each lap's speed at the track's nodes (laps ×
    nodes, m/s), linear in distance between its fixes; a lap with fewer
    than 50 usable fixes is left out."""
    laps: list[int] = []
    rows: list[np.ndarray] = []
    v = sig.geom.speed
    for lap_num in np.unique(lap_of[lap_of >= 0]):
        m = (lap_of == lap_num) & np.isfinite(s) & np.isfinite(v)
        if m.sum() < 50:
            continue
        laps.append(int(lap_num))
        rows.append(np.interp(track.s_m, s[m], v[m], period=track.length_m))
    shape = (len(rows), track.n)
    return laps, (np.vstack(rows).astype(np.float32) if rows else np.zeros(shape, np.float32))


def session_tables(
    root: Path,
    session_id: str,
    track: trk.TrackModel,
    laps: pd.DataFrame,
    clean_laps: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame] | None:
    """The four sample tables of one session's ``laps`` (lap-index rows:
    ``lap_num, t_start_s, t_end_s`` and a ``basis`` column), its one-row
    ``sessions`` table, and ``speed`` (laps × nodes) with the laps it holds
    in ``speed_laps``; None without geometry. Every table carries
    ``session_id`` and, per lap, ``basis``."""
    sig = session_signals(root, session_id, track.layout_key, clean_laps)
    if sig is None:
        return None
    s, grade, lap_of = track_position(sig, track, laps)
    sig.lat_tyre = in_tyres_frame(sig, track, s)
    basis = dict(zip(laps["lap_num"].astype(int), laps["basis"]))
    out = {
        "throttle": throttle_samples(sig, s, grade, lap_of),
        "coast": coast_samples(sig, s, grade, lap_of),
        "brake": brake_zones(sig, s, grade, lap_of),
        "apex": apex_rows(sig, track, s, lap_of),
    }
    lap_nums, v_grid = speed_on_track(sig, track, s, lap_of)
    out["speed_laps"] = pd.DataFrame({"lap_num": np.asarray(lap_nums, dtype=np.int32)})
    for tbl in out.values():
        tbl.insert(0, "session_id", session_id)
        tbl.insert(1, "basis", tbl["lap_num"].map(basis).astype(object))
    out["speed"] = v_grid  # type: ignore[assignment]
    info = dict(sig.info)
    info["n_laps"] = int(len(np.unique(lap_of[lap_of >= 0])))
    out["sessions"] = pd.DataFrame([info])
    return out


# ------------------------------------------------------------ every session


@dataclass
class CarTables:
    """The sample tables of every car (:data:`TABLE_NAMES`); every row has
    ``session_id, car, date, layout_id`` and, but for ``sessions``,
    ``basis`` (``dry`` or ``wet``) and ``lap_num``."""

    sessions: pd.DataFrame
    throttle: pd.DataFrame
    coast: pd.DataFrame
    brake: pd.DataFrame
    apex: pd.DataFrame
    #: One row per lap with a speed trace: its keys, and ``row``, its row in
    #: ``speed[layout_id]``.
    speed_laps: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: ``layout_id -> (laps × nodes)`` speed at the layout's track nodes.
    speed: dict[str, np.ndarray] = field(default_factory=dict)

    def save(self, out_dir: Path, signature: str = "") -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in TABLE_NAMES + ("speed_laps",):
            getattr(self, name).to_parquet(out_dir / f"{name}.parquet", index=False)
        for layout, v in self.speed.items():
            np.save(out_dir / f"speed_{layout}.npy", v)
        (out_dir / "signature.txt").write_text(signature)

    @classmethod
    def load(cls, out_dir: Path) -> "CarTables":
        out_dir = Path(out_dir)
        tables = {n: pd.read_parquet(out_dir / f"{n}.parquet") for n in TABLE_NAMES}
        laps = pd.read_parquet(out_dir / "speed_laps.parquet")
        speed = {f.stem[len("speed_") :]: np.load(f) for f in sorted(out_dir.glob("speed_*.npy"))}
        return cls(**tables, speed_laps=laps, speed=speed)

    def lap_speeds(
        self, car: str, layout_id: str, basis: str = "dry"
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """The keys and the speed traces (laps × nodes) of a car's laps at
        a layout."""
        laps = self.speed_laps
        if laps.empty or layout_id not in self.speed:
            return laps.iloc[:0], np.zeros((0, 0), np.float32)
        own = laps[
            (laps["car"] == car) & (laps["layout_id"] == layout_id) & (laps["basis"] == basis)
        ]
        return own.reset_index(drop=True), self.speed[layout_id][own["row"].to_numpy()]


def build_laps(index: pd.DataFrame) -> pd.DataFrame:
    """The lap-index rows the tables are built from, with ``basis``."""
    rows = index[index["in_build"] | index["in_build_wet"]].copy()
    rows["basis"] = np.where(rows["in_build"], "dry", "wet")
    return rows


def load_track(
    root: Path, layout_key: str, *, tracks_dir: Path | None = None, cache_dir: Path | None = None
) -> trk.TrackModel:
    """The layout's dry track: the JSON gate B2 saved in ``tracks_dir``
    when there is one, else built now (and saved there)."""
    path = Path(tracks_dir) / f"{layout_key}.json" if tracks_dir is not None else None
    if path is not None and path.exists():
        return trk.TrackModel.load(path)
    model = trk.build_track_model(
        root, layout_key, cache_dir=cache_dir, inertial=False, n_boot=0
    ).model
    if path is not None:
        model.save(path)
    return model


def _tables_signature(root: Path, tracks_dir: Path | None, layouts: list[str]) -> str:
    h = hashlib.sha256(f"tables-v{TABLES_VERSION}:{geo._index_signature(root)}".encode())
    for key in layouts:
        f = Path(tracks_dir) / f"{key}.json" if tracks_dir is not None else None
        if f is not None and f.exists():
            st = f.stat()
            h.update(f"{f.name}:{st.st_mtime_ns}:{st.st_size};".encode())
    return h.hexdigest()


def car_tables(
    root: Path, *, cache_dir: Path | None = None, tracks_dir: Path | None = None
) -> CarTables:
    """The sample tables over every session with build laps, dry or wet.

    Reading every session takes a minute or two, so the tables are kept in
    ``<cache_dir>/car_tables/`` and reused while the dataset's files and
    the tracks are the ones they were built from.
    """
    root = Path(root)
    index = geo.clean_lap_index(root, cache_dir=cache_dir)
    rows = build_laps(index)
    layouts = sorted(str(k) for k in rows["layout_id"].unique())
    store = Path(cache_dir) / "car_tables" if cache_dir is not None else None
    tracks = {
        key: load_track(root, key, tracks_dir=tracks_dir, cache_dir=cache_dir) for key in layouts
    }
    sig = _tables_signature(root, tracks_dir, layouts)
    if store is not None and (store / "signature.txt").exists():
        if (store / "signature.txt").read_text() == sig:
            return CarTables.load(store)
    clean = index[index["clean"]]
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLE_NAMES + ("speed_laps",)}
    speed: dict[str, list[np.ndarray]] = {}
    for sid, grp in rows.groupby("session_id", sort=True):
        first = grp.iloc[0]
        try:
            got = session_tables(
                root,
                str(sid),
                tracks[str(first["layout_id"])],
                grp,
                clean[clean["session_id"] == sid],
            )
        except Exception:  # noqa: BLE001 - one unreadable session must not stop the tables
            logger.warning("car tables: failed for %s", sid, exc_info=True)
            continue
        if got is None:
            continue
        layout = str(first["layout_id"])
        traces = speed.setdefault(layout, [])
        v_grid = got.pop("speed")
        got["speed_laps"]["row"] = sum(len(v) for v in traces) + np.arange(len(v_grid))
        traces.append(np.asarray(v_grid))
        for name, tbl in got.items():
            for i, col in enumerate(("car", "date", "layout_id", "logger_id")):
                tbl.insert(1 + i, col, str(first[col]))
            parts[name].append(tbl)
    tables = CarTables(
        **{
            name: (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
            for name, frames in parts.items()
        },
        speed={k: np.vstack(v) for k, v in speed.items() if v},
    )
    if store is not None:
        tables.save(store, sig)
    return tables


# ---------------------------------------------------------------- summaries


def straight_table(
    throttle: pd.DataFrame, *, by: tuple[str, ...] = ("car", "layout_id")
) -> pd.DataFrame:
    """Median full-throttle acceleration per :data:`SPEED_BIN_MS` bin of the
    straight-line samples: ``v_bin_ms`` (the bin's lower edge), ``a_g``,
    ``n``, ``sessions`` and ``days``, grouped by ``by``. The median is over
    samples."""
    cols = list(by) + ["v_bin_ms", "a_g", "v_ms", "n", "sessions", "days"]
    if throttle.empty:
        return pd.DataFrame(columns=cols)
    st = throttle[throttle["straight"]].copy()
    st["v_bin_ms"] = np.floor(st["v_ms"] / SPEED_BIN_MS) * SPEED_BIN_MS
    out = (
        st.groupby(list(by) + ["v_bin_ms"])
        .agg(
            a_g=("a_g", "median"),
            v_ms=("v_ms", "median"),
            n=("a_g", "size"),
            sessions=("session_id", "nunique"),
            days=("date", "nunique"),
        )
        .reset_index()
    )
    return out[cols]


def apex_summary(apex: pd.DataFrame) -> pd.DataFrame:
    """Per car, layout, basis and corner: laps, the median apex speed, the
    median and 90th percentile of the laps' ground-frame peak, the 90th
    percentile of the track's ``V²κ`` and of the accelerometer's peak, and
    ``grip_limited``: the corner's p90 is within 15 % of the highest of
    the car's corners at the layout (on that basis)."""
    keys = ["car", "layout_id", "basis", "corner"]
    cols = keys + ["sign", "s_apex_m", "laps", "days", "v_apex_ms", "lat_p50_g", "lat_p90_g"]
    cols += ["track_p90_g", "accel_p90_g", "grip_limited"]
    if apex.empty:
        return pd.DataFrame(columns=cols)
    cells = apex.groupby(keys)
    out = cells.agg(
        sign=("sign", "first"),
        s_apex_m=("s_apex_m", "first"),
        laps=("lat_doppler_g", "size"),
        days=("date", "nunique"),
        v_apex_ms=("v_apex_ms", "median"),
        lat_p50_g=("lat_doppler_g", "median"),
    )
    # the three 90th percentiles of every corner in one grouped call
    p90 = cells[["lat_doppler_g", "lat_track_g", "lat_accel_g"]].quantile(0.9)
    out["lat_p90_g"] = p90["lat_doppler_g"]
    out["track_p90_g"] = p90["lat_track_g"]
    out["accel_p90_g"] = p90["lat_accel_g"]
    out = out.reset_index()
    top = out.groupby(keys[:3])["lat_p90_g"].transform("max")
    out["grip_limited"] = out["lat_p90_g"] >= GRIP_LIMITED_SHARE * top
    return out[cols]


def lateral_level(summary: pd.DataFrame) -> pd.DataFrame:
    """Per car, layout and basis: the median over the grip-limited corners
    of the corner's p90 (``level_g``), how many corners that is, and the
    same from the accelerometer."""
    lim = summary[summary["grip_limited"]]
    return (
        lim.groupby(["car", "layout_id", "basis"])
        .agg(
            corners=("corner", "size"),
            laps=("laps", "max"),
            days=("days", "max"),
            level_g=("lat_p90_g", "median"),
            accel_g=("accel_p90_g", "median"),
        )
        .reset_index()
    )


# ------------------------------------------------------------------ the gate


def cell_counts(tables: CarTables, rows: pd.DataFrame) -> pd.DataFrame:
    """Per car, layout and basis: sessions, days and laps of the build
    population (``rows``) and what each table holds of them."""
    keys = ["car", "layout_id", "basis"]
    out = (
        rows.groupby(keys)
        .agg(sessions=("session_id", "nunique"), days=("date", "nunique"), laps=("lap_num", "size"))
        .reset_index()
    )

    def _add(name: str, frame: pd.DataFrame) -> None:
        cnt = frame.groupby(keys).size() if len(frame) else pd.Series(dtype=float)
        out[name] = [int(cnt.get(tuple(k), 0)) for k in out[keys].itertuples(index=False)]

    th = tables.throttle
    _add("full_throttle", th)
    _add("straight", th[th["straight"]] if len(th) else th)
    _add("coast", tables.coast)
    _add("brake_zones", tables.brake)
    _add("apex_rows", tables.apex)
    seen = (
        tables.apex.drop_duplicates(["session_id", "lap_num"]) if len(tables.apex) else tables.apex
    )
    _add("apex_laps", seen)
    return out


def check_envelope_samples(
    root: Path, *, cache_dir: Path | None = None, tracks_dir: Path | None = None
) -> GateReport:
    """Gate C1: the sample tables exist for every car and layout the model
    will be fitted on, and say what they cover."""
    index = geo.clean_lap_index(root, cache_dir=cache_dir)
    rows = build_laps(index)
    tables = car_tables(root, cache_dir=cache_dir, tracks_dir=tracks_dir)
    rep = GateReport("C1", "the car's sample tables (`lapsim check-envelope-samples`)")
    counts = cell_counts(tables, rows)
    sess = tables.sessions
    rep.notes.append(
        f"{len(rows)} build laps ({int((rows['basis'] == 'dry').sum())} dry, "
        f"{int((rows['basis'] == 'wet').sum())} wet) in {rows['session_id'].nunique()} sessions; "
        f"tables hold {len(tables.throttle)} full-throttle samples "
        f"({int(tables.throttle['straight'].sum()) if len(tables.throttle) else 0} in a straight "
        f"line), {len(tables.coast)} coast samples, {len(tables.brake)} braking zones and "
        f"{len(tables.apex)} apex rows from {len(sess)} sessions."
    )

    # 1. Completeness, on the cells the car model is fitted on.
    dry = counts[(counts["basis"] == "dry") & counts["car"].isin(MODEL_CARS)]
    cover = dry["apex_laps"] / dry["laps"].clip(lower=1)
    bad = dry[(dry["straight"] == 0) | (dry["brake_zones"] == 0) | (cover < APEX_COVER_MIN)]
    rep.items.append(
        GateItem(
            "Every model car's cell has its tables",
            bool(len(dry)) and bad.empty,
            f"{len(dry) - len(bad)} of {len(dry)} cells; apex rows for "
            f"{dry['apex_laps'].sum()} of {dry['laps'].sum()} laps",
            f"straight samples, braking zones, and an apex row for ≥ {APEX_COVER_MIN:.0%} of the "
            "build laps, per model car and layout (dry)",
            "Per car, layout and basis (`full_throttle`, `straight`, `coast`: samples; "
            "`apex_laps`: laps with apex rows):\n\n"
            + md_table(counts)
            + ("\n\nCells without:\n\n" + md_table(bad) if len(bad) else ""),
        )
    )

    # 2. Thin cells.
    thin = counts[(counts["basis"] == "dry") & (counts["days"] < MIN_VOTE_DAYS)]
    rep.items.append(
        GateItem(
            "Cells with fewer than three days",
            None,
            "; ".join(f"{r.car} at {r.layout_id} ({r.days})" for r in thin.itertuples()) or "none",
            "listed; no vote in C3",
        )
    )

    # 3. Channel coverage.
    if len(sess):
        s2 = sess.assign(
            throttle=sess["throttle_channel"].replace("", "none"),
            brake=sess["brake_channel"].replace("", "none"),
        )
        cov = (
            s2.groupby(["car", "logger_id", "throttle", "brake"])
            .agg(
                sessions=("session_id", "size"),
                closed=("throttle_closed", "median"),
                wide_open=("throttle_open", "median"),
                brake_hardest=("brake_max", "median"),
                gear=("has_gear", "sum"),
                rpm=("has_rpm", "sum"),
                accel=("has_accel", "sum"),
            )
            .reset_index()
        )
        cov["full_at"] = cov["closed"] + FULL_THROTTLE * (cov["wide_open"] - cov["closed"])
        cov["brake_on_at"] = BRAKE_ON_SHARE * cov["brake_hardest"]
        by_car = s2.groupby("car").agg(
            sessions=("session_id", "size"),
            throttle=("throttle_channel", lambda x: int((x != "").sum())),
            brake=("brake_channel", lambda x: int((x != "").sum())),
        )
        rep.items.append(
            GateItem(
                "Channel coverage",
                None,
                "; ".join(
                    f"{car} throttle {r.throttle}/{r.sessions}, brake {r.brake}/{r.sessions}"
                    for car, r in by_car.iterrows()
                ),
                "",
                "Sessions per car, logger and channel; `closed`, `wide_open` and "
                "`brake_hardest` are the median of the sessions' own readings (the channel's "
                "unit), `full_at` and `brake_on_at` the thresholds that follow; `gear`, `rpm` "
                "and `accel` count sessions.\n\n" + md_table(cov),
            )
        )

    # 4. Straight tables.
    th = tables.throttle
    st = straight_table(th[th["basis"] == "dry"] if len(th) else th)
    solid = st[st["n"] >= BIN_MIN_SAMPLES]
    span = (
        solid.groupby(["car", "layout_id"])
        .agg(
            v_min_ms=("v_bin_ms", "min"),
            v_max_ms=("v_bin_ms", lambda x: x.max() + SPEED_BIN_MS),
            bins=("v_bin_ms", "size"),
            samples=("n", "sum"),
            days=("days", "max"),
        )
        .reset_index()
    )
    wide = solid.pivot_table(index=["car", "layout_id"], columns="v_bin_ms", values="a_g")
    wide.columns = [f"{int(c)}" for c in wide.columns]
    rep.items.append(
        GateItem(
            "Speed range of each straight table",
            None,
            "; ".join(
                f"{r.car} {r.layout_id} {r.v_min_ms:.0f}–{r.v_max_ms:.0f}"
                for r in span.itertuples()
            )
            + " m/s",
            f"bins of {SPEED_BIN_MS:.0f} m/s with ≥ {BIN_MIN_SAMPLES} samples",
            md_table(span)
            + "\n\nMedian `dV/dt + g·grade` (g) per bin, by the bin's lower edge (m/s), dry:\n\n"
            + md_table(wide.reset_index(), floats="{:.3f}"),
        )
    )

    # 5. Lateral level at grip-limited corners.
    summary = apex_summary(tables.apex)
    level = lateral_level(summary)
    dry_level = level[level["basis"] == "dry"]
    ratio = (
        dry_level[dry_level["days"] >= MIN_VOTE_DAYS]
        .groupby("car")["level_g"]
        .agg(lambda x: x.max() / x.min() if len(x) > 1 else np.nan)
    )
    rep.items.append(
        GateItem(
            "Lateral level at grip-limited corners",
            None,
            "max / min across layouts with three days: "
            + (
                "; ".join(f"{car} {v:.3f}" for car, v in ratio.dropna().items())
                or "no car with two such layouts"
            ),
            "reported (design review: 1.06–1.12 on the Inferno 86, ≤ 1.03 on the KK-SII)",
            "`level_g`: median over the grip-limited corners of the corner's p90 of the laps' "
            "own ground-frame peak; `accel_g`: the same from the accelerometer, in the sensor "
            "frame.\n\n"
            + md_table(level, floats="{:.3f}")
            + "\n\nPer corner (dry):\n\n"
            + md_table(summary[summary["basis"] == "dry"], floats="{:.3f}"),
        )
    )

    # 6. Wet over dry.
    both = level.pivot_table(index=["car", "layout_id"], columns="basis", values="level_g")
    if {"dry", "wet"} <= set(both.columns):
        both = both.dropna().reset_index()
        both["wet_over_dry"] = both["wet"] / both["dry"]
        wet_laps = counts[counts["basis"] == "wet"].set_index(["car", "layout_id"])
        both["wet_laps"] = [
            int(wet_laps["laps"].get(k, 0)) for k in zip(both["car"], both["layout_id"])
        ]
        both["wet_days"] = [
            int(wet_laps["days"].get(k, 0)) for k in zip(both["car"], both["layout_id"])
        ]
        rep.items.append(
            GateItem(
                "Wet over dry lateral level",
                None,
                "; ".join(f"{r.car} {r.layout_id} {r.wet_over_dry:.2f}" for r in both.itertuples())
                or "no cell with both",
                "reported (design review: 0.79 in the median)",
                "The wet population is B's wet build laps (sessions labelled wet and off the "
                "dry pace, judged against each other), not the lateral-level rule of the "
                "issue's text: the level is what is being measured here.\n\n"
                + md_table(both, floats="{:.3f}"),
            )
        )

    # 7. Braking and coasting.
    if len(tables.brake):
        br = tables.brake[tables.brake["basis"] == "dry"]
        zone = (
            br.groupby(["car", "layout_id", "source"])
            .agg(
                zones=("decel_p95_g", "size"),
                peak_p50_g=("decel_p95_g", "median"),
                peak_p90_g=("decel_p95_g", lambda x: x.quantile(0.9)),
                sustained_p50_g=("decel_median_g", "median"),
                lat_at_peak_g=("lat_peak_g", "median"),
            )
            .reset_index()
        )
        n_laps = counts[counts["basis"] == "dry"].set_index(["car", "layout_id"])["laps"]
        zone.insert(
            4,
            "per_lap",
            [
                z / max(int(n_laps.get((c, lay), 1)), 1)
                for z, c, lay in zip(zone["zones"], zone["car"], zone["layout_id"])
            ],
        )
        rep.items.append(
            GateItem(
                "Braking zones",
                None,
                "; ".join(
                    f"{car} {g['peak_p90_g'].min():.2f}–{g['peak_p90_g'].max():.2f} g"
                    for car, g in zone.groupby("car")
                ),
                "p90 over zones of the zone's p95, grade removed (drag is not)",
                md_table(zone, floats="{:.3f}"),
            )
        )
    if len(tables.coast):
        co = tables.coast[tables.coast["basis"] == "dry"].copy()
        co["v_bin_ms"] = np.floor(co["v_ms"] / (2 * SPEED_BIN_MS)) * 2 * SPEED_BIN_MS
        coast = co.groupby(["car", "v_bin_ms"]).agg(n=("a_g", "size"), a_g=("a_g", "median"))
        coast = coast[coast["n"] >= BIN_MIN_SAMPLES].reset_index()
        rep.items.append(
            GateItem(
                "Lift-off deceleration",
                None,
                "; ".join(
                    f"{car} {-g['a_g'].max():.2f}–{-g['a_g'].min():.2f} g"
                    for car, g in coast.groupby("car")
                )
                or "no bin with enough samples",
                "throttle closed, brakes off, straight line; median per 10 m/s",
                md_table(coast, floats="{:.3f}"),
            )
        )
    return rep


__all__ = [
    "CarTables",
    "MODEL_CARS",
    "Signals",
    "apex_rows",
    "apex_summary",
    "brake_zones",
    "build_laps",
    "car_tables",
    "cell_counts",
    "check_envelope_samples",
    "coast_samples",
    "held_after",
    "held_for",
    "hold_last",
    "in_tyres_frame",
    "lateral_level",
    "load_track",
    "runs",
    "session_signals",
    "session_tables",
    "speed_on_track",
    "straight_table",
    "throttle_samples",
    "track_position",
]
