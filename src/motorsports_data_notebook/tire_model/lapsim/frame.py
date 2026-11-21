"""The session frame record (lap simulation segment C0, issue #205).

A track and a simulated lap live in the **ground frame**: lateral
acceleration is ``V·ψ̇`` with ``ψ`` the course over ground, longitudinal is
``dV/dt``. The pressure model's heat was fitted on what the sensor reads. A
chassis accelerometer rolls and pitches with the body and sits at whatever
angle it was mounted that day, so it reads the lateral at a slope under one
and the longitudinal with a zero that steps between days.

The record is measured per session, from the session's own GPS and
accelerometer and no car model:

- ``lat_slope`` and ``lat_zero_g``: the accelerometer's lateral against
  the ground-frame lateral;
- ``long_zero_g``: lap-mean accelerometer longitudinal minus lap-mean
  ``dV/dt`` (which is zero over a closed lap, so this is the sensor's zero
  and not the car's);
- the accelerometer's delay behind GPS position, removed before anything
  is compared (otherwise the slope absorbs it);
- how many of the session's 20 s windows each source has.

Both sides are compared at one bandwidth: the ground-frame series are
differences over :data:`GROUND_SMOOTH_S`, the accelerometer a mean over
the same window. A GPS-sourced lap has no record: the logged GPS series
low-passed over that window *is* the ground frame (to 0.1 % in heat).

:func:`check_frame` is gate C0: a lap's ground-frame series, put through
its session's record, must give the heat production computes from the
series as logged.

After the design reviews of D and E (#206, #207) the record is no longer
applied to a simulation, since simulated and observed heat are compared in
the ground frame on both sides. It remains the measurement of what the
sensor frame costs (#255).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ...tire_etl.kinematics import G_MS2, load_kinematics
from ..heat_input import ShareParams
from . import geometry as geo
from . import track as trk
from .gates import GateItem, GateReport, md_table
from .heat import CORNERS, series_heat

logger = logging.getLogger(__name__)

#: Window of the ground-frame derivatives and of the accelerometer's mean.
GROUND_SMOOTH_S = geo.SMOOTH_S
FIT_MIN_SPEED_MS = geo.LAG_MIN_SPEED_MS
FIT_MIN_SAMPLES = 500
IMU_SOURCES = trk.IMU_SOURCES
GPS_SOURCES = ("gps_channel", "gps_position")
#: A (session, source) cell is scored with at least this many laps.
MIN_CELL_LAPS = 3
#: A car is gated with at least this many scored cells.
MIN_GATE_CELLS = 10
WORST_TYRE_P90_MAX = 0.03
TYRE_MEAN_MAX = 0.02


RATIOS = ("rec", "none", "clock", "noise")
RATIO_COLUMNS = [f"{n}_{c}" for n in RATIOS for c in CORNERS]


# ------------------------------------------------------------ the two series


def ground_series(geom: geo.SessionGeometry) -> tuple[np.ndarray, np.ndarray]:
    """``(lat_g, long_g)`` in the ground frame on the position clock.

    Lateral is ``V·Δψ/Δt`` and longitudinal ``ΔV/Δt``, each a central
    difference over :data:`GROUND_SMOOTH_S` of the Doppler course and speed
    (both already on the position clock). Positive lateral is a right
    turn. NaN where the window leaves the sample's segment.
    """
    h = GROUND_SMOOTH_S / 2.0
    t, seg = geom.t, geom.seg

    def _rate(x: np.ndarray) -> np.ndarray:
        up, down = geo.at_offset(t, x, seg, h), geo.at_offset(t, x, seg, -h)
        return np.asarray((up - down) / (2.0 * h))

    with np.errstate(invalid="ignore"):
        lat = geom.speed * _rate(geom.course) / G_MS2
        lng = _rate(geom.speed) / G_MS2
    return np.asarray(lat), np.asarray(lng)


@dataclass
class Inertial:
    """A session's validated kinematics as arrays, time-ordered: the series
    production's heat is computed from, on the chosen sensor's own clock."""

    t: np.ndarray
    speed: np.ndarray  # GPS speed as logged (on the velocity clock)
    lat_g: np.ndarray
    long_g: np.ndarray
    source: np.ndarray  # per row: imu, imu_swapped, gps_channel, gps_position, discard

    @property
    def is_imu(self) -> np.ndarray:
        return np.isin(self.source, IMU_SOURCES)


def load_inertial(root: Path, session_id: str) -> Inertial | None:
    """The session's :func:`~...tire_etl.kinematics.load_kinematics` rows."""
    kin = load_kinematics(root, session_id)
    if kin is None or "lat_source" not in kin.schema.names or len(kin) == 0:
        return None

    def _col(name: str) -> np.ndarray:
        return np.asarray(kin.column(name).to_numpy(zero_copy_only=False), dtype=float)

    t = _col("t_session_s")
    order = np.argsort(t, kind="stable")
    src = np.asarray(kin.column("lat_source").to_numpy(zero_copy_only=False), dtype=object)
    return Inertial(
        t=t[order],
        speed=_col("speed_ms")[order],
        lat_g=_col("lat_g")[order],
        long_g=_col("long_g")[order],
        source=src[order],
    )


# ---------------------------------------------------------------- the record


@dataclass(frozen=True)
class FrameRecord:
    """What one session's accelerometer reads of a ground-frame motion."""

    session_id: str
    lat_slope: float = 1.0
    lat_zero_g: float = 0.0
    long_zero_g: float = 0.0
    #: The accelerometer's delay behind GPS position (s); NaN without one.
    accel_delay_s: float = float("nan")
    #: Slope of the accelerometer's longitudinal on ``dV/dt``; measured,
    #: not applied (pitch under braking; a per-car factor did not help).
    long_slope: float = float("nan")
    lat_resid_g: float = float("nan")  # robust sd of the lateral fit
    long_zero_sd_g: float = float("nan")  # sd of the per-lap zero
    n_samples: int = 0
    n_laps: int = 0
    #: 20 s windows of the session per source (``imu``, ``gps``, ``discard``).
    windows: dict[str, int] = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        """False for a session without enough accelerometer to measure: the
        record is then the identity."""
        return self.n_samples >= FIT_MIN_SAMPLES

    def to_sensor(
        self, lat_ground_g: np.ndarray, long_ground_g: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """A ground-frame series as this session's accelerometer reads it."""
        lat = self.lat_slope * np.asarray(lat_ground_g, float) + self.lat_zero_g
        return lat, np.asarray(long_ground_g, float) + self.long_zero_g

    def to_row(self) -> dict:
        row = asdict(self)
        win = row.pop("windows")
        for key in ("imu", "gps", "discard"):
            row[f"win_{key}"] = int(win.get(key, 0))
        row["fitted"] = self.fitted
        return row


def robust_line(x: np.ndarray, y: np.ndarray, *, rounds: int = 3) -> tuple[float, float, float]:
    """``(slope, zero, robust sd of the residual)`` of ``y`` on ``x`` by
    least squares, refitted after dropping residuals beyond 4 robust sd
    (a kerb strike or a spin is not the frame)."""
    keep = np.isfinite(x) & np.isfinite(y)
    slope, zero, sd = float("nan"), float("nan"), float("nan")
    for _ in range(rounds):
        if keep.sum() < 3 or np.ptp(x[keep]) <= 0:
            break
        slope, zero = (float(c) for c in np.polyfit(x[keep], y[keep], 1))
        resid = y - (slope * x + zero)
        sd = geo._robust_sd(resid[keep])
        if not np.isfinite(sd) or sd <= 0:
            break
        keep = np.isfinite(resid) & (np.abs(resid) <= 4.0 * sd)
    return slope, zero, sd


def _accel_on_position_clock(
    geom: geo.SessionGeometry, inertial: Inertial, delay_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The accelerometer's lateral and longitudinal, each a mean over
    :data:`GROUND_SMOOTH_S`, at the GPS fixes ``delay_s`` later; and whether
    the samples around each fix are the IMU's."""
    half = GROUND_SMOOTH_S / 2.0
    at = geom.t + delay_s
    lat = np.interp(at, inertial.t, geo.nan_window_mean(inertial.lat_g, inertial.t, half))
    lng = np.interp(at, inertial.t, geo.nan_window_mean(inertial.long_g, inertial.t, half))
    imu = np.interp(at, inertial.t, inertial.is_imu.astype(float)) > 0.999
    imu &= (at >= inertial.t[0]) & (at <= inertial.t[-1])
    return lat, lng, imu


def session_frame(
    geom: geo.SessionGeometry,
    inertial: Inertial,
    laps: pd.DataFrame,
    *,
    windows: dict[str, int] | None = None,
) -> FrameRecord:
    """Measure the record of one session on ``laps`` (``t_start_s,
    t_end_s`` of the laps to use: its clean laps).

    The lateral line is fitted over every IMU-sourced fix of those laps at
    :data:`FIT_MIN_SPEED_MS` or more; the longitudinal zero is the median
    over the laps of each lap's mean difference. A session with fewer than
    :data:`FIT_MIN_SAMPLES` such fixes gets the identity.
    """
    windows = dict(windows or {})
    half = GROUND_SMOOTH_S / 2.0
    delay = trk.accel_delay(
        geom,
        inertial.t,
        geo.nan_window_mean(inertial.lat_g, inertial.t, trk.INERTIAL_SMOOTH_S / 2.0),
        inertial.is_imu,
    )
    if not np.isfinite(delay) or inertial.t.size < 2:
        return FrameRecord(session_id=geom.session_id, windows=windows)
    lat_gr, long_gr = ground_series(geom)
    lat_acc, long_acc, imu = _accel_on_position_clock(geom, inertial, delay)
    ok = imu & np.isfinite(lat_gr) & np.isfinite(long_gr) & np.isfinite(lat_acc)
    ok &= np.isfinite(long_acc) & np.isfinite(geom.speed) & (geom.speed >= FIT_MIN_SPEED_MS)
    in_lap = np.zeros(len(geom), dtype=bool)
    zeros: list[float] = []
    for a, b in zip(laps["t_start_s"], laps["t_end_s"]):
        # the window of the derivatives stays inside the lap
        m = ok & (geom.t >= a + half) & (geom.t < b - half)
        in_lap |= m
        if m.sum() >= 0.9 * (b - a) / max(float(np.median(np.diff(geom.t))), 1e-3):
            zeros.append(float(np.mean(long_acc[m] - long_gr[m])))
    n = int(in_lap.sum())
    if n < FIT_MIN_SAMPLES or not zeros:
        return FrameRecord(
            session_id=geom.session_id, accel_delay_s=delay, n_samples=n, windows=windows
        )
    slope, zero, sd = robust_line(lat_gr[in_lap], lat_acc[in_lap])
    long_slope = robust_line(long_gr[in_lap], long_acc[in_lap])[0]
    return FrameRecord(
        session_id=geom.session_id,
        lat_slope=slope,
        lat_zero_g=zero,
        long_zero_g=float(np.median(zeros)),
        accel_delay_s=delay,
        long_slope=long_slope,
        lat_resid_g=sd,
        long_zero_sd_g=float(np.std(zeros)) if len(zeros) > 1 else float("nan"),
        n_samples=n,
        n_laps=len(zeros),
        windows=windows,
    )


# ---------------------------------------------------------------- the replay


def _log_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where((a > 0) & (b > 0), np.log(a / b), np.nan))


def replay_laps(
    geom: geo.SessionGeometry,
    inertial: Inertial,
    record: FrameRecord,
    laps: pd.DataFrame,
    share: ShareParams,
) -> pd.DataFrame:
    """Per lap of ``laps`` (``lap_num, t_start_s, t_end_s``): the lap's
    source and, per tyre, four log ratios of lap-mean heat.

    - ``rec_<tyre>``: the ground-frame series through the record over the
      series as logged. For a GPS-sourced lap the record is the identity
      and the logged series is low-passed over :data:`GROUND_SMOOTH_S`.
    - ``none_<tyre>``: the ground-frame series as it is, over the same.
    - ``clock_<tyre>``: the series as logged on its own clock, with the
      logged speed, over the series as logged on the position clock. What
      production's heat owes to the three clocks not being one.
    - ``noise_<tyre>`` (accelerometer laps): the series as logged over
      itself low-passed over :data:`GROUND_SMOOTH_S`. What the sensor's
      content above the ground frame's bandwidth (vibration, kerbs) adds
      through the positive parts; no slope or zero can carry it.

    The logged series is put on the position clock before the comparison
    (the accelerometer by its delay, a GPS series by the velocity lag), so
    the first two are about the frame alone. ``source`` is ``mixed`` for a
    lap that changes source and ``discard`` for one with a hole; neither is
    scored.
    """
    lat_gr, long_gr = ground_series(geom)
    lat_rec, long_rec = record.to_sensor(lat_gr, long_gr)
    half = GROUND_SMOOTH_S / 2.0
    speed_pos = np.nan_to_num(geom.speed, nan=0.0)
    lat_lp = geo.nan_window_mean(inertial.lat_g, inertial.t, half)
    long_lp = geo.nan_window_mean(inertial.long_g, inertial.t, half)
    rows: list[dict] = []
    # the logged series on the position clock, once per source: a lap is
    # then a slice of the session, not a mask over all of it
    logged: dict[bool, tuple[np.ndarray, np.ndarray]] = {}

    def _lap(t: np.ndarray, a: float, b: float) -> slice:
        """The samples with ``a ≤ t < b`` of a clock in order (all of it
        when the clock is not a number)."""
        if not np.isfinite(t[:1]).all():
            return slice(None)
        j0, j1 = np.searchsorted(t, [a, b])
        return slice(int(j0), int(j1))

    for lap_num, a, b in zip(laps["lap_num"], laps["t_start_s"], laps["t_end_s"]):
        i0, i1 = np.searchsorted(inertial.t, [a, b])
        kinds = set(inertial.source[i0:i1].tolist())
        row: dict = {"lap_num": int(lap_num)}
        if "discard" in kinds or not kinds:
            row["source"] = "discard" if kinds else "none"
            rows.append(row)
            continue
        if len(kinds) > 1:
            row["source"] = "mixed"
            rows.append(row)
            continue
        source = str(next(iter(kinds)))
        row["source"] = source
        imu = source in IMU_SOURCES
        if imu and not record.fitted:
            row["source"] = "unfitted"
            rows.append(row)
            continue
        if imu not in logged:
            shift = record.accel_delay_s if imu else geom.lag.tau_s
            t_shifted = inertial.t - shift
            logged[imu] = (t_shifted, np.interp(t_shifted, geom.t, speed_pos))
        t_log, v_log = logged[imu]
        lat_obs, long_obs = (inertial.lat_g, inertial.long_g) if imu else (lat_lp, long_lp)
        on_log, on_own, on_pos = _lap(t_log, a, b), _lap(inertial.t, a, b), _lap(geom.t, a, b)
        q_obs = series_heat(
            t_log[on_log], v_log[on_log], lat_obs[on_log], long_obs[on_log], share, a, b
        )
        q_own = series_heat(
            inertial.t[on_own], inertial.speed[on_own], lat_obs[on_own], long_obs[on_own],
            share, a, b,
        )  # fmt: skip
        q_none = series_heat(
            geom.t[on_pos], geom.speed[on_pos], lat_gr[on_pos], long_gr[on_pos], share, a, b
        )
        q_rec = (
            series_heat(
                geom.t[on_pos], geom.speed[on_pos], lat_rec[on_pos], long_rec[on_pos], share, a, b
            )
            if imu
            else q_none
        )
        for name, q in (("rec", q_rec), ("none", q_none), ("clock", q_own)):
            for c, v in zip(CORNERS, _log_ratio(q, q_obs)):
                row[f"{name}_{c}"] = float(v)
        if imu:
            q_lp = series_heat(
                t_log[on_log], v_log[on_log], lat_lp[on_log], long_lp[on_log], share, a, b
            )
            for c, v in zip(CORNERS, _log_ratio(q_obs, q_lp)):
                row[f"noise_{c}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows, columns=["lap_num", "source"] + RATIO_COLUMNS)


def session_replay(
    root: Path, session_id: str, layout_id: str, laps: pd.DataFrame, share: ShareParams
) -> tuple[FrameRecord, pd.DataFrame] | None:
    """The record of a dataset session and the replay of ``laps`` (its
    clean laps, from the lap index); None without geometry or kinematics."""
    from ...tire_etl.dataset import timeseries_metadata, timeseries_path
    from ...tire_etl.kinematics import KinematicsVerdict

    geom = geo.load_session_geometry(root, session_id, layout_id)
    inertial = load_inertial(root, session_id)
    if geom is None or inertial is None:
        return None
    windows: dict[str, int] = {}
    path = timeseries_path(root, session_id)
    meta = timeseries_metadata(path) if path is not None else {}
    if meta.get("kinematics"):
        windows = KinematicsVerdict.from_json(meta["kinematics"]).window_counts()
    record = session_frame(geom, inertial, laps, windows=windows)
    return record, replay_laps(geom, inertial, record, laps, share)


# ------------------------------------------------------------------ the gate


def frame_table(root: Path, *, cache_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(records, laps)`` over every session with clean laps: one row per
    session (the record with the session's car, date, layout, logger and
    ``has_tpms``) and one per clean lap (:func:`replay_laps` with the same
    keys)."""
    from .holdout import production_shares, share_for

    index = geo.clean_lap_index(root, cache_dir=cache_dir)
    sessions = geo.load_sessions(root).set_index("session_id")
    shares = production_shares(root)
    clean = index[index["clean"]]
    records: list[dict] = []
    laps: list[pd.DataFrame] = []
    for sid, grp in clean.groupby("session_id", sort=True):
        first = grp.iloc[0]
        try:
            share = share_for(str(first["car"]), shares)
        except KeyError:
            logger.warning("frame: no force shares for %s (%s)", first["car"], sid)
            continue
        try:
            got = session_replay(root, str(sid), str(first["layout_id"]), grp, share)
        except Exception:  # noqa: BLE001 - one unreadable session must not stop the gate
            logger.warning("frame: failed for %s", sid, exc_info=True)
            continue
        if got is None:
            continue
        record, replay = got
        keys: dict[str, str | bool] = {
            "session_id": str(sid),
            "car": str(first["car"]),
            "date": str(first["date"]),
            "layout_id": str(first["layout_id"]),
            "logger_id": str(first["logger_id"]),
            "has_tpms": bool(sessions.at[sid, "has_tpms"]) if sid in sessions.index else False,
        }
        records.append({**keys, **record.to_row()})
        for i, (key, val) in enumerate(keys.items()):
            replay.insert(i, key, val)
        laps.append(replay)
    return pd.DataFrame(records), (pd.concat(laps, ignore_index=True) if laps else pd.DataFrame())


def cell_scores(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per (session, source) with :data:`MIN_CELL_LAPS` scored
    laps: the mean over its laps of each log ratio per tyre, and
    ``worst_rec`` / ``worst_none``, the largest of the four in magnitude."""
    names = RATIO_COLUMNS
    scored = laps.dropna(subset=[f"rec_{c}" for c in CORNERS])
    keys = ["session_id", "car", "date", "layout_id", "logger_id", "has_tpms", "source"]
    if scored.empty:
        return pd.DataFrame(columns=keys + ["laps"] + names + ["worst_rec", "worst_none"])
    out = scored.groupby(keys, as_index=False)[names].mean()
    out.insert(len(keys), "laps", scored.groupby(keys).size().to_numpy())
    out = out[out["laps"] >= MIN_CELL_LAPS].reset_index(drop=True)
    for name in ("rec", "none"):
        out[f"worst_{name}"] = out[[f"{name}_{c}" for c in CORNERS]].abs().max(axis=1)
    return out


def _p90(s: pd.Series) -> float:
    return float(s.quantile(0.9)) if len(s) else float("nan")


def check_frame(root: Path, *, cache_dir: Path | None = None) -> GateReport:
    """Gate C0: the session frame record carries a ground-frame lap to the
    heat production computes from the series as logged."""
    records, laps = frame_table(root, cache_dir=cache_dir)
    rep = GateReport("C0", "the session frame record (`lapsim check-frame`)")
    cells = cell_scores(laps)
    tp = cells[cells["has_tpms"]]
    n_lap = laps["source"].value_counts().to_dict() if len(laps) else {}
    rep.notes.append(
        f"{len(records)} sessions with clean laps, {int(records['fitted'].sum())} with a fitted "
        f"record; {len(laps)} clean laps by source: "
        + ", ".join(f"{k} {v}" for k, v in sorted(n_lap.items()))
        + f". Scored: {len(cells)} (session, source) cells of {MIN_CELL_LAPS}+ laps, "
        f"{len(tp)} of them in sessions with TPMS ({int(tp['laps'].sum())} laps)."
    )
    rep.notes.append(
        "Heat is production's chain (`heat.lap_heat`: the share model on 1 s means of the four "
        "positive parts), not `pace_sim.corner_heat_means` as the issue first had it (#206 "
        "finding 11). The record is measured on the laps it is scored on: it is a measurement "
        "of the session, not a model."
    )

    # 1 and 2: per car, over its sessions with TPMS.
    by_car = (
        tp.groupby("car")
        .agg(
            cells=("laps", "size"),
            laps=("laps", "sum"),
            worst_p50=("worst_rec", "median"),
            worst_p90=("worst_rec", _p90),
            worst_max=("worst_rec", "max"),
            none_p50=("worst_none", "median"),
            none_p90=("worst_none", _p90),
            none_max=("worst_none", "max"),
            **{f"mean_{c}": (f"rec_{c}", "mean") for c in CORNERS},
        )
        .reset_index()
    )
    gated = by_car[by_car["cells"] >= MIN_GATE_CELLS]
    listed = by_car[by_car["cells"] < MIN_GATE_CELLS]
    p90_ok = bool(len(gated)) and bool((gated["worst_p90"] <= WORST_TYRE_P90_MAX).all())
    means = gated[[f"mean_{c}" for c in CORNERS]].abs().to_numpy()
    mean_ok = bool(len(gated)) and bool((means <= TYRE_MEAN_MAX).all())
    by_cell = (
        tp.groupby(["car", "layout_id", "source"])
        .agg(
            cells=("laps", "size"),
            laps=("laps", "sum"),
            worst_p50=("worst_rec", "median"),
            worst_p90=("worst_rec", _p90),
            none_p90=("worst_none", _p90),
            **{f"mean_{c}": (f"rec_{c}", "mean") for c in CORNERS},
        )
        .reset_index()
    )
    over = tp[tp["worst_rec"] > WORST_TYRE_P90_MAX].sort_values("worst_rec", ascending=False)
    rep.items.append(
        GateItem(
            "Worst tyre per session through the record",
            p90_ok if len(gated) else None,
            "; ".join(
                f"{r.car} {r.worst_p90:.3f} (no record {r.none_p90:.3f})"
                for r in gated.itertuples()
            )
            or "no car with enough sessions",
            f"P90 over a car's sessions with TPMS ≤ {WORST_TYRE_P90_MAX}",
            "By car (`worst_*`: through the record; `none_*`: the ground frame as it is):\n\n"
            + md_table(by_car, floats="{:.3f}")
            + (
                f"\n\nCars with fewer than {MIN_GATE_CELLS} cells are listed, not gated: "
                + ", ".join(listed["car"])
                + "."
                if len(listed)
                else ""
            )
            + "\n\nBy car, layout and source:\n\n"
            + md_table(by_cell, floats="{:.3f}")
            + f"\n\nCells over {WORST_TYRE_P90_MAX}:\n\n"
            + md_table(
                over[
                    ["session_id", "source", "laps", "worst_rec", "worst_none"]
                    + [f"rec_{c}" for c in CORNERS]
                ],
                floats="{:.3f}",
            ),
        )
    )
    rep.items.append(
        GateItem(
            "Per-tyre mean through the record",
            mean_ok if len(gated) else None,
            "; ".join(
                f"{r.car} " + " / ".join(f"{getattr(r, 'mean_' + c):+.3f}" for c in CORNERS)
                for r in gated.itertuples()
            )
            or "no car with enough sessions",
            f"|mean| ≤ {TYRE_MEAN_MAX} per tyre (fl / fr / rl / rr) and car",
        )
    )

    # Reported: sessions without TPMS, the clocks, the record's parameters.
    rest = cells[~cells["has_tpms"]]
    if len(rest):
        other = (
            rest.groupby(["car", "source"])
            .agg(
                cells=("laps", "size"),
                laps=("laps", "sum"),
                worst_p50=("worst_rec", "median"),
                worst_p90=("worst_rec", _p90),
                none_p90=("worst_none", _p90),
            )
            .reset_index()
        )
        rep.items.append(
            GateItem(
                "Sessions without TPMS",
                None,
                f"{len(rest)} cells; worst tyre P90 {_p90(rest['worst_rec']):.3f}",
                "",
                md_table(other, floats="{:.3f}"),
            )
        )
    clock = (
        cells.groupby(["car", "source"])
        .agg(cells=("laps", "size"), **{f"clock_{c}": (f"clock_{c}", "mean") for c in CORNERS})
        .reset_index()
    )
    span = cells[[f"clock_{c}" for c in CORNERS]].to_numpy()
    rep.items.append(
        GateItem(
            "What the clocks cost",
            None,
            (
                f"logged series on its own clock over the same on the position clock: "
                f"{np.nanmin(span):+.3f} to {np.nanmax(span):+.3f} per cell and tyre"
                if span.size
                else "none"
            ),
            "",
            "Mean log ratio per car and source (the accelerometer trails GPS position by its "
            "delay and the logged speed by the velocity lag):\n\n"
            + md_table(clock, floats="{:+.4f}"),
        )
    )
    acc = cells[cells["source"].isin(IMU_SOURCES)]
    noise = (
        acc.groupby(["car", "logger_id"])
        .agg(
            cells=("laps", "size"),
            first=("date", "min"),
            last=("date", "max"),
            **{f"noise_{c}": (f"noise_{c}", "mean") for c in CORNERS},
            **{f"rec_{c}": (f"rec_{c}", "mean") for c in CORNERS},
        )
        .reset_index()
    )
    worst_noise = noise[[f"noise_{c}" for c in CORNERS]].max(axis=1)
    rep.items.append(
        GateItem(
            "What the accelerometer's noise adds",
            None,
            "; ".join(
                f"{r.car} on {r.logger_id} {w:+.3f}"
                for r, w in zip(noise.itertuples(), worst_noise)
            ),
            "",
            f"Heat from the accelerometer as logged over the same series low-passed over "
            f"{GROUND_SMOOTH_S} s (`noise_*`, the largest tyre in the summary), beside what "
            "is left through the record (`rec_*`). Where the noise share is large the record "
            "reads low by about as much on all four tyres: a slope and a zero cannot carry "
            "vibration.\n\n" + md_table(noise, floats="{:+.3f}"),
        )
    )
    fit = records[records["fitted"]]
    slope = (
        fit.groupby(["car", "layout_id"])
        .agg(
            sessions=("lat_slope", "size"),
            slope_p10=("lat_slope", lambda s: s.quantile(0.1)),
            slope_median=("lat_slope", "median"),
            slope_p90=("lat_slope", lambda s: s.quantile(0.9)),
            lat_zero_median=("lat_zero_g", "median"),
            long_slope_median=("long_slope", "median"),
        )
        .reset_index()
    )
    rep.items.append(
        GateItem(
            "Lateral slope per layout",
            None,
            "; ".join(
                f"{car} {g['slope_median'].min():.3f}–{g['slope_median'].max():.3f}"
                for car, g in slope.groupby("car")
            ),
            "",
            md_table(slope, floats="{:.3f}"),
        )
    )
    day = (
        fit.groupby(["car", "date"])
        .agg(sessions=("long_zero_g", "size"), long_zero_g=("long_zero_g", "median"))
        .reset_index()
    )
    day_span = (
        day.groupby("car")
        .agg(
            days=("long_zero_g", "size"),
            zero_min=("long_zero_g", "min"),
            zero_median=("long_zero_g", "median"),
            zero_max=("long_zero_g", "max"),
            day_sd=("long_zero_g", "std"),
        )
        .reset_index()
    )
    within = fit.groupby(["car", "date"])["long_zero_g"].std().groupby("car").median()
    day_span["within_day_sd"] = day_span["car"].map(within)
    rep.items.append(
        GateItem(
            "Longitudinal zero per day",
            None,
            "; ".join(
                f"{r.car} {r.zero_min:+.3f} to {r.zero_max:+.3f} g" for r in day_span.itertuples()
            ),
            "",
            md_table(day_span, floats="{:+.4f}")
            + "\n\nPer day:\n\n"
            + md_table(day, floats="{:+.4f}"),
        )
    )
    delay = (
        fit.groupby(["logger_id", "car"])
        .agg(
            sessions=("accel_delay_s", "size"),
            delay_ms=("accel_delay_s", lambda s: 1e3 * s.median()),
            lat_resid_g=("lat_resid_g", "median"),
        )
        .reset_index()
    )
    rep.items.append(
        GateItem(
            "Accelerometer delay per logger",
            None,
            f"{delay['delay_ms'].min():.0f}–{delay['delay_ms'].max():.0f} ms",
            "",
            md_table(delay, floats="{:.3g}"),
        )
    )
    return rep


__all__ = [
    "FrameRecord",
    "Inertial",
    "cell_scores",
    "check_frame",
    "frame_table",
    "ground_series",
    "load_inertial",
    "replay_laps",
    "robust_line",
    "session_frame",
    "session_replay",
]
