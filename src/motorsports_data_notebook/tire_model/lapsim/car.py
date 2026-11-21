"""The car model (lap simulation segment C2, issue #205).

A car is a chassis with one or more **powertrains**. The chassis is what
its raw cars share: grip, braking, drag, the cost of cornering at full
throttle. A powertrain is an engine with its gearing; a car gets a second
one when its engine changes (the KK-F-V2 is the KK-F with a new intake;
the Inferno 86 had its engine replaced, and its final drive with it, in
mid-2026). :data:`FAMILY` and :data:`POWERTRAINS` say which raw car and
date is which.

How hard the gears are held is the **driver's**, not the car's: the
``shift_rpm`` of a day is read from that day's logs
(:func:`shift_rpm_by_day`) and handed to the solver beside the pace knob.
A new shift light is then a different way of driving one car.

Every number is read from the car's sample tables (:mod:`.samples`), not
fitted through a solver; the solver is the gate.

- ``power.c_drag`` (1/m, ``a_drag = c·V²``, at :data:`RHO_REF`): where the
  logs have engine rpm, from the same rpm in different gears
  (:func:`fit_drag_rpm`): one rpm is one power, so ``a·V`` falls with
  ``V³`` at ``−c``. Without rpm, from the cells whose straight table
  reaches :data:`DRAG_REACH_MS`, the slope of ``a·V`` on ``V³`` over the
  bins from :data:`DRAG_FIT_MIN_MS` up (:func:`fit_drag`), which takes
  the power for one number there. Held within :data:`DRAG_PRIOR_BAND` of
  a published value where the car has one.
- **Gears.** Where the logs have no gear channel but an engine rpm, the
  gear is made from speed over rpm (:func:`with_gears`): in one gear that
  ratio is one number. The gears a car ran are read per day
  (:func:`gear_sets_by_day`), because a formula car's gear set is changed
  between circuits, and completed from the other days that ran the same
  set (:func:`complete_sets`: a day shows only the gears it used at full
  throttle). Like the shift point, the day's set goes to the solver; a
  powertrain's own ``gears``, the set most of its days ran, are what it
  is driven on when a day has none.
- ``power.engines[name]``, per powertrain, in **standard air** (SAE J1349:
  99 kPa of dry air at 25 °C): ``curve``, the power at the wheels per mass
  against engine rpm, the median of ``(a + c·V²)·V`` over full-throttle
  straight-line samples in bins of :data:`RPM_BIN`; ``gears``, the speed
  per 1,000 rpm of each gear; ``p_specific``, one number for the solver
  of a car without an rpm or a gear channel (the median over layouts and
  speed bins); ``shift_rpm``, the median over its days; and ``scale``,
  its power over the car's first powertrain's at equal rpm. A powertrain
  with too few samples for a curve of its own is that curve at its scale:
  the chassis and the shape of the engine are shared, the level is not.
- ``power.k_corner_drag`` (1/g): the median of ``(a_straight − a) / lat²``
  over full-throttle samples above :data:`CORNER_DRAG_MIN_LAT_G`.
- ``grip.mu_b``: the 90th percentile over the zones braked in a straight
  line of the zone's peak, drag and grade removed. ``grip.n_brake`` is 1
  (#206's field).
- ``grip.brake_ellipse``: how far the braking limit under cornering is
  from the diamond towards the ellipse (:func:`braking_shape`).
- ``grip.onset_s``: the time between drive and full braking where a stop
  begins (:func:`brake_onset`); the driver's on this car, not the tyre's.
- ``grip.mu_y``: a convention, since the pace knob absorbs the level
  (:func:`.car_gate.set_convention`, which also fills ``ref_u``);
  :func:`fit_car` alone leaves the apex table's level there.
- ``wet.grip_scale``: the wet population's lateral level over the dry one.

**Air.** An engine makes less power in thin, hot or humid air and the car
meets less drag in thin air. :func:`air_state` gives each session the
density of its moist air and the SAE correction factor from the hourly
weather; a sample's power is corrected to standard air before anything is
pooled, and the solver is handed the day's air back.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ...tire_etl.kinematics import G_MS2
from ...tire_etl.paths import weather_dir
from . import geometry as geo
from . import samples as smp
from .gates import GateItem, GateReport, md_table

logger = logging.getLogger(__name__)

#: What is published about each car. ``mass_kg`` is with the driver: the
#: class minimum for the formula cars (JAF regulations, sources on #205);
#: for the Inferno 86 the mass behind the design review's drag prior
#: (Cd 0.27, about 2.0 m² and 2.4e-4 /m), to be replaced by the car's
#: weighed mass. ``crank_kw`` is the engine's published power.
PUBLISHED: dict[str, dict] = {
    "Inferno 86": {"mass_kg": 1350.0, "crank_kw": 147.0, "c_drag_prior": 2.4e-4},
    "KK-SII": {"mass_kg": 495.0, "crank_kw": 88.0, "c_drag_prior": None},
    "KK-F": {"mass_kg": 525.0, "crank_kw": None, "c_drag_prior": None},
}
#: Raw cars that are another car's chassis with a different powertrain.
FAMILY: dict[str, str] = {"KK-F-V2": "KK-F"}
#: A car's powertrains in the order they came: each is the one of a raw
#: car (``raw_car``) or of a span of dates (``from`` / ``until``,
#: inclusive). A car not listed has one, :data:`DEFAULT_ENGINE`.
#: The Inferno 86's engine was replaced between its sessions of 2026-04-06
#: and 2026-07-15 (the owner, 2026-11-15); the logs show the final drive
#: going from 4.3 to 4.1 at the same change.
POWERTRAINS: dict[str, list[dict]] = {
    "Inferno 86": [
        {"name": "first engine", "until": "2026-05-31"},
        {"name": "second engine", "from": "2026-06-01"},
    ],
    "KK-F": [
        {"name": "KK-F", "raw_car": "KK-F"},
        {"name": "KK-F-V2", "raw_car": "KK-F-V2"},
    ],
}
DEFAULT_ENGINE = "engine"

DRAG_REACH_MS = 50.0
DRAG_FIT_MIN_MS = 40.0
DRAG_FIT_MIN_BINS = 3
DRAG_PRIOR_BAND = (0.7, 1.3)
#: An rpm bin of a day says something about drag when its samples span
#: this much speed: the same rpm in two gears.
DRAG_RPM_SPREAD_MS = 6.0
CORNER_DRAG_MIN_LAT_G = 0.5
#: A day's level is read from at least this many samples in a bin.
DAY_BIN_MIN_SAMPLES = 30
DAY_MIN_BINS = 2
R_DRY_AIR = 287.05
R_VAPOUR = 461.5
#: Standard air (SAE J1349): dry-air pressure (hPa), temperature (°C) and
#: the density that goes with them.
STD_DRY_HPA = 990.0
STD_T_C = 25.0
RHO_REF = STD_DRY_HPA * 100.0 / (R_DRY_AIR * (STD_T_C + 273.15))
RPM_BIN = 250.0
#: Clustering speed per 1,000 rpm into gears (m/s per 1,000 rpm).
RATIO_BIN = 0.05
GEAR_TOL = 0.03
CLUSTER_MIN_SHARE = 0.02
CLUSTER_MIN_SAMPLES = 30
CURVE_MIN_SAMPLES = 150
CURVE_MIN_BINS = 5
GEAR_MIN_SAMPLES = 200
#: A day's shift point is this quantile of its full-throttle rpm.
SHIFT_QUANTILE = 0.99
SHIFT_MIN_SAMPLES = 300
POWER_BETWEEN_LAYOUTS_MAX = 0.07
POWER_OVER_CRANK_BAND = (0.55, 0.80)
MIN_VOTE_DAYS = smp.MIN_VOTE_DAYS
#: An upshift is two full-throttle samples of one lap, a gear apart, this
#: far apart in time (the table leaves out what is near a gear change).
SHIFT_GAP_S = (0.06, 2.5)
SHIFT_MIN_EVENTS = 20
#: The engine's power either side of an upshift: the median over this
#: many samples of the gear.
SHIFT_EDGE_SAMPLES = 6
#: The same test inside a gear, over this long, reads the method's bias.
SHIFT_CONTROL_S = 1.0
SHIFT_CONTROL_EVERY = 10
SHIFT_STEPS = 64
#: The braking limit's shape is read from the zones whose peak comes with
#: this much of the car's lateral level, in bands this wide.
#: A zone is braked in a straight line when its peak comes with less than
#: this share of the car's lateral level.
#: Where a stop begins is read from the stops from full throttle at this
#: speed or more.
ONSET_MIN_SPEED_MS = 30.0
ONSET_MIN_ZONES = 20
BRAKE_STRAIGHT_SHARE = 0.2
BRAKE_STRAIGHT_MIN_ZONES = 30
BRAKE_BANDS = (0.3, 0.9)
BRAKE_BAND_WIDTH = 0.1
BRAKE_BAND_MIN_ZONES = 15


def family_of(raw_car: str) -> str:
    """The car a raw car is a powertrain of (itself, for most)."""
    return FAMILY.get(raw_car, raw_car)


def powertrain_of(raw_car: str, date: str) -> str:
    """The name of the powertrain a raw car had on ``date`` (ISO)."""
    for spec in POWERTRAINS.get(family_of(raw_car), []):
        if "raw_car" in spec:
            if spec["raw_car"] == raw_car:
                return str(spec["name"])
        elif spec.get("from", "") <= str(date) <= spec.get("until", "9999"):
            return str(spec["name"])
    return DEFAULT_ENGINE


#: The mark :func:`as_family` leaves on a table (pandas carries ``attrs``
#: over to a selection of its rows), so that it is not labelled twice.
_FAMILIES = "lapsim_families"
#: What the fit reads of a full-throttle sample.
_FIT_COLS = (
    "car", "raw_car", "engine", "date", "layout_id", "session_id", "lap_num", "t_s",
    "v_ms", "a_g", "lat_g", "grade", "rpm", "gear", "straight",
)  # fmt: skip


def as_family(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` (rows with ``car`` and ``date``) with the raw car moved to
    ``raw_car``, ``car`` the family and ``engine`` the powertrain."""
    if frame.empty or "car" not in frame.columns or frame.attrs.get(_FAMILIES):
        return frame  # nothing to label, or labelled already (the rows of a labelled table)
    out = frame.copy()
    if "raw_car" not in out.columns:
        out["raw_car"] = out["car"]
    # one call per raw car and per (raw car, day), not per sample: the
    # throttle table has millions of rows and a dozen cars
    car_code, cars = pd.factorize(out["raw_car"], use_na_sentinel=False)
    out["car"] = pd.array([family_of(c) for c in cars], dtype="str").take(car_code)
    if "date" in out.columns:
        day_code, days = pd.factorize(out["date"].astype(str), use_na_sentinel=False)
        names = pd.array([powertrain_of(c, d) for c in cars for d in days], dtype="str")
        out["engine"] = names.take(car_code * len(days) + day_code)
    out.attrs[_FAMILIES] = True
    return out


def as_families(tables: smp.CarTables) -> smp.CarTables:
    """``tables`` with every table's cars as families (:func:`as_family`)."""
    return smp.CarTables(
        sessions=as_family(tables.sessions),
        throttle=as_family(tables.throttle),
        coast=as_family(tables.coast),
        brake=as_family(tables.brake),
        apex=as_family(tables.apex),
        speed_laps=as_family(tables.speed_laps),
        speed=tables.speed,
    )


@dataclass
class CarModel:
    """A car as the solver takes it (the issue's car JSON)."""

    car: str
    raw_car: str
    mass_kg: float | None  # with the driver; None when nothing is published
    grip: dict = field(default_factory=dict)  # mu_y, mu_b, n_brake
    #: c_drag, rho_ref, k_corner_drag, engines{name: p_specific, curve,
    #: gears, shift_rpm}, default_engine, and p_specific / a_ft_by_v of the
    #: default engine.
    power: dict = field(default_factory=dict)
    wet: dict = field(default_factory=dict)  # grip_scale
    support: dict = field(default_factory=dict)  # table -> {v_min, v_max}
    ref_u: dict = field(default_factory=dict)  # layout -> [median, day_sd]
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "car": self.car,
            "raw_car": self.raw_car,
            "mass_kg": self.mass_kg,
            "grip": self.grip,
            "power": self.power,
            "wet": self.wet,
            "support": self.support,
            "ref_u": self.ref_u,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, data: dict) -> "CarModel":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))
        return path

    @classmethod
    def load(cls, path: Path) -> "CarModel":
        return cls.from_json(json.loads(Path(path).read_text()))


def _round(v: float | None, nd: int) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


# ---------------------------------------------------------------------- air


def air_state(root: Path, sessions: pd.DataFrame) -> pd.DataFrame:
    """Per ``session_id``: the air of the hour the session started in,
    from the hourly weather of its track.

    ``rho`` is the density of the moist air (kg/m³), ``cf`` the SAE J1349
    factor by which a naturally aspirated engine's observed power is
    multiplied to give its power in standard air
    (``1.18 · (99 / p_dry) · √((T + 273) / 298) − 0.18``, ``p_dry`` the
    pressure less the water vapour's), with ``t_c`` and ``rh`` beside
    them. A session without weather within the hour is left out.
    """
    wx_root = weather_dir(Path(root))
    by_track: dict[str, pd.DataFrame] = {}
    need = {"ts_utc", "temperature_2m", "surface_pressure"}
    rows: list[dict] = []
    starts = pd.to_datetime(sessions["session_start_utc"], utc=True)
    for sid, track, start in zip(sessions["session_id"], sessions["track_canonical"], starts):
        key = str(track)
        if key not in by_track:
            files = sorted((wx_root / key).glob("*.parquet")) if (wx_root / key).is_dir() else []
            frames = [pq.read_table(f).to_pandas() for f in files]
            wx = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if len(wx) and need <= set(wx.columns):
                wx["ts"] = pd.to_datetime(wx["ts_utc"], utc=True)
                wx = wx.dropna(subset=["temperature_2m", "surface_pressure"]).sort_values("ts")
            else:
                wx = pd.DataFrame()
            by_track[key] = wx
        wx = by_track[key]
        if not len(wx) or pd.isna(start):
            continue
        i = int(np.argmin(np.abs((wx["ts"] - start).dt.total_seconds().to_numpy())))
        if abs((wx["ts"].iloc[i] - start).total_seconds()) > 3600.0:
            continue
        t_c = float(wx["temperature_2m"].iloc[i])
        rh = float(wx["relative_humidity_2m"].iloc[i]) if "relative_humidity_2m" in wx else 0.0
        rows.append(
            {"session_id": str(sid), **moist_air(t_c, float(wx["surface_pressure"].iloc[i]), rh)}
        )
    out = pd.DataFrame(rows, columns=["session_id", "rho", "cf", "t_c", "rh"])
    return out.set_index("session_id")


def moist_air(t_c: float, pressure_hpa: float, rh_pct: float = 0.0) -> dict:
    """``rho`` and ``cf`` (see :func:`air_state`) of air at ``t_c`` °C,
    ``pressure_hpa`` and ``rh_pct`` relative humidity."""
    rh = 0.0 if not np.isfinite(rh_pct) else rh_pct
    vapour = rh / 100.0 * 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))  # Magnus, hPa
    dry = pressure_hpa - vapour
    kelvin = t_c + 273.15
    rho = dry * 100.0 / (R_DRY_AIR * kelvin) + vapour * 100.0 / (R_VAPOUR * kelvin)
    cf = 1.18 * (STD_DRY_HPA / dry) * np.sqrt(kelvin / (STD_T_C + 273.15)) - 0.18
    return {"rho": float(rho), "cf": float(cf), "t_c": float(t_c), "rh": float(rh)}


def with_air(samples: pd.DataFrame, air: pd.DataFrame | None) -> pd.DataFrame:
    """``samples`` with each row's ``rho`` and ``cf`` from its session's
    air; standard air where there is none."""
    out = samples.copy()
    if air is not None and len(air) and "session_id" in out.columns:
        code, sessions = pd.factorize(out["session_id"], use_na_sentinel=False)
        for col in ("rho", "cf"):
            out[col] = air[col].reindex(sessions).to_numpy(dtype=float)[code]
    for col, std in (("rho", RHO_REF), ("cf", 1.0)):
        if col not in out.columns:
            out[col] = std
        out[col] = out[col].fillna(std)
    return out


# -------------------------------------------------------------------- power


def solid_bins(throttle: pd.DataFrame, by: tuple[str, ...] = ("layout_id",)) -> pd.DataFrame:
    """The straight table of a set of dry samples: per ``by`` and
    :data:`~.samples.SPEED_BIN_MS` bin the median ``a_g`` and ``v_ms``, the
    samples, sessions and days, and the mean ``rho`` and ``cf`` where the
    samples carry them; bins with :data:`~.samples.BIN_MIN_SAMPLES` only."""
    cols = list(by) + ["v_bin_ms", "a_g", "v_ms", "n", "sessions", "days", "rho", "cf"]
    if throttle.empty:
        return pd.DataFrame(columns=cols)
    st = _only(
        throttle, throttle["straight"], *by, "v_ms", "a_g", "session_id", "date", "rho", "cf"
    )
    st["v_bin_ms"] = np.floor(st["v_ms"] / smp.SPEED_BIN_MS) * smp.SPEED_BIN_MS
    for col, std in (("rho", RHO_REF), ("cf", 1.0)):
        if col not in st.columns:
            st[col] = std
    out = (
        st.groupby(list(by) + ["v_bin_ms"])
        .agg(
            a_g=("a_g", "median"),
            v_ms=("v_ms", "median"),
            n=("a_g", "size"),
            sessions=("session_id", "nunique"),
            days=("date", "nunique"),
            rho=("rho", "mean"),
            cf=("cf", "mean"),
        )
        .reset_index()
    )
    return out[out["n"] >= smp.BIN_MIN_SAMPLES].reset_index(drop=True)[cols]


def fit_drag(bins: pd.DataFrame, prior: float | None = None) -> dict:
    """Drag from the top of the straight tables, at :data:`RHO_REF`.

    With the engine at a power ``P``, ``a·V = P − c·V³``: over the bins
    from :data:`DRAG_FIT_MIN_MS` up of a layout that reaches
    :data:`DRAG_REACH_MS`, the slope of ``a·V`` on ``V³`` is ``−c`` in the
    air of those bins, and ``c`` goes with the density. ``by_layout``
    holds each such layout's ``c_drag``, the top-end power ``p_top``
    (W/kg, as observed) and the terminal speed the two imply; ``fitted``
    is their median, ``c_drag`` that value held within
    :data:`DRAG_PRIOR_BAND` of ``prior`` (the prior itself when no layout
    reaches; NaN with neither) and ``source`` says which.
    """
    by_layout: dict[str, dict] = {}
    for layout, grp in bins.groupby("layout_id"):
        top = grp[grp["v_bin_ms"] >= DRAG_FIT_MIN_MS]
        if len(top) < DRAG_FIT_MIN_BINS or top["v_bin_ms"].max() < DRAG_REACH_MS:
            continue
        v = top["v_ms"].to_numpy(float)
        slope, p_top = np.polyfit(v**3, top["a_g"].to_numpy(float) * G_MS2 * v, 1)
        there = -float(slope)
        rho = float(top["rho"].mean()) if "rho" in top.columns else RHO_REF
        by_layout[str(layout)] = {
            "c_drag": there * RHO_REF / rho,
            "p_top": float(p_top),
            "v_terminal_ms": (
                float((p_top / there) ** (1.0 / 3.0)) if there > 0 and p_top > 0 else np.nan
            ),
            "bins": int(len(top)),
            "rho": rho,
        }
    fitted = float(np.median([d["c_drag"] for d in by_layout.values()])) if by_layout else np.nan
    if np.isfinite(fitted) and prior:
        c_drag = float(np.clip(fitted, DRAG_PRIOR_BAND[0] * prior, DRAG_PRIOR_BAND[1] * prior))
        source = "fitted" if c_drag == fitted else "fitted, held at the edge of the prior's band"
    elif np.isfinite(fitted):
        c_drag, source = fitted, "fitted (no prior)"
    elif prior:
        c_drag, source = float(prior), "prior (no layout reaches 50 m/s)"
    else:
        c_drag, source = float("nan"), "none"
    return {
        "c_drag": c_drag,
        "fitted": fitted,
        "prior": prior,
        "source": source,
        "by_layout": by_layout,
    }


def fit_drag_rpm(throttle: pd.DataFrame) -> dict:
    """Drag from the same rpm in different gears, at :data:`RHO_REF`.

    At one rpm the engine makes one power, whatever gear it is in, so
    ``(a + c·V²)·V`` must agree between the gears: within a day, a
    powertrain and an rpm bin, ``a·V`` falls with ``V³`` at the slope
    ``−c``. That needs no assumption about the shape of the power curve,
    which the fit at the top of the straight table does (:func:`fit_drag`
    takes the power for one number over its bins; on simulated laps of an
    engine whose power falls off at high rpm it read 42 % high).

    Only cells (``engine``, ``date``, rpm bin) holding at least
    :data:`DAY_BIN_MIN_SAMPLES` samples that span :data:`DRAG_RPM_SPREAD_MS`
    of speed take part: two gears at the least. One refit after dropping
    residuals beyond four robust sd. ``c_drag`` is NaN without rpm or
    without such cells; ``cells`` and ``n`` say how much is behind it.
    """
    out = {"c_drag": float("nan"), "cells": 0, "n": 0}
    st = _with_rpm(throttle, "engine", "date", *_POWER_COLS).copy()
    if st.empty:
        return out
    for col, std in (("rho", RHO_REF), ("cf", 1.0), ("engine", DEFAULT_ENGINE)):
        if col not in st.columns:
            st[col] = std
    v = st["v_ms"].to_numpy(float)
    cf = st["cf"].to_numpy(float)
    st["y"] = st["a_g"].to_numpy(float) * G_MS2 * v * cf
    st["x"] = st["rho"].to_numpy(float) / RHO_REF * v**3 * cf
    st["bin"] = np.floor(st["rpm"] / RPM_BIN)
    grp = st.groupby(["engine", "date", "bin"])
    size = grp["v_ms"].transform("size")
    edges = grp["v_ms"].quantile(np.array([0.05, 0.95])).unstack()
    spread = (edges[0.95] - edges[0.05]).to_numpy(float)[grp.ngroup().to_numpy()]
    st = st[(size >= DAY_BIN_MIN_SAMPLES) & (spread >= DRAG_RPM_SPREAD_MS)]
    if st.empty:
        return out
    grp = st.groupby(["engine", "date", "bin"])
    x = (st["x"] - grp["x"].transform("mean")).to_numpy(float)
    y = (st["y"] - grp["y"].transform("mean")).to_numpy(float)
    keep = np.ones(len(x), dtype=bool)
    c = float("nan")
    for _ in range(2):
        den = float(np.sum(x[keep] ** 2))
        if den <= 0:
            return out
        c = -float(np.sum(x[keep] * y[keep]) / den)
        resid = y + c * x
        sd = 1.4826 * float(np.median(np.abs(resid[keep] - np.median(resid[keep]))))
        if sd <= 0:
            break
        keep = np.abs(resid) <= 4.0 * sd
    return {"c_drag": c, "cells": int(grp.ngroups), "n": int(keep.sum())}


def standard_power(frame: pd.DataFrame, c_drag: float) -> np.ndarray:
    """Power at the wheels per mass in standard air (W/kg) of each row of
    ``frame`` (``a_g``, ``v_ms`` and, where it has them, ``rho`` and
    ``cf``): ``(a + c·(rho/rho_ref)·V²)·V·cf``."""
    v = frame["v_ms"].to_numpy(float)
    rho = frame["rho"].to_numpy(float) if "rho" in frame.columns else np.full(len(v), RHO_REF)
    cf = frame["cf"].to_numpy(float) if "cf" in frame.columns else np.ones(len(v))
    c = c_drag if np.isfinite(c_drag) else 0.0
    out = (frame["a_g"].to_numpy(float) * G_MS2 + c * (rho / RHO_REF) * v * v) * v * cf
    return np.asarray(out)


def bin_power(bins: pd.DataFrame, c_drag: float) -> pd.DataFrame:
    """``bins`` with ``p_wkg``, each bin's :func:`standard_power`."""
    out = bins.copy()
    out["p_wkg"] = standard_power(out, c_drag)
    return out


def specific_power(bins: pd.DataFrame, c_drag: float) -> dict:
    """``p_specific``: the median over layouts and bins of the bins' power;
    ``by_layout`` each layout's own median with its bins and days, and
    ``matched`` the median over a layout's bins of its power over the
    pooled power of the same speed bin."""
    p = bin_power(bins, c_drag)
    if p.empty:
        return {"p_specific": float("nan"), "by_layout": {}}
    pooled_bin = p.groupby("v_bin_ms")["p_wkg"].median()
    p["rel"] = p["p_wkg"] / p["v_bin_ms"].map(pooled_bin)
    by_layout = {
        str(layout): {
            "p_wkg": float(g["p_wkg"].median()),
            "matched": float(g["rel"].median()),
            "bins": int(len(g)),
            "days": int(g["days"].max()),
        }
        for layout, g in p.groupby("layout_id")
    }
    return {"p_specific": float(p["p_wkg"].median()), "by_layout": by_layout}


def _only(frame: pd.DataFrame, rows: pd.Series | np.ndarray | slice, *cols: str) -> pd.DataFrame:
    """The ``rows`` of ``frame`` with the columns of ``cols`` it has: a
    statistic copies what it reads and not the whole table."""
    return frame.loc[rows, [c for c in dict.fromkeys(cols) if c in frame.columns]]


#: What the power statistics read of a sample.
_POWER_COLS = ("rpm", "v_ms", "a_g", "rho", "cf")


def _with_rpm(throttle: pd.DataFrame, *cols: str, gear: bool = False) -> pd.DataFrame:
    """The straight-line samples with an rpm (and a gear): every column,
    or those of ``cols``."""
    if throttle.empty or "rpm" not in throttle.columns:
        return throttle.iloc[:0]
    ok = throttle["straight"] & throttle["rpm"].notna() & (throttle["rpm"] > 0)
    if gear:
        ok &= throttle["gear"].notna() & (throttle["gear"] > 0)
    return _only(throttle, ok, *cols) if cols else throttle[ok]


def engine_curve(throttle: pd.DataFrame, c_drag: float) -> pd.DataFrame:
    """Power in standard air against engine rpm: per :data:`RPM_BIN` bin
    (``rpm``, its lower edge) the median :func:`standard_power` of the
    full-throttle straight-line samples (``p_wkg``) and their number
    (``n``); bins with :data:`CURVE_MIN_SAMPLES`. Empty when the samples
    have no rpm or fill fewer than :data:`CURVE_MIN_BINS` bins."""
    st = _with_rpm(throttle, *_POWER_COLS).copy()
    empty = pd.DataFrame(columns=["rpm", "p_wkg", "n"])
    if st.empty:
        return empty
    st["rpm"] = np.floor(st["rpm"] / RPM_BIN) * RPM_BIN
    st["p_wkg"] = standard_power(st, c_drag)
    out = st.groupby("rpm").agg(p_wkg=("p_wkg", "median"), n=("p_wkg", "size")).reset_index()
    out = out[out["n"] >= CURVE_MIN_SAMPLES].reset_index(drop=True)
    return out if len(out) >= CURVE_MIN_BINS else empty


def power_scale(throttle: pd.DataFrame, c_drag: float, base: pd.DataFrame) -> float:
    """How much power a set of samples shows against the curve ``base``
    of another powertrain of the same car: the median over rpm bins (with
    :data:`DAY_BIN_MIN_SAMPLES`) of the samples' power over the curve's at
    that rpm. NaN without rpm or a bin in common."""
    st = _with_rpm(throttle, *_POWER_COLS).copy()
    if st.empty or base.empty:
        return float("nan")
    st["rpm"] = np.floor(st["rpm"] / RPM_BIN) * RPM_BIN
    st["p_wkg"] = standard_power(st, c_drag)
    own = st.groupby("rpm")["p_wkg"].agg(["median", "size"])
    own = own[own["size"] >= DAY_BIN_MIN_SAMPLES]
    rel = (own["median"] / base.set_index("rpm")["p_wkg"]).dropna()
    return float(rel.median()) if len(rel) else float("nan")


def power_scale_by(
    throttle: pd.DataFrame, c_drag: float, base: pd.DataFrame, by: str = "layout_id"
) -> dict[str, float]:
    """:func:`power_scale` of the samples of each value of ``by``, all at
    once (NaN for one without rpm or a bin in common)."""
    out = {str(k): float("nan") for k in sorted(throttle[by].unique())} if len(throttle) else {}
    st = _with_rpm(throttle, by, *_POWER_COLS).copy()
    if st.empty or base.empty:
        return out
    st["rpm"] = np.floor(st["rpm"] / RPM_BIN) * RPM_BIN
    st["p_wkg"] = standard_power(st, c_drag)
    own = st.groupby([by, "rpm"])["p_wkg"].agg(["median", "size"]).reset_index()
    own = own[own["size"] >= DAY_BIN_MIN_SAMPLES]
    own["rel"] = own["median"] / own["rpm"].map(base.set_index("rpm")["p_wkg"])
    for k, v in own.dropna(subset=["rel"]).groupby(by)["rel"].median().items():
        out[str(k)] = float(v)
    return out


def gearing(throttle: pd.DataFrame) -> dict[int, float]:
    """Speed per 1,000 rpm (m/s) of each gear with
    :data:`GEAR_MIN_SAMPLES` full-throttle straight-line samples."""
    st = _with_rpm(throttle, gear=True)
    if st.empty:
        return {}
    ratio = st["v_ms"] / st["rpm"] * 1000.0
    out = ratio.groupby(st["gear"].round().astype(int)).agg(["median", "size"])
    return {
        int(g): float(m)
        for g, m, n in zip(out.index.to_numpy(), out["median"], out["size"])
        if n >= GEAR_MIN_SAMPLES
    }


def ratio_clusters(ratio: np.ndarray) -> list[float]:
    """The gears in a set of speed-per-1,000-rpm values: in one gear the
    ratio is one number, so the values fall into one tight cluster per
    gear. A cluster is a peak of the histogram (bins of
    :data:`RATIO_BIN`) holding :data:`CLUSTER_MIN_SHARE` of the values and
    :data:`CLUSTER_MIN_SAMPLES`; peaks within :data:`GEAR_TOL` of a lower
    one are that gear's. Returns the clusters' medians, lowest first."""
    r = np.asarray(ratio, float)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size < CLUSTER_MIN_SAMPLES:
        return []
    edges = np.arange(
        max(float(r.min()) - RATIO_BIN, 0.0), float(r.max()) + 2 * RATIO_BIN, RATIO_BIN
    )
    hist, _ = np.histogram(r, bins=edges)
    floor = max(CLUSTER_MIN_SHARE * r.size, CLUSTER_MIN_SAMPLES)
    out: list[float] = []
    for i in range(len(hist)):
        before = hist[i - 1] if i else 0
        after = hist[i + 1] if i + 1 < len(hist) else 0
        if hist[i] < floor or hist[i] < before or hist[i] <= after:
            continue
        centre = float(np.median(r[np.abs(r / (edges[i] + RATIO_BIN / 2.0) - 1.0) < GEAR_TOL]))
        if out and centre / out[-1] - 1.0 < GEAR_TOL:
            continue
        out.append(centre)
    return out


def _number_sets(sets: dict[str, list[float]]) -> dict[str, list[int]]:
    """Gear numbers for each day's ratios such that a ratio keeps its
    number from day to day: a day that shows only some of a set's gears
    (the lowest is not always used at full throttle) is numbered within
    the fullest set that holds all of its ratios. Numbers are ranks among
    the gears seen at full throttle, from 1; they are not the gearbox's."""
    catalogue: list[list[float]] = []
    out: dict[str, list[int]] = {}
    for date in sorted(sets, key=lambda d: (-len(sets[d]), d)):
        ratios = sets[date]
        numbers: list[int] | None = None
        for full in catalogue:
            match = [
                next((k for k, f in enumerate(full) if abs(r / f - 1.0) < GEAR_TOL), -1)
                for r in ratios
            ]
            if ratios and all(k >= 0 for k in match):
                numbers = [k + 1 for k in match]
                break
        if numbers is None:
            catalogue.append(list(ratios))
            numbers = list(range(1, len(ratios) + 1))
        out[date] = numbers
    return out


def with_gears(throttle: pd.DataFrame) -> pd.DataFrame:
    """``throttle`` with a ``gear`` on the samples that have an rpm and no
    logged gear: the day's speed per 1,000 rpm is clustered
    (:func:`ratio_clusters`) and each sample takes the number of the
    cluster within :data:`GEAR_TOL` of it (NaN outside every one: a
    slipping clutch, a missed sample). ``gear_inferred`` marks them. A
    logged gear is never replaced."""
    out = throttle.copy()
    if "gear" not in out.columns:
        out["gear"] = np.nan
    out["gear_inferred"] = False
    if out.empty or "rpm" not in out.columns:
        return out
    need = (out["gear"].isna() & out["rpm"].notna() & (out["rpm"] > 0)).to_numpy()
    if not need.any():
        return out
    at = np.flatnonzero(need)
    ratio = (out["v_ms"] / out["rpm"] * 1000.0).to_numpy(float)[at]
    day, dates = pd.factorize(out["date"].to_numpy()[at], sort=True)
    # the rows of each day, side by side
    order = np.argsort(day, kind="stable")
    ends = np.cumsum(np.bincount(day, minlength=len(dates)))
    rows = {str(d): order[e - n : e] for d, e, n in zip(dates, ends, np.diff(ends, prepend=0))}
    sets = {d: ratio_clusters(ratio[k]) for d, k in rows.items()}
    numbers = _number_sets({d: r for d, r in sets.items() if r})
    gear = np.full(at.size, np.nan)
    for d, k in rows.items():
        if not sets[d]:
            continue
        near = np.abs(ratio[k][:, None] / np.asarray(sets[d])[None, :] - 1.0)
        best = near.argmin(axis=1)
        gear[k] = np.where(
            near[np.arange(k.size), best] < GEAR_TOL, np.asarray(numbers[d], float)[best], np.nan
        )
    found = np.isfinite(gear)
    column = out["gear"].to_numpy(copy=True)
    column[at[found]] = gear[found]
    out["gear"] = column
    inferred = np.zeros(len(out), dtype=bool)
    inferred[at[found]] = True
    out["gear_inferred"] = inferred
    return out


def gear_sets_by_day(throttle: pd.DataFrame) -> dict[str, dict[int, float]]:
    """``date -> {gear: m/s per 1,000 rpm}``: the gears the car ran each
    day, from that day's full-throttle straight-line samples
    (:func:`gearing`). A formula car's gear set is changed between
    circuits, so the gearing is the day's, like the shift point."""
    st = _with_rpm(throttle, "date", "v_ms", "rpm", "gear", gear=True)
    if st.empty:
        return {}
    ratio = st["v_ms"] / st["rpm"] * 1000.0
    got = ratio.groupby([st["date"], st["gear"].round().astype(int)]).agg(["median", "size"])
    got = got[got["size"] >= DAY_BIN_MIN_SAMPLES]
    out: dict[str, dict[int, float]] = {}
    for (date, g), m in zip(got.index, got["median"]):
        out.setdefault(str(date), {})[int(g)] = float(m)
    return out


def complete_sets(
    day_sets: dict[str, dict[int, float]],
) -> tuple[dict[str, dict[int, float]], list[dict]]:
    """Each day's gears completed from the days that ran the same set.

    A day shows only the gears it used at full throttle in a straight
    line (the lowest often not), but the gearbox had them all. Days whose
    common gears agree within :data:`GEAR_TOL` ran one set; the set is
    the median over those days of each gear's ratio, and every one of the
    days gets all of it. Returns ``date -> set`` and the sets found
    (``gears``, ``days``, ``first`` and ``last``), most days first.
    """
    groups: list[dict] = []
    for date in sorted(day_sets, key=lambda d: (-len(day_sets[d]), d)):
        gears = day_sets[date]
        home = None
        for grp in groups:
            seen = grp["ratios"]
            common = [g for g in gears if g in seen]
            if common and all(
                abs(gears[g] / float(np.median(seen[g])) - 1.0) < GEAR_TOL for g in common
            ):
                home = grp
                break
        if home is None:
            home = {"ratios": {}, "dates": []}
            groups.append(home)
        home["dates"].append(date)
        for g, r in gears.items():
            home["ratios"].setdefault(g, []).append(r)
    out: dict[str, dict[int, float]] = {}
    found: list[dict] = []
    for grp in sorted(groups, key=lambda g: -len(g["dates"])):
        full = {g: float(np.median(r)) for g, r in sorted(grp["ratios"].items())}
        for date in grp["dates"]:
            out[date] = full
        found.append(
            {
                "gears": full,
                "days": len(grp["dates"]),
                "first": min(grp["dates"]),
                "last": max(grp["dates"]),
            }
        )
    return out, found


def shift_rpm_by_day(throttle: pd.DataFrame) -> pd.Series:
    """``date -> shift_rpm``: how hard the driver holds the gears that day.

    Where the gear is logged, the median over the gears the car shifts up
    out of (every gear but the day's highest) of the
    :data:`SHIFT_QUANTILE` of the full-throttle rpm in that gear; the top
    gear is run to wherever the straight ends, which is not a shift point.
    Without a gear channel, the same quantile over all of the day's
    full-throttle rpm. Days with :data:`SHIFT_MIN_SAMPLES`. The samples
    within 0.3 s of a gear change are not in the table, so it reads a
    little under the rpm the change is made at."""
    if throttle.empty or "rpm" not in throttle.columns:
        return pd.Series(dtype=float, name="shift_rpm")
    th = _only(throttle, throttle["rpm"].notna() & (throttle["rpm"] > 0), "date", "gear", "rpm")
    by_day = th.groupby("date")["rpm"]
    # every day at once: the quantile over the day, and over each of its gears
    out = by_day.quantile(SHIFT_QUANTILE)[by_day.size() >= SHIFT_MIN_SAMPLES].astype(float)
    geared = th[th["gear"].notna()] if "gear" in th.columns else th.iloc[:0]
    if len(geared) and len(out):
        gear = geared["gear"].round().rename("gear")
        cells = geared["rpm"].groupby([geared["date"], gear])
        tops = pd.DataFrame(
            {"rpm": cells.quantile(SHIFT_QUANTILE), "n": cells.size()}
        ).reset_index()
        top_gear = tops["date"].map(gear.groupby(geared["date"]).max())
        tops = tops[(tops["gear"] < top_gear) & (tops["n"] >= SHIFT_MIN_SAMPLES // 3)]
        by_gear = tops.groupby("date")["rpm"].median().astype(float).reindex(out.index)
        out = by_gear.where(by_gear.notna(), out)
    return pd.Series(
        out.to_numpy(float), index=out.index.astype(str), dtype=float, name="shift_rpm"
    )


def shift_times(throttle: pd.DataFrame, c_drag: float, k_corner: float = 0.0) -> dict[int, dict]:
    """Seconds of drive lost in each upshift, by the gear shifted out of.

    The full-throttle table has a gap around every gear change. Over an
    upshift's gap the car gains less speed than the engine's power either
    side of it would have given (integrated over the gap with the drag and
    the grade); that shortfall over the acceleration the engine alone
    gives there is the time it gave nothing. The same test over
    :data:`SHIFT_CONTROL_S` inside a gear, where nothing is lost, reads
    what the method itself adds, and that is taken off.

    Per gear with :data:`SHIFT_MIN_EVENTS` upshifts: ``shift_s`` (not
    under zero), ``n``, the median ``gap_s`` and ``short_ms`` and the
    ``bias_s`` taken off. Every upshift and every control pair of the
    table is worked at once.
    """
    need = ("session_id", "lap_num", "t_s", "v_ms", "a_g", "gear")
    if throttle.empty or any(col not in throttle.columns for col in need):
        return {}
    th = _only(throttle, throttle["gear"].notna(), *need, "lat_g", "grade", "rho")
    th = th.sort_values(["session_id", "lap_num", "t_s"], kind="stable")
    if len(th) < 2 * SHIFT_EDGE_SAMPLES:
        return {}
    lap = th.groupby(["session_id", "lap_num"], sort=False).ngroup().to_numpy()
    t, v = th["t_s"].to_numpy(float), th["v_ms"].to_numpy(float)
    gear = th["gear"].round().to_numpy(float)
    n = len(th)

    def _col(name: str, std: float) -> np.ndarray:
        return th[name].to_numpy(float) if name in th.columns else np.full(n, std)

    c = (c_drag if np.isfinite(c_drag) else 0.0) * _col("rho", RHO_REF) / RHO_REF
    grade = np.nan_to_num(_col("grade", 0.0))
    lat = np.nan_to_num(_col("lat_g", 0.0))
    # the power the engine is seen to give at each sample (W/kg)
    power = (th["a_g"].to_numpy(float) * G_MS2 + c * v * v + k_corner * lat * lat * G_MS2) * v

    def _edge(step: int) -> np.ndarray:
        """Per sample, the median power of it and the samples of its own
        lap and gear up to five before (``step`` −1) or after (+1) it."""
        near = np.full((n, SHIFT_EDGE_SAMPLES), np.nan)
        for k in range(SHIFT_EDGE_SAMPLES):
            other = np.clip(np.arange(n) + step * k, 0, n - 1)
            same = (lap[other] == lap) & (gear[other] == gear) & (other == np.arange(n) + step * k)
            near[:, k] = np.where(same, power[other], np.nan)
        return geo.nan_median(near, 1)

    before, after = _edge(-1), _edge(1)

    def _lost(i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(seconds, m/s)`` the car is short of full power from sample
        ``i`` to sample ``j``."""
        p = 0.5 * (before[i] + after[j])
        gap = t[j] - t[i]
        drag, slope = 0.5 * (c[i] + c[j]), 0.5 * (grade[i] + grade[j])
        speed = v[i].copy()
        for _ in range(SHIFT_STEPS):
            speed += gap / SHIFT_STEPS * (p / speed - drag * speed * speed - G_MS2 * slope)
        short = speed - v[j]
        return short / (p / v[i]), short

    first = np.arange(n - 1)
    gap = t[first + 1] - t[first]
    up = (
        (lap[first] == lap[first + 1])
        & (gear[first + 1] == gear[first] + 1)
        & (gap > SHIFT_GAP_S[0])
        & (gap < SHIFT_GAP_S[1])
    )
    shifts = first[up]
    step = np.median(np.diff(t)[np.diff(lap) == 0]) if n > 2 else np.nan
    span = int(round(SHIFT_CONTROL_S / step)) if np.isfinite(step) and step > 0 else 0
    control = np.arange(0, max(n - span, 0), SHIFT_CONTROL_EVERY) if span > 0 else np.zeros(0, int)
    control = control[
        (lap[control] == lap[control + span])
        & (gear[control] == gear[control + span])
        & (np.abs(t[control + span] - t[control] - SHIFT_CONTROL_S) < 0.5 * step)
    ]
    with np.errstate(all="ignore"):
        lost, short = _lost(shifts, shifts + 1)
        idle, _ = _lost(control, control + span)
    idle_of = {int(g): float(np.median(idle[gear[control] == g])) for g in np.unique(gear[control])}
    everywhere = float(np.median(idle)) if idle.size else 0.0
    out: dict[int, dict] = {}
    for g in np.unique(gear[shifts]):
        mine = gear[shifts] == g
        if int(mine.sum()) < SHIFT_MIN_EVENTS:
            continue
        either = [idle_of[k] for k in (int(g), int(g) + 1) if k in idle_of]
        bias = float(np.mean(either)) if either else everywhere
        out[int(g)] = {
            "shift_s": max(float(np.median(lost[mine])) - bias, 0.0),
            "n": int(mine.sum()),
            "gap_s": float(np.median(gap[up][mine])),
            "short_ms": float(np.median(short[mine])),
            "bias_s": bias,
        }
    return out


def corner_drag(throttle: pd.DataFrame, bins: pd.DataFrame) -> dict:
    """``k`` (1/g) with ``a = a_straight − k·lat²`` at full throttle: per
    layout and pooled, the median over the samples above
    :data:`CORNER_DRAG_MIN_LAT_G` of the deficit against the layout's own
    straight bin, over ``lat²``."""
    if throttle.empty or bins.empty:
        return {"k_corner_drag": float("nan"), "by_layout": {}, "n": 0}
    fast = np.abs(throttle["lat_g"]) >= CORNER_DRAG_MIN_LAT_G
    th = _only(throttle, fast, "layout_id", "v_ms", "a_g", "lat_g").copy()
    th["v_bin_ms"] = np.floor(th["v_ms"] / smp.SPEED_BIN_MS) * smp.SPEED_BIN_MS
    ref = bins.set_index(["layout_id", "v_bin_ms"])["a_g"]
    keys = pd.MultiIndex.from_frame(th[["layout_id", "v_bin_ms"]])
    th["a_straight"] = ref.reindex(keys).to_numpy()
    th = th.dropna(subset=["a_straight"])
    if th.empty:
        return {"k_corner_drag": float("nan"), "by_layout": {}, "n": 0}
    th["k"] = (th["a_straight"] - th["a_g"]) / th["lat_g"] ** 2
    return {
        "k_corner_drag": float(th["k"].median()),
        "by_layout": {str(k): float(v) for k, v in th.groupby("layout_id")["k"].median().items()},
        "n": int(len(th)),
    }


def day_levels(
    throttle: pd.DataFrame, bins: pd.DataFrame, c_drag: float, curve: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Each day's full-throttle level in standard air, per ``date`` and
    ``layout_id``.

    With an engine ``curve`` (:func:`engine_curve`) the level is the
    median over rpm bins of the day's power over the curve's at the same
    rpm: what the engine did, whatever gear and shift point it was driven
    in. Without one it is the median over speed bins of the day's power
    over the pooled power of the same layout and speed bin, which also
    moves when the driver shifts elsewhere. ``bins`` and ``samples`` say
    what is behind it; ``rho`` and ``cf`` are the day's air.
    """
    cols = ["date", "layout_id", "level", "bins", "samples", "rho", "cf"]
    if throttle.empty:
        return pd.DataFrame(columns=cols)
    by_rpm = curve is not None and len(curve) > 0
    read = ("date", "layout_id", *_POWER_COLS)
    st = (
        _with_rpm(throttle, *read) if by_rpm else _only(throttle, throttle["straight"], *read)
    ).copy()
    if st.empty:
        return pd.DataFrame(columns=cols)
    for col, std in (("rho", RHO_REF), ("cf", 1.0)):
        if col not in st.columns:
            st[col] = std
    st["p_wkg"] = standard_power(st, c_drag)
    if by_rpm:
        assert curve is not None
        st["bin"] = np.floor(st["rpm"] / RPM_BIN) * RPM_BIN
        ref = st["bin"].map(curve.set_index("rpm")["p_wkg"])
    else:
        st["bin"] = np.floor(st["v_ms"] / smp.SPEED_BIN_MS) * smp.SPEED_BIN_MS
        pooled = bin_power(bins, c_drag).set_index(["layout_id", "v_bin_ms"])["p_wkg"]
        ref = pd.Series(
            pooled.reindex(pd.MultiIndex.from_frame(st[["layout_id", "bin"]])).to_numpy(),
            index=st.index,
        )
    st["ref"] = ref
    day = (
        st.dropna(subset=["ref"])
        .groupby(["date", "layout_id", "bin"])
        .agg(
            p=("p_wkg", "median"),
            ref=("ref", "first"),
            n=("p_wkg", "size"),
            rho=("rho", "mean"),
            cf=("cf", "mean"),
        )
        .reset_index()
    )
    day = day[day["n"] >= DAY_BIN_MIN_SAMPLES]
    day["rel"] = day["p"] / day["ref"]
    out = (
        day.groupby(["date", "layout_id"])
        .agg(
            level=("rel", "median"),
            bins=("rel", "size"),
            samples=("n", "sum"),
            rho=("rho", "mean"),
            cf=("cf", "mean"),
        )
        .reset_index()
    )
    return out[out["bins"] >= DAY_MIN_BINS].reset_index(drop=True)[cols]


def density_slope(days: pd.DataFrame) -> dict:
    """What the air correction leaves: the slope of ``log level`` on
    ``log rho`` with one intercept per layout, its R² and the sd of
    ``log level`` before and after it is taken out."""
    d = days.dropna(subset=["level", "rho"])
    d = d[(d["level"] > 0) & (d["rho"] > 0)]
    out = {"days": int(len(d)), "slope": float("nan"), "r2": float("nan")}
    out["sd"] = float(np.log(days["level"]).std()) if len(days) > 1 else float("nan")
    out["sd_at_equal_density"] = float("nan")
    if len(d) < 5:
        return out
    y = np.log(d["level"]) - np.log(d["level"]).groupby(d["layout_id"]).transform("mean")
    x = np.log(d["rho"]) - np.log(d["rho"]).groupby(d["layout_id"]).transform("mean")
    den = float(np.sum(x * x))
    if den <= 0:
        return out
    slope = float(np.sum(x * y) / den)
    resid = y - slope * x
    total = float(np.sum(y * y))
    out.update(
        slope=slope,
        r2=float(1.0 - np.sum(resid * resid) / total) if total > 0 else float("nan"),
        sd_at_equal_density=float(resid.std()),
    )
    return out


# ------------------------------------------------------------------ the fit


def braking_level(brake: pd.DataFrame, c_drag: float, lat_level: float | None = None) -> dict:
    """``mu_b``: the 90th percentile over dry braking zones of the zone's
    peak less the drag at the speed of the peak; and the same per layout.

    With the car's lateral level the zones are those braked in a straight
    line: a peak that comes with under :data:`BRAKE_STRAIGHT_SHARE` of it
    in lateral. A stop made while cornering has less than ``mu_b`` to
    brake with, which is :func:`braking_shape`'s to say and not a lower
    ``mu_b``. Every zone where there are fewer than
    :data:`BRAKE_STRAIGHT_MIN_ZONES` of those, or no lateral to go by
    (``straight`` says which; ``zones`` are the ones used)."""
    if brake.empty:
        return {"mu_b": float("nan"), "by_layout": {}, "zones": 0, "straight": False}
    straight = False
    if lat_level is not None and np.isfinite(lat_level) and "lat_peak_g" in brake.columns:
        line = brake[brake["lat_peak_g"].abs() < BRAKE_STRAIGHT_SHARE * lat_level]
        if len(line) >= BRAKE_STRAIGHT_MIN_ZONES:
            brake, straight = line, True
    drag = (c_drag if np.isfinite(c_drag) else 0.0) * brake["v_peak_ms"] ** 2 / G_MS2
    peak = brake["decel_p95_g"] - drag
    return {
        "mu_b": float(peak.quantile(0.9)),
        "by_layout": {
            str(k): float(v) for k, v in peak.groupby(brake["layout_id"]).quantile(0.9).items()
        },
        "zones": int(len(peak)),
        "straight": straight,
    }


def brake_onset(brake: pd.DataFrame) -> dict:
    """How long the car goes without drive or full braking where a stop
    begins: ``onset_s``, the time from full throttle to the brake
    application (``lift_s``) plus half the time the brake takes to build
    (``build_s`` is to :data:`~.samples.BRAKE_BUILD_SHARE` of the zone's
    highest pressure, so the whole build is that over the share). Medians
    over the dry stops from full throttle at :data:`ONSET_MIN_SPEED_MS` or
    more; NaN with fewer than :data:`ONSET_MIN_ZONES`. Without a pressure
    channel the build counts as none."""
    out = {"onset_s": float("nan"), "lift_s": float("nan"), "build_s": float("nan"), "zones": 0}
    if brake.empty or "lift_s" not in brake.columns:
        return out
    stops = brake[(brake["v_on_ms"] >= ONSET_MIN_SPEED_MS) & brake["lift_s"].notna()]
    out["zones"] = int(len(stops))
    if len(stops) < ONSET_MIN_ZONES:
        return out
    lift = float(stops["lift_s"].median())
    build = float(stops["build_s"].median()) if "build_s" in stops.columns else float("nan")
    half = 0.5 * build / smp.BRAKE_BUILD_SHARE if np.isfinite(build) else 0.0
    out.update(onset_s=lift + half, lift_s=lift, build_s=build)
    return out


def onset_by_day(brake: pd.DataFrame) -> pd.DataFrame:
    """:func:`brake_onset` of every day at once: ``date, onset_s, lift_s,
    build_s, zones``, the days with :data:`ONSET_MIN_ZONES` stops."""
    cols = ["date", "onset_s", "lift_s", "build_s", "zones"]
    if brake.empty or "lift_s" not in brake.columns:
        return pd.DataFrame(columns=cols)
    stops = brake[(brake["v_on_ms"] >= ONSET_MIN_SPEED_MS) & brake["lift_s"].notna()]
    days = stops.groupby("date").agg(
        lift_s=("lift_s", "median"), build_s=("build_s", "median"), zones=("lift_s", "size")
    )
    days = days[days["zones"] >= ONSET_MIN_ZONES].reset_index()
    days["onset_s"] = days["lift_s"] + (0.5 * days["build_s"] / smp.BRAKE_BUILD_SHARE).fillna(0.0)
    return days[cols]


def braking_shape(
    brake: pd.DataFrame,
    c_drag: float,
    mu_b: float,
    lat_level: float,
    *,
    min_zones: int | None = None,
) -> dict:
    """How far the braking limit under cornering is from the diamond
    towards the ellipse.

    With ``x`` the lateral at a zone's peak deceleration over the car's
    lateral level and ``y`` that peak (less drag) over ``mu_b`` (the level
    of the zones braked in a straight line, :func:`braking_level`), the
    diamond allows ``y = 1 − x`` and the ellipse ``y = √(1 − x²)``. Over
    the dry zones in each band of ``x`` (:data:`BRAKE_BAND_WIDTH` wide,
    within :data:`BRAKE_BANDS`, with :data:`BRAKE_BAND_MIN_ZONES`) the
    90th percentile of ``y`` (as ``mu_b`` is the 90th percentile of the
    peaks in a straight line) lies some share of the way from the one to
    the other; ``brake_ellipse`` is the median of the bands' shares, held
    to 0–1. NaN without the columns or a band. ``bands`` are the workings.
    ``min_zones`` replaces :data:`BRAKE_BAND_MIN_ZONES` (a handful of
    simulated laps has few zones).

    Read against the straight-line level it gives a simulated car's share
    back (median +0.03 over fifteen cars from 0.2 to 0.8); read against a
    ``mu_b`` taken over every zone it reads 0.09 high, because that
    ``mu_b`` is itself lowered by the zones braked while cornering.
    """
    out: dict = {"brake_ellipse": float("nan"), "bands": pd.DataFrame()}
    need = ("lat_peak_g", "decel_p95_g", "v_peak_ms")
    ok = np.isfinite(mu_b) and np.isfinite(lat_level) and mu_b > 0 and lat_level > 0
    if brake.empty or not ok or any(col not in brake.columns for col in need):
        return out
    drag = (c_drag if np.isfinite(c_drag) else 0.0) * brake["v_peak_ms"] ** 2 / G_MS2
    zones = pd.DataFrame(
        {
            "x": (brake["lat_peak_g"].abs() / lat_level).to_numpy(float),
            "y": ((brake["decel_p95_g"] - drag) / mu_b).to_numpy(float),
        }
    ).dropna()
    zones = zones[(zones["x"] >= BRAKE_BANDS[0]) & (zones["x"] < BRAKE_BANDS[1])]
    if zones.empty:
        return out
    cells = zones.groupby(np.floor(zones["x"] / BRAKE_BAND_WIDTH) * BRAKE_BAND_WIDTH)
    bands = pd.DataFrame(
        {"zones": cells.size(), "x": cells["x"].median(), "y_p90": cells["y"].quantile(0.9)}
    )
    least = BRAKE_BAND_MIN_ZONES if min_zones is None else min_zones
    bands = bands[bands["zones"] >= least].rename_axis("band").reset_index()
    if bands.empty:
        return out
    bands["diamond"] = 1.0 - bands["x"]
    bands["ellipse"] = np.sqrt(1.0 - bands["x"] ** 2)
    bands["share"] = (bands["y_p90"] - bands["diamond"]) / (bands["ellipse"] - bands["diamond"])
    out["brake_ellipse"] = float(np.clip(bands["share"].median(), 0.0, 1.0))
    out["bands"] = bands
    return out


def _engine_names(car: str, throttle: pd.DataFrame) -> list[str]:
    """The car's powertrains that have samples, in the order they came."""
    seen = set(throttle["engine"].unique()) if "engine" in throttle.columns else set()
    listed = [str(s["name"]) for s in POWERTRAINS.get(car, [])]
    names = [n for n in listed if n in seen] + sorted(seen - set(listed))
    return names or [DEFAULT_ENGINE]


def fit_car(
    tables: smp.CarTables, car: str, *, air: pd.DataFrame | None = None
) -> tuple[CarModel, dict]:
    """The model of ``car`` (a family: its raw cars' laps together) from
    the sample tables, and the workings the gate reports.

    The chassis numbers (drag, cornering drag, braking, lateral level) are
    read from every powertrain's laps together; power, gearing and the
    shift point per powertrain. ``air`` is :func:`air_state`; without it
    every sample counts as driven in standard air.
    """
    fam = as_families(tables)

    def _own(tbl: pd.DataFrame, basis: str | None = "dry") -> pd.DataFrame:
        if tbl.empty:
            return tbl
        m = tbl["car"] == car
        if basis is not None:
            m &= tbl["basis"] == basis
        return tbl[m]

    own_throttle = _own(fam.throttle)
    throttle = with_gears(with_air(_only(own_throttle, slice(None), *_FIT_COLS), air))
    if "engine" not in throttle.columns:
        throttle["engine"] = DEFAULT_ENGINE
    names = _engine_names(car, throttle)
    facts = PUBLISHED.get(car, {})

    # Drag: every powertrain's long layouts vote.
    cells: dict[str, dict] = {}
    of_engine = {
        name: throttle if len(names) == 1 else throttle[throttle["engine"] == name]
        for name in names
    }
    bins_of = {name: solid_bins(own) for name, own in of_engine.items()}
    for name in names:
        for layout, d in fit_drag(bins_of[name]).get("by_layout", {}).items():
            cells[layout if len(names) == 1 else f"{layout} ({name})"] = d
    prior = facts.get("c_drag_prior")
    top = float(np.median([d["c_drag"] for d in cells.values()])) if cells else float("nan")
    across = fit_drag_rpm(throttle)
    # the same rpm in different gears where the logs allow; else the top of the table
    by_rpm = bool(np.isfinite(across["c_drag"]) and across["c_drag"] > 0)
    fitted = float(across["c_drag"]) if by_rpm else top
    how = "at equal rpm across gears" if by_rpm else "at the top of the straight table"
    if np.isfinite(fitted) and prior:
        c_drag = float(np.clip(fitted, DRAG_PRIOR_BAND[0] * prior, DRAG_PRIOR_BAND[1] * prior))
        source = (
            f"fitted {how}"
            if c_drag == fitted
            else f"fitted {how}, held at the edge of the prior's band"
        )
    elif np.isfinite(fitted):
        c_drag, source = fitted, f"fitted {how} (no prior)"
    elif prior:
        c_drag, source = float(prior), "prior (no layout reaches 50 m/s)"
    else:
        c_drag, source = float("nan"), "none"
    drag: dict = {
        "c_drag": c_drag,
        "fitted": fitted,
        "top_of_table": top,
        "across_gears": across,
        "prior": prior,
        "source": source,
        "by_layout": cells,
    }
    c = c_drag if np.isfinite(c_drag) else 0.0

    # Power, gearing and shift point per powertrain.
    engines: dict[str, dict] = {}
    work_engines: dict[str, dict] = {}
    day_frames: list[pd.DataFrame] = []
    k_vals: list[tuple[float, int]] = []
    k_layouts: dict[str, float] = {}
    base_name, base_curve = "", pd.DataFrame(columns=["rpm", "p_wkg", "n"])
    # The gearbox is the car's: a powertrain that kept it (the KK-F-V2) has
    # its gears completed from every day that ran the same set, and one
    # that changed it (the Inferno 86's final drive) falls into a set of
    # its own.
    all_sets, car_sets = complete_sets(gear_sets_by_day(throttle))
    for name in names:
        own, bins = of_engine[name], bins_of[name]
        power = specific_power(bins, c)
        curve = engine_curve(own, c)
        own_days = set(own["date"].astype(str).unique())
        day_sets = {d: g for d, g in all_sets.items() if d in own_days}
        inferred = bool(own["gear_inferred"].any()) if len(own) else False
        # the gear sets this powertrain ran, by how many of its days ran each
        sets_found = sorted(
            (
                {**f, "days": sum(1 for d in day_sets if day_sets[d] is f["gears"])}
                for f in car_sets
                if any(day_sets[d] is f["gears"] for d in day_sets)
            ),
            key=lambda f: -f["days"],
        )
        # its own gears: the set most of its days ran
        gears = sets_found[0]["gears"] if sets_found else gearing(own)
        # Against the car's first powertrain with a curve: how much more
        # (or less) power this one makes at equal rpm. One with too few
        # samples for a curve of its own takes that curve at this scale.
        scale, borrowed = 1.0, ""
        if base_curve.empty and len(curve):
            base_name, base_curve = name, curve
        elif len(base_curve):
            scale = power_scale(own, c, base_curve)
            if curve.empty and np.isfinite(scale):
                curve = base_curve.assign(p_wkg=base_curve["p_wkg"] * scale, n=0)
                borrowed = base_name
        if not gears and borrowed:
            gears = gearing(of_engine[base_name])
        shifts = shift_rpm_by_day(own)
        days = day_levels(own, bins, c, curve)
        days.insert(0, "engine", name)
        days["shift_rpm"] = days["date"].map(shifts).to_numpy(dtype=float)
        day_frames.append(days)
        kd = corner_drag(own, bins)
        if np.isfinite(kd["k_corner_drag"]):
            k_vals.append((kd["k_corner_drag"], kd["n"]))
        for layout, v in kd["by_layout"].items():
            k_layouts[layout if len(names) == 1 else f"{layout} ({name})"] = v
        p_spec = power["p_specific"]
        if not np.isfinite(p_spec) and borrowed:
            # too few samples for a straight table: the base's level at this scale
            p_spec = float(work_engines[base_name]["power"]["p_specific"]) * scale
        lost = shift_times(own, c, kd["k_corner_drag"] if np.isfinite(kd["k_corner_drag"]) else 0.0)
        if not lost and borrowed:  # too few upshifts of its own: the same gearbox and driver
            lost = work_engines[base_name]["shift_times"]
        engines[name] = {
            "p_specific": _round(p_spec, 2),
            "curve": [[float(r), round(float(p), 2)] for r, p in zip(curve["rpm"], curve["p_wkg"])]
            or None,
            "gears": {str(g): round(v, 4) for g, v in sorted(gears.items())} or None,
            "gears_inferred": inferred,
            "gear_sets": [
                {**f, "gears": {str(g): round(v, 4) for g, v in f["gears"].items()}}
                for f in sets_found
            ],
            "shift_rpm": _round(float(shifts.median()), 0) if len(shifts) else None,
            "shift_s": {str(g): round(d["shift_s"], 3) for g, d in sorted(lost.items())} or None,
            "scale": _round(scale, 4),
            "scale_against": base_name if name != base_name and base_name else None,
            "curve_from": borrowed or None,
            "days": int(own["date"].nunique()),
            "samples": int(own["straight"].sum()) if len(own) else 0,
        }
        # each layout's power over the powertrain's own curve, at equal rpm
        at_rpm = power_scale_by(own, c, curve) if len(curve) and not borrowed else {}
        work_engines[name] = {
            "power": power,
            "curve": curve,
            "bins": bins,
            "shifts": shifts,
            "at_rpm": at_rpm,
            "gear_sets": day_sets,
            "shift_times": lost,
        }
    with_data = [n for n in names if engines[n]["p_specific"] is not None]
    default = with_data[-1] if with_data else names[-1]
    # cornering drag: the powertrains' medians weighted by their samples
    k_corner = (
        float(np.average([k for k, _ in k_vals], weights=[n for _, n in k_vals]))
        if k_vals
        else float("nan")
    )
    cdrag: dict = {
        "k_corner_drag": k_corner,
        "by_layout": k_layouts,
        "n": int(sum(n for _, n in k_vals)),
    }

    level = smp.lateral_level(smp.apex_summary(_own(fam.apex, None)))
    dry = level[level["basis"] == "dry"]
    voting = dry[dry["days"] >= MIN_VOTE_DAYS]
    mu_y = float((voting if len(voting) else dry)["level_g"].median()) if len(dry) else float("nan")
    both = level.pivot_table(index="layout_id", columns="basis", values="level_g")
    wet_scale = (
        float((both["wet"] / both["dry"]).dropna().median())
        if {"dry", "wet"} <= set(both.columns)
        else float("nan")
    )
    # the lateral level the braking is read against is the apex table's:
    # mu_b from the zones braked in a straight line by it, and the limit's
    # shape from the others against the two, each at its 90th percentile
    braking = braking_level(_own(fam.brake), c, mu_y)
    shape = braking_shape(_own(fam.brake), c, braking["mu_b"], mu_y)
    braking["shape"] = shape
    onset = brake_onset(_own(fam.brake))
    braking["onset"] = onset
    braking["onset_days"] = onset_by_day(_own(fam.brake))
    days = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()
    if days.empty:
        days = pd.DataFrame(
            columns=["engine", "date", "layout_id", "level", "bins", "samples", "rho", "cf"]
        ).assign(shift_rpm=np.nan)
    density = density_slope(days)
    pooled = solid_bins(of_engine[default], by=("car",))
    brake = _own(fam.brake)
    apex = _own(fam.apex)
    support = {
        "straight": {
            "v_min": _round(pooled["v_bin_ms"].min(), 1) if len(pooled) else None,
            "v_max": (
                _round(pooled["v_bin_ms"].max() + smp.SPEED_BIN_MS, 1) if len(pooled) else None
            ),
        },
        "braking": {
            "v_min": _round(brake["v_min_ms"].quantile(0.01), 1) if len(brake) else None,
            "v_max": _round(brake["v_on_ms"].quantile(0.99), 1) if len(brake) else None,
        },
        "apex": {
            "v_min": _round(apex["v_apex_ms"].quantile(0.01), 1) if len(apex) else None,
            "v_max": _round(apex["v_apex_ms"].quantile(0.99), 1) if len(apex) else None,
        },
    }
    model = CarModel(
        car=car,
        raw_car=car,
        mass_kg=facts.get("mass_kg"),
        grip={
            "mu_y": _round(mu_y, 4),
            "mu_b": _round(braking["mu_b"], 4),
            "n_brake": 1.0,
            "brake_ellipse": _round(shape["brake_ellipse"], 3),
            "onset_s": _round(onset["onset_s"], 3),
        },
        power={
            "c_drag": _round(drag["c_drag"], 7),
            "rho_ref": round(RHO_REF, 4),
            "k_corner_drag": _round(cdrag["k_corner_drag"], 4),
            "default_engine": default,
            "engines": engines,
            "p_specific": engines[default]["p_specific"],
            "a_ft_by_v": [
                [round(float(v), 2), round(float(a), 4)]
                for v, a in zip(pooled["v_ms"], pooled["a_g"])
            ],
            "day_sd": _round(density["sd"], 4),
        },
        wet={"grip_scale": _round(wet_scale, 3)},
        support=support,
        ref_u={},
        meta={
            "mu_y_source": "apex table; provisional until the solver sets it (median ref_u = 1)",
            "c_drag_source": drag["source"],
            "air": "standard (SAE J1349: 99 kPa dry, 25 C)",
            "fitted_from": {
                "raw_cars": sorted(throttle["raw_car"].unique()) if len(throttle) else [],
                "layouts": (
                    sorted(str(k) for k in throttle["layout_id"].unique()) if len(throttle) else []
                ),
                "n_days": int(throttle["date"].nunique()) if len(throttle) else 0,
                "n_laps": (
                    int(len(apex[["session_id", "lap_num"]].drop_duplicates())) if len(apex) else 0
                ),
            },
            "fit_at_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        },
    )
    work = {
        "drag": drag,
        "engines": work_engines,
        "power": work_engines[default]["power"],
        "corner_drag": cdrag,
        "braking": braking,
        "level": level,
        "days": days,
        "density": density,
        "bins": work_engines[default]["bins"],
    }
    return model, work


def trap_speed(throttle: pd.DataFrame) -> pd.Series:
    """Per layout, the median over laps of the highest speed reached at
    full throttle (the logged speed-trap figure a drag has to reproduce)."""
    if throttle.empty:
        return pd.Series(dtype=float)
    top = throttle.groupby(["layout_id", "session_id", "lap_num"])["v_ms"].max()
    return top.groupby("layout_id").median()


# ------------------------------------------------------------------ the gate


def table_clauses(
    root: Path, tables: smp.CarTables
) -> tuple[GateReport, dict[str, tuple[CarModel, dict]]]:
    """The clauses of gate C2 that need no solver: every car's numbers
    from its tables, and whether a powertrain's power agrees between
    layouts. Returns the report so far and :func:`fit_car` of every car
    (family); :func:`.car_gate.check_car_fit` adds the clauses through the
    solver."""
    air = air_state(root, geo.load_sessions(root))
    rep = GateReport("C2", "the car model from the tables (`lapsim check-car-fit`)")
    present = sorted({family_of(str(c)) for c in tables.sessions["car"].unique()})
    cars = [c for c in smp.MODEL_CARS if c in present]
    listed = [c for c in present if c not in cars]
    fits = {c: fit_car(tables, c, air=air) for c in cars + listed}

    # 1. Power agrees between layouts, powertrain by powertrain.
    lay_rows = []
    for c, (model, work) in fits.items():
        for name, eng in work["engines"].items():
            pooled = eng["power"]["p_specific"]
            for layout, d in eng["power"]["by_layout"].items():
                at_rpm = eng["at_rpm"].get(layout, float("nan"))
                by_speed = d["p_wkg"] / pooled
                lay_rows.append(
                    {
                        "car": c,
                        "engine": name,
                        "layout_id": layout,
                        "days": d["days"],
                        "bins": d["bins"],
                        "p_wkg": d["p_wkg"],
                        "over_pooled": by_speed,
                        "at_equal_rpm": at_rpm,
                        "gated": at_rpm if np.isfinite(at_rpm) else by_speed,
                    }
                )
    lay = pd.DataFrame(lay_rows)
    gated = lay[lay["car"].isin(cars)] if len(lay) else lay
    off = (gated["gated"] - 1.0).abs() if len(gated) else pd.Series(dtype=float)
    rep.items.append(
        GateItem(
            "Power agrees between layouts",
            bool((off <= POWER_BETWEEN_LAYOUTS_MAX).all()) if len(gated) else None,
            "; ".join(
                f"{c} ({e}) " + " / ".join(f"{v - 1:+.1%}" for v in g["gated"])
                for (c, e), g in gated.groupby(["car", "engine"], sort=False)
            ),
            f"each layout's own within {POWER_BETWEEN_LAYOUTS_MAX:.0%} of its powertrain's, in "
            "standard air: at equal rpm where the powertrain has a curve of its own",
            "`at_equal_rpm`: the layout's power over the powertrain's curve, the median over "
            "rpm bins; `over_pooled`: the layout's median over its speed bins over the "
            "powertrain's `p_specific`, which also moves with the gears a layout is driven in. "
            "`gated` is the first where there is one.\n\n" + md_table(lay, floats="{:.3f}"),
        )
    )

    # 2. The chassis.
    rows = []
    for c, (model, work) in fits.items():
        rows.append(
            {
                "car": c,
                "model": c in cars,
                "raw_cars": ", ".join(model.meta["fitted_from"]["raw_cars"]),
                "c_drag_e4": 1e4 * work["drag"]["c_drag"],
                "c_fitted_e4": 1e4 * work["drag"]["fitted"],
                "drag_source": work["drag"]["source"],
                "k_corner": work["corner_drag"]["k_corner_drag"],
                "mu_y": model.grip["mu_y"],
                "mu_b": model.grip["mu_b"],
                "brake_ellipse": model.grip.get("brake_ellipse"),
                "onset_s": model.grip.get("onset_s"),
                "lift_s": work["braking"]["onset"]["lift_s"],
                "build_s": work["braking"]["onset"]["build_s"],
                "wet_scale": model.wet["grip_scale"],
            }
        )
    summary = pd.DataFrame(rows)
    rep.items.append(
        GateItem(
            "The chassis, shared by a car's powertrains",
            None,
            "; ".join(
                f"{r.car} c {r.c_drag_e4:.2f}e-4 /m, k {r.k_corner:.3f}, mu_b {r.mu_b:.2f}, "
                f"level {r.mu_y:.2f}"
                for r in summary[summary["model"]].itertuples()
            ),
            "",
            md_table(summary, floats="{:.3f}")
            + f"\n\n`c_drag` is at the density of standard air ({RHO_REF:.3f} kg/m³). `mu_y` "
            "is the apex table's level, provisional: the convention (median `ref_u` = 1) "
            "needs the solver.",
        )
    )

    # 3. The powertrains.
    eng_rows, curve_rows = [], []
    for c, (model, work) in fits.items():
        for name, e in model.power["engines"].items():
            curve = e["curve"] or []
            peak = max(curve, key=lambda rp: rp[1]) if curve else [np.nan, np.nan]
            eng_rows.append(
                {
                    "car": c,
                    "engine": name,
                    "default": name == model.power["default_engine"],
                    "days": e["days"],
                    "samples": e["samples"],
                    "p_specific": e["p_specific"],
                    "scale": e["scale"] if e["scale_against"] else np.nan,
                    "curve": (
                        ("of " + e["curve_from"]) if e["curve_from"] else ("own" if curve else "")
                    ),
                    "peak_wkg": peak[1],
                    "peak_rpm": peak[0],
                    "shift_rpm": e["shift_rpm"],
                    "gears": (
                        " / ".join(f"{v:.2f}" for v in e["gears"].values()) if e["gears"] else ""
                    ),
                    "shift_s": (
                        " / ".join(f"{v:.2f}" for v in e["shift_s"].values())
                        if e.get("shift_s")
                        else ""
                    ),
                }
            )
            for rpm, p in curve:
                curve_rows.append({"car": c, "engine": name, "rpm": int(rpm), "p_wkg": p})
    set_rows = []
    for c, (model, work) in fits.items():
        for name, e in model.power["engines"].items():
            for f in e.get("gear_sets") or []:
                set_rows.append(
                    {
                        "car": c,
                        "engine": name,
                        "from": "speed over rpm" if e["gears_inferred"] else "gear channel",
                        "days": f["days"],
                        "first": f["first"],
                        "last": f["last"],
                        "gears": " / ".join(f"{v:.2f}" for v in f["gears"].values()),
                    }
                )
    eng = pd.DataFrame(eng_rows)
    curves = pd.DataFrame(curve_rows)
    wide = (
        curves.pivot_table(index=["car", "engine"], columns="rpm", values="p_wkg").reset_index()
        if len(curves)
        else curves
    )
    rep.items.append(
        GateItem(
            "The powertrains",
            None,
            "; ".join(
                f"{c} ({e}) {p:.0f} W/kg" + (f", shifts at {sh:.0f}" if pd.notna(sh) else "")
                for c, e, p, sh in zip(
                    eng["car"], eng["engine"], eng["p_specific"], eng["shift_rpm"]
                )
                if c in cars
            ),
            "power at the wheels per mass in standard air; reported",
            md_table(eng, floats="{:.1f}")
            + "\n\n`shift_s`: seconds of drive lost in an upshift out of each gear but the top "
            "one (`shift_times`). "
            + "`gears`: m/s per 1,000 rpm, lowest gear seen first. `shift_rpm`: the median "
            "over the powertrain's days of the day's 99th percentile of full-throttle rpm. "
            "`scale`: its power over the car's first powertrain's at equal rpm; a powertrain "
            "with too few samples for a curve of its own has that one's at this scale."
            + "\n\nThe gear sets each powertrain ran (m/s per 1,000 rpm; `first` and `last` "
            "are of the set over the whole car, `days` this powertrain's). A gear made from "
            "speed over rpm is numbered among the gears seen at full throttle, not as the "
            "gearbox numbers them:\n\n"
            + md_table(pd.DataFrame(set_rows))
            + (
                "\n\nPower against rpm (W/kg, by the bin's lower edge):\n\n"
                + md_table(wide, floats="{:.1f}")
                if len(wide)
                else ""
            ),
        )
    )

    # 4. Drag.
    drag_rows = []
    for c, (model, work) in fits.items():
        for cell, d in work["drag"]["by_layout"].items():
            drag_rows.append(
                {
                    "car": c,
                    "cell": cell,
                    "bins": d["bins"],
                    "rho": d["rho"],
                    "c_drag_e4": 1e4 * d["c_drag"],
                    "p_top_wkg": d["p_top"],
                    "v_terminal_ms": d["v_terminal_ms"],
                }
            )
    rep.items.append(
        GateItem(
            "Drag from the top of the straight tables",
            None,
            "; ".join(f"{r['car']} {r['cell']} {r['c_drag_e4']:.2f}e-4" for r in drag_rows)
            or "no layout reaches 50 m/s",
            f"layouts reaching {DRAG_REACH_MS:.0f} m/s; bins from {DRAG_FIT_MIN_MS:.0f} m/s; "
            "brought to the density of standard air",
            md_table(pd.DataFrame(drag_rows), floats="{:.3f}")
            + "\n\n`rho`: the mean air density of the bins; `v_terminal_ms`: where the fitted "
            "line reaches zero acceleration in that air.",
        )
    )

    # 5. Sanity band: power over the published crank power per mass.
    band_rows = []
    for c in cars:
        facts = PUBLISHED.get(c, {})
        if not (facts.get("crank_kw") and facts.get("mass_kg")):
            continue
        for name, e in fits[c][0].power["engines"].items():
            if e["p_specific"] is None:
                continue
            ratio = e["p_specific"] * facts["mass_kg"] / (1e3 * facts["crank_kw"])
            band_rows.append(
                {
                    "car": c,
                    "engine": name,
                    "mass_kg": facts["mass_kg"],
                    "crank_kw": facts["crank_kw"],
                    "ratio": ratio,
                }
            )
    rep.items.append(
        GateItem(
            "Power at the wheels over published crank power",
            None,
            "; ".join(f"{r['car']} ({r['engine']}) {r['ratio']:.2f}" for r in band_rows)
            or "no published power",
            f"sanity band {POWER_OVER_CRANK_BAND[0]}–{POWER_OVER_CRANK_BAND[1]}, reported",
            md_table(pd.DataFrame(band_rows), floats="{:.3f}"),
        )
    )

    # 6. Power and shift point by day.
    dens_rows = []
    for c, (model, work) in fits.items():
        days = work["days"]
        for name, g in days.groupby("engine") if len(days) else []:
            dens_rows.append(
                {
                    "car": c,
                    "engine": name,
                    "days": int(len(g)),
                    "level_min": float(g["level"].min()),
                    "level_max": float(g["level"].max()),
                    "sd_log": float(np.log(g["level"]).std()) if len(g) > 1 else np.nan,
                    "shift_min": float(g["shift_rpm"].min()),
                    "shift_max": float(g["shift_rpm"].max()),
                }
            )
    dens = pd.DataFrame(dens_rows)
    frames = [work["days"].assign(car=c) for c, (_, work) in fits.items() if c in cars]
    frames = [f for f in frames if len(f)]
    per_day = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    left = {c: fits[c][1]["density"] for c in cars}
    rep.items.append(
        GateItem(
            "Power and shift point by day",
            None,
            "; ".join(
                f"{r.car} ({r.engine}) sd {r.sd_log:.3f}"
                for r in dens[dens["car"].isin(cars)].itertuples()
            ),
            "the day's power over its powertrain's, at equal rpm where there is a curve; "
            "reported",
            md_table(dens, floats="{:.3f}")
            + "\n\nWhat the air correction leaves, as a slope of log level on log density "
            "within a layout: "
            + "; ".join(f"{c} {d['slope']:+.2f} (R² {d['r2']:.2f})" for c, d in left.items())
            + ".\n\nPer day and layout:\n\n"
            + (
                md_table(
                    per_day[
                        [
                            "car",
                            "engine",
                            "date",
                            "layout_id",
                            "level",
                            "shift_rpm",
                            "bins",
                            "samples",
                            "rho",
                            "cf",
                        ]
                    ].sort_values(["car", "date"]),
                    floats="{:.3f}",
                )
                if len(per_day)
                else "_none_"
            ),
        )
    )

    # 7. By layout: braking, cornering drag.
    lvl_rows = []
    for c, (model, work) in fits.items():
        for layout, v in work["braking"]["by_layout"].items():
            lvl_rows.append({"car": c, "what": "mu_b", "cell": layout, "value": v})
        for layout, v in work["corner_drag"]["by_layout"].items():
            lvl_rows.append({"car": c, "what": "k_corner_drag", "cell": layout, "value": v})
    shape_rows: list[pd.DataFrame] = [
        work["braking"]["shape"]["bands"].assign(car=c)
        for c, (model, work) in fits.items()
        if c in cars and len(work["braking"]["shape"]["bands"])
    ]
    band_table = (
        pd.concat(shape_rows, ignore_index=True)[
            ["car", "band", "zones", "x", "y_p90", "diamond", "ellipse", "share"]
        ]
        if shape_rows
        else pd.DataFrame()
    )
    rep.items.append(
        GateItem(
            "Braking peak and cornering drag by layout",
            None,
            "; ".join(
                f"{r.car} mu_b {r.mu_b:.2f}, k {r.k_corner:.3f}"
                for r in summary[summary["model"]].itertuples()
            ),
            "reported",
            md_table(pd.DataFrame(lvl_rows), floats="{:.3f}")
            + "\n\nThe braking limit under cornering (`braking_shape`): per band of `x`, the "
            "lateral at a zone's peak deceleration over the car's lateral level, the 90th "
            "percentile of the peak over `mu_b` (`y_p90`) beside what the diamond and the "
            "ellipse allow there, and its share of the way from the one to the other. "
            "`brake_ellipse` is the median share.\n\n" + md_table(band_table, floats="{:.2f}"),
        )
    )
    return rep, fits


__all__ = [
    "CarModel",
    "DEFAULT_ENGINE",
    "FAMILY",
    "POWERTRAINS",
    "PUBLISHED",
    "RHO_REF",
    "air_state",
    "as_families",
    "as_family",
    "bin_power",
    "braking_level",
    "brake_onset",
    "braking_shape",
    "complete_sets",
    "corner_drag",
    "day_levels",
    "density_slope",
    "engine_curve",
    "family_of",
    "fit_car",
    "fit_drag",
    "fit_drag_rpm",
    "gear_sets_by_day",
    "gearing",
    "moist_air",
    "onset_by_day",
    "power_scale",
    "powertrain_of",
    "shift_rpm_by_day",
    "shift_times",
    "solid_bins",
    "specific_power",
    "standard_power",
    "table_clauses",
    "ratio_clusters",
    "trap_speed",
    "with_air",
    "with_gears",
]
