"""Gate C2 through the solver (lap simulation segment C2, issue #205).

:mod:`.car` reads a car from its tables. What needs the solver is here:

- **The convention for ``mu_y``** (:func:`set_convention`). The pace knob
  u absorbs the lateral level, so ``mu_y`` is defined: the value at which
  the median over the car's layouts of the reference lap's u is 1. A
  layout's reference lap is the median over its days of the day's best
  build lap. ``ref_u`` keeps each layout's u (the median over its days)
  and its day sd.
- **The driver's braking use** (:func:`set_use`, before the convention).
  The car's braking limit is the car's; the level the driver brakes at
  and the shape held under cornering are read from the logged
  deceleration of the car's laps, each at its own u, and go into the
  model's ``driver``. The laps are then braked as the driver brakes.
- **In-sample speed** (:func:`lap_scores`): each build lap simulated at
  its own lap time (its own u), the RMS difference of speed at the track's
  nodes, beside the **day floor** (:func:`day_floor`): the same difference
  between two logged laps of the car at the layout, from different days
  and of nearly the same lap time. A simulation cannot be asked to be
  closer to a lap than the car's own laps are to each other.
- **Recovery** (:func:`recover`): laps simulated from a known car, put
  through C1's selection rules and C2's estimator, must give that car
  back. A simulated lap becomes a session by the same definitions the
  logged ones use: 25 Hz, accelerations as 0.5 s differences, full
  throttle where the engine is what limits, brakes on where the tyres
  slow the car.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ...tire_etl.kinematics import G_MS2
from . import car as cm
from . import frame as frm
from . import geometry as geo
from . import limits as lim
from . import samples as smp
from . import solver as sv
from . import track as trk
from .gates import GateItem, GateReport, md_table

logger = logging.getLogger(__name__)

CONVENTION_TOL = 1e-3
CONVENTION_ROUNDS = 6
#: Two laps are of the same pace when their lap times differ by less.
FLOOR_LAP_TIME_TOL = 0.005
FLOOR_PAIRS_PER_LAP = 3
SPEED_OVER_FLOOR_MAX = 0.4
RECOVERY_POWER_MAX = 0.03
RECOVERY_DRAG_MAX = 0.20
RECOVERY_U = (0.97, 1.0, 1.03)
#: The driver's braking use read back from laps simulated with it: the
#: level within this share, the shape within this much.
RECOVERY_USE_MAX = 0.02
RECOVERY_USE_SHAPE_MAX = 0.05
#: Three simulated laps a layout have few braking zones in a band of lateral.
RECOVERY_BAND_MIN_ZONES = 5
SIM_HZ = 25.0
#: The driver's braking use is read on the nodes where the logged lap
#: slows by this much (g) or more (the braking measure's own zone,
#: :data:`~.car_holdout.BRAKE_MATCH_MIN_G`), this long (s) or more into
#: the stop: past the brake's build.
USE_MIN_G = 0.2
USE_BUILD_S = 0.5
#: The bands of lateral use (width, and how many from 0) the level is
#: read in, and the nodes a band needs.
USE_BAND = 0.1
USE_BANDS = 9
USE_BAND_MIN_NODES = 200
#: A lap's u depends on how it is braked, so the reading is repeated at
#: the u the laps have under it: until the level moves by less than
#: ``USE_TOL``, at most ``USE_ROUNDS`` times. Each repeat moves the level
#: by about a ninth of what the one before moved it (0.11–0.13 on the
#: three cars), so a move under 0.015 leaves the reading within 0.002 of
#: where it would settle: two readings from a car braked at its limit.
USE_ROUNDS = 6
USE_TOL = 0.015
#: Deceleration from node speeds is the change of V² over this distance (m).
DECEL_WINDOW_M = 16.0
#: A simulated step is at full throttle when its tyre force is this share
#: of what the engine can give at its speed.
SIM_FULL_SHARE = 0.98
#: The in-sample table's comparison column: cornering drag the other way.
OTHER_COLUMN = "without_corner_drag" if sv.CORNER_DRAG_DEFAULT else "with_corner_drag"


# ----------------------------------------------------- a lap's own settings


@dataclass
class Drive:
    """What turns a lap's keys into the car the solver drives: the model
    with the powertrain the raw car had that day, the driver's shift point
    of the day and the air of the session.

    ``shifts`` maps a date to the day's shift rpm
    (:func:`~.car.shift_rpm_by_day`; a day without one takes the
    powertrain's own median), ``gears`` a date to the gear set fitted
    that day (:func:`~.car.gear_sets_by_day`; a day without one takes the
    powertrain's own) and ``air`` is :func:`~.car.air_state` (a
    session without weather runs in standard air). ``power_scale`` and
    ``drag_scale`` plant a wrong power or drag; ``corner_drag`` is passed
    to :meth:`~.solver.SolverCar.from_model`. ``bank`` is the road's and
    not the car's, and rides here because it is a setting of the lap like
    the others: the laps are driven on the track's bank unless it is
    switched off here (:func:`profile_of`).
    """

    model: cm.CarModel
    air: pd.DataFrame | None = None
    shifts: dict[str, float] = field(default_factory=dict)
    #: ``date -> the gears fitted that day`` (m/s per 1,000 rpm).
    gears: dict[str, tuple[float, ...]] = field(default_factory=dict)
    power_scale: float = 1.0
    drag_scale: float = 1.0
    corner_drag: bool | None = None
    #: Whether the shift times are taken out of the drive (None: the default).
    shift_loss: bool | None = None
    #: Whether the braking limit has the car's measured shape (None: the default).
    brake_shape: bool | None = None
    #: Whether a stop begins with the car's measured coast (None: the default).
    brake_onset: bool | None = None
    #: Whether the laps are driven on the track's bank, where it has one.
    bank: bool = True
    #: Whether the laps are braked as the driver brakes, where the model
    #: carries the driver's use, or at the car's limit.
    brake_use: bool = True
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def car(self, raw_car: str = "", date: str = "", session_id: str = "") -> sv.SolverCar:
        key = (raw_car, date, session_id)
        got: sv.SolverCar | None = self._cache.get(key)
        if got is None:
            engines = self.model.power.get("engines") or {}
            name = cm.powertrain_of(raw_car, date) if raw_car else ""
            cf, rho = self._air_of(session_id)
            base = sv.SolverCar.from_model(
                self.model,
                engine=name if name in engines else None,
                shift_rpm=self.shifts.get(date),
                gears=self.gears.get(date),
                cf=cf,
                rho=rho,
                corner_drag=self.corner_drag,
                shift_loss=self.shift_loss,
                brake_shape=self.brake_shape,
                brake_onset=self.brake_onset,
                brake_use=self.brake_use,
            )
            got = replace(
                base,
                power_scale=base.power_scale * self.power_scale,
                c_drag=base.c_drag * self.drag_scale,
            )
            self._cache[key] = got
        return got

    def _air_of(self, session_id: str) -> tuple[float, float | None]:
        """A session's SAE factor and density (standard air without it),
        from one table read per :class:`Drive`."""
        if "air" not in self._cache:
            air = self.air
            self._cache["air"] = (
                {}
                if air is None
                else dict(
                    zip(air.index, zip(air["cf"].to_numpy(float), air["rho"].to_numpy(float)))
                )
            )
        cf, rho = self._cache["air"].get(session_id, (1.0, None))
        return float(cf), None if rho is None else float(rho)

    def but(self, **changes: object) -> "Drive":
        """The same with some fields changed."""
        kept: tuple[str, ...] = ("model", "air", "shifts", "gears", "power_scale", "drag_scale")
        kept += ("corner_drag", "shift_loss", "brake_shape", "brake_onset", "bank", "brake_use")
        return Drive(**{**{f: getattr(self, f) for f in kept}, **changes})  # type: ignore[arg-type]


Who = Drive | sv.SolverCar | cm.CarModel


def as_drive(who: Who) -> Drive | sv.SolverCar:
    """A model as a :class:`Drive` in standard air; the others as they are."""
    return Drive(who) if isinstance(who, cm.CarModel) else who


def profile_of(who: Drive | sv.SolverCar, track: trk.TrackModel) -> sv.Profile:
    """The track as the laps of ``who`` are driven on it: with its bank,
    unless a :class:`Drive` takes the road as flat."""
    return sv.Profile.from_track(track, bank=who.bank if isinstance(who, Drive) else True)


def _lap_car(who: Drive | sv.SolverCar, row: dict) -> sv.SolverCar:
    if isinstance(who, sv.SolverCar):
        return who
    return who.car(
        str(row.get("raw_car") or ""), str(row.get("date") or ""), str(row.get("session_id") or "")
    )


def _records(frame: pd.DataFrame) -> list[dict]:
    keep = [c for c in ("raw_car", "date", "session_id", "lap_time_s") if c in frame.columns]
    return [dict(zip(keep, vals)) for vals in zip(*(frame[c] for c in keep))]


# ------------------------------------------------------- the reference laps


def day_best_laps(rows: pd.DataFrame, car: str) -> pd.DataFrame:
    """Per layout and day, the best dry build lap of ``car``:
    ``layout_id, date, lap_time_s`` and, where ``rows`` has them, the
    lap's ``session_id`` and ``raw_car``."""
    own = rows[(rows["car"] == car) & (rows["basis"] == "dry")]
    cols = ["layout_id", "date", "lap_time_s"]
    cols += [c for c in ("session_id", "raw_car") if c in own.columns]
    if own.empty:
        return pd.DataFrame(columns=cols)
    best = own.loc[own.groupby(["layout_id", "date"])["lap_time_s"].idxmin(), cols]
    return best.sort_values(["layout_id", "date"]).reset_index(drop=True)


def reference_u(
    who: Who,
    tracks: dict[str, trk.TrackModel],
    best: pd.DataFrame,
    *,
    guess: float = 1.0,
) -> dict[str, dict]:
    """Per layout: the u of each day's best lap, each driven with that
    day's own powertrain, shift point and air (``by_day``); their median
    (``u_ref``, also ``median``), their sd and the median lap time."""
    drive = as_drive(who)
    out: dict[str, dict] = {}
    for layout, grp in best.groupby("layout_id"):
        if layout not in tracks:
            continue
        profile = profile_of(drive, tracks[str(layout)])
        recs = _records(grp)
        got = sv.solve_laps(
            profile,
            [_lap_car(drive, row) for row in recs],
            [float(row["lap_time_s"]) for row in recs],
            guess=guess,
        )
        us: dict[str, float] = {}
        for row, u in zip(recs, got.u):
            us[str(row.get("date", len(us)))] = float(u)
        vals = np.array([v for v in us.values() if np.isfinite(v)])
        median = float(np.median(vals)) if vals.size else float("nan")
        out[str(layout)] = {
            "u_ref": median,
            "lap_time_s": float(grp["lap_time_s"].median()),
            "median": median,
            "day_sd": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
            "days": int(vals.size),
            "by_day": us,
        }
    return out


def _with_mu_y(model: cm.CarModel, mu_y: float, *, hold_use: bool = False) -> cm.CarModel:
    """The model with ``mu_y`` moved. The driver's braking level is a
    share of ``mu_b·u`` and u goes as ``1 / mu_y``, so the share moves
    with ``mu_y``: every lap is then the same lap (the same cornering and
    braking in g, the same lateral use) at ``u`` over the ratio.
    ``hold_use`` leaves the share as it is: for a level that was read at
    the convention (u = 1 on the reference lap) and is to stay there."""
    out = cm.CarModel.from_json(model.to_json())
    new = round(mu_y, 4)
    out.grip = {**model.grip, "mu_y": new}
    use = model.driver.get("brake_use")
    if use is not None and model.grip.get("mu_y") and not hold_use:
        out.driver = {**model.driver, "brake_use": float(use) * new / float(model.grip["mu_y"])}
    return out


# ------------------------------------------------- the driver's braking use


def deceleration_g(v: np.ndarray, ds: float, window_m: float = DECEL_WINDOW_M) -> np.ndarray:
    """Deceleration along a closed lap (g, positive where the car slows)
    from its speed at the track's nodes: the change of ``V²`` over
    ``window_m`` centred on each node, ``−Δ(V²) / (2·Δs)``. The last axis
    is the nodes."""
    half = max(int(round(window_m / 2.0 / ds)), 1)
    v2 = np.asarray(v, float) ** 2
    drop = np.roll(v2, half, axis=-1) - np.roll(v2, -half, axis=-1)
    return np.asarray(drop / (2.0 * 2 * half * ds) / G_MS2)


def time_into_stop(slowing: np.ndarray, v: np.ndarray, ds: float) -> np.ndarray:
    """Seconds since the run of ``slowing`` nodes a node is in began
    (laps × nodes, the lap closed: a run may pass the start line); NaN
    outside a run."""
    slowing = np.asarray(slowing, bool)
    reach = np.cumsum(ds / v, axis=-1) - ds / v  # the time at which each node is reached
    lap = np.sum(ds / v, axis=-1, keepdims=True)
    begins = slowing & ~np.roll(slowing, 1, axis=-1)
    began = np.maximum.accumulate(np.where(begins, reach, -np.inf), axis=-1)
    # a run that was under way at the start line began on the lap before
    began = np.where(np.isfinite(began), began, began[..., -1:] - lap)
    return np.asarray(np.where(slowing & np.isfinite(began), reach - began, np.nan))


@dataclass
class UseLaps:
    """What the driver's braking use is read from at one layout: the laps
    (``session_id, lap_num, date, raw_car, lap_time_s``) and their braking
    nodes past the build (:data:`USE_MIN_G`, :data:`USE_BUILD_S`), in the
    solver's terms less the car's numbers: per node the lap it is of
    (``lap``, a row of ``frame``), the deceleration less the grade's over
    the load (``slowing``), ``V²`` over the load (``v2``: times the lap's
    drag coefficient it is the drag's part) and the tyres' lateral over
    the load (``lat``). ``bank`` is whether the road was taken with its
    bank (:func:`use_laps`)."""

    frame: pd.DataFrame
    lap: np.ndarray
    slowing: np.ndarray
    v2: np.ndarray
    lat: np.ndarray
    bank: bool = True

    def keep(self, rows: np.ndarray) -> "UseLaps":
        """The same with the laps where ``rows`` (a mask over ``frame``)
        is true."""
        rows = np.asarray(rows, bool)
        new = np.cumsum(rows) - 1
        nodes = rows[self.lap]
        return UseLaps(
            self.frame[rows].reset_index(drop=True),
            new[self.lap[nodes]],
            self.slowing[nodes],
            self.v2[nodes],
            self.lat[nodes],
            self.bank,
        )


def use_laps(
    track: trk.TrackModel, frame: pd.DataFrame, v_obs: np.ndarray, *, bank: bool = True
) -> UseLaps:
    """:class:`UseLaps` of a car's laps at a layout (``v_obs`` their
    logged speed at the track's nodes, gaps filled here). The braking
    force is what slows the car less the drag and the grade, as the
    solver's braking law has it, over the load on the tyres; the lateral
    is what the tyres hold on the road's bank (``bank=False``: on a flat
    road, for a lap driven so)."""
    profile = sv.Profile.from_track(track, bank=bank)
    kap, grd, kv, cos, gsb = sv._road(profile)
    if not len(frame):
        none = np.zeros(0)
        return UseLaps(frame.reset_index(drop=True), np.zeros(0, int), none, none, none, bank)
    v = np.vstack([trk._fill_nan_periodic(np.asarray(row, float)) for row in v_obs])
    decel = deceleration_g(v, profile.ds_m)
    since = time_into_stop(decel >= USE_MIN_G, v, profile.ds_m)
    load = G_MS2 * cos + v * v * kv
    lap, node = np.nonzero(since >= USE_BUILD_S)
    at = (lap, node)
    return UseLaps(
        frame.reset_index(drop=True),
        lap,
        ((decel * G_MS2 - G_MS2 * grd) / load)[at],
        (v * v / load)[at],
        (np.abs(v * v * kap - gsb) / load)[at],
        bank,
    )


def read_use(braking: np.ndarray, lateral: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    """The driver's level and held shape from braking nodes: ``braking``
    the braking force over the lap's braking limit (``mu_b·u`` times the
    load), ``lateral`` the lateral over the lap's cornering limit.

    Per band of lateral (:data:`USE_BAND` wide from 0, :data:`USE_BANDS`
    of them, each with :data:`USE_BAND_MIN_NODES` nodes or more) the
    median of each; then ``braking = A·(1 − x) + B·√(1 − x²)`` by least
    squares over the bands, the solver's own form: a level ``A + B`` in a
    straight line and ``B / (A + B)`` of the way from the diamond to the
    ellipse (held to 0..1). Returns ``(level, shape, bands)``; NaN with
    fewer than two bands. The median and not a high percentile: this is
    what the driver does, where :func:`~.car.braking_shape` reads what
    the car can."""
    rows = []
    for k in range(USE_BANDS):
        inside = (lateral >= k * USE_BAND) & (lateral < (k + 1) * USE_BAND)
        if int(inside.sum()) >= USE_BAND_MIN_NODES:
            rows.append(
                {
                    "lateral": float(np.median(lateral[inside])),
                    "braking": float(np.median(braking[inside])),
                    "nodes": int(inside.sum()),
                }
            )
    bands = pd.DataFrame(rows, columns=["lateral", "braking", "nodes"])
    if len(bands) < 2:
        return float("nan"), float("nan"), bands
    x = bands["lateral"].to_numpy()
    basis = np.column_stack([1.0 - x, np.sqrt(1.0 - x * x)])
    (a, b), *_ = np.linalg.lstsq(basis, bands["braking"].to_numpy(), rcond=None)
    level = float(a + b)
    if not level > 0.0:
        return float("nan"), float("nan"), bands
    return level, float(np.clip(b / level, 0.0, 1.0)), bands


def with_use(model: cm.CarModel, use: dict) -> cm.CarModel:
    """The model with its ``driver`` replaced."""
    out = cm.CarModel.from_json(model.to_json())
    out.driver = dict(use)
    return out


def set_use(
    who: Drive | cm.CarModel, tracks: dict[str, trk.TrackModel], laps: dict[str, UseLaps]
) -> tuple[cm.CarModel, dict]:
    """The model with the driver's braking use read from ``laps`` (per
    layout, :func:`use_laps`) into its ``driver``, and the workings.

    Each lap is simulated at its own lap time, which gives its u; its
    braking nodes are then put over the car's limits at that u
    (:func:`read_use`). How a lap is braked moves its u, so the reading is
    repeated with the laps braked as read, until the level holds
    (:data:`USE_TOL`). The level is a share of ``mu_b·u`` at the model's
    ``mu_y``: set the convention after this (:func:`set_convention` moves
    the share with ``mu_y``), not before. Without two bands of nodes the
    ``driver`` is left empty and the laps are braked at the limit.

    The workings: ``rounds`` (level and shape after each), ``bands`` (the
    last reading's), ``laps`` and ``nodes``."""
    drive = who if isinstance(who, Drive) else Drive(who)
    model = with_use(drive.model, {})
    guess: dict[str, np.ndarray] = {}
    use: dict = {}
    rounds: list[tuple[float, float]] = []
    bands = pd.DataFrame()
    count = {"laps": 0, "nodes": 0}
    for _ in range(USE_ROUNDS):
        driven = drive.but(model=with_use(model, use))
        braking, lateral = [], []
        count = {"laps": 0, "nodes": 0}
        for layout, own in laps.items():
            if layout not in tracks or not len(own.frame):
                continue
            recs = _records(own.frame)
            cars = [_lap_car(driven, rec) for rec in recs]
            got = sv.solve_laps(
                sv.Profile.from_track(tracks[layout], bank=own.bank),
                cars,
                [float(rec["lap_time_s"]) for rec in recs],
                guess=guess.get(layout, 1.0),
            )
            guess[layout] = np.where(got.ok, got.u, 1.0)
            good = got.ok[own.lap]
            at = own.lap[good]
            drag = np.array([car.c_drag for car in cars])[at]
            limit_b = (np.array([car.mu_b for car in cars]) * got.u)[at]
            limit_y = (np.array([car.mu_y for car in cars]) * got.u)[at]
            braking.append((own.slowing[good] - drag * own.v2[good]) / limit_b)
            lateral.append(own.lat[good] / limit_y)
            count["laps"] += int(got.ok.sum())
            count["nodes"] += int(good.sum())
        if not braking:
            break
        level, shape, bands = read_use(np.concatenate(braking), np.concatenate(lateral))
        if not np.isfinite(level):
            use = {}
            break
        moved = abs(level - float(use.get("brake_use", 1.0)))
        use = {"brake_use": level, "brake_use_ellipse": shape}
        rounds.append((level, shape))
        if len(rounds) > 1 and moved < USE_TOL:
            break
    return with_use(model, use), {"rounds": rounds, "bands": bands, **count}


def car_use_laps(
    tables: smp.CarTables,
    lap_time: pd.Series,
    car: str,
    tracks: dict[str, trk.TrackModel],
    *,
    bank: bool = True,
) -> dict[str, UseLaps]:
    """:func:`use_laps` of every dry build lap of ``car`` at each of
    ``tracks`` (``lap_time``: seconds by ``(session_id, lap_num)``)."""
    out: dict[str, UseLaps] = {}
    for layout, track in tracks.items():
        frame, v_obs = tables.lap_speeds(car, layout)
        if frame.empty:
            continue
        keys = pd.MultiIndex.from_frame(frame[["session_id", "lap_num"]])
        frame = frame.assign(lap_time_s=lap_time.reindex(keys).to_numpy())
        timed = frame["lap_time_s"].notna().to_numpy()
        out[layout] = use_laps(track, frame[timed], v_obs[timed], bank=bank)
    return out


def recover_use(
    drive: Drive, tracks: dict[str, trk.TrackModel], laps: dict[str, UseLaps]
) -> dict[str, float]:
    """The reading put to its own test: the laps of ``laps`` simulated at
    their lap times by a model that carries a braking use, and the use
    read back from the simulated speeds as it is from the logged ones.
    ``level`` and ``shape`` are the model's, ``level_back`` and
    ``shape_back`` what comes back."""
    planted = drive.model.driver
    sims: dict[str, UseLaps] = {}
    for layout, own in laps.items():
        if layout not in tracks or not len(own.frame):
            continue
        recs = _records(own.frame)
        got = sv.solve_laps(
            sv.Profile.from_track(tracks[layout], bank=own.bank),
            [_lap_car(drive, rec) for rec in recs],
            [float(rec["lap_time_s"]) for rec in recs],
        )
        sims[layout] = use_laps(
            tracks[layout], own.frame[got.ok], got.v_node_ms[got.ok], bank=own.bank
        )
    back = set_use(drive, tracks, sims)[0].driver
    return {
        "level": float(planted.get("brake_use", float("nan"))),
        "level_back": float(back.get("brake_use", float("nan"))),
        "shape": float(planted.get("brake_use_ellipse", float("nan"))),
        "shape_back": float(back.get("brake_use_ellipse", float("nan"))),
    }


def set_convention(
    who: Drive | cm.CarModel,
    tracks: dict[str, trk.TrackModel],
    best: pd.DataFrame,
    *,
    hold_use: bool = False,
) -> tuple[cm.CarModel, dict[str, dict]]:
    """The model with ``mu_y`` moved to where the median over layouts of
    the reference u is 1, and ``ref_u`` filled; and the workings of
    :func:`reference_u` at that value. A layout's reference u is the
    median over its days of the u of the day's best build lap, each lap
    driven with its day's powertrain, shift point and air (those of
    ``who``, a :class:`Drive`; a bare model is driven in standard air with
    its default powertrain). ``mu_b`` is a measurement and stays. A
    driver's braking use in the model moves with ``mu_y`` so that the
    laps stay as they are (:func:`set_use` reads it before the
    convention); ``hold_use`` keeps its share instead, for a use that was
    read under a convention already (a planted wrong car that keeps the
    fitted car's driver)."""
    drive = who if isinstance(who, Drive) else Drive(who)
    model = drive.model
    mu_y = float(model.grip["mu_y"])
    refs: dict[str, dict] = {}
    for _ in range(CONVENTION_ROUNDS):
        moved = _with_mu_y(model, mu_y, hold_use=hold_use)
        refs = reference_u(drive.but(model=moved), tracks, best)
        level = float(np.median([r["u_ref"] for r in refs.values()])) if refs else float("nan")
        if not np.isfinite(level) or abs(level - 1.0) < CONVENTION_TOL:
            break
        mu_y *= level
    out = _with_mu_y(model, mu_y, hold_use=hold_use)
    out.ref_u = {
        k: [
            _rounded(r["median"], 4),
            _rounded(r["day_sd"], 4),
        ]
        for k, r in refs.items()
    }
    out.meta = {
        **model.meta,
        "mu_y_source": "the value at which the median over layouts of the reference lap's u is 1",
        "mu_y_apex_table": model.grip["mu_y"],
    }
    return out, refs


def typical_use(who: Who) -> float:
    """The car's typical use of its cornering limit: the median over the
    layouts it was set on of the use of a day's best lap
    (:func:`set_limit`). 1 for a car set by the convention, where that
    lap's knob is 1 by definition, and for a bare solver's car."""
    model = who.model if isinstance(who, Drive) else who
    if not isinstance(model, cm.CarModel):
        return 1.0
    return float(model.meta.get("use_ref") or 1.0)


def set_limit(
    who: Drive | cm.CarModel,
    tracks: dict[str, trk.TrackModel],
    best: pd.DataFrame,
    *,
    hold: dict | None = None,
) -> tuple[cm.CarModel, dict[str, dict]]:
    """The model with ``mu_y`` at the car's cornering limit
    (``grip.mu_y_limit``, :func:`~.limits.car_limit`), and the workings
    of :func:`reference_u` there.

    The limit is the car's and no pace moves it; the solver's knob u is
    then the share of it the driver uses on a lap (q = −ln u), the same in
    cornering and braking. ``ref_u`` keeps each layout's use (the median
    over its days of the use of the day's best build lap) and
    ``meta.use_ref`` their median over the layouts, the car's *typical
    use*: what is carried to a layout the car has not run.

    A driver's braking use in the model was read at the model's present
    ``mu_y`` and is a share of ``mu_b`` times the use, so it moves with
    ``mu_y`` and every lap stays the lap it was (:func:`_with_mu_y`).
    ``hold`` is for a planted wrong car that keeps another model's driver
    (that model's ``driver``, as this function left it): the braking
    level in g at the typical use is held at that driver's, whatever the
    wrong car's typical use turns out to be.

    A model without a limit (no corner supports the statistic) is set by
    the convention instead (:func:`set_convention`)."""
    drive = who if isinstance(who, Drive) else Drive(who)
    model = drive.model
    limit = model.grip.get("mu_y_limit")
    if limit is None or not np.isfinite(limit):
        if hold:
            kept = {k: v for k, v in hold.items() if k != "use_ref"}
            drive = drive.but(model=with_use(model, kept))
        return set_convention(drive, tracks, best, hold_use=bool(hold))
    refs: dict[str, dict] = {}

    def _typical(of: cm.CarModel) -> float:
        nonlocal refs
        refs = reference_u(drive.but(model=of), tracks, best)
        return float(np.median([r["u_ref"] for r in refs.values()])) if refs else float("nan")

    if hold and hold.get("brake_use") is not None:
        level = float(hold["brake_use"]) * float(hold.get("use_ref") or 1.0)
        out = with_use(_with_mu_y(model, float(limit), hold_use=True), dict(hold))
        ref = float("nan")
        for _ in range(CONVENTION_ROUNDS):
            ref = _typical(out)
            if not np.isfinite(ref) or ref <= 0.0:
                break
            share = level / ref
            settled = abs(share / float(out.driver["brake_use"]) - 1.0) < CONVENTION_TOL
            out.driver = {**out.driver, "brake_use": share}
            if settled:
                break
    else:
        out = _with_mu_y(model, float(limit))
        ref = _typical(out)
    out.ref_u = {k: [_rounded(r["median"], 4), _rounded(r["day_sd"], 4)] for k, r in refs.items()}
    out.meta = {
        **model.meta,
        "mu_y_source": "the car's cornering limit: the expected best of m passes (limits.car_limit)",
        "mu_y_apex_table": model.grip["mu_y"],
        "use_ref": _rounded(ref, 5),
    }
    if out.driver:
        out.driver = {**out.driver, "use_ref": _rounded(ref, 5)}
    return out, refs


def _rounded(v: float, nd: int) -> float | None:
    return None if not np.isfinite(v) else round(float(v), nd)


# ------------------------------------------------------------ in-sample speed


def _rms(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def lap_scores(
    who: Who, track: trk.TrackModel, laps: pd.DataFrame, v_obs: np.ndarray
) -> pd.DataFrame:
    """Each lap of ``laps`` (``lap_time_s`` beside its keys; ``v_obs`` its
    speed at the track's nodes) simulated at its own lap time, with its
    day's powertrain, shift point and air when ``who`` is a
    :class:`Drive`: ``use`` (the lap's use of the car's limit, the
    solver's knob), ``u`` (that over the car's typical use,
    :func:`typical_use`: 1 is the car's reference pace), ``status`` and
    ``speed_rmse`` (m/s over the nodes)."""
    drive = as_drive(who)
    profile = profile_of(drive, track)
    typical = typical_use(drive)
    recs = _records(laps)
    got = sv.solve_laps(
        profile,
        [_lap_car(drive, row) for row in recs],
        [float(row["lap_time_s"]) for row in recs],
    )
    rows = [
        {
            "u": float(u) / typical,
            "use": float(u),
            "status": status,
            "speed_rmse": _rms(got.v_node_ms[i] - v_obs[i]) if status == "ok" else float("nan"),
        }
        for i, (u, status) in enumerate(zip(got.u, got.status))
    ]
    return pd.concat([laps.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def day_floor(laps: pd.DataFrame, v_obs: np.ndarray) -> dict:
    """How far apart two logged laps are: over pairs of laps from different
    days whose lap times agree within :data:`FLOOR_LAP_TIME_TOL` (each lap
    with its :data:`FLOOR_PAIRS_PER_LAP` nearest in lap time), the median
    RMS difference of speed at the nodes. ``floor`` is NaN without a pair."""
    t = laps["lap_time_s"].to_numpy(float)
    day = laps["date"].to_numpy()
    seen: set[tuple[int, int]] = set()
    vals: list[float] = []
    for i in range(len(t)):
        other = np.flatnonzero((day != day[i]) & (np.abs(t / t[i] - 1.0) <= FLOOR_LAP_TIME_TOL))
        for j in other[np.argsort(np.abs(t[other] - t[i]))][:FLOOR_PAIRS_PER_LAP]:
            pair = (min(i, int(j)), max(i, int(j)))
            if pair in seen:
                continue
            seen.add(pair)
            vals.append(_rms(v_obs[i] - v_obs[j]))
    return {"floor": float(np.median(vals)) if vals else float("nan"), "pairs": len(vals)}


# ------------------------------------------------------------------ recovery


def signals_from_sim(
    sim: sv.SimLap, profile: sv.Profile, car: sv.SolverCar
) -> tuple[smp.Signals, np.ndarray, np.ndarray, np.ndarray]:
    """A simulated lap as a logged session would look: ``(signals, s,
    grade, lap_of)`` for :mod:`.samples`.

    Three laps end to end at :data:`SIM_HZ`, the middle one the lap (so
    that a rule looking 0.8 s back sees what came before the line).
    Accelerations are those of :func:`~.frame.ground_series`: differences
    over 0.5 s. A car driven through gears has its ``gear`` and ``rpm``
    channels. The throttle is full where the tyre force is the engine's
    (:data:`SIM_FULL_SHARE` of what it gives at that speed and lateral g)
    and closed elsewhere; the brakes
    are on, in proportion, where the tyres slow the car.
    """
    n = len(sim.dt_s)
    lap = sim.lap_time_s
    length = profile.length_m
    t_node = np.concatenate([[0.0], np.cumsum(sim.dt_s)])
    t3 = np.concatenate([t_node[:-1] - lap, t_node[:-1], t_node + lap])
    v3 = np.concatenate([sim.v_node_ms, sim.v_node_ms, sim.v_node_ms, sim.v_node_ms[:1]])
    s3 = np.concatenate(
        [
            np.arange(n) * profile.ds_m - length,
            np.arange(n) * profile.ds_m,
            np.arange(n + 1) * profile.ds_m + length,
        ]
    )
    t = np.arange(-lap + 1.0, 2.0 * lap - 1.0, 1.0 / SIM_HZ)
    h = frm.GROUND_SMOOTH_S / 2.0
    speed = np.interp(t, t3, v3)
    s_abs = np.interp(t, t3, s3)
    s = np.mod(s_abs, length)
    long_g = (np.interp(t + h, t3, v3) - np.interp(t - h, t3, v3)) / (2.0 * h) / G_MS2
    # the course is the integral of κ along the lap; its difference over 0.5 s
    psi3 = np.concatenate([[0.0], np.cumsum(np.tile(profile.kappa, 3) * profile.ds_m)])
    psi_s = np.arange(3 * n + 1) * profile.ds_m - length

    def _psi(at: np.ndarray) -> np.ndarray:
        return np.asarray(np.interp(np.interp(at, t3, s3), psi_s, psi3))

    lat_g = speed * (_psi(t + h) - _psi(t - h)) / (2.0 * h) / G_MS2
    step = np.clip(np.searchsorted(t_node, np.mod(t, lap), side="right") - 1, 0, n - 1)
    tyre = sim.long_tyre_g[step] * G_MS2
    engine = car.power_at(sim.v_ms[step]) / sim.v_ms[step]
    engine = engine - car.k_corner_drag * sim.lat_g[step] ** 2 * G_MS2
    full = tyre >= SIM_FULL_SHARE * engine
    brake = np.maximum(-sim.long_tyre_g[step], 0.0)
    nan = np.full(len(t), np.nan)
    geom = geo.SessionGeometry(
        session_id="sim",
        layout_id="",
        t=t,
        x=np.zeros(len(t)),
        y=np.zeros(len(t)),
        alt=np.zeros(len(t)),
        lap_num=np.ones(len(t), dtype=np.int64),
        seg=np.zeros(len(t), dtype=np.int64),
        speed=speed,
        course=np.zeros(len(t)),
        lat_g_doppler=nan.copy(),
        lat_g_position=nan.copy(),
        discard=np.zeros(len(t), dtype=bool),
        lag=geo.LagFit(tau_s=0.0),
        crossings=np.array([0.0, lap]),
    )
    hardest = float(brake.max()) if brake.max() > 0 else 1.0
    info = {"throttle_channel": "sim", "brake_channel": "sim", "brake_on_at": 0.05 * hardest}
    sig = smp.Signals(
        geom=geom,
        lat_g=lat_g,
        long_g=long_g,
        throttle=full.astype(float),
        brake=brake,
        brake_on=brake >= info["brake_on_at"],
        gear=nan.copy(),
        rpm=nan.copy(),
        lat_acc=nan.copy(),
        info=info,
    )
    if car.geared:  # the channels a logged car with an ECU would have
        gear, rpm = car.gear_and_rpm(speed)
        sig.gear = np.asarray(car.gear_numbers, float)[gear]
        sig.rpm = rpm
    grade = np.interp(s, np.arange(n) * profile.ds_m, profile.grade, period=length)
    lap_of = np.where((t >= 0.0) & (t < lap), 1, -1).astype(np.int64)
    return sig, s, grade, lap_of


def simulated_tables(
    car: sv.SolverCar, tracks: dict[str, trk.TrackModel], name: str = "sim"
) -> smp.CarTables:
    """C1's tables from laps of ``car`` simulated on ``tracks``: one lap
    per layout and per u of :data:`RECOVERY_U`, each as a day of its own."""
    parts: dict[str, list[pd.DataFrame]] = {"throttle": [], "coast": [], "brake": [], "apex": []}
    for layout, track in tracks.items():
        profile = sv.Profile.from_track(track)
        for k, u in enumerate(RECOVERY_U):
            sim = sv.simulate_lap(profile, car, u)
            sig, s, grade, lap_of = signals_from_sim(sim, profile, car)
            sig.lat_tyre = smp.in_tyres_frame(sig, track, s)
            got = {
                "throttle": smp.throttle_samples(sig, s, grade, lap_of),
                "coast": smp.coast_samples(sig, s, grade, lap_of),
                "brake": smp.brake_zones(sig, s, grade, lap_of),
                "apex": smp.apex_rows(sig, track, s, lap_of),
            }
            for key, tbl in got.items():
                tbl.insert(0, "session_id", f"sim_{layout}_{k}")
                tbl.insert(1, "car", name)
                tbl.insert(2, "date", f"2000-01-{k + 1:02d}")
                tbl.insert(3, "layout_id", layout)
                tbl.insert(4, "basis", "dry")
                parts[key].append(tbl)
    frames = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in parts.items()}
    return smp.CarTables(
        sessions=pd.DataFrame({"car": [name]}),
        throttle=frames["throttle"],
        coast=frames["coast"],
        brake=frames["brake"],
        apex=frames["apex"],
    )


def recover(car: sv.SolverCar, tracks: dict[str, trk.TrackModel]) -> dict:
    """Simulate ``car`` on ``tracks``, apply C1 and C2 to the laps and
    compare: power and ``c_drag`` returned over the truth, the layouts
    whose simulated straight table reached 50 m/s, and the returned
    braking peak and lateral level over the truth (reported: a 0.5 s
    difference rounds a peak off).

    For a car driven through gears the power compared is the curve's, the
    median over its rpm bins of returned over true, and the gears and the
    shift point come back too (``gear_ratio``: the largest departure of a
    returned gear from the true one; ``shift_rpm`` beside ``shift_true``).
    """
    # The estimator is what is tested: the laps are driven without the time
    # lost in the shifts, which the solver spreads over a band of speed and
    # a logged car loses in the shift itself.
    car = replace(car, shift_s=())
    tables = simulated_tables(car, tracks)
    model, work = cm.fit_car(tables, "sim")
    spec = model.power["engines"][model.power["default_engine"]]
    p, c = spec["p_specific"], model.power["c_drag"]
    p_ratio = float(p) / car.p_specific if p is not None else float("nan")
    gear_ratio, shift = float("nan"), float("nan")
    if car.geared and spec["curve"]:
        centre = np.array([r for r, _ in spec["curve"]]) + sv.CURVE_BIN_RPM / 2.0
        inside = (centre >= car.curve_rpm[0]) & (centre <= car.curve_rpm[-1])
        true = np.interp(centre, car.curve_rpm, car.curve_p)
        got = np.array([v for _, v in spec["curve"]])
        if inside.any():
            p_ratio = float(np.median((got / true)[inside]))
        gears = spec["gears"] or {}
        off = [
            gears[str(n)] / k - 1.0 for n, k in zip(car.gear_numbers, car.gears) if str(n) in gears
        ]
        gear_ratio = float(np.max(np.abs(off))) if off else float("nan")
        shift = float(spec["shift_rpm"]) if spec["shift_rpm"] is not None else float("nan")
    return {
        "p_ratio": p_ratio,
        "c_ratio": float(c) / car.c_drag if c is not None and car.c_drag > 0 else float("nan"),
        "drag_layouts": sorted(work["drag"]["by_layout"]),
        "mu_b_ratio": float(model.grip["mu_b"]) / car.mu_b if model.grip["mu_b"] else float("nan"),
        "mu_y_ratio": float(model.grip["mu_y"]) / car.mu_y if model.grip["mu_y"] else float("nan"),
        "k_corner_drag": model.power["k_corner_drag"],
        "k_true": car.k_corner_drag,
        "brake_ellipse": _rounded(
            cm.braking_shape(
                tables.brake[tables.brake["basis"] == "dry"],
                float(c) if c is not None else float("nan"),
                float(model.grip["mu_b"] or float("nan")),
                float(model.meta.get("mu_y_apex_table") or model.grip["mu_y"] or float("nan")),
                min_zones=RECOVERY_BAND_MIN_ZONES,
            )["brake_ellipse"],
            3,
        ),
        "ellipse_true": car.brake_ellipse,
        "onset_s": model.grip.get("onset_s"),
        "onset_true": car.onset_s,
        "gear_off": gear_ratio,
        "shift_rpm": shift,
        "shift_true": car.shift_rpm,
        "samples": int(len(tables.throttle)),
    }


# ------------------------------------------- the limit's statistic on a plant

#: The plant of the cornering limit's statistic (clause 5.4 of the driver
#: committee's design): days and laps a day, the tops of a corner's
#: passes, and the rule it is held to.
PLANT_DAYS = 12
PLANT_LAPS = 10
PLANT_TOPS = ("bell", "capped", "mistakes")
#: A capped top: the driver's typical use sits this close under the limit.
PLANT_CAPPED_USE = 0.97
#: Slow-side mistakes: this share of passes, this many spreads down.
PLANT_MISTAKE_SHARE = 0.02
PLANT_MISTAKE_SDS = 3.0
#: The level read must be within this of the expected best of m of the
#: planted passes, and no more than this above the planted limit.
PLANT_MAX_OFF = 0.01


def _smooth_noise(rng: np.random.Generator, n: int, sd: float) -> np.ndarray:
    """Noise of standard deviation ``sd`` as a series smoothed over
    :data:`~.frame.GROUND_SMOOTH_S` at :data:`SIM_HZ` has it: neighbouring
    fixes share most of it, so the largest of a corner's fixes is raised
    by less than white noise would raise it."""
    if sd <= 0.0:
        return np.zeros(n)
    width = max(int(round(frm.GROUND_SMOOTH_S * SIM_HZ)), 1)
    raw = rng.normal(0.0, 1.0, n + width - 1)
    out = np.convolve(raw, np.ones(width) / width, mode="valid")
    return np.asarray(out / out.std() * sd)


def plant_limit(
    car: sv.SolverCar,
    track: trk.TrackModel,
    *,
    top: str = "bell",
    use: float = 0.90,
    sd_pass: float = 0.04,
    sd_shared: float = 0.0,
    noise_g: float = 0.03,
    accel_scale: float = 0.93,
    days: int = PLANT_DAYS,
    laps: int = PLANT_LAPS,
    m: int = lim.M_DEFAULT,
    seed: int = 0,
) -> dict:
    """The cornering limit's statistic put to a car whose limit is known.

    ``car`` (its ``mu_y`` the planted limit) is driven round ``track`` for
    ``days`` × ``laps`` laps. On each lap every corner is taken at its own
    use of the limit, drawn about ``use`` with the spread ``sd_pass`` (in
    logs): the lap is driven at the limit on the track with that corner's
    curvature divided by its use, which is the same lap. ``top`` is the
    shape of the top of a corner's passes: ``bell`` (as drawn), ``capped``
    (drawn about :data:`PLANT_CAPPED_USE` and cut off at the limit) or
    ``mistakes`` (``bell`` with :data:`PLANT_MISTAKE_SHARE` of the passes
    :data:`PLANT_MISTAKE_SDS` spreads slower). Each lap goes through the
    package's own table builder (:func:`signals_from_sim`,
    :func:`~.samples.apex_rows`) twice: as driven, and with noise of
    ``noise_g`` added to each of the two readings' series *before* the
    corner's largest value is taken, the accelerometer reading
    ``accel_scale`` of what the tyres hold. ``sd_shared`` is an error the
    two readings share, one per corner and day (the line, a kerb).

    Returns the planted ``limit``; ``truth``, the expected best of m of
    the planted passes (limit × use) by the statistic's own rule;
    ``clean``, the statistic on the tables without noise (what the table
    builder alone does to it); ``raw`` and ``level``, the statistic on the
    noisy tables before and after its noise correction; ``cross``, the
    same with each sensor read where the other peaks; ``sd_noise``, the
    noise it found in the two-reading mean; and ``corners`` and
    ``passes``."""
    if top not in PLANT_TOPS:
        raise ValueError(f"top must be one of {PLANT_TOPS}")
    rng = np.random.default_rng(seed)
    profile = sv.Profile.from_track(track)
    n = profile.n
    s_node = np.arange(n) * profile.ds_m
    spans = []
    for corner in track.corners:
        a, b = float(corner["s_in"]), float(corner["s_out"])
        spans.append((s_node >= a) & (s_node <= b) if a <= b else (s_node >= a) | (s_node <= b))
    k_corners = len(spans)
    centre = np.log(PLANT_CAPPED_USE if top == "capped" else use)
    shared = rng.normal(0.0, sd_shared, (days, k_corners)) if sd_shared > 0.0 else None
    rows: dict[str, list[pd.DataFrame]] = {"clean": [], "noisy": []}
    uses = []
    for day in range(days):
        for lap in range(laps):
            logw = centre + rng.normal(0.0, sd_pass, k_corners)
            if top == "capped":
                logw = np.minimum(logw, 0.0)
            elif top == "mistakes":
                logw -= PLANT_MISTAKE_SDS * sd_pass * (rng.random(k_corners) < PLANT_MISTAKE_SHARE)
            w = np.exp(logw)
            kappa = profile.kappa.copy()
            for k, span in enumerate(spans):
                kappa[span] = profile.kappa[span] / w[k]
            sim = sv.simulate_lap(replace(profile, kappa=kappa), car, 1.0)
            sig, s, _, lap_of = signals_from_sim(sim, profile, car)
            tyre = smp.in_tyres_frame(sig, track, s)
            for name in ("clean", "noisy"):
                with_noise = name == "noisy"
                sd = noise_g if with_noise else 0.0
                one = replace(sig, lat_g=sig.lat_g + _smooth_noise(rng, len(s), sd))
                one.lat_acc = accel_scale * tyre + _smooth_noise(rng, len(s), sd)
                got = smp.apex_rows(one, track, s, lap_of)
                if with_noise and shared is not None:
                    scale = np.exp(shared[day, got["corner"].to_numpy()])
                    for col in (
                        "lat_doppler_g",
                        "lat_accel_g",
                        "lat_tyre_g",
                        "lat_tyre_cross_g",
                        "lat_accel_cross_g",
                    ):
                        got[col] = got[col] * scale
                got.insert(0, "session_id", f"plant_{day}")
                got.insert(1, "car", "plant")
                got.insert(2, "date", f"2000-01-{day + 1:02d}")
                got.insert(3, "layout_id", track.layout_key)
                got.insert(4, "basis", "dry")
                got["lap_num"] = day * laps + lap
                rows[name].append(got)
            uses.append(w)
    noisy_rows = pd.concat(rows["noisy"], ignore_index=True)
    read = lim.car_limit(noisy_rows, m)
    crossed = lim.car_limit(noisy_rows, m, cross=True)
    bare = lim.car_limit(pd.concat(rows["clean"], ignore_index=True), m, correct=False)
    own = read.by_corner[read.by_corner["supports"]] if read.found else pd.DataFrame()
    w_all = np.vstack(uses)
    truth = float("nan")
    if len(own):
        truth = float(
            np.median([lim.expected_best(car.mu_y * w_all[:, int(k)], m) for k in own["corner"]])
        )
    return {
        "limit": float(car.mu_y),
        "truth": truth,
        "clean": bare.raw,
        "raw": read.raw,
        "level": read.level,
        "cross": crossed.level,
        "sd_noise": float(own["sd_noise"].median()) if len(own) else float("nan"),
        "sd": float(own["sd"].median()) if len(own) else float("nan"),
        "corners": read.corners,
        "passes": read.passes,
    }


#: The plant in the gate: the error the two readings share, by day and
#: corner (the whole of the committee's day-and-corner spread taken as
#: shared error: an upper size).
PLANT_SHARED_SD = 0.025


def limit_rows(car: str, apex: pd.DataFrame, layouts: list[str]) -> list[dict]:
    """The car's cornering limit at each m of :data:`~.limits.M_REPORTED`
    for the gate's table: the level with its day bootstrap, what it
    stands on, the layouts that borrow it, and beside it the same passes
    read where the other sensor peaks (``cross``) and on the Doppler
    alone (``doppler``)."""
    rows = []
    for m in lim.M_REPORTED:
        got = lim.car_limit(apex, m, boot=lim.BOOT_RESAMPLES)
        cross = lim.car_limit(apex, m, cross=True)
        alone = lim.car_limit(apex.assign(lat_accel_g=np.nan), m)
        rows.append(
            {
                "car": car,
                "m": m,
                "level": got.level,
                "p05": got.lo,
                "p95": got.hi,
                "as_read": got.raw,
                "corners": got.corners,
                "passes": got.passes,
                "one_sensor": got.one_sensor,
                "stands_on": ", ".join(got.layouts),
                "borrowed_by": ", ".join(k for k in layouts if k not in got.layouts),
                "cross": cross.level,
                "doppler": alone.level,
            }
        )
    return rows


def plant_rows(
    car_name: str, model: cm.CarModel, limit: lim.Limit, tracks: dict[str, trk.TrackModel]
) -> list[dict]:
    """:func:`plant_limit` for a car as the logs have it: its limit
    planted, on the layout most of its supporting corners are at, the
    passes' typical use and spread as read from its own corners, and the
    sensors' noise set so that the statistic finds in the plant the noise
    it finds in the logs. One row per top, without and with an error the
    two readings share (:data:`PLANT_SHARED_SD`)."""
    own = limit.by_corner[limit.by_corner["supports"]]
    if not limit.found or limit.one_sensor or own.empty:
        return []
    layout = str(own["layout_id"].value_counts().idxmax())
    if layout not in tracks:
        return []
    at = own[own["layout_id"] == layout]
    noise_logs = float(at["sd_noise"].median())
    sd_pass = float(np.sqrt(max(float(at["sd"].median()) ** 2 - noise_logs**2, 0.0)))
    # a corner's typical pass over its expected best: that far under the level
    use = float(np.exp(-lim.best_of_normal(limit.m) * sd_pass))
    car = replace(
        sv.SolverCar.from_model(model, brake_use=False), mu_y=float(limit.level), shift_s=()
    )
    noise_g = noise_logs * float(np.sqrt(2.0)) * float(limit.level) * use
    for _ in range(2):  # the noise found goes with the noise planted
        found = plant_limit(car, tracks[layout], use=use, sd_pass=sd_pass, noise_g=noise_g)
        if np.isfinite(found["sd_noise"]) and found["sd_noise"] > 0.0:
            noise_g *= noise_logs / found["sd_noise"]
    rows = []
    for top in PLANT_TOPS:
        for shared in (0.0, PLANT_SHARED_SD):
            got = plant_limit(
                car,
                tracks[layout],
                top=top,
                use=use,
                sd_pass=sd_pass,
                sd_shared=shared,
                noise_g=noise_g,
            )
            rows.append(
                {
                    "car": car_name,
                    "layout_id": layout,
                    "top": top,
                    "shared_sd": shared,
                    "limit": got["limit"],
                    "truth": got["truth"],
                    "no_noise": got["clean"] / got["truth"] - 1.0,
                    "as_read": got["raw"] / got["truth"] - 1.0,
                    "level": got["level"] / got["truth"] - 1.0,
                    "cross": got["cross"] / got["truth"] - 1.0,
                    "level_over_limit": got["level"] / got["limit"] - 1.0,
                    "cross_over_limit": got["cross"] / got["limit"] - 1.0,
                    "noise_logs": noise_logs,
                    "noise_found": got["sd_noise"],
                    "noise_g": noise_g,
                }
            )
    return rows


# ------------------------------------------------------------------ the gate


def day_shifts(work: dict) -> dict[str, float]:
    """``date -> shift rpm`` over every powertrain of a car, from the
    workings of :func:`~.car.fit_car`."""
    out: dict[str, float] = {}
    for eng in work.get("engines", {}).values():
        for date, rpm in eng["shifts"].items():
            out[str(date)] = float(rpm)
    return out


def day_gears(work: dict) -> dict[str, tuple[float, ...]]:
    """``date -> the gears fitted that day``, lowest first, over every
    powertrain of a car, from the workings of :func:`~.car.fit_car`."""
    out: dict[str, tuple[float, ...]] = {}
    for eng in work.get("engines", {}).values():
        for date, gears in eng.get("gear_sets", {}).items():
            out[str(date)] = tuple(sorted(float(v) for v in gears.values()))
    return out


def day_compounds(root: Path, rows: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    """``(car, layout_id, date) ->`` the tyre compound(s) the car's build
    sessions of that day are labelled with (production's labels,
    :func:`~..compounds.load_compound_labels`; joined by `` / `` where a
    day has more than one, absent where it has none)."""
    try:
        from ..compounds import load_compound_labels

        labels = load_compound_labels(Path(root))
    except Exception:  # noqa: BLE001 - a dataset without labels has an era ladder without them
        return {}
    if labels.empty or "session_id" not in rows.columns:
        return {}
    of = dict(zip(labels["session_id"], labels["compound"]))
    seen = rows[["car", "layout_id", "date", "session_id"]].drop_duplicates()
    seen = seen.assign(compound=seen["session_id"].map(of)).dropna(subset=["compound"])
    return {
        (str(c), str(k), str(d)): " / ".join(sorted(set(g["compound"])))
        for (c, k, d), g in seen.groupby(["car", "layout_id", "date"])
    }


def era_ladder(
    car: str,
    best: pd.DataFrame,
    refs: dict[str, dict],
    days: pd.DataFrame,
    compounds: dict[tuple[str, str, str], str],
) -> pd.DataFrame:
    """A car's days in order: ``date, layout_id, engine`` (the powertrain
    the raw car had that day), ``compound``, the day's best build lap
    (``best_lap_s``), its pace knob (``u``, from ``refs`` of
    :func:`set_convention`), the day's full-throttle level over its
    powertrain's at equal rpm (``level``) and the driver's shift point
    (``shift_rpm``), the last two from the fit's ``days``. What changed on
    a car, or in how it was driven, shows as a step in a column."""
    cols = ["car", "date", "layout_id", "engine", "compound", "best_lap_s", "u", "level"]
    cols += ["shift_rpm"]
    by_day = (
        {(str(r["date"]), str(r["layout_id"])): r for r in days.to_dict("records")}
        if len(days)
        else {}
    )
    rows: list[dict] = []
    for rec in best.to_dict("records"):
        date, layout = str(rec["date"]), str(rec["layout_id"])
        u = refs.get(layout, {}).get("by_day", {}).get(date)
        if u is None:
            continue
        day = by_day.get((date, layout), {})
        rows.append(
            {
                "car": car,
                "date": date,
                "layout_id": layout,
                "engine": cm.powertrain_of(str(rec.get("raw_car") or car), date),
                "compound": compounds.get((car, layout, date), ""),
                "best_lap_s": float(rec["lap_time_s"]),
                "u": float(u),
                "level": float(day.get("level", float("nan"))),
                "shift_rpm": float(day.get("shift_rpm", float("nan"))),
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["date", "layout_id"]).reset_index(drop=True)


def class_lap_rows(
    tables: smp.CarTables, rows: pd.DataFrame, models: dict[str, cm.CarModel]
) -> pd.DataFrame:
    """Another driver's lap in a car's class (:data:`~.car.CLASS_LAPS`)
    beside what the car's own dry build laps hold at the same layout: the
    best lap, the lateral level of the apex table (the median over the
    grip-limited corners of the corner's 90th percentile, ground frame)
    and the 90th percentile of the braking zones' peaks, with the model's
    ``mu_y`` and ``mu_b``."""
    out: list[dict] = []
    level = smp.lateral_level(smp.apex_summary(tables.apex)) if len(tables.apex) else None
    for car, lap in cm.CLASS_LAPS.items():
        if car not in models:
            continue
        layout = lap["layout_id"]
        own = rows[(rows["car"] == car) & (rows["layout_id"] == layout) & (rows["basis"] == "dry")]
        if own.empty:
            continue
        lat = float("nan")
        if level is not None and len(level):
            got = level[
                (level["car"] == car) & (level["layout_id"] == layout) & (level["basis"] == "dry")
            ]
            lat = float(got["level_g"].iloc[0]) if len(got) else float("nan")
        zones = tables.brake
        zones = zones[
            (zones["car"] == car) & (zones["layout_id"] == layout) & (zones["basis"] == "dry")
        ]
        out.append(
            {
                "car": car,
                "layout_id": layout,
                "class_lap": lap["what"],
                "class_lap_s": lap["lap_s"],
                "best_lap_s": float(own["lap_time_s"].min()),
                "class_lat_lo_g": lap["lat_g"][0],
                "class_lat_hi_g": lap["lat_g"][1],
                "apex_level_g": lat,
                "mu_y": float(models[car].grip["mu_y"]),
                "class_brake_g": lap["brake_g"],
                "brake_p90_g": float(zones["decel_p95_g"].quantile(0.9)) if len(zones) else np.nan,
                "mu_b": float(models[car].grip["mu_b"]),
            }
        )
    return pd.DataFrame(out)


def check_car_fit(
    root: Path,
    *,
    cache_dir: Path | None = None,
    tracks_dir: Path | None = None,
    cars_dir: Path | None = None,
) -> GateReport:
    """Gate C2: every model car's numbers from its tables
    (:func:`.car.table_clauses`), then through the solver: the convention
    for ``mu_y`` with ``ref_u``, the recovery of a known car, and the
    in-sample speed RMSE beside the day floor. Every logged lap is driven
    with its day's powertrain, shift point and air (:class:`Drive`)."""
    tables = cm.as_families(smp.car_tables(root, cache_dir=cache_dir, tracks_dir=tracks_dir))
    rows = cm.as_family(smp.build_laps(geo.clean_lap_index(root, cache_dir=cache_dir)))
    rep, fits = cm.table_clauses(root, tables)
    air = cm.air_state(root, geo.load_sessions(root))
    cars = [c for c in smp.MODEL_CARS if c in fits]
    layouts = sorted(str(k) for k in rows[rows["car"].isin(cars)]["layout_id"].unique())
    tracks = {
        k: smp.load_track(root, k, tracks_dir=tracks_dir, cache_dir=cache_dir) for k in layouts
    }
    lap_time = rows.set_index(["session_id", "lap_num"])["lap_time_s"]

    models: dict[str, cm.CarModel] = {}
    ref_rows, rec_rows, score_rows, use_rows = [], [], [], []
    lim_rows: list[dict] = []
    planted: list[dict] = []
    use_bands: dict[str, pd.DataFrame] = {}
    ladders: list[pd.DataFrame] = []
    compounds = day_compounds(root, rows)
    for car_name in cars:
        best = day_best_laps(rows, car_name)
        own = {k: v for k, v in tracks.items() if k in set(best["layout_id"])}
        fitted, work = fits[car_name]
        drive = Drive(fitted, air=air, shifts=day_shifts(work), gears=day_gears(work))
        # how the driver brakes, then the car's limit under it: the knob is
        # from here on the driver's use of that limit
        braked = car_use_laps(tables, lap_time, car_name, own)
        used, use_work = set_use(drive, own, braked)
        model, uses = set_limit(drive.but(model=used), own, best)
        drive = drive.but(model=model)
        typical = typical_use(model)
        # the pace against the car's typical use, as the convention had it
        refs = {
            k: {
                **r,
                "use_ref": r["u_ref"],
                "u_ref": r["u_ref"] / typical,
                "by_day": {d: v / typical for d, v in r["by_day"].items()},
            }
            for k, r in uses.items()
        }
        models[car_name] = model
        if model.driver:
            back = recover_use(drive, own, braked)
            use_bands[car_name] = use_work["bands"]
            use_rows.append(
                {
                    "car": car_name,
                    "laps": use_work["laps"],
                    "nodes": use_work["nodes"],
                    "rounds": len(use_work["rounds"]),
                    "brake_use": model.driver["brake_use"],
                    "level_g": model.driver["brake_use"] * float(model.grip["mu_b"]),
                    "held_shape": model.driver["brake_use_ellipse"],
                    "limit_shape": float(model.grip.get("brake_ellipse") or 0.0),
                    "use_back": back["level_back"] / back["level"],
                    "shape_back": back["shape_back"],
                }
            )
        if cars_dir is not None:
            model.save(Path(cars_dir) / f"{car_name.replace(' ', '_')}.json")
        ladders.append(era_ladder(car_name, best, refs, work["days"], compounds))
        lim_rows += limit_rows(car_name, tables.apex[tables.apex["car"] == car_name], sorted(own))
        planted += plant_rows(car_name, model, work["limit"], own)
        for layout, r in refs.items():
            ref_rows.append(
                {
                    "car": car_name,
                    "layout_id": layout,
                    "days": r["days"],
                    "ref_lap_s": r["lap_time_s"],
                    "use": r["use_ref"],
                    "q": -float(np.log(r["use_ref"])) if r["use_ref"] > 0 else float("nan"),
                    "u_ref": r["u_ref"],
                    "u_day_sd": r["day_sd"] / typical,
                }
            )
        for name in model.power["engines"]:
            if model.power["engines"][name]["p_specific"] is None:
                continue
            # a known car is one driven at its limits
            got = recover(sv.SolverCar.from_model(model, engine=name, brake_use=False), own)
            rec_rows.append(
                {
                    "car": car_name,
                    "engine": name,
                    **got,
                    "drag_layouts": ", ".join(got["drag_layouts"]),
                }
            )
        flipped = drive.but(corner_drag=not sv.CORNER_DRAG_DEFAULT)
        for layout, track in own.items():
            laps, v_obs = tables.lap_speeds(car_name, layout)
            if laps.empty:
                continue
            keys = pd.MultiIndex.from_frame(laps[["session_id", "lap_num"]])
            laps = laps.assign(lap_time_s=lap_time.reindex(keys).to_numpy())
            ok = laps["lap_time_s"].notna().to_numpy()
            laps, v_obs = laps[ok].reset_index(drop=True), v_obs[ok]
            scores = lap_scores(drive, track, laps, v_obs)
            other = lap_scores(flipped, track, laps, v_obs)
            floor = day_floor(laps, v_obs)
            score_rows.append(
                {
                    "car": car_name,
                    "layout_id": layout,
                    "laps": int(len(scores)),
                    "days": int(scores["date"].nunique()),
                    "speed_rmse": float(scores["speed_rmse"].median()),
                    "day_floor": floor["floor"],
                    "pairs": floor["pairs"],
                    "over_floor": float(scores["speed_rmse"].median()) - floor["floor"],
                    OTHER_COLUMN: float(other["speed_rmse"].median()),
                    "u_p05": float(scores["u"].quantile(0.05)),
                    "u_median": float(scores["u"].median()),
                    "u_p95": float(scores["u"].quantile(0.95)),
                    "no_root": int((scores["status"] != "ok").sum()),
                }
            )

    # The convention and ref_u.
    ref = pd.DataFrame(ref_rows)
    mu = pd.DataFrame(
        [
            {
                "car": c,
                "mu_y": m.grip["mu_y"],
                "apex_table_level": m.meta.get("mu_y_apex_table"),
                "typical_use": typical_use(m),
                "q_typical": -float(np.log(typical_use(m))),
                "at_typical_use": float(m.grip["mu_y"]) * typical_use(m),
                "mu_b": m.grip["mu_b"],
                "brake_use": m.driver.get("brake_use"),
                "braking_at_typical_use": (
                    float(m.driver["brake_use"]) * typical_use(m)
                    if m.driver.get("brake_use") is not None
                    else float("nan")
                ),
            }
            for c, m in models.items()
        ]
    )
    spread = (
        ref.groupby("car")["u_ref"].agg(lambda x: x.max() / x.min() - 1.0)
        if len(ref)
        else pd.Series(dtype=float)
    )
    rep.items.append(
        GateItem(
            "`mu_y` is the car's limit; the driver's use of it per layout",
            None,
            "; ".join(
                f"{r.car} limit {r.mu_y:.3f}, typical use {r.typical_use:.3f} (q "
                f"{r.q_typical:.3f}), u between layouts "
                f"{spread.get(r.car, float('nan')):.1%}"
                for r in mu.itertuples()
            ),
            "recorded (#312: the pace knob is the driver's use of a limit that pace does not "
            "move; owner, 2026-11-22)",
            "`mu_y`: the car's cornering limit (`grip.mu_y_limit`, the item above); the "
            "solver's knob is the share of it a lap uses, in cornering and braking alike, and "
            "q = −ln of that. `typical_use`: the median over the car's layouts of the use of a "
            "day's best build lap, which is what is carried to a layout the car has not run; "
            "`at_typical_use`: the limit times it, the number the convention for `mu_y` used "
            "to hold. `brake_use`: the driver's braking level as a share of `mu_b` at a use of "
            "1 (it was read as a share of `mu_b` times the knob, so it rose as `mu_y` rose to "
            "the limit and every lap stayed the lap it was); `braking_at_typical_use`: that "
            "share at the typical use. A share near 1 leaves a lap faster than the typical "
            "one no room to brake harder before it reaches the car's `mu_b`.\n\n"
            + md_table(mu, floats="{:.3f}")
            + "\n\n`use`: the median over the layout's days of the use of the day's best "
            "build lap, each driven with its day's powertrain, shift point and air, and `q` "
            "its −ln; `u_ref`: that over the car's typical use (1 is the car's reference "
            "pace, as under the convention); `u_day_sd`: the sd over the days; `ref_lap_s`: "
            "the median of those lap times.\n\n" + md_table(ref, floats="{:.4f}"),
        )
    )

    # The car's cornering limit as its best moments, and the statistic on a plant.
    limits = pd.DataFrame(lim_rows)
    if len(limits):
        main = limits[limits["m"] == lim.M_DEFAULT]
        rep.items.append(
            GateItem(
                "The car's cornering limit: the expected best of m passes through a corner",
                None,
                "; ".join(
                    f"{r.car} {r.level:.3f} [{r.p05:.3f}, {r.p95:.3f}] on {r.corners} corners"
                    + (" (the Doppler alone)" if r.one_sensor else "")
                    for r in main.itertuples()
                ),
                f"reported (#312; m = {lim.M_DEFAULT} by the owner's decision of 2026-11-23): "
                "the best the car has been shown to do, a floor under its true limit",
                "Per corner, every dry day pooled, the expected largest of m passes of the "
                "lateral the tyres hold over the load on them, each pass read by the Doppler and "
                "by the accelerometer (put on the Doppler's scale per session) and taken as "
                "their geometric mean, with the independent noise of the two taken out; then the "
                f"median over the grip-limited corners with {lim.SUPPORT_FACTOR}·m passes or "
                "more (`.limits`). `level`: that; `p05`, `p95`: the ends of a bootstrap over the "
                "car's days; `as_read`: before the noise is taken out; `stands_on`: the layouts "
                "with a supporting corner, and `borrowed_by` those that take the car's level "
                "without one. `cross`: the same statistic with each sensor read where the "
                "*other* one peaks, which its own noise has no part in choosing (the plant below "
                "says what each reading is worth); `doppler`: on the Doppler alone, uncorrected, "
                "as a car or a fold without two readings is read. The model keeps `level` at "
                f"m = {lim.M_DEFAULT} as `grip.mu_y_limit`; the solver does not use it yet.\n\n"
                + md_table(limits, floats="{:.3f}"),
            )
        )
    plants = pd.DataFrame(planted)
    if len(plants):
        bare = plants[plants["shared_sd"] == 0.0]
        off = bare.groupby("car")["level"].agg(lambda x: float(x.abs().max()))
        over = bare.groupby("car")["level_over_limit"].max()
        cross_off = bare.groupby("car")["cross"].agg(lambda x: float(x.abs().max()))
        plant_ok = bool((off <= PLANT_MAX_OFF).all() and (over <= PLANT_MAX_OFF).all())
        rep.items.append(
            GateItem(
                "The limit's statistic on a planted car (the committee's clause 5.4)",
                plant_ok,
                "; ".join(
                    f"{c} off by up to {off[c]:.1%} (read where the other sensor peaks: "
                    f"{cross_off[c]:.1%})"
                    for c in off.index
                ),
                f"within {PLANT_MAX_OFF:.0%} of the expected best of m of the planted passes, "
                f"and no more than {PLANT_MAX_OFF:.0%} above the planted limit, on every top "
                "without a shared error (the rule written on #312 before the run)",
                "Each car's limit planted on the layout most of its supporting corners are at: "
                f"{PLANT_DAYS} days of {PLANT_LAPS} laps, every corner pass at its own use of "
                "the limit (the lap is driven at the limit on the track with that corner's "
                "curvature divided by its use), through the package's own table builder, with "
                "noise added to both readings' series before the corner's largest value is "
                "taken. The passes' spread is the car's own; the noise is set so that the "
                "statistic finds in the plant what it finds in the logs (`noise_found` against "
                "`noise_logs`; `noise_g` is what that takes per reading). Tops: `bell` (the "
                f"passes drawn about a typical use), `capped` (drawn about {PLANT_CAPPED_USE} "
                f"and cut off at the limit), `mistakes` (`bell` with {PLANT_MISTAKE_SHARE:.0%} "
                f"of the passes {PLANT_MISTAKE_SDS:g} spreads slower). `shared_sd`: an error the "
                "two readings share by day and corner, which no reading can remove. Against "
                "`truth` (the expected best of m of the planted passes): `no_noise`, what the "
                "table builder alone does; `as_read` and `level`, the statistic before and "
                "after its noise correction; `cross`, the same with each sensor read where the "
                "other peaks. `*_over_limit` are against the planted limit itself. The "
                "correction takes out the spread the noise adds; it cannot take out what the "
                "noise adds to the *largest* value of a corner's fixes, which is why `level` "
                "reads high.\n\n" + md_table(plants, floats="{:.4f}"),
            )
        )

    # The era ladder: each day's pace level, in order.
    ladder = pd.concat(ladders, ignore_index=True) if ladders else pd.DataFrame()
    if len(ladder):
        rep.items.append(
            GateItem(
                "The era ladder: each day's pace level",
                None,
                "; ".join(
                    f"{c} {len(g)} days, u {g['u'].min():.3f} to {g['u'].max():.3f}"
                    for c, g in ladder.groupby("car", sort=False)
                ),
                "reported (issue #205)",
                "A car's days in order. `u`: the pace knob of the day's best build lap, driven "
                "with that day's powertrain, shift point and air (the same laps `ref_u` is the "
                "median of); `level`: the day's full-throttle level over its powertrain's at "
                "equal rpm, in standard air; `shift_rpm`: the driver's shift point that day; "
                "`compound`: the tyre the day's sessions are labelled with, where they are. A "
                "change to the car or to how it is driven shows as a step in a column: the "
                "engine in `engine` and `level`, a shift light in `shift_rpm`, tyres in "
                "`compound` and `u`. `u` holds the day's grip and the day's driving together; "
                "the logs cannot split them.\n\n" + md_table(ladder, floats="{:.3f}"),
            )
        )

    # Another driver's lap in the class.
    outside = class_lap_rows(tables, rows, models)
    if len(outside):
        rep.items.append(
            GateItem(
                "Against another driver's lap in the class",
                None,
                "; ".join(
                    f"{r['car']} at {r['layout_id']}: lap {float(r['best_lap_s']):.1f} s "
                    f"({float(r['class_lap_s']):.1f}), lateral {float(r['apex_level_g']):.2f} g "
                    f"({float(r['class_lat_lo_g']):.2f}–{float(r['class_lat_hi_g']):.2f}), "
                    f"braking {float(r['brake_p90_g']):.2f} g ({float(r['class_brake_g']):.2f})"
                    for r in outside.to_dict("records")
                ),
                "reported (issue #205); the other driver's figure in brackets",
                "An outside figure, never fitted to: one published dry reference lap of a car "
                "of the same class at the same layout (`class_lap`), with its lap time, the "
                "95th percentile of its lateral over 0.5 s and its braking, beside this car's "
                "best dry build lap, the lateral level of its apex table (`apex_level_g`: "
                "ground frame, on the car's own line, so the road's bank is in it) and the "
                "90th percentile of its braking zones' peaks (`brake_p90_g`: drag not taken "
                "out), with the model's `mu_y` (a convention) and `mu_b`. The frames differ, so "
                "this is a band to be near, not a number to match.\n\n"
                + md_table(outside, floats="{:.3f}"),
            )
        )

    # The driver's braking use.
    use = pd.DataFrame(use_rows)
    if len(use):
        back_ok = bool(
            ((use["use_back"] - 1.0).abs() <= RECOVERY_USE_MAX).all()
            and ((use["shape_back"] - use["held_shape"]).abs() <= RECOVERY_USE_SHAPE_MAX).all()
        )
        rep.items.append(
            GateItem(
                "The driver's braking use",
                back_ok,
                "; ".join(
                    f"{r['car']} {float(r['brake_use']):.2f} of mu_b "
                    f"({float(r['level_g']):.2f} g at u = 1), shape "
                    f"{float(r['held_shape']):.2f} (the limit's {float(r['limit_shape']):.2f}); "
                    f"read back {float(r['use_back']) - 1:+.1%}, {float(r['shape_back']):.2f}"
                    for r in use.to_dict("records")
                ),
                f"recorded; read back from laps simulated with it within {RECOVERY_USE_MAX:.0%} "
                f"of the level and {RECOVERY_USE_SHAPE_MAX} of the shape",
                "The car's braking limit (`mu_b`, `brake_ellipse`) is the car's. The laps are "
                "braked as the driver brakes: at `brake_use` of `mu_b·u`, with the shape held "
                "under cornering in place of the limit's (`driver` in the car's JSON; issue "
                "#312). Read from the logged speed of every dry build lap, each at its own u: "
                f"the nodes that slow by {USE_MIN_G} g or more, {USE_BUILD_S} s or more into the "
                "stop; the braking force (what slows the car less drag and grade) over "
                "`mu_b·u` times the load against the lateral over `mu_y·u` times the load; the "
                f"median of each per band of lateral {USE_BAND} wide; then `A·(1 − x) + "
                "B·√(1 − x²)` by least squares over the bands, the level `A + B` and the shape "
                "`B / (A + B)`. Repeated at the u the laps have when braked so (`rounds`), then "
                "the convention is set under it. `use_back`, `shape_back`: the same reading on "
                "the same laps simulated with it.\n\n"
                + md_table(use, floats="{:.3f}")
                + "".join(
                    f"\n\n{c}, the last reading's bands (before the convention):\n\n"
                    + md_table(b, floats="{:.3f}")
                    for c, b in use_bands.items()
                ),
            )
        )

    # Recovery.
    rec = pd.DataFrame(rec_rows)
    p_ok = bool(((rec["p_ratio"] - 1.0).abs() <= RECOVERY_POWER_MAX).all()) if len(rec) else None
    with_drag = rec[rec["drag_layouts"] != ""] if len(rec) else rec
    c_ok = (
        bool(((with_drag["c_ratio"] - 1.0).abs() <= RECOVERY_DRAG_MAX).all())
        if len(with_drag)
        else None
    )
    rep.items.append(
        GateItem(
            "Recovery on simulated laps",
            None if p_ok is None else bool(p_ok and c_ok is not False),
            "; ".join(
                f"{c} ({e}) power {p - 1:+.1%}, drag {d - 1:+.1%}"
                for c, e, p, d in zip(rec["car"], rec["engine"], rec["p_ratio"], rec["c_ratio"])
            ),
            f"power within {RECOVERY_POWER_MAX:.0%} (the curve against rpm where the car has "
            f"one); `c_drag` within {RECOVERY_DRAG_MAX:.0%} where a simulated layout reaches "
            "50 m/s",
            "Each powertrain simulated on the car's layouts at u = "
            + ", ".join(str(u) for u in RECOVERY_U)
            + ", in standard air at its own shift point, the laps put through C1's selection "
            "and C2's estimator (returned over true). `gear_off`: the largest departure of a "
            "returned gear from the true one; `shift_rpm` reads under `shift_true` by "
            "construction (the samples next to a gear change are not in the table). No speed "
            "slope of grip is estimated, so none can be found.\n\n"
            + md_table(rec, floats="{:.3f}"),
        )
    )

    # In-sample speed.
    score = pd.DataFrame(score_rows)
    scored = score[score["day_floor"].notna()] if len(score) else score
    rep.items.append(
        GateItem(
            "In-sample speed RMSE at each lap's own u",
            bool((scored["over_floor"] <= SPEED_OVER_FLOOR_MAX).all()) if len(scored) else None,
            "; ".join(
                f"{r.car} {r.layout_id} {r.speed_rmse:.2f} (floor {r.day_floor:.2f})"
                for r in scored.itertuples()
            )
            + " m/s",
            f"median over laps ≤ the cell's day floor + {SPEED_OVER_FLOOR_MAX} m/s",
            "`day_floor`: the same RMS difference between two logged laps from different days "
            f"within {FLOOR_LAP_TIME_TOL:.1%} of lap time. A cell without such a pair is listed "
            f"and not scored. `{OTHER_COLUMN}`: the same with `k_corner_drag` the other way "
            "round, for comparison.\n\n" + md_table(score, floats="{:.3f}"),
        )
    )
    return rep


__all__ = [
    "check_car_fit",
    "day_best_laps",
    "Drive",
    "UseLaps",
    "car_use_laps",
    "day_floor",
    "day_gears",
    "day_shifts",
    "lap_scores",
    "class_lap_rows",
    "day_compounds",
    "deceleration_g",
    "era_ladder",
    "read_use",
    "recover",
    "recover_use",
    "reference_u",
    "set_convention",
    "set_use",
    "time_into_stop",
    "use_laps",
    "with_use",
    "signals_from_sim",
    "simulated_tables",
]
