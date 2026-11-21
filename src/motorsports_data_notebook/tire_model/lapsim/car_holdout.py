"""Does a car's model predict days and layouts it never saw, and can the
gate tell it from a wrong car's? (lap simulation segment C3, issue #205)

**Folds.** Leave-one-day-out per layout, and leave-one-layout-out: the
car is read from its tables without the held-out laps (:func:`fold_car`),
``mu_y`` set by the convention on what is left, and every held-out build
lap is simulated at its own lap time, its own u, with the powertrain the
car had that day, the driver's shift point of the day and the session's
air (:class:`~.car_gate.Drive`). The shift point of a held-out day comes
from that day's own logs: it is the driver's, like the lap time.

**Statistics**, per lap, a fold reading the median over its laps:

- ``S1``: RMS difference of speed over the track's nodes;
- ``S2``: signed end-of-straight speed error, the mean over the layout's
  straights of the simulated top speed less the logged one;
- ``S3``: signed apex speed error, the mean over the corners of the
  simulated minimum less the logged one.

**Controls** through the same folds: (a) the fold's own car with
``p_specific × 0.85`` and ``mu_y`` raised by the convention until the
reference laps are matched again; (b) another car's parameter set. A lap
time alone cannot tell either from the car, because u is solved from it.

**Verdict.** A model is *inside* a fold when ``S1`` is within the fold's
day floor + 0.4 m/s and ``|S2|`` and ``|S3|`` within 1.0 m/s. A fold in
which a control is inside has no power against that control: it is listed
and not counted; so is a fold without a day floor (a layout the car ran
on one day). A car that is outside on any powered fold fails. One that
is not passes when it has a powered day fold and a powered layout fold
for each control, and is otherwise *not scored*, which is not a pass.

The layout's track is the one built from every car's laps, the held-out
ones included; a track without them is B3's subject.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..heat_input import ShareParams
from . import car as cm
from . import car_gate as cg
from . import geometry as geo
from . import holdout as ho
from . import samples as smp
from . import solver as sv
from . import track as trk
from .gates import GateItem, GateReport, md_table

logger = logging.getLogger(__name__)

S1_OVER_FLOOR_MAX = cg.SPEED_OVER_FLOOR_MAX
S2_MAX = 1.0
S3_MAX = 1.0
POWER_CONTROL = 0.85
#: The car whose parameters stand in as "another car".
OTHER_CAR = {"Inferno 86": "KK-SII", "KK-SII": "Inferno 86", "KK-F": "Inferno 86"}
#: A straight is a stretch between two apexes this long on which the car
#: gains this much speed over the faster of the two apexes.
STRAIGHT_MIN_M = 200.0
STRAIGHT_MIN_GAIN_MS = 5.0
APEX_WINDOW_M = 20.0
MIN_FOLD_LAPS = 3
TOP_BAND_QUANTILE = 0.75
DRAG_GRID = (0.0, 0.5, 1.0, 1.5)
TOP_BAND_MAX = 0.3
MODELS = ("fit", "low_power", "other_car")


# -------------------------------------------------------------- the layout


@dataclass
class Layout:
    """What the statistics need of a track: the nodes around each apex and
    the stretches between consecutive apexes."""

    track: trk.TrackModel
    apex_nodes: list[np.ndarray]
    stretches: list[np.ndarray]  # node indices from one apex to the next, in driving order

    @classmethod
    def of(cls, track: trk.TrackModel) -> "Layout":
        n, ds = track.n, track.ds_m
        apex = sorted(int(round(c["s_apex"] / ds)) % n for c in track.corners)
        half = max(int(round(APEX_WINDOW_M / ds)), 1)
        windows = [np.arange(a - half, a + half + 1) % n for a in apex]
        stretches = []
        for a, b in zip(apex, apex[1:] + apex[:1]):
            length = (b - a) % n or n
            stretches.append(np.arange(a, a + length + 1) % n)
        return cls(track=track, apex_nodes=windows, stretches=stretches if apex else [])


def lap_stats(layout: Layout, v_sim: np.ndarray, v_obs: np.ndarray) -> dict:
    """``S1, S2, S3`` of one lap (module docstring) and
    ``top``: the RMS difference over the nodes where the logged lap is in
    its own top quarter of speed."""
    ds = layout.track.ds_m
    ok = np.isfinite(v_obs)
    out = {"S1": cg._rms((v_sim - v_obs)[ok]), "S2": np.nan, "S3": np.nan}
    band = ok & (v_obs >= np.nanquantile(v_obs, TOP_BAND_QUANTILE))
    out["top"] = cg._rms((v_sim - v_obs)[band])
    if layout.apex_nodes:
        out["S3"] = float(
            np.mean([np.min(v_sim[w]) - np.nanmin(v_obs[w]) for w in layout.apex_nodes])
        )
    tops: list[float] = []
    for nodes in layout.stretches:
        if (len(nodes) - 1) * ds < STRAIGHT_MIN_M:
            continue
        obs = v_obs[nodes]
        if np.nanmax(obs) - max(obs[0], obs[-1]) < STRAIGHT_MIN_GAIN_MS:
            continue
        tops.append(float(np.max(v_sim[nodes]) - np.nanmax(obs)))
    if tops:
        out["S2"] = float(np.mean(tops))
    return out


def score_laps(who: cg.Who, layout: Layout, laps: pd.DataFrame, v_obs: np.ndarray) -> pd.DataFrame:
    """:func:`lap_stats` of every lap of ``laps`` (``session_id, lap_num,
    lap_time_s``; ``v_obs`` its speed at the nodes) simulated at its own
    lap time, with ``u``: by one car, or by a :class:`~.car_gate.Drive`
    with each lap's own powertrain, shift point and air. A lap whose time
    the car cannot do has NaN statistics."""
    drive = cg.as_drive(who)
    profile = cg.profile_of(drive, layout.track)
    recs = cg._records(laps)
    got = sv.solve_laps(
        profile,
        [cg._lap_car(drive, rec) for rec in recs],
        [float(rec["lap_time_s"]) for rec in recs],
    )
    rows = []
    for i in range(len(recs)):
        row = {"u": float(got.u[i]), "S1": np.nan, "S2": np.nan, "S3": np.nan, "top": np.nan}
        if got.ok[i]:
            row.update(lap_stats(layout, got.v_node_ms[i], v_obs[i]))
        rows.append(row)
    return pd.DataFrame(rows)


def heat_errors(
    who: cg.Who,
    layout: Layout,
    laps: pd.DataFrame,
    v_obs: np.ndarray,
    share: ShareParams,
    best: pd.DataFrame,
) -> pd.DataFrame:
    """Per lap of ``laps``, the log of simulated over logged lap-mean heat
    per tyre through production's share model
    (:func:`~.holdout.tyre_heat`), the lap driven two ways:

    - ``h_<tyre>``: at its own lap time (u solved from it);
    - ``hc_<tyre>``: with the pace level carried. ``best`` is the layout's
      day-best laps; their median u under ``who`` is the layout's
      reference, and the lap is driven at its own u over that. Where the
      car was set without this layout that takes the layout's reference
      lap as u = 1, which is all a car can be told of a layout it has not
      run, and leaves the lap only its pace against the reference.

    Both sides are on the track's curvature, each at its own speed, and
    in the tyres' frame where the laps are driven on the track's bank
    (the ground frame on a flat road). A lap the car cannot do is NaN."""
    drive = cg.as_drive(who)
    track = layout.track
    profile = cg.profile_of(drive, track)
    recs = cg._records(laps)
    cars = [cg._lap_car(drive, rec) for rec in recs]
    got = sv.solve_laps(profile, cars, [float(rec["lap_time_s"]) for rec in recs])
    ref = cg.reference_u(drive, {track.layout_key: track}, best).get(track.layout_key, {})
    u_ref = float(ref.get("u_ref", float("nan")))
    carried = sv.drive_laps(profile, cars, np.where(got.ok, got.u / u_ref, 1.0))
    out = np.full((len(recs), 2 * len(ho.CORNERS)), np.nan)
    for i in range(len(recs)):
        if not got.ok[i]:
            continue
        logged = ho.tyre_heat(
            share,
            profile.kappa,
            trk._fill_nan_periodic(np.asarray(v_obs[i], float)),
            profile.ds_m,
            profile.bank,
        )
        for k, sim in enumerate((got.v_node_ms[i], carried.v_node_ms[i])):
            if np.isfinite(sim).all():
                heat = ho.tyre_heat(share, profile.kappa, sim, profile.ds_m, profile.bank)
                out[i, k * len(ho.CORNERS) : (k + 1) * len(ho.CORNERS)] = np.log(heat / logged)
    cols = [f"{kind}_{c}" for kind in ("h", "hc") for c in ho.CORNERS]
    return pd.DataFrame(out, columns=cols)


def worst_tyre(laps: pd.DataFrame, kind: str, tag: str = "") -> float:
    """The largest over the tyres of the absolute median over ``laps`` of
    :func:`heat_errors`' ``kind`` (``h`` or ``hc``; ``tag`` the variant's
    suffix); NaN where the laps carry none."""
    cols = [f"{kind}_{c}{tag}" for c in ho.CORNERS]
    if not set(cols) <= set(laps.columns):
        return float("nan")
    med = laps[cols].median().abs()
    return float(med.max()) if med.notna().any() else float("nan")


# ---------------------------------------------------------------- the folds


def car_only(tables: smp.CarTables, car: str) -> smp.CarTables:
    """``tables`` with the rows of ``car`` alone (speed traces left as
    they are: they are looked up by row)."""

    def _own(tbl: pd.DataFrame) -> pd.DataFrame:
        return tbl[tbl["car"] == car] if len(tbl) and "car" in tbl.columns else tbl

    return smp.CarTables(
        sessions=_own(tables.sessions),
        throttle=_own(tables.throttle),
        coast=_own(tables.coast),
        brake=_own(tables.brake),
        apex=_own(tables.apex),
        speed_laps=_own(tables.speed_laps),
        speed=tables.speed,
    )


def without(tables: smp.CarTables, layout_id: str, date: str | None = None) -> smp.CarTables:
    """``tables`` without the rows of ``layout_id`` (on ``date`` only, when
    one is given)."""

    def _keep(tbl: pd.DataFrame) -> pd.DataFrame:
        if tbl.empty or "layout_id" not in tbl.columns:
            return tbl
        out = tbl["layout_id"] == layout_id
        if date is not None:
            out &= tbl["date"] == date
        return tbl[~out]

    return smp.CarTables(
        sessions=tables.sessions,
        throttle=_keep(tables.throttle),
        coast=_keep(tables.coast),
        brake=_keep(tables.brake),
        apex=_keep(tables.apex),
        speed_laps=tables.speed_laps,
        speed=tables.speed,
    )


def fold_car(
    tables: smp.CarTables,
    car: str,
    tracks: dict[str, trk.TrackModel],
    best: pd.DataFrame,
    *,
    air: pd.DataFrame | None = None,
    shifts: dict[str, float] | None = None,
    gears: dict[str, tuple[float, ...]] | None = None,
    power_scale: float = 1.0,
) -> cg.Drive | None:
    """The car read from ``tables`` (already without the held-out laps)
    with ``mu_y`` by the convention on ``best`` (the day-best laps that
    are left), as a :class:`~.car_gate.Drive` with the days' shift points
    and air; ``power_scale`` plants a wrong power before the convention is
    set. None when the tables give no power, drag or braking."""
    model = cm.fit_car(tables, car, air=air)[0]
    return convene(
        model, tracks, best, air=air, shifts=shifts, gears=gears, power_scale=power_scale
    )


def convene(
    model: cm.CarModel,
    tracks: dict[str, trk.TrackModel],
    best: pd.DataFrame,
    *,
    air: pd.DataFrame | None = None,
    shifts: dict[str, float] | None = None,
    gears: dict[str, tuple[float, ...]] | None = None,
    power_scale: float = 1.0,
    bank: bool = True,
) -> cg.Drive | None:
    """``model`` (as :func:`~.car.fit_car` returns it) as a
    :class:`~.car_gate.Drive` with its power scaled and ``mu_y`` then set
    by the convention on ``best``; None when the model lacks power, drag
    or braking, or no reference lap is left. ``bank=False`` has its
    laps, the reference laps among them, driven on a flat road."""
    need = (
        model.power["p_specific"],
        model.power["c_drag"],
        model.grip["mu_b"],
        model.grip["mu_y"],
    )
    if any(v is None for v in need) or best.empty:
        return None
    drive = cg.Drive(
        model,
        air=air,
        shifts=dict(shifts or {}),
        gears=dict(gears or {}),
        power_scale=power_scale,
        bank=bank,
    )
    left = {k: v for k, v in tracks.items() if k in set(best["layout_id"])}
    return drive.but(model=cg.set_convention(drive, left, best)[0])


def _pair_floor(
    held: pd.DataFrame, v_held: np.ndarray, rest: pd.DataFrame, v_rest: np.ndarray
) -> float:
    """The median RMS difference of speed between each held-out lap and
    the laps of other days within :data:`~.car_gate.FLOOR_LAP_TIME_TOL` of
    its lap time (its nearest three); NaN without a pair."""
    t_rest = rest["lap_time_s"].to_numpy(float)
    vals: list[float] = []
    for i, t in enumerate(held["lap_time_s"].to_numpy(float)):
        near = np.flatnonzero(np.abs(t_rest / t - 1.0) <= cg.FLOOR_LAP_TIME_TOL)
        for j in near[np.argsort(np.abs(t_rest[near] - t))][: cg.FLOOR_PAIRS_PER_LAP]:
            vals.append(cg._rms(v_held[i] - v_rest[j]))
    return float(np.median(vals)) if vals else float("nan")


def run_folds(
    tables: smp.CarTables,
    rows: pd.DataFrame,
    tracks: dict[str, trk.TrackModel],
    car: str,
    other: cg.Drive | cm.CarModel | None,
    *,
    air: pd.DataFrame | None = None,
    shifts: dict[str, float] | None = None,
    gears: dict[str, tuple[float, ...]] | None = None,
    shares: dict[str, ShareParams] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every fold of ``car``: ``(folds, laps)``.

    ``tables`` and ``rows`` are by family (:func:`~.car.as_families`). A
    held-out lap is driven with its own day's powertrain, shift point and
    gear set (``shifts`` and ``gears``, read from the logs of that day:
    the driver's and the setup's, like the lap time) and air; ``other`` is the car whose parameters stand in as
    another car's.

    ``folds`` has one row per fold and model (``fit``, ``low_power``,
    ``other_car``): ``kind`` (``day`` or ``layout``), ``layout_id``,
    ``date``, ``laps``, ``floor`` and the medians ``S1, S2, S3``.
    ``laps`` has the per-lap statistics of the fitted car with its
    variants on the layout folds: ``S1_other`` and ``top_other`` (the
    same car with its cornering drag the other way round: off when the
    car carries it, on when it does not), ``S1_flat``, ``S3_flat``,
    ``top_flat`` and ``u_flat`` (the fold's car with every road taken as
    flat, ``mu_y`` set again by the convention there, so that ``u``
    against ``u_flat`` says how the pace level carries to the held-out
    layout each way) and, on a layout that reaches 50 m/s, ``top_<k>``
    (the top-quarter RMS at k × the model's drag). With ``shares`` (the
    production force shares per car, :func:`~.holdout.production_shares`)
    the layout folds also carry :func:`heat_errors`, on the bank and with
    the suffix ``_flat`` without it.
    """
    own = car_only(tables, car)
    best_all = cg.day_best_laps(rows, car)
    lap_time = rows.set_index(["session_id", "lap_num"])["lap_time_s"]
    layouts = {k: Layout.of(v) for k, v in tracks.items() if k in set(best_all["layout_id"])}
    other_car = cg.Drive(other, air=air) if isinstance(other, cm.CarModel) else other
    any_bank = any(len(t.bank) for t in tracks.values())
    share = None
    if shares is not None:
        try:
            share = ho.share_for(car, shares)
        except KeyError:
            logger.info("holdout: no force shares for %s, its heat is not scored", car)
    fold_rows: list[dict] = []
    lap_rows: list[pd.DataFrame] = []
    for layout_id, layout in layouts.items():
        cell, v_cell = own.lap_speeds(car, layout_id)
        if cell.empty:
            continue
        keys = pd.MultiIndex.from_frame(cell[["session_id", "lap_num"]])
        cell = cell.assign(lap_time_s=lap_time.reindex(keys).to_numpy())
        cell_floor = cg.day_floor(cell, v_cell)["floor"]
        days = sorted(cell["date"].unique())
        plans: list[tuple[str, str | None]] = [("layout", None)]
        if len(days) > 1:
            plans += [("day", d) for d in days]
        for kind, date in plans:
            held_mask = (cell["date"] == date).to_numpy() if date else np.ones(len(cell), bool)
            held = cell[held_mask].reset_index(drop=True)
            v_held = v_cell[held_mask]
            if len(held) < MIN_FOLD_LAPS:
                continue
            best = best_all[
                ~(
                    (best_all["layout_id"] == layout_id)
                    & ((best_all["date"] == date) if date else True)
                )
            ]
            raw = cm.fit_car(without(own, layout_id, date), car, air=air)[0]
            fitted = convene(raw, tracks, best, air=air, shifts=shifts, gears=gears)
            if fitted is None:
                logger.info("holdout: no car for %s without %s %s", car, layout_id, date or "")
                continue
            low = convene(
                raw, tracks, best, air=air, shifts=shifts, gears=gears, power_scale=POWER_CONTROL
            )
            models: dict[str, cg.Drive | None] = {
                "fit": fitted,
                "low_power": low,
                "other_car": other_car,
            }
            floor = cell_floor
            if date:
                rest = cell[~held_mask].reset_index(drop=True)
                own_floor = _pair_floor(held, v_held, rest, v_cell[~held_mask])
                floor = own_floor if np.isfinite(own_floor) else cell_floor
            base = {
                "car": car,
                "kind": kind,
                "layout_id": layout_id,
                "date": date or "",
                "laps": int(len(held)),
                "floor": floor,
            }
            for name, driven in models.items():
                if driven is None:
                    continue
                scores = score_laps(driven, layout, held, v_held)
                fold_rows.append(
                    {
                        **base,
                        "model": name,
                        **{k: float(scores[k].median()) for k in ("S1", "S2", "S3")},
                        "u": float(scores["u"].median()),
                        "mu_y": driven.model.grip["mu_y"],
                        "p_specific": driven.model.power["p_specific"] * driven.power_scale,
                        "c_drag": driven.model.power["c_drag"],
                    }
                )
                if name != "fit":
                    continue
                per_lap = pd.concat([held[["session_id", "lap_num", "date"]], scores], axis=1)
                per_lap.insert(0, "car", car)
                per_lap.insert(1, "kind", kind)
                per_lap.insert(2, "layout_id", layout_id)
                if kind == "layout":
                    flipped = driven.but(corner_drag=not sv.CORNER_DRAG_DEFAULT)
                    variant = score_laps(flipped, layout, held, v_held)
                    per_lap["S1_other"] = variant["S1"]
                    per_lap["top_other"] = variant["top"]
                    # and with the shift times the other way round
                    shifted = score_laps(
                        driven.but(shift_loss=not sv.SHIFT_LOSS_DEFAULT), layout, held, v_held
                    )
                    per_lap["S1_shift"] = shifted["S1"]
                    per_lap["top_shift"] = shifted["top"]
                    # and with the braking limit's shape the other way round
                    braked = score_laps(
                        driven.but(brake_shape=not sv.BRAKE_SHAPE_DEFAULT), layout, held, v_held
                    )
                    per_lap["S1_brake"] = braked["S1"]
                    per_lap["S3_brake"] = braked["S3"]
                    # and with every road taken as flat, mu_y set again by
                    # the convention on the layouts that are left
                    level = (
                        convene(raw, tracks, best, air=air, shifts=shifts, gears=gears, bank=False)
                        if any_bank
                        else None
                    )
                    if level is not None:
                        on_flat = score_laps(level, layout, held, v_held)
                        for col in ("S1", "S3", "top", "u"):
                            per_lap[f"{col}_flat"] = on_flat[col]
                    # heat per tyre, at the lap's own time and with the pace carried
                    own_best = best_all[best_all["layout_id"] == layout_id]
                    for who, tag in ((driven, ""), (level, "_flat")):
                        if share is None or who is None:
                            continue
                        heat = heat_errors(who, layout, held, v_held, share, own_best)
                        for col in heat.columns:
                            per_lap[f"{col}{tag}"] = heat[col]
                    if np.nanmax(v_held) >= cm.DRAG_REACH_MS:
                        for k in DRAG_GRID:
                            if k == 1.0:
                                per_lap["top_1.0"] = per_lap["top"]
                                continue
                            grid = driven.but(drag_scale=k)
                            per_lap[f"top_{k}"] = score_laps(grid, layout, held, v_held)["top"]
                lap_rows.append(per_lap)
    folds = pd.DataFrame(fold_rows)
    laps = pd.concat(lap_rows, ignore_index=True) if lap_rows else pd.DataFrame()
    return folds, laps


# ---------------------------------------------------------------- the verdict


def inside(folds: pd.DataFrame) -> pd.Series:
    """Whether each row of ``folds`` is inside the limits."""
    with np.errstate(invalid="ignore"):
        return (
            (folds["S1"] <= folds["floor"] + S1_OVER_FLOOR_MAX)
            & (folds["S2"].abs() <= S2_MAX)
            & (folds["S3"].abs() <= S3_MAX)
        )


def verdict(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(wide, by_car)``: one row per fold with each model's statistics
    and ``inside`` side by side and ``powered_<control>``; and per car and
    kind the counts and ``result`` (``pass``, ``fail`` or ``not scored``)."""
    keys = ["car", "kind", "layout_id", "date"]
    f = folds.assign(inside=inside(folds))
    wide = f[f["model"] == "fit"][keys + ["laps", "floor", "S1", "S2", "S3", "inside"]]
    for name in MODELS[1:]:
        ctl = f[f["model"] == name][keys + ["S1", "S2", "S3", "inside"]]
        ctl = ctl.rename(columns={c: f"{c}_{name}" for c in ("S1", "S2", "S3", "inside")})
        wide = wide.merge(ctl, on=keys, how="left")
        wide[f"powered_{name}"] = wide[f"inside_{name}"].eq(False) & wide["floor"].notna()
    wide["powered"] = wide[[f"powered_{n}" for n in MODELS[1:]]].any(axis=1)
    rows = []
    for (car, kind), g in wide.groupby(["car", "kind"]):
        row = {
            "car": car,
            "kind": kind,
            "folds": int(len(g)),
            "no_floor": int(g["floor"].isna().sum()),
            "fit_inside": int(g["inside"].sum()),
        }
        for name in MODELS[1:]:
            row[f"powered_{name}"] = int(g[f"powered_{name}"].sum())
        row["fit_outside_on_powered"] = int((g["powered"] & ~g["inside"]).sum())
        rows.append(row)
    by = pd.DataFrame(rows)
    results = []
    for car, g in by.groupby("car"):
        kinds = set(g["kind"])
        scored = kinds == {"day", "layout"} and all(
            (g[f"powered_{name}"] > 0).all() for name in MODELS[1:]
        )
        if g["fit_outside_on_powered"].sum():
            results.append("fail")  # whatever else is missing, this sends the work back
        else:
            results.append("pass" if scored else "not scored")
    by["result"] = by["car"].map(dict(zip(sorted(by["car"].unique()), results)))
    return wide, by


def check_car_holdout(
    root: Path, *, cache_dir: Path | None = None, tracks_dir: Path | None = None
) -> GateReport:
    """Gate C3: a car's parameters predict its speed trace on days and at
    layouts the fit never saw, and two planted wrong cars are told apart."""
    tables = cm.as_families(smp.car_tables(root, cache_dir=cache_dir, tracks_dir=tracks_dir))
    rows = cm.as_family(smp.build_laps(geo.clean_lap_index(root, cache_dir=cache_dir)))
    air = cm.air_state(root, geo.load_sessions(root))
    cars = [c for c in smp.MODEL_CARS if (tables.sessions["car"] == c).any()]
    layouts = sorted(str(k) for k in rows[rows["car"].isin(cars)]["layout_id"].unique())
    tracks = {
        k: smp.load_track(root, k, tracks_dir=tracks_dir, cache_dir=cache_dir) for k in layouts
    }
    try:
        shares: dict[str, ShareParams] | None = ho.production_shares(root)
    except FileNotFoundError:
        shares = None  # no fitted pressure model beside the dataset: heat is not scored
    full: dict[str, cg.Drive | None] = {}
    shifts: dict[str, dict[str, float]] = {}
    gears: dict[str, dict[str, tuple[float, ...]]] = {}
    for c in cars:
        best = cg.day_best_laps(rows, c)
        model, work = cm.fit_car(car_only(tables, c), c, air=air)
        shifts[c], gears[c] = cg.day_shifts(work), cg.day_gears(work)
        full[c] = convene(model, tracks, best, air=air, shifts=shifts[c], gears=gears[c])
    all_folds, all_laps = [], []
    for c in cars:
        # another car's parameters, its own default powertrain and shift point
        stand_in = full.get(OTHER_CAR.get(c, ""))
        other = stand_in.but(shifts={}, gears={}) if stand_in is not None else None
        folds, laps = run_folds(
            tables,
            rows,
            tracks,
            c,
            other,
            air=air,
            shifts=shifts[c],
            gears=gears[c],
            shares=shares,
        )
        all_folds.append(folds)
        all_laps.append(laps)
    folds = pd.concat(all_folds, ignore_index=True)
    laps = pd.concat(all_laps, ignore_index=True)
    wide, by = verdict(folds)
    rep = GateReport("C3", "the car on held-out days and layouts (`lapsim check-car-holdout`)")
    rep.notes.append(
        f"{len(wide)} folds of {MIN_FOLD_LAPS}+ laps ({int((wide['kind'] == 'day').sum())} by "
        f"day, {int((wide['kind'] == 'layout').sum())} by layout), "
        f"{int(laps[laps['kind'] == 'day']['S1'].notna().sum())} held-out laps by day. Controls: "
        f"`low_power` is the fold's own car with its power × {POWER_CONTROL} and mu_y raised to "
        "hold the reference laps; `other_car` is "
        + ", ".join(f"{v} for the {k}" for k, v in OTHER_CAR.items() if k in cars)
        + "."
    )

    # 1. The verdict per car.
    results = by.groupby("car")["result"].first()
    for c in cars:
        res = results.get(c, "not scored")
        g = by[by["car"] == c]
        rep.items.append(
            GateItem(
                f"{c}: inside on every powered fold, both controls outside",
                {"pass": True, "fail": False}.get(res),
                res
                + "; "
                + "; ".join(
                    f"{r.kind}: fitted inside {r.fit_inside}/{r.folds}, low power powered "
                    f"{r.powered_low_power}, other car powered {r.powered_other_car}, fitted "
                    f"outside on a powered fold {r.fit_outside_on_powered}"
                    for r in g.itertuples()
                ),
                f"S1 ≤ day floor + {S1_OVER_FLOOR_MAX}, |S2| ≤ {S2_MAX}, |S3| ≤ {S3_MAX} m/s; a "
                "powered day fold and a powered layout fold per control",
                md_table(wide[wide["car"] == c].drop(columns=["car"]), floats="{:.2f}"),
            )
        )

    fit = folds[folds["model"] == "fit"]

    # 3. Top-speed band on the layouts that reach 50 m/s.
    top_cols = [f"top_{k}" for k in DRAG_GRID if f"top_{k}" in laps.columns]
    lay = laps[laps["kind"] == "layout"]
    if top_cols:
        band = lay.dropna(subset=top_cols).groupby(["car", "layout_id"])[top_cols].median()
        band["best"] = band[top_cols].min(axis=1)
        band["model_over_best"] = band["top_1.0"] - band["best"]
        band = band.reset_index()
        c_used = fit[fit["kind"] == "layout"].set_index(["car", "layout_id"])["c_drag"]
        band["c_drag_e4"] = [
            1e4 * c_used.get(k, np.nan) for k in zip(band["car"], band["layout_id"])
        ]
        rep.items.append(
            GateItem(
                "Top-speed band",
                bool((band["model_over_best"] <= TOP_BAND_MAX).all()) if len(band) else None,
                "; ".join(
                    f"{r.car} {r.layout_id} +{r.model_over_best:.2f}" for r in band.itertuples()
                )
                + " m/s over the best of the grid",
                f"the held-out layout's top quarter of speeds: RMSE with the model's drag within "
                f"{TOP_BAND_MAX} m/s of the best over {DRAG_GRID} × it",
                "The drag is the one the car has without the held-out layout (`c_drag_e4`); "
                "`top_k`: RMS over the nodes where the lap is in its top quarter of speed, at "
                "k × that drag.\n\n" + md_table(band, floats="{:.2f}"),
            )
        )

    # 4. Cornering drag: does it earn its place?
    if "S1_other" in lay.columns:
        on, off = ("", "_other") if sv.CORNER_DRAG_DEFAULT else ("_other", "")
        term = (
            lay.groupby(["car", "layout_id"])
            .agg(
                laps=("S1", "count"),
                S1_with=(f"S1{on}", "median"),
                S1_without=(f"S1{off}", "median"),
                top_with=(f"top{on}", "median"),
                top_without=(f"top{off}", "median"),
            )
            .reset_index()
        )
        lower = int((term["S1_with"] < term["S1_without"]).sum())
        top_lower = int((term["top_with"] < term["top_without"]).sum())
        rep.items.append(
            GateItem(
                "Cornering drag as a term of the default",
                None,
                f"with it, leave-one-layout-out S1 is lower in {lower} of {len(term)} cells and "
                f"the top quarter of speeds in {top_lower} of {len(term)}",
                "in the default only if it lowers the error on the top quarter of speeds and "
                "does not raise the leave-one-layout-out error; "
                + ("it is on" if sv.CORNER_DRAG_DEFAULT else "it is off"),
                md_table(term, floats="{:.2f}"),
            )
        )

    # 4b. The time lost in the upshifts: the same question.
    if "S1_shift" in lay.columns:
        on, off = ("", "_shift") if sv.SHIFT_LOSS_DEFAULT else ("_shift", "")
        term = (
            lay.groupby(["car", "layout_id"])
            .agg(
                laps=("S1", "count"),
                S1_with=(f"S1{on}", "median"),
                S1_without=(f"S1{off}", "median"),
                top_with=(f"top{on}", "median"),
                top_without=(f"top{off}", "median"),
            )
            .reset_index()
        )
        term = term[(term["S1_with"] != term["S1_without"])].reset_index(drop=True)
        lower = int((term["S1_with"] < term["S1_without"]).sum())
        top_lower = int((term["top_with"] < term["top_without"]).sum())
        rep.items.append(
            GateItem(
                "Shift time as a term of the default",
                None,
                f"with it, leave-one-layout-out S1 is lower in {lower} of {len(term)} cells and "
                f"the top quarter of speeds in {top_lower} of {len(term)}",
                "in the default only if it lowers the error on the top quarter of speeds and "
                "does not raise the leave-one-layout-out error; "
                + ("it is on" if sv.SHIFT_LOSS_DEFAULT else "it is off"),
                "The cells of the cars driven through their gears with a measured shift time "
                "(a car on one power level has no shift).\n\n" + md_table(term, floats="{:.2f}"),
            )
        )

    # 4c. The braking limit's shape: the same question.
    if "S1_brake" in lay.columns:
        on, off = ("", "_brake") if sv.BRAKE_SHAPE_DEFAULT else ("_brake", "")
        term = (
            lay.groupby(["car", "layout_id"])
            .agg(
                laps=("S1", "count"),
                S1_with=(f"S1{on}", "median"),
                S1_without=(f"S1{off}", "median"),
                S3_with=(f"S3{on}", "median"),
                S3_without=(f"S3{off}", "median"),
            )
            .reset_index()
        )
        lower = int((term["S1_with"] < term["S1_without"]).sum())
        rep.items.append(
            GateItem(
                "The braking limit's shape as a term of the default",
                None,
                f"with it, leave-one-layout-out S1 is lower in {lower} of {len(term)} cells",
                "in the default only if it does not raise the leave-one-layout-out error; "
                + ("it is on" if sv.BRAKE_SHAPE_DEFAULT else "it is off"),
                "With the car's own `brake_ellipse` against the diamond.\n\n"
                + md_table(term, floats="{:.2f}"),
            )
        )

    # 4d. The road's bank: against every road taken as flat.
    if "S1_flat" in lay.columns:
        term = (
            lay.groupby(["car", "layout_id"])
            .agg(
                laps=("S1", "count"),
                S1_with=("S1", "median"),
                S1_without=("S1_flat", "median"),
                S3_with=("S3", "median"),
                S3_without=("S3_flat", "median"),
                u_with=("u", "median"),
                u_without=("u_flat", "median"),
            )
            .reset_index()
        )
        cells = [g for _, g in lay.groupby(["car", "layout_id"])]
        for name, kind in (("heat", "h"), ("heat_carried", "hc")):
            term[f"{name}_with"] = [worst_tyre(g, kind) for g in cells]
            term[f"{name}_without"] = [worst_tyre(g, kind, "_flat") for g in cells]
        lower = int((term["S1_with"] < term["S1_without"]).sum())
        apex = int((term["S3_with"].abs() < term["S3_without"].abs()).sum())
        nearer = int(((term["u_with"] - 1).abs() < (term["u_without"] - 1).abs()).sum())
        measured = (
            f"with it, leave-one-layout-out S1 is lower in {lower} of {len(term)} cells, the "
            f"apex speed nearer in {apex} and the held-out layout's u nearer 1 in {nearer}; "
            f"median |u − 1| {float((term['u_with'] - 1).abs().median()):.3f} with it, "
            f"{float((term['u_without'] - 1).abs().median()):.3f} without"
        )
        if term["heat_carried_with"].notna().any():
            cooler = int((term["heat_carried_with"] < term["heat_carried_without"]).sum())
            measured += (
                f"; worst-tyre heat with the pace carried "
                f"{100 * float(term['heat_carried_with'].median()):.1f} % with it, "
                f"{100 * float(term['heat_carried_without'].median()):.1f} % without (lower in "
                f"{cooler}), and at the lap's own time "
                f"{100 * float(term['heat_with'].median()):.1f} % against "
                f"{100 * float(term['heat_without'].median()):.1f} %"
            )
        rep.items.append(
            GateItem(
                "The road's bank as a term of the default",
                None,
                measured,
                "reported; it is on by the owner's decision of 2026-11-17",
                "The bank is the track's, from the elevation model at full strength; without it "
                "every road is taken as flat (`Profile.from_track(track, bank=False)`). `mu_y` "
                "is set again by the convention on the layouts that are left, each way. `u`: "
                "the median over the held-out layout's laps of the pace knob, which is 1 where "
                "the car's level carries to a layout it was not set on. `heat`: the largest "
                "over the four tyres of the absolute median over the laps of log(simulated / "
                "logged lap-mean heat), through production's share model, in the tyres' frame "
                "on the bank and the ground frame without it; `heat_carried`: the same with "
                "each lap driven at its own u over the layout's reference u, the pace level "
                "carried from the car's other layouts.\n\n" + md_table(term, floats="{:.3f}"),
            )
        )

    # 5. Fold summaries.
    summary = (
        folds.groupby(["car", "kind", "model"])
        .agg(
            folds=("S1", "size"),
            S1=("S1", "median"),
            S2=("S2", "median"),
            S3=("S3", "median"),
            u=("u", "median"),
            mu_y=("mu_y", "median"),
            p_specific=("p_specific", "median"),
        )
        .reset_index()
    )
    rep.items.append(
        GateItem(
            "The three models through the folds",
            None,
            "; ".join(
                f"{r.car} {r.kind} {r.model} {r.S1:.2f} / {r.S2:+.2f} / {r.S3:+.2f}"
                for r in summary[summary["kind"] == "layout"].itertuples()
            ),
            "median over folds of S1 / S2 / S3 (m/s); reported",
            md_table(summary, floats="{:.3f}"),
        )
    )
    return rep


__all__ = [
    "Layout",
    "car_only",
    "check_car_holdout",
    "convene",
    "fold_car",
    "heat_errors",
    "inside",
    "lap_stats",
    "run_folds",
    "score_laps",
    "verdict",
    "without",
    "worst_tyre",
]
