"""Gate E1 and arm B of gate E2 (#207): a simulated lap's heat per tyre
against the heat production has for the logged lap.

It rides on gate C3's folds (:func:`~.car_holdout.run_folds`): a
held-out dry build lap is solved at its own lap time by the fold's car
(fitted without the lap's day: **E1**; without the lap's layout: **E2's
arm B**), and that lap, not a pace-table read, goes through production's
chain (:func:`~.heat.sim_heat`).

By the owner's rulings of 2026-11-24 production is not retrained for
this and the score is in gate C3's form:

- **Observed heat** is production's chain on the sensor's series of the
  lap (:func:`~.heat_gate.chain_lap_heat`). A lap with a discarded
  window or of mixed source is left out.
- **The frame factor** brings the simulated lap (the tyres' frame) to it.
  A fold reads it from logged laps that are not held out
  (:func:`fold_factors`): in a day fold the car's laps at the layout,
  from the same kind of source, without the day; in a layout fold the
  car's laps at its other layouts.
- **A cell** is a car at a layout at a level of pace on one tyre; it
  passes when the simulated lap's heat is no further from the logged
  lap's than a logged lap of another day and like lap time, *from the
  same kind of source*, is (:func:`yardsticks`).

Beside it, the same comparison with no sensor in it: simulated heat over
the logged lap's own heat in the tyres' frame
(:func:`~.heat_gate.tyres_frame_lap_heat`), no factor (``g_<tyre>``
where the sensor-frame measure is ``e_<tyre>``). And, so that the count
can be seen to rank or not, the fold's planted wrong cars and "the car's
other layouts, unscaled" on the same cells.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..heat_input import ShareParams
from . import car as cm
from . import car_gate as cg
from . import car_holdout as ch
from . import heat_gate as hg
from . import holdout as ho
from . import samples as smp
from . import solver as sv
from . import track as trk
from .heat import CORNERS, sim_heat

#: The measures: log of simulated over observed heat per tyre, in
#: production's frame through the factor (``e``) and in the tyres' (``g``).
E_MEASURES = tuple(f"e_{c}" for c in CORNERS)
G_MEASURES = tuple(f"g_{c}" for c in CORNERS)
#: Fewest laps a fold's frame factor is read from.
FACTOR_MIN_LAPS = 8
#: The simulated arms of a fold, and the logged one of a layout fold.
SIM_MODELS = ("fit", "low_power", "other_car")
OTHER_LAYOUTS = "other_layouts"


def observed(
    root: Path,
    laps: pd.DataFrame,
    shares: dict[str, ShareParams],
    *,
    tracks_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Per lap of ``laps`` (``car, session_id, lap_num``): production's
    heat (``q_<tyre>``, with ``source`` and ``hole``) and the lap's heat
    in the tyres' frame (``t_<tyre>``)."""
    keyed = laps[["car", "session_id", "lap_num"]].drop_duplicates()
    own: dict[str, ShareParams] = {}
    for car in keyed["car"].unique():
        try:
            own[str(car)] = ho.share_for(str(car), shares)
        except KeyError:
            continue
    sensor = hg.chain_lap_heat(root, keyed, own)
    cols = ["session_id", "lap_num", "source", "hole"] + [f"q_{c}" for c in CORNERS]
    if sensor.empty:
        return pd.DataFrame(columns=cols + [f"t_{c}" for c in CORNERS])
    frame = hg.tyres_frame_lap_heat(root, keyed, own, tracks_dir=tracks_dir, cache_dir=cache_dir)
    return sensor[cols].merge(frame, on=["session_id", "lap_num"], how="left")


def usable(obs: pd.DataFrame) -> pd.DataFrame:
    """The laps a heat can be scored on (no discarded window, one kind of
    source, a heat on every tyre), with their heats and source alone."""
    ok = (~obs["hole"].astype(bool)) & obs["source"].isin(["imu", "gps"])
    ok &= obs[[f"q_{c}" for c in CORNERS]].notna().all(axis=1)
    cols = ["session_id", "lap_num", "source"]
    cols += [f"q_{c}" for c in CORNERS] + [f"t_{c}" for c in CORNERS]
    return obs.loc[ok, cols]


def _factor(pool: pd.DataFrame) -> np.ndarray:
    """Per tyre, the median over ``pool`` of production's heat over the
    tyres'-frame heat (NaN under :data:`FACTOR_MIN_LAPS` laps)."""
    q = pool[[f"q_{c}" for c in CORNERS]].to_numpy(float)
    t = pool[[f"t_{c}" for c in CORNERS]].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(q / t)
    r = r[np.isfinite(r).all(axis=1)]
    if len(r) < FACTOR_MIN_LAPS:
        return np.full(len(CORNERS), np.nan)
    return np.asarray(np.exp(np.median(r, axis=0)))


def fold_factors(
    car_obs: pd.DataFrame, layout_id: str, kind: str, date: str
) -> dict[str, np.ndarray]:
    """``source -> factor per tyre`` for a fold of one car, from the car's
    logged laps (``car_obs``: ``layout_id, date, source, q_*, t_*``) that
    the fold does not hold out: the layout's without ``date`` (a day
    fold), or the other layouts' (a layout fold)."""
    if kind == "day":
        pool = car_obs[(car_obs["layout_id"] == layout_id) & (car_obs["date"] != date)]
    else:
        pool = car_obs[car_obs["layout_id"] != layout_id]
    return {str(src): _factor(grp) for src, grp in pool.groupby("source")}


def errors(
    laps: pd.DataFrame, obs: pd.DataFrame, where: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per held-out lap, fold kind and arm: ``e_<tyre>`` and
    ``g_<tyre>`` (module docstring).

    ``laps`` is gate C3's per-lap frame (``car, kind, layout_id,
    session_id, lap_num, date`` with the simulated heats ``sim_q_<tyre>``
    of the fitted car and ``sim_q_<tyre>_<name>`` of the planted ones;
    not gate C3's own ``h_<tyre>``, which is a heat error); ``obs``
    is :func:`observed` of the same laps, usable or not. ``model`` is one
    of :data:`SIM_MODELS` or, on a layout fold, :data:`OTHER_LAYOUTS`
    (the car's median logged heat at its other layouts from the same
    kind of source, unscaled: no simulated lap). ``where`` names the
    car's logged laps the factors and the other layouts are read from
    (``car, layout_id, session_id, lap_num, date``; ``laps`` itself
    when it is not given)."""
    keys = ["car", "kind", "layout_id", "session_id", "lap_num", "date"]
    cols = keys + ["source", "model", *E_MEASURES, *G_MEASURES]
    if laps.empty or "sim_q_fl" not in laps.columns:
        return pd.DataFrame(columns=cols)
    good = usable(obs)
    both = laps.merge(good, on=["session_id", "lap_num"], how="inner")
    known = laps if where is None else where
    known = known[["car", "layout_id", "session_id", "lap_num", "date"]].drop_duplicates(
        ["session_id", "lap_num"]
    )
    car_obs_all = known.merge(good, on=["session_id", "lap_num"], how="inner")
    q_cols, t_cols = [f"q_{c}" for c in CORNERS], [f"t_{c}" for c in CORNERS]
    out: list[pd.DataFrame] = []
    for (car, kind, layout_id), cell in both.groupby(["car", "kind", "layout_id"], sort=False):
        car_obs = car_obs_all[car_obs_all["car"] == car]
        folds = cell.groupby("date", sort=False) if kind == "day" else [("", cell)]
        for date, held in folds:
            factors = fold_factors(car_obs, str(layout_id), str(kind), str(date))
            src = held["source"].astype(str).to_numpy()
            factor = np.vstack([factors.get(s, np.full(len(CORNERS), np.nan)) for s in src])
            q, t = held[q_cols].to_numpy(float), held[t_cols].to_numpy(float)
            base = held[keys + ["source"]].reset_index(drop=True)
            for name in SIM_MODELS:
                tag = "" if name == "fit" else f"_{name}"
                h_cols = [f"sim_q_{c}{tag}" for c in CORNERS]
                if any(c not in held.columns for c in h_cols):
                    continue
                h = held[h_cols].to_numpy(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    e, g = np.log(h * factor / q), np.log(h / t)
                out.append(_rows(base, name, e, g))
            if kind == "layout":
                others = car_obs[car_obs["layout_id"] != layout_id]
                e = np.full(q.shape, np.nan)
                g = np.full(q.shape, np.nan)
                for s in np.unique(src):
                    pool = others[others["source"] == s]
                    if len(pool) < FACTOR_MIN_LAPS:
                        continue
                    on = src == s
                    with np.errstate(divide="ignore", invalid="ignore"):
                        e[on] = np.log(np.median(pool[q_cols].to_numpy(float), axis=0) / q[on])
                        g[on] = np.log(np.nanmedian(pool[t_cols].to_numpy(float), axis=0) / t[on])
                out.append(_rows(base, OTHER_LAYOUTS, e, g))
    if not out:
        return pd.DataFrame(columns=cols)
    got = pd.concat(out, ignore_index=True)
    return got.replace([np.inf, -np.inf], np.nan)[cols]


def _rows(base: pd.DataFrame, model: str, e: np.ndarray, g: np.ndarray) -> pd.DataFrame:
    frame = base.copy()
    frame["model"] = model
    for k, c in enumerate(CORNERS):
        frame[f"e_{c}"] = e[:, k]
        frame[f"g_{c}"] = g[:, k]
    return frame


def yardsticks(cell: pd.DataFrame, levels: tuple[float, ...] = ch.PACE_LEVELS) -> pd.DataFrame:
    """Reality against itself in heat, for the usable laps of one car at
    one layout (``session_id, lap_num, date, lap_time_s, source, q_*,
    t_*``): per lap above each level's percentile of pace, the median
    over its :data:`~.car_gate.FLOOR_PAIRS_PER_LAP` nearest laps in lap
    time of the size of the log ratio of their heats, per tyre, from
    another day and from its own. A pair is two laps of the same kind of
    source. ``e_<tyre>_other_day`` and ``_same_day`` compare production's
    heats, ``g_<tyre>_…`` the tyres'-frame ones."""
    t = cell["lap_time_s"].to_numpy(float)
    day = cell["date"].to_numpy()
    src = cell["source"].astype(str).to_numpy()
    heats = {
        "e": np.log(cell[[f"q_{c}" for c in CORNERS]].to_numpy(float)),
        "g": np.log(cell[[f"t_{c}" for c in CORNERS]].to_numpy(float)),
    }
    rows: list[dict] = []
    for level in levels:
        inset = ch.fast_laps(t, level)
        for i in np.flatnonzero(inset):
            row: dict = {
                "session_id": cell["session_id"].iloc[i],
                "lap_num": cell["lap_num"].iloc[i],
                "date": day[i],
                "level": float(level),
            }
            others = inset & (np.arange(len(t)) != i) & (src == src[i])
            for tag, pool in (("other_day", day != day[i]), ("same_day", day == day[i])):
                pick = np.flatnonzero(others & pool)
                pick = pick[np.argsort(np.abs(t[pick] - t[i]))][: cg.FLOOR_PAIRS_PER_LAP]
                for name, logs in heats.items():
                    gap = np.abs(logs[pick] - logs[i]) if len(pick) else np.empty((0, len(CORNERS)))
                    for k, c in enumerate(CORNERS):
                        vals = gap[:, k][np.isfinite(gap[:, k])]
                        row[f"{name}_{c}_{tag}"] = float(np.median(vals)) if len(vals) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def cell_yardsticks(laps: pd.DataFrame, obs: pd.DataFrame, lap_time: pd.Series) -> pd.DataFrame:
    """:func:`yardsticks` of every car and layout of gate C3's laps, keyed
    as gate C3's are (``car, layout_id, session_id, lap_num, date,
    level``). ``lap_time`` is by ``(session_id, lap_num)``."""
    where = laps[["car", "layout_id", "session_id", "lap_num", "date"]].drop_duplicates(
        ["session_id", "lap_num"]
    )
    own = where.merge(usable(obs), on=["session_id", "lap_num"], how="inner")
    keys = pd.MultiIndex.from_frame(own[["session_id", "lap_num"]])
    own = own.assign(lap_time_s=lap_time.reindex(keys).to_numpy())
    own = own[np.isfinite(own["lap_time_s"])]
    parts = []
    for (car, layout_id), cell in own.groupby(["car", "layout_id"], sort=False):
        got = yardsticks(cell.reset_index(drop=True))
        if len(got):
            got.insert(0, "car", car)
            got.insert(1, "layout_id", layout_id)
            parts.append(got)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def score(errs: pd.DataFrame, yard: pd.DataFrame, measures: tuple[str, ...]) -> pd.DataFrame:
    """Gate C3's attainment of one arm on ``measures`` (sizes compared):
    one row per car, layout, level and tyre (:func:`~.car_holdout.attainment`)."""
    if errs.empty or yard.empty:
        return pd.DataFrame()
    sized = errs.copy()
    sized[list(measures)] = sized[list(measures)].abs()
    pace = ch.by_pace(sized, yard, measures)
    return ch.attainment(pace, measures) if len(pace) else pd.DataFrame()


def tally(cells: pd.DataFrame) -> dict:
    """Of a score: ``passed`` of ``scored`` cells, and ``whole`` of
    ``groups``: the car × layout × level groups with every scored tyre
    inside (the four tyres of a lap move together)."""
    if cells.empty:
        return {"passed": 0, "scored": 0, "whole": 0, "groups": 0}
    own = cells[cells["scored"]]
    groups = own.groupby(["car", "layout_id", "level"])["passed"].all()
    return {
        "passed": int(own["passed"].sum()),
        "scored": int(len(own)),
        "whole": int(groups.sum()),
        "groups": int(len(groups)),
    }


def signed(errs: pd.DataFrame) -> pd.DataFrame:
    """Per car, layout and arm: the signed ``e`` and ``g`` per tyre (the
    median over days of the day's median lap), with ``laps`` and ``days``."""
    cols = [*E_MEASURES, *G_MEASURES]
    if errs.empty:
        return pd.DataFrame()
    cell = ["car", "layout_id", "model"]
    days = errs.groupby(cell + ["date"], sort=False)[cols].median()
    out = days.groupby(cell, sort=False)[cols].median()
    out.insert(0, "laps", errs.dropna(subset=list(E_MEASURES)).groupby(cell, sort=False).size())
    out.insert(1, "days", days.groupby(cell, sort=False).size())
    return out.reset_index()


# ------------------------------------------------- arms C and D of gate E2

#: The arms: the car fitted on everything or without the layout, on its
#: own track or on the track from other cars.
ARMS = ("A", "B", "C", "D")
ARM_NAMES = {
    "A": "the car's own fit, its own track",
    "B": "the car fitted without the layout, its own track",
    "C": "the car's own fit, the track from other cars",
    "D": "the car fitted without the layout, the track from other cars",
}
#: The same with the lap time not given: each day's best lap driven at
#: the car's typical use of its limit (what is carried to a layout it has
#: not run), on the arm's car and track.
CARRIED = {"A*": "A", "B*": "B", "D*": "D"}


def fold_track_name(car: str, layout_id: str) -> str:
    """The file stem of gate B3's saved track of ``layout_id`` built
    without the class of ``car`` (the strictest fold)."""
    return f"{layout_id}__without_{ho.CLASS_OF.get(car, car).replace(' ', '-')}__rung5"


def laid_track(own: trk.TrackModel, other: trk.TrackModel) -> trk.TrackModel:
    """``own`` with the curvature of ``other`` (the same layout from other
    cars' laps) laid on its nodes, so that a lap driven on it has its
    speed where the logged laps have theirs. The bank and the grade stay
    ``own``'s: the bank is the road's, from the elevation model."""
    from .wet_gate import wet_line

    return replace(own, kappa_heat=wet_line(own, other))


def transfer_laps(
    tables: smp.CarTables,
    rows: pd.DataFrame,
    tracks: dict[str, trk.TrackModel],
    others: dict[str, trk.TrackModel],
    car: str,
    full: cg.Drive,
    share: ShareParams,
    *,
    air: pd.DataFrame | None = None,
    shifts: dict[str, float] | None = None,
    gears: dict[str, tuple[float, ...]] | None = None,
) -> pd.DataFrame:
    """One row per dry build lap of ``car`` and arm of :data:`ARMS`, at
    each layout that has a track from other cars (``others``, by
    layout): the simulated lap's heat per tyre (``sim_q_<tyre>``; the
    tyres' frame) and the knob it solved to, with ``car, layout_id,
    session_id, lap_num, date, arm``. ``full`` is the car fitted on
    everything (gate C3's); the car without the layout is fitted here as
    a layout fold of gate C3 is."""
    own = ch.car_only(tables, car)
    best_all = cg.day_best_laps(rows, car)
    lap_time = rows.set_index(["session_id", "lap_num"])["lap_time_s"]
    use_tracks = {k: v for k, v in tracks.items() if k in set(best_all["layout_id"])}
    use_laps = cg.car_use_laps(own, lap_time, car, use_tracks)
    out: list[dict] = []
    for layout_id, other in others.items():
        if layout_id not in tracks:
            continue
        cell, _ = own.lap_speeds(car, layout_id)
        if cell.empty:
            continue
        keys = pd.MultiIndex.from_frame(cell[["session_id", "lap_num"]])
        cell = cell.assign(lap_time_s=lap_time.reindex(keys).to_numpy())
        cell = cell[np.isfinite(cell["lap_time_s"])].reset_index(drop=True)
        raw = cm.fit_car(ch.without(own, layout_id), car, air=air)[0]
        unseen = ch.convene(
            raw,
            tracks,
            best_all[best_all["layout_id"] != layout_id],
            air=air,
            shifts=shifts,
            gears=gears,
            laps=ch.laps_left(use_laps, layout_id),
        )
        roads = {"own": tracks[layout_id], "other": laid_track(tracks[layout_id], other)}
        recs = cg._records(cell)
        target = [float(rec["lap_time_s"]) for rec in recs]
        for arm, who, road in (
            ("A", full, "own"),
            ("B", unseen, "own"),
            ("C", full, "other"),
            ("D", unseen, "other"),
        ):
            if who is None:
                continue
            drive = who.but(use_cap=True)
            profile = cg.profile_of(drive, roads[road])
            cars = [cg._lap_car(drive, rec) for rec in recs]
            got = sv.solve_laps(profile, cars, target, guess=cg.typical_use(drive))
            for i in range(len(cell)):
                row = {
                    "car": car,
                    "layout_id": layout_id,
                    "session_id": cell["session_id"].iloc[i],
                    "lap_num": int(cell["lap_num"].iloc[i]),
                    "date": str(cell["date"].iloc[i]),
                    "arm": arm,
                    "use": float(got.u[i]) if got.ok[i] else np.nan,
                }
                if got.ok[i]:
                    sim = sv.lap_from_nodes(profile, cars[i], got.v_node_ms[i], float(got.u[i]))
                    for tyre, q in zip(CORNERS, sim_heat(sim, share)):
                        row[f"sim_q_{tyre}"] = float(q)
                out.append(row)
        # the lap time not given: each day's best lap at the car's typical use
        best = cell.loc[cell.groupby("date")["lap_time_s"].idxmin()].reset_index(drop=True)
        best_recs = cg._records(best)
        for arm, who, road in (("A*", full, "own"), ("B*", unseen, "own"), ("D*", unseen, "other")):
            if who is None:
                continue
            drive = who.but(use_cap=True)
            profile = cg.profile_of(drive, roads[road])
            typical = cg.typical_use(drive)
            cars = [cg._lap_car(drive, rec) for rec in best_recs]
            got = sv.drive_laps(profile, cars, np.full(len(cars), typical))
            for i in range(len(best)):
                sim = sv.lap_from_nodes(profile, cars[i], got.v_node_ms[i], typical)
                out.append(
                    {
                        "car": car,
                        "layout_id": layout_id,
                        "session_id": best["session_id"].iloc[i],
                        "lap_num": int(best["lap_num"].iloc[i]),
                        "date": str(best["date"].iloc[i]),
                        "arm": arm,
                        "use": float(typical),
                        "lap_time_s": float(best["lap_time_s"].iloc[i]),
                        "lap_time_sim_s": float(got.lap_time_s[i]),
                        **{f"sim_q_{t}": float(q) for t, q in zip(CORNERS, sim_heat(sim, share))},
                    }
                )
    return pd.DataFrame(out)


def pace_carried(arm_laps: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """The lap time not given (the arms of :data:`CARRIED`): per car,
    layout and arm, over the days' best laps, ``lap_time`` (the median of
    the log of the simulated lap time at the car's typical use over the
    day's best lap), ``lap_time_size`` (the median of its size),
    ``lap_time_other_day`` (the same size between a day's best lap and
    another day's best lap at the layout: reality against itself) and, per
    tyre, ``g_<tyre>``: the median of the log of that lap's simulated heat
    over the best lap's own heat in the tyres' frame."""
    if arm_laps.empty or "lap_time_sim_s" not in arm_laps.columns:
        return pd.DataFrame()
    own = arm_laps[arm_laps["arm"].isin(list(CARRIED))]
    both = own.merge(usable(obs), on=["session_id", "lap_num"], how="inner")
    if both.empty:
        return pd.DataFrame()
    with np.errstate(divide="ignore", invalid="ignore"):
        both["lap_time"] = np.log(both["lap_time_sim_s"] / both["lap_time_s"])
        for c in CORNERS:
            both[f"g_{c}"] = np.log(both[f"sim_q_{c}"] / both[f"t_{c}"])
    both["lap_time_size"] = both["lap_time"].abs()
    # reality against itself for a lap time: another day's best lap at the layout
    both["lap_time_other_day"] = np.nan
    for _, grp in both.groupby(["car", "layout_id", "arm"], sort=False):
        best = np.log(grp["lap_time_s"].to_numpy(float))
        if len(best) > 1:
            gaps = np.abs(best[:, None] - best[None, :])
            np.fill_diagonal(gaps, np.nan)
            both.loc[grp.index, "lap_time_other_day"] = np.nanmedian(gaps, axis=1)
    cols = ["lap_time", "lap_time_size", "lap_time_other_day", *G_MEASURES]
    cell = ["car", "layout_id", "arm"]
    out = both.groupby(cell, sort=False)[cols].median()
    out.insert(0, "days", both.groupby(cell, sort=False)["date"].nunique())
    return out.reset_index()


def transfers(arm_laps: pd.DataFrame, obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(per_lap, by_cell)`` of :func:`transfer_laps`: per lap and arm the
    log of simulated heat over the lap's own heat in the tyres' frame
    (``g_<tyre>``); and per car and layout, the median over days of the
    day's median lap of each arm's ``g`` and, lap by lap, of the
    transfers ``B-A``, ``C-A``, ``D-A`` and of ``D-B-C+A`` (whether they
    add), one row per quantity (``what``)."""
    t_cols = [f"t_{c}" for c in CORNERS]
    if arm_laps.empty or "sim_q_fl" not in arm_laps.columns:
        return pd.DataFrame(), pd.DataFrame()
    both = arm_laps.merge(usable(obs), on=["session_id", "lap_num"], how="inner")
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.log(both[[f"sim_q_{c}" for c in CORNERS]].to_numpy(float) / both[t_cols].to_numpy())
    per_lap = both[["car", "layout_id", "session_id", "lap_num", "date", "arm", "source"]].copy()
    for k, c in enumerate(CORNERS):
        per_lap[f"g_{c}"] = g[:, k]
    per_lap = per_lap.replace([np.inf, -np.inf], np.nan)
    keys = ["car", "layout_id", "session_id", "lap_num", "date"]
    wide = per_lap.pivot_table(index=keys, columns="arm", values=list(G_MEASURES))
    rows = []
    for (car, layout_id), cell in wide.groupby(level=["car", "layout_id"], sort=False):
        have = set(cell.columns.get_level_values("arm"))
        things: dict[str, pd.DataFrame] = {
            a: pd.DataFrame(cell.xs(a, axis=1, level="arm")) for a in ARMS if a in have
        }
        if {"A", "B"} <= have:
            things["B-A"] = things["B"] - things["A"]
        if {"A", "C"} <= have:
            things["C-A"] = things["C"] - things["A"]
        if set(ARMS) <= have:
            things["D-A"] = things["D"] - things["A"]
            things["D-B-C+A"] = things["D"] - things["B"] - things["C"] + things["A"]
        for what, frame in things.items():
            days = frame.groupby(level="date").median()
            mid = days.median()
            rows.append(
                {
                    "car": car,
                    "layout_id": layout_id,
                    "what": what,
                    "laps": int(frame.dropna().shape[0]),
                    "days": int(len(days)),
                    **{m: float(mid[m]) for m in G_MEASURES},
                }
            )
    return per_lap, pd.DataFrame(rows)


CARRIED_NAMES = {
    "A*": "no lap time given: the car's own fit, its own track",
    "B*": "no lap time given: the car fitted without the layout, its own track",
    "D*": "no lap time given: the car fitted without the layout, the track from other cars",
}


def arm_scores(
    arm_laps: pd.DataFrame, obs: pd.DataFrame, yard: pd.DataFrame, laps: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Each arm of :func:`transfer_laps` as paired runs against real laps:
    its cells in gate C3's form in the tyres' frame
    (:func:`score` on ``g``), by arm, with :data:`OTHER_LAYOUTS` (the
    car's logged heat at its other layouts, unscaled, on arm D's laps)
    last. ``laps`` is gate C3's per-lap frame, which says where the car's
    other layouts are."""
    out: dict[str, pd.DataFrame] = {}
    others = pd.DataFrame()
    for arm in (*ARMS, *CARRIED):
        own = arm_laps[arm_laps["arm"] == arm]
        if own.empty:
            continue
        errs = errors(own.assign(kind="layout"), obs, where=laps)
        out[arm] = score(errs[errs["model"] == "fit"], yard, G_MEASURES)
        if arm == "D":
            others = score(errs[errs["model"] == OTHER_LAYOUTS], yard, G_MEASURES)
    if len(others):
        out[OTHER_LAYOUTS] = others
    return out


# ------------------------------------- the pace response: bands of pace

#: Bands of a clean dry lap's time over its session's best: the build
#: laps (within 2 %), the other clean laps up to here, and the slow laps.
BANDS = ("fast", "mid", "slow")
MID_MAX = 1.04
SLOW_MAX = 1.12


def band_rows(index: pd.DataFrame, cars: tuple[str, ...]) -> pd.DataFrame:
    """The clean dry laps of ``cars`` within :data:`SLOW_MAX` of their
    session's best, from the clean-lap index (by family), with ``band``:
    ``fast`` (a build lap), ``mid`` (another clean lap up to
    :data:`MID_MAX`) or ``slow``. A session with no dry build lap (a wet
    one) is left out."""
    own = index[index["clean"] & index["car"].isin(cars)]
    dry = set(own.loc[own["in_build"], "session_id"])
    own = own[own["session_id"].isin(dry)].copy()
    over = own["lap_time_s"] / own.groupby("session_id")["lap_time_s"].transform("min")
    own["band"] = np.select(
        [
            own["in_build"].to_numpy(bool),
            (over <= MID_MAX).to_numpy(),
            (over <= SLOW_MAX).to_numpy(),
        ],
        list(BANDS),
        default="",
    )
    return own[own["band"] != ""].reset_index(drop=True)


def band_heat(
    rows: pd.DataFrame,
    tracks: dict[str, trk.TrackModel],
    full: dict[str, cg.Drive | None],
    shares: dict[str, ShareParams],
) -> pd.DataFrame:
    """Each lap of :func:`band_rows` solved to its own lap time by its
    car's own fit on its own track (its day's powertrain, shift point,
    gears and air): ``sim_q_<tyre>`` beside the lap's keys and ``band``.
    A lap the car cannot do, or of a car without shares or a track, is
    left out."""
    out: list[dict] = []
    for (car, layout_id), cell in rows.groupby(["car", "layout_id"], sort=False):
        drive = full.get(str(car))
        if drive is None or layout_id not in tracks:
            continue
        try:
            share = ho.share_for(str(car), shares)
        except KeyError:
            continue
        drive = drive.but(use_cap=True)
        profile = cg.profile_of(drive, tracks[str(layout_id)])
        cell = cell.reset_index(drop=True)
        recs = cg._records(cell)
        cars = [cg._lap_car(drive, rec) for rec in recs]
        got = sv.solve_laps(
            profile, cars, [float(r["lap_time_s"]) for r in recs], guess=cg.typical_use(drive)
        )
        for i in range(len(cell)):
            if not got.ok[i]:
                continue
            sim = sv.lap_from_nodes(profile, cars[i], got.v_node_ms[i], float(got.u[i]))
            out.append(
                {
                    "car": car,
                    "layout_id": layout_id,
                    "session_id": cell["session_id"].iloc[i],
                    "lap_num": int(cell["lap_num"].iloc[i]),
                    "date": str(cell["date"].iloc[i]),
                    "band": cell["band"].iloc[i],
                    "lap_time_s": float(cell["lap_time_s"].iloc[i]),
                    "use": float(got.u[i]),
                    **{f"sim_q_{t}": float(q) for t, q in zip(CORNERS, sim_heat(sim, share))},
                }
            )
    return pd.DataFrame(out)


def band_scores(heat: pd.DataFrame, obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(by_band, by_cell)`` of :func:`band_heat` against the logged
    laps' own heat in the tyres' frame.

    ``by_band``: per band, the cells (car, layout, tyre) in which the
    simulated lap is no further from the logged lap than a logged lap *of
    the same band*, another day and like lap time is (``passed`` of
    ``scored``), the two distances in the median scored cell
    (``simulated``, ``other_day``), ``same_day`` likewise, and ``laps``.
    ``by_cell``: per band, car and layout, the signed log of simulated
    over logged heat (the median over days of the day's median lap) per
    tyre, its four-tyre ``mean`` and ``right_less_left``, with ``laps``
    and ``days``."""
    if heat.empty:
        return pd.DataFrame(), pd.DataFrame()
    both = heat.merge(usable(obs), on=["session_id", "lap_num"], how="inner")
    g = list(G_MEASURES)
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in CORNERS:
            both[f"g_{c}"] = np.log(both[f"sim_q_{c}"] / both[f"t_{c}"])
    both = both.replace([np.inf, -np.inf], np.nan).dropna(subset=g)
    band_rows_, cell_rows = [], []
    for band in BANDS:
        own = both[both["band"] == band]
        if own.empty:
            continue
        parts = []
        for (car, layout_id), cell in own.groupby(["car", "layout_id"], sort=False):
            yard = yardsticks(cell.reset_index(drop=True), levels=(0.0,))
            if len(yard):
                yard.insert(0, "car", str(car))
                yard.insert(1, "layout_id", str(layout_id))
                parts.append(yard)
        cells = (
            score(own, pd.concat(parts, ignore_index=True), G_MEASURES) if parts else pd.DataFrame()
        )
        row: dict = {"band": band, "laps": int(len(own)), "passed": 0, "scored": 0}
        if len(cells):
            ok = cells[cells["scored"]]
            row.update(
                passed=int(ok["passed"].sum()),
                scored=int(len(ok)),
                simulated=float(ok["simulated"].median()) if len(ok) else np.nan,
                other_day=float(ok["other_day"].median()) if len(ok) else np.nan,
                same_day=float(cells["same_day"].median()),
            )
        band_rows_.append(row)
        days = own.groupby(["car", "layout_id", "date"], sort=False)[g].median()
        mid = days.groupby(["car", "layout_id"], sort=False)[g].median()
        mid.insert(0, "laps", own.groupby(["car", "layout_id"], sort=False).size())
        mid.insert(1, "days", days.groupby(["car", "layout_id"], sort=False).size())
        mid["mean"] = mid[g].mean(axis=1)
        mid["right_less_left"] = 0.5 * (mid["g_fr"] + mid["g_rr"] - mid["g_fl"] - mid["g_rl"])
        mid = mid.reset_index()
        mid.insert(0, "band", band)
        cell_rows.append(mid)
    by_cell = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    return pd.DataFrame(band_rows_), by_cell


def band_item(heat: pd.DataFrame, obs: pd.DataFrame) -> list:
    """The item gate C3's report carries for the pace response (#207:
    E3, and E1's slow laps)."""
    from .gates import GateItem, md_table

    by_band, by_cell = band_scores(heat, obs)
    if by_band.empty:
        return []
    spread = by_cell.groupby("band", sort=False)["mean"].agg(["median", "min", "max"])
    measured = "; ".join(
        f"{r['band']} ({r['laps']} laps) {r['passed']} of {r['scored']} cells, "
        f"{r.get('simulated', float('nan')):.3f} from the logged lap against "
        f"{r.get('other_day', float('nan')):.3f} for another day's lap of the band, the four-tyre "
        f"mean {spread.loc[r['band'], 'median']:+.3f} in the median cell"
        for r in by_band.to_dict("records")
    )
    return [
        GateItem(
            "Per-tyre heat as the lap gets slower (gates E3 and E1 of #207: the pace response)",
            None,
            measured,
            "a score in gate C3's form (no fixed limits): against a logged lap of the same "
            "band, another day and like lap time",
            "Every clean dry lap within "
            f"{SLOW_MAX:g} × its session's best, in three bands: `fast` the build laps (within "
            f"2 %), `mid` the other clean laps up to {MID_MAX:g} ×, `slow` beyond. Each is solved "
            "to its own lap time by the car's own fit on its own track and its heat set against "
            "the logged lap's own heat in the tyres' frame. A cell is a car at a layout on one "
            "tyre, passing when the simulated lap is no further from the logged lap than a logged "
            "lap of the same band, another day and like lap time is. A simulated slow lap uses "
            "less of the limit everywhere alike; a driven one is slow in places.\n\n"
            + md_table(by_band, floats="{:.4f}")
            + "\n\n**Signed, by band and cell** (`mean`: the four tyres' mean; above zero, the "
            "simulated lap has more heat).\n\n" + md_table(by_cell, floats="{:.3f}"),
        )
    ]


def carried_factors(laps: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """What carrying a frame factor to a layout costs by itself: per car,
    layout and source, the log of the layout's own factor over the one
    carried from the car's other layouts, per tyre (``c_<tyre>``)."""
    where = laps[["car", "layout_id", "session_id", "lap_num", "date"]].drop_duplicates(
        ["session_id", "lap_num"]
    )
    own = where.merge(usable(obs), on=["session_id", "lap_num"], how="inner")
    rows = []
    for (car, layout_id, source), here in own.groupby(["car", "layout_id", "source"], sort=False):
        other = own[
            (own["car"] == car) & (own["layout_id"] != layout_id) & (own["source"] == source)
        ]
        ratio = np.log(_factor(here) / _factor(other))
        if np.isfinite(ratio).all():
            rows.append(
                {
                    "car": car,
                    "layout_id": layout_id,
                    "source": source,
                    "laps": int(len(here)),
                    **{f"c_{c}": float(ratio[k]) for k, c in enumerate(CORNERS)},
                }
            )
    return pd.DataFrame(rows)


def transfer_item(
    arm_laps: pd.DataFrame,
    obs: pd.DataFrame,
    yard: pd.DataFrame,
    laps: pd.DataFrame,
    donors: pd.DataFrame,
) -> list:
    """The item gate C3's report carries for arms C and D of gate E2
    (:func:`transfer_laps`); ``donors`` says what each track from other
    cars was built from (``car, layout_id, donor_laps, donor_days,
    donor_cars``)."""
    from .gates import GateItem, md_table

    per_lap, by_cell = transfers(arm_laps, obs)
    if by_cell.empty:
        return []
    g = list(G_MEASURES)
    # laps arm D could not drive: under the cap the car cannot make the lap time
    lost = arm_laps[(arm_laps["arm"] == "D") & arm_laps["sim_q_fl"].isna()]
    unsolved = lost.groupby(["car", "layout_id"]).size().to_dict()
    cells = []
    for (car, layout_id), grp in by_cell.groupby(["car", "layout_id"], sort=False):

        def worst(what: str) -> float:
            row = grp[grp["what"] == what]
            return float(row[g].abs().to_numpy().max()) if len(row) else float("nan")

        days = int(grp["days"].max())
        donor = donors[(donors["car"] == car) & (donors["layout_id"] == layout_id)]
        d_laps = int(donor["donor_laps"].iloc[0]) if len(donor) else 0
        d_days = int(donor["donor_days"].iloc[0]) if len(donor) else 0
        cells.append(
            {
                "car": car,
                "layout_id": layout_id,
                "days": days,
                "track_from": str(donor["donor_cars"].iloc[0]) if len(donor) else "",
                "donor_laps": d_laps,
                "donor_days": d_days,
                "verdict": bool(days >= 3 and d_days >= 3 and d_laps >= 30),
                "B-A": worst("B-A"),
                "C-A": worst("C-A"),
                "D-A": worst("D-A"),
                "adds": worst("D-B-C+A"),
                "D_no_lap": int(unsolved.get((car, layout_id), 0)),
            }
        )
    table = pd.DataFrame(cells)
    # every arm as paired runs: against a logged lap of another day (gate C3's form)
    scores = arm_scores(arm_laps, obs, yard, laps)
    base = scores.get("A", pd.DataFrame())
    tallies = []
    for arm, cells_g in scores.items():
        got = tally(cells_g)
        row: dict = {
            "arm": arm,
            "what": ARM_NAMES.get(arm, CARRIED_NAMES.get(arm, "the car's other layouts, unscaled")),
            "cells": f"{got['passed']} of {got['scored']}",
            "whole": f"{got['whole']} of {got['groups']}",
        }
        if len(cells_g) and cells_g["scored"].any():
            own = cells_g[cells_g["scored"]]
            row["simulated"] = float(own["simulated"].median())
            row["other_day"] = float(own["other_day"].median())
        if arm != "A" and len(base) and len(cells_g):
            keys = ["car", "layout_id", "level", "measure"]
            both = cells_g.merge(base[keys + ["passed", "scored"]], on=keys, suffixes=("", "_A"))
            both = both[both["scored"] & both["scored_A"]]
            row["won_vs_A"] = int((both["passed"] & ~both["passed_A"]).sum())
            row["lost_vs_A"] = int((~both["passed"] & both["passed_A"]).sum())
        tallies.append(row)
    by_arm = {r["arm"]: r for r in tallies}
    measured = (
        "cells no further from the logged lap than a logged lap of another day (the tyres' "
        "frame): "
        + "; ".join(f"{a} {by_arm[a]['cells']}" for a in (*ARMS, OTHER_LAYOUTS) if a in by_arm)
        + "; against A, "
        + "; ".join(
            f"{a} won {by_arm[a].get('won_vs_A', 0)} lost {by_arm[a].get('lost_vs_A', 0)}"
            for a in ARMS[1:]
            if a in by_arm
        )
        + f"; the car's transfer up to {table['B-A'].max():.3f} on a tyre, the track's up to "
        f"{table['C-A'].max():.3f}, adding within {table['adds'].max():.4f}"
    )
    detail = (
        "Four arms on every dry build lap of the cell, each solved to the lap's own lap time: "
        + "; ".join(f"**{a}** {ARM_NAMES[a]}" for a in ARMS)
        + ". The track from other cars is gate B3's saved track of the layout built without "
        "the car's class, its curvature laid on the own track's nodes (the bank and the grade "
        "stay the own track's). Everything here is in the tyres' frame (simulated heat over "
        "the logged lap's own heat there, no factor).\n\n**Each arm against real laps.** A "
        "cell is a car at a layout at a level of pace on one tyre, passing when the arm's lap "
        "is no further from the logged lap than a logged lap of another day and like lap time "
        "is; `simulated` and `other_day` are the two distances in the median scored cell; "
        "`won_vs_A` and `lost_vs_A` the cells an arm passes and A does not, and the reverse.\n\n"
        + md_table(pd.DataFrame(tallies), floats="{:.4f}")
        + "\n\n**The cells, and the transfers between arms** (`B-A`, `C-A`, `D-A`: the largest "
        "on a tyre of the difference between two arms, lap by lap, the median over days of the "
        "day's median; `adds`: the largest of D-B-C+A). `verdict`: three scored dry days of "
        "the car, and three days and 30 laps of other cars behind the track. `D_no_lap`: laps "
        "arm D has no lap for (under the cap of a use of 1 the unseen car on the other cars' "
        "track cannot make the lap time).\n\n"
        + md_table(table, floats="{:.4f}")
        + "\n\n**Per tyre** (`what`: an arm's log of simulated over logged heat, or a "
        "transfer between arms).\n\n" + md_table(by_cell, floats="{:.4f}")
    )
    carried = pace_carried(arm_laps, obs)
    if len(carried):
        both_unseen = carried[carried["arm"] == "D*"]
        near = int((both_unseen["lap_time_size"] <= both_unseen["lap_time_other_day"]).sum())
        measured += (
            "; the lap time not given (each day's best lap at the car's typical use): "
            + "; ".join(f"{a} {by_arm[a]['cells']}" for a in CARRIED if a in by_arm)
            + f" cells; with nothing given the lap time is {both_unseen['lap_time_size'].median():.3f} "
            f"from the day's best in the median cell where another day's best lap is "
            f"{both_unseen['lap_time_other_day'].median():.3f} ({near} of {len(both_unseen)} "
            "cells no further)"
        )
        detail += (
            "\n\n**The lap time not given.** Each day's best lap of the cell is driven at the "
            "car's typical use of its limit, which is what is carried to a layout the car has "
            "not run: `A*` the car's own fit on its own track, `B*` the car fitted without the "
            "layout, `D*` that car on the track from other cars. Their cells are in the table "
            "of arms above. `lap_time`: the log of the simulated lap time over the day's best "
            "lap (the median over days; `lap_time_size` the median of its size); "
            "`lap_time_other_day`: the same between the day's best lap and another day's best "
            "lap at the layout, which is reality against itself for a lap time; `g_<tyre>`: the "
            "log of that lap's simulated heat over the best lap's own heat.\n\n"
            + md_table(carried, floats="{:.4f}")
        )
    return [
        GateItem(
            "Per-tyre heat on a track from other cars (gate E2 of #207, arms C and D)",
            None,
            measured,
            "a score in gate C3's form (owner, 2026-11-24: no fixed limits; paired against "
            "real laps of another day)",
            detail,
        )
    ]


def report_items(
    root: Path,
    laps: pd.DataFrame,
    shares: dict[str, ShareParams],
    lap_time: pd.Series,
    *,
    tracks_dir: Path | None = None,
    cache_dir: Path | None = None,
    transfer: dict | None = None,
    folds_dir: Path | None = None,
) -> list:
    """The items gate C3's report carries for gate E1 (the day folds) and
    arm B of gate E2 (the layout folds), from gate C3's per-lap frame.
    With ``transfer`` (``tables, rows, tracks, full, shifts, gears, air``
    of gate C3) arms C and D follow, at each layout with a saved track
    from other cars under ``folds_dir`` (``tracks_dir / "folds"`` when it
    is not given)."""
    from .gates import GateItem, md_table

    bands = pd.DataFrame()
    if transfer is not None and transfer.get("index") is not None:
        bands = band_rows(transfer["index"], tuple(transfer["full"]))
    seen_keys = ["car", "session_id", "lap_num"]
    every = pd.concat([laps[seen_keys], bands[seen_keys]] if len(bands) else [laps[seen_keys]])
    obs = observed(root, every, shares, tracks_dir=tracks_dir, cache_dir=cache_dir)
    errs = errors(laps, obs)
    yard = cell_yardsticks(laps, obs, lap_time)
    if errs.empty or yard.empty:
        return []
    held = laps.drop_duplicates(["session_id", "lap_num"])[["session_id", "lap_num"]]
    seen = held.merge(obs, on=["session_id", "lap_num"], how="left")
    left_out = (
        f"{len(held)} held-out laps: {int(seen['hole'].fillna(False).astype(bool).sum())} with a "
        f"discarded window, {int((seen['source'] == 'mixed').sum())} of mixed source and "
        f"{int(seen['source'].isna().sum())} without a heat left out"
    )
    items = []
    for kind, title in (
        ("day", "Per-tyre heat on held-out days (gate E1 of #207)"),
        ("layout", "Per-tyre heat at a layout the car was not fitted on (gate E2 of #207, arm B)"),
    ):
        own = errs[errs["kind"] == kind]
        if own.empty:
            continue
        arms = SIM_MODELS + ((OTHER_LAYOUTS,) if kind == "layout" else ())
        tallies, fit_cells = [], pd.DataFrame()
        for model in arms:
            arm = own[own["model"] == model]
            if arm.empty:
                continue
            row: dict = {"arm": model}
            for tag, measures in (("production", E_MEASURES), ("tyres_frame", G_MEASURES)):
                cells = score(arm, yard, measures)
                got = tally(cells)
                row[f"{tag}_cells"] = f"{got['passed']} of {got['scored']}"
                row[f"{tag}_whole"] = f"{got['whole']} of {got['groups']}"
                if model == "fit" and tag == "production":
                    fit_cells = cells
            tallies.append(row)
        table = signed(own)
        head = tallies[0] if tallies else {}
        paired = ""
        if len(fit_cells) and fit_cells["scored"].any():
            both = fit_cells[fit_cells["scored"]]
            paired = (
                f"; in the median cell the simulated lap is {both['simulated'].median():.3f} "
                f"from the logged lap and a logged lap of another day "
                f"{both['other_day'].median():.3f}"
            )
        detail = (
            "A cell is a car at a layout at a level of pace on one tyre: the size of the log of "
            "simulated over observed heat (the median over days of the day's median lap) "
            "against the same between two logged laps of like lap time from another day and the "
            "same kind of source. `production`: the simulated lap through the fold's frame "
            "factor against production's heat for the lap; `tyres_frame`: no factor, against the "
            "lap's own heat in the tyres' frame. `whole`: car × layout × level groups with every "
            "scored tyre inside (the four tyres of a lap move together). `low_power` and "
            "`other_car` are the fold's planted wrong cars"
            + (
                "; `other_layouts` is the car's median logged heat at its other layouts, "
                "unscaled, with no simulated lap"
                if kind == "layout"
                else ""
            )
            + ": if they pass about as many cells as `fit`, the count cannot tell a right car "
            "from a wrong one and the signed numbers lead.\n\n"
            + md_table(pd.DataFrame(tallies))
            + "\n\n**Signed** (the log of simulated over observed, the median over days of the "
            "day's median lap; `e_`: production's frame, `g_`: the tyres' frame).\n\n"
            + md_table(table, floats="{:.3f}")
        )
        if kind == "layout":
            carried = carried_factors(laps, obs)
            if len(carried):
                detail += (
                    "\n\n**What carrying the frame factor costs by itself** (the log of the "
                    "layout's own factor over the one carried from the car's other layouts; it "
                    "is in `e_` above and not in `g_`).\n\n" + md_table(carried, floats="{:.3f}")
                )
        if len(fit_cells):
            shown = fit_cells.copy()
            shown["level"] = [f"P{lv:g}" for lv in shown["level"]]
            detail += "\n\n**Every cell of the fitted car, production's frame.**\n\n" + md_table(
                shown[
                    ["car", "layout_id", "level", "measure", "laps", "days", "simulated"]
                    + ["other_day", "same_day", "scored", "passed"]
                ],
                floats="{:.3f}",
            )
        items.append(
            GateItem(
                title,
                None,
                f"fitted car {head.get('production_cells', '')} cells in production's frame "
                f"({head.get('production_whole', '')} whole), {head.get('tyres_frame_cells', '')} "
                f"in the tyres' frame; "
                + "; ".join(
                    f"{r['arm']} {r['production_cells']} / {r['tyres_frame_cells']}"
                    for r in tallies[1:]
                )
                + paired
                + f"; {left_out}",
                "a score in gate C3's form (owner, 2026-11-24): the simulated lap's heat no "
                "further from the logged lap's than a logged lap of another day is",
                detail,
            )
        )
    if folds_dir is None and tracks_dir is not None:
        folds_dir = Path(tracks_dir) / "folds"
    if transfer is not None and folds_dir is not None:
        parts, donor_rows = [], []
        for car, full in transfer["full"].items():
            try:
                share = ho.share_for(car, shares)
            except KeyError:
                continue
            if full is None:
                continue
            others = {}
            for layout_id in transfer["tracks"]:
                path = Path(folds_dir) / f"{fold_track_name(car, layout_id)}.json"
                if path.exists():
                    others[layout_id] = trk.TrackModel.load(path)
                    built = others[layout_id].meta.get("built_from") or {}
                    donor_rows.append(
                        {
                            "car": car,
                            "layout_id": layout_id,
                            "donor_laps": int(built.get("n_laps") or 0),
                            "donor_days": int(built.get("n_days") or 0),
                            "donor_cars": ", ".join(built.get("cars") or []),
                        }
                    )
            if not others:
                continue
            parts.append(
                transfer_laps(
                    transfer["tables"],
                    transfer["rows"],
                    transfer["tracks"],
                    others,
                    car,
                    full,
                    share,
                    air=transfer.get("air"),
                    shifts=transfer["shifts"].get(car),
                    gears=transfer["gears"].get(car),
                )
            )
        if parts:
            arm_laps = pd.concat(parts, ignore_index=True)
            items += transfer_item(arm_laps, obs, yard, laps, pd.DataFrame(donor_rows))
    if len(bands) and transfer is not None:
        items += band_item(band_heat(bands, transfer["tracks"], transfer["full"], shares), obs)
    return items


__all__ = [
    "E_MEASURES",
    "G_MEASURES",
    "OTHER_LAYOUTS",
    "SIM_MODELS",
    "carried_factors",
    "cell_yardsticks",
    "errors",
    "fold_factors",
    "observed",
    "report_items",
    "score",
    "signed",
    "tally",
    "usable",
    "yardsticks",
]
