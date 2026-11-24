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

from pathlib import Path

import numpy as np
import pandas as pd

from ..heat_input import ShareParams
from . import car_gate as cg
from . import car_holdout as ch
from . import heat_gate as hg
from . import holdout as ho
from .heat import CORNERS

#: The measures: log of simulated over observed heat per tyre, in
#: production's frame through the factor (``e``) and in the tyres' (``g``).
E_MEASURES = tuple(f"e_{c}" for c in CORNERS)
G_MEASURES = tuple(f"g_{c}" for c in CORNERS)
#: Fewest laps a fold's frame factor is read from.
FACTOR_MIN_LAPS = 8
#: The simulated arms of a fold, and the logged one of a layout fold.
SIM_MODELS = ("fit", "low_power", "other_car")
OTHER_LAYOUTS = "other_layouts"
#: #207's fixed limits, kept as reference numbers: in-sample and on the
#: worst tyre of a cell the car has not run.
REF_IN_SAMPLE = 0.03
REF_UNSEEN = 0.05


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


def errors(laps: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """One row per held-out lap, fold kind and arm: ``e_<tyre>`` and
    ``g_<tyre>`` (module docstring).

    ``laps`` is gate C3's per-lap frame (``car, kind, layout_id,
    session_id, lap_num, date`` with the simulated heats ``sim_q_<tyre>``
    of the fitted car and ``sim_q_<tyre>_<name>`` of the planted ones;
    not gate C3's own ``h_<tyre>``, which is a heat error); ``obs``
    is :func:`observed` of the same laps, usable or not. ``model`` is one
    of :data:`SIM_MODELS` or, on a layout fold, :data:`OTHER_LAYOUTS`
    (the car's median logged heat at its other layouts from the same
    kind of source, unscaled: no simulated lap)."""
    keys = ["car", "kind", "layout_id", "session_id", "lap_num", "date"]
    cols = keys + ["source", "model", *E_MEASURES, *G_MEASURES]
    if laps.empty or "sim_q_fl" not in laps.columns:
        return pd.DataFrame(columns=cols)
    good = usable(obs)
    both = laps.merge(good, on=["session_id", "lap_num"], how="inner")
    where = laps[["car", "layout_id", "session_id", "lap_num", "date"]].drop_duplicates(
        ["session_id", "lap_num"]
    )
    car_obs_all = where.merge(good, on=["session_id", "lap_num"], how="inner")
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


def report_items(
    root: Path,
    laps: pd.DataFrame,
    shares: dict[str, ShareParams],
    lap_time: pd.Series,
    *,
    tracks_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> list:
    """The items gate C3's report carries for gate E1 (the day folds) and
    arm B of gate E2 (the layout folds), from gate C3's per-lap frame."""
    from .gates import GateItem, md_table

    obs = observed(root, laps, shares, tracks_dir=tracks_dir, cache_dir=cache_dir)
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
    for kind, title, ref in (
        ("day", "Per-tyre heat on held-out days (gate E1 of #207)", REF_IN_SAMPLE),
        (
            "layout",
            "Per-tyre heat at a layout the car was not fitted on (gate E2 of #207, arm B)",
            REF_UNSEEN,
        ),
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
        fit_signed = table[table["model"] == "fit"]
        worst = fit_signed[list(E_MEASURES)].abs().max(axis=1)
        inside = int((worst <= ref).sum())
        head = tallies[0] if tallies else {}
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
                + f"; worst tyre within {ref:g} (the fixed limit of #207, a reference) in "
                f"{inside} of {len(fit_signed)} car × layout cells; {left_out}",
                "a score in gate C3's form (owner, 2026-11-24): the simulated lap's heat no "
                "further from the logged lap's than a logged lap of another day is",
                detail,
            )
        )
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
