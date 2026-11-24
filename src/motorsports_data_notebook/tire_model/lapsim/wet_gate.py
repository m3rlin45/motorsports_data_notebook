"""Gate D2, the wet (#206; ``lapsim check-wet``).

A wet lap is solved to its own lap time, so the wet grip scale, the day's
water level and the driver's use are one product that no simulated lap
depends on. Two things are not absorbed by the lap time, and they are
the arms of this gate, each wet build lap driven four ways with its own
day's powertrain, shift point, gears and air:

- **A**, the null: the dry track, the wet as a lower use and nothing else;
- **B**: A with the wet braking term (:mod:`.wet`: braking falls less
  than cornering), read without the lap's own wet day;
- **C**: A on the wet line, where the layout has a wet track (its
  curvature laid on the dry track's nodes, :func:`wet_line`, so that the
  logged laps are read where they are);
- **D**: both.

What a fold holds out is the wet term: it is read without the lap's own
wet day. The car is the dry car gate C2 saved (no wet lap is in its fit
but for the wet grip scale, which the lap time absorbs).

The measures are gate C3's four, at P75, P90 and P95 of pace among the
car's wet build laps at the layout. The score is against a logged lap of
the **same** wet day (the owner's ruling of 2026-11-24: another wet day has other
water, and only four cells per measure have two wet days at a level),
with the other-day form of gate C3 beside it. Against a same-day lap the
dry gate itself reaches 31 of 104, which is what a count here is read
beside.

The entry rule for a term was written before the first run: nearer than
the null on its own measure (braking for B, apex for C) on a number of
car-days a coin would reach less than one time in twenty, and no more
same-day cells lost than won.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from ..heat_input import ShareParams
from . import car as cm
from . import car_gate as cg
from . import car_holdout as ch
from . import geometry as geo
from . import holdout as ho
from . import samples as smp
from . import solver as sv
from . import track as trk
from . import wet
from .gates import GateItem, GateReport, md_table
from .heat import CORNERS, sim_heat

logger = logging.getLogger(__name__)

ARMS = ("A", "B", "C", "D")
ARM_NAMES = {
    "A": "the dry track, a lower use (the null)",
    "B": "with the wet braking term",
    "C": "on the wet line",
    "D": "both",
}
#: The measure a term is judged on, car-day by car-day.
OWN_MEASURE = {"B": "brake_g", "C": "S3", "D": "S1"}
#: A coin reaches the count less often than this.
SIGN_LEVEL = 0.05


def wet_line(dry: trk.TrackModel, wet_track: trk.TrackModel) -> np.ndarray:
    """The wet track's curvature at the dry track's nodes: each wet node
    laid on the dry reference line (the two share the start/finish gate)
    and the curvature read across, so that a lap driven on it has its
    speed where the logged laps have theirs. The distance along the dry
    line stands in for the wet line's, which is within a few per cent of
    it inside a corner."""
    guess = wet_track.s_m * (dry.length_m / wet_track.length_m)
    s, _ = trk.project_onto(dry.x_m, dry.y_m, wet_track.x_m, wet_track.y_m, guess, closed=True)
    s = np.mod(np.asarray(s, float), dry.length_m)
    kappa = np.nan_to_num(np.asarray(wet_track.kappa_heat, float))
    order = np.argsort(s)
    return np.asarray(np.interp(dry.s_m, s[order], kappa[order], period=dry.length_m))


def sign_count(n: int, level: float = SIGN_LEVEL) -> int:
    """The fewest wins out of ``n`` that a fair coin reaches with a
    probability under ``level`` (``n + 1`` when none does)."""
    for k in range(n + 1):
        if sum(comb(n, j) for j in range(k, n + 1)) / 2.0**n < level:
            return k
    return n + 1


def _same_day(pace: pd.DataFrame) -> pd.DataFrame:
    """:func:`~.car_holdout.attainment` with the same-day yardstick
    beside the other-day one: ``scored_same`` and ``passed_same``."""
    got = ch.attainment(pace)
    ok = np.isfinite(got["simulated"]) & np.isfinite(got["same_day"])
    got["scored_same"] = ok
    got["passed_same"] = ok & (got["simulated"] <= got["same_day"])
    return got


def run_wet(
    tables: smp.CarTables,
    rows: pd.DataFrame,
    tracks: dict[str, trk.TrackModel],
    wet_tracks: dict[str, trk.TrackModel],
    car: str,
    drive: cg.Drive,
    share: ShareParams | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(laps, yardsticks, terms)`` of one car's wet build laps: per lap
    and arm the four measures and the use it solved to; the yardsticks of
    :func:`~.car_holdout.pace_yardsticks` among the wet laps; and per wet
    day the wet readings a fold without it gives (``terms``)."""
    own = ch.car_only(tables, car)
    model = drive.model
    lap_time = rows.set_index(["session_id", "lap_num"])["lap_time_s"]
    c_drag = float(model.power["c_drag"])
    lat_level = float(model.meta.get("lat_level_braking") or model.grip["mu_y"])
    lap_rows, yard_rows, term_rows = [], [], []
    wet_layouts = sorted(
        str(k) for k in own.speed_laps.loc[own.speed_laps["basis"] == "wet", "layout_id"].unique()
    )
    for layout_id in wet_layouts:
        if layout_id not in tracks:
            continue
        track = tracks[layout_id]
        layout = ch.Layout.of(track)
        cell, v_cell = own.lap_speeds(car, layout_id, basis="wet")
        if cell.empty:
            continue
        keys = pd.MultiIndex.from_frame(cell[["session_id", "lap_num"]])
        cell = cell.assign(lap_time_s=lap_time.reindex(keys).to_numpy())
        yard = ch.pace_yardsticks(layout, cell, v_cell)
        keyed = cell.iloc[yard["lap"].to_numpy()][["session_id", "lap_num", "date"]]
        yard = pd.concat([keyed.reset_index(drop=True), yard.drop(columns="lap")], axis=1)
        yard.insert(0, "car", car)
        yard.insert(1, "layout_id", layout_id)
        yard_rows.append(yard)
        dry_road = cg.profile_of(drive, track)
        roads = {"A": dry_road, "B": dry_road}
        if layout_id in wet_tracks:
            on_wet = replace(dry_road, kappa=wet_line(track, wet_tracks[layout_id]))
            roads.update({"C": on_wet, "D": on_wet})
        for date in sorted(cell["date"].unique()):
            held_mask = (cell["date"] == date).to_numpy()
            held = cell[held_mask].reset_index(drop=True)
            v_held = v_cell[held_mask]
            terms = wet.wet_terms(own.apex, own.brake, c_drag, lat_level, without=str(date))
            term = terms.brake_term
            term_rows.append(
                {"car": car, "layout_id": layout_id, "date": str(date), **terms.to_dict()}
            )
            recs = cg._records(held)
            base = [cg._lap_car(drive, row) for row in recs]
            target = [float(row["lap_time_s"]) for row in recs]
            for arm, road in roads.items():
                if arm in ("B", "D") and not np.isfinite(term):
                    continue
                cars = (
                    [replace(c, wet_brake=float(term)) for c in base] if arm in ("B", "D") else base
                )
                got = sv.solve_laps(road, cars, target, guess=cg.typical_use(drive), wet=True)
                for i in range(len(held)):
                    row = {
                        "car": car,
                        "layout_id": layout_id,
                        "session_id": held["session_id"].iloc[i],
                        "lap_num": int(held["lap_num"].iloc[i]),
                        "date": str(date),
                        "arm": arm,
                        "lap_time_s": target[i],
                        "use": float(got.u[i]) if got.ok[i] else np.nan,
                        "term": float(term) if np.isfinite(term) else np.nan,
                    }
                    if got.ok[i]:
                        row.update(ch.lap_stats(layout, got.v_node_ms[i], v_held[i]))
                        if share is not None:
                            # the wet lap's heat per tyre (gate E2's wet clause, #207)
                            sim = sv.lap_from_nodes(
                                road, cars[i], got.v_node_ms[i], float(got.u[i])
                            )
                            for tyre, q in zip(CORNERS, sim_heat(sim, share)):
                                row[f"sim_q_{tyre}"] = float(q)
                    lap_rows.append(row)
    laps = pd.DataFrame(lap_rows)
    yards = pd.concat(yard_rows, ignore_index=True) if yard_rows else pd.DataFrame()
    return laps, yards, pd.DataFrame(term_rows)


def by_car_day(laps: pd.DataFrame) -> pd.DataFrame:
    """Per car, layout, wet day and arm: the median over the day's laps of
    each measure's size."""
    if laps.empty:
        return laps
    sized = laps.copy()
    cols = list(ch.YARDSTICK_MEASURES)
    sized[cols] = sized[cols].abs()
    return (
        sized.groupby(["car", "layout_id", "date", "arm"], sort=False)[cols].median().reset_index()
    )


def against_null(days: pd.DataFrame, arm: str, measure: str) -> tuple[int, int, pd.DataFrame]:
    """``(nearer, of, table)``: the car-days on which ``arm`` is nearer
    the logged lap than the null on ``measure``."""
    wide = days.pivot_table(index=["car", "layout_id", "date"], columns="arm", values=measure)
    if arm not in wide.columns or "A" not in wide.columns:
        return 0, 0, pd.DataFrame()
    both = wide[["A", arm]].dropna()
    both = both.assign(nearer=both[arm] < both["A"]).reset_index()
    return int(both["nearer"].sum()), int(len(both)), both


def wet_heat(laps: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Per wet build lap of the null arm: ``g_<tyre>``, the log of the
    simulated lap's heat over the logged wet lap's own heat in the tyres'
    frame (no factor), with the lap's keys, ``source`` and its heats."""
    from . import heat_match as hm

    null = laps[laps["arm"] == "A"]
    both = null.merge(hm.usable(obs), on=["session_id", "lap_num"], how="inner")
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in CORNERS:
            both[f"g_{c}"] = np.log(both[f"sim_q_{c}"] / both[f"t_{c}"])
    return both.replace([np.inf, -np.inf], np.nan)


def wet_heat_items(
    root: Path,
    laps: pd.DataFrame,
    shares: dict[str, ShareParams],
    rows: pd.DataFrame,
    *,
    tracks_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> list[GateItem]:
    """Gate E2's wet clause as a report: the heat of each wet build lap
    driven as the null drives it (the same car at a lower use on the dry
    track) against the logged wet lap's own heat in the tyres' frame, per
    tyre and right less left, with a logged lap of the same wet day
    beside it (the owner's form for the wet)."""
    from . import heat_match as hm

    obs = hm.observed(root, laps, shares, tracks_dir=tracks_dir, cache_dir=cache_dir)
    got = wet_heat(laps, obs)
    if got.empty:
        return []
    g = list(hm.G_MEASURES)
    cell = ["car", "layout_id"]
    days = got.groupby(cell + ["date"], sort=False)[g].median()
    signed = days.groupby(cell, sort=False)[g].median()
    signed.insert(0, "laps", got.groupby(cell, sort=False).size())
    signed.insert(1, "days", days.groupby(cell, sort=False).size())
    signed["mean"] = signed[g].mean(axis=1)
    signed["right_less_left"] = 0.5 * (
        signed["g_fr"] + signed["g_rr"] - signed["g_fl"] - signed["g_rl"]
    )
    signed["worst"] = signed[g].abs().max(axis=1)
    signed = signed.reset_index()
    # the same-day yardstick, per cell
    parts = []
    for (car, layout_id), own in got.groupby(cell, sort=False):
        yard = hm.yardsticks(own.reset_index(drop=True))
        if len(yard):
            yard.insert(0, "car", str(car))
            yard.insert(1, "layout_id", str(layout_id))
            parts.append(yard)
    tally = "no laps of one wet day to set a cell against"
    if parts:
        sized = got.copy()
        sized[g] = sized[g].abs()
        pace = ch.by_pace(sized, pd.concat(parts, ignore_index=True), hm.G_MEASURES)
        if len(pace):
            cells = ch.attainment(pace, hm.G_MEASURES)
            ok = np.isfinite(cells["simulated"]) & np.isfinite(cells["same_day"])
            passed = ok & (cells["simulated"] <= cells["same_day"])
            tally = f"{int(passed.sum())} of {int(ok.sum())} cells no further than a same-day lap"
    three = signed[signed["days"] >= 3]
    return [
        GateItem(
            "The wet laps' heat per tyre (gate E2 of #207, the wet clause)",
            None,
            f"worst tyre {signed['worst'].median():.3f} in the median cell (up to "
            f"{signed['worst'].max():.3f}); the four-tyre mean {signed['mean'].min():+.3f} to "
            f"{signed['mean'].max():+.3f}, right less left "
            f"{signed['right_less_left'].min():+.3f} to {signed['right_less_left'].max():+.3f}; "
            f"{len(three)} cells have three wet days; {tally}",
            "reported: a wet lap is the same car at a lower use on the dry track (gate D2), "
            "against the wet lap's own heat in the tyres' frame",
            "Per car and layout, the median over wet days of the day's median lap of the log of "
            "simulated over logged heat per tyre; `mean` the four tyres' mean, `right_less_left` "
            "the right-hand tyres less the left-hand ones, `worst` the largest size. The car is "
            "the dry car on every dry lap (the wet laps are not in its fit); the lap time is "
            "the wet lap's own. A cell is scored against a logged lap of the same wet day and "
            "like lap time (owner, 2026-11-24).\n\n" + md_table(signed, floats="{:.3f}"),
        )
    ]


def check_wet(
    root: Path,
    *,
    cache_dir: Path | None = None,
    tracks_dir: Path | None = None,
    cars_dir: Path | None = None,
) -> GateReport:
    """Gate D2 (module docstring)."""
    rep = GateReport("D2", "the wet: what a lap time does not absorb (`lapsim check-wet`)")
    tables = cm.as_families(smp.car_tables(root, cache_dir=cache_dir, tracks_dir=tracks_dir))
    rows = cm.as_family(smp.build_laps(geo.clean_lap_index(root, cache_dir=cache_dir)))
    _, fits = cm.table_clauses(root, tables)
    try:
        shares: dict[str, ShareParams] | None = ho.production_shares(root)
    except FileNotFoundError:
        shares = None
    air = cm.air_state(root, geo.load_sessions(root))
    layouts = sorted(str(k) for k in rows.loc[rows["basis"] == "wet", "layout_id"].unique())
    tracks = {
        k: smp.load_track(root, k, tracks_dir=tracks_dir, cache_dir=cache_dir) for k in layouts
    }
    wet_tracks = {}
    for k in layouts:
        path = Path(tracks_dir) / f"{k}__wet.json" if tracks_dir is not None else None
        if path is not None and path.exists():
            wet_tracks[k] = trk.TrackModel.load(path)
    lap_parts, yard_parts, term_parts, read_rows, day_parts = [], [], [], [], []
    for car in smp.MODEL_CARS:
        path = Path(cars_dir) / f"{car.replace(' ', '_')}.json" if cars_dir is not None else None
        if path is None or not path.exists() or car not in fits:
            continue
        model = cm.CarModel.load(path)
        work = fits[car][1]
        drive = cg.Drive(model, air=air, shifts=cg.day_shifts(work), gears=cg.day_gears(work))
        try:
            share = ho.share_for(car, shares) if shares is not None else None
        except KeyError:
            share = None
        laps, yards, terms = run_wet(tables, rows, tracks, wet_tracks, car, drive, share)
        if laps.empty:
            continue
        lap_parts.append(laps)
        yard_parts.append(yards)
        term_parts.append(terms)
        own = ch.car_only(tables, car)
        whole = wet.wet_terms(
            own.apex,
            own.brake,
            float(model.power["c_drag"]),
            float(model.meta.get("lat_level_braking") or model.grip["mu_y"]),
        )
        read_rows.append(
            {"car": car, **whole.to_dict(), "grip_scale_as_fitted": model.wet.get("grip_scale")}
        )
        if len(whole.days):
            day_parts.append(whole.days.assign(car=car))
    if not lap_parts:
        rep.items.append(GateItem("Wet build laps", False, "none", "a car with wet laps"))
        return rep
    laps = pd.concat(lap_parts, ignore_index=True)
    yards = pd.concat(yard_parts, ignore_index=True)
    terms = pd.concat(term_parts, ignore_index=True)

    # 1. The readings.
    reads = pd.DataFrame(read_rows)
    level_table = ""
    if day_parts:
        levels = pd.concat(day_parts, ignore_index=True)
        use = (
            laps[laps["arm"] == "A"]
            .groupby(["car", "layout_id", "date"])["use"]
            .median()
            .rename("use_of_wet_limit")
            .reset_index()
        )
        levels = levels.merge(use, on=["car", "layout_id", "date"], how="outer")
        level_table = (
            "\n\n**Each wet day.** `ratio`: the day's own wet cornering ratio (its water level "
            "as far as the logs give it); `use_of_wet_limit`: the median use its laps solve to "
            "on the null, as a share of the car's limit times `grip_scale_as_fitted`. The two "
            "and the driver's use are one product: nothing is solved with them.\n\n"
            + md_table(
                levels[
                    ["car", "layout_id", "date", "ratio", "corners", "passes"]
                    + ["use_of_wet_limit"]
                ],
                floats="{:.3f}",
            )
        )
    rep.items.append(
        GateItem(
            "The wet readings: cornering, braking and the wet braking term",
            None,
            "; ".join(
                f"{r['car']} cornering {r['cornering']} braking {r['braking']} "
                f"term {r['brake_term']}"
                for r in read_rows
            ),
            "reported",
            "Per car, its wet passes pooled over its wet days and layouts: the expected best "
            "of m wet passes over the expected best of the same m dry passes at the corner (m "
            f"half the wet passes, {wet.M_MIN} to {wet.M_MAX}), the median over corners; the "
            "same on the straight-line stops; `brake_term` their quotient. In the arms each "
            "lap's term is read without its own wet day (below).\n\n"
            + md_table(reads, floats="{:.3f}")
            + "\n\n**The term each fold reads** (without the day named).\n\n"
            + md_table(
                terms[
                    ["car", "layout_id", "date", "cornering", "braking", "brake_term"]
                    + ["corners", "places"]
                ],
                floats="{:.3f}",
            )
            + level_table,
        )
    )

    # 2. The score per arm, against same-day and other-day laps.
    cells = {}
    for arm in ARMS:
        own_laps = laps[laps["arm"] == arm]
        if own_laps.empty:
            continue
        got = _same_day(ch.by_pace(own_laps, yards))
        got["level"] = [f"P{lv:g}" for lv in got["level"]]
        cells[arm] = got.set_index(["car", "layout_id", "level", "measure"])
    score_rows = []
    base = cells["A"]
    for arm, got in cells.items():
        both = got.join(base[["passed_same", "scored_same"]], rsuffix="_A", how="inner")
        common = both[both["scored_same"] & both["scored_same_A"]]
        score_rows.append(
            {
                "arm": arm,
                "what": ARM_NAMES[arm],
                "same_day": f"{int(got['passed_same'].sum())} of {int(got['scored_same'].sum())}",
                "other_day": f"{int(got['passed'].sum())} of {int(got['scored'].sum())}",
                "won_vs_A": int((common["passed_same"] & ~common["passed_same_A"]).sum()),
                "lost_vs_A": int((~common["passed_same"] & common["passed_same_A"]).sum()),
                "nearer_cells": int(
                    (common["simulated"] < base.loc[common.index, "simulated"]).sum()
                ),
                "cells_in_common": int(len(common)),
            }
        )
    by_measure = (
        pd.concat({arm: got for arm, got in cells.items()}, names=["arm"])
        .reset_index()
        .groupby(["arm", "measure"], sort=False)
        .agg(same_day=("passed_same", "sum"), of=("scored_same", "sum"))
        .reset_index()
    )
    detail = (
        "A cell is one car at one layout at one level of pace on one measure; it passes when "
        "the simulated lap is no further from the logged lap than a logged lap of the same wet "
        "day and like lap time is (`same_day`), or of another wet day (`other_day`, gate C3's "
        "form). `won_vs_A` and `lost_vs_A`: same-day cells the arm passes and the null does not, "
        "and the reverse, among the cells both have; `nearer_cells`: cells where the arm's "
        "figure is smaller than the null's. Against a same-day lap the dry gate reaches 31 of "
        "104.\n\n"
        + md_table(pd.DataFrame(score_rows))
        + "\n\n**By measure** (same-day).\n\n"
        + md_table(by_measure)
        + "\n\n**Every cell of the null and of the arm with both terms.**\n\n"
        + md_table(
            pd.concat({a: cells[a] for a in ("A", "D") if a in cells}, names=["arm"]).reset_index()[
                ["arm", "car", "layout_id", "level", "measure", "laps", "days", "simulated"]
                + ["same_day", "other_day", "passed_same", "passed"]
            ],
            floats="{:.3f}",
        )
    )
    rep.score = f"{score_rows[0]['same_day']} same-day cells on the null"
    rep.items.append(
        GateItem(
            "The wet laps against a logged lap of the same wet day",
            None,
            "; ".join(
                f"{r['arm']} {r['same_day']} (other day {r['other_day']})" for r in score_rows
            ),
            "a score, read beside the dry gate's 31 of 104 against same-day laps",
            detail,
        )
    )

    # 3. The terms against the null, car-day by car-day, and the entry rule.
    days = by_car_day(laps)
    won = {str(r["arm"]): (int(r["won_vs_A"]), int(r["lost_vs_A"])) for r in score_rows}  # type: ignore[call-overload]
    for arm in ("B", "C", "D"):
        if arm not in cells:
            continue
        measure = OWN_MEASURE[arm]
        nearer, of, table = against_null(days, arm, measure)
        need = sign_count(of)
        enters = bool(of and nearer >= need and won[arm][1] <= won[arm][0])
        others = []
        for name in ch.YARDSTICK_MEASURES:
            wins, total, _ = against_null(days, arm, name)
            others.append(f"{ch.MEASURE_NAMES[name]} {wins} of {total}")
        rep.items.append(
            GateItem(
                f"Arm {arm} ({ARM_NAMES[arm]}) against the null, car-day by car-day",
                None,
                f"nearer on {ch.MEASURE_NAMES[measure]} on {nearer} of {of} car-days "
                f"({need} needed); same-day cells won {won[arm][0]}, lost {won[arm][1]}: "
                + ("meets the entry rule" if enters else "does not meet the entry rule"),
                "entry rule written before the run: a count a coin reaches under one time in "
                "twenty, and no more same-day cells lost than won",
                "The day's median lap (every wet build lap of the day), sizes compared. On every "
                "measure: " + "; ".join(others) + ".\n\n" + md_table(table, floats="{:.4f}"),
            )
        )
    # Gate E2's wet clause (#207): the wet laps' heat per tyre, on the null.
    if shares is not None and "sim_q_fl" in laps.columns:
        rep.items.extend(
            wet_heat_items(root, laps, shares, rows, tracks_dir=tracks_dir, cache_dir=cache_dir)
        )
    return rep
