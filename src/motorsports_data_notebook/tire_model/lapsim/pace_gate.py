"""Gate D1, the pace table (#206; ``lapsim check-pace-table``).

A table of every model car at every layout that has a track, built from
the car models gate C2 saved, and held to a direct solve: twenty lap
times evenly between the table's ends, each solved with the table's own
car and put through the heat chain, against the table read at the same
lap time. The limit is the issue's and was written before the run: per
tyre within 1 %. It is asked under two sets of force shares (the car's
own and another car's), because the table hands over laps and not heat.

Reported beside it: each table's ends, support and what it assumes of the
car; what the assumed powertrain is worth (the same lap time under the
car's other powertrain); and where the car's own logged laps fall in the
table (how many are faster than the lap at a use of 1).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..heat_input import ShareParams
from . import car as cm
from . import geometry as geo
from . import holdout as ho
from . import pace
from . import samples as smp
from . import solver as sv
from . import track as trk
from .gates import GateItem, GateReport, md_table
from .heat import CORNERS, sim_heat

logger = logging.getLogger(__name__)

#: Interpolated per-tyre heat against a direct solve (size of the log ratio).
HEAT_MAX = 0.01
#: The same for the use read off the table.
USE_MAX = 0.01
#: Lap times between a table's ends at which it is held to a direct solve.
CHECK_TIMES = 20
PIT_FIELDS = ("s_join_m", "v_join_ms", "v_join_p10_ms", "v_join_p90_ms", "rolling_s", "n")


def interpolation_error(
    table: pace.PaceTable, shares: dict[str, ShareParams], *, times: int = CHECK_TIMES
) -> dict:
    """The table against a direct solve at ``times`` lap times evenly
    between its ends: ``heat`` (the largest size of the log ratio of
    interpolated over direct heat, over the tyres, the lap times and the
    share sets), ``heat_by`` (the same per share set), ``use`` (likewise
    for the use read off the table) and ``unsolved`` (lap times the direct
    solve did not reach) and ``out`` (the out-lap's time and per-tyre heat,
    table against the out-lap driven at the directly solved use, under the
    first share set; NaN without a pit exit)."""
    at = np.linspace(table.lap_time_s[-1], table.lap_time_s[0], times + 2)[1:-1]
    use, laps = pace.direct(table, at)
    solved = [i for i, lap in enumerate(laps) if lap is not None]
    by: dict[str, float] = {}
    for name, share in shares.items():
        want = np.vstack([sim_heat(laps[i], share) for i in solved])  # type: ignore[arg-type]
        got = table.heat(at[solved], share)
        by[name] = float(np.abs(np.log(got / want)).max())
    read = np.array([table.use_at(float(t)) for t in at[solved]])
    out = float("nan")
    if table.v_out_ms is not None and solved:
        # the out-lap at the use the direct solve found, against the table's
        share = next(iter(shares.values()))
        driven = pace.out_laps(table.profile, table.driven, use[solved], table.pit)
        part = pace._out_profile(
            table.profile, pace._join_node(table.profile, table.pit), whole=False
        )
        worst = 0.0
        for row, i in enumerate(solved):
            lap = sv.lap_from_nodes(part, table.driven, driven[row], float(use[i]))  # type: ignore[index]
            got_out = table.read_out(float(at[i]), share)
            assert got_out is not None
            worst = max(
                worst,
                float(np.abs(np.log(got_out[1] / sim_heat(lap, share))).max()),
                abs(math.log(got_out[0] / lap.lap_time_s)),
            )
        out = worst
    return {
        "heat": max(by.values()) if by else float("nan"),
        "heat_by": by,
        "out": out,
        "use": float(np.abs(np.log(read / use[solved])).max()) if solved else float("nan"),
        "unsolved": len(laps) - len(solved),
    }


def load_models(cars_dir: Path) -> dict[str, cm.CarModel]:
    """The car models gate C2 saved, by car."""
    out = {}
    for name in smp.MODEL_CARS:
        path = Path(cars_dir) / f"{name.replace(' ', '_')}.json"
        if path.exists():
            out[name] = cm.CarModel.load(path)
    return out


def load_tracks(tracks_dir: Path) -> dict[str, trk.TrackModel]:
    """The dry tracks gate B2 saved, by layout."""
    return {
        p.stem: trk.TrackModel.load(p)
        for p in sorted(Path(tracks_dir).glob("*.json"))
        if "__" not in p.stem
    }


def check_pace_table(
    root: Path,
    *,
    tracks_dir: Path,
    cars_dir: Path,
    out_dir: Path | None = None,
) -> GateReport:
    """Gate D1's pace-table clause, and the tables written to ``out_dir``
    (``<car>__<layout>.json``)."""
    rep = GateReport("D1T", "the pace table (`lapsim check-pace-table`)")
    models, tracks = load_models(cars_dir), load_tracks(tracks_dir)
    if not models or not tracks:
        rep.items.append(
            GateItem("Car models and tracks", False, "none saved", "gates B2 and C2 run first")
        )
        return rep
    shares = ho.production_shares(root)
    rows = cm.as_family(smp.build_laps(geo.clean_lap_index(root)))
    dry = rows[rows["basis"] == "dry"]
    table_rows, engine_rows = [], []
    for car, model in models.items():
        try:
            own = ho.share_for(car, shares)
        except KeyError:
            logger.info("pace table: no force shares for %s", car)
            continue
        other = next((s for s in shares.values() if s != own), own)
        for layout, track in tracks.items():
            table = pace.build_table(model, track)
            if out_dir is not None:
                table.save(Path(out_dir) / f"{car.replace(' ', '_')}__{layout}.json")
            err = interpolation_error(table, {"own": own, "other": other})
            logged = dry.loc[
                (dry["car"] == car) & (dry["layout_id"] == layout), "lap_time_s"
            ].to_numpy(float)
            status = [table.status(float(t)) for t in logged]
            wet_logged = rows.loc[
                (rows["basis"] == "wet") & (rows["car"] == car) & (rows["layout_id"] == layout),
                "lap_time_s",
            ].to_numpy(float)
            wet_status = [table.status(float(t)) for t in wet_logged]
            sup = table.support
            table_rows.append(
                {
                    "car": car,
                    "layout_id": layout,
                    "powertrain": table.assumes["powertrain"],
                    "slow_s": float(table.lap_time_s[0]),
                    "at_one_s": float(table.lap_time_s[int(np.argmin(np.abs(table.use - 1.0)))]),
                    "fastest_s": table.fastest_lap_s,
                    "max_driver_s": table.max_driver_lap_s,
                    "between": int(
                        ((logged < table.fastest_lap_s) & (logged >= table.max_driver_lap_s)).sum()
                    ),
                    "use_min": sup.get("use_min"),
                    "use_max": sup.get("use_max"),
                    "run_here": layout in (model.meta.get("use_range") or {}),
                    "heat_own": err["heat_by"]["own"],
                    "heat_other": err["heat_by"]["other"],
                    "use": err["use"],
                    "out": err["out"],
                    "out_part_s": (
                        float(out_k[0][-1]) if (out_k := table.out_knots(own)) is not None else None
                    ),
                    "unsolved": err["unsolved"],
                    "logged": len(logged),
                    "no_solution": status.count(pace.NO_SOLUTION),
                    "outside": status.count(pace.OUTSIDE_SUPPORT),
                    "wet_logged": len(wet_logged),
                    "wet_outside": len(wet_logged) - wet_status.count(pace.OK),
                }
            )
            # what the assumed powertrain is worth at the layout's median logged lap
            engines = list((model.power.get("engines") or {}))
            if len(engines) > 1 and logged.size:
                t_mid = float(np.median(logged))
                base = table.heat(t_mid, own)
                for name in engines:
                    if name == table.assumes["powertrain"]:
                        continue
                    alt = pace.build_table(model, track, powertrain=name)
                    ratio = np.log(alt.heat(t_mid, own) / base)
                    engine_rows.append(
                        {
                            "car": car,
                            "layout_id": layout,
                            "table": table.assumes["powertrain"],
                            "other": name,
                            "lap_s": t_mid,
                            "use_table": table.use_at(t_mid),
                            "use_other": alt.use_at(t_mid),
                            **{c: float(r) for c, r in zip(CORNERS, ratio)},
                        }
                    )
    got = pd.DataFrame(table_rows)
    if got.empty:
        rep.items.append(GateItem("Tables", False, "none built", "a car with force shares"))
        return rep
    worst = got.loc[got[["heat_own", "heat_other"]].max(axis=1).idxmax()]
    heat_max = float(got[["heat_own", "heat_other"]].max().max())
    rep.items.append(
        GateItem(
            "Interpolated per-tyre heat against a direct solve",
            bool(heat_max <= HEAT_MAX and int(got["unsolved"].sum()) == 0),
            f"{heat_max:.4f} at most ({worst['car']} at {worst['layout_id']}); "
            f"{len(got)} tables, {CHECK_TIMES} lap times each, two share sets",
            f"<= {HEAT_MAX} (size of the log ratio, every tyre) and every lap time solved",
            "`heat_own` and `heat_other`: the largest size of the log ratio of the table's heat "
            "over the heat of the lap solved directly at the same lap time, over the four tyres "
            "and the lap times, with the car's own force shares and with another car's. `use`: "
            "the same for the use read off the table. `slow_s` and `fastest_s` are the "
            "table's ends (a use of "
            f"{pace.USE_SLOW:g}, and the knob's end lap: the cornering at the limit and the "
            "braking at the car's own level, past a use of 1); `at_one_s` the lap at a use of "
            "1; `max_driver_s` the max driver's lap beside it (the driver's habits off); "
            "`between`: the logged laps faster than the end lap and no faster than the max "
            "driver's, which no knob answers. `use_min` and `use_max`: the support, the car's logged "
            "range of use (at the layout where `run_here`, else over the layouts it has run). "
            "`logged`: the car's dry build laps at the layout; `no_solution`: those faster than "
            "the table's fastest lap (the knob's end lap); `outside`: those the table flags.\n\n"
            + md_table(got, floats="{:.4f}"),
        )
    )
    use_max = float(got["use"].max())
    rep.items.append(
        GateItem(
            "The use read off the table against a direct solve",
            bool(use_max <= USE_MAX),
            f"{use_max:.4f} at most",
            f"<= {USE_MAX}",
        )
    )
    with_out = got.dropna(subset=["out"])
    if len(with_out):
        out_max = float(with_out["out"].max())
        rep.items.append(
            GateItem(
                "The out-lap's time and per-tyre heat against the out-lap driven directly",
                bool(out_max <= HEAT_MAX),
                f"{out_max:.4f} at most; {len(with_out)} of {len(got)} tables have a pit exit",
                f"<= {HEAT_MAX}",
                "`out` in the table above: the part of a lap from the pit exit's join to the "
                "line, begun at the join speed of the logged out-laps and driven at the use of a "
                "flying lap of the given lap time; `out_part_s` is how long it takes at a use of "
                "1.\n\n"
                + md_table(
                    pd.DataFrame(
                        [
                            {"layout_id": k, **{f: v.meta["pit"].get(f) for f in PIT_FIELDS}}
                            for k, v in tracks.items()
                            if v.meta.get("pit")
                        ]
                    ),
                    floats="{:.1f}",
                ),
            )
        )
    faster = int(got["no_solution"].sum())
    rep.items.append(
        GateItem(
            "Logged laps faster than the table's fastest lap",
            None,
            f"{faster} of {int(got['logged'].sum())} dry build laps; "
            f"{int(got['outside'].sum())} flagged outside the support; of "
            f"{int(got['wet_logged'].sum())} wet build laps {int(got['wet_outside'].sum())} "
            "are not answered `ok`",
            "reported",
            "A table is made with the car's latest powertrain, its own shift point and standard "
            "air; a logged lap was driven with its day's. A lap the table has no solution for "
            "is one that its own day's car solves to a use near or above 1 (five of 2,231 "
            "held-out laps in gate C3), or one of an earlier powertrain that the table's car "
            "cannot be compared with at all. The wet laps are read off the same table (a wet "
            "lap is a lower use of the same car on the dry track, gate D2), and the support's "
            "slow end is the slowest of the car's wet build laps.",
        )
    )
    if engine_rows:
        eng = pd.DataFrame(engine_rows)
        span = float(eng[list(CORNERS)].abs().max().max())
        rep.items.append(
            GateItem(
                "What the assumed powertrain is worth",
                None,
                f"up to {span:.3f} in a tyre's heat at the layout's median logged lap time",
                "reported",
                "The log ratio of per-tyre heat, the car's other powertrain over the table's, "
                "at one lap time (the median of the car's logged laps at the layout), with the "
                "use each needs for it.\n\n" + md_table(eng, floats="{:.4f}"),
            )
        )
    notes = pd.DataFrame(
        [
            {
                "car": car,
                **{
                    k: v
                    for k, v in pace.assumed_car(model)[1].items()
                    if k not in ("car", "gears", "basis")
                },
            }
            for car, model in models.items()
        ]
    )
    rep.items.append(
        GateItem(
            "What a table assumes of the car",
            None,
            "the latest powertrain, its own shift point and gears, standard air, dry",
            "stated",
            md_table(notes.fillna(""), floats="{:.4g}"),
        )
    )
    return rep
