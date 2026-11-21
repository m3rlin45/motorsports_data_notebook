"""Is one track profile every car's? (lap simulation segment B3, issue #204)

**Statistic.** Profile against profile, never lap against profile: the
curvature built from one group of laps against the curvature built from
another, both driven with the *same* speed trace (the held-out group's
per-bin median) through production's own heat chain,
``heat_input.share_heat_rate`` with the held-out car's force shares:
lateral g from ``V²κ``, longitudinal g from the speed trace. The result is
the log ratio of lap-mean heat per tyre, ``Δq``; a gate reads the largest
of the four in magnitude. Beside it: lateral heat by the sign of κ, the
share under 0.004 /m and ``V²Δκ`` rms.

**Ladder.** The same statistic on ever wider splits, so that a difference
between cars can be read against the differences that have nothing to do
with the car:

1. random halves of a car's laps within each day;
2. random halves of a car's days (cells with four days or more);
3. a car's early days against its late days;
4. same class: the KK-F against the KK-SII;
5. other class: each class against every other car at the layout.

All of it is computed on one projection, the layout's build laps on the
build's reference line, by taking the per-bin median course of subsets of
laps; the reference line is a coordinate system and one more
re-projection moves V²κ by 0.0003–0.0013 g (gate B2). The voting
comparisons of rungs 4 and 5 are also rebuilt from scratch, each side on
its own reference line (``dq_max_strict``), and those donor tracks are
saved for C3 and D1.

**Votes.** A comparison between cars counts when the held-out group has
three days and the donor laps three days and thirty laps (#204, #207).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ...tire_etl.kinematics import G_MS2
from ..heat_input import CarFacts, ShareParams, share_heat_rate
from . import geometry as geo
from . import track as trk
from .gates import TRACK_LAYOUTS, GateItem, GateReport, md_table

logger = logging.getLogger(__name__)

CORNERS = ("fl", "fr", "rl", "rr")
#: Cars pooled as one class for rung 5; every other car is its own.
CLASS_OF = {"KK-F": "FJ", "KK-SII": "FJ", "KK-F-V2": "FJ"}
SAME_CLASS_PAIRS = (("KK-F", "KK-SII"), ("KK-SII", "KK-F"))
#: Whose production force shares a car without a fitted model is read
#: through (the statistic needs a lens, not that car's own heat model).
SHARE_OF = {"SFJ": "FJ", "EK9": "Inferno 86", "RX8": "Inferno 86", "718 Cayman": "Inferno 86"}
LONG_WINDOW_M = 12.0
LOW_KAPPA = 0.004
MIN_SUBSET_LAPS = 6
MIN_CELL_LAPS = 12
MIN_FLOOR_DAYS = 4
VOTE_DAYS = 3
VOTE_DONOR_LAPS = 30
ONE_TRACK_MAX = 0.02
STOP_ABOVE = 0.05
PACE_BAND = (1.04, 1.12)
LAP_METRIC_SMOOTH_S = 0.5


def production_shares(root: Path) -> dict[str, ShareParams]:
    """The fitted force shares of ``tire_model.json`` per fitted car, with
    the car aliases the model was fitted under."""
    model = json.loads((Path(root) / "tire_model.json").read_text())
    hi = model.get("heat_input", {})
    out: dict[str, ShareParams] = {}
    for car, fitted in hi.get("fitted_by_car", {}).items():
        facts = hi.get("car_facts_by_car", {}).get(car, {})
        cf = CarFacts(
            p_f=float(facts.get("p_f", 0.5)),
            brake_bias_front=float(facts.get("brake_bias_front", 0.6)),
            driven=str(facts.get("driven", "rear")),
        )
        out[car] = ShareParams(
            p_f=cf.p_f,
            g_transfer_front=float(fitted["g_transfer"]),
            g_transfer_rear=float(fitted["g_transfer"]),
            beta0=cf.beta0,
            beta1=0.0,
            eps_drive=float(fitted.get("eps_drive", 1.0)),
            eps_brake=float(fitted.get("eps_brake", 1.0)),
            driven=cf.driven,
        )
    for raw, fitted_as in model.get("car_aliases", {}).items():
        if fitted_as in out:
            out[raw] = out[fitted_as]
    return out


def share_for(car: str, shares: dict[str, ShareParams]) -> ShareParams:
    """The force shares ``car`` is read through; ``KeyError`` when neither
    the car nor its stand-in (:data:`SHARE_OF`) has any."""
    return shares[car] if car in shares else shares[SHARE_OF[car]]


def longitudinal_g(speed: np.ndarray, ds: float) -> np.ndarray:
    """``dV/dt = V·dV/ds`` (g) of a closed speed trace, the derivative over
    :data:`LONG_WINDOW_M`."""
    k = max(int(round(LONG_WINDOW_M / 2.0 / ds)), 1)
    dvds = (np.roll(speed, -k) - np.roll(speed, k)) / (2 * k * ds)
    return np.asarray(speed * dvds / G_MS2)


def tyre_heat(share: ShareParams, kappa: np.ndarray, speed: np.ndarray, ds: float) -> np.ndarray:
    """Lap-mean heat rate per tyre (``fl, fr, rl, rr``; G·m/s) of a lap on
    the profile ``kappa`` at the speed trace ``speed``."""
    lat = speed * speed * kappa / G_MS2
    long = longitudinal_g(speed, ds)
    dt = ds / np.maximum(speed, 1.0)
    parts = (np.maximum(lat, 0), np.maximum(-lat, 0), np.maximum(long, 0), np.maximum(-long, 0))
    return np.array(
        [float(np.sum(share_heat_rate(share, *parts, speed, c) * dt) / dt.sum()) for c in CORNERS]
    )


def _log_ratio(a: float, b: float) -> float:
    """``log(a / b)``; 0 when neither side has any (a layout that only
    turns one way), NaN when one has."""
    if a <= 0.0 and b <= 0.0:
        return 0.0
    if a <= 0.0 or b <= 0.0:
        return float("nan")
    return math.log(a / b)


def compare_profiles(
    kappa_a: np.ndarray, kappa_b: np.ndarray, speed: np.ndarray, ds: float, share: ShareParams
) -> dict:
    """``a`` against ``b`` at one speed trace: ``dq_<tyre>`` (log ratio of
    lap-mean heat), ``dq_max`` (the largest in magnitude), ``right`` /
    ``left`` (log ratio of lateral heat in right / left turns), ``low``
    (difference of the share of lateral heat under 0.004 /m) and
    ``v2dk_rms_g``."""
    qa, qb = tyre_heat(share, kappa_a, speed, ds), tyre_heat(share, kappa_b, speed, ds)
    dq = np.log(qa / qb)
    w = speed * speed

    def _side(k: np.ndarray, sign: float) -> float:
        return float(np.sum(w * np.maximum(sign * k, 0.0)))

    def _low(k: np.ndarray) -> float:
        heat = w * np.abs(k)
        return float(np.sum(heat[np.abs(k) < LOW_KAPPA]) / np.sum(heat))

    out = {f"dq_{c}": float(v) for c, v in zip(CORNERS, dq)}
    out["dq_max"] = float(np.max(np.abs(dq)))
    out["right"] = _log_ratio(_side(kappa_a, 1.0), _side(kappa_b, 1.0))
    out["left"] = _log_ratio(_side(kappa_a, -1.0), _side(kappa_b, -1.0))
    out["low"] = _low(kappa_a) - _low(kappa_b)
    out["v2dk_rms_g"] = float(np.sqrt(np.mean((w * (kappa_a - kappa_b) / G_MS2) ** 2)))
    return out


def _kappa(proj: trk.Projection, rows: np.ndarray, ds: float) -> np.ndarray:
    return trk._kappa_from_course(proj.course[rows], ds)[0]


def _speed(proj: trk.Projection, rows: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return trk._fill_nan_periodic(geo.nan_median(proj.speed[rows][:, :-1], 0))


def ladder(
    build: trk.TrackBuild,
    shares: dict[str, ShareParams],
    *,
    n_repeat: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Every comparison of the ladder at one layout, one row each:
    ``layout, rung, held_out, donor, laps_held, days_held, laps_donor,
    days_donor, vote`` and the columns of :func:`compare_profiles`. In a
    row the donor's profile is ``a`` and the held-out group's own ``b``,
    on the held-out group's speed and force shares."""
    proj, ds = build.proj, build.model.ds_m
    layout = build.model.layout_key
    cars = np.array([lap.car for lap in build.laps])
    days = np.array([lap.date for lap in build.laps])
    classes = np.array([CLASS_OF.get(c, c) for c in cars])
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    def _add(rung: int, held: str, donor: str, a: np.ndarray, b: np.ndarray, car: str) -> None:
        if a.sum() < MIN_SUBSET_LAPS or b.sum() < MIN_SUBSET_LAPS:
            return
        try:
            share = share_for(car, shares)
        except KeyError:
            return
        res = compare_profiles(_kappa(proj, a, ds), _kappa(proj, b, ds), _speed(proj, b), ds, share)
        n_days_b, n_days_a = len(np.unique(days[b])), len(np.unique(days[a]))
        rows.append(
            {
                "layout": layout,
                "rung": rung,
                "held_out": held,
                "donor": donor,
                "laps_held": int(b.sum()),
                "days_held": n_days_b,
                "laps_donor": int(a.sum()),
                "days_donor": n_days_a,
                "vote": bool(
                    rung >= 4
                    and n_days_b >= VOTE_DAYS
                    and n_days_a >= VOTE_DAYS
                    and a.sum() >= VOTE_DONOR_LAPS
                ),
                **res,
            }
        )

    for car in np.unique(cars):
        cell = cars == car
        if cell.sum() < MIN_CELL_LAPS:
            continue
        cell_days = np.unique(days[cell])
        for _ in range(n_repeat):  # rung 1: halves of laps within days
            a = np.zeros(len(cars), dtype=bool)
            for day in cell_days:
                idx = rng.permutation(np.flatnonzero(cell & (days == day)))
                a[idx[: len(idx) // 2 + (len(idx) % 2 and rng.integers(2))]] = True
            _add(1, car, "other half of each day", a, cell & ~a, car)
        if len(cell_days) >= MIN_FLOOR_DAYS:
            for _ in range(n_repeat):  # rung 2: halves of days
                pick = rng.permutation(cell_days)[: len(cell_days) // 2]
                a = cell & np.isin(days, pick)
                _add(2, car, "other half of the days", a, cell & ~a, car)
            early = cell & np.isin(days, np.sort(cell_days)[: len(cell_days) // 2])
            _add(3, f"{car} late days", "early days", early, cell & ~early, car)
    for donor, held in SAME_CLASS_PAIRS:  # rung 4
        _add(4, held, donor, cars == donor, cars == held, held)
    for cls in np.unique(classes):  # rung 5
        b = classes == cls
        if (~b).sum() == 0:
            continue
        members = np.unique(cars[b])
        lens_car = str(members[np.argmax([np.sum(cars == m) for m in members])])
        _add(5, cls, "every other car", ~b, b, lens_car)
    return pd.DataFrame(rows)


def layout_verdict(lad: pd.DataFrame) -> dict:
    """The gate's reading of one layout's ladder: the floor (P90 of rung 2,
    of rung 1 where no cell has four days), the largest voting ``dq_max``
    of rungs 4 and 5, and the verdict."""
    r2, r1 = lad[lad["rung"] == 2]["dq_max"], lad[lad["rung"] == 1]["dq_max"]
    floor_rung = 2 if len(r2) else 1
    floor = float((r2 if len(r2) else r1).quantile(0.9)) if len(r2) or len(r1) else float("nan")
    votes = lad[(lad["rung"] >= 4) & lad["vote"]]
    out: dict = {
        "floor_rung": floor_rung,
        "floor_p90": floor,
        "rung1_p90": float(r1.quantile(0.9)) if len(r1) else float("nan"),
        "voting": int(len(votes)),
        "between_cars_max": float(votes["dq_max"].max()) if len(votes) else float("nan"),
    }
    stat = out["between_cars_max"]
    if not len(votes):
        out["verdict"] = "no vote"
    elif stat <= floor:
        out["verdict"] = "one track (within the day-to-day floor)"
    elif stat <= ONE_TRACK_MAX:
        out["verdict"] = "one track, bias carried to E1"
    elif stat <= STOP_ABOVE:
        out["verdict"] = "per-class tracks"
    else:
        out["verdict"] = "above the stop bound"
    return out


def strict_comparison(
    laps_donor: list[trk.LapRecord],
    laps_held: list[trk.LapRecord],
    layout: str,
    share: ShareParams,
) -> tuple[dict, trk.TrackModel, np.ndarray]:
    """The same comparison with each side built on its own reference line:
    the result of :func:`compare_profiles`, the donor's track and the
    held-out group's speed trace (on its own track's bins)."""
    donor = trk.build_from_laps(laps_donor, layout, n_boot=0)
    held = trk.build_from_laps(laps_held, layout, n_boot=0)
    n = held.model.n
    frac = np.arange(n) / n
    k_donor = np.interp(
        frac, np.arange(donor.model.n) / donor.model.n, donor.model.kappa_heat, period=1.0
    )
    # one lap of the donor's track, stretched onto the held-out track's length
    k_donor = k_donor * (donor.model.length_m / held.model.length_m)
    res = compare_profiles(
        k_donor, held.model.kappa_heat, held.speed_median, held.model.ds_m, share
    )
    return res, donor.model, held.speed_median


def extra_laps_against_build(
    root: Path,
    build: trk.TrackBuild,
    rows: pd.DataFrame,
    shares: dict[str, ShareParams],
    label: str,
) -> list[dict]:
    """Laps outside the build (``rows`` of the lap index) against the
    build's own laps, car by car: the extra laps' median course on the
    build's reference line against that car's build profile, at the extra
    laps' speed."""
    layout, ds = build.model.layout_key, build.model.ds_m
    out: list[dict] = []
    if rows.empty:
        return out
    laps = trk.load_layout_laps(root, layout, rows, inertial=False)
    if len(laps) < MIN_SUBSET_LAPS:
        return out
    proj = trk.project_laps(laps, build.ref_x, build.ref_y)
    cars = np.array([lap.car for lap in laps])
    build_cars = np.array([lap.car for lap in build.laps])
    for car in np.unique(cars):
        a, b = cars == car, build_cars == car
        if a.sum() < MIN_CELL_LAPS or b.sum() < MIN_CELL_LAPS:
            continue
        try:
            share = share_for(car, shares)
        except KeyError:
            continue
        res = compare_profiles(
            _kappa(proj, a, ds), _kappa(build.proj, b, ds), _speed(proj, a), ds, share
        )
        out.append(
            {
                "layout": layout,
                "what": label,
                "car": car,
                "laps": int(a.sum()),
                "days": len({lap.date for lap, m in zip(laps, a) if m}),
                **res,
            }
        )
    return out


def _lap_metric(
    k_laps: np.ndarray, v_laps: np.ndarray, k_prof: np.ndarray, ds: float
) -> np.ndarray:
    """Per lap, ``log(Σ|V²κ_lap| / Σ|V²κ_profile|)`` at the lap's own speed,
    both lateral accelerations smoothed over :data:`LAP_METRIC_SMOOTH_S`
    of the lap's median speed first (unsmoothed, ``|·|`` turns the lap's
    own noise on the straights into heat and biases the ratio low by
    0.01–0.06)."""
    out = np.full(len(k_laps), np.nan)
    for i, (kl, vl) in enumerate(zip(k_laps, v_laps)):
        ok = np.isfinite(kl) & np.isfinite(vl)
        if ok.mean() < 0.9:
            continue
        v = np.where(ok, vl, 0.0)
        half = max(int(round(LAP_METRIC_SMOOTH_S * float(np.median(vl[ok])) / 2.0 / ds)), 1)
        a_lap = trk._periodic_mean(np.where(ok, v * v * kl, 0.0), half)
        a_prof = trk._periodic_mean(np.where(ok, v * v * k_prof, 0.0), half)
        den = float(np.sum(np.abs(a_prof[ok])))
        if den > 0:
            out[i] = math.log(float(np.sum(np.abs(a_lap[ok]))) / den)
    return out


def per_lap_metric(build: trk.TrackBuild) -> pd.DataFrame:
    """The first revision's lap-against-profile metric, as distance from
    the same-car floor: for each car, its laps against the profile of its
    *own other days* (``own``) and against the profile of *every other
    car* (``others``), medians over its laps, and their difference. A
    bias the two profiles share shows in both; the difference is the car's
    line against the other cars'."""
    from .gates import lap_curvature

    proj, ds = build.proj, build.model.ds_m
    cars = np.array([lap.car for lap in build.laps])
    days = np.array([lap.date for lap in build.laps])
    k_laps = lap_curvature(proj.course, ds, trk.KAPPA_WINDOW_M)
    v_laps = proj.speed[:, :-1]
    rows: list[dict] = []
    for car in np.unique(cars):
        cell = cars == car
        cell_days = np.unique(days[cell])
        if cell.sum() < MIN_CELL_LAPS or len(cell_days) < 2 or (~cell).sum() < MIN_SUBSET_LAPS:
            continue
        own = np.full(len(cars), np.nan)
        for day in cell_days:  # each day against the car's other days
            held, rest = cell & (days == day), cell & (days != day)
            if rest.sum() < MIN_SUBSET_LAPS:
                continue
            own[held] = _lap_metric(k_laps[held], v_laps[held], _kappa(proj, rest, ds), ds)
        others = _lap_metric(k_laps[cell], v_laps[cell], _kappa(proj, ~cell, ds), ds)
        if not np.isfinite(own[cell]).any():
            continue
        rows.append(
            {
                "layout": build.model.layout_key,
                "car": car,
                "laps": int(np.isfinite(others).sum()),
                "days": len(cell_days),
                "own_other_days": float(np.nanmedian(own[cell])),
                "every_other_car": float(np.nanmedian(others)),
                "difference": float(np.nanmedian(others) - np.nanmedian(own[cell])),
            }
        )
    return pd.DataFrame(rows)


def check_track_holdout(
    root: Path,
    *,
    cache_dir: Path | None = None,
    folds_dir: Path | None = None,
    layouts: tuple[str, ...] = TRACK_LAYOUTS,
    n_repeat: int = 20,
) -> GateReport:
    """Gate B3: one profile serves every car that runs the layout, to
    within what the heat model can tell apart."""
    rep = GateReport("B3", "one track for every car (`lapsim check-track-holdout`)")
    shares = production_shares(root)
    index = geo.clean_lap_index(root, cache_dir=cache_dir)
    ladders, verdicts, beside, strict_rows = [], [], [], []
    lap_metrics: list[pd.DataFrame] = []
    for layout in layouts:
        build = trk.build_track_model(root, layout, cache_dir=cache_dir, inertial=False, n_boot=0)
        lad = ladder(build, shares, n_repeat=n_repeat)
        if lad.empty:
            continue
        ladders.append(lad)
        verdicts.append({"layout": layout, **layout_verdict(lad)})
        lap_metrics.append(per_lap_metric(build))

        # The voting comparisons again, each side on its own reference line.
        cars = np.array([lap.car for lap in build.laps])
        classes = np.array([CLASS_OF.get(c, c) for c in cars])
        voting = lad[(lad["rung"] >= 4) & lad["vote"]]
        for rung, held_out, donor, dq_max in zip(
            voting["rung"], voting["held_out"], voting["donor"], voting["dq_max"]
        ):
            held_name = str(held_out)
            if int(rung) == 4:
                held_rows, donor_rows, lens = cars == held_name, cars == str(donor), held_name
            else:
                held_rows = classes == held_name
                donor_rows = ~held_rows
                members = np.unique(cars[held_rows])
                lens = str(members[np.argmax([np.sum(cars == m) for m in members])])
            res, donor_model, speed = strict_comparison(
                [lap for lap, m in zip(build.laps, donor_rows) if m],
                [lap for lap, m in zip(build.laps, held_rows) if m],
                layout,
                share_for(lens, shares),
            )
            strict_rows.append(
                {
                    "layout": layout,
                    "rung": int(rung),
                    "held_out": held_name,
                    "donor": str(donor),
                    "dq_max": float(dq_max),
                    "dq_max_strict": res["dq_max"],
                }
            )
            if folds_dir is not None:
                stem = f"{layout}__without_{held_name.replace(' ', '-')}__rung{int(rung)}"
                donor_model.save(Path(folds_dir) / f"{stem}.json")
                (Path(folds_dir) / f"{stem}__speed.json").write_text(
                    json.dumps(
                        {
                            "layout": layout,
                            "held_out": held_name,
                            "speed_ms": [round(float(v), 3) for v in speed],
                        }
                    )
                )

        # Beside the ladder: the line moving with pace, and wet sessions.
        here = index[(index["layout_id"] == layout) & index["clean"] & ~index["flagged"]]
        ratio = here["lap_time_s"] / here["session_best_s"]
        slow = here[(here["build_reason"] == "slow_lap") & ratio.between(*PACE_BAND)]
        beside += extra_laps_against_build(
            root, build, slow, shares, f"laps at {PACE_BAND[0]}–{PACE_BAND[1]} × session best"
        )
        # Build laps of sessions the wetness model labels wet against the
        # same car's other build laps (the build does not use the label).
        label = index.drop_duplicates("session_id").set_index("session_id")["condition"]
        is_wet = np.array([label.get(lap.session_id) == "wet" for lap in build.laps])
        for car in np.unique(cars):
            a, b = is_wet & (cars == car), ~is_wet & (cars == car)
            if a.sum() < MIN_CELL_LAPS or b.sum() < MIN_CELL_LAPS:
                continue
            res = compare_profiles(
                _kappa(build.proj, a, build.model.ds_m),
                _kappa(build.proj, b, build.model.ds_m),
                _speed(build.proj, a),
                build.model.ds_m,
                share_for(car, shares),
            )
            beside.append(
                {
                    "layout": layout,
                    "what": "build laps of sessions labelled wet",
                    "car": car,
                    "laps": int(a.sum()),
                    "days": len({lap.date for lap, m in zip(build.laps, a) if m}),
                    **res,
                }
            )

    if not ladders:
        rep.items.append(GateItem("Ladder", None, "no layout with enough laps", ""))
        return rep
    lad_all = pd.concat(ladders, ignore_index=True)
    ver = pd.DataFrame(verdicts)
    over_stop = ver[ver["between_cars_max"] > STOP_ABOVE]
    rep.items.append(
        GateItem(
            "Separability",
            bool(len(over_stop) < 2),
            "; ".join(
                (
                    f"{r.layout}: {r.between_cars_max:.3f} against a floor of {r.floor_p90:.3f}, {r.verdict}"
                    if r.voting
                    else f"{r.layout}: no vote"
                )
                for r in ver.itertuples()
            ),
            f"largest per-tyre |Δq| between cars ≤ the day-split P90 (one track) or ≤ "
            f"{ONE_TRACK_MAX} (one track, bias recorded); stop above {STOP_ABOVE} at two layouts",
            md_table(ver, floats="{:.3g}"),
        )
    )
    summary = (
        lad_all.groupby(["layout", "rung"])
        .agg(
            comparisons=("dq_max", "size"),
            dq_max_median=("dq_max", "median"),
            dq_max_p90=("dq_max", lambda s: s.quantile(0.9)),
            dq_max_largest=("dq_max", "max"),
            v2dk_rms_g=("v2dk_rms_g", "median"),
        )
        .reset_index()
    )
    rep.items.append(
        GateItem(
            "Ladder",
            None,
            "largest per-tyre |Δq| by rung, P90 over layouts: "
            + ", ".join(
                f"rung {rung} {lad_all[lad_all['rung'] == rung]['dq_max'].quantile(0.9):.3f}"
                for rung in sorted(int(r) for r in lad_all["rung"].unique())
            ),
            "reported",
            md_table(summary, floats="{:.3g}"),
        )
    )
    between = lad_all[lad_all["rung"] >= 3].drop(columns=["v2dk_rms_g"])
    rep.items.append(
        GateItem(
            "Between eras and cars",
            None,
            f"{int(between['vote'].sum())} voting comparisons of {int((between['rung'] >= 4).sum())} "
            "between cars",
            "reported: rungs 3–5, every comparison",
            md_table(between, floats="{:.3g}"),
        )
    )
    if strict_rows:
        st = pd.DataFrame(strict_rows)
        rep.items.append(
            GateItem(
                "Own reference lines",
                None,
                f"largest change of dq_max {float((st['dq_max_strict'] - st['dq_max']).abs().max()):.4f}",
                "reported: the voting comparisons rebuilt from scratch",
                md_table(st, floats="{:.3g}"),
            )
        )
    lm = pd.concat(lap_metrics, ignore_index=True) if lap_metrics else pd.DataFrame()
    if len(lm):
        rep.items.append(
            GateItem(
                "Per-lap metric",
                None,
                f"laps against their own car's other days: {lm['own_other_days'].min():+.3f} to "
                f"{lm['own_other_days'].max():+.3f}; against every other car, less that: "
                f"{lm['difference'].min():+.3f} to {lm['difference'].max():+.3f}",
                "reported: log ratio of lap-mean |V²κ|, lap against profile, both smoothed "
                f"over {LAP_METRIC_SMOOTH_S} s",
                md_table(lm, floats="{:.3g}"),
            )
        )
    if beside:
        bs = pd.DataFrame(beside)
        cols = [
            "layout",
            "what",
            "car",
            "laps",
            "days",
            "dq_max",
            "right",
            "left",
            "low",
            "v2dk_rms_g",
        ]
        rep.items.append(
            GateItem(
                "Pace and wet sessions",
                None,
                "; ".join(
                    f"{what}: dq_max {grp['dq_max'].min():.3f}–{grp['dq_max'].max():.3f}"
                    for what, grp in bs.groupby("what")
                ),
                "reported (the pace response goes to D2)",
                md_table(bs[cols], floats="{:.3g}"),
            )
        )
    return rep
