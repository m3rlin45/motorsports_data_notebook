"""Segment F (#208): the cold pressure an entry of the pressure model
leads to, and the entries that are set against each other.

The calculators recommend a cold pressure from an *entry*: what
``tire_model.json`` holds for a (track, car, condition) beside the car's
constants. An entry is a typical lap time, the typical heat per tyre at
it, a pace curve (heat against lap time, used as a ratio) and an out-lap
(its rolling time and heat). Gate F asks of a heat source what entry it
would write, places that entry in a fold's model and reads the
recommendation as the calculators do (``validate._calculator_lap_inputs``,
its interpolation included).

**Heat is of the fold's own shares.** Leaving a cell out moves the same
laps' heat by up to 11 % per tyre (#208, finding 4), so nothing here takes
a heat computed under another fit: a logged entry is built from laps whose
heat the fold model's shares gave (``validate._prepare_fold_laps``), a
simulated one from a table's exported laps split through the same shares
(:class:`~.export.TableExport`), brought to production's frame by a frame
factor read under them too (:func:`frame_factors`).

**The recommended cold pressure** (:func:`recommend`): for a pit-exit
stint and a tyre, the cold pressure that gives a stated hot pressure at
the end of flying lap N (5, the calculators' default), through the fold
model's K and tau (the compound's where the session has a human or seed
label), from the stint's own pit-exit temperature and the session's
weather. No TPMS pressure enters, and the stint need not have reached lap
N.

Entries, by who writes them:

- :func:`logged_entry`: production's own builders on a set of logged laps
  (a day's, a cell's without a day): what a logged entry is.
- :func:`day_entry`: a session as it was driven, the median of its
  *ordinary* flying laps (:func:`mark_laps`) at their median lap time.
- :class:`SimSource`: a simulated lap's, from a pace table's export.
- :func:`other_layouts_entry`: the car's own heat at its other layouts,
  unscaled (#260's last step), and the fold model as it stands, which is
  the fallback as coded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..energy_balance import P_ATM_BAR, gay_lussac_cold_pressure_bar, warmup_two_stage_c
from ..heat_input import ShareParams
from ..predict import RAW_CAR_TABLES
from ..validate import FoldLaps, _calculator_lap_inputs, _fold_constants
from .export import TableExport
from .heat import CORNERS, LapBins, lap_heat

#: The flying lap a recommendation is for: the calculators' default, and
#: the second one printed beside it.
FLYING_LAP_N = 5
FLYING_LAPS = (5, 10)
#: A session's *ordinary* flying laps: from this flying lap on, not a
#: cool-down lap (over this multiple of the stint's best on the full lap
#: list), with a heat on every tyre. At least this many make a session's
#: own heat.
ORDINARY_FROM_LAP = 4
COOL_DOWN = 1.12
MIN_ORDINARY_LAPS = 3
#: Another day's heat at a given lap time is the median of this many of
#: its ordinary laps nearest that lap time.
NEAR_LAPS = 5
#: Fewest logged laps a frame factor is read from (gate E's).
FACTOR_MIN_LAPS = 8

OK = "ok"
OUTSIDE_SUPPORT = "outside_support"
NO_SOLUTION = "no_solution"


# ------------------------------------------------------------------ entries


@dataclass(frozen=True)
class Entry:
    """What ``tire_model.json`` holds for one (track, car, condition).

    ``q`` is the typical heat per tyre (``fl, fr, rl, rr``; G·m/s) at
    ``lap_time_s``. ``curve_t`` (rising lap times) and ``curve_q`` (nodes
    × 4) are the pace curve, read linearly between nodes and as a ratio
    to its own value at ``lap_time_s``; without one the calculators fall
    back to the exponent. ``out_s`` and ``out_q`` are the out-lap's
    rolling time and mean heat; without one the fold model's own lookup
    answers. ``status`` is how a simulated entry's lap time was answered
    (``ok``, ``outside_support``, ``no_solution``)."""

    lap_time_s: float
    q: np.ndarray
    curve_t: np.ndarray | None = None
    curve_q: np.ndarray | None = None
    out_s: float | None = None
    out_q: np.ndarray | None = None
    source: str = "logged"
    n_laps: int = 0
    status: str = OK
    #: The values a model's rows carried beside the per-tyre ones (their
    #: means), kept so that an entry read from a model and placed again is
    #: the rows it was read from, to the last bit.
    extra: dict = field(default_factory=dict, compare=False)

    def at_lap_time(self, lap_time_s: float) -> "Entry":
        """The same heat as the typical heat of another lap time: an
        entry that is fed there unscaled."""
        return replace(self, lap_time_s=float(lap_time_s), curve_t=None, curve_q=None)

    def scaled(self, factor: float) -> "Entry":
        """Every heat of the entry times ``factor`` (a planted error)."""
        return replace(
            self,
            q=self.q * factor,
            out_q=None if self.out_q is None else self.out_q * factor,
            extra={},
        )


def _key(row: dict, track: str, car: str, condition: str) -> bool:
    return bool(
        row["track_canonical"] == track and row["car"] == car and row["condition"] == condition
    )


def entry_of(model: dict[str, Any], track: str, car: str, condition: str) -> Entry | None:
    """The entry a model holds for (track, car, condition); None without
    a heat and a lap time there."""
    heat = next(
        (r for r in model["g2_typ_by_track_car_cond"] if _key(r, track, car, condition)), None
    )
    lap = next(
        (r for r in model["lap_time_typ_by_track_car_cond"] if _key(r, track, car, condition)),
        None,
    )
    if heat is None or lap is None:
        return None
    per = heat.get("q_typ_by_corner") or {}
    q = np.array([float(per.get(c, heat["g2_typ"])) for c in CORNERS])
    extra: dict = {"g2_typ": heat["g2_typ"], "n_lap_time": lap["n_laps_used"]}
    curve_t: np.ndarray | None = None
    curve_q: np.ndarray | None = None
    curve = heat.get("g2_vs_lap_time")
    if isinstance(curve, dict):
        curve_t = np.array([float(x) for x in curve["lap_time_s"]])
        by = curve.get("g2_by_corner") or {}
        curve_q = np.column_stack(
            [np.array([float(x) for x in (by.get(c) or curve["g2"])]) for c in CORNERS]
        )
        extra["curve"] = {k: v for k, v in curve.items() if k not in ("lap_time_s", "g2_by_corner")}
    out_s = out_q = None
    out = next(
        (
            r
            for r in model.get("outlap_typ_by_track_car_cond") or []
            if _key(r, track, car, condition)
        ),
        None,
    )
    if out is not None:
        out_s = float(out["outlap_moving_s"])
        per_out = out.get("outlap_q_by_corner") or {}
        out_q = np.array([float(per_out.get(c, out["outlap_g2"])) for c in CORNERS])
        extra["outlap_g2"] = out["outlap_g2"]
        extra["n_out"] = out["n_laps_used"]
        extra["out_by_corner"] = "outlap_q_by_corner" in out
    extra["by_corner"] = "q_typ_by_corner" in heat
    return Entry(
        lap_time_s=float(lap["lap_time_typ_s"]),
        q=q,
        curve_t=curve_t,
        curve_q=curve_q,
        out_s=out_s,
        out_q=out_q,
        n_laps=int(heat["n_laps_used"]),
        extra=extra,
    )


def _raw_cars_of(model: dict[str, Any], car: str) -> set[str]:
    """The raw cars ``car_aliases`` pools under ``car``."""
    return {raw for raw, fit in (model.get("car_aliases") or {}).items() if fit == car != raw}


def _without_raw_rows(model: dict[str, Any], track: str, car: str, condition: str) -> dict:
    """The raw-car entry tables (``predict.RAW_CAR_TABLES``) of ``model``
    without the rows of the raw cars pooled under ``car`` at (track,
    condition); only the tables the model has."""
    raws = _raw_cars_of(model, car)
    out: dict[str, list[dict]] = {}
    for name in RAW_CAR_TABLES.values():
        if name in model:
            out[name] = [
                r
                for r in model[name] or []
                if not (
                    r["track_canonical"] == track
                    and r["condition"] == condition
                    and r["car"] in raws
                )
            ]
    return out


def place(
    model: dict[str, Any],
    track: str,
    car: str,
    condition: str,
    entry: Entry,
    *,
    raw: bool = True,
) -> dict:
    """``model`` with ``entry`` as its (track, car, condition): a shallow
    copy whose three entry tables hold the entry's rows in place of any
    the model had there (and no out-lap row when the entry has none).
    Every constant, and every other cell, is the model's own.

    ``raw``: the entry answers the cell for the raw cars pooled under
    ``car`` too, whose own rows there (#383) are taken out; without it they
    stay, and a lookup by a raw car still reads them first."""
    extra = entry.extra
    heat: dict[str, Any] = {
        "track_canonical": track,
        "car": car,
        "condition": condition,
        "g2_typ": extra.get("g2_typ", float(np.mean(entry.q))),
        "n_laps_used": int(entry.n_laps),
    }
    if entry.curve_t is not None and entry.curve_q is not None:
        kept = extra.get("curve") or {
            "g2": [float(x) for x in np.mean(entry.curve_q, axis=1)],
            "n_laps": int(entry.n_laps),
            "method": entry.source,
        }
        heat["g2_vs_lap_time"] = {
            "lap_time_s": [float(x) for x in entry.curve_t],
            **kept,
            "g2_by_corner": {
                c: [float(x) for x in entry.curve_q[:, k]] for k, c in enumerate(CORNERS)
            },
        }
    if extra.get("by_corner", True):
        heat["q_typ_by_corner"] = {c: float(entry.q[k]) for k, c in enumerate(CORNERS)}
    lap = {
        "track_canonical": track,
        "car": car,
        "condition": condition,
        "lap_time_typ_s": float(entry.lap_time_s),
        "n_laps_used": int(extra.get("n_lap_time", entry.n_laps)),
    }
    out_rows = []
    if entry.out_s is not None and entry.out_q is not None:
        row: dict[str, Any] = {
            "track_canonical": track,
            "car": car,
            "condition": condition,
            "outlap_moving_s": float(entry.out_s),
            "outlap_g2": extra.get("outlap_g2", float(np.mean(entry.out_q))),
            "n_laps_used": int(extra.get("n_out", entry.n_laps)),
        }
        if extra.get("out_by_corner", True):
            row["outlap_q_by_corner"] = {c: float(entry.out_q[k]) for k, c in enumerate(CORNERS)}
        out_rows.append(row)

    def _swap(table: list[dict], rows: list[dict]) -> list[dict]:
        kept_rows = [r for r in table if not _key(r, track, car, condition)]
        return sorted(
            kept_rows + rows, key=lambda r: (r["track_canonical"], r["car"], r["condition"])
        )

    out = dict(model)
    if raw:
        out.update(_without_raw_rows(model, track, car, condition))
    out["g2_typ_by_track_car_cond"] = _swap(model["g2_typ_by_track_car_cond"], [heat])
    out["lap_time_typ_by_track_car_cond"] = _swap(model["lap_time_typ_by_track_car_cond"], [lap])
    out["outlap_typ_by_track_car_cond"] = _swap(
        model.get("outlap_typ_by_track_car_cond") or [], out_rows
    )
    return out


def without_entry(model: dict[str, Any], track: str, car: str, condition: str) -> dict:
    """``model`` with nothing for (track, car, condition), nor for the raw
    cars pooled under ``car`` there: the lookups' fallback chain then
    answers for it."""
    out = dict(model)
    out.update(_without_raw_rows(model, track, car, condition))
    for name in (
        "g2_typ_by_track_car_cond",
        "lap_time_typ_by_track_car_cond",
        "outlap_typ_by_track_car_cond",
    ):
        out[name] = [r for r in model.get(name) or [] if not _key(r, track, car, condition)]
    return out


# ------------------------------------------------- the recommended pressure


def stints_of(prep: FoldLaps, targets: dict[tuple[str, str], float]) -> pd.DataFrame:
    """One row per held-out stint: ``session_id, stint_id, track_canonical,
    car, condition, t_eff_c, compound`` (the human or seed label; None
    without one), ``target_hot_bar`` (:func:`target_hot`) and per tyre
    ``t_start_<tyre>``, the pit-exit temperature (NaN where the stint is
    not anchored at a pit exit on that tyre)."""
    rows = []
    for (sid, stint), grp in prep.full.groupby(["session_id", "stint_id"], sort=True):
        first = grp.iloc[0]
        cond = str(first.get("condition", "dry"))
        compound = prep.label_by_session.get(str(sid))
        row: dict[str, Any] = {
            "session_id": str(sid),
            "stint_id": int(stint),  # type: ignore[call-overload]
            "track_canonical": str(first["track_canonical"]),
            "car": str(first["car"]),
            "condition": cond,
            "t_eff_c": float(first["t_eff_c"]) if pd.notna(first["t_eff_c"]) else float("nan"),
            "compound": compound,
            "target_hot_bar": target_hot(targets, str(first["car"]), compound),
        }
        for c in CORNERS:
            pit = str(first.get(f"anchor_kind_{c}") or "") == "pit_exit"
            t0 = first.get(f"t_start_{c}")
            row[f"t_start_{c}"] = float(t0) if pit and pd.notna(t0) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def target_hot(targets: dict[tuple[str, str], float], car: str, compound: str | None) -> float:
    """The hot pressure a recommendation is for: the team's target for the
    car and the session's compound, else the car's first listed target
    (NaN for a car with none). It only sets the unit: the same on every
    side of a comparison."""
    if compound is not None and (car, compound) in targets:
        return float(targets[(car, compound)])
    own = [bar for (name, _), bar in targets.items() if name == car]
    return float(own[0]) if own else float("nan")


def recommend(
    model: dict[str, Any],
    stints: pd.DataFrame,
    *,
    lap_time_s: float | None = None,
    n: int = FLYING_LAP_N,
) -> pd.DataFrame:
    """The cold pressure the calculators would recommend for each stint
    and tyre of ``stints`` (:func:`stints_of`) from ``model``'s entries and
    constants, at the end of flying lap ``n``: ``session_id, stint_id,
    corner, cold_bar, t_hot_c, q, lap_time_s``.

    ``lap_time_s`` is the lap time typed in; without one the entry's own
    typical lap time is the plan, as in the calculators. The curve is fed
    as ``validate._calculator_lap_inputs`` feeds it (the entry's heat
    along its pace curve, ``n`` laps of that lap time after the entry's
    out-lap), from the stint's pit-exit temperature, which is also the
    cold side of the gas law."""
    from ..predict import _lookup_lap_time

    constants = _fold_constants(model)
    rows = []
    for stint in stints.to_dict("records"):
        track, car, cond = stint["track_canonical"], stint["car"], stint["condition"]
        fed = lap_time_s
        if fed is None:
            fed = _lookup_lap_time(model, track, car, cond)[0]
        for c in CORNERS:
            t_start = stint[f"t_start_{c}"]
            K, tau = constants(car, c, cond, stint["compound"])
            if not np.isfinite(t_start) or not np.isfinite(stint["t_eff_c"]):
                continue
            if K is None or tau is None or not np.isfinite(stint["target_hot_bar"]):
                continue
            g2, t_fly, _scale, out_s, out_g2 = _calculator_lap_inputs(
                model, track, car, cond, n, float(fed), with_outlap=True, corner=c
            )
            _after, t_hot = warmup_two_stage_c(
                t_outlap_s=out_s,
                g2_outlap=out_g2,
                t_flying_s=t_fly,
                g2_flying=g2,
                t_eff_c=float(stint["t_eff_c"]),
                k_kelvin_per_g2=K,
                tau_sec=tau,
                t_start_c=float(t_start),
            )
            cold = gay_lussac_cold_pressure_bar(
                target_hot_pressure_bar=float(stint["target_hot_bar"]),
                t_hot_c=t_hot,
                t_cold_c=float(t_start),
                p_atm_bar=P_ATM_BAR,
            )
            rows.append(
                {
                    "session_id": stint["session_id"],
                    "stint_id": stint["stint_id"],
                    "corner": c,
                    "cold_bar": float(cold),
                    "t_hot_c": float(t_hot),
                    "q": float(g2),
                    "lap_time_s": float(fed),
                }
            )
    return pd.DataFrame(
        rows, columns=["session_id", "stint_id", "corner", "cold_bar", "t_hot_c", "q", "lap_time_s"]
    )


# ------------------------------------------------------------ logged laps


def mark_laps(prep: FoldLaps, raw_laps: pd.DataFrame) -> pd.DataFrame:
    """The held-out laps (``prep.full``) with what #208's rows rest on:

    - ``flying``: not an out-lap;
    - ``stint_best_s``: the stint's best lap on the full lap list
      (``raw_laps``: every lap of the dataset, usable or not; out-laps
      apart);
    - ``cool_down``: a flying lap over :data:`COOL_DOWN` × that;
    - ``after_cool_down``: at or after the stint's first cool-down lap;
    - ``after_dropped``: after a lap that cleaning dropped from the stint
      (the stint's clock and its lap count then skip it, #270);
    - ``ordinary``: a flying lap from flying lap
      :data:`ORDINARY_FROM_LAP` on that is not a cool-down lap and has a
      heat on every tyre."""
    df = prep.full.sort_values(["session_id", "stint_id", "lap_num"]).reset_index(drop=True)
    is_out = df["_is_out"].astype(bool) if "_is_out" in df.columns else pd.Series(False, df.index)
    df["flying"] = ~is_out
    raw = raw_laps[~raw_laps["is_outlap"].fillna(False).astype(bool)]
    best = raw.groupby(["session_id", "stint_id"])["lap_time_s"].min().rename("stint_best_s")
    df = df.merge(best.reset_index(), on=["session_id", "stint_id"], how="left")
    df["cool_down"] = df["flying"] & (df["lap_time_s"] > COOL_DOWN * df["stint_best_s"])
    stint = df.groupby(["session_id", "stint_id"], sort=False)
    df["after_cool_down"] = stint["cool_down"].cummax().astype(bool)
    gap = stint["lap_num"].diff().fillna(1.0) > 1.0
    df["after_dropped"] = gap.groupby([df["session_id"], df["stint_id"]]).cummax().astype(bool)
    heat = df[[f"q_lap_{c}" for c in CORNERS]].notna().all(axis=1)
    df["ordinary"] = (
        df["flying"] & (df["lap_within_stint"] >= ORDINARY_FROM_LAP) & ~df["cool_down"] & heat
    )
    return df


def _single(found: dict) -> Any:
    return next(iter(found.values())) if len(found) == 1 else None


def logged_entry(
    root: Path, laps: pd.DataFrame, share: ShareParams, *, curve: bool = True
) -> Entry | None:
    """The entry production's own builders write from ``laps`` (prepared
    laps of one track, car and condition, their heat of the fold's
    shares; out-laps among them): the median lap time and the median heat
    per tyre of the flying laps, their pace curve where five laps carry
    one (``curve=False`` skips it, for an entry that is only ever fed at
    its own lap time) and the median from-pit out-lap. None without a
    heat on every tyre."""
    from ..pace_sim import build_pace_model
    from ..warmup_table import (
        _build_lap_time_typ,
        _build_outlap_typ_with_corners,
        _build_q_typ_per_corner,
        _flying_laps,
    )

    if laps.empty:
        return None
    flying = _flying_laps(laps)
    flying = flying[flying["t_eff_c"].notna() & (flying["condition"] != "unknown")]
    typ = _single(_build_lap_time_typ(flying))
    heat = _single(_build_q_typ_per_corner(flying))
    if typ is None or heat is None:
        return None
    per, n = heat
    curve_t = curve_q = None
    if curve:
        car = str(flying["car"].iloc[0])
        got = _single(build_pace_model(root, flying, {car: share}))
        if got is not None:
            curve_t = np.array(got["lap_time_s"], float)
            curve_q = np.column_stack([got["g2_by_corner"][c] for c in CORNERS])
    out_s = out_q = None
    base, per_out = _build_outlap_typ_with_corners(laps)
    out_per = _single(per_out)
    if out_per is not None and _single(base) is not None:
        out_s = float(_single(base)[0])
        out_q = np.array([out_per[c] for c in CORNERS])
    return Entry(
        lap_time_s=float(typ[0]),
        q=np.array([per[c] for c in CORNERS]),
        curve_t=curve_t,
        curve_q=curve_q,
        out_s=out_s,
        out_q=out_q,
        n_laps=int(n),
    )


def day_entry(root: Path, laps: pd.DataFrame, share: ShareParams) -> Entry | None:
    """A session as it was driven (``laps``: its rows of :func:`mark_laps`):
    the median heat per tyre of its ordinary flying laps at their median
    lap time, with the session's own median out-lap. None with fewer than
    :data:`MIN_ORDINARY_LAPS` ordinary laps."""
    own = laps[laps["ordinary"]]
    if len(own) < MIN_ORDINARY_LAPS:
        return None
    out = logged_entry(root, laps, share, curve=False)
    return Entry(
        lap_time_s=float(own["on_track_s"].median()),
        q=np.array([float(own[f"q_lap_{c}"].median()) for c in CORNERS]),
        out_s=None if out is None else out.out_s,
        out_q=None if out is None else out.out_q,
        n_laps=int(len(own)),
        source="day",
    )


def near_heat(laps: pd.DataFrame, lap_time_s: float) -> tuple[np.ndarray, float] | None:
    """Another session's heat at a given lap time: the median per tyre of
    its :data:`NEAR_LAPS` ordinary laps nearest ``lap_time_s``, and the
    median of their lap times (the pair's lap-time gap is that over
    ``lap_time_s``). None with fewer than :data:`MIN_ORDINARY_LAPS`."""
    own = laps[laps["ordinary"]]
    if len(own) < MIN_ORDINARY_LAPS:
        return None
    near = own.loc[(own["on_track_s"] - lap_time_s).abs().sort_values().index[:NEAR_LAPS]]
    q = np.array([float(near[f"q_lap_{c}"].median()) for c in CORNERS])
    return q, float(near["on_track_s"].median())


# --------------------------------------------------------- simulated laps


def heats_frame(
    keys: pd.DataFrame,
    sensor: dict[tuple[str, int], tuple[LapBins, str, bool]],
    tyres: dict[tuple[str, int], LapBins],
    share: ShareParams,
) -> pd.DataFrame:
    """Per lap of ``keys`` (``session_id, lap_num`` and whatever else):
    production's heat on the sensor's series (``q_<tyre>``, with ``source``
    and ``hole``) and the lap's heat in the tyres' frame (``t_<tyre>``),
    both split from the laps' bins through ``share``."""
    rows = []
    for rec in keys.to_dict("records"):
        key = (str(rec["session_id"]), int(rec["lap_num"]))
        if key not in sensor:
            continue
        bins, source, hole = sensor[key]
        q = lap_heat(bins, share)
        t = lap_heat(tyres[key], share) if key in tyres else np.full(len(CORNERS), np.nan)
        row: dict[str, Any] = {str(k): v for k, v in rec.items()}
        row.update(source=source, hole=bool(hole))
        row.update({f"q_{c}": float(q[k]) for k, c in enumerate(CORNERS)})
        row.update({f"t_{c}": float(t[k]) for k, c in enumerate(CORNERS)})
        rows.append(row)
    return pd.DataFrame(rows)


def frame_factors(heats: pd.DataFrame) -> dict[str, np.ndarray]:
    """``source -> factor per tyre``: what brings a simulated lap's heat
    (the tyres' frame) to production's, from the logged laps of
    ``heats`` (:func:`heats_frame`; a fold gives the car's laps at its
    other layouts). Per kind of source the per-tyre median over the laps
    of production's heat over the tyres'-frame heat; laps with a hole or
    of mixed source are left out, and a source with fewer than
    :data:`FACTOR_MIN_LAPS` laps has no factor."""
    out: dict[str, np.ndarray] = {}
    if heats.empty:
        return out
    ok = (~heats["hole"].astype(bool)) & heats["source"].isin(["imu", "gps"])
    for source, pool in heats[ok].groupby("source"):
        q = pool[[f"q_{c}" for c in CORNERS]].to_numpy(float)
        t = pool[[f"t_{c}" for c in CORNERS]].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log(q / t)
        r = r[np.isfinite(r).all(axis=1)]
        if len(r) >= FACTOR_MIN_LAPS:
            out[str(source)] = np.asarray(np.exp(np.median(r, axis=0)))
    return out


@dataclass
class SimSource:
    """A pace table's export split through one fit's shares: what a
    simulated entry is read from. ``heat`` is knots × 4 in the tyres'
    frame; ``out_part_s`` and ``out_heat`` the out-lap's part from the
    join to the line (None without a pit exit)."""

    export: TableExport
    heat: np.ndarray
    out_part_s: np.ndarray | None = None
    out_heat: np.ndarray | None = None
    #: The shares the export was split through.
    share: ShareParams | None = None

    @classmethod
    def of(cls, export: TableExport, share: ShareParams) -> "SimSource":
        out = export.out_knots(share)
        return cls(
            export=export,
            heat=export.heat_knots(share),
            out_part_s=None if out is None else out[0],
            out_heat=None if out is None else out[1],
            share=share,
        )

    def status(self, lap_time_s: float) -> str:
        """How the table answers a lap time (:mod:`.pace`): ``ok``,
        ``outside_support`` or ``no_solution`` (faster than its fastest
        lap, or no lap time at all)."""
        exp = self.export
        if not lap_time_s >= float(exp.lap_time_s[-1]):
            return NO_SOLUTION
        lo, hi = exp.support.get("use_min"), exp.support.get("use_max")
        if lo is None or hi is None or lap_time_s > float(exp.lap_time_s[0]):
            return OUTSIDE_SUPPORT
        return OK if float(lo) <= exp.use_at(lap_time_s) <= float(hi) else OUTSIDE_SUPPORT

    def entry(
        self,
        factor: np.ndarray,
        *,
        use: float | None = None,
        lap_time_s: float | None = None,
        nodes: int | np.ndarray | None = None,
        source: str = "sim",
    ) -> Entry | None:
        """The entry a simulated lap writes, in production's frame
        (``factor`` per tyre, :func:`frame_factors`).

        Its typical lap is the lap of ``use`` (the car's fed use: the
        entry supplies its own lap time) or of ``lap_time_s``; its heat
        there is read log-log between the knots, as the table is. The
        pace curve is the knots themselves (rising lap times), or with
        ``nodes`` that many nodes from 0.75 to 1.5 × the typical lap time
        read off the table, which is the grid the calculators' logged
        curves have (an array: those multiples of the typical lap time,
        so that two cars' entries can share their nodes). The out-lap is the track's rolling time to the pit
        exit's join, with no heat, and the knot's part from the join to
        the line. None where the table has no lap for the use, or the
        factor is not known."""
        exp = self.export
        if not np.isfinite(factor).all():
            return None
        typ = exp.lap_time_at(float(use)) if lap_time_s is None and use is not None else lap_time_s
        if typ is None or not np.isfinite(typ):
            return None
        hot = self.heat * factor
        status = self.status(float(typ))
        # faster than the table there is no lap: the fastest knot's heat stands in, flagged
        read = max(float(typ), float(exp.lap_time_s[-1]))
        q = exp.at(read, hot)
        if nodes is None:
            curve_t, curve_q = exp.lap_time_s[::-1].copy(), hot[::-1].copy()
        else:
            grid = np.linspace(0.75, 1.5, nodes) if isinstance(nodes, int) else np.asarray(nodes)
            curve_t = float(typ) * grid
            curve_q = np.vstack(
                [exp.at(max(float(t), float(exp.lap_time_s[-1])), hot) for t in curve_t]
            )
        out_s = out_q = None
        if self.out_part_s is not None and self.out_heat is not None:
            part = float(exp.at(read, self.out_part_s))
            out_s = float(exp.rolling_s) + part
            out_q = exp.at(read, self.out_heat * factor) * part / out_s
        return Entry(
            lap_time_s=float(typ),
            q=np.asarray(q, float),
            curve_t=curve_t,
            curve_q=curve_q,
            out_s=out_s,
            out_q=out_q,
            source=source,
            status=status,
        )

    def direct(self, factor: np.ndarray, lap_time_s: float) -> Entry | None:
        """The table read log-log at ``lap_time_s`` and nothing else: an
        entry with that heat at that lap time and no curve. What the
        linear read of an entry's curve is held against."""
        got = self.entry(factor, lap_time_s=lap_time_s)
        return (
            None if got is None else replace(got, curve_t=None, curve_q=None, source="sim_direct")
        )


def pooled_entry(entries: list[Entry], weights: list[float]) -> Entry | None:
    """The one entry the calculators read for cars pooled under one name
    (#208: the formula cars are one FJ): per tyre the geometric mean of
    the cars' entries, weighted (by each car's share of the pooled car's
    training laps, the caller's). The entries are at one reference lap
    time and on the same curve nodes (:meth:`SimSource.entry` with
    ``lap_time_s`` and a ``nodes`` array). The out-lap is pooled where
    every entry has one; the status is the worst of them. None without an
    entry or a weight."""
    pairs = [(e, float(w)) for e, w in zip(entries, weights) if e is not None and w > 0.0]
    if not pairs:
        return None
    w = np.array([p[1] for p in pairs])
    w = w / w.sum()
    own = [p[0] for p in pairs]

    def _mean(values: list) -> np.ndarray:
        return np.asarray(np.exp(np.tensordot(w, np.log(np.array(values, float)), axes=1)))

    curve = all(e.curve_t is not None and e.curve_q is not None for e in own)
    out = all(e.out_s is not None and e.out_q is not None for e in own)
    order = [OK, OUTSIDE_SUPPORT, NO_SOLUTION]
    return Entry(
        lap_time_s=float(_mean([e.lap_time_s for e in own])),
        q=_mean([e.q for e in own]),
        curve_t=_mean([e.curve_t for e in own]) if curve else None,
        curve_q=_mean([e.curve_q for e in own]) if curve else None,
        out_s=float(_mean([e.out_s for e in own])) if out else None,
        out_q=_mean([e.out_q for e in own]) if out else None,
        source="sim_pooled",
        status=max((e.status for e in own), key=order.index),
    )


# ------------------------------------------- what the simulation replaces


def other_layouts_entry(model: dict[str, Any], track: str, car: str) -> Entry | None:
    """The car's own heat at its other layouts, unscaled (#260's last
    step): per tyre the mean of the dry entries ``model`` has for ``car``
    at other tracks. The lap time is another car's dry lap time at
    ``track`` times the two cars' ratio at the tracks they share (the
    mean over such cars; NaN where no other car has run the track: the
    calculators would ask for it). The out-lap is the car's own ratio of
    out-lap time and heat to a flying lap at its other tracks. None for a
    car with no other dry entry."""
    heats = {
        r["track_canonical"]: r
        for r in model["g2_typ_by_track_car_cond"]
        if r["car"] == car and r["condition"] == "dry" and r["track_canonical"] != track
    }
    heats = {k: r for k, r in heats.items() if r.get("q_typ_by_corner")}
    if not heats:
        return None
    q = np.mean([[r["q_typ_by_corner"][c] for c in CORNERS] for r in heats.values()], axis=0)
    laps = {
        (r["track_canonical"], r["car"]): float(r["lap_time_typ_s"])
        for r in model["lap_time_typ_by_track_car_cond"]
        if r["condition"] == "dry"
    }
    guesses = []
    for (where, other), there in laps.items():
        if where != track or other == car:
            continue
        ratios = [
            laps[(shared, car)] / laps[(shared, other)]
            for (shared, who) in laps
            if who == car and shared != track and (shared, other) in laps
        ]
        if ratios:
            guesses.append(there * float(np.median(ratios)))
    lap_time = float(np.mean(guesses)) if guesses else float("nan")
    out_s = out_q = None
    ratios_s, ratios_q = [], []
    for r in model.get("outlap_typ_by_track_car_cond") or []:
        where = r["track_canonical"]
        if r["car"] != car or r["condition"] != "dry" or where not in heats:
            continue
        if (where, car) not in laps or "outlap_q_by_corner" not in r:
            continue
        ratios_s.append(float(r["outlap_moving_s"]) / laps[(where, car)])
        ratios_q.append(
            [r["outlap_q_by_corner"][c] / heats[where]["q_typ_by_corner"][c] for c in CORNERS]
        )
    if ratios_s and math.isfinite(lap_time):
        out_s = lap_time * float(np.median(ratios_s))
        out_q = q * np.median(np.array(ratios_q), axis=0)
    return Entry(
        lap_time_s=lap_time,
        q=np.asarray(q, float),
        out_s=out_s,
        out_q=out_q,
        source="other_layouts",
        n_laps=int(sum(int(r["n_laps_used"]) for r in heats.values())),
    )


__all__ = [
    "COOL_DOWN",
    "Entry",
    "FLYING_LAPS",
    "FLYING_LAP_N",
    "MIN_ORDINARY_LAPS",
    "NEAR_LAPS",
    "ORDINARY_FROM_LAP",
    "SimSource",
    "day_entry",
    "entry_of",
    "frame_factors",
    "heats_frame",
    "logged_entry",
    "mark_laps",
    "near_heat",
    "other_layouts_entry",
    "place",
    "pooled_entry",
    "recommend",
    "stints_of",
    "target_hot",
    "without_entry",
]
