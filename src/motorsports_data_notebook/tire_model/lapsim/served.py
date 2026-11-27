"""A simulated entry as the calculators are served it (the cutover, #208's
adoption).

The standing holdout scores what ships, so the rows a retrain writes into
``tire_model.json`` and the holdout's ``served`` arm are made by the same
two functions here: :func:`sim_entry` reads a pace table's export
(:class:`~.pressure.SimSource`) at the cell's lap time, and :func:`serve`
places it in the model it is served from.

What is served at a (track, car, condition) the simulation covers:

- **the flying laps' heat and its curve** are the table's: the knots, and
  the heat read log-log at the row's lap time (in the tyres' frame, with
  no factor, #376);
- **the typical lap time** is, in the dry, the table's lap at the raw
  car's fed use (:func:`~.car_gate.fed_use`); in the wet and the damp the
  model's own logged lap time for the cell, where the chain finds one
  (:attr:`ServeRules.wet_lap_time`), with no simulated lap-time row;
- **the out-lap** is the cell's own logged out-lap where the model has one
  (the simulated out-lap is driven at the flying knob, which a real
  out-lap is not, #377), else, by :attr:`ServeRules.outlap`, the model's
  fallback chain (``"chain"``, the committee's ``table_logout``) or the
  simulated out-lap (``"sim"``);
- **the rows** of a raw car pooled under another (the FJ's KK-SII, KK-F,
  KK-F-V2; #383) go in its own ``*_raw_car_*`` tables, a car pooled under
  no other name (the Inferno 86) in the pooled ones, **beside** the logged
  rows (``"raw"``; schema §9: a simulated row answers only its own exact
  condition, so every answer the simulation does not give stays the logged
  one); or, as the committee placed them, in place of the pooled row with
  the cell's raw rows taken out (``"pooled"``).

Every row the simulation writes carries ``source: "sim"`` (a logged row
has none, or ``"logged"``) and a simulated curve its ``support``: the lap
times of the car's logged range of use (``use_max`` and ``use_min``; the
wet laps' slow end for a wet or damp row), which the calculators flag a
target outside of. Neither moves a number.

:class:`ServeRules` holds each rule apart, so that the arm can be moved
from the committee's rules to the shipping ones one rule at a time and
each step read on the holdout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from . import pressure as pr

OUTLAP_CHAIN = "chain"
OUTLAP_SIM = "sim"
ROWS_POOLED = "pooled"
ROWS_RAW = "raw"
LAP_FED = "fed"
LAP_CHAIN = "chain"

SIM = "sim"
LOGGED = "logged"
#: The three entry tables and their raw-car siblings (#383).
_TABLES = (
    "g2_typ_by_track_car_cond",
    "lap_time_typ_by_track_car_cond",
    "outlap_typ_by_track_car_cond",
)


@dataclass(frozen=True)
class ServeRules:
    """How a simulated entry is read and placed (module docstring)."""

    outlap: str = OUTLAP_CHAIN
    rows: str = ROWS_POOLED
    wet_lap_time: str = LAP_FED
    #: The table's numbers (and the fed use) to this many significant
    #: digits, as the committed export holds them; None: as driven.
    digits: int | None = None


#: The committee's arm (simfix's ``table_logout``): the table's heat, curve
#: and lap time at the fed use in every condition, in the pooled row, the
#: model's own out-lap chain.
COMMITTEE = ServeRules()
#: What ships: the export's numbers as committed (7 significant digits).
SHIPPING = ServeRules(outlap=OUTLAP_SIM, rows=ROWS_RAW, wet_lap_time=LAP_CHAIN, digits=7)
#: By name (``check-pressure-holdout --served-rules``); the steps between
#: the two, one rule each, are named after what they add, and
#: ``shipping_unrounded`` is what ships with the table as driven.
RULES = {
    "committee": COMMITTEE,
    "sim_outlap": replace(COMMITTEE, outlap=OUTLAP_SIM),
    "raw_rows": replace(COMMITTEE, outlap=OUTLAP_SIM, rows=ROWS_RAW),
    "shipping_unrounded": replace(SHIPPING, digits=None),
    "shipping": SHIPPING,
}


def as_served(src: pr.SimSource, fed_use: float, rules: ServeRules) -> tuple[pr.SimSource, float]:
    """``(src, fed_use)`` as ``rules`` read them: to the export's digits
    (idempotent: a committed export comes back as it is)."""
    if rules.digits is None or src.share is None:
        return src, fed_use
    from .export import sig

    return pr.SimSource.of(src.export.rounded(rules.digits), src.share), sig(fed_use, rules.digits)


def _raw_table(name: str) -> str:
    from ..predict import RAW_CAR_TABLES

    return RAW_CAR_TABLES[name]


def pooled_as(model: dict[str, Any], raw_car: str) -> str:
    """The car ``raw_car``'s constants and pooled entries are fitted as."""
    return str((model.get("car_aliases") or {}).get(raw_car, raw_car))


def support_of(src: pr.SimSource, condition: str) -> dict[str, float]:
    """The lap times of the export's logged range of use: ``lap_time_min_s``
    at ``use_max``, ``lap_time_max_s`` at the slow end (the dry laps' for a
    dry row, the wet laps' as well for a wet or damp one); empty without a
    range."""
    sup = src.export.support or {}
    hi = sup.get("use_max")
    lo = sup.get("use_min") if condition != "dry" else sup.get("use_min_dry", sup.get("use_min"))
    if hi is None or lo is None:
        return {}
    fast, slow = src.export.lap_time_at(float(hi)), src.export.lap_time_at(float(lo))
    if not (math.isfinite(fast) and math.isfinite(slow)):
        return {}
    return {"lap_time_min_s": float(fast), "lap_time_max_s": float(slow)}


def sim_entry(
    src: pr.SimSource,
    factor: np.ndarray,
    model: dict[str, Any],
    track: str,
    raw_car: str,
    condition: str,
    fed_use: float,
    rules: ServeRules,
) -> tuple[pr.Entry | None, bool]:
    """The simulated entry of (track, ``raw_car``, condition) and whether
    the model's own lap-time row is kept (the wet and damp under
    ``wet_lap_time="chain"``): ``(entry, keep_lap_row)``. In the dry, and
    under ``"fed"``, the entry's lap is the table's at the fed use. Under
    ``"chain"`` a wet or damp cell is served only where the model has a
    logged typical lap time of its own for exactly that condition (the
    raw car's or its pooled car's; the lead, 2026-11-27): the dry table is
    read at it (the owner's #208 ruling) and the row is kept. Elsewhere
    there is no entry, and the cell keeps the model's logged answer: a dry
    pace would be a wet prefill (None)."""
    if condition == "dry" or rules.wet_lap_time == LAP_FED:
        return src.entry(factor, use=fed_use), False
    lap = logged_lap_time(model, track, raw_car, condition)
    if lap is None:
        return None, False
    return src.entry(factor, lap_time_s=lap), True


def logged_lap_time(
    model: dict[str, Any], track: str, raw_car: str, condition: str
) -> float | None:
    """The model's logged typical lap time of the cell at exactly
    ``condition`` (the raw car's own row, else its pooled car's), or None."""
    from ..predict import LOGGED, _answering_rows, row_provenance

    rows, step = _answering_rows(
        model, "lap_time_typ_by_track_car_cond", track, raw_car, condition, global_step=False
    )
    if step != "exact" or not rows or row_provenance(rows) != LOGGED:
        return None
    lap = float(rows[0]["lap_time_typ_s"])
    return lap if math.isfinite(lap) else None


def _key(row: dict, track: str, car: str, condition: str) -> bool:
    return bool(
        row["track_canonical"] == track and row["car"] == car and row["condition"] == condition
    )


def logged_outlap(model: dict[str, Any], track: str, raw_car: str, condition: str) -> bool:
    """Does the model's chain answer the cell's out-lap with a logged row
    of the car's own (the raw car's or the car it is pooled as, at this
    condition or the next on the condition chain), rather than a track or
    global average, a simulated row or nothing?"""
    from ..predict import _AVERAGED_STEPS, LOGGED, _answering_rows, row_provenance

    rows, step = _answering_rows(
        model, "outlap_typ_by_track_car_cond", track, raw_car, condition, global_step=False
    )
    return bool(rows) and step not in _AVERAGED_STEPS and row_provenance(rows) == LOGGED


def _marked(row: dict[str, Any], support: dict[str, float] | None) -> dict[str, Any]:
    """A row the simulation wrote: ``source: "sim"``, and on a curve
    ``method: "sim"`` with its ``support``."""
    row = {**row, "source": SIM}
    curve = row.get("g2_vs_lap_time")
    if isinstance(curve, dict):
        row["g2_vs_lap_time"] = {
            **curve,
            "method": SIM,
            **({"support": dict(support)} if support else {}),
        }
    return row


def _sorted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (r["track_canonical"], r["car"], r["condition"]))


def serve(
    model: dict[str, Any],
    track: str,
    raw_car: str,
    condition: str,
    entry: pr.Entry,
    rules: ServeRules,
    *,
    keep_lap_row: bool = False,
    support: dict[str, float] | None = None,
) -> dict[str, Any]:
    """``model`` with the simulated ``entry`` served at (track, ``raw_car``,
    condition) under ``rules``: a shallow copy in which only that cell's
    rows differ (module docstring). ``keep_lap_row`` writes no lap-time
    row; ``support`` goes on the simulated curve.

    Under ``rows="raw"`` (what ships) the simulated rows are added **beside**
    the logged ones, never in their place (schema §9): the raw car's own
    tables for a raw car pooled under another, else the car's; a simulated
    row already at the cell is replaced. The calculators' chain reads a
    simulated row only at its own exact condition, so every other answer
    stays the logged one. The committee's ``rows="pooled"`` puts the entry
    in place of the pooled row, with the cell's raw rows taken out."""
    car = pooled_as(model, raw_car)
    keep_out = logged_outlap(model, track, raw_car, condition) or rules.outlap == OUTLAP_CHAIN
    skip = {
        "lap_time_typ_by_track_car_cond": keep_lap_row,
        "outlap_typ_by_track_car_cond": keep_out,
    }
    if rules.rows == ROWS_POOLED:
        placed = pr.place(model, track, car, condition, entry)
        if keep_lap_row or keep_out:
            for t in _TABLES:
                if skip.get(t):
                    for name in (t, _raw_table(t)):
                        if name in model or name in placed:
                            placed[name] = list(model.get(name) or [])
        for t in _TABLES:
            if not skip.get(t):
                placed[t] = [
                    _marked(r, support) if _key(r, track, car, condition) else r
                    for r in placed.get(t) or []
                ]
        return placed
    who = raw_car
    names = {t: (_raw_table(t) if car != raw_car else t) for t in _TABLES}
    new = pr.place({t: [] for t in _TABLES}, track, who, condition, entry, raw=False)
    placed = dict(model)
    for t in _TABLES:
        if skip.get(t):
            continue
        kept = [
            r
            for r in model.get(names[t]) or []
            if not (_key(r, track, who, condition) and r.get("source") == SIM)
        ]
        placed[names[t]] = _sorted(kept + [_marked(r, support) for r in new[t]])
    return placed


def serve_cell(
    model: dict[str, Any],
    src: pr.SimSource,
    factor: np.ndarray,
    track: str,
    raw_car: str,
    fed_use: float,
    conditions: tuple[str, ...],
    rules: ServeRules,
) -> tuple[dict[str, Any], dict[str, str]]:
    """``model`` with (track, ``raw_car``) served in each of ``conditions``
    from one table, the dry first, so that a wet or damp row is read at the
    lap time the chain gives it once the dry is served (what ships, and so
    what the holdout's ``served`` arm places): ``(model, how)``, ``how``
    saying per condition ``sim`` or why not."""
    how: dict[str, str] = {}
    out = model
    src, fed_use = as_served(src, fed_use, rules)
    for cond in sorted(dict.fromkeys(conditions), key=lambda c: c != "dry"):
        entry, keep = sim_entry(src, factor, out, track, raw_car, cond, fed_use, rules)
        if entry is None:
            how[cond] = f"no logged {cond} lap time at the cell: its logged answer stands"
            continue
        out = serve(
            out,
            track,
            raw_car,
            cond,
            entry,
            rules,
            keep_lap_row=keep,
            support=support_of(src, cond),
        )
        how[cond] = SIM
    return out, how
