"""Simulated entries in ``tire_model.json`` (the cutover; #208's adoption).

After the pressure model is fitted on logged heat in the tyres' frame
(#376), the cells the lap simulation covers are served from the committed
export of simulated laps (``data/lapsim/exports``,
:mod:`.lapsim.sim_export`): per raw car and committed track, the pace
table's laps split through this fit's own shares, read at the raw car's
fed use, placed by the shipping rules of :mod:`.lapsim.served`, the same
code the standing holdout's ``served`` arm is scored with.

- **Only the shipped artifact.** :func:`place_simulated` runs where
  ``tire_model.json`` is written (``build_warmup_table(write_artifacts=
  True)``), never in a fold's fit: a fold's calculator stays logged, and the
  export, which has seen every session, never answers a held-out one.
- **Only the tyres' frame.** A simulated lap meets the fit with no factor
  only there; a model fitted on the sensor's series gets no simulated row
  (``simulation.placed`` false, with the reason).
- **The cells.** A raw car's export at a layout serves the track the
  layout is the main layout of (its venue's ``default_layout``), in each of
  :data:`SIM_CONDITIONS`, the dry first (a wet or damp row is read at the
  lap time the chain then gives it).
- **What it says it did.** ``simulation`` records the export's making, the
  conditions, the rules and GSI's attribution; ``entry_coverage`` is, per
  cell the calculators offer, which kind of row each lookup answers with
  (:func:`coverage`). Every cell whose answers are all logged reads exactly
  as it did before the placement (:func:`unmoved`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: The conditions the simulation serves. Provisional: the lead sets it from
#: the re-evaluation's per-condition numbers (the holdout's ``served`` arm
#: scores all three).
SIM_CONDITIONS: tuple[str, ...] = ("dry", "damp", "wet")
CONDITIONS = ("dry", "damp", "wet")
#: The lookups a prediction reads an entry through.
LOOKUPS = ("heat", "lap_time", "outlap", "curve")
_TABLE_OF = {
    "heat": "g2_typ_by_track_car_cond",
    "lap_time": "lap_time_typ_by_track_car_cond",
    "outlap": "outlap_typ_by_track_car_cond",
    "curve": "g2_typ_by_track_car_cond",
}


def track_of(layout_id: str) -> str | None:
    """The track (``track_canonical``) whose main layout ``layout_id`` is,
    or None for another layout of a venue."""
    from ..tire_etl.layouts import VENUES

    for venue in VENUES.values():
        if venue.default_layout == layout_id:
            return str(venue.venue_id)
    return None


def offered_cells(model: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Every (track, car, condition) the calculators offer: each track with
    an entry, each car (a pooled car's raw cars in its place, #383), each
    condition."""
    from .predict import RAW_CAR_TABLES

    aliases = dict(model.get("car_aliases") or {})
    pooled = {v for k, v in aliases.items() if k != v}
    tracks: set[str] = set()
    cars: set[str] = set()
    for name, raw_name in RAW_CAR_TABLES.items():
        for r in model.get(name) or []:
            tracks.add(str(r["track_canonical"]))
            if r["car"] not in pooled:
                cars.add(str(r["car"]))
        for r in model.get(raw_name) or []:
            tracks.add(str(r["track_canonical"]))
    cars |= {k for k, v in aliases.items() if k != v}
    return [(t, c, k) for t in sorted(tracks) for c in sorted(cars) for k in CONDITIONS]


def answer(model: dict[str, Any], lookup: str, track: str, car: str, condition: str) -> str:
    """The kind of row a lookup answers (track, car, condition) with, by
    the calculators' one chain (:func:`~.predict._answering_rows`):
    ``sim`` or ``logged`` (a row found by the condition chain, the raw
    car's before the pooled car's), ``averaged`` (a track or global
    average, of logged rows only) or ``none`` (nothing answers, or for the
    curve a row without one: the exponent)."""
    from .predict import _AVERAGED_STEPS, _answering_rows, row_provenance

    rows, step = _answering_rows(
        model, _TABLE_OF[lookup], track, car, condition, global_step=lookup in ("heat", "curve")
    )
    if not rows:
        return "none"
    if step in _AVERAGED_STEPS:
        return "averaged"
    if lookup == "curve" and not isinstance(rows[0].get("g2_vs_lap_time"), dict):
        return "none"
    return str(row_provenance(rows))


def _reading(model: dict[str, Any], track: str, car: str, condition: str) -> tuple:
    """What a prediction reads of a cell: the heat per tyre, the lap time,
    the out-lap per tyre and the pace curve."""
    from .predict import _lookup_g2, _lookup_g2_pace_curve, _lookup_lap_time, _lookup_outlap

    corners = ("fl", "fr", "rl", "rr")

    def _try(fn, *a):  # type: ignore[no-untyped-def]
        try:
            return fn(*a)
        except (KeyError, ValueError, ZeroDivisionError):
            return None

    return (
        tuple(_try(_lookup_g2, model, track, car, condition, c) for c in corners),
        _try(_lookup_lap_time, model, track, car, condition),
        tuple(_try(_lookup_outlap, model, track, car, condition, c) for c in corners),
        repr(_try(_lookup_g2_pace_curve, model, track, car, condition)),
    )


def coverage(
    model: dict[str, Any],
    before: dict[str, Any],
    notes: dict[tuple, str],
    conditions: tuple[str, ...] = SIM_CONDITIONS,
) -> list[dict[str, Any]]:
    """``entry_coverage``: per offered cell, the kind of row each lookup
    answers with on the placed ``model``, its layout where the simulation
    has one, and the reason a cell the simulation does not answer is
    logged; and ``moved``, whether anything it reads changed from
    ``before`` (true only where something answers ``sim``)."""
    layout_of = {}
    from ..tire_etl.layouts import VENUES

    for venue in VENUES.values():
        layout_of[str(venue.venue_id)] = str(venue.default_layout)
    rows = []
    for track, car, cond in offered_cells(model):
        got = {k: answer(model, k, track, car, cond) for k in LOOKUPS}
        sim = any(v == "sim" for v in got.values())
        row: dict[str, Any] = {
            "track_canonical": track,
            "layout_id": layout_of.get(track),
            "car": car,
            "condition": cond,
            **got,
            "sim_only": sim
            and all(answer(before, k, track, car, cond) != "logged" for k in ("heat", "lap_time")),
        }
        if not sim:
            row["reason"] = (
                notes.get((track, car, cond))
                or notes.get((track, car))
                or notes.get(("", car))
                or "no simulated laps for the car at this track"
            )
            if cond not in conditions:
                row["reason"] = f"{cond}: logged by the policy (SIM_CONDITIONS)"
        rows.append(row)
    return rows


def unmoved(
    model: dict[str, Any], before: dict[str, Any], rows: list[dict[str, Any]]
) -> list[tuple[str, str, str]]:
    """The cells whose answers are all logged but whose reading moved with
    the placement: empty, or the chain lets a simulated row into a logged
    answer (an averaged fallback, say)."""
    bad = []
    for r in rows:
        if any(r[k] == "sim" for k in LOOKUPS):
            continue
        cell = (r["track_canonical"], r["car"], r["condition"])
        if _reading(model, *cell) != _reading(before, *cell):
            bad.append(cell)
    return bad


class LoggedCellMoved(RuntimeError):
    """A cell the simulation does not answer reads differently after the
    placement: the chain let a simulated row into a logged answer."""


def place_simulated(
    model: dict[str, Any],
    exports_dir: Path | None = None,
    *,
    conditions: tuple[str, ...] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """``model`` with the simulated entries of the committed export
    (module docstring), ``simulation`` and ``entry_coverage`` added. A
    model with no export to read, or in the sensor's frame, comes back with
    ``simulation.placed`` false and why. ``strict``: a logged cell that
    moved is a :class:`LoggedCellMoved`, not an artifact."""
    from .heat_frame import TYRES, frame_of
    from .lapsim import served as srv
    from .lapsim import sim_export as sx
    from .lapsim.holdout import shares_of
    from .lapsim.pressure import SimSource

    conditions = SIM_CONDITIONS if conditions is None else tuple(conditions)
    out = dict(model)
    if frame_of(model) != TYRES:
        out["simulation"] = {
            "placed": False,
            "why": "the model's heat is in the sensor's frame: a simulated lap meets it only "
            "through a frame factor, which is not served (#376)",
        }
        return out
    exports, index = sx.load_exports(exports_dir or sx.default_exports())
    if not exports:
        out["simulation"] = {"placed": False, "why": "no committed export of simulated laps"}
        return out
    shares = shares_of(model)
    before = model
    notes: dict[tuple, str] = {}
    for n in index.get("notes") or []:
        if not n.get("made") and n.get("car"):
            notes[("", str(n["car"]))] = str(n.get("why") or "not made")
    served_cells = []
    ones = np.ones(4)
    for exp in sorted(exports, key=lambda e: (e.raw_car, e.layout_id)):
        track = track_of(exp.layout_id)
        raw = exp.raw_car
        if track is None or not raw:
            continue
        if raw not in shares:
            notes[(track, raw)] = "the pressure model has no fitted car for it"
            continue
        use = float(exp.fed.get("use", float("nan")))
        if not np.isfinite(use):
            notes[(track, raw)] = "no fed use"
            continue
        src = SimSource.of(exp, shares[raw])
        out, why = srv.serve_cell(out, src, ones, track, raw, use, conditions, srv.SHIPPING)
        for cond, how in why.items():
            if how != srv.SIM:
                notes[(track, raw, cond)] = how
        served_cells.append(
            {"track_canonical": track, "layout_id": exp.layout_id, "car": raw, **why}
        )
    rows = coverage(out, before, notes, conditions)
    moved = unmoved(out, before, rows)
    if moved:
        logger.warning(
            "sim entries: %d logged cells read differently after the placement: %s",
            len(moved),
            moved[:10],
        )
        if strict:
            raise LoggedCellMoved(
                f"{len(moved)} cells the simulation does not answer read differently after "
                f"the placement (first {moved[:5]}): the averaged fallbacks must read logged "
                "rows only"
            )
    out["simulation"] = {
        "placed": True,
        "export": index.get("made") or {},
        "conditions": list(conditions),
        "rules": asdict(srv.SHIPPING),
        "cells": served_cells,
        "logged_cells_moved": [list(c) for c in moved],
        "attribution": sx.ATTRIBUTION,
    }
    out["entry_coverage"] = rows
    return out
