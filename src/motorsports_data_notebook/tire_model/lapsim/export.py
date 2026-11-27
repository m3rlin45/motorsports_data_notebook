"""A pace table as the laps the heat chain reads (#208, "Adoption").

A pace table (:mod:`.pace`) holds laps, not heat: a tyre's heat is of the
force shares of the fit in hand, and they move with every retrain (leaving
a cell out moves the same laps' heat by up to 11 % per tyre). So what a
simulated entry is built from cannot be a row of heats written once. It is
this: each knot's flying lap and out-lap **as production's chain bins
them**, one-second means of speed and of the four non-negative parts of
the accelerations (:class:`~.heat.LapBins`: the lateral the tyres hold,
the longitudinal an accelerometer reads), which is everything
:func:`~.heat.lap_heat` reads. A consumer splits the bins through its own
shares (:meth:`TableExport.heat_knots`) with no solver, no track and
nothing from ``workspace/``; the result is :meth:`.pace.PaceTable.heat_knots`
of the same shares to the last bit, because it is the same arithmetic on
the same bins.

Between the knots everything is read log-log against the lap time, as the
table is. :meth:`TableExport.lap_time_at` is the other way round: the lap
time of a use, which is how an entry that is given no lap time supplies
its own (the car's *fed use*, :func:`~.car_gate.fed_use`).

This module imports nothing of the solver, so that the pressure model's
build can read an export.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..heat_input import ShareParams
from .heat import CORNERS, LapBins, lap_heat

_BIN_FIELDS = ("dur_s", "v_ms", "lat_pos", "lat_neg", "long_pos", "long_neg")


def _bins_to_dict(bins: LapBins) -> dict:
    out: dict = {f: [float(x) for x in getattr(bins, f)] for f in _BIN_FIELDS}
    out["lap_time_s"] = float(bins.lap_time_s)
    return out


def sig(x: float, digits: int) -> float:
    """``x`` to ``digits`` significant digits (NaN and inf as they are)."""
    x = float(x)
    return float(f"{x:.{digits}g}") if math.isfinite(x) else x


def _sig_array(values: np.ndarray, digits: int) -> np.ndarray:
    return np.array([sig(v, digits) for v in np.asarray(values, float)], dtype=float)


def _sig_bins(bins: LapBins, digits: int) -> LapBins:
    return LapBins(
        **{f: _sig_array(getattr(bins, f), digits) for f in _BIN_FIELDS},
        lap_time_s=sig(bins.lap_time_s, digits),
    )


def _sig_dict(data: dict, digits: int) -> dict:
    return {
        k: (
            sig(v, digits)
            if isinstance(v, float)
            else _sig_dict(v, digits) if isinstance(v, dict) else v
        )
        for k, v in data.items()
    }


def _bins_from_dict(data: dict) -> LapBins:
    return LapBins(
        **{f: np.asarray(data[f], float) for f in _BIN_FIELDS},
        lap_time_s=float(data["lap_time_s"]),
    )


@dataclass
class TableExport:
    """The knots of a pace table as binned laps (module docstring).

    ``use`` rises along the knots and ``lap_time_s`` falls, as in the
    table. ``flying`` is each knot's lap; ``out`` each knot's part of an
    out-lap from the pit exit's join to the line (None without a pit
    exit), and ``rolling_s`` how long an out-lap has been rolling when it
    joins (the track's; no heat is counted there). ``support`` and
    ``assumes`` are the table's; ``made`` says when and at which commit
    the laps were driven. A committed export (:mod:`.sim_export`) also
    names its ``raw_car`` and holds its ``fed`` use (``{"use", ...}``: the
    lap an entry stands for)."""

    layout_id: str
    car: str
    use: np.ndarray
    lap_time_s: np.ndarray
    flying: list[LapBins]
    out: list[LapBins] | None = None
    rolling_s: float = 0.0
    support: dict = field(default_factory=dict)
    assumes: dict = field(default_factory=dict)
    fastest_lap_s: float = float("nan")
    max_driver_lap_s: float = float("nan")
    made: dict = field(default_factory=dict)
    raw_car: str = ""
    fed: dict = field(default_factory=dict)

    def rounded(self, digits: int = 7) -> "TableExport":
        """The export with every number to ``digits`` significant digits:
        what is committed (:mod:`.sim_export`), and what the holdout's
        ``served`` arm reads, so that the arm scored is the shipped one to
        the last bit."""
        return TableExport(
            layout_id=self.layout_id,
            car=self.car,
            use=_sig_array(self.use, digits),
            lap_time_s=_sig_array(self.lap_time_s, digits),
            flying=[_sig_bins(b, digits) for b in self.flying],
            out=None if self.out is None else [_sig_bins(b, digits) for b in self.out],
            rolling_s=sig(self.rolling_s, digits),
            support=_sig_dict(self.support, digits),
            assumes=dict(self.assumes),
            fastest_lap_s=sig(self.fastest_lap_s, digits),
            max_driver_lap_s=sig(self.max_driver_lap_s, digits),
            made=dict(self.made),
            raw_car=self.raw_car,
            fed=_sig_dict(self.fed, digits),
        )

    def heat_knots(self, share: ShareParams) -> np.ndarray:
        """Lap-mean heat rate per tyre at each knot (knots × 4; G·m/s)."""
        return np.vstack([lap_heat(bins, share) for bins in self.flying])

    def out_knots(self, share: ShareParams) -> tuple[np.ndarray, np.ndarray] | None:
        """``(part_s, heat)`` of the out-lap at each knot: how long the
        part from the join to the line takes and its mean heat rate per
        tyre over that time; None without a pit exit."""
        if self.out is None:
            return None
        part = np.array([bins.lap_time_s for bins in self.out])
        return part, np.vstack([lap_heat(bins, share) for bins in self.out])

    def at(self, lap_time_s: float, values: np.ndarray) -> np.ndarray:
        """``values`` (one row per knot, positive) at a lap time: log-log
        between the knots, off the last two beyond either end."""
        x = np.log(self.lap_time_s[::-1])  # rising
        y = np.log(np.asarray(values, float)[::-1])
        at = math.log(lap_time_s)
        k = int(np.clip(np.searchsorted(x, at) - 1, 0, len(x) - 2))
        w = (at - x[k]) / (x[k + 1] - x[k])
        return np.asarray(np.exp(y[k] + w * (y[k + 1] - y[k])))

    def use_at(self, lap_time_s: float) -> float:
        """The use a lap time is driven at (NaN: faster than the table)."""
        if not lap_time_s >= float(self.lap_time_s[-1]):
            return float("nan")
        return float(self.at(lap_time_s, self.use))

    def lap_time_at(self, use: float) -> float:
        """The lap time of a use: the inverse of :meth:`use_at` (NaN
        outside the knots)."""
        if not (float(self.use[0]) <= use <= float(self.use[-1])):
            return float("nan")
        x, y = np.log(self.use), np.log(self.lap_time_s)
        k = int(np.clip(np.searchsorted(x, math.log(use)) - 1, 0, len(x) - 2))
        w = (math.log(use) - x[k]) / (x[k + 1] - x[k])
        if w <= 0.0 or w >= 1.0:  # at a knot: the knot's own lap time, not its logarithm's
            return float(self.lap_time_s[k if w <= 0.0 else k + 1])
        return float(np.exp(y[k] + w * (y[k + 1] - y[k])))

    def to_dict(self) -> dict:
        return {
            "layout_id": self.layout_id,
            "car": self.car,
            "tyres": list(CORNERS),
            "series": (
                "one-second means from the lap's start, as production's heat chain bins a "
                "lap: dur_s (seconds a bin stands for), v_ms, and the non-negative parts of "
                "the lateral the tyres hold (lat_pos: right turns) and of the longitudinal an "
                "accelerometer reads (long_pos: drive), in g"
            ),
            "interpolation": "log-log against lap_time_s between neighbouring knots",
            "knots": [
                {
                    "use": float(self.use[k]),
                    "lap_time_s": float(self.lap_time_s[k]),
                    "flying": _bins_to_dict(self.flying[k]),
                    **({"out": _bins_to_dict(self.out[k])} if self.out is not None else {}),
                }
                for k in range(len(self.use))
            ],
            "rolling_s": float(self.rolling_s),
            "support": self.support,
            "assumes": self.assumes,
            "fastest_lap_s": float(self.fastest_lap_s),
            "max_driver_lap_s": float(self.max_driver_lap_s),
            "made": self.made,
            **({"raw_car": self.raw_car} if self.raw_car else {}),
            **({"fed": self.fed} if self.fed else {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TableExport":
        knots = data["knots"]
        has_out = bool(knots) and all("out" in k for k in knots)
        return cls(
            layout_id=str(data["layout_id"]),
            car=str(data["car"]),
            use=np.array([float(k["use"]) for k in knots]),
            lap_time_s=np.array([float(k["lap_time_s"]) for k in knots]),
            flying=[_bins_from_dict(k["flying"]) for k in knots],
            out=[_bins_from_dict(k["out"]) for k in knots] if has_out else None,
            rolling_s=float(data.get("rolling_s", 0.0)),
            support=dict(data.get("support") or {}),
            assumes=dict(data.get("assumes") or {}),
            fastest_lap_s=float(data.get("fastest_lap_s", float("nan"))),
            max_driver_lap_s=float(data.get("max_driver_lap_s", float("nan"))),
            made=dict(data.get("made") or {}),
            raw_car=str(data.get("raw_car") or ""),
            fed=dict(data.get("fed") or {}),
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> "TableExport":
        return cls.from_dict(json.loads(Path(path).read_text()))


__all__ = ["TableExport"]
