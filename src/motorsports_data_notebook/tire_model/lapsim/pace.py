"""The pace table of gate D1 (#206): a car's lap at a layout over the
whole range of pace, for whoever needs the heat of a lap of a given lap
time without a solver (#207, #208, the calculators).

**The axis is the driver's use of the car's limit** (#312; owner,
2026-11-23), ``q = -ln(use)``: twelve laps from a use of 0.40 to a use of
1, the knob's lap at q = 0. That lap is braked as the driver brakes, so
it is not the max driver's (the committee's clause 3.1: every use at 1
*and* the driver's habits off), which is a little faster and is kept
beside the table. A table holds each knot as its speed at the track's
nodes and nothing made from it:

- **No heat columns.** Per-tyre heat is defined by the force shares of
  the fit in hand (leaving a cell out moves the same laps' heat by up to
  11 % per tyre, #208), so the table hands over the lap and
  :meth:`PaceTable.heat` splits it with the caller's shares.
- **No chain.** The lap goes through production's chain as gate D1 has it
  (:func:`~.heat.sim_heat`) unless the caller gives another; the 0.5 s
  smoothing of #207's E0 is 0.6-1.3 % lower and is E's to settle.

Between the knots the heat and the use are read log-log against the lap
time; gate D1 holds that to 1 % of a direct solve.

**What a table assumes of the car** (a car is a chassis with powertrains,
and a lap also has its day's shift point, gear set and air, #205; none of
them is known of a lap that has not been driven): the powertrain named
(the car's latest unless one is asked for), that powertrain's own shift
point and gears (the median day's), standard air, the laps braked as the
driver brakes, the road's bank. They are written into
:attr:`PaceTable.assumes`. A powertrain that has run more than one gear
set (the KK-SII has per-circuit gearing) says so there; its gears are
inferred, so it is driven on one specific power and the set changes
nothing in the lap.

**Support.** The car's logged range of use at the layout
(:func:`~.car_gate.use_range`, saved by gate C2 as the model's
``meta.use_range``): a lap time whose use lies outside it is answered and
flagged ``outside_support``; so is one beyond the table's slow end, which
is read off the last two knots. At a layout the car has not run the range
is the car's over the layouts it has, and the table says so. A lap time
faster than the lap at a use of 1 has ``no_solution``: the table gives
no heat, and that lap's time and the max driver's beside it. A target
between those two is one the committee's knob would answer (below q = 0
the cornering use stays at 1 and the braking use rises to the car's
level); the kernel's knob scales the two together, so until that cap is
built such a target goes unanswered here.

**The out-lap.** Where the track knows its pit exit (``meta.pit`` of
gate B2: where an out-lap joins the driven line, how fast it is going
there and how long it had been rolling) each knot also has the part of a
lap from the join to the line, begun at the join speed and driven at the
knot's use: :meth:`PaceTable.out_lap`, its time and its heat. The car is
driven on round the lap past the line, so that it arrives at the line as
a lap that goes on, not one that ends there. The rolling time before the
join is the track's and carries no heat here.

Dry only: the wet entry is gate D2's.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..heat_input import ShareParams
from . import car as cm
from . import car_gate as cg
from . import solver as sv
from . import track as trk
from .heat import sim_heat

#: The knots: this many laps, evenly in ``q``, from this use to a use of 1.
KNOTS = 12
USE_SLOW = 0.40

OK = "ok"
OUTSIDE_SUPPORT = "outside_support"
NO_SOLUTION = "no_solution"

#: A lap's heat per tyre from the simulated lap and the shares.
Chain = Callable[[sv.SimLap, ShareParams], np.ndarray]


@dataclass(frozen=True)
class Reading:
    """What a table gives for one lap time: the ``use`` of the limit and
    ``q = -ln(use)``, the lap-mean heat rate per tyre (``fl, fr, rl, rr``;
    G·m/s), the ``status``, the table's fastest lap (a use of 1, braked
    as the driver brakes) and the max driver's (a use of 1 with the
    driver's habits off). With ``no_solution`` the use and the heat are
    NaN."""

    lap_time_s: float
    use: float
    q: float
    heat: np.ndarray
    status: str
    fastest_lap_s: float
    max_driver_lap_s: float


@dataclass
class PaceTable:
    """A car's laps at a layout from a use of :data:`USE_SLOW` to 1
    (module docstring). ``use`` rises along the knots, ``lap_time_s``
    falls; ``v_node_ms`` is knots × nodes. ``support`` is ``{"use_min",
    "use_max", "laps", "days", "from"}`` and ``assumes`` what the laps
    assume of the car."""

    layout_id: str
    car: str
    use: np.ndarray
    lap_time_s: np.ndarray
    v_node_ms: np.ndarray
    profile: sv.Profile
    driven: sv.SolverCar
    support: dict = field(default_factory=dict)
    assumes: dict = field(default_factory=dict)
    #: The track's pit exit (``s_join_m, v_join_ms, rolling_s, n``) and the
    #: out-lap of each knot: its speed at the nodes from the join to the
    #: line (knots × nodes of the part). Empty and None without one.
    pit: dict = field(default_factory=dict)
    v_out_ms: np.ndarray | None = None
    #: The max driver's lap time (clause 3.1: every use at 1 and the
    #: driver's habits off), with the table's car otherwise; the table's
    #: fastest lap for a car with no driver entry.
    max_driver_lap_s: float = float("nan")

    @property
    def q(self) -> np.ndarray:
        return -np.log(self.use)

    @property
    def fastest_lap_s(self) -> float:
        """The table's fast end: the lap at a use of 1, braked as the
        driver brakes (the knob's lap at q = 0)."""
        return float(self.lap_time_s[-1])

    def lap(self, knot: int) -> sv.SimLap:
        """The simulated lap of one knot."""
        return sv.lap_from_nodes(
            self.profile, self.driven, self.v_node_ms[knot], float(self.use[knot])
        )

    def out_lap(self, knot: int) -> sv.SimLap | None:
        """The out-lap of one knot: the simulated part of a lap from the
        pit exit's join to the line (None without a pit exit)."""
        if self.v_out_ms is None:
            return None
        part = _out_profile(self.profile, _join_node(self.profile, self.pit), whole=False)
        return sv.lap_from_nodes(part, self.driven, self.v_out_ms[knot], float(self.use[knot]))

    def out_knots(
        self, share: ShareParams, chain: Chain = sim_heat
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """``(part_s, heat)`` of the out-lap at each knot: how long the part
        from the join to the line takes (knots) and its mean heat rate per
        tyre over that time (knots × 4); None without a pit exit."""
        if self.v_out_ms is None:
            return None
        laps = [self.out_lap(k) for k in range(len(self.use))]
        part = np.array([lap.lap_time_s for lap in laps if lap is not None])
        return part, np.vstack([chain(lap, share) for lap in laps if lap is not None])

    def read_out(
        self, lap_time_s: float, share: ShareParams, *, chain: Chain = sim_heat
    ) -> tuple[float, np.ndarray] | None:
        """``(part_s, heat)`` of the out-lap driven at the use of a flying
        lap of ``lap_time_s`` (NaN where that has no solution); None
        without a pit exit."""
        got = self.out_knots(share, chain)
        if got is None:
            return None
        if self.status(lap_time_s) == NO_SOLUTION:
            return float("nan"), np.full(4, float("nan"))
        return float(self._at(lap_time_s, got[0])), self._at(lap_time_s, got[1])

    def heat_knots(self, share: ShareParams, chain: Chain = sim_heat) -> np.ndarray:
        """Lap-mean heat rate per tyre at each knot (knots × 4) through
        ``share`` and ``chain``."""
        return np.vstack([chain(self.lap(k), share) for k in range(len(self.use))])

    def _at(self, lap_time_s: float, values: np.ndarray) -> np.ndarray:
        """``values`` (one row per knot, positive) at a lap time, log-log
        between the knots and off the last two beyond the slow end."""
        x = np.log(self.lap_time_s[::-1])  # rising
        y = np.log(np.asarray(values, float)[::-1])
        at = math.log(lap_time_s)
        k = int(np.clip(np.searchsorted(x, at) - 1, 0, len(x) - 2))
        w = (at - x[k]) / (x[k + 1] - x[k])
        return np.asarray(np.exp(y[k] + w * (y[k + 1] - y[k])))

    def use_at(self, lap_time_s: float) -> float:
        """The use a lap time is driven at (NaN: faster than a use of 1)."""
        if not lap_time_s >= self.fastest_lap_s:
            return float("nan")
        return float(self._at(lap_time_s, self.use))

    def use_at_many(self, lap_time_s: np.ndarray) -> np.ndarray:
        """:meth:`use_at` of each lap time, 1 where there is no solution
        (a guess for a direct solve)."""
        use = np.array([self.use_at(float(t)) for t in np.asarray(lap_time_s, float)])
        return np.where(np.isfinite(use), use, 1.0)

    def status(self, lap_time_s: float) -> str:
        """``ok``, ``outside_support`` or ``no_solution`` (module docstring)."""
        if not lap_time_s >= self.fastest_lap_s:
            return NO_SOLUTION
        use = self.use_at(lap_time_s)
        lo, hi = self.support.get("use_min"), self.support.get("use_max")
        if lo is None or hi is None or lap_time_s > self.lap_time_s[0]:
            return OUTSIDE_SUPPORT
        return OK if float(lo) <= use <= float(hi) else OUTSIDE_SUPPORT

    def read(
        self,
        lap_time_s: float,
        share: ShareParams,
        *,
        chain: Chain = sim_heat,
        knots: np.ndarray | None = None,
    ) -> Reading:
        """The table at one lap time. ``knots`` is :meth:`heat_knots` of
        the same shares and chain, for a caller with many lap times."""
        status = self.status(lap_time_s)
        if status == NO_SOLUTION:
            nan = float("nan")
            return Reading(
                lap_time_s,
                nan,
                nan,
                np.full(4, nan),
                status,
                self.fastest_lap_s,
                self.max_driver_lap_s,
            )
        heat = self._at(lap_time_s, self.heat_knots(share, chain) if knots is None else knots)
        use = self.use_at(lap_time_s)
        return Reading(
            lap_time_s,
            use,
            -math.log(use),
            heat,
            status,
            self.fastest_lap_s,
            self.max_driver_lap_s,
        )

    def heat(
        self, lap_time_s: np.ndarray | float, share: ShareParams, *, chain: Chain = sim_heat
    ) -> np.ndarray:
        """Lap-mean heat rate per tyre at each lap time (… × 4; NaN where
        there is no solution)."""
        knots = self.heat_knots(share, chain)
        times = np.atleast_1d(np.asarray(lap_time_s, float))
        out = np.vstack([self.read(float(t), share, knots=knots).heat for t in times])
        return out[0] if np.ndim(lap_time_s) == 0 else out

    def to_dict(self) -> dict:
        """The table without the track (it is rebuilt against the layout's
        track by :func:`from_dict`)."""
        return {
            "layout_id": self.layout_id,
            "car": self.car,
            "use": [float(x) for x in self.use],
            "lap_time_s": [float(x) for x in self.lap_time_s],
            "v_node_ms": [[float(x) for x in row] for row in self.v_node_ms],
            "support": self.support,
            "assumes": self.assumes,
            "pit": self.pit,
            "max_driver_lap_s": float(self.max_driver_lap_s),
            "v_out_ms": (
                None
                if self.v_out_ms is None
                else [[float(x) for x in row] for row in self.v_out_ms]
            ),
        }

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()))


def _join_node(profile: sv.Profile, pit: dict) -> int:
    return int(round(float(pit["s_join_m"]) / profile.ds_m)) % profile.n


def _out_profile(profile: sv.Profile, join: int, *, whole: bool) -> sv.Profile:
    """The road from node ``join`` as an open profile: on to the line (its
    last node is the lap's first), or with ``whole`` on round the lap to
    the node before ``join`` again."""
    n = profile.n
    idx = (np.arange(join, join + n) % n) if whole else (np.arange(join, n + 1) % n)
    return sv.Profile(
        ds_m=profile.ds_m,
        kappa=np.ascontiguousarray(profile.kappa[idx]),
        grade=np.ascontiguousarray(profile.grade[idx]),
        kappa_v=np.ascontiguousarray(profile.kappa_v[idx]),
        closed=False,
        bank=None if profile.bank is None else np.ascontiguousarray(profile.bank[idx]),
    )


def out_laps(
    profile: sv.Profile, driven: sv.SolverCar, use: np.ndarray, pit: dict
) -> np.ndarray | None:
    """Each use's out-lap as its speed at the nodes from the pit exit's
    join to the line (uses × nodes of the part): begun at the join speed
    and driven on round the lap, of which the part to the line is kept.
    None where the track has no join or no join speed."""
    if not pit or not pit.get("n") or pit.get("v_join_ms") is None:
        return None
    join = _join_node(profile, pit)
    if join == 0:
        return None
    road = _out_profile(profile, join, whole=True)
    keep = profile.n - join + 1
    v_join = float(pit["v_join_ms"])
    return np.vstack(
        [sv.simulate_lap(road, driven, float(u), v_start=v_join).v_node_ms[:keep] for u in use]
    )


def assumed_car(
    model: cm.CarModel, powertrain: str | None = None, *, habits: bool = True
) -> tuple[sv.SolverCar, dict]:
    """The car a table's laps are driven by and what that assumes
    (module docstring): ``powertrain`` or the model's latest, with its own
    shift point and gears, in standard air, braked as the driver brakes
    (``habits=False``: braked at the car's limit, the max driver's car)."""
    engines = model.power.get("engines") or {}
    name = powertrain or model.power.get("default_engine")
    if powertrain is not None and powertrain not in engines:
        raise KeyError(f"{model.car} has no powertrain {powertrain!r}")
    driven = sv.SolverCar.from_model(
        model, engine=name if name in engines else None, brake_use=habits
    )
    spec = engines.get(name, {}) if name else {}
    sets = len(spec.get("gear_sets") or [])
    assumes = {
        "car": model.car,
        "powertrain": name,
        "shift_rpm": spec.get("shift_rpm"),
        "gears": "the powertrain's own (the median day's)",
        "gear_sets_logged": sets,
        "gears_inferred": bool(spec.get("gears_inferred")),
        "air": "standard (SAE J1349)",
        "braking": "as the driver brakes" if model.driver else "at the car's limit",
        "typical_use": cg.typical_use(model),
        "basis": "dry",
    }
    if sets > 1:
        assumes["gears_note"] = f"{sets} gear sets logged (per-circuit gearing); " + (
            "the gears are inferred and the car is driven on one specific power, so the "
            "set changes nothing in the lap"
            if assumes["gears_inferred"]
            else "the set at this layout is assumed"
        )
    return driven, assumes


def support_of(model: cm.CarModel, layout_id: str) -> dict:
    """The table's support from the model's logged range of use
    (``meta.use_range``): the layout's own, else the car's over the
    layouts it has run; empty for a model without one. The fast end is
    held at a use of 1, where the table ends."""
    ranges = model.meta.get("use_range") or {}
    if layout_id in ranges:
        own = [ranges[layout_id]]
        source = "the car's laps at this layout"
    else:
        own = list(ranges.values())
        source = "the car's laps at the layouts it has run (it has not run this one)"
    if not own:
        return {}
    return {
        "use_min": float(min(r["min"] for r in own)),
        "use_max": float(min(max(r["max"] for r in own), 1.0)),
        "laps": int(sum(r["laps"] for r in own)),
        "days": int(sum(r["days"] for r in own)),
        "from": source,
    }


def build_table(
    model: cm.CarModel,
    track: trk.TrackModel,
    *,
    powertrain: str | None = None,
    knots: int = KNOTS,
    bank: bool = True,
) -> PaceTable:
    """The pace table of ``model`` at ``track``: its knots driven as one
    batch (:func:`~.solver.drive_laps`)."""
    driven, assumes = assumed_car(model, powertrain)
    profile = sv.Profile.from_track(track, bank=bank)
    use = np.exp(-np.linspace(-math.log(USE_SLOW), 0.0, knots))
    use[-1] = 1.0
    got = sv.drive_laps(profile, driven, use)
    assumes = {**assumes, "bank": bool(bank and profile.bank is not None)}
    pit = dict(track.meta.get("pit") or {})
    v_out = out_laps(profile, driven, use, pit)
    limit_car, _ = assumed_car(model, powertrain, habits=False)
    max_driver = float(sv.drive_laps(profile, limit_car, np.ones(1)).lap_time_s[0])
    return PaceTable(
        layout_id=track.layout_key,
        car=model.car,
        use=use,
        lap_time_s=np.asarray(got.lap_time_s, float),
        v_node_ms=np.asarray(got.v_node_ms, float),
        profile=profile,
        driven=driven,
        support=support_of(model, track.layout_key),
        assumes=assumes,
        pit=pit if v_out is not None else {},
        v_out_ms=v_out,
        max_driver_lap_s=max_driver,
    )


def from_dict(data: dict, model: cm.CarModel, track: trk.TrackModel) -> PaceTable:
    """A saved table against the model and the track it was built on."""
    assumes = dict(data["assumes"])
    driven, _ = assumed_car(model, assumes.get("powertrain"))
    return PaceTable(
        layout_id=str(data["layout_id"]),
        car=str(data["car"]),
        use=np.asarray(data["use"], float),
        lap_time_s=np.asarray(data["lap_time_s"], float),
        v_node_ms=np.asarray(data["v_node_ms"], float),
        profile=sv.Profile.from_track(track, bank=bool(assumes.get("bank", True))),
        driven=driven,
        support=dict(data["support"]),
        assumes=assumes,
        pit=dict(data.get("pit") or {}),
        v_out_ms=None if data.get("v_out_ms") is None else np.asarray(data["v_out_ms"], float),
        max_driver_lap_s=float(data.get("max_driver_lap_s", float("nan"))),
    )


def direct(table: PaceTable, lap_time_s: np.ndarray) -> tuple[np.ndarray, list[sv.SimLap | None]]:
    """The laps of ``lap_time_s`` solved directly with the table's car,
    one batch: ``(use, laps)`` (None where a lap time has no solution)."""
    got = sv.solve_laps(
        table.profile, table.driven, lap_time_s, guess=table.use_at_many(lap_time_s)
    )
    laps: list[sv.SimLap | None] = [
        (
            sv.lap_from_nodes(table.profile, table.driven, got.v_node_ms[i], float(got.u[i]))
            if got.ok[i]
            else None
        )
        for i in range(len(got.u))
    ]
    return np.where(got.ok, got.u, np.nan), laps
