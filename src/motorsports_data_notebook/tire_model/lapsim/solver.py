"""The quasi-steady-state lap solver (lap simulation D1, issue #206; the
part segment C's gates need, #205).

A point mass on a track's curvature, grade and vertical curvature, in the
ground frame. With ``N = cos θ + V²κ_v / g`` the normal load per weight
and ``G = u · w · g · N`` (``u`` the pace knob, ``w`` the wet grip scale
on a wet track and 1 otherwise):

- cornering: ``V²|κ| ≤ μ_y G``, solved in closed form for V;
- braking: ``|a_tyre| / (μ_b G) + V²|κ| / (μ_y G) ≤ 1`` (a diamond);
- drive: ``a_tyre = min(P / V − k·lat², r·√((μ_y G)² − (V²κ)²))``; ``r``
  is the share of the grip the driven axle has (``drive_share``: the
  rear's share of the static weight on these rear-driven cars; 1 takes
  the whole car's; issue #312), the
  engine's power is not scaled by the knob, and ``k`` is the cost of
  cornering at full throttle (``k_corner_drag``, on by default since
  gate C3). ``P`` is one number, or the engine's curve against rpm at the
  speed's rpm in the gear the driver is in: the lowest gear that is under
  the driver's shift point there (:meth:`SolverCar.power_at`);
- both ways: ``dV/dt = a_tyre − c_d V² − g · grade``.

The knob multiplies ``μ_y`` and ``μ_b`` and nothing else.

Where the road is banked (a track's ``bank`` from the elevation model,
:mod:`.bank`; in the lap since the owner's decision of 2026-11-17), with
``b`` the lean into the turn: the tyres hold ``V²|κ|·cos b − g·sin b``
sideways instead of ``V²|κ|``, under the load ``g·cos θ·cos b + V²(κ_v +
|κ|·sin b)``. The road carries a share of the cornering and presses the
car down; nothing is fitted. A lean out of the turn does the opposite,
and on a straight the tyres hold the car against the cross-slope. A
track without a bank is a flat road, and
``Profile.from_track(track, bank=False)`` takes any track as one, which
is the comparison gate C3 reports.

Integration is by trapezoidal steps in ``V²`` over the track's own nodes
(``d(V²)/ds = 2 · dV/dt``, the step's two ends averaged, the far end found
by fixed-point iteration, and by bisection where that does not settle); a
step takes ``2·ds / (V_i + V_{i+1})``. A
forward pass under the drive rule and a backward pass under the braking
rule start from the slowest apex, and the speed is the lower of the two.
The forward pass is repeated from its own closing speed until the lap
closes.

The solver works on a batch of laps of one track, each lap with its own
car and its own u (:func:`solve_laps`, :func:`drive_laps`); one lap is a
batch of one. There are two forms of it and they give the same numbers:
:mod:`._tensor` is numpy with the laps as an axis (a step along the track
is taken by every lap at once; the reference, and what runs without
numba), and :mod:`._kernel` is the same statements compiled with numba,
one lane per lap, the lanes side by side on the machine's cores.

Output is per step, not per node: a step's mean speed times its time is
its length exactly, so distance, time and the closed-lap identity
(``Σ V·long·dt = 0``) hold to rounding.

The outside reference of #206's gate D1 (two stadium ovals lapped by an
ODE reference and a closed form) is in the tests: the solver reproduces
"ellipse both sides" to 0.03 % at 2 m and 0.001 % at 0.25 m with
``brake_ellipse = 1`` and ``drive_share = μ_x / μ_y``. Production's heat
chain on a simulated lap is :func:`.heat.sim_heat`.

Not here yet (#206): the pace table and the ``outside_support`` status.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from ...tire_etl.kinematics import G_MS2
from . import _kernel, _tensor
from .car import CarModel
from .track import TrackModel

V_MAX_MS = 150.0
V_FLOOR_MS = 1.0
FIXED_POINT_ROUNDS = 4
STEP_TOL_MS = 1e-7
CLOSE_TOL_MS = 1e-3
CLOSE_MAX_ROUNDS = 20
#: Whether a powertrain whose gear was made from speed over rpm
#: (``gears_inferred`` in the model) is driven through its curve and gears.
#: Off: on the KK-SII, the one car it was measured on, the curve through the
#: gears put the held-out speed trace further out on 19 of 22 folds (S1
#: +0.04 m/s) and the simulation 0.2–0.4 m/s faster over the top quarter of
#: speeds on 21, so by the rule for a new term it stays on its level
#: (docs/tire_model.md §2.12). Its gears still give the day's gear set, the
#: shift point and the report.
INFERRED_GEARS_DRIVE = False
#: Whether the braking limit takes the shape measured for the car
#: (``grip.brake_ellipse``: how far from the diamond towards the ellipse)
#: or stays the diamond #206 wrote. On by the owner's decision of
#: 2026-11-16, with the shift time (docs/tire_model.md §2.12 C4).
BRAKE_SHAPE_DEFAULT = True
#: Whether a lap coasts for the car's ``onset_s`` between drive and
#: braking (the driver's foot changing pedals and the brake building).
BRAKE_ONSET_DEFAULT = False
#: Whether the time an upshift takes is taken out of the drive
#: (:meth:`SolverCar.power_at`, a powertrain's ``shift_s``).
SHIFT_LOSS_DEFAULT = True
#: An upshift's lost drive is taken as this share of the engine's power
#: over a band of speed just above the shift.
SHIFT_BAND_SHARE = 0.25
#: Whether the laps run compiled. On where numba is installed; a test
#: turns it off to hold the compiled lanes to the numpy ones.
USE_COMPILED = _kernel.HAVE_NUMBA
#: From how much work (laps × nodes) a batch is spread over the cores
#: (:func:`solve_laps`, :func:`drive_laps`). Waking the threads costs about
#: 10 ms, which is eight laps of Tsukuba or two of Suzuka.
PARALLEL_MIN_NODES = 8000
#: The most laps the numpy form takes in one go (it holds a dozen arrays
#: of laps × nodes; a longer batch goes through in pieces).
TENSOR_MAX_LAPS = 1024
U_RANGE = (0.3, 2.5)
U_TOL = 1e-7
#: Whether a car taken from a model carries its cornering drag. On since
#: gate C3 (#205): the term lowers the leave-one-layout-out error and the
#: error on the top quarter of speeds in ten of ten cells.
CORNER_DRAG_DEFAULT = True
#: Whether the drive out of a corner goes through the driven axle (a
#: model's ``grip.drive_share``) or has the whole car's grip. On for all
#: three cars by the owner's decision of 2026-11-23 (issue #312).
DRIVEN_AXLE_DEFAULT = True
#: Whether a wet lap's braking falls less than its cornering by the
#: car's wet braking term (a model's ``wet.brake_term``, :mod:`.wet`). Off:
#: the term is read and scored by gate D2 (#206) and has not met its
#: entry rule.
WET_BRAKE_DEFAULT = False
#: Width of the bins of a model's power curve (:data:`.car.RPM_BIN`).
CURVE_BIN_RPM = 250.0
#: Step of the table of engine power against speed a lap is driven from.
POWER_TABLE_DV = 0.25


@dataclass(frozen=True)
class Profile:
    """What the solver needs of a track: nodes ``ds_m`` apart with the
    curvature (1/m, positive in a right turn), the grade and the vertical
    curvature (1/m, positive in a compression) at each. ``closed``: the
    node after the last is the first. ``bank`` is the road's bank at each
    node (rad, positive where the left side of the direction of travel is
    higher), or None for a road taken as flat."""

    ds_m: float
    kappa: np.ndarray
    grade: np.ndarray
    kappa_v: np.ndarray
    closed: bool = True
    bank: np.ndarray | None = None

    @classmethod
    def from_track(cls, track: TrackModel, *, bank: bool = True) -> "Profile":
        """The profile of ``track``, with its bank where it has one;
        ``bank=False`` takes the road as flat."""
        n = track.n
        kv = track.kappa_v if len(track.kappa_v) == n else np.zeros(n)
        banked = bank and len(track.bank) == n
        return cls(
            ds_m=float(track.ds_m),
            kappa=np.nan_to_num(np.asarray(track.kappa_heat, float)),
            grade=np.nan_to_num(np.asarray(track.grade, float)),
            kappa_v=np.nan_to_num(np.asarray(kv, float)),
            bank=np.nan_to_num(np.asarray(track.bank, float)) if banked else None,
        )

    @classmethod
    def flat(cls, kappa: np.ndarray, ds_m: float, *, closed: bool = True) -> "Profile":
        k = np.asarray(kappa, float)
        return cls(
            ds_m=ds_m, kappa=k, grade=np.zeros_like(k), kappa_v=np.zeros_like(k), closed=closed
        )

    @property
    def n(self) -> int:
        return int(len(self.kappa))

    @property
    def length_m(self) -> float:
        return self.ds_m * (self.n if self.closed else self.n - 1)

    @property
    def lean(self) -> np.ndarray:
        """The bank as the lean into the turn at each node (rad): the
        bank with the curvature's sign, zeros on a flat road."""
        if self.bank is None:
            return np.zeros(self.n)
        return np.asarray(self.bank * np.where(self.kappa >= 0.0, 1.0, -1.0))

    def mirrored(self) -> "Profile":
        return replace(self, kappa=-self.kappa, bank=None if self.bank is None else -self.bank)

    def rolled(self, shift: int) -> "Profile":
        """The same closed track with its first node ``shift`` nodes on."""
        return replace(
            self,
            kappa=np.roll(self.kappa, -shift),
            grade=np.roll(self.grade, -shift),
            kappa_v=np.roll(self.kappa_v, -shift),
            bank=None if self.bank is None else np.roll(self.bank, -shift),
        )

    def refined(self, factor: int) -> "Profile":
        """The same track on ``factor`` times as many nodes (linear between
        the nodes; a closed track wraps)."""
        n = self.n
        if self.closed:
            x = np.arange(n * factor) / factor

            def _fine(a: np.ndarray) -> np.ndarray:
                return np.asarray(np.interp(x, np.arange(n + 1), np.append(a, a[0])))

        else:
            x = np.arange((n - 1) * factor + 1) / factor

            def _fine(a: np.ndarray) -> np.ndarray:
                return np.asarray(np.interp(x, np.arange(n), a))

        return replace(
            self,
            ds_m=self.ds_m / factor,
            kappa=_fine(self.kappa),
            grade=_fine(self.grade),
            kappa_v=_fine(self.kappa_v),
            bank=None if self.bank is None else _fine(self.bank),
        )


@dataclass(frozen=True)
class SolverCar:
    """The car's numbers as the solver takes them."""

    mu_y: float
    mu_b: float
    p_specific: float  # W/kg
    c_drag: float  # 1/m
    wet_scale: float = 1.0
    #: How much less braking falls than cornering in the wet: a wet lane's
    #: braking level is multiplied by it on top of ``wet_scale`` (1: the
    #: two fall alike; :mod:`.wet`, #206 D2).
    wet_brake: float = 1.0
    #: Cost of cornering at full throttle (1/g): the engine gives
    #: ``P/V − k·lat²`` (lat in g, the result in g).
    k_corner_drag: float = 0.0
    #: The engine as power (W/kg) against rpm, and the gears as speed per
    #: 1,000 rpm (m/s), lowest first, with their numbers. With both and a
    #: shift point the power at a speed is the curve's at the rpm of the
    #: gear the driver is in; without, it is ``p_specific`` at any speed.
    curve_rpm: tuple[float, ...] = ()
    curve_p: tuple[float, ...] = ()
    gears: tuple[float, ...] = ()
    gear_numbers: tuple[int, ...] = ()
    #: The driver's: the rpm a gear is held to.
    shift_rpm: float = float("nan")
    #: On the engine's power: the day's air (1 / the SAE factor), or a
    #: planted wrong power.
    power_scale: float = 1.0
    #: How far the braking limit is from the diamond (0: what the cornering
    #: leaves of the grip, taken linearly) towards the ellipse (1): the
    #: car's own shape, or the one the driver holds when the lap is braked
    #: as the driver brakes (``brake_use`` under 1).
    brake_ellipse: float = 0.0
    #: The driver's: the level the lap is braked at, as a share of
    #: ``mu_b`` (1: at the car's limit). The pace knob multiplies it like
    #: the limit.
    brake_use: float = 1.0
    #: Seconds between full drive and full braking: every speed peak of the
    #: lap is flat for this long (:func:`._kernel.open_peaks`).
    onset_s: float = 0.0
    #: Seconds of drive lost in an upshift out of each gear (as ``gears``;
    #: nothing for the top one). Empty: shifts cost nothing.
    shift_s: tuple[float, ...] = ()
    #: The share of what the cornering leaves of the grip that the driven
    #: axle can put down as drive (1: all of it).
    drive_share: float = 1.0

    @property
    def geared(self) -> bool:
        """Whether the engine is a curve driven through gears."""
        return bool(self.curve_rpm) and bool(self.gears) and math.isfinite(self.shift_rpm)

    def gear_and_rpm(self, v: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """``(index of the gear, rpm)`` at speed ``v``: the lowest gear
        whose rpm there is under the shift point, the top gear beyond it."""
        speed = np.atleast_1d(np.asarray(v, float))
        ratios = np.asarray(self.gears, float)
        rpm_all = speed[:, None] / ratios[None, :] * 1000.0
        ok = rpm_all <= self.shift_rpm
        gear = np.where(ok.any(axis=1), ok.argmax(axis=1), len(ratios) - 1)
        return gear, rpm_all[np.arange(len(speed)), gear]

    def power_at(self, v: np.ndarray | float) -> np.ndarray:
        """Power at the wheels per mass (W/kg) at speed ``v``, at full
        throttle in a straight line. Under the curve's lowest rpm the
        torque is held (power falls with the rpm); over its highest the
        power is."""
        speed = np.atleast_1d(np.asarray(v, float))
        if not self.geared:
            return np.full(speed.shape, self.p_specific * self.power_scale)
        power = self.power_through_gears(speed)
        for v_shift, dv in self.shift_bands():
            band = (speed > v_shift) & (speed <= v_shift + dv)
            power[band] *= 1.0 - SHIFT_BAND_SHARE
        return np.asarray(power)

    def power_through_gears(self, v: np.ndarray | float) -> np.ndarray:
        """:meth:`power_at` before the time lost in the shifts is taken out."""
        speed = np.atleast_1d(np.asarray(v, float))
        if not self.geared:
            return np.full(speed.shape, self.p_specific * self.power_scale)
        _, rpm = self.gear_and_rpm(speed)
        return np.asarray(self._curve(rpm) * self.power_scale)

    def _curve(self, rpm: np.ndarray) -> np.ndarray:
        """The engine's curve at ``rpm`` (W/kg, before the day's scale)."""
        xs, ps = np.asarray(self.curve_rpm), np.asarray(self.curve_p)
        power = np.interp(rpm, xs, ps)
        low = rpm < xs[0]
        power[low] = ps[0] * rpm[low] / xs[0]
        return np.asarray(power)

    def shift_bands(self) -> list[tuple[float, float]]:
        """``(speed, width)`` of the band above each upshift in which the
        engine gives :data:`SHIFT_BAND_SHARE` less (m/s), one per gear but
        the top one; width 0 where the shift costs nothing.

        A shift is ``shift_s`` seconds without drive: the engine's work
        ``P · shift_s`` is lost. Taken at once, on a stretch of road, it
        would make the lap time jump as u moves a corner's speed across a
        shift point; taken as a share η of the power over a band of speed
        it is the same work wherever the car's speed takes it through
        the band. On the flat the car spends ``width / ((1 − η)·P/V −
        c·V²)`` in the band, giving up ``η · P`` all the while, so the
        width that loses ``P · shift_s`` is ``shift_s · ((1 − η)·P/V −
        c·V²) / η``. A grade changes the time spent in the band and with
        it the loss, by the ratio of the accelerations; and where the car
        could not pull the gear at all the band has no width.
        """
        if not self.geared or not self.shift_s:
            return []
        out = []
        for k, lost in enumerate(self.shift_s[: len(self.gears) - 1]):
            v_shift = self.shift_rpm * self.gears[k] / 1000.0
            rpm = np.array([v_shift / self.gears[k + 1] * 1000.0])
            pull = (1.0 - SHIFT_BAND_SHARE) * float(self._curve(rpm)[0]) * self.power_scale
            room = pull / v_shift - self.c_drag * v_shift * v_shift
            width = lost * room / SHIFT_BAND_SHARE if lost > 0.0 and room > 0.0 else 0.0
            # never into the next shift's band
            reach = self.shift_rpm * self.gears[k + 1] / 1000.0 - v_shift
            out.append((float(v_shift), float(min(width, reach))))
        return out

    @classmethod
    def from_model(
        cls,
        model: CarModel,
        *,
        corner_drag: bool | None = None,
        engine: str | None = None,
        shift_rpm: float | None = None,
        gears: Sequence[float] | None = None,
        cf: float = 1.0,
        rho: float | None = None,
        through_gears: bool | None = None,
        shift_loss: bool | None = None,
        brake_shape: bool | None = None,
        onset_s: float | None = None,
        brake_onset: bool | None = None,
        brake_use: bool = True,
        driven_axle: bool | None = None,
        wet_brake: bool | None = None,
    ) -> "SolverCar":
        """The solver's car from a model.

        ``engine`` names the powertrain (None: the model's default),
        ``shift_rpm`` the driver's shift point (None: that powertrain's
        own median) and ``gears`` the gear set fitted that day, as speed
        per 1,000 rpm (None: the powertrain's own). ``through_gears`` says
        whether the car is driven through its curve and gears where it has
        them, or on ``p_specific`` (None: through them where the gear was
        logged, and by :data:`INFERRED_GEARS_DRIVE` where it was made from
        speed over rpm). ``shift_loss`` says whether the powertrain's
        measured shift times are taken out of the drive (None:
        :data:`SHIFT_LOSS_DEFAULT`), ``brake_shape`` whether the braking
        limit has the car's measured shape or is the diamond (None:
        :data:`BRAKE_SHAPE_DEFAULT`). ``onset_s`` is the day's time from
        drive to braking (None: the car's own, ``grip.onset_s``) and
        ``brake_onset`` whether the lap takes it at all (None:
        :data:`BRAKE_ONSET_DEFAULT`). ``cf`` and ``rho`` are the air of the day
        (:func:`~.car.air_state`): the engine's power is divided by the
        SAE factor and the drag goes with the density; without them the
        car runs in standard air. ``corner_drag`` says whether
        ``k_corner_drag`` is carried (None: :data:`CORNER_DRAG_DEFAULT`).
        ``brake_use`` says whether the lap is braked as the driver brakes
        (the model's ``driver``: the driver's level as a share of ``mu_b`` and
        the shape held under cornering, in place of the limit's) or at the car's limit;
        a model without a ``driver`` is braked at the limit either way.
        ``driven_axle`` says whether the drive is held to the driven
        axle's share of the grip (``grip.drive_share``; None:
        :data:`DRIVEN_AXLE_DEFAULT`); a model without one drives on the
        whole car's.
        """
        on = CORNER_DRAG_DEFAULT if corner_drag is None else corner_drag
        wet = model.wet.get("grip_scale")
        k = model.power.get("k_corner_drag") if on else 0.0
        engines = model.power.get("engines") or {}
        spec = engines.get(engine or model.power.get("default_engine") or "") or {}
        p_spec = spec.get("p_specific") or model.power.get("p_specific")
        if p_spec is None:
            raise ValueError(f"{model.car}: no power for powertrain {engine!r}")
        curve, own_gears = spec.get("curve") or [], spec.get("gears") or {}
        shift = shift_rpm if shift_rpm is not None and math.isfinite(shift_rpm) else None
        if shift is None:
            shift = spec.get("shift_rpm")
        if gears:  # the day's own set, lowest first
            ratios = sorted(float(k) for k in gears)
            numbers = list(range(1, len(ratios) + 1))
        else:
            numbers = sorted(int(g) for g in own_gears)
            ratios = [float(own_gears[str(g)]) for g in numbers]
        if through_gears is None:
            through_gears = INFERRED_GEARS_DRIVE or not spec.get("gears_inferred")
        geared = bool(curve) and bool(ratios) and shift is not None and through_gears
        lost: tuple[float, ...] = ()
        if geared and (SHIFT_LOSS_DEFAULT if shift_loss is None else shift_loss):
            lost = _shift_times_for(spec.get("shift_s") or {}, own_gears, numbers, ratios, gears)
        driver = model.driver if brake_use else {}
        use, held = driver.get("brake_use"), driver.get("brake_use_ellipse")
        ellipse = held if use is not None and held is not None else model.grip.get("brake_ellipse")
        c_drag = float(model.power["c_drag"])
        if rho is not None and math.isfinite(rho) and model.power.get("rho_ref"):
            c_drag *= rho / float(model.power["rho_ref"])
        return cls(
            mu_y=float(model.grip["mu_y"]),
            mu_b=float(model.grip["mu_b"]),
            p_specific=float(p_spec),
            c_drag=c_drag,
            wet_scale=float(wet) if wet is not None else 1.0,
            wet_brake=(
                float(model.wet.get("brake_term") or 1.0)
                if (WET_BRAKE_DEFAULT if wet_brake is None else wet_brake)
                else 1.0
            ),
            k_corner_drag=float(k) if k is not None else 0.0,
            # the curve's bins are stored by their lower edge
            curve_rpm=tuple(float(r) + CURVE_BIN_RPM / 2.0 for r, _ in curve) if geared else (),
            curve_p=tuple(float(p) for _, p in curve) if geared else (),
            gears=tuple(ratios) if geared else (),
            gear_numbers=tuple(numbers) if geared else (),
            shift_rpm=float(shift) if geared and shift is not None else float("nan"),
            power_scale=1.0 / cf if cf and math.isfinite(cf) else 1.0,
            shift_s=lost,
            brake_ellipse=(
                float(ellipse or 0.0)
                if (BRAKE_SHAPE_DEFAULT if brake_shape is None else brake_shape)
                else 0.0
            ),
            brake_use=float(use) if use is not None else 1.0,
            drive_share=(
                float(model.grip.get("drive_share") or 1.0)
                if (DRIVEN_AXLE_DEFAULT if driven_axle is None else driven_axle)
                else 1.0
            ),
            onset_s=(
                float(
                    onset_s
                    if onset_s is not None and math.isfinite(onset_s)
                    else model.grip.get("onset_s") or 0.0
                )
                if (BRAKE_ONSET_DEFAULT if brake_onset is None else brake_onset)
                else 0.0
            ),
        )


def _shift_times_for(
    shift_s: dict,
    own_gears: dict,
    numbers: Sequence[int],
    ratios: Sequence[float],
    day_gears: Sequence[float] | None,
) -> tuple[float, ...]:
    """A powertrain's shift times (by its own gear numbers) laid on the
    gears the car is driven through. With the day's own gear set the gears
    are numbered from 1: a set as long as the powertrain's is matched in
    order, another by the nearest ratio. A gear without a measured time
    takes the median of those that have one."""
    if not shift_s:
        return ()
    known = {int(g): float(v) for g, v in shift_s.items()}
    typical = float(np.median(list(known.values())))
    own = sorted((float(r), int(g)) for g, r in own_gears.items())
    if not day_gears:
        got = [known.get(int(n), typical) for n in numbers]
    elif len(own) == len(ratios):
        got = [known.get(g, typical) for _, g in own]
    else:
        got = []
        for ratio in ratios:
            near = min(own, key=lambda rg: abs(rg[0] / ratio - 1.0)) if own else None
            hit = near is not None and abs(near[0] / ratio - 1.0) < 0.03
            got.append(known.get(near[1], typical) if hit and near is not None else typical)
    return tuple(got[:-1]) + (0.0,) if got else ()


@dataclass
class SimLap:
    """A simulated lap, one row per step of the track.

    ``s_m`` and ``t_s`` are where and when the step begins, ``dt_s`` how
    long it takes and ``v_ms`` its mean speed. ``long_kin_g`` is ``dV/dt``;
    ``long_sf_g`` adds the grade (what an accelerometer reads);
    ``long_tyre_g`` adds drag (what the tyres supply). ``v_node_ms`` is
    the speed at the nodes.
    """

    s_m: np.ndarray
    t_s: np.ndarray
    dt_s: np.ndarray
    v_ms: np.ndarray
    lat_g: np.ndarray
    long_kin_g: np.ndarray
    long_sf_g: np.ndarray
    long_tyre_g: np.ndarray
    v_node_ms: np.ndarray
    u: float
    status: str = "ok"
    #: What the tyres hold sideways (g, signed as ``lat_g``): ``lat_g`` on
    #: a flat road, less the bank's share on a banked one. None: as ``lat_g``.
    lat_tyre_g: np.ndarray | None = None

    @property
    def lap_time_s(self) -> float:
        return float(self.dt_s.sum())


def _road(profile: Profile) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The track's arrays as the passes take them: ``(kap, grd, kv, cos,
    gsb)``.

    On a flat road: ``|κ|``, the grade, the vertical curvature, the
    grade's cosine and zeros. On a road that leans into the turn by the
    angle b (:attr:`Profile.lean`) the bank is folded into them:
    ``|κ|·cos b``, the grade, ``κ_v + |κ|·sin b``, ``cos θ·cos b`` and
    ``g·sin b``. The passes then read the tyres' sideways hold as
    ``V²·kap − gsb`` and the load as ``g·cos + V²·kv``, which on the flat
    road are the numbers they always were, to the last bit."""
    kap = np.abs(profile.kappa)
    grd = np.ascontiguousarray(profile.grade, dtype=float)
    kv = np.ascontiguousarray(profile.kappa_v, dtype=float)
    cos = 1.0 / np.sqrt(1.0 + profile.grade**2)
    if profile.bank is None:
        return kap, grd, kv, cos, np.zeros(profile.n)
    lean = profile.lean
    sin_b, cos_b = np.sin(lean), np.cos(lean)
    return kap * cos_b, grd, kv + kap * sin_b, cos * cos_b, G_MS2 * sin_b


def corner_speed_limit(profile: Profile, mu: float) -> np.ndarray:
    """The speed at which cornering alone takes the whole of the grip
    ``mu`` at each node: ``V² (|κ| − mu·κ_v) = mu·g·cos θ`` on a flat
    road, and with the bank's share on a banked one (:func:`_road`);
    capped at :data:`V_MAX_MS` where the track does not limit it."""
    kap, _, kv, cos, gsb = _road(profile)
    den = kap - mu * kv
    with np.errstate(divide="ignore", invalid="ignore"):
        v2 = np.where(den > 1e-9, (mu * G_MS2 * cos + gsb) / np.maximum(den, 1e-9), V_MAX_MS**2)
    v2 = np.where(v2 < V_FLOOR_MS**2, V_FLOOR_MS**2, v2)
    return np.asarray(np.minimum(np.sqrt(v2), V_MAX_MS))


@lru_cache(maxsize=512)
def _power_table(car: SolverCar) -> np.ndarray:
    """The engine's power against speed, through the gears, at steps of
    :data:`POWER_TABLE_DV`: read linearly by the passes (a gear change is
    spread over one step of it). Kept per car: it does not change with u."""
    grid = np.arange(0.0, V_MAX_MS + 10.0, POWER_TABLE_DV)
    table = car.power_through_gears(grid)
    # A shift's band takes from each grid point the share of its own cell
    # that lies in the band, so the band read linearly off the table is as
    # wide as it is meant to be wherever its edges fall between points.
    half = POWER_TABLE_DV / 2.0
    for v_shift, dv in car.shift_bands():
        inside = np.minimum(grid + half, v_shift + dv) - np.maximum(grid - half, v_shift)
        table = table * (1.0 - SHIFT_BAND_SHARE * np.clip(inside / POWER_TABLE_DV, 0.0, 1.0))
    # The last entry is not power: it is the driven axle's share of the
    # grip, which the passes read from here so that it rides with the car
    # (no speed reaches it: the grid ends 10 m/s over :data:`V_MAX_MS`).
    table = np.append(table, car.drive_share)
    table.setflags(write=False)
    return table


def _passes(
    profile: Profile,
    car: SolverCar,
    u: float,
    wet: bool,
    v_start: float | None,
    v_end: float | None,
) -> np.ndarray:
    """Node speeds of ``profile`` (n nodes; a closed profile is returned
    from its own first node): the compiled passes where numba is installed
    (:mod:`._kernel`), else a batch of one through :mod:`._tensor`, which
    they reproduce."""
    if not USE_COMPILED:
        _, v = _tensor.drive_lanes(
            np.array([float(u)]),
            *_lanes(profile, [car], wet),
            closed=profile.closed,
            v_start=float("nan") if v_start is None else float(v_start),
            v_end=float("nan") if v_end is None else float(v_end),
        )
        return np.asarray(v[0])
    w = car.wet_scale if wet else 1.0
    mu = car.mu_y * u * w
    v_lim = corner_speed_limit(profile, mu)
    n = profile.n
    start = int(np.argmin(v_lim)) if profile.closed else 0
    idx = (np.arange(n) + start) % n
    kap, grd, kv, cos, gsb = _road(profile)
    v = _kernel.passes(
        np.ascontiguousarray(kap[idx]),
        np.ascontiguousarray(grd[idx]),
        np.ascontiguousarray(kv[idx]),
        np.ascontiguousarray(cos[idx]),
        np.ascontiguousarray(gsb[idx]),
        np.ascontiguousarray(v_lim[idx]),
        _power_table(car),
        1.0 / POWER_TABLE_DV,
        mu,
        car.mu_b * car.brake_use * u * w * (car.wet_brake if wet else 1.0),
        car.c_drag,
        car.k_corner_drag,
        car.brake_ellipse,
        G_MS2,
        profile.ds_m,
        profile.closed,
        float("nan") if v_start is None else float(v_start),
        float("nan") if v_end is None else float(v_end),
        V_FLOOR_MS,
        FIXED_POINT_ROUNDS,
        STEP_TOL_MS,
        CLOSE_TOL_MS,
        CLOSE_MAX_ROUNDS,
    )
    if car.onset_s > 0.0:
        flat = np.empty(n)
        _kernel.open_peaks(v, profile.ds_m, car.onset_s, profile.closed, flat)
        v = flat
    out = np.empty(n)
    out[idx] = v
    return out


def simulate_lap(
    track: TrackModel | Profile,
    car: SolverCar,
    u: float = 1.0,
    *,
    wet: bool = False,
    v_start: float | None = None,
    v_end: float | None = None,
) -> SimLap:
    """One lap of ``track`` by ``car`` at the pace knob ``u``.

    An open :class:`Profile` is driven from its first node (at ``v_start``,
    or the node's cornering limit) to its last (braked down to ``v_end``
    when given).
    """
    profile = track if isinstance(track, Profile) else Profile.from_track(track)
    return lap_from_nodes(profile, car, _passes(profile, car, u, wet, v_start, v_end), u)


def lap_from_nodes(profile: Profile, car: SolverCar, v: np.ndarray, u: float) -> SimLap:
    """The :class:`SimLap` of a lap given by its speed at the profile's
    nodes (one row of :func:`drive_laps` or :func:`solve_laps`): what
    :func:`simulate_lap` makes of the same speeds, step for step."""
    v = np.asarray(v, dtype=float)
    n = profile.n
    if profile.closed:
        nxt = np.roll(np.arange(n), -1)
        first = np.arange(n)
    else:
        first, nxt = np.arange(n - 1), np.arange(1, n)
    v_i, v_j = v[first], v[nxt]
    v_mid = 0.5 * (v_i + v_j)
    dt = profile.ds_m / v_mid
    kin = (v_j * v_j - v_i * v_i) / (2.0 * profile.ds_m * G_MS2)
    grade = 0.5 * (profile.grade[first] + profile.grade[nxt])
    lat = 0.5 * (v_i * v_i * profile.kappa[first] + v_j * v_j * profile.kappa[nxt]) / G_MS2
    lat_tyre = lat
    if profile.bank is not None:
        # signed as the curvature: what the tyres hold of it, the bank's share off
        kap, _, _, _, gsb = _road(profile)
        side = np.where(profile.kappa >= 0.0, 1.0, -1.0)
        hold_i = side[first] * (v_i * v_i * kap[first] - gsb[first])
        hold_j = side[nxt] * (v_j * v_j * kap[nxt] - gsb[nxt])
        lat_tyre = 0.5 * (hold_i + hold_j) / G_MS2
    drag = car.c_drag * 0.5 * (v_i * v_i + v_j * v_j) / G_MS2
    return SimLap(
        s_m=first * profile.ds_m,
        t_s=np.concatenate([[0.0], np.cumsum(dt)[:-1]]),
        dt_s=dt,
        v_ms=v_mid,
        lat_g=lat,
        long_kin_g=kin,
        long_sf_g=kin + grade,
        long_tyre_g=kin + grade + drag,
        v_node_ms=v,
        u=float(u),
        lat_tyre_g=lat_tyre,
    )


def lap_time(track: TrackModel | Profile, car: SolverCar, u: float, *, wet: bool = False) -> float:
    """The lap time at ``u`` (s): :attr:`SimLap.lap_time_s` without the
    rest of the lap being put together."""
    profile = track if isinstance(track, Profile) else Profile.from_track(track)
    v = _passes(profile, car, u, wet, None, None)
    if profile.closed:
        v_mid = 0.5 * (v + np.roll(v, -1))
    else:
        v_mid = 0.5 * (v[:-1] + v[1:])
    return float(np.sum(profile.ds_m / v_mid))


def solve_u(
    track: TrackModel | Profile, car: SolverCar, lap_time_s: float, *, wet: bool = False
) -> tuple[float, str]:
    """The pace knob at which the car laps in ``lap_time_s``, by bisection
    over :data:`U_RANGE` (lap time falls as u rises). ``status`` is ``ok``
    or ``no_root`` (the lap time is outside what the range can give; u is
    then NaN)."""
    profile = track if isinstance(track, Profile) else Profile.from_track(track)
    lo, hi = U_RANGE
    t_lo, t_hi = lap_time(profile, car, lo, wet=wet), lap_time(profile, car, hi, wet=wet)
    if not (t_hi <= lap_time_s <= t_lo):
        return float("nan"), "no_root"
    while hi - lo > U_TOL:
        mid = 0.5 * (lo + hi)
        if lap_time(profile, car, mid, wet=wet) > lap_time_s:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), "ok"


def solve_u_near(
    track: TrackModel | Profile,
    car: SolverCar,
    lap_time_s: float,
    guess: float = 1.0,
    *,
    wet: bool = False,
) -> tuple[float, str]:
    """:func:`solve_u` for a lap whose u is expected near ``guess``: a
    bracket is opened around the guess and widened until it holds the lap
    time, then Brent's method. About eight laps instead of two dozen; the
    answer is the same root. Falls back to :func:`solve_u` when the
    bracket reaches :data:`U_RANGE` without holding it. A batch of one
    (:func:`solve_laps`)."""
    profile = track if isinstance(track, Profile) else Profile.from_track(track)
    if USE_COMPILED and profile.closed:
        # the same search, compiled (`_kernel._solve_u`): what a lap of a
        # batch gets (:func:`solve_laps`)
        lanes = _lanes(profile, [car], wet)
        kap, grd, kv, cos, gsb, ptabs, _, mu_y, mu_b, w, c, kc, be, ons, cst = lanes
        u, status = _kernel._solve_u(
            float(lap_time_s), float(guess), kap, grd, kv, cos, gsb, ptabs[0], mu_y[0], mu_b[0],
            w[0], c[0], kc[0], be[0], ons[0], cst, np.empty(profile.n),
        )  # fmt: skip
        return float(u), "ok" if status == _kernel.OK else "no_root"
    us, found, _ = _tensor.solve_lanes(
        np.array([float(lap_time_s)]),
        np.array([float(guess)]),
        *_lanes(profile, [car], wet),
        closed=profile.closed,
    )
    return float(us[0]), "ok" if found[0] == _kernel.OK else "no_root"


@dataclass
class LapBatch:
    """A batch of laps of one track, one row per lap: its pace knob, whether
    the car can do it, its lap time and its speed at the track's nodes
    (NaN where it cannot)."""

    u: np.ndarray
    ok: np.ndarray
    lap_time_s: np.ndarray
    v_node_ms: np.ndarray

    @property
    def status(self) -> list[str]:
        """``ok`` or ``no_root`` per lap, as :func:`solve_u` names them."""
        return ["ok" if good else "no_root" for good in self.ok]


def _cars_of(cars: SolverCar | Sequence[SolverCar], laps: int) -> list[SolverCar]:
    got = [cars] * laps if isinstance(cars, SolverCar) else list(cars)
    if len(got) != laps:
        raise ValueError(f"{len(got)} cars for {laps} laps")
    return got


def _lanes(profile: Profile, cars: list[SolverCar], wet: bool) -> tuple:
    """What either form of the batch takes of a track and the cars of its laps:
    the track's arrays (:func:`_road`), the power tables (one row per
    distinct car), which of them each lap drives, the laps' own numbers
    and the constants."""
    index: dict[SolverCar, int] = {}
    which = np.array([index.setdefault(car, len(index)) for car in cars], dtype=np.int64)
    cst = np.zeros(12)
    cst[_kernel.INV_DV] = 1.0 / POWER_TABLE_DV
    cst[_kernel.G] = G_MS2
    cst[_kernel.DS] = profile.ds_m
    cst[_kernel.V_MAX] = V_MAX_MS
    cst[_kernel.FLOOR] = V_FLOOR_MS
    cst[_kernel.STEP_TOL] = STEP_TOL_MS
    cst[_kernel.CLOSE_TOL] = CLOSE_TOL_MS
    cst[_kernel.ROUNDS] = FIXED_POINT_ROUNDS
    cst[_kernel.CLOSE_ROUNDS] = CLOSE_MAX_ROUNDS
    cst[_kernel.U_MIN], cst[_kernel.U_MAX] = U_RANGE
    cst[_kernel.U_TOL] = U_TOL
    return (
        *_road(profile),
        np.ascontiguousarray(np.stack([_power_table(car) for car in index])),
        which,
        np.array([car.mu_y for car in cars], dtype=float),
        np.array(
            [car.mu_b * car.brake_use * (car.wet_brake if wet else 1.0) for car in cars],
            dtype=float,
        ),
        np.array([car.wet_scale if wet else 1.0 for car in cars], dtype=float),
        np.array([car.c_drag for car in cars], dtype=float),
        np.array([car.k_corner_drag for car in cars], dtype=float),
        np.array([car.brake_ellipse for car in cars], dtype=float),
        np.array([car.onset_s for car in cars], dtype=float),
        cst,
    )


def solve_laps(
    track: TrackModel | Profile,
    cars: SolverCar | Sequence[SolverCar],
    lap_time_s: Sequence[float] | np.ndarray,
    *,
    guess: float | Sequence[float] | np.ndarray = 1.0,
    wet: bool = False,
) -> LapBatch:
    """:func:`solve_u_near` for many laps of one track at once, and each
    lap driven at the u found.

    ``cars`` is one car for every lap or one per lap (a lap's own
    powertrain, shift point and air), ``guess`` likewise. The laps have
    nothing in common but the track: compiled, they are solved side by
    side on the machine's cores; in numpy they are an axis of every array
    and go round the track together. Either way every lap's u is what
    :func:`solve_u_near` gives it alone.
    """
    profile = track if isinstance(track, Profile) else Profile.from_track(track)
    target = np.ascontiguousarray(lap_time_s, dtype=float)
    laps = len(target)
    own = _cars_of(cars, laps)
    start = np.ascontiguousarray(np.broadcast_to(np.asarray(guess, dtype=float), (laps,)))
    if laps == 0:
        return LapBatch(np.empty(0), np.empty(0, dtype=bool), target, np.empty((0, profile.n)))
    if USE_COMPILED and profile.closed:
        spread = laps > 1 and laps * profile.n >= PARALLEL_MIN_NODES
        run = _kernel.solve_lanes_parallel if spread else _kernel.solve_lanes
        u, status, v = run(target, start, *_lanes(profile, own, wet))
    else:
        parts = [
            _tensor.solve_lanes(
                target[k : k + TENSOR_MAX_LAPS],
                start[k : k + TENSOR_MAX_LAPS],
                *_lanes(profile, own[k : k + TENSOR_MAX_LAPS], wet),
                closed=profile.closed,
            )
            for k in range(0, laps, TENSOR_MAX_LAPS)
        ]
        u, status, v = (np.concatenate([part[k] for part in parts]) for k in range(3))
    ok = status == _kernel.OK
    return LapBatch(u, ok, np.where(ok, target, np.nan), v)


def drive_laps(
    track: TrackModel | Profile,
    cars: SolverCar | Sequence[SolverCar],
    u: Sequence[float] | np.ndarray,
    *,
    wet: bool = False,
) -> LapBatch:
    """Many laps of one track, each at its own ``u`` (and its own car, as
    in :func:`solve_laps`): :func:`simulate_lap`'s node speeds and lap
    times, side by side on the cores where the passes are compiled."""
    profile = track if isinstance(track, Profile) else Profile.from_track(track)
    us = np.ascontiguousarray(u, dtype=float)
    laps = len(us)
    own = _cars_of(cars, laps)
    if laps == 0:
        return LapBatch(us, np.empty(0, dtype=bool), np.empty(0), np.empty((0, profile.n)))
    if USE_COMPILED and profile.closed:
        # a lap driven is about a tenth of a lap solved
        spread = laps > 1 and laps * profile.n >= 10 * PARALLEL_MIN_NODES
        run = _kernel.drive_lanes_parallel if spread else _kernel.drive_lanes
        t, v = run(us, *_lanes(profile, own, wet))
    else:
        parts = [
            _tensor.drive_lanes(
                us[k : k + TENSOR_MAX_LAPS],
                *_lanes(profile, own[k : k + TENSOR_MAX_LAPS], wet),
                closed=profile.closed,
            )
            for k in range(0, laps, TENSOR_MAX_LAPS)
        ]
        t, v = (np.concatenate([part[k] for part in parts]) for k in range(2))
    return LapBatch(us, np.ones(laps, dtype=bool), np.asarray(t), v)


@dataclass
class PaceCurve:
    """Lap time against the knob for one track and car, for solving many
    laps: the bracket comes from the curve and the root from Brent's
    method inside it, a handful of laps instead of the two dozen of
    :func:`solve_u`."""

    profile: Profile
    car: SolverCar
    wet: bool
    u: np.ndarray
    t: np.ndarray  # falls as u rises

    @classmethod
    def build(
        cls, track: TrackModel | Profile, car: SolverCar, *, wet: bool = False, n: int = 13
    ) -> "PaceCurve":
        profile = track if isinstance(track, Profile) else Profile.from_track(track)
        u = np.geomspace(0.4, 1.7, n)
        t = np.array([lap_time(profile, car, float(x), wet=wet) for x in u])
        return cls(profile=profile, car=car, wet=wet, u=u, t=t)

    def solve(self, lap_time_s: float) -> tuple[float, str]:
        """``(u, status)`` as :func:`solve_u`; a lap time outside the curve
        falls back to it."""
        from scipy.optimize import brentq

        k = int(np.searchsorted(-self.t, -lap_time_s)) - 1
        if not np.isfinite(lap_time_s) or k < 0 or k >= len(self.u) - 1:
            return solve_u(self.profile, self.car, lap_time_s, wet=self.wet)
        root = brentq(
            lambda x: lap_time(self.profile, self.car, x, wet=self.wet) - lap_time_s,
            float(self.u[k]),
            float(self.u[k + 1]),
            xtol=U_TOL,
        )
        return float(root), "ok"


def combined_use(sim: SimLap, profile: Profile, car: SolverCar, *, wet: bool = False) -> np.ndarray:
    """How much of the envelope each step uses under the rule that binds
    it: under braking the tyre's deceleration over what the cornering
    leaves of the braking limit (the diamond, or the car's own shape
    between it and the ellipse) where it leaves any, else the lateral over
    its limit; under drive the two combine as an ellipse. 1 is the limit
    the lap is driven to: under braking the driver's level where the car
    carries one (``brake_use``)."""
    w = car.wet_scale if wet else 1.0
    n = profile.n
    first = np.arange(n) if profile.closed else np.arange(n - 1)
    nxt = np.roll(np.arange(n), -1) if profile.closed else np.arange(1, n)
    cos = 1.0 / np.sqrt(1.0 + (0.5 * (profile.grade[first] + profile.grade[nxt])) ** 2)
    load = cos + sim.v_ms**2 * 0.5 * (profile.kappa_v[first] + profile.kappa_v[nxt]) / G_MS2
    lat = np.abs(sim.lat_g)
    if profile.bank is not None and sim.lat_tyre_g is not None:
        # the road's share: the load it adds and the lateral it takes
        lean = 0.5 * (profile.lean[first] + profile.lean[nxt])
        bend = 0.5 * (np.abs(profile.kappa[first]) + np.abs(profile.kappa[nxt]))
        load = (
            cos * np.cos(lean)
            + sim.v_ms**2
            * (0.5 * (profile.kappa_v[first] + profile.kappa_v[nxt]) + bend * np.sin(lean))
            / G_MS2
        )
        lat = np.abs(sim.lat_tyre_g)
    a_y = car.mu_y * sim.u * w * load
    a_b = car.mu_b * car.brake_use * sim.u * w * (car.wet_brake if wet else 1.0) * load
    tyre = sim.long_tyre_g
    used = np.minimum(lat / a_y, 1.0)
    left = (1.0 - car.brake_ellipse) * (1.0 - used) + car.brake_ellipse * np.sqrt(1.0 - used**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        braking = np.where(left > 0.0, -tyre / (a_b * left), lat / a_y)
    if car.brake_ellipse == 0.0:  # the diamond as a sum, as #206 writes it
        braking = -tyre / a_b + lat / a_y
    # under drive the driven axle has ``drive_share`` of what the cornering leaves
    driven = tyre / car.drive_share
    return np.asarray(np.where(tyre < 0.0, braking, np.sqrt(driven**2 + lat**2) / a_y))


__all__ = [
    "LapBatch",
    "PaceCurve",
    "Profile",
    "SimLap",
    "SolverCar",
    "combined_use",
    "corner_speed_limit",
    "drive_laps",
    "lap_time",
    "simulate_lap",
    "solve_laps",
    "solve_u",
    "solve_u_near",
]
