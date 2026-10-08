"""Heat-input forms for the tire energy balance.

The energy balance is ``m·c·dT/dt = α·q(t) − h·(T − T_eff)``. This
module defines ``q(t)``, the *driving intensity* the heat input is
proportional to, from the logger channels (lateral g, longitudinal g, speed).

Forms
-----
``g2`` (the v0–v0.27 proxy)
    ``q = lat_g² + long_g²`` in G². No speed, no slip: the fitted α silently
    contains ⟨V⟩ and the load²/stiffness ratio.

``sliding`` (sliding power, ChassisSim eq. 6 with one slip for both axes)
    Friction heating is force × slip velocity. The force magnitude is the
    friction-circle total ``g = √(lat_g² + long_g²)`` (per-corner load is a
    constant absorbed in α), the slip velocity is ``V · s(g)`` with ``s`` the
    slip proxy below, so

        q = V · g · s(g)          [units G² · m/s]

    Combining the lateral and longitudinal products in quadrature with one
    shared slip, as the paper does, reduces to exactly this expression.

    The slip proxy ``s(g)`` stands in for the slip angle / slip ratio the
    logger does not record. A tire's force saturates with slip; the inverse
    of a ``tanh`` saturation ``F = F_lim · tanh(C·s / F_lim)`` is

        s(g) = g_lim · atanh(min(g / g_lim, u_max))

    which is ``≈ g`` (linear slip, cornering stiffness in α) well below the
    car's grip limit ``g_lim`` and grows steeply as the tire is driven at the
    limit, where most of the sliding happens. ``g_lim = ∞`` recovers linear
    slip, ``q = V·g²``. ``u_max`` caps the activation so curb strikes and GPS
    glitches above ``g_lim`` stay finite.

    ``rr`` adds rolling-resistance / free-rolling hysteresis heating,
    ``q += V · rr``: power proportional to speed at zero g (``rr`` in G²,
    the g² whose sliding heat it equals).

``power``
    ``q = V · g^p``: a pure power law, the agnostic alternative to the
    activation (``p = 2`` is linear slip).

``gv`` (production since schema v5)
    ``q = g^p · V^m`` with ``p = 1, m = 1`` by default (G·m/s): **force ×
    slip fraction × speed**. A race tyre runs near its optimum slip, a
    kinematic fraction that does not grow with force, so the dissipated
    power is the force the tyre produces (∝ |g|) times that fraction times
    the rolling speed. ``p = 2`` would be the small-slip brush model
    (slip ∝ force), which double-counts the force in the racing regime;
    ``|g|·V`` beat it on held-out pressure, per-track constancy and the
    wet/dry ratio (2026-10-09).

All forms clip the total g at ``g_clip`` (default 3 G) before use: single
25 Hz samples above that are logger glitches (one 33 G sample was found in
the dataset), not driving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

FORMS = ("g2", "sliding", "power", "g2v", "gv")
DEFAULT_G_CLIP = 3.0


@dataclass(frozen=True)
class HeatInput:
    """A choice of heat-input form and its (non-fitted) constants.

    ``g_lim`` is per car (fit-pooled car label); a car missing from the map
    uses ``g_lim_default`` (``inf`` = linear slip).
    """

    form: str = "g2"
    g_lim: Mapping[str, float] = field(default_factory=dict)
    g_lim_default: float = float("inf")
    u_max: float = 0.98
    rr: float = 0.0
    p: float = 2.0
    m: float = 0.0  # speed exponent of the ``g2v`` form
    g_clip: float = DEFAULT_G_CLIP

    def __post_init__(self) -> None:
        if self.form not in FORMS:
            raise ValueError(f"form must be one of {FORMS}; got {self.form!r}")
        if not 0.0 < self.u_max < 1.0:
            raise ValueError(f"u_max must be in (0, 1); got {self.u_max}")

    @property
    def units(self) -> str:
        if self.form == "gv":
            return f"G^{self.p:g}*(m/s)^{self.m:g}"
        return "G^2" if self.form in ("g2", "g2v") else "G^2*m/s"

    @property
    def force_exp(self) -> float:
        """Exponent of the force magnitude in the per-corner split."""
        return self.p if self.form in ("gv", "power") else 2.0

    @property
    def speed_exp(self) -> float:
        """Exponent of ``V/V_ref`` in the per-corner split."""
        if self.form in ("gv", "g2v"):
            return self.m
        return 1.0 if self.form in ("sliding", "power") else 0.0

    def label(self, car: str | None = None) -> str:
        if self.form == "g2":
            return "g2"
        if self.form == "power":
            s = f"V*g^{self.p:g}"
        elif self.form == "g2v":
            s = f"g^2*V^{self.m:g}"
        elif self.form == "gv":
            s = f"g^{self.p:g}*V^{self.m:g}"
        else:
            gl = self.g_lim_for(car) if car else None
            s = "V*g^2" if gl is None or not np.isfinite(gl) else f"V*g*atanh(g/{gl:.2f})"
        if self.rr:
            s += f"+V*{self.rr:g}"
        return s

    def g_lim_for(self, car: str) -> float:
        return float(self.g_lim.get(car, self.g_lim_default))

    def slip(self, g: np.ndarray, car: str) -> np.ndarray:
        """Slip proxy ``s(g)`` (G) for the ``sliding`` form."""
        g_lim = self.g_lim_for(car)
        if not np.isfinite(g_lim):
            return g
        u = np.minimum(g / g_lim, self.u_max)
        return g_lim * np.arctanh(u)

    def rate(
        self,
        lat_g: np.ndarray,
        long_g: np.ndarray,
        speed_ms: np.ndarray,
        car: str,
    ) -> np.ndarray:
        """Driving intensity ``q`` per sample. NaN channels count as zero."""
        lat = np.nan_to_num(np.asarray(lat_g, dtype=float), nan=0.0)
        lng = np.nan_to_num(np.asarray(long_g, dtype=float), nan=0.0)
        g = np.minimum(np.sqrt(lat * lat + lng * lng), self.g_clip)
        if self.form == "g2":
            return np.asarray(g * g)
        v = np.nan_to_num(np.asarray(speed_ms, dtype=float), nan=0.0)
        v = np.maximum(v, 0.0)
        if self.form == "g2v":
            return np.asarray(g * g * v**self.m)
        if self.form == "gv":
            return np.asarray(g**self.p * v**self.m)
        if self.form == "power":
            heat = g**self.p
        else:
            heat = g * self.slip(g, car)
        return np.asarray(v * (heat + self.rr))


DEFAULT_HEAT_INPUT = HeatInput()
# Production (schema v5): force × slip fraction × speed, q = |g| · V / V_ref,
# with the per-corner force-path split of :func:`corner_heat_parts` on top.
PHYSICAL_HEAT_INPUT = HeatInput(form="gv", p=1.0, m=1.0)


# ------------------------------------------------------------ load transfer


@dataclass(frozen=True)
class CarGeometry:
    """Quasi-static load-transfer constants of a car (dimensionless).

    ``wdf``: static front axle share of the weight; ``xi_x = h/L`` (CG
    height over wheelbase) scales longitudinal transfer, ``xi_y = h/t`` (CG
    height over track) lateral transfer; ``rld`` is the front share of the
    lateral transfer (roll-stiffness distribution).
    """

    wdf: float
    xi_x: float
    xi_y: float
    rld: float
    driven: str = "rear"  # axle that produces the drive force ("rear" | "front" | "all")

    def is_driven(self, corner: str) -> bool:
        return self.driven == "all" or corner[0] == self.driven[0]

    def corner_load(self, lat_g: np.ndarray, long_g: np.ndarray, corner: str) -> np.ndarray:
        """Vertical load on ``corner`` as a fraction of the car's weight.
        Positive ``lat_g`` is a right turn (loads the LEFT tyres), positive
        ``long_g`` is acceleration (loads the rear). Clipped at zero (lift)."""
        lat = np.nan_to_num(np.asarray(lat_g, dtype=float), nan=0.0)
        lng = np.nan_to_num(np.asarray(long_g, dtype=float), nan=0.0)
        front = corner[0] == "f"
        left = corner[1] == "l"
        static = (self.wdf if front else 1.0 - self.wdf) / 2.0
        dx = -self.xi_x * lng / 2.0 if front else self.xi_x * lng / 2.0
        share = self.rld if front else 1.0 - self.rld
        dy = self.xi_y * share * lat * (1.0 if left else -1.0)
        return np.asarray(np.maximum(static + dx + dy, 0.0))

    def static_load(self, corner: str) -> float:
        return (self.wdf if corner[0] == "f" else 1.0 - self.wdf) / 2.0


# Nominal geometry per (fit-pooled) car. These are spec-sheet estimates, not
# fitted: Toyota 86 — wheelbase 2.57 m, track 1.52 m, CG ≈ 0.46 m, 53:47;
# FJ-class single-seater — wheelbase ≈ 2.4 m, track ≈ 1.35 m, CG ≈ 0.28 m,
# 42:58. Override per run when better numbers are known.
BUILTIN_GEOMETRY: dict[str, CarGeometry] = {
    "Inferno 86": CarGeometry(wdf=0.53, xi_x=0.18, xi_y=0.30, rld=0.55),
    "FJ": CarGeometry(wdf=0.42, xi_x=0.12, xi_y=0.21, rld=0.45),
}


def corner_load_ratio(
    lat_g: np.ndarray, long_g: np.ndarray, corner: str, geom: CarGeometry
) -> np.ndarray:
    """``W_i(t) / W_i,static`` for a corner: the load-transfer multiplier."""
    return np.asarray(geom.corner_load(lat_g, long_g, corner) / geom.static_load(corner))


def corner_heat_parts(
    lat_g: np.ndarray,
    long_g: np.ndarray,
    speed_ms: np.ndarray,
    corner: str,
    geom: CarGeometry,
    g_clip: float = DEFAULT_G_CLIP,
    v_ref_ms: float = 30.0,
    speed_exp: float = 1.0,
    force_exp: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-corner heat input split by how the force reaches the tyre.

    Dissipated power at a corner is its force × slip velocity. Lateral and
    braking force are shared in proportion to load (brake bias ≈ the
    dynamic load split), drive force sits on the driven axle only. With
    the slip a kinematic fraction near the tyre's optimum (``force_exp =
    1``, production) the slip velocity is that fraction times the rolling
    speed; with the small-slip brush model (``force_exp = 2``) it would be
    ∝ force/load instead. Returns ``(base, drive, brake_power)``::

        base        = r_i · g_lb^p · (V/V_ref)^m      g_lb = √(lat_g² + brake_g²)
        drive       = acc_g^p / r_i^(p−1) · (V/V_ref)^m   on driven corners, else 0
        brake_power = brake_g · V / V_ref              on front corners, else 0

    ``brake_power`` is the disc energy that reaches the hub and gas directly
    (a power, ∝ g·V, not a slip term); its coefficient is fitted per car.
    """
    lat = np.nan_to_num(np.asarray(lat_g, dtype=float), nan=0.0)
    lng = np.nan_to_num(np.asarray(long_g, dtype=float), nan=0.0)
    v = np.maximum(np.nan_to_num(np.asarray(speed_ms, dtype=float), nan=0.0), 0.0)
    lat = np.clip(lat, -g_clip, g_clip)
    lng = np.clip(lng, -g_clip, g_clip)
    brake = np.maximum(-lng, 0.0)
    acc = np.maximum(lng, 0.0)
    r = np.maximum(corner_load_ratio(lat, lng, corner, geom), 0.05)
    vm = (v / v_ref_ms) ** speed_exp if speed_exp else 1.0
    g_lb = np.sqrt(lat * lat + brake * brake)
    base = r * g_lb**force_exp * vm
    drive = (
        acc**force_exp / r ** (force_exp - 1.0) * vm
        if geom.is_driven(corner)
        else np.zeros_like(base)
    )
    brake_power = brake * v / v_ref_ms if corner[0] == "f" else np.zeros_like(base)
    return np.asarray(base), np.asarray(drive), np.asarray(brake_power)


def corner_heat_rate(
    heat: HeatInput,
    lat_g: np.ndarray,
    long_g: np.ndarray,
    speed_ms: np.ndarray,
    car: str,
    corner: str,
    geom: CarGeometry | None,
    load_exp: float,
) -> np.ndarray:
    """Per-corner driving intensity ``q_i = (W_i/W_i,static)^p · q``.

    With ``p = 1`` the corner's heat scales with its instantaneous load
    (sliding power at equal slip across the axle); ``p = 2`` is the
    deflection/hysteresis scaling (energy per revolution ∝ W²/P). ``p = 0``
    or no geometry reproduces the corner-blind input.
    """
    q = heat.rate(lat_g, long_g, speed_ms, car)
    if geom is None or load_exp == 0.0:
        return q
    r = corner_load_ratio(lat_g, long_g, corner, geom)
    return np.asarray(q * r**load_exp)


# ------------------------------------------------------------ bounded force shares


@dataclass(frozen=True)
class ShareParams:
    """Bounded force-share model of the per-corner force vector.

    The car's force (``lat_g``, ``long_g``) is split over the corners by
    fractions in [0, 1] that saturate when a wheel is fully unloaded:

    - lateral: the front axle carries ``p_f`` of the lateral force; within an
      axle the loaded (outer) tyre's share is
      ``λ = ½(1 + tanh(|lat|/g_transfer))`` and the inner tyre's ``1 − λ`` —
      ½ at rest, saturating at 1 as the inner tyre unloads. ``g_transfer``
      is the lateral transfer scale (G): the lateral g at which the outer
      tyre carries ≈ 88 % of its axle's load. The rigid-body lift
      acceleration ``track / (2·CG height)`` is its geometric floor; a
      higher fitted value means tyre load sensitivity or the roll-stiffness
      split soften the transfer the heat sees;
    - braking: the front axle's share is the logistic ``β = σ(β0 + β1·|long|)``
      (``β0`` the static brake bias, ``β1`` the forward shift), each tyre of
      an axle half of it;
    - acceleration: the driven axle takes all of it, half per tyre.

    Two per-car heat efficiencies sit outside the shares (how much heat a
    unit of force makes relative to cornering force): ``eps_drive`` on the
    driven axle (wheelspin) and ``eps_brake`` on both axles (slip plus
    brake-disc heat into the hub; the front/rear split is the bias).
    """

    p_f: float
    g_transfer_front: float
    g_transfer_rear: float
    beta0: float
    beta1: float
    eps_drive: float
    eps_brake: float
    driven: str = "rear"
    g_x_transfer: float = float("inf")  # longitudinal transfer scale (G); inf = no transfer

    def front_lateral_share(self, long_signed: np.ndarray) -> np.ndarray:
        """Front axle's share of the lateral force as a bounded function of the
        longitudinal acceleration: ``½(1 + tanh(z0 − long/g_x))`` with
        ``z0 = atanh(2·p_f − 1)`` so it equals ``p_f`` at zero; braking
        (long < 0) loads the front axle, acceleration unloads it."""
        if not np.isfinite(self.g_x_transfer):
            return np.full_like(np.asarray(long_signed, dtype=float), self.p_f)
        z0 = np.arctanh(2.0 * np.clip(self.p_f, 1e-6, 1 - 1e-6) - 1.0)
        return np.asarray(0.5 * (1.0 + np.tanh(z0 - np.asarray(long_signed) / self.g_x_transfer)))

    def lateral_share(
        self, lat_abs: np.ndarray, corner: str, loaded: bool, long_signed: np.ndarray | None = None
    ) -> np.ndarray:
        g_c = self.g_transfer_front if corner[0] == "f" else self.g_transfer_rear
        lam = 0.5 * (1.0 + np.tanh(lat_abs / g_c))
        if long_signed is None:
            p: np.ndarray | float = self.p_f if corner[0] == "f" else 1.0 - self.p_f
        else:
            pf = self.front_lateral_share(long_signed)
            p = pf if corner[0] == "f" else 1.0 - pf
        return np.asarray(p * (lam if loaded else 1.0 - lam))

    def braking_share(self, brake_abs: np.ndarray, corner: str) -> np.ndarray:
        beta = 1.0 / (1.0 + np.exp(-(self.beta0 + self.beta1 * brake_abs)))
        share = beta if corner[0] == "f" else 1.0 - beta
        return np.asarray(0.5 * share * self.eps_brake)

    def drive_share(self, corner: str) -> float:
        driven = self.driven == "all" or corner[0] == self.driven[0]
        return 0.5 * self.eps_drive if driven else 0.0


def share_heat_rate(
    p: ShareParams,
    lat_pos: np.ndarray,
    lat_neg: np.ndarray,
    long_pos: np.ndarray,
    long_neg: np.ndarray,
    v: np.ndarray,
    corner: str,
) -> np.ndarray:
    """``q_i = V · |F_i| / (m·g)`` (G·m/s) from per-second non-negative parts of
    the accelerations (``lat_pos``: right turns, which load the LEFT tyres;
    ``long_neg``: braking; ``long_pos``: acceleration)."""
    left = corner[1] == "l"
    long_signed = long_pos - long_neg
    fy = lat_pos * p.lateral_share(lat_pos, corner, left, long_signed) + lat_neg * p.lateral_share(
        lat_neg, corner, not left, long_signed
    )
    fx = long_neg * p.braking_share(long_neg, corner) + long_pos * p.drive_share(corner)
    return np.asarray(v * np.sqrt(fy * fy + fx * fx))


SHARE_PARAM_NAMES = (
    "p_f",
    "g_transfer_front",
    "g_transfer_rear",
    "beta0",
    "beta1",
    "eps_drive",
    "eps_brake",
    "g_x_transfer",
)
SHARE_BOUNDS = {
    "p_f": (0.2, 0.8),
    "g_transfer_front": (0.5, 6.0),
    "g_transfer_rear": (0.5, 6.0),
    "beta0": (-2.0, 3.0),
    "beta1": (0.0, 4.0),
    "eps_drive": (0.0, 10.0),
    "eps_brake": (0.0, 10.0),
    "g_x_transfer": (0.5, 50.0),
}
SHARE_INIT = {
    "p_f": 0.5,
    "g_transfer_front": 2.0,
    "g_transfer_rear": 2.0,
    "beta0": 0.5,
    "beta1": 0.5,
    "eps_drive": 1.0,
    "eps_brake": 1.0,
    "g_x_transfer": 4.0,
}


# ------------------------------------------------------------ production car facts


@dataclass(frozen=True)
class CarFacts:
    """What the production share model takes from the car rather than fits:
    static front weight share, front brake bias, driven axle."""

    p_f: float
    brake_bias_front: float
    driven: str = "rear"

    @property
    def beta0(self) -> float:
        """Logit of the brake bias (the share model's ``beta0`` with ``beta1 = 0``)."""
        b = min(max(self.brake_bias_front, 1e-6), 1 - 1e-6)
        return float(np.log(b / (1.0 - b)))


# Spec-sheet facts per (fit-pooled) car. Toyota 86: 53:47, ≈ 65 % front brake
# bias; FJ-class single-seater: 42:58, ≈ 58 % front bias. Both rear-drive.
CAR_FACTS: dict[str, CarFacts] = {
    "Inferno 86": CarFacts(p_f=0.53, brake_bias_front=0.65),
    "FJ": CarFacts(p_f=0.42, brake_bias_front=0.58),
}
DEFAULT_CAR_FACTS = CarFacts(p_f=0.5, brake_bias_front=0.6)
