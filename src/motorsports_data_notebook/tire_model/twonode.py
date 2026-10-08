"""Two-node tire thermal model: tread surface + carcass (gas), fitted
jointly on the IR tread temperature and the pressure-implied gas temperature.

Model (per stint, per corner, 1 s steps on the rolling clock)::

    dT_s/dt = a_s · c_track · q(t) − b_sa · (T_s − T_eff) − b_sc · (T_s − T_c)
    dT_c/dt = b_cs · (T_s − T_c) − b_c · (T_c − T_eff)

``T_s`` is the tread surface (observed by the IR array, mean of its zones),
``T_c`` the carcass whose temperature the cavity gas reads (observed through
the TPMS pressure, see :mod:`.statespace`). All heat enters at the surface
(ChassisSim's node); the carcass is heated only by conduction from the tread.
``b_sc / b_cs = C_c / C_s`` is the carcass-to-tread heat-capacity ratio, a
physical sanity check on the fit. Each step is integrated exactly with the
2×2 matrix exponential, inputs held constant over the second.

IR channels are gated per (stint, corner) by :func:`ir_channel_ok` — unplugged
sensors read a −200 sentinel; a sensor that sees bodywork instead of the
tread reads near ambient with no zone-to-zone spread, little swing, and sits
*below* the gas temperature, which a running tread cannot.

Run the comparison against the single node on the same stints::

    uv run python -m motorsports_data_notebook.tire_model.twonode --n-folds 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from ..tire_etl.paths import default_dataset_root
from .energy_balance import P_ATM_BAR, T_ZERO_C_TO_K, speed_pressure_factor
from .heat_input import HeatInput
from .statespace import (
    CORNERS,
    V_REF_MS,
    StintSeries,
    build_stint_series,
    cooling_factor,
    fit_cells,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------ IR quality gate

IR_MIN_SECONDS = 300
IR_MIN_MEDIAN_C = 25.0
IR_MAX_C = 150.0
IR_MIN_SWING_C = 15.0  # p95 − p5 over the stint
IR_MIN_STEP_STD_C = 0.8  # std of the 1 s differences
IR_MIN_ZONE_RANGE_C = 5.0  # median max−min across the 8 zones (a tread has a profile)
IR_MIN_CORR_G2 = 0.4  # against the 10 s trailing mean of g²
IR_MIN_SURF_MINUS_GAS_C = -3.0  # median over seconds with both; tread ≥ gas while running
IR_SECOND_MIN_SURF_MINUS_GAS_C = -15.0  # per-second sanity mask

SIGMA_SURF_C = 4.0
SIGMA_GAS_C = 2.0


@dataclass
class IrQuality:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    diag: dict[str, float] = field(default_factory=dict)


def ir_channel_ok(s: StintSeries, j: int) -> IrQuality:
    """Gate one (stint, corner) IR channel on the binned series."""
    start = int(s.anchor_idx[j])
    x = s.surf[:, j].copy()
    if start >= 0:
        x[:start] = np.nan
    fin = np.isfinite(x)
    diag: dict[str, float] = {"n_s": float(fin.sum())}
    reasons: list[str] = []
    if fin.sum() < IR_MIN_SECONDS:
        return IrQuality(False, ["too_few_seconds"], diag)
    xs = x[fin]
    diag["median"] = float(np.median(xs))
    diag["max"] = float(np.max(xs))
    diag["swing"] = float(np.percentile(xs, 95) - np.percentile(xs, 5))
    diag["step_std"] = float(np.std(np.diff(xs)))
    zr = s.surf_zone_range[:, j][fin]
    diag["zone_range"] = float(np.nanmedian(zr)) if np.isfinite(zr).any() else 0.0
    g2s = pd.Series(s.g2).rolling(10, min_periods=3).mean().to_numpy()
    m = fin & np.isfinite(g2s)
    diag["corr_g2"] = (
        float(np.corrcoef(x[m], g2s[m])[0, 1]) if m.sum() > 60 and np.std(x[m]) > 0 else 0.0
    )
    both = fin & np.isfinite(s.obs[:, j])
    diag["surf_minus_gas"] = (
        float(np.median(x[both] - s.obs[both, j])) if both.sum() > 60 else float("nan")
    )
    if diag["median"] < IR_MIN_MEDIAN_C:
        reasons.append("cold_median")
    if diag["max"] > IR_MAX_C:
        reasons.append("implausible_max")
    if diag["swing"] < IR_MIN_SWING_C:
        reasons.append("no_swing")
    if diag["step_std"] < IR_MIN_STEP_STD_C:
        reasons.append("flat")
    if diag["zone_range"] < IR_MIN_ZONE_RANGE_C:
        reasons.append("no_zone_profile")
    if diag["corr_g2"] < IR_MIN_CORR_G2:
        reasons.append("not_tracking_g2")
    if np.isfinite(diag["surf_minus_gas"]) and diag["surf_minus_gas"] < IR_MIN_SURF_MINUS_GAS_C:
        reasons.append("below_gas")
    return IrQuality(not reasons, reasons, diag)


def gate_ir(stints: list[StintSeries], cars: tuple[str, ...] = ("Inferno 86",)) -> pd.DataFrame:
    """Apply :func:`ir_channel_ok` in place (failed channels become NaN) and
    return the per-(stint, corner) decisions. Cars outside ``cars`` are
    masked outright (the KK-F IR data is unusable)."""
    rows = []
    for s in stints:
        for j, c in enumerate(CORNERS):
            if s.car not in cars:
                q = IrQuality(False, ["car_excluded"], {})
            else:
                q = ir_channel_ok(s, j)
            if not q.ok:
                s.surf[:, j] = np.nan
            else:
                bad = (s.surf[:, j] - s.obs[:, j]) < IR_SECOND_MIN_SURF_MINUS_GAS_C
                s.surf[bad & np.isfinite(s.obs[:, j]), j] = np.nan
            rows.append(
                {
                    "session_id": s.session_id,
                    "stint_id": s.stint_id,
                    "car": s.car,
                    "track": s.track,
                    "condition": s.condition,
                    "corner": c,
                    "ok": q.ok,
                    "reasons": ",".join(q.reasons),
                    **q.diag,
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------ model


# ------------------------------------------------------------ model


@dataclass(frozen=True)
class TwoNodeParams:
    a_s: float  # K/s per G² into the surface node (at c_track = 1)
    b_sa: float  # 1/s surface → ambient (at V_ref when beta > 0)
    b_sc: float  # 1/s surface → carcass (per surface capacity)
    b_cs: float  # 1/s carcass ← surface (per carcass capacity)
    b_c: float  # 1/s carcass → ambient
    beta: float = 0.0  # forced-convection slope: b_sa(V) = b_sa·(1 + β·V/V_ref)/(1 + β)

    @property
    def capacity_ratio(self) -> float:
        """``C_c / C_s = b_sc / b_cs``."""
        return self.b_sc / self.b_cs

    @property
    def tau_surface_s(self) -> float:
        return 1.0 / (self.b_sa + self.b_sc)

    @property
    def tau_carcass_s(self) -> float:
        return 1.0 / (self.b_cs + self.b_c)

    def matrix(self) -> np.ndarray:
        return np.array(
            [[-(self.b_sa + self.b_sc), self.b_sc], [self.b_cs, -(self.b_cs + self.b_c)]]
        )

    def steady_state_gain(self) -> tuple[float, float]:
        """Steady-state (T_s − T_eff, T_c − T_eff) per unit ``c_track·q``:
        the two-node equivalent of K."""
        x = np.linalg.solve(self.matrix(), -np.array([self.a_s, 0.0]))
        return float(x[0]), float(x[1])


def _eig2(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real eigen-decomposition ``A = P diag(λ) P⁻¹`` of the 2×2 compartment
    matrix (``a12·a21 > 0`` guarantees real, distinct eigenvalues)."""
    a11, a12, a21, a22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    tr = a11 + a22
    det = a11 * a22 - a12 * a21
    r = np.sqrt(max(tr * tr / 4.0 - det, 1e-300))
    lam = np.array([tr / 2.0 + r, tr / 2.0 - r])
    P = np.array([[a12, a12], [lam[0] - a11, lam[1] - a11]])
    Pinv = np.linalg.inv(P)
    return lam, P, Pinv


def _expm2(A: np.ndarray) -> np.ndarray:
    """Matrix exponential of the 2×2 compartment matrix (reference helper)."""
    lam, P, Pinv = _eig2(A)
    return np.asarray(P @ np.diag(np.exp(lam)) @ Pinv)


@dataclass
class _Batch:
    """All (stint, corner) series of one corner as padded ``(m, n)`` arrays,
    from each series' anchor; ``valid`` marks the real samples."""

    Q: np.ndarray  # c_track · q
    V: np.ndarray  # speed m/s
    teff: np.ndarray  # (m,)
    x0: np.ndarray  # (m, 2) initial (T_s, T_c)
    obs_s: np.ndarray  # NaN where unobserved
    obs_c: np.ndarray
    valid: np.ndarray  # (m, n) bool
    lap_end: np.ndarray  # (m, n) bool
    length: np.ndarray  # (m,)
    meta: list[dict[str, Any]]

    @property
    def mask_s(self) -> np.ndarray:
        return np.isfinite(self.obs_s)

    @property
    def mask_c(self) -> np.ndarray:
        return np.isfinite(self.obs_c)


def build_batch(
    stints: list[StintSeries],
    j: int,
    c_track: dict[str, float],
    kappa_by_car: dict[str, float] | None = None,
) -> _Batch | None:
    """``kappa_by_car`` puts the gas observable on the gas-law scale
    (``energy_balance.speed_pressure_factor``) before fitting."""
    rows = [s for s in stints if int(s.anchor_idx[j]) >= 0]
    if not rows:
        return None
    starts = [int(s.anchor_idx[j]) for s in rows]
    L = np.array([len(s.g2) - st for s, st in zip(rows, starts)])
    m, n = len(rows), int(L.max())
    Q = np.zeros((m, n))
    V = np.full((m, n), V_REF_MS)
    obs_s = np.full((m, n), np.nan)
    obs_c = np.full((m, n), np.nan)
    valid = np.zeros((m, n), bool)
    lap_end = np.zeros((m, n), bool)
    x0 = np.zeros((m, 2))
    meta = []
    for i, (s, st) in enumerate(zip(rows, starts)):
        k = L[i]
        Q[i, :k] = (s.q4[st:, j] if s.q4 is not None else s.g2[st:]) * c_track.get(s.track, 1.0)
        V[i, :k] = s.v[st:]
        obs_s[i, :k] = s.surf[st:, j]
        kap = (kappa_by_car or {}).get(s.car, 0.0)
        f = speed_pressure_factor(s.v[st:], kap) / float(speed_pressure_factor(s.v[st], kap))
        obs_c[i, :k] = (s.obs[st:, j] + T_ZERO_C_TO_K) * f - T_ZERO_C_TO_K
        valid[i, :k] = True
        lap_end[i, :k] = s.lap_ends[st:]
        tc0 = float(s.t_start[j])
        ts0 = float(obs_s[i, 0]) if np.isfinite(obs_s[i, 0]) else tc0
        x0[i] = (ts0, tc0)
        meta.append(
            {
                "session_id": s.session_id,
                "stint_id": s.stint_id,
                "car": s.car,
                "track": s.track,
                "condition": s.condition,
                "corner": CORNERS[j],
                "p_start": float(s.p_start[j]),
                "t_start": tc0,
            }
        )
    return _Batch(
        Q, V, np.array([s.t_eff_c for s in rows]), x0, obs_s, obs_c, valid, lap_end, L, meta
    )


def _modal_response(lam: float, z0: np.ndarray, zeq: np.ndarray) -> np.ndarray:
    """Scalar recurrence ``z[k+1] = e^λ z[k] + (1 − e^λ) zeq[k]`` from
    ``z[0] = z0`` in closed form (cumulative sum), vectorised over rows."""
    n = zeq.shape[1]
    k = np.arange(n, dtype=float)
    d = np.exp(lam)
    # z[k] = e^{λk} (z0 + Σ_{j<k} (1 − d) zeq_j e^{−λ(j+1)})
    w = (1.0 - d) * zeq * np.exp(np.minimum(-lam * (k + 1.0), 600.0))[None, :]
    acc = np.concatenate([np.zeros((zeq.shape[0], 1)), np.cumsum(w, axis=1)[:, :-1]], axis=1)
    return np.asarray(np.exp(lam * k)[None, :] * (z0[:, None] + acc))


def _solve_constant(
    p: TwoNodeParams, b: _Batch, u1: np.ndarray, u2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-second solution of ``x' = A x + u(t)`` with constant ``A``
    and ``u`` held constant over each second: diagonalise and solve the two
    decoupled modes in closed form (no loop over time)."""
    A = p.matrix()
    lam, P, Pinv = _eig2(A)
    Ainv = np.linalg.inv(A)
    xeq1 = -(Ainv[0, 0] * u1 + Ainv[0, 1] * u2)
    xeq2 = -(Ainv[1, 0] * u1 + Ainv[1, 1] * u2)
    z0 = b.x0 @ Pinv.T
    zeq1 = Pinv[0, 0] * xeq1 + Pinv[0, 1] * xeq2
    zeq2 = Pinv[1, 0] * xeq1 + Pinv[1, 1] * xeq2
    Z1 = _modal_response(float(lam[0]), z0[:, 0], zeq1)
    Z2 = _modal_response(float(lam[1]), z0[:, 1], zeq2)
    return P[0, 0] * Z1 + P[0, 1] * Z2, P[1, 0] * Z1 + P[1, 1] * Z2


PICARD_ITERATIONS = 4


def simulate_batch(p: TwoNodeParams, b: _Batch) -> tuple[np.ndarray, np.ndarray]:
    """``(T_s, T_c)`` as ``(m, n)`` arrays. With ``beta > 0`` the surface
    cooling coefficient varies with speed; the varying part is moved to the
    input side, ``−δ(t)·(T_s − T_eff)`` with ``δ = b_sa·(f(V) − 1)``, and
    the constant-coefficient solver is iterated (Picard) on the midpoint
    surface temperature of the previous pass. Loop-free throughout."""
    m, n = b.Q.shape
    u1 = p.a_s * b.Q + (p.b_sa * b.teff)[:, None]
    u2 = np.broadcast_to((p.b_c * b.teff)[:, None], (m, n))
    if not p.beta:
        return _solve_constant(p, b, u1, u2)
    delta = p.b_sa * (cooling_factor(b.V, p.beta) - 1.0)
    Ts, Tc = _solve_constant(p, b, u1, u2)
    for _ in range(PICARD_ITERATIONS):
        ts_mid = 0.5 * (Ts + np.concatenate([Ts[:, 1:], Ts[:, -1:]], axis=1))
        u1_eff = u1 - delta * (ts_mid - b.teff[:, None])
        Ts, Tc = _solve_constant(p, b, u1_eff, u2)
    return Ts, Tc


def residuals(p: TwoNodeParams, b: _Batch) -> np.ndarray:
    Ts, Tc = simulate_batch(p, b)
    ms, mc = b.mask_s, b.mask_c
    return np.concatenate(
        [(Ts[ms] - b.obs_s[ms]) / SIGMA_SURF_C, (Tc[mc] - b.obs_c[mc]) / SIGMA_GAS_C]
    )


def _simulate_reference_loop(
    p: TwoNodeParams, q: np.ndarray, v: np.ndarray, t_eff: float, ts0: float, tc0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Step-by-step reference (tests only): exact 2×2 exponential per second
    with the speed-dependent coefficient rebuilt each step."""
    n = len(q)
    Ts, Tc = np.empty(n), np.empty(n)
    x = np.array([ts0, tc0])
    for k in range(n):
        Ts[k], Tc[k] = x
        bsa = p.b_sa * float(cooling_factor(np.array([v[k]]), p.beta)[0]) if p.beta else p.b_sa
        A = np.array([[-(bsa + p.b_sc), p.b_sc], [p.b_cs, -(p.b_cs + p.b_c)]])
        u = np.array([p.a_s * q[k] + bsa * t_eff, p.b_c * t_eff])
        xeq = -np.linalg.solve(A, u)
        x = xeq + _expm2(A) @ (x - xeq)
    return Ts, Tc


# ------------------------------------------------------------ fit

BETA_BOUNDS = (0.0, 20.0)


def _bounds(q_typ: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Surface τ ~ 10–60 s, carcass τ ~ minutes; a_s sized for a ~40 K
    # steady surface rise at typical q.
    b_sa0, b_sc0 = 1 / 40.0, 1 / 40.0
    a0 = 40.0 * (b_sa0 + b_sc0) / max(q_typ, 1e-6)
    x0 = [a0, b_sa0, b_sc0, 1 / 300.0, 1 / 600.0]
    lo = [a0 * 1e-3, 1 / 600.0, 1 / 600.0, 1 / 3000.0, 1 / 3000.0]
    hi = [a0 * 1e3, 1.0, 1.0, 1 / 5.0, 1 / 60.0]
    return np.log(x0), np.log(lo), np.log(hi)


@dataclass
class TwoNodeFit:
    params: dict[str, TwoNodeParams]  # corner -> params
    c_track: dict[str, float]
    cost: dict[str, float]
    n_surf: dict[str, int]
    n_gas: dict[str, int]
    mode: str = "per_corner"
    kappa_by_car: dict[str, float] = field(default_factory=dict)


def _batches(
    stints: list[StintSeries],
    c_track: dict[str, float],
    kappa_by_car: dict[str, float] | None = None,
) -> dict[str, _Batch]:
    out: dict[str, _Batch] = {}
    for j, corner in enumerate(CORNERS):
        b = build_batch(stints, j, c_track, kappa_by_car)
        if b is not None:
            out[corner] = b
    return out


def _counts(batches: dict[str, _Batch]) -> tuple[dict[str, int], dict[str, int]]:
    return (
        {c: int(b.mask_s.sum()) for c, b in batches.items()},
        {c: int(b.mask_c.sum()) for c, b in batches.items()},
    )


def fit_two_node(
    stints: list[StintSeries],
    c_track: dict[str, float],
    kappa_by_car: dict[str, float] | None = None,
) -> TwoNodeFit:
    """Fit the five constants independently per corner (under-identified:
    kept as the unconstrained reference)."""
    batches = _batches(stints, c_track, kappa_by_car)
    params: dict[str, TwoNodeParams] = {}
    cost: dict[str, float] = {}
    for corner, b in batches.items():
        q_typ = float(b.Q[b.valid].mean())
        x0, lo, hi = _bounds(q_typ)
        res = least_squares(
            lambda x, bb=b: residuals(TwoNodeParams(*np.exp(x)), bb),
            x0,
            bounds=(lo, hi),
            method="trf",
            x_scale="jac",
            max_nfev=200,
        )
        params[corner] = TwoNodeParams(*np.exp(res.x))
        cost[corner] = float(res.cost)
        logger.info("two-node %s: %s cost %.0f", corner, params[corner], res.cost)
    n_surf, n_gas = _counts(batches)
    return TwoNodeFit(
        params, dict(c_track), cost, n_surf, n_gas, kappa_by_car=dict(kappa_by_car or {})
    )


def _joint_unpack(x: np.ndarray, corners: list[str], speed: bool) -> dict[str, TwoNodeParams]:
    n = len(corners)
    a = np.exp(x[:n])
    b_sa, b_sc, b_cs, b_c = np.exp(x[n : n + 4])
    beta = float(x[n + 4]) if speed else 0.0
    return {
        c: TwoNodeParams(float(a[i]), b_sa, b_sc, b_cs, b_c, beta) for i, c in enumerate(corners)
    }


def fit_two_node_shared(
    stints: list[StintSeries],
    c_track: dict[str, float],
    *,
    speed: bool = False,
    kappa_by_car: dict[str, float] | None = None,
) -> TwoNodeFit:
    """One tyre, four corners: the thermal constants ``b_sa, b_sc, b_cs,
    b_c`` (and ``β``) are shared across corners, only the heat input
    ``a_s`` is per corner (load and duty differ by corner)."""
    batches = _batches(stints, c_track, kappa_by_car)
    corners = list(batches)
    q_typ = float(np.mean([b.Q[b.valid].mean() for b in batches.values()]))
    x0, lo, hi = _bounds(q_typ)
    x0l = [x0[0]] * len(corners) + list(x0[1:])
    lol = [lo[0]] * len(corners) + list(lo[1:])
    hil = [hi[0]] * len(corners) + list(hi[1:])
    if speed:
        x0l.append(1.0)
        lol.append(BETA_BOUNDS[0])
        hil.append(BETA_BOUNDS[1])

    def joint(x: np.ndarray) -> np.ndarray:
        params = _joint_unpack(x, corners, speed)
        return np.concatenate([residuals(params[c], batches[c]) for c in corners])

    res = least_squares(
        joint,
        np.array(x0l),
        bounds=(np.array(lol), np.array(hil)),
        method="trf",
        x_scale="jac",
        max_nfev=200,
    )
    params = _joint_unpack(res.x, corners, speed)
    n_surf, n_gas = _counts(batches)
    logger.info("two-node shared%s: %s cost %.0f", " +speed" if speed else "", params, res.cost)
    return TwoNodeFit(
        params,
        dict(c_track),
        {c: float(res.cost) for c in corners},
        n_surf,
        n_gas,
        mode="shared_speed" if speed else "shared",
        kappa_by_car=dict(kappa_by_car or {}),
    )


# ------------------------------------------------------------ scoring


def score_two_node(stints: list[StintSeries], fit: TwoNodeFit) -> pd.DataFrame:
    """Lap-end gas/pressure residuals (anchor lap skipped) and per-stint
    surface MAE for held-out stints."""
    frames = []
    for corner, b in _batches(stints, fit.c_track, fit.kappa_by_car).items():
        p = fit.params.get(corner)
        if p is None:
            continue
        Ts, Tc = simulate_batch(p, b)
        ms = b.mask_s
        err_s = np.where(ms, np.abs(Ts - b.obs_s), 0.0)
        n_s = ms.sum(axis=1)
        surf_mae = np.where(n_s > 0, err_s.sum(axis=1) / np.maximum(n_s, 1), np.nan)
        ratio = np.array(
            [(d["p_start"] + P_ATM_BAR) / (d["t_start"] + T_ZERO_C_TO_K) for d in b.meta]
        )
        # lap ends after the anchor lap, with a finite gas observation
        first_end = np.argmax(b.lap_end, axis=1)
        is_first = np.zeros_like(b.lap_end)
        is_first[np.arange(len(first_end)), first_end] = b.lap_end[
            np.arange(len(first_end)), first_end
        ]
        sel = b.lap_end & ~is_first & b.mask_c
        rows_i, cols = np.nonzero(sel)
        lap_n = np.cumsum(b.lap_end, axis=1)[rows_i, cols] - 1
        frames.append(
            pd.DataFrame(
                {
                    **{k: [b.meta[i][k] for i in rows_i] for k in b.meta[0]},
                    "lap_within_stint": lap_n,
                    "t_s": cols.astype(float),
                    "T_pred_c": Tc[rows_i, cols],
                    "T_obs_c": b.obs_c[rows_i, cols],
                    "resid_c": Tc[rows_i, cols] - b.obs_c[rows_i, cols],
                    "resid_bar": (Tc[rows_i, cols] - b.obs_c[rows_i, cols]) * ratio[rows_i],
                    "Ts_pred_c": Ts[rows_i, cols],
                    "Ts_obs_c": b.obs_s[rows_i, cols],
                    "surf_mae_stint_c": surf_mae[rows_i],
                    "has_ir": n_s[rows_i] > 0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _summ(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return f"{label}: no rows"
    ir = df[df["has_ir"]]
    parts = [
        f"{label}: n={len(df)} P MAE {df['resid_bar'].abs().mean():.4f} bar bias "
        f"{df['resid_bar'].mean():+.4f}  T_gas MAE {df['resid_c'].abs().mean():.2f} C"
    ]
    if not ir.empty:
        parts.append(
            f"  IR-valid subset n={len(ir)}: P MAE {ir['resid_bar'].abs().mean():.4f}"
            + (
                f"  surface MAE {ir.drop_duplicates(['session_id','stint_id','corner'])['surf_mae_stint_c'].mean():.2f} C"
                if "surf_mae_stint_c" in ir
                else ""
            )
        )
    return "\n".join(parts)


# ------------------------------------------------------------ driver


def _fit(
    stints: list[StintSeries],
    c_track: dict[str, float],
    mode: str,
    kappa_by_car: dict[str, float] | None = None,
) -> TwoNodeFit:
    if mode == "per_corner":
        return fit_two_node(stints, c_track, kappa_by_car)
    if mode == "shared":
        return fit_two_node_shared(stints, c_track, speed=False, kappa_by_car=kappa_by_car)
    if mode == "shared_speed":
        return fit_two_node_shared(stints, c_track, speed=True, kappa_by_car=kappa_by_car)
    raise ValueError(mode)


def run(
    root: Path, *, n_folds: int, car: str, out_dir: Path | None, mode: str = "per_corner"
) -> None:
    from .heat_experiment import _folds, prepare_laps_for_fit, score_stints

    t0 = time.time()
    laps = prepare_laps_for_fit(root)
    laps = laps[(laps["car"] == car) & (laps["condition"] == "dry")].reset_index(drop=True)
    stints = build_stint_series(root, laps, HeatInput("g2"))
    gate = gate_ir(stints, cars=(car,))
    ok = gate[gate["ok"]]
    print(
        f"{len(stints)} dry {car} stints; IR channels passing the gate: {len(ok)}/{len(gate)} "
        f"({ok.drop_duplicates('session_id').shape[0]} sessions); rejections: "
        f"{gate[~gate['ok']]['reasons'].value_counts().to_dict()}"
    )
    # Track factors held at the production values (fitted on the whole fleet).
    import json

    model = json.load(open(root / "tire_model.json"))
    c_track = {d["track_canonical"]: float(d["value"]) for d in model["c_track_by_track"]}
    kappa_by_car = {
        k: float(v)
        for k, v in model["energy_balance"]
        .get("speed_pressure", {})
        .get("kappa_by_car", {})
        .items()
    }
    print(f"c_track {c_track}; kappa {kappa_by_car}")
    folds = [f for f in _folds(root, n_folds, 2, 10)]
    sids = {s.session_id for s in stints}
    folds = [f & sids for f in folds if f & sids]
    print(f"{len(folds)} folds; prep {time.time() - t0:.0f}s")
    two_frames, one_frames = [], []
    fold_fits: list[TwoNodeFit] = []
    for k, held in enumerate(folds):
        train = [s for s in stints if s.session_id not in held]
        test = [s for s in stints if s.session_id in held]
        t1 = time.time()
        f2 = _fit(train, c_track, mode, kappa_by_car)
        d2 = score_two_node(test, f2)
        d2["fold"] = k
        two_frames.append(d2)
        fold_fits.append(f2)
        f1 = fit_cells(
            train,
            anchor_track="tsukuba_2000",
            fixed_c_track=c_track,
            v_corr=False,
            kappa=kappa_by_car,
        )
        d1 = score_stints(test, f1)
        d1["fold"] = k
        # mark the IR-valid subset in the single-node rows too
        ir_keys = {(r.session_id, r.stint_id, r.corner) for r in ok.itertuples()}
        d1["has_ir"] = [
            (a, b, c) in ir_keys for a, b, c in zip(d1["session_id"], d1["stint_id"], d1["corner"])
        ]
        one_frames.append(d1)
        print(f"fold {k}: {len(held)} sessions, {time.time() - t1:.0f}s")
    two = pd.concat(two_frames, ignore_index=True)
    one = pd.concat(one_frames, ignore_index=True)
    print("\n" + _summ(one, "single node (gas only)"))
    print(_summ(two, "two node (surface + gas)"))
    full = _fit(stints, c_track, mode, kappa_by_car)
    print(f"\nfull-data two-node constants per corner (mode {mode}):")
    print(
        f"{'corner':6s} {'a_s':>8s} {'tau_s(s)':>9s} {'tau_c(s)':>9s} {'C_c/C_s':>8s} {'K_s':>7s} {'K_c':>7s} {'b_sa':>8s} {'b_sc':>8s} {'b_cs':>8s} {'b_c':>8s} {'beta':>6s}"
    )
    for c, p in full.params.items():
        ks, kc = p.steady_state_gain()
        print(
            f"{c:6s} {p.a_s:8.4f} {p.tau_surface_s:9.1f} {p.tau_carcass_s:9.1f} {p.capacity_ratio:8.2f} "
            f"{ks:7.1f} {kc:7.1f} {p.b_sa:8.5f} {p.b_sc:8.5f} {p.b_cs:8.5f} {p.b_c:8.5f} {p.beta:6.2f}"
        )
    for k, f in enumerate(fold_fits):
        p = next(iter(f.params.values()))
        print(
            f"fold {k}: tau_s {p.tau_surface_s:.1f} tau_c {p.tau_carcass_s:.1f} "
            f"C_c/C_s {p.capacity_ratio:.2f} beta {p.beta:.2f}"
        )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        two.to_parquet(out_dir / "holdout_two_node.parquet")
        one.to_parquet(out_dir / "holdout_single_node.parquet")
        gate.to_parquet(out_dir / "ir_gate.parquet")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--car", default="Inferno 86")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--mode", choices=["per_corner", "shared", "shared_speed"], default="per_corner"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    root = Path(args.dataset_root) if args.dataset_root else default_dataset_root()
    run(root, n_folds=args.n_folds, car=args.car, out_dir=args.out_dir, mode=args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
