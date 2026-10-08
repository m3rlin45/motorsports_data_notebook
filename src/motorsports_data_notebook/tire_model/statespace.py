"""Per-second (1 Hz) fit of the tire energy balance on the pressure-implied
cavity-gas temperature.

Model (per stint, per corner), integrated exactly with the inputs held
constant over each 1 s step of the stint's rolling clock (standstill
excluded; a lap the usability filters dropped mid-stint still heats the
tire and advances the clock, it is only not scored)::

    dT_i/dt = a · q_i(t) − b_axle(i) · (T_i − T_eff),   T(t_anchor) = T_start

``q_i`` is the per-corner driving intensity of :mod:`.heat_input` (schema
v5: |g|·V/V_ref with the weight-transfer / drive / brake split). The
production fit (:func:`fit_physical`) shares the gain ``a`` across the
corners of a (car, condition), fits the cooling ``b`` per (car, axle,
condition), the drive and brake coefficients and the speed-pressure
constant κ per car, and carries **no track constants**.

Observation: the pressure-implied cavity-gas temperature
``T_gas_K = T_start_K · P(t)_abs / P_start_abs · (1 + κV²)/(1 + κV_anchor²)``
from the stint's pit-exit (T, P) anchor, for every 1 s bin with a finite
pressure at or after the anchor (a (stint, corner) needs ≥ 60 scored
seconds). Rain conditions are fitted after dry with ``b_rain ≥ b_dry``
(τ_rain ≤ τ_dry) as a bound. Parameter recovery on synthetic stints is
unit-tested.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from ..tire_etl.paths import timeseries_dir
from .energy_balance import P_ATM_BAR, T_ZERO_C_TO_K, speed_pressure_factor
from .heat_input import (
    BUILTIN_GEOMETRY,
    DEFAULT_HEAT_INPUT,
    PHYSICAL_HEAT_INPUT,
    CarGeometry,
    HeatInput,
    corner_heat_parts,
    corner_heat_rate,
)

logger = logging.getLogger(__name__)

CORNERS = ("fl", "fr", "rl", "rr")
MOVING_SPEED_MS = 5.0 / 3.6
MIN_SCORED_SECONDS = 60  # a (stint, corner) needs this much observed data to count
MIN_RAIN_SESSIONS = 3  # per (car, condition) for its own tau/K (else the condition chain)
TAU_BOUNDS_S = (30.0, 3000.0)
A_BOUNDS = (1e-3, 1e3)  # relative to the initial guess (see _Design.bounds)
P_REF_ABS_BAR = 2.5  # the heat-input gain ``a`` is quoted at this absolute pressure
P_EXP_BOUNDS = (0.0, 4.0)
P_MODES = ("anchor", "instant")
V_REF_MS = 30.0  # the cooling coefficient ``b`` is quoted at this speed
IR_VALID_C = (-40.0, 200.0)  # IR samples outside this range are sentinels (−200 = unplugged)
COOL_MODES = ("speed",)
KAPPA_BOUNDS = (0.0, 5e-5)  # per (m/s)²: pressure observable P_obs = P_gas / (1 + κ·V²)
COOL_BETA_BOUNDS = (0.0, 20.0)


@dataclass
class StintSeries:
    session_id: str
    stint_id: int
    car: str
    track: str
    condition: str
    t_eff_c: float
    g2: np.ndarray  # (n,) mean heat input q per rolling second (G² for the g2 form)
    v: np.ndarray  # (n,) mean speed per rolling second (m/s)
    surf: np.ndarray  # (n, 4) IR tread temperature (mean of the zones), NaN when absent/invalid
    surf_zone_range: np.ndarray  # (n, 4) max − min across the IR zones
    obs: np.ndarray  # (n, 4) pressure-implied gas temperature, NaN where missing
    anchor_idx: np.ndarray  # (4,) first scored bin per corner, -1 when none
    t_start: np.ndarray  # (4,) anchor temperature per corner
    p_start: np.ndarray  # (4,) anchor gauge pressure per corner (bar)
    n_laps: int
    lap_ends: np.ndarray  # (n,) bool: last bin of a lap
    q4: np.ndarray | None = None  # (n, 4) per-corner heat input with load transfer, else None
    q4_drive: np.ndarray | None = None  # (n, 4) drive-slip part (driven corners), see heat_input
    q4_brake: np.ndarray | None = None  # (n, 4) brake-power part (front corners)
    lap_lo: int = 0  # lap_num of the stint's first bin (lap index k ↔ lap_num lap_lo + k)


# ---------------------------------------------------------------- data prep


@lru_cache(maxsize=4)
def _session_timeseries(root_str: str, session_id: str) -> pd.DataFrame | None:
    files = list(timeseries_dir(Path(root_str)).glob(f"*/{session_id}.parquet"))
    if not files:
        return None
    cols = ["lap_num", "sample_idx", "t_session_s", "speed_ms", "lat_g", "long_g"]
    cols += [f"tpms_press_{c}_bar" for c in CORNERS]
    cols += [f"surf_temp_{c}_ch{i}_c" for c in CORNERS for i in range(1, 9)]
    schema = pq.read_schema(files[0])
    cols = [c for c in cols if c in schema.names]
    df: pd.DataFrame = pq.read_table(files[0], columns=cols).to_pandas()
    return df


def _bin_stint(
    ts: pd.DataFrame,
    lap_nums: list[int],
    heat: HeatInput = DEFAULT_HEAT_INPUT,
    car: str = "",
    geometry: CarGeometry | None = None,
    load_exp: float = 0.0,
    long_split: bool = False,
) -> dict[str, Any] | None:
    """1 Hz bins on the rolling clock for the given laps of one stint.

    ``heat`` picks the heat-input form; ``q`` is evaluated per sample before
    binning so a nonlinear slip activation sees the native-rate g. With a
    ``geometry`` and ``load_exp > 0`` the per-corner input
    ``q_i = (W_i/W_i,static)^p · q`` is binned too (``"q4"``)."""
    # The clock and g² span every lap from the first to the last fitted one (a lap
    # the usability filters dropped mid-stint still heats the tire and advances the
    # clock); observations are scored only in the fitted laps.
    lo, hi = min(lap_nums), max(lap_nums)
    sub = ts[(ts["lap_num"] >= lo) & (ts["lap_num"] <= hi)].sort_values(["lap_num", "sample_idx"])
    if len(sub) < 10:
        return None
    t = sub["t_session_s"].to_numpy(dtype=float)
    lap = sub["lap_num"].to_numpy()
    dt = np.diff(t, prepend=t[0])
    dt = np.where((dt < 0) | (dt > 2.0), 0.0, dt)  # a recording gap, not a lap boundary
    v = (
        sub["speed_ms"].to_numpy(dtype=float)
        if "speed_ms" in sub.columns
        else np.full(len(sub), 10.0)
    )
    moving = np.nan_to_num(v, nan=0.0) > MOVING_SPEED_MS
    in_fit = np.isin(lap, lap_nums)
    roll = np.cumsum(np.where(moving, dt, 0.0))
    b = np.floor(roll).astype(int)
    n = int(b.max()) + 1
    if n < 2:
        return None
    cnt = np.bincount(b[moving], minlength=n).astype(float)
    lat_s = sub["lat_g"].to_numpy(dtype=float)
    lng_s = sub["long_g"].to_numpy(dtype=float)
    g2 = heat.rate(lat_s, lng_s, v, car)
    q4 = None
    q4_drive = None
    q4_brake = None
    if geometry is not None and long_split:
        parts = [
            corner_heat_parts(
                lat_s,
                lng_s,
                v,
                c,
                geometry,
                heat.g_clip,
                speed_exp=heat.speed_exp,
                force_exp=heat.force_exp,
            )
            for c in CORNERS
        ]
        q4 = np.column_stack([p[0] for p in parts])
        q4_drive = np.column_stack([p[1] for p in parts])
        q4_brake = np.column_stack([p[2] for p in parts])
    elif geometry is not None and load_exp:
        cols = []
        for c in CORNERS:
            qc = corner_heat_rate(heat, lat_s, lng_s, v, car, c, geometry, load_exp)
            cols.append(qc)
        q4 = np.column_stack(cols)
    g2_bin = np.bincount(b[moving], weights=g2[moving], minlength=n) / np.maximum(cnt, 1)
    g2_bin = np.where(cnt > 0, g2_bin, 0.0)
    v_bin = np.bincount(b[moving], weights=np.nan_to_num(v[moving]), minlength=n) / np.maximum(
        cnt, 1
    )
    v_bin = np.where(cnt > 0, v_bin, 0.0)

    def _bin4(x: np.ndarray | None) -> np.ndarray | None:
        if x is None:
            return None
        out = np.zeros((n, 4))
        for j in range(4):
            out[:, j] = np.bincount(b[moving], weights=x[moving, j], minlength=n) / np.maximum(
                cnt, 1
            )
        return np.where(cnt[:, None] > 0, out, 0.0)

    q4_bin = _bin4(q4)
    q4_drive_bin = _bin4(q4_drive)
    q4_brake_bin = _bin4(q4_brake)
    press = np.full((n, 4), np.nan)
    for j, c in enumerate(CORNERS):
        col = f"tpms_press_{c}_bar"
        if col not in sub.columns:
            continue
        p = sub[col].to_numpy(dtype=float)
        ok = np.isfinite(p) & moving & in_fit
        s = np.bincount(b[ok], weights=p[ok], minlength=n)
        k = np.bincount(b[ok], minlength=n)
        press[:, j] = np.where(k > 0, s / np.maximum(k, 1), np.nan)
    lap_end = np.zeros(n, dtype=bool)
    for ln in np.unique(lap):
        m = lap == ln
        lap_end[int(b[m].max())] = True
    # IR tread temperature: mean and spread across the zones, sentinel-masked.
    surf = np.full((n, 4), np.nan)
    zr = np.full((n, 4), np.nan)
    for j, c in enumerate(CORNERS):
        zcols = [f"surf_temp_{c}_ch{i}_c" for i in range(1, 9) if f"surf_temp_{c}_ch{i}_c" in sub]
        if not zcols:
            continue
        z = sub[zcols].to_numpy(dtype=float)
        z = np.where((z > IR_VALID_C[0]) & (z < IR_VALID_C[1]), z, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
            zmean = np.nanmean(z, axis=1)
            zrange = np.nanmax(z, axis=1) - np.nanmin(z, axis=1)
        ok = np.isfinite(zmean) & moving
        k = np.bincount(b[ok], minlength=n)
        surf[:, j] = np.where(
            k > 0, np.bincount(b[ok], weights=zmean[ok], minlength=n) / np.maximum(k, 1), np.nan
        )
        zr[:, j] = np.where(
            k > 0, np.bincount(b[ok], weights=zrange[ok], minlength=n) / np.maximum(k, 1), np.nan
        )
    return {
        "g2": g2_bin,
        "v": v_bin,
        "press": press,
        "lap_end": lap_end,
        "surf": surf,
        "surf_zone_range": zr,
        "q4": q4_bin,
        "q4_drive": q4_drive_bin,
        "q4_brake": q4_brake_bin,
    }


def build_stint_series(
    root: Path,
    laps_for_fit: pd.DataFrame,
    heat: HeatInput = DEFAULT_HEAT_INPUT,
    geometry: dict[str, CarGeometry] | None = None,
    load_exp: float = 0.0,
    long_split: bool = False,
) -> list[StintSeries]:
    """One :class:`StintSeries` per (session, stint) present in ``laps_for_fit``
    (which carries the anchors, T_eff, condition, car and track). With a
    ``geometry`` map and ``load_exp`` the per-corner load-transfer input is
    attached as ``q4``."""
    out: list[StintSeries] = []
    need = ["t_eff_c", "condition", "car", "track_canonical"] + [
        f"{p}_{c}" for c in CORNERS for p in ("t_anchor", "t_start", "p_start")
    ]
    for col in need:
        if col not in laps_for_fit.columns:
            raise ValueError(f"laps_for_fit lacks {col!r}; run the warmup_table prep first")
    for key, grp in laps_for_fit.groupby(["session_id", "stint_id"], sort=False):
        sid, stint = str(key[0]), int(grp["stint_id"].iloc[0])  # type: ignore[index]
        ts = _session_timeseries(str(root), str(sid))
        if ts is None:
            continue
        first = grp.iloc[0]
        if pd.isna(first["t_eff_c"]) or first["condition"] == "unknown":
            continue
        binned = _bin_stint(
            ts,
            [int(x) for x in grp["lap_num"]],
            heat,
            str(first["car"]),
            (geometry or {}).get(str(first["car"])),
            load_exp,
            long_split,
        )
        if binned is None:
            continue
        n = len(binned["g2"])
        obs = np.full((n, 4), np.nan)
        anchor_idx = np.full(4, -1, dtype=int)
        t_start = np.full(4, np.nan)
        p_start = np.full(4, np.nan)
        roll_t = np.arange(n, dtype=float)
        for j, c in enumerate(CORNERS):
            t0, p0, ta = first[f"t_start_{c}"], first[f"p_start_{c}"], first[f"t_anchor_{c}"]
            if pd.isna(t0) or pd.isna(p0) or pd.isna(ta) or float(p0) <= 0.3:
                continue
            p = binned["press"][:, j]
            tg = (float(t0) + T_ZERO_C_TO_K) * (p + P_ATM_BAR) / (
                float(p0) + P_ATM_BAR
            ) - T_ZERO_C_TO_K
            scored = np.isfinite(p) & (roll_t >= float(ta))
            if scored.sum() < MIN_SCORED_SECONDS:
                continue
            obs[:, j] = np.where(scored, tg, np.nan)
            anchor_idx[j] = int(np.argmax(scored))
            t_start[j] = float(t0)
            p_start[j] = float(p0)
        if (anchor_idx < 0).all():
            continue
        out.append(
            StintSeries(
                session_id=str(sid),
                stint_id=int(stint),
                car=str(first["car"]),
                track=str(first["track_canonical"]),
                condition=str(first["condition"]),
                t_eff_c=float(first["t_eff_c"]),
                g2=binned["g2"],
                v=binned["v"],
                surf=binned["surf"],
                surf_zone_range=binned["surf_zone_range"],
                q4=binned["q4"],
                q4_drive=binned["q4_drive"],
                q4_brake=binned["q4_brake"],
                lap_lo=int(min(int(x) for x in grp["lap_num"])),
                obs=obs,
                anchor_idx=anchor_idx,
                t_start=t_start,
                p_start=p_start,
                n_laps=int(len(grp)),
                lap_ends=binned["lap_end"],
            )
        )
    return out


def stint_log_pressure_ratio(s: StintSeries, mode: str) -> np.ndarray:
    """``log(P_abs(t) / P_REF_ABS_BAR)`` per (second, corner) for the
    pressure dependence of the heat input.

    ``"anchor"``: the stint's pit-exit absolute pressure, constant.
    ``"instant"``: the measured pressure at each second (from the observed
    gas temperature and the anchor), forward-filled over gaps.
    """
    n = len(s.g2)
    out = np.zeros((n, 4))
    for j in range(4):
        p0 = float(s.p_start[j])
        if not np.isfinite(p0):
            continue
        p_abs = np.full(n, p0 + P_ATM_BAR)
        if mode == "instant":
            t0 = float(s.t_start[j])
            pk = (p0 + P_ATM_BAR) * (s.obs[:, j] + T_ZERO_C_TO_K) / (t0 + T_ZERO_C_TO_K)
            fin = np.isfinite(pk)
            if fin.any():
                idx = np.where(fin, np.arange(n), -1)
                idx = np.maximum.accumulate(idx)
                p_abs = np.where(idx >= 0, pk[np.maximum(idx, 0)], p0 + P_ATM_BAR)
        out[:, j] = np.log(np.maximum(p_abs, 0.5) / P_REF_ABS_BAR)
    return out


def cooling_factor(v: np.ndarray, beta: float) -> np.ndarray:
    """``h(V) / h(V_ref)`` for forced convection ``h = h_0 + h_1·V``:
    ``(1 + β·V/V_ref) / (1 + β)``; ``β = 0`` is speed-independent cooling."""
    return np.asarray((1.0 + beta * np.asarray(v, dtype=float) / V_REF_MS) / (1.0 + beta))


# ---------------------------------------------------------------- design + fit


class _Design:
    """Flattened arrays for a set of stints and the (car, condition) cells
    being fitted; other cells' parameters are held fixed."""

    def __init__(
        self,
        stints: list[StintSeries],
        cells: list[tuple[str, str]],
        fixed: dict[tuple[str, str, str], tuple[float, float]],
        p_mode: str | None = None,
        fixed_p_exp: float | None = None,
        cool_mode: str | None = None,
        fixed_cool_beta: float | None = None,
        v_corr: bool = False,
        fixed_kappa: dict[str, float] | None = None,
        share_a: bool = False,
        share_b: bool | str = False,
        drive_term: bool = False,
        brake_term: bool = False,
        fixed_drive: dict[str, float] | None = None,
        fixed_brake: dict[str, float] | None = None,
    ):
        self.cells = cells
        self.fixed = fixed
        self.p_mode = p_mode
        self.fixed_p_exp = fixed_p_exp
        self.cool_mode = cool_mode
        self.fixed_cool_beta = fixed_cool_beta
        self.v_corr = v_corr
        self.fixed_kappa = dict(fixed_kappa or {})
        self.share_a = share_a
        self.share_b = share_b  # False | True (all corners) | "axle" (front / rear)
        self.drive_term = drive_term
        self.brake_term = brake_term
        self.fixed_drive = dict(fixed_drive or {})
        self.fixed_brake = dict(fixed_brake or {})
        q4s = []
        q4d = []
        q4b = []
        self.cars = sorted({s.car for s in stints})
        car_i = []
        v_anchor = []
        g2, teff, obs, seg_start, t0, cell_i, logp = [], [], [], [], [], [], []
        vv = []
        n0 = 0
        for s in stints:
            n = len(s.g2)
            ss = np.zeros((n, 4), dtype=int)
            tt = np.zeros((n, 4))
            for j in range(4):
                a_i = s.anchor_idx[j] if s.anchor_idx[j] >= 0 else 0
                ss[:, j] = n0 + a_i
                tt[:, j] = s.t_start[j] if np.isfinite(s.t_start[j]) else s.t_eff_c
            g2.append(s.g2)
            q4s.append(s.q4 if s.q4 is not None else np.repeat(s.g2[:, None], 4, axis=1))
            q4d.append(s.q4_drive if s.q4_drive is not None else np.zeros((n, 4)))
            q4b.append(s.q4_brake if s.q4_brake is not None else np.zeros((n, 4)))
            vv.append(s.v)
            teff.append(np.full(n, s.t_eff_c))
            obs.append(s.obs)
            seg_start.append(ss)
            t0.append(tt)
            cell_i.append(np.full(n, self._cell_index(s.car, s.condition)))
            car_i.append(np.full(n, self.cars.index(s.car)))
            va = np.zeros((n, 4))
            for j in range(4):
                a_i = s.anchor_idx[j] if s.anchor_idx[j] >= 0 else 0
                va[:, j] = s.v[a_i]
            v_anchor.append(va)
            if p_mode is not None:
                logp.append(stint_log_pressure_ratio(s, p_mode))
            n0 += n
        self.g2 = np.concatenate(g2)
        self.q4 = np.concatenate(q4s)
        self.q4_drive = np.concatenate(q4d)
        self.q4_brake = np.concatenate(q4b)
        self.v = np.concatenate(vv)
        self.car_idx = np.concatenate(car_i)
        self.v_anchor = np.concatenate(v_anchor)
        self.logp = np.concatenate(logp) if logp else None
        self.teff = np.concatenate(teff)
        self.obs = np.concatenate(obs)
        self.seg_start = np.concatenate(seg_start)
        self.t0 = np.concatenate(t0)
        self.cell = np.concatenate(cell_i)
        self.fin = np.isfinite(self.obs)
        self.names: list[str] = []
        for car, cond in cells:
            self.names += [f"a|{car}|{c}|{cond}" for c in (("*",) if share_a else CORNERS)]
            self.names += [f"b|{car}|{c}|{cond}" for c in self._b_keys()]
        if p_mode is not None and fixed_p_exp is None:
            self.names.append("n|p")
        if cool_mode is not None and fixed_cool_beta is None:
            self.names.append("m|v")
        if v_corr:
            self.names += [f"k|{car}" for car in self.cars if car not in self.fixed_kappa]
        if drive_term:
            self.names += [f"d|{car}" for car in self.cars if car not in self.fixed_drive]
        if brake_term:
            self.names += [f"cb|{car}" for car in self.cars if car not in self.fixed_brake]

    def _b_keys(self) -> tuple[str, ...]:
        if self.share_b == "axle":
            return ("f*", "r*")
        return ("*",) if self.share_b else CORNERS

    def _b_key(self, corner: str) -> str:
        if self.share_b == "axle":
            return corner[0] + "*"
        return "*" if self.share_b else corner

    def _cell_index(self, car: str, cond: str) -> int:
        try:
            return self.cells.index((car, cond))
        except ValueError:
            return -1

    def p_exp(self, x: np.ndarray) -> float:
        """The heat-input pressure exponent ``n`` (``q ∝ (P_ref / P)^n``)."""
        if self.p_mode is None:
            return 0.0
        if self.fixed_p_exp is not None:
            return self.fixed_p_exp
        return float(x[self.names.index("n|p")])  # linear, not log-scaled

    def kappa(self, x: np.ndarray) -> dict[str, float]:
        """Per-car speed-pressure constant κ (linear, not log-scaled)."""
        out = dict(self.fixed_kappa)
        if self.v_corr:
            for car in self.cars:
                if car not in out:
                    out[car] = float(x[self.names.index(f"k|{car}")])
        return out

    def _per_car_linear(
        self, x: np.ndarray, prefix: str, fixed: dict[str, float]
    ) -> dict[str, float]:
        out = dict(fixed)
        for car in self.cars:
            nme = f"{prefix}|{car}"
            if car not in out and nme in self.names:
                out[car] = float(x[self.names.index(nme)])
        return out

    def drive_coef(self, x: np.ndarray) -> dict[str, float]:
        """Per-car drive-slip heat coefficient ``d`` (0 when off)."""
        return (
            self._per_car_linear(x, "d", self.fixed_drive)
            if (self.drive_term or self.fixed_drive)
            else {}
        )

    def brake_coef(self, x: np.ndarray) -> dict[str, float]:
        """Per-car brake-power heat coefficient ``c_b`` (0 when off)."""
        return (
            self._per_car_linear(x, "cb", self.fixed_brake)
            if (self.brake_term or self.fixed_brake)
            else {}
        )

    def heat_input(self, x: np.ndarray) -> np.ndarray:
        """``(n, 4)`` per-corner driving intensity including the fitted
        drive and brake terms."""
        q = self.q4
        d = self.drive_coef(x)
        cb = self.brake_coef(x)
        if d:
            q = (
                q
                + np.array([d.get(c, 0.0) for c in self.cars])[self.car_idx][:, None]
                * self.q4_drive
            )
        if cb:
            q = (
                q
                + np.array([cb.get(c, 0.0) for c in self.cars])[self.car_idx][:, None]
                * self.q4_brake
            )
        return np.asarray(q)

    def observed(self, x: np.ndarray) -> np.ndarray:
        """The gas temperature observations, corrected for the speed effect
        on the pressure reading when ``v_corr`` is on."""
        if not self.v_corr and not self.fixed_kappa:
            return self.obs
        kap = self.kappa(x)
        k = np.array([kap.get(c, 0.0) for c in self.cars])[self.car_idx]
        f = (1.0 + k[:, None] * self.v[:, None] ** 2) / (1.0 + k[:, None] * self.v_anchor**2)
        return np.asarray((self.obs + T_ZERO_C_TO_K) * f - T_ZERO_C_TO_K)

    def cool_beta(self, x: np.ndarray) -> float:
        """The forced-convection slope ``β`` (see :func:`cooling_factor`)."""
        if self.cool_mode is None:
            return 0.0
        if self.fixed_cool_beta is not None:
            return self.fixed_cool_beta
        return float(x[self.names.index("m|v")])  # linear, not log-scaled

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = dict(zip(self.names, np.exp(x)))
        n_cells = len(self.cells)
        a = np.zeros((n_cells, 4))
        b = np.zeros((n_cells, 4))
        for i, (car, cond) in enumerate(self.cells):
            for j, c in enumerate(CORNERS):
                a[i, j] = p[f"a|{car}|{'*' if self.share_a else c}|{cond}"]
                b[i, j] = p[f"b|{car}|{self._b_key(c)}|{cond}"]
        return a, b

    def simulate(self, x: np.ndarray) -> np.ndarray:
        a_t, b_t = self.unpack(x)
        rows = self.cell >= 0
        a = np.zeros((len(self.g2), 4))
        b = np.full((len(self.g2), 4), 1.0 / 600.0)
        a[rows] = a_t[self.cell[rows]]
        b[rows] = b_t[self.cell[rows]]
        if self.cool_mode == "speed":
            b = b * cooling_factor(self.v, self.cool_beta(x))[:, None]
        q = self.heat_input(x)
        if self.logp is not None:
            q = q * np.exp(-self.p_exp(x) * self.logp)
        # exact per-second update, segment-wise via cumulative sums
        S = np.cumsum(b, 0)
        S_prev = S - b
        d = np.exp(-b)
        teq = self.teff[:, None] + a * q / b
        s_at = np.take_along_axis(S_prev, self.seg_start, 0)
        srel_next = S - s_at
        srel = S_prev - s_at
        G = np.exp(np.minimum(srel_next, 600.0)) * (1 - d) * teq
        CG = np.cumsum(G, 0)
        CG_prev = CG - G
        cg_at = np.take_along_axis(CG_prev, self.seg_start, 0)
        T: np.ndarray = np.exp(-srel) * (self.t0 + (CG_prev - cg_at))
        return T

    def residuals(self, x: np.ndarray) -> np.ndarray:
        T = self.simulate(x)
        r: np.ndarray = (T - self.observed(x))[self.fin & (self.cell >= 0)[:, None]]
        return r

    def bounds(
        self, b_lower: dict[tuple[str, str, str], float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # The scale of ``a`` depends on the heat-input units (G² or G²·m/s):
        # start where a typical stint reaches ~25 K above T_eff in 400 s.
        q_typ = float(np.mean(self.g2[self.cell >= 0])) if (self.cell >= 0).any() else 1.0
        a0 = 25.0 / 400.0 / max(q_typ, 1e-9)
        x0, lo, hi = [], [], []
        for nme in self.names:
            kind = nme.split("|")[0]
            if kind == "a":
                x0.append(a0)
                lo.append(a0 * A_BOUNDS[0])
                hi.append(a0 * A_BOUNDS[1])
            elif kind == "b":
                _, car, corner, cond = nme.split("|")
                group = [c for c in CORNERS if corner in ("*", c) or corner == c[0] + "*"]
                lo_b = max(
                    1.0 / TAU_BOUNDS_S[1], max(b_lower.get((car, c, cond), 0.0) for c in group)
                )
                x0.append(max(1.0 / 400.0, lo_b * 1.05))
                lo.append(lo_b)
                hi.append(1.0 / TAU_BOUNDS_S[0])
        x0l, lol, hil = list(np.log(x0)), list(np.log(lo)), list(np.log(hi))
        if "n|p" in self.names:  # the exponent is carried linearly
            x0l.append(1.0)
            lol.append(P_EXP_BOUNDS[0])
            hil.append(P_EXP_BOUNDS[1])
        if "m|v" in self.names:
            x0l.append(1.0)
            lol.append(COOL_BETA_BOUNDS[0])
            hil.append(COOL_BETA_BOUNDS[1])
        for nme in self.names:
            if nme.startswith("k|"):
                x0l.append(2e-6)
                lol.append(KAPPA_BOUNDS[0])
                hil.append(KAPPA_BOUNDS[1])
            elif nme.startswith("d|") or nme.startswith("cb|"):
                x0l.append(0.5)
                lol.append(0.0)
                hil.append(20.0)
        return np.array(x0l), np.array(lol), np.array(hil)


def lap_heat_components(stints: list[StintSeries]) -> pd.DataFrame:
    """Per (session, stint, lap_num, corner): the lap sums of the three
    heat-input parts (``base``, ``drive``, ``brake``; see
    :func:`heat_input.corner_heat_parts`) and the lap's moving seconds.
    A corner-blind stint (no geometry) carries its ``g2`` as ``base``.
    The lap's driving intensity is ``(base + d·drive + c_b·brake) / seconds``."""
    rows = []
    for s in stints:
        lap_idx = np.cumsum(s.lap_ends) - s.lap_ends  # 0-based lap index per bin
        n_laps = int(lap_idx.max()) + 1 if len(lap_idx) else 0
        seconds = np.bincount(lap_idx, minlength=n_laps).astype(float)
        for j, c in enumerate(CORNERS):
            base = s.q4[:, j] if s.q4 is not None else s.g2
            drive = s.q4_drive[:, j] if s.q4_drive is not None else np.zeros_like(base)
            brake = s.q4_brake[:, j] if s.q4_brake is not None else np.zeros_like(base)
            b_sum = np.bincount(lap_idx, weights=base, minlength=n_laps)
            d_sum = np.bincount(lap_idx, weights=drive, minlength=n_laps)
            k_sum = np.bincount(lap_idx, weights=brake, minlength=n_laps)
            for k in range(n_laps):
                rows.append(
                    {
                        "session_id": s.session_id,
                        "stint_id": s.stint_id,
                        "lap_num": s.lap_lo + k,
                        "corner": c,
                        "base": float(b_sum[k]),
                        "drive": float(d_sum[k]),
                        "brake": float(k_sum[k]),
                        "seconds": float(seconds[k]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "session_id",
            "stint_id",
            "lap_num",
            "corner",
            "base",
            "drive",
            "brake",
            "seconds",
        ],
    )


def lap_heat_frame(
    components: pd.DataFrame,
    drive_by_car: dict[str, float],
    brake_by_car: dict[str, float],
    car_by_session: dict[str, str],
) -> pd.DataFrame:
    """Wide per-lap frame ``q_lap_{corner}`` (driving intensity per corner,
    units of the fit's q) from :func:`lap_heat_components`."""
    if components.empty:
        return pd.DataFrame(columns=["session_id", "stint_id", "lap_num"])
    df = components.copy()
    car = df["session_id"].map(car_by_session)
    d = car.map(drive_by_car).fillna(0.0).to_numpy(dtype=float)
    cb = car.map(brake_by_car).fillna(0.0).to_numpy(dtype=float)
    sec = df["seconds"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["q_lap"] = np.where(
            sec > 0,
            (df["base"] + d * df["drive"] + cb * df["brake"]) / np.maximum(sec, 1e-9),
            np.nan,
        )
    wide = df.pivot_table(
        index=["session_id", "stint_id", "lap_num"],
        columns="corner",
        values="q_lap",
        aggfunc="first",
    )
    wide.columns = [f"q_lap_{c}" for c in wide.columns]
    return wide.reset_index()


@dataclass
class PhysicalFit:
    """Result of :func:`fit_physical`: the schema-v5 production model."""

    tau: dict[tuple[str, str, str], Any]  # (car, corner, cond) -> FitParam seconds
    k: dict[tuple[str, str, str], Any]  # (car, corner, cond) -> FitParam K per unit q
    n_samples: dict[tuple[str, str, str], int]
    kappa: dict[str, float]
    drive: dict[str, float]
    brake: dict[str, float]
    geometry: dict[str, CarGeometry]
    components: pd.DataFrame  # lap_heat_components of the fitted stints
    cost_dry: float


def fit_physical(
    root: Path,
    laps_for_fit: pd.DataFrame,
    *,
    geometry: dict[str, CarGeometry] | None = None,
) -> PhysicalFit:
    """The production fit (schema v5): the force × slip-fraction × speed
    input ``|g|·V/V_ref`` with the per-corner force-path split, one gain per (car, condition),
    one cooling coefficient per (car, axle, condition), fitted drive and
    brake coefficients and the speed-pressure constant per car, and **no
    track constants**."""
    from .warmup_table import FitParam

    geom = dict(geometry or BUILTIN_GEOMETRY)
    stints = build_stint_series(root, laps_for_fit, PHYSICAL_HEAT_INPUT, geom, 0.0, long_split=True)
    if not stints:
        return PhysicalFit({}, {}, {}, {}, {}, {}, geom, pd.DataFrame(), 0.0)
    cf = fit_cells(
        stints,
        v_corr=True,
        share_a=True,
        share_b="axle",
        drive_term=True,
        brake_term=True,
    )
    scored: dict[tuple[str, str, str], int] = {}
    for s in stints:
        for j, c in enumerate(CORNERS):
            if s.anchor_idx[j] >= 0:
                scored[(s.car, c, s.condition)] = scored.get((s.car, c, s.condition), 0) + s.n_laps
    tau: dict[tuple[str, str, str], Any] = {}
    k: dict[tuple[str, str, str], Any] = {}
    for (car, c, cond), (a_v, b_v) in cf.ab.items():
        n = scored.get((car, c, cond), 0)
        tau[(car, c, cond)] = FitParam(value=1.0 / b_v, stderr=0.0, n_samples=n)
        k[(car, c, cond)] = FitParam(value=a_v / b_v, stderr=0.0, n_samples=n)
    logger.info(
        "physical fit: %d stints, kappa %s, drive %s, brake %s",
        len(stints),
        cf.kappa,
        cf.drive,
        cf.brake,
    )
    return PhysicalFit(
        tau=tau,
        k=k,
        n_samples=scored,
        kappa=dict(cf.kappa),
        drive=dict(cf.drive),
        brake=dict(cf.brake),
        geometry=geom,
        components=lap_heat_components(stints),
        cost_dry=cf.cost_dry,
    )


def stint_speed_terms(root: Path, laps_for_fit: pd.DataFrame) -> pd.DataFrame:
    """Per (session, stint, lap_num): the speed over the lap's last rolling
    second (``speed_end_ms``, where ``tpms_press_{c}_end`` is read) and per
    corner the speed at the stint anchor (``speed_anchor_{c}_ms``), for the
    speed-pressure correction of lap-level observables."""
    rows = []
    for key, grp in laps_for_fit.groupby(["session_id", "stint_id"], sort=False):
        sid, stint = str(key[0]), int(grp["stint_id"].iloc[0])  # type: ignore[index]
        ts = _session_timeseries(str(root), sid)
        if ts is None:
            continue
        lap_nums = [int(x) for x in grp["lap_num"]]
        binned = _bin_stint(ts, lap_nums)
        if binned is None:
            continue
        v = binned["v"]
        ends = np.flatnonzero(binned["lap_end"])
        lo = min(lap_nums)
        first = grp.iloc[0]
        anchors = {}
        for c in CORNERS:
            ta = first.get(f"t_anchor_{c}")
            if ta is None or pd.isna(ta):
                anchors[c] = np.nan
            else:
                i = min(max(int(np.ceil(float(ta))), 0), len(v) - 1)
                anchors[c] = float(v[i])
        for k, e in enumerate(ends):
            rows.append(
                {
                    "session_id": sid,
                    "stint_id": stint,
                    "lap_num": lo + k,
                    "speed_end_ms": float(v[e]),
                    **{f"speed_anchor_{c}_ms": anchors[c] for c in CORNERS},
                }
            )
    cols = ["session_id", "stint_id", "lap_num", "speed_end_ms"] + [
        f"speed_anchor_{c}_ms" for c in CORNERS
    ]
    return pd.DataFrame(rows, columns=cols)


@dataclass
class CellFit:
    """Result of :func:`fit_cells`: per-(car, corner, condition) ``(a, b)`` and
    the per-car constants."""

    ab: dict[tuple[str, str, str], tuple[float, float]]
    cost_dry: float
    n_residuals_dry: int
    p_mode: str | None = None
    p_exp: float = 0.0
    cool_mode: str | None = None
    cool_beta: float = 0.0
    kappa: dict[str, float] = field(default_factory=dict)  # per car, 0 when not fitted
    drive: dict[str, float] = field(default_factory=dict)  # per car drive-slip heat coefficient
    brake: dict[str, float] = field(default_factory=dict)  # per car brake-power heat coefficient

    def params_for(self, car: str, corner: str, condition: str) -> tuple[float, float] | None:
        """``(a, b)`` for the cell, falling back along the condition chain."""
        for cond in _condition_chain(condition):
            ab = self.ab.get((car, corner, cond))
            if ab is not None:
                return ab
        return None


def _condition_chain(condition: str) -> tuple[str, ...]:
    if condition == "wet":
        return ("wet", "damp", "dry")
    if condition == "damp":
        return ("damp", "dry")
    return ("dry",)


def fit_cells(
    stints: list[StintSeries],
    *,
    p_mode: str | None = None,
    p_exp: float | None = None,
    cool_mode: str | None = None,
    cool_beta: float | None = None,
    v_corr: bool = False,
    kappa: dict[str, float] | None = None,
    share_a: bool = False,
    share_b: bool | str = False,
    drive_term: bool = False,
    brake_term: bool = False,
) -> CellFit:
    """Fit ``(a, b)`` per (car, corner, condition) on the given stints (dry
    first; rain cells after with ``b_rain ≥ b_dry`` and the per-car
    constants held).

    ``p_mode`` enables the pressure dependence of the heat input,
    ``q ∝ (P_ref / P)^n`` with one global exponent ``n`` fitted with the dry
    cells (or fixed at ``p_exp``) and held for the rain cells.
    """
    if p_mode is not None and p_mode not in P_MODES:
        raise ValueError(f"p_mode must be one of {P_MODES}; got {p_mode!r}")
    if cool_mode is not None and cool_mode not in COOL_MODES:
        raise ValueError(f"cool_mode must be one of {COOL_MODES}; got {cool_mode!r}")
    sess_by_cell: dict[tuple[str, str], set[str]] = {}
    for s in stints:
        sess_by_cell.setdefault((s.car, s.condition), set()).add(s.session_id)
    dry_cells = sorted(c for c in sess_by_cell if c[1] == "dry")
    rain_cells = sorted(
        c for c in sess_by_cell if c[1] != "dry" and len(sess_by_cell[c]) >= MIN_RAIN_SESSIONS
    )
    fitted: dict[tuple[str, str, str], tuple[float, float]] = {}
    dry_stints = [s for s in stints if s.condition == "dry"]
    d1 = _Design(
        dry_stints,
        dry_cells,
        {},
        p_mode,
        p_exp,
        cool_mode,
        cool_beta,
        v_corr,
        kappa,
        share_a,
        share_b,
        drive_term,
        brake_term,
    )
    x0, lo, hi = d1.bounds({})
    res = least_squares(
        d1.residuals, x0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=300
    )
    a, b = d1.unpack(res.x)
    n_p = d1.p_exp(res.x)
    beta = d1.cool_beta(res.x)
    kap = d1.kappa(res.x)
    drive = d1.drive_coef(res.x)
    brake = d1.brake_coef(res.x)
    for i, (car, cond) in enumerate(dry_cells):
        for j, c in enumerate(CORNERS):
            fitted[(car, c, cond)] = (float(a[i, j]), float(b[i, j]))
    logger.info(
        "per-second fit (dry): %d stints, %d residuals, cost %.1f",
        len(dry_stints),
        res.fun.size,
        res.cost,
    )
    if rain_cells:
        rain_stints = [s for s in stints if (s.car, s.condition) in rain_cells]
        d2 = _Design(
            rain_stints,
            rain_cells,
            fitted,
            p_mode,
            n_p,
            cool_mode,
            beta,
            False,
            kap,
            share_a,
            share_b,
            False,
            False,
            drive,
            brake,
        )
        b_lower = {}
        for car, cond in rain_cells:
            for c in CORNERS:
                dry = fitted.get((car, c, "dry"))
                if dry is not None:
                    b_lower[(car, c, cond)] = dry[1]
        x0, lo, hi = d2.bounds(b_lower)
        res2 = least_squares(
            d2.residuals, x0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=300
        )
        a2, b2 = d2.unpack(res2.x)
        for i, (car, cond) in enumerate(rain_cells):
            for j, c in enumerate(CORNERS):
                fitted[(car, c, cond)] = (float(a2[i, j]), float(b2[i, j]))
    return CellFit(
        ab=fitted,
        cost_dry=float(res.cost),
        n_residuals_dry=int(res.fun.size),
        p_mode=p_mode,
        p_exp=float(n_p),
        cool_mode=cool_mode,
        cool_beta=float(beta),
        kappa=kap,
        drive=drive,
        brake=brake,
    )
