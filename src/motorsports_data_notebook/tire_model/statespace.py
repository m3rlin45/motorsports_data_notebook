"""Per-second (1 Hz) fit of the lumped-capacity energy balance.

The per-lap closed form (``warmup_table._pass1_fit_tau_and_gains``) sees one
pressure sample per lap. The TPMS reports pressure in 0.03 bar steps, so a
lap-end sample is uncertain by ±0.015 bar (≈ 2 K of gas temperature) and the
pit-exit anchor sample the whole stint is referenced to is uncertain by the
same amount. At 1 Hz a rising pressure crosses a step every 10–30 s early in
a stint and the *time* of each crossing locates the pressure to a fraction
of a step, so the per-second series carries far more information than its
lap-end samples; measured g²(t) through the lap also identifies τ and K from
the shape of each lap rather than from end-of-lap levels only.

Model (per stint, per corner), integrated exactly with inputs held constant
over each 1 s step on the stint's rolling clock (standstill excluded):

    dT/dt = a · c_track · g²(t) − b · (T − T_eff),   T(t_anchor) = T_start
    K = a / b,  τ = 1 / b

Observation: the pressure-implied cavity-gas temperature
``T_gas_K = T_start_K · P(t)_abs / P_start_abs`` from the stint's pit-exit
(T, P) anchor, for every 1 s bin with a finite pressure at or after the
anchor. Parameters: ``a, b`` per (car, corner, condition) and ``c_track`` per
track (Tsukuba anchored at 1), fitted by bounded least squares in log space.
Rain conditions are fitted after dry with ``b_rain ≥ b_dry`` (τ_rain ≤ τ_dry)
as a bound. Outputs the same ``tau`` / per-track ``gain = K · c_track`` tables
Pass 1 produces, so Pass 2, the compound EM and the artifact are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from ..tire_etl.paths import timeseries_dir
from .energy_balance import P_ATM_BAR, T_ZERO_C_TO_K

logger = logging.getLogger(__name__)

CORNERS = ("fl", "fr", "rl", "rr")
MOVING_SPEED_MS = 5.0 / 3.6
MIN_SCORED_SECONDS = 60  # a (stint, corner) needs this much observed data to count
MIN_RAIN_SESSIONS = 3  # per (car, condition) for its own tau/K (else the condition chain)
TAU_BOUNDS_S = (30.0, 3000.0)
A_BOUNDS = (1e-4, 10.0)
C_TRACK_BOUNDS = (0.3, 3.0)


@dataclass
class StintSeries:
    session_id: str
    stint_id: int
    car: str
    track: str
    condition: str
    t_eff_c: float
    g2: np.ndarray  # (n,) mean lat²+long² per rolling second
    obs: np.ndarray  # (n, 4) pressure-implied gas temperature, NaN where missing
    anchor_idx: np.ndarray  # (4,) first scored bin per corner, -1 when none
    t_start: np.ndarray  # (4,) anchor temperature per corner
    n_laps: int
    lap_ends: np.ndarray  # (n,) bool: last bin of a lap


# ---------------------------------------------------------------- data prep


@lru_cache(maxsize=4)
def _session_timeseries(root_str: str, session_id: str) -> pd.DataFrame | None:
    files = list(timeseries_dir(Path(root_str)).glob(f"*/{session_id}.parquet"))
    if not files:
        return None
    cols = ["lap_num", "sample_idx", "t_session_s", "speed_ms", "lat_g", "long_g"]
    cols += [f"tpms_press_{c}_bar" for c in CORNERS]
    schema = pq.read_schema(files[0])
    cols = [c for c in cols if c in schema.names]
    df: pd.DataFrame = pq.read_table(files[0], columns=cols).to_pandas()
    return df


def _bin_stint(ts: pd.DataFrame, lap_nums: list[int]) -> dict[str, np.ndarray] | None:
    """1 Hz bins on the rolling clock for the given laps of one stint."""
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
    g2 = sub["lat_g"].to_numpy(dtype=float) ** 2 + sub["long_g"].to_numpy(dtype=float) ** 2
    g2 = np.nan_to_num(g2, nan=0.0)
    g2_bin = np.bincount(b[moving], weights=g2[moving], minlength=n) / np.maximum(cnt, 1)
    g2_bin = np.where(cnt > 0, g2_bin, 0.0)
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
    return {"g2": g2_bin, "press": press, "lap_end": lap_end}


def build_stint_series(root: Path, laps_for_fit: pd.DataFrame) -> list[StintSeries]:
    """One :class:`StintSeries` per (session, stint) present in ``laps_for_fit``
    (which carries the anchors, T_eff, condition, car and track)."""
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
        binned = _bin_stint(ts, [int(x) for x in grp["lap_num"]])
        if binned is None:
            continue
        n = len(binned["g2"])
        obs = np.full((n, 4), np.nan)
        anchor_idx = np.full(4, -1, dtype=int)
        t_start = np.full(4, np.nan)
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
                obs=obs,
                anchor_idx=anchor_idx,
                t_start=t_start,
                n_laps=int(len(grp)),
                lap_ends=binned["lap_end"],
            )
        )
    return out


# ---------------------------------------------------------------- design + fit


class _Design:
    """Flattened arrays for a set of stints and the (car, condition) cells
    being fitted; other cells' parameters are held fixed."""

    def __init__(
        self,
        stints: list[StintSeries],
        cells: list[tuple[str, str]],
        tracks: list[str],
        anchor_track: str,
        fixed: dict[tuple[str, str, str], tuple[float, float]],
        fixed_c: dict[str, float],
    ):
        self.cells = cells
        self.tracks = [t for t in tracks if t != anchor_track and t not in fixed_c]
        self.anchor_track = anchor_track
        self.fixed = fixed
        self.fixed_c = dict(fixed_c)
        g2, teff, obs, seg_start, t0, cell_i, trk_i = [], [], [], [], [], [], []
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
            teff.append(np.full(n, s.t_eff_c))
            obs.append(s.obs)
            seg_start.append(ss)
            t0.append(tt)
            cell_i.append(np.full(n, self._cell_index(s.car, s.condition)))
            trk_i.append(np.full(n, tracks.index(s.track)))
            n0 += n
        self.g2 = np.concatenate(g2)
        self.teff = np.concatenate(teff)
        self.obs = np.concatenate(obs)
        self.seg_start = np.concatenate(seg_start)
        self.t0 = np.concatenate(t0)
        self.cell = np.concatenate(cell_i)
        self.trk = np.concatenate(trk_i)
        self.all_tracks = tracks
        self.fin = np.isfinite(self.obs)
        self.names: list[str] = []
        for car, cond in cells:
            self.names += [f"a|{car}|{c}|{cond}" for c in CORNERS]
            self.names += [f"b|{car}|{c}|{cond}" for c in CORNERS]
        self.names += [f"c|{t}" for t in self.tracks]

    def _cell_index(self, car: str, cond: str) -> int:
        try:
            return self.cells.index((car, cond))
        except ValueError:
            return -1

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = dict(zip(self.names, np.exp(x)))
        n_cells = len(self.cells)
        a = np.zeros((n_cells, 4))
        b = np.zeros((n_cells, 4))
        for i, (car, cond) in enumerate(self.cells):
            for j, c in enumerate(CORNERS):
                a[i, j] = p[f"a|{car}|{c}|{cond}"]
                b[i, j] = p[f"b|{car}|{c}|{cond}"]
        ct = np.ones(len(self.all_tracks))
        for i, t in enumerate(self.all_tracks):
            if t in self.fixed_c:
                ct[i] = self.fixed_c[t]
            elif t != self.anchor_track:
                ct[i] = p[f"c|{t}"]
        return a, b, ct

    def simulate(self, x: np.ndarray) -> np.ndarray:
        a_t, b_t, ct = self.unpack(x)
        rows = self.cell >= 0
        a = np.zeros((len(self.g2), 4))
        b = np.full((len(self.g2), 4), 1.0 / 600.0)
        a[rows] = a_t[self.cell[rows]]
        b[rows] = b_t[self.cell[rows]]
        q = (self.g2 * ct[self.trk])[:, None]
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
        r: np.ndarray = (T - self.obs)[self.fin & (self.cell >= 0)[:, None]]
        return r

    def bounds(
        self, b_lower: dict[tuple[str, str, str], float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0, lo, hi = [], [], []
        for nme in self.names:
            kind = nme.split("|")[0]
            if kind == "a":
                x0.append(0.08)
                lo.append(A_BOUNDS[0])
                hi.append(A_BOUNDS[1])
            elif kind == "b":
                _, car, corner, cond = nme.split("|")
                lo_b = max(1.0 / TAU_BOUNDS_S[1], b_lower.get((car, corner, cond), 0.0))
                x0.append(max(1.0 / 400.0, lo_b * 1.05))
                lo.append(lo_b)
                hi.append(1.0 / TAU_BOUNDS_S[0])
            else:
                x0.append(1.0)
                lo.append(C_TRACK_BOUNDS[0])
                hi.append(C_TRACK_BOUNDS[1])
        return np.log(x0), np.log(lo), np.log(hi)


def fit_tau_and_gains(
    root: Path,
    laps_for_fit: pd.DataFrame,
    *,
    anchor_track: str,
    min_laps_for_fit: int = 30,
) -> tuple[
    dict[tuple[str, str, str], Any],
    dict[tuple[str, str, str, str], Any],
    dict[tuple[str, str, str, str], int],
]:
    """Per-second fit. Returns ``(tau_by_car_corner_cond, bucket_gains,
    bucket_n_samples)`` in the same shapes as Pass 1 (FitParam values)."""
    from .warmup_table import FitParam

    stints = build_stint_series(root, laps_for_fit)
    if not stints:
        return {}, {}, {}
    tracks = sorted({s.track for s in stints})
    if anchor_track not in tracks:
        tracks = [anchor_track] + tracks
    # Which (car, condition) cells have enough data to fit on their own.
    sess_by_cell: dict[tuple[str, str], set[str]] = {}
    laps_by_bucket: dict[tuple[str, str, str], int] = {}
    for s in stints:
        sess_by_cell.setdefault((s.car, s.condition), set()).add(s.session_id)
        laps_by_bucket[(s.car, s.track, s.condition)] = (
            laps_by_bucket.get((s.car, s.track, s.condition), 0) + s.n_laps
        )
    dry_cells = sorted(c for c in sess_by_cell if c[1] == "dry")
    rain_cells = sorted(
        c for c in sess_by_cell if c[1] != "dry" and len(sess_by_cell[c]) >= MIN_RAIN_SESSIONS
    )

    fitted: dict[tuple[str, str, str], tuple[float, float]] = {}  # (car, corner, cond) -> (a, b)
    fixed_c: dict[str, float] = {}
    # ---- phase 1: dry cells + c_track ----
    dry_stints = [s for s in stints if s.condition == "dry"]
    d1 = _Design(dry_stints, dry_cells, tracks, anchor_track, {}, {})
    x0, lo, hi = d1.bounds({})
    res = least_squares(
        d1.residuals, x0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=300
    )
    a, b, ct = d1.unpack(res.x)
    for i, (car, cond) in enumerate(dry_cells):
        for j, c in enumerate(CORNERS):
            fitted[(car, c, cond)] = (float(a[i, j]), float(b[i, j]))
    for i, t in enumerate(tracks):
        fixed_c[t] = float(ct[i])
    logger.info(
        "per-second fit (dry): %d stints, %d residuals, cost %.1f, c_track %s",
        len(dry_stints),
        res.fun.size,
        res.cost,
        {t: round(v, 3) for t, v in fixed_c.items()},
    )
    # ---- phase 2: rain cells with b >= b_dry (tau_rain <= tau_dry), c_track fixed ----
    if rain_cells:
        rain_stints = [s for s in stints if (s.car, s.condition) in rain_cells]
        d2 = _Design(rain_stints, rain_cells, tracks, anchor_track, fitted, fixed_c)
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
        a2, b2, _ = d2.unpack(res2.x)
        for i, (car, cond) in enumerate(rain_cells):
            for j, c in enumerate(CORNERS):
                fitted[(car, c, cond)] = (float(a2[i, j]), float(b2[i, j]))

    # ---- outputs in Pass 1 shape ----
    tau_out: dict[tuple[str, str, str], Any] = {}
    gains: dict[tuple[str, str, str, str], Any] = {}
    n_samples: dict[tuple[str, str, str, str], int] = {}
    scored_laps: dict[tuple[str, str, str], int] = {}
    for s in stints:
        for j, c in enumerate(CORNERS):
            if s.anchor_idx[j] >= 0:
                scored_laps[(s.car, c, s.condition)] = (
                    scored_laps.get((s.car, c, s.condition), 0) + s.n_laps
                )
    for (car, c, cond), (a_v, b_v) in fitted.items():
        n_cell = scored_laps.get((car, c, cond), 0)
        tau_out[(car, c, cond)] = FitParam(value=1.0 / b_v, stderr=0.0, n_samples=n_cell)
        k_v = a_v / b_v
        for t in tracks:
            n_b = laps_by_bucket.get((car, t, cond), 0)
            if n_b < min_laps_for_fit:
                continue
            gains[(car, t, c, cond)] = FitParam(
                value=k_v * fixed_c.get(t, 1.0), stderr=0.0, n_samples=n_b
            )
            n_samples[(car, t, c, cond)] = n_b
    return tau_out, gains, n_samples
