"""Compare heat-input forms for the per-second energy-balance fit.

For each candidate :class:`~.heat_input.HeatInput` this harness

1. builds the 1 Hz stint series (the same ones ``statespace`` fits),
2. fits ``(a, b)`` per (car, corner, condition) on the
   training sessions of each fold, simulates every held-out stint from its
   own measured trace and pit-exit anchor, and scores the lap-end residual
   in the pressure domain (what the driver gets) and in gas temperature,
3. refits on everything and reports how *constant* the constants are: the
   the spread of per-stint implied gains around the
   pooled value (per car and per track) and the wet/dry gain ratio.

Held-out numbers here are *oracle* inputs (the real driving), i.e. the
thermal model's accuracy. The calculator-input number comes from
``tire-predict-holdout`` once a form is wired through the lookups.

Run::

    uv run python -m motorsports_data_notebook.tire_model.heat_experiment \\
        --form g2 --form sliding --form "sliding:glim=1.15" --n-folds 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..tire_etl.paths import default_dataset_root
from .energy_balance import P_ATM_BAR, T_ZERO_C_TO_K
from .heat_input import BUILTIN_GEOMETRY, SHARE_PARAM_NAMES, HeatInput, share_heat_rate
from .statespace import (
    CORNERS,
    P_REF_ABS_BAR,
    CellFit,
    StintSeries,
    build_stint_series,
    cooling_factor,
    fit_cells,
    speed_pressure_factor,
    stint_log_pressure_ratio,
    COOL_MODES,
    cooling_power_factor,
    sink_temperature,
)

logger = logging.getLogger(__name__)

G_REF_PERCENTILE = 99.5  # per-car grip reference: this percentile of moving total g (dry)


# ---------------------------------------------------------------- data prep


def prepare_laps_for_fit(root: Path) -> pd.DataFrame:
    """The ``laps_for_fit`` frame of ``build_warmup_table`` for every session
    (anchors, T_eff, condition); folds split it by session afterwards."""
    from .warmup_table import (
        _apply_blacklist,
        _attach_weather,
        _build_g2_typ,
        _compute_delta_t,
        _compute_stint_anchor,
        _compute_stint_clock,
        _flying_laps,
        _laps_for_fit,
        _load_filtered_laps,
        _load_weather,
        load_sensor_blacklist,
    )

    laps = _load_filtered_laps(root)
    laps = _attach_weather(laps, _load_weather(root))
    laps = _compute_stint_clock(laps)
    laps, _ = _apply_blacklist(laps, load_sensor_blacklist(root))
    laps = _compute_stint_anchor(laps)
    laps = _compute_delta_t(laps)
    return _laps_for_fit(laps, _build_g2_typ(_flying_laps(laps)))


def car_g_reference(root: Path, laps_for_fit: pd.DataFrame) -> dict[str, float]:
    """Per (pooled) car: the ``G_REF_PERCENTILE`` of moving total g over its
    dry sessions — the grip limit the slip activation is scaled to."""
    from .statespace import MOVING_SPEED_MS, _session_timeseries

    parts: dict[str, list[np.ndarray]] = {}
    dry = laps_for_fit[laps_for_fit["condition"] == "dry"]
    for sid, car in dry.drop_duplicates("session_id")[["session_id", "car"]].itertuples(
        index=False
    ):
        ts = _session_timeseries(str(root), str(sid))
        if ts is None or "speed_ms" not in ts.columns:
            continue
        moving = ts["speed_ms"].to_numpy(dtype=float) > MOVING_SPEED_MS
        lat = np.nan_to_num(ts["lat_g"].to_numpy(dtype=float)[moving], nan=0.0)
        lng = np.nan_to_num(ts["long_g"].to_numpy(dtype=float)[moving], nan=0.0)
        parts.setdefault(str(car), []).append(np.sqrt(lat * lat + lng * lng))
    return {
        car: float(np.percentile(np.concatenate(v), G_REF_PERCENTILE)) for car, v in parts.items()
    }


# ---------------------------------------------------------------- simulation


def simulate_corner(
    q: np.ndarray,
    t_eff: float,
    a: float,
    b: float | np.ndarray,
    t_start: float,
    start: int,
) -> np.ndarray:
    """Exact per-second integration of ``dT/dt = a·c·q − b·(T − T_eff)``
    from ``T[start] = t_start``; NaN before ``start``. ``b`` may be a
    per-second array (speed-dependent cooling)."""
    n = len(q)
    T = np.full(n, np.nan)
    if start < 0 or start >= n:
        return T
    bb = np.broadcast_to(np.asarray(b, dtype=float), (n,))[start:]
    teq = t_eff + a * q[start:] / bb
    d = np.exp(-bb)
    # With S_k = Σ_{j<k} b_j:  T[s+k] = e^{-S_k} (T_s + Σ_{j<k} (1-d_j) teq_j e^{S_{j+1}})
    S_next = np.cumsum(bb)
    S = S_next - bb
    w = (1.0 - d) * teq * np.exp(np.minimum(S_next, 600.0))
    acc = np.concatenate([[0.0], np.cumsum(w)[:-1]])
    T[start:] = np.exp(-S) * (t_start + acc)
    return T


def simulate_corner_own_pressure(
    q: np.ndarray,
    t_eff: float,
    a: float,
    b: float | np.ndarray,
    t_start: float,
    start: int,
    p_start_abs: float,
    p_exp: float,
) -> np.ndarray:
    """As :func:`simulate_corner` (``b`` scalar or per-second) with ``q`` scaled by ``(P_ref / P)^n`` where
    ``P`` is the model's *own* gas-law pressure at each second (no use of the
    observed pressure): the prediction-time form of the ``instant`` mode."""
    n = len(q)
    T = np.full(n, np.nan)
    if start < 0 or start >= n:
        return T
    bb = np.broadcast_to(np.asarray(b, dtype=float), (n,))
    t0_k = t_start + T_ZERO_C_TO_K
    T[start] = t_start
    for k in range(start, n - 1):
        p_abs = p_start_abs * (T[k] + T_ZERO_C_TO_K) / t0_k
        teq = t_eff + a * q[k] * (P_REF_ABS_BAR / p_abs) ** p_exp / bb[k]
        T[k + 1] = teq + (T[k] - teq) * np.exp(-bb[k])
    return T


def _corner_params(fit: CellFit, s: StintSeries, corner: str) -> tuple[float, float] | None:
    """Cell parameters for a stint, with a ``car/compound`` label falling
    back to the pooled car."""
    ab = fit.params_for(s.car, corner, s.condition)
    if ab is None and "/" in s.car:
        ab = fit.params_for(s.car.split("/", 1)[0], corner, s.condition)
    return ab


def _effective_q(s: StintSeries, j: int, fit: CellFit) -> np.ndarray:
    """``q`` for a corner, including the anchor-pressure factor when the fit
    used ``p_mode == "anchor"``."""
    q = _corner_q(s, j, fit)
    if fit.p_mode == "anchor":
        q = q * np.exp(-fit.p_exp * stint_log_pressure_ratio(s, "anchor")[:, j])
    return q


def _corner_q(s: StintSeries, j: int, fit: CellFit) -> np.ndarray:
    """Per-corner driving intensity with the fitted drive/brake terms (or the
    bounded force-share model when the fit carries one)."""
    car = s.car.split("/", 1)[0]
    if fit.share and car in fit.share and s.lat_pos is not None:
        return share_heat_rate(
            fit.share[car], s.lat_pos, s.lat_neg, s.long_pos, s.long_neg, s.v, CORNERS[j]  # type: ignore[arg-type]
        )
    q = s.q4[:, j] if s.q4 is not None else s.g2
    if s.q4_drive is not None and fit.drive.get(car):
        q = q + fit.drive[car] * s.q4_drive[:, j]
    if s.q4_brake is not None and fit.brake.get(car):
        q = q + fit.brake[car] * s.q4_brake[:, j]
    return np.asarray(q)


def _b_series(s: StintSeries, fit: CellFit, b: float, j: int) -> float | np.ndarray:
    cp = fit.cool_params
    if fit.cool_mode == "speed":
        return b * cooling_factor(s.v, fit.cool_beta)
    if fit.cool_mode == "speed_axle":
        return b * cooling_factor(s.v, cp["m|v|f*" if j < 2 else "m|v|r*"])
    if fit.cool_mode == "speed_car":
        return b * cooling_factor(s.v, cp[f"m|v|{s.car.split('/', 1)[0]}"])
    return b


def _simulate(s: StintSeries, j: int, fit: CellFit, a: float, b: float) -> np.ndarray:
    t0 = float(s.t_start[j])
    start = int(s.anchor_idx[j])
    b_t = _b_series(s, fit, b, j)
    t_eff = sink_temperature(s, fit.cool_params, CORNERS[j])
    if fit.p_mode == "instant":
        return simulate_corner_own_pressure(
            _corner_q(s, j, fit),
            t_eff,
            a,
            b_t,
            t0,
            start,
            float(s.p_start[j]) + P_ATM_BAR,
            fit.p_exp,
        )
    q = _effective_q(s, j, fit)
    T = simulate_corner(q, t_eff, a, b_t, t0, start)
    if fit.cool_mode == "power":
        n = fit.cool_params["n|cool"]
        for _ in range(3):
            T = simulate_corner(q, t_eff, a, b_t * cooling_power_factor(T, t_eff, n), t0, start)
    return T


@dataclass
class ScoredRow:
    session_id: str
    stint_id: int
    car: str
    track: str
    condition: str
    corner: str
    lap_within_stint: int
    t_s: float
    T_pred_c: float
    T_obs_c: float
    P_pred_bar: float
    P_obs_bar: float


def score_stints(stints: list[StintSeries], fit: CellFit) -> pd.DataFrame:
    """Simulate each stint/corner from its anchor with ``fit`` and score the
    lap ends after the anchor lap (the first lap end after the anchor is the
    anchor's own lap and is skipped, as in ``tire-predict-holdout``)."""
    rows: list[dict[str, Any]] = []
    for s in stints:
        ends = np.flatnonzero(s.lap_ends)
        for j, corner in enumerate(CORNERS):
            start = int(s.anchor_idx[j])
            if start < 0:
                continue
            ab = _corner_params(fit, s, corner)
            if ab is None:
                continue
            a, b = ab
            T = _simulate(s, j, fit, a, b)
            after = ends[ends > start]
            ratio = (float(s.p_start[j]) + P_ATM_BAR) / (float(s.t_start[j]) + T_ZERO_C_TO_K)
            kap = fit.kappa.get(s.car, 0.0)
            f_anchor = float(speed_pressure_factor(np.array([s.v[start]]), kap)[0])
            for n_lap, e in enumerate(after[1:], start=1):
                obs = s.obs[e, j]
                if not np.isfinite(obs):
                    continue
                # The reading at speed sits below the gas-law pressure by
                # (1 + κV²): compare in the *observed* pressure domain.
                f_e = float(speed_pressure_factor(np.array([s.v[e]]), kap)[0]) / f_anchor
                obs = (obs + T_ZERO_C_TO_K) * f_e - T_ZERO_C_TO_K  # gas temperature implied
                rows.append(
                    {
                        "session_id": s.session_id,
                        "stint_id": s.stint_id,
                        "car": s.car,
                        "track": s.track,
                        "condition": s.condition,
                        "corner": corner,
                        "lap_within_stint": n_lap,
                        "t_s": float(e - start),
                        "T_pred_c": float(T[e]),
                        "T_obs_c": float(obs),
                        "P_pred_bar": (T[e] + T_ZERO_C_TO_K) * ratio / f_e - P_ATM_BAR,
                        "P_obs_bar": (obs + T_ZERO_C_TO_K) * ratio / f_e - P_ATM_BAR,
                        "v_ms": float(s.v[e]),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["resid_c"] = df["T_pred_c"] - df["T_obs_c"]
        df["resid_bar"] = df["P_pred_bar"] - df["P_obs_bar"]
    return df


# ---------------------------------------------------------------- stability


def implied_gains(stints: list[StintSeries], fit: CellFit) -> pd.DataFrame:
    """Per (stint, corner): the gain ``a`` that best explains that stint
    alone with ``b`` held at the pooled fit, as a ratio to
    the pooled ``a``. A perfect model has ratios of 1 everywhere."""
    rows: list[dict[str, Any]] = []
    for s in stints:
        for j, corner in enumerate(CORNERS):
            start = int(s.anchor_idx[j])
            if start < 0:
                continue
            ab = _corner_params(fit, s, corner)
            if ab is None:
                continue
            a, b = ab
            t0 = float(s.t_start[j])
            # Implied gain uses the measured pressure for the instant mode
            # (an input, like g²), keeping T affine in a.
            q = _corner_q(s, j, fit)
            if fit.p_mode is not None:
                q = q * np.exp(-fit.p_exp * stint_log_pressure_ratio(s, fit.p_mode)[:, j])
            b_t = _b_series(s, fit, b, j)
            t_eff = sink_temperature(s, fit.cool_params, CORNERS[j])
            T0 = simulate_corner(q, t_eff, 0.0, b_t, t0, start)
            T1 = simulate_corner(q, t_eff, 1.0, b_t, t0, start) - T0
            m = np.isfinite(s.obs[:, j]) & np.isfinite(T0)
            if m.sum() < 60 or float(np.sum(T1[m] ** 2)) <= 0:
                continue
            kap = fit.kappa.get(s.car, 0.0)
            fcorr = speed_pressure_factor(s.v, kap) / float(
                speed_pressure_factor(np.array([s.v[start]]), kap)[0]
            )
            obs_c = (s.obs[:, j] + T_ZERO_C_TO_K) * fcorr - T_ZERO_C_TO_K
            a_hat = float(np.sum(T1[m] * (obs_c[m] - T0[m])) / np.sum(T1[m] ** 2))
            rows.append(
                {
                    "session_id": s.session_id,
                    "stint_id": s.stint_id,
                    "car": s.car,
                    "track": s.track,
                    "condition": s.condition,
                    "corner": corner,
                    "a_ratio": a_hat / a,
                    "n": int(m.sum()),
                }
            )
    return pd.DataFrame(rows)


def stability_report(stints: list[StintSeries], fit: CellFit) -> dict[str, Any]:
    ig = implied_gains(stints, fit)
    out: dict[str, Any] = {}
    if ig.empty:
        return out
    dry = ig[ig["condition"] == "dry"].copy()
    dry["car"] = dry["car"].str.split("/", n=1).str[0]
    # Spread of per-stint gains around the pooled value, per car (robust: IQR/median).
    per_car = {}
    for car, g in dry.groupby("car"):
        q25, q50, q75 = np.percentile(g["a_ratio"], [25, 50, 75])
        per_car[str(car)] = {"median": q50, "iqr_rel": (q75 - q25) / max(q50, 1e-9), "n": len(g)}
    out["stint_gain_spread_dry"] = per_car
    # Between-track consistency: median ratio per (car, track). With the right
    # heat input these all sit at 1.
    out["track_median_ratio_dry"] = {
        f"{car}|{trk}": float(np.median(g["a_ratio"]))
        for (car, trk), g in dry.groupby(["car", "track"])
        if len(g) >= 8
    }
    # Wet/dry gain ratio from the fitted cells, averaged over corners.
    wd: dict[str, list[float]] = {}
    for (car, corner, cond), (a, _b) in fit.ab.items():
        if cond == "dry":
            continue
        dry_ab = fit.ab.get((car, corner, "dry"))
        if dry_ab:
            wd.setdefault(f"{car}|{cond}", []).append(a / dry_ab[0])
    out["gain_ratio_vs_dry"] = {k: float(np.mean(v)) for k, v in wd.items()}
    return out


# ---------------------------------------------------------------- driver


def parse_form(spec: str, g_ref: dict[str, float]) -> HeatInput:
    """``"g2"``, ``"sliding"``, ``"sliding:glim=1.15"`` (× the car's g
    reference), ``"sliding:glim=FJ:2.2,Inferno 86:1.5"`` (absolute),
    ``"sliding:rr=0.2"``, ``"power:p=2.5"``; options separated by ``:``."""
    parts = spec.split(":")
    form = parts[0]
    kw: dict[str, Any] = {}
    i = 1
    while i < len(parts):
        key, _, val = parts[i].partition("=")
        if key == "glim":
            if "," in val or (":" in "".join(parts[i + 1 : i + 2]) and not _is_float(val)):
                # absolute per-car map: "FJ:2.2,Inferno 86:1.5" spans ':'-split parts
                raw = ":".join(parts[i:])
                _, _, maplist = raw.partition("=")
                kw["g_lim"] = {
                    k.strip(): float(v)
                    for k, v in (item.rsplit(":", 1) for item in maplist.split(","))
                }
                i = len(parts)
                continue
            mult = float(val)
            kw["g_lim"] = {car: mult * ref for car, ref in g_ref.items()}
            kw["g_lim_default"] = (
                mult * float(np.median(list(g_ref.values()))) if g_ref else float("inf")
            )
        elif key == "rr":
            kw["rr"] = float(val)
        elif key == "p":
            kw["p"] = float(val)
        elif key == "m":
            kw["m"] = float(val)
        elif key == "umax":
            kw["u_max"] = float(val)
        elif key == "clip":
            kw["g_clip"] = float(val)
        else:
            raise ValueError(f"unknown option {key!r} in {spec!r}")
        i += 1
    return HeatInput(form=form, **kw)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _folds(root: Path, n_folds: int, n_per_bucket: int, min_bucket_size: int) -> list[set[str]]:
    from .validate import _load_sessions_and_laps, _pick_holdout_sessions, _session_conditions

    sessions, laps = _load_sessions_and_laps(root)
    cond = _session_conditions(root)
    out: list[set[str]] = []
    for fold in range(n_folds):
        ids = _pick_holdout_sessions(
            sessions,
            laps,
            n_per_bucket=n_per_bucket,
            min_bucket_size=min_bucket_size,
            fold=fold,
            session_condition=cond,
        )
        if not ids:
            break
        out.append(set(ids))
    return out


def _mae(x: pd.Series) -> float:
    return float(np.mean(np.abs(x)))


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": len(df),
        "P_mae_bar": _mae(df["resid_bar"]),
        "P_bias_bar": float(df["resid_bar"].mean()),
        "T_mae_c": _mae(df["resid_c"]),
    }
    for car, g in df.groupby("car"):
        out[f"P_mae_bar[{car}]"] = _mae(g["resid_bar"])
    for cond, g in df.groupby("condition"):
        out[f"P_mae_bar[{cond}]"] = _mae(g["resid_bar"])
    return out


def run(
    root: Path,
    specs: list[str],
    *,
    n_folds: int,
    n_per_bucket: int,
    min_bucket_size: int,
    out_dir: Path | None,
    p_mode: str | None = None,
    p_exp: float | None = None,
    split_compound: bool = False,
    cars: list[str] | None = None,
    cool_mode: str | None = None,
    cool_beta: float | None = None,
    v_corr: bool = False,
    load_exp: float = 0.0,
    share: str = "none",
    long_split: bool = False,
    drive_term: bool = False,
    brake_term: bool = False,
    loto: bool = False,
    loto_min_sessions: int = 3,
    share_model: bool = False,
    share_fixed: dict[str, dict[str, float]] | None = None,
    share_tie_gc: bool = False,
    conditions_shared: bool = False,
    fixed_a: float | None = None,
    fixed_b: float | None = None,
    fit_w_road: bool | str = False,
) -> pd.DataFrame:
    t0 = time.time()
    laps_for_fit = prepare_laps_for_fit(root)
    if cars:
        laps_for_fit = laps_for_fit[laps_for_fit["car"].isin(cars)].reset_index(drop=True)
    compound_by_session: dict[str, str] = {}
    if split_compound:
        from .compounds import load_compound_labels

        lab = load_compound_labels(root)
        compound_by_session = {
            str(r.session_id): str(r.compound)
            for r in lab.itertuples()
            if isinstance(r.compound, str)
        }
    g_ref = car_g_reference(root, laps_for_fit)
    print(
        f"g reference (p{G_REF_PERCENTILE} moving total g, dry): "
        + ", ".join(f"{c} {v:.2f}" for c, v in sorted(g_ref.items()))
    )
    folds = _folds(root, n_folds, n_per_bucket, min_bucket_size)
    print(
        f"{len(folds)} folds, {sum(len(f) for f in folds)} session×fold holdouts; "
        f"prep {time.time() - t0:.0f}s"
    )
    results: list[dict[str, Any]] = []
    for spec in specs:
        heat = parse_form(spec, g_ref)
        t1 = time.time()
        stints = build_stint_series(
            root,
            laps_for_fit,
            heat,
            BUILTIN_GEOMETRY if (load_exp or long_split) else None,
            load_exp,
            long_split,
        )
        for s in stints:  # a labeled session trains its own (car/compound) cell
            comp = compound_by_session.get(s.session_id)
            if comp:
                s.car = f"{s.car}/{comp}"
        full = fit_cells(  # full-data fit first: the folds warm-start from it
            stints,
            p_mode=p_mode,
            p_exp=p_exp,
            cool_mode=cool_mode,
            cool_beta=cool_beta,
            v_corr=v_corr,
            share_a=("cars" if share in ("cars", "cars-b") else share in ("a", "ab", "a+axle")),
            share_b=(
                "cars"
                if share == "cars-b"
                else "axle" if share in ("a+axle", "cars") else share == "ab"
            ),
            drive_term=drive_term,
            brake_term=brake_term,
            share_model=share_model,
            share_fixed=share_fixed,
            share_tie_gc=share_tie_gc,
            conditions_shared=conditions_shared,
            fixed_a=fixed_a,
            fixed_b=fixed_b,
            fit_w_road=fit_w_road,
        )
        frames = []
        if loto:
            # Leave one track out: the circuit is unseen by the fit (both cars);
            # its track factor is the session-weighted mean of the fitted ones.
            by_track: dict[str, set[str]] = {}
            for s in stints:
                if s.condition == "dry":
                    by_track.setdefault(s.track, set()).add(s.session_id)
            held_tracks = sorted(t for t, ids in by_track.items() if len(ids) >= loto_min_sessions)
            folds = []
            fold_tracks: list[str] = []
            for t in held_tracks:
                folds.append({s.session_id for s in stints if s.track == t})
                fold_tracks.append(t)
        for k, held in enumerate(folds):
            train = [s for s in stints if s.session_id not in held]
            test = [s for s in stints if s.session_id in held]
            kw: dict[str, Any] = dict(
                p_mode=p_mode,
                p_exp=p_exp,
                cool_mode=cool_mode,
                cool_beta=cool_beta,
                v_corr=v_corr,
                share_a=("cars" if share in ("cars", "cars-b") else share in ("a", "ab", "a+axle")),
                share_b=(
                    "cars"
                    if share == "cars-b"
                    else "axle" if share in ("a+axle", "cars") else share == "ab"
                ),
                drive_term=drive_term,
                brake_term=brake_term,
                share_model=share_model,
                share_fixed=share_fixed,
                share_tie_gc=share_tie_gc,
                conditions_shared=conditions_shared,
                fixed_a=fixed_a,
                fixed_b=fixed_b,
                fit_w_road=fit_w_road,
            )
            fit = fit_cells(train, warm=full.x_by_name, **kw)
            df = score_stints(test, fit)
            df["fold"] = k
            df["p_exp"] = fit.p_exp
            df["cool_beta"] = fit.cool_beta
            df.attrs["cool"] = dict(fit.cool_params)
            df["kappa"] = [fit.kappa.get(c, 0.0) for c in df["car"]]
            df.attrs["drive"] = dict(fit.drive)
            df.attrs["brake"] = dict(fit.brake)
            df.attrs["share"] = dict(fit.share)
            frames.append(df)
        scored = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        stab = stability_report(stints, full)
        tag = (
            spec
            + (f" p:{p_mode}" if p_mode else "")
            + (f" cool:{cool_mode}" if cool_mode else "")
            + (" vcorr" if v_corr else "")
            + (f" load^{load_exp:g}" if load_exp else "")
            + (" longsplit" if long_split else "")
            + (" +drive" if drive_term else "")
            + (" +brake" if brake_term else "")
            + (" share-model" if share_model else "")
            + (
                " fix:" + ";".join(f"{c}:{','.join(v)}" for c, v in share_fixed.items())
                if share_fixed
                else ""
            )
            + (" tie-gc" if share_tie_gc else "")
            + (f" share:{share}" if share != "none" else "")
            + (" split" if split_compound else "")
        )
        row: dict[str, Any] = {"form": tag, "label": heat.label(next(iter(g_ref), None))}
        row["p_exp"] = round(full.p_exp, 3)
        row["cool_beta"] = round(full.cool_beta, 3)
        row["cool"] = {k: round(v, 3) for k, v in full.cool_params.items()}
        row["kappa"] = {c: f"{v:.2e}" for c, v in full.kappa.items()}
        row["drive"] = {c: round(v, 3) for c, v in full.drive.items()}
        row["share"] = {
            c: {k: round(getattr(p, k), 3) for k in SHARE_PARAM_NAMES}
            for c, p in full.share.items()
        }
        if frames and share_model:
            row["share_folds"] = {
                c: [
                    {k: round(getattr(f.attrs["share"][c], k), 2) for k in SHARE_PARAM_NAMES}
                    for f in frames
                    if c in f.attrs.get("share", {})
                ]
                for c in full.share
            }
        row["brake"] = {c: round(v, 3) for c, v in full.brake.items()}
        if frames:
            row["drive_folds"] = {
                c: [round(float(f.attrs.get("drive", {}).get(c, float("nan"))), 2) for f in frames]
                for c in full.drive
            }
        if frames:
            row["kappa_folds"] = {
                c: [
                    f"{float(f[f['car'] == c]['kappa'].iloc[0]):.2e}"
                    for f in frames
                    if (f["car"] == c).any()
                ]
                for c in full.kappa
            }
        if frames:
            row["p_exp_folds"] = [round(float(f["p_exp"].iloc[0]), 2) for f in frames if len(f)]
            row["cool_beta_folds"] = [
                round(float(f["cool_beta"].iloc[0]), 2) for f in frames if len(f)
            ]
            row["cool_folds"] = [
                {k: round(v, 3) for k, v in f.attrs.get("cool", {}).items()} for f in frames
            ]
        if not scored.empty:
            row.update(summarize(scored))
            if loto:
                row["loto"] = {
                    f"{car}|{trk}": (
                        f"MAE {g['resid_bar'].abs().mean():.4f} bias {g['resid_bar'].mean():+.4f} "
                        f"(n={len(g)}; corner bias "
                        + " ".join(
                            f"{c}{g[g['corner'] == c]['resid_bar'].mean():+.3f}" for c in CORNERS
                        )
                        + ")"
                    )
                    for (car, trk), g in scored[scored["condition"] == "dry"].groupby(
                        ["car", "track"]
                    )
                    if len(g) >= 20
                }
        row["gain_spread"] = {
            c: round(v["iqr_rel"], 3) for c, v in stab.get("stint_gain_spread_dry", {}).items()
        }
        row["track_ratio"] = {
            k: round(v, 3) for k, v in stab.get("track_median_ratio_dry", {}).items()
        }
        row["wet_dry"] = {k: round(v, 3) for k, v in stab.get("gain_ratio_vs_dry", {}).items()}
        row["tau_dry"] = {
            f"{car}|{c}": round(1.0 / b)
            for (car, c, cond), (_a, b) in full.ab.items()
            if cond == "dry"
        }
        row["a_dry"] = {
            f"{car}|{c}": round(a, 4)
            for (car, c, cond), (a, _b) in full.ab.items()
            if cond == "dry"
        }
        row["fit_s"] = round(time.time() - t1)
        results.append(row)
        _print_row(row)
        if out_dir is not None and not scored.empty:
            out_dir.mkdir(parents=True, exist_ok=True)
            safe = tag.replace(":", "_").replace("=", "-").replace(",", "_").replace(" ", "_")
            scored.to_parquet(out_dir / f"holdout_{safe}.parquet")
    return pd.DataFrame(results)


def _print_row(row: dict[str, Any]) -> None:
    print(f"\n=== {row['form']}  ({row['label']})  fit {row['fit_s']}s ===")
    if "P_mae_bar" in row:
        print(
            f"held-out lap ends: n={row['n']}  P MAE {row['P_mae_bar']:.4f} bar  "
            f"bias {row['P_bias_bar']:+.4f}  T MAE {row['T_mae_c']:.2f} C"
        )
        print(
            "  "
            + "  ".join(
                f"{k[10:-1]}: {v:.4f}" for k, v in row.items() if k.startswith("P_mae_bar[")
            )
        )
    print(f"p_exp: {row['p_exp']}  folds {row.get('p_exp_folds')}")
    print(f"cool_beta: {row['cool_beta']}  folds {row.get('cool_beta_folds')}")
    print(f"kappa per car: {row.get('kappa')}  folds {row.get('kappa_folds')}")
    print(
        f"drive coef: {row.get('drive')} folds {row.get('drive_folds')}   brake coef: {row.get('brake')}"
    )
    print(f"a dry: {row['a_dry']}")
    if row.get("share"):
        print(f"share model: {row['share']}")
        for c, folds in (row.get("share_folds") or {}).items():
            for i, f in enumerate(folds):
                print(f"  fold {i} {c}: {f}")
    if row.get("loto"):
        print("leave-one-track-out (dry, unseen circuit):")
        for k, v in row["loto"].items():
            print(f"  {k:28s} {v}")
    print(f"stint gain IQR/median (dry): {row['gain_spread']}")
    print(f"track median gain ratio (dry): {row['track_ratio']}")
    print(f"gain vs dry: {row['wet_dry']}")
    print(f"tau dry: {row['tau_dry']}")


def _parse_share_fix(specs: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for spec in specs:
        name, _, val = spec.partition("=")
        name = name.strip()
        if ":" in val:
            for item in val.split(","):
                car, _, v = item.rpartition(":")
                out.setdefault(car.strip(), {})[name] = float(v)
        else:
            out.setdefault("*", {})[name] = float(val)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--form", action="append", default=[], help="heat-input spec (repeatable)")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-per-bucket", type=int, default=2)
    ap.add_argument("--min-bucket-size", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=None, help="write per-form held-out rows here")
    ap.add_argument(
        "--p-mode",
        choices=["anchor", "instant"],
        default=None,
        help="pressure dependence of the heat input, q ∝ (P_ref/P)^n",
    )
    ap.add_argument(
        "--p-exp", type=float, default=None, help="fix the exponent n instead of fitting it"
    )
    ap.add_argument(
        "--split-compound",
        action="store_true",
        help="labeled sessions train their own (car/compound) cells",
    )
    ap.add_argument("--car", action="append", default=[], help="restrict to these (pooled) cars")
    ap.add_argument(
        "--cool-mode",
        choices=list(COOL_MODES),
        default=None,
        help=(
            "speed: b(t) = b * (1 + beta*V/V_ref)/(1 + beta); speed_axle / speed_car: one beta "
            "per axle / per car; power: b_eff = b * (dT/20K)^(n-1), nonlinear in the gap"
        ),
    )
    ap.add_argument(
        "--fit-w-road",
        nargs="?",
        const="shared",
        choices=["shared", "axle", "car"],
        default=None,
        help="fit the sink weight w in T_eff = T_air + w*(T_road - T_air) (prepped: 0.2): "
        "one shared value, one per axle, or one per car",
    )
    ap.add_argument("--cool-beta", type=float, default=None, help="fix beta instead of fitting it")
    ap.add_argument(
        "--load-exp",
        type=float,
        default=0.0,
        help="per-corner heat q_i = (W_i/W_static)^p * q from quasi-static load transfer (0 = off)",
    )
    ap.add_argument(
        "--share",
        choices=["none", "a", "ab", "a+axle", "cars", "cars-b"],
        default="none",
        help="share the gain a across corners; 'ab' shares the cooling b too, 'a+axle' lets b differ by axle",
    )
    ap.add_argument(
        "--long-split",
        action="store_true",
        help="per-corner input split by force path: r_i*(lat²+brake²) + d*acc²/r_i (driven) [+ c_b*brake*V (fronts)]",
    )
    ap.add_argument(
        "--drive-term", action="store_true", help="fit the drive-slip coefficient d per car"
    )
    ap.add_argument(
        "--brake-term", action="store_true", help="fit the brake-power coefficient c_b per car"
    )
    ap.add_argument(
        "--share-model",
        action="store_true",
        help="bounded force-share heat input (fitted p_f, g_c, brake bias, slip efficiencies)",
    )
    ap.add_argument(
        "--share-fix",
        action="append",
        default=[],
        help=(
            "fix a share parameter: 'name=value' for all cars, or 'name=CAR:value,CAR:value'; "
            "repeatable (e.g. --share-fix beta1=0 --share-fix 'beta0=FJ:0.32,Inferno 86:0.62')"
        ),
    )
    ap.add_argument(
        "--share-tie-gc", action="store_true", help="one lift acceleration for both axles"
    )
    ap.add_argument(
        "--conditions-shared",
        action="store_true",
        help="one (a, b) per car for every condition (rain uses the dry constants)",
    )
    ap.add_argument(
        "--fix-a",
        type=float,
        default=None,
        help="ablation: the gain a is this constant, not fitted",
    )
    ap.add_argument(
        "--fix-b",
        type=float,
        default=None,
        help="ablation: the cooling rate b is this constant (1/s), not fitted",
    )
    ap.add_argument(
        "--loto",
        action="store_true",
        help="leave-one-track-out instead of session folds (cross-track transfer)",
    )
    ap.add_argument(
        "--v-corr",
        action="store_true",
        help="fit a per-car speed-pressure constant: P_obs = P_gas / (1 + kappa*V^2)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    root = Path(args.dataset_root) if args.dataset_root else default_dataset_root()
    specs = args.form or ["g2", "sliding"]
    res = run(
        root,
        specs,
        n_folds=args.n_folds,
        n_per_bucket=args.n_per_bucket,
        min_bucket_size=args.min_bucket_size,
        out_dir=args.out_dir,
        p_mode=args.p_mode,
        p_exp=args.p_exp,
        split_compound=args.split_compound,
        cars=args.car or None,
        cool_mode=args.cool_mode,
        cool_beta=args.cool_beta,
        v_corr=args.v_corr,
        load_exp=args.load_exp,
        share=args.share,
        long_split=args.long_split,
        drive_term=args.drive_term,
        brake_term=args.brake_term,
        loto=args.loto,
        share_model=args.share_model,
        share_fixed=_parse_share_fix(args.share_fix) or None,
        share_tie_gc=args.share_tie_gc,
        conditions_shared=args.conditions_shared,
        fixed_a=args.fix_a,
        fixed_b=args.fix_b,
        fit_w_road=(
            False
            if args.fit_w_road is None
            else True if args.fit_w_road == "shared" else args.fit_w_road
        ),
    )
    if args.out_dir is not None:
        res.to_json(args.out_dir / "summary.json", orient="records", indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
