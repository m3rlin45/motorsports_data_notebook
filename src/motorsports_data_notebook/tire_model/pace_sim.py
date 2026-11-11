"""Pace model: a faster (or slower) lap simulated per second through the
production heat equations.

The calculator's target lap time asks a counterfactual — *the same car on
the same circuit, lapping at a different pace*. This module answers it from
the logged laps themselves instead of a fitted pace→heat curve:

1. Take the bucket's reference laps (the ``N_REFERENCE_LAPS`` training laps
   nearest its typical lap time) as logged, sample by sample: speed ``V``,
   lateral g, longitudinal g, duration ``dt``.
2. Scale the whole lap uniformly to the target lap time with the speed
   multiplier ``m = T_lap / T_target``: ``V' = m·V``, ``lat' = m²·lat`` (same
   line, same corner radius, so lateral acceleration goes with speed
   squared), longitudinal g unchanged (braking and traction already at
   their limits), ``dt' = dt / m`` (the same piece of track at ``m`` times
   the speed). The lap time is then exactly ``T_lap / m``.
3. Push the rescaled samples through the schema-v5 per-corner heat
   equations (:func:`heat_input.share_heat_rate`: the car's fitted force
   shares, ``q_i = V·|F_i|``) and take each corner's lap-mean heat input
   relative to the unscaled lap.

The median over the reference laps of those per-corner ratios, on a grid of
lap times around the typical lap, is stored in the artifact as the bucket's
``g2_vs_lap_time`` curve (``g2`` = corner mean for consumers that predate
per-corner curves, ``g2_by_corner`` per corner). Nothing in it is fitted:
the only inputs are the logged laps and the heat model's own force shares.

Why uniform scaling: checked within sessions (fastest clean laps against the
session's median laps, observed per-corner heat ratio from the per-second
heat model; see :func:`fast_lap_check`), the uniform transform reproduces
the real fast laps' heat with no bias on all four corners. Scaling only the
corner-limited samples, or a dynamically consistent point-mass lap with
braking and traction held at the lap's limits, over-predicts the heat by
1–3 %: a median lap is not at those limits, and a faster lap gains speed
everywhere. A fitted lap-time exponent of 4 (the between-session optimum)
over-predicts by 4 %, so it was absorbing day effects, not pace.

Buckets without usable timeseries fall back to the physics exponent
``DEFAULT_EXPONENT`` (``q = |F|·V ∝ V³``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .heat_input import DEFAULT_G_CLIP, ShareParams, share_heat_rate

logger = logging.getLogger(__name__)

CORNERS = ("fl", "fr", "rl", "rr")

#: Lap-time grid of the stored curve, as multiples of the bucket's typical lap.
GRID_FACTORS: tuple[float, ...] = tuple(float(x) for x in np.linspace(0.75, 1.5, 9))
#: Training laps nearest the typical lap time that are simulated per bucket.
N_REFERENCE_LAPS = 15
#: Fewer simulated laps than this and the bucket gets no curve (exponent fallback).
MIN_REFERENCE_LAPS = 5
#: Physics fallback for a bucket with no curve: heat ∝ V·|F| with F ∝ V².
DEFAULT_EXPONENT = 3.0
METHOD = "uniform_scaled_lap_shares"
#: Samples slower than this (pit lane, stops) carry no usable curvature.
MIN_SPEED_MS = 3.0

_TS_COLUMNS = ["lap_num", "t_lap_s", "speed_ms", "lat_g", "long_g"]


@dataclass(frozen=True)
class LapSamples:
    """One logged lap, sample by sample (time-ordered)."""

    dt_s: np.ndarray
    speed_ms: np.ndarray
    lat_g: np.ndarray
    long_g: np.ndarray

    @property
    def lap_time_s(self) -> float:
        return float(self.dt_s.sum())


# One sample table per dataset root per process: the timeseries is immutable
# during a build / CV run and reading it is the expensive part.
_SAMPLE_CACHE: dict[str, dict[tuple[str, int], LapSamples]] = {}


def load_lap_samples(root: Path) -> dict[tuple[str, int], LapSamples]:
    """Per-(session_id, lap_num) samples of every session's validated
    kinematics (:func:`~..tire_etl.kinematics.load_kinematics`: speed and
    the chosen lateral / longitudinal g on the sensor's own clock), cached."""
    from ..tire_etl.kinematics import load_kinematics

    key = str(Path(root).resolve())
    cached = _SAMPLE_CACHE.get(key)
    if cached is not None:
        return cached
    out: dict[tuple[str, int], LapSamples] = {}
    for f in sorted((Path(root) / "timeseries").glob("*/*.parquet")):
        try:
            kin = load_kinematics(Path(root), f.stem)
            if kin is None or not set(_TS_COLUMNS).issubset(kin.schema.names):
                continue
            ts = kin.select(_TS_COLUMNS).to_pandas()
        except Exception:  # noqa: BLE001 — one unreadable file shouldn't kill the build
            logger.warning("pace model: failed to read %s — skipping", f)
            continue
        for lap_num, lap in ts.groupby("lap_num"):
            samples = _lap_samples(lap)
            if samples is not None:
                out[(f.stem, int(lap_num))] = samples
    _SAMPLE_CACHE[key] = out
    return out


def _lap_samples(lap: pd.DataFrame) -> LapSamples | None:
    t = lap["t_lap_s"].to_numpy(dtype=float)
    order = np.argsort(t)
    t = t[order]
    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isnan(dt) | (dt < 0), 0.0, dt)
    v = np.maximum(np.nan_to_num(lap["speed_ms"].to_numpy(dtype=float)[order], nan=0.0), 0.0)
    lat = np.clip(
        np.nan_to_num(lap["lat_g"].to_numpy(dtype=float)[order]), -DEFAULT_G_CLIP, DEFAULT_G_CLIP
    )
    lng = np.clip(
        np.nan_to_num(lap["long_g"].to_numpy(dtype=float)[order]), -DEFAULT_G_CLIP, DEFAULT_G_CLIP
    )
    keep = (dt > 0) & (v >= MIN_SPEED_MS)
    if keep.sum() < 20 or dt[keep].sum() <= 0:
        return None
    return LapSamples(dt[keep], v[keep], lat[keep], lng[keep])


def scaled_lap(samples: LapSamples, m: float) -> LapSamples:
    """The lap driven ``m`` times faster everywhere (uniform scaling)."""
    if m <= 0:
        raise ValueError(f"speed multiplier must be > 0; got {m}")
    return LapSamples(samples.dt_s / m, samples.speed_ms * m, samples.lat_g * m * m, samples.long_g)


def corner_heat_means(share: ShareParams, samples: LapSamples) -> np.ndarray:
    """Lap-mean per-corner driving intensity ``q_i`` (G·m/s) under the share
    model, in :data:`CORNERS` order."""
    lat_pos = np.maximum(samples.lat_g, 0.0)
    lat_neg = np.maximum(-samples.lat_g, 0.0)
    long_pos = np.maximum(samples.long_g, 0.0)
    long_neg = np.maximum(-samples.long_g, 0.0)
    total = samples.lap_time_s
    return np.array(
        [
            float(
                np.sum(
                    share_heat_rate(
                        share, lat_pos, lat_neg, long_pos, long_neg, samples.speed_ms, c
                    )
                    * samples.dt_s
                )
                / total
            )
            for c in CORNERS
        ]
    )


def lap_heat_ratios(
    share: ShareParams, samples: LapSamples, lap_times_s: Sequence[float]
) -> np.ndarray | None:
    """Per-corner heat-input ratio ``q_i(target) / q_i(lap)`` for each target
    lap time, shape ``(len(lap_times_s), 4)``; ``None`` when the lap carries
    no heat (all-zero accelerations)."""
    base = corner_heat_means(share, samples)
    if np.any(base <= 0) or not np.all(np.isfinite(base)):
        return None
    t0 = samples.lap_time_s
    rows = []
    for target in lap_times_s:
        if target <= 0:
            raise ValueError(f"target lap time must be > 0; got {target}")
        rows.append(corner_heat_means(share, scaled_lap(samples, t0 / float(target))) / base)
    return np.array(rows)


def build_pace_model(
    root: Path,
    laps: pd.DataFrame,
    share_by_car: dict[str, ShareParams],
    *,
    n_reference: int = N_REFERENCE_LAPS,
    grid_factors: Sequence[float] = GRID_FACTORS,
) -> dict[tuple[str, str, str], dict]:
    """Per-(track, car, condition) ``g2_vs_lap_time`` curves from simulated laps.

    ``laps`` is the prepped training frame of flying laps (``session_id``,
    ``lap_num``, ``track_canonical``, ``car``, ``condition``, ``on_track_s``);
    only its laps are simulated, so held-out CV folds never leak in.
    ``share_by_car`` are the fitted force shares (:class:`PhysicalFit.share`).
    A bucket with fewer than :data:`MIN_REFERENCE_LAPS` simulable laps, or a
    car without shares, gets no curve.
    """
    needed = {"session_id", "lap_num", "track_canonical", "car", "condition", "on_track_s"}
    if laps.empty or not needed.issubset(laps.columns):
        return {}
    samples_by_lap = load_lap_samples(root)
    if not samples_by_lap:
        return {}
    meta = laps[sorted(needed)].drop_duplicates()
    meta = meta[(meta["condition"] != "unknown") & meta["on_track_s"].notna()]
    curves: dict[tuple[str, str, str], dict] = {}
    for (track, car, cond), bucket in meta.groupby(["track_canonical", "car", "condition"]):
        share = share_by_car.get(str(car))
        if share is None:
            continue
        typ = float(bucket["on_track_s"].median())
        if not typ > 0:
            continue
        nearest = bucket.assign(_d=(bucket["on_track_s"] - typ).abs()).nsmallest(n_reference, "_d")
        grid = [typ * float(f) for f in grid_factors]
        ratios = []
        for sid, lap_num in zip(nearest["session_id"], nearest["lap_num"]):
            samples = samples_by_lap.get((str(sid), int(lap_num)))
            if samples is None:
                continue
            r = lap_heat_ratios(share, samples, grid)
            if r is not None and np.all(np.isfinite(r)):
                ratios.append(r)
        if len(ratios) < MIN_REFERENCE_LAPS:
            continue
        med = np.median(np.array(ratios), axis=0)  # (n_grid, 4)
        curves[(str(track), str(car), str(cond))] = {
            "lap_time_s": [round(float(x), 3) for x in grid],
            "g2": [round(float(y), 5) for y in med.mean(axis=1)],
            "g2_by_corner": {
                c: [round(float(y), 5) for y in med[:, j]] for j, c in enumerate(CORNERS)
            },
            "n_laps": len(ratios),
            "method": METHOD,
        }
    return curves


# ------------------------------------------------------------ within-session check


def fast_lap_check(
    laps: pd.DataFrame,
    samples_by_lap: dict[tuple[str, int], LapSamples],
    share_by_car: dict[str, ShareParams],
    *,
    faster_than: float = 0.985,
    not_faster_than: float = 0.94,
    reference_band: float = 0.02,
    min_laps: int = 8,
) -> pd.DataFrame:
    """Does the simulation reproduce how real fast laps heat the tyres?

    Per session with ``min_laps`` flying laps: reference laps within
    ``reference_band`` of the session's median lap time, fast laps between
    ``faster_than`` and ``not_faster_than`` times the median. Observed
    per-corner ratio = mean ``q_lap_c`` (fast) / mean ``q_lap_c``
    (reference), from the per-second heat model's own per-lap intensities;
    predicted = the simulation from each reference lap to the fast laps'
    mean time (median over reference laps). Returns one row per session with
    ``log_obs_{c}`` and ``log_pred_{c}``; an unbiased simulation has
    ``mean(log_obs - log_pred) ≈ 0`` and slope ≈ 1 on every corner.
    """
    q_cols = [f"q_lap_{c}" for c in CORNERS]
    needed = {"session_id", "lap_num", "car", "on_track_s", *q_cols}
    if laps.empty or not needed.issubset(laps.columns):
        return pd.DataFrame()
    rows = []
    for sid, s in laps.dropna(subset=q_cols).groupby("session_id"):
        if len(s) < min_laps:
            continue
        share = share_by_car.get(str(s["car"].iloc[0]))
        if share is None:
            continue
        t_med = float(s["on_track_s"].median())
        ref = s[(s["on_track_s"] - t_med).abs() <= reference_band * t_med]
        fast = s[
            (s["on_track_s"] < faster_than * t_med) & (s["on_track_s"] > not_faster_than * t_med)
        ]
        if len(ref) < 3 or len(fast) < 2:
            continue
        t_fast = float(fast["on_track_s"].mean())
        obs = np.array([fast[c].mean() / ref[c].mean() for c in q_cols])
        preds = []
        for lap_sid, lap_num in zip(ref["session_id"], ref["lap_num"]):
            samples = samples_by_lap.get((str(lap_sid), int(lap_num)))
            if samples is None:
                continue
            r = lap_heat_ratios(share, samples, [t_fast])
            if r is not None and np.all(np.isfinite(r)):
                preds.append(r[0])
        if len(preds) < 2 or np.any(obs <= 0):
            continue
        pred = np.median(np.array(preds), axis=0)
        row: dict[str, object] = {
            "session_id": sid,
            "car": s["car"].iloc[0],
            "track_canonical": s["track_canonical"].iloc[0] if "track_canonical" in s else None,
            "lap_time_median_s": t_med,
            "lap_time_fast_s": t_fast,
            "n_reference": len(preds),
            "n_fast": len(fast),
        }
        for j, c in enumerate(CORNERS):
            row[f"log_obs_{c}"] = float(np.log(obs[j]))
            row[f"log_pred_{c}"] = float(np.log(pred[j]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_fast_lap_check(frame: pd.DataFrame) -> pd.DataFrame:
    """Bias, MAE and slope (observed on predicted, log ratios) per corner."""
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame(rows)
    for c in CORNERS:
        o = frame[f"log_obs_{c}"].to_numpy(dtype=float)
        p = frame[f"log_pred_{c}"].to_numpy(dtype=float)
        slope = float(np.polyfit(p, o, 1)[0]) if len(o) >= 3 and np.ptp(p) > 1e-9 else float("nan")
        rows.append(
            {
                "corner": c,
                "n_sessions": len(o),
                "bias_log": float(np.mean(o - p)),
                "mae_log": float(np.mean(np.abs(o - p))),
                "slope_obs_on_pred": slope,
                "mean_log_obs": float(np.mean(o)),
            }
        )
    return pd.DataFrame(rows)
