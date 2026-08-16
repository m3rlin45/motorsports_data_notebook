"""Per-sector pace→energy model for the target-lap-time feature.

Whole-lap (lap_time, g²) points are fragile: one bad turn on an otherwise
aggressive lap drags both numbers and skews a whole-lap fit. Instead each
lap is split into ``N_SECTORS`` distance-based sectors from the timeseries
(same ``(lat_g² + long_g²)·dt`` integrand as the lap-level ``heat_proxy``),
and the pace→energy mapping is built sector-wise:

For a curve sample at total lap time T, take the ``K_NEIGHBORS`` nearest
historical laps by total time, per sector the *median* sector time and
median sector g² (a botched sector is an outlier the median discards),
rescale sector times to sum to T, and recombine
``g²_eff(T) = Σ g²_s · t_s / T``.

The result is stored in the artifact as a small piecewise-linear
``g2_vs_lap_time`` curve per (track, car, condition) so the calculators
only ever do a linear interpolation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

N_SECTORS = 3
K_NEIGHBORS = 15
N_GRID = 9
MIN_LAPS_FOR_CURVE = 25
# Grid support: trim pace outliers so a tow-in lap doesn't stretch the curve.
GRID_QUANTILES = (0.02, 0.98)
# Pooled fallback exponent clamp (see warmup_table.G2_LAP_TIME_EXPONENT_CLAMP).
EXPONENT_CLAMP = (0.0, 6.0)

_TS_COLUMNS = ["lap_num", "t_lap_s", "speed_ms", "lat_g", "long_g"]

# One sector table per dataset root per process: the timeseries is immutable
# during a build/CV run and sector extraction is the expensive part.
_SECTOR_CACHE: dict[str, pd.DataFrame] = {}


def compute_sector_table(root: Path) -> pd.DataFrame:
    """Per-(session, lap, sector) time + g² from the raw timeseries.

    Sectors are equal thirds of the lap by *distance* (integrated speed) so
    they map to the same piece of track across laps; laps without speed
    data fall back to equal thirds by time.

    Returns columns: session_id, lap_num, lap_time_s, sector, t_s, g2_s.
    """
    key = str(root.resolve())
    cached = _SECTOR_CACHE.get(key)
    if cached is not None:
        return cached

    rows: list[tuple[str, int, float, int, float, float]] = []
    files = sorted((root / "timeseries").glob("*/*.parquet"))
    for f in files:
        session_id = f.stem
        try:
            schema_names = pq.read_schema(f).names
            cols = [c for c in _TS_COLUMNS if c in schema_names]
            if not {"lap_num", "t_lap_s", "lat_g", "long_g"}.issubset(cols):
                continue
            ts = pq.read_table(f, columns=cols).to_pandas()
        except Exception:  # noqa: BLE001 — one unreadable file shouldn't kill the fit
            logger.warning("sector extraction: failed to read %s — skipping", f)
            continue

        for lap_num, lap in ts.groupby("lap_num"):
            t = lap["t_lap_s"].to_numpy(dtype=float)
            if t.size < N_SECTORS * 2 or np.all(np.isnan(t)):
                continue
            order = np.argsort(t)
            t = t[order]
            lat = np.nan_to_num(lap["lat_g"].to_numpy(dtype=float)[order], nan=0.0)
            lng = np.nan_to_num(lap["long_g"].to_numpy(dtype=float)[order], nan=0.0)
            dt = np.diff(t, prepend=t[0])
            dt = np.where(np.isnan(dt) | (dt < 0), 0.0, dt)
            lap_time_s = float(t[-1] - t[0])
            if lap_time_s <= 0:
                continue

            if "speed_ms" in lap.columns:
                speed = np.nan_to_num(lap["speed_ms"].to_numpy(dtype=float)[order], nan=0.0)
                progress = np.cumsum(speed * dt)
            else:
                progress = t - t[0]
            total = float(progress[-1])
            if total <= 0:
                progress = t - t[0]
                total = lap_time_s
            sector = np.minimum((progress / total * N_SECTORS).astype(int), N_SECTORS - 1)

            g2 = lat * lat + lng * lng
            for s in range(N_SECTORS):
                mask = sector == s
                t_s = float(np.sum(dt[mask]))
                if t_s <= 0:
                    break
                heat_s = float(np.sum(g2[mask] * dt[mask]))
                rows.append((session_id, int(lap_num), lap_time_s, s, t_s, heat_s / t_s))

    table = pd.DataFrame(
        rows, columns=["session_id", "lap_num", "lap_time_s", "sector", "t_s", "g2_s"]
    )
    _SECTOR_CACHE[key] = table
    return table


def _curve_for_bucket(bucket: pd.DataFrame) -> dict | None:
    """kNN-median g² curve for one (track, car, condition) sector frame."""
    per_lap = bucket.groupby(["session_id", "lap_num"])["lap_time_s"].first().reset_index()
    n_laps = len(per_lap)
    if n_laps < MIN_LAPS_FOR_CURVE:
        return None

    lap_times = per_lap["lap_time_s"].to_numpy(dtype=float)
    lo = float(np.quantile(lap_times, GRID_QUANTILES[0]))
    hi = float(np.quantile(lap_times, GRID_QUANTILES[1]))
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, N_GRID)
    k = min(K_NEIGHBORS, n_laps)

    # Wide-format sector arrays aligned with per_lap rows
    sector_t = np.full((n_laps, N_SECTORS), np.nan)
    sector_g2 = np.full((n_laps, N_SECTORS), np.nan)
    idx_of = {
        (sid, ln): i for i, (sid, ln) in enumerate(zip(per_lap["session_id"], per_lap["lap_num"]))
    }
    for sid, ln, sec, t_s, g2_s in zip(
        bucket["session_id"], bucket["lap_num"], bucket["sector"], bucket["t_s"], bucket["g2_s"]
    ):
        i = idx_of[(sid, ln)]
        sector_t[i, int(sec)] = t_s
        sector_g2[i, int(sec)] = g2_s
    complete = ~np.isnan(sector_t).any(axis=1) & ~np.isnan(sector_g2).any(axis=1)
    if complete.sum() < MIN_LAPS_FOR_CURVE:
        return None
    lap_times = lap_times[complete]
    sector_t = sector_t[complete]
    sector_g2 = sector_g2[complete]
    k = min(k, len(lap_times))

    g2_curve: list[float] = []
    for target in grid:
        nearest = np.argsort(np.abs(lap_times - target))[:k]
        med_t = np.median(sector_t[nearest], axis=0)
        med_g2 = np.median(sector_g2[nearest], axis=0)
        med_t = med_t * (target / float(med_t.sum()))  # rescale to sum to target
        g2_curve.append(float(np.sum(med_g2 * med_t) / target))

    return {
        "lap_time_s": [round(float(x), 3) for x in grid],
        "g2": [round(float(y), 5) for y in g2_curve],
        "n_laps": int(complete.sum()),
    }


def build_pace_model(
    root: Path, laps: pd.DataFrame
) -> tuple[dict[tuple[str, str, str], dict], float]:
    """Build per-bucket g²-vs-lap-time curves + the pooled fallback exponent.

    ``laps`` is the prepped training frame (already condition-classified and
    filtered/excluded); only its (session_id, lap_num) laps contribute, so
    held-out CV folds never leak into the curves.

    Returns ``({(track, car, cond): curve_dict}, default_exponent)``.
    """
    sector_tbl = compute_sector_table(root)
    if sector_tbl.empty:
        return {}, 3.0

    meta_cols = ["session_id", "lap_num", "track_canonical", "car", "condition"]
    meta = laps[meta_cols].drop_duplicates()
    merged = sector_tbl.merge(meta, on=["session_id", "lap_num"], how="inner")
    merged = merged[merged["condition"] != "unknown"]
    if merged.empty:
        return {}, 3.0

    curves: dict[tuple[str, str, str], dict] = {}
    pooled_x: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    for (track, car, cond), bucket in merged.groupby(["track_canonical", "car", "condition"]):
        # Pooled fallback exponent from per-sector points, centered per
        # (bucket, sector) so it estimates a common within-sector slope.
        for _s, sec in bucket.groupby("sector"):
            ok = (sec["t_s"] > 0) & (sec["g2_s"] > 0)
            if ok.sum() < 8:
                continue
            log_t = np.log(sec.loc[ok, "t_s"].to_numpy(dtype=float))
            log_g2 = np.log(sec.loc[ok, "g2_s"].to_numpy(dtype=float))
            pooled_x.append(log_t - float(np.median(log_t)))
            pooled_y.append(log_g2 - float(np.median(log_g2)))

        curve = _curve_for_bucket(bucket)
        if curve is not None:
            curves[(str(track), str(car), str(cond))] = curve

    default_exponent = 3.0
    if pooled_x:
        x = np.concatenate(pooled_x)
        y = np.concatenate(pooled_y)
        if len(x) >= 50 and float(x.max() - x.min()) > 0:
            default_exponent = float(
                np.clip(-np.polyfit(x, y, 1)[0], EXPONENT_CLAMP[0], EXPONENT_CLAMP[1])
            )

    return curves, default_exponent
