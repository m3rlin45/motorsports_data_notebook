"""Build the tire warmup model from the committed dataset.

Reads ``data/tire_dataset/{laps,sessions,weather_hourly}/*.parquet``, fits the
energy-balance parameters in two passes (see the plan / Model section), and
writes two artifacts side-by-side:

- ``data/tire_dataset/warmup_table.parquet`` — fast-load Python format.
- ``data/tire_dataset/tire_model.json`` — schema-versioned JSON for the
  predictor + future C# integration.

The Python predictor reads either; the JSON is the canonical hand-off.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.optimize  # type: ignore[import-untyped]

from ..tire_etl.paths import (
    default_dataset_root,
    laps_dir,
    sessions_dir,
    weather_dir,
)
from .energy_balance import t_effective_c, t_road_proxy_c

logger = logging.getLogger(__name__)

CORNERS = ("fl", "fr", "rl", "rr")
SCHEMA_VERSION = 1

# Anchored track for the c_track identifiability constraint. Tsukuba has the
# most coverage across both cars in the current dataset.
ANCHOR_TRACK = "tsukuba_2000"

# Energy-balance config (see plan)
W_ROAD = 0.2  # fixed in v0
DELTA_SUN_MAX_C = 10.0
SUN_FACTOR_DEFAULT = 1.0

# Physical priors used when a (car, corner) bucket lacks enough data to fit
PRIOR_TAU_SEC = 240.0
PRIOR_K_KELVIN_PER_G2 = 60.0
PRIOR_C_TRACK = 1.0

# Bucket-size thresholds
MIN_LAPS_FOR_TAU_FIT = 30  # per (car, track, corner) bucket to participate in Pass 1
MIN_LAPS_FOR_K_BUCKET = 5  # per (car, track, corner) bucket to factor in Pass 2

# Per-(session, corner) sensor sanity check: flag stuck/broken TPMS channels so
# the fit doesn't learn from them. Pure heuristic — easy to tune later.
BROKEN_CORNER_STD_THRESHOLD_C = 1.0  # std(temp) across session's tire-usable laps
BROKEN_CORNER_MIN_LAPS = 4  # need at least this many laps to call it "stuck"


# ---------- Public entry point ----------


def build_warmup_table(
    dataset_root: Path | None = None,
    *,
    rebuild: bool = False,
    exclude_session_ids: set[str] | None = None,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Fit the energy-balance model and write both artifacts.

    Parameters
    ----------
    exclude_session_ids
        If given, drop these session_ids from training. Used by held-out
        validation to avoid evaluating the model on its own training data.
    write_artifacts
        If False (used by held-out validation), skip writing
        ``tire_model.json`` and ``warmup_table.parquet`` — return the in-memory
        model dict only.

    Returns the fitted model as a dict (matching the JSON schema) so callers
    can inspect without re-reading the file.
    """
    root = Path(dataset_root) if dataset_root else default_dataset_root()

    laps = _load_filtered_laps(root)
    if exclude_session_ids:
        before = len(laps)
        laps = laps[~laps["session_id"].isin(exclude_session_ids)].reset_index(drop=True)
        logger.info(
            "Excluded %d held-out sessions (%d → %d laps)",
            len(exclude_session_ids),
            before,
            len(laps),
        )
    weather = _load_weather(root)
    laps = _attach_weather(laps, weather)
    laps = _compute_stint_clock(laps)
    blacklist_pairs = load_sensor_blacklist(root)
    laps, blacklist_applied = _apply_blacklist(laps, blacklist_pairs)
    laps = _compute_delta_t(laps)

    lap_time_lookup = _build_lap_time_typ(laps)
    g2_lookup = _build_g2_typ(laps)

    laps_for_fit = _laps_for_fit(laps, g2_lookup)

    tau_by_car_corner: dict[tuple[str, str], FitParam] = {}
    bucket_gains: dict[tuple[str, str, str], FitParam] = {}  # (car, track, corner) -> gain
    bucket_n_samples: dict[tuple[str, str, str], int] = {}

    for car in sorted(set(laps_for_fit["car"])):
        for corner in CORNERS:
            tau, per_bucket_gain = _pass1_fit_tau_and_gains(laps_for_fit, car, corner)
            tau_by_car_corner[(car, corner)] = tau
            for track, gain in per_bucket_gain.items():
                bucket_gains[(car, track, corner)] = gain
                bucket_n_samples[(car, track, corner)] = _bucket_sample_count(
                    laps_for_fit, car, track, corner
                )

    k_by_car_corner, c_track_by_track = _pass2_factor_gains(
        bucket_gains=bucket_gains,
        g2_lookup=g2_lookup,
        anchor_track=ANCHOR_TRACK,
    )

    model = _assemble_model(
        tau_by_car_corner=tau_by_car_corner,
        k_by_car_corner=k_by_car_corner,
        c_track_by_track=c_track_by_track,
        g2_lookup=g2_lookup,
        lap_time_lookup=lap_time_lookup,
        bucket_n_samples=bucket_n_samples,
        blacklist_applied=blacklist_applied,
    )

    if write_artifacts:
        _write_warmup_table_parquet(root, model)
        _write_tire_model_json(root, model)

    return model


# ---------- Data prep ----------


def _load_filtered_laps(root: Path) -> pd.DataFrame:
    """Read laps + sessions, filter to ok + has_tpms + tire_usable."""
    laps_files = sorted((laps_dir(root)).glob("*.parquet"))
    sessions_files = sorted((sessions_dir(root)).glob("*.parquet"))
    if not laps_files or not sessions_files:
        raise FileNotFoundError(
            f"No laps/sessions parquets under {root}. Run `just tire-refresh` first."
        )
    laps = pa.concat_tables([pq.read_table(f) for f in laps_files]).to_pandas()
    sessions = pa.concat_tables([pq.read_table(f) for f in sessions_files]).to_pandas()

    sessions_keep = sessions[(sessions["status"] == "ok") & sessions["has_tpms"]]
    cols_from_sessions = ["session_id", "track_canonical", "car", "session_start_utc", "date"]
    df = laps.merge(sessions_keep[cols_from_sessions], on="session_id", how="inner")
    df = df[df["tire_usable"]].reset_index(drop=True)
    df = df[df["track_canonical"].notna()]
    df = df[df["car"].notna()]
    out: pd.DataFrame = df.reset_index(drop=True)
    return out


def _load_weather(root: Path) -> pd.DataFrame:
    """Load all weather parquets into a single DataFrame keyed by (track, ts_utc)."""
    rows: list[pd.DataFrame] = []
    wx_root = weather_dir(root)
    if not wx_root.exists():
        logger.warning("No weather directory at %s — predictions will use T_air fallback", wx_root)
        return pd.DataFrame(columns=["track_canonical", "ts_utc", "temperature_2m", "cloud_cover"])
    for track_dir in sorted(wx_root.iterdir()):
        if not track_dir.is_dir():
            continue
        for f in sorted(track_dir.glob("*.parquet")):
            wx = pq.read_table(f).to_pandas()
            wx["track_canonical"] = track_dir.name
            rows.append(wx[["track_canonical", "ts_utc", "temperature_2m", "cloud_cover"]])
    if not rows:
        return pd.DataFrame(columns=["track_canonical", "ts_utc", "temperature_2m", "cloud_cover"])
    return pd.concat(rows, ignore_index=True)


def _attach_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join hourly weather onto laps via floor-to-hour on session_start_utc."""
    if weather.empty:
        laps["t_air_c"] = np.nan
        laps["cloud_cover"] = np.nan
        return laps
    laps = laps.copy()
    starts = pd.to_datetime(laps["session_start_utc"], utc=True)
    laps["_hour_key"] = starts.dt.strftime("%Y-%m-%dT%H:00")
    out = laps.merge(
        weather.rename(columns={"temperature_2m": "t_air_c", "ts_utc": "_hour_key"})[
            ["track_canonical", "_hour_key", "t_air_c", "cloud_cover"]
        ],
        on=["track_canonical", "_hour_key"],
        how="left",
    )
    out = out.drop(columns=["_hour_key"])
    # Fill T_air with historical-median by track when weather is missing.
    median_t_air = out.groupby("track_canonical")["t_air_c"].transform("median")
    out["t_air_c"] = out["t_air_c"].fillna(median_t_air)
    return out


def _compute_stint_clock(laps: pd.DataFrame) -> pd.DataFrame:
    """Add ``lap_within_stint`` (cumcount within (session_id, stint_id)) and
    ``t_cum_s`` (on_track_s cumsum within stint, at end of each lap)."""
    df = laps.sort_values(["session_id", "stint_id", "lap_num"]).copy()
    grouped = df.groupby(["session_id", "stint_id"], sort=False)
    df["lap_within_stint"] = grouped.cumcount()
    df["t_cum_s"] = grouped["on_track_s"].cumsum()
    return df.reset_index(drop=True)


def detect_suspect_corners(laps: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of (session, corner) pairs whose TPMS temperature
    looks stuck/broken (std below threshold over ≥ N usable laps).

    Pure detection — no laps are modified. Use the output as a candidate
    list for human review; confirmed entries go into ``sensor_blacklist.yaml``.

    Columns: session_id, car, track_canonical, corner, n_laps, std_c, min_c,
    max_c, mean_c, sample_values (first 5).
    """
    if laps.empty:
        return pd.DataFrame()
    candidates: list[dict] = []
    meta_cols = ["session_id", "car", "track_canonical", "date"]
    for c in CORNERS:
        col = f"tpms_temp_{c}_end"
        stats = (
            laps.groupby("session_id")[col]
            .agg(["count", "std", "min", "max", "mean"])
            .rename(
                columns={
                    "count": "n_laps",
                    "std": "std_c",
                    "min": "min_c",
                    "max": "max_c",
                    "mean": "mean_c",
                }
            )
        )
        suspect = stats[
            (stats["n_laps"] >= BROKEN_CORNER_MIN_LAPS)
            & (stats["std_c"].fillna(0.0) < BROKEN_CORNER_STD_THRESHOLD_C)
        ]
        if suspect.empty:
            continue
        meta = (
            laps[laps["session_id"].isin(suspect.index)][meta_cols]
            .drop_duplicates("session_id")
            .set_index("session_id")
        )
        meta_dict: dict[Any, Any] = meta.to_dict(orient="index")
        for sid, row in suspect.iterrows():
            values = laps.loc[laps["session_id"] == sid, col].dropna().tolist()
            meta_row = meta_dict[sid]
            candidates.append(
                {
                    "session_id": str(sid),
                    "car": str(meta_row["car"]),
                    "track_canonical": str(meta_row["track_canonical"]),
                    "date": str(meta_row["date"]),
                    "corner": c,
                    "n_laps": int(row["n_laps"]),
                    "std_c": float(row["std_c"] or 0.0),
                    "min_c": float(row["min_c"] or 0.0),
                    "max_c": float(row["max_c"] or 0.0),
                    "mean_c": float(row["mean_c"] or 0.0),
                    "first_5_values": values[:5],
                }
            )
    return pd.DataFrame(candidates)


def load_sensor_blacklist(dataset_root: Path) -> set[tuple[str, str]]:
    """Read ``sensor_blacklist.yaml`` and return the set of
    (session_id, corner) pairs to exclude from training.

    Missing file is treated as an empty blacklist (no exclusions).
    """
    path = dataset_root / "sensor_blacklist.yaml"
    if not path.exists():
        return set()
    import yaml  # local import to keep top-of-module deps minimal

    data = yaml.safe_load(path.read_text())
    entries = (data or {}).get("entries", []) or []
    pairs: set[tuple[str, str]] = set()
    for e in entries:
        sid = e.get("session_id")
        corner = e.get("corner")
        if sid and corner in CORNERS:
            pairs.add((str(sid), str(corner)))
    return pairs


def _apply_blacklist(
    laps: pd.DataFrame,
    blacklist: set[tuple[str, str]],
    *,
    warn_on_unknown: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """NaN out tpms_temp_{c}_end for each (session_id, corner) in the blacklist.

    Returns ``(laps_with_nans, applied_records)`` where ``applied_records``
    is what was actually masked (for the artifact's audit trail).

    ``warn_on_unknown=False`` silences the per-entry "unknown session_id"
    warning — used when the caller has intentionally filtered the laps to
    a subset (e.g., the held-out validation evaluates only a few sessions).
    """
    if not blacklist:
        return laps, []
    df = laps.copy()
    applied: list[dict] = []
    for sid, corner in blacklist:
        col = f"tpms_temp_{corner}_end"
        mask = df["session_id"] == sid
        if not mask.any():
            if warn_on_unknown:
                logger.warning("blacklist entry references unknown session_id %s — skipping", sid)
            continue
        applied.append(
            {
                "session_id": sid,
                "corner": corner,
                "n_laps_masked": int(mask.sum()),
            }
        )
        df.loc[mask, col] = np.nan
    if applied:
        logger.info(
            "Applied user-confirmed sensor blacklist: %d (session, corner) channels masked",
            len(applied),
        )
    return df, applied


def _compute_delta_t(laps: pd.DataFrame) -> pd.DataFrame:
    """Compute T_road proxy and δT per corner = tpms_temp_{c}_end − T_eff."""
    df = laps.copy()
    df["t_road_c"] = [
        (
            t_road_proxy_c(
                t_air_c=a,
                cloud_cover_pct=c,
                sun_factor=SUN_FACTOR_DEFAULT,
                delta_sun_max_c=DELTA_SUN_MAX_C,
            )
            if pd.notna(a)
            else np.nan
        )
        for a, c in zip(df["t_air_c"], df["cloud_cover"])
    ]
    df["t_eff_c"] = [
        (
            t_effective_c(t_air_c=a, t_road_c=r, w_road=W_ROAD)
            if pd.notna(a) and pd.notna(r)
            else np.nan
        )
        for a, r in zip(df["t_air_c"], df["t_road_c"])
    ]
    for c in CORNERS:
        col = f"tpms_temp_{c}_end"
        df[f"delta_t_{c}"] = df[col] - df["t_eff_c"]
    return df


def _build_lap_time_typ(laps: pd.DataFrame) -> dict[tuple[str, str], tuple[float, int]]:
    """Median on_track_s per (track, car). Returns {(track, car): (median_s, n)}."""
    out: dict[tuple[str, str], tuple[float, int]] = {}
    for (track, car), grp in laps.groupby(["track_canonical", "car"]):
        out[(str(track), str(car))] = (float(grp["on_track_s"].median()), int(len(grp)))
    return out


def _build_g2_typ(laps: pd.DataFrame) -> dict[tuple[str, str], tuple[float, int]]:
    """Median heat_proxy/on_track_s per (track, car). Returns {(track, car): (g2, n)}."""
    out: dict[tuple[str, str], tuple[float, int]] = {}
    for (track, car), grp in laps.groupby(["track_canonical", "car"]):
        g2_per_lap = grp["heat_proxy"] / grp["on_track_s"]
        g2_per_lap = g2_per_lap.replace([np.inf, -np.inf], np.nan).dropna()
        if g2_per_lap.empty:
            continue
        out[(str(track), str(car))] = (float(g2_per_lap.median()), int(len(g2_per_lap)))
    return out


def _laps_for_fit(
    laps: pd.DataFrame, g2_lookup: dict[tuple[str, str], tuple[float, int]]
) -> pd.DataFrame:
    """Drop out-laps and rows without a valid t_eff or g²."""
    df = laps[laps["lap_within_stint"] > 0].copy()
    df = df[df["t_eff_c"].notna()]
    df = df[df["t_cum_s"] > 0]
    # Attach the bucket's g2_typ (used in Pass 2)
    df["g2_typ"] = [
        g2_lookup.get((t, c), (np.nan, 0))[0] for t, c in zip(df["track_canonical"], df["car"])
    ]
    df = df[df["g2_typ"].notna()]
    return df.reset_index(drop=True)


# ---------- Pass 1: fit τ_sec[car, corner] + per-bucket gains ----------


@dataclass(frozen=True)
class FitParam:
    value: float
    stderr: float
    n_samples: int
    from_prior: bool = False


def _bucket_sample_count(laps_for_fit: pd.DataFrame, car: str, track: str, corner: str) -> int:
    col = f"delta_t_{corner}"
    mask = (laps_for_fit["car"] == car) & (laps_for_fit["track_canonical"] == track)
    return int(laps_for_fit.loc[mask, col].notna().sum())


def _pass1_fit_tau_and_gains(
    laps_for_fit: pd.DataFrame, car: str, corner: str
) -> tuple[FitParam, dict[str, FitParam]]:
    """Fit τ_sec[car, corner] jointly across that car's (track) buckets.

    Returns (tau_FitParam, {track: gain_FitParam}). Buckets with fewer than
    ``MIN_LAPS_FOR_TAU_FIT`` lap samples are excluded from this pass; they get
    a gain in Pass 2 only if they meet ``MIN_LAPS_FOR_K_BUCKET``.
    """
    delta_col = f"delta_t_{corner}"
    car_df = laps_for_fit[(laps_for_fit["car"] == car) & laps_for_fit[delta_col].notna()]
    if car_df.empty:
        return (FitParam(PRIOR_TAU_SEC, 0.0, 0, from_prior=True), {})

    # Build per-bucket arrays
    buckets: list[tuple[str, np.ndarray, np.ndarray]] = []
    for track, grp in car_df.groupby("track_canonical"):
        if len(grp) < MIN_LAPS_FOR_TAU_FIT:
            continue
        buckets.append((str(track), grp["t_cum_s"].to_numpy(), grp[delta_col].to_numpy()))

    if not buckets:
        return (FitParam(PRIOR_TAU_SEC, 0.0, 0, from_prior=True), {})

    n_buckets = len(buckets)
    # Concatenate all samples; remember which bucket each belongs to
    all_t = np.concatenate([b[1] for b in buckets])
    all_y = np.concatenate([b[2] for b in buckets])
    bucket_idx = np.concatenate([np.full(len(b[1]), i, dtype=int) for i, b in enumerate(buckets)])

    def model(_x: np.ndarray, *params: float) -> np.ndarray:
        # params: tau_sec, gain_0, gain_1, ..., gain_{n_buckets-1}
        tau = params[0]
        gains = np.array(params[1:])
        out: np.ndarray = gains[bucket_idx] * (1.0 - np.exp(-all_t / tau))
        return out

    p0 = [PRIOR_TAU_SEC] + [40.0] * n_buckets  # gain ≈ ΔT_∞ ≈ 40 K
    bounds_lower = [60.0] + [0.0] * n_buckets
    bounds_upper = [1200.0] + [200.0] * n_buckets

    try:
        popt, pcov = scipy.optimize.curve_fit(
            model, all_t, all_y, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=10000
        )
    except Exception as e:  # noqa: BLE001 — convert any optimizer failure to a prior
        logger.warning("Pass-1 fit failed for (%s, %s): %s — using prior", car, corner, e)
        return (FitParam(PRIOR_TAU_SEC, 0.0, int(len(all_t)), from_prior=True), {})

    perr = np.sqrt(np.diag(pcov))
    tau_fit = FitParam(float(popt[0]), float(perr[0]), int(len(all_t)))

    per_bucket: dict[str, FitParam] = {}
    for i, (track, _t, _y) in enumerate(buckets):
        per_bucket[track] = FitParam(
            value=float(popt[1 + i]),
            stderr=float(perr[1 + i]),
            n_samples=int(len(buckets[i][1])),
        )
    return tau_fit, per_bucket


# ---------- Pass 2: factor per-bucket gains into K × c_track ----------


def _pass2_factor_gains(
    *,
    bucket_gains: dict[tuple[str, str, str], FitParam],
    g2_lookup: dict[tuple[str, str], tuple[float, int]],
    anchor_track: str,
) -> tuple[dict[tuple[str, str], FitParam], dict[str, FitParam]]:
    """Decompose ``gain_b = K[car, corner] · c_track[track] · ⟨g²⟩[track, car]``.

    Uses alternating least squares in log space with ``c_track[anchor] ≡ 1.0``.
    Returns (k_by_car_corner, c_track_by_track).
    """
    if not bucket_gains:
        return {}, {}

    # effective_gain = gain / g2_typ  =  K · c_track
    log_eff: dict[tuple[str, str, str], float] = {}
    stderr_eff: dict[tuple[str, str, str], float] = {}
    for (car, track, corner), gain in bucket_gains.items():
        g2 = g2_lookup.get((track, car), (np.nan, 0))[0]
        if not np.isfinite(g2) or g2 <= 0 or gain.value <= 0:
            continue
        eff = gain.value / g2
        log_eff[(car, track, corner)] = math.log(eff)
        # δ(log eff) ≈ (δ gain)/gain  (g2 is treated as exact)
        stderr_eff[(car, track, corner)] = gain.stderr / gain.value if gain.value > 0 else 1.0

    tracks = sorted({t for (_, t, _) in log_eff})
    cc_pairs = sorted({(c, k) for (c, _, k) in log_eff})

    log_c_track: dict[str, float] = {t: 0.0 for t in tracks}  # log(c_track[anchor]) = 0
    log_k: dict[tuple[str, str], float] = {p: 0.0 for p in cc_pairs}

    # Alternating LS (anchor c_track[ANCHOR_TRACK] = 1.0 ⇒ log = 0)
    for _ in range(20):
        # Solve for log_k holding log_c_track fixed: log_k[c,k] = mean over tracks (log_eff − log_c_track)
        for car, corner in cc_pairs:
            vals: list[float] = []
            for track in tracks:
                key = (car, track, corner)
                if key in log_eff:
                    vals.append(log_eff[key] - log_c_track[track])
            if vals:
                log_k[(car, corner)] = float(np.mean(vals))

        # Solve for log_c_track holding log_k fixed; anchor stays at 0
        for track in tracks:
            if track == anchor_track:
                log_c_track[track] = 0.0
                continue
            vals = []
            for car, corner in cc_pairs:
                key = (car, track, corner)
                if key in log_eff:
                    vals.append(log_eff[key] - log_k[(car, corner)])
            if vals:
                log_c_track[track] = float(np.mean(vals))

    # Convert back from log space; collect stderr from residuals
    k_by_car_corner: dict[tuple[str, str], FitParam] = {}
    for car, corner in cc_pairs:
        residuals: list[float] = []
        n_total = 0
        from_single_track = True
        seen_tracks = set()
        for track in tracks:
            key = (car, track, corner)
            if key in log_eff:
                residuals.append(log_eff[key] - log_k[(car, corner)] - log_c_track[track])
                n_total += bucket_gains[key].n_samples
                seen_tracks.add(track)
        if len(seen_tracks) >= 2:
            from_single_track = False
        rmse_log = float(np.std(residuals, ddof=0)) if residuals else 0.0
        k_val = math.exp(log_k[(car, corner)])
        k_stderr = k_val * rmse_log  # propagate via δ(K) = K · δ(log K)
        # Use bucket size as the n_samples count (most informative single number)
        n_max = max(
            (bucket_gains[(car, t, corner)].n_samples for t in seen_tracks),
            default=0,
        )
        k_by_car_corner[(car, corner)] = FitParam(
            value=k_val,
            stderr=k_stderr,
            n_samples=max(n_total, n_max),
        )
        # Stash side-info via metadata-free trick: encode "from_single_track" in
        # the from_prior field's nuance — keep it explicit instead.
        # (Handled below when serializing K_buckets.)

    c_track_by_track: dict[str, FitParam] = {}
    for track in tracks:
        residuals = []
        for car, corner in cc_pairs:
            key = (car, track, corner)
            if key in log_eff:
                residuals.append(log_eff[key] - log_k[(car, corner)] - log_c_track[track])
        rmse_log = float(np.std(residuals, ddof=0)) if residuals else 0.0
        val = math.exp(log_c_track[track])
        stderr = 0.0 if track == anchor_track else val * rmse_log
        n_buckets = sum(1 for cc in cc_pairs if (cc[0], track, cc[1]) in log_eff)
        c_track_by_track[track] = FitParam(value=val, stderr=stderr, n_samples=n_buckets)

    # Re-flag K params whose data came from a single track
    for car, corner in list(k_by_car_corner.keys()):
        seen = {t for t in tracks if (car, t, corner) in log_eff}
        if len(seen) < 2:
            # store flag in a sibling dict — we'll thread it through assemble_model
            k_by_car_corner[(car, corner)] = FitParam(
                value=k_by_car_corner[(car, corner)].value,
                stderr=k_by_car_corner[(car, corner)].stderr,
                n_samples=k_by_car_corner[(car, corner)].n_samples,
                from_prior=False,
            )

    return k_by_car_corner, c_track_by_track


# ---------- Assemble + write artifacts ----------


def _assemble_model(
    *,
    tau_by_car_corner: dict[tuple[str, str], FitParam],
    k_by_car_corner: dict[tuple[str, str], FitParam],
    c_track_by_track: dict[str, FitParam],
    g2_lookup: dict[tuple[str, str], tuple[float, int]],
    lap_time_lookup: dict[tuple[str, str], tuple[float, int]],
    bucket_n_samples: dict[tuple[str, str, str], int],
    blacklist_applied: list[dict] | None = None,
) -> dict[str, Any]:
    """Build the in-memory model dict that matches the JSON artifact schema."""
    fit_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # tracks where K-bucket data was seen for a (car, corner)
    seen_tracks_per_cc: dict[tuple[str, str], set[str]] = {}
    for car, track, corner in bucket_n_samples:
        seen_tracks_per_cc.setdefault((car, corner), set()).add(track)

    return {
        "schema_version": SCHEMA_VERSION,
        "fit_at_utc": fit_at,
        "model_form": (
            "T_hot - T_eff = K * c_track * g2_typ * (1 - exp(-t / tau_sec))   "
            "where T_eff = (1-w_road)*T_air + w_road*T_road, t = N * lap_time_typ_s"
        ),
        "gay_lussac": {
            "p_atm_bar": 1.0,
            "t_zero_c_to_k": 273.15,
            "t_cold_uses": "T_air",
        },
        "energy_balance": {
            "w_road": W_ROAD,
            "w_road_fitted": False,
            "t_road_proxy": {
                "formula": "T_air + delta_sun_max_c * (1 - cloud_cover/100) * sun_factor",
                "delta_sun_max_c": DELTA_SUN_MAX_C,
                "sun_factor_default": SUN_FACTOR_DEFAULT,
            },
        },
        "corners": list(CORNERS),
        "min_samples_per_bucket": MIN_LAPS_FOR_K_BUCKET,
        "priors_when_no_fit": {
            "tau_sec_seconds": PRIOR_TAU_SEC,
            "K_kelvin_per_g2": PRIOR_K_KELVIN_PER_G2,
            "c_track": PRIOR_C_TRACK,
        },
        "tau_sec_by_car_corner": [
            {
                "car": car,
                "corner": corner,
                "value_seconds": fp.value,
                "stderr_seconds": fp.stderr,
                "n_samples_used": fp.n_samples,
                "from_prior": fp.from_prior,
            }
            for (car, corner), fp in sorted(tau_by_car_corner.items())
        ],
        "K_buckets": [
            {
                "key": {"car": car, "corner": corner},
                "value_kelvin_per_g2": fp.value,
                "stderr_kelvin_per_g2": fp.stderr,
                "n_samples": fp.n_samples,
                "from_prior": fp.from_prior,
                "from_single_track": len(seen_tracks_per_cc.get((car, corner), set())) < 2,
            }
            for (car, corner), fp in sorted(k_by_car_corner.items())
        ],
        "c_track_by_track": [
            {
                "track_canonical": track,
                "value": fp.value,
                "stderr": fp.stderr,
                "n_buckets_used": fp.n_samples,
                "anchor": track == ANCHOR_TRACK,
            }
            for track, fp in sorted(c_track_by_track.items())
        ],
        "g2_typ_by_track_car": [
            {
                "track_canonical": track,
                "car": car,
                "g2_typ": value,
                "n_laps_used": n,
            }
            for (track, car), (value, n) in sorted(g2_lookup.items())
        ],
        "lap_time_typ_by_track_car": [
            {
                "track_canonical": track,
                "car": car,
                "lap_time_typ_s": value,
                "n_laps_used": n,
            }
            for (track, car), (value, n) in sorted(lap_time_lookup.items())
        ],
        "fallback_order_for_K": [
            ["car", "corner"],
            ["car"],
            [],
        ],
        "sensor_blacklist_applied": sorted(
            blacklist_applied or [], key=lambda r: (r["session_id"], r["corner"])
        ),
    }


def _write_warmup_table_parquet(root: Path, model: dict[str, Any]) -> None:
    """Flatten the model into a single per-bucket table for Python fast-load."""
    rows: list[dict[str, Any]] = []
    tau_idx = {(d["car"], d["corner"]): d for d in model["tau_sec_by_car_corner"]}
    c_track_idx = {d["track_canonical"]: d for d in model["c_track_by_track"]}
    g2_idx = {(d["track_canonical"], d["car"]): d for d in model["g2_typ_by_track_car"]}
    lt_idx = {(d["track_canonical"], d["car"]): d for d in model["lap_time_typ_by_track_car"]}
    for kb in model["K_buckets"]:
        car = kb["key"]["car"]
        corner = kb["key"]["corner"]
        tau = tau_idx.get((car, corner), {})
        # Cross-product over tracks where this (car, corner) appears
        for track, ct in c_track_idx.items():
            g2 = g2_idx.get((track, car), {})
            lt = lt_idx.get((track, car), {})
            if not g2:
                continue
            rows.append(
                {
                    "track_canonical": track,
                    "car": car,
                    "corner": corner,
                    "K_kelvin_per_g2": kb["value_kelvin_per_g2"],
                    "K_stderr": kb["stderr_kelvin_per_g2"],
                    "tau_sec": tau.get("value_seconds", np.nan),
                    "tau_stderr": tau.get("stderr_seconds", np.nan),
                    "c_track": ct["value"],
                    "c_track_stderr": ct["stderr"],
                    "g2_typ": g2.get("g2_typ", np.nan),
                    "lap_time_typ_s": lt.get("lap_time_typ_s", np.nan),
                    "n_samples_K": kb["n_samples"],
                    "from_prior": kb["from_prior"],
                }
            )
    if not rows:
        logger.warning("No K buckets to write — empty warmup_table.parquet")
    table = pa.Table.from_pylist(rows)
    out = root / "warmup_table.parquet"
    pq.write_table(table, out, compression="zstd", compression_level=3)


def _write_tire_model_json(root: Path, model: dict[str, Any]) -> None:
    out = root / "tire_model.json"
    out.write_text(json.dumps(model, indent=2, sort_keys=False))
