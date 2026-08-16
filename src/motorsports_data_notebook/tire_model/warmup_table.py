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
SCHEMA_VERSION = 3

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

# Percentile (0–100) of per-lap heat_proxy/on_track_s used as the bucket's
# representative ⟨g²⟩. The median underweights fast on-pace laps because
# `tire_usable` already drops out-laps + in-laps + slow laps (> 1.40×
# session-best), but still keeps mid-pace and recovery laps that pull the
# centre of mass down vs. the asymptote a hot lap actually reaches. CV
# diagnostic on the 5-fold residuals shows per-lap g² (relative to bucket
# median) is the single biggest univariate signal in unexplained residual
# variance (R² ≈ 0.065, β ≈ −44 K/(g²·s)/s), so shifting the
# representative up reduces under-prediction on fast laps without
# changing the structural model.
G2_TYP_PERCENTILE = 75.0

# Bucket-size thresholds
MIN_LAPS_FOR_TAU_FIT = 30  # per (car, track, corner) bucket to participate in Pass 1
MIN_LAPS_FOR_K_BUCKET = 5  # per (car, track, corner) bucket to factor in Pass 2

# ---- Target-lap-time feature: g² as a function of pace ----
# Energy into the tires scales strongly with pace (log(g²) vs log(lap time)
# slopes of -2.4…-3.6, |r| 0.8-0.97 on the 2026-08 dataset; pure v²-scaling
# physics would give -4). Fitted sector-wise — see tire_model.sectors — so
# one bad turn on an otherwise aggressive lap can't skew the mapping. Each
# ⟨g²⟩ bucket carries a piecewise-linear g2_vs_lap_time curve; buckets
# without one fall back to the pooled sector-fit exponent below.
G2_LAP_TIME_EXPONENT_FALLBACK = 3.0  # used when even the pooled sector fit is empty
# Prediction-time clamp on the g² multiplier so an unrealistic target lap
# time can't extrapolate the asymptote into nonsense.
G2_SCALE_MULTIPLIER_CLAMP = (0.4, 2.5)

# ---- Compound-aware K (Inferno 86 runs A052 and RE-71RS interchangeably) ----
# Labeled sessions (tire_compounds.yaml sidecar + notes extraction) get a
# per-(car, compound, corner, condition) K fitted with the pooled τ and
# c_track held fixed — closed-form weighted least squares, so sparse
# compound buckets stay stable. Unlabeled sessions keep the pooled K.
MIN_LAPS_FOR_COMPOUND_K = 10

# Damp/wet τ should physically be ≤ the dry τ (faster cooling in rain). When a
# rain bucket fits a τ_sec much larger than the same (car, corner)'s dry τ,
# the fit is almost certainly picking up warmup-interrupted short stints
# rather than real thermal physics. Cap to this multiple of dry τ and flag.
MAX_WET_TAU_VS_DRY_RATIO = 1.5

# Damp/wet K should physically be ≤ dry K (less friction, less heat). Cap to
# this multiple of the same (car, corner)'s dry K when the fit comes back
# higher. 1.2 gives a small upside band — empirical sweet spot in v0.3
# held-out validation. A hard 1.0 cap under-predicts; uncapped over-predicts.
MAX_WET_K_VS_DRY_RATIO = 1.2

# Per-(session, corner) sensor sanity check: flag stuck/broken TPMS channels so
# the fit doesn't learn from them. Pure heuristic — easy to tune later.
BROKEN_CORNER_STD_THRESHOLD_C = 1.0  # std(temp) across session's tire-usable laps
BROKEN_CORNER_MIN_LAPS = 4  # need at least this many laps to call it "stuck"

# Precipitation thresholds for condition classification (mm/hr).
# Lower bound for "damp" follows Open-Meteo's "trace precipitation" magnitude.
# Upper bound for "damp" is the start of light rain.
CONDITION_DRY_MAX_PRECIP_MM_HR = 0.1
CONDITION_DAMP_MAX_PRECIP_MM_HR = 1.0
CONDITIONS = ("dry", "damp", "wet")
DEFAULT_CONDITION = "dry"  # used at inference when caller doesn't supply one


def _clip_wet_tau_to_dry_ratio(
    tau_by_car_corner_cond: dict[tuple[str, str, str], "FitParam"],
) -> dict[tuple[str, str, str], "FitParam"]:
    """Cap damp/wet τ_sec at MAX_WET_TAU_VS_DRY_RATIO × the same (car, corner)'s
    dry τ. Marks any clipped entry with ``from_prior=True``.

    Physically, rain ⇒ more cooling ⇒ smaller τ; a damp/wet τ that's much
    larger than dry is the fit's way of saying "the stint was too short to
    actually see warmup complete, so I extrapolate that τ is huge". Clipping
    keeps such buckets from blowing up predictions at long stints.
    """
    out = dict(tau_by_car_corner_cond)
    for (car, corner, cond), fp in tau_by_car_corner_cond.items():
        if cond == "dry":
            continue
        dry_fp = tau_by_car_corner_cond.get((car, corner, "dry"))
        if dry_fp is None:
            continue
        cap = MAX_WET_TAU_VS_DRY_RATIO * dry_fp.value
        if fp.value > cap and not fp.from_prior:
            logger.warning(
                "Clipping τ_sec[%s, %s, %s] = %.0f s → %.0f s "
                "(dry τ = %.0f s, ratio cap = %.1f×). Likely warmup-incomplete "
                "buckets — investigate data quality.",
                car,
                corner,
                cond,
                fp.value,
                cap,
                dry_fp.value,
                MAX_WET_TAU_VS_DRY_RATIO,
            )
            out[(car, corner, cond)] = FitParam(
                value=cap,
                stderr=fp.stderr,
                n_samples=fp.n_samples,
                from_prior=True,
            )
    return out


def _clip_wet_k_to_dry_ratio(
    k_by_car_corner_cond: dict[tuple[str, str, str], "FitParam"],
) -> dict[tuple[str, str, str], "FitParam"]:
    """Cap damp/wet K at MAX_WET_K_VS_DRY_RATIO × same (car, corner)'s dry K.

    Physically, rain ⇒ lower μ ⇒ less heat per G² ⇒ K should drop. A fitted
    K[damp] > K[dry] is the optimizer compensating for some unmodeled effect
    (often a too-large fitted τ that drives a low warmup_frac, which then
    needs a big K to match the observed temps). Clipping prevents that
    chain from blowing up rain predictions.
    """
    out = dict(k_by_car_corner_cond)
    for (car, corner, cond), fp in k_by_car_corner_cond.items():
        if cond == "dry":
            continue
        dry_fp = k_by_car_corner_cond.get((car, corner, "dry"))
        if dry_fp is None:
            continue
        cap = MAX_WET_K_VS_DRY_RATIO * dry_fp.value
        if fp.value > cap and not fp.from_prior:
            logger.warning(
                "Clipping K[%s, %s, %s] = %.1f K/G² → %.1f K/G² "
                "(dry K = %.1f, ratio cap = %.1f×).",
                car,
                corner,
                cond,
                fp.value,
                cap,
                dry_fp.value,
                MAX_WET_K_VS_DRY_RATIO,
            )
            out[(car, corner, cond)] = FitParam(
                value=cap,
                stderr=fp.stderr,
                n_samples=fp.n_samples,
                from_prior=True,
            )
    return out


def classify_condition(precipitation_mm_hr: float | None) -> str:
    """Map precipitation rate to a categorical condition.

    - dry    : precipitation < 0.1 mm/hr  (effectively no rain)
    - damp   : 0.1 ≤ precipitation < 1.0  (trace to light drizzle)
    - wet    : precipitation ≥ 1.0        (light rain or heavier)
    - unknown: precipitation is None / NaN (no weather data for this session)

    Sessions with `unknown` condition are excluded from training so we don't
    leak ambiguity into the fit; at inference the user supplies a category
    directly via `--condition`.
    """
    if precipitation_mm_hr is None:
        return "unknown"
    try:
        if not math.isfinite(precipitation_mm_hr):
            return "unknown"
    except TypeError:
        return "unknown"
    if precipitation_mm_hr < CONDITION_DRY_MAX_PRECIP_MM_HR:
        return "dry"
    if precipitation_mm_hr < CONDITION_DAMP_MAX_PRECIP_MM_HR:
        return "damp"
    return "wet"


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
    from .sectors import build_pace_model

    g2_curves, g2_exponent_default = build_pace_model(root, laps)

    laps_for_fit = _laps_for_fit(laps, g2_lookup)

    # τ and per-bucket gains are now per (car, corner, condition).
    tau_by_car_corner_cond: dict[tuple[str, str, str], FitParam] = {}
    bucket_gains: dict[tuple[str, str, str, str], FitParam] = {}
    bucket_n_samples: dict[tuple[str, str, str, str], int] = {}

    seen_conditions = sorted(set(laps_for_fit["condition"]))
    for car in sorted(set(laps_for_fit["car"])):
        for corner in CORNERS:
            for cond in seen_conditions:
                tau, per_bucket_gain = _pass1_fit_tau_and_gains(laps_for_fit, car, corner, cond)
                if tau.n_samples == 0 and not per_bucket_gain:
                    # Skip empty (car, corner, condition) combos — no data at all
                    continue
                tau_by_car_corner_cond[(car, corner, cond)] = tau
                for track, gain in per_bucket_gain.items():
                    bucket_gains[(car, track, corner, cond)] = gain
                    bucket_n_samples[(car, track, corner, cond)] = _bucket_sample_count(
                        laps_for_fit, car, track, corner, cond
                    )

    # Sanity-clip rain-condition τ to a sane multiple of dry τ. Sparse wet/damp
    # buckets can fit pathologically large τ when stints are short and the
    # warmup-curve fit is undersampled; this cap prevents that from silently
    # producing wildly wrong predictions. Clipped entries keep their stderr
    # and get `from_prior=True` so the user sees the override.
    tau_by_car_corner_cond = _clip_wet_tau_to_dry_ratio(tau_by_car_corner_cond)

    k_by_car_corner_cond, c_track_by_track = _pass2_factor_gains(
        bucket_gains=bucket_gains,
        g2_lookup=g2_lookup,
        anchor_track=ANCHOR_TRACK,
    )
    # Same physical-prior clip on K (rain ⇒ less heat ⇒ K ≤ dry K).
    k_by_car_corner_cond = _clip_wet_k_to_dry_ratio(k_by_car_corner_cond)

    # Compound-aware K: multi-task fit with partial supervision. The
    # compound-assignment task is supervised where labels exist (sidecar +
    # notes, plus weather-driven condition seeds like the KK-SII's
    # DRY/WET) and latent elsewhere; the temperature-regression task
    # shares the per-compound K parameters. Solved jointly by EM
    # (mixture of regressions with pinned responsibilities). Soft
    # assignments are training-only — held-out evaluation uses human/seed
    # labels exclusively (see validate._evaluate_fold).
    from .compound_infer import apply_condition_seeds, fit_compounds_em
    from .compounds import load_compound_labels, load_condition_seeds

    compound_labels = load_compound_labels(root)
    if exclude_session_ids:
        compound_labels = compound_labels[
            ~compound_labels["session_id"].isin(exclude_session_ids)
        ].reset_index(drop=True)
    compound_labels = apply_condition_seeds(
        compound_labels, laps_for_fit, load_condition_seeds(root)
    )
    k_em, em_assignments = fit_compounds_em(
        laps_for_fit, compound_labels, tau_by_car_corner_cond, c_track_by_track
    )
    k_by_compound = {
        key: FitParam(value=k, stderr=stderr, n_samples=int(round(n_eff)))
        for key, (k, stderr, n_eff) in k_em.items()
    }
    if k_by_compound:
        inferred = [a for a in em_assignments if not a.pinned and a.responsibility >= 0.9]
        logger.info(
            "Fitted %d compound K buckets (EM over %d session-axles, "
            "%d pinned, %d confidently inferred)",
            len(k_by_compound),
            len(em_assignments),
            sum(1 for a in em_assignments if a.pinned),
            len(inferred),
        )

    model = _assemble_model(
        tau_by_car_corner_cond=tau_by_car_corner_cond,
        k_by_car_corner_cond=k_by_car_corner_cond,
        c_track_by_track=c_track_by_track,
        g2_lookup=g2_lookup,
        lap_time_lookup=lap_time_lookup,
        bucket_n_samples=bucket_n_samples,
        blacklist_applied=blacklist_applied,
        g2_curves=g2_curves,
        g2_exponent_default=g2_exponent_default,
        k_by_compound=k_by_compound,
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
    cols = ["track_canonical", "ts_utc", "temperature_2m", "cloud_cover", "precipitation"]
    rows: list[pd.DataFrame] = []
    wx_root = weather_dir(root)
    if not wx_root.exists():
        logger.warning("No weather directory at %s — predictions will use T_air fallback", wx_root)
        return pd.DataFrame(columns=cols)
    for track_dir in sorted(wx_root.iterdir()):
        if not track_dir.is_dir():
            continue
        for f in sorted(track_dir.glob("*.parquet")):
            wx = pq.read_table(f).to_pandas()
            wx["track_canonical"] = track_dir.name
            rows.append(wx[cols])
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(rows, ignore_index=True)


def _attach_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join hourly weather onto laps via floor-to-hour on session_start_utc.

    Adds three columns: ``t_air_c``, ``cloud_cover``, ``precipitation``
    (mm/hr). Also derives ``condition`` from precipitation via
    :func:`classify_condition` — used as a model dimension.
    """
    if weather.empty:
        laps["t_air_c"] = np.nan
        laps["cloud_cover"] = np.nan
        laps["precipitation"] = np.nan
        laps["condition"] = "unknown"
        return laps
    laps = laps.copy()
    starts = pd.to_datetime(laps["session_start_utc"], utc=True)
    laps["_hour_key"] = starts.dt.strftime("%Y-%m-%dT%H:00")
    out = laps.merge(
        weather.rename(columns={"temperature_2m": "t_air_c", "ts_utc": "_hour_key"})[
            ["track_canonical", "_hour_key", "t_air_c", "cloud_cover", "precipitation"]
        ],
        on=["track_canonical", "_hour_key"],
        how="left",
    )
    out = out.drop(columns=["_hour_key"])
    # Fill T_air with historical-median by track when weather is missing.
    median_t_air = out.groupby("track_canonical")["t_air_c"].transform("median")
    out["t_air_c"] = out["t_air_c"].fillna(median_t_air)
    out["condition"] = out["precipitation"].apply(classify_condition)
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


def _build_lap_time_typ(
    laps: pd.DataFrame,
) -> dict[tuple[str, str, str], tuple[float, int]]:
    """Median on_track_s per (track, car, condition). Drops `unknown` condition.

    Returns ``{(track, car, condition): (median_s, n)}``.
    """
    out: dict[tuple[str, str, str], tuple[float, int]] = {}
    for (track, car, cond), grp in laps.groupby(["track_canonical", "car", "condition"]):
        if cond == "unknown":
            continue
        out[(str(track), str(car), str(cond))] = (
            float(grp["on_track_s"].median()),
            int(len(grp)),
        )
    return out


def _build_g2_typ(
    laps: pd.DataFrame,
    *,
    percentile: float = G2_TYP_PERCENTILE,
) -> dict[tuple[str, str, str], tuple[float, int]]:
    """Representative total ``heat_proxy/on_track_s`` per (track, car, condition).

    Total (un-decomposed) ⟨g²⟩ kept here as a production-prediction fallback
    when the per-corner statistic (see :func:`_build_g2_typ_per_corner`)
    isn't available. ``percentile=50`` recovers the legacy median; the
    default :data:`G2_TYP_PERCENTILE` shifts toward the hot-lap end of the
    distribution so the warmup asymptote reflects what tires actually see
    on pace.

    Drops `unknown` condition. Returns ``{(track, car, condition): (g2, n)}``.
    """
    out: dict[tuple[str, str, str], tuple[float, int]] = {}
    for (track, car, cond), grp in laps.groupby(["track_canonical", "car", "condition"]):
        if cond == "unknown":
            continue
        g2_per_lap = grp["heat_proxy"] / grp["on_track_s"]
        g2_per_lap = g2_per_lap.replace([np.inf, -np.inf], np.nan).dropna()
        if g2_per_lap.empty:
            continue
        out[(str(track), str(car), str(cond))] = (
            float(np.percentile(g2_per_lap, percentile)),
            int(len(g2_per_lap)),
        )
    return out


def _build_g2_typ_per_corner(
    laps: pd.DataFrame,
    *,
    percentile: float = G2_TYP_PERCENTILE,
) -> dict[tuple[str, str, str, str], tuple[float, int]]:
    """Per-corner percentile of ``heat_proxy_{corner}/on_track_s``.

    Falls back to the total ``heat_proxy`` when a corner column is missing
    (older extracts). Returns ``{(track, car, condition, corner): (g2, n)}``.
    """
    out: dict[tuple[str, str, str, str], tuple[float, int]] = {}
    for (track, car, cond), grp in laps.groupby(["track_canonical", "car", "condition"]):
        if cond == "unknown":
            continue
        for c in CORNERS:
            col = f"heat_proxy_{c}"
            src = grp[col] if col in grp.columns else grp["heat_proxy"]
            g2_per_lap = src / grp["on_track_s"]
            g2_per_lap = g2_per_lap.replace([np.inf, -np.inf], np.nan).dropna()
            if g2_per_lap.empty:
                continue
            out[(str(track), str(car), str(cond), c)] = (
                float(np.percentile(g2_per_lap, percentile)),
                int(len(g2_per_lap)),
            )
    return out


def _laps_for_fit(
    laps: pd.DataFrame,
    g2_lookup: dict[tuple[str, str, str], tuple[float, int]],
) -> pd.DataFrame:
    """Drop out-laps and rows without a valid t_eff, g², or known condition."""
    df = laps[laps["lap_within_stint"] > 0].copy()
    df = df[df["t_eff_c"].notna()]
    df = df[df["t_cum_s"] > 0]
    df = df[df["condition"] != "unknown"]
    # Attach the bucket's condition-specific g2_typ (used in Pass 2)
    df["g2_typ"] = [
        g2_lookup.get((t, c, cond), (np.nan, 0))[0]
        for t, c, cond in zip(df["track_canonical"], df["car"], df["condition"])
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


def _bucket_sample_count(
    laps_for_fit: pd.DataFrame, car: str, track: str, corner: str, condition: str
) -> int:
    col = f"delta_t_{corner}"
    mask = (
        (laps_for_fit["car"] == car)
        & (laps_for_fit["track_canonical"] == track)
        & (laps_for_fit["condition"] == condition)
    )
    return int(laps_for_fit.loc[mask, col].notna().sum())


def _pass1_fit_tau_and_gains(
    laps_for_fit: pd.DataFrame, car: str, corner: str, condition: str
) -> tuple[FitParam, dict[str, FitParam]]:
    """Fit τ_sec[car, corner, condition] jointly across that car's (track) buckets
    in the given condition.

    The closed-form warmup model uses **per-lap g²** as a known feature:
    ``ΔT_i = (K · c_track) · g²_i · (1 - exp(-t_i / τ))``, where g²_i is
    ``heat_proxy_i / on_track_s_i`` for lap i. The fitted "gain" per bucket
    is therefore ``K · c_track`` (no ⟨g²⟩ factor); Pass 2 decomposes it
    into the per-car K and per-track c_track without the prior division
    by a bucket statistic. Laps with higher actual g² get a higher
    asymptote, which matches the field observation that on-pace laps run
    hotter than the bucket median.

    Returns (tau_FitParam, {track: gain_FitParam}). Buckets with fewer than
    ``MIN_LAPS_FOR_TAU_FIT`` lap samples are excluded from this pass; they get
    a gain in Pass 2 only if they meet ``MIN_LAPS_FOR_K_BUCKET``.
    """
    delta_col = f"delta_t_{corner}"
    car_df = laps_for_fit[
        (laps_for_fit["car"] == car)
        & (laps_for_fit["condition"] == condition)
        & laps_for_fit[delta_col].notna()
    ].copy()
    if car_df.empty:
        return (FitParam(PRIOR_TAU_SEC, 0.0, 0, from_prior=True), {})

    # Use total per-lap g² (heat_proxy / on_track_s); a per-corner
    # signed-G decomposition was tried but the crude sign-splitting hurt
    # FR / FL MAE more than it helped RL / RR, so leave it as future work
    # gated on a chassis-aware load-transfer model.
    car_df["g2_lap"] = car_df["heat_proxy"] / car_df["on_track_s"]
    car_df = car_df[car_df["g2_lap"].notna() & (car_df["g2_lap"] > 0)]
    if car_df.empty:
        return (FitParam(PRIOR_TAU_SEC, 0.0, 0, from_prior=True), {})

    # Build per-bucket arrays
    buckets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for track, grp in car_df.groupby("track_canonical"):
        if len(grp) < MIN_LAPS_FOR_TAU_FIT:
            continue
        buckets.append(
            (
                str(track),
                grp["t_cum_s"].to_numpy(),
                grp[delta_col].to_numpy(),
                grp["g2_lap"].to_numpy(),
            )
        )

    if not buckets:
        return (FitParam(PRIOR_TAU_SEC, 0.0, 0, from_prior=True), {})

    n_buckets = len(buckets)
    # Concatenate all samples; remember which bucket each belongs to
    all_t = np.concatenate([b[1] for b in buckets])
    all_y = np.concatenate([b[2] for b in buckets])
    all_g2 = np.concatenate([b[3] for b in buckets])
    bucket_idx = np.concatenate([np.full(len(b[1]), i, dtype=int) for i, b in enumerate(buckets)])

    def model(_x: np.ndarray, *params: float) -> np.ndarray:
        # params: tau_sec, Kc_0, Kc_1, ..., Kc_{n_buckets-1}
        # where Kc_b = K[car, corner, cond] · c_track[track of bucket b]
        tau = params[0]
        Kc = np.array(params[1:])
        out: np.ndarray = Kc[bucket_idx] * all_g2 * (1.0 - np.exp(-all_t / tau))
        return out

    # Initial Kc guess: ΔT_∞ / typical g² ≈ 40 / 0.5 ≈ 80 K/G²
    p0 = [PRIOR_TAU_SEC] + [PRIOR_K_KELVIN_PER_G2] * n_buckets
    bounds_lower = [60.0] + [0.0] * n_buckets
    bounds_upper = [1200.0] + [500.0] * n_buckets

    try:
        popt, pcov = scipy.optimize.curve_fit(
            model, all_t, all_y, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=10000
        )
    except Exception as e:  # noqa: BLE001 — convert any optimizer failure to a prior
        logger.warning(
            "Pass-1 fit failed for (%s, %s, %s): %s — using prior", car, corner, condition, e
        )
        return (FitParam(PRIOR_TAU_SEC, 0.0, int(len(all_t)), from_prior=True), {})

    perr = np.sqrt(np.diag(pcov))
    tau_fit = FitParam(float(popt[0]), float(perr[0]), int(len(all_t)))

    per_bucket: dict[str, FitParam] = {}
    for i, (track, _t, _y, _g2) in enumerate(buckets):
        per_bucket[track] = FitParam(
            value=float(popt[1 + i]),
            stderr=float(perr[1 + i]),
            n_samples=int(len(buckets[i][1])),
        )
    return tau_fit, per_bucket


# ---------- Pass 2: factor per-bucket gains into K × c_track ----------


def _pass2_factor_gains(
    *,
    bucket_gains: dict[tuple[str, str, str, str], FitParam],
    g2_lookup: dict[tuple[str, str, str], tuple[float, int]],
    anchor_track: str,
) -> tuple[dict[tuple[str, str, str], FitParam], dict[str, FitParam]]:
    """Decompose ``gain_b = K[car, corner, condition] · c_track[track]``.

    Pass 1 now fits the warmup curve with per-lap g² as a known feature,
    so the bucket ``gain`` it returns is already ``K · c_track`` (no ⟨g²⟩
    factor). Pass 2 just factors that product into the per (car, corner,
    condition) K and the per-track c_track. ``g2_lookup`` is no longer
    used in the decomposition but is kept in the signature so callers
    don't have to change.

    ``c_track`` is shared across conditions (the asphalt's surface
    character is a property of the venue; condition's effect lives in
    ``K`` and ⟨g²⟩). Returns (k_by_car_corner_condition, c_track_by_track).
    """
    del g2_lookup  # unused; retained in signature for API stability
    if not bucket_gains:
        return {}, {}

    log_eff: dict[tuple[str, str, str, str], float] = {}
    for (car, track, corner, cond), gain in bucket_gains.items():
        if gain.value <= 0:
            continue
        log_eff[(car, track, corner, cond)] = math.log(gain.value)

    tracks = sorted({t for (_, t, _, _) in log_eff})
    # Condition-aware "K cell" = (car, corner, condition); c_track is per-track only.
    cc_cond_keys = sorted({(c, k, cond) for (c, _, k, cond) in log_eff})

    log_c_track: dict[str, float] = {t: 0.0 for t in tracks}  # log(c_track[anchor]) = 0
    log_k: dict[tuple[str, str, str], float] = {p: 0.0 for p in cc_cond_keys}

    # Alternating LS (anchor c_track[ANCHOR_TRACK] = 1.0 ⇒ log = 0)
    for _ in range(20):
        # Solve for log_k holding log_c_track fixed
        for car, corner, cond in cc_cond_keys:
            vals: list[float] = []
            for track in tracks:
                key = (car, track, corner, cond)
                if key in log_eff:
                    vals.append(log_eff[key] - log_c_track[track])
            if vals:
                log_k[(car, corner, cond)] = float(np.mean(vals))

        # Solve for log_c_track holding log_k fixed; anchor stays at 0
        for track in tracks:
            if track == anchor_track:
                log_c_track[track] = 0.0
                continue
            vals = []
            for car, corner, cond in cc_cond_keys:
                key = (car, track, corner, cond)
                if key in log_eff:
                    vals.append(log_eff[key] - log_k[(car, corner, cond)])
            if vals:
                log_c_track[track] = float(np.mean(vals))

    # Convert back from log space; collect stderr from residuals
    k_by_car_corner_cond: dict[tuple[str, str, str], FitParam] = {}
    for car, corner, cond in cc_cond_keys:
        residuals: list[float] = []
        n_total = 0
        seen_tracks: set[str] = set()
        for track in tracks:
            key = (car, track, corner, cond)
            if key in log_eff:
                residuals.append(log_eff[key] - log_k[(car, corner, cond)] - log_c_track[track])
                n_total += bucket_gains[key].n_samples
                seen_tracks.add(track)
        rmse_log = float(np.std(residuals, ddof=0)) if residuals else 0.0
        k_val = math.exp(log_k[(car, corner, cond)])
        k_stderr = k_val * rmse_log  # propagate via δ(K) = K · δ(log K)
        n_max = max(
            (bucket_gains[(car, t, corner, cond)].n_samples for t in seen_tracks),
            default=0,
        )
        k_by_car_corner_cond[(car, corner, cond)] = FitParam(
            value=k_val,
            stderr=k_stderr,
            n_samples=max(n_total, n_max),
        )

    c_track_by_track: dict[str, FitParam] = {}
    for track in tracks:
        residuals = []
        for car, corner, cond in cc_cond_keys:
            key = (car, track, corner, cond)
            if key in log_eff:
                residuals.append(log_eff[key] - log_k[(car, corner, cond)] - log_c_track[track])
        rmse_log = float(np.std(residuals, ddof=0)) if residuals else 0.0
        val = math.exp(log_c_track[track])
        stderr = 0.0 if track == anchor_track else val * rmse_log
        n_buckets = sum(1 for (c, k, cond) in cc_cond_keys if (c, track, k, cond) in log_eff)
        c_track_by_track[track] = FitParam(value=val, stderr=stderr, n_samples=n_buckets)

    return k_by_car_corner_cond, c_track_by_track


# ---------- Assemble + write artifacts ----------


def _assemble_model(
    *,
    tau_by_car_corner_cond: dict[tuple[str, str, str], FitParam],
    k_by_car_corner_cond: dict[tuple[str, str, str], FitParam],
    c_track_by_track: dict[str, FitParam],
    g2_lookup: dict[tuple[str, str, str], tuple[float, int]],
    lap_time_lookup: dict[tuple[str, str, str], tuple[float, int]],
    bucket_n_samples: dict[tuple[str, str, str, str], int],
    blacklist_applied: list[dict] | None = None,
    g2_curves: dict[tuple[str, str, str], dict] | None = None,
    g2_exponent_default: float = G2_LAP_TIME_EXPONENT_FALLBACK,
    k_by_compound: dict[tuple[str, str, str, str], "FitParam"] | None = None,
) -> dict[str, Any]:
    """Build the in-memory model dict that matches the JSON artifact schema.

    Schema version 3: adds the target-lap-time feature — a per-bucket
    ``g2_lap_time_exponent`` on each ⟨g²⟩ entry plus the top-level
    ``g2_lap_time_model`` block (default exponent + multiplier clamp).
    Version 2 keyed K and τ_sec by (car, corner, condition) and the ⟨g²⟩ +
    lap_time_typ lookups by (track, car, condition).
    """
    g2_curves = g2_curves or {}
    k_by_compound = k_by_compound or {}

    def _g2_entry(track: str, car: str, cond: str, value: float, n: int) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "track_canonical": track,
            "car": car,
            "condition": cond,
            "g2_typ": value,
            "n_laps_used": n,
        }
        curve = g2_curves.get((track, car, cond))
        if curve is not None:
            entry["g2_vs_lap_time"] = curve
        return entry

    fit_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # tracks where K-bucket data was seen for a (car, corner, condition)
    seen_tracks_per_kcell: dict[tuple[str, str, str], set[str]] = {}
    for car, track, corner, cond in bucket_n_samples:
        seen_tracks_per_kcell.setdefault((car, corner, cond), set()).add(track)

    return {
        "schema_version": SCHEMA_VERSION,
        "fit_at_utc": fit_at,
        "model_form": (
            "T_hot - T_eff = K[car,corner,cond] * c_track[track] * g2 "
            "* (1 - exp(-t / tau_sec[car,corner,cond]))   "
            "where T_eff = (1-w_road)*T_air + w_road*T_road, t = N * lap_time_s, "
            "g2 = g2_typ[track,car,cond] * clamp((lap_time_typ_s / target_lap_time_s)"
            "^g2_lap_time_exponent) when a target lap time is given, else g2_typ"
        ),
        "g2_lap_time_model": {
            "method": "sector_knn_median_curve",
            "formula": (
                "g2 = g2_typ * interp(target_lap_time_s, g2_vs_lap_time) / "
                "interp(lap_time_typ_s, g2_vs_lap_time); buckets without a curve "
                "fall back to g2_typ * (lap_time_typ_s / target_lap_time_s) ** "
                "default_exponent. The multiplier is clamped either way."
            ),
            "default_exponent": g2_exponent_default,
            "multiplier_clamp": {
                "min": G2_SCALE_MULTIPLIER_CLAMP[0],
                "max": G2_SCALE_MULTIPLIER_CLAMP[1],
            },
        },
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
        "conditions": {
            "values": list(CONDITIONS),
            "default": DEFAULT_CONDITION,
            "classification": {
                "from_field": "precipitation_mm_hr",
                "thresholds": {
                    "dry_max": CONDITION_DRY_MAX_PRECIP_MM_HR,
                    "damp_max": CONDITION_DAMP_MAX_PRECIP_MM_HR,
                },
                "rule": (
                    "p < dry_max → dry; dry_max ≤ p < damp_max → damp; "
                    "p ≥ damp_max → wet; missing → unknown (excluded from training)"
                ),
            },
        },
        "corners": list(CORNERS),
        "min_samples_per_bucket": MIN_LAPS_FOR_K_BUCKET,
        "priors_when_no_fit": {
            "tau_sec_seconds": PRIOR_TAU_SEC,
            "K_kelvin_per_g2": PRIOR_K_KELVIN_PER_G2,
            "c_track": PRIOR_C_TRACK,
        },
        "tau_sec_by_car_corner_cond": [
            {
                "car": car,
                "corner": corner,
                "condition": cond,
                "value_seconds": fp.value,
                "stderr_seconds": fp.stderr,
                "n_samples_used": fp.n_samples,
                "from_prior": fp.from_prior,
            }
            for (car, corner, cond), fp in sorted(tau_by_car_corner_cond.items())
        ],
        "K_buckets": [
            {
                "key": {"car": car, "corner": corner, "condition": cond},
                "value_kelvin_per_g2": fp.value,
                "stderr_kelvin_per_g2": fp.stderr,
                "n_samples": fp.n_samples,
                "from_prior": fp.from_prior,
                "from_single_track": (
                    len(seen_tracks_per_kcell.get((car, corner, cond), set())) < 2
                ),
            }
            for (car, corner, cond), fp in sorted(k_by_car_corner_cond.items())
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
        "g2_typ_by_track_car_cond": [
            _g2_entry(track, car, cond, value, n)
            for (track, car, cond), (value, n) in sorted(g2_lookup.items())
        ],
        # Compound-specific K overrides (schema v3 additive; consumers that
        # don't know about compounds ignore this table and use K_buckets).
        "K_by_car_compound_corner_cond": [
            {
                "car": car,
                "compound": compound,
                "corner": corner,
                "condition": cond,
                "value_kelvin_per_g2": fp.value,
                "stderr_kelvin_per_g2": fp.stderr,
                "n_laps": fp.n_samples,
            }
            for (car, compound, corner, cond), fp in sorted(k_by_compound.items())
        ],
        "lap_time_typ_by_track_car_cond": [
            {
                "track_canonical": track,
                "car": car,
                "condition": cond,
                "lap_time_typ_s": value,
                "n_laps_used": n,
            }
            for (track, car, cond), (value, n) in sorted(lap_time_lookup.items())
        ],
        "fallback_order_for_K": [
            ["car", "corner", "condition"],
            ["car", "corner"],
            ["car"],
            [],
        ],
        "fallback_order_for_condition_lookups": [
            ["track", "car", "condition"],
            ["track", "car", "dry"],
            ["track", "car"],
            ["track"],
        ],
        "sensor_blacklist_applied": sorted(
            blacklist_applied or [], key=lambda r: (r["session_id"], r["corner"])
        ),
    }


def _write_warmup_table_parquet(root: Path, model: dict[str, Any]) -> None:
    """Flatten the model into a single per-bucket table for Python fast-load.

    Rows are the cross-product (K bucket × c_track entry), filtered to those
    with matching ⟨g²⟩ and lap_time_typ entries for the same (track, car,
    condition).
    """
    rows: list[dict[str, Any]] = []
    tau_idx = {
        (d["car"], d["corner"], d["condition"]): d for d in model["tau_sec_by_car_corner_cond"]
    }
    c_track_idx = {d["track_canonical"]: d for d in model["c_track_by_track"]}
    g2_idx = {
        (d["track_canonical"], d["car"], d["condition"]): d
        for d in model["g2_typ_by_track_car_cond"]
    }
    lt_idx = {
        (d["track_canonical"], d["car"], d["condition"]): d
        for d in model["lap_time_typ_by_track_car_cond"]
    }
    for kb in model["K_buckets"]:
        car = kb["key"]["car"]
        corner = kb["key"]["corner"]
        cond = kb["key"]["condition"]
        tau = tau_idx.get((car, corner, cond), {})
        for track, ct in c_track_idx.items():
            g2 = g2_idx.get((track, car, cond), {})
            lt = lt_idx.get((track, car, cond), {})
            if not g2:
                continue
            rows.append(
                {
                    "track_canonical": track,
                    "car": car,
                    "corner": corner,
                    "condition": cond,
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
