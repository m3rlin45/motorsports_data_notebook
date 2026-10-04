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
from . import wetness as _wetness
from .energy_balance import P_ATM_BAR, T_ZERO_C_TO_K, t_effective_c, t_road_proxy_c

logger = logging.getLogger(__name__)

CORNERS = ("fl", "fr", "rl", "rr")
SCHEMA_VERSION = 4

# Cars pooled under one label for every fitted quantity. KK-F and KK-SII are
# near-identical FJ-series machines running the same tires, so their laps
# train a single "FJ" car. The map is emitted into tire_model.json as
# ``car_aliases`` so predictors keep accepting the raw car names.
CAR_FIT_ALIASES = {"KK-F": "FJ", "KK-SII": "FJ"}

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
# representative ⟨g²⟩. This was 75 while the fit target was the TPMS
# temperature: the sensor lags the gas by 2–3 min, which depressed the
# fitted K, and a hot ⟨g²⟩ compensated so the predicted pressures came out
# unbiased. With the pressure-implied gas temperature as the observable
# (2026-10) K is honest and the 75th percentile over-predicted held-out hot
# pressure by +0.015 bar, so the representative is the median; pace enters
# through the entered target lap time and the sector curve instead.
G2_TYP_PERCENTILE = 50.0

# Bucket-size thresholds
MIN_LAPS_FOR_TAU_FIT = 30  # per (car, track, corner) bucket to participate in Pass 1
MIN_LAPS_FOR_K_BUCKET = 5  # per (car, track, corner) bucket to factor in Pass 2

# Laps this deep into a stint count as steady-state for the UI prefill
# medians (hot temp / hot pressure per car+corner+condition). Buckets with
# fewer than CORNER_DEFAULTS_MIN_LAPS steady laps are dropped so a couple
# of outlier laps (e.g. two crawling wet laps) can't seed the prefill —
# the UI's condition chain falls back to a denser bucket instead.
CORNER_DEFAULTS_MIN_LAP_IN_STINT = 4
CORNER_DEFAULTS_MIN_LAPS = 5

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

# Rain thermal parameters are fitted per condition (dry / damp / wet) with
# the one bound the physics supports placed inside the fit: rain only adds
# cooling (evaporation off the tread, conduction into a wet, cold surface),
# so τ_rain ≤ τ_dry for any tire; K = α/h is left free because a rain
# compound has a different hysteresis α. A rain bucket is fitted on its own
# only when it has enough independent sessions; otherwise the predictors'
# wet → damp → dry fallback chain resolves it to the dry parameters.
MIN_SESSIONS_FOR_RAIN_FIT = 3

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

# ---------- Condition classification (from Open-Meteo precipitation) ----------


def classify_condition(precipitation_mm_hr: float | None) -> str:
    """Legacy: map a precipitation rate to a categorical condition.

    Production classifies from the surface water balance instead
    (:mod:`.wetness`, see :func:`_attach_weather`); this stays as the
    documented rain-rate convention behind the category names.

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
    Rain conditions are fitted on their own (τ_rain ≤ τ_dry bounded inside
    the fit, K free) when they have ``MIN_SESSIONS_FOR_RAIN_FIT`` sessions
    in a (car, track) bucket; otherwise they fall back to dry at prediction.
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
    laps = _compute_stint_anchor(laps)
    laps = _compute_delta_t(laps)

    flying = _flying_laps(laps)
    lap_time_lookup = _build_lap_time_typ(flying)
    g2_lookup = _build_g2_typ(flying)
    corner_defaults = _build_corner_defaults(flying)
    outlap_lookup = _build_outlap_typ(laps)
    from .sectors import build_pace_model

    g2_curves, g2_exponent_default = build_pace_model(root, flying)

    laps_for_fit = _laps_for_fit(laps, g2_lookup)

    # τ and per-bucket gains are now per (car, corner, condition).
    tau_by_car_corner_cond: dict[tuple[str, str, str], FitParam] = {}
    bucket_gains: dict[tuple[str, str, str, str], FitParam] = {}
    bucket_n_samples: dict[tuple[str, str, str, str], int] = {}

    seen_conditions = sorted(set(laps_for_fit["condition"]))
    # Dry first: its τ is the upper bound for the rain buckets.
    seen_conditions = [c for c in seen_conditions if c == "dry"] + [
        c for c in seen_conditions if c != "dry"
    ]
    for car in sorted(set(laps_for_fit["car"])):
        for corner in CORNERS:
            for cond in seen_conditions:
                tau_upper = _rain_tau_upper(tau_by_car_corner_cond, car, corner, cond)
                tau, per_bucket_gain = _pass1_fit_tau_and_gains(
                    laps_for_fit, car, corner, cond, tau_upper=tau_upper
                )
                if tau.n_samples == 0 and not per_bucket_gain:
                    # Skip empty (car, corner, condition) combos — no data at all
                    continue
                tau_by_car_corner_cond[(car, corner, cond)] = tau
                for track, gain in per_bucket_gain.items():
                    bucket_gains[(car, track, corner, cond)] = gain
                    bucket_n_samples[(car, track, corner, cond)] = _bucket_sample_count(
                        laps_for_fit, car, track, corner, cond
                    )

    k_by_car_corner_cond, c_track_by_track = _pass2_factor_gains(
        bucket_gains=bucket_gains,
        g2_lookup=g2_lookup,
        anchor_track=ANCHOR_TRACK,
    )

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
        compound_labels, laps_for_fit, alias_condition_seeds(load_condition_seeds(root))
    )
    k_em, em_assignments, compound_multipliers = fit_compounds_em(
        laps_for_fit, compound_labels, tau_by_car_corner_cond, c_track_by_track
    )
    k_by_compound = {
        key: FitParam(value=k, stderr=stderr, n_samples=int(round(n_eff)))
        for key, (k, stderr, n_eff) in k_em.items()
    }
    if k_by_compound:
        inferred = [a for a in em_assignments if not a.pinned and a.responsibility >= 0.9]
        logger.info(
            "Fitted %d compound K buckets (decomposed base×m; EM over %d "
            "session-axles, %d pinned, %d confidently inferred; multipliers %s)",
            len(k_by_compound),
            len(em_assignments),
            sum(1 for a in em_assignments if a.pinned),
            len(inferred),
            compound_multipliers,
        )

    _data_through = _data_through_for_fit(laps_for_fit)
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
        compound_multipliers=compound_multipliers,
        corner_defaults=corner_defaults,
        data_through_date=_data_through[0],
        data_through_local=_data_through[1],
        outlap_lookup=outlap_lookup,
    )

    if write_artifacts:
        _write_warmup_table_parquet(root, model)
        _write_tire_model_json(root, model)

    return model


# ---------- Data prep ----------


def apply_car_aliases(laps: pd.DataFrame) -> pd.DataFrame:
    """Map aliased car labels (see CAR_FIT_ALIASES) onto their pooled name."""
    if laps.empty or "car" not in laps.columns:
        return laps
    df = laps.copy()
    df["car"] = df["car"].replace(CAR_FIT_ALIASES)
    return df


def alias_condition_seeds(seeds: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Re-key the sidecar's condition_seeds through CAR_FIT_ALIASES.

    The sidecar records seeds against the raw car name (e.g. "KK-SII");
    after aliasing the laps carry the pooled label, so the seeds must be
    pooled the same way for :func:`compound_infer.apply_condition_seeds`
    to match them.
    """
    out: dict[str, dict[str, str]] = {}
    for car, mapping in seeds.items():
        key = CAR_FIT_ALIASES.get(car, car)
        out.setdefault(key, {}).update(mapping)
    return out


def _load_filtered_laps(root: Path) -> pd.DataFrame:
    """Read laps + sessions, filter to ok + has_tpms + tire_usable.

    Car labels are alias-pooled here (see CAR_FIT_ALIASES) so every
    consumer — the fit, held-out evaluation, sensor audit — sees the
    pooled car names.
    """
    laps_files = sorted((laps_dir(root)).glob("*.parquet"))
    sessions_files = sorted((sessions_dir(root)).glob("*.parquet"))
    if not laps_files or not sessions_files:
        raise FileNotFoundError(
            f"No laps/sessions parquets under {root}. Run `just tire-refresh` first."
        )
    laps = pa.concat_tables(
        [pq.read_table(f) for f in laps_files], promote_options="default"
    ).to_pandas()
    sessions = pa.concat_tables(
        [pq.read_table(f) for f in sessions_files], promote_options="default"
    ).to_pandas()

    sessions_keep = sessions[(sessions["status"] == "ok") & sessions["has_tpms"]]
    cols_from_sessions = ["session_id", "track_canonical", "car", "session_start_utc", "date"]
    df = laps.merge(sessions_keep[cols_from_sessions], on="session_id", how="inner")
    df = df[df["tire_usable"]].reset_index(drop=True)
    df = df[df["track_canonical"].notna()]
    df = df[df["car"].notna()]
    out: pd.DataFrame = apply_car_aliases(df.reset_index(drop=True))
    return out


def _load_weather(root: Path) -> pd.DataFrame:
    """Load all weather parquets into a single DataFrame keyed by (track, ts_utc)."""
    cols = [
        "track_canonical",
        "ts_utc",
        "temperature_2m",
        "cloud_cover",
        "precipitation",
        "relative_humidity_2m",
        "wind_speed_10m",
    ]
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
            for c in cols:
                if c not in wx.columns:
                    wx[c] = np.nan
            rows.append(wx[cols])
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(rows, ignore_index=True)


def _attach_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join hourly weather onto laps and derive the track condition.

    ``t_air_c`` and ``cloud_cover`` are the instantaneous values of the hour
    containing the session start; ``precipitation`` (mm/hr, start hour,
    preceding-hour sum) is kept for reference. ``condition`` comes from the
    surface water balance in :mod:`.wetness`: the film depth on the track at
    roll-out and over the run (``track_wetness_start_mm``,
    ``track_wetness_run_max_mm``), classified by :func:`wetness.classify_wetness`.
    """
    if weather.empty:
        laps["t_air_c"] = np.nan
        laps["cloud_cover"] = np.nan
        laps["precipitation"] = np.nan
        laps["track_wetness_start_mm"] = np.nan
        laps["track_wetness_run_max_mm"] = np.nan
        laps["condition"] = "unknown"
        return laps
    from .wetness import classify_wetness, session_wetness, surface_water_series

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

    # Expected track wetness per session from the surface water balance.
    depth_by_track = {
        str(track): surface_water_series(grp) for track, grp in weather.groupby("track_canonical")
    }
    duration = out.groupby("session_id")["on_track_s"].sum()
    feats: dict[str, dict[str, float]] = {}
    for sid, grp in out.groupby("session_id"):
        track = str(grp["track_canonical"].iloc[0])
        series = depth_by_track.get(track, pd.Series(dtype=float))
        feats[str(sid)] = session_wetness(
            series,
            pd.Timestamp(grp["session_start_utc"].iloc[0]),
            float(duration.get(sid, 0.0)),
        )
    out["track_wetness_start_mm"] = out["session_id"].map(
        lambda s: feats.get(str(s), {}).get("depth_start_mm", np.nan)
    )
    out["track_wetness_run_max_mm"] = out["session_id"].map(
        lambda s: feats.get(str(s), {}).get("depth_run_max_mm", np.nan)
    )
    out["condition"] = out["track_wetness_run_max_mm"].apply(classify_wetness)
    return out


def _is_outlap_col(df: pd.DataFrame) -> pd.Series:
    if "is_outlap" in df.columns:
        return df["is_outlap"].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


def _flying_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Laps that are not out-laps: what the per-bucket lookups (typical lap
    time, ⟨g²⟩, pace curves, corner defaults) describe. The out-lap has its
    own lookup (:func:`_build_outlap_typ`)."""
    return laps[~_is_outlap_col(laps)]


def _rolling_time_s(df: pd.DataFrame) -> pd.Series:
    """Per-lap rolling time: ``moving_s`` when the dataset carries it (schema
    v3), else ``on_track_s``. An out-lap's ``on_track_s`` includes the grid /
    pit-lane wait, which is not warm-up time."""
    if "moving_s" in df.columns:
        return df["moving_s"].where(df["moving_s"].notna(), df["on_track_s"])
    return df["on_track_s"]


def _compute_stint_clock(laps: pd.DataFrame) -> pd.DataFrame:
    """Add ``lap_within_stint`` (cumcount within (session_id, stint_id); the
    out-lap, lap 0, is index 0 when present) and ``t_cum_s`` (rolling time
    cumsum within the stint, at the end of each lap)."""
    df = laps.sort_values(["session_id", "stint_id", "lap_num"]).copy()
    df["_roll_s"] = _rolling_time_s(df)
    grouped = df.groupby(["session_id", "stint_id"], sort=False)
    df["lap_within_stint"] = grouped.cumcount()
    df["t_cum_s"] = grouped["_roll_s"].cumsum()
    df = df.drop(columns=["_roll_s"])
    return df.reset_index(drop=True)


def _compute_stint_anchor(laps: pd.DataFrame) -> pd.DataFrame:
    """Per (session, stint, corner): the stint's initial condition.

    Adds ``t_anchor_{c}`` (rolling seconds, same clock as ``t_cum_s``),
    ``t_start_{c}`` (°C), ``p_start_{c}`` (gauge bar at the same reading, for
    the pressure-domain evaluation) and ``anchor_kind_{c}``:

    - ``"pit_exit"``: the stint begins with an out-lap that left the pits
      and the channel's first valid reading in it — the temperature and
      the cold pressure the driver actually set. Its time on the rolling
      clock is the reading's lap-relative time minus the standstill before
      the car moved (clipped at 0).
    - ``"first_lap"``: no usable out-lap; the first finite reading of the
      first lap(s), the pre-schema-v3 behaviour. The warmup curve is integrated from that point:
    ``T(t) = T_eff + K·c·g²·(1 − e^{−(t−t_a)/τ}) + (T_a − T_eff)·e^{−(t−t_a)/τ}``.

    Candidates in order: the first lap's start reading (t_a = 0), its end
    reading (t_a = on_track_s), then the next lap's start/end, and so on —
    TPMS channels are NaN for the first samples of nearly every session
    (stale-prefix masking in the ETL), so the anchor is often a few seconds
    to one lap in. Laps at or before the anchor cannot be scored and are
    dropped by the fit.
    """
    df = laps.sort_values(["session_id", "stint_id", "lap_num"]).copy()
    df["_roll_s"] = _rolling_time_s(df)
    df["_is_out"] = _is_outlap_col(df)
    df["_from_pit"] = (
        df["outlap_from_pit"].fillna(False).astype(bool)
        if "outlap_from_pit" in df.columns
        else False
    )
    for c in CORNERS:
        df[f"t_anchor_{c}"] = np.nan
        df[f"t_start_{c}"] = np.nan
        df[f"p_start_{c}"] = np.nan
        df[f"anchor_kind_{c}"] = None
    for (_sid, _stint), grp in df.groupby(["session_id", "stint_id"], sort=False):
        first = grp.iloc[0]
        for c in CORNERS:
            start_col, end_col = f"tpms_temp_{c}_start", f"tpms_temp_{c}_end"
            p_start_col, p_end_col = f"tpms_press_{c}_start", f"tpms_press_{c}_end"
            anchor: tuple[float, float, float] | None = None
            kind = "first_lap"
            if (
                bool(first["_is_out"])
                and bool(first["_from_pit"])
                and start_col in grp.columns
                and pd.notna(first[start_col])
            ):
                fv_col = f"tpms_temp_{c}_first_valid_s"
                first_valid = float(first[fv_col]) if fv_col in grp.columns else 0.0
                if not np.isfinite(first_valid):
                    first_valid = 0.0
                wait_s = max(float(first["on_track_s"]) - float(first["_roll_s"]), 0.0)
                t_read = max(first_valid - wait_s, 0.0)
                p_val = first[p_start_col] if p_start_col in grp.columns else np.nan
                anchor = (
                    t_read,
                    float(first[start_col]),
                    float(p_val) if pd.notna(p_val) else np.nan,
                )
                kind = "pit_exit"
            for row in grp.itertuples(index=False):
                if anchor is not None:
                    break
                t_end = float(getattr(row, "t_cum_s"))
                t_begin = max(t_end - float(getattr(row, "on_track_s")), 0.0)
                t_s = getattr(row, start_col, np.nan) if start_col in grp.columns else np.nan
                t_e = getattr(row, end_col, np.nan)
                if pd.notna(t_s):
                    p_s = (
                        getattr(row, p_start_col, np.nan) if p_start_col in grp.columns else np.nan
                    )
                    anchor = (t_begin, float(t_s), float(p_s) if pd.notna(p_s) else np.nan)
                    break
                if pd.notna(t_e):
                    p_e = getattr(row, p_end_col, np.nan) if p_end_col in grp.columns else np.nan
                    anchor = (t_end, float(t_e), float(p_e) if pd.notna(p_e) else np.nan)
                    break
            if anchor is not None:
                df.loc[grp.index, f"t_anchor_{c}"] = anchor[0]
                df.loc[grp.index, f"t_start_{c}"] = anchor[1]
                df.loc[grp.index, f"p_start_{c}"] = anchor[2]
                df.loc[grp.index, f"anchor_kind_{c}"] = kind
    df = df.drop(columns=["_roll_s", "_is_out", "_from_pit"])
    return df.reset_index(drop=True)


def _anchor_terms(df: pd.DataFrame, corner: str) -> tuple[np.ndarray, np.ndarray]:
    """``(t_anchor, start_excess)`` arrays for ``corner``; the excess is
    ``T_start − T_eff``. Frames without anchor columns (synthetic tests,
    legacy callers) get the v0 behaviour: anchor at t = 0 with T_start = T_eff."""
    n = len(df)
    if f"t_anchor_{corner}" not in df.columns or f"t_start_{corner}" not in df.columns:
        return np.zeros(n), np.zeros(n)
    t_a = df[f"t_anchor_{corner}"].to_numpy(dtype=float)
    excess = df[f"t_start_{corner}"].to_numpy(dtype=float) - df["t_eff_c"].to_numpy(dtype=float)
    return t_a, excess


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
    """NaN out tpms_temp_{c}_end / _start for each (session_id, corner) in the blacklist.

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
        start_col = f"tpms_temp_{corner}_start"
        if start_col in df.columns:
            df.loc[mask, start_col] = np.nan
    if applied:
        logger.info(
            "Applied user-confirmed sensor blacklist: %d (session, corner) channels masked",
            len(applied),
        )
    return df, applied


def gas_temperature_c(
    t_anchor_c: np.ndarray, p_anchor_bar: np.ndarray, p_bar: np.ndarray
) -> np.ndarray:
    """Cavity-gas temperature implied by pressure, from an anchor state.

    Constant-volume ideal gas: ``T_gas_K = T_anchor_K · P_abs / P_anchor_abs``.
    The anchor is the stint's pit-exit reading, where the tire has rested
    long enough for the separate temperature measurement and the gas to
    agree. The pressure channel responds to the tread within seconds while
    the valve-mounted TPMS temperature lags it by 2–3 minutes, and pressure
    is what the calculator predicts — so this, not the TPMS temperature, is
    the model's observable (2026-10). NaN where any input is missing.
    """
    t_k = np.asarray(t_anchor_c, dtype=float) + T_ZERO_C_TO_K
    p_a = np.asarray(p_anchor_bar, dtype=float) + P_ATM_BAR
    p = np.asarray(p_bar, dtype=float) + P_ATM_BAR
    with np.errstate(invalid="ignore", divide="ignore"):
        out = t_k * p / p_a - T_ZERO_C_TO_K
    ok = np.isfinite(t_k) & np.isfinite(p_a) & np.isfinite(p) & (p_a > 0.3) & (p > 0.3)
    out = np.where(ok, out, np.nan)
    return np.asarray(out, dtype=float)


def _compute_delta_t(laps: pd.DataFrame) -> pd.DataFrame:
    """Compute T_road proxy, the pressure-implied gas temperature at lap end
    (``t_gas_{c}_end``) and the regression target
    ``delta_t_{c} = t_gas_{c}_end − T_eff``.

    Falls back to the TPMS end temperature for frames without anchor
    pressures (synthetic tests, pre-schema-v3 data)."""
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
        if f"p_start_{c}" in df.columns and f"tpms_press_{c}_end" in df.columns:
            df[f"t_gas_{c}_end"] = gas_temperature_c(
                df[f"t_start_{c}"].to_numpy(dtype=float),
                df[f"p_start_{c}"].to_numpy(dtype=float),
                df[f"tpms_press_{c}_end"].to_numpy(dtype=float),
            )
        else:
            df[f"t_gas_{c}_end"] = df[col]
        df[f"delta_t_{c}"] = df[f"t_gas_{c}_end"] - df["t_eff_c"]
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


def _build_outlap_typ(
    laps: pd.DataFrame,
) -> dict[tuple[str, str, str], tuple[float, float, int]]:
    """Typical out-lap per (track, car, condition): median rolling time and
    median g² (``heat_proxy / moving_s``) over from-pit out-laps.

    The calculator integrates this segment first, from the typed pit-exit
    temperature, before the N flying laps. Returns
    ``{(track, car, condition): (moving_s, g2, n)}``; empty when the dataset
    predates schema v3.
    """
    out: dict[tuple[str, str, str], tuple[float, float, int]] = {}
    if "is_outlap" not in laps.columns or "moving_s" not in laps.columns:
        return out
    from_pit = (
        laps["outlap_from_pit"].fillna(False).astype(bool)
        if "outlap_from_pit" in laps.columns
        else True
    )
    outs = laps[_is_outlap_col(laps) & from_pit]
    for (track, car, cond), grp in outs.groupby(["track_canonical", "car", "condition"]):
        if cond == "unknown":
            continue
        mv = grp["moving_s"].astype(float)
        g2 = (grp["heat_proxy"] / mv).replace([np.inf, -np.inf], np.nan)
        ok = mv.notna() & (mv > 0) & g2.notna()
        if ok.sum() == 0:
            continue
        out[(str(track), str(car), str(cond))] = (
            float(mv[ok].median()),
            float(g2[ok].median()),
            int(ok.sum()),
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


def _build_corner_defaults(
    laps: pd.DataFrame,
    *,
    min_lap_within_stint: int = CORNER_DEFAULTS_MIN_LAP_IN_STINT,
    min_laps: int = CORNER_DEFAULTS_MIN_LAPS,
) -> dict[tuple[str, str, str], tuple[float, float, int]]:
    """Median steady-state hot temp/pressure per (car, corner, condition).

    UI prefill values: what this car's tires actually settle at once warm
    (``lap_within_stint >= min_lap_within_stint``). The hot temperature is
    the pressure-implied gas temperature (the model's observable), not the
    TPMS display. Blacklisted corners are already NaN-masked upstream, so
    they drop out of the medians.

    Returns ``{(car, corner, condition): (hot_temp_c, hot_pressure_bar, n)}``.
    """
    steady = laps[laps["lap_within_stint"] >= min_lap_within_stint]
    steady = steady[steady["condition"] != "unknown"]
    out: dict[tuple[str, str, str], tuple[float, float, int]] = {}
    for (car, cond), grp in steady.groupby(["car", "condition"]):
        for c in CORNERS:
            temp_col = f"t_gas_{c}_end" if f"t_gas_{c}_end" in grp.columns else f"tpms_temp_{c}_end"
            paired = grp[[temp_col, f"tpms_press_{c}_mean"]].dropna()
            if len(paired) < min_laps:
                continue
            out[(str(car), c, str(cond))] = (
                float(paired[temp_col].median()),
                float(paired[f"tpms_press_{c}_mean"].median()),
                int(len(paired)),
            )
    return out


def _laps_for_fit(
    laps: pd.DataFrame,
    g2_lookup: dict[tuple[str, str, str], tuple[float, int]],
) -> pd.DataFrame:
    """Drop rows without a valid t_eff, g², or known condition.

    The first lap of a stint is kept: with the stint anchored on its first
    finite TPMS reading it is a legitimate warmup observation (the fit drops
    per-corner rows at or before the anchor itself)."""
    df = laps.copy()
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
    laps_for_fit: pd.DataFrame,
    car: str,
    corner: str,
    condition: str,
    *,
    tau_upper: float | None = None,
) -> tuple[FitParam, dict[str, FitParam]]:
    """Fit τ_sec[car, corner, condition] jointly across that car's (track) buckets
    in the given condition.

    The closed-form warmup model uses **per-lap g²** as a known feature and
    the stint's first finite TPMS reading as the initial condition:
    ``ΔT_i = (K · c_track) · g²_i · (1 - exp(-Δt_i / τ)) + (T_a − T_eff) · exp(-Δt_i / τ)``,
    with ``Δt_i = t_i − t_a`` and g²_i = ``heat_proxy_i / on_track_s_i`` for
    lap i (see :func:`_compute_stint_anchor`). The fitted "gain" per bucket
    is therefore ``K · c_track`` (no ⟨g²⟩ factor); Pass 2 decomposes it
    into the per-car K and per-track c_track without the prior division
    by a bucket statistic. Laps with higher actual g² get a higher
    asymptote, which matches the field observation that on-pace laps run
    hotter than the bucket median.

    ``tau_upper`` bounds τ from above *inside* the fit (used for rain
    buckets: τ_rain ≤ τ_dry), so the gains are estimated consistently with
    the bound rather than clipped afterwards.

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
    t_anchor, start_excess = _anchor_terms(car_df, corner)
    car_df["_dt"] = car_df["t_cum_s"].to_numpy(dtype=float) - t_anchor
    car_df["_excess"] = start_excess
    car_df = car_df[
        car_df["g2_lap"].notna()
        & (car_df["g2_lap"] > 0)
        & car_df["_dt"].notna()
        & (car_df["_dt"] > 0)
        & car_df["_excess"].notna()
    ]
    if car_df.empty:
        return (FitParam(PRIOR_TAU_SEC, 0.0, 0, from_prior=True), {})

    # Build per-bucket arrays
    buckets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for track, grp in car_df.groupby("track_canonical"):
        if len(grp) < MIN_LAPS_FOR_TAU_FIT:
            continue
        if condition != "dry" and grp["session_id"].nunique() < MIN_SESSIONS_FOR_RAIN_FIT:
            continue  # one rainy day cannot carry its own τ/K; fall back to dry
        buckets.append(
            (
                str(track),
                grp["_dt"].to_numpy(),
                grp[delta_col].to_numpy(),
                grp["g2_lap"].to_numpy(),
                grp["_excess"].to_numpy(),
            )
        )

    if not buckets:
        return (FitParam(PRIOR_TAU_SEC, 0.0, 0, from_prior=True), {})

    n_buckets = len(buckets)
    # Concatenate all samples; remember which bucket each belongs to
    all_t = np.concatenate([b[1] for b in buckets])
    all_y = np.concatenate([b[2] for b in buckets])
    all_g2 = np.concatenate([b[3] for b in buckets])
    all_excess = np.concatenate([b[4] for b in buckets])
    bucket_idx = np.concatenate([np.full(len(b[1]), i, dtype=int) for i, b in enumerate(buckets)])

    def model(_x: np.ndarray, *params: float) -> np.ndarray:
        # params: tau_sec, Kc_0, Kc_1, ..., Kc_{n_buckets-1}
        # where Kc_b = K[car, corner, cond] · c_track[track of bucket b]
        tau = params[0]
        Kc = np.array(params[1:])
        decay = np.exp(-all_t / tau)
        out: np.ndarray = Kc[bucket_idx] * all_g2 * (1.0 - decay) + all_excess * decay
        return out

    # Initial Kc guess: ΔT_∞ / typical g² ≈ 40 / 0.5 ≈ 80 K/G²
    tau_hi = 1200.0 if tau_upper is None else max(61.0, min(1200.0, float(tau_upper)))
    p0 = [min(PRIOR_TAU_SEC, 0.9 * tau_hi)] + [PRIOR_K_KELVIN_PER_G2] * n_buckets
    bounds_lower = [60.0] + [0.0] * n_buckets
    bounds_upper = [tau_hi] + [500.0] * n_buckets

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
    for i, (track, _t, _y, _g2, _ex) in enumerate(buckets):
        per_bucket[track] = FitParam(
            value=float(popt[1 + i]),
            stderr=float(perr[1 + i]),
            n_samples=int(len(buckets[i][1])),
        )
    return tau_fit, per_bucket


def _rain_tau_upper(
    tau_by_car_corner_cond: dict[tuple[str, str, str], FitParam],
    car: str,
    corner: str,
    condition: str,
) -> float | None:
    """τ_rain ≤ τ_dry (physics: rain only adds cooling). None for dry or when
    the dry τ is itself a prior."""
    if condition == "dry":
        return None
    dry = tau_by_car_corner_cond.get((car, corner, "dry"))
    if dry is None or dry.from_prior:
        return None
    return dry.value


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


def _data_through_for_fit(laps: "pd.DataFrame") -> tuple[str | None, str | None]:
    """(date, local datetime) of the newest session that fed the fit.

    Returns ``("2026-10-02", "2026-10-02 14:23 JST")``: the first is the
    track-local calendar date of the newest session, the second its start time
    rendered in the track's time zone for display. Both None when no laps.
    """
    if laps is None or len(laps) == 0 or "date" not in laps.columns:
        return None, None
    df = laps.dropna(subset=["date"])
    if df.empty:
        return None, None
    df = df.assign(_d=df["date"].map(lambda d: str(d)[:10]))
    newest_date = str(df["_d"].max())
    local_str: str | None = None
    if "session_start_utc" in df.columns:
        from ..tire_etl.tracks import get_track

        newest = df[df["_d"] == newest_date].dropna(subset=["session_start_utc"])
        if not newest.empty:
            row = newest.sort_values("session_start_utc").iloc[-1]
            ts = pd.Timestamp(row["session_start_utc"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ti = get_track(str(row.get("track_canonical", "")))
            if ti is not None:
                ts = ts.tz_convert(ti.timezone)
            local_str = ts.strftime("%Y-%m-%d %H:%M %Z")
    return newest_date, local_str


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
    compound_multipliers: dict[str, dict[str, float]] | None = None,
    corner_defaults: dict[tuple[str, str, str], tuple[float, float, int]] | None = None,
    data_through_date: str | None = None,
    data_through_local: str | None = None,
    outlap_lookup: dict[tuple[str, str, str], tuple[float, float, int]] | None = None,
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
    compound_multipliers = compound_multipliers or {}
    corner_defaults = corner_defaults or {}

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
        # Newest session that fed the fit: its track-local calendar date and
        # a display-ready local start time ("2026-10-02 14:23 JST"). Shown
        # in the calculators' footer so a user can tell whether the build in
        # front of them has the latest data.
        "data_through_date": data_through_date,
        "data_through_local": data_through_local,
        # Raw car label -> pooled fit label. Predictors resolve an input car
        # through this map before any lookup, so old car names keep working.
        "car_aliases": dict(CAR_FIT_ALIASES),
        "model_form": (
            "T_hot - T_eff = K[car,corner,cond] * c_track[track] * g2 "
            "* (1 - exp(-t / tau_sec[car,corner,cond])) "
            "+ (T_start - T_eff) * exp(-t / tau_sec[car,corner,cond]), where T is the "
            "pressure-implied cavity-gas temperature T_K = T_start_K * P_abs / P_start_abs "
            "(so P_hot_abs = P_start_abs * T_hot_K / T_start_K exactly), integrated in two "
            "segments: the out-lap (outlap_moving_s at outlap_g2) from T_start = the pit-exit "
            "tire temperature (fit: first valid TPMS reading of the out-lap; predict: the "
            "entered current tire temp, default T_air), then N flying laps (t = N * lap_time_s) "
            "from the temperature at the end of the out-lap. T_eff = (1-w_road)*T_air + w_road*T_road, "
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
            "observable": (
                "pressure-implied cavity-gas temperature: T_gas_K = T_start_K * P_abs / "
                "P_start_abs from the stint's pit-exit (T, P). The TPMS temperature lags "
                "the gas by 2-3 min and is used only as the initial condition."
            ),
            "initial_condition": {
                "fit": "first valid TPMS reading of the out-lap at pit exit (t_anchor, T_start, P_start)",
                "predict": "cold_tire_temp_c (a separate measurement at standstill), default T_air",
            },
            "t_road_proxy": {
                "formula": "T_air + delta_sun_max_c * (1 - cloud_cover/100) * sun_factor",
                "delta_sun_max_c": DELTA_SUN_MAX_C,
                "sun_factor_default": SUN_FACTOR_DEFAULT,
            },
        },
        "rain_thermal": {
            "method": (
                "tau/K fitted per condition; tau_rain <= tau_dry bounded inside the fit "
                "(rain only adds cooling), K free (a rain compound has its own hysteresis). "
                "Rain buckets with fewer sessions fall back to dry via the condition chain."
            ),
            "min_sessions_for_rain_fit": MIN_SESSIONS_FOR_RAIN_FIT,
        },
        "conditions": {
            "values": list(CONDITIONS),
            "default": DEFAULT_CONDITION,
            "classification": {
                "from_field": "track_wetness_run_max_mm",
                "method": "surface water balance over the hourly weather (tire_model/wetness.py)",
                "formula": (
                    "d(t+1h) = clamp(d(t) + P - E, 0, surface_storage_mm); "
                    "E = evap_coeff * VPD(T_surface, RH) * (1 + 0.54 * wind_m_s), "
                    "T_surface = T_air + surface_sun_excess_c * (1 - cloud/100); "
                    "P is Open-Meteo's preceding-hour precipitation"
                ),
                "params": {
                    "surface_storage_mm": _wetness.SURFACE_STORAGE_MM,
                    "evap_coeff_mm_per_h_kpa": _wetness.EVAP_COEFF_MM_PER_H_KPA,
                    "wind_factor_per_m_s": _wetness.WIND_FACTOR_PER_M_S,
                    "surface_sun_excess_c": _wetness.SURFACE_SUN_EXCESS_C,
                },
                "thresholds": {
                    "damp_min_mm": _wetness.DAMP_FILM_MM,
                    "wet_min_mm": _wetness.WET_FILM_MM,
                },
                "rule": (
                    "max film depth over the run < damp_min → dry; ≥ damp_min → damp; "
                    "≥ wet_min → wet; no weather coverage → unknown (excluded from training)"
                ),
                "legacy_precipitation_rule_mm_hr": {
                    "dry_max": CONDITION_DRY_MAX_PRECIP_MM_HR,
                    "damp_max": CONDITION_DAMP_MAX_PRECIP_MM_HR,
                },
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
        # Compound decomposition: K_effective = c_track × K_base × m[compound].
        # The multipliers document the fitted per-compound ratios; the table
        # below carries the ready-to-use products.
        "K_compound_multipliers": [
            {"car": car, "compound": comp, "multiplier": mult}
            for car, comps in sorted(compound_multipliers.items())
            for comp, mult in sorted(comps.items())
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
        # UI prefill medians: what the tires settle at once warm (additive;
        # calculators use these to seed the target hot temp / hot pressure
        # inputs when the car or condition selection changes).
        "corner_defaults_by_car_corner_cond": [
            {
                "car": car,
                "corner": corner,
                "condition": cond,
                "hot_temp_c": temp,
                "hot_pressure_bar": press,
                "n_laps_used": n,
            }
            for (car, corner, cond), (temp, press, n) in sorted(corner_defaults.items())
        ],
        # Typical out-lap (pit exit to the first start/finish crossing):
        # rolling time and g², integrated first from the typed pit-exit
        # temperature. Consumers without this table treat the out-lap as
        # zero-length (pre-v0.26 behaviour).
        "outlap_typ_by_track_car_cond": [
            {
                "track_canonical": track,
                "car": car,
                "condition": cond,
                "outlap_moving_s": mv,
                "outlap_g2": g2,
                "n_laps_used": n,
            }
            for (track, car, cond), (mv, g2, n) in sorted((outlap_lookup or {}).items())
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
