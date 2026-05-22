"""Validation utilities for the tire warmup model.

Two distinct validations:

- :func:`run_validation` — compares predicted cold pressures against
  notes-recorded cold pressures. Uses the production (full-data) model,
  so this is a *consistency* check, not a held-out test.
- :func:`run_holdout_validation` — held-out test. Excludes a few sessions
  from training, then predicts per-lap T_hot for those sessions and
  reports per-corner per-lap residuals. This is the honest measure of
  generalization.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..tire_etl.paths import default_dataset_root, laps_dir, sessions_dir
from .energy_balance import (
    t_effective_c,
    t_road_proxy_c,
    warmup_curve_c,
)
from .predict import CORNERS, predict_cold_pressure
from .warmup_table import (
    CORNERS as _WT_CORNERS,
    W_ROAD,
    build_warmup_table,
)

logger = logging.getLogger(__name__)


def run_validation(dataset_root: Path | None = None) -> int:
    root = Path(dataset_root) if dataset_root else default_dataset_root()
    notes_path = root / "notes_matches.parquet"
    if not notes_path.exists():
        logger.error("No notes_matches.parquet at %s", notes_path)
        return 1
    notes = pq.read_table(notes_path).to_pandas()
    sessions = pd.concat(
        [pq.read_table(f).to_pandas() for f in sorted(sessions_dir(root).glob("*.parquet"))],
        ignore_index=True,
    )
    laps = pd.concat(
        [pq.read_table(f).to_pandas() for f in sorted(laps_dir(root).glob("*.parquet"))],
        ignore_index=True,
    )

    # Merge notes -> sessions for track/car/ambient
    merged = notes.merge(
        sessions[["session_id", "track_canonical", "car", "status", "has_tpms"]],
        on="session_id",
        how="left",
    )
    merged = merged[
        (merged["status"] == "ok") & merged["has_tpms"] & merged["track_canonical"].notna()
    ]

    # Pre-cache model
    with (root / "tire_model.json").open() as f:
        model = json.load(f)

    rows: list[dict] = []
    for _, row in merged.iterrows():
        actual = {c: row.get(f"cold_pressure_bar_{c}") for c in CORNERS}
        if any(pd.isna(v) for v in actual.values()):
            continue
        # Target hot pressure: median of tpms_press_{c}_mean over mid-stint laps of this session
        sess_laps = laps[(laps["session_id"] == row["session_id"]) & laps["tire_usable"]]
        if sess_laps.empty:
            continue
        hot_targets: dict[str, float] = {}
        skip = False
        for c in CORNERS:
            col = f"tpms_press_{c}_mean"
            vals = sess_laps[col].dropna()
            if vals.empty:
                skip = True
                break
            hot_targets[c] = float(vals.median())
        if skip:
            continue
        # Use lap 5 within stint as the representative warm point (or the median lap_within_stint)
        # Use ambient from notes if present; else weather-derived from laps
        ambient = row.get("ambient_temp_c")
        if pd.isna(ambient):
            continue
        try:
            pred = predict_cold_pressure(
                track=row["track_canonical"],
                car=row["car"],
                lap_within_stint=5,
                target_hot_pressure_bar=hot_targets,
                ambient_temp_c=float(ambient),
                dataset_root=root,
                _model=model,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping session %s: %s", row["session_id"], e)
            continue
        record = {
            "session_id": row["session_id"],
            "track": row["track_canonical"],
            "car": row["car"],
            "ambient_c": float(ambient),
        }
        for c in CORNERS:
            record[f"actual_{c}"] = float(actual[c])
            record[f"pred_{c}"] = float(pred[c].cold_pressure_bar)
            record[f"resid_{c}"] = float(pred[c].cold_pressure_bar - actual[c])
        rows.append(record)

    if not rows:
        print("No validation rows produced (no notes had complete actual + hot-pressure data).")
        return 0
    df = pd.DataFrame(rows)
    print(f"Validation set: {len(df)} sessions\n")
    for c in CORNERS:
        resid = df[f"resid_{c}"]
        mae = float(resid.abs().mean())
        mean_bias = float(resid.mean())
        print(
            f"  {c.upper():3}  MAE = {mae:.3f} bar    "
            f"mean signed residual = {mean_bias:+.3f} bar    "
            f"(n={len(resid)})"
        )

    print("\nPer-session breakdown (first 15):")
    cols = ["session_id", "track", "car", "ambient_c"]
    for c in CORNERS:
        cols += [f"actual_{c}", f"pred_{c}", f"resid_{c}"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(df[cols].head(15).to_string(index=False))
    return 0


# ---------- Held-out (true generalization) validation ----------


def _pick_holdout_sessions(
    sessions: pd.DataFrame,
    laps: pd.DataFrame,
    n_per_bucket: int = 2,
    min_bucket_size: int = 10,
    fold: int = 0,
) -> list[str]:
    """Pick deterministic held-out session_ids from each (track, car) bucket
    that has enough sessions to afford excluding ``n_per_bucket`` without
    breaking the fit.

    Sessions are sorted by session_id (stable hash) within each bucket and
    ``fold`` selects which contiguous slice of size ``n_per_bucket`` to hold
    out. Different folds produce disjoint slices, so a k-fold CV sweeps
    every session through the held-out set exactly once (until the bucket
    runs out, at which point that bucket is silently skipped for later
    folds).
    """
    ok = sessions[(sessions["status"] == "ok") & sessions["has_tpms"]]
    ok = ok[ok["track_canonical"].notna() & ok["car"].notna()]
    # Restrict to sessions that have at least 3 usable laps with TPMS data
    usable_per_session = (
        laps[laps["tire_usable"]].groupby("session_id").size().rename("n_usable_laps").reset_index()
    )
    ok = ok.merge(usable_per_session, on="session_id", how="left")
    ok = ok[ok["n_usable_laps"].fillna(0) >= 3]

    held_out: list[str] = []
    start = fold * n_per_bucket
    stop = start + n_per_bucket
    for (track, car), grp in ok.groupby(["track_canonical", "car"]):
        if len(grp) < min_bucket_size:
            continue
        ordered = sorted(grp["session_id"].tolist())
        if start >= len(ordered):
            continue  # this bucket has been exhausted by earlier folds
        held_out.extend(ordered[start:stop])
    return held_out


def _evaluate_fold(root: Path, holdout_ids: list[str]) -> pd.DataFrame:
    """Train a model excluding ``holdout_ids`` and return per-(lap, corner)
    residual rows for the held-out sessions."""
    model = build_warmup_table(root, exclude_session_ids=set(holdout_ids), write_artifacts=False)

    # Build per-(track, car) lookups from the held-out model
    g2_lookup = {
        (d["track_canonical"], d["car"], d["condition"]): d["g2_typ"]
        for d in model["g2_typ_by_track_car_cond"]
    }
    c_track_lookup = {d["track_canonical"]: d["value"] for d in model["c_track_by_track"]}
    k_lookup = {
        (d["key"]["car"], d["key"]["corner"], d["key"]["condition"]): d["value_kelvin_per_g2"]
        for d in model["K_buckets"]
    }
    tau_lookup = {
        (d["car"], d["corner"], d["condition"]): d["value_seconds"]
        for d in model["tau_sec_by_car_corner_cond"]
    }

    # Build (session_id → ambient_temp_c) from weather attached during training prep.
    # Simpler: re-run the same weather lookup logic for held-out sessions only.
    from .warmup_table import (
        _apply_blacklist,
        _attach_weather,
        _compute_delta_t,
        _compute_stint_clock,
        _load_filtered_laps,
        _load_weather,
        load_sensor_blacklist,
    )

    all_laps = _load_filtered_laps(root)
    all_laps = all_laps[all_laps["session_id"].isin(holdout_ids)].copy()
    if all_laps.empty:
        return pd.DataFrame()
    weather = _load_weather(root)
    all_laps = _attach_weather(all_laps, weather)
    all_laps = _compute_stint_clock(all_laps)
    # Apply the same blacklist used at training — don't grade predictions
    # against channels we already know are broken. Held-out laps are a
    # subset, so most blacklist entries won't match — silence that noise.
    blacklist_pairs = load_sensor_blacklist(root)
    all_laps, _ = _apply_blacklist(all_laps, blacklist_pairs, warn_on_unknown=False)
    all_laps = _compute_delta_t(all_laps)
    # Drop out-laps from evaluation (same convention as training)
    all_laps = all_laps[all_laps["lap_within_stint"] > 0].reset_index(drop=True)

    # Per-lap predictions
    rows: list[dict] = []
    for _, lap in all_laps.iterrows():
        track = lap["track_canonical"]
        car = lap["car"]
        cond = lap.get("condition", "dry")
        if cond == "unknown":
            cond = "dry"  # held-out evaluation defaults to dry when condition unknown
        t_cum_s = float(lap["t_cum_s"])
        t_eff = float(lap["t_eff_c"]) if pd.notna(lap["t_eff_c"]) else None
        if t_eff is None:
            continue
        g2 = g2_lookup.get((track, car, cond)) or g2_lookup.get((track, car, "dry"))
        if g2 is None:
            continue
        c_track = c_track_lookup.get(track, 1.0)
        for c in CORNERS:
            K = k_lookup.get((car, c, cond)) or k_lookup.get((car, c, "dry"))
            tau = tau_lookup.get((car, c, cond)) or tau_lookup.get((car, c, "dry"))
            if K is None or tau is None:
                continue
            obs = lap.get(f"tpms_temp_{c}_end")
            if pd.isna(obs):
                continue
            t_hot_pred = warmup_curve_c(
                t_seconds=t_cum_s,
                t_eff_c=t_eff,
                k_kelvin_per_g2=K,
                c_track=c_track,
                g2_typ=g2,
                tau_sec=tau,
            )
            rows.append(
                {
                    "session_id": lap["session_id"],
                    "track": track,
                    "car": car,
                    "condition": cond,
                    "lap_num": int(lap["lap_num"]),
                    "stint_id": int(lap["stint_id"]),
                    "lap_within_stint": int(lap["lap_within_stint"]),
                    "t_cum_s": t_cum_s,
                    "corner": c,
                    "T_hot_pred_c": t_hot_pred,
                    "T_hot_obs_c": float(obs),
                    "resid_c": t_hot_pred - float(obs),
                }
            )
    return pd.DataFrame(rows)


def _print_corner_table(label: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    print(f"\n=== {label} ({len(frame)} (lap × corner) points) ===")
    for c in CORNERS:
        sub = frame[frame["corner"] == c]
        if sub.empty:
            continue
        mae = float(sub["resid_c"].abs().mean())
        rmse = float(np.sqrt((sub["resid_c"] ** 2).mean()))
        bias = float(sub["resid_c"].mean())
        print(
            f"  {c.upper():>3}   MAE = {mae:>5.2f} °C   RMSE = {rmse:>5.2f} °C   "
            f"mean bias = {bias:+.2f} °C    n = {len(sub)}"
        )


def _print_summary(df: pd.DataFrame, *, summary_label: str = "Held-out") -> None:
    _print_corner_table(f"{summary_label} per-corner T_hot residuals — POOLED", df)
    for car_name, car_frame in df.groupby("car"):
        _print_corner_table(f"{summary_label} — {car_name}", car_frame)
    for cond_name, cond_frame in df.groupby("condition"):
        _print_corner_table(f"{summary_label} — condition={cond_name}", cond_frame)


def run_holdout_validation(
    dataset_root: Path | None = None,
    *,
    n_per_bucket: int = 2,
    min_bucket_size: int = 10,
    n_folds: int = 1,
) -> int:
    """Train on all-minus-held-out, predict per-lap T_hot for held-out sessions.

    With ``n_folds == 1`` (default) this is a single deterministic holdout
    — the legacy behavior. With ``n_folds > 1`` it sweeps disjoint
    n_per_bucket-sized slices through every bucket as k-fold CV: refits k
    times, predicts on each fold's held-out set, then aggregates residuals
    across all folds so every session appears in the held-out set roughly
    once. Useful when a single fold's MAE is too noisy to compare model
    revisions (e.g. tiny per-bucket holdouts).
    """
    root = Path(dataset_root) if dataset_root else default_dataset_root()
    sessions = pd.concat(
        [pq.read_table(f).to_pandas() for f in sorted(sessions_dir(root).glob("*.parquet"))],
        ignore_index=True,
    )
    laps = pd.concat(
        [pq.read_table(f).to_pandas() for f in sorted(laps_dir(root).glob("*.parquet"))],
        ignore_index=True,
    )

    fold_frames: list[pd.DataFrame] = []
    total_holdouts = 0
    for fold in range(max(1, n_folds)):
        holdout_ids = _pick_holdout_sessions(
            sessions,
            laps,
            n_per_bucket=n_per_bucket,
            min_bucket_size=min_bucket_size,
            fold=fold,
        )
        if not holdout_ids:
            if fold == 0:
                print("No (track, car) bucket has enough sessions to hold out cleanly.")
                return 1
            # No more buckets have unused sessions for this fold; stop.
            break
        print(
            f"Fold {fold + 1}/{n_folds}: holding out {len(holdout_ids)} sessions"
            if n_folds > 1
            else f"Holding out {len(holdout_ids)} sessions "
            f"({n_per_bucket} per bucket, min bucket size = {min_bucket_size})"
        )
        fold_df = _evaluate_fold(root, holdout_ids)
        if not fold_df.empty:
            fold_df["fold"] = fold
            fold_frames.append(fold_df)
        total_holdouts += len(holdout_ids)

    if not fold_frames:
        print("No predictable laps in any held-out fold.")
        return 0
    df = pd.concat(fold_frames, ignore_index=True)

    label = "Held-out" if n_folds <= 1 else f"{n_folds}-fold CV"
    if n_folds > 1:
        unique_sessions = df["session_id"].nunique()
        print(
            f"\n{n_folds}-fold CV: {total_holdouts} (session × fold) holdouts → "
            f"{unique_sessions} unique sessions evaluated"
        )
    _print_summary(df, summary_label=label)

    # Per-(session, lap) table — only useful for a single fold; CV mode skips
    # it because dumping residuals for every session in every fold is noise.
    if n_folds <= 1:
        print("\n=== Per-(session, lap) breakdown ===")
        pivot = df.pivot_table(
            index=["session_id", "track", "car", "lap_num", "lap_within_stint", "t_cum_s"],
            columns="corner",
            values=["T_hot_pred_c", "T_hot_obs_c", "resid_c"],
        )
        pivot.columns = [f"{tup[0]}_{tup[1]}" for tup in pivot.columns]
        pivot = pivot.reset_index().sort_values(["session_id", "lap_num"])
        pd.set_option("display.width", 240)
        pd.set_option("display.max_columns", 30)
        pd.set_option("display.float_format", lambda x: f"{x:.1f}")
        cols = ["session_id", "track", "car", "lap_num", "lap_within_stint", "t_cum_s"]
        for c in CORNERS:
            cols += [f"T_hot_obs_c_{c}", f"T_hot_pred_c_{c}", f"resid_c_{c}"]
        print(pivot[cols].to_string(index=False))
    return 0
