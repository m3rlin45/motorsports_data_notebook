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
    P_ATM_BAR,
    T_ZERO_C_TO_K,
    t_effective_c,
    t_road_proxy_c,
    warmup_curve_c,
    warmup_two_stage_c,
)
from .predict import (
    CORNERS,
    _g2_pace_scale,
    _lookup_g2,
    _lookup_lap_time,
    _lookup_outlap,
    predict_cold_pressure,
)
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


def _load_sessions_and_laps(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = pd.concat(
        [pq.read_table(f).to_pandas() for f in sorted(sessions_dir(root).glob("*.parquet"))],
        ignore_index=True,
    )
    laps = pd.concat(
        [pq.read_table(f).to_pandas() for f in sorted(laps_dir(root).glob("*.parquet"))],
        ignore_index=True,
    )
    return sessions, laps


def _collect_holdout_frame(
    root: Path,
    *,
    n_per_bucket: int,
    min_bucket_size: int,
    n_folds: int,
    inputs: str = "calculator",
    quiet: bool = False,
) -> tuple[pd.DataFrame | None, int]:
    """Run the k-fold holdout and return ``(residual rows, n session×fold
    holdouts)``. ``None`` rows means no bucket could be held out."""
    sessions, laps = _load_sessions_and_laps(root)
    session_condition = _session_conditions(root)
    fold_frames: list[pd.DataFrame] = []
    total_holdouts = 0
    for fold in range(max(1, n_folds)):
        holdout_ids = _pick_holdout_sessions(
            sessions,
            laps,
            n_per_bucket=n_per_bucket,
            min_bucket_size=min_bucket_size,
            fold=fold,
            session_condition=session_condition,
        )
        if not holdout_ids:
            if fold == 0:
                print("No (track, car) bucket has enough sessions to hold out cleanly.")
                return None, 0
            # No more buckets have unused sessions for this fold; stop.
            break
        if not quiet:
            by_cond = pd.Series([session_condition.get(s, "?") for s in holdout_ids]).value_counts()
            cond_txt = ", ".join(f"{k} {v}" for k, v in by_cond.items())
            print(
                f"Fold {fold + 1}/{n_folds}: holding out {len(holdout_ids)} sessions ({cond_txt})"
                if n_folds > 1
                else f"Holding out {len(holdout_ids)} sessions ({cond_txt}; "
                f"{n_per_bucket} per (track, car, condition) bucket, min bucket size = "
                f"{min_bucket_size} dry / 3 rain)"
            )
        fold_df = _evaluate_fold(root, holdout_ids, inputs=inputs)
        if not fold_df.empty:
            fold_df["fold"] = fold
            fold_frames.append(fold_df)
        total_holdouts += len(holdout_ids)
    if not fold_frames:
        return pd.DataFrame(), total_holdouts
    return pd.concat(fold_frames, ignore_index=True), total_holdouts


def _session_conditions(root: Path) -> dict[str, str]:
    """Per-session track condition (dry/damp/wet/unknown) from the same
    weather classification the fit uses (see ``warmup_table._attach_weather``)."""
    from .warmup_table import _attach_weather, _load_filtered_laps, _load_weather

    laps = _attach_weather(_load_filtered_laps(root), _load_weather(root))
    first = laps.drop_duplicates("session_id")
    return dict(zip(first["session_id"], first["condition"]))


def _pick_holdout_sessions(
    sessions: pd.DataFrame,
    laps: pd.DataFrame,
    n_per_bucket: int = 2,
    min_bucket_size: int = 10,
    fold: int = 0,
    *,
    session_condition: dict[str, str] | None = None,
    rain_min_bucket_size: int = 3,
) -> list[str]:
    """Pick deterministic held-out session_ids, stratified by condition.

    Buckets are ``(track, car, condition)`` when ``session_condition`` is
    given (the production path), else ``(track, car)``. Dry buckets need
    ``min_bucket_size`` sessions to be eligible; damp/wet buckets are far
    smaller, so they use ``rain_min_bucket_size`` — without that, no rain
    session would ever be held out and the rain numbers would be whatever
    happened to fall into the dry slices. Sessions with ``unknown``
    condition are never held out (they are excluded from training too).

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

    if session_condition is not None:
        ok = ok.assign(_cond=ok["session_id"].map(session_condition).fillna("unknown"))
        ok = ok[ok["_cond"] != "unknown"]
        keys = ["track_canonical", "car", "_cond"]
    else:
        ok = ok.assign(_cond="dry")
        keys = ["track_canonical", "car"]

    held_out: list[str] = []
    start = fold * n_per_bucket
    stop = start + n_per_bucket
    for key, grp in ok.groupby(keys):
        cond = key[2] if len(key) == 3 else "dry"
        needed = min_bucket_size if cond == "dry" else rain_min_bucket_size
        if len(grp) < needed:
            continue
        ordered = sorted(grp["session_id"].tolist())
        if start >= len(ordered):
            continue  # this bucket has been exhausted by earlier folds
        held_out.extend(ordered[start:stop])
    return held_out


INPUT_MODES = ("calculator", "oracle")


def _calculator_lap_inputs(
    model: dict,
    track: str,
    car: str,
    condition: str,
    flying_lap_n: int,
    session_lap_time_s: float,
    *,
    with_outlap: bool,
) -> tuple[float, float, float, float, float]:
    """What the calculator would feed the warmup curve for this lap.

    The driver enters the track, car, condition, the flying-lap number N and
    a target lap time; we stand in the session's own median lap time for
    the target (a driver knows their pace to a few seconds). Returns
    ``(g2, t_flying_s, g2_scale, outlap_s, outlap_g2)``: the bucket's ⟨g²⟩
    scaled along the pace curve, the flying clock ``N × target``, and the
    bucket's typical out-lap segment (zero when ``with_outlap`` is False —
    the stint's anchor was not a pit exit, so the clock starts at the first
    flying lap as before).
    """
    lap_time_typ_s, _n, _src = _lookup_lap_time(model, track, car, condition)
    g2_typ, _n2, _src2 = _lookup_g2(model, track, car, condition)
    scale, _pace_src = _g2_pace_scale(
        model, track, car, condition, lap_time_typ_s, session_lap_time_s
    )
    t_flying_s = float(flying_lap_n) * float(session_lap_time_s)
    out_s, out_g2 = 0.0, 0.0
    if with_outlap:
        hit = _lookup_outlap(model, track, car, condition)
        if hit is not None:
            out_s, out_g2 = hit[0], hit[1]
    return g2_typ * scale, t_flying_s, scale, out_s, out_g2


def _evaluate_fold(
    root: Path,
    holdout_ids: list[str],
    *,
    inputs: str = "calculator",
) -> pd.DataFrame:
    """Train a model excluding ``holdout_ids`` and return per-(lap, corner)
    residual rows for the held-out sessions.

    ``inputs`` decides what the warmup curve is fed:

    - ``"calculator"`` (default): only what the calculator has — the
      bucket's typical out-lap (rolling time, g²) integrated from the
      pit-exit reading the driver types in (stood in by the out-lap's first
      valid TPMS reading), then the bucket's ⟨g²⟩ scaled by the pace curve
      at the session's median lap time for ``N × lap time`` of flying laps.
      Stints without a usable pit-exit out-lap anchor on their first lap as
      before (``anchor_kind == "first_lap"``).
    - ``"oracle"``: the measured out-lap and the lap's own measured g² and
      clock from the same anchor — the thermal model's accuracy given the
      real driving, an upper bound on what the calculator can do.
    """
    if inputs not in INPUT_MODES:
        raise ValueError(f"inputs must be one of {INPUT_MODES}; got {inputs!r}")
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
    k_compound_lookup = {
        (d["car"], d["compound"], d["corner"], d["condition"]): d["value_kelvin_per_g2"]
        for d in model.get("K_by_car_compound_corner_cond", [])
    }
    # Held-out sessions' compound labels (metadata, not telemetry — using
    # them mirrors a driver entering the compound in the calculator).
    # Condition seeds count as metadata too (the driver knows whether wets
    # are bolted on); signature-INFERRED labels do not — classifying a
    # held-out session from its own temps would leak the target.
    from .compound_infer import apply_condition_seeds
    from .compounds import load_compound_labels, load_condition_seeds

    labels = load_compound_labels(root)
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
        _compute_stint_anchor,
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
    all_laps = _compute_stint_anchor(all_laps)
    all_laps = _compute_delta_t(all_laps)
    # Score the flying laps (the out-lap, lap_within_stint 0 when present, is
    # the anchor's own lap). With pit-exit anchors N = lap_within_stint is
    # the calculator's lap number; for first-lap anchors the first scored
    # lap is the one after the anchor lap, as in the v0 reports.
    all_laps["_is_out"] = (
        all_laps["is_outlap"].fillna(False).astype(bool)
        if "is_outlap" in all_laps.columns
        else False
    )
    all_laps_full = all_laps.copy()
    all_laps = all_laps[all_laps["lap_within_stint"] > 0].reset_index(drop=True)

    from .warmup_table import alias_condition_seeds

    labels = apply_condition_seeds(
        labels, all_laps, alias_condition_seeds(load_condition_seeds(root))
    )
    label_by_session = {
        r.session_id: r.compound for r in labels.itertuples() if isinstance(r.compound, str)
    }
    flying_only = all_laps[~all_laps["_is_out"]] if "_is_out" in all_laps.columns else all_laps
    session_lap_time = flying_only.groupby("session_id")["on_track_s"].median().to_dict()
    # Measured out-lap per stint for the oracle: (rolling-clock end, g²).
    stint_outlap: dict[tuple[str, int], tuple[float, float]] = {}
    if "_is_out" in all_laps_full.columns and "moving_s" in all_laps_full.columns:
        outs = all_laps_full[all_laps_full["_is_out"]]
        for sid_o, stint_o, mv, hp, t_end in zip(
            outs["session_id"].tolist(),
            outs["stint_id"].tolist(),
            outs["moving_s"].to_numpy(dtype=float),
            outs["heat_proxy"].to_numpy(dtype=float),
            outs["t_cum_s"].to_numpy(dtype=float),
        ):
            if np.isfinite(mv) and mv > 0 and np.isfinite(hp):
                stint_outlap[(str(sid_o), int(stint_o))] = (float(t_end), float(hp / mv))
    gamma_by_car = _fit_pressure_gain_by_car(root, exclude_session_ids=set(holdout_ids))

    # Per-lap predictions. Per-lap g² (heat_proxy / on_track_s) is the
    # held-out lap's actual driving intensity; falls back to the bucket
    # statistic only when that ratio is unavailable.
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
        on_track_s = lap.get("on_track_s")
        heat_proxy_total = lap.get("heat_proxy")
        g2: float | None
        if (
            pd.notna(heat_proxy_total)
            and pd.notna(on_track_s)
            and float(on_track_s) > 0
            and float(heat_proxy_total) > 0
        ):
            g2 = float(heat_proxy_total) / float(on_track_s)
        else:
            g2 = g2_lookup.get((track, car, cond)) or g2_lookup.get((track, car, "dry"))
        if g2 is None:
            continue
        g2_scale = 1.0
        t_pred_s = t_cum_s
        c_track = c_track_lookup.get(track, 1.0)
        session_compound = label_by_session.get(lap["session_id"])
        for c in CORNERS:
            K = k_lookup.get((car, c, cond)) or k_lookup.get((car, c, "dry"))
            if session_compound is not None:
                K = (
                    k_compound_lookup.get((car, session_compound, c, cond))
                    or k_compound_lookup.get((car, session_compound, c, "dry"))
                    or K
                )
            tau = tau_lookup.get((car, c, cond)) or tau_lookup.get((car, c, "dry"))
            if K is None or tau is None:
                continue
            obs = lap.get(f"tpms_temp_{c}_end")
            if pd.isna(obs):
                continue
            t_anchor = lap.get(f"t_anchor_{c}")
            t_start = lap.get(f"t_start_{c}")
            if pd.isna(t_anchor) or pd.isna(t_start) or t_cum_s <= float(t_anchor):
                continue
            anchor_kind = str(lap.get(f"anchor_kind_{c}") or "first_lap")
            pit_exit = anchor_kind == "pit_exit"
            t_start_used = float(t_start)
            out_info = stint_outlap.get((lap["session_id"], lap["stint_id"]))
            if inputs == "calculator":
                # The driver types the pit-exit temperature; the calculator
                # integrates the bucket's typical out-lap, then N flying laps
                # at the pace-scaled bucket g². Without a pit-exit anchor the
                # clock starts at the first flying lap (first_lap anchors are
                # on that lap's start).
                n_flying = (
                    int(lap["lap_within_stint"]) if pit_exit else int(lap["lap_within_stint"])
                )
                g2_c, t_fly, g2_scale, out_s, out_g2 = _calculator_lap_inputs(
                    model,
                    track,
                    car,
                    cond,
                    n_flying,
                    float(session_lap_time.get(lap["session_id"], lap["on_track_s"])),
                    with_outlap=pit_exit,
                )
                g2 = g2_c
                t_pred_s = t_fly
                _t_after, t_hot_pred = warmup_two_stage_c(
                    t_outlap_s=out_s,
                    g2_outlap=out_g2,
                    t_flying_s=t_fly,
                    g2_flying=g2,
                    t_eff_c=t_eff,
                    k_kelvin_per_g2=K,
                    c_track=c_track,
                    tau_sec=tau,
                    t_start_c=t_start_used,
                )
                t_from_start = out_s + t_fly
            else:
                # Oracle: the measured out-lap (its own g² over its rolling
                # time after the reading), then the flying laps at this
                # lap's measured g².
                if pit_exit and out_info is not None:
                    out_end_s, out_g2_meas = out_info
                    out_seg = max(out_end_s - float(t_anchor), 0.0)
                    _t_after, t_hot_pred = warmup_two_stage_c(
                        t_outlap_s=out_seg,
                        g2_outlap=out_g2_meas,
                        t_flying_s=max(t_cum_s - out_end_s, 0.0),
                        g2_flying=g2,
                        t_eff_c=t_eff,
                        k_kelvin_per_g2=K,
                        c_track=c_track,
                        tau_sec=tau,
                        t_start_c=t_start_used,
                    )
                else:
                    t_hot_pred = warmup_curve_c(
                        t_seconds=t_cum_s - float(t_anchor),
                        t_eff_c=t_eff,
                        k_kelvin_per_g2=K,
                        c_track=c_track,
                        g2_typ=g2,
                        tau_sec=tau,
                        t_start_c=t_start_used,
                    )
                t_from_start = t_cum_s - float(t_anchor)
            # Pressure domain: what the driver actually gets. Push the
            # predicted hot temperature through the same constant-volume
            # step the calculators use, from the pressure/temperature at
            # the stint anchor, and compare with the TPMS hot pressure.
            p_anchor = lap.get(f"p_start_{c}")
            p_obs = lap.get(f"tpms_press_{c}_end")
            p_pred = p_pred_gamma = float("nan")
            gamma_car = float(gamma_by_car.get(car, 1.0))
            if pd.notna(p_anchor) and pd.notna(p_obs) and float(t_start) > -273.15:
                ratio = (t_hot_pred + T_ZERO_C_TO_K) / (float(t_start) + T_ZERO_C_TO_K)
                p_pred = (float(p_anchor) + P_ATM_BAR) * ratio - P_ATM_BAR
                p_pred_gamma = (float(p_anchor) + P_ATM_BAR) * ratio**gamma_car - P_ATM_BAR
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
                    "t_anchor_s": float(t_anchor),
                    "t_start_c": t_start_used,
                    "anchor_kind": anchor_kind,
                    "inputs": inputs,
                    "g2_used": g2,
                    "g2_scale": g2_scale,
                    "t_used_s": t_from_start,
                    "corner": c,
                    "T_hot_pred_c": t_hot_pred,
                    "T_hot_obs_c": float(obs),
                    "resid_c": t_hot_pred - float(obs),
                    "P_anchor_bar": float(p_anchor) if pd.notna(p_anchor) else float("nan"),
                    "P_hot_obs_bar": float(p_obs) if pd.notna(p_obs) else float("nan"),
                    "P_hot_pred_bar": p_pred,
                    "resid_bar": p_pred - float(p_obs) if pd.notna(p_obs) else float("nan"),
                    "gamma_car": gamma_car,
                    "resid_bar_gamma": (
                        p_pred_gamma - float(p_obs) if pd.notna(p_obs) else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _fit_pressure_gain_by_car(root: Path, *, exclude_session_ids: set[str]) -> dict[str, float]:
    """Per-car exponent γ in P_abs ∝ T_abs^γ, fitted on the training sessions.

    Constant-volume ideal gas says γ = 1 for the true gas temperature, but
    the TPMS temperature is read at the valve and runs warmer than the mean
    cavity gas, so the pressure rises less than the temperature implies.
    Fitted on exactly the quantity the evaluation applies it to: the ratio
    from the stint anchor (first finite reading) to each later lap end,
    slope through the origin of ln(P_end/P_anchor) on ln(T_end/T_anchor).
    """
    from .warmup_table import _compute_stint_anchor, _compute_stint_clock, _load_filtered_laps

    laps = _load_filtered_laps(root)
    laps = laps[~laps["session_id"].isin(exclude_session_ids)]
    laps = _compute_stint_anchor(_compute_stint_clock(laps))
    out: dict[str, float] = {}
    for car, grp in laps.groupby("car"):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for c in CORNERS:
            t_e = grp[f"tpms_temp_{c}_end"].to_numpy(dtype=float) + T_ZERO_C_TO_K
            t_a = grp[f"t_start_{c}"].to_numpy(dtype=float) + T_ZERO_C_TO_K
            p_e = grp[f"tpms_press_{c}_end"].to_numpy(dtype=float) + P_ATM_BAR
            p_a = grp[f"p_start_{c}"].to_numpy(dtype=float) + P_ATM_BAR
            after = grp["t_cum_s"].to_numpy(dtype=float) > grp[f"t_anchor_{c}"].to_numpy(
                dtype=float
            )
            ok = np.isfinite(t_e) & np.isfinite(t_a) & np.isfinite(p_e) & np.isfinite(p_a) & after
            ok &= (p_e > 0.3) & (p_a > 0.3) & (np.abs(t_e - t_a) > 2)
            xs.append(np.log(t_e[ok] / t_a[ok]))
            ys.append(np.log(p_e[ok] / p_a[ok]))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        if len(x) >= 20 and float(np.sum(x * x)) > 0:
            out[str(car)] = float(np.sum(x * y) / np.sum(x * x))
        else:
            out[str(car)] = 1.0
    return out


def _print_corner_table(label: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    has_p = "resid_bar" in frame.columns and frame["resid_bar"].notna().any()
    print(f"\n=== {label} ({len(frame)} (lap × corner) points) ===")
    if has_p:
        print(
            "         temperature                                 hot pressure, calculator (γ=1)   with per-car γ"
        )
    for c in CORNERS:
        sub = frame[frame["corner"] == c]
        if sub.empty:
            continue
        mae = float(sub["resid_c"].abs().mean())
        rmse = float(np.sqrt((sub["resid_c"] ** 2).mean()))
        bias = float(sub["resid_c"].mean())
        line = (
            f"  {c.upper():>3}   MAE = {mae:>5.2f} °C   RMSE = {rmse:>5.2f} °C   "
            f"mean bias = {bias:+.2f} °C    n = {len(sub)}"
        )
        if has_p:
            pb = sub["resid_bar"].dropna()
            pg = sub["resid_bar_gamma"].dropna()
            if not pb.empty:
                line += (
                    f"   |  MAE = {pb.abs().mean():.3f} bar  bias = {pb.mean():+.3f} bar"
                    f"   |  MAE = {pg.abs().mean():.3f} bar  bias = {pg.mean():+.3f} bar"
                )
        print(line)


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
    inputs: str = "calculator",
) -> int:
    """Train on all-minus-held-out, predict per-lap T_hot for held-out sessions.

    ``inputs``: see :func:`_evaluate_fold`. The default
    scores what the calculator would have told the driver; ``inputs="oracle"``
    scores the thermal model with the lap's real g² and clock.

    With ``n_folds == 1`` (default) this is a single deterministic holdout
    — the legacy behavior. With ``n_folds > 1`` it sweeps disjoint
    n_per_bucket-sized slices through every bucket as k-fold CV: refits k
    times, predicts on each fold's held-out set, then aggregates residuals
    across all folds so every session appears in the held-out set roughly
    once. Useful when a single fold's MAE is too noisy to compare model
    revisions (e.g. tiny per-bucket holdouts).
    """
    root = Path(dataset_root) if dataset_root else default_dataset_root()
    df, total_holdouts = _collect_holdout_frame(
        root,
        n_per_bucket=n_per_bucket,
        min_bucket_size=min_bucket_size,
        n_folds=n_folds,
        inputs=inputs,
    )
    if df is None:
        return 1
    if df.empty:
        print("No predictable laps in any held-out fold.")
        return 0
    print(
        f"inputs: {inputs}"
        + (
            " (bucket g² at the session's pace, N × lap time, typed-in start temperature)"
            if inputs == "calculator"
            else " (measured g², clock, anchor)"
        )
    )

    if "gamma_car" in df.columns:
        gam = df.drop_duplicates("car")[["car", "gamma_car"]]
        print(
            "pressure gain γ (P_abs ∝ T_abs^γ, fitted on training sessions; the calculators use 1): "
            + ", ".join(f"{r.car} {r.gamma_car:.3f}" for r in gam.itertuples())
        )
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
