"""CLI for the tire warmup model: build-warmup-table, predict, validate."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..tire_etl.paths import default_dataset_root
from .predict import CORNERS, Prediction, predict_cold_pressure
from .warmup_table import build_warmup_table

logger = logging.getLogger(__name__)


def _add_dataset_root_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the dataset root (defaults to repo data/tire_dataset).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tire-model", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser(
        "build-warmup-table",
        help="Fit the energy-balance model and write tire_model.json + warmup_table.parquet.",
    )
    _add_dataset_root_arg(p_build)
    p_build.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild (currently always rebuilds; flag reserved for future)",
    )

    p_predict = sub.add_parser(
        "predict",
        help="Predict per-corner cold pressures for a target lap.",
    )
    _add_dataset_root_arg(p_predict)
    p_predict.add_argument("--track", required=True)
    p_predict.add_argument("--car", required=True)
    p_predict.add_argument(
        "--lap",
        type=int,
        required=True,
        help="Target lap_within_stint (0 = out-lap; 5 = 5 laps in).",
    )
    p_predict.add_argument(
        "--ambient", type=float, required=True, help="Ambient air temperature in °C."
    )
    p_predict.add_argument(
        "--track-temp",
        type=float,
        default=None,
        help="Optional measured track-surface temperature in °C.",
    )
    p_predict.add_argument(
        "--cloud-cover",
        type=float,
        default=None,
        help="0..100; used only if --track-temp is omitted.",
    )
    p_predict.add_argument(
        "--g2-typ",
        type=float,
        default=None,
        help="Override the looked-up <g²> for this (track, car).",
    )
    p_predict.add_argument(
        "--lap-time-s",
        type=float,
        default=None,
        help="Override the looked-up median lap time in seconds.",
    )
    group = p_predict.add_argument_group("Target hot pressures (bar gauge)")
    group.add_argument(
        "--hot-all", type=float, default=None, help="Shorthand: same target for all 4 corners."
    )
    group.add_argument("--hot-fl", type=float, default=None)
    group.add_argument("--hot-fr", type=float, default=None)
    group.add_argument("--hot-rl", type=float, default=None)
    group.add_argument("--hot-rr", type=float, default=None)

    p_validate = sub.add_parser(
        "validate",
        help="MAE report vs. notes-recorded cold pressures (uses production model).",
    )
    _add_dataset_root_arg(p_validate)

    p_audit = sub.add_parser(
        "audit-sensors",
        help=(
            "Detect (session, corner) channels that look stuck/broken (low std). "
            "Lists candidates with evidence; you decide which to add to sensor_blacklist.yaml."
        ),
    )
    _add_dataset_root_arg(p_audit)

    p_holdout = sub.add_parser(
        "holdout",
        help="Held-out validation: exclude sessions from training, predict per-lap T_hot, report residuals.",
    )
    _add_dataset_root_arg(p_holdout)
    p_holdout.add_argument(
        "--n-per-bucket",
        type=int,
        default=2,
        help="Number of held-out sessions per (track, car) bucket.",
    )
    p_holdout.add_argument(
        "--min-bucket-size",
        type=int,
        default=10,
        help="Only hold out from buckets with at least this many sessions.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "build-warmup-table":
        return _cmd_build(args)
    if args.cmd == "predict":
        return _cmd_predict(args)
    if args.cmd == "validate":
        return _cmd_validate(args)
    if args.cmd == "holdout":
        return _cmd_holdout(args)
    if args.cmd == "audit-sensors":
        return _cmd_audit_sensors(args)
    parser.print_help()
    return 2


def _cmd_build(args: argparse.Namespace) -> int:
    root = args.dataset_root or default_dataset_root()
    logger.info("Building tire model artifacts at %s", root)
    model = build_warmup_table(root, rebuild=args.rebuild)
    n_k = len(model["K_buckets"])
    n_tau = len(model["tau_sec_by_car_corner"])
    n_c = len(model["c_track_by_track"])
    logger.info(
        "Wrote tire_model.json + warmup_table.parquet: %d K, %d τ, %d c_track entries",
        n_k,
        n_tau,
        n_c,
    )
    return 0


def _resolve_hot_pressures(args: argparse.Namespace) -> dict[str, float]:
    if args.hot_all is not None:
        return {c: float(args.hot_all) for c in CORNERS}
    by_corner = {
        "fl": args.hot_fl,
        "fr": args.hot_fr,
        "rl": args.hot_rl,
        "rr": args.hot_rr,
    }
    missing = [c for c, v in by_corner.items() if v is None]
    if missing:
        raise SystemExit(
            f"Must provide --hot-all OR all of --hot-fl/--hot-fr/--hot-rl/--hot-rr; missing: {missing}"
        )
    return {c: float(v) for c, v in by_corner.items()}


def _cmd_predict(args: argparse.Namespace) -> int:
    targets = _resolve_hot_pressures(args)
    result = predict_cold_pressure(
        track=args.track,
        car=args.car,
        lap_within_stint=args.lap,
        target_hot_pressure_bar=targets,
        ambient_temp_c=args.ambient,
        track_temp_c=args.track_temp,
        cloud_cover_pct=args.cloud_cover,
        g2_typ_override=args.g2_typ,
        lap_time_typ_override_s=args.lap_time_s,
        dataset_root=args.dataset_root,
    )
    _print_prediction(result, args)
    return 0


def _print_prediction(result: dict[str, Prediction], args: argparse.Namespace) -> None:
    # Header from the first corner (all share track-level lookups)
    any_p = next(iter(result.values()))
    print(
        f"Track:        {args.track:<20} c_track = {any_p.c_track:.2f}"
        f" ± {any_p.c_track_stderr:.3f}     ⟨g²⟩ = {any_p.g2_typ:.2f} G²"
    )
    print(f"Car:          {args.car:<20} lap_time_typ = {any_p.lap_time_typ_s:.1f} s")
    print(f"                                  t at lap {args.lap} = {any_p.t_at_lap_n_s:.1f} s")
    print(
        f"T_air = {any_p.t_air_c:.1f} °C    T_road = {any_p.t_road_c:.1f} °C    "
        f"T_eff = {any_p.t_eff_c:.1f} °C (w_road=0.20)"
    )
    print()
    header = (
        f"{'Corner':6} {'K(K/G²)':>9} {'τ_sec(s)':>9} {'warmup':>7} {'ΔT∞(K)':>8} "
        f"{'HotT(°C)':>9} {'HotIn(bar)':>11} {'Cold(bar)':>10} {'±(bar)':>7} "
        f"{'K source':<18} {'n':>4}"
    )
    print(header)
    for c in CORNERS:
        p = result[c]
        # Propagate uncertainty: dominant term is K stderr × c_track × g²
        dT_inf_stderr = p.K_stderr * p.c_track * p.g2_typ
        # Roughly: |dCold/dT_hot| · stderr of T_hot
        t_hot_k = p.predicted_hot_temp_c + 273.15
        t_cold_k = p.t_air_c + 273.15
        # P_cold = (HotIn+1) · T_cold_K / T_hot_K  →  ∂/∂T_hot = -(HotIn+1) · T_cold_K / T_hot_K²
        d_cold_d_thot = -(p.target_hot_pressure_bar + 1.0) * t_cold_k / (t_hot_k**2)
        cold_stderr = abs(d_cold_d_thot) * (dT_inf_stderr * p.warmup_frac)
        src = ",".join(p.K_source_bucket) if p.K_source_bucket else "prior"
        print(
            f"{c.upper():6} {p.K_kelvin_per_g2:>9.1f} {p.tau_sec:>9.0f} {p.warmup_frac:>7.2f} "
            f"{p.delta_t_inf_kelvin:>8.1f} {p.predicted_hot_temp_c:>9.1f} "
            f"{p.target_hot_pressure_bar:>11.3f} {p.cold_pressure_bar:>10.3f} "
            f"{cold_stderr:>7.3f} ({src})".ljust(0) + f"{'':<4}{p.K_n_samples:>4}"
        )
    print()
    print("ΔT_∞ = K · c_track · ⟨g²⟩       Hot T = T_eff + ΔT_∞ · warmup_frac")
    print("Cold  = (HotIn + 1)·(T_air+273)/(HotT+273) − 1")


def _cmd_validate(args: argparse.Namespace) -> int:
    from .validate import run_validation  # local import keeps CLI cold-load fast

    return run_validation(dataset_root=args.dataset_root)


def _cmd_holdout(args: argparse.Namespace) -> int:
    from .validate import run_holdout_validation

    return run_holdout_validation(
        dataset_root=args.dataset_root,
        n_per_bucket=args.n_per_bucket,
        min_bucket_size=args.min_bucket_size,
    )


def _cmd_audit_sensors(args: argparse.Namespace) -> int:
    """Detect candidate broken sensors and print them for human review."""
    from .warmup_table import (
        _attach_weather,
        _compute_stint_clock,
        _load_filtered_laps,
        _load_weather,
        detect_suspect_corners,
        load_sensor_blacklist,
    )

    root = args.dataset_root or default_dataset_root()
    laps = _load_filtered_laps(root)
    laps = _attach_weather(laps, _load_weather(root))
    laps = _compute_stint_clock(laps)
    candidates = detect_suspect_corners(laps)

    blacklisted = load_sensor_blacklist(root)

    if candidates.empty:
        print("No suspect (session, corner) channels detected.")
        return 0

    print(f"=== Sensor audit: {len(candidates)} candidate (session, corner) channels ===")
    print(
        "Heuristic: tpms_temp_{c}_end has std < 1.0 °C across ≥ 4 tire-usable laps "
        "(channel looks stuck)."
    )
    print(
        "Review each entry; add confirmed broken ones to "
        "`data/tire_dataset/sensor_blacklist.yaml` to exclude from training."
    )
    print()
    candidates = candidates.sort_values(["car", "track_canonical", "date", "session_id", "corner"])

    import pandas as _pd

    _pd.set_option("display.width", 240)
    _pd.set_option("display.max_columns", 30)
    _pd.set_option("display.max_colwidth", 80)
    # Show whether each candidate is already in the blacklist
    candidates = candidates.copy()
    candidates["already_blacklisted"] = [
        (sid, c) in blacklisted for sid, c in zip(candidates["session_id"], candidates["corner"])
    ]
    cols = [
        "session_id",
        "car",
        "track_canonical",
        "date",
        "corner",
        "n_laps",
        "std_c",
        "min_c",
        "max_c",
        "mean_c",
        "first_5_values",
        "already_blacklisted",
    ]
    print(candidates[cols].to_string(index=False))
    print()
    n_new = int((~candidates["already_blacklisted"]).sum())
    n_known = int(candidates["already_blacklisted"].sum())
    print(f"  {n_new} not yet in sensor_blacklist.yaml    {n_known} already blacklisted")
    print()
    if n_new > 0:
        print("Suggested YAML entries (copy into data/tire_dataset/sensor_blacklist.yaml):")
        print()
        for _, row in candidates[~candidates["already_blacklisted"]].iterrows():
            print(f"  - session_id: {row['session_id']}")
            print(f"    corner: {row['corner']}")
            print(
                f"    reason: \"std={row['std_c']:.2f} °C across {row['n_laps']} laps "
                f"(min={row['min_c']:.1f}, max={row['max_c']:.1f})\""
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
