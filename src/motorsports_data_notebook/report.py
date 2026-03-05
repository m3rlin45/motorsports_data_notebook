"""Session report generation for motorsports telemetry analysis.

Orchestrates all analysis functions into a single structured report
with JSON-serializable output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from motorsports_data_notebook.channels import get_best_lap, get_top_laps
from motorsports_data_notebook.corners import identify_corners
from motorsports_data_notebook.driver_analysis import (
    find_throttle_acceptance_from_arrays,
    prepare_throttle_acceptance,
)
from motorsports_data_notebook.profiles import get_logger_id, get_profile_for_logger
from motorsports_data_notebook.suspension import analyze_suspension_velocity_multi_lap
from motorsports_data_notebook.tire_grip import analyze_tire_grip_multi_lap
from motorsports_data_notebook.zones import (
    compute_segment_stats,
    create_track_segments,
    detect_zones_averaged,
)

if TYPE_CHECKING:
    from motorsports_data_notebook._types import LogFile
    from motorsports_data_notebook.corners import Corner
    from motorsports_data_notebook.suspension import MotionRatios, VelocityHistogramResult
    from motorsports_data_notebook.tire_grip import TireGripResult


# ── Dataclasses ──────────────────────────────────────────────────────────────────


@dataclass
class SessionMetadata:
    """Top-level session information."""

    file_name: str
    logger_id: str | None
    vehicle_profile: str | None
    total_laps: int
    valid_laps: int
    top_lap_count: int


@dataclass
class LapTimeSummary:
    """Lap time analysis results."""

    best_lap_num: int
    best_lap_time_s: float
    best_lap_time_fmt: str
    top_laps: list[dict]
    mean_top_lap_time_s: float
    mean_top_lap_time_fmt: str
    std_top_lap_time_s: float
    all_lap_times: list[dict]


@dataclass
class CornerInfo:
    """JSON-serializable corner data (mirrors corners.Corner)."""

    id: int
    name: str
    direction: str
    start_dist: float
    end_dist: float
    apex_dist: float
    length: float
    radius: float


@dataclass
class CornerBestLap:
    """Best execution at a specific corner."""

    lap_num: int
    selection_reason: str
    braking_point: float | None
    min_speed: float
    exit_speed: float
    throttle_acceptance_pct: float | None
    throttle_point: float | None
    vs_mean: dict[str, float]
    peak_brake: float | None = None
    entry_speed: float | None = None
    brake_release_point: float | None = None
    braking_distance: float | None = None
    mean_decel_g: float | None = None
    g_utilization_pct: float | None = None
    total_g_min: float | None = None
    total_g_min_dist: float | None = None
    total_g_min_phase: str | None = None


@dataclass
class CornerConsistency:
    """Per-corner aggregated consistency metrics across top laps."""

    corner: CornerInfo
    ta_mean: float
    ta_std: float
    bp_mean: float
    bp_std: float
    min_speed_mean: float
    min_speed_std: float
    exit_speed_mean: float
    exit_speed_std: float
    accel_zone_length: float
    opportunity_score: float
    best_lap: CornerBestLap | None
    peak_brake_mean: float | None = None
    peak_brake_std: float | None = None
    entry_speed_mean: float | None = None
    entry_speed_std: float | None = None
    brake_release_mean: float | None = None
    brake_release_std: float | None = None
    braking_distance_mean: float | None = None
    braking_distance_std: float | None = None
    mean_decel_g_mean: float | None = None
    mean_decel_g_std: float | None = None
    # G utilization fields
    g_utilization_mean: float | None = None
    g_utilization_std: float | None = None
    total_g_min_mean: float | None = None
    total_g_min_phase: str | None = None
    braking_g_mean_val: float | None = None
    entry_g_mean_val: float | None = None
    mid_g_mean_val: float | None = None
    exit_g_mean_val: float | None = None
    # Off-track detection
    excluded_laps: list[int] = field(default_factory=list)


@dataclass
class SuspensionSummary:
    """Per-wheel suspension velocity distribution stats."""

    per_wheel: dict[str, dict[str, float]]
    symmetry: dict[str, float]


@dataclass
class TireGripSummary:
    """Per-wheel tire grip stats."""

    metric_mode: str
    units: dict[str, str]
    per_wheel: dict[str, dict[str, float]]


@dataclass
class BrakingBalanceSummary:
    """Braking balance analysis between front and rear brake circuits."""

    available: bool
    front_channel: str
    rear_channel: str
    per_corner: list[dict]
    overall_balance_pct: float
    overall_balance_std: float
    min_brake_threshold: float = 0.0


@dataclass
class SessionReport:
    """Complete session analysis report."""

    metadata: SessionMetadata
    lap_times: LapTimeSummary
    corners: list[CornerInfo]
    corner_consistency: list[CornerConsistency]
    track_length_m: float
    suspension: SuspensionSummary | None
    tire_grip: TireGripSummary | None
    available_analyses: list[str]
    skipped_analyses: dict[str, str]
    braking_balance: BrakingBalanceSummary | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _check_channels_available(
    log: LogFile, channel_names: dict[str, str], required_keys: list[str]
) -> tuple[bool, list[str]]:
    """Check if required channel keys exist and map to available channels."""
    missing = []
    for key in required_keys:
        if key not in channel_names:
            missing.append(key)
        elif channel_names[key] not in log.channels:
            missing.append(key)
    return (len(missing) == 0, missing)


def _corner_to_info(corner: Corner) -> CornerInfo:
    """Convert a Corner dataclass to JSON-serializable CornerInfo."""
    return CornerInfo(
        id=corner.id,
        name=corner.name,
        direction=corner.direction,
        start_dist=float(corner.start_dist),
        end_dist=float(corner.end_dist),
        apex_dist=float(corner.apex_dist),
        length=float(corner.length),
        radius=float(corner.radius),
    )


def _lap_time_to_seconds(lap_time) -> float:
    """Convert a lap_time value (Timedelta or numeric) to seconds."""
    if isinstance(lap_time, pd.Timedelta):
        return lap_time.total_seconds()
    return float(lap_time)


def _format_lap_time(seconds: float) -> str:
    """Format seconds as M:SS.mmm (e.g. 117.566 -> '1:57.566')."""
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}:{remainder:06.3f}"


def _get_per_lap_metric(
    stats_df: pd.DataFrame, corner_id: int, segment_type: str, metric_name: str
) -> list[tuple[int, float]]:
    """Extract per-lap values of a metric from segment stats DataFrame."""
    mask = (stats_df["corner_id"] == corner_id) & (stats_df["segment_type"] == segment_type)
    rows = stats_df[mask]
    if len(rows) == 0 or metric_name not in rows.columns:
        return []
    result = []
    for _, row in rows.iterrows():
        val = row[metric_name]
        if pd.notna(val):
            result.append((int(row["lap_num"]), float(val)))
    return result


def _compute_best_lap(
    bp_vals: list[tuple[int, float]],
    min_speed_vals: list[tuple[int, float]],
    exit_speed_vals: list[tuple[int, float]],
    tp_vals: list[tuple[int, float]],
    ta_entries: list[tuple[int, float]],
    peak_brake_vals: list[tuple[int, float]] | None = None,
    entry_speed_vals: list[tuple[int, float]] | None = None,
    brake_release_vals: list[tuple[int, float]] | None = None,
    braking_distance_vals: list[tuple[int, float]] | None = None,
    mean_decel_g_vals: list[tuple[int, float]] | None = None,
) -> CornerBestLap | None:
    """Compute the best lap at a corner using composite z-score ranking.

    Score = 0.5 * exit_speed_z + 0.3 * min_speed_z + 0.2 * braking_point_z
    """
    if not exit_speed_vals:
        return None

    # Build per-lap metric dict
    lap_data: dict[int, dict[str, float]] = {}
    for lap_num, val in exit_speed_vals:
        lap_data.setdefault(lap_num, {})["exit_speed"] = val
    for lap_num, val in min_speed_vals:
        lap_data.setdefault(lap_num, {})["min_speed"] = val
    for lap_num, val in bp_vals:
        lap_data.setdefault(lap_num, {})["braking_point"] = val
    for src, key in [
        (peak_brake_vals, "peak_brake"),
        (entry_speed_vals, "entry_speed"),
        (brake_release_vals, "brake_release_point"),
        (braking_distance_vals, "braking_distance"),
        (mean_decel_g_vals, "mean_decel_g"),
    ]:
        if src:
            for lap_num, val in src:
                lap_data.setdefault(lap_num, {})[key] = val

    valid_laps = {k: v for k, v in lap_data.items() if "exit_speed" in v}
    if not valid_laps:
        return None

    ta_dict = dict(ta_entries)
    tp_dict = dict(tp_vals)

    if len(valid_laps) == 1:
        lap_num = next(iter(valid_laps))
        d = valid_laps[lap_num]
        return CornerBestLap(
            lap_num=lap_num,
            selection_reason="only lap with data",
            braking_point=d.get("braking_point"),
            min_speed=d.get("min_speed", 0.0),
            exit_speed=d["exit_speed"],
            throttle_acceptance_pct=ta_dict.get(lap_num),
            throttle_point=tp_dict.get(lap_num),
            vs_mean={},
            peak_brake=d.get("peak_brake"),
            entry_speed=d.get("entry_speed"),
            brake_release_point=d.get("brake_release_point"),
            braking_distance=d.get("braking_distance"),
            mean_decel_g=d.get("mean_decel_g"),
        )

    def _z_scores(arr: np.ndarray) -> np.ndarray:
        std = float(np.std(arr))
        if std == 0:
            return np.zeros_like(arr)
        return (arr - np.mean(arr)) / std  # type: ignore[no-any-return]

    laps_list = sorted(valid_laps.keys())
    exit_speeds = np.array([valid_laps[n]["exit_speed"] for n in laps_list])
    exit_z = _z_scores(exit_speeds)

    # Min speed (optional — fill NaN with mean for z-score)
    min_speeds = np.array([valid_laps[n].get("min_speed", np.nan) for n in laps_list])
    has_min_speed = not np.all(np.isnan(min_speeds))
    if has_min_speed:
        filled = np.where(np.isnan(min_speeds), np.nanmean(min_speeds), min_speeds)
        min_speed_z = _z_scores(filled)
    else:
        min_speed_z = np.zeros(len(laps_list))

    # Braking point (optional — later braking = higher distance = better)
    bp_arr = np.array([valid_laps[n].get("braking_point", np.nan) for n in laps_list])
    has_bp = not np.all(np.isnan(bp_arr))
    if has_bp:
        filled_bp = np.where(np.isnan(bp_arr), np.nanmean(bp_arr), bp_arr)
        bp_z = _z_scores(filled_bp)
    else:
        bp_z = np.zeros(len(laps_list))

    scores = 0.5 * exit_z + 0.3 * min_speed_z + 0.2 * bp_z
    best_idx = int(np.argmax(scores))
    best_lap_num = laps_list[best_idx]
    d = valid_laps[best_lap_num]

    # vs_mean deltas
    exit_mean = float(np.mean(exit_speeds))
    vs_mean: dict[str, float] = {"exit_speed": round(d["exit_speed"] - exit_mean, 2)}
    if has_min_speed and "min_speed" in d:
        vs_mean["min_speed"] = round(d["min_speed"] - float(np.nanmean(min_speeds)), 2)
    if has_bp and "braking_point" in d:
        vs_mean["braking_point"] = round(d["braking_point"] - float(np.nanmean(bp_arr)), 2)
    for key in [
        "peak_brake",
        "entry_speed",
        "brake_release_point",
        "braking_distance",
        "mean_decel_g",
    ]:
        vals = [valid_laps[n].get(key) for n in laps_list]
        if d.get(key) is not None and any(v is not None for v in vals):
            arr = np.array([v if v is not None else np.nan for v in vals])
            if not np.all(np.isnan(arr)):
                vs_mean[key] = round(d[key] - float(np.nanmean(arr)), 2)

    # Selection reason
    reasons = []
    if exit_z[best_idx] > 0.5:
        reasons.append("fastest exit speed")
    if has_min_speed and min_speed_z[best_idx] > 0.5:
        reasons.append("high min speed")
    if has_bp and bp_z[best_idx] > 0.5:
        reasons.append("later braking")
    reason = ", ".join(reasons) if reasons else "best composite score"

    return CornerBestLap(
        lap_num=best_lap_num,
        selection_reason=reason,
        braking_point=d.get("braking_point"),
        min_speed=d.get("min_speed", 0.0),
        exit_speed=d["exit_speed"],
        throttle_acceptance_pct=ta_dict.get(best_lap_num),
        throttle_point=tp_dict.get(best_lap_num),
        vs_mean=vs_mean,
        peak_brake=d.get("peak_brake"),
        entry_speed=d.get("entry_speed"),
        brake_release_point=d.get("brake_release_point"),
        braking_distance=d.get("braking_distance"),
        mean_decel_g=d.get("mean_decel_g"),
    )


def _detect_off_track_laps(
    per_lap_gps: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    corners: list[Corner],
    deviation_threshold: float = 10.0,
) -> dict[int, set[int]]:
    """Detect laps that went off-track at each corner via GPS deviation.

    For each corner, interpolates all laps onto a common distance grid,
    computes the median path as reference, and flags laps whose max
    deviation exceeds the threshold.

    Parameters
    ----------
    per_lap_gps : dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
        Per-lap (distance_m, latitude, longitude) arrays.
    corners : list[Corner]
        Detected corners.
    deviation_threshold : float
        Max allowed deviation from median path in meters.

    Returns
    -------
    dict[int, set[int]]
        Mapping of corner_id -> set of off-track lap numbers.
    """
    if not per_lap_gps or not corners:
        return {}

    R = 6371000.0

    # Compute shared reference center from all GPS data
    all_lat = np.concatenate([d[1] for d in per_lap_gps.values()])
    all_lon = np.concatenate([d[2] for d in per_lap_gps.values()])
    valid = (all_lat != 0.0) | (all_lon != 0.0)
    if not np.any(valid):
        return {}
    lat0 = np.radians(float(np.mean(all_lat[valid])))
    lon0 = np.radians(float(np.mean(all_lon[valid])))

    off_track: dict[int, set[int]] = {}

    for corner in corners:
        margin = 50.0
        c_start = corner.start_dist - margin
        c_end = corner.end_dist + margin
        grid = np.arange(c_start, c_end, 1.0)
        if len(grid) < 5:
            continue

        # Interpolate each lap's XY onto the common grid
        lap_xys: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for lap_num, (dist_arr, lat_arr, lon_arr) in per_lap_gps.items():
            mask = (dist_arr >= c_start - 10) & (dist_arr <= c_end + 10)
            if np.sum(mask) < 3:
                continue
            lx = R * (np.radians(lon_arr[mask]) - lon0) * np.cos(lat0)
            ly = R * (np.radians(lat_arr[mask]) - lat0)
            x_interp = np.interp(grid, dist_arr[mask], lx)
            y_interp = np.interp(grid, dist_arr[mask], ly)
            lap_xys[lap_num] = (x_interp, y_interp)

        if len(lap_xys) < 3:
            continue

        # Median reference path
        x_stack = np.stack([xy[0] for xy in lap_xys.values()])
        y_stack = np.stack([xy[1] for xy in lap_xys.values()])
        ref_x = np.median(x_stack, axis=0)
        ref_y = np.median(y_stack, axis=0)

        for lap_num, (lx, ly) in lap_xys.items():
            deviations = np.sqrt((lx - ref_x) ** 2 + (ly - ref_y) ** 2)
            if float(np.max(deviations)) > deviation_threshold:
                off_track.setdefault(corner.id, set()).add(lap_num)

    return off_track


def _extract_suspension_summary(result: VelocityHistogramResult) -> SuspensionSummary:
    """Extract JSON-serializable suspension stats from VelocityHistogramResult."""
    per_wheel = {}
    for name, corner_data in [
        ("FL", result.front_left),
        ("FR", result.front_right),
        ("RL", result.rear_left),
        ("RR", result.rear_right),
    ]:
        per_wheel[name] = {
            "skew": float(corner_data.skew),
            "kurtosis": float(corner_data.kurtosis),
            "mean": float(corner_data.mean),
            "std": float(corner_data.std),
            "pct_friction": float(corner_data.pct_friction),
            "pct_slow_bump": float(corner_data.pct_slow_bump),
            "pct_slow_rebound": float(corner_data.pct_slow_rebound),
            "pct_fast_bump": float(corner_data.pct_fast_bump),
            "pct_fast_rebound": float(corner_data.pct_fast_rebound),
            "pct_curb": float(corner_data.pct_curb),
        }

    fl, fr, rl, rr = per_wheel["FL"], per_wheel["FR"], per_wheel["RL"], per_wheel["RR"]
    symmetry = {
        "front_lr_friction_diff": round(fl["pct_friction"] - fr["pct_friction"], 2),
        "rear_lr_friction_diff": round(rl["pct_friction"] - rr["pct_friction"], 2),
        "front_lr_bump_diff": round(fl["pct_slow_bump"] - fr["pct_slow_bump"], 2),
        "rear_lr_bump_diff": round(rl["pct_slow_bump"] - rr["pct_slow_bump"], 2),
        "front_lr_rebound_diff": round(fl["pct_slow_rebound"] - fr["pct_slow_rebound"], 2),
        "rear_lr_rebound_diff": round(rl["pct_slow_rebound"] - rr["pct_slow_rebound"], 2),
    }

    return SuspensionSummary(per_wheel=per_wheel, symmetry=symmetry)


def _extract_tire_grip_summary(result: TireGripResult) -> TireGripSummary:
    """Extract JSON-serializable tire grip stats from TireGripResult."""
    per_wheel = {}
    for name, corner_data in [
        ("FL", result.front_left),
        ("FR", result.front_right),
        ("RL", result.rear_left),
        ("RR", result.rear_right),
    ]:
        per_wheel[name] = {
            "mean_g": float(corner_data.mean_g),
            "std_g": float(corner_data.std_g),
            "mean_metric": float(corner_data.mean_metric),
            "std_metric": float(corner_data.std_metric),
        }

    return TireGripSummary(
        metric_mode=result.metric_mode,
        units={"metric_unit": result.metric_unit, "accel_unit": result.accel_unit},
        per_wheel=per_wheel,
    )


# ── Main Function ───────────────────────────────────────────────────────────────


def generate_session_report(
    log: LogFile,
    channel_names: dict[str, str],
    motion_ratios: MotionRatios | None = None,
    top_lap_threshold: float = 1.03,
    file_name: str = "",
) -> SessionReport:
    """Generate a complete session analysis report.

    Orchestrates all available analysis functions and produces a
    JSON-serializable report. Each analysis checks channel availability
    first; missing channels are skipped gracefully with reasons recorded.

    Parameters
    ----------
    log : LogFile
        Loaded and enriched session data (from load_session()).
    channel_names : dict[str, str]
        Channel name mapping (canonical key -> actual channel name).
    motion_ratios : MotionRatios, optional
        Suspension motion ratios for wheel velocity conversion.
    top_lap_threshold : float, default=1.03
        Include laps within this fraction of the best lap time.
    file_name : str, default=""
        Source file name for metadata.

    Returns
    -------
    SessionReport
        Complete analysis with all available data.
    """
    skipped: dict[str, str] = {}
    available: list[str] = []

    # ── 1. Session metadata ──────────────────────────────────────────────────
    logger_id = get_logger_id(log)
    profile = get_profile_for_logger(logger_id) if logger_id else None
    laps_df = log.laps.to_pandas()

    # ── 2. Lap time analysis ─────────────────────────────────────────────────
    best_lap = get_best_lap(laps_df)
    top_laps_df = get_top_laps(laps_df, threshold_pct=top_lap_threshold)
    top_lap_nums = [int(n) for n in top_laps_df["num"].tolist()]
    best_lap_num = int(best_lap["num"])
    available.append("lap_times")

    top_lap_entries = []
    top_times: list[float] = []
    for _, row in top_laps_df.iterrows():
        t = _lap_time_to_seconds(row["lap_time"])
        top_lap_entries.append(
            {"num": int(row["num"]), "lap_time_s": t, "lap_time_fmt": _format_lap_time(t)}
        )
        top_times.append(t)

    all_lap_entries = []
    for _, row in laps_df.iterrows():
        if int(row["num"]) == 0:
            continue
        t = _lap_time_to_seconds(row["lap_time"])
        all_lap_entries.append(
            {"num": int(row["num"]), "lap_time_s": t, "lap_time_fmt": _format_lap_time(t)}
        )

    best_time = _lap_time_to_seconds(best_lap["lap_time"])
    mean_top = float(np.mean(top_times)) if top_times else 0.0
    lap_times = LapTimeSummary(
        best_lap_num=best_lap_num,
        best_lap_time_s=best_time,
        best_lap_time_fmt=_format_lap_time(best_time),
        top_laps=top_lap_entries,
        mean_top_lap_time_s=mean_top,
        mean_top_lap_time_fmt=_format_lap_time(mean_top),
        std_top_lap_time_s=float(np.std(top_times)) if top_times else 0.0,
        all_lap_times=all_lap_entries,
    )

    # ── 3. Corner detection ──────────────────────────────────────────────────
    corners_raw: list[Corner] = []
    corners_info: list[CornerInfo] = []
    track_length = 0.0

    gps_ok, gps_missing = _check_channels_available(
        log, channel_names, ["gps_latitude", "gps_longitude"]
    )

    if gps_ok:
        lap_log = log.filter_by_lap(best_lap_num)
        lat_ch = channel_names["gps_latitude"]
        lon_ch = channel_names["gps_longitude"]
        gps_chs = [lat_ch, lon_ch, "distance_m"]
        gps_data = lap_log.select_channels(gps_chs).resample_to_channel(lat_ch).channels

        lat = gps_data[lat_ch].column(lat_ch).to_numpy()
        lon = gps_data[lon_ch].column(lon_ch).to_numpy()
        dist = gps_data["distance_m"].column("distance_m").to_numpy()

        # Filter to valid GPS samples
        valid_gps = (lat != 0.0) | (lon != 0.0)
        if not np.all(valid_gps):
            dist = dist[valid_gps]

        track_length = float(dist[-1]) if len(dist) > 0 else 0.0
        corners_raw = identify_corners(lat, lon)
        if corners_raw:
            corners_info = [_corner_to_info(c) for c in corners_raw]
            available.append("corners")
        else:
            skipped["corners"] = "No corners detected from GPS data"
    else:
        skipped["corners"] = f"Missing channels: {gps_missing}"

    # ── 4-6. Zones, segments, stats, driver analysis ─────────────────────────
    corner_consistency: list[CornerConsistency] = []

    zones_ok, zones_missing = _check_channels_available(
        log, channel_names, ["throttle", "brake", "gps_speed"]
    )

    if corners_raw and zones_ok:
        braking_zones, accel_zones = detect_zones_averaged(log, top_laps_df, channel_names)
        segments = create_track_segments(corners_raw, braking_zones, accel_zones, track_length)
        stats_df = compute_segment_stats(log, top_laps_df, segments, channel_names)
        available.append("zones")
        available.append("segments")

        # Throttle acceptance per corner per lap
        ta_ok, ta_missing = _check_channels_available(log, channel_names, ["lateral_g"])
        ta_per_corner: dict[int, list[tuple[int, float]]] = {}

        if ta_ok:
            throttle_ch = channel_names["throttle"]
            lateral_g_ch = channel_names["lateral_g"]
            extract_chs = list(dict.fromkeys(["distance_m", throttle_ch, lateral_g_ch]))

            for lap_num in top_lap_nums:
                try:
                    aligned = (
                        log.filter_by_lap(lap_num)
                        .select_channels(extract_chs)
                        .resample_to_channel("distance_m")
                        .channels
                    )
                except Exception:
                    continue

                ref = aligned["distance_m"]
                distance_arr = ref.column("distance_m").to_numpy()
                timecodes_arr = ref.column("timecodes").to_numpy()
                if len(distance_arr) == 0:
                    continue

                throttle_arr = aligned[throttle_ch].column(throttle_ch).to_numpy()
                lateral_g_arr = aligned[lateral_g_ch].column(lateral_g_ch).to_numpy()

                smoothed, eff_thresh = prepare_throttle_acceptance(throttle_arr, lateral_g_arr)

                for corner in corners_raw:
                    ta_result = find_throttle_acceptance_from_arrays(
                        distance_arr,
                        timecodes_arr,
                        throttle_arr,
                        lateral_g_arr,
                        corner,
                        smoothed_lateral_g=smoothed,
                        effective_threshold=eff_thresh,
                    )
                    if ta_result is not None:
                        ta_per_corner.setdefault(corner.id, []).append(
                            (lap_num, ta_result["throttle_acceptance_pct"])
                        )

            available.append("driver_analysis")
        else:
            skipped["driver_analysis"] = f"Missing channels: {ta_missing}"

        # ── GPS off-track detection ─────────────────────────────────────────
        off_track_laps: dict[int, set[int]] = {}
        if gps_ok and len(top_lap_nums) >= 3:
            lat_ch = channel_names["gps_latitude"]
            lon_ch = channel_names["gps_longitude"]
            gps_extract_chs = list(dict.fromkeys(["distance_m", lat_ch, lon_ch]))

            per_lap_gps_raw: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for lap_num in top_lap_nums:
                try:
                    aligned = (
                        log.filter_by_lap(lap_num)
                        .select_channels(gps_extract_chs)
                        .resample_to_channel("distance_m")
                        .channels
                    )
                except Exception:
                    continue
                d = aligned["distance_m"].column("distance_m").to_numpy()
                if len(d) == 0:
                    continue
                la = aligned[lat_ch].column(lat_ch).to_numpy()
                lo = aligned[lon_ch].column(lon_ch).to_numpy()
                per_lap_gps_raw[lap_num] = (d, la, lo)

            if len(per_lap_gps_raw) >= 3:
                off_track_laps = _detect_off_track_laps(
                    per_lap_gps_raw, corners_raw, deviation_threshold=10.0
                )

        # Build CornerConsistency for each corner
        for corner_raw, corner_info in zip(corners_raw, corners_info):
            excluded = sorted(off_track_laps.get(corner_raw.id, set()))

            bp_vals = _get_per_lap_metric(stats_df, corner_raw.id, "braking", "braking_point")
            min_speed_vals = _get_per_lap_metric(stats_df, corner_raw.id, "corner", "min_speed")
            exit_speed_vals = _get_per_lap_metric(stats_df, corner_raw.id, "corner", "exit_speed")
            tp_vals = _get_per_lap_metric(stats_df, corner_raw.id, "acceleration", "throttle_point")
            ta_entries = ta_per_corner.get(corner_raw.id, [])

            # New braking metrics from segment stats
            peak_brake_vals = _get_per_lap_metric(stats_df, corner_raw.id, "braking", "peak_brake")
            entry_speed_vals = _get_per_lap_metric(
                stats_df, corner_raw.id, "braking", "entry_speed"
            )
            brake_release_vals = _get_per_lap_metric(
                stats_df, corner_raw.id, "braking", "brake_release_point"
            )
            braking_distance_vals = _get_per_lap_metric(
                stats_df, corner_raw.id, "braking", "braking_distance"
            )
            mean_decel_g_vals = _get_per_lap_metric(
                stats_df, corner_raw.id, "braking", "mean_decel_g"
            )

            # Filter off-track laps from all metric lists
            if excluded:
                exc_set = set(excluded)
                bp_vals = [(n, v) for n, v in bp_vals if n not in exc_set]
                min_speed_vals = [(n, v) for n, v in min_speed_vals if n not in exc_set]
                exit_speed_vals = [(n, v) for n, v in exit_speed_vals if n not in exc_set]
                tp_vals = [(n, v) for n, v in tp_vals if n not in exc_set]
                ta_entries = [(n, v) for n, v in ta_entries if n not in exc_set]
                peak_brake_vals = [(n, v) for n, v in peak_brake_vals if n not in exc_set]
                entry_speed_vals = [(n, v) for n, v in entry_speed_vals if n not in exc_set]
                brake_release_vals = [(n, v) for n, v in brake_release_vals if n not in exc_set]
                braking_distance_vals = [
                    (n, v) for n, v in braking_distance_vals if n not in exc_set
                ]
                mean_decel_g_vals = [(n, v) for n, v in mean_decel_g_vals if n not in exc_set]

            bp_values = [v for _, v in bp_vals]
            ms_values = [v for _, v in min_speed_vals]
            es_values = [v for _, v in exit_speed_vals]
            ta_values = [v for _, v in ta_entries]
            pb_values = [v for _, v in peak_brake_vals]
            es_brk_values = [v for _, v in entry_speed_vals]
            br_values = [v for _, v in brake_release_vals]
            bd_values = [v for _, v in braking_distance_vals]
            mdg_values = [v for _, v in mean_decel_g_vals]

            accel_seg = next(
                (
                    s
                    for s in segments
                    if s.corner_id == corner_raw.id and s.segment_type == "acceleration"
                ),
                None,
            )
            accel_len = accel_seg.length if accel_seg else 0.0
            es_std = float(np.std(es_values)) if es_values else 0.0

            best_lap_data = _compute_best_lap(
                bp_vals,
                min_speed_vals,
                exit_speed_vals,
                tp_vals,
                ta_entries,
                peak_brake_vals=peak_brake_vals,
                entry_speed_vals=entry_speed_vals,
                brake_release_vals=brake_release_vals,
                braking_distance_vals=braking_distance_vals,
                mean_decel_g_vals=mean_decel_g_vals,
            )

            corner_consistency.append(
                CornerConsistency(
                    corner=corner_info,
                    ta_mean=float(np.mean(ta_values)) if ta_values else 0.0,
                    ta_std=float(np.std(ta_values)) if ta_values else 0.0,
                    bp_mean=float(np.mean(bp_values)) if bp_values else 0.0,
                    bp_std=float(np.std(bp_values)) if bp_values else 0.0,
                    min_speed_mean=float(np.mean(ms_values)) if ms_values else 0.0,
                    min_speed_std=float(np.std(ms_values)) if ms_values else 0.0,
                    exit_speed_mean=float(np.mean(es_values)) if es_values else 0.0,
                    exit_speed_std=es_std,
                    accel_zone_length=accel_len,
                    opportunity_score=es_std * accel_len if es_std > 0 else 0.0,
                    best_lap=best_lap_data,
                    peak_brake_mean=float(np.mean(pb_values)) if pb_values else None,
                    peak_brake_std=float(np.std(pb_values)) if pb_values else None,
                    entry_speed_mean=float(np.mean(es_brk_values)) if es_brk_values else None,
                    entry_speed_std=float(np.std(es_brk_values)) if es_brk_values else None,
                    brake_release_mean=float(np.mean(br_values)) if br_values else None,
                    brake_release_std=float(np.std(br_values)) if br_values else None,
                    braking_distance_mean=float(np.mean(bd_values)) if bd_values else None,
                    braking_distance_std=float(np.std(bd_values)) if bd_values else None,
                    mean_decel_g_mean=float(np.mean(mdg_values)) if mdg_values else None,
                    mean_decel_g_std=float(np.std(mdg_values)) if mdg_values else None,
                    excluded_laps=excluded,
                )
            )
    elif not corners_raw:
        skipped["zones"] = "Corners not available (prerequisite)"
    else:
        skipped["zones"] = f"Missing channels: {zones_missing}"

    # ── Braking balance ─────────────────────────────────────────────────────
    braking_balance: BrakingBalanceSummary | None = None
    brake_rear_key = channel_names.get("brake_rear", "")
    front_brake_ch = channel_names.get("brake", "")
    if (
        brake_rear_key
        and front_brake_ch
        and brake_rear_key in log.channels
        and front_brake_ch in log.channels
        and corners_raw
        and zones_ok
    ):
        # Collect per-corner balance data with individual pcts for later filtering
        per_corner_data: list[tuple[dict, list[float]]] = []

        for corner_raw, corner_info in zip(corners_raw, corners_info):
            braking_seg = next(
                (
                    s
                    for s in segments
                    if s.corner_id == corner_raw.id and s.segment_type == "braking"
                ),
                None,
            )
            if braking_seg is None:
                continue

            corner_pcts: list[float] = []
            corner_peaks: list[float] = []
            for lap_num in top_lap_nums:
                try:
                    aligned = (
                        log.filter_by_lap(lap_num)
                        .select_channels(["distance_m", front_brake_ch, brake_rear_key])
                        .resample_to_channel("distance_m")
                        .channels
                    )
                except Exception:
                    continue

                dist_arr = aligned["distance_m"].column("distance_m").to_numpy()
                mask = (dist_arr >= braking_seg.start_dist) & (dist_arr <= braking_seg.end_dist)
                if not np.any(mask):
                    continue

                front_arr = aligned[front_brake_ch].column(front_brake_ch).to_numpy()[mask]
                rear_arr = aligned[brake_rear_key].column(brake_rear_key).to_numpy()[mask]
                front_peak = float(np.max(front_arr)) if len(front_arr) > 0 else 0.0
                rear_peak = float(np.max(rear_arr)) if len(rear_arr) > 0 else 0.0
                total = front_peak + rear_peak
                if total > 0:
                    corner_pcts.append(front_peak / total * 100.0)
                    corner_peaks.append(total)

            if corner_pcts:
                entry = {
                    "corner_id": corner_raw.id,
                    "corner_name": corner_raw.name,
                    "balance_pct_mean": round(float(np.mean(corner_pcts)), 1),
                    "balance_pct_std": round(float(np.std(corner_pcts)), 1),
                    "n_samples": len(corner_pcts),
                    "peak_brake_mean": round(float(np.mean(corner_peaks)), 1),
                }
                per_corner_data.append((entry, corner_pcts))

        if per_corner_data:
            # Filter out low-brake corners (lift-and-turn) that add noise
            session_max_peak = max(e["peak_brake_mean"] for e, _ in per_corner_data)
            brake_threshold = session_max_peak * 0.2
            filtered_pcts: list[float] = []
            filtered_corners: list[dict] = []
            for entry, pcts in per_corner_data:
                if entry["peak_brake_mean"] >= brake_threshold:
                    filtered_corners.append(entry)
                    filtered_pcts.extend(pcts)

            if filtered_pcts:
                braking_balance = BrakingBalanceSummary(
                    available=True,
                    front_channel=front_brake_ch,
                    rear_channel=brake_rear_key,
                    per_corner=filtered_corners,
                    overall_balance_pct=round(float(np.mean(filtered_pcts)), 1),
                    overall_balance_std=round(float(np.std(filtered_pcts)), 1),
                    min_brake_threshold=round(brake_threshold, 1),
                )
                available.append("braking_balance")

    # ── G utilization ────────────────────────────────────────────────────────
    g_ok, g_missing = _check_channels_available(log, channel_names, ["lateral_g"])
    if g_ok and corners_raw and zones_ok and corner_consistency:
        try:
            from motorsports_data_notebook.zones import compute_g_utilization

            lateral_g_ch = channel_names["lateral_g"]
            inline_g_ch = channel_names.get("inline_g", "")
            has_inline = inline_g_ch and inline_g_ch in log.channels
            speed_ch = channel_names["gps_speed"]

            distances_list: list[np.ndarray] = []
            speeds_list: list[np.ndarray] = []
            lat_g_list: list[np.ndarray] = []
            inl_g_list: list[np.ndarray] | None = [] if has_inline else None
            valid_lap_nums: list[int] = []

            extract_chs = list(
                dict.fromkeys(
                    ["distance_m", speed_ch, lateral_g_ch] + ([inline_g_ch] if has_inline else [])
                )
            )

            for lap_num in top_lap_nums:
                try:
                    aligned = (
                        log.filter_by_lap(lap_num)
                        .select_channels(extract_chs)
                        .resample_to_channel("distance_m")
                        .channels
                    )
                except Exception:
                    continue

                dist_arr = aligned["distance_m"].column("distance_m").to_numpy()
                if len(dist_arr) == 0:
                    continue

                distances_list.append(dist_arr)
                speeds_list.append(aligned[speed_ch].column(speed_ch).to_numpy())
                lat_g_list.append(aligned[lateral_g_ch].column(lateral_g_ch).to_numpy())
                if has_inline and inl_g_list is not None:
                    inl_g_list.append(aligned[inline_g_ch].column(inline_g_ch).to_numpy())
                valid_lap_nums.append(lap_num)

            if valid_lap_nums:
                g_df = compute_g_utilization(
                    distances_list,
                    speeds_list,
                    lat_g_list,
                    inl_g_list,
                    valid_lap_nums,
                    segments,
                    corners_raw,
                )

                # Populate G utilization fields on each CornerConsistency
                for cc in corner_consistency:
                    cid = cc.corner.id
                    corner_rows = g_df[g_df["corner_id"] == cid]
                    if len(corner_rows) == 0:
                        continue
                    g_util_arr = np.asarray(corner_rows["g_utilization_pct"].dropna().values)
                    if len(g_util_arr) > 0:
                        cc.g_utilization_mean = round(float(np.mean(g_util_arr)), 1)
                        cc.g_utilization_std = round(float(np.std(g_util_arr)), 1)
                    tg_min_arr = np.asarray(corner_rows["total_g_min"].dropna().values)
                    if len(tg_min_arr) > 0:
                        cc.total_g_min_mean = round(float(np.mean(tg_min_arr)), 2)
                    # Phase of min G (most common)
                    phases = corner_rows["total_g_min_phase"].dropna().values
                    if len(phases) > 0:
                        from collections import Counter

                        cc.total_g_min_phase = Counter(phases).most_common(1)[0][0]
                    for col, attr in [
                        ("braking_g_mean", "braking_g_mean_val"),
                        ("entry_g_mean", "entry_g_mean_val"),
                        ("mid_g_mean", "mid_g_mean_val"),
                        ("exit_g_mean", "exit_g_mean_val"),
                    ]:
                        phase_vals = np.asarray(corner_rows[col].dropna().values)
                        if len(phase_vals) > 0:
                            setattr(cc, attr, round(float(np.mean(phase_vals)), 2))

                    # Populate best lap G data
                    if cc.best_lap is not None:
                        bl_rows = corner_rows[corner_rows["lap_num"] == cc.best_lap.lap_num]
                        if len(bl_rows) > 0:
                            row = bl_rows.iloc[0]
                            cc.best_lap.g_utilization_pct = round(
                                float(row["g_utilization_pct"]), 1
                            )
                            cc.best_lap.total_g_min = round(float(row["total_g_min"]), 3)
                            cc.best_lap.total_g_min_dist = round(float(row["total_g_min_dist"]), 1)
                            cc.best_lap.total_g_min_phase = str(row["total_g_min_phase"])

                available.append("g_utilization")
        except Exception as e:
            skipped["g_utilization"] = str(e)
    elif not g_ok:
        skipped["g_utilization"] = f"Missing channels: {g_missing}"

    # ── 7. Suspension ────────────────────────────────────────────────────────
    suspension = None
    susp_ok, susp_missing = _check_channels_available(
        log, channel_names, ["shock_fl", "shock_fr", "shock_rl", "shock_rr"]
    )

    if susp_ok:
        try:
            susp_result = analyze_suspension_velocity_multi_lap(
                log, top_lap_nums, channel_names=channel_names, motion_ratios=motion_ratios
            )
            suspension = _extract_suspension_summary(susp_result)
            available.append("suspension")
        except Exception as e:
            skipped["suspension"] = str(e)
    else:
        skipped["suspension"] = f"Missing channels: {susp_missing}"

    # ── 8. Tire grip ─────────────────────────────────────────────────────────
    tire_grip = None
    accel_ok, _ = _check_channels_available(log, channel_names, ["lateral_g", "inline_g"])
    press_ok, _ = _check_channels_available(
        log,
        channel_names,
        ["tpms_press_fl", "tpms_press_fr", "tpms_press_rl", "tpms_press_rr"],
    )
    temp_ok, _ = _check_channels_available(
        log,
        channel_names,
        ["tpms_temp_fl", "tpms_temp_fr", "tpms_temp_rl", "tpms_temp_rr"],
    )

    if accel_ok and (press_ok or temp_ok):
        metric_mode = "pressure" if press_ok else "temperature"
        try:
            grip_result = analyze_tire_grip_multi_lap(
                log, top_lap_nums, channel_names, metric_mode=metric_mode
            )
            tire_grip = _extract_tire_grip_summary(grip_result)
            available.append("tire_grip")
        except Exception as e:
            skipped["tire_grip"] = str(e)
    else:
        missing_parts = []
        if not accel_ok:
            missing_parts.append("lateral_g/inline_g")
        if not press_ok and not temp_ok:
            missing_parts.append("TPMS pressure/temperature")
        skipped["tire_grip"] = f"Missing channels: {', '.join(missing_parts)}"

    # ── Build report ─────────────────────────────────────────────────────────
    total_laps = len(laps_df)
    valid_laps_count = len(laps_df[laps_df["num"] > 0]) if "num" in laps_df.columns else total_laps

    metadata = SessionMetadata(
        file_name=file_name,
        logger_id=logger_id,
        vehicle_profile=profile.name if profile else None,
        total_laps=total_laps,
        valid_laps=valid_laps_count,
        top_lap_count=len(top_lap_nums),
    )

    return SessionReport(
        metadata=metadata,
        lap_times=lap_times,
        corners=corners_info,
        corner_consistency=corner_consistency,
        track_length_m=track_length,
        suspension=suspension,
        tire_grip=tire_grip,
        available_analyses=available,
        skipped_analyses=skipped,
        braking_balance=braking_balance,
    )
