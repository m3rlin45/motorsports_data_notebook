"""Analysis pipeline for driver consistency across corners.

Orchestrates existing motorsports_data_notebook functions into a
corner-by-corner consistency analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from motorsports_data_notebook.corners import Corner, identify_corners
from motorsports_data_notebook.driver_analysis import (
    find_throttle_acceptance_from_arrays,
    prepare_throttle_acceptance,
)
from motorsports_data_notebook.zones import (
    TrackSegment,
    compute_segment_stats_from_arrays,
    create_track_segments,
    detect_zones_from_arrays,
)

if TYPE_CHECKING:
    from motorsports_data_notebook._types import LogFile


@dataclass
class LapTraceData:
    """Per-lap trace data for detail overlay visualization.

    Attributes
    ----------
    lap_num : int
        Lap number.
    distance : np.ndarray
        Distance values in meters.
    throttle : np.ndarray
        Throttle position values (%).
    brake : np.ndarray
        Brake pressure values.
    lateral_g : np.ndarray
        Lateral G values.
    """

    lap_num: int
    distance: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    lateral_g: np.ndarray


@dataclass
class CornerConsistencyData:
    """Consistency data for a single corner across multiple laps.

    Attributes
    ----------
    corner : Corner
        The corner definition.
    ta_values : list[float]
        Throttle acceptance % per lap (excluding None results).
    ta_mean : float
        Mean throttle acceptance %.
    ta_std : float
        Std dev of throttle acceptance %.
    bp_values : list[float]
        Braking distance per lap.
    bp_std : float
        Braking point std dev (lower = more consistent).
    speed_values : list[float]
        Minimum corner speed per lap.
    speed_mean : float
        Mean minimum corner speed.
    speed_std : float
        Std dev of minimum corner speed.
    lap_traces : list[LapTraceData]
        Per-lap trace data for detail overlay.
    braking_start : float | None
        Start distance of braking zone (for detail view markers).
    """

    corner: Corner
    ta_values: list[float] = field(default_factory=list)
    ta_mean: float = 0.0
    ta_std: float = 0.0
    bp_values: list[float] = field(default_factory=list)
    bp_std: float = 0.0
    speed_values: list[float] = field(default_factory=list)
    speed_mean: float = 0.0
    speed_std: float = 0.0
    exit_speed_values: list[float] = field(default_factory=list)
    exit_speed_mean: float = 0.0
    exit_speed_std: float = 0.0
    accel_zone_length: float = 0.0
    opportunity_score: float = 0.0
    lap_traces: list[LapTraceData] = field(default_factory=list)
    braking_start: float | None = None


@dataclass
class DriverConsistencyResult:
    """Complete result of driver consistency analysis.

    Attributes
    ----------
    corners : list[Corner]
        Detected corners.
    corner_data : list[CornerConsistencyData]
        Per-corner consistency data.
    segments : list[TrackSegment]
        Track segments.
    stats_df : pd.DataFrame
        Per-lap segment statistics.
    ref_lat : np.ndarray
        Reference lap GPS latitude.
    ref_lon : np.ndarray
        Reference lap GPS longitude.
    ref_distance : np.ndarray
        Reference lap distance in meters.
    """

    corners: list[Corner]
    corner_data: list[CornerConsistencyData]
    segments: list[TrackSegment]
    stats_df: pd.DataFrame
    ref_lat: np.ndarray
    ref_lon: np.ndarray
    ref_distance: np.ndarray


def analyze_driver_consistency(
    log: "LogFile",
    selected_laps: list[int],
    channel_names: dict[str, str],
    corner_threshold: float = 0.006,
    throttle_threshold: float = 98.0,
    sustain_time_ms: float = 500.0,
) -> DriverConsistencyResult:
    """Run the full driver consistency analysis pipeline.

    Steps:
    1. Get best lap GPS data for corner detection
    2. Detect corners from GPS curvature
    3. Detect braking/acceleration zones across selected laps
    4. Create track segments combining corners and zones
    5. Compute per-lap segment statistics
    6. Compute throttle acceptance per lap per corner
    7. Extract per-lap trace data for detail overlays
    8. Aggregate into CornerConsistencyData

    Parameters
    ----------
    log : LogFile
        Loaded and enriched session data.
    selected_laps : list[int]
        Lap numbers to include in analysis.
    channel_names : dict[str, str]
        Channel name mapping with keys: throttle, brake, lateral_g,
        gps_lat, gps_lon, gps_speed.
    corner_threshold : float, default=0.006
        Curvature threshold for corner detection.
    throttle_threshold : float, default=98.0
        Throttle % threshold for throttle acceptance.
    sustain_time_ms : float, default=500.0
        Time in ms that throttle must be sustained.

    Returns
    -------
    DriverConsistencyResult
        Complete analysis results.
    """
    laps_df = log.laps.to_pandas()

    # Filter to selected laps
    selected_laps_df = laps_df[laps_df["num"].isin(selected_laps)].copy()
    if len(selected_laps_df) == 0:
        raise ValueError("No valid laps selected")

    # Step 1: Get best lap GPS for corner detection
    gps_channels = [channel_names["gps_lat"], channel_names["gps_lon"], "distance_m"]
    # Use fastest selected lap for GPS reference (avoids out-laps / incomplete laps)
    selected_laps_df["_duration"] = selected_laps_df["end_time"] - selected_laps_df["start_time"]
    best_lap = selected_laps_df.loc[selected_laps_df["_duration"].idxmin()]
    best_lap_num = int(best_lap["num"])
    best_lap_log = log.filter_by_lap(best_lap_num)
    best_channels = (
        best_lap_log.select_channels(gps_channels)
        .resample_to_channel(channel_names["gps_lat"])
        .channels
    )
    lat_table = best_channels[channel_names["gps_lat"]]
    lon_table = best_channels[channel_names["gps_lon"]]
    lat = lat_table.column(channel_names["gps_lat"]).to_numpy()
    lon = lon_table.column(channel_names["gps_lon"]).to_numpy()
    dist_table = best_channels["distance_m"]
    ref_dist = dist_table.column("distance_m").to_numpy()

    # Filter ref_dist to match valid GPS samples (identify_corners filters lat/lon internally)
    valid_gps = (lat != 0.0) | (lon != 0.0)
    if not np.all(valid_gps):
        ref_dist = ref_dist[valid_gps]

    # Step 2: Detect corners
    corners = identify_corners(lat, lon, threshold=corner_threshold)
    if not corners:
        raise ValueError("No corners detected. Try adjusting the corner detection threshold.")

    # Get track length from best lap distance
    track_length = float(ref_dist[-1])

    # Step 3-5: Single-pass per-lap extraction + zone detection + segment stats + TA
    throttle_ch = channel_names["throttle"]
    brake_ch = channel_names["brake"]
    lateral_g_ch = channel_names["lateral_g"]
    gps_speed_ch = channel_names["gps_speed"]

    # Build deduplicated channel list for extraction
    all_channels = ["distance_m", throttle_ch, brake_ch, lateral_g_ch, "speed_kmh", gps_speed_ch]
    all_channels = list(dict.fromkeys(all_channels))

    # Per-lap caches
    lap_distances: list[np.ndarray] = []
    lap_speeds: list[np.ndarray] = []
    lap_brakes: list[np.ndarray] = []
    lap_throttles: list[np.ndarray] = []
    lap_gps_speeds: list[np.ndarray] = []
    lap_lateral_gs: list[np.ndarray] = []
    lap_timecodes_list: list[np.ndarray] = []
    lap_num_list: list[int] = []
    lap_time_list: list[float] = []

    for _, lap_row in selected_laps_df.iterrows():
        lap_num = int(lap_row["num"])
        try:
            aligned = (
                log.filter_by_lap(lap_num)
                .select_channels(all_channels)
                .resample_to_channel("distance_m")
                .channels
            )
        except Exception:
            continue

        ref_table = aligned["distance_m"]
        distance = ref_table.column("distance_m").to_numpy()
        if len(distance) == 0:
            continue

        lap_distances.append(distance)
        lap_timecodes_list.append(ref_table.column("timecodes").to_numpy())
        lap_brakes.append(aligned[brake_ch].column(brake_ch).to_numpy())
        lap_throttles.append(aligned[throttle_ch].column(throttle_ch).to_numpy())
        lap_speeds.append(aligned["speed_kmh"].column("speed_kmh").to_numpy())
        lap_gps_speeds.append(aligned[gps_speed_ch].column(gps_speed_ch).to_numpy())
        lap_lateral_gs.append(aligned[lateral_g_ch].column(lateral_g_ch).to_numpy())
        lap_num_list.append(lap_num)
        lap_time_list.append(lap_row["lap_time"])

    # Zone detection (no filter_by_lap)
    braking_zones, accel_zones = detect_zones_from_arrays(
        lap_distances, lap_brakes, lap_throttles, lap_gps_speeds,
    )

    # Create segments (unchanged)
    segments = create_track_segments(corners, braking_zones, accel_zones, track_length)

    # Segment stats (no filter_by_lap)
    stats_df = compute_segment_stats_from_arrays(
        lap_distances, lap_speeds, lap_brakes, lap_throttles,
        lap_num_list, lap_time_list, segments,
    )

    # Pre-compute per-corner metadata
    corner_data_list: list[CornerConsistencyData] = []
    trace_bounds: list[tuple[float, float]] = []

    for corner in corners:
        cd = CornerConsistencyData(corner=corner)

        corner_braking = stats_df[
            (stats_df["corner_id"] == corner.id) & (stats_df["segment_type"] == "braking")
        ]
        if len(corner_braking) > 0 and "braking_point" in corner_braking.columns:
            bp_vals = corner_braking["braking_point"].dropna().tolist()
            if bp_vals:
                cd.bp_values = bp_vals
                cd.bp_std = float(np.std(bp_vals))

        corner_seg = stats_df[
            (stats_df["corner_id"] == corner.id) & (stats_df["segment_type"] == "corner")
        ]
        if len(corner_seg) > 0 and "min_speed" in corner_seg.columns:
            speed_vals = corner_seg["min_speed"].dropna().tolist()
            if speed_vals:
                cd.speed_values = speed_vals
                cd.speed_mean = float(np.mean(speed_vals))
                cd.speed_std = float(np.std(speed_vals))

        if len(corner_seg) > 0 and "exit_speed" in corner_seg.columns:
            exit_vals = corner_seg["exit_speed"].dropna().tolist()
            if exit_vals:
                cd.exit_speed_values = exit_vals
                cd.exit_speed_mean = float(np.mean(exit_vals))
                cd.exit_speed_std = float(np.std(exit_vals))

        accel_seg = next(
            (s for s in segments if s.corner_id == corner.id and s.segment_type == "acceleration"),
            None,
        )
        if accel_seg is not None:
            cd.accel_zone_length = accel_seg.length
            if cd.exit_speed_std > 0:
                cd.opportunity_score = cd.exit_speed_std * cd.accel_zone_length

        braking_seg = next(
            (s for s in segments if s.corner_id == corner.id and s.segment_type == "braking"),
            None,
        )
        trace_margin_before = 50.0
        if braking_seg is not None:
            trace_start = braking_seg.start_dist - trace_margin_before
            cd.braking_start = braking_seg.start_dist
        else:
            trace_start = corner.start_dist - 150.0
        trace_end = corner.end_dist + 50.0
        trace_bounds.append((trace_start, trace_end))

        corner_data_list.append(cd)

    # Per-lap TA + trace extraction (using cached arrays, no filter_by_lap)
    for i, (lap_distance, lap_timecodes, lap_throttle, lap_brake, lap_lateral_g) in enumerate(
        zip(lap_distances, lap_timecodes_list, lap_throttles, lap_brakes, lap_lateral_gs)
    ):
        lap_num = lap_num_list[i]

        # Prepare TA shared data once per lap
        smoothed, eff_thresh = prepare_throttle_acceptance(
            lap_throttle, lap_lateral_g, throttle_threshold=throttle_threshold,
        )

        for ci, corner in enumerate(corners):
            trace_start, trace_end = trace_bounds[ci]
            si = np.searchsorted(lap_distance, trace_start)
            ei = np.searchsorted(lap_distance, trace_end, side="right")
            if si >= ei:
                continue

            corner_data_list[ci].lap_traces.append(
                LapTraceData(
                    lap_num=lap_num,
                    distance=lap_distance[si:ei].copy(),
                    throttle=lap_throttle[si:ei].copy(),
                    brake=lap_brake[si:ei].copy(),
                    lateral_g=lap_lateral_g[si:ei].copy(),
                )
            )

            result = find_throttle_acceptance_from_arrays(
                lap_distance, lap_timecodes, lap_throttle, lap_lateral_g, corner,
                smoothed_lateral_g=smoothed, effective_threshold=eff_thresh,
                sustain_time_ms=sustain_time_ms,
            )
            if result is not None:
                corner_data_list[ci].ta_values.append(result["throttle_acceptance_pct"])

    # Post-loop: compute TA aggregates
    for cd in corner_data_list:
        if cd.ta_values:
            cd.ta_mean = float(np.mean(cd.ta_values))
            cd.ta_std = float(np.std(cd.ta_values))

    return DriverConsistencyResult(
        corners=corners,
        corner_data=corner_data_list,
        segments=segments,
        stats_df=stats_df,
        ref_lat=lat,
        ref_lon=lon,
        ref_distance=ref_dist,
    )
