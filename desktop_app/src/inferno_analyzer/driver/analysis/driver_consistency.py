"""Analysis pipeline for driver consistency across corners.

Orchestrates existing motorsports_data_notebook functions into a
corner-by-corner consistency analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from motorsports_data_notebook.channels import get_best_lap_channels, get_top_laps
from motorsports_data_notebook.corners import Corner, identify_corners
from motorsports_data_notebook.driver_analysis import find_throttle_acceptance
from motorsports_data_notebook.zones import (
    TrackSegment,
    compute_segment_stats,
    create_track_segments,
    detect_zones_averaged,
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
    # Use get_top_laps to get laps for zone averaging
    top_laps = selected_laps_df

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

    # Step 3: Detect zones
    zone_channel_names = {
        "throttle": channel_names["throttle"],
        "brake": channel_names["brake"],
        "gps_speed": channel_names["gps_speed"],
    }
    braking_zones, accel_zones = detect_zones_averaged(log, top_laps, zone_channel_names)

    # Step 4: Create segments
    segments = create_track_segments(corners, braking_zones, accel_zones, track_length)

    # Step 5: Compute segment stats
    seg_channel_names = {
        "throttle": channel_names["throttle"],
        "brake": channel_names["brake"],
    }
    stats_df = compute_segment_stats(log, selected_laps_df, segments, seg_channel_names)

    # Step 6 & 7: Per-corner analysis
    corner_data_list = []
    # Channels needed for trace extraction (including timecodes for TA calculation)
    extract_channels = [
        "distance_m",
        channel_names["throttle"],
        channel_names["brake"],
        channel_names["lateral_g"],
    ]
    ta_channel_names = {
        "throttle": channel_names["throttle"],
        "lateral_g": channel_names["lateral_g"],
    }

    for corner in corners:
        cd = CornerConsistencyData(corner=corner)

        # Extract braking point stats from stats_df for this corner
        corner_braking = stats_df[
            (stats_df["corner_id"] == corner.id) & (stats_df["segment_type"] == "braking")
        ]
        if len(corner_braking) > 0 and "braking_point" in corner_braking.columns:
            bp_vals = corner_braking["braking_point"].dropna().tolist()
            if bp_vals:
                cd.bp_values = bp_vals
                cd.bp_std = float(np.std(bp_vals))

        # Extract min speed stats
        corner_seg = stats_df[
            (stats_df["corner_id"] == corner.id) & (stats_df["segment_type"] == "corner")
        ]
        if len(corner_seg) > 0 and "min_speed" in corner_seg.columns:
            speed_vals = corner_seg["min_speed"].dropna().tolist()
            if speed_vals:
                cd.speed_values = speed_vals
                cd.speed_mean = float(np.mean(speed_vals))
                cd.speed_std = float(np.std(speed_vals))

        # Find braking zone start for this corner to extend trace view
        braking_seg = next(
            (s for s in segments if s.corner_id == corner.id and s.segment_type == "braking"),
            None,
        )
        # Trace starts at braking zone start (+ 50m extra), or 150m before corner
        trace_margin_before = 50.0
        if braking_seg is not None:
            trace_start = braking_seg.start_dist - trace_margin_before
            cd.braking_start = braking_seg.start_dist
        else:
            trace_start = corner.start_dist - 150.0
        trace_end = corner.end_dist + 50.0

        # Per-lap throttle acceptance and trace extraction
        for lap_num in selected_laps:
            try:
                lap_log = log.filter_by_lap(lap_num)
                # Extract channels aligned to distance_m, preserving timecodes
                aligned = (
                    lap_log.select_channels(extract_channels)
                    .resample_to_channel("distance_m")
                    .channels
                )
                # Build DataFrame with timecodes (from any channel table)
                ref_table = aligned["distance_m"]
                corner_df = pd.DataFrame(
                    {
                        "timecodes": ref_table.column("timecodes").to_numpy(),
                        **{
                            name: aligned[name].column(name).to_numpy() for name in extract_channels
                        },
                    }
                )

                # Filter to extended region (braking zone through corner exit)
                trace_mask = (corner_df["distance_m"] >= trace_start) & (
                    corner_df["distance_m"] <= trace_end
                )
                trace_df = corner_df[trace_mask].copy()

                if len(trace_df) == 0:
                    continue

                # Throttle acceptance (uses corner boundaries internally)
                ta_result = find_throttle_acceptance(
                    trace_df,
                    corner,
                    ta_channel_names,
                    throttle_threshold=throttle_threshold,
                    sustain_time_ms=sustain_time_ms,
                )
                if ta_result is not None:
                    cd.ta_values.append(ta_result["throttle_acceptance_pct"])

                # Extract trace data
                trace = LapTraceData(
                    lap_num=lap_num,
                    distance=trace_df["distance_m"].to_numpy(),
                    throttle=trace_df[channel_names["throttle"]].to_numpy(),
                    brake=trace_df[channel_names["brake"]].to_numpy(),
                    lateral_g=trace_df[channel_names["lateral_g"]].to_numpy(),
                )
                cd.lap_traces.append(trace)

            except Exception:
                continue

        # Compute TA aggregates
        if cd.ta_values:
            cd.ta_mean = float(np.mean(cd.ta_values))
            cd.ta_std = float(np.std(cd.ta_values))

        corner_data_list.append(cd)

    return DriverConsistencyResult(
        corners=corners,
        corner_data=corner_data_list,
        segments=segments,
        stats_df=stats_df,
        ref_lat=lat,
        ref_lon=lon,
        ref_distance=ref_dist,
    )
