#!/usr/bin/env python3
"""CLI entry point for session report generation.

Usage:
    uv run python scripts/analyze_session.py path/to/session.xrz [options]

Options:
    --output FILE            Write JSON to file instead of stdout
    --profile PROFILE        Override auto-detected vehicle profile name
    --threshold FLOAT        Top lap threshold (default: 1.03 = 3% slower)
    --track-map FILE         Save a track map image (PNG) with labeled corners
    --comparison-dir DIR     Save corner comparison images to this directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from motorsports_data_notebook._util import load_session
from motorsports_data_notebook.profiles import (
    DEFAULT_CHANNEL_NAMES,
    get_logger_id,
    get_profile_for_logger,
    load_builtin_profiles,
    load_user_profiles,
)
from motorsports_data_notebook.report import generate_session_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a structured session analysis report from AIM telemetry data."
    )
    parser.add_argument("session_file", help="Path to XRK/XRZ/IBT telemetry file")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument("--profile", "-p", help="Vehicle profile name (overrides auto-detect)")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=1.03,
        help="Top lap threshold (default: 1.03)",
    )
    parser.add_argument(
        "--track-map",
        help="Save a track map image (PNG) with labeled corners",
    )
    parser.add_argument(
        "--comparison-dir",
        help="Save corner comparison images (inputs + map) to this directory",
    )
    args = parser.parse_args()

    session_path = Path(args.session_file)
    if not session_path.exists():
        print(f"Error: file not found: {session_path}", file=sys.stderr)
        return 1

    # Load session
    try:
        log = load_session(str(session_path))
    except Exception as e:
        print(f"Error loading session: {e}", file=sys.stderr)
        return 2

    # Resolve profile and channel names
    if args.profile:
        # Look up by name in builtin + user profiles
        all_profiles = {**load_builtin_profiles(), **load_user_profiles()}
        profile = all_profiles.get(args.profile)
        if profile is None:
            print(
                f"Error: profile '{args.profile}' not found. "
                f"Available: {', '.join(sorted(all_profiles))}",
                file=sys.stderr,
            )
            return 3
        channel_names = profile.channel_names
        motion_ratios = profile.motion_ratios
    else:
        logger_id = get_logger_id(log)
        profile = get_profile_for_logger(logger_id) if logger_id else None
        if profile:
            channel_names = profile.channel_names
            motion_ratios = profile.motion_ratios
        else:
            channel_names = DEFAULT_CHANNEL_NAMES.copy()
            motion_ratios = None

    # Generate report
    try:
        report = generate_session_report(
            log,
            channel_names,
            motion_ratios=motion_ratios,
            top_lap_threshold=args.threshold,
            file_name=session_path.name,
        )
    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        return 4

    # Shared reconstruction for --track-map and --comparison-dir
    corners_raw = None
    segments = None
    lat = lon = dist = None
    top_laps_df = None
    top_lap_nums = None

    if (args.track_map or args.comparison_dir) and report.corners:
        try:
            from motorsports_data_notebook.channels import get_top_laps
            from motorsports_data_notebook.corners import identify_corners
            from motorsports_data_notebook.zones import create_track_segments, detect_zones_averaged

            lat_ch = channel_names["gps_latitude"]
            lon_ch = channel_names["gps_longitude"]
            best_lap_num = report.lap_times.best_lap_num
            gps_chs = [lat_ch, lon_ch, "distance_m"]
            gps_data = (
                log.filter_by_lap(best_lap_num)
                .select_channels(gps_chs)
                .resample_to_channel(lat_ch)
                .channels
            )
            lat = gps_data[lat_ch].column(lat_ch).to_numpy()
            lon = gps_data[lon_ch].column(lon_ch).to_numpy()
            dist = gps_data["distance_m"].column("distance_m").to_numpy()

            corners_raw = identify_corners(lat, lon)
            if corners_raw:
                laps_df = log.laps.to_pandas()
                top_laps_df = get_top_laps(laps_df, threshold_pct=args.threshold)
                top_lap_nums = [int(n) for n in top_laps_df["num"].tolist()]
                braking_zones, accel_zones = detect_zones_averaged(log, top_laps_df, channel_names)
                valid_gps = (lat != 0.0) | (lon != 0.0)
                track_length = float(dist[valid_gps][-1]) if np.any(valid_gps) else float(dist[-1])
                segments = create_track_segments(
                    corners_raw, braking_zones, accel_zones, track_length
                )
        except Exception as e:
            print(f"Warning: corner/segment reconstruction failed: {e}", file=sys.stderr)

    # Track map
    if args.track_map and corners_raw and segments is not None:
        try:
            from motorsports_data_notebook.visualization import save_track_map

            save_track_map(
                lat,
                lon,
                dist,
                segments,
                args.track_map,
                title=f"Track Map — {session_path.stem}",
            )
            print(f"Track map saved to {args.track_map}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: track map generation failed: {e}", file=sys.stderr)

    # Corner comparison images
    if args.comparison_dir and corners_raw and segments is not None and top_lap_nums:
        try:
            _generate_comparison_images(
                log, report, corners_raw, segments, top_lap_nums, channel_names, args.comparison_dir
            )
        except Exception as e:
            print(f"Warning: comparison image generation failed: {e}", file=sys.stderr)

    # Output
    json_str = report.to_json()
    if args.output:
        Path(args.output).write_text(json_str)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(json_str)

    return 0


def _generate_comparison_images(
    log,
    report,
    corners_raw,
    segments,
    top_lap_nums,
    channel_names,
    comparison_dir,
):
    """Generate corner comparison images for top corners by opportunity score."""
    from motorsports_data_notebook.corners import gps_to_local_xy
    from motorsports_data_notebook.visualization import save_corner_comparison, save_corner_map

    out_dir = Path(comparison_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort corners by opportunity_score, take top 5
    scored = sorted(report.corner_consistency, key=lambda c: c.opportunity_score, reverse=True)
    top_corners = scored[:5]
    if not top_corners:
        return

    # Build corner lookup
    corner_by_id = {c.id: c for c in corners_raw}

    # Determine channels to extract
    throttle_ch = channel_names.get("throttle", "")
    brake_ch = channel_names.get("brake", "")
    speed_ch = channel_names.get("gps_speed", "")
    lat_ch = channel_names["gps_latitude"]
    lon_ch = channel_names["gps_longitude"]
    steering_ch = channel_names.get("steering", "")
    has_steering = steering_ch and steering_ch in log.channels
    lateral_g_ch = channel_names.get("lateral_g", "")
    has_lateral_g = lateral_g_ch and lateral_g_ch in log.channels
    inline_g_ch = channel_names.get("inline_g", "")
    has_inline_g = inline_g_ch and inline_g_ch in log.channels

    extract_chs = list(
        dict.fromkeys(
            ["distance_m", speed_ch, throttle_ch, brake_ch, lat_ch, lon_ch]
            + ([steering_ch] if has_steering else [])
            + ([lateral_g_ch] if has_lateral_g else [])
            + ([inline_g_ch] if has_inline_g else [])
        )
    )

    # Extract per-lap data once for all top laps
    per_lap_all: dict[int, dict[str, np.ndarray]] = {}
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

        lap_data: dict[str, np.ndarray] = {
            "distance_m": dist_arr,
            "speed": aligned[speed_ch].column(speed_ch).to_numpy(),
            "throttle": aligned[throttle_ch].column(throttle_ch).to_numpy(),
            "brake": aligned[brake_ch].column(brake_ch).to_numpy(),
            "lat": aligned[lat_ch].column(lat_ch).to_numpy(),
            "lon": aligned[lon_ch].column(lon_ch).to_numpy(),
        }
        if has_steering:
            lap_data["steering"] = aligned[steering_ch].column(steering_ch).to_numpy()
        if has_lateral_g:
            lat_g = aligned[lateral_g_ch].column(lateral_g_ch).to_numpy()
            if has_inline_g:
                inl_g = aligned[inline_g_ch].column(inline_g_ch).to_numpy()
                lap_data["total_g"] = np.sqrt(lat_g**2 + inl_g**2)
            else:
                lap_data["total_g"] = np.abs(lat_g)

        per_lap_all[lap_num] = lap_data

    if not per_lap_all:
        return

    count = 0
    for cc in top_corners:
        corner = corner_by_id.get(cc.corner.id)
        if corner is None:
            continue

        # Find braking and acceleration segments for this corner
        braking_seg = next(
            (s for s in segments if s.corner_id == corner.id and s.segment_type == "braking"),
            None,
        )
        accel_seg = next(
            (s for s in segments if s.corner_id == corner.id and s.segment_type == "acceleration"),
            None,
        )

        # Skip if no braking or acceleration segment
        if braking_seg is None or accel_seg is None:
            continue

        x_start = braking_seg.start_dist
        x_end = accel_seg.end_dist
        best_lap_num = cc.best_lap.lap_num if cc.best_lap else report.lap_times.best_lap_num

        # Slice per-lap data for this corner region (with margin for GPS map)
        margin = 50.0  # meters of margin for map context
        per_lap_corner: dict[int, dict[str, np.ndarray]] = {}
        per_lap_gps: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Use first available lap's lat/lon as reference for gps_to_local_xy
        ref_lat = None
        ref_lon = None
        for lap_num, data in per_lap_all.items():
            d = data["distance_m"]
            mask = (d >= x_start - margin) & (d <= x_end + margin)
            if not np.any(mask):
                continue
            if ref_lat is None:
                ref_lat = data["lat"][mask]
                ref_lon = data["lon"][mask]
                break

        if ref_lat is None:
            continue

        # Compute reference center from best lap (or first available)
        lat_center = np.mean(ref_lat)
        lon_center = np.mean(ref_lon)

        for lap_num, data in per_lap_all.items():
            d = data["distance_m"]
            mask = (d >= x_start) & (d <= x_end)
            if not np.any(mask):
                continue

            sliced: dict[str, np.ndarray] = {}
            for key in ["distance_m", "speed", "throttle", "brake"]:
                sliced[key] = data[key][mask]
            if "steering" in data:
                sliced["steering"] = data["steering"][mask]
            if "total_g" in data:
                sliced["total_g"] = data["total_g"][mask]
            per_lap_corner[lap_num] = sliced

            # GPS data with margin for map
            map_mask = (d >= x_start - margin) & (d <= x_end + margin)
            if np.any(map_mask):
                lap_lat = data["lat"][map_mask]
                lap_lon = data["lon"][map_mask]
                # Local XY using shared reference center
                lat_rad = np.radians(lap_lat)
                lon_rad = np.radians(lap_lon)
                lat0 = np.radians(lat_center)
                lon0 = np.radians(lon_center)
                R = 6371000.0
                lx = R * (lon_rad - lon0) * np.cos(lat0)
                ly = R * (lat_rad - lat0)
                per_lap_gps[lap_num] = (lx, ly)

        if not per_lap_corner:
            continue

        # Apex XY on best lap
        apex_xy = None
        if best_lap_num in per_lap_all:
            best_data = per_lap_all[best_lap_num]
            apex_idx = int(np.argmin(np.abs(best_data["distance_m"] - corner.apex_dist)))
            a_lat = best_data["lat"][apex_idx]
            a_lon = best_data["lon"][apex_idx]
            lat0 = np.radians(lat_center)
            lon0 = np.radians(lon_center)
            R = 6371000.0
            ax = R * (np.radians(a_lon) - lon0) * np.cos(lat0)
            ay = R * (np.radians(a_lat) - lat0)
            apex_xy = (float(ax), float(ay))

        corner_seg = next(
            (s for s in segments if s.corner_id == corner.id and s.segment_type == "corner"),
            None,
        )

        # Save input comparison
        save_corner_comparison(
            per_lap_corner,
            best_lap_num,
            corner.name,
            x_start,
            x_end,
            str(out_dir / f"comparison_t{corner.id}_inputs.png"),
            braking_start=braking_seg.start_dist,
            corner_start=corner_seg.start_dist if corner_seg else None,
            corner_end=corner_seg.end_dist if corner_seg else None,
            apex_dist=corner.apex_dist,
        )

        # Save corner map
        if per_lap_gps:
            save_corner_map(
                per_lap_gps,
                best_lap_num,
                corner.name,
                str(out_dir / f"comparison_t{corner.id}_map.png"),
                apex_xy=apex_xy,
            )

        count += 1

    if count > 0:
        print(f"Corner comparison images saved to {comparison_dir} ({count} corners)", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
