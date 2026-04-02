"""High-level session analysis functions.

Shared logic used by both CLI scripts and the MCP server.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from motorsports_data_notebook._util import MergedLogFile, load_session
from motorsports_data_notebook.profiles import (
    DEFAULT_CHANNEL_NAMES,
    get_logger_id,
    get_profile_for_logger,
    load_builtin_profiles,
    load_user_profiles,
)
from motorsports_data_notebook.report import generate_session_report

if TYPE_CHECKING:
    from motorsports_data_notebook.report import SessionReport
    from motorsports_data_notebook.suspension import MotionRatios


@dataclass
class SessionAnalysisResult:
    """Result of a session analysis run."""

    report: SessionReport
    image_paths: list[str]


def resolve_profile(
    log,
    profile_name: str | None = None,
) -> tuple[dict[str, str], "MotionRatios | None"]:
    """Resolve channel names and motion ratios for a session.

    Parameters
    ----------
    log : LogFile
        Loaded telemetry session.
    profile_name : str or None
        Explicit profile name to use. If None, auto-detects from logger ID.

    Returns
    -------
    tuple[dict[str, str], MotionRatios | None]
        (channel_names, motion_ratios) tuple.

    Raises
    ------
    ValueError
        If an explicit profile name is given but not found.
    """
    if profile_name:
        all_profiles = {**load_builtin_profiles(), **load_user_profiles()}
        profile = all_profiles.get(profile_name)
        if profile is None:
            available = ", ".join(sorted(all_profiles))
            raise ValueError(f"Profile '{profile_name}' not found. Available: {available}")
        return profile.channel_names, profile.motion_ratios

    logger_id = get_logger_id(log)
    profile = get_profile_for_logger(logger_id) if logger_id else None
    if profile:
        return profile.channel_names, profile.motion_ratios
    return DEFAULT_CHANNEL_NAMES.copy(), None


def load_and_merge_sessions(session_files: list[str]) -> tuple:  # (LogFile, Path)
    """Load one or more session files, merging if multiple.

    Parameters
    ----------
    session_files : list[str]
        Paths to telemetry files. Multiple files are merged into one session.

    Returns
    -------
    tuple[LogFile, Path]
        (log, primary_path) — the loaded/merged session and the first file's path.
    """
    session_paths = [Path(f) for f in session_files]
    for p in session_paths:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    logs = [load_session(str(p)) for p in session_paths]
    if len(logs) == 1:
        log = logs[0]
    else:
        log = MergedLogFile(logs)
        print(
            f"Merged {len(logs)} files into one session " f"({len(log.laps)} total laps)",
            file=sys.stderr,
        )
    return log, session_paths[0]


def run_session_analysis(
    session_files: list[str],
    profile: str | None = None,
    threshold: float = 1.03,
    session_num: int = 1,
    track_map: str | None = None,
    comparison_dir: str | None = None,
) -> SessionAnalysisResult:
    """Run full session analysis and optionally generate images.

    Parameters
    ----------
    session_files : list[str]
        Paths to telemetry files. Multiple files are merged.
    profile : str or None
        Vehicle profile name (overrides auto-detect).
    threshold : float
        Top lap threshold multiplier (default: 1.03 = 3% slower).
    session_num : int
        Session number for the day (1-based).
    track_map : str or None
        If set, save a track map PNG to this path.
    comparison_dir : str or None
        If set, save corner comparison images to this directory.

    Returns
    -------
    SessionAnalysisResult
        Report and list of generated image paths.
    """
    log, session_path = load_and_merge_sessions(session_files)
    channel_names, motion_ratios = resolve_profile(log, profile)

    report = generate_session_report(
        log,
        channel_names,
        motion_ratios=motion_ratios,
        top_lap_threshold=threshold,
        file_name=session_path.name,
        session_num=session_num,
    )

    image_paths: list[str] = []

    # Shared reconstruction for track map and comparison images
    corners_raw = None
    segments = None
    lat = lon = dist = None
    top_laps_df = None
    top_lap_nums = None

    if (track_map or comparison_dir) and report.corners:
        try:
            from motorsports_data_notebook.channels import get_top_laps
            from motorsports_data_notebook.corners import identify_corners
            from motorsports_data_notebook.zones import (
                create_track_segments,
                detect_zones_averaged,
            )

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
                top_laps_df = get_top_laps(laps_df, threshold_pct=threshold)
                top_lap_nums = [int(n) for n in top_laps_df["num"].tolist()]
                braking_zones, accel_zones = detect_zones_averaged(log, top_laps_df, channel_names)
                valid_gps = (lat != 0.0) | (lon != 0.0)
                track_length = float(dist[valid_gps][-1]) if np.any(valid_gps) else float(dist[-1])
                segments = create_track_segments(
                    corners_raw, braking_zones, accel_zones, track_length
                )
        except Exception as e:
            print(
                f"Warning: corner/segment reconstruction failed: {e}",
                file=sys.stderr,
            )

    # Track map
    if track_map and corners_raw and segments is not None:
        assert lat is not None and lon is not None and dist is not None
        try:
            from motorsports_data_notebook.visualization import save_track_map

            save_track_map(
                lat,
                lon,
                dist,
                segments,
                track_map,
                title=f"Track Map — {session_path.stem}",
            )
            image_paths.append(track_map)
            print(f"Track map saved to {track_map}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: track map generation failed: {e}", file=sys.stderr)

    # Corner comparison images
    if comparison_dir and corners_raw and segments is not None and top_lap_nums:
        try:
            new_paths = _generate_comparison_images(
                log,
                report,
                corners_raw,
                segments,
                top_lap_nums,
                channel_names,
                comparison_dir,
            )
            image_paths.extend(new_paths)
        except Exception as e:
            print(
                f"Warning: comparison image generation failed: {e}",
                file=sys.stderr,
            )

    return SessionAnalysisResult(report=report, image_paths=image_paths)


def _parse_aim_datetime(date_str: str, time_str: str) -> datetime | None:
    """Parse AIM metadata date/time strings into a datetime.

    AIM uses MM/DD/YYYY for dates and HH:MM:SS for times.
    """
    if not date_str or not time_str:
        return None
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M:%S")
    except ValueError:
        return None


def group_session_files(
    files: list[str],
    max_gap_minutes: float = 30.0,
) -> list[dict]:
    """Group consecutive session files into logical sessions.

    Files are grouped when they have the same driver, same venue,
    and a time gap of less than max_gap_minutes between them.

    Parameters
    ----------
    files : list[str]
        Sorted list of file paths.
    max_gap_minutes : float
        Maximum time gap (in minutes) to consider files part of the
        same session. Default: 30.

    Returns
    -------
    list[dict]
        Groups with metadata. Each group has:
        - files: list of {path, start, laps} dicts
        - gap_minutes: list of gaps between consecutive files in the group
        - driver: driver name
        - venue: venue name
    """
    if not files:
        return []

    file_info: list[dict] = []
    for f in files:
        log = load_session(f)
        meta = log.metadata or {}
        dt = _parse_aim_datetime(meta.get("Log Date", ""), meta.get("Log Time", ""))
        file_info.append(
            {
                "path": f,
                "datetime": dt,
                "start": meta.get("Log Time", ""),
                "laps": len(log.laps),
                "driver": meta.get("Driver", ""),
                "venue": meta.get("Venue", ""),
            }
        )

    def _make_file_entry(info: dict) -> dict:
        return {"path": info["path"], "start": info["start"], "laps": info["laps"]}

    groups: list[dict] = [
        {
            "files": [_make_file_entry(file_info[0])],
            "gap_minutes": [],
            "driver": file_info[0]["driver"],
            "venue": file_info[0]["venue"],
        }
    ]

    for i in range(1, len(file_info)):
        prev = file_info[i - 1]
        curr = file_info[i]

        same_driver = prev["driver"] == curr["driver"]
        same_venue = prev["venue"] == curr["venue"]

        gap_min = None
        within_gap = False
        if prev["datetime"] and curr["datetime"]:
            gap_min = (curr["datetime"] - prev["datetime"]).total_seconds() / 60.0
            within_gap = gap_min < max_gap_minutes

        if same_driver and same_venue and within_gap:
            assert gap_min is not None
            groups[-1]["files"].append(_make_file_entry(curr))
            groups[-1]["gap_minutes"].append(round(gap_min, 1))
        else:
            groups.append(
                {
                    "files": [_make_file_entry(curr)],
                    "gap_minutes": [],
                    "driver": curr["driver"],
                    "venue": curr["venue"],
                }
            )

    return groups


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_comparison_images(
    log,
    report: "SessionReport",
    corners_raw,
    segments,
    top_lap_nums: list[int],
    channel_names: dict[str, str],
    comparison_dir: str,
) -> list[str]:
    """Generate corner comparison images for top corners by opportunity score.

    Returns list of generated image paths.
    """
    from motorsports_data_notebook.visualization import (
        save_corner_comparison,
        save_corner_map,
    )

    out_dir = Path(comparison_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort corners by opportunity_score, take top 5
    scored = sorted(
        report.corner_consistency,
        key=lambda c: c.opportunity_score,
        reverse=True,
    )
    top_corners = scored[:5]
    if not top_corners:
        return []

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
        return []

    generated: list[str] = []
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

        if braking_seg is None or accel_seg is None:
            continue

        x_start = braking_seg.start_dist
        x_end = accel_seg.end_dist
        best_lap_num = cc.best_lap.lap_num if cc.best_lap else report.lap_times.best_lap_num

        # Slice per-lap data for this corner region (with margin for GPS map)
        margin = 50.0
        per_lap_corner: dict[int, dict[str, np.ndarray]] = {}
        per_lap_gps: dict[int, tuple[np.ndarray, np.ndarray]] = {}

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

        if ref_lat is None or ref_lon is None:
            continue

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
        inputs_path = str(out_dir / f"comparison_t{corner.id}_inputs.png")
        save_corner_comparison(
            per_lap_corner,
            best_lap_num,
            corner.name,
            x_start,
            x_end,
            inputs_path,
            braking_start=braking_seg.start_dist,
            corner_start=corner_seg.start_dist if corner_seg else None,
            corner_end=corner_seg.end_dist if corner_seg else None,
            apex_dist=corner.apex_dist,
        )
        generated.append(inputs_path)

        # Save corner map
        if per_lap_gps:
            map_path = str(out_dir / f"comparison_t{corner.id}_map.png")
            save_corner_map(
                per_lap_gps,
                best_lap_num,
                corner.name,
                map_path,
                apex_xy=apex_xy,
            )
            generated.append(map_path)

    if generated:
        print(
            f"Corner comparison images saved to {comparison_dir} "
            f"({len([p for p in generated if '_inputs.png' in p])} corners)",
            file=sys.stderr,
        )

    return generated
