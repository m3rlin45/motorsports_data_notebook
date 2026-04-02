"""MCP tool definitions for inferno-driving-coach."""

from __future__ import annotations

import json

from . import mcp


@mcp.tool()
def analyze_session(
    session_files: list[str],
    profile: str | None = None,
    threshold: float = 1.03,
    session_num: int = 1,
    output_dir: str | None = None,
) -> str:
    """Run full session analysis on telemetry data.

    Analyzes AIM (XRK/XRZ) or iRacing (IBT) telemetry files and produces
    a structured report with lap times, corner consistency, braking analysis,
    suspension data, and more.

    Parameters
    ----------
    session_files : list[str]
        Paths to telemetry files. Multiple files from the same session
        (e.g., split by a car restart) are merged automatically.
    profile : str or None
        Vehicle profile name to use. If omitted, auto-detects from logger ID.
    threshold : float
        Top lap threshold multiplier. Default 1.03 means laps within 3%
        of the best lap time are considered "top laps" for analysis.
    session_num : int
        Session number for the day (1-based). Used in the session_id field.
    output_dir : str or None
        If set, saves track map and corner comparison images to this directory.
        The track map is saved as ``<output_dir>/track_map.png`` and comparisons
        as ``<output_dir>/comparisons/comparison_t{id}_inputs.png``.

    Returns
    -------
    str
        JSON string containing the full SessionReport with image paths.
    """
    from motorsports_data_notebook.session_runner import run_session_analysis

    track_map = None
    comparison_dir = None
    if output_dir:
        from pathlib import Path

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        track_map = str(Path(output_dir) / "track_map.png")
        comparison_dir = str(Path(output_dir) / "comparisons")

    result = run_session_analysis(
        session_files=session_files,
        profile=profile,
        threshold=threshold,
        session_num=session_num,
        track_map=track_map,
        comparison_dir=comparison_dir,
    )

    output = {
        "report": result.report.to_dict(),
        "image_paths": result.image_paths,
    }
    return json.dumps(output, indent=2)


@mcp.tool()
def group_sessions(
    files: list[str],
    max_gap_minutes: float = 30.0,
) -> str:
    """Detect and group split session files into logical sessions.

    AIM loggers create a new file when the car is restarted (e.g., after
    a spin). This tool detects split sessions by comparing metadata
    timestamps and groups files that belong to the same on-track session.

    Files are grouped when they share the same driver, same venue, and
    have a time gap less than max_gap_minutes.

    Parameters
    ----------
    files : list[str]
        Sorted list of telemetry file paths.
    max_gap_minutes : float
        Maximum time gap in minutes to consider files part of the same
        session. Default: 30.

    Returns
    -------
    str
        JSON array of session groups, each with files, gap_minutes,
        driver, and venue.
    """
    from motorsports_data_notebook.session_runner import group_session_files

    groups = group_session_files(files, max_gap_minutes=max_gap_minutes)
    return json.dumps(groups, indent=2)


@mcp.tool()
def list_profiles() -> str:
    """List available vehicle profiles and their channel mappings.

    Returns all builtin and user-defined vehicle profiles, showing
    the profile name and its channel name mapping.

    Returns
    -------
    str
        JSON object with profile names as keys and channel mappings as values.
    """
    from motorsports_data_notebook.profiles import load_builtin_profiles, load_user_profiles

    all_profiles = {**load_builtin_profiles(), **load_user_profiles()}
    output = {}
    for name, profile in sorted(all_profiles.items()):
        output[name] = {
            "vehicle_name": profile.name,
            "channel_names": profile.channel_names,
        }
    return json.dumps(output, indent=2)
