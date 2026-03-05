#!/usr/bin/env python3
"""CLI entry point for session report generation.

Usage:
    uv run python scripts/analyze_session.py path/to/session.xrz [options]

Options:
    --output FILE       Write JSON to file instead of stdout
    --profile PROFILE   Override auto-detected vehicle profile name
    --threshold FLOAT   Top lap threshold (default: 1.03 = 3% slower)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

    # Output
    json_str = report.to_json()
    if args.output:
        Path(args.output).write_text(json_str)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
