#!/usr/bin/env python3
"""CLI entry point for session report generation.

Usage:
    uv run python scripts/analyze_session.py path/to/session.xrz [options]
    uv run python scripts/analyze_session.py file1.xrk file2.xrk [options]

When multiple files are provided, they are merged into a single session
(e.g., when a session is split across files due to a car restart). Laps
are renumbered sequentially across all files.

Options:
    --output FILE            Write JSON to file instead of stdout
    --profile PROFILE        Override auto-detected vehicle profile name
    --threshold FLOAT        Top lap threshold (default: 1.03 = 3% slower)
    --track-map FILE         Save a track map image (PNG) with labeled corners
    --comparison-dir DIR     Save corner comparison images to this directory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from motorsports_data_notebook.session_runner import run_session_analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a structured session analysis report from AIM telemetry data."
    )
    parser.add_argument(
        "session_file",
        nargs="+",
        help="Path(s) to XRK/XRZ/IBT telemetry file(s). Multiple files are merged into one session.",
    )
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
    parser.add_argument(
        "--session-num",
        type=int,
        default=1,
        help="Session number for the day (1-based, used in session_id)",
    )
    args = parser.parse_args()

    try:
        result = run_session_analysis(
            session_files=args.session_file,
            profile=args.profile,
            threshold=args.threshold,
            session_num=args.session_num,
            track_map=args.track_map,
            comparison_dir=args.comparison_dir,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Output
    json_str = result.report.to_json()
    if args.output:
        Path(args.output).write_text(json_str)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
