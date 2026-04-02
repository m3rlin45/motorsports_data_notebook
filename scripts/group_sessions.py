#!/usr/bin/env python3
"""Detect and group split session files into logical sessions.

AIM loggers create a new file when the car is restarted (e.g., after a spin).
This script detects split sessions by comparing metadata timestamps and groups
files that belong to the same on-track session.

Usage:
    uv run python scripts/group_sessions.py file1.xrk file2.xrk file3.xrk ...

Output:
    JSON array of session groups with metadata for display:

    [
      {
        "files": [
          {"path": "file_0095.xrk", "start": "08:51:22", "laps": 9},
          {"path": "file_0096.xrk", "start": "09:11:32", "laps": 4}
        ],
        "gap_minutes": [20.2],
        "driver": "CMD",
        "venue": "Fuji GP Sh"
      },
      ...
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from motorsports_data_notebook.session_runner import group_session_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect and group split session files into logical sessions."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Telemetry file paths, sorted by session order.",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=30.0,
        help="Maximum time gap in minutes to merge sessions (default: 30).",
    )
    args = parser.parse_args()

    for f in args.files:
        if not Path(f).exists():
            print(f"Error: file not found: {f}", file=sys.stderr)
            return 1

    groups = group_session_files(args.files, max_gap_minutes=args.max_gap)
    print(json.dumps(groups, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
