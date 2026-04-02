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
from datetime import datetime
from pathlib import Path

from motorsports_data_notebook._util import load_session


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

    # Extract metadata from each file
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

    # Group consecutive files
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
