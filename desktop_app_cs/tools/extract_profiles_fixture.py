"""Dump the Python profile resolution results as a parity fixture for the C# port.

Run from the repo root::

    uv run python desktop_app_cs/tools/extract_profiles_fixture.py

Writes ``desktop_app_cs/Tests/Fixtures/profiles.json`` with the builtin
profiles as Python resolves them, plus lookup and vehicle-name cases, so the
C# ``ProfileStore`` can be checked against the reference implementation.
User profiles are excluded (the user config path is pointed at nothing).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from motorsports_data_notebook import profiles as P

OUT = Path("desktop_app_cs/Tests/Fixtures/profiles.json")

LOOKUPS = [
    ("6519522", None),
    ("6701209", None),
    ("15074", None),
    ("6603435", None),
    ("8401203", None),
    ("8401203", "KK-F"),
    ("8401203", "kk-f"),
    ("8401203", "KK-SII"),
    ("8401203", "Something Else"),
    ("iracing", None),
    ("999999", None),
    ("999999", "KK-F"),
]

VEHICLE_CASES = [
    ({"Logger ID": 8401203, "Vehicle": "KK-F"}, "CMD_KK-SII_Suzuka Car_Race_a_0358.xrk"),
    ({"Logger ID": 8401203, "Vehicle": ""}, "/data/2026-10-02/CMD_KK-F_Suzuka West_Generic testing_a_0351.xrz"),
    ({"Logger ID": 8401203}, "CMD_KK-F_Tsukuba_Car_Race_a_0344.xrz"),
    ({"Logger ID": 8401203}, ""),
    ({"Logger ID": 1}, "session.xrk"),
    ({"session_info_yaml": "x"}, "mx5_okayama_2024.ibt"),
    ({}, "CMD_KK-F_Tsukuba_Car_Race_a_0344.xrz"),
    ({"logger_id": "6701209", "Vehicle Name": "  Inferno 86 "}, ""),
    ({"Logger ID": ""}, "CMD_KK-F_Tsukuba_Car_Race_a_0344.xrz"),
]


@dataclass
class MockLog:
    metadata: dict
    file_name: str = ""


def axle(a) -> dict:
    return {
        "wheel_per_sensor": a.wheel_per_sensor,
        "damper_per_sensor": a.damper_per_sensor,
        "damper_per_wheel_fit": a.damper_per_wheel_fit,
        "pot_per_wheel": a.pot_per_wheel,
        "damper_per_wheel": a.damper_per_wheel,
        "wheel_r2": a.wheel_r2,
        "damper_r2": a.damper_r2,
        "damper_wheel_r2": a.damper_wheel_r2,
        "points": a.points,
        "excluded_points": a.excluded_points,
    }


def profile(p: P.VehicleProfile) -> dict:
    return {
        "name": p.name,
        "channel_names": dict(p.channel_names),
        "motion_ratios": {
            "front_left": p.motion_ratios.front_left,
            "front_right": p.motion_ratios.front_right,
            "rear_left": p.motion_ratios.rear_left,
            "rear_right": p.motion_ratios.rear_right,
        },
        "shock_calibration": (
            None
            if p.shock_calibration is None
            else {"front": axle(p.shock_calibration.front), "rear": axle(p.shock_calibration.rear)}
        ),
        "track_widths": (
            None
            if p.track_widths is None
            else {"front_mm": p.track_widths.front_mm, "rear_mm": p.track_widths.rear_mm}
        ),
    }


def main() -> int:
    P._get_user_profiles_path = lambda: Path("/nonexistent/profiles.yaml")  # builtins only
    builtin = P.load_builtin_profiles()
    definitions = sorted(P.load_profile_definitions())

    fixture = {
        "builtin": {key: profile(p) for key, p in builtin.items()},
        "definitions": definitions,
        "lookups": [
            {
                "logger_id": logger_id,
                "vehicle": vehicle,
                "name": (p.name if (p := P.get_profile_for_logger(logger_id, vehicle)) else None),
            }
            for logger_id, vehicle in LOOKUPS
        ],
        "vehicle_names": [
            {
                "metadata": {k: str(v) for k, v in metadata.items()},
                "file_name": file_name,
                "logger_id": P.get_logger_id(MockLog(metadata, file_name)),
                "vehicle": P.get_vehicle_name(MockLog(metadata, file_name)),
                "profile": (p.name if (p := P.get_profile_for_log(MockLog(metadata, file_name))) else None),
            }
            for metadata, file_name in VEHICLE_CASES
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {OUT}: {len(builtin)} builtin keys, {len(definitions)} definitions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
