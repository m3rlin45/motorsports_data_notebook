"""Dump a Python analysis kernel's inputs and outputs for the C# port's parity tests.

Run from the repo root (Rust parser backend, as the C# port binds)::

    LIBXRK_BACKEND=rust uv run python desktop_app_cs/tools/extract_kernel_fixture.py --module corners

Each module writes ``desktop_app_cs/Tests/Fixtures/<module>_86.json`` from the
Fuji 86 sample (laps 14 and 12 unless the module says otherwise). Inputs are
dumped verbatim so the C# kernel can be driven with exactly the arrays Python
saw; outputs are dumped in full where small and as summaries where large.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

SAMPLE = Path("workspace_template/data/CMD_Inferno 86_Fuji GP Sh_Generic testing_a_2248.xrz")
FIXTURES = Path("desktop_app_cs/Tests/Fixtures")
LAPS = [14, 12]


def summary(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "length": int(len(v)),
        "first": float(v[0]),
        "last": float(v[-1]),
        "sum": float(np.sum(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def corners_fixture(log, profile) -> dict:
    from motorsports_data_notebook import corners as C

    lat_name = profile.channel_names["gps_latitude"]
    lon_name = profile.channel_names["gps_longitude"]
    laps = {}
    for num in LAPS:
        lap = log.filter_by_lap(num)
        lat = lap.channels[lat_name].column(lat_name).to_numpy()
        lon = lap.channels[lon_name].column(lon_name).to_numpy()
        x, y = C.gps_to_local_xy(lat, lon)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        distance = np.cumsum(np.sqrt(dx**2 + dy**2))
        curvature, signed, radius = C.compute_curvature(x, y)
        detected = C.identify_corners(lat, lon)
        laps[str(num)] = {
            "lat": lat.tolist(),
            "lon": lon.tolist(),
            "x": summary(x),
            "y": summary(y),
            "path_distance": summary(distance),
            "curvature": summary(curvature),
            "signed_curvature": summary(signed),
            "radius": summary(radius),
            "corners": [dataclasses.asdict(c) for c in detected],
        }
    return {"channels": {"lat": lat_name, "lon": lon_name}, "laps": laps}


def _none_safe(d: dict) -> dict:
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in d.items()}


def zones_fixture(log, profile) -> dict:
    """Zone detection, segments, segment stats and G utilization over laps 14 and 12."""
    import dataclasses

    from motorsports_data_notebook import corners as C
    from motorsports_data_notebook import zones as Z

    names = profile.channel_names
    throttle, brake, speed = names["throttle"], names["brake"], names["gps_speed"]
    lat, lon, lateral = names["gps_latitude"], names["gps_longitude"], names["lateral_g"]
    inline = names.get("inline_g", "")
    has_inline = inline in log.channels

    laps_out = {}
    arrays = {}
    for num in LAPS:
        lap_row = log.laps.to_pandas().set_index("num").loc[num]
        wanted = list(
            dict.fromkeys(
                ["distance_m", brake, throttle, speed, "speed_kmh", lateral]
                + ([inline] if has_inline else [])
            )
        )
        aligned = (
            log.filter_by_lap(num)
            .select_channels(wanted)
            .resample_to_channel("distance_m")
            .channels
        )
        col = (
            lambda n: aligned[n].column(n).to_numpy(zero_copy_only=False).astype(np.float64)
        )  # noqa: E731
        arrays[num] = {
            "distance": col("distance_m"),
            "brake": col(brake),
            "throttle": col(throttle),
            "speed_ms": col(speed),
            "speed_kmh": col("speed_kmh"),
            "lateral_g": col(lateral),
            "inline_g": col(inline) if has_inline else None,
            "lap_time_s": float(lap_row["lap_time"].total_seconds()),
        }
        braking, accel = Z.identify_zones_single_lap(
            arrays[num]["distance"],
            arrays[num]["brake"],
            arrays[num]["throttle"],
            arrays[num]["speed_ms"],
        )
        laps_out[str(num)] = {
            "lap_time_s": arrays[num]["lap_time_s"],
            "inputs": {k: v.tolist() for k, v in arrays[num].items() if isinstance(v, np.ndarray)},
            "braking_zones": braking,
            "accel_zones": accel,
        }

    order = [arrays[n] for n in LAPS]
    braking_avg, accel_avg = Z.detect_zones_from_arrays(
        [a["distance"] for a in order],
        [a["brake"] for a in order],
        [a["throttle"] for a in order],
        [a["speed_ms"] for a in order],
    )
    ref_lap = log.filter_by_lap(LAPS[0])
    corners = C.identify_corners(
        ref_lap.channels[lat].column(lat).to_numpy(), ref_lap.channels[lon].column(lon).to_numpy()
    )
    track_length = float(order[0]["distance"].max())
    segments = Z.create_track_segments(corners, braking_avg, accel_avg, track_length)
    stats = Z.compute_segment_stats_from_arrays(
        [a["distance"] for a in order],
        [a["speed_kmh"] for a in order],
        [a["brake"] for a in order],
        [a["throttle"] for a in order],
        LAPS,
        [a["lap_time_s"] for a in order],
        segments,
    )
    g_util = Z.compute_g_utilization(
        [a["distance"] for a in order],
        [a["speed_kmh"] for a in order],
        [a["lateral_g"] for a in order],
        [a["inline_g"] for a in order] if has_inline else None,
        LAPS,
        segments,
        corners,
    )
    g_util_derived = Z.compute_g_utilization(
        [a["distance"] for a in order],
        [a["speed_kmh"] for a in order],
        [a["lateral_g"] for a in order],
        None,
        LAPS,
        segments,
        corners,
    )
    return {
        "channels": {
            "throttle": throttle,
            "brake": brake,
            "speed": speed,
            "lateral_g": lateral,
            "inline_g": inline if has_inline else None,
        },
        "laps": laps_out,
        "braking_zones": braking_avg,
        "accel_zones": accel_avg,
        "track_length": track_length,
        "corners": [dataclasses.asdict(c) for c in corners],
        "segments": [dataclasses.asdict(s) for s in segments],
        "segment_stats": [_none_safe(r) for r in stats.to_dict(orient="records")],
        "g_utilization": [_none_safe(r) for r in g_util.to_dict(orient="records")],
        "g_utilization_derived_inline": [
            _none_safe(r) for r in g_util_derived.to_dict(orient="records")
        ],
    }


MODULES = {"corners": corners_fixture, "zones": zones_fixture}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--module", required=True, choices=sorted(MODULES))
    args = parser.parse_args(argv)
    if os.environ.get("LIBXRK_BACKEND") != "rust":
        print("set LIBXRK_BACKEND=rust: the C# port binds the Rust parser", file=sys.stderr)
        return 2

    from motorsports_data_notebook import profiles as P
    from motorsports_data_notebook._util import load_session

    P._get_user_profiles_path = lambda: Path("/nonexistent/profiles.yaml")  # builtins only
    log = load_session(str(SAMPLE))
    profile = P.get_profile_for_log(log)
    assert profile is not None

    fixture = {"sample": SAMPLE.name, "laps": LAPS, **MODULES[args.module](log, profile)}
    out = FIXTURES / f"{args.module}_86.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture) + "\n")
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
