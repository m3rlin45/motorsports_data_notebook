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


MODULES = {"corners": corners_fixture}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
