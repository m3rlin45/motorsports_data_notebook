"""Dump what the Python stack reads from a session file, for the C# data layer's parity tests.

Run from the repo root. For AIM files the Rust backend is required, since
that is what the C# port binds to::

    LIBXRK_BACKEND=rust uv run python desktop_app_cs/tools/extract_session_fixture.py
    uv run python desktop_app_cs/tools/extract_session_fixture.py path/to/session.ibt \
        --out desktop_app_cs/Tests/Fixtures/session_ibt.json

Writes (default ``desktop_app_cs/Tests/Fixtures/session_86.json``), per channel
of the raw parser result, its metadata and length plus first/last/sum/min/max
of timecodes and values; the raw laps table; the metadata (every key, the
scalar entries as text, the nested ones as the JSON the FFI library writes);
and the ``load_session`` outcome (cleaned laps, derived channels).

Vector-valued channels (libibt's FixedSizeList columns for per-car arrays)
are left out, as the C# reader leaves them out.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

DEFAULT_SAMPLE = Path(
    "workspace_template/data/CMD_Inferno 86_Fuji GP Sh_Generic testing_a_2248.xrz"
)
DEFAULT_OUT = Path("desktop_app_cs/Tests/Fixtures/session_86.json")


def summary(values: np.ndarray) -> dict:
    v = values.astype(np.float64)
    return {
        "length": int(len(v)),
        "first": float(v[0]) if len(v) else None,
        "last": float(v[-1]) if len(v) else None,
        "sum": float(np.sum(v)),
        "min": float(np.min(v)) if len(v) else None,
        "max": float(np.max(v)) if len(v) else None,
    }


def ffi_json(value: object) -> object:
    """``value`` as the parsers' FFI libraries write it to JSON.

    Dates are ISO 8601 strings, non-finite numbers null, and a mapping key
    that is not a string is its JSON text.
    """
    if isinstance(value, dict):
        return {
            k if isinstance(k, str) else json.dumps(ffi_json(k)): ffi_json(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [ffi_json(v) for v in value]
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sample", type=Path, nargs="?", default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    SAMPLE, OUT = args.sample, args.out
    is_ibt = SAMPLE.suffix.lower() == ".ibt"

    if not is_ibt and os.environ.get("LIBXRK_BACKEND") != "rust":
        print("set LIBXRK_BACKEND=rust: the C# port binds the Rust parser", file=sys.stderr)
        return 2

    from motorsports_data_notebook._util import load_session

    if is_ibt:
        from libibt import ChannelMetadata, ibt

        raw = ibt(str(SAMPLE))
    else:
        from libxrk import ChannelMetadata, aim_xrk

        raw = aim_xrk(str(SAMPLE))
    channels = {}
    for name, table in raw.channels.items():
        field = table.schema.field(name)
        if pa.types.is_fixed_size_list(field.type) or pa.types.is_list(field.type):
            continue
        meta = ChannelMetadata.from_field(field)
        channels[name] = {
            "units": meta.units,
            "dec_pts": getattr(meta, "dec_pts", 0),
            "interpolate": meta.interpolate,
            "timecodes": summary(table.column("timecodes").to_numpy()),
            "values": summary(table.column(name).to_numpy(zero_copy_only=False)),
        }

    laps_raw = [
        {
            "num": int(r["num"]),
            "start_time": int(r["start_time"]),
            "end_time": int(r["end_time"]),
            "lap_type": r["lap_type"],
            "session": r.get("session"),
        }
        for r in raw.laps.to_pylist()
    ]
    metadata = {
        k: ("" if v is None else str(v))
        for k, v in raw.metadata.items()
        if (isinstance(v, (str, int, float)) or v is None) and not isinstance(v, bool)
    }
    # Compact JSON text: the parsed iRacing session info alone is 40 kB.
    metadata_nested = {
        k: json.dumps(ffi_json(v), separators=(",", ":"))
        for k, v in raw.metadata.items()
        if isinstance(v, (dict, list, tuple))
    }

    session = load_session(str(SAMPLE))
    laps_clean = [
        {"num": int(r["num"]), "lap_time_s": r["lap_time"].total_seconds()}
        for r in session.laps.to_pylist()
    ]
    distance = session.channels["distance_m"]
    t = distance.column("timecodes").to_numpy()
    d = distance.column("distance_m").to_numpy()
    lap_distance = {}
    for r in session.laps.to_pylist():
        mask = (t >= r["start_time"]) & (t < r["end_time"])
        lap_distance[str(r["num"])] = float(d[mask][-1]) if mask.any() else None

    fixture = {
        "sample": SAMPLE.name,
        "channel_order": list(channels),
        "channels": channels,
        "laps_raw": laps_raw,
        "metadata_keys": list(raw.metadata),
        "metadata": metadata,
        "metadata_nested": metadata_nested,
        "session": {
            "laps": laps_clean,
            "speed_kmh": summary(session.channels["speed_kmh"].column("speed_kmh").to_numpy()),
            "distance_m": summary(d),
            "lap_end_distance_m": lap_distance,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1) + "\n")
    print(
        f"wrote {OUT}: {len(channels)} channels, {len(laps_raw)} raw laps, {len(laps_clean)} clean laps",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
