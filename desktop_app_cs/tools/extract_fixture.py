"""Dump lap channel data from a session file as JSON for the C# charting spike.

Run from the repo root with the project environment::

    uv run python desktop_app_cs/tools/extract_fixture.py \
        "workspace_template/data/CMD_Inferno 86_Fuji GP Sh_Generic testing_a_2248.xrz" \
        --laps 14 12 --out desktop_app_cs/App/fixtures/laps.json

Options:

``--laps N [N ...]``
    Lap numbers to include, in draw order (the first is the delta-time
    reference). Default: the two fastest laps.
``--top N``
    Instead of ``--laps``, take the N fastest laps (via ``get_top_laps``).
``--timebase CHANNEL``
    Channel whose timecodes every series is resampled onto. Default is the
    speed channel (~25 Hz on AIM GPS loggers). Pass a fast channel (e.g. a
    shock pot or the throttle) for a high-density stress fixture.
``--channels KEY [KEY ...]``
    Profile channel keys to export. Default: ``gps_speed throttle brake``.
``--no-context``
    Skip the neighbouring laps. By default each lap carries its predecessor
    and successor (``before``/``after``) so the chart can scroll past the
    lap boundaries.

The JSON shape is read by ``App/Fixture.cs``; keep the two in sync.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

from motorsports_data_notebook._util import get_channel_unit, load_session
from motorsports_data_notebook.channels import get_top_laps
from motorsports_data_notebook.profiles import DEFAULT_CHANNEL_NAMES, get_profile_for_log

DISPLAY_NAMES = {"gps_speed": "Speed", "throttle": "Throttle", "brake": "Brake"}


def _timecodes_to_seconds(timecodes: np.ndarray, lap_time_s: float) -> np.ndarray:
    """Return lap-relative seconds, whatever unit the logger's timecodes are in."""
    rel = timecodes.astype(np.float64) - float(timecodes[0])
    span = float(rel[-1]) if len(rel) > 1 else 0.0
    if span <= 0 or lap_time_s <= 0:
        return rel
    # Timecodes may be s, ms or us; pick the power of ten that makes the lap
    # span match the lap time.
    scale = 10 ** round(np.log10(span / lap_time_s))
    return rel / scale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--laps", type=int, nargs="+")
    parser.add_argument("--top", type=int)
    parser.add_argument("--timebase")
    parser.add_argument("--channels", nargs="+", default=["gps_speed", "throttle", "brake"])
    parser.add_argument("--out", type=Path, default=Path("desktop_app_cs/App/fixtures/laps.json"))
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args(argv)

    log = load_session(str(args.session))
    profile = get_profile_for_log(log)
    names = profile.channel_names if profile is not None else DEFAULT_CHANNEL_NAMES
    channel_names = {key: names[key] for key in args.channels}
    # The logger's GPS speed is m/s; load_session derives speed_kmh, which is
    # what every existing view shows. The raw channel still serves as timebase.
    timebase = args.timebase or names["gps_speed"]
    if "gps_speed" in channel_names:
        channel_names["gps_speed"] = "speed_kmh"
    forced_units = {"speed_kmh": "km/h"}

    laps_df = log.laps.to_pandas()
    laps_df["lap_time_s"] = laps_df["lap_time"].dt.total_seconds()
    if args.laps:
        lap_nums = args.laps
    else:
        top = get_top_laps(laps_df).sort_values("lap_time")
        lap_nums = [int(n) for n in top["num"].head(args.top or 2)]
    lap_times = dict(zip(laps_df["num"].astype(int), laps_df["lap_time_s"]))

    units: dict[str, str] = {}
    wanted = list(dict.fromkeys([*channel_names.values(), "distance_m", timebase]))
    dumped: dict[int, dict] = {}

    def dump_lap(num: int) -> dict:
        """Lap ``num`` as a JSON-ready dict (cached: a lap can be a neighbour twice)."""
        if num in dumped:
            return dumped[num]
        lap = log.filter_by_lap(num)
        aligned = lap.select_channels(wanted).resample_to_channel(timebase).channels

        def column(name: str) -> np.ndarray:
            table: pa.Table = aligned[name]
            return table.column(name).to_numpy(zero_copy_only=False).astype(np.float64)

        timecodes = aligned["distance_m"].column("timecodes").to_numpy()
        lap_time_s = float(lap_times[num])
        series = {}
        for key, name in channel_names.items():
            display = DISPLAY_NAMES.get(key, name)
            series[display] = np.round(column(name), 3).tolist()
            units.setdefault(display, forced_units.get(name) or get_channel_unit(aligned[name], name))

        dumped[num] = {
            "num": num,
            "lap_time_s": lap_time_s,
            "time_s": np.round(_timecodes_to_seconds(timecodes, lap_time_s), 4).tolist(),
            "distance_m": np.round(column("distance_m"), 2).tolist(),
            "series": series,
        }
        print(f"lap {num}: {len(timecodes)} samples, {lap_time_s:.3f} s", file=sys.stderr)
        return dumped[num]

    laps_out = []
    for num in lap_nums:
        entry = dict(dump_lap(num))
        if not args.no_context:
            for key, neighbour in (("before", num - 1), ("after", num + 1)):
                if neighbour in lap_times and lap_times[neighbour] > 0:
                    entry[key] = dump_lap(neighbour)
        laps_out.append(entry)

    fixture = {
        "source": args.session.name,
        "timebase": timebase,
        "units": units,
        "channels": [DISPLAY_NAMES.get(k, channel_names[k]) for k in channel_names],
        "laps": laps_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, separators=(",", ":")))
    total = sum(len(lap["time_s"]) for lap in laps_out)
    print(f"wrote {args.out} ({total} samples/channel across {len(laps_out)} laps)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
