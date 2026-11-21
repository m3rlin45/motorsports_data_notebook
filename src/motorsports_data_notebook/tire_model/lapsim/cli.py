"""CLI of the lap simulation's gates (``just tire-lapsim <command>``).

Reports go to ``workspace/lapsim/reports/`` and the lap index cache to
``workspace/lapsim/cache/`` (neither is committed). Run under a memory cap,
like every job that reads the timeseries: ``ulimit -v 10000000``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ...tire_etl.paths import default_dataset_root, repo_root


def default_workspace() -> Path:
    return repo_root() / "workspace" / "lapsim"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lapsim", description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Where reports and caches go (default: workspace/lapsim).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "check-geometry",
        help="Gate B1: lap geometry on one clock, the clean laps and the build population.",
    )
    p_truth = sub.add_parser(
        "check-track-truth",
        help="Gate B2: build every layout's track, check it against the inertial sensors "
        "and the GSI elevation model, and give it its bank from that model.",
    )
    p_truth.add_argument(
        "--no-dem",
        action="store_true",
        help="Skip the altitude check and the bank (they fetch elevation tiles on first "
        "use); the tracks are then saved without a bank.",
    )
    p_truth.add_argument(
        "--dem-offline",
        action="store_true",
        help="Use only the elevation tiles already in workspace/lapsim/gsi/.",
    )
    sub.add_parser(
        "check-track-holdout",
        help="Gate B3: the ladder of holdouts, day against day up to car against car.",
    )
    sub.add_parser(
        "check-frame",
        help="Gate C0: the session frame record carries a ground-frame lap to the logged heat.",
    )
    sub.add_parser(
        "check-envelope-samples",
        help="Gate C1: the car's sample tables (full throttle, coast, braking zones, apexes).",
    )
    sub.add_parser(
        "check-car-fit",
        help="Gate C2: the car model from the sample tables, and through the solver.",
    )
    sub.add_parser(
        "check-car-holdout",
        help="Gate C3: the car on held-out days and layouts, against two planted wrong cars.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = args.dataset_root or default_dataset_root()
    workspace = args.workspace or default_workspace()

    from . import gates

    if args.cmd == "check-geometry":
        report = gates.check_geometry(root, cache_dir=workspace / "cache")
    elif args.cmd == "check-track-truth":
        ground = None
        if not args.no_dem:
            from .dem import GsiDem

            ground = GsiDem(workspace / "gsi", fetch=not args.dem_offline)
        from .drawings import DRAWINGS

        report = gates.check_track_truth(
            root,
            cache_dir=workspace / "cache",
            tracks_dir=workspace / "tracks",
            dem=ground,
            drawings=DRAWINGS,
        )
    elif args.cmd == "check-track-holdout":
        from . import holdout

        report = holdout.check_track_holdout(
            root, cache_dir=workspace / "cache", folds_dir=workspace / "tracks" / "folds"
        )
    elif args.cmd == "check-frame":
        from . import frame

        report = frame.check_frame(root, cache_dir=workspace / "cache")
    elif args.cmd == "check-envelope-samples":
        from . import samples

        report = samples.check_envelope_samples(
            root, cache_dir=workspace / "cache", tracks_dir=workspace / "tracks"
        )
    elif args.cmd == "check-car-fit":
        from . import car_gate

        report = car_gate.check_car_fit(
            root,
            cache_dir=workspace / "cache",
            tracks_dir=workspace / "tracks",
            cars_dir=workspace / "cars",
        )
    elif args.cmd == "check-car-holdout":
        from . import car_holdout

        report = car_holdout.check_car_holdout(
            root, cache_dir=workspace / "cache", tracks_dir=workspace / "tracks"
        )
    else:  # pragma: no cover - argparse rejects unknown commands
        parser.error(f"unknown command {args.cmd}")
    path = gates.write_report(report, workspace / "reports")
    print(report.to_markdown())
    print(f"report: {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
