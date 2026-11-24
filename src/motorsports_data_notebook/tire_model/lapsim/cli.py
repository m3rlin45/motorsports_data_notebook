"""CLI of the lap simulation's gates (``just tire-lapsim <command>``).

Reports go to ``workspace/lapsim/reports/`` and the lap index cache to
``workspace/lapsim/cache/`` (neither is committed). The tracks gate B2
builds are committed, with their bank: they are read from and written to
``data/lapsim/tracks/`` (owner, 2026-11-24; the attribution to GSI is in
``data/lapsim/README.md``). The tracks of gate B3's folds are caches and
stay in ``workspace/lapsim/tracks/folds/``. Run under a memory cap, like
every job that reads the timeseries: ``ulimit -v 10000000``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ...tire_etl.paths import default_dataset_root, repo_root


def default_workspace() -> Path:
    return repo_root() / "workspace" / "lapsim"


def default_tracks() -> Path:
    """Where the committed tracks are (``data/lapsim/tracks``)."""
    return repo_root() / "data" / "lapsim" / "tracks"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lapsim", description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Where reports and caches go (default: workspace/lapsim).",
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=None,
        help="Where the tracks are read from and saved (default: data/lapsim/tracks, "
        "which is committed).",
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
        "check-heat-chain",
        help="Gate D1 on logged laps: production's heat chain on a lap against production's "
        "own per-lap heat, holes, the phase of the bins and the 0.5 s smoothing.",
    )
    sub.add_parser(
        "check-pace-table",
        help="Gate D1, the pace table: every model car at every layout with a track, held to "
        "a direct solve at twenty lap times; the tables go to workspace/lapsim/pace/.",
    )
    sub.add_parser(
        "check-wet",
        help="Gate D2: each wet build lap driven on the dry track, with the wet braking term, "
        "on the wet line and with both, against a logged lap of the same wet day.",
    )
    sub.add_parser(
        "check-car-holdout",
        help="Gate C3: the car on held-out days and layouts, against two planted wrong cars.",
    )
    p_pressure = sub.add_parser(
        "check-pressure",
        help="Gate F: the cold pressure a simulated lap leads to at a track the car has not "
        "run, against paired real days; one refit of the pressure model per fold, kept in "
        "workspace/lapsim/cache/pressure_folds/.",
    )
    p_pressure.add_argument(
        "--yardstick",
        action="store_true",
        help="Stop at the yardstick: the folds, the cells, another day against a day, the "
        "ceiling and what the simulation replaces, with no simulated lap.",
    )
    p_pressure.add_argument(
        "--no-dates",
        action="store_true",
        help="Leave the leave-date-out folds out (ten more fits: the reference level on "
        "cells the fit has seen, and pressure against the TPMS with the cell in the fit).",
    )
    p_holdout = sub.add_parser(
        "check-pressure-holdout",
        help="The calculator's own holdout (MAE over folds of held-out sessions, hot pressure "
        "against the TPMS) for a calculator whose entries are simulated laps, beside the "
        "calculator as it is on the same folds and rows.",
    )
    p_holdout.add_argument("--n-folds", type=int, default=20)
    p_holdout.add_argument("--n-per-bucket", type=int, default=2)
    p_holdout.add_argument("--min-bucket-size", type=int, default=10)
    p_holdout.add_argument(
        "--split",
        nargs="+",
        default=[],
        metavar="CAR",
        help="An experiment: fit these raw cars as a pooled car of their own, named after "
        "the first (e.g. --split KK-F KK-F-V2), on the same folds.",
    )
    p_holdout.add_argument(
        "--fastest",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help="A holdout by pace in place of the folds: per raw car and track the fastest "
        "fraction of the dry sessions (by the lap time typed in) is held out of every model.",
    )
    p_holdout.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Folds scored side by side (default: half the cores; a fold's fit and the "
        "solver use about two each, and a fold needs about 2.3 GB).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = args.dataset_root or default_dataset_root()
    workspace = args.workspace or default_workspace()
    tracks = args.tracks_dir or default_tracks()
    folds = workspace / "tracks" / "folds"

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
            tracks_dir=tracks,
            dem=ground,
            drawings=DRAWINGS,
        )
    elif args.cmd == "check-track-holdout":
        from . import holdout

        report = holdout.check_track_holdout(root, cache_dir=workspace / "cache", folds_dir=folds)
    elif args.cmd == "check-frame":
        from . import frame

        report = frame.check_frame(root, cache_dir=workspace / "cache")
    elif args.cmd == "check-envelope-samples":
        from . import samples

        report = samples.check_envelope_samples(
            root, cache_dir=workspace / "cache", tracks_dir=tracks
        )
    elif args.cmd == "check-car-fit":
        from . import car_gate

        report = car_gate.check_car_fit(
            root,
            cache_dir=workspace / "cache",
            tracks_dir=tracks,
            cars_dir=workspace / "cars",
        )
    elif args.cmd == "check-car-holdout":
        from . import car_holdout

        report = car_holdout.check_car_holdout(
            root, cache_dir=workspace / "cache", tracks_dir=tracks, folds_dir=folds
        )
    elif args.cmd == "check-heat-chain":
        from . import heat_gate

        report = heat_gate.check_heat_chain(
            root,
            cache_dir=workspace / "cache",
            tracks_dir=tracks,
            cars_dir=workspace / "cars",
        )
    elif args.cmd == "check-pace-table":
        from . import pace_gate

        report = pace_gate.check_pace_table(
            root,
            tracks_dir=tracks,
            cars_dir=workspace / "cars",
            out_dir=workspace / "pace",
        )
    elif args.cmd == "check-wet":
        from . import wet_gate

        report = wet_gate.check_wet(
            root,
            cache_dir=workspace / "cache",
            tracks_dir=tracks,
            cars_dir=workspace / "cars",
        )
    elif args.cmd == "check-pressure":
        from . import pressure_gate

        report = pressure_gate.check_pressure(
            root,
            cache_dir=workspace / "cache",
            tracks_dir=tracks,
            folds_dir=folds,
            simulate=not args.yardstick,
            dates=not args.no_dates,
        )
    elif args.cmd == "check-pressure-holdout":
        from . import pressure_gate

        report = pressure_gate.check_pressure_holdout(
            root,
            cache_dir=workspace / "cache",
            tracks_dir=tracks,
            folds_dir=folds,
            n_folds=args.n_folds,
            n_per_bucket=args.n_per_bucket,
            min_bucket_size=args.min_bucket_size,
            workers=args.workers,
            split=tuple(args.split),
            fastest=args.fastest,
        )
    else:  # pragma: no cover - argparse rejects unknown commands
        parser.error(f"unknown command {args.cmd}")
    path = gates.write_report(report, workspace / "reports")
    print(report.to_markdown())
    print(f"report: {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
