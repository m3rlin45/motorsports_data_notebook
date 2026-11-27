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
from typing import Any

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
        "--anchor",
        choices=("first", "window", "settled"),
        default="first",
        help="An option (#421), for every fit and held-out stint alike: how a stint anchor reads "
        "the TPMS. first (default): the first valid reading. window: the means of the readings "
        "over their first 60 s, at their mean time. settled: the window, with the temperature "
        "the sensor reads 60-120 s on.",
    )
    p_holdout.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Folds scored side by side (default: half the cores; a fold's fit and the "
        "solver use about two each, and a fold needs about 2.3 GB).",
    )
    p_holdout.add_argument(
        "--heat-frame",
        choices=("tyres", "sensor"),
        default="tyres",
        help="tyres (default): every fold fitted, and the calculator's entries built, on heat "
        "rebuilt from GPS and the committed track, the simulated laps served with no frame "
        "factor (#376). sensor: the pressure model fitted on the sensor verdict's series, the "
        "simulated laps brought to it by a frame factor (for comparison).",
    )
    p_holdout.add_argument(
        "--served-rules",
        choices=("committee", "sim_outlap", "raw_rows", "shipping_unrounded", "shipping"),
        default="shipping",
        help="How the `served` arm reads and places its entry (lapsim/served.py): committee "
        "(the table's heat at the fed use in the pooled row, the model's own out-lap chain), "
        "the steps sim_outlap and raw_rows, or shipping (what tire_model.json gets).",
    )
    p_holdout.add_argument(
        "--arms",
        nargs="+",
        default=["calculator", "table", "direct", "direct_setup", "served"],
        choices=("calculator", "table", "direct", "direct_setup", "served"),
        help="The arms to score (the calculator always): e.g. `--arms calculator served` for "
        "the shipped arm alone, with no lap driven directly.",
    )
    p_holdout.add_argument(
        "--served-steps",
        nargs="*",
        default=[],
        choices=("committee", "sim_outlap", "raw_rows", "shipping_unrounded", "shipping"),
        help="More rules scored beside `served` in the same run, as arms served:<name> (kept "
        "out of the reports' common rows): the steps from the committee's rules to the "
        "shipping ones.",
    )
    p_margins = sub.add_parser(
        "pressure-pace-margins",
        help="What `check-pressure-holdout --fastest F` would ask for at several fractions "
        "(how far under the training pace the held-out sessions are); no fit is made.",
    )
    p_margins.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.8, 0.9])
    p_export = sub.add_parser(
        "export-entries",
        help="Write the committed export of the simulated laps the pressure model's simulated "
        "entries are made from (data/lapsim/exports; lapsim/sim_export.py), from every raw car's "
        "fit on all its laps on the committed tracks.",
    )
    p_export.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where the export goes (default: data/lapsim/exports, which is committed).",
    )
    p_compare = sub.add_parser(
        "compare-pressure-holdouts",
        help="Two saved runs of check-pressure-holdout on the same folds, row for row "
        "(workspace/lapsim/cache/pressure_rows/holdout_rows<tag>.parquet).",
    )
    p_compare.add_argument("--a", default="", help="The first run's tag (default: the plain run).")
    p_compare.add_argument("--b", default="__split", help="The second run's tag.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = args.dataset_root or default_dataset_root()
    workspace = args.workspace or default_workspace()
    tracks = args.tracks_dir or default_tracks()
    folds = workspace / "tracks" / "folds"

    from contextlib import nullcontext

    from ..heat_frame import model_frame, using

    # The gates compare with production, so they read heat in the frame of the
    # committed tire_model.json (#376); the standing holdout has its own
    # --heat-frame (the tyres' frame by default) and sets it per fold.
    own_frame = args.cmd in ("check-pressure-holdout", "compare-pressure-holdouts")
    frame = None if own_frame else model_frame(root)
    if frame is not None:
        logging.getLogger(__name__).info("heat frame: %s (the committed model's)", frame)
    with nullcontext() if frame is None else using(frame):
        report = _run(args, parser, root, workspace, tracks, folds)
    from . import gates

    path = gates.write_report(report, workspace / "reports")
    print(report.to_markdown())
    print(f"report: {path}")
    return 0 if report.passed else 1


def _run(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    root: Path,
    workspace: Path,
    tracks: Path,
    folds: Path,
) -> Any:
    """The report of the command ``args.cmd``."""
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
            heat_frame=args.heat_frame,
            served_rules=args.served_rules,
            served_steps=tuple(args.served_steps),
            arms=tuple(dict.fromkeys(["calculator", *args.arms])),
            anchor=args.anchor,
        )
    elif args.cmd == "pressure-pace-margins":
        from . import pressure_gate

        report = pressure_gate.check_pace_margins(root, tuple(args.fractions))
    elif args.cmd == "export-entries":
        from . import sim_export

        report = sim_export.export_entries(
            root,
            cache_dir=workspace / "cache",
            tracks_dir=tracks,
            folds_dir=folds,
            out_dir=args.out_dir,
        )
    elif args.cmd == "compare-pressure-holdouts":
        import pandas as pd

        from . import pressure_gate

        saved = workspace / "cache" / "pressure_rows"
        report = pressure_gate.compare_holdouts(
            pd.read_parquet(saved / f"holdout_rows{args.a}.parquet"),
            pd.read_parquet(saved / f"holdout_rows{args.b}.parquet"),
            (f"holdout_rows{args.a}", f"holdout_rows{args.b}"),
        )
    else:  # pragma: no cover - argparse rejects unknown commands
        parser.error(f"unknown command {args.cmd}")
    return report


if __name__ == "__main__":
    sys.exit(main())
