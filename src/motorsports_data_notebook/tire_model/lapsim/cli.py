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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = args.dataset_root or default_dataset_root()
    workspace = args.workspace or default_workspace()

    from . import gates

    if args.cmd == "check-geometry":
        report = gates.check_geometry(root, cache_dir=workspace / "cache")
    else:  # pragma: no cover - argparse rejects unknown commands
        parser.error(f"unknown command {args.cmd}")
    path = gates.write_report(report, workspace / "reports")
    print(report.to_markdown())
    print(f"report: {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
