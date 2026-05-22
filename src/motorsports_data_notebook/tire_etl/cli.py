"""Command-line entry points for the tire ETL pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as _date
from pathlib import Path

from .paths import (
    DEFAULT_AIM_ROOT,
    DEFAULT_NOTES_ROOT,
    default_dataset_root,
)

logger = logging.getLogger(__name__)


def _parse_since(s: str | None) -> _date | None:
    if not s:
        return None
    return _date.fromisoformat(s)


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--aim-root", type=Path, default=DEFAULT_AIM_ROOT)
    p.add_argument("--notes-root", type=Path, default=DEFAULT_NOTES_ROOT)
    p.add_argument("--dataset-root", type=Path, default=default_dataset_root())


def _cmd_extract(args: argparse.Namespace) -> int:
    from .extract import run_extract

    counts = run_extract(
        aim_root=args.aim_root,
        dataset_root=args.dataset_root,
        since=_parse_since(args.since),
        only_car=args.only_car,
        force=args.force,
        retry_errors=args.retry_errors,
    )
    print(
        f"extract: scanned={counts['scanned']} skipped={counts['skipped']}"
        f" extracted={counts['extracted']} errors={counts['errors']}"
    )
    return 0 if counts["errors"] == 0 else 1


def _cmd_enrich_notes(args: argparse.Namespace) -> int:
    from .notes_parser import DEFAULT_MODEL, run_enrich_notes

    counts = run_enrich_notes(
        notes_root=args.notes_root,
        dataset_root=args.dataset_root,
        model=args.model,
        force=args.force,
    )
    print(
        f"notes: scanned={counts['scanned']} cached={counts['cached']}"
        f" parsed={counts['parsed']} errors={counts['errors']} model={args.model or DEFAULT_MODEL}"
    )
    return 0 if counts["errors"] == 0 else 1


def _cmd_enrich_weather(args: argparse.Namespace) -> int:
    from .weather import run_enrich_weather

    counts = run_enrich_weather(dataset_root=args.dataset_root)
    print(f"weather: tracks={counts['tracks']} dates={counts['dates']}")
    return 0


def _cmd_match_notes(args: argparse.Namespace) -> int:
    """Match parsed notes to sessions and write notes_matches.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .notes_match import match_notes_to_sessions, write_matches
    from .notes_parser import ParsedNotes, NotesData
    from .paths import notes_extracted_dir, sessions_dir
    import json
    from datetime import datetime as _dt

    sess_root = sessions_dir(args.dataset_root)
    if not sess_root.exists():
        print("no sessions partitions found", file=sys.stderr)
        return 1
    tables = [pq.read_table(p) for p in sorted(sess_root.glob("*.parquet"))]
    if not tables:
        print("no sessions found", file=sys.stderr)
        return 1
    sessions = pa.concat_tables(tables, promote_options="default")

    parsed: list[ParsedNotes] = []
    nex = notes_extracted_dir(args.dataset_root)
    for p in sorted(nex.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            data = NotesData.model_validate(payload["data"])
            parsed.append(
                ParsedNotes(
                    source_file=Path(payload.get("_source_file", str(p))),
                    source_sha1=payload.get("_source_sha1", ""),
                    prompt_sha1=payload.get("_prompt_sha1", ""),
                    model=payload.get("_model", ""),
                    extracted_at=payload.get("_extracted_at", ""),
                    data=data,
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("skipping malformed %s: %s", p, e)

    matches = match_notes_to_sessions(sessions, parsed)
    write_matches(args.dataset_root, matches)
    print(f"match-notes: sessions={len(sessions)} notes={len(parsed)} matches={len(matches)}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    import duckdb

    # Register session/laps/timeseries as DuckDB views.
    con = duckdb.connect(":memory:")
    root = args.dataset_root
    con.execute(f"CREATE VIEW sessions AS SELECT * FROM read_parquet('{root}/sessions/*.parquet')")
    con.execute(f"CREATE VIEW laps AS SELECT * FROM read_parquet('{root}/laps/*.parquet')")
    con.execute(
        f"CREATE VIEW timeseries AS SELECT * FROM read_parquet('{root}/timeseries/*/*.parquet')"
    )
    result = con.execute(args.sql).fetch_arrow_table()
    print(result.to_pandas().to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tire_etl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="Extract AIM sessions into the dataset")
    _common_args(p)
    p.add_argument("--since", help="YYYY-MM-DD: only sessions on/after this date")
    p.add_argument("--only-car", help="Only process files whose car field matches this string")
    p.add_argument("--force", action="store_true", help="Re-extract even if manifest matches")
    p.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-try sessions previously recorded as errors",
    )
    p.set_defaults(func=_cmd_extract)

    p = sub.add_parser("enrich-notes", help="Parse run notes via `claude -p`")
    _common_args(p)
    p.add_argument("--model", default="claude-opus-4-7", help="Claude model to use")
    p.add_argument("--force", action="store_true", help="Ignore cached JSON and re-run")
    p.set_defaults(func=_cmd_enrich_notes)

    p = sub.add_parser("enrich-weather", help="Fetch Open-Meteo historical weather")
    _common_args(p)
    p.set_defaults(func=_cmd_enrich_weather)

    p = sub.add_parser("match-notes", help="Match parsed notes to sessions")
    _common_args(p)
    p.set_defaults(func=_cmd_match_notes)

    p = sub.add_parser("query", help="Run a DuckDB SQL query over the dataset")
    _common_args(p)
    p.add_argument("sql", help="SQL query")
    p.set_defaults(func=_cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
