"""Tests for the tire_etl CLI argument parser."""

from __future__ import annotations

from datetime import date

import pytest

from motorsports_data_notebook.tire_etl.cli import _parse_since, build_parser


def test_build_parser_lists_all_subcommands() -> None:
    parser = build_parser()
    # argparse exposes subparsers via the internal `_SubParsersAction`
    # choices map.
    subactions = [
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    ]
    assert subactions, "expected at least one subparsers action"
    sub = subactions[0]
    assert set(sub.choices.keys()) >= {
        "extract",
        "enrich-notes",
        "enrich-weather",
        "match-notes",
        "query",
    }


def test_extract_subcommand_accepts_common_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "extract",
            "--aim-root",
            "/tmp/aim",
            "--dataset-root",
            "/tmp/dataset",
            "--since",
            "2026-04-01",
            "--only-car",
            "Inferno 86",
            "--force",
            "--retry-errors",
        ]
    )
    assert args.cmd == "extract"
    assert args.since == "2026-04-01"
    assert args.only_car == "Inferno 86"
    assert args.force is True
    assert args.retry_errors is True


def test_enrich_notes_subcommand_has_model_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["enrich-notes"])
    assert args.cmd == "enrich-notes"
    assert args.model == "claude-opus-4-7"
    assert args.force is False


def test_query_subcommand_requires_sql_positional() -> None:
    parser = build_parser()
    args = parser.parse_args(["query", "SELECT 1"])
    assert args.cmd == "query"
    assert args.sql == "SELECT 1"
    with pytest.raises(SystemExit):
        parser.parse_args(["query"])  # missing positional


def test_parse_since_returns_date_or_none() -> None:
    assert _parse_since(None) is None
    assert _parse_since("") is None
    assert _parse_since("2026-04-01") == date(2026, 4, 1)
    with pytest.raises(ValueError):
        _parse_since("not-a-date")
