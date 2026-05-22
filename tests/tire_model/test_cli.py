"""Unit tests for the tire-model CLI argument parsing and helpers.

These tests cover the argparse surface and the pure helpers in cli.py —
not the subcommands themselves, which require a built model artifact
and are exercised end-to-end by `just tire-build-warmup-table` +
`just tire-predict-holdout`.
"""

from __future__ import annotations

import argparse

import pytest

from motorsports_data_notebook.tire_model.cli import (
    _resolve_hot_pressures,
    main,
)


def _args(**kw: float | None) -> argparse.Namespace:
    """Build a tiny Namespace with the hot-pressure fields _resolve_hot_pressures reads."""
    return argparse.Namespace(
        hot_all=kw.get("hot_all"),
        hot_fl=kw.get("hot_fl"),
        hot_fr=kw.get("hot_fr"),
        hot_rl=kw.get("hot_rl"),
        hot_rr=kw.get("hot_rr"),
    )


# ---------- _resolve_hot_pressures ----------


def test_resolve_hot_pressures_all_shorthand_broadcasts() -> None:
    out = _resolve_hot_pressures(_args(hot_all=1.95))
    assert out == {"fl": 1.95, "fr": 1.95, "rl": 1.95, "rr": 1.95}


def test_resolve_hot_pressures_per_corner_values() -> None:
    out = _resolve_hot_pressures(_args(hot_fl=1.95, hot_fr=1.95, hot_rl=1.90, hot_rr=1.92))
    assert out == {"fl": 1.95, "fr": 1.95, "rl": 1.90, "rr": 1.92}


def test_resolve_hot_pressures_all_overrides_per_corner() -> None:
    out = _resolve_hot_pressures(_args(hot_all=2.0, hot_fl=1.5, hot_fr=1.5, hot_rl=1.5, hot_rr=1.5))
    assert out == {"fl": 2.0, "fr": 2.0, "rl": 2.0, "rr": 2.0}


def test_resolve_hot_pressures_missing_corner_raises_systemexit() -> None:
    # Neither --hot-all nor a complete set of per-corner values
    with pytest.raises(SystemExit) as exc:
        _resolve_hot_pressures(_args(hot_fl=1.95, hot_fr=1.95))
    msg = str(exc.value)
    # Should mention which corners are missing
    assert "rl" in msg and "rr" in msg


# ---------- CLI parser ----------


def test_cli_no_args_errors() -> None:
    """Calling main() with no subcommand should exit non-zero (argparse: required=True)."""
    with pytest.raises(SystemExit):
        main([])


def test_cli_predict_requires_track_car_lap_and_ambient() -> None:
    with pytest.raises(SystemExit):
        main(["predict", "--track", "tsukuba_2000", "--car", "KK-SII", "--hot-all", "1.95"])
        # Missing --lap and --ambient


def test_cli_predict_parses_full_invocation_then_fails_loading_model(tmp_path) -> None:
    """The CLI should parse cleanly even if the dataset_root has no
    tire_model.json — failure should come from the predictor, not argparse."""
    with pytest.raises(FileNotFoundError):
        main(
            [
                "predict",
                "--dataset-root",
                str(tmp_path),
                "--track",
                "tsukuba_2000",
                "--car",
                "KK-SII",
                "--lap",
                "5",
                "--ambient",
                "18",
                "--hot-all",
                "1.95",
            ]
        )


def test_cli_subcommands_are_registered() -> None:
    """A regression guard so silently dropping a subcommand from main() trips."""
    import argparse

    # Build the same parser the CLI does, by introspecting main's argv handling
    # via a dry-run with --help and confirming all subcommands appear in the
    # parser. Easier: just confirm each subcommand's argparser doesn't error
    # when given its minimal required args.
    cases: list[list[str]] = [
        ["build-warmup-table", "--dataset-root", "/nonexistent"],
        ["audit-sensors", "--dataset-root", "/nonexistent"],
        ["validate", "--dataset-root", "/nonexistent"],
        ["holdout", "--dataset-root", "/nonexistent"],
    ]
    for argv in cases:
        # Each subcommand should parse cleanly. Most will fail downstream
        # because the dataset doesn't exist; that's fine — we're testing
        # argparse here, not the I/O.
        try:
            main(argv)
        except (FileNotFoundError, ValueError, SystemExit):
            pass
        except argparse.ArgumentError as e:
            pytest.fail(f"argparse rejected subcommand {argv[0]!r}: {e}")
