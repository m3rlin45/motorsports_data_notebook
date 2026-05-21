"""Tests for filename discovery + parsing."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from motorsports_data_notebook.tire_etl.discovery import parse_filename, scan_aim_tree


def _mk(tmp_path: Path, name: str, date_str: str) -> Path:
    d = tmp_path / date_str
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"\x00")  # minimal file; we don't call load_session here
    return p


def test_parse_filename_standard(tmp_path: Path) -> None:
    p = _mk(tmp_path, "CMD_Inferno 86_Tsukuba_Generic testing_a_0192.xrk", "2026-03-15")
    c = parse_filename(p)
    assert c.date == date(2026, 3, 15)
    assert c.driver == "CMD"
    assert c.car == "Inferno 86"
    assert c.track_raw == "Tsukuba"
    assert c.track_canonical == "tsukuba_2000"
    assert c.session_type == "Generic testing"
    assert c.run_num == 192


def test_parse_filename_with_plus_driver(tmp_path: Path) -> None:
    p = _mk(tmp_path, "CMD + Maruyama_Inferno 86_Sodegaura_a_0045.xrk", "2025-06-15")
    c = parse_filename(p)
    assert c.driver == "CMD + Maruyama"
    assert c.track_canonical == "sodegaura"
    assert c.run_num == 45


def test_parse_filename_unknown_track(tmp_path: Path) -> None:
    p = _mk(tmp_path, "CMD_Something_UnknownTrack_a_0001.xrk", "2025-01-01")
    c = parse_filename(p)
    assert c.track_raw == "UnknownTrack"
    assert c.track_canonical is None


def test_scan_skips_tmp_dirs(tmp_path: Path) -> None:
    _mk(tmp_path, "CMD_Inferno 86_Tsukuba_a_0001.xrk", "2026-01-01")
    tmp_dir = tmp_path / ".tmp.drivedownload" / "2026-01-02"
    tmp_dir.mkdir(parents=True)
    (tmp_dir / "CMD_Inferno 86_Fuji_a_0002.xrk").write_bytes(b"\x00")
    results = scan_aim_tree(tmp_path)
    assert len(results) == 1
    assert results[0].track_canonical == "tsukuba_2000"


def test_scan_prefers_xrk_over_xrz(tmp_path: Path) -> None:
    d = tmp_path / "2026-01-01"
    d.mkdir()
    (d / "CMD_Inferno 86_Tsukuba_a_0001.xrk").write_bytes(b"\x00")
    (d / "CMD_Inferno 86_Tsukuba_a_0001.xrz").write_bytes(b"\x00")
    results = scan_aim_tree(tmp_path)
    assert len(results) == 1
    assert results[0].path.suffix == ".xrk"


def test_scan_since_filter(tmp_path: Path) -> None:
    _mk(tmp_path, "CMD_Inferno 86_Tsukuba_a_0001.xrk", "2025-01-01")
    _mk(tmp_path, "CMD_Inferno 86_Tsukuba_a_0002.xrk", "2026-04-01")
    results = scan_aim_tree(tmp_path, since=date(2026, 1, 1))
    assert len(results) == 1
    assert results[0].date == date(2026, 4, 1)


def test_scan_only_car_filter(tmp_path: Path) -> None:
    _mk(tmp_path, "CMD_Inferno 86_Tsukuba_a_0001.xrk", "2026-01-01")
    _mk(tmp_path, "CMD_KKSII_Tsukuba_a_0001.xrk", "2026-01-02")
    results = scan_aim_tree(tmp_path, only_car="Inferno 86")
    assert len(results) == 1
    assert results[0].car == "Inferno 86"


def test_parse_filename_missing_suffix(tmp_path: Path) -> None:
    """Names without the _letter_0000.xrk anchor still parse, just with run=None."""
    d = tmp_path / "2026-01-01"
    d.mkdir()
    p = d / "CMD_Inferno 86_Tsukuba.xrk"
    p.write_bytes(b"\x00")
    c = parse_filename(p)
    assert c.date == date(2026, 1, 1)
    assert c.run_num is None
    assert c.track_canonical == "tsukuba_2000"
