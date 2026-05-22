"""Tests for filename discovery + parsing."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from motorsports_data_notebook.tire_etl.discovery import (
    SessionCandidate,
    group_split_sessions,
    parse_filename,
    scan_aim_tree,
)


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


def _make_cand(
    run_num: int, *, date_v=date(2026, 5, 22), car="KK-SII", session_type="Qualifying testing"
) -> SessionCandidate:
    return SessionCandidate(
        path=Path(f"/fake/{date_v}/CMD_{car}_Tsukuba_Car_{session_type}_a_{run_num:04d}.xrk"),
        date=date_v,
        driver="CMD",
        car=car,
        track_raw="Tsukuba",
        track_canonical="tsukuba_2000",
        session_type=session_type,
        run_num=run_num,
    )


def test_group_split_sessions_merges_consecutive_run_nums() -> None:
    # Three files in a row with run_num 139, 140, 141 → one merged group.
    groups = group_split_sessions([_make_cand(139), _make_cand(140), _make_cand(141)])
    assert len(groups) == 1
    assert [c.run_num for c in groups[0]] == [139, 140, 141]


def test_group_split_sessions_splits_on_run_num_gap() -> None:
    # 139, 140 are contiguous; 143 has a gap of 3 from 140, so it's its own group.
    groups = group_split_sessions([_make_cand(139), _make_cand(140), _make_cand(143)])
    assert [[c.run_num for c in g] for g in groups] == [[139, 140], [143]]


def test_group_split_sessions_splits_on_session_type_change() -> None:
    # Same date / car but different session_type → never merged even if run_nums are consecutive.
    a = _make_cand(139, session_type="Qualifying testing")
    b = _make_cand(140, session_type="Generic testing")
    groups = group_split_sessions([a, b])
    # Two singleton groups; order depends on the sort key — only the partitioning matters.
    assert sorted([tuple(c.run_num for c in g) for g in groups]) == [(139,), (140,)]


def test_group_split_sessions_singleton_groups() -> None:
    groups = group_split_sessions([_make_cand(139)])
    assert len(groups) == 1
    assert len(groups[0]) == 1


def test_group_split_sessions_empty_input() -> None:
    assert group_split_sessions([]) == []
