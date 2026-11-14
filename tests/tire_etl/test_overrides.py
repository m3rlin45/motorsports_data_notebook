"""Tests for :mod:`tire_etl.overrides` and its use by discovery and the extractor."""

from __future__ import annotations

from pathlib import Path

import pytest

from motorsports_data_notebook.tire_etl.discovery import (
    group_split_sessions,
    parse_filename,
    scan_aim_tree,
)
from motorsports_data_notebook.tire_etl.extract import (
    _build_empty_session_row,
    session_channel_names,
)
from motorsports_data_notebook.tire_etl.overrides import (
    OVERRIDES_FILE,
    apply_override,
    load_overrides,
    override_for,
)

_REPO_DATASET = Path(__file__).resolve().parents[2] / "data" / "tire_dataset"


def _write(root: Path, body: str) -> Path:
    (root / OVERRIDES_FILE).write_text(body, encoding="utf-8")
    return root


_ONE = """
files:
  - date: 2026-10-02
    file: CMD_KK-F_Suzuka West_Generic testing_a_0350
    car: KK-SII
    reason: carries the KK-SII's channel set
"""


class TestLoadOverrides:
    def test_missing_file_is_no_overrides(self, tmp_path: Path) -> None:
        assert load_overrides(tmp_path) == {}

    def test_entry_is_keyed_by_day_and_stem(self, tmp_path: Path) -> None:
        overrides = load_overrides(_write(tmp_path, _ONE))
        assert len(overrides) == 1
        ov = next(iter(overrides.values()))
        assert (ov.date, ov.car) == ("2026-10-02", "KK-SII")
        assert ov.key == ("2026-10-02", "cmd_kk-f_suzuka west_generic testing_a_0350")

    def test_matches_either_extension_and_any_letter_case(self, tmp_path: Path) -> None:
        overrides = load_overrides(_write(tmp_path, _ONE))
        day = Path("/aim/2026-10-02")
        for name in (
            "CMD_KK-F_Suzuka West_Generic testing_a_0350.xrk",
            "CMD_KK-F_Suzuka West_Generic testing_a_0350.xrz",
            "cmd_kk-f_suzuka west_generic testing_a_0350.xrk",
        ):
            found = override_for(overrides, day / name)
            assert found is not None and found.car == "KK-SII"
        # Another run of the same day, and the same name on another day, are not corrected.
        assert (
            override_for(overrides, day / "CMD_KK-F_Suzuka West_Generic testing_a_0351.xrk") is None
        )
        other_day = Path("/aim/2026-10-03") / "CMD_KK-F_Suzuka West_Generic testing_a_0350.xrk"
        assert override_for(overrides, other_day) is None

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ("files:\n  - {date: 2026-10-02, file: x_a_0001, car: KK-SII}\n", "missing"),
            ("files:\n  - {date: 2026-10-02, file: x_a_0001, car: '', reason: r}\n", "missing"),
            (
                "files:\n  - {date: 2026-10-02, file: x_a_0001, car: A, reason: r, track: t}\n",
                "unknown key",
            ),
            (
                "files:\n  - {date: 2026-10-02, file: x_a_0001.xrk, car: A, reason: r}\n",
                "without its extension",
            ),
            (
                "files:\n"
                "  - {date: 2026-10-02, file: x_a_0001, car: A, reason: r}\n"
                "  - {date: 2026-10-02, file: X_A_0001, car: B, reason: r}\n",
                "second entry",
            ),
            ("sessions: []\n", "unknown top-level"),
        ],
    )
    def test_a_malformed_entry_is_an_error_not_a_skipped_correction(
        self, tmp_path: Path, body: str, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            load_overrides(_write(tmp_path, body))


class TestOverridesInDiscovery:
    def _tree(self, tmp_path: Path) -> Path:
        day = tmp_path / "aim" / "2026-10-02"
        day.mkdir(parents=True)
        for name in (
            "CMD_KK-F_Suzuka West_Generic testing_a_0350.xrk",
            "CMD_KK-F_Suzuka West_Generic testing_a_0351.xrk",
            "CMD_KK-SII_Suzuka Car_Generic testing_a_0353.xrk",
        ):
            (day / name).write_bytes(b"")
        return tmp_path / "aim"

    def test_scan_replaces_the_car_before_the_only_car_filter(self, tmp_path: Path) -> None:
        aim = self._tree(tmp_path)
        overrides = load_overrides(_write(tmp_path, _ONE))
        cars = {c.run_num: c.car for c in scan_aim_tree(aim, overrides=overrides)}
        assert cars == {350: "KK-SII", 351: "KK-F", 353: "KK-SII"}
        only = scan_aim_tree(aim, only_car="KK-SII", overrides=overrides)
        assert sorted(c.run_num for c in only) == [350, 353]
        assert {c.run_num: c.car for c in scan_aim_tree(aim)} == {
            350: "KK-F",
            351: "KK-F",
            353: "KK-SII",
        }

    def test_a_restart_pair_stays_one_group_only_when_both_files_are_corrected(
        self, tmp_path: Path
    ) -> None:
        aim = self._tree(tmp_path)
        half = load_overrides(_write(tmp_path, _ONE))
        groups = group_split_sessions(scan_aim_tree(aim, overrides=half))
        assert sorted(sorted(c.run_num for c in g) for g in groups) == [[350], [351], [353]]
        both = load_overrides(
            _write(
                tmp_path,
                _ONE
                + "  - date: 2026-10-02\n"
                + "    file: CMD_KK-F_Suzuka West_Generic testing_a_0351\n"
                + "    car: KK-SII\n"
                + "    reason: restart of a_0350\n",
            )
        )
        groups = group_split_sessions(scan_aim_tree(aim, overrides=both))
        assert sorted(sorted(c.run_num for c in g) for g in groups) == [[350, 351], [353]]

    def test_apply_override_leaves_other_fields_alone(self, tmp_path: Path) -> None:
        overrides = load_overrides(_write(tmp_path, _ONE))
        path = Path("/aim/2026-10-02/CMD_KK-F_Suzuka West_Generic testing_a_0350.xrk")
        parsed = parse_filename(path)
        fixed = apply_override(parsed, overrides)
        assert (parsed.car, fixed.car) == ("KK-F", "KK-SII")
        assert (fixed.track_raw, fixed.run_num, fixed.date) == (
            parsed.track_raw,
            parsed.run_num,
            parsed.date,
        )


class _FakeAimLog:
    def __init__(self, logger_id: int, vehicle: str) -> None:
        self.metadata = {"Logger ID": logger_id, "Vehicle": vehicle}
        self.file_name = ""


class TestOverridesInExtraction:
    def test_profile_follows_the_corrected_car_not_the_file(self) -> None:
        """The file says KK-F; the config that recorded it was the KK-SII's."""
        log = _FakeAimLog(8401203, "KK-F")
        assert session_channel_names(log)[1] == "KK-F"
        names, profile, _ = session_channel_names(log, vehicle="KK-SII")
        assert profile == "KK-SII"
        assert names["brake"] == "F_BRK"

    def test_session_row_keeps_what_the_file_declared(self, tmp_path: Path) -> None:
        overrides = load_overrides(_write(tmp_path, _ONE))
        path = Path("/aim/2026-10-02/CMD_KK-F_Suzuka West_Generic testing_a_0350.xrk")
        parsed = parse_filename(path)
        row = _build_empty_session_row(
            session_id="s",
            path=path,
            mtime_ns=0,
            file_size=0,
            cand=apply_override(parsed, overrides),
            extractor_version="0",
            car_declared=parsed.car,
        )
        assert row.column("car").to_pylist() == ["KK-SII"]
        assert row.column("car_declared").to_pylist() == ["KK-F"]

    def test_car_declared_defaults_to_the_car(self) -> None:
        path = Path("/aim/2026-10-03/CMD_KK-SII_Suzuka Car_Race_a_0355.xrk")
        row = _build_empty_session_row(
            session_id="s",
            path=path,
            mtime_ns=0,
            file_size=0,
            cand=parse_filename(path),
            extractor_version="0",
        )
        assert row.column("car_declared").to_pylist() == ["KK-SII"]


class TestCommittedOverrides:
    """The file in the repository parses and names cars the profiles know."""

    def test_committed_file_loads(self) -> None:
        overrides = load_overrides(_REPO_DATASET)
        assert overrides, "data/tire_dataset/session_overrides.yaml is missing or empty"
        from motorsports_data_notebook.tire_model.warmup_table import CAR_FIT_ALIASES

        known = set(CAR_FIT_ALIASES) | {"Inferno 86", "EK9", "RX8", "SFJ", "718 Cayman"}
        for ov in overrides.values():
            assert ov.car in known, f"{ov.file}: unknown car {ov.car!r}"
            # A correction names a different car than the file does.
            declared = parse_filename(Path(f"/aim/{ov.date}/{ov.file}.xrk")).car
            assert declared is not None and declared.casefold() != ov.car.casefold()
