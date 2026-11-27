"""The committed export of the simulated laps (lapsim/sim_export.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from motorsports_data_notebook.tire_model.lapsim import pace
from motorsports_data_notebook.tire_model.lapsim import sim_export as sx

from .test_car_gate import _model, _oval_track
from .test_holdout import SHARE
from .test_pace import _ranged


@pytest.fixture(scope="module")
def export():
    track = _oval_track()
    track.meta["pit"] = {"s_join_m": 400.0, "v_join_ms": 15.0, "rolling_s": 20.0, "n": 12}
    exp = pace.build_table(_ranged(_model(), oval=(0.80, 0.97)), track).laps_export(
        {"commit": "abc"}
    )
    exp.raw_car = "KK-F"
    exp.fed = {"use": 0.9, "from": "the raw car's sessions at this layout", "sessions": 4}
    return exp


class TestWriteAndLoad:
    def test_the_export_comes_back_as_it_was_written(self, export, tmp_path: Path) -> None:
        sx.write_exports({("KK-F", "oval"): export}, tmp_path, [], {"commit": "abc"})
        (got,), index = sx.load_exports(tmp_path)
        assert got.raw_car == "KK-F" and got.fed == export.fed
        np.testing.assert_array_equal(got.heat_knots(SHARE), export.heat_knots(SHARE))
        assert got.lap_time_at(0.9) == export.lap_time_at(0.9)
        assert index["files"] == [
            {"raw_car": "KK-F", "layout_id": "oval", "file": "KK-F__oval.json"}
        ]
        assert index["attribution"] == sx.ATTRIBUTION and "国土地理院" in index["attribution"]["ja"]

    def test_a_file_no_longer_made_is_removed(self, export, tmp_path: Path) -> None:
        (tmp_path / "Old_car__oval.json").write_text("{}")
        sx.write_exports({("KK-F", "oval"): export}, tmp_path, [], {})
        assert sorted(p.name for p in tmp_path.iterdir()) == ["KK-F__oval.json", "index.json"]

    def test_no_index_is_no_export(self, tmp_path: Path) -> None:
        assert sx.load_exports(tmp_path) == ([], {})

    def test_the_index_is_plain_json(self, export, tmp_path: Path) -> None:
        path = sx.write_exports({("Inferno 86", "oval"): export}, tmp_path, [{"x": 1}], {})
        data = json.loads(path.read_text())
        assert data["files"][0]["file"] == "Inferno_86__oval.json" and data["notes"] == [{"x": 1}]


class TestPowertrain:
    def test_the_latest_powertrain_of_each_raw_car(self) -> None:
        assert sx.latest_powertrain("Inferno 86", "2026-11-27") == "second engine"
        assert sx.latest_powertrain("KK-F-V2", "2026-11-27") == "KK-F-V2"
        assert sx.latest_powertrain("KK-F", "2026-11-27") == "KK-F"


class TestRounded:
    def test_seven_digits_and_once_is_enough(self, export) -> None:
        got = export.rounded(7)
        v = float(export.flying[3].v_ms[5])
        assert got.flying[3].v_ms[5] == float(f"{v:.7g}")
        again = got.rounded(7)
        np.testing.assert_array_equal(again.heat_knots(SHARE), got.heat_knots(SHARE))
        assert again.lap_time_at(0.9) == got.lap_time_at(0.9)
        assert got.fed == {**export.fed, "use": 0.9}

    def test_the_committed_json_reads_back_bit_for_bit(self, export, tmp_path: Path) -> None:
        got = export.rounded(7)
        sx.write_exports({("KK-F", "oval"): got}, tmp_path, [], {})
        (back,), _ = sx.load_exports(tmp_path)
        np.testing.assert_array_equal(back.heat_knots(SHARE), got.heat_knots(SHARE))
        np.testing.assert_array_equal(back.use, got.use)

    def test_rounding_moves_the_heat_by_less_than_a_millionth(self, export) -> None:
        a, b = export.heat_knots(SHARE), export.rounded(7).heat_knots(SHARE)
        assert np.abs(b / a - 1.0).max() < 1e-6
