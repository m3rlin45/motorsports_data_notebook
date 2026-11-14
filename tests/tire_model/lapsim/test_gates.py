"""Gate B1 of the lap simulation (`lapsim check-geometry`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import cli, gates
from motorsports_data_notebook.tire_model.lapsim import geometry as geo

from ._synthetic import LAYOUT_ID, write_dataset


@pytest.fixture()
def root(tmp_path):
    geo._INDEX_CACHE.clear()
    sessions = {
        f"2026-01-1{i}_s{i}": {
            "car": "car",
            "date": f"2026-01-1{i}",
            "kwargs": {"n_laps": 3, "seed": i},
        }
        for i in range(3)
    }
    sessions["2026-01-12_s2"]["kwargs"]["slew"] = (2, 4.0)
    write_dataset(tmp_path / "data", sessions)
    return tmp_path / "data"


class TestMdTable:
    def test_table(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": [1.23456, np.nan], "c": [1, 2]})
        text = gates.md_table(df, floats="{:.3g}")
        assert text.splitlines() == [
            "| a | b | c |",
            "|---|---|---|",
            "| x | 1.23 | 1 |",
            "| y |  | 2 |",
        ]

    def test_empty(self):
        assert gates.md_table(pd.DataFrame({"a": []})) == "_none_"


class TestUnmatchedCauses:
    def test_causes(self):
        df = pd.DataFrame(
            {
                "session_id": ["a"] * 5 + ["b"],
                "lap_num": [0, 1, 2, 3, 4, 1],
                "lap_type": "full",
                "layout_id": LAYOUT_ID,
                "lap_time_s": [300.0, 60.0, 61.0, 5.0, 62.0, 60.0],
                "matched": [False, True, True, False, False, False],
                "n_crossings": [5, 5, 5, 5, 5, 1],
                "lap_source": "logger",
            }
        )
        out = gates.unmatched_causes(df).set_index(["session_id", "lap_num"])["cause"]
        assert out.to_dict() == {
            ("a", 0): "not_flying",
            ("a", 3): "fragment",
            ("a", 4): "unexplained",
            ("b", 1): "no_crossings",
        }


class TestCheckGeometry:
    def test_gate_on_a_sound_dataset(self, root):
        report = gates.check_geometry(root)
        items = {i.name: i for i in report.items}
        assert items["Doppler course closes"].passed is True
        assert items["Path length = ∫V dt"].passed is True
        assert items["Lap length = session row"].passed is True
        assert items["Lap length spread"].passed is True
        assert items["No unflagged clean lap over 3 g"].passed is True
        assert items["Laps map to crossing intervals"].passed is True
        assert items["Gyro scale spread"].passed is True
        assert items["Lag removed"].passed is None
        assert items["Build population"].passed is None
        # one flagged lap of nine is over the 2 % bound
        assert items["Flagged laps"].passed is False
        assert "1 of 9 clean laps" in items["Flagged laps"].measured
        assert not report.passed
        assert gates.expected_turns(geo.clean_lap_index(root))[LAYOUT_ID] == 1.0

    def test_markdown_and_report_file(self, root, tmp_path):
        report = gates.check_geometry(root)
        text = report.to_markdown()
        assert text.startswith("# B1: ")
        assert "| Doppler course closes |" in text
        assert "## Build population" in text
        path = gates.write_report(report, tmp_path / "reports")
        assert path.name == "B1_geometry.md"
        assert path.read_text() == text

    def test_path_excess_profile(self, root):
        build = geo.clean_lap_index(root)
        build = build[build["in_build"]]
        prof = gates.path_excess_profile(root, build, LAYOUT_ID, bin_m=100.0)
        assert prof.attrs["n_laps"] == len(build)
        # exact positions and speeds: no bin gains more than a few centimetres
        assert prof["excess_m"].abs().max() < 0.2

    def test_report_passes_when_every_gated_item_does(self):
        rep = gates.GateReport(
            "B1", "t", [gates.GateItem("a", True, "1"), gates.GateItem("b", None, "2")]
        )
        assert rep.passed
        rep.items.append(gates.GateItem("c", False, "3"))
        assert not rep.passed


class TestCli:
    def test_check_geometry_writes_the_report(self, root, tmp_path, capsys):
        code = cli.main(
            ["--dataset-root", str(root), "--workspace", str(tmp_path / "ws"), "check-geometry"]
        )
        assert code == 1  # the flagged-lap share of this tiny dataset
        assert (tmp_path / "ws" / "reports" / "B1_geometry.md").exists()
        assert (tmp_path / "ws" / "cache" / "lap_index.parquet").exists()
        assert "# B1: " in capsys.readouterr().out
