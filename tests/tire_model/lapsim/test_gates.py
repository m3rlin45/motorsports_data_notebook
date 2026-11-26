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

    def test_a_gate_that_is_a_score_heads_its_report_with_it(self):
        rep = gates.GateReport("C3", "t", [gates.GateItem("a", None, "1")], score="41 of 96")
        text = rep.to_markdown()
        assert "Attainment: 41 of 96" in text and "Overall:" not in text
        assert rep.passed  # nothing in it is a pass or a fail
        assert "Overall: pass" in gates.GateReport("B1", "t").to_markdown()


class TestCli:
    def test_check_geometry_writes_the_report(self, root, tmp_path, capsys):
        code = cli.main(
            ["--dataset-root", str(root), "--workspace", str(tmp_path / "ws"), "check-geometry"]
        )
        assert code == 1  # the flagged-lap share of this tiny dataset
        assert (tmp_path / "ws" / "reports" / "B1_geometry.md").exists()
        assert (tmp_path / "ws" / "cache" / "lap_index.parquet").exists()
        assert "# B1: " in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("model", "cmd", "want"),
        [
            ({"heat_input": {"form": "x"}}, "check-geometry", "sensor"),  # fitted before #376
            ({"heat_input": {"frame": "tyres"}}, "check-geometry", "tyres"),
            ({"heat_input": {"form": "x"}}, "check-pressure-holdout", "tyres"),  # its own flag
        ],
    )
    def test_a_gate_reads_heat_in_the_committed_models_frame(
        self, monkeypatch, tmp_path, model, cmd, want
    ):
        import json

        from motorsports_data_notebook.tire_model import heat_frame

        (tmp_path / "tire_model.json").write_text(json.dumps(model))
        seen = []

        def run(*args):
            seen.append(heat_frame.current())
            return gates.GateReport("B1", "t")

        monkeypatch.setattr(cli, "_run", run)
        monkeypatch.setattr(gates, "write_report", lambda rep, out: out / "r.md")
        argv = ["--dataset-root", str(tmp_path), "--workspace", str(tmp_path / "ws"), cmd]
        assert cli.main(argv) == 0
        assert seen == [want] and heat_frame.current() == heat_frame.TYRES


class TestBankAgainstSensor:
    def test_the_slope_the_roll_and_the_rest(self):
        rng = np.random.default_rng(5)
        bank = rng.normal(0.0, 0.03, 600)
        lat_g = rng.normal(0.0, 0.8, 600) + 8.0 * bank  # banked where the car corners
        proxy = 0.9 * bank - 0.02 * lat_g + 0.013 + rng.normal(0.0, 0.004, 600)
        got = gates.bank_against_sensor(bank, proxy, lat_g * 9.80665)
        assert got["nodes"] == 600
        assert got["slope"] == pytest.approx(0.9, abs=0.03)
        assert got["per_g"] == pytest.approx(-0.02, abs=0.002)
        assert got["partial_r"] > 0.9
        assert got["rms_g_fitted"] == pytest.approx(0.004, abs=0.0005) and got["rms_g"] > 0.01

    def test_too_few_nodes_is_no_reading(self):
        few = np.zeros(gates.BANK_SENSOR_MIN_NODES - 1)
        assert gates.bank_against_sensor(few, few, few) is None


class TestCheckTrackTruth:
    @pytest.fixture()
    def inertial_root(self, tmp_path):
        geo._INDEX_CACHE.clear()
        sessions = {
            f"2026-01-1{i}_s{i}": {
                "car": "car",
                "date": f"2026-01-1{i}",
                "kwargs": {"n_laps": 12, "seed": i, "noise_m": 0.02, "accel_delay_s": 0.05},
            }
            for i in range(3)
        }
        write_dataset(tmp_path / "data", sessions)
        return tmp_path / "data"

    def test_gate_on_a_sound_track(self, inertial_root, tmp_path):
        report = gates.check_track_truth(
            inertial_root, tracks_dir=tmp_path / "tracks", layouts=(LAYOUT_ID,)
        )
        items = {i.name: i for i in report.items}
        assert items["Profile closes"].passed is True
        assert items["Re-referencing converged"].passed is True
        assert items["Against the inertial profile"].passed is True
        assert items["Apexes not flattened"].passed is True
        assert items["Apex κ against the accelerometer"].passed is True
        assert items["Fills"].measured.startswith("0 stretches")
        assert items["Grade between cars"].passed is None  # one car
        assert report.passed
        assert (tmp_path / "tracks" / f"{LAYOUT_ID}.json").exists()
        assert gates.write_report(report, tmp_path / "reports").name == "B2_track_truth.md"

    def test_corner_apexes_and_lap_curvature(self):
        ds = 2.0
        s = np.arange(0.0, 402.0, ds)
        course = np.vstack([0.02 * np.clip(s - 100.0, 0.0, 100.0), np.full(s.size, np.nan)])
        k = gates.lap_curvature(course, ds, 10.0)
        assert k.shape == (2, s.size - 1)
        assert k[0, 75] == pytest.approx(0.02) and k[0, 20] == pytest.approx(0.0)
        assert np.isnan(k[0, :3]).all() and np.isnan(k[1]).all()
        corners = [{"s_in": 100.0, "s_out": 200.0, "sign": 1}]
        apex = gates.corner_apexes(k, corners, ds)
        assert apex[0, 0] == pytest.approx(0.02) and np.isnan(apex[1, 0])
        assert gates.corner_apexes(k[0], corners, ds)[0] == pytest.approx(0.02)
