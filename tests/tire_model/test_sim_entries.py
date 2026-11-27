"""Simulated entries in tire_model.json (tire_model/sim_entries.py)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from motorsports_data_notebook.tire_model import sim_entries as se
from motorsports_data_notebook.tire_model.lapsim import pace
from motorsports_data_notebook.tire_model.lapsim import sim_export as sx
from motorsports_data_notebook.tire_model.lapsim.holdout import shares_of
from motorsports_data_notebook.tire_model.lapsim.pressure import SimSource
from motorsports_data_notebook.tire_model.predict import _lookup_g2, _lookup_lap_time
from tests.tire_model.lapsim.test_car_gate import _model, _oval_track
from tests.tire_model.lapsim.test_pace import _ranged
from tests.tire_model.lapsim.test_pressure import _fold_model

TRACK = "tsukuba_2000"  # the oval stands in for its main layout
HEAT = "g2_typ_by_track_car_cond"
RAW_HEAT = "g2_typ_by_track_raw_car_cond"
RAW_LAP = "lap_time_typ_by_track_raw_car_cond"


@pytest.fixture(scope="module")
def exports_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    track = _oval_track()
    track.meta["pit"] = {"s_join_m": 400.0, "v_join_ms": 15.0, "rolling_s": 20.0, "n": 12}
    exp = pace.build_table(_ranged(_model(), oval=(0.80, 0.97)), track).laps_export()
    exp.layout_id = TRACK
    exp.raw_car = "r"
    exp.fed = {"use": 0.9, "from": "test", "sessions": 3}
    out = tmp_path_factory.mktemp("exports")
    sx.write_exports({("r", TRACK): exp}, out, [{"car": "z", "why": "#378: never logged"}], {})
    return out


def _tyres_model() -> dict:
    """A fold-like model in the tyres' frame: pooled car "c" with raw cars
    "r" and "s", logged entries at TRACK (renamed from "a") and "b"."""
    model = copy.deepcopy(_fold_model())
    for name in (HEAT, "lap_time_typ_by_track_car_cond", "outlap_typ_by_track_car_cond"):
        for r in model[name]:
            if r["track_canonical"] == "a":
                r["track_canonical"] = TRACK
    model["car_aliases"] = {"r": "c", "s": "c"}
    # a logged wet lap time at TRACK (the pooled car's); none in the damp
    model["lap_time_typ_by_track_car_cond"].append(
        {
            "track_canonical": TRACK,
            "car": "c",
            "condition": "wet",
            "lap_time_typ_s": 68.0,
            "n_laps_used": 9,
        }
    )
    model["heat_input"] = {
        "frame": "tyres",
        "fitted_by_car": {"c": {"g_transfer": 2.0}},
        "car_facts_by_car": {"c": {"p_f": 0.5, "brake_bias_front": 0.6, "driven": "rear"}},
    }
    return model


class TestPlace:
    def test_the_raw_cars_cells_are_served_and_nothing_else_moves(self, exports_dir) -> None:
        model = _tyres_model()
        got = se.place_simulated(model, exports_dir)
        assert got["simulation"]["placed"] and got["simulation"]["logged_cells_moved"] == []
        raw = [r for r in got[RAW_HEAT] if r["car"] == "r"]
        assert sorted(r["condition"] for r in raw) == ["dry", "wet"]  # no logged damp lap time
        assert all(r["source"] == "sim" and r["g2_vs_lap_time"]["method"] == "sim" for r in raw)
        assert got[HEAT] == model[HEAT]  # the pooled rows stay the logged ones
        # the chain: the raw car reads its own simulated heat, at its fed use's lap
        dry = next(r for r in raw if r["condition"] == "dry")
        lap = _lookup_lap_time(got, TRACK, "r", "dry")[0]
        (exp,), _ = sx.load_exports(exports_dir)
        exp = exp.rounded(sx.DIGITS)  # as the export is committed and served
        assert lap == exp.lap_time_at(0.9)
        assert _lookup_g2(got, TRACK, "r", "dry", "fl")[0] == dry["q_typ_by_corner"]["fl"]
        # the wet: the dry table read at the logged wet lap time, which is kept
        wet = next(r for r in raw if r["condition"] == "wet")
        assert _lookup_lap_time(got, TRACK, "r", "wet")[0] == 68.0
        src = SimSource.of(exp, shares_of(model)["r"])
        want = src.entry(np.ones(4), lap_time_s=68.0).q
        assert [wet["q_typ_by_corner"][c] for c in ("fl", "fr", "rl", "rr")] == list(want)
        assert [r["source"] for r in got[RAW_LAP] if r["car"] == "r"] == ["sim"]  # the dry only
        assert got["simulation"]["attribution"] == sx.ATTRIBUTION

    def test_coverage_says_which_rows_answer(self, exports_dir) -> None:
        got = se.place_simulated(_tyres_model(), exports_dir)
        rows = {(r["track_canonical"], r["car"], r["condition"]): r for r in got["entry_coverage"]}
        sim = rows[(TRACK, "r", "dry")]
        assert (sim["heat"], sim["lap_time"], sim["outlap"]) == ("sim", "sim", "logged")
        assert sim["layout_id"] == TRACK and not sim["sim_only"]
        logged = rows[("b", "r", "dry")]
        assert logged["heat"] == "logged" and "reason" in logged
        other = rows[(TRACK, "s", "dry")]  # a raw car with no export
        assert other["heat"] == "logged"

    def test_only_the_chosen_conditions(self, exports_dir) -> None:
        got = se.place_simulated(_tyres_model(), exports_dir, conditions=("dry",))
        raw = [r for r in got[RAW_HEAT] if r["car"] == "r"]
        assert [r["condition"] for r in raw] == ["dry"]
        wet = next(
            r
            for r in got["entry_coverage"]
            if (r["track_canonical"], r["car"], r["condition"]) == (TRACK, "r", "wet")
        )
        # a simulated row answers only its own condition: the wet reads the logged rows
        assert (wet["heat"], wet["lap_time"]) == ("logged", "logged") and "reason" in wet


class TestNotPlaced:
    def test_a_sensor_frame_model_gets_no_simulated_row(self, exports_dir) -> None:
        model = _tyres_model()
        del model["heat_input"]["frame"]
        got = se.place_simulated(model, exports_dir)
        assert not got["simulation"]["placed"] and "sensor" in got["simulation"]["why"]
        assert "entry_coverage" not in got and RAW_HEAT not in got

    def test_no_export_no_simulated_row(self, tmp_path: Path) -> None:
        got = se.place_simulated(_tyres_model(), tmp_path)
        assert not got["simulation"]["placed"] and RAW_HEAT not in got

    def test_a_main_layout_maps_to_its_track_and_another_layout_to_none(self) -> None:
        assert se.track_of("fuji_gp") == "fuji" and se.track_of("fuji_short") is None
        assert se.track_of("tsukuba_2000") == "tsukuba_2000"
