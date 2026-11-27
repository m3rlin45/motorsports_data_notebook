"""A simulated entry as the calculators are served it (lapsim/served.py)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from motorsports_data_notebook.tire_model.lapsim import pressure as pr
from motorsports_data_notebook.tire_model.lapsim import served as srv
from motorsports_data_notebook.tire_model.predict import (
    _lookup_g2,
    _lookup_lap_time,
    _lookup_outlap,
)

from .test_pressure import TYRES, _fold_model

HEAT = "g2_typ_by_track_car_cond"
LAP = "lap_time_typ_by_track_car_cond"
OUT = "outlap_typ_by_track_car_cond"
RAW = {t: t.replace("_car_", "_raw_car_") for t in (HEAT, LAP, OUT)}


def _entry() -> pr.Entry:
    return pr.Entry(
        lap_time_s=58.0,
        q=np.array([3.0, 3.5, 4.0, 4.5]),
        curve_t=np.array([50.0, 58.0, 70.0]),
        curve_q=np.array([[4.0] * 4, [3.0, 3.5, 4.0, 4.5], [2.0] * 4]),
        out_s=95.0,
        out_q=np.array([1.0] * 4),
        source="sim",
    )


def _pooled_model() -> dict:
    """The fold model with raw cars "r" and "s" pooled as "c"; "s" has its
    own logged rows at track "a"."""
    model = _fold_model()
    model["car_aliases"] = {"r": "c", "s": "c"}
    for t in (HEAT, LAP, OUT):
        model[RAW[t]] = [
            {**copy.deepcopy(r), "car": "s"} for r in model[t] if r["track_canonical"] == "a"
        ]
    return model


def _cell(model: dict, table: str, track: str, car: str) -> list[dict]:
    return [r for r in model.get(table) or [] if r["track_canonical"] == track and r["car"] == car]


def _sims(model: dict, table: str, track: str, car: str) -> list[dict]:
    return [r for r in _cell(model, table, track, car) if r.get("source") == "sim"]


class TestServe:
    def test_a_cell_with_a_logged_out_lap_keeps_it_under_either_rules(self) -> None:
        model = _fold_model()
        for rules in (srv.COMMITTEE, srv.SHIPPING):
            got = srv.serve(model, "a", "c", "dry", _entry(), rules)
            assert _cell(got, OUT, "a", "c") == _cell(model, OUT, "a", "c")
            (heat,) = _sims(got, HEAT, "a", "c")
            assert heat["q_typ_by_corner"] == dict(zip(TYRES, [3.0, 3.5, 4.0, 4.5]))
            assert heat["g2_vs_lap_time"]["method"] == "sim"
            (lap,) = _sims(got, LAP, "a", "c")
            assert lap["lap_time_typ_s"] == 58.0
            assert _lookup_g2(got, "a", "c", "dry", "fl")[0] == 3.0
            assert _lookup_lap_time(got, "a", "c", "dry")[0] == 58.0
        # what ships adds beside the logged rows; the committee's replaces them
        shipping = srv.serve(model, "a", "c", "dry", _entry(), srv.SHIPPING)
        logged = [r for r in _cell(shipping, HEAT, "a", "c") if "source" not in r]
        assert logged == _cell(model, HEAT, "a", "c")
        committee = srv.serve(model, "a", "c", "dry", _entry(), srv.COMMITTEE)
        assert [r.get("source") for r in _cell(committee, HEAT, "a", "c")] == ["sim"]

    def test_without_one_the_committee_reads_the_chain_and_shipping_the_simulation(
        self,
    ) -> None:
        model = _fold_model()
        assert _cell(model, OUT, "b", "c") == []
        committee = srv.serve(model, "b", "c", "dry", _entry(), srv.COMMITTEE)
        assert _cell(committee, OUT, "b", "c") == []  # the fallback chain answers
        shipping = srv.serve(model, "b", "c", "dry", _entry(), srv.SHIPPING)
        (row,) = _cell(shipping, OUT, "b", "c")
        assert row["outlap_moving_s"] == 95.0 and row["source"] == "sim"

    def test_only_the_cell_changes(self) -> None:
        model = _fold_model()
        got = srv.serve(model, "b", "c", "dry", _entry(), srv.SHIPPING)
        for t in (HEAT, LAP, OUT):
            assert _cell(got, t, "a", "c") == _cell(model, t, "a", "c")
        assert {k: v for k, v in got.items() if not k.endswith("_cond")} == {
            k: v for k, v in model.items() if not k.endswith("_cond")
        }

    def test_support_goes_on_the_curve(self) -> None:
        sup = {"lap_time_min_s": 52.0, "lap_time_max_s": 66.0}
        got = srv.serve(_fold_model(), "b", "c", "dry", _entry(), srv.SHIPPING, support=sup)
        (heat,) = _sims(got, HEAT, "b", "c")
        assert heat["g2_vs_lap_time"]["support"] == sup

    def test_serving_again_replaces_the_simulated_row_only(self) -> None:
        model = _fold_model()
        once = srv.serve(model, "a", "c", "dry", _entry(), srv.SHIPPING)
        twice = srv.serve(once, "a", "c", "dry", _entry(), srv.SHIPPING)
        assert len(_sims(twice, HEAT, "a", "c")) == 1
        assert len(_cell(twice, HEAT, "a", "c")) == 2  # the logged row beside it


class TestRawRows:
    def test_shipping_writes_the_raw_cars_own_rows_and_leaves_the_pooled_one(self) -> None:
        model = _pooled_model()
        got = srv.serve(model, "a", "r", "dry", _entry(), srv.SHIPPING)
        for t in (HEAT, LAP, OUT):
            assert _cell(got, t, "a", "c") == _cell(model, t, "a", "c")  # the pooled FJ row
            assert _cell(got, RAW[t], "a", "s") == _cell(model, RAW[t], "a", "s")  # another raw car
        assert [r["source"] for r in _cell(got, RAW[HEAT], "a", "r")] == ["sim"]
        assert [r["source"] for r in _cell(got, RAW[LAP], "a", "r")] == ["sim"]
        assert _cell(got, RAW[OUT], "a", "r") == []  # the pooled car's logged out-lap answers
        # read by the calculators' chain: the raw car's own simulated heat, the logged out-lap
        assert _lookup_g2(got, "a", "r", "dry", "fl")[0] == 3.0
        assert _lookup_lap_time(got, "a", "r", "dry")[0] == 58.0
        hit = _lookup_outlap(got, "a", "r", "dry", "fl")
        assert hit is not None and hit[0] == 80.0
        assert _lookup_g2(got, "a", "c", "dry", "fl")[0] == 4.0  # a saved FJ: logged

    def test_the_committee_places_the_pooled_row_and_clears_the_raw_ones(self) -> None:
        model = _pooled_model()
        got = srv.serve(model, "a", "s", "dry", _entry(), srv.COMMITTEE)
        assert _cell(got, RAW[HEAT], "a", "s") == []
        assert _lookup_g2(got, "a", "s", "dry", "fl")[0] == 3.0
        # the model's own out-lap chain: the raw car's logged row
        assert _cell(got, RAW[OUT], "a", "s") == _cell(model, RAW[OUT], "a", "s")

    def test_a_raw_cars_logged_out_lap_counts_as_the_cells(self) -> None:
        model = _pooled_model()
        model[OUT] = []  # no pooled out-lap at all
        assert srv.logged_outlap(model, "a", "s", "dry")
        assert not srv.logged_outlap(model, "a", "r", "dry")
        got = srv.serve(model, "a", "r", "dry", _entry(), srv.SHIPPING)
        (row,) = _cell(got, RAW[OUT], "a", "r")
        assert row["source"] == "sim"


class TestWetOutLap:
    def test_a_wet_cell_reads_the_logged_dry_out_lap_down_the_chain(self) -> None:
        model = _fold_model()  # a logged dry out-lap at "a", none in the wet
        assert srv.logged_outlap(model, "a", "c", "wet")
        got = srv.serve(model, "a", "c", "wet", _entry(), srv.SHIPPING)
        assert _cell(got, OUT, "a", "c") == _cell(model, OUT, "a", "c")
        hit = _lookup_outlap(got, "a", "c", "wet", "fl")
        assert hit is not None and hit[0] == 80.0

    def test_a_simulated_row_is_not_a_logged_out_lap(self) -> None:
        model = srv.serve(_fold_model(), "b", "c", "dry", _entry(), srv.SHIPPING)
        assert not srv.logged_outlap(model, "b", "c", "wet")


class TestWetLapTime:
    def test_the_wet_keeps_the_models_lap_time_row(self) -> None:
        model = _fold_model()
        model[LAP].append(
            {
                "track_canonical": "a",
                "car": "c",
                "condition": "wet",
                "lap_time_typ_s": 66.0,
                "n_laps_used": 9,
            }
        )
        got = srv.serve(model, "a", "c", "wet", _entry(), srv.SHIPPING, keep_lap_row=True)
        assert _cell(got, LAP, "a", "c") == _cell(model, LAP, "a", "c")
        assert _lookup_lap_time(got, "a", "c", "wet")[0] == 66.0
        assert [r["source"] for r in _cell(got, HEAT, "a", "c") if r["condition"] == "wet"] == [
            "sim"
        ]


class TestScored:
    """The holdout's scorer reads what is served: a raw car's own row."""

    def test_a_raw_cars_session_is_scored_on_its_served_row(self) -> None:
        from motorsports_data_notebook.tire_model.validate import _score_fold_laps
        from tests.tire_model.test_validate import _harness_laps, _harness_model

        prep = _harness_laps()
        prep.flying["raw_car"] = "r"
        model = _harness_model()
        model["car_aliases"] = {"r": "c"}
        entry = pr.Entry(lap_time_s=60.0, q=np.array([8.0] * 4), source="sim")
        calc = _score_fold_laps(prep, model)
        shipping = _score_fold_laps(prep, srv.serve(model, "t", "r", "dry", entry, srv.SHIPPING))
        committee = _score_fold_laps(prep, srv.serve(model, "t", "r", "dry", entry, srv.COMMITTEE))
        assert (calc["g2_used"] == 4.0).all()  # the logged entry
        assert (shipping["g2_used"] == 8.0).all()  # the raw car's own simulated row
        pd.testing.assert_frame_equal(shipping, committee)  # the same heat, either way


class TestBesideTheLoggedRows:
    """Schema §9: a simulated row answers only its own exact condition."""

    def test_a_wet_cell_falls_back_to_the_logged_dry_row_not_the_simulated_one(self) -> None:
        model = _pooled_model()
        before = (
            _lookup_g2(model, "a", "s", "wet", "fl"),
            _lookup_lap_time(model, "a", "s", "wet"),
        )
        got = srv.serve(model, "a", "s", "dry", _entry(), srv.SHIPPING)
        assert _lookup_g2(got, "a", "s", "dry", "fl")[0] == 3.0  # the exact step: simulated
        after = (_lookup_g2(got, "a", "s", "wet", "fl"), _lookup_lap_time(got, "a", "s", "wet"))
        assert after == before  # the wet reads exactly what it read

    def test_the_source_decides_not_the_order(self) -> None:
        model = srv.serve(_pooled_model(), "a", "s", "dry", _entry(), srv.SHIPPING)
        for t in (RAW[HEAT],):
            model[t] = list(reversed(model[t]))
        assert _lookup_g2(model, "a", "s", "dry", "fl")[0] == 3.0
        model[RAW[HEAT]] = sorted(model[RAW[HEAT]], key=lambda r: r.get("source") == "sim")
        assert _lookup_g2(model, "a", "s", "dry", "fl")[0] == 3.0

    def test_the_raw_cars_sim_row_before_its_logged_one_before_the_pooled_rows(self) -> None:
        model = _pooled_model()
        model[HEAT] = model[HEAT] + [
            {**model[HEAT][0], "q_typ_by_corner": dict(zip(TYRES, [9.0] * 4)), "source": "sim"}
        ]
        # a pooled sim row does not beat the raw car's own logged row
        assert _lookup_g2(model, "a", "s", "dry", "fl")[0] == 4.0
        # a raw car with no row of its own reads the pooled sim row before the pooled logged one
        assert _lookup_g2(model, "a", "r", "dry", "fl")[0] == 9.0


class TestRuleC:
    def test_no_logged_lap_time_in_the_condition_no_wet_entry(self) -> None:
        from motorsports_data_notebook.tire_model.lapsim.export import TableExport

        model = _fold_model()
        assert srv.logged_lap_time(model, "a", "c", "wet") is None
        assert srv.logged_lap_time(model, "a", "c", "dry") == 60.0
        src = pr.SimSource(export=TableExport.__new__(TableExport), heat=np.ones((2, 4)))
        got = srv.sim_entry(src, np.ones(4), model, "a", "c", "wet", 0.9, srv.SHIPPING)
        assert got == (None, False)

    def test_a_simulated_lap_time_is_not_a_logged_one(self) -> None:
        model = srv.serve(_fold_model(), "b", "c", "dry", _entry(), srv.SHIPPING)
        assert srv.logged_lap_time(model, "b", "c", "dry") is None
