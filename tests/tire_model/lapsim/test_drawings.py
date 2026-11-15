"""The operators' published geometry and the check against it (gate B2)."""

from __future__ import annotations

import math

import pytest

from motorsports_data_notebook.tire_model.lapsim import drawings as dr
from motorsports_data_notebook.tire_model.lapsim import gates
from motorsports_data_notebook.tire_model.lapsim import geometry as geo
from motorsports_data_notebook.tire_model.lapsim import track as trk

from ._synthetic import LAYOUT_ID, LENGTH_M, RADIUS_M, STRAIGHT_M, write_dataset

ARC_M = math.pi * RADIUS_M


def _stadium_drawing(direction: str = "R", width_m: float = 10.0) -> dict:
    return {
        "source": "synthetic",
        "length_m": LENGTH_M,
        "width_m": (width_m, width_m),
        "elements": dr._elements(
            [
                (direction, RADIUS_M, ARC_M, STRAIGHT_M / 2, None),
                (direction, RADIUS_M, ARC_M, 1.5 * STRAIGHT_M + ARC_M, None),
            ]
        ),
    }


class TestTranscription:
    def test_sodegaura_arcs_close_the_lap(self):
        els = dr.DRAWINGS["sodegaura"]["elements"]
        turn = sum(e["sign"] * math.degrees(e["arc_m"] / e["radius_m"]) for e in els)
        assert turn == pytest.approx(360.0, abs=0.1)
        # stations are consistent with the element lengths
        assert els[1]["station_m"] - (els[0]["station_m"] + els[0]["arc_m"]) == pytest.approx(156.4)
        assert els[9]["radius_m"] == 55  # printed R=55; 65 would break the closure by 25°

    @pytest.mark.parametrize(
        ("layout", "n_elements", "n_groups"),
        [
            ("sodegaura", 14, 5),
            ("tsukuba_2000", 12, 7),
            ("suzuka_full", 32, 13),
            ("motegi_full", 13, 9),
        ],
    )
    def test_elements_and_groups(self, layout, n_elements, n_groups):
        els = dr.DRAWINGS[layout]["elements"]
        assert len(els) == n_elements
        assert len(dr.turn_sequence([e["sign"] for e in els])) == n_groups
        assert len(dr.groups(els)) == n_groups
        assert dr.DRAWINGS[layout]["source"].startswith("https://")

    def test_suzuka_is_the_only_disclaimed_one(self):
        flagged = [k for k, v in dr.DRAWINGS.items() if v.get("dimensions_disclaimed")]
        assert flagged == ["suzuka_full"]


class TestTools:
    def test_turn_sequence_and_groups(self):
        assert dr.turn_sequence([1, 1, -1, 1, 1, 1, -1, -1]) == [1, -1, 1, -1]
        assert dr.turn_sequence([]) == []
        grouped = dr.groups(dr._elements([("R", 40), ("R", 95), ("L", 60), ("R", 25)]))
        assert [len(g) for g in grouped] == [2, 1, 1]

    def test_straight_to_straight_cuts_at_long_straights(self):
        pieces = dr.straight_to_straight(dr.DRAWINGS["sodegaura"])
        assert len(pieces) == 4
        assert sum(p["turn_deg"] for p in pieces) == pytest.approx(360.0, abs=0.1)
        assert pieces[0]["turn_deg"] == pytest.approx(57.1, abs=0.1)  # the 95R alone
        # the last piece runs through the start line and holds the first corner
        assert pieces[-1]["station_to_m"] > dr.DRAWINGS["sodegaura"]["length_m"]
        assert pieces[-1]["turn_deg"] == pytest.approx(93.1, abs=0.1)

    def test_no_pieces_without_arc_lengths(self):
        assert dr.straight_to_straight(dr.DRAWINGS["tsukuba_2000"]) == []

    def test_inside_edge_radius(self):
        group = dr._elements([("R", 55, None, None, 15.05), ("R", 35, None, None, 10.10)])
        assert dr.inside_edge_radius(group, 12.0) == pytest.approx(35 - 5.05)
        assert dr.inside_edge_radius(dr._elements([("R", 30)]), 15.0) == pytest.approx(22.5)
        assert dr.inside_edge_radius(dr._elements([("L", None)]), 15.0) is None


class TestAgainstDrawing:
    @pytest.fixture()
    def root(self, tmp_path):
        geo._INDEX_CACHE.clear()
        write_dataset(
            tmp_path,
            {
                f"2026-01-1{i}_s{i}": {
                    "car": "car",
                    "date": f"2026-01-1{i}",
                    "kwargs": {"n_laps": 5, "seed": i, "noise_m": 0.02},
                }
                for i in range(3)
            },
        )
        return tmp_path

    def test_track_agrees_with_its_drawing(self, root):
        model = trk.build_track_model(root, LAYOUT_ID, inertial=False, n_boot=0).model
        row, text = gates.against_drawing(model, _stadium_drawing())
        assert row["sequence_matches"] and row["groups_drawing"] == 1
        assert row["turn_pieces"] == 2
        assert abs(row["turn_worst_deg"]) < 2.0
        # driven on the 50 m centre line of a 10 m wide track: 50 / 45
        assert row["apex_vs_inside_edge_min"] == pytest.approx(RADIUS_M / 45.0, rel=0.05)
        assert "Net turning between long straights" in text

    def test_gate_passes_and_fails_on_the_drawing(self, root):
        good = gates.check_track_truth(
            root, layouts=(LAYOUT_ID,), drawings={LAYOUT_ID: _stadium_drawing()}
        )
        item = {i.name: i for i in good.items}["Against the operators' drawings"]
        assert item.passed is True
        # the same circuit drawn turning left, and one drawn with 80 m
        # corners: the driven 50 m apex would be inside its inside edge
        wide = _stadium_drawing()
        for el in wide["elements"]:
            el["radius_m"], el["arc_m"] = 80.0, math.pi * 80.0
        for wrong in (_stadium_drawing("L"), wide):
            bad = gates.check_track_truth(root, layouts=(LAYOUT_ID,), drawings={LAYOUT_ID: wrong})
            assert {i.name: i for i in bad.items}["Against the operators' drawings"].passed is False

    def test_without_drawings_the_check_is_not_measured(self, root):
        report = gates.check_track_truth(root, layouts=(LAYOUT_ID,))
        item = {i.name: i for i in report.items}["Against the operators' drawings"]
        assert item.passed is None
