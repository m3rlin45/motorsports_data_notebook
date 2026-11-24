"""Tests for the wet readings (#206 D2)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import limits as lim
from motorsports_data_notebook.tire_model.lapsim import solver as sv
from motorsports_data_notebook.tire_model.lapsim import wet

from .test_car_gate import CAR, _model, _oval_track


def _passes(rng, corner, basis, n, level, date="d", layout="a", sd=0.03):
    return pd.DataFrame(
        {
            "layout_id": layout,
            "corner": corner,
            "date": date,
            "basis": basis,
            "value": level * np.exp(rng.normal(0.0, sd, n)),
        }
    )


class TestBestRatio:
    def test_a_planted_ratio_comes_back(self):
        rng = np.random.default_rng(3)
        frames = []
        for corner in range(6):
            frames.append(_passes(rng, corner, "dry", 400, 1.4))
            frames.append(_passes(rng, corner, "wet", 40, 1.4 * 0.85))
        ratio, by_corner = wet.best_ratio(pd.concat(frames))
        assert ratio == pytest.approx(0.85, abs=0.015)
        assert len(by_corner) == 6 and (by_corner["m"] == 20).all()
        # the same statistic on both sides: m of the dry passes, not their overall best
        assert (by_corner["dry"] < 1.4 * np.exp(3 * 0.03)).all()

    def test_m_is_half_the_wet_passes_and_a_thin_corner_is_left_out(self):
        rng = np.random.default_rng(4)
        frames = [
            _passes(rng, 0, "dry", 100, 1.2),
            _passes(rng, 0, "wet", 9, 1.0),
            _passes(rng, 1, "dry", 100, 1.2),
            _passes(rng, 1, "wet", 5, 1.0),  # m would be 2
            _passes(rng, 2, "dry", 3, 1.2),  # fewer dry passes than m
            _passes(rng, 2, "wet", 12, 1.0),
        ]
        ratio, by_corner = wet.best_ratio(pd.concat(frames))
        assert list(by_corner["corner"]) == [0] and int(by_corner["m"].iloc[0]) == 4
        assert ratio == pytest.approx(1.0 / 1.2, abs=0.03)
        empty, none = wet.best_ratio(pd.concat(frames[2:4]))
        assert np.isnan(empty) and none.empty

    def test_each_wet_days_own_level(self):
        rng = np.random.default_rng(5)
        frames = []
        for corner in range(4):
            frames.append(_passes(rng, corner, "dry", 300, 1.4))
            frames.append(_passes(rng, corner, "wet", 16, 1.4 * 0.90, date="damp"))
            frames.append(_passes(rng, corner, "wet", 16, 1.4 * 0.75, date="soaked"))
        days = wet.day_levels(pd.concat(frames)).set_index("date")
        assert days.loc["damp", "ratio"] == pytest.approx(0.90, abs=0.02)
        assert days.loc["soaked", "ratio"] == pytest.approx(0.75, abs=0.02)
        assert (days["corners"] == 4).all() and (days["passes"] == 64).all()
        # pooled, the best moments are the driest day's
        pooled, _ = wet.best_ratio(pd.concat(frames))
        assert pooled > 0.86


def _tables(rng, corner_ratio=0.85, brake_ratio=0.95, wet_days=("w1", "w2")):
    apex, brake = [], []
    for basis, days, n, c_scale, b_scale in (
        ("dry", ("d1", "d2", "d3"), 60, 1.0, 1.0),
        ("wet", wet_days, 12, corner_ratio, brake_ratio),
    ):
        for date in days:
            for corner in range(4):
                peak = 1.4 * c_scale * np.exp(rng.normal(0.0, 0.02, n))
                apex.append(
                    pd.DataFrame(
                        {
                            "car": "car",
                            "layout_id": "a",
                            "corner": corner,
                            "basis": basis,
                            "date": date,
                            "session_id": f"s_{date}",
                            "lap_num": np.arange(n),
                            "v_apex_ms": 30.0,
                            "s_apex_m": 100.0 * corner,
                            "sign": 1,
                            "lat_doppler_g": peak,
                            "lat_tyre_g": peak,
                            "lat_track_g": peak,
                            "lat_accel_g": peak,
                            "load": 1.0,
                        }
                    )
                )
            for place in range(3):
                brake.append(
                    pd.DataFrame(
                        {
                            "layout_id": "a",
                            "basis": basis,
                            "date": date,
                            "session_id": f"s_{date}",
                            "s_on_m": 500.0 * place + rng.uniform(0.0, 20.0, n),
                            "held_s": 1.0,
                            "lat_held_tyre_g": 0.05,
                            "v_held_ms": 40.0,
                            "decel_held_g": 1.1 * b_scale * np.exp(rng.normal(0.0, 0.02, n)),
                        }
                    )
                )
    return pd.concat(apex, ignore_index=True), pd.concat(brake, ignore_index=True)


class TestWetTerms:
    def test_braking_falls_less_than_cornering(self):
        apex, brake = _tables(np.random.default_rng(6))
        got = wet.wet_terms(apex, brake, c_drag=0.0, lat_level=1.4)
        assert got.cornering == pytest.approx(0.85, abs=0.02)
        assert got.braking == pytest.approx(0.95, abs=0.02)
        assert got.brake_term == pytest.approx(0.95 / 0.85, abs=0.03)
        told = got.to_dict()
        assert told["corners"] == 4 and told["places"] == 3 and told["wet_passes"] == 96
        assert set(got.days["date"]) == {"w1", "w2"}

    def test_drag_is_taken_out_of_a_stop(self):
        apex, brake = _tables(np.random.default_rng(6))
        stops = wet.braking_stops(brake, c_drag=3.0e-4, lat_level=1.4)
        plain = wet.braking_stops(brake, c_drag=0.0, lat_level=1.4)
        assert (plain["value"].mean() - stops["value"].mean()) == pytest.approx(
            3.0e-4 * 40.0**2 / 9.80665, rel=1e-6
        )
        # a stop under cornering is not a straight-line stop
        assert wet.braking_stops(brake.assign(lat_held_tyre_g=0.6), 0.0, 1.4).empty

    def test_a_fold_reads_without_its_wet_day(self):
        rng = np.random.default_rng(7)
        apex, brake = _tables(rng, wet_days=("w1",))
        none = wet.wet_terms(apex, brake, 0.0, 1.4, without="w1")
        assert np.isnan(none.cornering) and np.isnan(none.brake_term)
        assert none.to_dict()["brake_term"] is None
        both = wet.wet_terms(*_tables(rng), 0.0, 1.4, without="w1")
        assert set(both.days["date"]) == {"w2"}
        assert wet.wet_terms(pd.DataFrame(), pd.DataFrame(), 0.0, 1.4).to_dict()["corners"] == 0

    def test_statistic_is_limits_expected_best(self):
        x = np.random.default_rng(8).normal(1.0, 0.1, 30)
        frame = pd.concat(
            [
                pd.DataFrame(
                    {"layout_id": "a", "corner": 0, "date": "d", "basis": "wet", "value": x}
                ),
                pd.DataFrame(
                    {"layout_id": "a", "corner": 0, "date": "d", "basis": "dry", "value": x}
                ),
            ]
        )
        ratio, by_corner = wet.best_ratio(frame)
        assert ratio == pytest.approx(1.0)
        assert by_corner["wet"].iloc[0] == pytest.approx(lim.expected_best(x, 15))


class TestSolver:
    def test_a_wet_scale_alone_changes_no_lap_solved_to_its_lap_time(self):
        track = _oval_track()
        target = sv.lap_time(track, CAR, 0.80)
        scaled = replace(CAR, wet_scale=0.85)
        dry = sv.solve_laps(track, CAR, [target])
        wet_lap = sv.solve_laps(track, scaled, [target], wet=True)
        assert wet_lap.u[0] == pytest.approx(dry.u[0] / 0.85, rel=1e-6)
        np.testing.assert_allclose(wet_lap.v_node_ms, dry.v_node_ms, atol=2e-4)

    def test_the_wet_braking_term_brakes_later_and_only_in_the_wet(self):
        track = _oval_track()
        car = replace(CAR, wet_scale=0.85, wet_brake=1.10)
        np.testing.assert_array_equal(
            sv.drive_laps(track, car, [0.9]).v_node_ms, sv.drive_laps(track, CAR, [0.9]).v_node_ms
        )
        target = sv.lap_time(track, CAR, 0.80)
        plain = sv.solve_laps(track, replace(car, wet_brake=1.0), [target], wet=True)
        termed = sv.solve_laps(track, car, [target], wet=True)
        assert termed.ok[0] and termed.u[0] < plain.u[0]  # less cornering for the same lap time
        # harder braking: the highest deceleration of the lap rises
        ds = track.ds_m

        def peak(v):
            return float(np.max(-(np.roll(v, -1) ** 2 - v**2) / (2 * ds * sv.G_MS2)))

        assert peak(termed.v_node_ms[0]) > peak(plain.v_node_ms[0]) * 1.03
        one = sv.simulate_lap(track, car, float(termed.u[0]), wet=True)
        np.testing.assert_allclose(one.v_node_ms, termed.v_node_ms[0], atol=2e-4)

    def test_from_a_model_it_is_off_unless_asked_for(self):
        model = _model()
        model.wet = {"grip_scale": 0.85, "brake_term": 1.08}
        assert sv.SolverCar.from_model(model).wet_brake == 1.0
        assert sv.SolverCar.from_model(model, wet_brake=True).wet_brake == 1.08
        model.wet = {"grip_scale": 0.85}
        assert sv.SolverCar.from_model(model, wet_brake=True).wet_brake == 1.0
