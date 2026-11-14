"""The track profile (lap simulation B2)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import geometry as geo
from motorsports_data_notebook.tire_model.lapsim import track as trk

from ._synthetic import LAYOUT_ID, LENGTH_M, RADIUS_M, STRAIGHT_M, stadium, write_dataset

ARC_M = math.pi * RADIUS_M


def _dataset(tmp_path, **kwargs):
    geo._INDEX_CACHE.clear()
    sessions = {
        f"2026-01-1{i}_s{i}": {
            "car": "car",
            "date": f"2026-01-1{i}",
            "kwargs": {"n_laps": 4, "seed": i, "noise_m": 0.02, **kwargs},
        }
        for i in range(3)
    }
    write_dataset(tmp_path, sessions)
    return tmp_path


class TestLineTools:
    def test_resample_line_is_evenly_spaced(self):
        x = np.array([0.0, 1.0, 1.5, 10.0])
        xr, yr = trk.resample_line(x, np.zeros(4), 2.0)
        assert len(xr) == 6
        assert np.diff(xr) == pytest.approx(2.0)
        assert xr[0] == 0.0 and xr[-1] == 10.0

    def test_projection_distance_and_side(self):
        # a line heading north; a point 3 m east of it is 3 m to the right
        yr = np.arange(0.0, 101.0, 2.0)
        xr = np.zeros_like(yr)
        s, off = trk.project_onto(
            xr, yr, np.array([3.0, -2.0]), np.array([41.0, 60.5]), np.array([40.0, 60.0])
        )
        assert s == pytest.approx([41.0, 60.5])
        assert off == pytest.approx([3.0, -2.0])

    def test_projection_stays_within_the_window(self):
        # out along y = 0 and back along y = 4: a point nearer the return
        # leg is still matched to the outbound leg its guess points at
        out = np.column_stack([np.arange(0.0, 400.0, 2.0), np.zeros(200)])
        back = np.column_stack([np.arange(398.0, -2.0, -2.0), np.full(200, 4.0)])
        line = np.vstack([out, [[400.0, 2.0]], back])
        s, off = trk.project_onto(
            line[:, 0], line[:, 1], np.array([100.0]), np.array([3.0]), np.array([100.0])
        )
        assert s[0] == pytest.approx(100.0, abs=0.01)
        assert off[0] == pytest.approx(-3.0, abs=0.01)  # left of an eastbound line

    def test_closed_line_continues_past_its_ends(self):
        s_true = np.linspace(0.0, LENGTH_M, 358)
        x, y, _ = stadium(s_true)
        x[-1], y[-1] = x[0], y[0]
        px, py, _ = stadium(np.array([-3.0, LENGTH_M + 4.0]))
        s, _ = trk.project_onto(x, y, px, py, np.array([0.0, LENGTH_M]), closed=True)
        assert s == pytest.approx([-3.0, LENGTH_M + 4.0], abs=0.05)
        s_open, _ = trk.project_onto(x, y, px, py, np.array([0.0, LENGTH_M]))
        assert s_open[0] == pytest.approx(0.0, abs=1e-6)

    def test_bins_a_lap_does_not_reach_come_from_its_other_end(self):
        grid = np.arange(0.0, 101.0, 2.0)
        s = np.linspace(3.0, 101.0, 200)  # starts 3 m late, ends 1 m past the line
        out = trk._bin_series(s, s + 10.0, grid, lap_m=100.0, wrap_value=7.0)
        assert out[2:] == pytest.approx(grid[2:] + 10.0)
        # s = 0 is s = 100 one lap on, less what the value gains over a lap
        assert out[0] == pytest.approx(110.0 - 7.0)
        assert np.isnan(out[1])  # 2 m: neither end of the lap covers it
        assert np.isnan(trk._bin_series(s, s, grid)[0])


class TestProfileTools:
    def test_curvature_of_a_circle(self):
        n, r = 200, 80.0
        ds = 2 * math.pi * r / n
        psi = np.arange(n + 1) * ds / r
        assert trk.curvature(psi, ds, 1, 10.0) == pytest.approx(1 / r, rel=1e-9)

    def test_circle_closes(self):
        n, r = 400, 60.0
        ds = 2 * math.pi * r / n
        psi = np.arange(n + 1) * ds / r
        x, y, gap = trk.integrate_line(psi, ds)
        assert gap < 0.01
        assert np.hypot(x - x.mean(), y - y.mean()) == pytest.approx(r, rel=1e-3)

    def test_close_course_spreads_the_miss(self):
        psi = np.linspace(0.0, 2 * math.pi + 0.02, 101)
        closed, miss = trk.close_course(psi, 1)
        assert miss == pytest.approx(0.02 / (2 * math.pi))
        assert closed[-1] - closed[0] == pytest.approx(2 * math.pi)

    def test_median_course_ignores_whole_turns_and_outliers(self):
        s = np.linspace(0.0, 1.0, 201)
        truth = 2 * math.pi * s + 0.3 * np.sin(2 * math.pi * s)
        rng = np.random.default_rng(1)
        laps = np.vstack(
            [truth + 2 * math.pi * k + rng.normal(0, 0.002, s.size) for k in (0, 1, -2, 3, 0, 5, 1)]
        )
        laps[3, 80:120] += 0.4  # one lap goes wide
        laps[4, :10] = np.nan
        psi, turns = trk.median_course(laps)
        assert turns == 1
        assert psi - psi[0] == pytest.approx(truth - truth[0], abs=0.01)

    def test_corner_table_of_a_stadium(self):
        n = 357
        ds = LENGTH_M / n
        s = np.arange(n) * ds
        in_arc = ((s >= STRAIGHT_M / 2) & (s < STRAIGHT_M / 2 + ARC_M)) | (
            (s >= 1.5 * STRAIGHT_M + ARC_M) & (s < 1.5 * STRAIGHT_M + 2 * ARC_M)
        )
        corners = trk.corner_table(np.where(in_arc, 1 / RADIUS_M, 0.0), ds)
        assert len(corners) == 2
        for c in corners:
            assert c["sign"] == 1
            assert c["turn_angle"] == pytest.approx(180.0, abs=3.0)
            assert c["kappa_apex"] == pytest.approx(1 / RADIUS_M)
        assert corners[0]["s_in"] == pytest.approx(STRAIGHT_M / 2, abs=ds)
        assert corners[0]["s_out"] == pytest.approx(STRAIGHT_M / 2 + ARC_M, abs=ds)

    def test_corner_across_the_start_line(self):
        kappa = np.zeros(200)
        kappa[190:] = kappa[:12] = -0.02
        (corner,) = trk.corner_table(kappa, 2.0)
        assert corner["sign"] == -1
        assert corner["s_in"] == 380.0 and corner["s_out"] == 24.0
        assert corner["turn_angle"] == pytest.approx(-math.degrees(0.02 * 22 * 2.0))

    def test_gentle_or_short_bends_are_not_corners(self):
        kappa = np.zeros(300)
        kappa[50:150] = 0.003  # never reaches the apex threshold
        kappa[200:204] = 0.02  # 9 degrees
        assert trk.corner_table(kappa, 2.0) == []


class TestFill:
    def _profile(self):
        n, ds = 600, 2.0
        s = np.arange(n) * ds
        truth = 0.01 * np.exp(-(((s - 400.0) / 60.0) ** 2))
        return truth, np.full(n, 40.0), ds, s

    @staticmethod
    def _excursion(s: np.ndarray, size: float) -> np.ndarray:
        # the course leaves the truth and comes back: no net turning
        u = (s - 800.0) / 15.0
        return size * u * np.exp(-u * u)

    def test_fixed_place_error_is_replaced(self):
        truth, v, ds, s = self._profile()
        bad = truth + self._excursion(s, 0.008)  # about ±0.55 g at 40 m/s
        filled, fills = trk.fill_from_inertial(bad, truth, v, ds)
        assert len(fills) == 1  # both lobes in one zone
        assert fills[0]["s_from"] < 780.0 and fills[0]["s_to"] > 820.0
        zone = (s >= fills[0]["s_from"]) & (s <= fills[0]["s_to"])
        # the course change across the zone is still the Doppler one
        assert np.sum(filled[zone]) == pytest.approx(np.sum(bad[zone]))
        assert np.array_equal(filled[~zone], bad[~zone])
        assert np.abs(filled - truth).max() < 0.05 * np.abs(bad - truth).max()

    def test_no_fill_under_the_threshold_or_without_a_reference(self):
        truth, v, ds, s = self._profile()
        small = truth + self._excursion(s, 0.002)
        assert trk.fill_from_inertial(small, truth, v, ds)[1] == []
        bad = truth + self._excursion(s, 0.008)
        ref = truth.copy()
        ref[(s > 790) & (s < 810)] = np.nan
        filled, fills = trk.fill_from_inertial(bad, ref, v, ds)
        assert fills == [] and np.array_equal(filled, bad)

    def test_zones_wrap_the_seam(self):
        over = np.zeros(100, dtype=bool)
        over[[98, 99, 0, 1]] = True
        over[40:43] = True
        assert trk.find_zones(over, 2.0) == [(40, 42), (98, 1)]
        assert trk.find_zones(over, 2.0, widen_m=4.0) == [(38, 44), (96, 3)]


class TestBuild:
    def test_stadium_is_recovered(self, tmp_path):
        root = _dataset(tmp_path)
        build = trk.build_track_model(root, LAYOUT_ID, inertial=False, measure_convergence=True)
        model = build.model
        assert len(build.laps) == 12
        assert model.length_m == pytest.approx(LENGTH_M, rel=2e-3)
        assert model.n * model.ds_m == pytest.approx(model.length_m)
        assert model.turns == 1
        assert abs(build.closure_turns) < 0.002
        assert build.line_gap_m < 1.0
        assert build.reproject_rms_g is not None and build.reproject_rms_g < 0.01
        s = model.s_m
        mid_arc = (np.abs(s - (STRAIGHT_M / 2 + ARC_M / 2)) < 40) | (
            np.abs(s - (1.5 * STRAIGHT_M + 1.5 * ARC_M)) < 40
        )
        assert model.kappa_heat[mid_arc] == pytest.approx(1 / RADIUS_M, rel=0.03)
        straight = np.abs(s - (STRAIGHT_M + ARC_M)) < 60
        assert np.abs(model.kappa_heat[straight]).max() < 5e-4
        assert np.sum(model.kappa_heat) * model.ds_m == pytest.approx(2 * math.pi, abs=1e-6)
        assert [c["sign"] for c in model.corners] == [1, 1]
        assert model.corners[0]["turn_angle"] == pytest.approx(180.0, abs=4.0)
        # a lap can miss the bin at the line itself by a fraction of a metre
        assert np.median(model.n_laps) == 12 and model.n_laps.min() >= 6
        assert model.alt_m == pytest.approx(20.0, abs=1e-3)
        assert np.abs(model.grade).max() < 1e-6
        assert np.isfinite(model.kappa_se).all() and np.median(model.kappa_se) < 5e-4
        # the stored line is the circuit
        sx, sy, _ = stadium(s)
        assert np.hypot(model.x_m - sx, model.y_m - sy).max() < 1.5
        assert model.meta["built_from"] == {
            "n_laps": 12,
            "n_sessions": 3,
            "n_days": 3,
            "cars": ["car"],
        }

    def test_anticlockwise_reads_negative(self, tmp_path):
        root = _dataset(tmp_path, mirror=True)
        model = trk.build_track_model(root, LAYOUT_ID, inertial=False, n_boot=0).model
        assert model.turns == -1
        s = model.s_m
        mid_arc = np.abs(s - (STRAIGHT_M / 2 + ARC_M / 2)) < 40
        assert model.kappa_heat[mid_arc] == pytest.approx(-1 / RADIUS_M, rel=0.03)
        assert [c["sign"] for c in model.corners] == [-1, -1]

    def test_sessions_and_cars_can_be_left_out(self, tmp_path):
        root = _dataset(tmp_path)
        build = trk.build_track_model(
            root, LAYOUT_ID, inertial=False, n_boot=0, exclude_session_ids=["2026-01-10_s0"]
        )
        assert {lap.session_id for lap in build.laps} == {"2026-01-11_s1", "2026-01-12_s2"}
        with pytest.raises(ValueError, match="need 3"):
            trk.build_track_model(root, LAYOUT_ID, cars=["other"], inertial=False)

    def test_json_round_trip(self, tmp_path):
        root = _dataset(tmp_path / "data")
        model = trk.build_track_model(root, LAYOUT_ID, inertial=False, n_boot=0).model
        path = model.save(tmp_path / "tracks" / "t.json")
        back = trk.TrackModel.load(path)
        assert back.layout_key == LAYOUT_ID and back.track_canonical == model.track_canonical
        assert back.kappa_heat == pytest.approx(model.kappa_heat, abs=1e-7)
        assert back.x_m == pytest.approx(model.x_m, abs=1e-3)
        assert np.isnan(back.kappa_se).all()  # no bootstrap: stored as nulls
        assert back.corners == model.corners and back.gate == model.gate
        assert back.turns == 1

    def test_project_lap(self, tmp_path):
        root = _dataset(tmp_path)
        model = trk.build_track_model(root, LAYOUT_ID, inertial=False, n_boot=0).model
        s_true = np.arange(5.0, LENGTH_M - 5.0, 7.0)
        x, y, _ = stadium(s_true)
        assert trk.project_lap(model, x, y, s_prev=s_true + 10.0) == pytest.approx(s_true, abs=1.5)


class TestWetBasis:
    @pytest.fixture()
    def root(self, tmp_path, monkeypatch):
        geo._INDEX_CACHE.clear()
        sessions = {}
        for d in range(3):
            sessions[f"2026-01-1{d}_dry"] = {
                "car": "car",
                "date": f"2026-01-1{d}",
                "kwargs": {"n_laps": 4, "seed": d, "noise_m": 0.02},
            }
            sessions[f"2026-02-1{d}_wet"] = {
                "car": "car",
                "date": f"2026-02-1{d}",
                "kwargs": {"n_laps": 12, "seed": 10 + d, "noise_m": 0.02, "pace": 1.1},
            }
        write_dataset(tmp_path, sessions)

        def _conditions(_root, frame, _laps):
            ids = frame["session_id"].astype(str)
            return pd.Series(
                ["wet" if i.endswith("_wet") else "dry" for i in ids], index=ids, name="condition"
            )

        monkeypatch.setattr(geo, "session_conditions", _conditions)
        return tmp_path

    def test_a_layout_has_a_track_per_basis(self, root):
        dry = trk.build_track_model(root, LAYOUT_ID, inertial=False, n_boot=0)
        wet = trk.build_track_model(root, LAYOUT_ID, basis="wet", inertial=False, n_boot=0)
        assert dry.model.meta["basis"] == "dry" and wet.model.meta["basis"] == "wet"
        assert {lap.session_id[-3:] for lap in dry.laps} == {"dry"}
        assert {lap.session_id[-3:] for lap in wet.laps} == {"wet"}
        assert len(wet.laps) == 36
        # the same road at a slower pace
        assert wet.model.length_m == pytest.approx(dry.model.length_m, rel=2e-3)
        assert np.median(wet.speed_median / dry.speed_median) == pytest.approx(1 / 1.1, rel=0.01)
        assert trk.model_summary(wet.model)["basis"] == "wet"

    def test_gate_builds_and_compares_the_wet_track(self, root, tmp_path):
        from motorsports_data_notebook.tire_model.lapsim import gates

        report = gates.check_track_truth(root, tracks_dir=tmp_path / "tracks", layouts=(LAYOUT_ID,))
        item = {i.name: i for i in report.items}["Wet tracks"]
        assert item.passed is True
        assert "1 of 1 layouts" in item.measured
        saved = trk.TrackModel.load(tmp_path / "tracks" / f"{LAYOUT_ID}__wet.json")
        assert saved.meta["basis"] == "wet"
        assert saved.meta["built_from"]["n_days"] == 3

    def test_unknown_basis(self, root):
        with pytest.raises(KeyError):
            trk.build_track_model(root, LAYOUT_ID, basis="damp")


class TestInertial:
    def test_accelerometer_delay_and_profile(self, tmp_path):
        root = _dataset(tmp_path, accel_delay_s=0.1)
        build = trk.build_track_model(root, LAYOUT_ID, n_boot=0)
        assert all(lap.lat_g is not None for lap in build.laps)
        assert build.laps[0].accel_delay_s == pytest.approx(0.1, abs=0.01)
        k_in, count = trk.inertial_kappa(build)
        assert np.median(count) == 12
        v = build.speed_median
        diff = v * v * (build.model.kappa_heat - k_in) / 9.80665
        assert np.sqrt(np.mean(diff**2)) < 0.06
        assert build.model.fills == []
        rows = np.array([lap.session_id == "2026-01-10_s0" for lap in build.laps])
        assert np.median(trk.inertial_kappa(build, rows)[1]) == 4

    def test_no_accelerometer_no_profile(self, tmp_path):
        root = _dataset(tmp_path)
        build = trk.build_track_model(root, LAYOUT_ID, n_boot=0)
        assert all(lap.lat_g is None for lap in build.laps)
        assert np.isnan(trk.inertial_kappa(build)[0]).all()
