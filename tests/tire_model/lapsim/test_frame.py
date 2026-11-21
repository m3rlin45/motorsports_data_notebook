"""The session frame record (lap simulation C0)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from motorsports_data_notebook.tire_model.heat_input import ShareParams
from motorsports_data_notebook.tire_model.lapsim import frame as frm
from motorsports_data_notebook.tire_model.lapsim import geometry as geo

from ._synthetic import LAYOUT_ID, LENGTH_M, RADIUS_M, speed_at, write_dataset, write_model_json

SHARE = ShareParams(
    p_f=0.45,
    g_transfer_front=2.5,
    g_transfer_rear=2.5,
    beta0=math.log(0.6 / 0.4),
    beta1=0.0,
    eps_drive=1.0,
    eps_brake=1.0,
)
SID = "2026-01-10_s0"
PLANTED = (0.93, -0.02, 0.05)  # lateral slope, lateral zero, longitudinal zero
RATIOS = ("rec", "none")


def _dataset(tmp_path, sessions=None, **kwargs):
    geo._INDEX_CACHE.clear()
    if sessions is None:
        sessions = {
            SID: {"car": "car", "date": "2026-01-10", "kwargs": {"n_laps": 5, **kwargs}},
        }
    write_dataset(tmp_path, sessions)
    write_model_json(tmp_path)
    return tmp_path


def _clean_laps(root, sid=SID):
    index = geo.clean_lap_index(root)
    return index[(index["session_id"] == sid) & index["clean"]]


class TestGroundSeries:
    @pytest.mark.parametrize("mirror", [False, True])
    def test_lateral_is_v_squared_over_r_and_signed(self, tmp_path, mirror):
        root = _dataset(tmp_path, mirror=mirror, lag_s=0.1)
        geom = geo.load_session_geometry(root, SID, LAYOUT_ID)
        lat, lng = frm.ground_series(geom)
        # the middle of the first corner of lap 2: 20 m/s on a 50 m radius
        tr = geo.lap_trace(geom, 2)
        at = float(np.interp(100.0 + math.pi * RADIUS_M / 2.0, tr.s, tr.t))
        i = int(np.argmin(np.abs(geom.t - at)))
        want = 20.0**2 / RADIUS_M / 9.80665
        assert lat[i] == pytest.approx(-want if mirror else want, rel=0.02)
        assert abs(lng[i]) < 0.02

    def test_longitudinal_is_the_speed_traces_derivative(self, tmp_path):
        root = _dataset(tmp_path, lag_s=0.1)
        geom = geo.load_session_geometry(root, SID, LAYOUT_ID)
        _, lng = frm.ground_series(geom)
        tr = geo.lap_trace(geom, 2)
        # dV/dt = V·dV/ds of the planted speed trace, an eighth of a lap in
        s = LENGTH_M / 8.0
        at = float(np.interp(s, tr.s, tr.t))
        i = int(np.argmin(np.abs(geom.t - at)))
        dvds = -8.0 * 4 * math.pi / LENGTH_M * math.sin(4 * math.pi * s / LENGTH_M)
        assert lng[i] == pytest.approx(float(speed_at(s)) * dvds / 9.80665, abs=0.01)


class TestRobustLine:
    def test_outliers_do_not_move_the_line(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-1.5, 1.5, 4000)
        y = 0.93 * x - 0.02 + rng.normal(0.0, 0.03, x.size)
        y[:40] += 2.0  # kerb strikes
        slope, zero, sd = frm.robust_line(x, y)
        assert slope == pytest.approx(0.93, abs=0.003)
        assert zero == pytest.approx(-0.02, abs=0.003)
        assert sd == pytest.approx(0.03, rel=0.15)

    def test_too_few_points(self):
        assert np.isnan(frm.robust_line(np.array([1.0]), np.array([1.0]))[0])


class TestRecord:
    def test_recovers_a_planted_frame(self, tmp_path):
        root = _dataset(tmp_path, accel_delay_s=0.1, accel_frame=PLANTED)
        geom = geo.load_session_geometry(root, SID, LAYOUT_ID)
        inertial = frm.load_inertial(root, SID)
        rec = frm.session_frame(geom, inertial, _clean_laps(root))
        assert rec.fitted and rec.n_laps == 5
        assert rec.accel_delay_s == pytest.approx(0.1, abs=0.01)
        assert rec.lat_slope == pytest.approx(PLANTED[0], abs=0.01)
        assert rec.lat_zero_g == pytest.approx(PLANTED[1], abs=0.005)
        assert rec.long_zero_g == pytest.approx(PLANTED[2], abs=0.005)
        assert rec.long_slope == pytest.approx(1.0, abs=0.03)

    def test_to_sensor_applies_slope_and_zeros(self):
        rec = frm.FrameRecord("s", lat_slope=0.9, lat_zero_g=0.1, long_zero_g=-0.05)
        lat, lng = rec.to_sensor(np.array([1.0, -1.0]), np.array([0.2, 0.0]))
        assert lat == pytest.approx([1.0, -0.8])
        assert lng == pytest.approx([0.15, -0.05])

    def test_a_session_without_an_accelerometer_gets_the_identity(self, tmp_path):
        sessions = {SID: {"car": "car", "date": "2026-01-10", "accel_source": "gps_position"}}
        root = _dataset(tmp_path, sessions)
        geom = geo.load_session_geometry(root, SID, LAYOUT_ID)
        rec = frm.session_frame(geom, frm.load_inertial(root, SID), _clean_laps(root))
        assert not rec.fitted
        assert (rec.lat_slope, rec.lat_zero_g, rec.long_zero_g) == (1.0, 0.0, 0.0)


class TestReplay:
    def test_the_record_carries_the_frame_and_nothing_does_not(self, tmp_path):
        root = _dataset(tmp_path, accel_delay_s=0.1, accel_frame=PLANTED)
        record, laps = frm.session_replay(root, SID, LAYOUT_ID, _clean_laps(root), SHARE)
        assert list(laps["source"]) == ["imu"] * 5
        rec = laps[[f"rec_{c}" for c in frm.CORNERS]].to_numpy()
        none = laps[[f"none_{c}" for c in frm.CORNERS]].to_numpy()
        assert np.abs(rec).max() < 0.01
        assert np.abs(none).max() > 0.04
        # a clean synthetic sensor has no content above the ground frame's band
        noise = laps[[f"noise_{c}" for c in frm.CORNERS]].to_numpy()
        assert np.abs(noise).max() < 0.02

    def test_a_gps_sourced_lap_needs_no_record(self, tmp_path):
        sessions = {SID: {"car": "car", "date": "2026-01-10", "accel_source": "gps_position"}}
        root = _dataset(tmp_path, sessions)
        record, laps = frm.session_replay(root, SID, LAYOUT_ID, _clean_laps(root), SHARE)
        assert set(laps["source"]) == {"gps_position"}
        assert np.abs(laps[[f"rec_{c}" for c in frm.CORNERS]].to_numpy()).max() < 0.03
        assert laps[[f"noise_{c}" for c in frm.CORNERS]].isna().all().all()


class TestGate:
    def _root(self, tmp_path, n=3, frame=PLANTED):
        sessions = {
            f"2026-01-1{i}_s{i}": {
                "car": "car",
                "date": f"2026-01-1{i}",
                "has_tpms": i != 2,
                "kwargs": {
                    "n_laps": 4,
                    "seed": i,
                    "accel_delay_s": 0.05,
                    "accel_frame": (frame[0], frame[1], frame[2] + 0.02 * i),
                },
            }
            for i in range(n)
        }
        return _dataset(tmp_path, sessions)

    def test_tables(self, tmp_path):
        root = self._root(tmp_path)
        records, laps = frm.frame_table(root)
        assert len(records) == 3 and records["fitted"].all()
        assert list(records["has_tpms"]) == [True, True, False]
        assert records["long_zero_g"].to_numpy() == pytest.approx([0.05, 0.07, 0.09], abs=0.005)
        cells = frm.cell_scores(laps)
        assert len(cells) == 3 and (cells["laps"] == 4).all()
        assert cells["worst_rec"].max() < 0.01 < cells["worst_none"].min()

    def test_a_cell_needs_three_laps(self):
        import pandas as pd

        rows = []
        for sid, n in (("a", 2), ("b", 3)):
            for lap in range(n):
                row = {
                    "session_id": sid,
                    "car": "car",
                    "date": "2026-01-10",
                    "layout_id": LAYOUT_ID,
                    "logger_id": "1",
                    "has_tpms": True,
                    "source": "imu",
                    "lap_num": lap,
                }
                row.update({col: 0.01 for col in frm.RATIO_COLUMNS})
                rows.append(row)
        cells = frm.cell_scores(pd.DataFrame(rows))
        assert list(cells["session_id"]) == ["b"]

    def test_gate_report(self, tmp_path, monkeypatch):
        root = self._root(tmp_path)
        monkeypatch.setattr(frm, "MIN_GATE_CELLS", 2)
        rep = frm.check_frame(root)
        assert rep.gate == "C0" and rep.passed
        names = [it.name for it in rep.items]
        assert names[:2] == [
            "Worst tyre per session through the record",
            "Per-tyre mean through the record",
        ]
        assert rep.items[0].passed is True and rep.items[1].passed is True
        assert "Sessions without TPMS" in names
        assert "Longitudinal zero per day" in names
        assert "car" in rep.to_markdown()

    def test_too_few_sessions_is_not_scored(self, tmp_path):
        rep = frm.check_frame(self._root(tmp_path))
        assert rep.items[0].passed is None and rep.items[1].passed is None
