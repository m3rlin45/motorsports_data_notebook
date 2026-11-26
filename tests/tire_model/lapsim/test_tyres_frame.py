"""The heat input rebuilt in the tyres' frame from GPS and the track (#376)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from motorsports_data_notebook.tire_model.lapsim import bank as bankmod
from motorsports_data_notebook.tire_model.lapsim import geometry as geo
from motorsports_data_notebook.tire_model.lapsim import track as trk
from motorsports_data_notebook.tire_model.lapsim import tyres_frame as tf

G = 9.80665
R = 100.0  # m, a clockwise circle: a right turn all the way round
V = 25.0  # m/s
BANK = 0.05  # rad, the left (outer) side higher: leaning into the right turn
GRADE = 0.02


def _track(n: int = 628) -> trk.TrackModel:
    length = 2 * np.pi * R
    ds = length / n
    s = np.arange(n) * ds
    theta = s / R
    return trk.TrackModel(
        layout_key="circle",
        track_canonical="circle",
        length_m=float(length),
        ds_m=float(ds),
        kappa_heat=np.full(n, 1.0 / R),
        x_m=R * np.sin(theta),
        y_m=R * np.cos(theta),
        alt_m=np.zeros(n),
        grade=np.full(n, GRADE),
        n_laps=np.ones(n, dtype=int),
        kappa_se=np.zeros(n),
        bank=np.full(n, BANK),
        bank_flag=np.zeros(n, dtype=np.int64),
    )


def _geometry(t: np.ndarray, radius: np.ndarray | float = R, speed: np.ndarray | float = V):
    """A car going clockwise round the circle (or one of ``radius``) at
    ``speed``: compass course = bearing from the centre + 90 degrees."""
    n = len(t)
    theta = V * t / R
    rad = np.broadcast_to(np.asarray(radius, float), (n,))
    nan = np.full(n, np.nan)
    return geo.SessionGeometry(
        session_id="s",
        layout_id="circle",
        t=t,
        x=rad * np.sin(theta),
        y=rad * np.cos(theta),
        alt=np.zeros(n),
        lap_num=np.ones(n, dtype=np.int64),
        seg=np.zeros(n, dtype=np.int64),
        speed=np.broadcast_to(np.asarray(speed, float), (n,)).copy(),
        course=theta + np.pi / 2,
        lat_g_doppler=nan.copy(),
        lat_g_position=nan.copy(),
        discard=np.zeros(n, dtype=bool),
        lag=geo.LagFit(tau_s=0.0),
        crossings=np.zeros(0),
    )


class TestProjection:
    def test_fixes_on_and_beside_the_line(self) -> None:
        track = _track()
        theta = np.linspace(0.1, 6.0, 50)
        for radius, expect in ((R, 0.0), (R + 5.0, -5.0), (R - 5.0, 5.0)):
            s, off = tf.project_session(track, radius * np.sin(theta), radius * np.cos(theta))
            assert np.allclose(s, theta * R, atol=0.6)
            # positive to the right of the direction of travel: the centre of a right turn
            assert np.allclose(off, expect, atol=0.1)

    def test_nan_fixes_stay_nan(self) -> None:
        s, off = tf.project_session(_track(), np.array([np.nan, 0.0]), np.array([0.0, R]))
        assert np.isnan(s[0]) and np.isnan(off[0]) and np.isfinite(s[1])


class TestSessionSeries:
    def test_on_the_line_the_tyres_hold_less_on_a_bank_into_the_turn(self) -> None:
        t = np.arange(0.0, 20.0, 0.04)
        series = tf.session_series(_geometry(t), _track())
        inner = slice(25, -25)
        expect = bankmod.to_tyres_frame(V * V / R / G, BANK)
        assert series.on_line[inner].all()
        assert np.allclose(series.lat_g[inner], expect, rtol=1e-3)
        assert np.allclose(series.long_g[inner], GRADE, atol=1e-6)

    def test_off_the_line_it_is_the_ground_frame_flat(self) -> None:
        t = np.arange(0.0, 20.0, 0.04)
        geom = _geometry(t, radius=R + 40.0)  # the pit lane, say: 40 m outside the line
        series = tf.session_series(geom, _track())
        inner = slice(25, -25)
        assert not series.on_line[inner].any()
        # the same yaw rate and speed as the line: V * omega / g, no bank, no grade
        assert np.allclose(series.lat_g[inner], V * (V / R) / G, rtol=1e-3)
        assert np.allclose(series.long_g[inner], 0.0, atol=1e-9)

    def test_no_track_is_the_ground_frame(self) -> None:
        t = np.arange(0.0, 20.0, 0.04)
        series = tf.session_series(_geometry(t), None)
        assert not series.on_line.any()
        assert np.allclose(series.lat_g[25:-25], V * V / R / G, rtol=1e-3)

    def test_standing_still_and_a_discarded_fix(self) -> None:
        t = np.arange(0.0, 20.0, 0.04)
        geom = _geometry(t)
        geom.speed[:100] = 0.0
        geom.course[:100] = np.nan  # the course is undefined at a standstill
        geom.discard[300] = True
        series = tf.session_series(geom, _track())
        # standing, the car's own acceleration is 0; on the road's bank and grade the
        # tyres still hold the slope (and the heat, V times the force, is 0 there)
        assert np.allclose(series.lat_g[:90], bankmod.to_tyres_frame(0.0, BANK))
        assert np.allclose(series.long_g[:90], GRADE)
        assert np.isnan(series.lat_g[300]) and np.isnan(series.long_g[300])

    def test_standing_off_the_line_is_zero(self) -> None:
        t = np.arange(0.0, 20.0, 0.04)
        geom = _geometry(t, radius=R + 40.0)
        geom.speed[:100] = 0.0
        geom.course[:100] = np.nan
        series = tf.session_series(geom, _track())
        assert np.all(series.lat_g[:90] == 0.0) and np.all(series.long_g[:90] == 0.0)


class TestOnClock:
    def _series(self, **over) -> tf.TyresSeries:  # type: ignore[no-untyped-def]
        t = np.array([0.0, 0.04, 0.08, 0.12, 5.0, 5.04, 5.4, 5.44, 5.48])
        fields = dict(
            t=t,
            seg=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]),
            lat_g=np.array([0.0, 0.4, 0.8, 1.2, 1.0, 1.0, np.nan, np.nan, 2.0]),
            long_g=np.array([0.0, 0.1, 0.2, 0.3, 0.0, 0.0, np.nan, np.nan, 0.4]),
            on_line=np.array([True, True, True, False, True, True, True, True, False]),
            hard=np.zeros(len(t), dtype=bool),
        )
        fields.update(over)
        return tf.TyresSeries(**fields)

    def test_linear_between_usable_fixes(self) -> None:
        lat, lng, on = tf.on_clock(self._series(), np.array([0.02, 0.06, 0.09, 0.11]))
        assert np.allclose(lat, [0.2, 0.6, 0.9, 1.1])
        assert np.allclose(lng, [0.05, 0.15, 0.225, 0.275])
        assert on.tolist() == [True, True, True, False]  # the nearer fix's

    def test_a_short_stretch_without_a_value_is_bridged(self) -> None:
        # 5.04 to 5.48: two fixes without a course in between, and a segment change at 5.04
        lat, lng, _ = tf.on_clock(self._series(), np.array([5.02, 5.26, 5.48]))
        assert np.allclose(lat, [1.0, 1.5, 2.0]) and np.allclose(lng, [0.0, 0.2, 0.4])

    def test_a_discarded_fix_is_never_bridged(self) -> None:
        hard = np.zeros(9, dtype=bool)
        hard[6] = True
        lat, lng, _ = tf.on_clock(self._series(hard=hard), np.array([5.26, 5.02]))
        assert np.isnan(lat[0]) and np.isnan(lng[0]) and np.isfinite(lat[1])

    def test_a_long_gap_is_a_hole(self) -> None:
        lat, lng, on = tf.on_clock(self._series(), np.array([2.0, 4.5]))
        assert np.isnan(lat).all() and np.isnan(lng).all() and not on.any()

    def test_the_edges_hold_by_a_little_and_no_further(self) -> None:
        # a file that starts or ends at speed: the first and last usable values hold
        lat, lng, on = tf.on_clock(self._series(), np.array([-0.5, 6.2, -1.5, 7.0]))
        assert np.allclose(lat[:2], [0.0, 2.0]) and np.allclose(lng[:2], [0.0, 0.4])
        assert on[:2].tolist() == [True, False]
        assert np.isnan(lat[2:]).all() and np.isnan(lng[2:]).all()


class TestTyresKinematics:
    def test_the_kinematics_clock_with_the_tyres_frame_series(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        t_gps = np.arange(0.0, 20.0, 0.04)
        geom = _geometry(t_gps)
        monkeypatch.setattr(geo, "load_session_geometry", lambda root, sid, lay: geom)
        t = np.arange(1.0, 25.0, 0.01)  # an IMU clock, running past the GPS's end
        speed = np.full(len(t), V)
        speed[t > 22.0] = 0.0  # standing in the pit after the GPS stopped
        kin = pa.table(
            {
                "t_session_s": t,
                "lap_num": np.ones(len(t), dtype=np.int64),
                "speed_ms": speed,
                "lat_g": np.full(len(t), 9.9),
                "long_g": np.full(len(t), 9.9),
                "lat_source": ["imu"] * len(t),
                "long_source": ["imu"] * len(t),
            }
        )
        out = tf.tyres_kinematics(Path("."), "s", kin, "circle", _track())
        assert out is not None
        assert out.schema.names[: len(kin.schema.names)] == kin.schema.names
        lat = np.asarray(out.column("lat_g").to_numpy(zero_copy_only=False), float)
        src = np.asarray(out.column("lat_source").to_numpy(zero_copy_only=False)).astype(str)
        on = np.asarray(out.column("on_line").to_numpy(zero_copy_only=False), bool)
        inside = t < 19.0
        assert np.allclose(lat[inside], bankmod.to_tyres_frame(V * V / R / G, BANK), rtol=1e-3)
        assert (src[inside] == tf.TYRES).all() and on[inside].all()
        # the sensor verdict stays beside, as a label only
        kind = np.asarray(out.column("sensor_source").to_numpy(zero_copy_only=False)).astype(str)
        assert (kind == "imu").all()
        # just past the GPS's last usable value the file runs on at speed: it holds
        held = (t > 19.98) & (t < 20.6)  # the last usable value is about 0.25 s before the last fix
        assert np.isfinite(lat[held]).all() and (src[held] == tf.TYRES).all()
        assert np.allclose(lat[held], lat[inside][-1], rtol=1e-3)
        # further on, still moving, there is no GPS: a hole
        moving_no_gps = (t > 20.8) & (t < 22.0)
        assert (src[moving_no_gps] == tf.DISCARD).all() and np.isnan(lat[moving_no_gps]).all()
        standing = t > 22.5
        assert (src[standing] == tf.TYRES).all() and (lat[standing] == 0.0).all()
        assert not on[standing].any()

    def test_no_geometry_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(geo, "load_session_geometry", lambda root, sid, lay: None)
        kin = pa.table({"t_session_s": [0.0], "speed_ms": [0.0]})
        assert tf.tyres_kinematics(Path("."), "s", kin, "circle", None) is None
