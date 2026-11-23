"""The car's sample tables (lap simulation C1)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import geometry as geo
from motorsports_data_notebook.tire_model.lapsim import samples as smp
from motorsports_data_notebook.tire_model.lapsim import track as trk

from ._synthetic import LAYOUT_ID, LENGTH_M, RADIUS_M, speed_at, write_dataset, write_model_json

G = 9.80665
DT = 0.04


def _signals(n=500, **over):
    """A hand-made session: 25 Hz, 40 m/s, nothing happening."""
    t = np.arange(n) * DT
    nan = np.full(n, np.nan)
    geom = geo.SessionGeometry(
        session_id="s",
        layout_id=LAYOUT_ID,
        t=t,
        x=np.zeros(n),
        y=np.zeros(n),
        alt=np.zeros(n),
        lap_num=np.ones(n, dtype=np.int64),
        seg=np.zeros(n, dtype=np.int64),
        speed=over.pop("speed", np.full(n, 40.0)),
        course=np.zeros(n),
        lat_g_doppler=nan.copy(),
        lat_g_position=nan.copy(),
        discard=np.zeros(n, dtype=bool),
        lag=geo.LagFit(tau_s=0.0),
        crossings=np.zeros(0),
    )
    info = {"throttle_channel": "throttle_pct", "brake_channel": "brake_bar", "brake_on_at": 1.0}
    info.update(over.pop("info", {}))
    fields = {
        "lat_g": np.zeros(n),
        "long_g": np.zeros(n),
        "throttle": np.zeros(n),
        "brake": np.zeros(n),
        "brake_on": np.zeros(n, dtype=bool),
        "gear": nan.copy(),
        "rpm": nan.copy(),
        "lat_acc": nan.copy(),
    }
    fields.update(over)
    return smp.Signals(geom=geom, info=info, **fields)


def _where(n=500, grade=0.0):
    """Track position of a hand-made session: 1.6 m per fix, one lap."""
    return np.arange(n) * 1.6, np.full(n, grade), np.ones(n, dtype=np.int64)


class TestHelpers:
    def test_hold_last(self):
        t = np.array([0.0, 1.0, 5.0])
        x = np.array([10.0, 20.0, 30.0])
        got = smp.hold_last(np.array([-0.5, 0.0, 0.9, 1.0, 3.0, 4.0, 5.1]), t, x, max_age_s=2.5)
        assert got[1:5] == pytest.approx([10.0, 10.0, 20.0, 20.0])
        assert np.isnan(got[0]) and np.isnan(got[5])  # before the first; older than the hold
        assert got[6] == 30.0

    def test_held_for_and_after(self):
        t = np.arange(10) * 0.5
        cond = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
        assert smp.held_for(t, cond) == pytest.approx([0, 0.5, 0, 0.5, 1.0, 1.5, 2.0, 0, 0.5, 1.0])
        assert smp.held_after(t, cond) == pytest.approx(
            [1.0, 0.5, 0, 2.0, 1.5, 1.0, 0.5, 0, 0.5, 0]
        )

    def test_runs(self):
        assert smp.runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 2), (4, 4)]
        assert smp.runs(np.zeros(3, dtype=bool)) == []


class TestThrottleSamples:
    def test_held_full_throttle_with_the_grade_taken_out(self):
        n = 500
        thr = np.zeros(n)
        thr[100:300] = 1.0  # t = 4.0 .. 11.96 s
        sig = _signals(throttle=thr, long_g=np.full(n, 0.10))
        got = smp.throttle_samples(sig, *_where(grade=0.03))
        # full for 0.8 s before and 0.25 s after
        assert got["t_s"].min() == pytest.approx(4.0 + 0.8, abs=DT)
        assert got["t_s"].max() == pytest.approx(11.96 - 0.25, abs=DT)
        assert got["a_g"].to_numpy() == pytest.approx(0.13)
        assert got["straight"].all()

    def test_cornering_samples_are_kept_but_not_straight(self):
        n = 500
        lat = np.where(np.arange(n) >= 250, 0.6, 0.0)
        sig = _signals(throttle=np.ones(n), lat_g=lat)
        got = smp.throttle_samples(sig, *_where())
        assert (got["straight"] == (got["t_s"] < 250 * DT)).all()
        assert (~got["straight"]).sum() > 100
        # the tyres' frame beside the ground frame: the same on a flat road
        assert np.array_equal(got["lat_tyre_g"], got["lat_g"])
        sig.lat_tyre = lat * 0.9
        banked = smp.throttle_samples(sig, *_where())
        assert banked["lat_tyre_g"].max() == pytest.approx(0.54)
        assert np.array_equal(banked["straight"], got["straight"])  # still by the ground frame

    def test_a_flat_shift_is_cut_out_by_the_gear(self):
        n = 500
        gear = np.where(np.arange(n) >= 250, 4.0, 3.0)  # changes at t = 10 s
        sig = _signals(throttle=np.ones(n), gear=gear)
        got = smp.throttle_samples(sig, *_where())
        near = (got["t_s"] > 10.0 - 0.3 + DT) & (got["t_s"] < 10.0 + 0.3 - DT)
        assert not near.any()
        assert ((got["t_s"] > 9.0) & (got["t_s"] < 9.6)).any()
        assert set(got["gear"]) == {3.0, 4.0}

    def test_slow_or_outside_a_lap_is_not_sampled(self):
        n = 500
        speed = np.where(np.arange(n) < 250, 5.0, 40.0)
        s, grade, lap_of = _where()
        lap_of[400:] = -1
        got = smp.throttle_samples(_signals(throttle=np.ones(n), speed=speed), s, grade, lap_of)
        assert got["t_s"].min() >= 250 * DT and got["t_s"].max() < 400 * DT


class TestCoast:
    def test_closed_throttle_brakes_off_straight_line(self):
        n = 500
        thr = np.ones(n)
        thr[100:200] = 0.0
        brake_on = np.zeros(n, dtype=bool)
        brake_on[150:200] = True
        sig = _signals(throttle=thr, brake_on=brake_on, long_g=np.full(n, -0.12))
        got = smp.coast_samples(sig, *_where())
        assert got["t_s"].min() == pytest.approx(100 * DT + 0.3, abs=DT)
        assert got["t_s"].max() == pytest.approx(149 * DT - 0.25, abs=DT)
        assert got["a_g"].to_numpy() == pytest.approx(-0.12)

    def test_needs_both_channels(self):
        sig = _signals(info={"brake_channel": ""})
        assert smp.coast_samples(sig, *_where()).empty


class TestBrakeZones:
    def _sig(self):
        n = 500
        on = np.zeros(n, dtype=bool)
        on[100:150] = True  # 2 s
        on[153:175] = True  # 0.12 s later: the same application
        on[300:305] = True  # a dab
        on[400:450] = True
        long = np.where(on, -0.8, 0.0)
        long[120] = -1.0
        lat = np.zeros(n)
        lat[120] = -0.3
        speed = np.full(n, 40.0)
        speed[100:175] = np.linspace(40.0, 22.0, 75)
        return _signals(brake_on=on, long_g=long, lat_g=lat, speed=speed, brake=on * 20.0)

    def test_zones(self):
        got = smp.brake_zones(self._sig(), *_where(grade=0.05))
        assert len(got) == 2
        z = got.iloc[0]
        assert z["source"] == "pressure"
        assert z["t_on_s"] == pytest.approx(100 * DT)
        assert z["duration_s"] == pytest.approx(74 * DT)
        assert z["s_on_m"] == pytest.approx(160.0) and z["s_off_m"] == pytest.approx(174 * 1.6)
        assert z["v_on_ms"] == pytest.approx(40.0) and z["v_min_ms"] == pytest.approx(22.0)
        # uphill at 5 %: the slope gives 0.05 g of the 0.8 g
        assert z["decel_median_g"] == pytest.approx(0.75)
        assert z["lat_peak_g"] == pytest.approx(0.3)
        assert z["brake_max"] == pytest.approx(20.0)

    def test_how_a_stop_begins(self):
        sig = self._sig()
        sig.throttle[:92] = 1.0  # full throttle until 8 samples before the first application
        sig.throttle[340:372] = 1.0
        sig.brake[100:106] = np.linspace(2.0, 14.0, 6)  # the pressure builds over 6 samples
        sig.brake[106:150] = 20.0
        got = smp.brake_zones(sig, *_where())
        first, second = got.iloc[0], got.iloc[1]
        # between sample 91, the last at full, and 100, the first on the brake: the
        # throttle left full within the sample after 91 and the brake came on within
        # the one before 100, so 8 samples between the middles
        assert first["lift_s"] == pytest.approx(8 * DT)
        assert first["build_s"] == pytest.approx(6 * DT)  # 16 of 20 is reached at sample 106
        assert np.isnan(second["lift_s"]) and second["build_s"] == pytest.approx(0.0)
        sig.info["brake_channel"] = ""
        bare = smp.brake_zones(sig, *_where())
        assert bare["build_s"].isna().all() and bare.iloc[0]["lift_s"] == pytest.approx(8 * DT)

    def test_a_zone_belongs_to_the_lap_it_begins_in(self):
        s, grade, lap_of = _where()
        lap_of[:120] = 1
        lap_of[120:] = -1
        got = smp.brake_zones(self._sig(), s, grade, lap_of)
        assert len(got) == 1 and got.iloc[0]["lap_num"] == 1

    def test_source_without_a_brake_channel(self):
        sig = self._sig()
        sig.info["brake_channel"] = ""
        got = smp.brake_zones(sig, *_where())
        assert set(got["source"]) == {"decel"} and got["brake_max"].isna().all()

    def test_the_held_part_of_a_stop_and_its_second_reading(self):
        sig = self._sig()
        sig.brake[100:110] = 10.0  # building: under 80 % of the 20 it reaches
        sig.long_acc = np.where(sig.brake_on, -0.9, np.nan)  # the accelerometer's own scale
        sig.long_acc[400:450] = np.nan  # the second stop is not IMU-sourced
        sig.lat_tyre = np.full(500, 0.2)
        got = smp.brake_zones(sig, *_where(grade=0.05))
        first, second = got.iloc[0], got.iloc[1]
        held = np.r_[110:150, 153:175]  # the fixes at 80 % of the pressure or more
        assert first["held_s"] == pytest.approx(len(held) * DT)
        assert first["decel_held_g"] == pytest.approx(np.mean(-(sig.long_g[held] + 0.05)))
        assert first["decel_held_accel_g"] == pytest.approx(0.9)  # as the sensor reads it
        assert first["v_held_ms"] == pytest.approx(np.mean(sig.geom.speed[held]))
        assert first["lat_held_tyre_g"] == pytest.approx(0.2)
        assert first["lat_peak_tyre_g"] == pytest.approx(0.2)
        assert first["lat_mean_tyre_g"] == pytest.approx(0.2)
        assert second["decel_held_g"] == pytest.approx(0.75) and np.isnan(
            second["decel_held_accel_g"]
        )

    def test_no_held_part_without_a_pressure_channel_and_the_flat_road(self):
        sig = self._sig()
        flat = smp.brake_zones(sig, *_where())
        assert flat.iloc[0]["lat_peak_tyre_g"] == pytest.approx(flat.iloc[0]["lat_peak_g"])
        sig.info["brake_channel"] = ""
        bare = smp.brake_zones(sig, *_where())
        assert bare[["held_s", "decel_held_g", "decel_held_accel_g"]].isna().all().all()


class TestApexRows:
    def _track(self):
        n = 400
        return trk.TrackModel(
            layout_key=LAYOUT_ID,
            track_canonical=LAYOUT_ID,
            length_m=800.0,
            ds_m=2.0,
            kappa_heat=np.zeros(n),
            x_m=np.zeros(n),
            y_m=np.zeros(n),
            alt_m=np.zeros(n),
            grade=np.zeros(n),
            n_laps=np.ones(n, dtype=int),
            kappa_se=np.zeros(n),
            corners=[
                {"s_in": 100.0, "s_apex": 200.0, "s_out": 300.0, "sign": -1, "kappa_apex": -0.02},
                {"s_in": 760.0, "s_apex": 790.0, "s_out": 40.0, "sign": 1, "kappa_apex": 0.01},
            ],
        )

    def test_a_left_corner_and_one_across_the_line(self):
        n = 500
        s, _, lap_of = _where()  # 0 .. 798.4 m
        lat = np.zeros(n)
        left = (s >= 100) & (s <= 300)
        lat[left] = -1.2
        lat[int(190 / 1.6)] = -1.5
        lat[s >= 760] = 0.7
        lat[s <= 40] = 0.9
        speed = 40.0 - 15.0 * np.exp(-(((s - 200.0) / 60.0) ** 2))
        acc = np.where(left, -1.1, np.nan)
        got = smp.apex_rows(_signals(lat_g=lat, speed=speed, lat_acc=acc), self._track(), s, lap_of)
        assert list(got["corner"]) == [0, 1]
        a, b = got.iloc[0], got.iloc[1]
        assert a["v_apex_ms"] == pytest.approx(25.0, abs=0.01)
        assert a["v_min_ms"] == pytest.approx(25.0, abs=0.01)
        assert a["lat_track_g"] == pytest.approx(25.0**2 * 0.02 / G, rel=1e-3)
        assert a["lat_doppler_g"] == pytest.approx(1.5)  # positive into the corner
        assert a["lat_accel_g"] == pytest.approx(1.1)
        assert b["lat_doppler_g"] == pytest.approx(0.9)  # from both ends of the lap
        assert np.isnan(b["lat_accel_g"])
        # a flat road: the tyres hold what the ground frame reads, under the car's weight
        assert a["lat_tyre_g"] == pytest.approx(1.5) and a["load"] == pytest.approx(1.0)
        # each sensor where the other peaks: the Doppler's spike is its own
        assert a["lat_tyre_cross_g"] == pytest.approx(1.2)
        assert a["lat_accel_cross_g"] == pytest.approx(1.1)
        assert np.isnan(b["lat_tyre_cross_g"]) and np.isnan(b["lat_accel_cross_g"])

    def test_the_tyres_frame_and_the_load_at_a_banked_apex(self):
        track = self._track()
        n = track.n
        track.kappa_heat = np.where((track.s_m >= 100) & (track.s_m <= 300), -0.02, 0.0)
        track.grade = np.full(n, 0.03)
        track.kappa_v = np.full(n, 1e-4)
        track.bank = np.full(n, -0.05)  # the right side higher: into the left corner
        track.bank[(track.s_m > 195) & (track.s_m < 205)] = -0.5  # a spike the median leaves out
        s, _, lap_of = _where()
        lat = np.where((s >= 100) & (s <= 300), -1.2, 0.0)
        speed = np.full(500, 25.0)
        got = smp.apex_rows(_signals(lat_g=lat, speed=speed), track, s, lap_of)
        a = got.iloc[0]
        # leaning into the corner by 0.05 rad, the road carries part of the cornering
        assert a["lat_tyre_g"] == pytest.approx(1.2 * math.cos(0.05) - math.sin(0.05))
        # the same pass on the track's line: the track's curvature at the lap's apex speed
        on_line = 25.0**2 * 0.02 / G
        assert a["lat_track_tyre_g"] == pytest.approx(on_line * math.cos(0.05) - math.sin(0.05))
        cos = math.cos(0.05) / math.sqrt(1 + 0.03**2)
        assert a["load"] == pytest.approx(cos + 25.0**2 * (1e-4 + 0.02 * math.sin(0.05)) / G)

    def test_a_sessions_lateral_in_the_tyres_frame(self):
        track = self._track()
        track.bank = np.full(track.n, 0.04)  # the left side higher
        s, _, _ = _where()
        s[:10] = np.nan  # before the first lap
        lat = np.full(500, 0.8)
        got = smp.in_tyres_frame(_signals(lat_g=lat), track, s)
        assert got[100] == pytest.approx(0.8 * math.cos(0.04) - math.sin(0.04))  # a right turn
        assert got[0] == pytest.approx(0.8)
        left = smp.in_tyres_frame(_signals(lat_g=-lat), track, s)
        assert left[100] == pytest.approx(-(0.8 * math.cos(0.04) + math.sin(0.04)))
        track.bank = np.zeros(0)
        assert np.array_equal(smp.in_tyres_frame(_signals(lat_g=lat), track, s), lat)


def _dataset(tmp_path, **kwargs):
    geo._INDEX_CACHE.clear()
    sessions = {
        f"2026-01-1{i}_s{i}": {
            "car": "car",
            "date": f"2026-01-1{i}",
            "kwargs": {"n_laps": 4, "seed": i, "controls": True, "accel_delay_s": 0.05, **kwargs},
        }
        for i in range(3)
    }
    write_dataset(tmp_path, sessions)
    write_model_json(tmp_path)
    return tmp_path


class TestSession:
    def test_signals_read_the_controls_relative_to_the_sessions_own_range(self, tmp_path):
        root = _dataset(tmp_path)
        sig = smp.session_signals(root, "2026-01-10_s0", LAYOUT_ID)
        assert sig.info["throttle_channel"] == "throttle_pct"
        assert (sig.info["throttle_closed"], sig.info["throttle_open"]) == (0.0, 100.0)
        assert set(np.unique(sig.throttle[np.isfinite(sig.throttle)])) == {0.0, 1.0}
        assert sig.info["brake_on_at"] == pytest.approx(0.05 * sig.info["brake_max"])
        assert sig.info["has_gear"] and not sig.info["has_rpm"]
        assert sig.info["has_accel"] and sig.info["accel_delay_s"] == pytest.approx(0.05, abs=0.01)

    def test_tables_of_a_session(self, tmp_path):
        root = _dataset(tmp_path)
        index = geo.clean_lap_index(root)
        laps = smp.build_laps(index)
        laps = laps[laps["session_id"] == "2026-01-10_s0"]
        track = smp.load_track(root, LAYOUT_ID)
        got = smp.session_tables(root, "2026-01-10_s0", track, laps)
        assert got["sessions"].iloc[0]["n_laps"] == len(laps) > 0
        th = got["throttle"]
        assert set(th["basis"]) == {"dry"}
        # the planted speed trace: dV/dt = V·dV/ds, on a flat track
        k = 4 * math.pi / LENGTH_M
        want = speed_at(th["s_m"]) * (-8.0 * k * np.sin(k * th["s_m"])) / G
        assert (th["a_g"] - want).abs().max() < 0.02
        assert (th["a_g"] > 0).all()
        # the car gains speed out of each corner and down the straight after it
        straight = th[th["straight"]]
        assert len(straight) > 50
        assert ((straight["s_m"] > 250) & (straight["s_m"] < 360) | (straight["s_m"] > 600)).all()
        # braking: one application into each corner, peak about 0.39 g
        br = got["brake"]
        assert len(br) == 2 * len(laps)
        assert br["decel_p95_g"].median() == pytest.approx(0.39, abs=0.04)
        apex = got["apex"]
        assert len(apex) == 2 * len(laps)
        # 20 m/s at the apex of a 50 m radius; the lap's own peak is at the
        # corner's ends, where the car is still (or again) doing 26 m/s
        assert apex["v_apex_ms"].median() == pytest.approx(20.0, abs=0.3)
        assert apex["lat_track_g"].median() == pytest.approx(20.0**2 / RADIUS_M / G, rel=0.05)
        assert 1.2 < apex["lat_doppler_g"].median() < 26.5**2 / RADIUS_M / G
        assert apex["lat_accel_g"].notna().all()
        # each lap's own lateral at the nodes, row for row with its speed
        speed, own = np.asarray(got["speed"]), np.asarray(got["lateral"])
        assert own.shape == speed.shape == (len(got["speed_laps"]), track.n)
        kappa = np.asarray(track.kappa_heat, float)
        turning = np.abs(kappa) > 0.5 * np.nanmax(np.abs(kappa))
        want = speed**2 * kappa / G  # a flat track: what the tyres hold is V²κ
        assert np.nanmedian(np.abs(own - want)[:, turning]) < 0.05
        assert (np.sign(own[:, turning]) == np.sign(kappa[turning])).mean() > 0.98

    def test_a_lateral_row_belongs_to_its_speed_rows_lap(self, tmp_path):
        root = _dataset(tmp_path)
        track = smp.load_track(root, LAYOUT_ID)
        sig = smp.session_signals(root, "2026-01-10_s0", LAYOUT_ID)
        laps = smp.build_laps(geo.clean_lap_index(root))
        s, _, lap_of = smp.track_position(sig, track, laps[laps["session_id"] == "2026-01-10_s0"])
        lap_nums, v = smp.speed_on_track(sig, track, s, lap_of)
        whole = smp.lateral_on_track(sig, track, s, lap_of, lap_nums)
        # without one lap's lateral its row is empty and the others stay where they were
        sig.lat_tyre = np.where(lap_of == lap_nums[1], np.nan, sig.lat_tyre_g)
        holed = smp.lateral_on_track(sig, track, s, lap_of, lap_nums)
        assert holed.shape == v.shape and np.isnan(holed[1]).all()
        keep = [k for k in range(len(lap_nums)) if k != 1]
        np.testing.assert_array_equal(holed[keep], whole[keep])


class TestTablesAndGate:
    def test_tables_are_cached_and_summarised(self, tmp_path):
        root = _dataset(tmp_path)
        cache = tmp_path / "cache"
        tables = smp.car_tables(root, cache_dir=cache, tracks_dir=tmp_path / "tracks")
        assert (tmp_path / "tracks" / f"{LAYOUT_ID}.json").exists()
        assert len(tables.sessions) == 3
        assert {"car", "date", "layout_id", "logger_id", "basis"} <= set(tables.throttle.columns)
        again = smp.car_tables(root, cache_dir=cache, tracks_dir=tmp_path / "tracks")
        pd.testing.assert_frame_equal(again.apex, tables.apex, check_dtype=False)
        np.testing.assert_array_equal(again.lateral[LAYOUT_ID], tables.lateral[LAYOUT_ID])
        own = tables.lap_laterals("car", LAYOUT_ID)
        laps, v = tables.lap_speeds("car", LAYOUT_ID)
        assert own is not None and own.shape == v.shape == (len(laps), v.shape[1])
        (cache / "car_tables" / "signature.txt").write_text("stale")
        rebuilt = smp.car_tables(root, cache_dir=cache, tracks_dir=tmp_path / "tracks")
        assert len(rebuilt.apex) == len(tables.apex)

        st = smp.straight_table(tables.throttle)
        assert set(st["car"]) == {"car"} and (st["days"] <= 3).all()
        assert (np.diff(st["v_bin_ms"]) == smp.SPEED_BIN_MS).all()
        summary = smp.apex_summary(tables.apex)
        assert len(summary) == 2 and summary["grip_limited"].all()
        level = smp.lateral_level(summary)
        assert 1.2 < level["level_g"].iloc[0] < 26.5**2 / RADIUS_M / G
        rows = smp.build_laps(geo.clean_lap_index(root))
        counts = smp.cell_counts(tables, rows)
        assert counts["apex_laps"].iloc[0] == counts["laps"].iloc[0] == len(rows)

    def test_grip_limited_is_relative_to_the_layouts_highest(self):
        rows = []
        for corner, level in ((0, 1.5), (1, 1.3), (2, 1.0)):
            for lap in range(10):
                rows.append(
                    {
                        "session_id": "s",
                        "car": "car",
                        "date": "2026-01-10",
                        "layout_id": LAYOUT_ID,
                        "basis": "dry",
                        "lap_num": lap,
                        "corner": corner,
                        "sign": 1,
                        "s_apex_m": 100.0 * corner,
                        "v_apex_ms": 30.0,
                        "lat_track_g": level,
                        "lat_doppler_g": level,
                        "lat_accel_g": np.nan,
                    }
                )
        summary = smp.apex_summary(pd.DataFrame(rows))
        assert list(summary["grip_limited"]) == [True, True, False]
        assert smp.lateral_level(summary)["level_g"].iloc[0] == pytest.approx(1.4)

    def test_gate(self, tmp_path, monkeypatch):
        root = _dataset(tmp_path)
        monkeypatch.setattr(smp, "MODEL_CARS", ("car",))
        rep = smp.check_envelope_samples(root, tracks_dir=tmp_path / "tracks")
        assert rep.gate == "C1" and rep.passed
        names = [it.name for it in rep.items]
        assert names[0] == "Every model car's cell has its tables" and rep.items[0].passed is True
        for want in (
            "Channel coverage",
            "Speed range of each straight table",
            "Lateral level at grip-limited corners",
            "Braking zones",
        ):
            assert want in names
        assert "throttle_pct" in rep.to_markdown()

    def test_gate_fails_a_model_car_without_a_throttle(self, tmp_path, monkeypatch):
        root = _dataset(tmp_path, controls=False)
        monkeypatch.setattr(smp, "MODEL_CARS", ("car",))
        rep = smp.check_envelope_samples(root)
        assert rep.items[0].passed is False and not rep.passed
