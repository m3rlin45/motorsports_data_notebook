"""Lap geometry on one clock (lap simulation B1)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from motorsports_data_notebook.tire_model.lapsim import geometry as geo

from ._synthetic import LAYOUT_ID, LENGTH_M, RADIUS_M, STRAIGHT_M, synthetic_session, write_dataset


def _geometry(**kwargs) -> tuple[geo.SessionGeometry, dict, pd.DataFrame]:
    frame, truth = synthetic_session(**kwargs)
    gps = pa.Table.from_pandas(frame.drop(columns=["yaw_rate_dps"]))
    geom = geo.session_geometry(gps, "s1", LAYOUT_ID)
    assert geom is not None
    return geom, truth, frame


class TestLocalXY:
    def test_degree_lengths_on_the_ellipsoid(self):
        # Lengths of one degree at 36° N on WGS84 (series expansion).
        xy = geo.local_xy(np.array([37.0, 36.0]), np.array([140.0, 141.0]), 36.0, 140.0)
        assert xy[0, 1] == pytest.approx(110_959.0, rel=2e-4)  # evaluated at 36°, not 36.5°
        assert xy[1, 0] == pytest.approx(90_163.7, rel=1e-5)
        assert xy[0, 0] == 0.0 and xy[1, 1] == 0.0

    def test_differs_from_the_sphere_by_direction(self):
        from motorsports_data_notebook.tire_etl.layouts import _local_xy

        ell = geo.local_xy(np.array([36.001]), np.array([140.001]), 36.0, 140.0)
        sph = _local_xy(np.array([36.001]), np.array([140.001]), 36.0, 140.0)
        assert sph[0, 1] / ell[0, 1] == pytest.approx(1.0021, abs=2e-4)  # north-south long
        assert sph[0, 0] / ell[0, 0] == pytest.approx(0.9977, abs=2e-4)  # east-west short


class TestDopplerCourse:
    def test_sum_recovers_the_heading(self):
        t = np.arange(0.0, 10.0, 0.04)
        heading = 0.3 * np.sin(t) + 0.2 * t
        yaw = np.zeros_like(t)
        yaw[1:] = np.degrees(np.diff(heading)) / np.diff(t)
        course, seg = geo.doppler_course(t, yaw)
        assert np.all(seg == 0)
        assert course - course[0] == pytest.approx(heading - heading[0], abs=1e-9)

    def test_irregular_steps_are_summed_as_logged(self):
        # a step of 120 ms carries the heading difference over its own duration
        t = np.array([0.0, 0.04, 0.16, 0.2])
        heading = np.array([0.0, 0.01, 0.05, 0.06])
        yaw = np.concatenate([[0.0], np.degrees(np.diff(heading)) / np.diff(t)])
        course, seg = geo.doppler_course(t, yaw)
        assert np.all(seg == 0)
        assert course == pytest.approx(heading, abs=1e-12)

    def test_cut_at_gap_and_at_missing_yaw(self):
        t = np.array([0.0, 0.04, 0.08, 0.5, 0.54, 0.58, 0.62])
        yaw = np.array([0.0, 10.0, 10.0, 10.0, 10.0, np.nan, 10.0])
        course, seg = geo.doppler_course(t, yaw)
        assert seg.tolist() == [0, 0, 0, 1, 1, 2, 2]
        # nothing is added across a cut
        assert course[3] == course[2]
        assert course[5] == course[4]


class TestAtOffset:
    def test_linear_shift_within_a_segment(self):
        t = np.arange(0.0, 1.0, 0.04)
        seg = np.zeros(t.size, dtype=int)
        out = geo.at_offset(t, 3.0 * t, seg, 0.1)
        ok = np.isfinite(out)
        assert out[ok] == pytest.approx(3.0 * (t[ok] + 0.1))
        assert not ok[-1] and ok[0]  # no value past the segment's end

    def test_nothing_is_carried_across_a_cut(self):
        t = np.arange(0.0, 1.0, 0.04)
        seg = (t >= 0.5).astype(int)
        out = geo.at_offset(t, t, seg, 0.1)
        last_of_first = np.flatnonzero(seg == 0)[-1]
        assert np.isnan(out[last_of_first])
        assert np.isfinite(out[last_of_first + 1])


class TestSessionGeometry:
    def test_lag_is_recovered(self):
        geom, _, _ = _geometry(lag_s=0.1)
        assert geom.lag.tau_s == pytest.approx(0.1, abs=0.003)
        assert geom.lag.scatter_deg < 0.1
        assert geom.lag.scatter_raw_deg > 5 * geom.lag.scatter_deg

    @pytest.mark.parametrize("lag_s", [0.065, 0.2, 0.4])
    def test_lag_range_of_the_loggers(self, lag_s):
        geom, _, _ = _geometry(lag_s=lag_s, n_laps=3)
        assert geom.lag.tau_s == pytest.approx(lag_s, abs=0.005)

    def test_speed_and_course_are_on_the_position_clock(self):
        geom, truth, _ = _geometry(lag_s=0.15)
        ok = np.isfinite(geom.course) & np.isfinite(geom.speed)
        assert ok.mean() > 0.99
        # the frame's speed column is 0.15 s late; the geometry's is not
        assert geom.speed[ok] == pytest.approx(truth["speed"][ok], abs=0.05)
        err = geo.wrap_angle(geom.course[ok] - truth["heading"][ok])
        assert np.abs(err).max() < math.radians(0.3)

    def test_course_increases_clockwise(self):
        geom, _, _ = _geometry(n_laps=2)
        ok = np.isfinite(geom.course)
        assert geom.course[ok][-1] - geom.course[ok][0] > 2 * math.pi

    def test_position_has_the_gate_at_the_origin(self):
        geom, _, _ = _geometry(n_laps=2)
        assert geom.x.min() == pytest.approx(0.0, abs=0.01)
        assert geom.x.max() == pytest.approx(2 * RADIUS_M, abs=0.01)
        assert geom.y.max() == pytest.approx(STRAIGHT_M / 2 + RADIUS_M, abs=0.01)

    def test_crossings_at_the_true_times(self):
        geom, truth, _ = _geometry(n_laps=4)
        assert geom.crossings == pytest.approx(truth["crossings"], abs=0.005)
        assert not geom.crossing_edge.any()

    def test_missing_columns_or_too_little_data(self):
        frame, _ = synthetic_session(n_laps=2)
        gps = pa.Table.from_pandas(frame)
        assert geo.session_geometry(gps.drop_columns(["speed_ms"]), "s", LAYOUT_ID) is None
        assert geo.session_geometry(gps.slice(0, 50), "s", LAYOUT_ID) is None

    def test_gap_splits_the_trace(self):
        frame, _ = synthetic_session(n_laps=3)
        t = frame["t_session_s"].to_numpy()
        frame = frame[(t < 40.0) | (t > 41.0)]
        geom = geo.session_geometry(pa.Table.from_pandas(frame), "s", LAYOUT_ID)
        assert geom is not None
        assert len(np.unique(geom.seg)) == 2
        assert geom.lag.tau_s == pytest.approx(0.1, abs=0.005)


class TestLapTable:
    def test_full_laps_match_and_measure_the_circuit(self):
        geom, truth, _ = _geometry(n_laps=5)
        rows = geo.lap_table(geom).set_index("lap_num")
        assert rows.loc[[1, 2, 3, 4, 5], "matched"].all()
        assert not rows.loc[0, "matched"] and not rows.loc[6, "matched"]
        full = rows.loc[[1, 2, 3, 4, 5]]
        assert full["length_m"].to_numpy() == pytest.approx(LENGTH_M, rel=1e-3)
        assert full["v_integral_m"].to_numpy() == pytest.approx(LENGTH_M, rel=1e-3)
        assert (full["turns"] == 1.0).all()
        assert full["closure"].abs().max() < 5e-4
        assert full["one_segment"].all()
        assert (full["n_implausible"] == 0).all()
        assert full["max_gap_s"].max() == pytest.approx(0.04, abs=1e-6)
        assert full["min_speed_ms"].min() == pytest.approx(20.0, abs=0.1)
        assert not full["edge"].any()

    def test_anticlockwise_turns_minus_one(self):
        geom, _, _ = _geometry(n_laps=3, mirror=True)
        rows = geo.lap_table(geom).set_index("lap_num")
        assert (rows.loc[[1, 2, 3], "turns"] == -1.0).all()
        assert rows.loc[[1, 2, 3], "closure"].abs().max() < 5e-4

    def test_peaks_are_the_cornering_acceleration(self):
        geom, _, _ = _geometry(n_laps=3)
        rows = geo.lap_table(geom).set_index("lap_num")
        # 20-28 m/s on a 50 m radius: about 1 g at the corner's ends
        assert 0.8 < rows.loc[2, "peak_doppler_g"] < 1.8
        assert rows.loc[2, "peak_position_g"] == pytest.approx(
            rows.loc[2, "peak_doppler_g"], abs=0.25
        )
        assert rows.loc[2, "spike_g"] < 0.5

    def test_position_slew_is_flagged_on_its_lap_only(self):
        geom, _, _ = _geometry(n_laps=4, slew=(2, 4.0))
        rows = geo.lap_table(geom).set_index("lap_num")
        assert rows.loc[2, "spike_g"] > geo.SPIKE_FLAG_G
        assert rows.loc[[1, 3, 4], "spike_g"].max() < 0.5
        # in the middle of the second straight
        assert rows.loc[2, "spike_s_m"] == pytest.approx(STRAIGHT_M + math.pi * RADIUS_M, abs=40.0)

    def test_recording_that_stops_before_the_line_gets_an_edge_crossing(self):
        frame, truth = synthetic_session(n_laps=3)
        # stop 4 m before the line that would close lap 3
        cut = float(truth["crossings"][3]) - 4.0 / 36.0
        frame = frame[frame["t_session_s"] <= cut]
        geom = geo.session_geometry(pa.Table.from_pandas(frame), "s", LAYOUT_ID)
        assert geom is not None
        assert geom.crossing_edge.sum() == 1
        assert geom.crossings[-1] == pytest.approx(truth["crossings"][3], abs=0.02)
        rows = geo.lap_table(geom).set_index("lap_num")
        assert rows.loc[3, "matched"] and rows.loc[3, "edge"]
        # the trace stops at the last fix, a few metres short of the line
        assert LENGTH_M - 6.0 < rows.loc[3, "length_m"] < LENGTH_M
        assert rows.loc[3, "turns"] == 1.0
        assert abs(rows.loc[3, "closure"]) < 1e-3

    def test_lap_with_a_gap_is_not_one_segment(self):
        frame, truth = synthetic_session(n_laps=3)
        mid = 0.5 * (truth["crossings"][1] + truth["crossings"][2])
        t = frame["t_session_s"].to_numpy()
        frame = frame[(t < mid) | (t > mid + 0.5)]
        geom = geo.session_geometry(pa.Table.from_pandas(frame), "s", LAYOUT_ID)
        assert geom is not None
        rows = geo.lap_table(geom).set_index("lap_num")
        assert not rows.loc[2, "one_segment"]
        assert rows.loc[2, "max_gap_s"] > 0.5
        assert rows.loc[1, "one_segment"] and rows.loc[3, "one_segment"]


class TestLapTrace:
    def test_trace_runs_from_crossing_to_crossing(self):
        geom, truth, _ = _geometry(n_laps=3)
        tr = geo.lap_trace(geom, 2)
        assert tr is not None
        assert tr.t[0] == pytest.approx(truth["crossings"][1], abs=0.005)
        assert tr.t[-1] == pytest.approx(truth["crossings"][2], abs=0.005)
        assert tr.s[0] == 0.0
        assert tr.length_m == pytest.approx(LENGTH_M, rel=1e-3)
        assert tr.lap_time_s == pytest.approx(np.diff(truth["crossings"])[1], abs=0.01)
        assert abs(tr.y[0]) < 0.05 and abs(tr.x[0]) < 0.05
        assert np.all(np.diff(tr.t) > 0)
        assert np.isfinite(tr.course).all() and np.isfinite(tr.speed).all()

    def test_unmatched_lap_has_no_trace(self):
        geom, _, _ = _geometry(n_laps=2)
        assert geo.lap_trace(geom, 0) is None
        assert geo.lap_trace(geom, 99) is None


class TestGyroScale:
    def test_scale_with_a_zero_offset(self):
        geom, truth, frame = _geometry(n_laps=3, gyro_scale=0.88, gyro_offset_dps=0.8)
        a, b = truth["crossings"][1], truth["crossings"][2]
        got = geo.gyro_scale(
            geom, frame["t_session_s"].to_numpy(), frame["yaw_rate_dps"].to_numpy(), a, b
        )
        assert got == pytest.approx(0.88, abs=0.005)

    def test_closure_alone_would_read_the_offset_as_scale(self):
        # the lap's own closure: 0.8 °/s over a 26 s lap is 21° of 360°
        frame, truth = synthetic_session(n_laps=2, gyro_scale=1.0, gyro_offset_dps=0.8)
        a, b = truth["crossings"][0], truth["crossings"][1]
        t = frame["t_session_s"].to_numpy()
        m = (t >= a) & (t <= b)
        closure = np.trapezoid(frame["yaw_rate_dps"].to_numpy()[m], t[m]) / 360.0
        assert closure > 1.04
        geom = geo.session_geometry(
            pa.Table.from_pandas(frame.drop(columns=["yaw_rate_dps"])), "s", LAYOUT_ID
        )
        assert geom is not None
        got = geo.gyro_scale(geom, t, frame["yaw_rate_dps"].to_numpy(), a, b)
        assert got == pytest.approx(1.0, abs=0.005)

    def test_no_scale_without_gyro_cover(self):
        geom, truth, frame = _geometry(n_laps=2)
        a, b = truth["crossings"][0], truth["crossings"][1]
        t = frame["t_session_s"].to_numpy()
        w = frame["yaw_rate_dps"].to_numpy()
        hole = (t < a + 5.0) | (t > a + 8.0)
        assert math.isnan(geo.gyro_scale(geom, t[hole], w[hole], a, b))
        assert math.isnan(geo.gyro_scale(geom, t[:50], w[:50], a, b))


def _index_rows(session_id, car, times, *, condition="dry", layout="tsukuba_2000", **over):
    n = len(times)
    base = {
        "session_id": session_id,
        "car": car,
        "layout_id": layout,
        "lap_num": np.arange(1, n + 1),
        "lap_type": "full",
        "lap_time_s": np.asarray(times, float),
        "matched": True,
        "one_segment": True,
        "n_implausible": 0,
        "max_gap_s": 0.04,
        "has_discard": False,
        "min_speed_ms": 15.0,
        "length_m": 2080.0,
        "spike_g": 0.2,
        "lag_scatter_deg": 0.2,
        "condition": condition,
    }
    base.update(over)
    return pd.DataFrame(base)


class TestClassifyLaps:
    def test_reasons_in_the_order_of_the_definition(self):
        df = _index_rows("a", "car", [60.0] * 9)
        df.loc[0, "lap_type"] = "out"
        df.loc[1, "matched"] = False
        df.loc[2, "n_implausible"] = 1
        df.loc[3, "max_gap_s"] = 0.2
        df.loc[4, "has_discard"] = True
        df.loc[5, "min_speed_ms"] = 1.0
        df.loc[6, "length_m"] = 2080.0 * 1.03
        df.loc[7, "lap_time_s"] = 60.0 * 1.2
        out = geo.classify_laps(df)
        assert out["clean_reason"].tolist() == [
            "not_full",
            "no_interval",
            "implausible_fix",
            "gap",
            "discard",
            "slow",
            "length",
            "pace",
            "",
        ]
        assert out["clean"].tolist() == [False] * 8 + [True]

    def test_first_failing_test_names_the_reason(self):
        df = _index_rows("a", "car", [60.0, 60.0])
        df.loc[0, ["lap_type", "has_discard"]] = ["in", True]
        assert geo.classify_laps(df)["clean_reason"].tolist() == ["not_full", ""]

    def test_excluded_layout(self):
        df = _index_rows("a", "car", [60.0, 60.0], layout="minami")
        assert (geo.classify_laps(df)["clean_reason"] == "layout").all()

    def test_session_best_ignores_laps_that_fail_another_test(self):
        df = _index_rows("a", "car", [50.0, 60.0, 61.0])
        df.loc[0, "length_m"] = 1500.0  # a cut lap is not the session's best
        out = geo.classify_laps(df)
        assert (out["session_best_s"] == 60.0).all()
        assert out["clean"].tolist() == [False, True, True]

    def test_flag_is_separate_from_clean(self):
        df = _index_rows("a", "car", [60.0, 60.5])
        df.loc[1, "spike_g"] = 2.0
        out = geo.classify_laps(df)
        assert out["clean"].all()
        assert out["flagged"].tolist() == [False, True]
        assert out["build_reason"].tolist() == ["", "flagged"]

    def test_reference_pace_is_the_third_fastest_session(self):
        parts = [
            _index_rows(f"s{i}", "car", [best, best * 1.01, best * 1.05])
            for i, best in enumerate([60.0, 61.0, 62.0, 64.0, 66.0])
        ]
        out = geo.classify_laps(pd.concat(parts, ignore_index=True))
        assert (out["reference_pace_s"] == 62.0).all()
        by = out.groupby("session_id")["build_reason"].apply(list)
        # 1.05 × 62 = 65.1: the 66 s session is out; in the others the lap at 1.05 × is
        assert by["s0"] == ["", "", "slow_lap"]
        assert by["s3"] == ["", "", "slow_lap"]
        assert by["s4"] == ["slow_session"] * 3

    def test_fastest_session_when_fewer_than_three(self):
        parts = [_index_rows("s0", "car", [60.0]), _index_rows("s1", "car", [62.0])]
        out = geo.classify_laps(pd.concat(parts, ignore_index=True))
        assert (out["reference_pace_s"] == 60.0).all()
        assert out["in_build"].all()

    def test_reference_pace_is_per_car_and_layout(self):
        parts = [
            _index_rows("s0", "fast", [60.0]),
            _index_rows("s1", "slow", [75.0]),
            _index_rows("s2", "slow", [100.0], layout="fuji_gp", length_m=4500.0),
        ]
        out = geo.classify_laps(pd.concat(parts, ignore_index=True))
        assert out["reference_pace_s"].tolist() == [60.0, 75.0, 100.0]
        assert out["in_build"].all()

    def test_condition_label_plays_no_part(self):
        # a session labelled wet but driven at pace builds and sets the
        # reference like any other; one that was really wet is slow
        parts = [
            _index_rows("s0", "car", [60.0]),
            _index_rows("s1", "car", [61.0], condition="wet"),
            _index_rows("s2", "car", [62.0], condition="unknown"),
            _index_rows("s3", "car", [70.0], condition="wet"),
            _index_rows("s4", "car", [64.0], condition="damp"),
        ]
        out = geo.classify_laps(pd.concat(parts, ignore_index=True))
        assert (out["reference_pace_s"] == 62.0).all()
        assert out["build_reason"].tolist() == ["", "", "", "slow_session", ""]

    def test_wet_days_are_their_own_basis(self):
        parts = [
            _index_rows("d0", "car", [60.0, 60.5]),
            _index_rows("d1", "car", [61.0], condition="wet"),  # labelled wet, at dry pace
            _index_rows("d2", "car", [61.5]),
            _index_rows("w4", "car", [95.0], condition="wet"),  # not driven for time
            _index_rows("w0", "car", [66.0, 67.0, 68.0], condition="wet"),
            _index_rows("w1", "car", [67.0], condition="wet"),
            _index_rows("w2", "car", [68.0], condition="wet"),
            _index_rows("w3", "car", [75.0], condition="wet"),  # slow even for the wet
            _index_rows("s0", "car", [70.0]),  # slow, not labelled wet
        ]
        out = geo.classify_laps(pd.concat(parts, ignore_index=True))
        by = out.groupby("session_id")["wet_build_reason"].apply(list)
        # wet reference: the third-fastest wet session, 68 s; 1.05 × is 71.4
        assert out.loc[out["session_id"] == "w0", "wet_reference_pace_s"].iloc[0] == 68.0
        assert by["w0"] == ["", "", "slow_lap"]
        assert by["w1"] == [""] and by["w2"] == [""]
        assert by["w3"] == ["slow_session"]
        assert by["d0"] == ["not_wet", "not_wet"] and by["d1"] == ["not_wet"]
        assert by["s0"] == ["not_wet"] and by["w4"] == ["not_wet"]
        # the two bases never share a lap
        assert not (out["in_build"] & out["in_build_wet"]).any()
        assert out["in_build"].sum() == 4 and out["in_build_wet"].sum() == 4

    def test_wet_basis_is_per_car(self):
        parts = [
            _index_rows("a0", "fast", [60.0]),
            _index_rows("a1", "fast", [66.0], condition="wet"),
            _index_rows("b0", "slow", [80.0]),
            _index_rows("b1", "slow", [90.0], condition="wet"),
        ]
        out = geo.classify_laps(pd.concat(parts, ignore_index=True))
        assert out["wet_reference_pace_s"].tolist()[1] == 66.0
        assert out["wet_reference_pace_s"].tolist()[3] == 90.0
        assert out["in_build_wet"].tolist() == [False, True, False, True]

    def test_unsound_receiver_is_left_out(self):
        df = _index_rows("a", "car", [60.0, 60.2], lag_scatter_deg=0.8)
        out = geo.classify_laps(df)
        assert out["clean"].all()
        assert (out["build_reason"] == "receiver").all()


class TestCleanLapIndex:
    @pytest.fixture()
    def root(self, tmp_path):
        write_dataset(
            tmp_path,
            {
                "2026-01-10_a": {"car": "car", "date": "2026-01-10", "kwargs": {"n_laps": 4}},
                "2026-01-11_b": {
                    "car": "car",
                    "date": "2026-01-11",
                    "kwargs": {"n_laps": 3, "slew": (2, 4.0), "gyro_scale": 0.9},
                },
            },
        )
        return tmp_path

    def test_index_of_a_dataset(self, root):
        geo._INDEX_CACHE.clear()
        ix = geo.clean_lap_index(root)
        assert len(ix) == 6 + 5
        a = ix[ix["session_id"] == "2026-01-10_a"].set_index("lap_num")
        assert a["clean"].tolist() == [False, True, True, True, True, False]
        assert a.loc[0, "clean_reason"] == "not_full"
        assert a["in_build"].sum() == 4
        assert a.loc[1, "lag_s"] == pytest.approx(0.1, abs=0.003)
        assert a.loc[1, "gyro_scale"] == pytest.approx(1.0, abs=0.005)
        assert a.loc[1, "condition"] == "unknown"
        assert a.loc[1, "date"] == "2026-01-10"
        b = ix[ix["session_id"] == "2026-01-11_b"].set_index("lap_num")
        assert b.loc[2, "flagged"] and b.loc[2, "clean"]
        assert b.loc[2, "build_reason"] == "flagged"
        assert b.loc[1, "gyro_scale"] == pytest.approx(0.9, abs=0.005)
        assert b["in_build"].sum() == 2

    def test_cache_file_is_written_and_reused(self, root, tmp_path, monkeypatch):
        geo._INDEX_CACHE.clear()
        cache = tmp_path / "cache"
        first = geo.clean_lap_index(root, cache_dir=cache)
        assert (cache / "lap_index.parquet").exists()
        geo._INDEX_CACHE.clear()

        def _no_build(_root):
            raise AssertionError("the index was rebuilt instead of read from its cache")

        monkeypatch.setattr(geo, "_build_index", _no_build)
        second = geo.clean_lap_index(root, cache_dir=cache)
        pd.testing.assert_frame_equal(first, second)

    def test_cache_is_dropped_when_the_dataset_changes(self, root, tmp_path):
        geo._INDEX_CACHE.clear()
        cache = tmp_path / "cache"
        geo.clean_lap_index(root, cache_dir=cache)
        write_dataset(
            root, {"2026-01-10_a": {"car": "car", "date": "2026-01-10", "kwargs": {"n_laps": 2}}}
        )
        geo._INDEX_CACHE.clear()
        ix = geo.clean_lap_index(root, cache_dir=cache)
        assert set(ix["session_id"]) == {"2026-01-10_a"}
        assert len(ix) == 4

    def test_summary_counts(self, root):
        import json

        geo._INDEX_CACHE.clear()
        summary = json.loads(geo.index_summary_json(geo.clean_lap_index(root)))
        assert summary["laps"] == 11
        assert summary["clean"] == 7
        assert summary["flagged_clean"] == 1
        assert summary["in_build"] == 6
