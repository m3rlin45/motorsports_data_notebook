"""Unit tests for helpers inside :mod:`tire_etl.extract`.

Full end-to-end extraction requires libxrk + a real AIM file; those are
exercised by an opt-in integration test. This file covers the pure
PyArrow transforms that are straightforward to unit-test.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from motorsports_data_notebook.tire_etl.dataset import narrowest_float

from datetime import datetime, timezone

import pyarrow.compute as pc

from motorsports_data_notebook.tire_etl.aggregates import channel_series
from motorsports_data_notebook.tire_etl.extract import (
    RawTimebase,
    _build_empty_session_row,
    _extract_session_datetime_utc,
    _pressure_to_bar,
    _timebase_names,
    _wanted_channels,
    build_session_timeseries,
    session_channel_names,
)
from motorsports_data_notebook.tire_etl.discovery import SessionCandidate

from datetime import date as _date
from pathlib import Path


class _FakeLog:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata


class _FakeLogFile:
    """Duck-typed LogFile: ``channels`` of ``{"timecodes", name}`` tables and a laps table."""

    def __init__(self, channels: dict[str, tuple[list[int], list[float]]], laps: list[tuple]):
        self.channels = {
            name: pa.table({"timecodes": pa.array(t, type=pa.int64()), name: pa.array(v)})
            for name, (t, v) in channels.items()
        }
        self.laps = pa.table(
            {
                "num": pa.array([n for n, _, _ in laps], type=pa.int64()),
                "start_time": pa.array([s for _, s, _ in laps], type=pa.int64()),
                "end_time": pa.array([e for _, _, e in laps], type=pa.int64()),
            }
        )
        self.metadata: dict = {}


class _FakeMergedLog:
    def __init__(self, parts: list[tuple[_FakeLogFile, int]]):
        self._parts = parts
        self.channels = {}
        for lg, _ in parts:
            for k, v in lg.channels.items():
                self.channels.setdefault(k, v)

    def parts(self):
        return list(self._parts)


_NAMES = {
    "gps_speed": "GPS Speed",
    "gps_latitude": "GPS Latitude",
    "gps_longitude": "GPS Longitude",
    "imu_lateral_g": "LateralAcc",
    "imu_inline_g": "InlineAcc",
    "tpms_press_fl": "TPMS_Press_LF",
    "tpms_temp_fl": "TPMS_Temp_LF",
    "tpms_press_fr": "TPMS_Press_RF",
}


def _two_lap_log() -> _FakeLogFile:
    t_gps = list(range(0, 20000, 40))  # 25 Hz
    t_imu = list(range(0, 20000, 20))  # 50 Hz, its own timebase
    t_tp = list(range(0, 20000, 800))  # 1.25 Hz
    t_tp2 = list(range(100, 20000, 800))  # the other corner, 100 ms later
    return _FakeLogFile(
        {
            "GPS Speed": (t_gps, [20.0 + i * 0.01 for i in range(len(t_gps))]),
            "GPS Latitude": (t_gps, [35.0 + i * 1e-6 for i in range(len(t_gps))]),
            "GPS Longitude": (t_gps, [139.0] * len(t_gps)),
            "LateralAcc": (t_imu, [0.5] * len(t_imu)),
            "InlineAcc": (t_imu, [-0.2] * len(t_imu)),
            "TPMS_Press_LF": (t_tp, [200.0 + i for i in range(len(t_tp))]),  # kPa
            "TPMS_Temp_LF": (t_tp, [30.0] * len(t_tp)),
            "TPMS_Press_RF": (t_tp2, [2.0] * len(t_tp2)),
        },
        laps=[(1, 1000, 11000), (2, 11000, 19000)],
    )


def test_build_session_timeseries_keeps_every_timebase_verbatim() -> None:
    log = _two_lap_log()
    ts, summary = build_session_timeseries(
        log,
        _NAMES,
        session_id="s",
        lap_nums=[1, 2],
        lap_starts_ms=[1000, 11000],
        lap_ends_ms=[11000, 19000],
        stint_ids=[1, 1],
    )
    names = {e["timebase"]: e for e in summary}
    assert set(names) == {"gps", "accel", "tpms/fl", "tpms/fr"}
    assert names["gps"]["rate_hz"] == pytest.approx(25.0)
    assert names["accel"]["rate_hz"] == pytest.approx(50.0)
    assert names["tpms/fl"]["channels"] == ["tpms_press_fl_bar", "tpms_temp_fl_c"]
    # samples inside [start, end) of a lap only, lap by window
    t_v, v = channel_series(ts, "speed_ms", "t_session_s")
    assert t_v.min() == pytest.approx(1.0) and t_v.max() < 19.0
    assert names["gps"]["n"] == len([t for t in range(0, 20000, 40) if 1000 <= t < 19000])
    lap_of = ts.filter(pc.is_valid(ts.column("speed_ms"))).column("lap_num").to_numpy()
    assert set(lap_of[t_v < 11.0]) == {1} and set(lap_of[t_v >= 11.0]) == {2}
    # t_lap_s counts from the lap start, sample_idx from 0 per (timebase, lap)
    gps_rows = ts.filter(pc.equal(ts.column("timebase"), "gps"))
    t_lap = gps_rows.column("t_lap_s").to_numpy()
    assert t_lap.min() == pytest.approx(0.0) and gps_rows.column("sample_idx")[0].as_py() == 0
    # a channel is null outside its timebase
    assert ts.filter(pc.equal(ts.column("timebase"), "gps")).column("lat_g").null_count == len(
        gps_rows
    )
    # kPa normalised to bar; position kept in double precision; the other
    # channels at the narrowest float width that holds them exactly
    _, p = channel_series(ts, "tpms_press_fl_bar")
    assert p[0] == pytest.approx(2.02, abs=1e-6) and p.max() < 3.0  # 202 kPa at t = 1.6 s
    assert ts.schema.field("gps_lat").type == pa.float64()
    raw_lat_g = log.channels[_NAMES["imu_lateral_g"]].column(_NAMES["imu_lateral_g"]).to_numpy()
    assert ts.schema.field("lat_g").type == narrowest_float(raw_lat_g).type
    _, stored_lat_g = channel_series(ts, "lat_g")
    assert np.array_equal(stored_lat_g, raw_lat_g.astype(np.float64)[: stored_lat_g.size])


def test_merged_parts_land_on_the_merged_clock() -> None:
    """A restart pair: the second file's raw clock starts at 0 again; its
    samples must carry t_session_s = raw + offset and a non-negative t_lap_s."""
    a = _FakeLogFile({"GPS Speed": (list(range(0, 10000, 40)), [10.0] * 250)}, [(1, 0, 10000)])
    b = _FakeLogFile({"GPS Speed": (list(range(0, 10000, 40)), [20.0] * 250)}, [(1, 0, 10000)])
    merged = _FakeMergedLog([(a, 0), (b, 600_000)])
    ts, summary = build_session_timeseries(
        merged,
        {"gps_speed": "GPS Speed"},
        session_id="m",
        lap_nums=[1, 2],
        lap_starts_ms=[0, 600_000],
        lap_ends_ms=[10_000, 610_000],
        stint_ids=[1, 1],
    )
    assert [e["timebase"] for e in summary] == ["gps"] and summary[0]["n"] == 500
    t = ts.column("t_session_s").to_numpy()
    lap = ts.column("lap_num").to_numpy()
    assert t[lap == 2].min() == pytest.approx(600.0)
    assert ts.column("t_lap_s").to_numpy().min() >= 0.0
    _, v = channel_series(ts, "speed_ms", "t_session_s")
    assert v[lap == 2].mean() == pytest.approx(20.0)


def test_timebase_names_split_families_by_channel_set() -> None:
    tbs = [
        RawTimebase("accel", {"lat_g": np.zeros(2)}, np.array([0, 2])),
        RawTimebase("accel", {"long_g": np.zeros(2), "vert_g": np.zeros(2)}, np.array([0, 2])),
        RawTimebase(
            "surface", {f"surf_temp_fl_ch{i}_c": np.zeros(1) for i in (1, 2, 3, 4)}, np.array([0])
        ),
        RawTimebase(
            "surface", {f"surf_temp_fl_ch{i}_c": np.zeros(1) for i in (5, 6, 7, 8)}, np.array([0])
        ),
        RawTimebase("gyro", {"yaw_rate_dps": np.zeros(1)}, np.array([0])),
    ]
    assert _timebase_names(tbs) == [
        "accel/lat_g",
        "accel/long_g+vert_g",
        "surface/fl_ch1-4",
        "surface/fl_ch5-8",
        "gyro",
    ]


def test_pressure_to_bar() -> None:
    np.testing.assert_allclose(_pressure_to_bar(np.array([180.0, 200.0])), [1.8, 2.0])
    np.testing.assert_allclose(_pressure_to_bar(np.array([1.8, 2.0])), [1.8, 2.0])


def test_session_time_from_aim_log_date_and_log_time() -> None:
    """AIM's `Log Date` (MM/DD/YYYY) + `Log Time` must be parsed and converted
    from track-local tz to UTC. Real AIM fleet data uses these keys; the old
    fallback (midnight local) was dropping within-day resolution and causing
    weather joins to land on the wrong hour (e.g. missing rain)."""
    log = _FakeLog({"Log Date": "04/04/2026", "Log Time": "12:57:49"})
    dt = _extract_session_datetime_utc(log, "2026-04-04", "tsukuba_2000")
    # 12:57:49 JST = 03:57:49 UTC
    assert dt == datetime(2026, 4, 4, 3, 57, 49, tzinfo=timezone.utc)


def test_session_time_from_aim_log_date_unambiguous_month_day() -> None:
    """MM/DD vs DD/MM was resolved empirically against a 04/17 file — 04/17 is
    only valid as MM/DD, so we must parse that format."""
    log = _FakeLog({"Log Date": "04/17/2026", "Log Time": "12:23:55"})
    dt = _extract_session_datetime_utc(log, "2026-04-17", "tsukuba_2000")
    assert dt == datetime(2026, 4, 17, 3, 23, 55, tzinfo=timezone.utc)


def test_session_time_falls_back_to_midnight_local_when_metadata_missing() -> None:
    log = _FakeLog({})
    dt = _extract_session_datetime_utc(log, "2026-04-04", "tsukuba_2000")
    # Midnight JST = 15:00 UTC previous day
    assert dt == datetime(2026, 4, 3, 15, 0, 0, tzinfo=timezone.utc)


def test_session_row_error_msg_is_always_string_typed() -> None:
    """Regression: error_msg must be pa.string() even when value is None.

    Otherwise PyArrow infers pa.null() for all-None monthly partitions, and
    cross-month concat/DuckDB reads fail with a schema-mismatch error.
    """
    cand = SessionCandidate(
        path=Path("/tmp/foo.xrk"),
        date=_date(2026, 4, 1),
        driver="CMD",
        car="Inferno 86",
        track_raw="Tsukuba",
        track_canonical="tsukuba_2000",
        session_type="testing",
        run_num=1,
    )
    row = _build_empty_session_row(
        session_id="abc123",
        path=Path("/tmp/foo.xrk"),
        mtime_ns=0,
        file_size=0,
        cand=cand,
        extractor_version="0.3.0",
    )
    assert row.schema.field("error_msg").type == pa.string()


def test_restint_at_pit_exits_splits_short_pit_visits() -> None:
    import pyarrow as pa
    from motorsports_data_notebook.tire_etl.extract import _restint_at_pit_exits

    laps = pa.table(
        {
            "lap_num": pa.array([0, 1, 2, 3, 4, 5], type=pa.int16()),
            "stint_id": pa.array([1, 1, 1, 1, 1, 1], type=pa.int16()),
            "outlap_from_pit": [True, False, False, True, False, False],
        }
    )
    ts = pa.table(
        {
            "lap_num": pa.array([0, 0, 1, 2, 3, 4, 5], type=pa.int16()),
            "stint_id": pa.array([1] * 7, type=pa.int16()),
        }
    )
    laps2, ts2 = _restint_at_pit_exits(laps, ts)
    assert laps2.column("stint_id").to_pylist() == [1, 1, 1, 2, 2, 2]
    assert ts2.column("stint_id").to_pylist() == [1, 1, 1, 1, 2, 2, 2]
    # Existing gap-based stints are preserved and renumbered monotonically.
    laps3 = laps.set_column(1, "stint_id", pa.array([1, 1, 1, 2, 2, 3], type=pa.int16()))
    out3, _ = _restint_at_pit_exits(laps3, ts)
    assert out3.column("stint_id").to_pylist() == [1, 1, 1, 2, 2, 3]


def test_car_name_takes_vehicle_metadata_spelling_only_for_case_differences() -> None:
    from motorsports_data_notebook.tire_etl.extract import _car_name

    assert _car_name("kk-f", "KK-F") == "KK-F"
    assert _car_name("KK-F", "KK-F") == "KK-F"
    assert _car_name("KK-F", "") == "KK-F"
    assert _car_name("KK-F", None) == "KK-F"
    assert _car_name("Inferno 86", "KK-F") == "Inferno 86"  # substance differs: filename wins
    assert _car_name(None, "KK-F") == "KK-F"
    assert _car_name(None, None) is None


class TestFileComment:
    def test_long_comment_is_normalised(self) -> None:
        from motorsports_data_notebook.tire_etl.extract import file_comment, session_comment

        class _Log:
            def __init__(self, meta):
                self.metadata = meta

        assert file_comment(_Log({"Long Comment": "Race\r\nstarted 4 \r\nslicks"})) == (
            "Race\nstarted 4\nslicks"
        )
        assert file_comment(_Log({"Long Comment": "   "})) is None
        assert file_comment(_Log({})) is None
        assert file_comment(_Log(None)) is None
        # Merged restart pair: identical comments collapse, distinct ones join in order.
        logs = [_Log({"Long Comment": "re-71rs"}), _Log({"Long Comment": "re-71rs"})]
        assert session_comment(logs) == "re-71rs"
        logs.append(_Log({"Long Comment": "wet tires"}))
        assert session_comment(logs) == "re-71rs\nwet tires"
        assert session_comment([_Log({})]) is None

    def test_session_row_carries_comment(self, tmp_path) -> None:
        from datetime import date

        from motorsports_data_notebook.tire_etl.discovery import parse_filename
        from motorsports_data_notebook.tire_etl.extract import _build_empty_session_row

        p = tmp_path / "2026-10-25" / "CMD_KK-F_Tsukuba_Car_Race_a_0381.xrk"
        row = _build_empty_session_row(
            session_id="x",
            path=p,
            mtime_ns=1,
            file_size=1,
            cand=parse_filename(p, date_hint=date(2026, 10, 25)),
            extractor_version="t",
        )
        assert row.column("comment").to_pylist() == [None]


def test_session_id_is_readable_and_sortable() -> None:
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    from motorsports_data_notebook.tire_etl.extract import (
        _local_wallclock,
        session_id_for,
    )

    start = datetime(2026, 10, 24, 12, 20, 53, tzinfo=ZoneInfo("Asia/Tokyo"))
    assert (
        session_id_for("2026-10-24", start, "KK-F", "tsukuba", [370, 371, 372])
        == "2026-10-24_122053_KK-F_tsukuba_a0370-0372"
    )
    # one run, non-consecutive runs, spaces in the car, underscore in the track (the id separator)
    assert session_id_for("2025-03-08", start, "Inferno 86", "sodegaura", [5079]).endswith(
        "_Inferno-86_sodegaura_a5079"
    )
    assert session_id_for("2025-03-08", start, "718 Cayman", "fuji_gp", [10, 12]).endswith(
        "_718-Cayman_fuji-gp_a0010+0012"
    )
    # unknown start / car / track still give a unique, sortable id
    assert session_id_for("2025-03-08", None, None, None, [None]) == (
        "2025-03-08_000000_unknown-car_unknown-track"
    )
    # the clock in the id is the logger's wall clock at the track; an
    # unresolved track is still in Japan
    utc = datetime(2026, 10, 24, 3, 20, 53, tzinfo=timezone.utc)
    assert _local_wallclock(utc, "tsukuba_2000").strftime("%H%M%S") == "122053"
    assert _local_wallclock(utc, None).strftime("%H%M%S") == "122053"
    assert _local_wallclock(None, "tsukuba_2000") is None


class _FakeAimLog:
    """Only what profile resolution reads: metadata and the file name."""

    def __init__(self, logger_id: int | None, vehicle: str, file_name: str = "") -> None:
        self.metadata = {"Vehicle": vehicle}
        if logger_id is not None:
            self.metadata["Logger ID"] = logger_id
        self.file_name = file_name


class TestSessionChannelNames:
    """The extractor resolves the profile by logger ID plus vehicle (#254)."""

    def test_shared_sw4_logger_is_split_by_vehicle(self) -> None:
        names_f, profile_f, logger_f = session_channel_names(_FakeAimLog(8401203, "KK-F"))
        names_s, profile_s, _ = session_channel_names(_FakeAimLog(8401203, "KK-SII"))
        assert (profile_f, logger_f) == ("KK-F", "8401203")
        assert profile_s == "KK-SII"
        assert names_f["throttle"] == "Throttle_PCT"
        assert names_f["brake"] == "P_Brake_Actual"
        assert names_s["throttle"] == "Throttle"
        assert names_s["brake"] == "F_BRK"

    def test_vehicle_from_the_file_name_when_metadata_is_blank(self) -> None:
        log = _FakeAimLog(8401203, "", "CMD_KK-F-V2_Tsukuba_Car_一般テスト_a_0392.xrz")
        assert session_channel_names(log)[1] == "KK-F-V2"

    def test_pre_sw4_kk_f_loggers_are_mapped(self) -> None:
        for logger_id in (6521016, 6520348):
            names, profile, _ = session_channel_names(_FakeAimLog(logger_id, "KK-F"))
            assert profile == "KK-F (data logger)"
            assert names["brake"] == "P_Brake"
            assert names["throttle_angle"] == "TA"
            assert names["steering"] == "Steering"
            assert names["vertical_g"] == "VerticalAcc"

    def test_lent_logger_has_no_profile_for_another_car(self) -> None:
        names, profile, logger_id = session_channel_names(_FakeAimLog(6520348, "718 Cayman"))
        assert (profile, logger_id) == ("default", "6520348")
        assert names["throttle"] == "PPS"

    def test_no_logger_id_means_defaults(self) -> None:
        names, profile, logger_id = session_channel_names(_FakeAimLog(None, "KK-F"))
        assert (profile, logger_id) == ("default", "")
        assert names["brake"] == "BrakePress"


def test_wanted_channels_for_the_kk_f_cover_controls_and_drivetrain() -> None:
    names, _, _ = session_channel_names(_FakeAimLog(8401203, "KK-F"))
    wanted = _wanted_channels(names)
    assert wanted["Throttle_PCT"] == ("controls", "throttle_pct")
    assert wanted["TA"] == ("controls", "throttle_deg")
    assert wanted["P_Brake_Actual"] == ("controls", "brake_bar")
    assert wanted["ENGRPM"] == ("drivetrain", "engine_rpm")
    assert wanted["GP"] == ("drivetrain", "gear")
    assert wanted["SPD_FL"] == ("drivetrain", "wheel_speed_fl_kmh")
    # The ECU's sign-inverted brake count and the SW4's own gear are not stored.
    assert "Pbrake_Value" not in wanted
    assert "Gear" not in wanted


def test_wanted_channels_for_the_pre_sw4_kk_f_store_the_angle_not_a_percentage() -> None:
    names, _, _ = session_channel_names(_FakeAimLog(6521016, "KK-F"))
    wanted = _wanted_channels(names)
    assert wanted["TA"] == ("controls", "throttle_deg")
    assert wanted["P_Brake"] == ("controls", "brake_bar")
    assert not [aim for aim, (_, col) in wanted.items() if col == "throttle_pct" and aim == "TA"]


def test_drivetrain_channels_are_stored_on_their_own_timebases() -> None:
    t_ecu = list(range(0, 20000, 100))  # 10 Hz ECU stream
    t_gear = list(range(0, 20000, 200))  # 5 Hz
    log = _FakeLogFile(
        {
            "GPS Speed": (list(range(0, 20000, 40)), [30.0] * 500),
            "RPM": (t_ecu, [5000.5 + i for i in range(len(t_ecu))]),
            "WheelSpdFL": (t_ecu, [100.0] * len(t_ecu)),
            "Gear": (t_gear, [3.0] * len(t_gear)),
            "TA": (t_ecu, [84.0] * len(t_ecu)),
        },
        laps=[(1, 1000, 11000), (2, 11000, 19000)],
    )
    names = {
        "gps_speed": "GPS Speed",
        "rpm": "RPM",
        "gear": "Gear",
        "wheel_speed_fl": "WheelSpdFL",
        "throttle_angle": "TA",
    }
    table, summary = build_session_timeseries(
        log,
        names,
        session_id="s",
        lap_nums=[1, 2],
        lap_starts_ms=[1000, 11000],
        lap_ends_ms=[11000, 19000],
        stint_ids=[1, 1],
    )
    by_name = {e["timebase"]: e for e in summary}
    assert by_name["drivetrain/engine_rpm+wheel_speed_fl_kmh"]["rate_hz"] == pytest.approx(10.0)
    assert by_name["drivetrain/gear"]["rate_hz"] == pytest.approx(5.0)
    assert by_name["controls"]["channels"] == ["throttle_deg"]
    # RPM with a half is not float16-representable and stays float32; the gear is exact in float16.
    assert table.schema.field("engine_rpm").type == pa.float32()
    assert table.schema.field("gear").type == pa.float16()
    assert table.schema.field("wheel_speed_fl_kmh").type == pa.float16()
    # Columns: keys, then the families in their fixed order, drivetrain last.
    assert table.schema.names[-4:] == ["throttle_deg", "engine_rpm", "gear", "wheel_speed_fl_kmh"]
    _, rpm = channel_series(table.filter(pc.equal(table.column("lap_num"), 1)), "engine_rpm")
    assert rpm[0] == pytest.approx(5010.5)
