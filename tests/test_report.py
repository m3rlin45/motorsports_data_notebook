"""Tests for the report module.

Uses mocked LogFile and analysis functions to test orchestration
and graceful degradation.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from motorsports_data_notebook.corners import Corner
from motorsports_data_notebook.report import (
    BrakingBalanceSummary,
    CornerBestLap,
    CornerConsistency,
    CornerInfo,
    LapTimeSummary,
    SessionMetadata,
    SessionReport,
    SuspensionSummary,
    TireGripSummary,
    _check_channels_available,
    _compute_best_lap,
    _corner_to_info,
    _extract_suspension_summary,
    _extract_tire_grip_summary,
    generate_session_report,
)


def _make_corner(
    id=1,
    name="Turn 1",
    direction="L",
    start_dist=100.0,
    end_dist=200.0,
    apex_dist=150.0,
    max_curvature=0.01,
) -> Corner:
    """Create a test Corner."""
    return Corner(
        id=id,
        name=name,
        direction=direction,
        start_idx=int(start_dist),
        end_idx=int(end_dist),
        start_dist=start_dist,
        end_dist=end_dist,
        apex_idx=int(apex_dist),
        apex_dist=apex_dist,
        max_curvature=max_curvature,
    )


def _make_log_mock(channel_names_present=None, laps=None):
    """Create a mock LogFile with specified channels available."""
    log = MagicMock()
    if channel_names_present is None:
        channel_names_present = []
    log.channels = {name: MagicMock() for name in channel_names_present}

    if laps is None:
        laps = pd.DataFrame(
            {
                "num": [0, 1, 2, 3],
                "start_time": [0, 100, 200, 300],
                "end_time": [100, 200, 300, 400],
                "lap_time": pd.to_timedelta([100, 90, 91, 92], unit="s"),
            }
        )
    log.laps.to_pandas.return_value = laps
    return log


class TestCheckChannelsAvailable:
    """Tests for _check_channels_available."""

    def test_all_present(self):
        log = _make_log_mock(["PPS", "BrakePress"])
        channel_names = {"throttle": "PPS", "brake": "BrakePress"}
        ok, missing = _check_channels_available(log, channel_names, ["throttle", "brake"])
        assert ok is True
        assert missing == []

    def test_missing_from_log(self):
        log = _make_log_mock(["PPS"])
        channel_names = {"throttle": "PPS", "brake": "BrakePress"}
        ok, missing = _check_channels_available(log, channel_names, ["throttle", "brake"])
        assert ok is False
        assert missing == ["brake"]

    def test_missing_from_channel_names(self):
        log = _make_log_mock(["PPS"])
        channel_names = {"throttle": "PPS"}
        ok, missing = _check_channels_available(log, channel_names, ["throttle", "lateral_g"])
        assert ok is False
        assert missing == ["lateral_g"]

    def test_empty_required(self):
        log = _make_log_mock()
        ok, missing = _check_channels_available(log, {}, [])
        assert ok is True
        assert missing == []


class TestCornerToInfo:
    """Tests for _corner_to_info."""

    def test_correct_field_mapping(self):
        corner = _make_corner(
            id=3,
            name="Turn 3",
            direction="R",
            start_dist=500.0,
            end_dist=600.0,
            apex_dist=550.0,
            max_curvature=0.02,
        )
        info = _corner_to_info(corner)
        assert isinstance(info, CornerInfo)
        assert info.id == 3
        assert info.name == "Turn 3"
        assert info.direction == "R"
        assert info.start_dist == 500.0
        assert info.end_dist == 600.0
        assert info.apex_dist == 550.0
        assert info.length == pytest.approx(100.0)
        assert info.radius == pytest.approx(50.0)

    def test_numpy_float_conversion(self):
        """Ensure numpy floats are converted to Python floats."""
        corner = _make_corner(start_dist=np.float64(100.0), end_dist=np.float64(200.0))
        info = _corner_to_info(corner)
        assert type(info.start_dist) is float
        assert type(info.end_dist) is float


class TestComputeBestLap:
    """Tests for _compute_best_lap."""

    def test_returns_none_when_no_exit_speeds(self):
        result = _compute_best_lap([], [], [], [], [])
        assert result is None

    def test_single_lap(self):
        result = _compute_best_lap(
            bp_vals=[(1, 450.0)],
            min_speed_vals=[(1, 80.0)],
            exit_speed_vals=[(1, 120.0)],
            tp_vals=[(1, 510.0)],
            ta_entries=[(1, 65.0)],
        )
        assert result is not None
        assert result.lap_num == 1
        assert result.selection_reason == "only lap with data"
        assert result.exit_speed == 120.0
        assert result.throttle_acceptance_pct == 65.0

    def test_selects_best_composite(self):
        # Lap 2 has the best exit speed, lap 3 has best min speed
        result = _compute_best_lap(
            bp_vals=[(1, 450.0), (2, 452.0), (3, 448.0)],
            min_speed_vals=[(1, 80.0), (2, 82.0), (3, 85.0)],
            exit_speed_vals=[(1, 120.0), (2, 128.0), (3, 118.0)],
            tp_vals=[(1, 510.0), (2, 508.0), (3, 512.0)],
            ta_entries=[(1, 65.0), (2, 60.0), (3, 70.0)],
        )
        assert result is not None
        # Lap 2 should win (highest exit speed, weighted 50%)
        assert result.lap_num == 2
        assert result.exit_speed == 128.0
        assert "exit_speed" in result.vs_mean

    def test_vs_mean_deltas(self):
        result = _compute_best_lap(
            bp_vals=[(1, 100.0), (2, 104.0)],
            min_speed_vals=[(1, 80.0), (2, 84.0)],
            exit_speed_vals=[(1, 120.0), (2, 130.0)],
            tp_vals=[],
            ta_entries=[],
        )
        assert result is not None
        # Lap 2 is better in all metrics
        assert result.vs_mean["exit_speed"] > 0
        assert result.vs_mean["min_speed"] > 0
        assert result.vs_mean["braking_point"] > 0


class TestExtractSuspensionSummary:
    """Tests for _extract_suspension_summary."""

    def _make_corner_velocity_data(self, **overrides):
        data = MagicMock()
        defaults = {
            "skew": 0.1,
            "kurtosis": 2.5,
            "mean": 5.0,
            "std": 15.0,
            "pct_friction": 20.0,
            "pct_slow_bump": 15.0,
            "pct_slow_rebound": 18.0,
            "pct_fast_bump": 10.0,
            "pct_fast_rebound": 12.0,
            "pct_curb": 2.0,
        }
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(data, k, v)
        return data

    def test_extracts_all_corners(self):
        result = MagicMock()
        result.front_left = self._make_corner_velocity_data(pct_friction=20.0)
        result.front_right = self._make_corner_velocity_data(pct_friction=22.0)
        result.rear_left = self._make_corner_velocity_data(pct_friction=18.0)
        result.rear_right = self._make_corner_velocity_data(pct_friction=19.0)

        summary = _extract_suspension_summary(result)
        assert isinstance(summary, SuspensionSummary)
        assert set(summary.per_wheel.keys()) == {"FL", "FR", "RL", "RR"}
        assert summary.per_wheel["FL"]["pct_friction"] == 20.0

    def test_symmetry_computation(self):
        result = MagicMock()
        result.front_left = self._make_corner_velocity_data(pct_friction=20.0, pct_slow_bump=15.0)
        result.front_right = self._make_corner_velocity_data(pct_friction=25.0, pct_slow_bump=10.0)
        result.rear_left = self._make_corner_velocity_data(pct_friction=18.0)
        result.rear_right = self._make_corner_velocity_data(pct_friction=18.0)

        summary = _extract_suspension_summary(result)
        assert summary.symmetry["front_lr_friction_diff"] == pytest.approx(-5.0)
        assert summary.symmetry["front_lr_bump_diff"] == pytest.approx(5.0)
        assert summary.symmetry["rear_lr_friction_diff"] == pytest.approx(0.0)


class TestExtractTireGripSummary:
    """Tests for _extract_tire_grip_summary."""

    def test_extracts_stats_and_units(self):
        result = MagicMock()
        result.metric_mode = "pressure"
        result.metric_unit = "bar"
        result.accel_unit = "g"

        for attr in ("front_left", "front_right", "rear_left", "rear_right"):
            corner = MagicMock()
            corner.mean_g = 1.2
            corner.std_g = 0.1
            corner.mean_metric = 2.0
            corner.std_metric = 0.05
            setattr(result, attr, corner)

        summary = _extract_tire_grip_summary(result)
        assert isinstance(summary, TireGripSummary)
        assert summary.metric_mode == "pressure"
        assert summary.units["metric_unit"] == "bar"
        assert summary.per_wheel["FL"]["mean_g"] == pytest.approx(1.2)


class TestGenerateSessionReport:
    """Integration tests for generate_session_report using mocked analysis functions."""

    def _make_channel_names(self):
        return {
            "throttle": "PPS",
            "brake": "BrakePress",
            "gps_speed": "GPS Speed",
            "gps_latitude": "GPS Latitude",
            "gps_longitude": "GPS Longitude",
            "lateral_g": "LateralAcc",
            "inline_g": "InlineAcc",
            "shock_fl": "LF_Shock_Pot",
            "shock_fr": "RF_Shock_Pot",
            "shock_rl": "LR_Shock_Pot",
            "shock_rr": "RR_Shock_Pot",
        }

    @patch("motorsports_data_notebook.report.get_profile_for_logger", return_value=None)
    @patch("motorsports_data_notebook.report.get_logger_id", return_value=None)
    def test_minimal_report_no_channels(self, mock_logger, mock_profile):
        """Report should work with just lap times when no channels are available."""
        log = _make_log_mock(channel_names_present=[])
        channel_names = self._make_channel_names()

        report = generate_session_report(log, channel_names, file_name="test.xrz")

        assert isinstance(report, SessionReport)
        assert report.metadata.file_name == "test.xrz"
        assert "lap_times" in report.available_analyses
        assert "corners" in report.skipped_analyses
        assert report.corners == []
        assert report.corner_consistency == []
        assert report.suspension is None
        assert report.tire_grip is None

    @patch("motorsports_data_notebook.report.get_profile_for_logger", return_value=None)
    @patch("motorsports_data_notebook.report.get_logger_id", return_value="12345")
    @patch("motorsports_data_notebook.report.identify_corners")
    def test_corners_detected(self, mock_corners, mock_logger, mock_profile):
        """Report should include corners when GPS channels are available."""
        corners = [_make_corner(id=1), _make_corner(id=2, name="Turn 2", start_dist=400.0)]
        mock_corners.return_value = corners

        log = _make_log_mock(channel_names_present=["GPS Latitude", "GPS Longitude"])

        # Mock the filter_by_lap chain
        dist_arr = np.linspace(0, 2000, 100)
        lat_arr = np.linspace(35.0, 35.1, 100)
        lon_arr = np.linspace(139.0, 139.1, 100)

        mock_table = MagicMock()

        def mock_column(name):
            if name == "GPS Latitude":
                return MagicMock(to_numpy=lambda: lat_arr)
            elif name == "GPS Longitude":
                return MagicMock(to_numpy=lambda: lon_arr)
            elif name == "distance_m":
                return MagicMock(to_numpy=lambda: dist_arr)
            elif name == "timecodes":
                return MagicMock(to_numpy=lambda: np.linspace(0, 10000, 100))
            return MagicMock(to_numpy=lambda: np.zeros(100))

        # Set up all channel mocks to return column data
        channels_dict = {}
        for ch_name in ["GPS Latitude", "GPS Longitude", "distance_m"]:
            ch_table = MagicMock()
            ch_table.column = mock_column
            channels_dict[ch_name] = ch_table

        mock_resampled = MagicMock()
        mock_resampled.channels = channels_dict
        mock_selected = MagicMock()
        mock_selected.resample_to_channel.return_value = mock_resampled
        mock_filtered = MagicMock()
        mock_filtered.select_channels.return_value = mock_selected
        log.filter_by_lap.return_value = mock_filtered

        channel_names = self._make_channel_names()
        report = generate_session_report(log, channel_names)

        assert "corners" in report.available_analyses
        assert len(report.corners) == 2
        assert report.track_length_m == pytest.approx(dist_arr[-1])

    @patch("motorsports_data_notebook.report.get_profile_for_logger", return_value=None)
    @patch("motorsports_data_notebook.report.get_logger_id", return_value=None)
    def test_skipped_analyses_recorded(self, mock_logger, mock_profile):
        """All skipped analyses should have explanations."""
        log = _make_log_mock(channel_names_present=[])
        channel_names = self._make_channel_names()

        report = generate_session_report(log, channel_names)

        assert "corners" in report.skipped_analyses
        assert "suspension" in report.skipped_analyses
        assert "tire_grip" in report.skipped_analyses

    @patch("motorsports_data_notebook.report.get_profile_for_logger", return_value=None)
    @patch("motorsports_data_notebook.report.get_logger_id", return_value=None)
    def test_lap_time_summary(self, mock_logger, mock_profile):
        """Lap time summary should correctly compute stats."""
        laps = pd.DataFrame(
            {
                "num": [0, 1, 2, 3, 4],
                "start_time": [0, 100, 200, 300, 400],
                "end_time": [100, 200, 300, 400, 500],
                "lap_time": pd.to_timedelta([120, 90, 91, 92, 95], unit="s"),
            }
        )
        log = _make_log_mock(channel_names_present=[], laps=laps)
        channel_names = self._make_channel_names()

        report = generate_session_report(log, channel_names, top_lap_threshold=1.05)

        assert report.lap_times.best_lap_num == 1
        assert report.lap_times.best_lap_time_s == pytest.approx(90.0)
        assert report.metadata.total_laps == 5
        assert report.metadata.valid_laps == 4  # excludes lap 0
        assert len(report.lap_times.all_lap_times) == 4  # excludes lap 0


class TestSessionReportSerialization:
    """Tests for to_dict() and to_json() methods."""

    def _make_minimal_report(self):
        return SessionReport(
            metadata=SessionMetadata(
                file_name="test.xrz",
                logger_id="12345",
                vehicle_profile="Test Car",
                total_laps=5,
                valid_laps=4,
                top_lap_count=3,
            ),
            lap_times=LapTimeSummary(
                best_lap_num=2,
                best_lap_time_s=90.5,
                top_laps=[{"num": 2, "lap_time_s": 90.5}],
                mean_top_lap_time_s=91.0,
                std_top_lap_time_s=0.5,
                all_lap_times=[{"num": 1, "lap_time_s": 95.0}, {"num": 2, "lap_time_s": 90.5}],
            ),
            corners=[
                CornerInfo(
                    id=1,
                    name="Turn 1",
                    direction="L",
                    start_dist=100.0,
                    end_dist=200.0,
                    apex_dist=150.0,
                    length=100.0,
                    radius=50.0,
                )
            ],
            corner_consistency=[
                CornerConsistency(
                    corner=CornerInfo(
                        id=1,
                        name="Turn 1",
                        direction="L",
                        start_dist=100.0,
                        end_dist=200.0,
                        apex_dist=150.0,
                        length=100.0,
                        radius=50.0,
                    ),
                    ta_mean=65.0,
                    ta_std=3.0,
                    bp_mean=450.0,
                    bp_std=2.5,
                    min_speed_mean=80.0,
                    min_speed_std=1.5,
                    exit_speed_mean=120.0,
                    exit_speed_std=3.0,
                    accel_zone_length=150.0,
                    opportunity_score=450.0,
                    best_lap=CornerBestLap(
                        lap_num=2,
                        selection_reason="fastest exit speed",
                        braking_point=452.0,
                        min_speed=81.0,
                        exit_speed=123.0,
                        throttle_acceptance_pct=62.0,
                        throttle_point=510.0,
                        vs_mean={"exit_speed": 3.0, "min_speed": 1.0},
                    ),
                )
            ],
            track_length_m=4500.0,
            suspension=SuspensionSummary(
                per_wheel={"FL": {"pct_friction": 20.0}, "FR": {"pct_friction": 22.0}},
                symmetry={"front_lr_friction_diff": -2.0},
            ),
            tire_grip=TireGripSummary(
                metric_mode="pressure",
                units={"metric_unit": "bar", "accel_unit": "g"},
                per_wheel={"FL": {"mean_g": 1.2}},
            ),
            available_analyses=["lap_times", "corners", "zones"],
            skipped_analyses={"suspension": "Missing channels: shock_fl"},
        )

    def test_to_json_produces_valid_json(self):
        report = self._make_minimal_report()
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["metadata"]["file_name"] == "test.xrz"

    def test_to_dict_round_trips(self):
        report = self._make_minimal_report()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["track_length_m"] == 4500.0
        assert d["corner_consistency"][0]["best_lap"]["lap_num"] == 2

    def test_to_json_with_none_fields(self):
        report = self._make_minimal_report()
        report.suspension = None
        report.tire_grip = None
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["suspension"] is None
        assert parsed["tire_grip"] is None


class TestBrakingMetrics:
    """Tests for new braking fields on CornerConsistency and CornerBestLap."""

    def _make_corner_info(self):
        return CornerInfo(
            id=1,
            name="Turn 1",
            direction="L",
            start_dist=100.0,
            end_dist=200.0,
            apex_dist=150.0,
            length=100.0,
            radius=50.0,
        )

    def test_corner_consistency_braking_fields_serialize(self):
        """Test that CornerConsistency with braking fields serializes correctly."""
        cc = CornerConsistency(
            corner=self._make_corner_info(),
            ta_mean=65.0,
            ta_std=3.0,
            bp_mean=450.0,
            bp_std=2.5,
            min_speed_mean=80.0,
            min_speed_std=1.5,
            exit_speed_mean=120.0,
            exit_speed_std=3.0,
            accel_zone_length=150.0,
            opportunity_score=450.0,
            best_lap=None,
            peak_brake_mean=55.0,
            peak_brake_std=2.0,
            entry_speed_mean=195.0,
            entry_speed_std=3.5,
            brake_release_mean=180.0,
            brake_release_std=5.0,
            braking_distance_mean=80.0,
            braking_distance_std=4.0,
            mean_decel_g_mean=1.2,
            mean_decel_g_std=0.1,
        )

        d = cc.__dict__
        assert d["peak_brake_mean"] == 55.0
        assert d["entry_speed_mean"] == 195.0
        assert d["brake_release_mean"] == 180.0
        assert d["braking_distance_mean"] == 80.0
        assert d["mean_decel_g_mean"] == 1.2

    def test_corner_best_lap_braking_fields(self):
        """Test CornerBestLap with new braking fields."""
        best = CornerBestLap(
            lap_num=2,
            selection_reason="fastest exit speed",
            braking_point=452.0,
            min_speed=81.0,
            exit_speed=123.0,
            throttle_acceptance_pct=62.0,
            throttle_point=510.0,
            vs_mean={"exit_speed": 3.0},
            peak_brake=58.0,
            entry_speed=198.0,
            brake_release_point=175.0,
            braking_distance=77.0,
            mean_decel_g=1.15,
        )

        assert best.peak_brake == 58.0
        assert best.entry_speed == 198.0
        assert best.brake_release_point == 175.0
        assert best.braking_distance == 77.0
        assert best.mean_decel_g == 1.15

    def test_compute_best_lap_includes_braking_vs_mean(self):
        """Test _compute_best_lap includes new braking metrics in vs_mean."""
        result = _compute_best_lap(
            bp_vals=[(1, 450.0), (2, 452.0)],
            min_speed_vals=[(1, 80.0), (2, 82.0)],
            exit_speed_vals=[(1, 120.0), (2, 128.0)],
            tp_vals=[],
            ta_entries=[],
            peak_brake_vals=[(1, 55.0), (2, 60.0)],
            entry_speed_vals=[(1, 195.0), (2, 200.0)],
            brake_release_vals=[(1, 180.0), (2, 175.0)],
            braking_distance_vals=[(1, 80.0), (2, 75.0)],
            mean_decel_g_vals=[(1, 1.1), (2, 1.3)],
        )

        assert result is not None
        assert result.peak_brake is not None
        assert result.entry_speed is not None
        assert result.brake_release_point is not None
        assert result.braking_distance is not None
        assert result.mean_decel_g is not None
        # vs_mean should include the new braking metrics
        assert "peak_brake" in result.vs_mean
        assert "entry_speed" in result.vs_mean


class TestBrakingBalance:
    """Tests for BrakingBalanceSummary dataclass and serialization."""

    def test_create_braking_balance_summary(self):
        """Test creating a BrakingBalanceSummary with test data."""
        summary = BrakingBalanceSummary(
            available=True,
            front_channel="BrakePress",
            rear_channel="BrakePressRear",
            per_corner=[
                {
                    "corner_id": 1,
                    "corner_name": "Turn 1",
                    "balance_pct_mean": 62.5,
                    "balance_pct_std": 1.2,
                    "n_samples": 3,
                }
            ],
            overall_balance_pct=62.5,
            overall_balance_std=1.2,
        )

        assert summary.available is True
        assert summary.front_channel == "BrakePress"
        assert summary.overall_balance_pct == 62.5

    def test_session_report_with_braking_balance(self):
        """Test SessionReport serialization with braking_balance populated."""
        report = SessionReport(
            metadata=SessionMetadata(
                file_name="test.xrz",
                logger_id="12345",
                vehicle_profile="Test Car",
                total_laps=5,
                valid_laps=4,
                top_lap_count=3,
            ),
            lap_times=LapTimeSummary(
                best_lap_num=2,
                best_lap_time_s=90.5,
                top_laps=[],
                mean_top_lap_time_s=91.0,
                std_top_lap_time_s=0.5,
                all_lap_times=[],
            ),
            corners=[],
            corner_consistency=[],
            track_length_m=4500.0,
            suspension=None,
            tire_grip=None,
            available_analyses=["lap_times"],
            skipped_analyses={},
            braking_balance=BrakingBalanceSummary(
                available=True,
                front_channel="BrakePress",
                rear_channel="BrakePressRear",
                per_corner=[],
                overall_balance_pct=60.0,
                overall_balance_std=2.0,
            ),
        )

        d = report.to_dict()
        assert d["braking_balance"]["available"] is True
        assert d["braking_balance"]["overall_balance_pct"] == 60.0

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["braking_balance"]["front_channel"] == "BrakePress"

    def test_session_report_braking_balance_none(self):
        """Test SessionReport serialization with braking_balance=None."""
        report = SessionReport(
            metadata=SessionMetadata(
                file_name="test.xrz",
                logger_id=None,
                vehicle_profile=None,
                total_laps=3,
                valid_laps=2,
                top_lap_count=2,
            ),
            lap_times=LapTimeSummary(
                best_lap_num=1,
                best_lap_time_s=95.0,
                top_laps=[],
                mean_top_lap_time_s=95.0,
                std_top_lap_time_s=0.0,
                all_lap_times=[],
            ),
            corners=[],
            corner_consistency=[],
            track_length_m=0.0,
            suspension=None,
            tire_grip=None,
            available_analyses=["lap_times"],
            skipped_analyses={},
            braking_balance=None,
        )

        d = report.to_dict()
        assert d["braking_balance"] is None

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["braking_balance"] is None
