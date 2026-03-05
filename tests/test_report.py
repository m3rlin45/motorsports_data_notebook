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
    CornerLapMetrics,
    LapTimeSummary,
    SessionMetadata,
    SessionReport,
    SuspensionSummary,
    TireGripSummary,
    _check_channels_available,
    _collate_per_lap_metrics,
    _compute_best_lap,
    _corner_to_info,
    _detect_off_track_laps,
    _extract_suspension_summary,
    _extract_tire_grip_summary,
    _make_session_id,
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
                best_lap_time_fmt="1:30.500",
                top_laps=[{"num": 2, "lap_time_s": 90.5}],
                mean_top_lap_time_s=91.0,
                mean_top_lap_time_fmt="1:31.000",
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
                best_lap_time_fmt="1:30.500",
                top_laps=[],
                mean_top_lap_time_s=91.0,
                mean_top_lap_time_fmt="1:31.000",
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
                best_lap_time_fmt="1:35.000",
                top_laps=[],
                mean_top_lap_time_s=95.0,
                mean_top_lap_time_fmt="1:35.000",
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

    def test_brake_balance_filtering_excludes_low_brake_corners(self):
        """Low-brake corners (< 20% of session max) are excluded from balance stats."""
        # Corner 1: high brake (100 bar total) -> 60% front bias
        # Corner 2: high brake (90 bar total) -> 70% front bias
        # Corner 3: low brake (15 bar total, < 20% of 100) -> 50% front bias (noise)
        per_corner_data = [
            (
                {
                    "corner_id": 1,
                    "corner_name": "Turn 1",
                    "balance_pct_mean": 60.0,
                    "balance_pct_std": 1.0,
                    "n_samples": 5,
                    "peak_brake_mean": 100.0,
                },
                [59.0, 60.0, 61.0, 60.0, 60.0],
            ),
            (
                {
                    "corner_id": 2,
                    "corner_name": "Turn 2",
                    "balance_pct_mean": 70.0,
                    "balance_pct_std": 1.5,
                    "n_samples": 5,
                    "peak_brake_mean": 90.0,
                },
                [69.0, 70.0, 71.0, 70.0, 70.0],
            ),
            (
                {
                    "corner_id": 3,
                    "corner_name": "Turn 3",
                    "balance_pct_mean": 50.0,
                    "balance_pct_std": 5.0,
                    "n_samples": 5,
                    "peak_brake_mean": 15.0,
                },
                [45.0, 50.0, 55.0, 50.0, 50.0],
            ),
        ]

        # Reproduce the filtering logic from report.py
        session_max_peak = max(e["peak_brake_mean"] for e, _ in per_corner_data)
        assert session_max_peak == 100.0

        brake_threshold = session_max_peak * 0.2
        assert brake_threshold == 20.0

        filtered_pcts: list[float] = []
        filtered_corners: list[dict] = []
        for entry, pcts in per_corner_data:
            if entry["peak_brake_mean"] >= brake_threshold:
                filtered_corners.append(entry)
                filtered_pcts.extend(pcts)

        # Corner 3 (15 bar) should be excluded
        assert len(filtered_corners) == 2
        assert all(c["corner_id"] in (1, 2) for c in filtered_corners)

        # Overall stats computed from filtered corners only (no 50% noise)
        overall_mean = float(np.mean(filtered_pcts))
        assert 60.0 < overall_mean < 70.0  # weighted mean of 60% and 70% groups

        summary = BrakingBalanceSummary(
            available=True,
            front_channel="BrakePress",
            rear_channel="BrakePressRear",
            per_corner=filtered_corners,
            overall_balance_pct=round(overall_mean, 1),
            overall_balance_std=round(float(np.std(filtered_pcts)), 1),
            min_brake_threshold=round(brake_threshold, 1),
        )
        assert summary.min_brake_threshold == 20.0
        assert len(summary.per_corner) == 2

    def test_brake_balance_filtering_keeps_all_when_all_high(self):
        """All corners kept when all have significant brake pressure."""
        per_corner_data = [
            (
                {
                    "corner_id": 1,
                    "corner_name": "Turn 1",
                    "balance_pct_mean": 65.0,
                    "balance_pct_std": 1.0,
                    "n_samples": 3,
                    "peak_brake_mean": 80.0,
                },
                [64.0, 65.0, 66.0],
            ),
            (
                {
                    "corner_id": 2,
                    "corner_name": "Turn 2",
                    "balance_pct_mean": 62.0,
                    "balance_pct_std": 1.0,
                    "n_samples": 3,
                    "peak_brake_mean": 75.0,
                },
                [61.0, 62.0, 63.0],
            ),
        ]

        session_max_peak = max(e["peak_brake_mean"] for e, _ in per_corner_data)
        brake_threshold = session_max_peak * 0.2  # 16.0

        filtered_corners = [
            e for e, _ in per_corner_data if e["peak_brake_mean"] >= brake_threshold
        ]
        assert len(filtered_corners) == 2

    def test_brake_balance_threshold_serialization(self):
        """min_brake_threshold round-trips through JSON."""
        summary = BrakingBalanceSummary(
            available=True,
            front_channel="BrakeF",
            rear_channel="BrakeR",
            per_corner=[
                {
                    "corner_id": 1,
                    "corner_name": "Turn 1",
                    "balance_pct_mean": 60.0,
                    "balance_pct_std": 1.0,
                    "n_samples": 3,
                    "peak_brake_mean": 100.0,
                }
            ],
            overall_balance_pct=60.0,
            overall_balance_std=1.0,
            min_brake_threshold=20.0,
        )

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
                best_lap_time_s=90.0,
                best_lap_time_fmt="1:30.000",
                top_laps=[],
                mean_top_lap_time_s=90.0,
                mean_top_lap_time_fmt="1:30.000",
                std_top_lap_time_s=0.0,
                all_lap_times=[],
            ),
            corners=[],
            corner_consistency=[],
            track_length_m=4000.0,
            suspension=None,
            tire_grip=None,
            available_analyses=["braking_balance"],
            skipped_analyses={},
            braking_balance=summary,
        )

        parsed = json.loads(report.to_json())
        assert parsed["braking_balance"]["min_brake_threshold"] == 20.0


class TestDetectOffTrackLaps:
    """Tests for _detect_off_track_laps GPS deviation detection."""

    @staticmethod
    def _make_gps_data(
        n_laps: int,
        corner_start: float,
        corner_end: float,
        *,
        offsets: dict[int, float] | None = None,
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Build synthetic per-lap GPS data.

        Creates a straight east-west track line. Each lap has the same path
        unless an offset (in meters, north-south) is specified for that lap.
        """
        offsets = offsets or {}
        # Track line: 0 to 500m, heading east at lat=35.0, lon=136.0
        base_lat = 35.0
        base_lon = 136.0
        R = 6371000.0

        per_lap: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for lap in range(1, n_laps + 1):
            dist = np.arange(0.0, 500.0, 1.0)
            lat = np.full_like(dist, base_lat)
            # Convert distance to longitude delta
            lon = base_lon + np.degrees(dist / (R * np.cos(np.radians(base_lat))))

            # Apply north-south offset for this lap (shift lat)
            if lap in offsets:
                offset_m = offsets[lap]
                lat = lat + np.degrees(offset_m / R)

            per_lap[lap] = (dist, lat, lon)

        return per_lap

    def test_no_off_track_when_all_laps_on_path(self):
        """All laps on the same line -> no off-track detections."""
        corner = _make_corner(id=1, start_dist=100.0, end_dist=200.0)
        gps = self._make_gps_data(5, 100.0, 200.0)
        result = _detect_off_track_laps(gps, [corner])
        assert result == {}

    def test_detects_large_deviation(self):
        """One lap offset 20m from the path is flagged."""
        corner = _make_corner(id=1, start_dist=100.0, end_dist=200.0)
        gps = self._make_gps_data(5, 100.0, 200.0, offsets={3: 20.0})
        result = _detect_off_track_laps(gps, [corner])
        assert 1 in result
        assert 3 in result[1]

    def test_small_deviation_not_flagged(self):
        """5m offset is within the default 10m threshold."""
        corner = _make_corner(id=1, start_dist=100.0, end_dist=200.0)
        gps = self._make_gps_data(5, 100.0, 200.0, offsets={3: 5.0})
        result = _detect_off_track_laps(gps, [corner])
        assert result == {}

    def test_custom_threshold(self):
        """8m offset detected with a tighter 6m threshold."""
        corner = _make_corner(id=1, start_dist=100.0, end_dist=200.0)
        gps = self._make_gps_data(5, 100.0, 200.0, offsets={2: 8.0})
        result = _detect_off_track_laps(gps, [corner], deviation_threshold=6.0)
        assert 1 in result
        assert 2 in result[1]

    def test_multiple_corners(self):
        """Off-track detection works independently per corner."""
        corner1 = _make_corner(id=1, start_dist=50.0, end_dist=120.0)
        corner2 = _make_corner(id=2, start_dist=300.0, end_dist=400.0)

        # Lap 2: offset only in corner 1 region, lap 4: offset everywhere
        gps = self._make_gps_data(5, 50.0, 400.0)
        # Manually shift lap 2 only in the corner 1 region
        dist2, lat2, lon2 = gps[2]
        R = 6371000.0
        mask_c1 = (dist2 >= 0.0) & (dist2 <= 170.0)  # corner1 + margin
        lat2_mod = lat2.copy()
        lat2_mod[mask_c1] += np.degrees(25.0 / R)
        gps[2] = (dist2, lat2_mod, lon2)

        # Shift lap 4 everywhere (affects both corners)
        dist4, lat4, lon4 = gps[4]
        lat4_shifted = lat4 + np.degrees(25.0 / R)
        gps[4] = (dist4, lat4_shifted, lon4)

        result = _detect_off_track_laps(gps, [corner1, corner2])

        # Corner 1: laps 2 and 4 off-track
        assert 1 in result
        assert {2, 4}.issubset(result[1])

        # Corner 2: only lap 4 off-track
        assert 2 in result
        assert 4 in result[2]
        assert 2 not in result[2]

    def test_too_few_laps_skips_corner(self):
        """Need at least 3 laps for median reference; 2 laps -> skip."""
        corner = _make_corner(id=1, start_dist=100.0, end_dist=200.0)
        gps = self._make_gps_data(2, 100.0, 200.0, offsets={2: 30.0})
        result = _detect_off_track_laps(gps, [corner])
        assert result == {}

    def test_corner_outside_gps_range_skipped(self):
        """Corner beyond GPS distance range produces no detections."""
        corner = _make_corner(id=1, start_dist=600.0, end_dist=700.0)
        gps = self._make_gps_data(5, 600.0, 700.0)  # GPS only covers 0-499m
        result = _detect_off_track_laps(gps, [corner])
        # Not enough interpolation points in-range -> skipped
        assert result == {}

    def test_empty_inputs(self):
        """Empty GPS data or empty corners list -> empty result."""
        corner = _make_corner(id=1, start_dist=100.0, end_dist=200.0)
        assert _detect_off_track_laps({}, [corner]) == {}
        gps = self._make_gps_data(5, 100.0, 200.0)
        assert _detect_off_track_laps(gps, []) == {}

    def test_negative_offset_detected(self):
        """Negative (south) offset is also detected."""
        corner = _make_corner(id=1, start_dist=100.0, end_dist=200.0)
        gps = self._make_gps_data(5, 100.0, 200.0, offsets={1: -15.0})
        result = _detect_off_track_laps(gps, [corner])
        assert 1 in result
        assert 1 in result[1]

    def test_majority_off_track_flags_minority(self):
        """When 3 of 5 laps are offset, the 2 normal laps get flagged instead.

        The median path follows the majority, so the outlier detection
        is relative — this tests that the function correctly identifies
        laps deviating from the median regardless of which group is larger.
        """
        corner = _make_corner(id=1, start_dist=100.0, end_dist=200.0)
        # Offset 3 laps by 20m; the 2 "normal" laps become the outliers
        gps = self._make_gps_data(5, 100.0, 200.0, offsets={1: 20.0, 2: 20.0, 3: 20.0})
        result = _detect_off_track_laps(gps, [corner])
        assert 1 in result
        # Laps 4,5 are now deviating from the median (which follows 1,2,3)
        assert {4, 5}.issubset(result[1])


class TestMakeSessionId:
    """Tests for _make_session_id."""

    def test_standard_date_time(self):
        assert _make_session_id("01/12/2026", "13:00:05", 1) == "2026-01-12_1300_s01"

    def test_session_num_padding(self):
        assert _make_session_id("12/05/2025", "09:30:00", 3) == "2025-12-05_0930_s03"

    def test_single_digit_month_day(self):
        assert _make_session_id("1/5/2026", "8:05:00", 2) == "2026-01-05_0805_s02"

    def test_none_date(self):
        assert _make_session_id(None, "13:00:00", 1) is None

    def test_none_time(self):
        assert _make_session_id("01/12/2026", None, 1) is None

    def test_both_none(self):
        assert _make_session_id(None, None, 1) is None

    def test_empty_strings(self):
        assert _make_session_id("", "13:00:00", 1) is None
        assert _make_session_id("01/12/2026", "", 1) is None

    def test_malformed_date(self):
        assert _make_session_id("2026-01-12", "13:00:00", 1) is None


class TestCornerLapMetrics:
    """Tests for CornerLapMetrics dataclass."""

    def test_serialization_round_trip(self):
        """CornerLapMetrics should round-trip through dict/JSON."""
        from dataclasses import asdict

        m = CornerLapMetrics(
            lap_num=5,
            braking_point=450.0,
            entry_speed=195.0,
            min_speed=80.0,
            exit_speed=120.0,
            throttle_point=510.0,
            throttle_acceptance_pct=65.0,
            peak_brake=55.0,
            brake_release_point=180.0,
            braking_distance=80.0,
            mean_decel_g=1.2,
        )
        d = asdict(m)
        assert d["lap_num"] == 5
        assert d["exit_speed"] == 120.0
        assert d["peak_brake"] == 55.0

        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["lap_num"] == 5
        assert parsed["braking_point"] == 450.0

    def test_none_fields(self):
        """CornerLapMetrics with None fields serializes correctly."""
        from dataclasses import asdict

        m = CornerLapMetrics(
            lap_num=3,
            braking_point=None,
            entry_speed=None,
            min_speed=80.0,
            exit_speed=120.0,
            throttle_point=None,
            throttle_acceptance_pct=None,
            peak_brake=None,
            brake_release_point=None,
            braking_distance=None,
            mean_decel_g=None,
        )
        d = asdict(m)
        assert d["braking_point"] is None
        assert d["min_speed"] == 80.0

        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["braking_point"] is None


class TestCollatePerLapMetrics:
    """Tests for _collate_per_lap_metrics."""

    def test_basic_collation(self):
        result = _collate_per_lap_metrics(
            bp_vals=[(1, 450.0), (2, 452.0)],
            entry_speed_vals=[(1, 195.0), (2, 200.0)],
            min_speed_vals=[(1, 80.0), (2, 82.0)],
            exit_speed_vals=[(1, 120.0), (2, 128.0)],
            tp_vals=[(1, 510.0)],
            ta_entries=[(1, 65.0)],
            peak_brake_vals=[(1, 55.0), (2, 60.0)],
            brake_release_vals=[(2, 175.0)],
            braking_distance_vals=[(1, 80.0), (2, 75.0)],
            mean_decel_g_vals=[(1, 1.1)],
        )
        assert len(result) == 2
        assert result[0].lap_num == 1
        assert result[0].braking_point == 450.0
        assert result[0].throttle_point == 510.0
        assert result[0].throttle_acceptance_pct == 65.0
        assert result[0].brake_release_point is None  # not in lap 1

        assert result[1].lap_num == 2
        assert result[1].exit_speed == 128.0
        assert result[1].throttle_point is None  # not in lap 2
        assert result[1].brake_release_point == 175.0
        assert result[1].mean_decel_g is None  # not in lap 2

    def test_empty_inputs(self):
        result = _collate_per_lap_metrics(
            bp_vals=[],
            entry_speed_vals=[],
            min_speed_vals=[],
            exit_speed_vals=[],
            tp_vals=[],
            ta_entries=[],
            peak_brake_vals=[],
            brake_release_vals=[],
            braking_distance_vals=[],
            mean_decel_g_vals=[],
        )
        assert result == []

    def test_sorted_by_lap_num(self):
        result = _collate_per_lap_metrics(
            bp_vals=[(5, 1.0), (2, 2.0), (8, 3.0)],
            entry_speed_vals=[],
            min_speed_vals=[],
            exit_speed_vals=[],
            tp_vals=[],
            ta_entries=[],
            peak_brake_vals=[],
            brake_release_vals=[],
            braking_distance_vals=[],
            mean_decel_g_vals=[],
        )
        assert [m.lap_num for m in result] == [2, 5, 8]


class TestPerLapMetricsInCornerConsistency:
    """Tests that per_lap_metrics appears in CornerConsistency serialized output."""

    def test_per_lap_metrics_serializes(self):
        from dataclasses import asdict

        cc = CornerConsistency(
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
            best_lap=None,
            per_lap_metrics=[
                CornerLapMetrics(
                    lap_num=1,
                    braking_point=450.0,
                    entry_speed=195.0,
                    min_speed=80.0,
                    exit_speed=120.0,
                    throttle_point=510.0,
                    throttle_acceptance_pct=65.0,
                    peak_brake=55.0,
                    brake_release_point=180.0,
                    braking_distance=80.0,
                    mean_decel_g=1.2,
                ),
            ],
        )
        d = asdict(cc)
        assert "per_lap_metrics" in d
        assert len(d["per_lap_metrics"]) == 1
        assert d["per_lap_metrics"][0]["lap_num"] == 1
        assert d["per_lap_metrics"][0]["exit_speed"] == 120.0

        # JSON round-trip
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["per_lap_metrics"][0]["braking_point"] == 450.0


class TestSessionIdInMetadata:
    """Tests that session_id appears in SessionMetadata."""

    def test_session_id_in_serialized_output(self):
        from dataclasses import asdict

        meta = SessionMetadata(
            file_name="test.xrz",
            logger_id="12345",
            vehicle_profile="Test Car",
            total_laps=5,
            valid_laps=4,
            top_lap_count=3,
            driver="CMD",
            vehicle="Inferno 86",
            venue="Sodegaura",
            log_date="01/12/2026",
            log_time="13:00:00",
            session_id="2026-01-12_1300_s01",
        )
        d = asdict(meta)
        assert d["session_id"] == "2026-01-12_1300_s01"

        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["session_id"] == "2026-01-12_1300_s01"

    def test_session_id_none_by_default(self):
        from dataclasses import asdict

        meta = SessionMetadata(
            file_name="test.xrz",
            logger_id=None,
            vehicle_profile=None,
            total_laps=3,
            valid_laps=2,
            top_lap_count=2,
            driver=None,
            vehicle=None,
            venue=None,
            log_date=None,
            log_time=None,
        )
        d = asdict(meta)
        assert d["session_id"] is None
