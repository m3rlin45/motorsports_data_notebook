"""Tests for the session_runner module."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from motorsports_data_notebook.session_runner import (
    SessionAnalysisResult,
    _parse_aim_datetime,
    group_session_files,
    load_and_merge_sessions,
    resolve_profile,
)

# ── _parse_aim_datetime ──────────────────────────────────────────────────────


class TestParseAimDatetime:
    def test_valid(self):
        result = _parse_aim_datetime("01/15/2026", "13:30:00")
        assert result == datetime(2026, 1, 15, 13, 30, 0)

    def test_empty_date(self):
        assert _parse_aim_datetime("", "13:30:00") is None

    def test_empty_time(self):
        assert _parse_aim_datetime("01/15/2026", "") is None

    def test_both_empty(self):
        assert _parse_aim_datetime("", "") is None

    def test_invalid_format(self):
        assert _parse_aim_datetime("2026-01-15", "13:30:00") is None

    def test_garbage(self):
        assert _parse_aim_datetime("not-a-date", "not-a-time") is None


# ── resolve_profile ──────────────────────────────────────────────────────────


class TestResolveProfile:
    def test_auto_detect_no_profile(self):
        """When no profile matches, returns DEFAULT_CHANNEL_NAMES."""
        log = MagicMock()
        log.metadata = {}
        with patch("motorsports_data_notebook.session_runner.get_logger_id", return_value=None):
            channel_names, motion_ratios = resolve_profile(log)
        assert "gps_speed" in channel_names
        assert motion_ratios is None

    def test_explicit_profile_not_found(self):
        """Raises ValueError for unknown profile name."""
        log = MagicMock()
        with pytest.raises(ValueError, match="not found"):
            resolve_profile(log, profile_name="nonexistent_profile_xyz")

    def test_explicit_profile_found(self):
        """Returns profile channel names when found."""
        log = MagicMock()
        mock_profile = MagicMock()
        mock_profile.channel_names = {"throttle": "TPS", "gps_speed": "GPS_Speed"}
        mock_profile.motion_ratios = MagicMock()
        with (
            patch(
                "motorsports_data_notebook.session_runner.load_builtin_profiles",
                return_value={"test_profile": mock_profile},
            ),
            patch(
                "motorsports_data_notebook.session_runner.load_user_profiles",
                return_value={},
            ),
        ):
            channel_names, motion_ratios = resolve_profile(log, profile_name="test_profile")
        assert channel_names == {"throttle": "TPS", "gps_speed": "GPS_Speed"}
        assert motion_ratios is mock_profile.motion_ratios

    def test_auto_detect_with_profile(self):
        """Auto-detects profile from logger ID."""
        log = MagicMock()
        mock_profile = MagicMock()
        mock_profile.channel_names = {"throttle": "PPS"}
        mock_profile.motion_ratios = None
        with (
            patch("motorsports_data_notebook.session_runner.get_logger_id", return_value="12345"),
            patch(
                "motorsports_data_notebook.session_runner.get_profile_for_logger",
                return_value=mock_profile,
            ),
        ):
            channel_names, motion_ratios = resolve_profile(log)
        assert channel_names == {"throttle": "PPS"}


# ── load_and_merge_sessions ──────────────────────────────────────────────────


class TestLoadAndMergeSessions:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_and_merge_sessions([str(tmp_path / "nonexistent.xrk")])

    def test_single_file(self, tmp_path):
        """Single file returns the log directly."""
        fake_file = tmp_path / "test.xrk"
        fake_file.touch()
        mock_log = MagicMock()
        with patch("motorsports_data_notebook.session_runner.load_session", return_value=mock_log):
            log, path = load_and_merge_sessions([str(fake_file)])
        assert log is mock_log
        assert path == fake_file

    def test_multiple_files_merges(self, tmp_path):
        """Multiple files creates a MergedLogFile."""
        f1 = tmp_path / "a.xrk"
        f2 = tmp_path / "b.xrk"
        f1.touch()
        f2.touch()
        mock_log1 = MagicMock()
        mock_log2 = MagicMock()
        mock_merged = MagicMock()
        mock_merged.laps = pa.table({"num": [1, 2, 3]})
        with (
            patch(
                "motorsports_data_notebook.session_runner.load_session",
                side_effect=[mock_log1, mock_log2],
            ),
            patch(
                "motorsports_data_notebook.session_runner.MergedLogFile",
                return_value=mock_merged,
            ) as merge_cls,
        ):
            log, path = load_and_merge_sessions([str(f1), str(f2)])
        merge_cls.assert_called_once_with([mock_log1, mock_log2])
        assert log is mock_merged
        assert path == f1


# ── group_session_files ──────────────────────────────────────────────────────


def _make_mock_log(driver="CMD", venue="Fuji", date="01/15/2026", time="08:00:00", laps=5):
    """Create a mock log with metadata for group_session_files tests."""
    log = MagicMock()
    log.metadata = {"Driver": driver, "Venue": venue, "Log Date": date, "Log Time": time}
    log.laps = pa.table({"num": list(range(1, laps + 1))})
    return log


class TestGroupSessionFiles:
    def test_empty_list(self):
        assert group_session_files([]) == []

    def test_single_file(self):
        mock_log = _make_mock_log()
        with patch("motorsports_data_notebook.session_runner.load_session", return_value=mock_log):
            groups = group_session_files(["file1.xrk"])
        assert len(groups) == 1
        assert len(groups[0]["files"]) == 1
        assert groups[0]["driver"] == "CMD"
        assert groups[0]["venue"] == "Fuji"

    def test_merge_close_files(self):
        """Files within gap threshold are merged."""
        log1 = _make_mock_log(time="08:00:00", laps=5)
        log2 = _make_mock_log(time="08:20:00", laps=3)
        with patch(
            "motorsports_data_notebook.session_runner.load_session", side_effect=[log1, log2]
        ):
            groups = group_session_files(["a.xrk", "b.xrk"], max_gap_minutes=30)
        assert len(groups) == 1
        assert len(groups[0]["files"]) == 2
        assert groups[0]["gap_minutes"] == [20.0]

    def test_split_on_large_gap(self):
        """Files beyond gap threshold are separate sessions."""
        log1 = _make_mock_log(time="08:00:00")
        log2 = _make_mock_log(time="10:00:00")
        with patch(
            "motorsports_data_notebook.session_runner.load_session", side_effect=[log1, log2]
        ):
            groups = group_session_files(["a.xrk", "b.xrk"], max_gap_minutes=30)
        assert len(groups) == 2

    def test_split_on_different_driver(self):
        """Different drivers are separate sessions even if close in time."""
        log1 = _make_mock_log(driver="CMD", time="08:00:00")
        log2 = _make_mock_log(driver="SBU", time="08:10:00")
        with patch(
            "motorsports_data_notebook.session_runner.load_session", side_effect=[log1, log2]
        ):
            groups = group_session_files(["a.xrk", "b.xrk"])
        assert len(groups) == 2
        assert groups[0]["driver"] == "CMD"
        assert groups[1]["driver"] == "SBU"

    def test_split_on_different_venue(self):
        """Different venues are separate sessions."""
        log1 = _make_mock_log(venue="Fuji", time="08:00:00")
        log2 = _make_mock_log(venue="Suzuka", time="08:10:00")
        with patch(
            "motorsports_data_notebook.session_runner.load_session", side_effect=[log1, log2]
        ):
            groups = group_session_files(["a.xrk", "b.xrk"])
        assert len(groups) == 2

    def test_three_files_two_groups(self):
        """Three files: first two merge, third is separate."""
        log1 = _make_mock_log(time="08:00:00", laps=5)
        log2 = _make_mock_log(time="08:15:00", laps=3)
        log3 = _make_mock_log(time="10:00:00", laps=7)
        with patch(
            "motorsports_data_notebook.session_runner.load_session",
            side_effect=[log1, log2, log3],
        ):
            groups = group_session_files(["a.xrk", "b.xrk", "c.xrk"], max_gap_minutes=30)
        assert len(groups) == 2
        assert len(groups[0]["files"]) == 2
        assert len(groups[1]["files"]) == 1

    def test_missing_datetime(self):
        """Files without timestamps become separate groups."""
        log1 = _make_mock_log(date="", time="")
        log2 = _make_mock_log(date="", time="")
        with patch(
            "motorsports_data_notebook.session_runner.load_session", side_effect=[log1, log2]
        ):
            groups = group_session_files(["a.xrk", "b.xrk"])
        assert len(groups) == 2


# ── run_session_analysis ─────────────────────────────────────────────────────


class TestRunSessionAnalysis:
    def test_basic_report_generation(self, tmp_path):
        """run_session_analysis returns a report without images when no image args."""
        from motorsports_data_notebook.session_runner import run_session_analysis

        fake_file = tmp_path / "test.xrk"
        fake_file.touch()

        mock_log = MagicMock()
        mock_report = MagicMock()
        mock_report.corners = []

        with (
            patch(
                "motorsports_data_notebook.session_runner.load_and_merge_sessions",
                return_value=(mock_log, fake_file),
            ),
            patch(
                "motorsports_data_notebook.session_runner.resolve_profile",
                return_value=({"gps_speed": "GPS_Speed"}, None),
            ),
            patch(
                "motorsports_data_notebook.session_runner.generate_session_report",
                return_value=mock_report,
            ),
        ):
            result = run_session_analysis(session_files=[str(fake_file)])

        assert isinstance(result, SessionAnalysisResult)
        assert result.report is mock_report
        assert result.image_paths == []
