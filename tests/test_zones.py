"""Tests for the zones module.

Uses synthetic data to test braking/acceleration zone detection and segment creation.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from helpers import MockLogFile, make_channel_table
from motorsports_data_notebook.corners import Corner
from motorsports_data_notebook.zones import (
    TrackSegment,
    average_zones_across_laps,
    compute_segment_stats,
    create_track_segments,
    detect_zones_averaged,
    get_corner_data,
    get_segment_mask,
    identify_zones_single_lap,
    merge_accel_zones_by_time,
)

# Test channel names mapping (matches the mock data channel names)
TEST_CHANNEL_NAMES = {
    "throttle": "PPS",
    "brake": "BrakePress",
    "gps_speed": "GPS Speed",
    "lateral_g": "LateralAcc",
    "gps_latitude": "GPS Latitude",
    "gps_longitude": "GPS Longitude",
    "steering": "SteerAngle",
}


class TestIdentifyZonesSingleLap:
    """Tests for identify_zones_single_lap function."""

    def test_no_braking_no_acceleration(self):
        """Test with no braking or acceleration zones."""
        distance = np.linspace(0, 1000, 500)
        brake_press = np.zeros(500)  # No braking
        throttle = np.zeros(500)  # No throttle
        speed = np.ones(500) * 30  # Constant 30 m/s

        braking_zones, accel_zones = identify_zones_single_lap(
            distance, brake_press, throttle, speed
        )

        assert len(braking_zones) == 0
        assert len(accel_zones) == 0

    def test_single_braking_zone(self):
        """Test detection of a single braking zone."""
        distance = np.linspace(0, 1000, 500)
        brake_press = np.zeros(500)
        throttle = np.zeros(500)
        speed = np.ones(500) * 30

        # Add braking from 200m to 300m
        brake_mask = (distance >= 200) & (distance <= 300)
        brake_press[brake_mask] = 50  # 50% brake pressure

        braking_zones, accel_zones = identify_zones_single_lap(
            distance, brake_press, throttle, speed, brake_threshold=5
        )

        assert len(braking_zones) == 1
        assert braking_zones[0][0] == pytest.approx(200, abs=5)
        assert braking_zones[0][1] == pytest.approx(300, abs=5)

    def test_single_acceleration_zone(self):
        """Test detection of a single acceleration zone."""
        distance = np.linspace(0, 1000, 500)
        brake_press = np.zeros(500)
        throttle = np.zeros(500)
        speed = np.ones(500) * 30

        # Add acceleration from 400m to 600m
        accel_mask = (distance >= 400) & (distance <= 600)
        throttle[accel_mask] = 80  # 80% throttle

        braking_zones, accel_zones = identify_zones_single_lap(
            distance, brake_press, throttle, speed, throttle_threshold=20
        )

        assert len(accel_zones) == 1
        assert accel_zones[0][0] == pytest.approx(400, abs=5)
        assert accel_zones[0][1] == pytest.approx(600, abs=5)

    def test_multiple_braking_zones(self):
        """Test detection of multiple braking zones."""
        distance = np.linspace(0, 1000, 500)
        brake_press = np.zeros(500)
        throttle = np.zeros(500)
        speed = np.ones(500) * 30

        # Braking zone 1: 100-150m
        mask1 = (distance >= 100) & (distance <= 150)
        brake_press[mask1] = 60

        # Braking zone 2: 500-550m
        mask2 = (distance >= 500) & (distance <= 550)
        brake_press[mask2] = 70

        braking_zones, accel_zones = identify_zones_single_lap(
            distance, brake_press, throttle, speed
        )

        assert len(braking_zones) == 2

    def test_accel_zone_excludes_braking(self):
        """Test that acceleration zones exclude areas with braking."""
        distance = np.linspace(0, 500, 250)
        brake_press = np.zeros(250)
        throttle = np.ones(250) * 50  # Throttle everywhere
        speed = np.ones(250) * 30

        # Add braking in the middle
        brake_mask = (distance >= 200) & (distance <= 250)
        brake_press[brake_mask] = 30

        braking_zones, accel_zones = identify_zones_single_lap(
            distance, brake_press, throttle, speed
        )

        # Should have two accel zones (before and after braking)
        # The braking area should not be counted as acceleration
        for zone in accel_zones:
            # No accel zone should overlap with braking zone
            assert not (zone[0] < 250 and zone[1] > 200)

    def test_auto_detect_zero_to_one_scale(self):
        """Test that zones are found with 0-1 scale data (iRacing)."""
        distance = np.linspace(0, 1000, 500)
        brake_press = np.zeros(500)
        throttle = np.zeros(500)
        speed = np.ones(500) * 30

        # Braking zone at 200-300m with 0-1 scale values
        brake_mask = (distance >= 200) & (distance <= 300)
        brake_press[brake_mask] = 0.5  # 50% brake in 0-1 scale

        # Acceleration zone at 400-600m with 0-1 scale values
        accel_mask = (distance >= 400) & (distance <= 600)
        throttle[accel_mask] = 0.8  # 80% throttle in 0-1 scale

        # Auto-detect (no explicit thresholds)
        braking_zones, accel_zones = identify_zones_single_lap(
            distance, brake_press, throttle, speed
        )

        assert len(braking_zones) >= 1
        assert braking_zones[0][0] == pytest.approx(200, abs=5)
        assert len(accel_zones) >= 1
        assert accel_zones[0][0] == pytest.approx(400, abs=5)

    def test_gear_change_gap_merging(self):
        """Test that short gaps in throttle (gear changes) are merged."""
        distance = np.linspace(0, 500, 250)
        brake_press = np.zeros(250)
        throttle = np.zeros(250)
        speed = np.ones(250) * 40  # 40 m/s constant

        # Two throttle zones with a small gap (simulating gear change)
        mask1 = (distance >= 100) & (distance <= 150)
        mask2 = (distance >= 160) & (distance <= 200)  # 10m gap at 40m/s = 0.25s
        throttle[mask1] = 80
        throttle[mask2] = 80

        braking_zones, accel_zones = identify_zones_single_lap(
            distance, brake_press, throttle, speed, gear_change_time=1.5
        )

        # Should merge into one zone since gap time < gear_change_time
        assert len(accel_zones) == 1
        assert accel_zones[0][0] == pytest.approx(100, abs=5)
        assert accel_zones[0][1] == pytest.approx(200, abs=5)


class TestAverageZonesAcrossLaps:
    """Tests for average_zones_across_laps function."""

    def test_identical_laps(self):
        """Test averaging when all laps have identical zones."""
        # All laps have braking at 100-200m
        all_braking = [[(100.0, 200.0)] for _ in range(5)]
        all_accel = [[(300.0, 400.0)] for _ in range(5)]

        braking_zones, accel_zones = average_zones_across_laps(
            all_braking, all_accel, track_length=500, resolution=1.0, threshold=0.5
        )

        assert len(braking_zones) == 1
        assert braking_zones[0][0] == pytest.approx(100, abs=2)
        assert braking_zones[0][1] == pytest.approx(200, abs=2)

        assert len(accel_zones) == 1
        assert accel_zones[0][0] == pytest.approx(300, abs=2)
        assert accel_zones[0][1] == pytest.approx(400, abs=2)

    def test_threshold_filtering(self):
        """Test that zones appearing in less than threshold laps are filtered."""
        # Only 1 out of 5 laps has braking at 100-200m
        all_braking = [
            [(100.0, 200.0)],  # Lap 1
            [],  # Lap 2
            [],  # Lap 3
            [],  # Lap 4
            [],  # Lap 5
        ]
        all_accel = [[] for _ in range(5)]

        braking_zones, accel_zones = average_zones_across_laps(
            all_braking, all_accel, track_length=500, resolution=1.0, threshold=0.5
        )

        # Should not appear (1/5 = 0.2 < 0.5 threshold)
        assert len(braking_zones) == 0

    def test_overlapping_zones_merged(self):
        """Test that overlapping zones from different laps are merged."""
        # Slightly different braking points across laps
        all_braking = [
            [(100.0, 200.0)],
            [(105.0, 195.0)],
            [(95.0, 205.0)],
            [(100.0, 200.0)],
            [(100.0, 200.0)],
        ]
        all_accel = [[] for _ in range(5)]

        braking_zones, accel_zones = average_zones_across_laps(
            all_braking, all_accel, track_length=500, resolution=1.0, threshold=0.5
        )

        # Should merge into one zone covering the overlap
        assert len(braking_zones) == 1


class TestMergeAccelZonesByTime:
    """Tests for merge_accel_zones_by_time function."""

    def test_no_merging_needed(self):
        """Test when zones are far apart and don't need merging."""
        accel_zones = [(100.0, 200.0), (500.0, 600.0)]
        braking_zones = []
        distance = np.linspace(0, 1000, 500)
        speed = np.ones(500) * 30

        result = merge_accel_zones_by_time(
            accel_zones, braking_zones, distance, speed, max_gap_time=1.5
        )

        assert len(result) == 2

    def test_merge_short_gap(self):
        """Test merging zones with short time gap."""
        accel_zones = [(100.0, 200.0), (210.0, 300.0)]  # 10m gap
        braking_zones = []
        distance = np.linspace(0, 500, 250)
        speed = np.ones(250) * 50  # 50 m/s -> 10m gap = 0.2s

        result = merge_accel_zones_by_time(
            accel_zones, braking_zones, distance, speed, max_gap_time=1.5
        )

        # Should merge since 0.2s < 1.5s
        assert len(result) == 1
        assert result[0][0] == pytest.approx(100.0)
        assert result[0][1] == pytest.approx(300.0)

    def test_no_merge_when_braking_in_gap(self):
        """Test that zones aren't merged if there's braking in the gap."""
        accel_zones = [(100.0, 200.0), (210.0, 300.0)]
        braking_zones = [(195.0, 215.0)]  # Braking in the gap
        distance = np.linspace(0, 500, 250)
        speed = np.ones(250) * 50

        result = merge_accel_zones_by_time(
            accel_zones, braking_zones, distance, speed, max_gap_time=1.5
        )

        # Should NOT merge due to braking in gap
        assert len(result) == 2

    def test_single_zone_unchanged(self):
        """Test that a single zone is returned unchanged."""
        accel_zones = [(100.0, 200.0)]
        braking_zones = []
        distance = np.linspace(0, 500, 250)
        speed = np.ones(250) * 30

        result = merge_accel_zones_by_time(
            accel_zones, braking_zones, distance, speed, max_gap_time=1.5
        )

        assert len(result) == 1
        assert result[0] == (100.0, 200.0)


class TestCreateTrackSegments:
    """Tests for create_track_segments function."""

    def test_single_corner_creates_three_segments(self):
        """Test that a single corner creates braking, corner, and accel segments."""
        corners = [
            Corner(
                id=1,
                name="Turn 1",
                direction="L",
                start_idx=100,
                end_idx=150,
                start_dist=200.0,
                end_dist=300.0,
                apex_idx=125,
                apex_dist=250.0,
                max_curvature=0.01,
            )
        ]
        braking_zones = [(100.0, 200.0)]
        accel_zones = [(300.0, 450.0)]

        segments = create_track_segments(corners, braking_zones, accel_zones, track_length=500)

        # Should have 3 segments: braking, corner, acceleration
        assert len(segments) == 3

        # Find each type
        braking_seg = next(s for s in segments if s.segment_type == "braking")
        corner_seg = next(s for s in segments if s.segment_type == "corner")
        accel_seg = next(s for s in segments if s.segment_type == "acceleration")

        assert braking_seg.name == "Turn 1 Braking"
        assert corner_seg.name == "Turn 1"
        assert accel_seg.name == "Turn 1 Exit"

        assert corner_seg.apex_dist == 250.0

    def test_segments_sorted_by_distance(self):
        """Test that segments are sorted by start distance."""
        corners = [
            Corner(
                id=1,
                name="Turn 1",
                direction="L",
                start_idx=100,
                end_idx=150,
                start_dist=200.0,
                end_dist=300.0,
                apex_idx=125,
                apex_dist=250.0,
                max_curvature=0.01,
            ),
            Corner(
                id=2,
                name="Turn 2",
                direction="R",
                start_idx=200,
                end_idx=250,
                start_dist=400.0,
                end_dist=500.0,
                apex_idx=225,
                apex_dist=450.0,
                max_curvature=0.01,
            ),
        ]
        braking_zones = [(100.0, 200.0), (300.0, 400.0)]
        accel_zones = [(300.0, 380.0), (500.0, 600.0)]

        segments = create_track_segments(corners, braking_zones, accel_zones, track_length=700)

        # Check that segments are sorted by start_dist
        for i in range(len(segments) - 1):
            assert segments[i].start_dist <= segments[i + 1].start_dist

    def test_default_braking_zone_when_none_found(self):
        """Test that a default braking zone is created when none found."""
        corners = [
            Corner(
                id=1,
                name="Turn 1",
                direction="L",
                start_idx=100,
                end_idx=150,
                start_dist=200.0,
                end_dist=300.0,
                apex_idx=125,
                apex_dist=250.0,
                max_curvature=0.01,
            )
        ]
        braking_zones = []  # No braking zones
        accel_zones = [(300.0, 450.0)]

        segments = create_track_segments(corners, braking_zones, accel_zones, track_length=500)

        braking_seg = next(s for s in segments if s.segment_type == "braking")

        # Default: 100m before corner start
        assert braking_seg.start_dist == pytest.approx(100.0)
        assert braking_seg.end_dist == pytest.approx(200.0)


class TestTrackSegmentDataclass:
    """Tests for TrackSegment dataclass."""

    def test_length_property(self):
        """Test that length property returns correct value."""
        segment = TrackSegment(
            id=1,
            segment_type="corner",
            start_dist=100.0,
            end_dist=200.0,
            name="Turn 1",
            corner_id=1,
            apex_dist=150.0,
        )

        assert segment.length == 100.0

    def test_apex_dist_optional(self):
        """Test that apex_dist is optional (for braking/accel segments)."""
        segment = TrackSegment(
            id=1,
            segment_type="braking",
            start_dist=50.0,
            end_dist=100.0,
            name="Turn 1 Braking",
            corner_id=1,
        )

        assert segment.apex_dist is None


# ============================================================================
# Tests for detect_zones_averaged
# ============================================================================


class TestDetectZonesAveraged:
    """Tests for detect_zones_averaged function."""

    @pytest.fixture
    def mock_log_for_zones(self):
        """Create a mock LogFile with clear braking/accel patterns."""
        # 3 laps of data, each lap 1000m
        n_samples_per_lap = 200
        n_laps = 3

        # Build continuous timecodes across all laps
        all_timecodes = []
        all_distance = []
        all_speed = []
        all_brake = []
        all_pps = []

        for lap_num in range(n_laps):
            lap_start = lap_num * 60000
            lap_end = (lap_num + 1) * 60000

            timecodes = np.linspace(lap_start, lap_end - 1, n_samples_per_lap, dtype=np.int64)
            distance_m = np.linspace(0, 1000, n_samples_per_lap)
            speed = np.ones(n_samples_per_lap) * 30  # 30 m/s

            # Brake pressure: high at 200-300m
            brake_press = np.zeros(n_samples_per_lap)
            brake_mask = (distance_m >= 200) & (distance_m <= 300)
            brake_press[brake_mask] = 60

            # Throttle: high at 400-800m
            pps = np.zeros(n_samples_per_lap)
            throttle_mask = (distance_m >= 400) & (distance_m <= 800)
            pps[throttle_mask] = 80

            all_timecodes.append(timecodes)
            all_distance.append(distance_m)
            all_speed.append(speed)
            all_brake.append(brake_press)
            all_pps.append(pps)

        timecodes = np.concatenate(all_timecodes)
        distance_m = np.concatenate(all_distance)
        speed = np.concatenate(all_speed)
        brake = np.concatenate(all_brake)
        pps = np.concatenate(all_pps)

        channels = {
            "distance_m": make_channel_table(timecodes, "distance_m", distance_m),
            "GPS Speed": make_channel_table(timecodes, "GPS Speed", speed),
            "BrakePress": make_channel_table(timecodes, "BrakePress", brake),
            "PPS": make_channel_table(timecodes, "PPS", pps),
        }

        return MockLogFile(channels)

    @pytest.fixture
    def sample_laps_for_zones(self):
        """Create sample laps DataFrame."""
        return pd.DataFrame(
            {
                "num": [1, 2, 3],
                "start_time": [0, 60000, 120000],
                "end_time": [60000, 120000, 180000],
                "lap_time": pd.to_timedelta([60, 58, 59], unit="s"),
            }
        )

    def test_detect_zones_averaged_basic(self, mock_log_for_zones, sample_laps_for_zones):
        """Test basic zone detection."""
        braking_zones, accel_zones = detect_zones_averaged(
            mock_log_for_zones,
            sample_laps_for_zones,
            TEST_CHANNEL_NAMES,
        )

        # Should find one braking zone around 200-300m
        assert len(braking_zones) >= 1

        # Should find acceleration zone around 400-800m
        assert len(accel_zones) >= 1

    def test_detect_zones_averaged_braking_location(
        self, mock_log_for_zones, sample_laps_for_zones
    ):
        """Test that braking zone is in expected location."""
        braking_zones, _ = detect_zones_averaged(
            mock_log_for_zones,
            sample_laps_for_zones,
            TEST_CHANNEL_NAMES,
        )

        # Braking zone should be around 200-300m
        found_braking = False
        for start, end in braking_zones:
            if start <= 250 <= end:
                found_braking = True
                break
        assert found_braking

    def test_detect_zones_averaged_returns_tuple(self, mock_log_for_zones, sample_laps_for_zones):
        """Test that function returns correct tuple structure."""
        result = detect_zones_averaged(
            mock_log_for_zones,
            sample_laps_for_zones,
            TEST_CHANNEL_NAMES,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)


# ============================================================================
# Tests for compute_segment_stats
# ============================================================================


class TestComputeSegmentStats:
    """Tests for compute_segment_stats function."""

    @pytest.fixture
    def mock_log_for_stats(self):
        """Create a mock LogFile with known patterns."""
        # 2 laps of data
        n_samples_per_lap = 200

        all_timecodes = []
        all_distance = []
        all_speed = []
        all_brake = []
        all_pps = []

        for lap_num in range(2):
            lap_start = lap_num * 60000
            lap_end = (lap_num + 1) * 60000

            timecodes = np.linspace(lap_start, lap_end - 1, n_samples_per_lap, dtype=np.int64)
            distance_m = np.linspace(0, 1000, n_samples_per_lap)

            # Speed: slower in corner (150-250m), faster elsewhere
            speed_kmh = np.ones(n_samples_per_lap) * 150
            corner_mask = (distance_m >= 150) & (distance_m <= 250)
            speed_kmh[corner_mask] = 80  # Slower through corner

            # Brake at 100-150m
            brake_press = np.zeros(n_samples_per_lap)
            brake_mask = (distance_m >= 100) & (distance_m <= 150)
            brake_press[brake_mask] = 60

            # Throttle at 250-400m
            pps = np.zeros(n_samples_per_lap)
            throttle_mask = (distance_m >= 250) & (distance_m <= 400)
            pps[throttle_mask] = 80

            all_timecodes.append(timecodes)
            all_distance.append(distance_m)
            all_speed.append(speed_kmh)
            all_brake.append(brake_press)
            all_pps.append(pps)

        timecodes = np.concatenate(all_timecodes)
        distance_m = np.concatenate(all_distance)
        speed = np.concatenate(all_speed)
        brake = np.concatenate(all_brake)
        pps = np.concatenate(all_pps)

        channels = {
            "distance_m": make_channel_table(timecodes, "distance_m", distance_m),
            "speed_kmh": make_channel_table(timecodes, "speed_kmh", speed),
            "BrakePress": make_channel_table(timecodes, "BrakePress", brake),
            "PPS": make_channel_table(timecodes, "PPS", pps),
        }

        return MockLogFile(channels)

    @pytest.fixture
    def sample_laps_for_stats(self):
        """Create sample laps."""
        return pd.DataFrame(
            {
                "num": [1, 2],
                "start_time": [0, 60000],
                "end_time": [60000, 120000],
                "lap_time": pd.to_timedelta([60, 58], unit="s"),
            }
        )

    @pytest.fixture
    def sample_segments_for_stats(self):
        """Create sample segments matching the data."""
        return [
            TrackSegment(
                id=1,
                segment_type="braking",
                start_dist=50,
                end_dist=150,
                name="Turn 1 Braking",
                corner_id=1,
            ),
            TrackSegment(
                id=2,
                segment_type="corner",
                start_dist=150,
                end_dist=250,
                name="Turn 1",
                corner_id=1,
                apex_dist=200,
            ),
            TrackSegment(
                id=3,
                segment_type="acceleration",
                start_dist=250,
                end_dist=400,
                name="Turn 1 Exit",
                corner_id=1,
            ),
        ]

    def test_compute_segment_stats_returns_dataframe(
        self, mock_log_for_stats, sample_laps_for_stats, sample_segments_for_stats
    ):
        """Test that function returns a DataFrame."""
        stats_df = compute_segment_stats(
            mock_log_for_stats,
            sample_laps_for_stats,
            sample_segments_for_stats,
            TEST_CHANNEL_NAMES,
        )

        assert isinstance(stats_df, pd.DataFrame)

    def test_compute_segment_stats_has_expected_columns(
        self, mock_log_for_stats, sample_laps_for_stats, sample_segments_for_stats
    ):
        """Test that DataFrame has expected columns."""
        stats_df = compute_segment_stats(
            mock_log_for_stats,
            sample_laps_for_stats,
            sample_segments_for_stats,
            TEST_CHANNEL_NAMES,
        )

        expected_cols = ["segment_id", "segment_name", "segment_type", "corner_id", "lap_num"]
        for col in expected_cols:
            assert col in stats_df.columns

    def test_compute_segment_stats_braking_points(
        self, mock_log_for_stats, sample_laps_for_stats, sample_segments_for_stats
    ):
        """Test that braking points are detected."""
        stats_df = compute_segment_stats(
            mock_log_for_stats,
            sample_laps_for_stats,
            sample_segments_for_stats,
            TEST_CHANNEL_NAMES,
        )

        braking_stats = stats_df[stats_df["segment_type"] == "braking"]
        assert len(braking_stats) > 0
        assert "braking_point" in braking_stats.columns

        # Should have found braking points around 100m
        valid_braking = braking_stats.dropna(subset=["braking_point"])
        assert len(valid_braking) > 0
        assert all(valid_braking["braking_point"] >= 100)

    def test_compute_segment_stats_min_speed(
        self, mock_log_for_stats, sample_laps_for_stats, sample_segments_for_stats
    ):
        """Test that corner min speed is computed."""
        stats_df = compute_segment_stats(
            mock_log_for_stats,
            sample_laps_for_stats,
            sample_segments_for_stats,
            TEST_CHANNEL_NAMES,
        )

        corner_stats = stats_df[stats_df["segment_type"] == "corner"]
        assert len(corner_stats) > 0
        assert "min_speed" in corner_stats.columns

        # Min speed should be around 80 km/h (our synthetic slow speed)
        valid_corners = corner_stats.dropna(subset=["min_speed"])
        assert len(valid_corners) > 0
        assert all(valid_corners["min_speed"] < 100)

    def test_compute_segment_stats_throttle_points(
        self, mock_log_for_stats, sample_laps_for_stats, sample_segments_for_stats
    ):
        """Test that throttle points are detected."""
        stats_df = compute_segment_stats(
            mock_log_for_stats,
            sample_laps_for_stats,
            sample_segments_for_stats,
            TEST_CHANNEL_NAMES,
        )

        accel_stats = stats_df[stats_df["segment_type"] == "acceleration"]
        assert len(accel_stats) > 0
        assert "throttle_point" in accel_stats.columns

    def test_compute_segment_stats_rows_per_lap(
        self, mock_log_for_stats, sample_laps_for_stats, sample_segments_for_stats
    ):
        """Test that we get correct number of rows (segments × laps)."""
        stats_df = compute_segment_stats(
            mock_log_for_stats,
            sample_laps_for_stats,
            sample_segments_for_stats,
            TEST_CHANNEL_NAMES,
        )

        n_laps = len(sample_laps_for_stats)
        n_segments = len(sample_segments_for_stats)
        assert len(stats_df) == n_laps * n_segments


class TestGetSegmentMask:
    """Tests for get_segment_mask function."""

    @pytest.fixture
    def sample_channels(self):
        """Create sample channel data with distance."""
        return pd.DataFrame(
            {
                "distance_m": np.linspace(0, 1000, 500),
                "speed_kmh": np.random.uniform(50, 150, 500),
            }
        )

    @pytest.fixture
    def sample_segment(self):
        """Create a sample segment."""
        return TrackSegment(
            id=1,
            segment_type="corner",
            start_dist=200,
            end_dist=400,
            name="Turn 1",
            corner_id=1,
            apex_dist=300,
        )

    def test_basic_mask(self, sample_channels, sample_segment):
        """Test that mask correctly filters to segment distance range."""
        mask = get_segment_mask(sample_channels, sample_segment)

        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool

        # Check that masked data is within segment bounds
        masked_data = sample_channels[mask]
        assert masked_data["distance_m"].min() >= 200
        assert masked_data["distance_m"].max() <= 400

    def test_mask_with_margin(self, sample_channels, sample_segment):
        """Test that margin extends the masked region."""
        mask = get_segment_mask(sample_channels, sample_segment, margin=50)
        masked_data = sample_channels[mask]

        # With 50m margin, range should be 150-450
        assert masked_data["distance_m"].min() >= 150
        assert masked_data["distance_m"].max() <= 450

    def test_mask_no_margin_vs_margin(self, sample_channels, sample_segment):
        """Test that margin includes more data."""
        mask_no_margin = get_segment_mask(sample_channels, sample_segment)
        mask_with_margin = get_segment_mask(sample_channels, sample_segment, margin=100)

        assert mask_with_margin.sum() > mask_no_margin.sum()

    def test_mask_empty_when_no_overlap(self, sample_channels):
        """Test that mask is empty when segment is outside data range."""
        segment_outside = TrackSegment(
            id=1,
            segment_type="corner",
            start_dist=1500,  # Beyond our 0-1000 data
            end_dist=1700,
            name="Turn Far",
            corner_id=1,
        )

        mask = get_segment_mask(sample_channels, segment_outside)
        assert mask.sum() == 0

    def test_mask_preserves_index(self, sample_channels, sample_segment):
        """Test that mask can be used to filter original DataFrame."""
        mask = get_segment_mask(sample_channels, sample_segment)
        filtered = sample_channels[mask]

        # Should be able to access filtered data without errors
        assert len(filtered) > 0
        assert "distance_m" in filtered.columns
        assert "speed_kmh" in filtered.columns


class TestGetCornerData:
    """Tests for get_corner_data function.

    Note: The function now expects a pre-filtered LogFile (via log.filter_by_lap()).
    The caller is responsible for filtering before calling get_corner_data.
    """

    @pytest.fixture
    def mock_log_for_corner(self):
        """Create a mock LogFile for a single lap."""
        np.random.seed(42)  # For reproducible tests
        n_samples = 200

        timecodes = np.linspace(0, 60000, n_samples, dtype=np.int64)
        distance_m = np.linspace(0, 1000, n_samples)

        channels = {
            "distance_m": make_channel_table(timecodes, "distance_m", distance_m),
            "PPS": make_channel_table(timecodes, "PPS", np.random.uniform(0, 100, n_samples)),
            "BrakePress": make_channel_table(
                timecodes, "BrakePress", np.random.uniform(0, 100, n_samples)
            ),
            "LateralAcc": make_channel_table(
                timecodes, "LateralAcc", np.random.uniform(-1.5, 1.5, n_samples)
            ),
            "SteerAngle": make_channel_table(
                timecodes, "SteerAngle", np.random.uniform(-180, 180, n_samples)
            ),
        }

        return MockLogFile(channels)

    @pytest.fixture
    def sample_corners(self):
        """Create sample Corner objects."""
        return [
            Corner(
                id=1,
                name="Turn 1",
                direction="L",
                start_idx=100,
                end_idx=200,
                start_dist=200,
                end_dist=400,
                apex_idx=150,
                apex_dist=300,
                max_curvature=0.01,
            ),
            Corner(
                id=2,
                name="Turn 2",
                direction="R",
                start_idx=350,
                end_idx=450,
                start_dist=700,
                end_dist=900,
                apex_idx=400,
                apex_dist=800,
                max_curvature=0.01,
            ),
        ]

    def test_get_corner_data_returns_dataframe(self, mock_log_for_corner, sample_corners):
        """Test that get_corner_data returns a DataFrame."""
        result = get_corner_data(mock_log_for_corner, sample_corners[0])
        assert isinstance(result, pd.DataFrame)

    def test_get_corner_data_filters_by_corner(self, mock_log_for_corner, sample_corners):
        """Test that data is filtered to the correct corner distance range."""
        result = get_corner_data(mock_log_for_corner, sample_corners[0])

        # Should have data (the test verifies filtering works by not raising errors)
        assert len(result) > 0
        # Distance should be within corner range with default margin (50m)
        assert result["distance_m"].min() >= sample_corners[0].start_dist - 50
        assert result["distance_m"].max() <= sample_corners[0].end_dist + 50

    def test_get_corner_data_with_zero_margin(self, mock_log_for_corner, sample_corners):
        """Test that data is filtered to exact corner distance range."""
        result = get_corner_data(mock_log_for_corner, sample_corners[0], margin=0)

        # All distances should be in Turn 1 range (200-400)
        assert result["distance_m"].min() >= 200
        assert result["distance_m"].max() <= 400

    def test_get_corner_data_with_margin(self, mock_log_for_corner, sample_corners):
        """Test that margin extends the distance range."""
        result = get_corner_data(mock_log_for_corner, sample_corners[0], margin=50)

        # With 50m margin, distances should be in range 150-450
        assert result["distance_m"].min() >= 150
        assert result["distance_m"].max() <= 450

    def test_get_corner_data_different_corners(self, mock_log_for_corner, sample_corners):
        """Test selecting different corners."""
        turn1_data = get_corner_data(mock_log_for_corner, sample_corners[0], margin=0)
        turn2_data = get_corner_data(mock_log_for_corner, sample_corners[1], margin=0)

        # Turn 1 is 200-400, Turn 2 is 700-900
        assert turn1_data["distance_m"].max() < turn2_data["distance_m"].min()

    def test_get_corner_data_has_expected_columns(self, mock_log_for_corner, sample_corners):
        """Test that result has expected columns from CORNER_DATA_CHANNELS."""
        result = get_corner_data(mock_log_for_corner, sample_corners[0])

        # Default channels from CORNER_DATA_CHANNELS
        assert "distance_m" in result.columns
        assert "PPS" in result.columns
        assert "BrakePress" in result.columns
        assert "LateralAcc" in result.columns
        assert "SteerAngle" in result.columns
