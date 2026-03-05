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
    _compute_deceleration,
    _find_brake_release_np,
    _find_peak_brake_np,
    average_zones_across_laps,
    compute_g_utilization,
    compute_segment_stats,
    compute_segment_stats_from_arrays,
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

    def test_compute_segment_stats_exit_speed(
        self, mock_log_for_stats, sample_laps_for_stats, sample_segments_for_stats
    ):
        """Test that corner exit speed is computed."""
        stats_df = compute_segment_stats(
            mock_log_for_stats,
            sample_laps_for_stats,
            sample_segments_for_stats,
            TEST_CHANNEL_NAMES,
        )

        corner_stats = stats_df[stats_df["segment_type"] == "corner"]
        assert len(corner_stats) > 0
        assert "exit_speed" in corner_stats.columns

        # Exit speed should be around 80 km/h (synthetic corner speed in fixture)
        valid_corners = corner_stats.dropna(subset=["exit_speed"])
        assert len(valid_corners) > 0
        for val in valid_corners["exit_speed"]:
            assert val == pytest.approx(80, abs=5)

    def test_compute_segment_stats_exit_speed_gte_min_speed(
        self, mock_log_for_stats, sample_laps_for_stats, sample_segments_for_stats
    ):
        """Test that exit speed >= min speed for all corner rows."""
        stats_df = compute_segment_stats(
            mock_log_for_stats,
            sample_laps_for_stats,
            sample_segments_for_stats,
            TEST_CHANNEL_NAMES,
        )

        corner_stats = stats_df[stats_df["segment_type"] == "corner"].dropna(
            subset=["exit_speed", "min_speed"]
        )
        assert len(corner_stats) > 0
        assert all(corner_stats["exit_speed"] >= corner_stats["min_speed"])

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


# ============================================================================
# Tests for braking helper functions
# ============================================================================


class TestFindPeakBrake:
    """Tests for _find_peak_brake_np function."""

    def test_clear_peak(self):
        """Test with a clear peak at known distance."""
        distance = np.linspace(0, 500, 250)
        brake = np.zeros(250)
        # Peak at ~200m
        peak_mask = (distance >= 180) & (distance <= 220)
        brake[peak_mask] = 60
        # Make the actual peak at ~200m
        peak_idx = np.argmin(np.abs(distance - 200))
        brake[peak_idx] = 80

        result = _find_peak_brake_np(brake, distance, 100, 300)

        assert result is not None
        peak_value, peak_dist = result
        assert peak_value == pytest.approx(80.0)
        assert peak_dist == pytest.approx(200.0, abs=3)

    def test_returns_none_for_empty_segment(self):
        """Test returns None when si >= ei."""
        distance = np.linspace(0, 500, 250)
        brake = np.ones(250) * 50

        # start_dist > end_dist => si >= ei
        result = _find_peak_brake_np(brake, distance, 300, 100)
        assert result is None

    def test_returns_none_for_equal_bounds(self):
        """Test returns None when start == end."""
        distance = np.linspace(0, 500, 250)
        brake = np.ones(250) * 50

        result = _find_peak_brake_np(brake, distance, 200, 200)
        assert result is None

    def test_peak_value_and_distance_correct(self):
        """Test that both value and distance are correct."""
        distance = np.arange(0, 100, dtype=float)
        brake = np.zeros(100)
        brake[50] = 90.0  # Peak at distance 50

        result = _find_peak_brake_np(brake, distance, 0, 100)

        assert result is not None
        assert result[0] == pytest.approx(90.0)
        assert result[1] == pytest.approx(50.0)


class TestFindBrakeRelease:
    """Tests for _find_brake_release_np function."""

    def test_release_detected(self):
        """Test with brake that rises, peaks at 50 bar, then drops to 0."""
        distance = np.arange(0, 200, dtype=float)
        brake = np.zeros(200)
        # Rise to peak at index 80
        brake[60:80] = np.linspace(0, 50, 20)
        brake[80] = 50
        # Drop from peak
        brake[81:120] = np.linspace(45, 0, 39)

        result = _find_brake_release_np(brake, distance, 0, 200)

        assert result is not None
        # Release should occur where brake drops below 10% of 50 = 5 bar
        # That happens somewhere around index 115-120
        assert result > 80  # After the peak
        assert result < 200  # Before end

    def test_threshold_pct_behavior(self):
        """Test with different threshold_pct values."""
        distance = np.arange(0, 100, dtype=float)
        brake = np.zeros(100)
        brake[20] = 50  # Peak at 50
        brake[21:60] = np.linspace(48, 0, 39)  # Linear drop

        result_10 = _find_brake_release_np(brake, distance, 0, 100, threshold_pct=0.10)
        result_50 = _find_brake_release_np(brake, distance, 0, 100, threshold_pct=0.50)

        assert result_10 is not None
        assert result_50 is not None
        # Higher threshold => release detected earlier
        assert result_50 < result_10

    def test_returns_none_when_brake_held(self):
        """Test returns None when brake never drops below threshold."""
        distance = np.arange(0, 100, dtype=float)
        brake = np.ones(100) * 50  # Constant high pressure

        result = _find_brake_release_np(brake, distance, 0, 100)
        assert result is None

    def test_returns_none_for_empty_segment(self):
        """Test returns None when si >= ei."""
        distance = np.arange(0, 100, dtype=float)
        brake = np.ones(100) * 50

        result = _find_brake_release_np(brake, distance, 80, 20)
        assert result is None

    def test_returns_none_for_zero_brake(self):
        """Test returns None when peak brake is zero."""
        distance = np.arange(0, 100, dtype=float)
        brake = np.zeros(100)

        result = _find_brake_release_np(brake, distance, 0, 100)
        assert result is None


class TestComputeDeceleration:
    """Tests for _compute_deceleration function."""

    def test_known_deceleration(self):
        """Test with synthetic speed drop: 200 -> 100 km/h over 100m."""
        distance = np.linspace(0, 200, 1000)
        # Speed drops linearly from 200 to 100 km/h between 50m and 150m
        speed = np.ones(1000) * 200.0
        mask = distance >= 50
        speed[mask] = 200 - (distance[mask] - 50) * (100 / 100)
        speed[distance >= 150] = 100

        result = _compute_deceleration(speed, distance, 50, 150)

        # v0 = 200/3.6 ≈ 55.56 m/s, v1 = 100/3.6 ≈ 27.78 m/s
        # a = (55.56² - 27.78²) / (2 * 100) / 9.81 ≈ 1.18 g
        assert result is not None
        assert result == pytest.approx(1.18, abs=0.1)

    def test_returns_none_for_zero_distance(self):
        """Test returns None when distance is zero."""
        distance = np.array([100.0, 100.0, 100.0])
        speed = np.array([200.0, 150.0, 100.0])

        result = _compute_deceleration(speed, distance, 100, 100)
        assert result is None

    def test_returns_none_for_acceleration(self):
        """Test returns None when speed is increasing (negative decel)."""
        distance = np.linspace(0, 200, 100)
        # Speed increases from 100 to 200 km/h
        speed = np.linspace(100, 200, 100)

        result = _compute_deceleration(speed, distance, 0, 200)
        assert result is None


# ============================================================================
# Tests for compute_segment_stats_from_arrays (braking segment stats)
# ============================================================================


class TestBrakingSegmentStats:
    """Tests for braking segment fields in compute_segment_stats_from_arrays."""

    @pytest.fixture
    def braking_arrays(self):
        """Create synthetic arrays with clear braking pattern."""
        n = 500
        distance = np.linspace(0, 1000, n)

        # Speed: 200 km/h, drops through braking zone 50-150m to 80 km/h
        speed = np.ones(n) * 200.0
        braking_mask = (distance >= 80) & (distance <= 150)
        speed[braking_mask] = np.linspace(200, 80, int(braking_mask.sum()))
        speed[distance > 150] = 80

        # Brake pressure: peaks at ~100m
        brake = np.zeros(n)
        brake_mask = (distance >= 80) & (distance <= 180)
        brake_indices = np.where(brake_mask)[0]
        # Rise to peak at ~100m then decay
        for i, idx in enumerate(brake_indices):
            d = distance[idx]
            if d < 100:
                brake[idx] = (d - 80) / 20 * 60  # rise to 60
            elif d < 140:
                brake[idx] = 60 - (d - 100) / 40 * 60  # decay from 60 to 0
            else:
                brake[idx] = 0

        # Throttle: picks up after corner at 250m
        throttle = np.zeros(n)
        throttle_mask = distance >= 250
        throttle[throttle_mask] = 80

        return distance, speed, brake, throttle

    @pytest.fixture
    def braking_segments(self):
        """Create segments matching the synthetic data."""
        return [
            TrackSegment(
                id=1,
                segment_type="braking",
                start_dist=50,
                end_dist=180,
                name="Turn 1 Braking",
                corner_id=1,
            ),
            TrackSegment(
                id=2,
                segment_type="corner",
                start_dist=180,
                end_dist=250,
                name="Turn 1",
                corner_id=1,
                apex_dist=215,
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

    def test_braking_segment_has_new_fields(self, braking_arrays, braking_segments):
        """Test that braking segments have the new stat fields."""
        distance, speed, brake, throttle = braking_arrays

        df = compute_segment_stats_from_arrays(
            distances=[distance],
            speeds=[speed],
            brakes=[brake],
            throttles=[throttle],
            lap_nums=[1],
            lap_times=[60.0],
            segments=braking_segments,
        )

        braking_rows = df[df["segment_type"] == "braking"]
        assert len(braking_rows) == 1
        row = braking_rows.iloc[0]

        # Check new fields exist
        assert "peak_brake" in df.columns
        assert "peak_brake_dist" in df.columns
        assert "brake_release_point" in df.columns
        assert "entry_speed" in df.columns
        assert "braking_distance" in df.columns
        assert "mean_decel_g" in df.columns

    def test_peak_brake_reasonable(self, braking_arrays, braking_segments):
        """Test peak brake is a reasonable value."""
        distance, speed, brake, throttle = braking_arrays

        df = compute_segment_stats_from_arrays(
            distances=[distance],
            speeds=[speed],
            brakes=[brake],
            throttles=[throttle],
            lap_nums=[1],
            lap_times=[60.0],
            segments=braking_segments,
        )

        row = df[df["segment_type"] == "braking"].iloc[0]
        assert pd.notna(row["peak_brake"])
        assert row["peak_brake"] == pytest.approx(60.0, abs=5)

    def test_entry_speed_reasonable(self, braking_arrays, braking_segments):
        """Test entry speed is close to initial speed."""
        distance, speed, brake, throttle = braking_arrays

        df = compute_segment_stats_from_arrays(
            distances=[distance],
            speeds=[speed],
            brakes=[brake],
            throttles=[throttle],
            lap_nums=[1],
            lap_times=[60.0],
            segments=braking_segments,
        )

        row = df[df["segment_type"] == "braking"].iloc[0]
        if pd.notna(row.get("entry_speed")):
            # Entry speed should be near 200 km/h (braking starts at ~80m)
            assert row["entry_speed"] >= 150

    def test_mean_decel_positive(self, braking_arrays, braking_segments):
        """Test mean deceleration is positive for actual braking."""
        distance, speed, brake, throttle = braking_arrays

        df = compute_segment_stats_from_arrays(
            distances=[distance],
            speeds=[speed],
            brakes=[brake],
            throttles=[throttle],
            lap_nums=[1],
            lap_times=[60.0],
            segments=braking_segments,
        )

        row = df[df["segment_type"] == "braking"].iloc[0]
        if pd.notna(row.get("mean_decel_g")):
            assert row["mean_decel_g"] > 0


# ============================================================================
# Tests for compute_g_utilization
# ============================================================================


class TestComputeGUtilization:
    """Tests for compute_g_utilization function."""

    @pytest.fixture
    def g_util_data(self):
        """Create synthetic data with known G patterns."""
        n = 500
        distance = np.linspace(0, 1000, n)
        speed = np.ones(n) * 150.0  # 150 km/h constant

        # Lateral G: 0 in straights, 1.2g in corner (200-400m)
        lateral_g = np.zeros(n)
        corner_mask = (distance >= 200) & (distance <= 400)
        lateral_g[corner_mask] = 1.2

        # Inline G: -1.0g braking (100-200m), +0.5g accel (400-600m)
        inline_g = np.zeros(n)
        braking_mask = (distance >= 100) & (distance <= 200)
        inline_g[braking_mask] = -1.0
        accel_mask = (distance >= 400) & (distance <= 600)
        inline_g[accel_mask] = 0.5

        return distance, speed, lateral_g, inline_g

    @pytest.fixture
    def g_util_segments(self):
        """Create segments for G utilization test."""
        return [
            TrackSegment(
                id=1,
                segment_type="braking",
                start_dist=100,
                end_dist=200,
                name="Turn 1 Braking",
                corner_id=1,
            ),
            TrackSegment(
                id=2,
                segment_type="corner",
                start_dist=200,
                end_dist=400,
                name="Turn 1",
                corner_id=1,
                apex_dist=300,
            ),
            TrackSegment(
                id=3,
                segment_type="acceleration",
                start_dist=400,
                end_dist=600,
                name="Turn 1 Exit",
                corner_id=1,
            ),
        ]

    @pytest.fixture
    def g_util_corners(self):
        """Create corners for G utilization test."""
        return [
            Corner(
                id=1,
                name="Turn 1",
                direction="L",
                start_idx=100,
                end_idx=200,
                start_dist=200.0,
                end_dist=400.0,
                apex_idx=150,
                apex_dist=300.0,
                max_curvature=0.01,
            )
        ]

    def test_output_has_expected_columns(self, g_util_data, g_util_segments, g_util_corners):
        """Test output DataFrame has expected columns."""
        distance, speed, lateral_g, inline_g = g_util_data

        df = compute_g_utilization(
            distances=[distance],
            speeds=[speed],
            lateral_gs=[lateral_g],
            inline_gs=[inline_g],
            lap_nums=[1],
            segments=g_util_segments,
            corners=g_util_corners,
        )

        expected_cols = [
            "corner_id",
            "corner_name",
            "lap_num",
            "total_g_mean",
            "total_g_max",
            "total_g_min",
            "total_g_min_phase",
            "g_utilization_pct",
            "braking_g_mean",
            "entry_g_mean",
            "mid_g_mean",
            "exit_g_mean",
        ]
        for col in expected_cols:
            assert col in df.columns

    def test_one_row_per_corner_per_lap(self, g_util_data, g_util_segments, g_util_corners):
        """Test one row per corner per lap."""
        distance, speed, lateral_g, inline_g = g_util_data

        df = compute_g_utilization(
            distances=[distance, distance],
            speeds=[speed, speed],
            lateral_gs=[lateral_g, lateral_g],
            inline_gs=[inline_g, inline_g],
            lap_nums=[1, 2],
            segments=g_util_segments,
            corners=g_util_corners,
        )

        assert len(df) == 2  # 1 corner × 2 laps

    def test_phase_means_computed(self, g_util_data, g_util_segments, g_util_corners):
        """Test that phase means are computed correctly."""
        distance, speed, lateral_g, inline_g = g_util_data

        df = compute_g_utilization(
            distances=[distance],
            speeds=[speed],
            lateral_gs=[lateral_g],
            inline_gs=[inline_g],
            lap_nums=[1],
            segments=g_util_segments,
            corners=g_util_corners,
        )

        row = df.iloc[0]
        # Braking phase (100-200m): inline_g = -1.0, lateral_g = 0 => total_g = 1.0
        assert row["braking_g_mean"] is not None
        assert row["braking_g_mean"] == pytest.approx(1.0, abs=0.1)

    def test_with_inline_gs_none(self, g_util_data, g_util_segments, g_util_corners):
        """Test with inline_gs=None (derived from speed) — should not crash."""
        distance, speed, lateral_g, _ = g_util_data

        df = compute_g_utilization(
            distances=[distance],
            speeds=[speed],
            lateral_gs=[lateral_g],
            inline_gs=None,
            lap_nums=[1],
            segments=g_util_segments,
            corners=g_util_corners,
        )

        assert len(df) == 1
        assert pd.notna(df.iloc[0]["total_g_mean"])

    def test_g_utilization_pct_in_range(self, g_util_data, g_util_segments, g_util_corners):
        """Test g_utilization_pct is in reasonable range (0-100)."""
        distance, speed, lateral_g, inline_g = g_util_data

        df = compute_g_utilization(
            distances=[distance],
            speeds=[speed],
            lateral_gs=[lateral_g],
            inline_gs=[inline_g],
            lap_nums=[1],
            segments=g_util_segments,
            corners=g_util_corners,
        )

        pct = df.iloc[0]["g_utilization_pct"]
        assert 0 <= pct <= 100
