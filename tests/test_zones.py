"""Tests for the zones module.

Uses synthetic data to test braking/acceleration zone detection and segment creation.
"""

import numpy as np
import pytest

from motorsports_data_notebook.corners import Corner
from motorsports_data_notebook.zones import (
    TrackSegment,
    average_zones_across_laps,
    create_track_segments,
    identify_zones_single_lap,
    merge_accel_zones_by_time,
)


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
