"""Tests for the corners module.

Uses synthetic data to test corner detection and curvature computation algorithms.
"""

import numpy as np
import pytest

from motorsports_data_notebook.corners import (
    Corner,
    compute_curvature,
    compute_lap_distance,
    gps_to_local_xy,
    identify_corners_from_curvature,
)


class TestComputeLapDistance:
    """Tests for compute_lap_distance function."""

    def test_stationary_vehicle(self):
        """Test that stationary vehicle has zero distance."""
        timecodes = np.array([0, 100, 200, 300, 400])  # ms
        speed = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # m/s

        distance = compute_lap_distance(timecodes, speed)

        assert len(distance) == 5
        np.testing.assert_array_almost_equal(distance, [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_constant_speed(self):
        """Test distance calculation at constant speed."""
        # 10 m/s for 1 second (1000ms) = 10 meters
        timecodes = np.array([0, 500, 1000])  # ms
        speed = np.array([10.0, 10.0, 10.0])  # m/s

        distance = compute_lap_distance(timecodes, speed)

        # First point is 0, then 5m at 0.5s, then 10m at 1s
        assert distance[0] == pytest.approx(0.0)
        assert distance[1] == pytest.approx(5.0)
        assert distance[2] == pytest.approx(10.0)

    def test_accelerating_vehicle(self):
        """Test distance calculation with changing speed."""
        timecodes = np.array([0, 1000, 2000])  # ms
        speed = np.array([0.0, 10.0, 20.0])  # m/s

        distance = compute_lap_distance(timecodes, speed)

        # At t=1s: 10 m/s * 1s = 10m
        # At t=2s: 10m + 20 m/s * 1s = 30m
        assert distance[0] == pytest.approx(0.0)
        assert distance[1] == pytest.approx(10.0)
        assert distance[2] == pytest.approx(30.0)


class TestGpsToLocalXY:
    """Tests for gps_to_local_xy function."""

    def test_single_point_at_origin(self):
        """Test that single point converts to origin."""
        lat = np.array([35.0])
        lon = np.array([139.0])

        x, y = gps_to_local_xy(lat, lon)

        # Single point should be at origin (relative to itself)
        assert x[0] == pytest.approx(0.0)
        assert y[0] == pytest.approx(0.0)

    def test_north_south_movement(self):
        """Test that north-south movement changes y coordinate."""
        # Two points, same longitude, different latitude
        lat = np.array([35.0, 35.001])  # ~111m apart
        lon = np.array([139.0, 139.0])

        x, y = gps_to_local_xy(lat, lon)

        # Y should change (north-south)
        assert abs(y[1] - y[0]) > 100  # Should be roughly 111m
        # X should be approximately same
        assert abs(x[1] - x[0]) < 1

    def test_east_west_movement(self):
        """Test that east-west movement changes x coordinate."""
        # Two points, same latitude, different longitude
        lat = np.array([35.0, 35.0])
        lon = np.array([139.0, 139.001])

        x, y = gps_to_local_xy(lat, lon)

        # X should change (east-west)
        assert abs(x[1] - x[0]) > 50  # Should be roughly 91m at 35deg lat
        # Y should be approximately same
        assert abs(y[1] - y[0]) < 1


class TestComputeCurvature:
    """Tests for compute_curvature function."""

    def test_straight_line_zero_curvature(self):
        """Test that a straight line has zero curvature."""
        # Straight line along x-axis
        x = np.linspace(0, 100, 50)
        y = np.zeros(50)

        curvature, signed_curvature, radius = compute_curvature(
            x, y, pos_smooth_window=1, curv_smooth_window=1
        )

        # Curvature should be essentially zero (allowing for numerical noise)
        assert np.mean(curvature[5:-5]) < 0.001  # Exclude edge effects

    def test_circle_constant_curvature(self):
        """Test that a circle has constant curvature equal to 1/radius."""
        # Circle with radius 100m
        R = 100.0
        theta = np.linspace(0, np.pi, 100)  # Half circle
        x = R * np.cos(theta)
        y = R * np.sin(theta)

        curvature, signed_curvature, radius_out = compute_curvature(
            x, y, pos_smooth_window=1, curv_smooth_window=1
        )

        # Middle portion should have curvature ~ 1/R = 0.01
        mid_curvature = np.mean(curvature[20:80])
        assert mid_curvature == pytest.approx(1 / R, rel=0.1)

    def test_left_turn_positive_curvature(self):
        """Test that a left turn has positive signed curvature."""
        # Quarter circle turning left (counterclockwise)
        R = 50.0
        theta = np.linspace(0, np.pi / 2, 50)
        x = R * np.cos(theta)
        y = R * np.sin(theta)

        curvature, signed_curvature, radius = compute_curvature(
            x, y, pos_smooth_window=1, curv_smooth_window=1
        )

        # Signed curvature should be positive for left turn
        mid_signed = np.mean(signed_curvature[10:40])
        assert mid_signed > 0

    def test_right_turn_negative_curvature(self):
        """Test that a right turn has negative signed curvature."""
        # Quarter circle turning right (clockwise)
        R = 50.0
        theta = np.linspace(0, -np.pi / 2, 50)
        x = R * np.cos(theta)
        y = R * np.sin(theta)

        curvature, signed_curvature, radius = compute_curvature(
            x, y, pos_smooth_window=1, curv_smooth_window=1
        )

        # Signed curvature should be negative for right turn
        mid_signed = np.mean(signed_curvature[10:40])
        assert mid_signed < 0


class TestIdentifyCornersFromCurvature:
    """Tests for identify_corners_from_curvature function."""

    def test_no_corners_on_straight(self):
        """Test that no corners are detected on a straight track."""
        distance = np.linspace(0, 1000, 500)
        curvature = np.zeros(500)
        signed_curvature = np.zeros(500)

        corners = identify_corners_from_curvature(
            distance, curvature, signed_curvature, threshold=0.006
        )

        assert len(corners) == 0

    def test_single_left_corner(self):
        """Test detection of a single left corner."""
        distance = np.linspace(0, 500, 250)
        curvature = np.zeros(250)
        signed_curvature = np.zeros(250)

        # Add a corner from 100m to 200m with curvature 0.01 (100m radius)
        corner_mask = (distance >= 100) & (distance <= 200)
        curvature[corner_mask] = 0.01
        signed_curvature[corner_mask] = 0.01  # Positive = left

        corners = identify_corners_from_curvature(
            distance, curvature, signed_curvature, threshold=0.006, min_corner_length=10
        )

        assert len(corners) == 1
        assert corners[0].direction == "L"
        assert corners[0].id == 1
        assert corners[0].name == "Turn 1"
        assert corners[0].start_dist == pytest.approx(100, abs=5)
        assert corners[0].end_dist == pytest.approx(200, abs=5)

    def test_single_right_corner(self):
        """Test detection of a single right corner."""
        distance = np.linspace(0, 500, 250)
        curvature = np.zeros(250)
        signed_curvature = np.zeros(250)

        # Add a right corner from 150m to 250m
        corner_mask = (distance >= 150) & (distance <= 250)
        curvature[corner_mask] = 0.008
        signed_curvature[corner_mask] = -0.008  # Negative = right

        corners = identify_corners_from_curvature(
            distance, curvature, signed_curvature, threshold=0.006, min_corner_length=10
        )

        assert len(corners) == 1
        assert corners[0].direction == "R"

    def test_multiple_corners_numbered_sequentially(self):
        """Test that multiple corners are numbered sequentially."""
        distance = np.linspace(0, 1000, 500)
        curvature = np.zeros(500)
        signed_curvature = np.zeros(500)

        # Corner 1: 100-150m (left)
        mask1 = (distance >= 100) & (distance <= 150)
        curvature[mask1] = 0.01
        signed_curvature[mask1] = 0.01

        # Corner 2: 300-350m (right)
        mask2 = (distance >= 300) & (distance <= 350)
        curvature[mask2] = 0.01
        signed_curvature[mask2] = -0.01

        # Corner 3: 600-700m (left)
        mask3 = (distance >= 600) & (distance <= 700)
        curvature[mask3] = 0.01
        signed_curvature[mask3] = 0.01

        corners = identify_corners_from_curvature(
            distance, curvature, signed_curvature, threshold=0.006, min_corner_length=10
        )

        assert len(corners) == 3
        assert corners[0].id == 1
        assert corners[0].name == "Turn 1"
        assert corners[1].id == 2
        assert corners[1].name == "Turn 2"
        assert corners[2].id == 3
        assert corners[2].name == "Turn 3"

    def test_corner_merging_same_direction(self):
        """Test that close corners with same direction are merged."""
        distance = np.linspace(0, 500, 250)
        curvature = np.zeros(250)
        signed_curvature = np.zeros(250)

        # Two left corners very close together (should merge)
        mask1 = (distance >= 100) & (distance <= 150)
        mask2 = (distance >= 170) & (distance <= 220)  # Only 20m gap
        curvature[mask1] = 0.01
        curvature[mask2] = 0.01
        signed_curvature[mask1] = 0.01
        signed_curvature[mask2] = 0.01

        corners = identify_corners_from_curvature(
            distance, curvature, signed_curvature, threshold=0.006, min_corner_length=10, min_gap=80
        )

        # Should merge into one corner
        assert len(corners) == 1

    def test_corner_not_merged_different_direction(self):
        """Test that close corners with different directions are not merged."""
        distance = np.linspace(0, 500, 250)
        curvature = np.zeros(250)
        signed_curvature = np.zeros(250)

        # Left then right corner close together
        mask1 = (distance >= 100) & (distance <= 150)
        mask2 = (distance >= 170) & (distance <= 220)
        curvature[mask1] = 0.01
        curvature[mask2] = 0.01
        signed_curvature[mask1] = 0.01  # Left
        signed_curvature[mask2] = -0.01  # Right

        corners = identify_corners_from_curvature(
            distance, curvature, signed_curvature, threshold=0.006, min_corner_length=10, min_gap=80
        )

        # Should NOT merge - different directions
        assert len(corners) == 2
        assert corners[0].direction == "L"
        assert corners[1].direction == "R"

    def test_corner_too_short_filtered(self):
        """Test that corners shorter than min_corner_length are filtered out."""
        distance = np.linspace(0, 500, 500)  # 1m resolution
        curvature = np.zeros(500)
        signed_curvature = np.zeros(500)

        # Very short corner (5m) - should be filtered
        mask = (distance >= 100) & (distance <= 105)
        curvature[mask] = 0.01
        signed_curvature[mask] = 0.01

        corners = identify_corners_from_curvature(
            distance, curvature, signed_curvature, threshold=0.006, min_corner_length=15
        )

        assert len(corners) == 0

    def test_apex_at_max_curvature(self):
        """Test that apex is correctly identified at maximum curvature point."""
        distance = np.linspace(0, 300, 150)
        curvature = np.zeros(150)
        signed_curvature = np.zeros(150)

        # Corner with varying curvature, peak at 150m
        corner_mask = (distance >= 100) & (distance <= 200)
        corner_indices = np.where(corner_mask)[0]

        for idx in corner_indices:
            d = distance[idx]
            # Curvature peaks at d=150
            curvature[idx] = 0.01 * (1 - abs(d - 150) / 50)
            signed_curvature[idx] = curvature[idx]

        corners = identify_corners_from_curvature(
            distance, curvature, signed_curvature, threshold=0.005, min_corner_length=10
        )

        assert len(corners) == 1
        assert corners[0].apex_dist == pytest.approx(150, abs=5)


class TestCornerDataclass:
    """Tests for Corner dataclass."""

    def test_length_property(self):
        """Test that length property returns correct value."""
        corner = Corner(
            id=1,
            name="Turn 1",
            direction="L",
            start_idx=0,
            end_idx=100,
            start_dist=100.0,
            end_dist=250.0,
            apex_idx=50,
            apex_dist=175.0,
            max_curvature=0.01,
        )

        assert corner.length == 150.0

    def test_radius_property(self):
        """Test that radius property returns 1/curvature."""
        corner = Corner(
            id=1,
            name="Turn 1",
            direction="L",
            start_idx=0,
            end_idx=100,
            start_dist=100.0,
            end_dist=250.0,
            apex_idx=50,
            apex_dist=175.0,
            max_curvature=0.01,  # 100m radius
        )

        assert corner.radius == pytest.approx(100.0)

    def test_radius_capped_for_zero_curvature(self):
        """Test that radius is capped at 10000m for near-zero curvature."""
        corner = Corner(
            id=1,
            name="Turn 1",
            direction="L",
            start_idx=0,
            end_idx=100,
            start_dist=100.0,
            end_dist=250.0,
            apex_idx=50,
            apex_dist=175.0,
            max_curvature=0.0,  # Zero curvature (straight)
        )

        assert corner.radius == 10000.0
