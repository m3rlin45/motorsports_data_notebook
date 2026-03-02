"""Tests for the driver_analysis module.

Uses synthetic data to test throttle acceptance analysis.
"""

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.corners import Corner
from motorsports_data_notebook.driver_analysis import (
    find_throttle_acceptance,
    find_throttle_acceptance_from_arrays,
    prepare_throttle_acceptance,
)

# Test channel names mapping (matches the test data column names)
TEST_CHANNEL_NAMES = {
    "throttle": "PPS",
    "brake": "BrakePress",
    "lateral_g": "LateralAcc",
}


class TestFindThrottleAcceptance:
    """Tests for find_throttle_acceptance function."""

    def create_corner(
        self,
        start_dist=100.0,
        end_dist=200.0,
        apex_dist=150.0,
    ) -> Corner:
        """Helper to create a test corner."""
        return Corner(
            id=1,
            name="Turn 1",
            direction="L",
            start_idx=100,
            end_idx=200,
            start_dist=start_dist,
            end_dist=end_dist,
            apex_idx=150,
            apex_dist=apex_dist,
            max_curvature=0.01,
        )

    def test_finds_throttle_acceptance_basic(self):
        """Test basic throttle acceptance detection."""
        corner = self.create_corner(start_dist=100.0, end_dist=300.0, apex_dist=200.0)

        # Create lap data with:
        # - Corner from 100m to 300m
        # - Peak lateral G in corner
        # - Full throttle from 250m onwards
        n_points = 200
        lap_data = pd.DataFrame(
            {
                "distance_m": np.linspace(50, 350, n_points),
                "timecodes": np.linspace(0, 10000, n_points),  # 10 seconds
                "PPS": np.zeros(n_points),  # Throttle %
                "LateralAcc": np.zeros(n_points),  # Lateral G
            }
        )

        # Add lateral G profile (peaks in corner, decreases towards exit)
        for i, row in lap_data.iterrows():
            d = row["distance_m"]
            if 100 <= d <= 300:
                # Peak at 150m, decreases towards exit
                if d <= 200:
                    lap_data.loc[i, "LateralAcc"] = 1.5  # Peak in corner
                else:
                    # Linear decrease from apex to exit
                    progress = (d - 200) / 100
                    lap_data.loc[i, "LateralAcc"] = 1.5 * (1 - progress * 0.7)

        # Add throttle (full throttle at 250m which is 50% through exit zone)
        throttle_start_idx = lap_data[lap_data["distance_m"] >= 250].index[0]
        lap_data.loc[throttle_start_idx:, "PPS"] = 100

        result = find_throttle_acceptance(
            lap_data,
            corner,
            TEST_CHANNEL_NAMES,
            throttle_threshold=98.0,
            sustain_time_ms=500,
            smoothing_window=5,
        )

        assert result is not None
        assert "throttle_acceptance_pct" in result
        assert "lateral_g_at_throttle" in result
        assert "peak_lateral_g" in result
        assert "full_throttle_dist" in result

        # Throttle acceptance should be > 0 (driver is still cornering when going full throttle)
        assert result["throttle_acceptance_pct"] > 0

    def test_returns_none_when_no_full_throttle(self):
        """Test that None is returned when full throttle is never sustained."""
        corner = self.create_corner(start_dist=100.0, end_dist=200.0, apex_dist=150.0)

        lap_data = pd.DataFrame(
            {
                "distance_m": np.linspace(50, 250, 100),
                "timecodes": np.linspace(0, 5000, 100),
                "PPS": np.ones(100) * 50,  # Only 50% throttle
                "LateralAcc": np.ones(100) * 1.0,
            }
        )

        result = find_throttle_acceptance(
            lap_data, corner, TEST_CHANNEL_NAMES, throttle_threshold=98.0
        )

        assert result is None

    def test_returns_none_when_no_corner_data(self):
        """Test that None is returned when no data falls within corner bounds."""
        corner = self.create_corner(start_dist=100.0, end_dist=200.0, apex_dist=150.0)

        # Data outside corner bounds
        lap_data = pd.DataFrame(
            {
                "distance_m": np.linspace(300, 400, 50),
                "timecodes": np.linspace(0, 2500, 50),
                "PPS": np.ones(50) * 100,
                "LateralAcc": np.ones(50) * 1.0,
            }
        )

        result = find_throttle_acceptance(lap_data, corner, TEST_CHANNEL_NAMES)

        assert result is None

    def test_returns_none_for_negligible_lateral_g(self):
        """Test that None is returned when lateral G is negligible."""
        corner = self.create_corner(start_dist=100.0, end_dist=200.0, apex_dist=150.0)

        lap_data = pd.DataFrame(
            {
                "distance_m": np.linspace(50, 250, 100),
                "timecodes": np.linspace(0, 5000, 100),
                "PPS": np.ones(100) * 100,
                "LateralAcc": np.ones(100) * 0.05,  # Very low lateral G
            }
        )

        result = find_throttle_acceptance(lap_data, corner, TEST_CHANNEL_NAMES)

        assert result is None

    def test_throttle_must_be_sustained(self):
        """Test that throttle must be sustained for the required time."""
        corner = self.create_corner(start_dist=100.0, end_dist=300.0, apex_dist=200.0)

        n_points = 100
        lap_data = pd.DataFrame(
            {
                "distance_m": np.linspace(50, 350, n_points),
                "timecodes": np.linspace(0, 5000, n_points),
                "PPS": np.zeros(n_points),
                "LateralAcc": np.ones(n_points) * 1.0,
            }
        )

        # Brief throttle blip (not sustained)
        throttle_idx = lap_data[lap_data["distance_m"] >= 250].index[:3]  # Only 3 points
        lap_data.loc[throttle_idx, "PPS"] = 100

        result = find_throttle_acceptance(lap_data, corner, TEST_CHANNEL_NAMES, sustain_time_ms=500)

        # Should be None because throttle wasn't sustained for 500ms
        assert result is None

    def test_higher_lateral_g_gives_higher_acceptance_pct(self):
        """Test that higher lateral G at throttle gives higher acceptance percentage."""
        corner = self.create_corner(start_dist=100.0, end_dist=300.0, apex_dist=200.0)

        def create_data(lateral_g_at_throttle_point: float):
            n_points = 100
            lap_data = pd.DataFrame(
                {
                    "distance_m": np.linspace(50, 350, n_points),
                    "timecodes": np.linspace(0, 10000, n_points),
                    "PPS": np.zeros(n_points),
                    "LateralAcc": np.zeros(n_points),
                }
            )

            # Peak lateral G = 2.0 at apex
            for i, row in lap_data.iterrows():
                d = row["distance_m"]
                if 100 <= d <= 200:  # Entry to apex
                    lap_data.loc[i, "LateralAcc"] = 2.0
                elif 200 < d <= 300:  # Exit
                    # Linearly decrease
                    progress = (d - 200) / 100
                    lap_data.loc[i, "LateralAcc"] = 2.0 * (1 - progress * 0.8)

            # Override lateral G at throttle point specifically
            throttle_start_idx = lap_data[lap_data["distance_m"] >= 250].index[0]
            lap_data.loc[throttle_start_idx:, "PPS"] = 100
            lap_data.loc[throttle_start_idx, "LateralAcc"] = lateral_g_at_throttle_point

            return lap_data

        # Test with different lateral G values at throttle point
        result_low = find_throttle_acceptance(
            create_data(0.5), corner, TEST_CHANNEL_NAMES, sustain_time_ms=500, smoothing_window=3
        )
        result_high = find_throttle_acceptance(
            create_data(1.5), corner, TEST_CHANNEL_NAMES, sustain_time_ms=500, smoothing_window=3
        )

        assert result_low is not None
        assert result_high is not None
        assert result_high["throttle_acceptance_pct"] > result_low["throttle_acceptance_pct"]

    def test_auto_detect_zero_to_one_scale(self):
        """Test that throttle acceptance works with 0-1 scale data (iRacing)."""
        corner = self.create_corner(start_dist=100.0, end_dist=300.0, apex_dist=200.0)

        n_points = 200
        lap_data = pd.DataFrame(
            {
                "distance_m": np.linspace(50, 350, n_points),
                "timecodes": np.linspace(0, 10000, n_points),
                "PPS": np.zeros(n_points),  # 0-1 scale throttle
                "LateralAcc": np.zeros(n_points),
            }
        )

        # Add lateral G profile
        for i, row in lap_data.iterrows():
            d = row["distance_m"]
            if 100 <= d <= 300:
                if d <= 200:
                    lap_data.loc[i, "LateralAcc"] = 1.5
                else:
                    progress = (d - 200) / 100
                    lap_data.loc[i, "LateralAcc"] = 1.5 * (1 - progress * 0.7)

        # Full throttle at 250m in 0-1 scale
        throttle_start_idx = lap_data[lap_data["distance_m"] >= 250].index[0]
        lap_data.loc[throttle_start_idx:, "PPS"] = 1.0  # 0-1 scale

        # Auto-detect (no explicit throttle_threshold)
        result = find_throttle_acceptance(
            lap_data,
            corner,
            TEST_CHANNEL_NAMES,
            sustain_time_ms=500,
            smoothing_window=5,
        )

        assert result is not None
        assert result["throttle_acceptance_pct"] > 0

    def test_adapts_threshold_for_capped_sensor(self):
        """Test that threshold adapts when sensor max is below 100%."""
        corner = self.create_corner(start_dist=100.0, end_dist=300.0, apex_dist=200.0)

        n_points = 200
        lap_data = pd.DataFrame(
            {
                "distance_m": np.linspace(50, 350, n_points),
                "timecodes": np.linspace(0, 10000, n_points),
                "PPS": np.zeros(n_points),
                "LateralAcc": np.zeros(n_points),
            }
        )

        # Lateral G profile
        for i, row in lap_data.iterrows():
            d = row["distance_m"]
            if 100 <= d <= 300:
                if d <= 200:
                    lap_data.loc[i, "LateralAcc"] = 1.5
                else:
                    progress = (d - 200) / 100
                    lap_data.loc[i, "LateralAcc"] = 1.5 * (1 - progress * 0.7)

        # Sensor caps at 98.7% — full throttle plateau at 97%
        throttle_start_idx = lap_data[lap_data["distance_m"] >= 250].index[0]
        lap_data.loc[throttle_start_idx:, "PPS"] = 97.0
        # A few samples at the sensor max
        lap_data.loc[throttle_start_idx, "PPS"] = 98.7

        # With fixed 98% threshold, this would fail (97 < 98)
        # With adaptive threshold, effective = 98.7 * 0.98 = 96.7, so 97 passes
        result = find_throttle_acceptance(
            lap_data,
            corner,
            TEST_CHANNEL_NAMES,
            throttle_threshold=98.0,
            sustain_time_ms=500,
            smoothing_window=5,
        )

        assert result is not None
        assert result["throttle_acceptance_pct"] > 0

    def test_smoothing_reduces_noise(self):
        """Test that smoothing window affects the result."""
        corner = self.create_corner(start_dist=100.0, end_dist=300.0, apex_dist=200.0)

        n_points = 200
        np.random.seed(42)

        lap_data = pd.DataFrame(
            {
                "distance_m": np.linspace(50, 350, n_points),
                "timecodes": np.linspace(0, 10000, n_points),
                "PPS": np.zeros(n_points),
                "LateralAcc": np.ones(n_points) * 1.5 + np.random.normal(0, 0.3, n_points),
            }
        )

        throttle_start_idx = lap_data[lap_data["distance_m"] >= 250].index[0]
        lap_data.loc[throttle_start_idx:, "PPS"] = 100

        result_no_smooth = find_throttle_acceptance(
            lap_data.copy(), corner, TEST_CHANNEL_NAMES, smoothing_window=1
        )
        result_smoothed = find_throttle_acceptance(
            lap_data.copy(), corner, TEST_CHANNEL_NAMES, smoothing_window=25
        )

        # Both should return results
        assert result_no_smooth is not None
        assert result_smoothed is not None

        # Smoothed version should have peak_lateral_g closer to 1.5 (the true value)
        assert abs(result_smoothed["peak_lateral_g"] - 1.5) < abs(
            result_no_smooth["peak_lateral_g"] - 1.5
        )


class TestPrepareThrottleAcceptance:
    """Tests for prepare_throttle_acceptance function."""

    def test_returns_smoothed_and_threshold(self):
        """Test that it returns smoothed lateral G and effective threshold."""
        throttle = np.ones(100) * 100.0
        lateral_g = np.ones(100) * 1.5

        smoothed, eff_thresh = prepare_throttle_acceptance(
            throttle, lateral_g, throttle_threshold=98.0, smoothing_window=5
        )

        assert len(smoothed) == len(lateral_g)
        assert eff_thresh == pytest.approx(98.0)

    def test_adapts_threshold_for_capped_sensor(self):
        """Test threshold adaptation when sensor max is below 100%."""
        throttle = np.ones(100) * 97.0
        throttle[0] = 98.7  # Sensor max
        lateral_g = np.ones(100) * 1.0

        _, eff_thresh = prepare_throttle_acceptance(throttle, lateral_g, throttle_threshold=98.0)

        # effective = 98.7 * (98.0 / 100.0) = 96.726
        assert eff_thresh == pytest.approx(98.7 * 0.98, rel=1e-3)

    def test_auto_detects_scale(self):
        """Test auto-detection of 0-1 vs 0-100 scale."""
        throttle_01 = np.ones(100) * 1.0
        throttle_100 = np.ones(100) * 100.0
        lateral_g = np.ones(100) * 1.0

        _, thresh_01 = prepare_throttle_acceptance(throttle_01, lateral_g)
        _, thresh_100 = prepare_throttle_acceptance(throttle_100, lateral_g)

        # 0-1 scale: threshold = 0.98 * 1.0 = 0.98
        assert thresh_01 == pytest.approx(0.98, rel=1e-2)
        # 0-100 scale: threshold = 0.98 * 100 = 98.0
        assert thresh_100 == pytest.approx(98.0, rel=1e-2)

    def test_smoothing_window_affects_output(self):
        """Test that larger smoothing window produces smoother output."""
        np.random.seed(42)
        lateral_g = np.ones(200) * 1.5 + np.random.normal(0, 0.3, 200)
        throttle = np.ones(200) * 100.0

        smoothed_small, _ = prepare_throttle_acceptance(throttle, lateral_g, smoothing_window=3)
        smoothed_large, _ = prepare_throttle_acceptance(throttle, lateral_g, smoothing_window=25)

        # Larger window should produce smaller variance
        assert np.std(smoothed_large) < np.std(smoothed_small)


class TestFindThrottleAcceptanceFromArrays:
    """Tests for find_throttle_acceptance_from_arrays function."""

    def create_corner(
        self,
        start_dist=100.0,
        end_dist=200.0,
        apex_dist=150.0,
    ) -> Corner:
        """Helper to create a test corner."""
        return Corner(
            id=1,
            name="Turn 1",
            direction="L",
            start_idx=100,
            end_idx=200,
            start_dist=start_dist,
            end_dist=end_dist,
            apex_idx=150,
            apex_dist=apex_dist,
            max_curvature=0.01,
        )

    def _make_arrays(self, n_points=200, throttle_start_dist=250.0, peak_lat_g=1.5):
        """Helper to create test arrays matching DataFrame-based tests."""
        distance = np.linspace(50, 350, n_points)
        timecodes = np.linspace(0, 10000, n_points)
        throttle = np.zeros(n_points)
        lateral_g = np.zeros(n_points)

        for i in range(n_points):
            d = distance[i]
            if 100 <= d <= 300:
                if d <= 200:
                    lateral_g[i] = peak_lat_g
                else:
                    progress = (d - 200) / 100
                    lateral_g[i] = peak_lat_g * (1 - progress * 0.7)

        throttle_start_idx = np.searchsorted(distance, throttle_start_dist)
        throttle[throttle_start_idx:] = 100.0

        return distance, timecodes, throttle, lateral_g

    def test_basic_detection(self):
        """Test basic throttle acceptance detection from arrays."""
        corner = self.create_corner(start_dist=100.0, end_dist=300.0, apex_dist=200.0)
        distance, timecodes, throttle, lateral_g = self._make_arrays()

        result = find_throttle_acceptance_from_arrays(
            distance,
            timecodes,
            throttle,
            lateral_g,
            corner,
            throttle_threshold=98.0,
            sustain_time_ms=500,
            smoothing_window=5,
        )

        assert result is not None
        assert result["throttle_acceptance_pct"] > 0
        assert "lateral_g_at_throttle" in result
        assert "peak_lateral_g" in result
        assert "full_throttle_dist" in result

    def test_matches_dataframe_version(self):
        """Test that array version gives same result as DataFrame version."""
        corner = self.create_corner(start_dist=100.0, end_dist=300.0, apex_dist=200.0)
        distance, timecodes, throttle, lateral_g = self._make_arrays()

        # DataFrame version
        lap_data = pd.DataFrame(
            {
                "distance_m": distance,
                "timecodes": timecodes,
                "PPS": throttle,
                "LateralAcc": lateral_g,
            }
        )
        result_df = find_throttle_acceptance(
            lap_data,
            corner,
            TEST_CHANNEL_NAMES,
            throttle_threshold=98.0,
            sustain_time_ms=500,
            smoothing_window=5,
        )

        # Array version
        result_arr = find_throttle_acceptance_from_arrays(
            distance,
            timecodes,
            throttle,
            lateral_g,
            corner,
            throttle_threshold=98.0,
            sustain_time_ms=500,
            smoothing_window=5,
        )

        assert result_df is not None
        assert result_arr is not None
        assert result_arr["throttle_acceptance_pct"] == pytest.approx(
            result_df["throttle_acceptance_pct"], rel=0.05
        )

    def test_with_precomputed_data(self):
        """Test using prepare_throttle_acceptance + find_throttle_acceptance_from_arrays."""
        corner = self.create_corner(start_dist=100.0, end_dist=300.0, apex_dist=200.0)
        distance, timecodes, throttle, lateral_g = self._make_arrays()

        smoothed, eff_thresh = prepare_throttle_acceptance(
            throttle,
            lateral_g,
            throttle_threshold=98.0,
            smoothing_window=5,
        )

        result = find_throttle_acceptance_from_arrays(
            distance,
            timecodes,
            throttle,
            lateral_g,
            corner,
            smoothed_lateral_g=smoothed,
            effective_threshold=eff_thresh,
            sustain_time_ms=500,
        )

        assert result is not None
        assert result["throttle_acceptance_pct"] > 0

    def test_returns_none_no_corner_data(self):
        """Test None when no data in corner bounds."""
        corner = self.create_corner(start_dist=100.0, end_dist=200.0, apex_dist=150.0)

        distance = np.linspace(300, 400, 50)
        timecodes = np.linspace(0, 2500, 50)
        throttle = np.ones(50) * 100.0
        lateral_g = np.ones(50) * 1.0

        result = find_throttle_acceptance_from_arrays(
            distance,
            timecodes,
            throttle,
            lateral_g,
            corner,
        )

        assert result is None

    def test_returns_none_negligible_lateral_g(self):
        """Test None for negligible lateral G."""
        corner = self.create_corner(start_dist=100.0, end_dist=200.0, apex_dist=150.0)

        distance = np.linspace(50, 250, 100)
        timecodes = np.linspace(0, 5000, 100)
        throttle = np.ones(100) * 100.0
        lateral_g = np.ones(100) * 0.05

        result = find_throttle_acceptance_from_arrays(
            distance,
            timecodes,
            throttle,
            lateral_g,
            corner,
        )

        assert result is None
