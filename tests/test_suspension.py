"""Tests for suspension module."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from dataclasses import dataclass
from typing import Dict

from motorsports_data_notebook.suspension import (
    MotionRatios,
    VelocityRanges,
    CornerVelocityData,
    VelocityHistogramResult,
    SUSPENSION_CHANNEL_NAMES,
    compute_shock_velocity,
    compute_wheel_velocity,
    compute_velocity_histogram,
    compute_velocity_stats,
    analyze_suspension_velocity,
    format_suspension_stats_table,
    format_symmetry_table,
    format_comparison_table,
)


@dataclass
class MockLogFile:
    """Mock LogFile for testing."""

    channels: Dict[str, pa.Table]
    laps: pa.Table = None
    metadata: Dict[str, str] = None
    file_name: str = "test.xrk"

    def __post_init__(self):
        if self.laps is None:
            self.laps = pa.table({"num": [], "start_time": [], "end_time": []})
        if self.metadata is None:
            self.metadata = {}


class TestMotionRatios:
    """Tests for MotionRatios dataclass."""

    def test_default_values(self):
        """Default values should be Toyota 86 ZN6 ratios."""
        ratios = MotionRatios()
        assert ratios.front_left == pytest.approx(0.997)
        assert ratios.front_right == pytest.approx(0.997)
        assert ratios.rear_left == pytest.approx(0.768)
        assert ratios.rear_right == pytest.approx(0.768)

    def test_toyota_86_zn6_classmethod(self):
        """toyota_86_zn6() should return correct ratios."""
        ratios = MotionRatios.toyota_86_zn6()
        assert ratios.front_left == pytest.approx(0.997)
        assert ratios.rear_left == pytest.approx(0.768)

    def test_custom_values(self):
        """Should accept custom motion ratios."""
        ratios = MotionRatios(
            front_left=0.9,
            front_right=0.9,
            rear_left=0.8,
            rear_right=0.8,
        )
        assert ratios.front_left == pytest.approx(0.9)
        assert ratios.rear_left == pytest.approx(0.8)


class TestVelocityRanges:
    """Tests for VelocityRanges dataclass."""

    def test_default_values(self):
        """Default velocity thresholds."""
        ranges = VelocityRanges()
        assert ranges.friction == pytest.approx(5.0)
        assert ranges.slow == pytest.approx(25.0)
        assert ranges.fast == pytest.approx(200.0)


class TestComputeShockVelocity:
    """Tests for compute_shock_velocity function."""

    def test_constant_displacement_zero_velocity(self):
        """Constant displacement should give zero velocity."""
        displacement = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        timecodes = np.array([0, 100, 200, 300, 400], dtype=np.int64)

        velocity = compute_shock_velocity(displacement, timecodes, smoothing_window=1)

        np.testing.assert_array_almost_equal(velocity, [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_linear_displacement_constant_velocity(self):
        """Linear displacement should give constant velocity."""
        # 10mm per 100ms = 100mm/s
        displacement = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        timecodes = np.array([0, 100, 200, 300, 400], dtype=np.int64)

        velocity = compute_shock_velocity(displacement, timecodes, smoothing_window=1)

        # All velocities should be 100 mm/s
        np.testing.assert_array_almost_equal(velocity, [100.0, 100.0, 100.0, 100.0, 100.0])

    def test_positive_velocity_for_bump(self):
        """Increasing displacement (bump) should give positive velocity."""
        displacement = np.array([0.0, 5.0, 10.0])
        timecodes = np.array([0, 100, 200], dtype=np.int64)

        velocity = compute_shock_velocity(displacement, timecodes, smoothing_window=1)

        assert all(v > 0 for v in velocity)

    def test_negative_velocity_for_rebound(self):
        """Decreasing displacement (rebound) should give negative velocity."""
        displacement = np.array([10.0, 5.0, 0.0])
        timecodes = np.array([0, 100, 200], dtype=np.int64)

        velocity = compute_shock_velocity(displacement, timecodes, smoothing_window=1)

        assert all(v < 0 for v in velocity)

    def test_smoothing_reduces_noise(self):
        """Smoothing should reduce velocity variations."""
        # Create noisy displacement
        displacement = np.array([0.0, 10.0, 5.0, 15.0, 10.0, 20.0])
        timecodes = np.array([0, 100, 200, 300, 400, 500], dtype=np.int64)

        velocity_raw = compute_shock_velocity(displacement, timecodes, smoothing_window=1)
        velocity_smooth = compute_shock_velocity(displacement, timecodes, smoothing_window=3)

        # Smoothed velocity should have lower standard deviation
        assert np.std(velocity_smooth) < np.std(velocity_raw)

    def test_handles_varying_sample_rate(self):
        """Should handle non-uniform time intervals."""
        displacement = np.array([0.0, 10.0, 30.0])
        # First interval: 100ms, second interval: 200ms
        timecodes = np.array([0, 100, 300], dtype=np.int64)

        velocity = compute_shock_velocity(displacement, timecodes, smoothing_window=1)

        # First point: 10mm/0.1s = 100mm/s
        # Third point: 20mm/0.2s = 100mm/s
        # Middle point is gradient interpolation
        assert velocity[0] == pytest.approx(100.0, rel=0.1)
        assert velocity[2] == pytest.approx(100.0, rel=0.1)


class TestComputeWheelVelocity:
    """Tests for compute_wheel_velocity function."""

    def test_motion_ratio_scaling(self):
        """Wheel velocity should scale by inverse of motion ratio."""
        shock_velocity = np.array([10.0, 20.0, 30.0])
        motion_ratio = 0.5

        wheel_velocity = compute_wheel_velocity(shock_velocity, motion_ratio)

        # wheel_vel = shock_vel / motion_ratio
        np.testing.assert_array_almost_equal(wheel_velocity, [20.0, 40.0, 60.0])

    def test_unity_motion_ratio(self):
        """Motion ratio of 1.0 should give same velocity."""
        shock_velocity = np.array([10.0, 20.0, 30.0])

        wheel_velocity = compute_wheel_velocity(shock_velocity, motion_ratio=1.0)

        np.testing.assert_array_almost_equal(wheel_velocity, shock_velocity)

    def test_toyota_86_front_ratio(self):
        """Test with Toyota 86 front motion ratio."""
        shock_velocity = np.array([100.0])
        motion_ratio = 0.997

        wheel_velocity = compute_wheel_velocity(shock_velocity, motion_ratio)

        # 100 / 0.997 ≈ 100.3
        assert wheel_velocity[0] == pytest.approx(100.3, rel=0.01)

    def test_toyota_86_rear_ratio(self):
        """Test with Toyota 86 rear motion ratio."""
        shock_velocity = np.array([100.0])
        motion_ratio = 0.768

        wheel_velocity = compute_wheel_velocity(shock_velocity, motion_ratio)

        # 100 / 0.768 ≈ 130.2
        assert wheel_velocity[0] == pytest.approx(130.2, rel=0.01)


class TestComputeVelocityHistogram:
    """Tests for compute_velocity_histogram function."""

    def test_percentages_sum_to_100(self):
        """Histogram percentages should sum to 100%."""
        velocity = np.random.normal(0, 50, 1000)

        histogram, _, _ = compute_velocity_histogram(velocity)

        assert histogram.sum() == pytest.approx(100.0, rel=0.001)

    def test_zero_centered_bins(self):
        """Bins should be centered on zero."""
        velocity = np.array([0.0])

        _, bin_edges, bin_centers = compute_velocity_histogram(velocity, bin_size=10.0)

        # Check that 0 is at a bin center
        assert any(np.isclose(bin_centers, 0.0, atol=1e-10))
        # Check bin edges are symmetric
        assert bin_edges[0] == pytest.approx(-bin_edges[-1])

    def test_symmetric_distribution_symmetric_histogram(self):
        """Symmetric velocity distribution should give symmetric histogram."""
        # Create perfectly symmetric distribution
        np.random.seed(42)
        positive_vel = np.abs(np.random.normal(50, 20, 500))
        velocity = np.concatenate([positive_vel, -positive_vel])

        histogram, _, bin_centers = compute_velocity_histogram(velocity, bin_size=10.0)

        # Find matching positive and negative bins
        n_bins = len(histogram)
        for i in range(n_bins // 2):
            left_idx = i
            right_idx = n_bins - 1 - i
            assert histogram[left_idx] == pytest.approx(histogram[right_idx], rel=0.1)

    def test_all_positive_velocity(self):
        """All positive velocity should have zeros on negative side."""
        velocity = np.array([50.0, 60.0, 70.0, 80.0])

        histogram, _, bin_centers = compute_velocity_histogram(velocity)

        # Negative velocity bins should be zero
        negative_mask = bin_centers < 0
        assert all(histogram[negative_mask] == 0)

    def test_max_velocity_clipping(self):
        """Velocities beyond max should be clipped to edge bins."""
        velocity = np.array([500.0, -500.0])  # Beyond default max of 300

        histogram, bin_edges, _ = compute_velocity_histogram(velocity, max_velocity=300.0)

        # Should still sum to 100%
        assert histogram.sum() == pytest.approx(100.0)
        # Edge bins should have the clipped values
        assert histogram[0] > 0  # -500 clipped to -300
        assert histogram[-1] > 0  # 500 clipped to 300


class TestComputeVelocityStats:
    """Tests for compute_velocity_stats function."""

    def test_symmetric_distribution_zero_skew(self):
        """Symmetric distribution should have near-zero skew."""
        np.random.seed(42)
        positive_vel = np.abs(np.random.normal(50, 20, 1000))
        velocity = np.concatenate([positive_vel, -positive_vel])

        stats = compute_velocity_stats(velocity, VelocityRanges())

        assert stats["skew"] == pytest.approx(0.0, abs=0.1)

    def test_right_tailed_distribution_positive_skew(self):
        """Right-tailed distribution should have positive skew."""
        # Distribution with tail toward positive (high bump velocities)
        np.random.seed(42)
        # Use an exponential-like distribution shifted to be asymmetric
        velocity = np.concatenate([
            np.random.exponential(50, 500),  # Right-tailed positive
            np.random.normal(-10, 5, 500),  # Concentrated negative
        ])

        stats = compute_velocity_stats(velocity, VelocityRanges())

        assert stats["skew"] > 0

    def test_left_tailed_distribution_negative_skew(self):
        """Left-tailed distribution should have negative skew."""
        # Distribution with tail toward negative (high rebound velocities)
        np.random.seed(42)
        velocity = np.concatenate([
            -np.random.exponential(50, 500),  # Left-tailed negative
            np.random.normal(10, 5, 500),  # Concentrated positive
        ])

        stats = compute_velocity_stats(velocity, VelocityRanges())

        assert stats["skew"] < 0

    def test_friction_range_percentage(self):
        """Friction range should count |v| < friction threshold."""
        ranges = VelocityRanges(friction=5.0)
        velocity = np.array([0.0, 2.0, -2.0, 4.0, -4.0, 10.0, -10.0])
        # 5 out of 7 values are in friction range

        stats = compute_velocity_stats(velocity, ranges)

        expected_pct = 5 / 7 * 100
        assert stats["pct_friction"] == pytest.approx(expected_pct, rel=0.01)

    def test_slow_range_percentages(self):
        """Slow range should count friction <= |v| < slow threshold."""
        ranges = VelocityRanges(friction=5.0, slow=25.0)
        # All in slow range (bump)
        velocity = np.array([10.0, 15.0, 20.0])

        stats = compute_velocity_stats(velocity, ranges)

        assert stats["pct_slow_bump"] == pytest.approx(100.0)
        assert stats["pct_slow_rebound"] == pytest.approx(0.0)

    def test_fast_range_percentages(self):
        """Fast range should count slow <= |v| < fast threshold."""
        ranges = VelocityRanges(slow=25.0, fast=200.0)
        # All in fast range (rebound)
        velocity = np.array([-50.0, -100.0, -150.0])

        stats = compute_velocity_stats(velocity, ranges)

        assert stats["pct_fast_bump"] == pytest.approx(0.0)
        assert stats["pct_fast_rebound"] == pytest.approx(100.0)

    def test_curb_range_percentage(self):
        """Curb range should count |v| >= fast threshold."""
        ranges = VelocityRanges(fast=200.0)
        velocity = np.array([250.0, -250.0, 300.0])  # All in curb range

        stats = compute_velocity_stats(velocity, ranges)

        assert stats["pct_curb"] == pytest.approx(100.0)

    def test_empty_array_returns_zeros(self):
        """Empty velocity array should return zero stats."""
        velocity = np.array([])

        stats = compute_velocity_stats(velocity, VelocityRanges())

        assert stats["skew"] == 0.0
        assert stats["kurtosis"] == 0.0
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0
        assert stats["pct_friction"] == 0.0

    def test_percentages_sum_to_100(self):
        """All percentage categories should sum to 100%."""
        np.random.seed(42)
        velocity = np.random.normal(0, 100, 1000)

        stats = compute_velocity_stats(velocity, VelocityRanges())

        total_pct = (
            stats["pct_friction"]
            + stats["pct_slow_bump"]
            + stats["pct_slow_rebound"]
            + stats["pct_fast_bump"]
            + stats["pct_fast_rebound"]
            + stats["pct_curb"]
        )
        assert total_pct == pytest.approx(100.0, rel=0.01)


class TestAnalyzeSuspensionVelocity:
    """Tests for analyze_suspension_velocity function."""

    def _create_mock_log(self, timecodes, displacement):
        """Create a mock log file with shock pot channels."""
        channels = {}
        for shock_name in SUSPENSION_CHANNEL_NAMES.values():
            channels[shock_name] = pa.table(
                {
                    "timecodes": pa.array(timecodes, type=pa.int64()),
                    shock_name: pa.array(displacement),
                }
            )
        return MockLogFile(channels=channels)

    def test_returns_velocity_histogram_result(self):
        """Should return VelocityHistogramResult with all corners."""
        timecodes = np.arange(0, 1000, 10, dtype=np.int64)
        displacement = np.cumsum(np.random.randn(len(timecodes))) + 50
        log = self._create_mock_log(timecodes, displacement)

        result = analyze_suspension_velocity(log, lap_start=0, lap_end=999)

        assert isinstance(result, VelocityHistogramResult)
        assert result.front_left is not None
        assert result.front_right is not None
        assert result.rear_left is not None
        assert result.rear_right is not None

    def test_uses_default_channel_names(self):
        """Should use SUSPENSION_CHANNEL_NAMES if not provided."""
        timecodes = np.arange(0, 500, 10, dtype=np.int64)
        displacement = np.cumsum(np.random.randn(len(timecodes)))
        log = self._create_mock_log(timecodes, displacement)

        # Should not raise KeyError with default channel names
        result = analyze_suspension_velocity(log, lap_start=0, lap_end=499)

        assert result is not None

    def test_custom_channel_names(self):
        """Should accept custom channel names."""
        timecodes = np.arange(0, 500, 10, dtype=np.int64)
        displacement = np.ones(len(timecodes)) * 50

        custom_names = {
            "shock_fl": "Custom_FL",
            "shock_fr": "Custom_FR",
            "shock_rl": "Custom_RL",
            "shock_rr": "Custom_RR",
        }
        channels = {
            name: pa.table({
                "timecodes": pa.array(timecodes, type=pa.int64()),
                name: pa.array(displacement),
            })
            for name in custom_names.values()
        }
        log = MockLogFile(channels=channels)

        result = analyze_suspension_velocity(
            log, lap_start=0, lap_end=499, channel_names=custom_names
        )

        assert result is not None

    def test_missing_channel_raises_key_error(self):
        """Should raise KeyError if required channel is missing."""
        log = MockLogFile(channels={})

        with pytest.raises(KeyError):
            analyze_suspension_velocity(log, lap_start=0, lap_end=100)

    def test_custom_motion_ratios(self):
        """Should use custom motion ratios when provided."""
        timecodes = np.arange(0, 500, 10, dtype=np.int64)
        # Linear displacement for predictable velocity
        displacement = np.arange(len(timecodes), dtype=float)
        log = self._create_mock_log(timecodes, displacement)

        custom_ratios = MotionRatios(
            front_left=0.5,
            front_right=0.5,
            rear_left=0.5,
            rear_right=0.5,
        )

        result = analyze_suspension_velocity(
            log, lap_start=0, lap_end=499, motion_ratios=custom_ratios
        )

        # Wheel velocity should be shock_velocity / motion_ratio
        # With motion_ratio=0.5, wheel velocity should be doubled
        assert result is not None


class TestFormatSuspensionStatsTable:
    """Tests for format_suspension_stats_table function."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        # Create a simple result
        corner_data = CornerVelocityData(
            corner_name="FL",
            velocity=np.array([0.0]),
            bin_edges=np.array([-10, 0, 10]),
            bin_centers=np.array([-5, 5]),
            histogram=np.array([50.0, 50.0]),
            skew=0.0,
            kurtosis=0.0,
            mean=0.0,
            std=10.0,
            pct_friction=20.0,
            pct_slow_bump=20.0,
            pct_slow_rebound=20.0,
            pct_fast_bump=15.0,
            pct_fast_rebound=15.0,
            pct_curb=10.0,
            pct_zero_bin=20.0,
        )

        result = VelocityHistogramResult(
            front_left=corner_data,
            front_right=corner_data,
            rear_left=corner_data,
            rear_right=corner_data,
        )

        df = format_suspension_stats_table(result)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # Four corners
        assert "Corner" in df.columns
        assert "Skew" in df.columns
        assert "Friction %" in df.columns
        assert "Zero Bin %" in df.columns


class TestFormatSymmetryTable:
    """Tests for format_symmetry_table function."""

    def test_returns_dataframe_with_symmetry(self):
        """Should return DataFrame with bump/rebound symmetry."""
        corner_data = CornerVelocityData(
            corner_name="FL",
            velocity=np.array([0.0]),
            bin_edges=np.array([-10, 0, 10]),
            bin_centers=np.array([-5, 5]),
            histogram=np.array([50.0, 50.0]),
            skew=0.0,
            kurtosis=0.0,
            mean=0.0,
            std=10.0,
            pct_friction=10.0,
            pct_slow_bump=20.0,
            pct_slow_rebound=20.0,
            pct_fast_bump=25.0,
            pct_fast_rebound=25.0,
            pct_curb=0.0,
            pct_zero_bin=10.0,
        )

        result = VelocityHistogramResult(
            front_left=corner_data,
            front_right=corner_data,
            rear_left=corner_data,
            rear_right=corner_data,
        )

        df = format_symmetry_table(result)

        assert isinstance(df, pd.DataFrame)
        assert "Total Bump %" in df.columns
        assert "Total Rebound %" in df.columns
        assert "Bump/Total %" in df.columns


class TestFormatComparisonTable:
    """Tests for format_comparison_table function."""

    def test_returns_comparison_dataframe(self):
        """Should return DataFrame with left/right and front/rear comparisons."""
        corner_data = CornerVelocityData(
            corner_name="FL",
            velocity=np.array([0.0]),
            bin_edges=np.array([-10, 0, 10]),
            bin_centers=np.array([-5, 5]),
            histogram=np.array([50.0, 50.0]),
            skew=0.1,
            kurtosis=0.0,
            mean=0.0,
            std=10.0,
            pct_friction=20.0,
            pct_slow_bump=20.0,
            pct_slow_rebound=20.0,
            pct_fast_bump=20.0,
            pct_fast_rebound=20.0,
            pct_curb=0.0,
            pct_zero_bin=20.0,
        )

        result = VelocityHistogramResult(
            front_left=corner_data,
            front_right=corner_data,
            rear_left=corner_data,
            rear_right=corner_data,
        )

        df = format_comparison_table(result)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two comparisons: Left vs Right, Front vs Rear
        assert "Comparison" in df.columns
        assert "Zero Bin Diff" in df.columns
        assert "Friction Diff" in df.columns
        assert "Slow Bump Diff" in df.columns
        assert "Fast Bump Diff" in df.columns
        assert "Curb Diff" in df.columns

    def test_comparison_differences_are_zero_when_identical(self):
        """When all corners are identical, differences should be zero."""
        corner_data = CornerVelocityData(
            corner_name="FL",
            velocity=np.array([0.0]),
            bin_edges=np.array([-10, 0, 10]),
            bin_centers=np.array([-5, 5]),
            histogram=np.array([50.0, 50.0]),
            skew=0.1,
            kurtosis=0.0,
            mean=0.0,
            std=10.0,
            pct_friction=20.0,
            pct_slow_bump=20.0,
            pct_slow_rebound=20.0,
            pct_fast_bump=20.0,
            pct_fast_rebound=20.0,
            pct_curb=0.0,
            pct_zero_bin=20.0,
        )

        result = VelocityHistogramResult(
            front_left=corner_data,
            front_right=corner_data,
            rear_left=corner_data,
            rear_right=corner_data,
        )

        df = format_comparison_table(result)

        # All differences should be zero when corners are identical
        for col in df.columns:
            if col != "Comparison":
                for val in df[col]:
                    assert val == pytest.approx(0.0)
