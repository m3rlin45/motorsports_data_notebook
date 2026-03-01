"""Tests for tire grip analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow as pa
import pytest
from dataclasses import dataclass
from typing import Dict

from motorsports_data_notebook.tire_grip import (
    CornerTireGripData,
    TireGripResult,
    _compute_corner,
    _get_channel_unit,
    analyze_tire_grip,
    analyze_tire_grip_multi_lap,
    format_tire_grip_stats_table,
)
from motorsports_data_notebook.visualization import plot_tire_grip_scatter


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

    def filter_by_lap(self, lap_num: int):
        """Mock filter_by_lap returning self for method chaining."""
        return self

    def select_channels(self, channel_names: list):
        """Mock select_channels returning self for method chaining."""
        return self

    def resample_to_channel(self, reference_channel: str):
        """Mock resample_to_channel returning self for method chaining."""
        return self


# Default channel name mapping matching profiles.py DEFAULT_CHANNEL_NAMES
_DEFAULT_CHANNEL_NAMES = {
    "lateral_g": "LateralAcc",
    "inline_g": "InlineAcc",
    "tpms_press_fl": "TPMS_Press_LF",
    "tpms_press_fr": "TPMS_Press_RF",
    "tpms_press_rl": "TPMS_Press_LR",
    "tpms_press_rr": "TPMS_Press_RR",
    "tpms_temp_fl": "TPMS_Temp_LF",
    "tpms_temp_fr": "TPMS_Temp_RF",
    "tpms_temp_rl": "TPMS_Temp_LR",
    "tpms_temp_rr": "TPMS_Temp_RR",
}


def _make_table_with_unit(
    timecodes: np.ndarray, name: str, values: np.ndarray, unit: str
) -> pa.Table:
    """Create a PyArrow table with unit metadata on the value field."""
    schema = pa.schema(
        [
            pa.field("timecodes", pa.int64()),
            pa.field(name, pa.float64(), metadata={"units": unit}),
        ]
    )
    return pa.table({"timecodes": timecodes, name: values}, schema=schema)


def _make_pressure_log(
    lateral: np.ndarray,
    inline: np.ndarray,
    press_fl: np.ndarray,
    press_fr: np.ndarray,
    press_rl: np.ndarray,
    press_rr: np.ndarray,
) -> MockLogFile:
    """Create a MockLogFile with acceleration and pressure channels."""
    n = len(lateral)
    timecodes = np.arange(0, n * 10, 10, dtype=np.int64)
    channels = {
        "LateralAcc": _make_table_with_unit(timecodes, "LateralAcc", lateral, "g"),
        "InlineAcc": _make_table_with_unit(timecodes, "InlineAcc", inline, "g"),
        "TPMS_Press_LF": _make_table_with_unit(timecodes, "TPMS_Press_LF", press_fl, "bar"),
        "TPMS_Press_RF": _make_table_with_unit(timecodes, "TPMS_Press_RF", press_fr, "bar"),
        "TPMS_Press_LR": _make_table_with_unit(timecodes, "TPMS_Press_LR", press_rl, "bar"),
        "TPMS_Press_RR": _make_table_with_unit(timecodes, "TPMS_Press_RR", press_rr, "bar"),
    }
    return MockLogFile(channels=channels)


def _make_temperature_log(
    lateral: np.ndarray,
    inline: np.ndarray,
    temp_fl: np.ndarray,
    temp_fr: np.ndarray,
    temp_rl: np.ndarray,
    temp_rr: np.ndarray,
) -> MockLogFile:
    """Create a MockLogFile with acceleration and temperature channels."""
    n = len(lateral)
    timecodes = np.arange(0, n * 10, 10, dtype=np.int64)
    channels = {
        "LateralAcc": _make_table_with_unit(timecodes, "LateralAcc", lateral, "g"),
        "InlineAcc": _make_table_with_unit(timecodes, "InlineAcc", inline, "g"),
        "TPMS_Temp_LF": _make_table_with_unit(timecodes, "TPMS_Temp_LF", temp_fl, "C"),
        "TPMS_Temp_RF": _make_table_with_unit(timecodes, "TPMS_Temp_RF", temp_fr, "C"),
        "TPMS_Temp_LR": _make_table_with_unit(timecodes, "TPMS_Temp_LR", temp_rl, "C"),
        "TPMS_Temp_RR": _make_table_with_unit(timecodes, "TPMS_Temp_RR", temp_rr, "C"),
    }
    return MockLogFile(channels=channels)


class TestAnalyzeTireGrip:
    """Tests for analyze_tire_grip function."""

    def test_pressure_mode(self):
        """Happy path with pressure mode should return TireGripResult with all 4 corners."""
        n = 100
        np.random.seed(42)
        lateral = np.random.normal(0.5, 0.3, n)
        inline = np.random.normal(0.2, 0.1, n)
        press = np.random.normal(32.0, 0.5, n)
        log = _make_pressure_log(lateral, inline, press, press, press, press)

        result = analyze_tire_grip(log, _DEFAULT_CHANNEL_NAMES, metric_mode="pressure")

        assert isinstance(result, TireGripResult)
        assert isinstance(result.front_left, CornerTireGripData)
        assert isinstance(result.front_right, CornerTireGripData)
        assert isinstance(result.rear_left, CornerTireGripData)
        assert isinstance(result.rear_right, CornerTireGripData)
        assert result.metric_mode == "pressure"
        assert result.metric_unit == "bar"
        assert result.accel_unit == "g"

    def test_temperature_mode(self):
        """Temperature mode should use temp channels and set correct metric_mode and unit."""
        n = 100
        np.random.seed(42)
        lateral = np.random.normal(0.5, 0.3, n)
        inline = np.random.normal(0.2, 0.1, n)
        temp = np.random.normal(180.0, 5.0, n)
        log = _make_temperature_log(lateral, inline, temp, temp, temp, temp)

        result = analyze_tire_grip(log, _DEFAULT_CHANNEL_NAMES, metric_mode="temperature")

        assert isinstance(result, TireGripResult)
        assert result.metric_mode == "temperature"
        assert result.metric_unit == "C"
        assert result.accel_unit == "g"
        assert result.front_left.corner_name == "FL"
        assert result.rear_right.corner_name == "RR"

    def test_total_g_computation(self):
        """Total G should be sqrt(lateral^2 + inline^2); lat=3, inline=4 -> total=5."""
        n = 50
        lateral = np.full(n, 3.0)
        inline = np.full(n, 4.0)
        press = np.full(n, 30.0)
        log = _make_pressure_log(lateral, inline, press, press, press, press)

        result = analyze_tire_grip(log, _DEFAULT_CHANNEL_NAMES, metric_mode="pressure")

        # All total_g values should be 5.0 (3-4-5 triangle)
        np.testing.assert_array_almost_equal(result.front_left.total_g, np.full(n, 5.0))
        np.testing.assert_array_almost_equal(result.front_right.total_g, np.full(n, 5.0))
        np.testing.assert_array_almost_equal(result.rear_left.total_g, np.full(n, 5.0))
        np.testing.assert_array_almost_equal(result.rear_right.total_g, np.full(n, 5.0))
        assert result.front_left.mean_g == pytest.approx(5.0)

    def test_missing_channel_raises(self):
        """Empty channel_names should raise KeyError for missing required keys."""
        log = MockLogFile(channels={})

        with pytest.raises(KeyError):
            analyze_tire_grip(log, {}, metric_mode="pressure")

    def test_invalid_metric_mode(self):
        """Invalid metric_mode should raise ValueError."""
        log = MockLogFile(channels={})

        with pytest.raises(ValueError, match="metric_mode must be"):
            analyze_tire_grip(log, _DEFAULT_CHANNEL_NAMES, metric_mode="invalid")

    def test_empty_data(self):
        """Empty arrays should be handled gracefully with zero statistics."""
        channels = {
            "LateralAcc": pa.table(
                {
                    "timecodes": pa.array([], type=pa.int64()),
                    "LateralAcc": pa.array([], type=pa.float64()),
                }
            ),
            "InlineAcc": pa.table(
                {
                    "timecodes": pa.array([], type=pa.int64()),
                    "InlineAcc": pa.array([], type=pa.float64()),
                }
            ),
            "TPMS_Press_LF": pa.table(
                {
                    "timecodes": pa.array([], type=pa.int64()),
                    "TPMS_Press_LF": pa.array([], type=pa.float64()),
                }
            ),
            "TPMS_Press_RF": pa.table(
                {
                    "timecodes": pa.array([], type=pa.int64()),
                    "TPMS_Press_RF": pa.array([], type=pa.float64()),
                }
            ),
            "TPMS_Press_LR": pa.table(
                {
                    "timecodes": pa.array([], type=pa.int64()),
                    "TPMS_Press_LR": pa.array([], type=pa.float64()),
                }
            ),
            "TPMS_Press_RR": pa.table(
                {
                    "timecodes": pa.array([], type=pa.int64()),
                    "TPMS_Press_RR": pa.array([], type=pa.float64()),
                }
            ),
        }
        log = MockLogFile(channels=channels)

        result = analyze_tire_grip(log, _DEFAULT_CHANNEL_NAMES, metric_mode="pressure")

        assert result.front_left.mean_g == pytest.approx(0.0)
        assert result.front_left.std_g == pytest.approx(0.0)
        assert result.front_left.mean_metric == pytest.approx(0.0)
        assert result.front_left.std_metric == pytest.approx(0.0)
        assert len(result.front_left.total_g) == 0
        assert len(result.front_left.tire_metric) == 0
        assert len(result.front_left.bucket_centers) == 0
        assert len(result.front_left.bucket_values) == 0
        assert len(result.front_left.bucket_counts) == 0


class TestPlotTireGripScatter:
    """Tests for plot_tire_grip_scatter visualization function."""

    def _make_result(self, metric_mode: str = "pressure") -> TireGripResult:
        """Create a minimal TireGripResult for plotting tests."""
        n = 50
        total_g = np.random.uniform(0.5, 2.0, n)
        tire_metric = np.random.uniform(30.0, 35.0, n)
        metric_unit = "bar" if metric_mode == "pressure" else "C"
        bucket_centers = np.array([31.0, 32.0, 33.0, 34.0])
        bucket_values = np.array([1.5, 1.8, 1.6, 1.7])
        bucket_counts = np.array([10, 15, 12, 13], dtype=np.int64)

        corners = {}
        for name in ("FL", "FR", "RL", "RR"):
            corners[name] = CornerTireGripData(
                corner_name=name,
                total_g=total_g.copy(),
                tire_metric=tire_metric.copy(),
                mean_g=float(np.mean(total_g)),
                mean_metric=float(np.mean(tire_metric)),
                std_g=float(np.std(total_g)),
                std_metric=float(np.std(tire_metric)),
                bucket_centers=bucket_centers.copy(),
                bucket_values=bucket_values.copy(),
                bucket_counts=bucket_counts.copy(),
                percentile=99.9,
            )

        return TireGripResult(
            front_left=corners["FL"],
            front_right=corners["FR"],
            rear_left=corners["RL"],
            rear_right=corners["RR"],
            metric_mode=metric_mode,
            metric_unit=metric_unit,
            accel_unit="g",
        )

    def test_returns_figure(self):
        """Should return a plotly go.Figure."""
        result = self._make_result()

        fig = plot_tire_grip_scatter(result)

        assert isinstance(fig, go.Figure)

    def test_four_subplots(self):
        """Should have 4 traces (one per corner)."""
        result = self._make_result()

        fig = plot_tire_grip_scatter(result)

        assert len(fig.data) == 4

    def test_traces_are_lines(self):
        """Traces should be line+marker plots using bucket data."""
        result = self._make_result()

        fig = plot_tire_grip_scatter(result)

        for trace in fig.data:
            assert trace.mode == "lines+markers"
            assert len(trace.x) == 4  # 4 bucket centers

    def test_axis_labels_pressure_mode(self):
        """X-axis label should contain 'Pressure' in pressure mode."""
        result = self._make_result(metric_mode="pressure")

        fig = plot_tire_grip_scatter(result)

        # Check that at least one x-axis has "Pressure" in its title
        x_axis_titles = []
        for key, val in fig.layout.to_plotly_json().items():
            if key.startswith("xaxis") and isinstance(val, dict) and "title" in val:
                title = val["title"]
                if isinstance(title, dict):
                    x_axis_titles.append(title.get("text", ""))
                else:
                    x_axis_titles.append(str(title))

        assert any(
            "Pressure" in t for t in x_axis_titles
        ), f"Expected 'Pressure' in x-axis labels, got: {x_axis_titles}"

    def test_axis_labels_temperature_mode(self):
        """X-axis label should contain 'Temperature' in temperature mode."""
        result = self._make_result(metric_mode="temperature")

        fig = plot_tire_grip_scatter(result)

        x_axis_titles = []
        for key, val in fig.layout.to_plotly_json().items():
            if key.startswith("xaxis") and isinstance(val, dict) and "title" in val:
                title = val["title"]
                if isinstance(title, dict):
                    x_axis_titles.append(title.get("text", ""))
                else:
                    x_axis_titles.append(str(title))

        assert any(
            "Temperature" in t for t in x_axis_titles
        ), f"Expected 'Temperature' in x-axis labels, got: {x_axis_titles}"


class TestFormatTireGripStatsTable:
    """Tests for format_tire_grip_stats_table function."""

    def _make_result(self, metric_mode: str = "pressure") -> TireGripResult:
        """Create a minimal TireGripResult for table formatting tests."""
        metric_unit = "bar" if metric_mode == "pressure" else "C"

        corners = {}
        for name in ("FL", "FR", "RL", "RR"):
            corners[name] = CornerTireGripData(
                corner_name=name,
                total_g=np.array([1.0, 1.5, 2.0]),
                tire_metric=np.array([32.0, 32.5, 33.0]),
                mean_g=1.5,
                mean_metric=32.5,
                std_g=0.5,
                std_metric=0.5,
                bucket_centers=np.array([32.25, 32.75]),
                bucket_values=np.array([1.8, 1.9]),
                bucket_counts=np.array([1, 2], dtype=np.int64),
                percentile=99.9,
            )

        return TireGripResult(
            front_left=corners["FL"],
            front_right=corners["FR"],
            rear_left=corners["RL"],
            rear_right=corners["RR"],
            metric_mode=metric_mode,
            metric_unit=metric_unit,
            accel_unit="g",
        )

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame with 4 rows and expected columns."""
        result = self._make_result()

        df = format_tire_grip_stats_table(result)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "Corner" in df.columns
        assert "Mean Accel (g)" in df.columns
        assert "Std Accel (g)" in df.columns

    def test_pressure_columns(self):
        """Pressure mode should have 'Pressure' in metric column names."""
        result = self._make_result(metric_mode="pressure")

        df = format_tire_grip_stats_table(result)

        pressure_cols = [col for col in df.columns if "Pressure" in col]
        assert len(pressure_cols) == 2  # Mean Pressure and Std Pressure
        assert any("Mean Pressure" in col for col in df.columns)
        assert any("Std Pressure" in col for col in df.columns)

    def test_temperature_columns(self):
        """Temperature mode should have 'Temperature' in metric column names."""
        result = self._make_result(metric_mode="temperature")

        df = format_tire_grip_stats_table(result)

        temp_cols = [col for col in df.columns if "Temperature" in col]
        assert len(temp_cols) == 2  # Mean Temperature and Std Temperature
        assert any("Mean Temperature" in col for col in df.columns)
        assert any("Std Temperature" in col for col in df.columns)

    def test_corner_names(self):
        """DataFrame should contain all four corner names."""
        result = self._make_result()

        df = format_tire_grip_stats_table(result)

        corner_values = df["Corner"].tolist()
        assert corner_values == ["FL", "FR", "RL", "RR"]

    def test_values_match_input(self):
        """DataFrame values should match the input TireGripResult data."""
        result = self._make_result()

        df = format_tire_grip_stats_table(result)

        # All corners have the same values in our test fixture
        for _, row in df.iterrows():
            assert row["Mean Accel (g)"] == pytest.approx(1.5)
            assert row["Std Accel (g)"] == pytest.approx(0.5)


class TestBucketComputation:
    """Tests for bucketed percentile computation in _compute_corner."""

    def test_bucket_computation(self):
        """Known linear data should produce expected bucket percentile values."""
        # 100 points with tire_metric from 0 to 9, total_g = tire_metric + 1
        n = 100
        tire_metric = np.linspace(0, 9, n)
        total_g = tire_metric + 1.0

        result = _compute_corner(
            total_g,
            tire_metric,
            "FL",
            num_buckets=10,
            percentile=100.0,
            min_count=1,
        )

        # Each bucket should have ~10 points, bucket centers at 0.45, 1.35, ...
        assert len(result.bucket_centers) == 10
        assert len(result.bucket_values) == 10
        assert len(result.bucket_counts) == 10
        assert result.percentile == 100.0

        # p100 = max; for last bucket (center ~8.55), max total_g ~ 10.0
        assert result.bucket_values[-1] == pytest.approx(10.0, abs=0.2)
        # First bucket max should be close to ~1.9
        assert result.bucket_values[0] == pytest.approx(1.9, abs=0.2)

    def test_min_count_filtering(self):
        """Buckets with fewer than min_count samples should be excluded."""
        # 5 points clustered at metric=1, 1 point at metric=10
        tire_metric = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0])
        total_g = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 2.0])

        result = _compute_corner(
            total_g,
            tire_metric,
            "FL",
            num_buckets=5,
            percentile=99.9,
            min_count=3,
        )

        # Only the bucket containing the 5 clustered points should survive
        assert len(result.bucket_centers) == 1
        assert result.bucket_counts[0] == 5

    def test_custom_percentile_p50(self):
        """p50 should give the median of total_g in each bucket."""
        n = 100
        tire_metric = np.full(n, 5.0)  # All same metric -> 1 bucket
        total_g = np.arange(1, n + 1, dtype=float)

        result = _compute_corner(
            total_g,
            tire_metric,
            "FL",
            num_buckets=1,
            percentile=50.0,
            min_count=1,
        )

        assert len(result.bucket_values) == 1
        assert result.bucket_values[0] == pytest.approx(np.median(total_g), abs=0.1)
        assert result.percentile == 50.0

    def test_pressure_mode_has_buckets(self):
        """analyze_tire_grip should populate bucket fields."""
        n = 100
        np.random.seed(42)
        lateral = np.random.normal(0.5, 0.3, n)
        inline = np.random.normal(0.2, 0.1, n)
        press = np.random.normal(32.0, 0.5, n)
        log = _make_pressure_log(lateral, inline, press, press, press, press)

        result = analyze_tire_grip(log, _DEFAULT_CHANNEL_NAMES, metric_mode="pressure")

        for corner in (result.front_left, result.front_right, result.rear_left, result.rear_right):
            assert len(corner.bucket_centers) > 0
            assert len(corner.bucket_values) == len(corner.bucket_centers)
            assert len(corner.bucket_counts) == len(corner.bucket_centers)
            assert corner.percentile == 99.9


class TestAnalyzeTireGripMultiLap:
    """Tests for analyze_tire_grip_multi_lap function."""

    def test_multi_lap_aggregates_data(self):
        """Multi-lap analysis should aggregate data from all laps."""
        n = 50
        np.random.seed(42)
        lateral = np.random.normal(0.5, 0.3, n)
        inline = np.random.normal(0.2, 0.1, n)
        press = np.random.normal(32.0, 0.5, n)
        log = _make_pressure_log(lateral, inline, press, press, press, press)

        result = analyze_tire_grip_multi_lap(
            log, [1, 2, 3], _DEFAULT_CHANNEL_NAMES, metric_mode="pressure"
        )

        assert isinstance(result, TireGripResult)
        # 3 laps of 50 points each = 150 total data points
        assert len(result.front_left.total_g) == n * 3
        assert len(result.front_left.tire_metric) == n * 3
        assert result.metric_mode == "pressure"
        assert result.metric_unit == "bar"
        assert result.accel_unit == "g"

    def test_multi_lap_empty_raises(self):
        """Empty lap_numbers should raise ValueError."""
        log = MockLogFile(channels={})

        with pytest.raises(ValueError, match="lap_numbers cannot be empty"):
            analyze_tire_grip_multi_lap(log, [], _DEFAULT_CHANNEL_NAMES)

    def test_multi_lap_invalid_metric_mode(self):
        """Invalid metric_mode should raise ValueError."""
        log = MockLogFile(channels={})

        with pytest.raises(ValueError, match="metric_mode must be"):
            analyze_tire_grip_multi_lap(log, [1], _DEFAULT_CHANNEL_NAMES, metric_mode="invalid")

    def test_multi_lap_has_buckets(self):
        """Multi-lap analysis should populate bucket fields."""
        n = 100
        np.random.seed(42)
        lateral = np.random.normal(0.5, 0.3, n)
        inline = np.random.normal(0.2, 0.1, n)
        press = np.random.normal(32.0, 0.5, n)
        log = _make_pressure_log(lateral, inline, press, press, press, press)

        result = analyze_tire_grip_multi_lap(
            log, [1, 2], _DEFAULT_CHANNEL_NAMES, metric_mode="pressure"
        )

        for corner in (result.front_left, result.front_right, result.rear_left, result.rear_right):
            assert len(corner.bucket_centers) > 0
            assert len(corner.bucket_values) == len(corner.bucket_centers)
            assert len(corner.bucket_counts) == len(corner.bucket_centers)

    def test_multi_lap_temperature_mode(self):
        """Multi-lap analysis should work with temperature mode."""
        n = 50
        np.random.seed(42)
        lateral = np.random.normal(0.5, 0.3, n)
        inline = np.random.normal(0.2, 0.1, n)
        temp = np.random.normal(180.0, 5.0, n)
        log = _make_temperature_log(lateral, inline, temp, temp, temp, temp)

        result = analyze_tire_grip_multi_lap(
            log, [1, 2], _DEFAULT_CHANNEL_NAMES, metric_mode="temperature"
        )

        assert result.metric_mode == "temperature"
        assert result.metric_unit == "C"
        assert result.accel_unit == "g"
        assert len(result.front_left.total_g) == n * 2


class TestGetChannelUnit:
    """Tests for _get_channel_unit helper."""

    def test_reads_unit_from_metadata(self):
        """Should read the 'units' key from field metadata."""
        schema = pa.schema([pa.field("ch", pa.float64(), metadata={"units": "kPa"})])
        table = pa.table({"ch": [1.0, 2.0]}, schema=schema)

        assert _get_channel_unit(table, "ch") == "kPa"

    def test_returns_empty_when_no_metadata(self):
        """Should return empty string when field has no metadata."""
        table = pa.table({"ch": [1.0, 2.0]})

        assert _get_channel_unit(table, "ch") == ""

    def test_returns_empty_for_missing_field(self):
        """Should return empty string when field name doesn't exist."""
        table = pa.table({"ch": [1.0, 2.0]})

        assert _get_channel_unit(table, "nonexistent") == ""

    def test_returns_empty_when_units_key_missing(self):
        """Should return empty string when metadata exists but has no 'units' key."""
        schema = pa.schema([pa.field("ch", pa.float64(), metadata={"desc": "test"})])
        table = pa.table({"ch": [1.0, 2.0]}, schema=schema)

        assert _get_channel_unit(table, "ch") == ""
