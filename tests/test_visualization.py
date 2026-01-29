"""Tests for visualization module."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow as pa
import pytest

from motorsports_data_notebook.channels import (
    get_best_lap,
    get_best_lap_channels,
    get_top_laps,
)
from motorsports_data_notebook.visualization import (
    format_lap_time,
    plot_corner_inputs,
    plot_gps_channels,
    plot_tire_thermography,
    plot_track_segments,
)
from motorsports_data_notebook.corners import Corner
from motorsports_data_notebook.zones import TrackSegment


# ============================================================================
# Fixtures for synthetic data
# ============================================================================


@pytest.fixture
def sample_laps():
    """Create sample laps DataFrame."""
    return pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5],
            "start_time": [0, 60000, 120000, 180000, 240000],
            "end_time": [60000, 120000, 180000, 240000, 300000],
            "lap_time": pd.to_timedelta([60, 58, 57, 59, 61], unit="s"),
        }
    )


@pytest.fixture
def sample_channels(sample_laps):
    """Create sample channels DataFrame with data for multiple laps."""
    n_samples = 1500  # 300 samples per lap, 5 laps
    timecodes = np.linspace(0, 300000, n_samples)

    # Create per-lap distance (resets at each lap boundary)
    distance_m = np.zeros(n_samples)
    for i, lap in sample_laps.iterrows():
        mask = (timecodes >= lap["start_time"]) & (timecodes < lap["end_time"])
        lap_indices = np.where(mask)[0]
        if len(lap_indices) > 0:
            distance_m[lap_indices] = np.linspace(0, 4500, len(lap_indices))

    return pd.DataFrame(
        {
            "timecodes": timecodes,
            "distance_m": distance_m,
            "GPS Speed": np.random.uniform(20, 60, n_samples),  # m/s
            "speed_kmh": np.random.uniform(72, 216, n_samples),  # km/h
            "GPS Latitude": 35.36 + np.sin(np.linspace(0, 2 * np.pi, n_samples)) * 0.01,
            "GPS Longitude": 138.92 + np.cos(np.linspace(0, 2 * np.pi, n_samples)) * 0.01,
        }
    )


@pytest.fixture
def sample_lap_channels():
    """Create sample channel tables for a single lap with tire temps."""
    n_samples = 300
    timecodes = np.linspace(60000, 120000, n_samples, dtype=np.int64)

    # Build dict of channel tables
    channels: dict[str, pa.Table] = {}

    # Helper to create a channel table
    def make_table(name: str, values: np.ndarray) -> pa.Table:
        return pa.table(
            {
                "timecodes": pa.array(timecodes, type=pa.int64()),
                name: pa.array(values),
            }
        )

    channels["distance_m"] = make_table("distance_m", np.linspace(0, 4500, n_samples))
    channels["speed_kmh"] = make_table(
        "speed_kmh", 100 + 50 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    )
    channels["GPS Latitude"] = make_table(
        "GPS Latitude", 35.36 + np.sin(np.linspace(0, 2 * np.pi, n_samples)) * 0.01
    )
    channels["GPS Longitude"] = make_table(
        "GPS Longitude", 138.92 + np.cos(np.linspace(0, 2 * np.pi, n_samples)) * 0.01
    )
    channels["LateralAcc"] = make_table("LateralAcc", np.random.uniform(-1.5, 1.5, n_samples))
    channels["InlineAcc"] = make_table("InlineAcc", np.random.uniform(-1.5, 1.5, n_samples))
    channels["BrakePress"] = make_table("BrakePress", np.random.uniform(0, 100, n_samples))
    channels["PPS"] = make_table("PPS", np.random.uniform(0, 100, n_samples))
    channels["SteerAngle"] = make_table("SteerAngle", np.random.uniform(-180, 180, n_samples))

    # Add tire temperature channels
    for pos in ["FL", "FR", "RL", "RR"]:
        for ch in range(1, 9):
            channels[f"{pos}_Ch{ch}"] = make_table(
                f"{pos}_Ch{ch}", np.random.uniform(60, 100, n_samples)
            )

    return channels


@pytest.fixture
def sample_lap_channels_df():
    """Create sample channel data as DataFrame for functions that still need it."""
    n_samples = 300
    distance_m = np.linspace(0, 4500, n_samples)

    data = {
        "timecodes": np.linspace(60000, 120000, n_samples),
        "distance_m": distance_m,
        "speed_kmh": 100 + 50 * np.sin(np.linspace(0, 4 * np.pi, n_samples)),
        "GPS Latitude": 35.36 + np.sin(np.linspace(0, 2 * np.pi, n_samples)) * 0.01,
        "GPS Longitude": 138.92 + np.cos(np.linspace(0, 2 * np.pi, n_samples)) * 0.01,
        "LateralAcc": np.random.uniform(-1.5, 1.5, n_samples),
        "InlineAcc": np.random.uniform(-1.5, 1.5, n_samples),
        "BrakePress": np.random.uniform(0, 100, n_samples),
        "PPS": np.random.uniform(0, 100, n_samples),
        "SteerAngle": np.random.uniform(-180, 180, n_samples),
    }

    # Add tire temperature channels
    for pos in ["FL", "FR", "RL", "RR"]:
        for ch in range(1, 9):
            data[f"{pos}_Ch{ch}"] = np.random.uniform(60, 100, n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def sample_segments():
    """Create sample track segments."""
    return [
        TrackSegment(
            id=1,
            segment_type="braking",
            start_dist=400,
            end_dist=500,
            name="Turn 1 Braking",
            corner_id=1,
        ),
        TrackSegment(
            id=2,
            segment_type="corner",
            start_dist=500,
            end_dist=700,
            name="Turn 1",
            corner_id=1,
            apex_dist=600,
        ),
        TrackSegment(
            id=3,
            segment_type="acceleration",
            start_dist=700,
            end_dist=1000,
            name="Turn 1 Exit",
            corner_id=1,
        ),
    ]


# ============================================================================
# Tests for format_lap_time
# ============================================================================


def test_format_lap_time_basic():
    """Test basic lap time formatting."""
    lap_time = pd.Timedelta(minutes=1, seconds=23, milliseconds=456)
    result = format_lap_time(lap_time)
    assert result == "1:23.456"


def test_format_lap_time_no_minutes():
    """Test lap time under one minute."""
    lap_time = pd.Timedelta(seconds=45, milliseconds=123)
    result = format_lap_time(lap_time)
    assert result == "0:45.123"


def test_format_lap_time_long():
    """Test lap time over two minutes."""
    lap_time = pd.Timedelta(minutes=2, seconds=5, milliseconds=789)
    result = format_lap_time(lap_time)
    assert result == "2:05.789"


# ============================================================================
# Tests for get_top_laps
# ============================================================================


def test_get_top_laps_basic(sample_laps):
    """Test basic top laps selection."""
    top_laps = get_top_laps(sample_laps, threshold_pct=1.03)

    # Should exclude first and last laps (1 and 5)
    assert 1 not in top_laps["num"].values
    assert 5 not in top_laps["num"].values


def test_get_top_laps_threshold(sample_laps):
    """Test that threshold filtering works correctly."""
    # Best lap is 57s, 103% = 58.71s
    # Lap 2 (58s) and Lap 3 (57s) should be included
    # Lap 4 (59s) should be excluded
    top_laps = get_top_laps(sample_laps, threshold_pct=1.03)

    assert 3 in top_laps["num"].values  # 57s - best
    assert 2 in top_laps["num"].values  # 58s - within 103%
    assert 4 not in top_laps["num"].values  # 59s - outside 103%


def test_get_top_laps_missing_lap_time():
    """Test that missing lap_time column raises error."""
    laps = pd.DataFrame({"num": [1, 2], "start_time": [0, 60000], "end_time": [60000, 120000]})

    with pytest.raises(ValueError, match="Expected lap_time column"):
        get_top_laps(laps)


def test_get_top_laps_empty_result():
    """Test handling when no laps meet criteria after filtering."""
    laps = pd.DataFrame(
        {
            "num": [1, 2],
            "start_time": [0, 60000],
            "end_time": [60000, 120000],
            "lap_time": pd.to_timedelta([60, 60], unit="s"),
        }
    )

    # With only 2 laps, excluding first and last leaves none
    top_laps = get_top_laps(laps)
    assert len(top_laps) == 0


# ============================================================================
# Tests for plot_tire_thermography
# ============================================================================


def test_plot_tire_thermography_returns_figure(sample_lap_channels):
    """Test that plot_tire_thermography returns a Plotly figure."""
    fig = plot_tire_thermography(sample_lap_channels)

    assert isinstance(fig, go.Figure)


def test_plot_tire_thermography_has_traces(sample_lap_channels):
    """Test that figure has expected number of traces."""
    fig = plot_tire_thermography(sample_lap_channels)

    # 4 heatmaps + 2 speed/G traces + 3 driver input traces = 9 traces
    assert len(fig.data) == 9


def test_plot_tire_thermography_missing_channels():
    """Test that missing tire channels raises error."""
    timecodes = pa.array([60000, 70000, 80000], type=pa.int64())
    channels = {
        "distance_m": pa.table({"timecodes": timecodes, "distance_m": pa.array([0, 100, 200])}),
        "speed_kmh": pa.table({"timecodes": timecodes, "speed_kmh": pa.array([100, 110, 105])}),
    }

    with pytest.raises(ValueError, match="Missing channels"):
        plot_tire_thermography(channels)


def test_plot_tire_thermography_custom_title(sample_lap_channels):
    """Test custom title is applied."""
    fig = plot_tire_thermography(sample_lap_channels, title="Custom Title")

    assert fig.layout.title.text == "Custom Title"


# ============================================================================
# Tests for plot_track_segments
# ============================================================================


def test_plot_track_segments_returns_figure(sample_lap_channels_df, sample_segments):
    """Test that plot_track_segments returns a Plotly figure."""
    fig = plot_track_segments(sample_lap_channels_df, sample_segments)

    assert isinstance(fig, go.Figure)


def test_plot_track_segments_has_base_track(sample_lap_channels_df, sample_segments):
    """Test that figure includes base track trace."""
    fig = plot_track_segments(sample_lap_channels_df, sample_segments)

    # First trace should be the base track
    assert fig.data[0].name == "Track"


def test_plot_track_segments_has_segment_traces(sample_lap_channels_df, sample_segments):
    """Test that figure includes traces for each segment type."""
    fig = plot_track_segments(sample_lap_channels_df, sample_segments)

    trace_names = [t.name for t in fig.data if t.name]
    assert "Track" in trace_names
    assert "Braking Zone" in trace_names
    assert "Corner" in trace_names
    assert "Acceleration Zone" in trace_names


def test_plot_track_segments_has_apex_markers(sample_lap_channels_df, sample_segments):
    """Test that corner apex markers are included."""
    fig = plot_track_segments(sample_lap_channels_df, sample_segments)

    # Find traces with text (apex labels)
    text_traces = [t for t in fig.data if hasattr(t, "text") and t.text]
    assert len(text_traces) > 0


def test_plot_track_segments_custom_dimensions(sample_lap_channels_df, sample_segments):
    """Test custom width and height are applied."""
    fig = plot_track_segments(sample_lap_channels_df, sample_segments, width=1200, height=800)

    assert fig.layout.width == 1200
    assert fig.layout.height == 800


# ============================================================================
# Tests for plot_corner_inputs
# ============================================================================


@pytest.fixture
def sample_corner():
    """Create a sample corner for plotting tests."""
    return Corner(
        id=1,
        name="Turn 1",
        direction="L",
        start_idx=50,
        end_idx=150,
        start_dist=500,
        end_dist=700,
        apex_idx=100,
        apex_dist=600,
        max_curvature=0.01,
    )


@pytest.fixture
def sample_corner_data():
    """Create sample corner data arrays."""
    n_samples = 100
    distance = np.linspace(450, 750, n_samples)
    throttle = np.concatenate(
        [
            np.linspace(100, 0, 30),  # Lift off
            np.zeros(40),  # Coast
            np.linspace(0, 100, 30),  # Back on throttle
        ]
    )
    brake = np.concatenate(
        [
            np.zeros(10),
            np.linspace(0, 80, 20),  # Braking
            np.linspace(80, 0, 30),  # Trail brake
            np.zeros(40),
        ]
    )
    steering = np.concatenate(
        [
            np.zeros(20),
            np.linspace(0, -90, 30),  # Turn in
            np.linspace(-90, -45, 30),  # Unwinding
            np.linspace(-45, 0, 20),  # Exit
        ]
    )
    return {
        "distance": distance,
        "throttle": throttle,
        "brake": brake,
        "steering": steering,
    }


def test_plot_corner_inputs_returns_figure(sample_corner_data, sample_corner):
    """Test that plot_corner_inputs returns a Plotly figure."""
    fig = plot_corner_inputs(
        sample_corner_data["distance"],
        sample_corner,
        throttle=sample_corner_data["throttle"],
        brake=sample_corner_data["brake"],
        steering=sample_corner_data["steering"],
    )

    assert isinstance(fig, go.Figure)


def test_plot_corner_inputs_has_three_subplots(sample_corner_data, sample_corner):
    """Test that figure has traces for all three channels."""
    fig = plot_corner_inputs(
        sample_corner_data["distance"],
        sample_corner,
        throttle=sample_corner_data["throttle"],
        brake=sample_corner_data["brake"],
        steering=sample_corner_data["steering"],
    )

    # Should have at least 3 traces (one per channel)
    assert len(fig.data) >= 3


def test_plot_corner_inputs_custom_title(sample_corner_data, sample_corner):
    """Test that custom title is applied."""
    fig = plot_corner_inputs(
        sample_corner_data["distance"],
        sample_corner,
        throttle=sample_corner_data["throttle"],
        title="Custom Title",
    )

    assert fig.layout.title.text == "Custom Title"


def test_plot_corner_inputs_default_title(sample_corner_data, sample_corner):
    """Test that default title includes corner name."""
    fig = plot_corner_inputs(
        sample_corner_data["distance"],
        sample_corner,
        throttle=sample_corner_data["throttle"],
    )

    assert "Turn 1" in fig.layout.title.text


def test_plot_corner_inputs_custom_dimensions(sample_corner_data, sample_corner):
    """Test that custom dimensions are applied."""
    fig = plot_corner_inputs(
        sample_corner_data["distance"],
        sample_corner,
        throttle=sample_corner_data["throttle"],
        width=800,
        height=400,
    )

    assert fig.layout.width == 800
    assert fig.layout.height == 400


def test_plot_corner_inputs_single_channel(sample_corner_data, sample_corner):
    """Test that plot works with only one channel."""
    fig = plot_corner_inputs(
        sample_corner_data["distance"],
        sample_corner,
        throttle=sample_corner_data["throttle"],
    )

    # Should have 1 trace
    assert len(fig.data) == 1


def test_plot_corner_inputs_two_channels(sample_corner_data, sample_corner):
    """Test that plot works with two channels."""
    fig = plot_corner_inputs(
        sample_corner_data["distance"],
        sample_corner,
        throttle=sample_corner_data["throttle"],
        brake=sample_corner_data["brake"],
    )

    # Should have 2 traces
    assert len(fig.data) == 2


def test_plot_corner_inputs_no_channels_raises(sample_corner_data, sample_corner):
    """Test that having no input channels raises an error."""
    with pytest.raises(ValueError, match="At least one input channel"):
        plot_corner_inputs(sample_corner_data["distance"], sample_corner)


def test_plot_corner_inputs_has_corner_annotations(sample_corner_data, sample_corner):
    """Test that figure includes corner boundary and apex annotations."""
    fig = plot_corner_inputs(
        sample_corner_data["distance"],
        sample_corner,
        throttle=sample_corner_data["throttle"],
        brake=sample_corner_data["brake"],
        steering=sample_corner_data["steering"],
    )

    # Check for vrect (corner boundary) and vline (apex) shapes
    shapes = fig.layout.shapes if fig.layout.shapes else []
    assert len(shapes) > 0


# ============================================================================
# Fixtures for channel-based functions (PyArrow tables)
# ============================================================================


class MockLogFile:
    """Mock LogFile for testing channel-based functions."""

    def __init__(self, channels: dict[str, pa.Table]):
        self.channels = channels


@pytest.fixture
def sample_channel_tables():
    """Create sample PyArrow channel tables with different sample rates."""
    # GPS channels at 10Hz (100ms intervals) - 100 samples for 10 seconds
    gps_timecodes = np.arange(60000, 70000, 100, dtype=np.int64)  # 60s to 70s
    n_gps = len(gps_timecodes)

    gps_lat = pa.table(
        {
            "timecodes": pa.array(gps_timecodes, type=pa.int64()),
            "GPS Latitude": pa.array(35.36 + np.sin(np.linspace(0, 2 * np.pi, n_gps)) * 0.01),
        }
    )

    gps_lon = pa.table(
        {
            "timecodes": pa.array(gps_timecodes, type=pa.int64()),
            "GPS Longitude": pa.array(138.92 + np.cos(np.linspace(0, 2 * np.pi, n_gps)) * 0.01),
        }
    )

    speed_kmh = pa.table(
        {
            "timecodes": pa.array(gps_timecodes, type=pa.int64()),
            "speed_kmh": pa.array(100 + 50 * np.sin(np.linspace(0, 4 * np.pi, n_gps))),
        }
    )

    # Non-GPS channels at 50Hz (20ms intervals) - 500 samples for 10 seconds
    other_timecodes = np.arange(60000, 70000, 20, dtype=np.int64)
    n_other = len(other_timecodes)

    brake_press = pa.table(
        {
            "timecodes": pa.array(other_timecodes, type=pa.int64()),
            "BrakePress": pa.array(np.random.uniform(0, 100, n_other)),
        }
    )

    pps = pa.table(
        {
            "timecodes": pa.array(other_timecodes, type=pa.int64()),
            "PPS": pa.array(np.random.uniform(0, 100, n_other)),
        }
    )

    return {
        "GPS Latitude": gps_lat,
        "GPS Longitude": gps_lon,
        "speed_kmh": speed_kmh,
        "BrakePress": brake_press,
        "PPS": pps,
    }


@pytest.fixture
def mock_log(sample_channel_tables):
    """Create a mock LogFile with sample channel tables."""
    return MockLogFile(sample_channel_tables)


@pytest.fixture
def sample_laps_for_channels():
    """Create sample laps DataFrame for channel-based tests."""
    return pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5],
            "start_time": [0, 60000, 120000, 180000, 240000],
            "end_time": [60000, 120000, 180000, 240000, 300000],
            "lap_time": pd.to_timedelta([60, 58, 57, 59, 61], unit="s"),
        }
    )




# ============================================================================
# Tests for plot_gps_channels
# ============================================================================


def test_plot_gps_channels_returns_figure(sample_channel_tables):
    """Test that plot_gps_channels returns a Plotly figure."""
    fig = plot_gps_channels(
        sample_channel_tables,
        lat_channel="GPS Latitude",
        lon_channel="GPS Longitude",
        color_channels=[("speed_kmh", "Speed (km/h)", "Viridis")],
    )

    assert isinstance(fig, go.Figure)


def test_plot_gps_channels_multiple_colors(sample_channel_tables):
    """Test plot with multiple color channels."""
    fig = plot_gps_channels(
        sample_channel_tables,
        lat_channel="GPS Latitude",
        lon_channel="GPS Longitude",
        color_channels=[
            ("BrakePress", "Brake", "Reds"),
            ("PPS", "Throttle", "Greens"),
        ],
    )

    # Should have multiple traces (color layers + track outline + start line)
    assert len(fig.data) >= 2


def test_plot_gps_channels_with_title(sample_channel_tables):
    """Test that title is applied to figure."""
    fig = plot_gps_channels(
        sample_channel_tables,
        lat_channel="GPS Latitude",
        lon_channel="GPS Longitude",
        color_channels=[("speed_kmh", "Speed", "Viridis")],
        title="Test Title",
    )

    assert fig.layout.title.text == "Test Title"
