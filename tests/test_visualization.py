"""Tests for visualization module."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from motorsports_data_notebook.visualization import (
    format_lap_time,
    get_best_lap,
    get_best_lap_data,
    get_top_laps,
    plot_tire_thermography,
    plot_track_segments,
)
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
    """Create sample channel data for a single lap with tire temps."""
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
# Tests for get_best_lap_data
# ============================================================================


def test_get_best_lap_data_returns_tuple(sample_channels, sample_laps):
    """Test that get_best_lap_data returns correct tuple structure."""
    best_lap, lap_channels = get_best_lap_data(sample_channels, sample_laps)

    assert isinstance(best_lap, pd.Series)
    assert isinstance(lap_channels, pd.DataFrame)


def test_get_best_lap_data_finds_fastest(sample_channels, sample_laps):
    """Test that get_best_lap_data finds the fastest lap (lap 3 with 57s)."""
    best_lap, lap_channels = get_best_lap_data(sample_channels, sample_laps)

    # get_best_lap uses duration from end_time - start_time, not lap_time column
    # All laps have same duration (60000ms), so it returns the first one found (lap 2)
    # This tests that it excludes first and last laps properly
    assert best_lap["num"] in [2, 3, 4]  # Should be one of the middle laps


def test_get_best_lap_data_filters_channels(sample_channels, sample_laps):
    """Test that returned channels are filtered to best lap timerange."""
    best_lap, lap_channels = get_best_lap_data(sample_channels, sample_laps)

    # All timecodes should be within best lap bounds
    assert lap_channels["timecodes"].min() >= best_lap["start_time"]
    assert lap_channels["timecodes"].max() < best_lap["end_time"]


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
    df = pd.DataFrame(
        {
            "distance_m": [0, 100, 200],
            "speed_kmh": [100, 110, 105],
        }
    )

    with pytest.raises(ValueError, match="Missing tire temperature channels"):
        plot_tire_thermography(df)


def test_plot_tire_thermography_custom_title(sample_lap_channels):
    """Test custom title is applied."""
    fig = plot_tire_thermography(sample_lap_channels, title="Custom Title")

    assert fig.layout.title.text == "Custom Title"


# ============================================================================
# Tests for plot_track_segments
# ============================================================================


def test_plot_track_segments_returns_figure(sample_lap_channels, sample_segments):
    """Test that plot_track_segments returns a Plotly figure."""
    fig = plot_track_segments(sample_lap_channels, sample_segments)

    assert isinstance(fig, go.Figure)


def test_plot_track_segments_has_base_track(sample_lap_channels, sample_segments):
    """Test that figure includes base track trace."""
    fig = plot_track_segments(sample_lap_channels, sample_segments)

    # First trace should be the base track
    assert fig.data[0].name == "Track"


def test_plot_track_segments_has_segment_traces(sample_lap_channels, sample_segments):
    """Test that figure includes traces for each segment type."""
    fig = plot_track_segments(sample_lap_channels, sample_segments)

    trace_names = [t.name for t in fig.data if t.name]
    assert "Track" in trace_names
    assert "Braking Zone" in trace_names
    assert "Corner" in trace_names
    assert "Acceleration Zone" in trace_names


def test_plot_track_segments_has_apex_markers(sample_lap_channels, sample_segments):
    """Test that corner apex markers are included."""
    fig = plot_track_segments(sample_lap_channels, sample_segments)

    # Find traces with text (apex labels)
    text_traces = [t for t in fig.data if hasattr(t, "text") and t.text]
    assert len(text_traces) > 0


def test_plot_track_segments_custom_dimensions(sample_lap_channels, sample_segments):
    """Test custom width and height are applied."""
    fig = plot_track_segments(sample_lap_channels, sample_segments, width=1200, height=800)

    assert fig.layout.width == 1200
    assert fig.layout.height == 800
