"""Tests for the roll gradient module."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from helpers import MockLogFile, make_table_with_unit
from motorsports_data_notebook.roll_gradient import (
    RollGradientResult,
    TrackWidths,
    analyze_roll_gradient,
    analyze_roll_gradient_multi_lap,
    bin_roll_by_lateral_g,
    compute_roll_angle_deg,
    fit_roll_gradient,
    format_roll_gradient_table,
)
from motorsports_data_notebook.suspension import MotionRatios

TRACK = TrackWidths(front_mm=1500.0, rear_mm=1500.0)


def _synthetic_log(
    gradient_front: float = 1.0,
    gradient_rear: float = 1.5,
    ratios: MotionRatios | None = None,
    with_gps: bool = True,
    shock_unit: str = "mm",
    lat_unit: str = "g",
    n: int = 2000,
) -> MockLogFile:
    """Log whose pots encode a known roll gradient (deg/g) per axle.

    roll = atan((wl - wr) / track) -> for a target gradient k at lateral g a:
    wl - wr = track * tan(k * a); split symmetrically; pot = wheel * ratio.
    """
    ratios = ratios or MotionRatios(1.0, 1.0, 1.0, 1.0)
    t = np.arange(n, dtype=np.int64) * 10  # 100 Hz
    lat = 1.5 * np.sin(np.linspace(0, 4 * np.pi, n))
    lat_values = lat * 9.80665 if "m/s" in lat_unit else lat

    def pots(k: float, track: float, rl: float, rr: float):
        delta = track * np.tan(np.radians(k * lat))
        left, right = delta / 2 + 3.0, -delta / 2 + 3.0  # +3 mm common heave offset
        return left * rl, right * rr

    fl, fr = pots(gradient_front, TRACK.front_mm, ratios.front_left, ratios.front_right)
    rl, rr = pots(gradient_rear, TRACK.rear_mm, ratios.rear_left, ratios.rear_right)
    channels = {
        "LF_Shock_Pot": make_table_with_unit(t, "LF_Shock_Pot", fl, shock_unit),
        "RF_Shock_Pot": make_table_with_unit(t, "RF_Shock_Pot", fr, shock_unit),
        "LR_Shock_Pot": make_table_with_unit(t, "LR_Shock_Pot", rl, shock_unit),
        "RR_Shock_Pot": make_table_with_unit(t, "RR_Shock_Pot", rr, shock_unit),
        "LateralAcc": make_table_with_unit(t, "LateralAcc", lat_values, lat_unit),
    }
    if with_gps:
        ang = np.linspace(0, 2 * np.pi, n)
        channels["GPS Latitude"] = make_table_with_unit(
            t, "GPS Latitude", 35.0 + 0.002 * np.sin(ang), "deg"
        )
        channels["GPS Longitude"] = make_table_with_unit(
            t, "GPS Longitude", 139.0 + 0.003 * np.cos(ang), "deg"
        )
    laps = pa.table({"num": [1, 2], "start_time": [0, 10000], "end_time": [10000, 20000]})
    return MockLogFile(channels=channels, laps=laps, metadata={"Logger ID": "1"})


CHANNELS = {
    "shock_fl": "LF_Shock_Pot",
    "shock_fr": "RF_Shock_Pot",
    "shock_rl": "LR_Shock_Pot",
    "shock_rr": "RR_Shock_Pot",
    "lateral_g": "LateralAcc",
    "gps_latitude": "GPS Latitude",
    "gps_longitude": "GPS Longitude",
}


class TestTrackWidths:
    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            TrackWidths(front_mm=0, rear_mm=1500)
        with pytest.raises(ValueError):
            TrackWidths(front_mm=1500, rear_mm=-1)

    def test_coerces_to_float(self):
        tw = TrackWidths(front_mm="1520", rear_mm=1540)  # type: ignore[arg-type]
        assert tw.front_mm == pytest.approx(1520.0)


class TestComputeRollAngle:
    def test_small_angle(self):
        roll = compute_roll_angle_deg(np.array([10.0]), np.array([-10.0]), 1500.0)
        assert roll[0] == pytest.approx(np.degrees(np.arctan(20 / 1500)))
        assert roll[0] > 0  # left compressed more -> positive

    def test_symmetric_travel_is_zero_roll(self):
        roll = compute_roll_angle_deg(np.array([5.0, 20.0]), np.array([5.0, 20.0]), 1500.0)
        assert np.allclose(roll, 0.0)

    def test_invalid_track(self):
        with pytest.raises(ValueError):
            compute_roll_angle_deg(np.array([1.0]), np.array([0.0]), 0.0)


class TestFitRollGradient:
    def test_recovers_known_line(self):
        x = np.linspace(-1.5, 1.5, 200)
        y = 1.25 * x - 0.1
        slope, intercept, r2 = fit_roll_gradient(x, y)
        assert slope == pytest.approx(1.25)
        assert intercept == pytest.approx(-0.1)
        assert r2 == pytest.approx(1.0)

    def test_min_lateral_g_excludes_center(self):
        x = np.linspace(-1.5, 1.5, 201)
        y = 1.0 * x
        y[np.abs(x) < 0.2] += 5.0  # corrupt the straight-line cluster
        _, intercept_all, r2_all = fit_roll_gradient(x, y)
        slope_cut, intercept_cut, r2 = fit_roll_gradient(x, y, min_lateral_g=0.2)
        assert slope_cut == pytest.approx(1.0)
        assert intercept_cut == pytest.approx(0.0, abs=1e-9)
        assert r2 == pytest.approx(1.0)
        assert intercept_all > 0.1 and r2_all < 0.99

    def test_degenerate_inputs_return_nan(self):
        slope, intercept, r2 = fit_roll_gradient(np.array([0.5]), np.array([0.5]))
        assert np.isnan(slope) and np.isnan(intercept) and np.isnan(r2)
        slope, _, _ = fit_roll_gradient(np.full(10, 0.3), np.linspace(0, 1, 10))
        assert np.isnan(slope)


class TestBinRoll:
    def test_bins_follow_line(self):
        x = np.linspace(-1, 1, 1001)
        centers, means, counts = bin_roll_by_lateral_g(x, 2 * x, bin_size_g=0.25)
        assert len(centers) == len(means) == len(counts) > 0
        assert np.allclose(means, 2 * centers, atol=0.05)
        assert counts.sum() == 1001

    def test_empty(self):
        centers, means, counts = bin_roll_by_lateral_g(np.array([]), np.array([]))
        assert len(centers) == 0 and len(means) == 0 and len(counts) == 0


class TestAnalyzeRollGradient:
    def test_recovers_gradients_and_ratio(self):
        log = _synthetic_log(gradient_front=1.0, gradient_rear=1.5)
        result = analyze_roll_gradient(
            log, CHANNELS, MotionRatios(1, 1, 1, 1), TRACK, smoothing_window=1
        )
        assert isinstance(result, RollGradientResult)
        assert result.front.gradient_deg_per_g == pytest.approx(1.0, abs=1e-3)
        assert result.rear.gradient_deg_per_g == pytest.approx(1.5, abs=1e-3)
        assert result.front.r2 == pytest.approx(1.0, abs=1e-4)
        assert result.roll_ratio == pytest.approx(1.0 / 1.5, abs=1e-3)
        assert result.roll_ratio_r2 == pytest.approx(1.0, abs=1e-4)
        assert result.gradient_ratio == pytest.approx(result.roll_ratio, abs=1e-3)
        assert result.signs_consistent
        assert result.mean_gradient_deg_per_g == pytest.approx(1.25, abs=1e-3)
        assert result.front_minus_rear_deg_per_g == pytest.approx(-0.5, abs=1e-3)

    def test_motion_ratios_convert_pot_to_wheel(self):
        """Pots scaled by 0.5 must be divided back out to recover the wheel-space gradient."""
        ratios = MotionRatios(0.5, 0.5, 0.8, 0.8)
        log = _synthetic_log(gradient_front=1.0, gradient_rear=1.5, ratios=ratios)
        result = analyze_roll_gradient(log, CHANNELS, ratios, TRACK, smoothing_window=1)
        assert result.front.gradient_deg_per_g == pytest.approx(1.0, abs=1e-3)
        assert result.rear.gradient_deg_per_g == pytest.approx(1.5, abs=1e-3)
        # Wrong (unity) ratios would under-read the front by 2x
        wrong = analyze_roll_gradient(
            log, CHANNELS, MotionRatios(1, 1, 1, 1), TRACK, smoothing_window=1
        )
        assert wrong.front.gradient_deg_per_g == pytest.approx(0.5, abs=1e-2)

    def test_lateral_in_ms2_is_converted_to_g(self):
        log = _synthetic_log(lat_unit="m/s^2")
        result = analyze_roll_gradient(
            log, CHANNELS, MotionRatios(1, 1, 1, 1), TRACK, smoothing_window=1
        )
        assert result.front.gradient_deg_per_g == pytest.approx(1.0, abs=1e-3)
        assert np.nanmax(np.abs(result.front.lateral_g)) == pytest.approx(1.5, abs=1e-6)

    def test_track_map_positions(self):
        log = _synthetic_log(with_gps=True)
        result = analyze_roll_gradient(
            log, CHANNELS, MotionRatios(1, 1, 1, 1), TRACK, smoothing_window=1
        )
        assert result.has_track_map
        assert len(result.x_m) == len(result.front.lateral_g)

    def test_without_gps_channels(self):
        log = _synthetic_log(with_gps=False)
        channels = {k: v for k, v in CHANNELS.items() if not k.startswith("gps")}
        result = analyze_roll_gradient(log, channels, MotionRatios(1, 1, 1, 1), TRACK)
        assert not result.has_track_map
        # Configured but absent GPS channels are also tolerated
        result2 = analyze_roll_gradient(log, CHANNELS, MotionRatios(1, 1, 1, 1), TRACK)
        assert not result2.has_track_map

    def test_velocity_channels_rejected(self):
        log = _synthetic_log(shock_unit="m/s")
        with pytest.raises(ValueError, match="velocities"):
            analyze_roll_gradient(log, CHANNELS, MotionRatios(1, 1, 1, 1), TRACK)

    def test_missing_channel_key(self):
        log = _synthetic_log()
        bad = {k: v for k, v in CHANNELS.items() if k != "lateral_g"}
        with pytest.raises(KeyError):
            analyze_roll_gradient(log, bad, MotionRatios(1, 1, 1, 1), TRACK)


class TestAnalyzeRollGradientMultiLap:
    def test_concatenates_laps(self):
        log = _synthetic_log()
        single = analyze_roll_gradient(
            log, CHANNELS, MotionRatios(1, 1, 1, 1), TRACK, smoothing_window=1
        )
        multi = analyze_roll_gradient_multi_lap(
            log, [1, 2], CHANNELS, MotionRatios(1, 1, 1, 1), TRACK, smoothing_window=1
        )
        # MockLogFile.filter_by_lap returns the whole log, so two laps = 2x samples
        assert len(multi.front.lateral_g) == 2 * len(single.front.lateral_g)
        assert multi.front.gradient_deg_per_g == pytest.approx(1.0, abs=1e-3)
        assert multi.has_track_map and len(multi.x_m) == len(multi.front.lateral_g)

    def test_empty_laps(self):
        with pytest.raises(ValueError, match="empty"):
            analyze_roll_gradient_multi_lap(
                _synthetic_log(), [], CHANNELS, MotionRatios(1, 1, 1, 1), TRACK
            )

    def test_table(self):
        log = _synthetic_log()
        result = analyze_roll_gradient_multi_lap(
            log, [1], CHANNELS, MotionRatios(1, 1, 1, 1), TRACK, smoothing_window=1
        )
        df = format_roll_gradient_table(result)
        assert list(df["Axle"]) == ["Front", "Rear", "Roll ratio (front/rear)"]
        assert df.loc[0, "Gradient (deg/g)"] == pytest.approx(1.0, abs=1e-3)
        assert df.loc[2, "Gradient (deg/g)"] == pytest.approx(1 / 1.5, abs=1e-3)
