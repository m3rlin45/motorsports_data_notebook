"""Roll gradient and roll ratio analysis from shock pot travel.

Body roll per g of lateral acceleration, estimated per axle from the
difference in wheel travel between the left and right shock pots::

    roll_deg = degrees(atan((wheel_left - wheel_right) / track_width))
    roll_deg = gradient * lateral_g + intercept        (least-squares fit)

The roll ratio is the slope of front roll against rear roll (front / rear),
fitted sample by sample; it equals the ratio of the two gradients when both
fits are good and shows how the front and rear axles share the body roll.

Positive roll means the left side is compressed more than the right (the body
leans to the left, as in a right-hand turn). The sign of the fitted gradient
therefore depends on the logger's lateral-G sign convention; front and rear
should agree, and :attr:`RollGradientResult.signs_consistent` flags when
they do not (typically swapped left/right channels).

Pot travel is converted to wheel travel with the vehicle's pot/wheel
:class:`~.suspension.MotionRatios` (``wheel = pot / ratio``), so the result is
in wheel space. Because only left-right differences are used, static pot
zero offsets cancel as long as both sides were zeroed at the same ride height.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._util import get_channel_unit as _get_channel_unit
from ._util import validate_channel_names as _validate_channel_names
from .corners import gps_to_local_xy
from .suspension import MotionRatios, _channels_are_velocity

if TYPE_CHECKING:
    from ._types import LogFile

_G_MS2 = 9.80665

REQUIRED_CHANNEL_KEYS = ["shock_fl", "shock_fr", "shock_rl", "shock_rr", "lateral_g"]
# Optional: when present, each sample also gets a track-map position
GPS_CHANNEL_KEYS = ["gps_latitude", "gps_longitude"]


@dataclass
class TrackWidths:
    """Front and rear track (tread) widths in mm, hub centre to hub centre.

    Attributes
    ----------
    front_mm : float
        Front track width in mm.
    rear_mm : float
        Rear track width in mm.
    """

    front_mm: float
    rear_mm: float

    def __post_init__(self) -> None:
        self.front_mm = float(self.front_mm)
        self.rear_mm = float(self.rear_mm)
        if self.front_mm <= 0 or self.rear_mm <= 0:
            raise ValueError("Track widths must be positive (mm)")


@dataclass
class AxleRollData:
    """Roll-vs-lateral-G data and fit for one axle.

    Attributes
    ----------
    axle : str
        "front" or "rear".
    lateral_g : np.ndarray
        Lateral acceleration samples in g.
    roll_deg : np.ndarray
        Axle roll angle samples in degrees (positive = left side compressed).
    gradient_deg_per_g : float
        Fitted slope, degrees of roll per g (signed).
    intercept_deg : float
        Fitted roll at zero lateral g (static cross-weight / ride height asymmetry).
    r2 : float
        Coefficient of determination of the linear fit.
    bin_centers_g : np.ndarray
        Centres of the lateral-g bins used for the binned means.
    bin_mean_deg : np.ndarray
        Mean roll angle per bin.
    bin_counts : np.ndarray
        Sample count per bin.
    n_samples : int
        Number of samples used in the fit.
    """

    axle: str
    lateral_g: np.ndarray
    roll_deg: np.ndarray
    gradient_deg_per_g: float
    intercept_deg: float
    r2: float
    bin_centers_g: np.ndarray
    bin_mean_deg: np.ndarray
    bin_counts: np.ndarray
    n_samples: int

    @property
    def abs_gradient_deg_per_g(self) -> float:
        """Magnitude of the roll gradient in deg/g."""
        return abs(self.gradient_deg_per_g)


@dataclass
class RollGradientResult:
    """Roll gradient analysis for both axles.

    Attributes
    ----------
    front : AxleRollData
        Front axle data and fit.
    rear : AxleRollData
        Rear axle data and fit.
    track_widths : TrackWidths
        Track widths used for the roll angle.
    lateral_g_unit : str
        Unit of the lateral acceleration channel as logged.
    roll_ratio : float
        Slope of front roll vs rear roll (front / rear), least squares.
    roll_ratio_intercept_deg : float
        Intercept of that fit.
    roll_ratio_r2 : float
        R² of that fit.
    ratio_bin_centers_deg, ratio_bin_mean_deg, ratio_bin_counts : np.ndarray
        Front roll binned by rear roll (for plotting).
    x_m, y_m : np.ndarray or None
        Local track position of every sample in metres (from GPS), aligned
        with ``front.lateral_g`` etc., when GPS channels were available.
    """

    front: AxleRollData
    rear: AxleRollData
    track_widths: TrackWidths
    lateral_g_unit: str = "g"
    roll_ratio: float = float("nan")
    roll_ratio_intercept_deg: float = float("nan")
    roll_ratio_r2: float = float("nan")
    ratio_bin_centers_deg: np.ndarray = None  # type: ignore[assignment]
    ratio_bin_mean_deg: np.ndarray = None  # type: ignore[assignment]
    ratio_bin_counts: np.ndarray = None  # type: ignore[assignment]
    x_m: np.ndarray | None = None
    y_m: np.ndarray | None = None

    @property
    def has_track_map(self) -> bool:
        """True if per-sample track positions (local XY, metres) are available."""
        return self.x_m is not None and self.y_m is not None and len(self.x_m) > 0

    def __post_init__(self) -> None:
        if self.ratio_bin_centers_deg is None:
            self.ratio_bin_centers_deg = np.array([])
        if self.ratio_bin_mean_deg is None:
            self.ratio_bin_mean_deg = np.array([])
        if self.ratio_bin_counts is None:
            self.ratio_bin_counts = np.array([], dtype=int)

    @property
    def gradient_ratio(self) -> float:
        """Front gradient / rear gradient (alternative roll ratio estimate)."""
        if self.rear.gradient_deg_per_g == 0 or not np.isfinite(self.rear.gradient_deg_per_g):
            return float("nan")
        return self.front.gradient_deg_per_g / self.rear.gradient_deg_per_g

    @property
    def mean_gradient_deg_per_g(self) -> float:
        """Mean of the front and rear gradient magnitudes (chassis roll gradient)."""
        return 0.5 * (self.front.abs_gradient_deg_per_g + self.rear.abs_gradient_deg_per_g)

    @property
    def front_minus_rear_deg_per_g(self) -> float:
        """Front gradient magnitude minus rear; non-zero means chassis/measurement twist."""
        return self.front.abs_gradient_deg_per_g - self.rear.abs_gradient_deg_per_g

    @property
    def signs_consistent(self) -> bool:
        """True if front and rear roll in the same direction for a given lateral g."""
        return bool(np.sign(self.front.gradient_deg_per_g) == np.sign(self.rear.gradient_deg_per_g))


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def compute_roll_angle_deg(
    left_travel_mm: np.ndarray,
    right_travel_mm: np.ndarray,
    track_mm: float,
) -> np.ndarray:
    """Axle roll angle from left/right wheel travel.

    Parameters
    ----------
    left_travel_mm, right_travel_mm : np.ndarray
        Wheel travel in mm, positive = bump.
    track_mm : float
        Track width in mm.

    Returns
    -------
    np.ndarray
        Roll angle in degrees, positive when the left side is compressed more.

    Examples
    --------
    >>> float(compute_roll_angle_deg(np.array([10.0]), np.array([-10.0]), 1500.0)[0])
    0.7639...
    """
    if track_mm <= 0:
        raise ValueError("track_mm must be positive")
    delta = np.asarray(left_travel_mm, dtype=float) - np.asarray(right_travel_mm, dtype=float)
    return np.asarray(np.degrees(np.arctan(delta / track_mm)))


def fit_roll_gradient(
    lateral_g: np.ndarray,
    roll_deg: np.ndarray,
    min_lateral_g: float = 0.0,
) -> tuple[float, float, float]:
    """Least-squares line ``roll = gradient * lateral_g + intercept``.

    Parameters
    ----------
    lateral_g : np.ndarray
        Lateral acceleration in g.
    roll_deg : np.ndarray
        Roll angle in degrees.
    min_lateral_g : float, default=0.0
        Ignore samples with ``|lateral_g|`` below this (e.g. 0.2 to drop the
        straight-line cluster).

    Returns
    -------
    tuple[float, float, float]
        (gradient_deg_per_g, intercept_deg, r2). All NaN if fewer than two
        usable samples or no spread in lateral g.
    """
    x = np.asarray(lateral_g, dtype=float)
    y = np.asarray(roll_deg, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x) >= min_lateral_g)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def bin_roll_by_lateral_g(
    lateral_g: np.ndarray,
    roll_deg: np.ndarray,
    bin_size_g: float = 0.1,
    min_count: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean roll angle in fixed-width lateral-g bins (for plotting).

    Parameters
    ----------
    lateral_g, roll_deg : np.ndarray
        Samples.
    bin_size_g : float, default=0.1
        Bin width in g.
    min_count : int, default=5
        Bins with fewer samples are dropped.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (bin_centers_g, bin_mean_deg, bin_counts)
    """
    x = np.asarray(lateral_g, dtype=float)
    y = np.asarray(roll_deg, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0 or bin_size_g <= 0:
        return np.array([]), np.array([]), np.array([], dtype=int)
    lo = np.floor(x.min() / bin_size_g) * bin_size_g
    hi = np.ceil(x.max() / bin_size_g) * bin_size_g + bin_size_g * 0.5
    edges = np.arange(lo, hi, bin_size_g)
    if len(edges) < 2:
        edges = np.array([lo, lo + bin_size_g])
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    sums = np.bincount(idx, weights=y, minlength=len(edges) - 1)
    keep = counts >= max(1, min_count)
    centers = (edges[:-1] + edges[1:]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    return centers[keep], means[keep], counts[keep]


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding (no-op for window <= 1)."""
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
    return np.asarray(np.convolve(padded, kernel, mode="valid"))


def _lateral_to_g(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert a lateral acceleration channel to g if it is logged in m/s²."""
    u = unit.lower().replace(" ", "")
    if "m/s" in u:
        return values / _G_MS2
    return values


def _build_result(
    lateral: np.ndarray,
    roll_front: np.ndarray,
    roll_rear: np.ndarray,
    track_widths: TrackWidths,
    unit: str,
    min_lateral_g: float,
    bin_size_g: float,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
) -> RollGradientResult:
    """Assemble the result: per-axle fits plus the front-vs-rear roll ratio fit."""
    x_m = y_m = None
    if lat is not None and lon is not None and len(lat) == len(lateral) and len(lat) > 0:
        x_m, y_m = gps_to_local_xy(lat, lon)
    front = _build_axle("front", lateral, roll_front, min_lateral_g, bin_size_g)
    rear = _build_axle("rear", lateral, roll_rear, min_lateral_g, bin_size_g)
    # Roll ratio: front roll against rear roll on the samples used for the fits
    mask = np.isfinite(lateral) & (np.abs(lateral) >= min_lateral_g)
    ratio, ratio_icpt, ratio_r2 = fit_roll_gradient(roll_rear[mask], roll_front[mask])
    # Bin front roll by rear roll with a bin width that gives ~0.1 g resolution
    rear_bin = abs(rear.gradient_deg_per_g) * bin_size_g
    if not np.isfinite(rear_bin) or rear_bin <= 0:
        rear_bin = 0.1
    centers, means, counts = bin_roll_by_lateral_g(roll_rear[mask], roll_front[mask], rear_bin)
    return RollGradientResult(
        front=front,
        rear=rear,
        track_widths=track_widths,
        lateral_g_unit=unit,
        roll_ratio=ratio,
        roll_ratio_intercept_deg=ratio_icpt,
        roll_ratio_r2=ratio_r2,
        ratio_bin_centers_deg=centers,
        ratio_bin_mean_deg=means,
        ratio_bin_counts=counts,
        x_m=x_m,
        y_m=y_m,
    )


def _build_axle(
    axle: str,
    lateral_g: np.ndarray,
    roll_deg: np.ndarray,
    min_lateral_g: float,
    bin_size_g: float,
) -> AxleRollData:
    gradient, intercept, r2 = fit_roll_gradient(lateral_g, roll_deg, min_lateral_g)
    centers, means, counts = bin_roll_by_lateral_g(lateral_g, roll_deg, bin_size_g)
    used = int(np.sum(np.isfinite(lateral_g) & (np.abs(lateral_g) >= min_lateral_g)))
    return AxleRollData(
        axle=axle,
        lateral_g=lateral_g,
        roll_deg=roll_deg,
        gradient_deg_per_g=gradient,
        intercept_deg=intercept,
        r2=r2,
        bin_centers_g=centers,
        bin_mean_deg=means,
        bin_counts=counts,
        n_samples=used,
    )


def _gps_channels(log: "LogFile", channel_names: dict[str, str]) -> tuple[str, str] | None:
    """GPS lat/lon channel names if configured and present in the log, else None."""
    lat_key = channel_names.get("gps_latitude")
    lon_key = channel_names.get("gps_longitude")
    if lat_key and lon_key and lat_key in log.channels and lon_key in log.channels:
        return lat_key, lon_key
    return None


def _collect_aligned(
    log: "LogFile",
    channel_names: dict[str, str],
    motion_ratios: MotionRatios,
    track_widths: TrackWidths,
    smoothing_window: int,
    gps: tuple[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray | None, np.ndarray | None]:
    """Return (lateral_g, roll_front_deg, roll_rear_deg, lateral_unit, lat, lon) for one lap.

    ``lat`` / ``lon`` are None when GPS channels are not requested.
    """
    fl, fr, rl, rr = (channel_names[k] for k in ("shock_fl", "shock_fr", "shock_rl", "shock_rr"))
    lat = channel_names["lateral_g"]
    wanted = [fl, fr, rl, rr, lat] + (list(gps) if gps else [])
    aligned = log.select_channels(wanted).resample_to_channel(fl).channels

    unit = _get_channel_unit(aligned[lat], lat)
    lateral = _lateral_to_g(aligned[lat].column(lat).to_numpy().astype(float), unit)
    gps_lat = gps_lon = None
    if gps:
        gps_lat = aligned[gps[0]].column(gps[0]).to_numpy().astype(float)
        gps_lon = aligned[gps[1]].column(gps[1]).to_numpy().astype(float)

    def wheel(name: str, ratio: float) -> np.ndarray:
        return _smooth(
            aligned[name].column(name).to_numpy().astype(float) / ratio, smoothing_window
        )

    roll_front = compute_roll_angle_deg(
        wheel(fl, motion_ratios.front_left),
        wheel(fr, motion_ratios.front_right),
        track_widths.front_mm,
    )
    roll_rear = compute_roll_angle_deg(
        wheel(rl, motion_ratios.rear_left),
        wheel(rr, motion_ratios.rear_right),
        track_widths.rear_mm,
    )
    return _smooth(lateral, smoothing_window), roll_front, roll_rear, unit, gps_lat, gps_lon


def _check_inputs(log: "LogFile", channel_names: dict[str, str], caller: str) -> None:
    _validate_channel_names(channel_names, REQUIRED_CHANNEL_KEYS, caller)
    if _channels_are_velocity(log, channel_names["shock_fl"]):
        raise ValueError(
            "Shock channels contain velocities (e.g. iRacing); roll gradient needs "
            "shock pot displacement"
        )


# ---------------------------------------------------------------------------
# Public analysis entry points
# ---------------------------------------------------------------------------


def analyze_roll_gradient(
    log: "LogFile",
    channel_names: dict[str, str],
    motion_ratios: MotionRatios,
    track_widths: TrackWidths,
    smoothing_window: int = 5,
    min_lateral_g: float = 0.0,
    bin_size_g: float = 0.1,
) -> RollGradientResult:
    """Roll gradient for a single lap (``log`` pre-filtered with ``filter_by_lap``).

    Parameters
    ----------
    log : LogFile
        LogFile filtered to one lap.
    channel_names : dict[str, str]
        Channel mapping with keys ``shock_fl``, ``shock_fr``, ``shock_rl``,
        ``shock_rr`` and ``lateral_g``. ``gps_latitude`` / ``gps_longitude``
        are optional and add a track position per sample.
    motion_ratios : MotionRatios
        Pot/wheel motion ratios (``wheel = pot / ratio``).
    track_widths : TrackWidths
        Front and rear track widths in mm.
    smoothing_window : int, default=5
        Moving-average window (samples) applied to wheel travel and lateral g.
    min_lateral_g : float, default=0.0
        Exclude samples with ``|lateral_g|`` below this from the fit.
    bin_size_g : float, default=0.1
        Bin width for the binned means.

    Returns
    -------
    RollGradientResult

    Raises
    ------
    KeyError
        If a required channel key is missing.
    ValueError
        If the shock channels are velocities rather than displacement.
    """
    _check_inputs(log, channel_names, "analyze_roll_gradient")
    lateral, roll_f, roll_r, unit, lat, lon = _collect_aligned(
        log,
        channel_names,
        motion_ratios,
        track_widths,
        smoothing_window,
        gps=_gps_channels(log, channel_names),
    )
    return _build_result(
        lateral, roll_f, roll_r, track_widths, unit, min_lateral_g, bin_size_g, lat, lon
    )


def analyze_roll_gradient_multi_lap(
    log: "LogFile",
    lap_numbers: list[int],
    channel_names: dict[str, str],
    motion_ratios: MotionRatios,
    track_widths: TrackWidths,
    smoothing_window: int = 5,
    min_lateral_g: float = 0.0,
    bin_size_g: float = 0.1,
) -> RollGradientResult:
    """Roll gradient across several laps (samples concatenated before fitting).

    Parameters
    ----------
    log : LogFile
        Unfiltered LogFile containing all laps.
    lap_numbers : list[int]
        Laps to include.
    channel_names, motion_ratios, track_widths, smoothing_window, min_lateral_g, bin_size_g
        See :func:`analyze_roll_gradient`.

    Returns
    -------
    RollGradientResult

    Raises
    ------
    ValueError
        If ``lap_numbers`` is empty, the shock channels are velocities, or no
        lap could be processed.
    KeyError
        If a required channel key is missing.
    """
    if not lap_numbers:
        raise ValueError("lap_numbers cannot be empty")
    _check_inputs(log, channel_names, "analyze_roll_gradient_multi_lap")

    gps = _gps_channels(log, channel_names)
    lat_parts: list[np.ndarray] = []
    front_parts: list[np.ndarray] = []
    rear_parts: list[np.ndarray] = []
    gps_lat_parts: list[np.ndarray] = []
    gps_lon_parts: list[np.ndarray] = []
    unit = ""
    for lap_num in lap_numbers:
        try:
            lateral, roll_f, roll_r, lap_unit, g_lat, g_lon = _collect_aligned(
                log.filter_by_lap(lap_num),
                channel_names,
                motion_ratios,
                track_widths,
                smoothing_window,
                gps=gps,
            )
        except Exception:
            continue
        unit = unit or lap_unit
        lat_parts.append(lateral)
        front_parts.append(roll_f)
        rear_parts.append(roll_r)
        if g_lat is not None and g_lon is not None:
            gps_lat_parts.append(g_lat)
            gps_lon_parts.append(g_lon)

    if not lat_parts:
        raise ValueError("No laps could be processed for roll gradient analysis")

    have_gps = gps is not None and len(gps_lat_parts) == len(lat_parts)
    return _build_result(
        np.concatenate(lat_parts),
        np.concatenate(front_parts),
        np.concatenate(rear_parts),
        track_widths,
        unit,
        min_lateral_g,
        bin_size_g,
        np.concatenate(gps_lat_parts) if have_gps else None,
        np.concatenate(gps_lon_parts) if have_gps else None,
    )


def format_roll_gradient_table(result: RollGradientResult) -> pd.DataFrame:
    """Summary table with one row per axle plus the chassis mean.

    Parameters
    ----------
    result : RollGradientResult
        Analysis result.

    Returns
    -------
    pd.DataFrame
        Columns: Axle, Gradient (deg/g), |Gradient| (deg/g), Intercept (deg), R², Samples.
        The last row is the roll ratio (front roll / rear roll, dimensionless).
    """
    rows = []
    for axle in (result.front, result.rear):
        rows.append(
            {
                "Axle": axle.axle.capitalize(),
                "Gradient (deg/g)": axle.gradient_deg_per_g,
                "|Gradient| (deg/g)": axle.abs_gradient_deg_per_g,
                "Intercept (deg)": axle.intercept_deg,
                "R²": axle.r2,
                "Samples": axle.n_samples,
            }
        )
    rows.append(
        {
            "Axle": "Roll ratio (front/rear)",
            "Gradient (deg/g)": result.roll_ratio,
            "|Gradient| (deg/g)": abs(result.roll_ratio),
            "Intercept (deg)": result.roll_ratio_intercept_deg,
            "R²": result.roll_ratio_r2,
            "Samples": min(result.front.n_samples, result.rear.n_samples),
        }
    )
    return pd.DataFrame(rows)
