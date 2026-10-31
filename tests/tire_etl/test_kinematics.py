"""IMU validation: synthetic sessions with a known-good and known-bad IMU."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest

from motorsports_data_notebook.tire_etl import kinematics as k
from motorsports_data_notebook.tire_etl.dataset import build_timeseries_table

G = k.G_MS2
LAP_S = 60.0


def _synthetic_lap(seed: int = 0):
    """A 60 s lap: speed 20-45 m/s, alternating right/left corners.

    Returns the 25 Hz GPS arrays (t, lat, lon, v) and the true lateral /
    longitudinal g and yaw rate (deg/s, right-positive) on that clock.
    """
    rng = np.random.default_rng(seed)
    hz = 25.0
    t = np.arange(0.0, LAP_S, 1.0 / hz)
    v = 32.0 + 12.0 * np.sin(2 * math.pi * t / 20.0)  # 20..44 m/s
    # curvature profile: right (positive) and left corners of radius ~60-120 m
    kappa = np.zeros_like(t)
    for start, dur, radius, sign in (
        (5, 6, 80.0, 1),
        (18, 5, 60.0, -1),
        (31, 7, 120.0, 1),
        (46, 6, 70.0, -1),
    ):
        m = (t >= start) & (t < start + dur)
        kappa[m] = sign / radius
    # smooth the curvature edges a little (a car cannot step its curvature)
    kern = np.ones(13) / 13.0
    kappa = np.convolve(kappa, kern, mode="same")
    heading = np.cumsum(-kappa * v / hz)  # right turn = clockwise = negative heading rate
    x = np.cumsum(v * np.cos(heading) / hz)
    y = np.cumsum(v * np.sin(heading) / hz)
    lat = 35.0 + y / 111194.9
    lon = 139.0 + x / (111194.9 * math.cos(math.radians(35.0)))
    a_lat = v * v * kappa / G
    a_long = np.gradient(v, t) / G
    yaw = v * kappa * 180.0 / math.pi
    return t, lat, lon, v, a_lat, a_long, yaw, rng


def _session_table(
    *,
    lat_scale: float = 1.0,
    long_scale: float = 1.0,
    flip_lat: bool = False,
    swap_axes: bool = False,
    flat_lat: bool = False,
    drop_imu: bool = False,
    yaw_scale: float = 1.0,
    with_gps_channels: bool = True,
    gps_channel_scale: float = 1.0,
) -> pa.Table:
    t, lat, lon, v, a_lat, a_long, yaw, rng = _synthetic_lap()
    n = len(t)
    keys = lambda tt: {  # noqa: E731
        "lap_num": pa.array(np.ones(len(tt), dtype=np.int16)),
        "stint_id": pa.array(np.ones(len(tt), dtype=np.int16)),
        "sample_idx": pa.array(np.arange(len(tt), dtype=np.int32)),
        "t_session_s": pa.array(tt, type=pa.float64()),
        "t_lap_s": pa.array(tt.astype(np.float32)),
    }
    gps = dict(keys(t))
    gps["speed_ms"] = pa.array(v.astype(np.float32))
    gps["gps_lat"] = pa.array(lat, type=pa.float64())
    gps["gps_lon"] = pa.array(lon, type=pa.float64())
    if with_gps_channels:
        gps["gps_lat_g"] = pa.array((a_lat * gps_channel_scale).astype(np.float32))
        gps["gps_long_g"] = pa.array((a_long * gps_channel_scale).astype(np.float32))
    parts = [("gps", "gps", pa.table(gps))]
    if not drop_imu:
        t_imu = np.arange(0.0, LAP_S, 1.0 / 50.0)
        lat_i = np.interp(t_imu, t, a_lat) * lat_scale + rng.normal(0, 0.03, len(t_imu))
        long_i = np.interp(t_imu, t, a_long) * long_scale + rng.normal(0, 0.03, len(t_imu))
        if flip_lat:
            lat_i = -lat_i
        if swap_axes:
            lat_i, long_i = long_i, lat_i
        if flat_lat:
            lat_i = np.full(len(t_imu), 0.02)
        acc = dict(keys(t_imu))
        acc["lat_g"] = pa.array(lat_i.astype(np.float32))
        acc["long_g"] = pa.array(long_i.astype(np.float32))
        parts.append(("accel", "accel", pa.table(acc)))
        gyr = dict(keys(t_imu))
        gyr["yaw_rate_dps"] = pa.array(
            (np.interp(t_imu, t, yaw) * yaw_scale + rng.normal(0, 0.5, len(t_imu))).astype(
                np.float32
            )
        )
        parts.append(("gyro", "gyro", pa.table(gyr)))
    return build_timeseries_table("s", parts)


def test_position_reference_recovers_the_true_kinematics() -> None:
    t, lat, lon, v, a_lat, a_long, yaw, _ = _synthetic_lap()
    al, ag, yw = k.position_kinematics(t, lat, lon, v)
    m = (v >= k.MOVING_MS) & (t > 1.0) & (t < LAP_S - 1.0)  # dV/dt is one-sided at the ends
    for est, true in ((al, a_lat), (ag, a_long), (yw, yaw)):
        sl = np.polyfit(true[m], est[m], 1)[0]
        assert 0.95 <= sl <= 1.05
        assert np.corrcoef(true[m], est[m])[0, 1] > 0.98
        # the ±5 m smoothing rounds the synthetic corner entries a little
        assert np.max(np.abs(est[m] - true[m])) < 0.15 * np.max(np.abs(true))


def test_healthy_imu_is_chosen() -> None:
    ts = _session_table()
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "imu" and verdict.gyro_source == "imu"
    for key in ("imu_lat", "imu_long", "imu_yaw"):
        chk = verdict.checks[key]
        assert chk.passed, (key, chk)
        assert 0.95 <= chk.slope <= 1.05 and chk.corr >= 0.97


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"lat_scale": 0.5}, "slope"),
        ({"flip_lat": True}, "flipped"),
        ({"flat_lat": True}, "flat"),
    ],
)
def test_bad_imu_falls_back_to_the_gps_channel(kwargs, reason) -> None:
    ts = _session_table(**kwargs)
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "gps_channel"
    assert not verdict.checks["imu_lat"].passed
    if reason:
        assert verdict.checks["imu_lat"].reason == reason
    assert verdict.gyro_source == "imu"  # the gyro was fine


def test_bad_imu_and_bad_gps_channel_fall_back_to_position() -> None:
    ts = _session_table(lat_scale=0.45, gps_channel_scale=0.6)
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "gps_position"


def test_missing_imu_uses_position() -> None:
    ts = _session_table(drop_imu=True, with_gps_channels=False)
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "gps_position" and verdict.gyro_source == "gps_position"
    kin = k.session_kinematics(ts, verdict)
    assert kin is not None and len(kin) == len(ts)
    assert kin.column("lat_source")[0].as_py() == "gps_position"
    assert np.nanmax(np.abs(kin.column("lat_g").to_numpy())) > 0.5


def test_bad_gyro_falls_back_to_position_yaw() -> None:
    ts = _session_table(yaw_scale=0.5)
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "imu" and verdict.gyro_source == "gps_position"
    kin = k.session_kinematics(ts, verdict)
    assert kin is not None
    # on the IMU clock (50 Hz) with the yaw from position interpolated onto it
    assert len(kin) == 50 * int(LAP_S)
    assert kin.column("yaw_source")[0].as_py() == "gps_position"
    assert np.nanmax(np.abs(kin.column("yaw_rate_dps").to_numpy())) > 10.0


def test_no_gps_reference_keeps_the_imu_unverified() -> None:
    t_imu = np.arange(0.0, 10.0, 0.02)
    acc = pa.table(
        {
            "lap_num": pa.array(np.ones(len(t_imu), dtype=np.int16)),
            "stint_id": pa.array(np.ones(len(t_imu), dtype=np.int16)),
            "sample_idx": pa.array(np.arange(len(t_imu), dtype=np.int32)),
            "t_session_s": pa.array(t_imu, type=pa.float64()),
            "t_lap_s": pa.array(t_imu.astype(np.float32)),
            "lat_g": pa.array(np.sin(t_imu).astype(np.float32)),
            "long_g": pa.array(np.cos(t_imu).astype(np.float32)),
        }
    )
    ts = build_timeseries_table("s", [("accel", "accel", acc)])
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "imu" and verdict.note == "no_gps_reference"
    row = verdict.to_row()
    assert row["accel_source"] == "imu" and row["kin_imu_lat_slope"] is None


def test_verdict_round_trips_through_json_and_row() -> None:
    ts = _session_table()
    verdict = k.validate_session(ts, ref_lap=1)
    back = k.KinematicsVerdict.from_json(verdict.to_json())
    assert back.accel_source == verdict.accel_source
    assert back.checks["imu_lat"].slope == pytest.approx(verdict.checks["imu_lat"].slope)
    row = verdict.to_row()
    assert set(row) == {name for name, _ in k.KIN_ROW_COLUMNS}
    assert row["kin_imu_lat_ok"] is True


def test_choose_reference_lap_prefers_fastest_full_lap() -> None:
    assert (
        k.choose_reference_lap(
            [0, 1, 2, 3], [90.0, 61.0, 60.5, 80.0], ["out", "full", "full", "in"]
        )
        == 2
    )
    assert k.choose_reference_lap([0, 1], [90.0, 61.0], ["out", "in"]) == 1
    assert k.choose_reference_lap([], [], []) is None


def test_position_reference_ignores_fix_acquisition_garbage() -> None:
    """A receiver acquiring its fix reports positions anywhere on the globe
    (lat 45, lon ±179 at zero speed for the first seconds of a file) and an
    occasional spike mid-lap; the reference must interpolate across them
    instead of laying a curvature grid over the planet."""
    t, lat, lon, v, a_lat, _, yaw, rng = _synthetic_lap()
    clean_lat, clean_long, clean_yaw = k.position_kinematics(t, lat, lon, v)
    # Warm-up garbage before the lap: 4 s of (0, 0) then wandering fixes.
    hz = 25
    pre_n = 6 * hz
    t_pre = np.arange(-6.0, 0.0, 1.0 / hz)
    lat_pre = np.where(np.arange(pre_n) < 4 * hz, 0.0, 45.0 + rng.normal(0, 0.2, pre_n))
    lon_pre = np.where(np.arange(pre_n) < 4 * hz, 0.0, rng.choice([-179.0, 179.5], pre_n))
    tt = np.concatenate([t_pre, t])
    la = np.concatenate([lat_pre, lat])
    lo = np.concatenate([lon_pre, lon])
    vv = np.concatenate([np.zeros(pre_n), v])
    # And one isolated spike 500 m off the path mid-lap.
    la[pre_n + 700] += 500.0 / 111194.9
    got_lat, got_long, got_yaw = k.position_kinematics(tt, la, lo, vv)
    sel = np.ones(len(t), dtype=bool)
    sel[690:712] = False  # the spike's smoothing footprint
    assert np.allclose(got_lat[pre_n:][sel], clean_lat[sel], atol=0.05)
    assert np.allclose(got_yaw[pre_n:][sel], clean_yaw[sel], atol=2.0)
    assert np.allclose(got_long[pre_n:][25:], clean_long[25:], atol=0.05)
    assert np.all(got_lat[:pre_n] == 0.0)  # zero speed: no lateral g
    assert np.isfinite(got_lat).all()


def test_position_reference_grid_cap_never_allocates(monkeypatch) -> None:
    t, lat, lon, v, *_ = _synthetic_lap()
    monkeypatch.setattr(k, "MAX_PATH_POINTS", 100)
    a_lat, a_long, yaw = k.position_kinematics(t, lat, lon, v)
    assert np.isnan(a_lat).all() and np.isnan(yaw).all()


def test_gyro_scale_outside_the_gate_band_falls_back() -> None:
    """A gyro reading 0.87× the yaw rate (the Inferno 86 since 2025-06-09)
    is a calibration fault: the position-derived yaw replaces it, while
    the accelerometer keeps its wider band."""
    ts = _session_table(yaw_scale=0.87)
    v = k.validate_session(ts, 1)
    assert v.gyro_source == "gps_position" and v.checks["imu_yaw"].reason == "slope"
    assert v.accel_source == "imu"
    ts = _session_table(yaw_scale=0.95, lat_scale=0.9)
    v = k.validate_session(ts, 1)
    assert v.gyro_source == "imu" and v.accel_source == "imu"


def test_vibration_does_not_fail_a_healthy_imu() -> None:
    """The reference is a 0.5 s window over 25 Hz GPS; a 50 Hz IMU carrying
    0.3 g of vibration is the same sensor and must be compared at the
    reference's bandwidth, not penalised for its own."""
    import pyarrow.compute as pc

    ts = _session_table()
    rng = np.random.default_rng(7)
    n = int(pc.sum(pc.is_valid(ts.column("long_g"))).as_py())
    noisy = ts.column("long_g").to_numpy(zero_copy_only=False).astype(np.float64)
    noisy[np.isfinite(noisy)] += rng.normal(0, 0.3, n) * np.sign(
        np.sin(np.arange(n) * 2.3)
    )  # broadband vibration well above the reference's bandwidth
    ts = ts.set_column(
        ts.schema.get_field_index("long_g"), "long_g", pa.array(noisy.astype(np.float32))
    )
    v = k.validate_session(ts, 1)
    assert v.accel_source == "imu", v.checks["imu_long"]


def test_swapped_axes_are_recovered_as_imu_swapped() -> None:
    """A dash logging the longitudinal acceleration as LateralAcc and vice
    versa (the KK-SII since 2026-03-18) keeps its 50 Hz accelerometer: the
    verdict is ``imu_swapped`` and the loader hands the columns out
    exchanged, matching the true lateral g, while the stored table is
    untouched."""
    ts = _session_table(swap_axes=True)
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "imu_swapped"
    assert verdict.checks["imu_lat"].reason == "twisted"
    assert verdict.checks["imu_swapped_lat"].passed and verdict.checks["imu_swapped_long"].passed
    assert verdict.gyro_source == "imu"
    kin = k.session_kinematics(ts, verdict)
    t, _, _, _, a_lat, a_long, _, _ = _synthetic_lap()
    lat = np.interp(t, kin.column("t_session_s").to_numpy(), kin.column("lat_g").to_numpy())
    long = np.interp(t, kin.column("t_session_s").to_numpy(), kin.column("long_g").to_numpy())
    assert np.corrcoef(lat, a_lat)[0, 1] > 0.99 and np.corrcoef(long, a_long)[0, 1] > 0.95
    assert kin.column("lat_source")[0].as_py() == "imu_swapped"
    # the stored channels are still the logged (swapped) ones
    stored = ts.filter(ts.column("lat_g").is_valid()).column("lat_g").to_numpy(zero_copy_only=False)
    assert (
        np.corrcoef(
            np.interp(
                t, ts.filter(ts.column("lat_g").is_valid()).column("t_session_s").to_numpy(), stored
            ),
            a_long,
        )[0, 1]
        > 0.95
    )
    row = verdict.to_row()
    assert row["kin_imu_swapped_lat_ok"] is True and row["kin_imu_lat_ok"] is False


def test_swapped_and_scaled_imu_is_not_recovered() -> None:
    ts = _session_table(swap_axes=True, lat_scale=0.5)
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "gps_channel"
    # the scaled axis is the one that lands on the lateral reference after the exchange
    assert not verdict.checks["imu_swapped_lat"].passed
