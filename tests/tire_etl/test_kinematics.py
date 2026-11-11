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
    n_laps: int = 1,
    imu_stops_at_s: float | None = None,
    imu_frozen_s: tuple[float, float] | None = None,
    gps_spike_at_s: float | None = None,
) -> pa.Table:
    """One session of ``n_laps`` identical laps (lap numbers 1..n) driven as
    one continuous path.

    ``imu_stops_at_s`` ends the accelerometer and gyro streams at that
    session time; ``imu_frozen_s`` holds the accelerometer on one value
    inside ``(start, end)``; ``gps_spike_at_s`` puts a 12 g spike of half a
    second into the logger's GPS-derived lateral channel.
    """
    t, lat, lon, v, a_lat, a_long, yaw, rng = _synthetic_lap()
    if n_laps > 1:
        t = np.concatenate([t + i * LAP_S for i in range(n_laps)])
        v, a_lat, a_long, yaw = (np.tile(x, n_laps) for x in (v, a_lat, a_long, yaw))
        # one continuous path: the laps follow each other, they do not restart
        heading = np.cumsum(-(a_lat * G / (v * v)) * v / 25.0)
        x_m = np.cumsum(v * np.cos(heading) / 25.0)
        y_m = np.cumsum(v * np.sin(heading) / 25.0)
        lat = 35.0 + y_m / 111194.9
        lon = 139.0 + x_m / (111194.9 * math.cos(math.radians(35.0)))
    n = len(t)
    keys = lambda tt: {  # noqa: E731
        "lap_num": pa.array((np.floor(tt / LAP_S) + 1).astype(np.int16)),
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
        ch_lat = a_lat * gps_channel_scale
        if gps_spike_at_s is not None:
            ch_lat = np.where((t >= gps_spike_at_s) & (t < gps_spike_at_s + 0.5), 12.0, ch_lat)
        gps["gps_lat_g"] = pa.array(ch_lat.astype(np.float32))
        gps["gps_long_g"] = pa.array((a_long * gps_channel_scale).astype(np.float32))
    parts = [("gps", "gps", pa.table(gps))]
    if not drop_imu:
        t_imu = np.arange(0.0, LAP_S * n_laps, 1.0 / 50.0)
        lat_i = np.interp(t_imu, t, a_lat) * lat_scale + rng.normal(0, 0.03, len(t_imu))
        long_i = np.interp(t_imu, t, a_long) * long_scale + rng.normal(0, 0.03, len(t_imu))
        if flip_lat:
            lat_i = -lat_i
        if swap_axes:
            lat_i, long_i = long_i, lat_i
        if flat_lat:
            lat_i = np.full(len(t_imu), 0.02)
        yaw_i = np.interp(t_imu, t, yaw) * yaw_scale + rng.normal(0, 0.5, len(t_imu))
        live = np.ones(len(t_imu), dtype=bool)
        if imu_stops_at_s is not None:
            live &= t_imu < imu_stops_at_s
        if imu_frozen_s is not None:
            stuck = (t_imu > imu_frozen_s[0]) & (t_imu < imu_frozen_s[1])
            lat_i = np.where(stuck, 0.4, lat_i)
            long_i = np.where(stuck, -0.1, long_i)
        t_imu, lat_i, long_i, yaw_i = t_imu[live], lat_i[live], long_i[live], yaw_i[live]
        acc = dict(keys(t_imu))
        acc["lat_g"] = pa.array(lat_i.astype(np.float32))
        acc["long_g"] = pa.array(long_i.astype(np.float32))
        parts.append(("accel", "accel", pa.table(acc)))
        gyr = dict(keys(t_imu))
        gyr["yaw_rate_dps"] = pa.array(yaw_i.astype(np.float32))
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


# ------------------------------------------------------------ per-window source

N_WIN = 15  # five 60 s laps in 20 s windows


def _sources(kin: pa.Table) -> dict[int, set[str]]:
    """The lat_source values of the kinematics rows, per 20 s window."""
    w = np.floor(kin.column("t_session_s").to_numpy() / k.WINDOW_S).astype(int)
    src = np.array(kin.column("lat_source").to_pylist())
    return {int(i): set(src[w == i]) for i in np.unique(w)}


def test_healthy_session_uses_the_imu_in_every_window() -> None:
    ts = _session_table(n_laps=5)
    verdict = k.validate_session(ts, ref_lap=3)
    assert verdict.accel_source == "imu" and verdict.gps_fallback == "gps_channel"
    assert verdict.windows == "I" * N_WIN and verdict.window_first == 0
    assert verdict.n_mismatch == 0
    kin = k.session_kinematics(ts, verdict)
    assert kin is not None and set(kin.column("lat_source").to_pylist()) == {"imu"}
    assert len(kin) == 50 * 5 * int(LAP_S)  # the IMU's own clock


def test_imu_that_stops_mid_run_hands_over_to_gps() -> None:
    # The stream ends half-way through lap 3 (150 s): the windows it covers
    # use it, the one it dies in and every later one use the GPS channels.
    ts = _session_table(n_laps=5, imu_stops_at_s=2.5 * LAP_S)
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "imu" and verdict.gyro_source == "imu"
    assert verdict.windows == "I" * 7 + "G" * 8
    assert verdict.n_mismatch == 0  # a window without IMU samples is not a mismatch
    kin = k.session_kinematics(ts, verdict)
    assert kin is not None
    by_window = _sources(kin)
    assert all(by_window[i] == {"imu"} for i in range(7))
    assert all(by_window[i] == {"gps_channel"} for i in range(7, N_WIN))
    assert sorted(set(kin.column("lap_num").to_pylist())) == [1, 2, 3, 4, 5]
    assert not np.isnan(kin.column("lat_g").to_numpy()).any()
    # the yaw rate follows the gyro while it lasts, the position after
    yaw_src = np.array(kin.column("yaw_source").to_pylist())
    t = kin.column("t_session_s").to_numpy()
    assert set(yaw_src[t < 140.0]) == {"imu"} and set(yaw_src[t > 160.0]) == {"gps_position"}
    row = verdict.to_row()
    assert (row["kin_win_imu"], row["kin_win_gps"], row["kin_win_discard"]) == (7, 8, 0)
    assert row["kin_gps_fallback"] == "gps_channel"


def test_imu_dead_by_the_fastest_lap_is_scored_on_a_lap_it_covers() -> None:
    ts = _session_table(n_laps=5, imu_stops_at_s=2.5 * LAP_S)
    # Scored on lap 5 alone the IMU has no samples: the session is all GPS.
    alone = k.validate_session(ts, ref_lap=5)
    assert alone.accel_source == "gps_channel" and alone.windows == "G" * N_WIN
    assert alone.checks["imu_lat"].reason in ("coverage", "no_samples")
    # Given the lap order it is scored on lap 2, the fastest it covers
    # (lap 3 is only half covered), and used for the windows it has.
    verdict = k.validate_session(ts, ref_lap=5, ref_laps=[5, 4, 3, 2, 1])
    assert verdict.imu_ref_lap == 2 and verdict.ref_lap == 5
    assert verdict.accel_source == "imu" and verdict.windows == "I" * 7 + "G" * 8
    assert verdict.to_row()["kin_imu_ref_lap"] == 2


def test_frozen_imu_window_uses_gps() -> None:
    # The accelerometer holds one value for two windows while the car corners.
    ts = _session_table(n_laps=5, imu_frozen_s=(200.0, 240.0))
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "imu"
    assert verdict.windows == "I" * 10 + "GG" + "I" * 3
    assert verdict.n_mismatch == 2
    kin = k.session_kinematics(ts, verdict)
    assert kin is not None
    by_window = _sources(kin)
    assert by_window[10] == {"gps_channel"} and by_window[11] == {"gps_channel"}
    assert by_window[9] == {"imu"} and by_window[12] == {"imu"}


def test_too_many_mismatched_windows_distrust_the_imu_for_the_session() -> None:
    ts = _session_table(n_laps=5, imu_frozen_s=(100.0, 200.0))  # five windows
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.n_mismatch == 5 > k.MISMATCH_CAP
    assert verdict.accel_source == "gps_channel" and verdict.note == "imu_window_mismatch"
    assert verdict.windows == "G" * N_WIN
    kin = k.session_kinematics(ts, verdict)
    assert kin is not None and set(kin.column("lat_source").to_pylist()) == {"gps_channel"}


def test_gps_spike_without_an_imu_is_discarded() -> None:
    from motorsports_data_notebook.tire_etl.aggregates import compute_lap_dynamics

    ts = _session_table(n_laps=5, drop_imu=True, gps_spike_at_s=130.0)
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "gps_channel"
    assert verdict.windows == "G" * 6 + "D" + "G" * 8
    kin = k.session_kinematics(ts, verdict)
    assert kin is not None and len(kin) == len(ts)  # the hole keeps its rows
    by_window = _sources(kin)
    assert by_window[6] == {k.DISCARD} and by_window[5] == {"gps_channel"}
    lat = kin.column("lat_g").to_numpy()
    w = np.floor(kin.column("t_session_s").to_numpy() / k.WINDOW_S).astype(int)
    assert np.isnan(lat[w == 6]).all() and not np.isnan(lat[w != 6]).any()
    assert np.nanmax(np.abs(lat)) < k.GPS_SANE_MAX_G  # the spike never reaches a consumer
    # the lap with the hole has no heat proxy; its neighbours do
    lap_of = kin.column("lap_num").to_numpy()
    dyn = {
        n: compute_lap_dynamics(
            ts.filter(pa.array(ts.column("lap_num").to_numpy() == n)),
            kin.filter(pa.array(lap_of == n)),
        )
        for n in (2, 3)
    }
    assert np.isnan(dyn[3]["heat_proxy"]) and np.isnan(dyn[3]["lat_g_peak"])
    assert dyn[2]["heat_proxy"] > 0


def test_gps_spike_is_harmless_while_the_imu_is_live() -> None:
    ts = _session_table(n_laps=5, gps_spike_at_s=130.0)
    verdict = k.validate_session(ts, ref_lap=1)
    assert verdict.accel_source == "imu" and verdict.windows == "I" * N_WIN


def test_window_codes_and_json_round_trip() -> None:
    ts = _session_table(n_laps=5, imu_stops_at_s=2.5 * LAP_S)
    verdict = k.validate_session(ts, ref_lap=1)
    back = k.KinematicsVerdict.from_json(verdict.to_json())
    assert back.windows == verdict.windows and back.gps_fallback == "gps_channel"
    codes = back.window_codes(np.array([0.0, 139.9, 140.0, 299.9, 300.0, -1.0, np.nan]))
    assert codes.tolist() == [b"I", b"I", b"G", b"G", b"-", b"-", b"-"]
    # a verdict without windows (the whole session from accel_source) reads as empty
    plain = k.KinematicsVerdict(accel_source="imu")
    assert plain.window_codes(np.array([10.0])).tolist() == [b"-"]
    assert k.session_kinematics(ts, plain) is not None


def test_discard_mark_is_a_sparse_column_on_every_timebase() -> None:
    ts = _session_table(n_laps=5, drop_imu=True, gps_spike_at_s=130.0)
    verdict = k.validate_session(ts, ref_lap=1)
    marked = k.mark_discarded(ts, verdict)
    assert marked.schema.field("discard").type == pa.bool_()
    # after the key columns, before the channels
    names = marked.schema.names
    assert names.index("t_lap_s") < names.index("discard") < names.index("speed_ms")
    col = marked.column("discard")
    t = marked.column("t_session_s").to_numpy()
    in_window = (t >= 120.0) & (t < 140.0)  # window 6, the one with the spike
    assert col.null_count == int((~in_window).sum())  # null everywhere else
    assert all(v is True for v in col.filter(pa.array(in_window)).to_pylist())
    # the samples of the marked rows are still there
    spike = marked.filter(pa.array(in_window)).column("gps_lat_g").to_numpy()
    assert np.nanmax(spike) == pytest.approx(12.0)
    # a session without a discarded window has the column, all null
    clean = k.mark_discarded(
        _session_table(n_laps=5), k.validate_session(_session_table(n_laps=5), 1)
    )
    assert clean.column("discard").null_count == len(clean)
    # an IMU that stopped earlier has no rows there: the mark sits on the rows that exist
    frozen_and_spiky = _session_table(n_laps=5, imu_stops_at_s=100.0, gps_spike_at_s=130.0)
    v2 = k.validate_session(frozen_and_spiky, ref_lap=1)
    m2 = k.mark_discarded(frozen_and_spiky, v2)
    hit = m2.filter(pc_is_true(m2.column("discard")))
    assert v2.windows[6] == "D" and set(hit.column("family").to_pylist()) == {"gps"}
    again = k.mark_discarded(m2, v2)  # idempotent
    assert again.schema.names == m2.schema.names and again.column("discard").equals(
        m2.column("discard")
    )


def pc_is_true(col: pa.ChunkedArray) -> pa.ChunkedArray:
    import pyarrow.compute as pc

    return pc.fill_null(col, False)
