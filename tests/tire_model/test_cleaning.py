"""Tests for the training-side cleaning: stale-TPMS masking, rebuilt
per-lap TPMS aggregates and the lap usability
filters. The dataset on disk is raw; these rules run when the model loads it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from motorsports_data_notebook.tire_etl.aggregates import compute_corner_aggregates
from motorsports_data_notebook.tire_etl.dataset import load_group
from tests._timeseries_fixture import write_session_timeseries
from motorsports_data_notebook.tire_model.cleaning import (
    TPMS_PRESSURE_STEP_BAR,
    FilterConfig,
    _stale_prefix_len,
    apply_filters,
    clean_laps,
    load_tpms_timeseries,
    mask_stale_session_prefix,
    mask_stale_stint_starts,
    mask_stale_tpms,
)


def _laps(rows: list[dict]) -> pa.Table:
    # Column-orient; align on keys.
    if not rows:
        return pa.table({})
    cols: dict[str, list] = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k in cols:
            cols[k].append(r.get(k, None))
    return pa.table(cols)


def _base_lap(**overrides) -> dict:
    d = {
        "lap_num": 1,
        "stint_id": 1,
        "lap_time_s": 60.0,
        "is_outlap": False,
        "is_inlap": False,
        "speed_kmh_mean": 120.0,
        "on_track_s": 60.0,
        # Healthy warmup span of 0.10 bar (100 kPa equivalent) — far above the
        # 0.003 bar stuck threshold.
        "tpms_press_fl_min": 2.00,
        "tpms_press_fl_max": 2.10,
        "tpms_press_fr_min": 2.00,
        "tpms_press_fr_max": 2.10,
        "tpms_press_rl_min": 2.00,
        "tpms_press_rl_max": 2.10,
        "tpms_press_rr_min": 2.00,
        "tpms_press_rr_max": 2.10,
    }
    d.update(overrides)
    return d


def test_outlap_is_usable_when_it_rolled_long_enough() -> None:
    """Out-laps are tire-model data (pit-exit reading = stint start). They are
    judged on rolling time: lap time and mean speed include the grid wait."""
    t = _laps(
        [
            # 4 min on the grid then a 70 s roll to the line: usable
            _base_lap(is_outlap=True, lap_time_s=310.0, speed_kmh_mean=25.0, moving_s=70.0),
            # rolled only 20 s (recording started just before the line): not usable
            _base_lap(lap_num=2, is_outlap=True, lap_time_s=20.0, moving_s=20.0),
            _base_lap(lap_num=3, moving_s=60.0),
        ]
    )
    out = apply_filters(t, FilterConfig())
    assert out.column("tire_usable").to_pylist() == [True, False, True]
    assert out.column("exclude_reason").to_pylist() == [None, "outlap_too_short", None]
    # The out-lap's long lap time must not count as the stint's best lap nor
    # make the flying lap look slow.
    legacy = apply_filters(t, FilterConfig(exclude_outlap=True))
    assert legacy.column("exclude_reason").to_pylist()[0] == "outlap"


def test_short_lap_excluded() -> None:
    t = _laps([_base_lap(lap_time_s=10.0)])
    out = apply_filters(t, FilterConfig())
    assert out.column("exclude_reason").to_pylist() == ["lap_too_short"]


def test_stuck_tpms_excluded() -> None:
    # All four corners stuck within tolerance
    base = _base_lap()
    for c in ("fl", "fr", "rl", "rr"):
        base[f"tpms_press_{c}_min"] = 2.00
        base[f"tpms_press_{c}_max"] = 2.001  # 0.001 bar range < 0.003 tolerance
    t = _laps([base])
    out = apply_filters(t, FilterConfig())
    assert out.column("exclude_reason").to_pylist() == ["tpms_stuck"]


def test_missing_tpms_excluded() -> None:
    base = _base_lap()
    for c in ("fl", "fr", "rl", "rr"):
        base[f"tpms_press_{c}_min"] = float("nan")
        base[f"tpms_press_{c}_max"] = float("nan")
    t = _laps([base])
    out = apply_filters(t, FilterConfig())
    assert out.column("exclude_reason").to_pylist() == ["no_tpms"]


def test_normal_lap_passes() -> None:
    t = _laps([_base_lap()])
    out = apply_filters(t, FilterConfig())
    assert out.column("tire_usable").to_pylist() == [True]
    assert out.column("exclude_reason").to_pylist() == [None]


def test_slow_relative_to_best_excluded() -> None:
    # Best lap 60s, one lap at 120s -> 2x best -> exceeds 1.4x default
    t = _laps(
        [
            _base_lap(lap_num=1, lap_time_s=60.0),
            _base_lap(lap_num=2, lap_time_s=120.0),
        ]
    )
    out = apply_filters(t, FilterConfig())
    reasons = out.column("exclude_reason").to_pylist()
    assert reasons[0] is None
    assert reasons[1] == "lap_too_slow"


def test_empty_table() -> None:
    t = pa.table(
        {
            "lap_num": pa.array([], type=pa.int16()),
            "stint_id": pa.array([], type=pa.int16()),
            "lap_time_s": pa.array([], type=pa.float32()),
            "is_outlap": pa.array([], type=pa.bool_()),
            "is_inlap": pa.array([], type=pa.bool_()),
            "speed_kmh_mean": pa.array([], type=pa.float32()),
            "on_track_s": pa.array([], type=pa.float32()),
        }
    )
    out = apply_filters(t, FilterConfig())
    assert len(out) == 0
    assert "tire_usable" in out.schema.names


def test_stale_prefix_len_detects_constant_run() -> None:
    # 4 bit-exact samples then a real change → prefix is 4.
    assert _stale_prefix_len(np.array([2.40, 2.40, 2.40, 2.40, 2.45, 2.50])) == 4


def test_stale_prefix_len_zero_when_data_starts_moving() -> None:
    # Sample 1 already differs from sample 0 → no stale prefix, return 0.
    assert _stale_prefix_len(np.array([2.40, 2.45, 2.50, 2.55])) == 0


def test_stale_prefix_len_returns_full_size_when_all_constant() -> None:
    arr = np.array([2.40, 2.40, 2.40, 2.40])
    assert _stale_prefix_len(arr) == arr.size


def test_mask_stale_session_prefix_masks_each_tpms_channel_independently() -> None:
    # FL sleeps for 3 samples then wakes; FR is alive from the start.
    ts = pa.table(
        {
            "t_lap_s": pa.array([0.0, 0.1, 0.2, 0.3, 0.4], type=pa.float32()),
            "tpms_press_fl_bar": pa.array([2.40, 2.40, 2.40, 2.45, 2.45], type=pa.float32()),
            "tpms_press_fr_bar": pa.array([2.30, 2.31, 2.32, 2.33, 2.34], type=pa.float32()),
            "tpms_temp_fl_c": pa.array([30.0, 30.0, 30.0, 32.0, 32.0], type=pa.float32()),
        }
    )
    out = mask_stale_session_prefix(ts)
    fl = out.column("tpms_press_fl_bar").to_pylist()
    assert all(v is None or v != v for v in fl[:3])  # NaN
    assert fl[3] == pytest.approx(2.45)
    fr = out.column("tpms_press_fr_bar").to_pylist()
    # FR was alive from sample 0 — no mask applied anywhere.
    assert fr[0] == pytest.approx(2.30)
    # FL temp also masked at the same 3-sample prefix length (per-channel).
    temp = out.column("tpms_temp_fl_c").to_pylist()
    assert all(v is None or v != v for v in temp[:3])
    assert temp[3] == pytest.approx(32.0)


def test_stale_mask_is_per_corner_first_change_on_either_stream() -> None:
    """One module reports a corner's pressure and temperature. As soon as
    either stream changes the module is awake, so the other stream's
    constant reading from that sample on is real and must not be masked."""
    import numpy as np
    import pyarrow as pa
    from motorsports_data_notebook.tire_model.cleaning import mask_stale_session_prefix

    n = 12
    # FL: pressure changes at sample 3, temperature only at sample 8.
    press_fl = np.array([2.0] * 3 + [2.03] * 9)
    temp_fl = np.array([20.0] * 8 + [21.0] * 4)
    # RR: both change at sample 5.
    press_rr = np.array([2.0] * 5 + [2.03] * 7)
    temp_rr = np.array([20.0] * 5 + [21.0] * 7)
    ts = pa.table(
        {
            "tpms_press_fl_bar": press_fl,
            "tpms_temp_fl_c": temp_fl,
            "tpms_press_rr_bar": press_rr,
            "tpms_temp_rr_c": temp_rr,
        }
    )
    out = mask_stale_session_prefix(ts)
    fl_t = out.column("tpms_temp_fl_c").to_numpy(zero_copy_only=False)
    fl_p = out.column("tpms_press_fl_bar").to_numpy(zero_copy_only=False)
    # FL temperature is valid from sample 3 (pressure proved the module awake),
    # still reading its constant 20 °C until it ticks up at sample 8.
    assert np.isnan(fl_t[:3]).all() and np.isnan(fl_p[:3]).all()
    assert fl_t[3] == 20.0 and fl_p[3] == 2.03
    rr_t = out.column("tpms_temp_rr_c").to_numpy(zero_copy_only=False)
    assert np.isnan(rr_t[:5]).all() and rr_t[5] == 21.0


def test_clean_laps_rebuilds_tpms_summary_from_masked_timeseries(tmp_path) -> None:
    """A raw dataset whose out-lap starts with a stale hot reading (the
    module asleep from an earlier run): the committed lap row carries the
    stale start, clean_laps replaces it with the first live reading and
    flags the laps."""
    root = tmp_path
    (root / "laps").mkdir()
    (root / "timeseries" / "2026-10").mkdir(parents=True)
    sid = "abc123"
    n0, n1 = 8, 6
    press = [2.19, 2.19, 2.19, 1.60, 1.60, 1.60, 1.63, 1.63] + [1.66] * n1
    temp = [36.0, 36.0, 36.0, 27.0, 27.0, 27.0, 27.0, 28.0] + [31.0] * n1
    ts = pa.table(
        {
            "session_id": [sid] * (n0 + n1),
            "lap_num": pa.array([0] * n0 + [1] * n1, type=pa.int16()),
            "t_lap_s": pa.array(
                list(np.arange(n0, dtype=np.float32)) + list(np.arange(n1, dtype=np.float32)),
                type=pa.float32(),
            ),
            "tpms_press_rl_bar": pa.array(press, type=pa.float32()),
            "tpms_temp_rl_c": pa.array(temp, type=pa.float32()),
        }
    )
    write_session_timeseries(root, sid, ts, partition="2026-10")
    laps = pd.DataFrame(
        [
            {
                "session_id": sid,
                "lap_num": 0,
                "stint_id": 1,
                "lap_time_s": 120.0,
                "is_outlap": True,
                "is_inlap": False,
                "on_track_s": 120.0,
                "moving_s": 100.0,
                "speed_kmh_mean": 60.0,
                "tpms_press_rl_start": 2.19,
                "tpms_temp_rl_start": 36.0,
                "tpms_press_rl_min": 1.60,
                "tpms_press_rl_max": 2.19,
            },
            {
                "session_id": sid,
                "lap_num": 1,
                "stint_id": 1,
                "lap_time_s": 80.0,
                "is_outlap": False,
                "is_inlap": False,
                "on_track_s": 80.0,
                "moving_s": 80.0,
                "speed_kmh_mean": 100.0,
                "tpms_press_rl_start": 1.66,
                "tpms_temp_rl_start": 31.0,
                "tpms_press_rl_min": 1.66,
                "tpms_press_rl_max": 1.66,
            },
        ]
    )
    pq.write_table(
        pa.Table.from_pandas(laps, preserve_index=False), root / "laps" / "2026-10.parquet"
    )

    out = clean_laps(root, laps)
    lap0 = out[out["lap_num"] == 0].iloc[0]
    assert lap0["tpms_press_rl_start"] == pytest.approx(1.60)
    assert lap0["tpms_temp_rl_start"] == pytest.approx(27.0)
    assert lap0["tpms_press_rl_first_valid_s"] == pytest.approx(3.0)
    assert lap0["tpms_press_rl_max"] == pytest.approx(1.63)
    assert bool(lap0["tire_usable"])
    lap1 = out[out["lap_num"] == 1].iloc[0]
    # Flat pressure over the lap: the filters flag it as stuck.
    assert lap1["exclude_reason"] == "tpms_stuck"
    assert not bool(lap1["tire_usable"])


def test_clean_laps_passes_through_synthetic_frames_with_usability() -> None:
    laps = pd.DataFrame({"session_id": ["s"], "lap_num": [1], "tire_usable": [True]})
    out = clean_laps(Path("/nonexistent"), laps)
    assert out.equals(laps)


def test_apply_filters_best_lap_is_per_session_and_stint() -> None:
    """Two sessions whose stints both carry id 1: a 100 s lap is slow next to
    an 80 s best only within its own session."""
    rows = []
    for sid, times in (("a", [80.0, 100.0]), ("b", [100.0, 101.0])):
        for i, t in enumerate(times):
            rows.append(
                {
                    "session_id": sid,
                    "lap_num": i + 1,
                    "stint_id": 1,
                    "lap_time_s": t,
                    "is_outlap": False,
                    "is_inlap": False,
                    "on_track_s": t,
                    "speed_kmh_mean": 90.0,
                    "tpms_press_fl_min": 1.6,
                    "tpms_press_fl_max": 1.8,
                }
            )
    out = apply_filters(pd.DataFrame(rows), FilterConfig(max_lap_time_vs_best_pct=1.2))
    assert isinstance(out, pd.DataFrame)
    assert out["exclude_reason"].tolist() == [None, "lap_too_slow", None, None]


# ------------------------------------------- stale packet at a stint's start (#371)

TPMS_DT_S = 0.8  # the modules' 1.25 Hz clock


def _tpms_table(segments: list[tuple[int, int, float, list[float], list[float]]]) -> pa.Table:
    """One corner (RR) on the TPMS clock. ``segments`` are ``(stint_id,
    lap_num, t_session_s of the first sample, pressures, temperatures)``."""
    cols: dict[str, list] = {"stint_id": [], "lap_num": [], "t_session_s": [], "p": [], "t": []}
    for stint, lap, t0, press, temp in segments:
        assert len(press) == len(temp)
        cols["stint_id"] += [stint] * len(press)
        cols["lap_num"] += [lap] * len(press)
        cols["t_session_s"] += list(t0 + TPMS_DT_S * np.arange(len(press)))
        cols["p"] += press
        cols["t"] += temp
    return pa.table(
        {
            "stint_id": pa.array(cols["stint_id"], type=pa.int16()),
            "lap_num": pa.array(cols["lap_num"], type=pa.int16()),
            "t_session_s": pa.array(cols["t_session_s"], type=pa.float64()),
            "tpms_press_rr_bar": pa.array(cols["p"], type=pa.float32()),
            "tpms_temp_rr_c": pa.array(cols["t"], type=pa.float32()),
        }
    )


def _col(ts: pa.Table, name: str) -> np.ndarray:
    return ts.column(name).to_numpy(zero_copy_only=False).astype(np.float64)


def _dither(base: float, n: int) -> list[float]:
    """A live reading: one count of dither around ``base`` (0.03 bar as logged)."""
    return [round(base + 0.03 * (i % 2), 2) for i in range(n)]


def test_three_counts_is_between_the_quantised_step_sizes() -> None:
    """The pressure is logged to 0.01 bar, so a count shows as 0.03 or 0.04 bar:
    two counts read 0.06–0.07 and three 0.09–0.10. Rule A's threshold (2.5
    counts) must fall between them."""
    threshold = 2.5 * TPMS_PRESSURE_STEP_BAR
    assert 0.07 < threshold < 0.09


def test_stint_start_after_a_mid_session_stop_is_masked() -> None:
    """Stint 2 starts on the packet the sensor sent before the car stopped (hot,
    held and re-sent by the logger); the first fresh reading is four counts
    lower, 8 s after roll-out. The session-start mask cannot see it."""
    ts = _tpms_table(
        [
            (1, 0, 0.0, _dither(1.30, 100), [20.0] * 100),
            (1, 1, 100.0, _dither(1.60, 100), [45.0] * 100),
            # after a 3 min stop: 10 samples of the held packet, then live
            (2, 2, 400.0, [1.72] * 10 + _dither(1.60, 90), [46.0] * 10 + [41.0] * 90),
            (2, 3, 500.0, _dither(1.66, 100), [47.0] * 100),
        ]
    )
    session_only = mask_stale_session_prefix(ts)
    assert np.isfinite(_col(session_only, "tpms_press_rr_bar")).all()

    out = mask_stale_tpms(ts)
    assert out.schema == ts.schema
    p, temp = _col(out, "tpms_press_rr_bar"), _col(out, "tpms_temp_rr_c")
    lap = _col(out, "lap_num")
    stale = np.flatnonzero(lap == 2)[:10]
    assert np.isnan(p[stale]).all() and np.isnan(temp[stale]).all()
    # the stint's first fresh reading, with its own time
    first = np.flatnonzero((lap == 2) & np.isfinite(p))[0]
    assert p[first] == pytest.approx(1.60)
    assert _col(out, "t_session_s")[first] == pytest.approx(400.0 + 10 * TPMS_DT_S)
    # nothing else is touched: stint 1 starts live, a later lap is not a stint start
    assert np.isfinite(p[lap != 2]).all() and np.isfinite(temp[lap != 2]).all()
    assert np.isfinite(p[lap == 2][10:]).all()


def test_pressure_is_unmasked_at_its_own_first_change() -> None:
    """a1603 FR: the temperature refreshes at 22 s while the pressure holds its
    stale 1.41 bar until 44 s. The session-start mask unmasks both streams
    at the temperature's change; rule A keeps both masked until the
    pressure's own."""
    n_t, n_p = 28, 55  # first temperature change at 22.4 s, pressure's at 44.0 s
    press = [1.41] * n_p + _dither(1.22, 45)
    temp = [31.0] * n_t + [27.0] * (100 - n_t)
    ts = _tpms_table([(1, 0, 0.0, press, temp)])

    session_only = mask_stale_session_prefix(ts)
    p0 = _col(session_only, "tpms_press_rr_bar")
    assert int(np.argmax(np.isfinite(p0))) == n_t  # unmasked at the temperature's change
    assert p0[n_t] == pytest.approx(1.41)

    out = mask_stale_tpms(ts)
    p, t = _col(out, "tpms_press_rr_bar"), _col(out, "tpms_temp_rr_c")
    assert np.isnan(p[:n_p]).all() and np.isnan(t[:n_p]).all()
    assert p[n_p] == pytest.approx(1.22) and t[n_p] == pytest.approx(27.0)


def test_first_of_day_single_stale_packet_is_masked() -> None:
    """The first packet of the day is the one the module stored at the last
    stop (2.19 bar): one sample, which the session-start mask lets through
    (a change at the second sample reads as a module awake from the start)."""
    ts = _tpms_table([(1, 0, 5.0, [2.19] + _dither(1.60, 99), [36.0] + [27.0] * 99)])
    session_only = mask_stale_session_prefix(ts)
    assert _col(session_only, "tpms_press_rr_bar")[0] == pytest.approx(2.19)

    out = mask_stale_tpms(ts)
    p, t = _col(out, "tpms_press_rr_bar"), _col(out, "tpms_temp_rr_c")
    assert np.isnan(p[0]) and np.isnan(t[0])
    assert p[1] == pytest.approx(1.60) and t[1] == pytest.approx(27.0)
    assert np.isfinite(p[1:]).all()


@pytest.mark.parametrize(
    "press",
    [
        # two counts (0.07 bar logged): a hot restart can do it
        [1.72] * 10 + _dither(1.65, 90),
        # three counts, but 61.6 s after the first sample
        [1.72] * 78 + _dither(1.62, 22),
        # the first change is one count of dither; the later step is live
        [1.72] * 5 + [1.69] * 5 + _dither(1.55, 90),
    ],
    ids=["two_counts", "after_60_s", "dither_first"],
)
def test_stint_start_is_left_alone_unless_three_counts_within_60_s(press: list[float]) -> None:
    ts = _tpms_table(
        [
            (1, 0, 0.0, _dither(1.30, 100), [20.0] * 100),
            (2, 1, 400.0, press, [40.0] * len(press)),
        ]
    )
    out = mask_stale_stint_starts(ts)
    assert np.isfinite(_col(out, "tpms_press_rr_bar")).all()
    assert np.isfinite(_col(out, "tpms_temp_rr_c")).all()


def test_stint_start_mask_needs_the_stint_and_the_session_clock() -> None:
    ts = _tpms_table([(1, 0, 0.0, [1.3, 1.3], [20.0, 20.0])])
    with pytest.raises(ValueError, match="stint_id"):
        mask_stale_stint_starts(ts.drop_columns(["stint_id"]))


def _write_wet_outlap_session(root: Path, sid: str) -> None:
    """a4512 RR (Sodegaura, wet, 2026-02-07) as logged: 1.88 bar and 20 °C from
    the first sample; the temperature refreshes to 19 °C at 8.3 s, which the
    session-start mask takes as the module awake; the pressure holds 1.88
    until a three-count fall to 1.79 at 19.5 s on the wet out-lap, and
    climbs again from there."""
    n0, n1 = 140, 100
    t0 = 0.1 + TPMS_DT_S * np.arange(n0)
    press = np.full(n0, 1.82)
    press[t0 < 19.4] = 1.88
    press[(t0 > 19.4) & (t0 < 24.0)] = 1.79
    press[t0 > 84.0] = 1.85
    temp = np.where(t0 < 8.3, 20.0, 19.0)
    t_sess = np.concatenate([t0, 115.0 + TPMS_DT_S * np.arange(n1)])
    lap = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    frame = pd.DataFrame(
        {
            "lap_num": lap,
            "stint_id": 1,
            "t_session_s": t_sess,
            "t_lap_s": np.where(lap == 0, t_sess, t_sess - 115.0),
            "speed_ms": 20.0,
            "lat_g": 0.5,
            "long_g": 0.0,
            "tpms_press_rr_bar": np.r_[press, _dither(1.88, n1)],
            "tpms_temp_rr_c": np.r_[temp, np.full(n1, 21.0)],
        }
    )
    write_session_timeseries(root, sid, frame, partition="2026-02")
    laps = pd.DataFrame(
        [
            {
                "session_id": sid,
                "lap_num": ln,
                "stint_id": 1,
                "lap_time_s": 115.0 if ln == 0 else 80.0,
                "is_outlap": ln == 0,
                "is_inlap": False,
                "outlap_from_pit": ln == 0,
                "on_track_s": 115.0 if ln == 0 else 80.0,
                "moving_s": 105.0 if ln == 0 else 80.0,
                "speed_kmh_mean": 70.0,
            }
            for ln in (0, 1)
        ]
    )
    (root / "laps").mkdir()
    pq.write_table(
        pa.Table.from_pandas(laps, preserve_index=False), root / "laps" / "2026-02.parquet"
    )


def test_a_real_cooling_step_is_read_with_its_time(tmp_path) -> None:
    """a4512 RR, the committee's one misfire: a value-only repair gave the
    anchor the value 1.79 bar but kept the time of the 1.88 reading, as if
    the tyre had read 1.79 at roll-out. Rule A masks samples, so the stint's
    anchor (the out-lap's start and first-valid time, on both streams) moves
    to the reading at 20.1 s, value and time together. (Out of sample the
    1.88 bar anchor of before scores better there; the owner accepted the
    rule with that case: docs/tire_model.md §1.0a′.)"""
    sid = "2026-02-07_103020_Inferno-86_sodegaura_a4512"
    _write_wet_outlap_session(tmp_path, sid)
    # what the session-start mask alone leaves: 1.88 bar from the temperature's refresh
    raw = load_group(tmp_path, sid, "tpms").select(
        ["t_session_s", "tpms_press_rr_bar", "tpms_temp_rr_c"]
    )
    p_old = _col(mask_stale_session_prefix(raw), "tpms_press_rr_bar")
    k_old = int(np.argmax(np.isfinite(p_old)))
    assert p_old[k_old] == pytest.approx(1.88)
    assert _col(raw, "t_session_s")[k_old] == pytest.approx(8.9)

    laps = pq.read_table(tmp_path / "laps" / "2026-02.parquet").to_pandas()
    out = clean_laps(tmp_path, laps)
    out_lap = out[out["lap_num"] == 0].iloc[0]
    assert out_lap["tpms_press_rr_start"] == pytest.approx(1.79)
    assert out_lap["tpms_temp_rr_start"] == pytest.approx(19.0)
    assert out_lap["tpms_press_rr_first_valid_s"] == pytest.approx(20.1, abs=1e-4)
    assert out_lap["tpms_temp_rr_first_valid_s"] == pytest.approx(20.1, abs=1e-4)
    # the next lap is not a stint start
    assert out[out["lap_num"] == 1].iloc[0]["tpms_press_rr_start"] == pytest.approx(1.88)


def test_both_consumers_mask_the_same_samples(tmp_path) -> None:
    """The lap aggregates (``load_tpms_timeseries``) and the per-second fit
    (``statespace._session_timeseries``) read the streams through one mask.
    The synthetic session has no sessions table or GPS family, so the fit's
    loader runs in the sensor's frame: the TPMS streams it reads are the
    same in either frame (#376)."""
    from motorsports_data_notebook.tire_model import heat_frame, statespace

    sid = "2026-02-07_103020_Inferno-86_sodegaura_a4512"
    _write_wet_outlap_session(tmp_path, sid)
    statespace._session_timeseries_in.cache_clear()
    try:
        with heat_frame.using(heat_frame.SENSOR):
            series = statespace._session_timeseries(str(tmp_path), sid)
    finally:
        statespace._session_timeseries_in.cache_clear()
    assert series is not None and series.tpms is not None
    agg = load_tpms_timeseries(tmp_path, sid)
    assert agg is not None
    fit = series.tpms
    for name in ("tpms_press_rr_bar", "tpms_temp_rr_c"):
        a = _col(agg, name)
        b = fit[name].to_numpy(dtype=np.float64)
        assert a.size == b.size
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
        assert int(np.isnan(a).sum()) == 25  # 11 by the session-start mask, 14 by rule A
