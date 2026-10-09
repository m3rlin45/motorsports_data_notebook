"""Synthetic-data round-trip tests for the warmup-table fit.

Generate per-lap (t_cum_s, δT) samples from a known set of
(K[car, corner], τ_sec[car, corner]) parameters and verify
that ``build_warmup_table`` recovers them within physically reasonable
tolerance. This guards against regressions in either pass of the fit.

These tests don't read the dataset — they exercise the fit logic in
isolation against the in-memory laps DataFrame.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model import warmup_table as wt


def _synth_laps(
    *,
    cars: list[str],
    tracks: list[str],
    K_true: dict[tuple[str, str], float],  # (car, corner) -> K
    tau_true: dict[tuple[str, str], float],  # (car, corner) -> tau_sec
    g2_true: dict[tuple[str, str], float],  # (track, car) -> g2
    lap_time_s: float = 80.0,
    laps_per_stint: int = 12,
    stints_per_session: int = 2,
    sessions_per_bucket: int = 8,
    noise_std_c: float = 1.0,
    t_air_c: float = 20.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a fake laps DataFrame with known ground-truth parameters."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    session_counter = 0
    for car in cars:
        for track in tracks:
            if (track, car) not in g2_true:
                continue
            for _s in range(sessions_per_bucket):
                session_counter += 1
                sid = f"sess_{session_counter:04d}"
                for stint_id in range(1, stints_per_session + 1):
                    t_cum = 0.0
                    for lap_within_stint in range(0, laps_per_stint):
                        t_cum += lap_time_s
                        row: dict = {
                            "session_id": sid,
                            "track_canonical": track,
                            "car": car,
                            "stint_id": stint_id,
                            "lap_num": (stint_id - 1) * laps_per_stint + lap_within_stint + 1,
                            "lap_within_stint": lap_within_stint,
                            "on_track_s": lap_time_s,
                            "t_cum_s": t_cum,
                            "heat_proxy": g2_true[(track, car)] * lap_time_s,
                            "tire_usable": True,
                            "t_air_c": t_air_c,
                            "cloud_cover": 50.0,
                            "precipitation": 0.0,  # synthetic data is all-dry
                            "condition": "dry",
                            "t_road_c": t_air_c,
                            "t_eff_c": t_air_c,
                            "session_start_utc": "2026-01-01T00:00:00Z",
                            "date": "2026-01-01",
                        }
                        # δT per corner = K · g² · (1 − exp(−t / τ)) + noise
                        for c in ("fl", "fr", "rl", "rr"):
                            K = K_true[(car, c)]
                            tau = tau_true[(car, c)]
                            g2 = g2_true[(track, car)]
                            delta_t = K * g2 * (1.0 - math.exp(-t_cum / tau))
                            tpms_temp = t_air_c + delta_t + rng.normal(0.0, noise_std_c)
                            row[f"tpms_temp_{c}_end"] = tpms_temp
                            row[f"delta_t_{c}"] = tpms_temp - t_air_c
                        rows.append(row)
    return pd.DataFrame(rows)


def test_compute_stint_anchor_prefers_first_lap_start_then_falls_back() -> None:
    laps = pd.DataFrame(
        [
            # stint 1: first lap start reading is stale (NaN) -> anchor on its end
            {
                "session_id": "s",
                "stint_id": 1,
                "lap_num": 1,
                "on_track_s": 60.0,
                "t_cum_s": 60.0,
                "tpms_temp_fl_start": np.nan,
                "tpms_temp_fl_end": 31.0,
            },
            {
                "session_id": "s",
                "stint_id": 1,
                "lap_num": 2,
                "on_track_s": 60.0,
                "t_cum_s": 120.0,
                "tpms_temp_fl_start": 31.5,
                "tpms_temp_fl_end": 38.0,
            },
            # stint 2: start reading present -> anchor at t = 0
            {
                "session_id": "s",
                "stint_id": 2,
                "lap_num": 3,
                "on_track_s": 60.0,
                "t_cum_s": 60.0,
                "tpms_temp_fl_start": 44.0,
                "tpms_temp_fl_end": 50.0,
            },
        ]
    )
    for c in ("fr", "rl", "rr"):
        laps[f"tpms_temp_{c}_start"] = np.nan
        laps[f"tpms_temp_{c}_end"] = np.nan
    out = wt._compute_stint_anchor(laps)
    s1 = out[out.stint_id == 1]
    assert (s1["t_anchor_fl"] == 60.0).all()
    assert (s1["t_start_fl"] == 31.0).all()
    s2 = out[out.stint_id == 2]
    assert (s2["t_anchor_fl"] == 0.0).all()
    assert (s2["t_start_fl"] == 44.0).all()
    assert out["t_anchor_rr"].isna().all()


def test_classify_condition_thresholds() -> None:
    """Three-level classification from precipitation in mm/hr."""
    assert wt.classify_condition(0.0) == "dry"
    assert wt.classify_condition(0.05) == "dry"
    assert wt.classify_condition(0.1) == "damp"
    assert wt.classify_condition(0.5) == "damp"
    assert wt.classify_condition(0.99) == "damp"
    assert wt.classify_condition(1.0) == "wet"
    assert wt.classify_condition(4.4) == "wet"
    assert wt.classify_condition(None) == "unknown"
    assert wt.classify_condition(float("nan")) == "unknown"


def test_w_road_default_is_zero_point_two() -> None:
    """v0 fixes w_road = 0.2; if this changes, lots of other things break."""
    assert wt.W_ROAD == pytest.approx(0.2)


def test_build_corner_defaults_medians_and_steady_state_filter() -> None:
    """Prefills use only steady-state laps and take per-corner medians."""
    rows = []
    for lap_within_stint, temp, press in [
        (1, 40.0, 1.5),  # warmup lap — must be excluded
        (4, 70.0, 1.9),
        (5, 71.0, 1.95),
        (6, 72.0, 2.0),
        (7, 73.0, 2.05),
        (8, 74.0, 2.1),
    ]:
        row: dict = {
            "car": "KK-SII",
            "condition": "dry",
            "lap_within_stint": lap_within_stint,
        }
        for c in ("fl", "fr", "rl", "rr"):
            row[f"tpms_temp_{c}_end"] = temp
            row[f"tpms_press_{c}_mean"] = press
        rows.append(row)
    # An unknown-condition steady lap must be excluded too.
    unknown = dict(rows[-1], condition="unknown")
    laps = pd.DataFrame(rows + [unknown])

    out = wt._build_corner_defaults(laps)

    assert set(out) == {("KK-SII", c, "dry") for c in ("fl", "fr", "rl", "rr")}
    temp, press, n = out[("KK-SII", "fl", "dry")]
    assert temp == pytest.approx(72.0)
    assert press == pytest.approx(2.0)
    assert n == 5


def test_build_corner_defaults_drops_thin_buckets() -> None:
    """Fewer than min_laps steady laps -> no prefill row for that bucket."""
    row: dict = {"car": "Inferno 86", "condition": "wet", "lap_within_stint": 5}
    for c in ("fl", "fr", "rl", "rr"):
        row[f"tpms_temp_{c}_end"] = 23.0
        row[f"tpms_press_{c}_mean"] = 2.5
    laps = pd.DataFrame([row, dict(row)])  # only 2 steady wet laps

    assert wt._build_corner_defaults(laps) == {}


def test_build_corner_defaults_skips_nan_masked_corners() -> None:
    """A blacklist-masked (NaN) corner drops out; the others still fit."""
    row: dict = {"car": "Inferno 86", "condition": "dry", "lap_within_stint": 5}
    for c in ("fl", "fr", "rl", "rr"):
        row[f"tpms_temp_{c}_end"] = 80.0
        row[f"tpms_press_{c}_mean"] = 1.8
    row["tpms_temp_fr_end"] = float("nan")
    laps = pd.DataFrame([row])

    out = wt._build_corner_defaults(laps, min_laps=1)

    assert ("Inferno 86", "fr", "dry") not in out
    assert out[("Inferno 86", "rl", "dry")][0] == pytest.approx(80.0)


def test_data_through_is_newest_fitted_session_in_track_local_time() -> None:
    import datetime as _dt

    import pandas as pd

    from motorsports_data_notebook.tire_model.warmup_table import _data_through_for_fit

    laps = pd.DataFrame(
        {
            "date": [_dt.date(2026, 8, 30), _dt.date(2026, 10, 2), _dt.date(2026, 10, 2), None],
            "session_start_utc": pd.to_datetime(
                ["2026-08-30T00:00:00Z", "2026-10-02T01:10:00Z", "2026-10-02T05:23:00Z", None],
                utc=True,
            ),
            "track_canonical": ["tsukuba_2000", "suzuka", "suzuka", None],
        }
    )
    date, local = _data_through_for_fit(laps)
    assert date == "2026-10-02"
    assert local == "2026-10-02 14:23 JST"
    assert _data_through_for_fit(pd.DataFrame({"date": []})) == (None, None)
    assert _data_through_for_fit(pd.DataFrame({"x": [1]})) == (None, None)


def test_compute_stint_anchor_uses_pit_exit_reading_on_the_rolling_clock() -> None:
    """Schema v3: the stint starts with an out-lap from the pits. Its first
    valid reading is the anchor, placed on the rolling clock (lap-relative
    time minus the standstill before the car moved). Stints without a
    from-pit out-lap keep the first-lap anchor."""
    base = {"session_id": "s", "car": "FJ"}
    laps = pd.DataFrame(
        [
            # stint 1: 200 s out-lap, 120 s of which standing on the grid; the
            # TPMS woke 150 s into the lap -> 30 s into the rolling clock.
            {
                **base,
                "stint_id": 1,
                "lap_num": 0,
                "is_outlap": True,
                "outlap_from_pit": True,
                "on_track_s": 200.0,
                "moving_s": 80.0,
                "t_cum_s": 80.0,
                "tpms_temp_fl_start": 26.0,
                "tpms_temp_fl_end": 31.0,
                "tpms_press_fl_start": 1.30,
                "tpms_temp_fl_first_valid_s": 150.0,
            },
            {
                **base,
                "stint_id": 1,
                "lap_num": 1,
                "is_outlap": False,
                "outlap_from_pit": False,
                "on_track_s": 60.0,
                "moving_s": 60.0,
                "t_cum_s": 140.0,
                "tpms_temp_fl_start": 31.5,
                "tpms_temp_fl_end": 38.0,
                "tpms_press_fl_start": 1.35,
                "tpms_temp_fl_first_valid_s": 0.0,
            },
            # stint 2: out-lap that started mid-track (file boundary) -> not a pit exit
            {
                **base,
                "stint_id": 2,
                "lap_num": 2,
                "is_outlap": True,
                "outlap_from_pit": False,
                "on_track_s": 70.0,
                "moving_s": 70.0,
                "t_cum_s": 70.0,
                "tpms_temp_fl_start": 44.0,
                "tpms_temp_fl_end": 50.0,
                "tpms_press_fl_start": 1.5,
                "tpms_temp_fl_first_valid_s": 0.0,
            },
        ]
    )
    for c in ("fr", "rl", "rr"):
        laps[f"tpms_temp_{c}_start"] = np.nan
        laps[f"tpms_temp_{c}_end"] = np.nan
        laps[f"tpms_press_{c}_start"] = np.nan
        laps[f"tpms_temp_{c}_first_valid_s"] = np.nan
    out = wt._compute_stint_anchor(laps)
    s1 = out[out.stint_id == 1]
    assert (s1["anchor_kind_fl"] == "pit_exit").all()
    assert (s1["t_anchor_fl"] == 30.0).all()
    assert (s1["t_start_fl"] == 26.0).all()
    assert (s1["p_start_fl"] == 1.30).all()
    s2 = out[out.stint_id == 2]
    assert (s2["anchor_kind_fl"] == "first_lap").all()
    assert (s2["t_anchor_fl"] == 0.0).all() and (s2["t_start_fl"] == 44.0).all()


def test_rolling_clock_and_flying_lookups_exclude_outlap_wait() -> None:
    laps = pd.DataFrame(
        [
            {
                "session_id": "s",
                "stint_id": 1,
                "lap_num": 0,
                "is_outlap": True,
                "outlap_from_pit": True,
                "on_track_s": 300.0,
                "moving_s": 90.0,
                "heat_proxy": 0.3 * 90,
                "track_canonical": "t",
                "car": "FJ",
                "condition": "dry",
            },
            {
                "session_id": "s",
                "stint_id": 1,
                "lap_num": 1,
                "is_outlap": False,
                "outlap_from_pit": False,
                "on_track_s": 60.0,
                "moving_s": 60.0,
                "heat_proxy": 0.9 * 60,
                "track_canonical": "t",
                "car": "FJ",
                "condition": "dry",
            },
            {
                "session_id": "s",
                "stint_id": 1,
                "lap_num": 2,
                "is_outlap": False,
                "outlap_from_pit": False,
                "on_track_s": 62.0,
                "moving_s": 62.0,
                "heat_proxy": 1.0 * 62,
                "track_canonical": "t",
                "car": "FJ",
                "condition": "dry",
            },
        ]
    )
    clocked = wt._compute_stint_clock(laps)
    assert clocked["t_cum_s"].tolist() == [90.0, 150.0, 212.0]
    assert clocked["lap_within_stint"].tolist() == [0, 1, 2]
    lt = wt._build_lap_time_typ(wt._flying_laps(laps))
    assert lt[("t", "FJ", "dry")][0] == pytest.approx(61.0)
    g2 = wt._build_g2_typ(wt._flying_laps(laps), percentile=50)
    assert g2[("t", "FJ", "dry")][0] == pytest.approx(0.95)
    out = wt._build_outlap_typ(laps)
    assert out[("t", "FJ", "dry")] == (90.0, pytest.approx(0.3), 1)


def test_delta_t_targets_the_pressure_implied_gas_temperature() -> None:
    """The model's observable is the gas temperature implied by pressure
    from the pit-exit (T, P); the TPMS end temperature is only a fallback
    when no anchor pressure exists."""
    laps = pd.DataFrame(
        {
            "t_air_c": [20.0, 20.0],
            "cloud_cover": [100.0, 100.0],
            "t_start_fl": [25.0, 25.0],
            "p_start_fl": [1.5, np.nan],
            "tpms_press_fl_end": [1.75, 1.75],
            "tpms_temp_fl_end": [40.0, 40.0],
        }
    )
    for c in ("fr", "rl", "rr"):
        laps[f"t_start_{c}"] = np.nan
        laps[f"p_start_{c}"] = np.nan
        laps[f"tpms_press_{c}_end"] = np.nan
        laps[f"tpms_temp_{c}_end"] = np.nan
    out = wt._compute_delta_t(laps)
    t_gas = (25.0 + 273.15) * 2.75 / 2.5 - 273.15  # pressure rose 10%
    assert out["t_gas_fl_end"].iloc[0] == pytest.approx(t_gas)
    assert out["delta_t_fl"].iloc[0] == pytest.approx(t_gas - out["t_eff_c"].iloc[0])
    assert np.isnan(out["t_gas_fl_end"].iloc[1])  # no anchor pressure -> no target
    g = wt.gas_temperature_c(np.array([25.0]), np.array([1.5]), np.array([1.5]))
    assert g[0] == pytest.approx(25.0)  # unchanged pressure -> anchor temperature


def test_per_corner_q_lookup_and_outlap_corners() -> None:
    """Schema v5 lookups: per-corner percentile of ``q_lap_{corner}`` for the
    flying laps and per-corner median for the pit out-laps."""
    rows = []
    for i in range(10):
        rows.append(
            {
                "session_id": "s1",
                "stint_id": 1,
                "lap_num": i,
                "track_canonical": "t",
                "car": "c",
                "condition": "dry",
                "is_outlap": i == 0,
                "outlap_from_pit": i == 0,
                "moving_s": 60.0,
                "heat_proxy": 60.0 * 0.8,
                "on_track_s": 60.0,
                "q_lap_fl": 1.0 + 0.01 * i if i else 0.4,
                "q_lap_fr": 0.5 + 0.01 * i if i else 0.2,
                "q_lap_rl": 0.9 if i else 0.3,
                "q_lap_rr": 0.4 if i else 0.1,
            }
        )
    laps = pd.DataFrame(rows)
    q = wt._build_q_typ_per_corner(wt._flying_laps(laps), percentile=50)
    per, n = q[("t", "c", "dry")]
    assert n == 9 and per["fl"] == pytest.approx(1.05) and per["rr"] == pytest.approx(0.4)
    base, corners = wt._build_outlap_typ_with_corners(laps)
    assert corners[("t", "c", "dry")] == {"fl": 0.4, "fr": 0.2, "rl": 0.3, "rr": 0.1}
    mv, g2_mean, n_out = base[("t", "c", "dry")]
    assert mv == 60.0 and g2_mean == pytest.approx(0.25) and n_out == 1
