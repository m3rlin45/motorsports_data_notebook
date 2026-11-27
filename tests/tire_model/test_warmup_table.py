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


def test_load_hot_pressure_targets_reads_car_compound_table(tmp_path) -> None:
    (tmp_path / "hot_pressure_targets.yaml").write_text(
        "targets:\n  FJ:\n    DRY: 1.5\n    WET: 1.7\n  Inferno 86:\n    A052: 1.8\n    A050: null\n"
    )
    assert wt.load_hot_pressure_targets(tmp_path) == {
        ("FJ", "DRY"): 1.5,
        ("FJ", "WET"): 1.7,
        ("Inferno 86", "A052"): 1.8,
    }
    assert wt.load_hot_pressure_targets(tmp_path / "missing") == {}
    (tmp_path / "hot_pressure_targets.yaml").write_text("targets:\n  FJ:\n    DRY: 15\n")
    with pytest.raises(ValueError):
        wt.load_hot_pressure_targets(tmp_path)


def test_committed_hot_pressure_targets_match_the_team_numbers() -> None:
    from motorsports_data_notebook.tire_etl.paths import default_dataset_root

    assert wt.load_hot_pressure_targets(default_dataset_root()) == {
        ("FJ", "DRY"): 1.5,
        ("FJ", "WET"): 1.7,
        ("Inferno 86", "A052"): 1.8,
        ("Inferno 86", "RE-71RS"): 2.0,
        ("Inferno 86", "A050"): 1.5,
    }


def test_first_lap_anchor_is_stamped_at_the_readings_own_time() -> None:
    """#371: when the masked TPMS streams leave a stint's out-lap without a
    reading, the anchor is the next lap's first live reading, which comes
    mid-lap. It is placed at its own first-valid time on the rolling clock
    (from the lap's start, less any standstill), not at the lap's start."""
    base = {"session_id": "s", "car": "FJ", "stint_id": 1}
    laps = pd.DataFrame(
        [
            {
                **base,
                "lap_num": 0,
                "is_outlap": True,
                "outlap_from_pit": True,
                "on_track_s": 150.0,
                "moving_s": 100.0,
                "t_cum_s": 100.0,
                "tpms_temp_fl_start": np.nan,
                "tpms_temp_fl_end": np.nan,
                "tpms_press_fl_start": np.nan,
                "tpms_temp_fl_first_valid_s": np.nan,
            },
            {
                **base,
                "lap_num": 1,
                "is_outlap": False,
                "outlap_from_pit": False,
                "on_track_s": 60.0,
                "moving_s": 60.0,
                "t_cum_s": 160.0,
                "tpms_temp_fl_start": 33.0,
                "tpms_temp_fl_end": 38.0,
                "tpms_press_fl_start": 1.42,
                "tpms_temp_fl_first_valid_s": 40.0,
            },
        ]
    )
    for c in ("fr", "rl", "rr"):
        laps[f"tpms_temp_{c}_start"] = np.nan
        laps[f"tpms_temp_{c}_end"] = np.nan
        laps[f"tpms_press_{c}_start"] = np.nan
        laps[f"tpms_temp_{c}_first_valid_s"] = np.nan
    out = wt._compute_stint_anchor(laps)
    assert (out["anchor_kind_fl"] == "first_lap").all()
    assert (out["t_anchor_fl"] == 140.0).all()
    assert (out["t_start_fl"] == 33.0).all() and (out["p_start_fl"] == 1.42).all()


def _two_raw_car_laps() -> pd.DataFrame:
    """Flying laps and a pit out-lap of two raw cars pooled as one ("P") at
    track "t", plus a car that is not pooled ("solo"); the raw cars drive
    with different heats and lap times."""
    rows = []
    for raw, q, lap_s, sessions in (
        ("A", 1.0, 60.0, 1),
        ("B", 2.0, 70.0, 3),
        ("solo", 3.0, 80.0, 1),
    ):
        for s in range(sessions):
            for i in range(6):
                rows.append(
                    {
                        "session_id": f"{raw}{s}",
                        "stint_id": 1,
                        "lap_num": i,
                        "track_canonical": "t",
                        "car": raw,
                        "condition": "dry",
                        "is_outlap": i == 0,
                        "outlap_from_pit": i == 0,
                        "moving_s": lap_s,
                        "heat_proxy": lap_s * q,
                        "on_track_s": lap_s,
                        **{f"q_lap_{c}": q if i else q / 4 for c in wt.CORNERS},
                    }
                )
    laps = pd.DataFrame(rows)
    return laps.assign(raw_car=laps["car"], car=laps["car"].replace({"A": "P", "B": "P"}))


def test_raw_car_entries_are_each_raw_cars_own_and_leave_the_pooled_tables(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    # the builders, with the guard off (A has one session and neither car a curve)
    monkeypatch.setattr(wt, "RAW_CAR_MIN_SESSIONS", 1)
    monkeypatch.setattr(wt, "RAW_CAR_NEEDS_CURVE", False)
    laps = _two_raw_car_laps()
    raw = wt._build_raw_car_entries(tmp_path, wt._flying_laps(laps), laps, {})
    # one entry per raw car pooled under another name; the car that is not
    # pooled has only its pooled entry
    assert sorted(raw.g2) == [("t", "A", "dry"), ("t", "B", "dry")]
    assert raw.g2[("t", "A", "dry")] == (pytest.approx(1.0), 5)
    assert raw.g2[("t", "B", "dry")] == (pytest.approx(2.0), 15)
    assert raw.lap_time[("t", "A", "dry")] == (60.0, 5)
    assert raw.outlap[("t", "B", "dry")][0] == 70.0
    assert raw.outlap_corner[("t", "A", "dry")]["fl"] == pytest.approx(0.25)
    # the pooled entry is the mix the raw entries take apart (B's laps outnumber A's)
    pooled = wt._build_q_typ_per_corner(wt._flying_laps(laps))
    assert pooled[("t", "P", "dry")][0]["fl"] == pytest.approx(2.0)

    def assemble(raw_entries):  # type: ignore[no-untyped-def]
        return wt._assemble_model(
            tau_by_car_corner_cond={},
            k_by_car_corner_cond={},
            g2_lookup={("t", "P", "dry"): (2.0, 20)},
            lap_time_lookup=wt._build_lap_time_typ(wt._flying_laps(laps)),
            bucket_n_samples={},
            q_corner_lookup=pooled,
            raw_car_entries=raw_entries,
        )

    with_raw, without = assemble(raw), assemble(None)
    assert [r["car"] for r in with_raw["g2_typ_by_track_raw_car_cond"]] == ["A", "B"]
    assert [r["car"] for r in with_raw["lap_time_typ_by_track_raw_car_cond"]] == ["A", "B"]
    assert with_raw["g2_typ_by_track_raw_car_cond"][0]["q_typ_by_corner"]["rr"] == 1.0
    assert "g2_typ_by_track_raw_car_cond" not in without
    # the pooled tables are the same rows with and without the raw cars'
    for name in (
        "g2_typ_by_track_car_cond",
        "lap_time_typ_by_track_car_cond",
        "outlap_typ_by_track_car_cond",
    ):
        assert with_raw[name] == without[name]


def test_raw_car_pace_curves_go_through_the_pooled_cars_shares(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A raw car's curve is its own laps simulated through the shares of
    the car it is pooled under (the constants stay pooled)."""
    from motorsports_data_notebook.tire_model import pace_sim

    seen = {}

    def fake(root, laps, share_by_car):  # type: ignore[no-untyped-def]
        seen["cars"] = sorted(set(laps["car"]))
        seen["shares"] = share_by_car
        return {(t, c, k): {"lap_time_s": [1.0], "g2": [1.0]} for t, c, k in [("t", "B", "dry")]}

    monkeypatch.setattr(pace_sim, "build_pace_model", fake)
    laps = _two_raw_car_laps()
    raw = wt._build_raw_car_entries(tmp_path, wt._flying_laps(laps), laps, {"P": "p-share"})
    assert seen == {"cars": ["A", "B"], "shares": {"A": "p-share", "B": "p-share"}}
    assert list(raw.curves) == [("t", "B", "dry")]


def test_apply_car_aliases_keeps_the_raw_car() -> None:
    laps = pd.DataFrame({"car": ["KK-F", "KK-SII", "Inferno 86"]})
    got = wt.apply_car_aliases(laps)
    assert list(got["car"]) == ["FJ", "FJ", "Inferno 86"]
    assert list(got["raw_car"]) == ["KK-F", "KK-SII", "Inferno 86"]


def test_raw_car_entries_need_two_sessions_and_a_pace_curve(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The guard (rule (b)): a raw car has its own entry only with at least
    two training sessions behind it and a pace curve; without them it has
    none (heat, lap time, out-lap or curve), so it reads the pooled one."""
    from motorsports_data_notebook.tire_model import pace_sim

    assert wt.RAW_CAR_MIN_SESSIONS == 2 and wt.RAW_CAR_NEEDS_CURVE
    laps = _two_raw_car_laps()  # A: 1 session, B: 3
    curves: dict = {}
    monkeypatch.setattr(pace_sim, "build_pace_model", lambda root, laps, shares: dict(curves))

    def build():  # type: ignore[no-untyped-def]
        return wt._build_raw_car_entries(tmp_path, wt._flying_laps(laps), laps, {"P": "s"})

    # both with a curve: A rests on one session, so only B has its own entry
    curves.update(
        {key: {"lap_time_s": [1.0], "g2": [1.0]} for key in [("t", "A", "dry"), ("t", "B", "dry")]}
    )
    both = build()
    assert sorted(both.g2) == sorted(both.lap_time) == sorted(both.outlap) == [("t", "B", "dry")]
    assert (
        sorted(both.curves)
        == sorted(both.q_corner)
        == sorted(both.outlap_corner)
        == [("t", "B", "dry")]
    )
    # B without a curve: neither car has its own entry
    del curves[("t", "B", "dry")]
    none = build()
    assert none.g2 == none.lap_time == none.outlap == none.curves == none.q_corner == {}
    # the guard off: every raw car with a heat has its own entry
    monkeypatch.setattr(wt, "RAW_CAR_MIN_SESSIONS", 1)
    monkeypatch.setattr(wt, "RAW_CAR_NEEDS_CURVE", False)
    every = build()
    assert sorted(every.g2) == [("t", "A", "dry"), ("t", "B", "dry")]


def test_the_model_says_when_kappa_was_read_within_the_lap() -> None:
    """With the option off the speed-pressure block is the block it always
    was; with it on the block carries the reading per car (#417)."""

    def block(within):  # type: ignore[no-untyped-def]
        model = wt._assemble_model(
            tau_by_car_corner_cond={},
            k_by_car_corner_cond={},
            g2_lookup={},
            lap_time_lookup={},
            bucket_n_samples={},
            kappa_by_car={"FJ": 1.9e-6},
            kappa_within_lap=within,
        )
        return model["energy_balance"]["speed_pressure"]

    plain = {"form": block(None)["form"], "kappa_by_car": {"FJ": 1.9e-6}, "fitted": True}
    assert block(None) == plain and block({}) == plain
    reading = {"FJ": {"kappa": 1.9e-6, "stderr": 7e-8, "n_segments": 2710, "held": True}}
    assert block(reading) == {**plain, "source": "within_lap", "within_lap": reading}
