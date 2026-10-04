"""Synthetic-data round-trip tests for the warmup-table fit.

Generate per-lap (t_cum_s, δT) samples from a known set of
(K[car, corner], τ_sec[car, corner], c_track[track]) parameters and verify
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
    c_track_true: dict[str, float],  # track -> c_track
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
                        # δT per corner = K · c_track · g² · (1 − exp(−t / τ)) + noise
                        for c in ("fl", "fr", "rl", "rr"):
                            K = K_true[(car, c)]
                            tau = tau_true[(car, c)]
                            c_t = c_track_true[track]
                            g2 = g2_true[(track, car)]
                            delta_t = K * c_t * g2 * (1.0 - math.exp(-t_cum / tau))
                            tpms_temp = t_air_c + delta_t + rng.normal(0.0, noise_std_c)
                            row[f"tpms_temp_{c}_end"] = tpms_temp
                            row[f"delta_t_{c}"] = tpms_temp - t_air_c
                        rows.append(row)
    return pd.DataFrame(rows)


def test_pass1_recovers_tau_and_gain_from_warm_starts() -> None:
    """Stints that start warm (previous run's heat still in the tire) must
    not bias τ short: with the stint anchor in the frame, Pass 1 recovers
    both τ and K·c_track from data generated with a +25 °C start excess."""
    K_true, tau_true, c_t, g2, t_air = 60.0, 250.0, 1.0, 0.9, 20.0
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    for sess in range(12):
        start_excess = 25.0 if sess % 2 else 0.0  # alternate warm / cold starts
        t_cum = 0.0
        for lap in range(0, 12):
            t_cum += 60.0
            decay = math.exp(-t_cum / tau_true)
            temp = t_air + K_true * c_t * g2 * (1 - decay) + start_excess * decay
            rows.append(
                {
                    "session_id": f"s{sess}",
                    "track_canonical": "track_x",
                    "car": "CarA",
                    "stint_id": 1,
                    "lap_num": lap + 1,
                    "lap_within_stint": lap,
                    "on_track_s": 60.0,
                    "t_cum_s": t_cum,
                    "heat_proxy": g2 * 60.0,
                    "condition": "dry",
                    "t_eff_c": t_air,
                    "tpms_temp_fl_end": temp + rng.normal(0.0, 0.3),
                    "delta_t_fl": temp + rng.normal(0.0, 0.3) - t_air,
                    "t_anchor_fl": 0.0,
                    "t_start_fl": t_air + start_excess,
                }
            )
    laps_for_fit = pd.DataFrame(rows)
    tau_fit, gains = wt._pass1_fit_tau_and_gains(laps_for_fit, "CarA", "fl", "dry")
    assert tau_fit.value == pytest.approx(tau_true, rel=0.05)
    assert gains["track_x"].value == pytest.approx(K_true * c_t, rel=0.05)

    # Without the anchor columns the same data fits the v0 form, which has
    # to explain the warm starts as a fast warmup: τ comes out biased short.
    naive = laps_for_fit.drop(columns=["t_anchor_fl", "t_start_fl"])
    tau_naive, _ = wt._pass1_fit_tau_and_gains(naive, "CarA", "fl", "dry")
    assert tau_naive.value < 0.8 * tau_true


def test_pass1_tau_upper_bounds_tau_inside_the_fit_and_refits_gain() -> None:
    """Rain buckets are fitted with τ ≤ τ_dry as a bound *inside* curve_fit, so
    the gain is estimated consistently with the bound (unlike a post-hoc clip,
    which leaves a gain that was fitted jointly with a longer τ)."""
    K_true, tau_true, g2, t_air = 60.0, 600.0, 0.9, 20.0
    rows: list[dict] = []
    for sess in range(8):
        t_cum = 0.0
        for lap in range(0, 6):  # short stints: never reach the plateau
            t_cum += 60.0
            temp = t_air + K_true * g2 * (1 - math.exp(-t_cum / tau_true))
            rows.append(
                {
                    "session_id": f"s{sess}",
                    "track_canonical": "track_x",
                    "car": "CarA",
                    "stint_id": 1,
                    "lap_num": lap + 1,
                    "lap_within_stint": lap,
                    "on_track_s": 60.0,
                    "t_cum_s": t_cum,
                    "heat_proxy": g2 * 60.0,
                    "condition": "damp",
                    "t_eff_c": t_air,
                    "tpms_temp_fl_end": temp,
                    "delta_t_fl": temp - t_air,
                    "t_anchor_fl": 0.0,
                    "t_start_fl": t_air,
                }
            )
    laps = pd.DataFrame(rows)
    free_tau, free_gain = wt._pass1_fit_tau_and_gains(laps, "CarA", "fl", "damp")
    bound_tau, bound_gain = wt._pass1_fit_tau_and_gains(laps, "CarA", "fl", "damp", tau_upper=300.0)
    assert free_tau.value == pytest.approx(tau_true, rel=0.05)
    assert bound_tau.value <= 300.0 + 1e-6
    # With τ forced short, the gain must come down to match the same early
    # temperatures: a clip that kept the free gain would over-predict.
    assert bound_gain["track_x"].value < free_gain["track_x"].value
    # Both fits still track the observed range (the bound trades asymptote
    # for speed; a post-hoc clip of τ with the free gain would sit ~5 °C high).
    t = 360.0
    truth = t_air + K_true * g2 * (1 - math.exp(-t / tau_true))
    for tau_fp, gain in ((free_tau, free_gain), (bound_tau, bound_gain)):
        pred = t_air + gain["track_x"].value * g2 * (1 - math.exp(-t / tau_fp.value))
        assert pred == pytest.approx(truth, abs=2.0)
    clipped = t_air + free_gain["track_x"].value * g2 * (1 - math.exp(-t / 300.0))
    assert clipped - truth > 3.0


def test_rain_tau_upper_is_dry_tau_only_for_rain_buckets() -> None:
    taus = {("CarA", "fl", "dry"): wt.FitParam(300.0, 1.0, 100)}
    assert wt._rain_tau_upper(taus, "CarA", "fl", "dry") is None
    assert wt._rain_tau_upper(taus, "CarA", "fl", "damp") == 300.0
    assert wt._rain_tau_upper(taus, "CarA", "fr", "damp") is None  # no dry fit
    taus[("CarA", "fl", "dry")] = wt.FitParam(240.0, 0.0, 0, from_prior=True)
    assert wt._rain_tau_upper(taus, "CarA", "fl", "wet") is None  # dry is a prior


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


def test_pass1_recovers_tau_sec_per_car_corner() -> None:
    """Pass 1 should recover τ_sec[car, corner] from synthetic data."""
    K_true = {
        ("CarA", "fl"): 60.0,
        ("CarA", "fr"): 65.0,
        ("CarA", "rl"): 70.0,
        ("CarA", "rr"): 72.0,
    }
    tau_true = {
        ("CarA", "fl"): 220.0,
        ("CarA", "fr"): 230.0,
        ("CarA", "rl"): 280.0,
        ("CarA", "rr"): 285.0,
    }
    c_track_true = {"track_x": 1.0, "track_y": 0.85}
    g2_true = {("track_x", "CarA"): 0.9, ("track_y", "CarA"): 0.7}

    laps = _synth_laps(
        cars=["CarA"],
        tracks=["track_x", "track_y"],
        K_true=K_true,
        tau_true=tau_true,
        c_track_true=c_track_true,
        g2_true=g2_true,
        sessions_per_bucket=10,
        laps_per_stint=15,
        noise_std_c=0.5,
        seed=42,
    )
    laps_for_fit = laps[laps["lap_within_stint"] > 0].reset_index(drop=True)
    # Attach g2_typ to each row (normally _laps_for_fit does this)
    laps_for_fit = laps_for_fit.copy()
    laps_for_fit["g2_typ"] = [
        g2_true[(t, c)] for t, c in zip(laps_for_fit["track_canonical"], laps_for_fit["car"])
    ]

    for corner in ("fl", "fr", "rl", "rr"):
        tau_fit, _ = wt._pass1_fit_tau_and_gains(laps_for_fit, "CarA", corner, "dry")
        assert tau_fit.value == pytest.approx(
            tau_true[("CarA", corner)], rel=0.10
        ), f"τ for CarA/{corner}: got {tau_fit.value:.1f}, expected {tau_true[('CarA', corner)]:.1f}"


def test_pass2_recovers_k_and_c_track_with_anchor() -> None:
    """Pass 2 alternating LS should recover K and c_track with track_x anchored at 1.0.

    Pass 1 now folds per-lap g² into the curve fit, so the bucket gains it
    feeds to Pass 2 are already ``K · c_track`` (no ⟨g²⟩ factor).
    """
    K_true = {
        ("CarA", "fl"): 60.0,
        ("CarA", "fr"): 65.0,
        ("CarA", "rl"): 70.0,
        ("CarA", "rr"): 72.0,
    }
    c_track_true = {"track_x": 1.0, "track_y": 0.85}

    bucket_gains: dict[tuple[str, str, str, str], wt.FitParam] = {}
    for (car, corner), K in K_true.items():
        for track, c_t in c_track_true.items():
            gain = K * c_t  # gain = K · c_track (no g² factor)
            bucket_gains[(car, track, corner, "dry")] = wt.FitParam(
                value=gain, stderr=gain * 0.01, n_samples=120
            )
    # g2_lookup is still passed (kept in signature) but unused by Pass 2.
    g2_lookup: dict[tuple[str, str, str], tuple[float, int]] = {}

    K_fit, c_track_fit = wt._pass2_factor_gains(
        bucket_gains=bucket_gains,
        g2_lookup=g2_lookup,
        anchor_track="track_x",
    )
    for (car, corner), K_expected in K_true.items():
        assert K_fit[(car, corner, "dry")].value == pytest.approx(K_expected, rel=0.001)
    assert c_track_fit["track_x"].value == pytest.approx(1.0, abs=1e-9)
    assert c_track_fit["track_y"].value == pytest.approx(0.85, rel=0.001)


def test_pass2_handles_single_track_bucket_gracefully() -> None:
    """If a (car, corner, cond) has data only at one track, K · c_track is
    unidentifiable on its own; the alternating-LS should still produce some
    K value rather than crashing.
    """
    bucket_gains = {
        ("CarA", "track_x", "fl", "dry"): wt.FitParam(60.0, 1.0, 100),  # K · c_track = 60 · 1.0
    }
    K_fit, c_track_fit = wt._pass2_factor_gains(
        bucket_gains=bucket_gains, g2_lookup={}, anchor_track="track_x"
    )
    assert K_fit[("CarA", "fl", "dry")].value == pytest.approx(60.0, rel=0.001)
    assert c_track_fit["track_x"].value == pytest.approx(1.0, abs=1e-9)


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


def test_pass1_returns_prior_when_no_dense_bucket() -> None:
    """If every (track) bucket for a (car, corner) has fewer than
    MIN_LAPS_FOR_TAU_FIT samples, return the physical prior with from_prior=True."""
    # Make a tiny dataset: 10 laps total per (car, corner) — under the 30-lap threshold
    laps = _synth_laps(
        cars=["CarA"],
        tracks=["track_x"],
        K_true={("CarA", c): 60.0 for c in ("fl", "fr", "rl", "rr")},
        tau_true={("CarA", c): 240.0 for c in ("fl", "fr", "rl", "rr")},
        c_track_true={"track_x": 1.0},
        g2_true={("track_x", "CarA"): 0.9},
        sessions_per_bucket=1,
        laps_per_stint=10,
        stints_per_session=1,
        noise_std_c=0.0,
        seed=0,
    )
    laps_for_fit = laps[laps["lap_within_stint"] > 0].copy()
    laps_for_fit["g2_typ"] = 0.9

    tau_fit, per_bucket = wt._pass1_fit_tau_and_gains(laps_for_fit, "CarA", "fl", "dry")
    assert tau_fit.from_prior is True
    assert tau_fit.value == pytest.approx(wt.PRIOR_TAU_SEC)
    assert per_bucket == {}


def test_w_road_default_is_zero_point_two() -> None:
    """v0 fixes w_road = 0.2; if this changes, lots of other things break."""
    assert wt.W_ROAD == pytest.approx(0.2)


def test_anchor_track_is_tsukuba_2000() -> None:
    """The c_track identifiability anchor must remain stable across runs."""
    assert wt.ANCHOR_TRACK == "tsukuba_2000"


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
