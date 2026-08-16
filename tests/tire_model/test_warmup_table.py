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
