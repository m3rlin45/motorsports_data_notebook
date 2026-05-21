"""Unit tests for the predictor and its fallback chain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motorsports_data_notebook.tire_model.predict import (
    CORNERS,
    Prediction,
    predict_cold_pressure,
)


def _minimal_model() -> dict:
    """Build a synthetic in-memory model that exercises the fallback chain.

    - Cars: "ToyCar" and "OtherCar"
    - Tracks: "track_a" (anchor), "track_b"
    - K_buckets cover every (ToyCar, corner) but only RR for OtherCar
    - τ covers all 8 (ToyCar) + (OtherCar) entries
    """
    return {
        "schema_version": 1,
        "model_form": "T_hot - T_eff = K * c_track * g2_typ * (1 - exp(-t/tau_sec))",
        "gay_lussac": {"p_atm_bar": 1.0, "t_zero_c_to_k": 273.15, "t_cold_uses": "T_air"},
        "energy_balance": {
            "w_road": 0.2,
            "w_road_fitted": False,
            "t_road_proxy": {
                "formula": "T_air + delta_sun_max_c * (1 - cloud_cover/100) * sun_factor",
                "delta_sun_max_c": 10.0,
                "sun_factor_default": 1.0,
            },
        },
        "corners": list(CORNERS),
        "min_samples_per_bucket": 5,
        "priors_when_no_fit": {
            "tau_sec_seconds": 240.0,
            "K_kelvin_per_g2": 60.0,
            "c_track": 1.0,
        },
        "tau_sec_by_car_corner": [
            {
                "car": "ToyCar",
                "corner": c,
                "value_seconds": 200.0 + 10.0 * i,
                "stderr_seconds": 5.0,
                "n_samples_used": 100,
                "from_prior": False,
            }
            for i, c in enumerate(CORNERS)
        ]
        + [
            {
                "car": "OtherCar",
                "corner": c,
                "value_seconds": 250.0,
                "stderr_seconds": 8.0,
                "n_samples_used": 50,
                "from_prior": False,
            }
            for c in CORNERS
        ],
        "K_buckets": [
            {
                "key": {"car": "ToyCar", "corner": c},
                "value_kelvin_per_g2": 50.0 + 5.0 * i,
                "stderr_kelvin_per_g2": 2.0,
                "n_samples": 100,
                "from_prior": False,
                "from_single_track": False,
            }
            for i, c in enumerate(CORNERS)
        ]
        + [
            {
                "key": {"car": "OtherCar", "corner": "rr"},
                "value_kelvin_per_g2": 45.0,
                "stderr_kelvin_per_g2": 3.0,
                "n_samples": 50,
                "from_prior": False,
                "from_single_track": True,
            },
        ],
        "c_track_by_track": [
            {
                "track_canonical": "track_a",
                "value": 1.0,
                "stderr": 0.0,
                "n_buckets_used": 8,
                "anchor": True,
            },
            {
                "track_canonical": "track_b",
                "value": 0.85,
                "stderr": 0.04,
                "n_buckets_used": 4,
                "anchor": False,
            },
        ],
        "g2_typ_by_track_car": [
            {"track_canonical": "track_a", "car": "ToyCar", "g2_typ": 0.9, "n_laps_used": 100},
            {"track_canonical": "track_b", "car": "ToyCar", "g2_typ": 0.7, "n_laps_used": 50},
            {"track_canonical": "track_a", "car": "OtherCar", "g2_typ": 0.8, "n_laps_used": 30},
        ],
        "lap_time_typ_by_track_car": [
            {
                "track_canonical": "track_a",
                "car": "ToyCar",
                "lap_time_typ_s": 60.0,
                "n_laps_used": 100,
            },
            {
                "track_canonical": "track_b",
                "car": "ToyCar",
                "lap_time_typ_s": 100.0,
                "n_laps_used": 50,
            },
            {
                "track_canonical": "track_a",
                "car": "OtherCar",
                "lap_time_typ_s": 80.0,
                "n_laps_used": 30,
            },
        ],
        "fallback_order_for_K": [["car", "corner"], ["car"], []],
    }


def test_predict_exact_bucket_match() -> None:
    model = _minimal_model()
    result = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    fl = result["fl"]
    assert isinstance(fl, Prediction)
    assert fl.K_source_bucket == ("ToyCar", "fl")
    assert not fl.K_from_prior
    assert fl.c_track == pytest.approx(1.0)  # anchor track
    assert fl.g2_typ == pytest.approx(0.9)
    assert fl.predicted_hot_temp_c > 20.0  # warming up


def test_predict_falls_back_to_car_level_when_corner_missing() -> None:
    """OtherCar has only RR in K_buckets — FL/FR/RL should fall back to (car) mean."""
    model = _minimal_model()
    result = predict_cold_pressure(
        track="track_a",
        car="OtherCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    fl = result["fl"]
    assert fl.K_source_bucket == ("OtherCar",)
    assert fl.K_kelvin_per_g2 == pytest.approx(45.0)  # only RR exists → mean is RR


def test_predict_falls_back_to_global_when_car_unknown() -> None:
    model = _minimal_model()
    result = predict_cold_pressure(
        track="track_a",
        car="UnknownCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        # No (car) data → falls back to prior K
        # Also no (car, corner) τ_sec → prior τ
        # ⟨g²⟩ fallback to (track) mean
        # lap_time_typ fallback to (track) mean
        _model=model,
    )
    fl = result["fl"]
    assert fl.K_from_prior is True
    assert fl.K_source_bucket == ()  # global fallback


def test_predict_falls_back_to_c_track_one_when_track_unknown() -> None:
    model = _minimal_model()
    result = predict_cold_pressure(
        track="unknown_track",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    fl = result["fl"]
    assert fl.c_track == pytest.approx(1.0)  # prior


def test_predict_g2_override_bypasses_lookup() -> None:
    model = _minimal_model()
    result = predict_cold_pressure(
        track="unknown_track",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        g2_typ_override=1.2,
        _model=model,
    )
    assert result["fl"].g2_typ == pytest.approx(1.2)


def test_predict_uses_user_provided_track_temp_when_given() -> None:
    model = _minimal_model()
    # Without track_temp: T_road defaults to T_air (no cloud cover provided)
    r_default = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    # With track_temp = 40 °C: T_eff blends up
    r_with = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        track_temp_c=40.0,
        _model=model,
    )
    assert r_default["fl"].t_eff_c == pytest.approx(20.0)  # T_road == T_air → T_eff == T_air
    # T_eff = 0.8·20 + 0.2·40 = 24
    assert r_with["fl"].t_eff_c == pytest.approx(24.0)
    # T_hot prediction is higher with the hotter T_eff baseline
    assert r_with["fl"].predicted_hot_temp_c > r_default["fl"].predicted_hot_temp_c


def test_predict_cold_uses_t_air_not_t_eff_for_gay_lussac() -> None:
    """Even when T_eff blends in road heat, the cold side of Gay-Lussac uses
    T_air — cold tires equilibrate to pit air, not hot asphalt."""
    model = _minimal_model()
    r = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        track_temp_c=50.0,
        _model=model,
    )
    # T_cold for the Gay-Lussac inversion should be 15 °C, even though T_eff is higher
    # We can verify by recomputing with the closed-form
    p = r["fl"]
    from motorsports_data_notebook.tire_model.energy_balance import (
        gay_lussac_cold_pressure_bar,
    )

    expected_cold = gay_lussac_cold_pressure_bar(
        target_hot_pressure_bar=1.95,
        t_hot_c=p.predicted_hot_temp_c,
        t_cold_c=15.0,  # T_air, not T_eff
    )
    assert p.cold_pressure_bar == pytest.approx(expected_cold)


def test_predict_missing_corner_in_target_raises_keyerror() -> None:
    model = _minimal_model()
    with pytest.raises(KeyError):
        predict_cold_pressure(
            track="track_a",
            car="ToyCar",
            lap_within_stint=5,
            target_hot_pressure_bar={"fl": 1.95, "fr": 1.95, "rl": 1.95},  # missing rr
            ambient_temp_c=20.0,
            _model=model,
        )


def test_predict_cross_track_uses_same_k_different_c_track() -> None:
    """Whole point of the energy-balance design: track shouldn't change K."""
    model = _minimal_model()
    r_a = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    r_b = predict_cold_pressure(
        track="track_b",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    # Same K, same τ
    assert r_a["fl"].K_kelvin_per_g2 == r_b["fl"].K_kelvin_per_g2
    assert r_a["fl"].tau_sec == r_b["fl"].tau_sec
    # Different c_track
    assert r_a["fl"].c_track != r_b["fl"].c_track


def test_predict_loads_from_disk_when_no_model_kwarg(tmp_path: Path) -> None:
    """End-to-end: predict_cold_pressure reads tire_model.json from dataset_root."""
    model = _minimal_model()
    (tmp_path / "tire_model.json").write_text(json.dumps(model))
    result = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        dataset_root=tmp_path,
    )
    assert result["fl"].cold_pressure_bar > 0
