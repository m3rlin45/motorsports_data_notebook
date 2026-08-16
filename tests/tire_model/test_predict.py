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
    """Synthetic schema-v2 (condition-aware) model exercising the fallback chain.

    - Cars: "ToyCar" and "OtherCar"
    - Tracks: "track_a" (anchor), "track_b"
    - Conditions: dry (covered for ToyCar + OtherCar) + damp (covered for ToyCar only)
    - K_buckets cover every (ToyCar, corner, dry) and (ToyCar, corner, damp);
      only (OtherCar, rr, dry) is present so the OtherCar (car) fallback is
      exercised on other corners.
    """
    return {
        "schema_version": 2,
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
        "conditions": {
            "values": ["dry", "damp", "wet"],
            "default": "dry",
            "classification": {
                "from_field": "precipitation_mm_hr",
                "thresholds": {"dry_max": 0.1, "damp_max": 1.0},
                "rule": "test",
            },
        },
        "corners": list(CORNERS),
        "min_samples_per_bucket": 5,
        "priors_when_no_fit": {
            "tau_sec_seconds": 240.0,
            "K_kelvin_per_g2": 60.0,
            "c_track": 1.0,
        },
        "tau_sec_by_car_corner_cond": (
            [
                {
                    "car": "ToyCar",
                    "corner": c,
                    "condition": "dry",
                    "value_seconds": 200.0 + 10.0 * i,
                    "stderr_seconds": 5.0,
                    "n_samples_used": 100,
                    "from_prior": False,
                }
                for i, c in enumerate(CORNERS)
            ]
            + [
                {
                    "car": "ToyCar",
                    "corner": c,
                    "condition": "damp",
                    "value_seconds": 150.0 + 10.0 * i,  # lower τ in damp
                    "stderr_seconds": 10.0,
                    "n_samples_used": 30,
                    "from_prior": False,
                }
                for i, c in enumerate(CORNERS)
            ]
            + [
                {
                    "car": "OtherCar",
                    "corner": c,
                    "condition": "dry",
                    "value_seconds": 250.0,
                    "stderr_seconds": 8.0,
                    "n_samples_used": 50,
                    "from_prior": False,
                }
                for c in CORNERS
            ]
        ),
        "K_buckets": (
            [
                {
                    "key": {"car": "ToyCar", "corner": c, "condition": "dry"},
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
                    "key": {"car": "ToyCar", "corner": c, "condition": "damp"},
                    "value_kelvin_per_g2": 20.0 + 2.0 * i,  # lower K in damp
                    "stderr_kelvin_per_g2": 3.0,
                    "n_samples": 30,
                    "from_prior": False,
                    "from_single_track": False,
                }
                for i, c in enumerate(CORNERS)
            ]
            + [
                {
                    "key": {"car": "OtherCar", "corner": "rr", "condition": "dry"},
                    "value_kelvin_per_g2": 45.0,
                    "stderr_kelvin_per_g2": 3.0,
                    "n_samples": 50,
                    "from_prior": False,
                    "from_single_track": True,
                },
            ]
        ),
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
        "g2_typ_by_track_car_cond": [
            {
                "track_canonical": "track_a",
                "car": "ToyCar",
                "condition": "dry",
                "g2_typ": 0.9,
                "n_laps_used": 100,
            },
            {
                "track_canonical": "track_a",
                "car": "ToyCar",
                "condition": "damp",
                "g2_typ": 0.6,  # drivers go slower in damp
                "n_laps_used": 30,
            },
            {
                "track_canonical": "track_b",
                "car": "ToyCar",
                "condition": "dry",
                "g2_typ": 0.7,
                "n_laps_used": 50,
            },
            {
                "track_canonical": "track_a",
                "car": "OtherCar",
                "condition": "dry",
                "g2_typ": 0.8,
                "n_laps_used": 30,
            },
        ],
        "lap_time_typ_by_track_car_cond": [
            {
                "track_canonical": "track_a",
                "car": "ToyCar",
                "condition": "dry",
                "lap_time_typ_s": 60.0,
                "n_laps_used": 100,
            },
            {
                "track_canonical": "track_a",
                "car": "ToyCar",
                "condition": "damp",
                "lap_time_typ_s": 65.0,
                "n_laps_used": 30,
            },
            {
                "track_canonical": "track_b",
                "car": "ToyCar",
                "condition": "dry",
                "lap_time_typ_s": 100.0,
                "n_laps_used": 50,
            },
            {
                "track_canonical": "track_a",
                "car": "OtherCar",
                "condition": "dry",
                "lap_time_typ_s": 80.0,
                "n_laps_used": 30,
            },
        ],
        "fallback_order_for_K": [
            ["car", "corner", "condition"],
            ["car", "corner"],
            ["car"],
            [],
        ],
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
    assert fl.K_source_bucket == ("ToyCar", "fl", "dry")
    assert not fl.K_from_prior
    assert fl.c_track == pytest.approx(1.0)
    assert fl.g2_typ == pytest.approx(0.9)
    assert fl.predicted_hot_temp_c > 20.0


def test_predict_damp_uses_damp_K() -> None:
    """The damp K (20-26 K/G²) is much lower than dry (50-65), so the predicted
    hot temp must drop and the resulting cold pressure must rise."""
    model = _minimal_model()
    dry = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        track_condition="dry",
        _model=model,
    )
    damp = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        track_condition="damp",
        _model=model,
    )
    assert damp["fl"].K_kelvin_per_g2 < dry["fl"].K_kelvin_per_g2
    assert damp["fl"].predicted_hot_temp_c < dry["fl"].predicted_hot_temp_c
    # Cold pressure higher in damp because the tire heats less
    assert damp["fl"].cold_pressure_bar > dry["fl"].cold_pressure_bar
    assert damp["fl"].K_source_bucket == ("ToyCar", "fl", "damp")


def test_predict_wet_with_no_wet_data_falls_back_to_damp_K() -> None:
    """The synthetic model has no `wet` entries but does have `damp`;
    requesting wet should prefer damp over dry (physically closer)."""
    model = _minimal_model()
    result = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        track_condition="wet",
        _model=model,
    )
    # wet → damp (closest neighbor) → dry. Synthetic model has damp data.
    assert result["fl"].K_source_bucket == ("ToyCar", "fl", "damp")


def test_predict_wet_with_no_rain_data_falls_back_to_dry_K() -> None:
    """If neither wet nor damp has data, fall back all the way to dry."""
    model = _minimal_model()
    # Strip out damp buckets so the chain has to walk through to dry
    model = {
        **model,
        "K_buckets": [r for r in model["K_buckets"] if r["key"]["condition"] != "damp"],
        "tau_sec_by_car_corner_cond": [
            r for r in model["tau_sec_by_car_corner_cond"] if r["condition"] != "damp"
        ],
        "g2_typ_by_track_car_cond": [
            r for r in model["g2_typ_by_track_car_cond"] if r["condition"] != "damp"
        ],
        "lap_time_typ_by_track_car_cond": [
            r for r in model["lap_time_typ_by_track_car_cond"] if r["condition"] != "damp"
        ],
    }
    result = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        track_condition="wet",
        _model=model,
    )
    assert result["fl"].K_source_bucket == ("ToyCar", "fl", "dry")


def test_predict_falls_back_to_car_level_when_corner_missing() -> None:
    """OtherCar only has K[rr, dry]; FL/FR/RL fall back to (car, corner) or (car)."""
    model = _minimal_model()
    result = predict_cold_pressure(
        track="track_a",
        car="OtherCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    # No (OtherCar, fl, *) in the table → falls back to (OtherCar) mean
    fl = result["fl"]
    assert fl.K_source_bucket == ("OtherCar",)
    assert fl.K_kelvin_per_g2 == pytest.approx(45.0)


def test_predict_falls_back_to_global_when_car_unknown() -> None:
    model = _minimal_model()
    result = predict_cold_pressure(
        track="track_a",
        car="UnknownCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    fl = result["fl"]
    assert fl.K_from_prior is True
    assert fl.K_source_bucket == ()


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
    assert result["fl"].c_track == pytest.approx(1.0)


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
    r_default = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    r_with = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=20.0,
        track_temp_c=40.0,
        _model=model,
    )
    assert r_default["fl"].t_eff_c == pytest.approx(20.0)
    assert r_with["fl"].t_eff_c == pytest.approx(24.0)
    assert r_with["fl"].predicted_hot_temp_c > r_default["fl"].predicted_hot_temp_c


def test_predict_cold_uses_t_air_not_t_eff_for_gay_lussac() -> None:
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
    p = r["fl"]
    from motorsports_data_notebook.tire_model.energy_balance import (
        gay_lussac_cold_pressure_bar,
    )

    expected_cold = gay_lussac_cold_pressure_bar(
        target_hot_pressure_bar=1.95,
        t_hot_c=p.predicted_hot_temp_c,
        t_cold_c=15.0,
    )
    assert p.cold_pressure_bar == pytest.approx(expected_cold)


def test_predict_cold_tire_temp_overrides_ambient_in_gay_lussac_only() -> None:
    """When the tire isn't at air temperature (e.g., warm garage), the user
    can pin T_cold separately. It MUST only affect the Gay-Lussac inversion
    — the warmup-curve target T_eff is still computed from ambient_temp_c."""
    model = _minimal_model()
    base = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        _model=model,
    )
    # Tire is 10 °C warmer than air (sat in a heated garage). Cold side of
    # Gay-Lussac uses 25 °C; T_eff and predicted hot temp are unchanged.
    warm_tire = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        cold_tire_temp_c=25.0,
        _model=model,
    )
    p_base = base["fl"]
    p_warm = warm_tire["fl"]
    # T_eff and predicted hot temp unchanged
    assert p_base.t_eff_c == pytest.approx(p_warm.t_eff_c)
    assert p_base.predicted_hot_temp_c == pytest.approx(p_warm.predicted_hot_temp_c)
    # T_cold differs
    assert p_base.t_cold_c == pytest.approx(15.0)
    assert p_warm.t_cold_c == pytest.approx(25.0)
    # Cold pressure HIGHER for a warmer cold-side (P/T constant ⇒ larger
    # T_cold ⇒ larger P_cold for the same P_hot/T_hot)
    assert p_warm.cold_pressure_bar > p_base.cold_pressure_bar


def test_predict_missing_corner_in_target_raises_keyerror() -> None:
    model = _minimal_model()
    with pytest.raises(KeyError):
        predict_cold_pressure(
            track="track_a",
            car="ToyCar",
            lap_within_stint=5,
            target_hot_pressure_bar={"fl": 1.95, "fr": 1.95, "rl": 1.95},
            ambient_temp_c=20.0,
            _model=model,
        )


def test_predict_cross_track_uses_same_k_different_c_track() -> None:
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
    assert r_a["fl"].K_kelvin_per_g2 == r_b["fl"].K_kelvin_per_g2
    assert r_a["fl"].tau_sec == r_b["fl"].tau_sec
    assert r_a["fl"].c_track != r_b["fl"].c_track


def test_predict_rejects_unknown_condition() -> None:
    model = _minimal_model()
    with pytest.raises(ValueError):
        predict_cold_pressure(
            track="track_a",
            car="ToyCar",
            lap_within_stint=5,
            target_hot_pressure_bar={c: 1.95 for c in CORNERS},
            ambient_temp_c=20.0,
            track_condition="monsoon",
            _model=model,
        )


def test_predict_loads_from_disk_when_no_model_kwarg(tmp_path: Path) -> None:
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


# ---------- Target-lap-time feature (schema v3) ----------


def _model_with_pace_curve() -> dict:
    """Minimal model + pace block: track_a/ToyCar/dry carries a curve."""
    model = _minimal_model()
    model["schema_version"] = 3
    model["g2_lap_time_model"] = {
        "method": "sector_knn_median_curve",
        "default_exponent": 2.0,
        "multiplier_clamp": {"min": 0.4, "max": 2.5},
    }
    for r in model["g2_typ_by_track_car_cond"]:
        if r["track_canonical"] == "track_a" and r["car"] == "ToyCar" and r["condition"] == "dry":
            # Linear curve through (55 s, 1.2) … (65 s, 0.6); typical lap is 60 s.
            r["g2_vs_lap_time"] = {
                "lap_time_s": [55.0, 60.0, 65.0],
                "g2": [1.2, 0.9, 0.6],
                "n_laps": 80,
            }
    return model


def test_target_lap_time_none_is_v2_behavior() -> None:
    model = _model_with_pace_curve()
    kwargs = dict(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.8 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    base = predict_cold_pressure(**kwargs)
    assert base["fl"].g2_scale == 1.0
    assert base["fl"].g2_pace_source is None
    assert base["fl"].target_lap_time_s is None


def test_target_lap_time_equal_to_typical_matches_no_target() -> None:
    model = _model_with_pace_curve()
    kwargs = dict(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.8 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    base = predict_cold_pressure(**kwargs)
    at_typ = predict_cold_pressure(target_lap_time_s=60.0, **kwargs)
    for c in CORNERS:
        assert at_typ[c].predicted_hot_temp_c == pytest.approx(base[c].predicted_hot_temp_c)
        assert at_typ[c].cold_pressure_bar == pytest.approx(base[c].cold_pressure_bar)
    assert at_typ["fl"].g2_scale == pytest.approx(1.0)
    assert at_typ["fl"].g2_pace_source == "curve"


def test_faster_target_heats_more_and_lowers_cold_pressure() -> None:
    model = _model_with_pace_curve()
    kwargs = dict(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.8 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    slow = predict_cold_pressure(target_lap_time_s=64.0, **kwargs)
    fast = predict_cold_pressure(target_lap_time_s=56.0, **kwargs)
    for c in CORNERS:
        assert fast[c].predicted_hot_temp_c > slow[c].predicted_hot_temp_c
        assert fast[c].cold_pressure_bar < slow[c].cold_pressure_bar
    # Curve is linear here: at 56 s g2 = 1.14, ratio vs 0.9 at 60 s
    assert fast["fl"].g2_scale == pytest.approx(1.14 / 0.9)
    # Time-on-track follows the target, not the typical lap time
    assert fast["fl"].t_at_lap_n_s == pytest.approx(5 * 56.0)
    assert slow["fl"].t_at_lap_n_s == pytest.approx(5 * 64.0)


def test_target_beyond_curve_range_clamps_to_endpoint() -> None:
    model = _model_with_pace_curve()
    kwargs = dict(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.8 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    at_edge = predict_cold_pressure(target_lap_time_s=55.0, **kwargs)
    beyond = predict_cold_pressure(target_lap_time_s=40.0, **kwargs)
    assert beyond["fl"].g2_scale == pytest.approx(at_edge["fl"].g2_scale)


def test_bucket_without_curve_falls_back_to_default_exponent() -> None:
    model = _model_with_pace_curve()
    kwargs = dict(
        track="track_b",  # no curve on this bucket
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.8 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    lap_typ = predict_cold_pressure(**kwargs)["fl"].lap_time_typ_s
    target = lap_typ * 0.9
    p = predict_cold_pressure(target_lap_time_s=target, **kwargs)
    assert p["fl"].g2_pace_source == "exponent"
    assert p["fl"].g2_scale == pytest.approx((lap_typ / target) ** 2.0)


def test_extreme_target_is_clamped_by_multiplier_clamp() -> None:
    model = _model_with_pace_curve()
    kwargs = dict(
        track="track_b",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.8 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    p = predict_cold_pressure(target_lap_time_s=10.0, **kwargs)
    assert p["fl"].g2_scale == pytest.approx(2.5)  # clamp max


def test_nonpositive_target_lap_time_raises() -> None:
    model = _model_with_pace_curve()
    with pytest.raises(ValueError, match="target_lap_time_s"):
        predict_cold_pressure(
            track="track_a",
            car="ToyCar",
            lap_within_stint=5,
            target_hot_pressure_bar={c: 1.8 for c in CORNERS},
            ambient_temp_c=20.0,
            target_lap_time_s=0.0,
            _model=model,
        )
