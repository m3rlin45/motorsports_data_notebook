"""Unit tests for the predictor and its fallback chain."""

from __future__ import annotations

import math

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
        "model_form": "T_hot - T_eff = K * g2_typ * (1 - exp(-t/tau_sec))",
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


def test_predict_cold_tire_temp_is_the_warmup_initial_condition() -> None:
    """The entered current tire temperature is used twice, consistently: as
    the Gay-Lussac cold side and as the warmup curve's starting point. A
    tire still warm from the previous run ends the lap hotter than one that
    started at air temperature, by the decayed start excess."""
    model = _minimal_model()
    base = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        _model=model,
    )
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
    # T_eff is an ambient property and does not move.
    assert p_base.t_eff_c == pytest.approx(p_warm.t_eff_c)
    # Default start temperature is T_air (not T_eff); the override replaces it.
    assert p_base.t_start_c == pytest.approx(15.0)
    assert p_base.t_cold_c == pytest.approx(15.0)
    assert p_warm.t_start_c == pytest.approx(25.0)
    assert p_warm.t_cold_c == pytest.approx(25.0)
    # The hot temperature moves by the decayed start excess.
    decay = math.exp(-p_base.t_at_lap_n_s / p_base.tau_sec)
    assert p_warm.predicted_hot_temp_c - p_base.predicted_hot_temp_c == pytest.approx(10.0 * decay)
    assert p_warm.predicted_hot_temp_c > p_base.predicted_hot_temp_c
    # Lap 0 returns exactly the start temperature.
    lap0 = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=0,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        cold_tire_temp_c=25.0,
        _model=model,
    )
    assert lap0["fl"].predicted_hot_temp_c == pytest.approx(25.0)
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
        "method": "uniform_scaled_lap_shares",
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


def test_per_corner_curve_scales_each_corner_by_its_own_series() -> None:
    """A simulated-lap curve carries g2_by_corner: a corner with its own
    series uses it, a corner without falls back to the corner-mean g2."""
    model = _model_with_pace_curve()
    for r in model["g2_typ_by_track_car_cond"]:
        if "g2_vs_lap_time" in r:
            # fl rises faster with pace than the mean: 1.5/0.9 at 55 s vs 1.2/0.9.
            r["g2_vs_lap_time"]["g2_by_corner"] = {"fl": [1.5, 0.9, 0.3]}
    p = predict_cold_pressure(
        _model=model,
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.8 for c in CORNERS},
        ambient_temp_c=20.0,
        target_lap_time_s=55.0,
    )
    assert p["fl"].g2_scale == pytest.approx(1.5 / 0.9)
    assert p["fr"].g2_scale == pytest.approx(1.2 / 0.9)
    assert p["fl"].g2_pace_source == "curve" and p["fr"].g2_pace_source == "curve"
    assert p["fl"].predicted_hot_temp_c > p["fr"].predicted_hot_temp_c


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


# ---------- Compound-aware K ----------


def _model_with_compound_k() -> dict:
    model = _model_with_pace_curve()
    model["K_by_car_compound_corner_cond"] = [
        {
            "car": "ToyCar",
            "compound": "A052",
            "corner": c,
            "condition": "dry",
            "value_kelvin_per_g2": 40.0,
            "stderr_kelvin_per_g2": 1.0,
            "n_laps": 50,
        }
        for c in CORNERS
    ]
    return model


def test_compound_k_overrides_pooled_k() -> None:
    model = _model_with_compound_k()
    kwargs = dict(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.8 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    pooled = predict_cold_pressure(**kwargs)
    comp = predict_cold_pressure(compound="A052", **kwargs)
    for c in CORNERS:
        assert comp[c].K_kelvin_per_g2 == pytest.approx(40.0)
        assert comp[c].K_source_bucket == ("ToyCar", "A052", c, "dry")
        assert comp[c].K_kelvin_per_g2 != pooled[c].K_kelvin_per_g2


def test_compound_applies_to_all_four_corners() -> None:
    model = _model_with_compound_k()
    kwargs = dict(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.8 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    p = predict_cold_pressure(compound="A052", **kwargs)
    for c in CORNERS:
        assert p[c].K_kelvin_per_g2 == pytest.approx(40.0)


def test_unknown_compound_falls_back_to_pooled_k() -> None:
    model = _model_with_compound_k()
    kwargs = dict(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.8 for c in CORNERS},
        ambient_temp_c=20.0,
        _model=model,
    )
    pooled = predict_cold_pressure(**kwargs)
    unknown = predict_cold_pressure(compound="RE-71RS", **kwargs)
    for c in CORNERS:
        assert unknown[c].K_kelvin_per_g2 == pooled[c].K_kelvin_per_g2
        assert unknown[c].K_source_bucket == pooled[c].K_source_bucket


def test_compound_k_condition_chain_falls_back_to_dry() -> None:
    model = _model_with_compound_k()  # only dry compound entries exist
    p = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=5,
        target_hot_pressure_bar={c: 1.8 for c in CORNERS},
        ambient_temp_c=20.0,
        track_condition="damp",
        compound="A052",
        _model=model,
    )
    assert p["fl"].K_source_bucket == ("ToyCar", "A052", "fl", "dry")


def test_predict_integrates_the_outlap_first_when_the_artifact_has_one() -> None:
    """With an out-lap table, lap N is preceded by the bucket's typical
    out-lap from the typed start temperature; --no-outlap and a missing
    table reproduce the single-segment behaviour."""
    model = _minimal_model()
    base = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=3,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        _model=model,
    )["fl"]
    assert base.outlap_time_s == 0.0 and base.outlap_source is None
    model2 = dict(model)
    model2["outlap_typ_by_track_car_cond"] = [
        {
            "track_canonical": "track_a",
            "car": "ToyCar",
            "condition": "dry",
            "outlap_moving_s": 90.0,
            "outlap_g2": 0.3,
            "n_laps_used": 12,
        },
    ]
    with_out = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=3,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        _model=model2,
    )["fl"]
    assert with_out.outlap_time_s == 90.0 and with_out.outlap_g2 == 0.3
    assert with_out.outlap_source == "exact"
    assert with_out.t_after_outlap_c is not None and with_out.t_after_outlap_c > with_out.t_start_c
    assert with_out.predicted_hot_temp_c > base.predicted_hot_temp_c
    assert with_out.cold_pressure_bar < base.cold_pressure_bar
    without = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=3,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        _model=model2,
        include_outlap=False,
    )["fl"]
    assert without.predicted_hot_temp_c == pytest.approx(base.predicted_hot_temp_c)
    overridden = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=3,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        _model=model2,
        outlap_time_s=0.0,
    )["fl"]
    assert overridden.predicted_hot_temp_c == pytest.approx(base.predicted_hot_temp_c)


def test_schema_v5_per_corner_intensity_is_used_with_fallback() -> None:
    """A ``q_typ_by_corner`` entry drives each corner with its own driving
    intensity (and ``outlap_q_by_corner`` its own out-lap); entries without
    the fields keep the corner-blind ``g2_typ`` / ``outlap_g2``."""
    model = _minimal_model()
    model = dict(model)
    entries = []
    for r in model["g2_typ_by_track_car_cond"]:
        r = dict(r)
        if r["track_canonical"] == "track_a" and r["condition"] == "dry":
            r["q_typ_by_corner"] = {"fl": 1.2, "fr": 0.6, "rl": 1.0, "rr": 0.5}
        entries.append(r)
    model["g2_typ_by_track_car_cond"] = entries
    model["outlap_typ_by_track_car_cond"] = [
        {
            "track_canonical": "track_a",
            "car": "ToyCar",
            "condition": "dry",
            "outlap_moving_s": 90.0,
            "outlap_g2": 0.3,
            "outlap_q_by_corner": {"fl": 0.5, "fr": 0.2, "rl": 0.4, "rr": 0.2},
            "n_laps_used": 12,
        },
    ]
    pred = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=3,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        _model=model,
    )
    assert pred["fl"].g2_typ == pytest.approx(1.2) and pred["fr"].g2_typ == pytest.approx(0.6)
    assert pred["fl"].outlap_g2 == pytest.approx(0.5) and pred["fr"].outlap_g2 == pytest.approx(0.2)
    # same K and tau on both fronts in the minimal model → the hotter corner is the loaded one
    assert pred["fl"].predicted_hot_temp_c > pred["fr"].predicted_hot_temp_c
    assert pred["fl"].cold_pressure_bar < pred["fr"].cold_pressure_bar
    # an override still applies to every corner
    over = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=3,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        g2_typ_override=0.8,
        _model=model,
    )
    assert over["fl"].g2_typ == pytest.approx(0.8) and over["rr"].g2_typ == pytest.approx(0.8)
    # fallback: no per-corner dict → corner-blind value for every corner
    plain = predict_cold_pressure(
        track="track_a",
        car="ToyCar",
        lap_within_stint=3,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        _model=_minimal_model(),
    )
    assert len({round(p.g2_typ, 9) for p in plain.values()}) == 1


# ---------- logged entries per raw car (#383) ----------


def _raw_car_model() -> dict:
    """``_minimal_model`` with ToyCar the pooled name of two raw cars, RawA
    and RawB; RawA has its own dry entry at track_a (heat, lap time, a
    pace curve, an out-lap) and nothing in the damp, RawB none."""
    model = dict(_minimal_model())
    model["car_aliases"] = {"RawA": "ToyCar", "RawB": "ToyCar"}
    model["outlap_typ_by_track_car_cond"] = [
        {
            "track_canonical": "track_a",
            "car": "ToyCar",
            "condition": "dry",
            "outlap_moving_s": 90.0,
            "outlap_g2": 0.3,
            "n_laps_used": 12,
        },
    ]
    model["g2_typ_by_track_raw_car_cond"] = [
        {
            "track_canonical": "track_a",
            "car": "RawA",
            "condition": "dry",
            "g2_typ": 1.2,
            "n_laps_used": 20,
            "g2_vs_lap_time": {"lap_time_s": [50.0, 70.0], "g2": [2.0, 1.0]},
        },
    ]
    model["lap_time_typ_by_track_raw_car_cond"] = [
        {
            "track_canonical": "track_a",
            "car": "RawA",
            "condition": "dry",
            "lap_time_typ_s": 58.0,
            "n_laps_used": 20,
        },
    ]
    model["outlap_typ_by_track_raw_car_cond"] = [
        {
            "track_canonical": "track_a",
            "car": "RawA",
            "condition": "dry",
            "outlap_moving_s": 80.0,
            "outlap_g2": 0.5,
            "n_laps_used": 4,
        },
    ]
    return model


def _predict(model: dict, car: str, **kw) -> Prediction:  # type: ignore[no-untyped-def]
    return predict_cold_pressure(
        track=kw.pop("track", "track_a"),
        car=car,
        lap_within_stint=3,
        target_hot_pressure_bar={c: 1.95 for c in CORNERS},
        ambient_temp_c=15.0,
        _model=model,
        **kw,
    )["fl"]


def test_a_raw_car_reads_its_own_entry_and_the_pooled_constants() -> None:
    model = _raw_car_model()
    own = _predict(model, "RawA", target_lap_time_s=60.0)
    pooled = _predict(model, "ToyCar", target_lap_time_s=60.0)
    # its own heat along its own curve, anchored at its own lap time (1.6 at 58 s)
    assert own.g2_typ == pytest.approx(1.2 * 1.5 / 1.6)
    assert own.lap_time_typ_s == 58.0 and own.g2_pace_source == "curve"
    assert own.outlap_time_s == 80.0 and own.outlap_g2 == 0.5
    assert pooled.g2_typ == pytest.approx(0.9) and pooled.lap_time_typ_s == 60.0
    assert pooled.outlap_time_s == 90.0
    # the constants are the pooled car's whichever name is given
    assert own.K_source_bucket == pooled.K_source_bucket == ("ToyCar", "fl", "dry")
    assert own.K_kelvin_per_g2 == pooled.K_kelvin_per_g2 and own.tau_sec == pooled.tau_sec


def test_a_raw_car_without_its_own_entry_reads_the_pooled_one() -> None:
    model = _raw_car_model()
    for kw in ({}, {"track": "track_b"}, {"track_condition": "damp"}):
        assert _predict(model, "RawB", **kw) == _predict(model, "ToyCar", **kw)
    # the condition chain comes first: RawA's damp reads the pooled damp
    # entry, not its own dry one ...
    damp = _predict(model, "RawA", track_condition="damp")
    pooled = _predict(model, "ToyCar", track_condition="damp")
    assert damp.g2_typ == pooled.g2_typ == pytest.approx(0.6)
    assert damp.lap_time_typ_s == pooled.lap_time_typ_s == 65.0
    # ... and each table walks the chain alone: with no damp out-lap at all,
    # the dry step reads RawA's own out-lap before the pooled one
    assert damp.outlap_source == pooled.outlap_source == "fallback(dry)"
    assert (damp.outlap_time_s, pooled.outlap_time_s) == (80.0, 90.0)


def test_an_artifact_without_raw_car_tables_answers_a_raw_car_as_before() -> None:
    model = _raw_car_model()
    for name in (
        "g2_typ_by_track_raw_car_cond",
        "lap_time_typ_by_track_raw_car_cond",
        "outlap_typ_by_track_raw_car_cond",
    ):
        del model[name]
    for kw in ({}, {"target_lap_time_s": 55.0}, {"track_condition": "wet"}, {"track": "x"}):
        assert _predict(model, "RawA", **kw) == _predict(model, "ToyCar", **kw)


def test_the_track_fallbacks_average_the_pooled_entries_only() -> None:
    """A car with no entry at a track reads the mean of the pooled entries
    there; the raw cars' rows are not counted twice."""
    from motorsports_data_notebook.tire_model.predict import _lookup_g2, _lookup_lap_time

    model = _raw_car_model()
    bare = {k: v for k, v in model.items() if not k.endswith("_by_track_raw_car_cond")}
    for m in (model, bare):
        assert _lookup_g2(m, "track_a", "Nobody", "dry") == _lookup_g2(
            bare, "track_a", "Nobody", "dry"
        )
        assert _lookup_lap_time(m, "track_a", "Nobody", "dry")[2] == "track_pooled"
    # a raw car whose own rows at the track are in another condition reads them first
    model["g2_typ_by_track_raw_car_cond"][0]["condition"] = "damp"
    model["lap_time_typ_by_track_raw_car_cond"][0]["condition"] = "damp"
    model["g2_typ_by_track_car_cond"] = [
        r for r in model["g2_typ_by_track_car_cond"] if r["track_canonical"] != "track_a"
    ] + [dict(r, condition="damp") for r in _minimal_model()["g2_typ_by_track_car_cond"][:1]]
    value, _n, source = _lookup_g2(model, "track_a", "RawA", "dry")
    assert source == "track_car_pooled" and value == pytest.approx(1.2)


def test_the_committed_artifact_answers_each_raw_car_as_its_pooled_car() -> None:
    """Until a retrain writes the raw-car tables, a raw car name gives the
    pooled car's prediction everywhere, as the calculators have always had it."""
    from motorsports_data_notebook.tire_etl.paths import default_dataset_root

    path = default_dataset_root() / "tire_model.json"
    if not path.exists():
        pytest.skip("no committed tire_model.json")
    model = json.loads(path.read_text())
    if any(name.endswith("_by_track_raw_car_cond") for name in model):
        pytest.skip("the committed artifact has raw-car entries")
    tracks = sorted({r["track_canonical"] for r in model["g2_typ_by_track_car_cond"]})
    for raw, pooled in model["car_aliases"].items():
        for track in tracks:
            for cond in ("dry", "damp", "wet"):
                kw = dict(track=track, track_condition=cond, target_lap_time_s=70.0)
                assert _predict(model, raw, **kw) == _predict(model, pooled, **kw)


# ------------------------------------------------- simulated entries (cutover)

#: RawA's dry entry simulated (the cutover's rows, schema §9: beside the
#: logged ones, which stay): 14 knots from 50 s to 76 s, supported from 54 s
#: to 70 s, at a fed-use lap time of 57 s; its out-lap stays logged. The
#: same numbers as the ports' tests.
SIM_KNOTS = [50.0 + 2 * i for i in range(14)]
SIM_TYP = 57.0


def _sim_series(k: float) -> list[float]:
    return [k / (1 + 0.07 * i) for i in range(len(SIM_KNOTS))]


def _sim_model(
    support: dict | None = None, *, with_support: bool = True, sim_first: bool = False
) -> dict:
    model = _raw_car_model()
    by_corner = {"fl": _sim_series(2.6), "fr": _sim_series(1.9)}
    by_corner |= {"rl": _sim_series(2.3), "rr": _sim_series(1.7)}
    curve: dict = {
        "lap_time_s": SIM_KNOTS,
        "g2": [sum(v[i] for v in by_corner.values()) / 4 for i in range(len(SIM_KNOTS))],
        "g2_by_corner": by_corner,
        "n_laps": 0,
        "method": "sim",
    }
    if with_support:
        curve["support"] = support or {"lap_time_min_s": 54.0, "lap_time_max_s": 70.0}
    key = {"track_canonical": "track_a", "car": "RawA", "condition": "dry", "source": "sim"}
    heat = key | {
        "g2_typ": 2.0,
        "n_laps_used": 0,
        "q_typ_by_corner": {"fl": 2.3, "fr": 1.7, "rl": 2.0, "rr": 1.5},
        "g2_vs_lap_time": curve,
    }
    lap = key | {"lap_time_typ_s": SIM_TYP, "n_laps_used": 0}
    for name, row in (
        ("g2_typ_by_track_raw_car_cond", heat),
        ("lap_time_typ_by_track_raw_car_cond", lap),
    ):
        model[name] = [row, *model[name]] if sim_first else [*model[name], row]
    return model


def _linear(x: float, xs: list[float], ys: list[float]) -> float:
    i = next(k for k, v in enumerate(xs) if v >= x)
    return ys[i - 1] + (x - xs[i - 1]) / (xs[i] - xs[i - 1]) * (ys[i] - ys[i - 1])


def test_a_simulated_entry_reports_its_provenance_and_reads_its_knots_linearly() -> None:
    from motorsports_data_notebook.tire_model.predict import _lookup_lap_time

    model = _sim_model()
    p = _predict(model, "RawA", target_lap_time_s=61.0)
    assert (p.heat_provenance, p.lap_time_provenance, p.outlap_provenance) == (
        "sim",
        "sim",
        "logged",
    )
    assert p.uses_simulation and p.pace_status == "ok"
    # the fed-use lap time is the prefill, and the 14 knots are read linearly
    assert _lookup_lap_time(model, "track_a", "RawA", "dry")[0] == SIM_TYP
    fl = _sim_series(2.6)
    expected = 2.3 * _linear(61.0, SIM_KNOTS, fl) / _linear(SIM_TYP, SIM_KNOTS, fl)
    assert p.g2_typ == pytest.approx(expected, abs=1e-12)
    # no target: the typical lap time's status
    assert _predict(model, "RawA").pace_status == "ok"
    # no out-lap asked for: none read
    assert _predict(model, "RawA", include_outlap=False).outlap_provenance is None


@pytest.mark.parametrize("target", [52.0, 71.0, 80.0])
def test_a_lap_time_outside_the_logged_pace_is_answered_and_flagged(target: float) -> None:
    p = _predict(_sim_model(), "RawA", target_lap_time_s=target)
    assert p.pace_status == "outside_support" and math.isfinite(p.cold_pressure_bar)


def test_a_lap_time_faster_than_the_table_is_answered_at_its_fastest_knot() -> None:
    p = _predict(_sim_model(), "RawA", target_lap_time_s=47.0)
    fl = _sim_series(2.6)
    assert p.pace_status == "no_solution"
    assert p.g2_scale == pytest.approx(fl[0] / _linear(SIM_TYP, SIM_KNOTS, fl), abs=1e-12)
    assert p.t_at_lap_n_s == 3 * 47.0
    # a sim curve without its support is outside it
    bare = _sim_model(with_support=False)
    assert _predict(bare, "RawA", target_lap_time_s=61.0).pace_status == "outside_support"


def test_a_sim_row_answers_its_exact_condition_only_whatever_its_place() -> None:
    """Schema §9: a simulated row sits beside the logged one and answers
    only its exact condition; every other step reads logged rows, so a wet
    or damp cell without a simulated row of its own reads what it reads
    today, and so does a saved pooled car."""
    from motorsports_data_notebook.tire_model.predict import _lookup_g2, _lookup_lap_time

    for sim_first in (False, True):
        model = _sim_model(sim_first=sim_first)
        assert _predict(model, "RawA", target_lap_time_s=61.0).heat_provenance == "sim"
        assert _lookup_lap_time(model, "track_a", "RawA", "dry")[0] == SIM_TYP

    def strip(model: dict) -> dict:
        for name in ("g2_typ_by_track_car_cond", "lap_time_typ_by_track_car_cond"):
            model[name] = [r for r in model[name] if r["condition"] != "damp"]
        return model

    # with the pooled damp rows gone, RawA's wet and damp fall back to its
    # own dry rows: the logged ones, as today, not the simulated one beside
    today, sim = strip(_raw_car_model()), strip(_sim_model())
    for cond in ("wet", "damp"):
        p = _predict(sim, "RawA", track_condition=cond, target_lap_time_s=61.0)
        assert p == _predict(today, "RawA", track_condition=cond, target_lap_time_s=61.0)
        assert p.g2_pace_source == "curve" and not p.uses_simulation
        assert _lookup_g2(sim, "track_a", "RawA", cond)[2] == "fallback(dry)"
    # with them, the pooled damp entry, as today; and a saved pooled car reads
    # the pooled logged rows
    for car, conds in (("RawA", ("wet", "damp")), ("ToyCar", ("dry", "damp", "wet"))):
        for cond in conds:
            kw = dict(track_condition=cond, target_lap_time_s=61.0)
            assert _predict(_sim_model(), car, **kw) == _predict(_raw_car_model(), car, **kw)


def test_logged_rows_carry_no_attribution() -> None:
    model = _sim_model()
    for car in ("ToyCar", "RawB"):
        p = _predict(model, car, target_lap_time_s=61.0)
        assert (p.heat_provenance, p.lap_time_provenance, p.outlap_provenance) == (
            "logged",
            "logged",
            "logged",
        )
        assert not p.uses_simulation and p.pace_status is None
    # a simulated out-lap alone carries the attribution
    out = _raw_car_model()
    rows = out["outlap_typ_by_track_raw_car_cond"]
    rows.append(rows[0] | {"source": "sim", "outlap_moving_s": 70.0})
    o = _predict(out, "RawA", target_lap_time_s=61.0)
    assert (o.heat_provenance, o.outlap_provenance, o.outlap_time_s, o.uses_simulation) == (
        "logged",
        "sim",
        70.0,
        True,
    )
    assert o.pace_status is None


def test_the_averaged_fallbacks_read_logged_rows_only() -> None:
    """The lead's ruling of 2026-11-27: a car without an entry of its own
    reads the mean of the logged rows only, a step with none is skipped,
    and an averaged answer is never simulated."""
    from motorsports_data_notebook.tire_model.predict import _lookup_g2, _lookup_lap_time

    def heat(model: dict, track: str, cond: str) -> dict:
        return next(
            r
            for r in model["g2_typ_by_track_car_cond"]
            if r["track_canonical"] == track and r["car"] == "ToyCar" and r["condition"] == cond
        )

    model = _minimal_model()
    assert _lookup_g2(model, "track_a", "Nobody", "dry")[0] == pytest.approx((0.9 + 0.6 + 0.8) / 3)
    # track_pooled: the simulated damp row is not counted
    heat(model, "track_a", "damp")["source"] = "sim"
    value, _n, step = _lookup_g2(model, "track_a", "Nobody", "dry")
    assert step == "track_pooled" and value == pytest.approx((0.9 + 0.8) / 2)
    nobody = _predict(model, "Nobody", target_lap_time_s=61.0)
    assert not nobody.uses_simulation and nobody.pace_status is None
    # track_car_pooled with only a simulated row is skipped for the track's logged rows
    model["g2_typ_by_track_car_cond"].remove(heat(model, "track_a", "dry"))
    value, _n, step = _lookup_g2(model, "track_a", "ToyCar", "dry")
    assert step == "track_pooled" and value == pytest.approx(0.8)
    # a track whose rows are all simulated is skipped for the global logged mean
    for r in model["g2_typ_by_track_car_cond"]:
        if r["track_canonical"] == "track_a":
            r["source"] = "sim"
    value, n, step = _lookup_g2(model, "track_a", "Nobody", "dry")
    assert step == "global" and value == pytest.approx(0.7) and n == 0
    # and the lap time the same: a simulated row is no term of an average
    for r in model["lap_time_typ_by_track_car_cond"]:
        r["source"] = "sim"
    assert _lookup_lap_time(model, "track_a", "Nobody", "dry") == (90.0, 0, "global")


def test_the_refactored_lookups_answer_as_before_on_the_committed_artifact() -> None:
    """The one chain walker leaves every lookup's answer where it was: the
    committed artifact reads as logged everywhere and its predictions match
    the parity fixture the ports are pinned to."""
    from motorsports_data_notebook.tire_etl.paths import default_dataset_root

    path = default_dataset_root() / "tire_model.json"
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tire_pressure_calculator"
        / "Tests"
        / "Fixtures"
        / "python_predictions.json"
    )
    if not path.exists() or not fixture.exists():
        pytest.skip("no committed tire_model.json or parity fixture")
    model = json.loads(path.read_text())
    if any("source" in r for r in model["g2_typ_by_track_car_cond"]):
        pytest.skip("the committed artifact has simulated entries")
    for case in json.loads(fixture.read_text()):
        inputs = dict(case["inputs"])
        target = inputs.pop("target_hot_pressure_bar")
        got = predict_cold_pressure(
            target_hot_pressure_bar={c: target for c in CORNERS}, _model=model, **inputs
        )
        for c in CORNERS:
            assert got[c].cold_pressure_bar == case["corners"][c]["cold_pressure_bar"]
            assert not got[c].uses_simulation and got[c].pace_status is None
