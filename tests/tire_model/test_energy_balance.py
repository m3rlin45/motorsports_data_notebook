"""Unit tests for the pure energy-balance physics functions."""

from __future__ import annotations

import math

import pytest

from motorsports_data_notebook.tire_model.energy_balance import (
    P_ATM_BAR,
    T_ZERO_C_TO_K,
    gay_lussac_cold_pressure_bar,
    t_effective_c,
    t_road_proxy_c,
    warmup_curve_c,
    warmup_recurrence_step_c,
)

# ---------- t_effective_c ----------


def test_t_effective_blends_air_and_road_by_w_road() -> None:
    # w_road=0 → all air
    assert t_effective_c(t_air_c=20.0, t_road_c=40.0, w_road=0.0) == pytest.approx(20.0)
    # w_road=1 → all road
    assert t_effective_c(t_air_c=20.0, t_road_c=40.0, w_road=1.0) == pytest.approx(40.0)
    # w_road=0.2 → 80/20 blend
    assert t_effective_c(t_air_c=20.0, t_road_c=40.0, w_road=0.2) == pytest.approx(24.0)


def test_t_effective_rejects_out_of_range_w_road() -> None:
    with pytest.raises(ValueError):
        t_effective_c(t_air_c=20.0, t_road_c=40.0, w_road=-0.1)
    with pytest.raises(ValueError):
        t_effective_c(t_air_c=20.0, t_road_c=40.0, w_road=1.1)


# ---------- t_road_proxy_c ----------


def test_t_road_proxy_returns_air_when_cloud_cover_unknown() -> None:
    assert t_road_proxy_c(t_air_c=22.0, cloud_cover_pct=None) == pytest.approx(22.0)


def test_t_road_proxy_full_sun_adds_default_offset() -> None:
    # 0% clouds, sun_factor=1, default delta_max=10 K → +10 K
    assert t_road_proxy_c(
        t_air_c=20.0, cloud_cover_pct=0.0, sun_factor=1.0, delta_sun_max_c=10.0
    ) == pytest.approx(30.0)


def test_t_road_proxy_full_clouds_no_offset() -> None:
    assert t_road_proxy_c(
        t_air_c=20.0, cloud_cover_pct=100.0, sun_factor=1.0, delta_sun_max_c=10.0
    ) == pytest.approx(20.0)


def test_t_road_proxy_half_clouds_half_offset() -> None:
    assert t_road_proxy_c(
        t_air_c=20.0, cloud_cover_pct=50.0, sun_factor=1.0, delta_sun_max_c=10.0
    ) == pytest.approx(25.0)


def test_t_road_proxy_clamps_cloud_cover_to_unit_range() -> None:
    # Open-Meteo can return weird values occasionally; clamp gracefully
    assert t_road_proxy_c(
        t_air_c=20.0, cloud_cover_pct=-10.0, sun_factor=1.0, delta_sun_max_c=10.0
    ) == pytest.approx(30.0)
    assert t_road_proxy_c(
        t_air_c=20.0, cloud_cover_pct=200.0, sun_factor=1.0, delta_sun_max_c=10.0
    ) == pytest.approx(20.0)


# ---------- warmup_curve_c ----------


def test_warmup_curve_at_t_zero_returns_t_eff() -> None:
    t = warmup_curve_c(
        t_seconds=0.0, t_eff_c=20.0, k_kelvin_per_g2=60.0, c_track=1.0, g2_typ=0.8, tau_sec=240.0
    )
    assert t == pytest.approx(20.0)


def test_warmup_curve_at_infinity_returns_t_eff_plus_delta_inf() -> None:
    # At t = 10·τ the exponential is ~ 5e-5; close to ΔT_∞
    t = warmup_curve_c(
        t_seconds=2400.0,  # 10·τ
        t_eff_c=20.0,
        k_kelvin_per_g2=60.0,
        c_track=1.0,
        g2_typ=0.8,
        tau_sec=240.0,
    )
    expected = 20.0 + 60.0 * 1.0 * 0.8  # = 68 °C
    assert t == pytest.approx(expected, abs=0.01)


def test_warmup_curve_at_one_tau_reaches_63_percent() -> None:
    # (1 − e⁻¹) ≈ 0.6321
    t = warmup_curve_c(
        t_seconds=240.0, t_eff_c=20.0, k_kelvin_per_g2=60.0, c_track=1.0, g2_typ=0.8, tau_sec=240.0
    )
    delta_inf = 60.0 * 1.0 * 0.8
    expected = 20.0 + delta_inf * (1.0 - math.exp(-1.0))
    assert t == pytest.approx(expected)


def test_warmup_curve_rejects_invalid_tau_or_time() -> None:
    with pytest.raises(ValueError):
        warmup_curve_c(
            t_seconds=10.0, t_eff_c=20.0, k_kelvin_per_g2=60.0, c_track=1.0, g2_typ=0.8, tau_sec=0.0
        )
    with pytest.raises(ValueError):
        warmup_curve_c(
            t_seconds=-1.0,
            t_eff_c=20.0,
            k_kelvin_per_g2=60.0,
            c_track=1.0,
            g2_typ=0.8,
            tau_sec=240.0,
        )


# ---------- gay_lussac_cold_pressure_bar ----------


def test_gay_lussac_matches_csharp_calculator_convention() -> None:
    # Worked example: target hot 1.95 bar at 65 °C, ambient 18 °C
    # T_cold_K = 291.15, T_hot_K = 338.15
    # P_hot_abs = 2.95
    # P_cold_abs = 2.95 · 291.15 / 338.15 ≈ 2.5403
    # P_cold_gauge ≈ 1.5403
    p = gay_lussac_cold_pressure_bar(target_hot_pressure_bar=1.95, t_hot_c=65.0, t_cold_c=18.0)
    p_hot_abs = 1.95 + 1.0
    t_cold_k = 18.0 + T_ZERO_C_TO_K
    t_hot_k = 65.0 + T_ZERO_C_TO_K
    expected = p_hot_abs * (t_cold_k / t_hot_k) - 1.0
    assert p == pytest.approx(expected)


def test_gay_lussac_round_trip_no_temperature_change() -> None:
    # If hot_temp == cold_temp, cold_pressure == hot_pressure
    p = gay_lussac_cold_pressure_bar(target_hot_pressure_bar=1.95, t_hot_c=18.0, t_cold_c=18.0)
    assert p == pytest.approx(1.95)


def test_gay_lussac_hotter_tire_means_lower_cold_pressure() -> None:
    # Common-sense direction check
    p_cold_cool = gay_lussac_cold_pressure_bar(
        target_hot_pressure_bar=1.95, t_hot_c=60.0, t_cold_c=20.0
    )
    p_cold_hotter = gay_lussac_cold_pressure_bar(
        target_hot_pressure_bar=1.95, t_hot_c=80.0, t_cold_c=20.0
    )
    assert p_cold_hotter < p_cold_cool


def test_gay_lussac_rejects_zero_absolute_temperature() -> None:
    with pytest.raises(ValueError):
        gay_lussac_cold_pressure_bar(
            target_hot_pressure_bar=1.95, t_hot_c=-T_ZERO_C_TO_K, t_cold_c=20.0
        )


# ---------- discretized recurrence matches closed-form ----------


def test_discretized_recurrence_matches_closed_form() -> None:
    """Iterating the per-Δt recurrence with constant g² recovers the
    closed-form warmup curve to within float precision (this is the
    correctness invariant for any future within-lap fitting)."""
    t_eff = 20.0
    k = 60.0
    c_track = 1.0
    g2 = 0.8
    tau = 240.0
    dt = 1.0
    t_iter = t_eff
    for step in range(1, 600):  # 600 seconds = 2.5·τ
        t_iter = warmup_recurrence_step_c(
            t_current_c=t_iter,
            g2_current=g2,
            dt_seconds=dt,
            t_eff_c=t_eff,
            k_kelvin_per_g2=k,
            c_track=c_track,
            tau_sec=tau,
        )
        t_closed = warmup_curve_c(
            t_seconds=float(step),
            t_eff_c=t_eff,
            k_kelvin_per_g2=k,
            c_track=c_track,
            g2_typ=g2,
            tau_sec=tau,
        )
        assert t_iter == pytest.approx(t_closed, abs=1e-9)


def test_p_atm_constant_matches_csharp() -> None:
    # tire_pressure_calculator/Core/ViewModels/TireCornerViewModel.cs:79
    # uses a +1.0 / -1.0 bar atmospheric offset.
    assert P_ATM_BAR == 1.0
