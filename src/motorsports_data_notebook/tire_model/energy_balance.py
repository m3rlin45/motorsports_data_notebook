"""Pure physics functions for the lumped-capacity tire thermal model.

Energy balance:

    m·c · dT/dt  =  c_track · α · g²(t)  −  h_air·(T−T_air)  −  h_road·(T−T_road)

Closed-form solution at constant g² ≈ ⟨g²⟩ starting from T = T_eff:

    T_hot(t) − T_eff  =  K · c_track · ⟨g²⟩ · (1 − exp(−t / τ_sec))

    with K     = α / (h_air + h_road)        [Kelvin per G²]
         τ_sec = m·c / (h_air + h_road)      [seconds]
         T_eff = (1 − w_road)·T_air + w_road·T_road

Gay-Lussac inversion at constant volume (absolute pressure / absolute temp):

    P_cold_abs  =  P_hot_abs · T_cold_K / T_hot_K

T_cold uses **T_air** (not T_eff) — cold tires equilibrate to ambient air, not
to the hot track surface.

All functions in this module are pure: no I/O, no global state.
"""

from __future__ import annotations

import math

# Atmospheric assumption — matches the C# calculator
# (tire_pressure_calculator/Core/ViewModels/TireCornerViewModel.cs:77-89)
P_ATM_BAR = 1.0
T_ZERO_C_TO_K = 273.15


def t_effective_c(
    t_air_c: float,
    t_road_c: float,
    w_road: float,
) -> float:
    """Blend air and road temperatures by the energy-OUT path weighting.

    ``T_eff = (1 − w_road) · T_air + w_road · T_road``

    ``w_road = h_road / (h_air + h_road)`` is the fraction of energy-OUT going
    to the track via conduction at the contact patch. ``0 ≤ w_road ≤ 1``;
    v0 fixes it at 0.2 (convection dominates).
    """
    if not 0.0 <= w_road <= 1.0:
        raise ValueError(f"w_road must be in [0, 1]; got {w_road}")
    return (1.0 - w_road) * t_air_c + w_road * t_road_c


def warmup_curve_c(
    t_seconds: float,
    *,
    t_eff_c: float,
    k_kelvin_per_g2: float,
    c_track: float,
    g2_typ: float,
    tau_sec: float,
) -> float:
    """Predicted tire temperature at on-track time ``t_seconds`` from stint start.

    Implements the closed-form solution of the lumped-capacity energy balance
    with constant ``g² ≈ g2_typ``:

        T(t) = T_eff + K · c_track · g2_typ · (1 − exp(−t / τ_sec))

    Parameters
    ----------
    t_seconds
        Cumulative on-track seconds since stint start. Must be ≥ 0.
    t_eff_c
        Effective ambient temperature (blended air + road) in °C.
    k_kelvin_per_g2
        Warmup gain K = α / (h_air + h_road), units Kelvin per G².
    c_track
        Per-track surface scalar (dimensionless), anchored at 1.0 for reference.
    g2_typ
        Session-average squared total acceleration, units G² (dimensionless).
    tau_sec
        Thermal time constant in seconds. Must be > 0.

    Returns
    -------
    Predicted tire temperature in °C.
    """
    if tau_sec <= 0:
        raise ValueError(f"tau_sec must be > 0; got {tau_sec}")
    if t_seconds < 0:
        raise ValueError(f"t_seconds must be >= 0; got {t_seconds}")
    warmup_frac = 1.0 - math.exp(-t_seconds / tau_sec)
    delta_t_inf = k_kelvin_per_g2 * c_track * g2_typ
    return t_eff_c + delta_t_inf * warmup_frac


def gay_lussac_cold_pressure_bar(
    *,
    target_hot_pressure_bar: float,
    t_hot_c: float,
    t_cold_c: float,
    p_atm_bar: float = P_ATM_BAR,
) -> float:
    """Invert Gay-Lussac's Law to compute cold pressure.

    At constant volume, P/T = const using **absolute** pressure and temperature.
    We convert gauge ↔ absolute by adding/subtracting ``p_atm_bar`` and °C ↔ K
    by adding/subtracting ``T_ZERO_C_TO_K``.

    Matches the C# calculator's convention exactly so round-trip is
    bit-identical (within float rounding):
    ``tire_pressure_calculator/Core/ViewModels/TireCornerViewModel.cs:77-89``.

    Parameters
    ----------
    target_hot_pressure_bar
        Desired hot gauge pressure in bar.
    t_hot_c
        Predicted hot tire temperature in °C.
    t_cold_c
        Cold (ambient) temperature in °C — use **T_air**, not T_eff.
    p_atm_bar
        Atmospheric pressure offset (defaults to 1.0 bar).

    Returns
    -------
    Cold gauge pressure in bar (not rounded; caller can round to display).
    """
    t_cold_k = t_cold_c + T_ZERO_C_TO_K
    t_hot_k = t_hot_c + T_ZERO_C_TO_K
    if t_hot_k <= 0 or t_cold_k <= 0:
        raise ValueError(
            f"Absolute temperatures must be positive (got T_hot_K={t_hot_k}, "
            f"T_cold_K={t_cold_k})"
        )
    p_hot_abs = target_hot_pressure_bar + p_atm_bar
    p_cold_abs = p_hot_abs * (t_cold_k / t_hot_k)
    return p_cold_abs - p_atm_bar


def t_road_proxy_c(
    *,
    t_air_c: float,
    cloud_cover_pct: float | None,
    sun_factor: float = 1.0,
    delta_sun_max_c: float = 10.0,
) -> float:
    """Estimate track surface temperature from air temp + cloud cover.

    On a clear day, asphalt can be 10+ K above the air. Cloud cover blocks
    incoming solar, so we scale a peak offset by (1 − cloud_cover/100). The
    ``sun_factor`` lets callers vary the peak by season / latitude / time of
    day if they want; the default 1.0 corresponds to a midday summer sun at
    Japan-typical latitudes.

    If ``cloud_cover_pct`` is None, fall back to ``T_road = T_air`` (no offset).
    """
    if cloud_cover_pct is None:
        return t_air_c
    clamped = max(0.0, min(100.0, cloud_cover_pct))
    return t_air_c + delta_sun_max_c * (1.0 - clamped / 100.0) * sun_factor


def warmup_recurrence_step_c(
    *,
    t_current_c: float,
    g2_current: float,
    dt_seconds: float,
    t_eff_c: float,
    k_kelvin_per_g2: float,
    c_track: float,
    tau_sec: float,
) -> float:
    """One step of the discretized energy-balance ODE (for sanity checks).

    The exact integrator of the linear ODE over one Δt step under the
    assumption ``g²`` is constant within the step:

        T_{i+1} = T_eff + (T_i − T_eff) · exp(−Δt/τ)
                        + K · c_track · g²_i · (1 − exp(−Δt/τ))

    Run repeatedly with a constant ``g²`` starting from ``T_0 = T_eff`` to
    recover :func:`warmup_curve_c` to within float precision (asserted in
    tests).

    Not used at inference in v0 — the closed-form is faster — but useful
    for tests and a future within-lap mode.
    """
    if tau_sec <= 0:
        raise ValueError(f"tau_sec must be > 0; got {tau_sec}")
    if dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be > 0; got {dt_seconds}")
    decay = math.exp(-dt_seconds / tau_sec)
    growth = 1.0 - decay
    return (
        t_eff_c + (t_current_c - t_eff_c) * decay + k_kelvin_per_g2 * c_track * g2_current * growth
    )
