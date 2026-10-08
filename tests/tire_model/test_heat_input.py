"""Heat-input forms: units, linear-slip limit, activation and glitch clipping."""

from __future__ import annotations

import numpy as np
import pytest

from motorsports_data_notebook.tire_model.heat_input import HeatInput


def test_g2_form_is_the_legacy_proxy() -> None:
    h = HeatInput(form="g2")
    q = h.rate(np.array([0.6, -0.8]), np.array([0.0, 0.3]), np.array([30.0, 10.0]), "FJ")
    assert q == pytest.approx([0.36, 0.73])
    assert h.units == "G^2"


def test_sliding_linear_slip_is_speed_times_g2() -> None:
    h = HeatInput(form="sliding")  # g_lim = inf → s(g) = g
    lat, lng, v = np.array([0.6, 1.0]), np.array([0.8, 0.0]), np.array([30.0, 20.0])
    assert h.rate(lat, lng, v, "FJ") == pytest.approx([30.0 * 1.0, 20.0 * 1.0])
    assert h.units == "G^2*m/s"


def test_activation_is_linear_far_below_the_limit_and_steep_near_it() -> None:
    h = HeatInput(form="sliding", g_lim={"FJ": 2.0}, u_max=0.98)
    g = np.array([0.1, 1.0, 1.9, 1.96, 5.0])
    s = h.slip(g, "FJ")
    assert s[0] == pytest.approx(0.1, rel=1e-2)  # ≈ g far below the limit
    assert s[1] > 1.0  # strictly above linear
    assert s[2] > 2.0 * s[1]  # steep near the limit
    assert s[3] == s[4]  # capped at u_max
    assert np.isfinite(s).all()


def test_unknown_car_uses_default_limit() -> None:
    h = HeatInput(form="sliding", g_lim={"FJ": 2.0}, g_lim_default=float("inf"))
    g = np.array([1.5])
    assert h.slip(g, "RX8") == pytest.approx(g)


def test_rolling_resistance_adds_speed_proportional_heat_at_zero_g() -> None:
    h = HeatInput(form="sliding", rr=0.2)
    q = h.rate(np.zeros(2), np.zeros(2), np.array([10.0, 30.0]), "FJ")
    assert q == pytest.approx([2.0, 6.0])


def test_power_form() -> None:
    h = HeatInput(form="power", p=3.0)
    assert h.rate(np.array([2.0]), np.array([0.0]), np.array([10.0]), "FJ") == pytest.approx([80.0])


def test_glitches_are_clipped_and_nans_are_zero() -> None:
    h = HeatInput(form="g2", g_clip=3.0)
    q = h.rate(np.array([33.0, np.nan]), np.array([0.0, np.nan]), np.array([20.0, 20.0]), "FJ")
    assert q == pytest.approx([9.0, 0.0])
    h2 = HeatInput(form="sliding")
    assert h2.rate(np.array([1.0]), np.array([0.0]), np.array([np.nan]), "FJ") == pytest.approx(
        [0.0]
    )


def test_invalid_form_rejected() -> None:
    with pytest.raises(ValueError):
        HeatInput(form="nope")
    with pytest.raises(ValueError):
        HeatInput(form="sliding", u_max=1.0)


def test_label() -> None:
    assert HeatInput().label() == "g2"
    assert HeatInput(form="sliding").label("FJ") == "V*g^2"
    assert HeatInput(form="sliding", g_lim={"FJ": 2.0}, rr=0.1).label("FJ") == (
        "V*g*atanh(g/2.00)+V*0.1"
    )


def test_load_transfer_conserves_weight_and_loads_the_outer_tyres() -> None:
    from motorsports_data_notebook.tire_model.heat_input import (
        BUILTIN_GEOMETRY,
        CarGeometry,
        corner_heat_rate,
        corner_load_ratio,
    )

    g = CarGeometry(wdf=0.5, xi_x=0.2, xi_y=0.3, rld=0.5)
    lat = np.array([0.0, 1.0, -1.0, 0.0])
    lng = np.array([0.0, 0.0, 0.0, -1.0])
    total = sum(g.corner_load(lat, lng, c) for c in ("fl", "fr", "rl", "rr"))
    assert total == pytest.approx([1.0, 1.0, 1.0, 1.0])
    # right turn (lat_g > 0) loads the left tyres; braking loads the fronts
    assert g.corner_load(lat, lng, "fl")[1] > g.corner_load(lat, lng, "fr")[1]
    assert g.corner_load(lat, lng, "fl")[3] > g.corner_load(lat, lng, "rl")[3]
    assert corner_load_ratio(lat, lng, "fl", g)[0] == pytest.approx(1.0)
    # the per-corner heat input scales with the load ratio^p
    h = HeatInput(form="g2")
    v = np.full(4, 30.0)
    q = h.rate(lat, lng, v, "X")
    q1 = corner_heat_rate(h, lat, lng, v, "X", "fl", g, 1.0)
    q2 = corner_heat_rate(h, lat, lng, v, "X", "fl", g, 2.0)
    r = corner_load_ratio(lat, lng, "fl", g)
    assert q1 == pytest.approx(q * r)
    assert q2 == pytest.approx(q * r**2)
    assert corner_heat_rate(h, lat, lng, v, "X", "fl", None, 1.0) == pytest.approx(q)
    assert set(BUILTIN_GEOMETRY) >= {"FJ", "Inferno 86"}
