"""Heat-input forms: units, linear-slip limit, activation and glitch clipping."""

from __future__ import annotations

import numpy as np
import pytest

from motorsports_data_notebook.tire_model.heat_input import HeatInput, ShareParams


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


# ---------------------------------------------------------------- heat paths (#380)


def _share(**kw: float) -> ShareParams:
    base: dict[str, float] = dict(
        p_f=0.53,
        g_transfer_front=3.0,
        g_transfer_rear=3.0,
        beta0=float(np.log(0.65 / 0.35)),
        beta1=0.0,
        eps_drive=1.0,
        eps_brake=1.0,
    )
    base.update(kw)
    return ShareParams(**base)


def _parts(n: int = 400, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    lat = rng.normal(0.0, 0.7, n)
    lng = rng.normal(0.0, 0.5, n)
    v = rng.uniform(10.0, 60.0, n)
    return (
        np.maximum(lat, 0.0),
        np.maximum(-lat, 0.0),
        np.maximum(lng, 0.0),
        np.maximum(-lng, 0.0),
        v,
    )


def test_heat_paths_off_leave_the_production_input_unchanged() -> None:
    from motorsports_data_notebook.tire_model.heat_input import share_heat_rate

    parts = _parts()
    for c in ("fl", "fr", "rl", "rr"):
        q0 = share_heat_rate(_share(), *parts, c)
        q1 = share_heat_rate(_share(r_carcass=0.0, c_rim=0.0, tau_rim_s=300.0), *parts, c)
        assert np.array_equal(q0, q1)


def test_load_share_conserves_weight_and_loads_the_outer_tyres() -> None:
    p = _share()
    lat_pos, lat_neg = np.array([0.0, 1.2, 0.0]), np.array([0.0, 0.0, 1.2])
    shares = {c: p.load_share(lat_pos, lat_neg, c) for c in ("fl", "fr", "rl", "rr")}
    assert sum(shares.values()) == pytest.approx([1.0, 1.0, 1.0])
    assert shares["fl"][0] == pytest.approx(0.53 / 2)
    assert shares["rr"][0] == pytest.approx(0.47 / 2)
    # a right turn (lat_pos) loads the left tyres, a left turn the right ones
    assert shares["fl"][1] > shares["fr"][1]
    assert shares["fr"][2] > shares["fl"][2]
    assert shares["fl"][1] + shares["fr"][1] == pytest.approx(0.53)


def test_carcass_path_adds_speed_times_load() -> None:
    from motorsports_data_notebook.tire_model.heat_input import share_heat_rate

    parts = _parts()
    lat_pos, lat_neg, _, _, v = parts
    p0, p1 = _share(), _share(r_carcass=0.2)
    for c in ("fl", "rr"):
        extra = share_heat_rate(p1, *parts, c) - share_heat_rate(p0, *parts, c)
        assert extra == pytest.approx(0.2 * v * p1.load_share(lat_pos, lat_neg, c))


def test_rim_path_is_the_brake_power_through_a_first_order_lag() -> None:
    from motorsports_data_notebook.tire_model.heat_input import share_heat_rate

    n = 900
    one = np.ones(n)
    zero = np.zeros(n)
    v, brake = 40.0 * one, 0.8 * one  # steady braking at 40 m/s
    tau = 120.0
    p0, p1 = _share(), _share(c_rim=0.5, tau_rim_s=tau)
    for c, bias in (("fl", 0.65), ("rl", 0.35)):
        power = v * brake * bias / 2.0
        steady = share_heat_rate(p1, zero, zero, zero, brake, v, c, dt_s=None)
        assert steady - share_heat_rate(p0, zero, zero, zero, brake, v, c) == pytest.approx(
            0.5 * power
        )
        lagged = share_heat_rate(p1, zero, zero, zero, brake, v, c)
        rise = (lagged - share_heat_rate(p0, zero, zero, zero, brake, v, c)) / (0.5 * power)
        t = np.arange(1, n + 1)
        assert rise == pytest.approx(1.0 - np.exp(-t / tau))
    # a new segment restarts the rim from cold
    starts = np.zeros(n, dtype=bool)
    starts[[0, 450]] = True
    y = share_heat_rate(p1, zero, zero, zero, brake, v, "fl", segment_starts=starts)
    y0 = share_heat_rate(p0, zero, zero, zero, brake, v, "fl")
    assert (y - y0)[450] == pytest.approx((y - y0)[0])


def test_first_order_lag_is_exact_for_a_held_input() -> None:
    from motorsports_data_notebook.tire_model.heat_input import first_order_lag

    u = np.r_[np.ones(50), np.zeros(50)]
    y = first_order_lag(u, 20.0, 1.0)
    assert y[49] == pytest.approx(1.0 - np.exp(-50 / 20.0))
    assert y[99] == pytest.approx(y[49] * np.exp(-50 / 20.0))
    assert first_order_lag(u, 20.0, 2.0)[49] == pytest.approx(1.0 - np.exp(-100 / 20.0))
