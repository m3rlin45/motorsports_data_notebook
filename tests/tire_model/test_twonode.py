"""Two-node tread + carcass model: integrator, parameter recovery, IR gate."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from motorsports_data_notebook.tire_model import twonode as tn
from motorsports_data_notebook.tire_model.statespace import StintSeries

CORNERS = ("fl", "fr", "rl", "rr")
TRUE = tn.TwoNodeParams(a_s=2.5, b_sa=1 / 25.0, b_sc=1 / 40.0, b_cs=1 / 320.0, b_c=1 / 900.0)
T_EFF = 22.0


def test_expm2_matches_scipy() -> None:
    A = TRUE.matrix()
    assert np.abs(tn._expm2(A) - expm(A)).max() < 1e-12


def _one_batch(
    p: tn.TwoNodeParams, q: np.ndarray, v: np.ndarray, ts0: float, tc0: float
) -> tn._Batch:
    n = len(q)
    return tn._Batch(
        Q=q[None, :],
        V=v[None, :],
        teff=np.array([T_EFF]),
        x0=np.array([[ts0, tc0]]),
        obs_s=np.full((1, n), np.nan),
        obs_c=np.full((1, n), np.nan),
        valid=np.ones((1, n), bool),
        lap_end=np.zeros((1, n), bool),
        length=np.array([n]),
        meta=[{}],
    )


def test_modal_solution_matches_stepwise_reference() -> None:
    n = 900
    q = 0.3 + 0.6 * np.sin(np.arange(n) / 9.0) ** 2
    v = np.full(n, 30.0)
    Ts, Tc = tn.simulate_batch(TRUE, _one_batch(TRUE, q, v, 30.0, 28.0))
    rs, rc = tn._simulate_reference_loop(TRUE, q, v, T_EFF, 30.0, 28.0)
    assert np.abs(Ts[0] - rs).max() < 1e-9
    assert np.abs(Tc[0] - rc).max() < 1e-9


def test_picard_matches_stepwise_reference_with_speed_cooling() -> None:
    p = tn.TwoNodeParams(2.5, 1 / 25.0, 1 / 40.0, 1 / 320.0, 1 / 900.0, beta=1.5)
    n = 900
    q = 0.3 + 0.6 * np.sin(np.arange(n) / 9.0) ** 2
    v = 10 + 30 * np.abs(np.sin(np.arange(n) / 20.0))
    Ts, Tc = tn.simulate_batch(p, _one_batch(p, q, v, 30.0, 28.0))
    rs, rc = tn._simulate_reference_loop(p, q, v, T_EFF, 30.0, 28.0)
    assert np.abs(Ts[0] - rs).max() < 0.05
    assert np.abs(Tc[0] - rc).max() < 0.02


def _synthetic_stint(sid: str, n_laps: int, seed: int) -> StintSeries:
    rng = np.random.default_rng(seed)
    lap_s = 70
    n = n_laps * lap_s
    t = np.arange(n)
    q = 0.2 + 0.8 * np.sin(2 * np.pi * t / lap_s) ** 2 * (0.8 + 0.4 * rng.random())
    v = 15 + 25 * np.cos(2 * np.pi * t / lap_s) ** 2
    ts0 = tc0 = 25.0 + 5 * rng.random()
    Ts, Tc = tn._simulate_reference_loop(TRUE, q, v, T_EFF, ts0, tc0)
    surf = np.tile((Ts + rng.normal(0, 2.0, n))[:, None], (1, 4))
    # gas observed through 0.03 bar pressure steps from (tc0, 1.8 bar)
    p_abs = 2.8 * (Tc + 273.15) / (tc0 + 273.15)
    p_q = np.round(p_abs / 0.03) * 0.03
    obs = np.tile(((tc0 + 273.15) * p_q / 2.8 - 273.15)[:, None], (1, 4))
    lap_end = np.zeros(n, bool)
    lap_end[lap_s - 1 :: lap_s] = True
    return StintSeries(
        session_id=sid,
        stint_id=1,
        car="Inferno 86",
        track="tsukuba_2000",
        condition="dry",
        t_eff_c=T_EFF,
        g2=q,
        v=v,
        surf=surf,
        surf_zone_range=np.full((n, 4), 15.0),
        obs=obs,
        anchor_idx=np.zeros(4, int),
        t_start=np.full(4, tc0),
        p_start=np.full(4, 1.8),
        n_laps=n_laps,
        lap_ends=lap_end,
    )


def test_recovers_shared_constants_from_surface_and_gas() -> None:
    stints = [_synthetic_stint(f"s{i}", 14, i) for i in range(4)]
    fit = tn.fit_two_node_shared(stints, {"tsukuba_2000": 1.0})
    p = fit.params["fl"]
    assert p.a_s == pytest.approx(TRUE.a_s, rel=0.1)
    assert p.tau_surface_s == pytest.approx(TRUE.tau_surface_s, rel=0.1)
    assert p.capacity_ratio == pytest.approx(TRUE.capacity_ratio, rel=0.25)
    assert p.b_c == pytest.approx(TRUE.b_c, rel=0.3)
    ks, kc = p.steady_state_gain()
    ks_t, kc_t = TRUE.steady_state_gain()
    assert kc == pytest.approx(kc_t, rel=0.05)
    assert ks == pytest.approx(ks_t, rel=0.05)


def test_ir_gate_accepts_a_tread_and_rejects_bodywork_and_sentinels() -> None:
    good = _synthetic_stint("g", 10, 1)
    assert tn.ir_channel_ok(good, 0).ok
    panel = _synthetic_stint("p", 10, 2)
    # a covered sensor: near ambient, flat, no zone profile, below the gas
    panel.surf[:] = 24.0 + np.random.default_rng(0).normal(0, 0.3, panel.surf.shape)
    panel.surf_zone_range[:] = 1.0
    q = tn.ir_channel_ok(panel, 0)
    assert not q.ok and "below_gas" in q.reasons and "no_zone_profile" in q.reasons
    unplugged = _synthetic_stint("u", 10, 3)
    unplugged.surf[:] = np.nan  # the −200 sentinel is masked to NaN at binning
    assert tn.ir_channel_ok(unplugged, 0).reasons == ["too_few_seconds"]
    short = _synthetic_stint("s", 3, 4)
    assert not tn.ir_channel_ok(short, 0).ok


def test_gate_masks_failed_channels_and_excludes_other_cars() -> None:
    a = _synthetic_stint("a", 10, 5)
    b = _synthetic_stint("b", 10, 6)
    b.car = "KK-F"
    df = tn.gate_ir([a, b])
    assert df[df.session_id == "a"]["ok"].all()
    assert (df[df.session_id == "b"]["reasons"] == "car_excluded").all()
    assert np.isnan(b.surf).all()
    assert np.isfinite(a.surf).any()
