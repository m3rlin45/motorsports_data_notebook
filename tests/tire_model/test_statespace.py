"""Per-second (1 Hz) fit: parameter recovery on synthetic stints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from motorsports_data_notebook.tire_model import statespace as ss
from motorsports_data_notebook.tire_model.cleaning import mask_stale_session_prefix
from motorsports_data_notebook.tire_model.energy_balance import P_ATM_BAR, T_ZERO_C_TO_K

CORNERS = ("fl", "fr", "rl", "rr")
TRUE_K = 30.0
TRUE_TAU = 300.0
T_EFF = 25.0
T_START = 28.0
P_START = 1.80
LAP_S = 60
HZ = 5


def _make_stint(
    root: Path,
    sid: str,
    track: str,
    n_laps: int,
    *,
    quantise: bool,
    drop_laps: set[int] = frozenset(),
    p_start: float = P_START,
    p_exp: float = 0.0,
    kappa: float = 0.0,
    lag_s: float = 0.0,
    anchor_error_bar: float = 0.0,
) -> pd.DataFrame:
    """Write one session's timeseries parquet and return its lap rows.

    ``p_exp`` > 0 generates the data with the pressure-dependent heat input
    ``q · (P_ref / P(t))^n`` where ``P(t)`` is the gas-law pressure. ``lag_s``
    > 0 heats the gas with ``q`` through a first-order lag of that time
    constant, from zero at the stint start (the tread-to-gas lag).
    ``anchor_error_bar`` is added to the recorded anchor pressure (a wrong
    pit-exit reading; the series itself stays true)."""
    n_s = n_laps * LAP_S
    t_1hz = np.arange(n_s)
    g2 = 0.4 + 0.8 * np.sin(2 * np.pi * t_1hz / LAP_S) ** 2  # constant within each second
    q_eff = ss.lowpass_heat(g2, lag_s) if lag_s else g2
    a, b = TRUE_K / TRUE_TAU, 1.0 / TRUE_TAU
    T = np.empty(n_s + 1)
    T[0] = T_START
    for i in range(n_s):
        p_abs = (p_start + P_ATM_BAR) * (T[i] + T_ZERO_C_TO_K) / (T_START + T_ZERO_C_TO_K)
        pf = (ss.P_REF_ABS_BAR / p_abs) ** p_exp
        teq = T_EFF + a * q_eff[i] * pf / b
        T[i + 1] = teq + (T[i] - teq) * np.exp(-b)
    # gas-law pressure from the anchor (T_START, p_start), sampled at 5 Hz;
    # read at speed it sits below the gas-law value by (1 + κ v²)
    # speed varies on a different period from g² so κ is separable from K/τ
    v_1hz = 15.0 + 25.0 * np.cos(2 * np.pi * t_1hz / (0.37 * LAP_S)) ** 2
    t_k = T[:-1] + T_ZERO_C_TO_K
    p = (p_start + P_ATM_BAR) * t_k / (T_START + T_ZERO_C_TO_K) / (1 + kappa * v_1hz**2) - P_ATM_BAR
    p_true = p.copy()
    if quantise:
        p = np.round(p / 0.03) * 0.03
    rows = []
    for i in range(n_s):
        lap = i // LAP_S
        for k in range(HZ):
            rows.append(
                {
                    "session_id": sid,
                    "lap_num": lap,
                    "sample_idx": i * HZ + k,
                    "t_session_s": i + k / HZ,
                    "speed_ms": v_1hz[i] if kappa else 25.0,
                    "lat_g": np.sqrt(g2[i]),
                    "long_g": 0.0,
                    **{f"tpms_press_{c}_bar": p[i] for c in CORNERS},
                }
            )
    ts = pd.DataFrame(rows)
    out = root / "timeseries" / "2026-01"
    out.mkdir(parents=True, exist_ok=True)
    ts.to_parquet(out / f"{sid}.parquet")
    # The dataset is raw and the fit masks the TPMS streams on load; the
    # production anchor is the first *live* reading of the cleaned laps.
    # Mirror that: anchor where the masked series becomes finite (quantised
    # pressure sits flat for the first seconds and reads as a stale prefix).
    masked = mask_stale_session_prefix(pa.Table.from_pandas(ts[["tpms_press_fl_bar"]]))
    pm = masked.column("tpms_press_fl_bar").to_numpy(zero_copy_only=False)
    k0 = int(np.argmax(np.isfinite(pm)))
    i0 = k0 // HZ
    t_anchor_s = float(ts["t_session_s"].iloc[k0])
    laps = pd.DataFrame(
        {
            "session_id": sid,
            "stint_id": 1,
            "lap_num": [l for l in range(n_laps) if l not in drop_laps],
            "car": "FJ",
            "track_canonical": track,
            "condition": "dry",
            "t_eff_c": T_EFF,
        }
    )
    # The anchor is the exact (T, P) pair at that instant. A noise-free
    # quantised series first changes exactly at a step edge, where the TPMS
    # reading is half a quantum (0.015 bar ≈ 1.8 K) off the true pressure —
    # a worst case the stale rule lands on by construction here, not a
    # property of the fit under test.
    for c in CORNERS:
        laps[f"t_anchor_{c}"] = t_anchor_s
        laps[f"t_start_{c}"] = float(T[i0])
        laps[f"p_start_{c}"] = float(p_true[i0]) + anchor_error_bar  # first live reading
    return laps


@pytest.mark.parametrize("quantise", [False, True])
def test_recovers_tau_and_k(tmp_path: Path, quantise: bool) -> None:
    laps = pd.concat(
        [
            _make_stint(tmp_path, "s_tsukuba", "tsukuba_2000", 12, quantise=quantise),
            _make_stint(tmp_path, "s_fuji", "fuji", 12, quantise=quantise),
        ],
        ignore_index=True,
    )
    stints = ss.build_stint_series(tmp_path, laps)
    fit = ss.fit_cells(stints, v_corr=True)
    tol = 0.03 if not quantise else 0.08
    assert 0.0 <= fit.kappa["FJ"] <= ss.KAPPA_BOUNDS[1]  # constant speed: κ cancels, fit unaffected
    for c in CORNERS:
        a, b = fit.ab[("FJ", c, "dry")]
        assert 1.0 / b == pytest.approx(TRUE_TAU, rel=tol)
        assert a / b == pytest.approx(TRUE_K, rel=tol)


def test_dropped_mid_stint_lap_keeps_the_clock(tmp_path: Path) -> None:
    """A lap the usability filters removed still advances the stint clock and
    heats the tire; only its observations are left out."""
    laps = pd.concat(
        [
            _make_stint(
                tmp_path, "s_tsukuba", "tsukuba_2000", 12, quantise=False, drop_laps={4, 5}
            ),
            _make_stint(tmp_path, "s_fuji", "fuji", 12, quantise=False, drop_laps={3}),
        ],
        ignore_index=True,
    )
    stints = ss.build_stint_series(tmp_path, laps)
    assert [s.n_laps for s in stints] == [10, 11]
    assert len(stints[0].g2) == 12 * LAP_S  # clock spans the dropped laps
    assert np.isnan(stints[0].obs[4 * LAP_S : 6 * LAP_S, 0]).all()  # not scored there
    fit = ss.fit_cells(stints)
    a, b = fit.ab[("FJ", "fl", "dry")]
    assert 1.0 / b == pytest.approx(TRUE_TAU, rel=0.03)
    assert a / b == pytest.approx(TRUE_K, rel=0.03)


def test_recovers_pressure_exponent(tmp_path: Path) -> None:
    """Stints at different set pressures identify ``q ∝ (P_ref/P)^n``."""
    true_n = 1.2
    laps = pd.concat(
        [
            _make_stint(
                tmp_path, "s_lo", "tsukuba_2000", 12, quantise=False, p_start=1.2, p_exp=true_n
            ),
            _make_stint(
                tmp_path, "s_hi", "tsukuba_2000", 12, quantise=False, p_start=2.0, p_exp=true_n
            ),
            _make_stint(tmp_path, "s_fuji", "fuji", 12, quantise=False, p_start=1.6, p_exp=true_n),
        ],
        ignore_index=True,
    )
    stints = ss.build_stint_series(tmp_path, laps)
    fit = ss.fit_cells(stints, p_mode="instant")
    assert fit.p_exp == pytest.approx(true_n, abs=0.05)
    a, b = fit.ab[("FJ", "fl", "dry")]
    assert 1.0 / b == pytest.approx(TRUE_TAU, rel=0.03)
    assert a / b == pytest.approx(TRUE_K, rel=0.03)
    # Without the pressure term the same data cannot be explained by one gain.
    plain = ss.fit_cells(stints)
    assert plain.cost_dry > 20 * fit.cost_dry


def test_recovers_speed_pressure_constant(tmp_path: Path) -> None:
    """A reading that drops with speed² is absorbed by κ, not by τ or K."""
    true_kappa = 8e-6
    laps = pd.concat(
        [
            _make_stint(tmp_path, "s_a", "tsukuba_2000", 12, quantise=False, kappa=true_kappa),
            _make_stint(tmp_path, "s_b", "fuji", 12, quantise=False, kappa=true_kappa),
        ],
        ignore_index=True,
    )
    stints = ss.build_stint_series(tmp_path, laps)
    fit = ss.fit_cells(stints, v_corr=True)
    assert fit.kappa["FJ"] == pytest.approx(true_kappa, rel=0.1)
    a, b = fit.ab[("FJ", "fl", "dry")]
    assert 1.0 / b == pytest.approx(TRUE_TAU, rel=0.03)
    assert a / b == pytest.approx(TRUE_K, rel=0.03)
    terms = ss.stint_speed_terms(tmp_path, laps)
    assert len(terms) == 24 and terms["speed_end_ms"].between(14.0, 41.0).all()
    assert terms["speed_anchor_fl_ms"].notna().all()


def test_recovers_heat_input_lag(tmp_path: Path) -> None:
    """Gas heated through a tread-to-gas lag: the lag's time constant is
    recovered per car and tau / K are not biased by it."""
    true_lag = 90.0
    laps = pd.concat(
        [
            _make_stint(tmp_path, "s_a", "tsukuba_2000", 12, quantise=False, lag_s=true_lag),
            _make_stint(tmp_path, "s_b", "fuji", 12, quantise=False, lag_s=true_lag),
        ],
        ignore_index=True,
    )
    stints = ss.build_stint_series(tmp_path, laps)
    fit = ss.fit_cells(stints, lag="car")
    assert fit.lag["FJ"] == pytest.approx(true_lag, rel=0.15)
    for c in CORNERS:
        a, b = fit.ab[("FJ", c, "dry")]
        assert 1.0 / b == pytest.approx(TRUE_TAU, rel=0.05)
        assert a / b == pytest.approx(TRUE_K, rel=0.05)
    # the same data fitted without the lag is biased (the lag is not a free lunch)
    plain = ss.fit_cells(stints)
    a0, b0 = plain.ab[("FJ", "fl", "dry")]
    assert not plain.lag
    assert abs(1.0 / b0 - TRUE_TAU) > abs(1.0 / fit.ab[("FJ", "fl", "dry")][1] - TRUE_TAU)


def test_lowpass_heat_is_the_exact_first_order_response() -> None:
    q = np.r_[np.zeros(5), np.ones(200)]
    y = ss.lowpass_heat(q, 50.0)
    assert y[:5] == pytest.approx(0.0)
    # step response from zero: 1 − e^{−k/τ} after k seconds of input
    assert y[5 + 50 - 1] == pytest.approx(1 - np.exp(-50 / 50.0), rel=0.02)
    assert ss.lowpass_heat(q, 0.0) is q  # no lag → untouched
    y2 = ss.lowpass_heat(np.repeat(q[:, None], 4, axis=1), 50.0)
    assert y2.shape == (205, 4) and y2[:, 2] == pytest.approx(y)


def test_anchor_nuisance_absorbs_a_wrong_anchor_reading(tmp_path: Path) -> None:
    """A pit-exit anchor pressure that is half a TPMS quantum high refers the
    whole gas-temperature observable low; without the nuisance the fit bends
    tau / K to follow it, with it the offset lands on δ and the constants
    come back."""
    err_bar = 0.015
    laps = pd.concat(
        [
            _make_stint(
                tmp_path, "s_a", "tsukuba_2000", 12, quantise=False, anchor_error_bar=err_bar
            ),
            _make_stint(tmp_path, "s_b", "fuji", 12, quantise=False, anchor_error_bar=err_bar),
        ],
        ignore_index=True,
    )
    stints = ss.build_stint_series(tmp_path, laps)
    plain = ss.fit_cells(stints)
    fit = ss.fit_cells(stints, anchor_nuisance=True)
    assert len(fit.anchor_delta) == 8  # 2 stints × 4 corners
    # T_gas ∝ 1 / P_anchor: the offset that undoes the wrong reading is
    # log((P_true + P_atm) / (P_wrong + P_atm)) ≈ −err / P_abs
    p_abs = P_START + P_ATM_BAR
    expected = float(np.log(p_abs / (p_abs + err_bar)))
    for (sid, stint, c), d in fit.anchor_delta.items():
        assert sid in ("s_a", "s_b") and stint == 1 and c in CORNERS
        assert d == pytest.approx(expected, abs=0.25 * abs(expected))
    for c in CORNERS:
        a, b = fit.ab[("FJ", c, "dry")]
        assert 1.0 / b == pytest.approx(TRUE_TAU, rel=0.03)
        assert a / b == pytest.approx(TRUE_K, rel=0.03)
    a0, b0 = plain.ab[("FJ", "fl", "dry")]
    assert abs(a0 / b0 - TRUE_K) > abs(
        fit.ab[("FJ", "fl", "dry")][0] / fit.ab[("FJ", "fl", "dry")][1] - TRUE_K
    )
