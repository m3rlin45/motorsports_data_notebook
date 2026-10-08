"""Per-second (1 Hz) fit: parameter recovery on synthetic stints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model import statespace as ss
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
) -> pd.DataFrame:
    """Write one session's timeseries parquet and return its lap rows.

    ``p_exp`` > 0 generates the data with the pressure-dependent heat input
    ``q · (P_ref / P(t))^n`` where ``P(t)`` is the gas-law pressure."""
    n_s = n_laps * LAP_S
    t_1hz = np.arange(n_s)
    g2 = 0.4 + 0.8 * np.sin(2 * np.pi * t_1hz / LAP_S) ** 2  # constant within each second
    a, b = TRUE_K / TRUE_TAU, 1.0 / TRUE_TAU
    T = np.empty(n_s + 1)
    T[0] = T_START
    for i in range(n_s):
        p_abs = (p_start + P_ATM_BAR) * (T[i] + T_ZERO_C_TO_K) / (T_START + T_ZERO_C_TO_K)
        pf = (ss.P_REF_ABS_BAR / p_abs) ** p_exp
        teq = T_EFF + a * g2[i] * pf / b
        T[i + 1] = teq + (T[i] - teq) * np.exp(-b)
    # gas-law pressure from the anchor (T_START, p_start), sampled at 5 Hz;
    # read at speed it sits below the gas-law value by (1 + κ v²)
    # speed varies on a different period from g² so κ is separable from K/τ
    v_1hz = 15.0 + 25.0 * np.cos(2 * np.pi * t_1hz / (0.37 * LAP_S)) ** 2
    t_k = T[:-1] + T_ZERO_C_TO_K
    p = (p_start + P_ATM_BAR) * t_k / (T_START + T_ZERO_C_TO_K) / (1 + kappa * v_1hz**2) - P_ATM_BAR
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
    for c in CORNERS:
        laps[f"t_anchor_{c}"] = 0.0
        laps[f"t_start_{c}"] = T_START
        laps[f"p_start_{c}"] = float(p[0])  # the anchor is the first reading (at its speed)
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
