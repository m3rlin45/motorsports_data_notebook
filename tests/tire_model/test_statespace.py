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
from tests._timeseries_fixture import write_session_timeseries

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
    discard_windows: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Write one session's timeseries parquet and return its lap rows.

    ``discard_windows`` are 20 s windows of the session the sensor verdict
    discards; the logged accelerations there are garbage (4 g), so a fit
    that read them would be thrown far off.

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
    windows = ""
    if discard_windows:
        n_win = int(np.ceil(n_s / 20.0))
        windows = "".join("D" if i in discard_windows else "I" for i in range(n_win))
        w = np.floor(ts["t_session_s"].to_numpy() / 20.0).astype(int)
        ts.loc[np.isin(w, discard_windows), "lat_g"] = 4.0
    write_session_timeseries(root, sid, ts, windows=windows)
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
        laps[f"p_start_{c}"] = float(p_true[i0])  # first live reading (at its speed)
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


def test_segment_scored_reanchors_after_each_hole() -> None:
    n = 400
    scored = np.zeros(n, dtype=bool)
    scored[20:] = True
    valid = np.ones(n, dtype=bool)
    valid[100:120] = False  # a 20 s hole
    valid[300:320] = False  # another, leaving 320..399 (80 s) after it
    scored &= valid
    out, anchor = ss._segment_scored(scored, valid)
    assert (anchor[:100] == -1).all()  # the stretch before the first hole: pit-exit anchor
    assert (anchor[120:300] == 120).all() and (anchor[320:] == 320).all()
    assert out[120:300].all() and out[320:].all() and not out[100:120].any()
    # a stretch too short to score is dropped
    valid2 = valid.copy()
    valid2[150:160] = False  # splits 120..299 into 30 s and 140 s
    out2, anchor2 = ss._segment_scored(scored & valid2, valid2)
    assert not out2[120:150].any() and (anchor2[160:300] == 160).all()


def test_fit_trains_around_a_discarded_window(tmp_path: Path) -> None:
    """A window the sensor verdict discarded leaves a hole in the heat input:
    its seconds are not scored, the simulation restarts from the measured
    state after it, and the fit still recovers the constants."""
    laps = pd.concat(
        [
            _make_stint(
                tmp_path, "s_tsukuba", "tsukuba_2000", 12, quantise=False, discard_windows=(9,)
            ),
            _make_stint(tmp_path, "s_fuji", "fuji", 12, quantise=False, discard_windows=(5, 20)),
        ],
        ignore_index=True,
    )
    stints = ss.build_stint_series(tmp_path, laps)
    s = stints[0]
    assert s.valid is not None and s.seg_anchor is not None
    assert not s.valid[180:200].any() and s.valid[:180].all() and s.valid[200:].all()
    assert np.isnan(s.obs[180:200, 0]).all()  # the hole is not scored
    assert (s.seg_anchor[:180, 0] == -1).all() and (s.seg_anchor[200:, 0] == 200).all()
    assert np.nanmax(s.g2) < 3.0  # the 4 g garbage never reaches the heat input
    fit = ss.fit_cells(stints)
    for c in CORNERS:
        a, b = fit.ab[("FJ", c, "dry")]
        assert 1.0 / b == pytest.approx(TRUE_TAU, rel=0.03)
        assert a / b == pytest.approx(TRUE_K, rel=0.03)
    # the laps holding a hole have no per-lap heat row
    comp = ss.lap_heat_components(stints)
    holed = comp[comp["session_id"] == "s_tsukuba"]["lap_num"].unique()
    assert 180 // LAP_S not in holed and 0 in holed
