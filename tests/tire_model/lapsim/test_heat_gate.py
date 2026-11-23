"""Gate D1's clauses on logged laps: the chain against production's own (#206)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model import statespace as ss
from motorsports_data_notebook.tire_model.heat_input import PHYSICAL_HEAT_INPUT, ShareParams
from motorsports_data_notebook.tire_model.lapsim import heat, heat_gate

SHARE = ShareParams(
    p_f=0.45,
    g_transfer_front=2.5,
    g_transfer_rear=2.5,
    beta0=math.log(0.6 / 0.4),
    beta1=0.0,
    eps_drive=1.0,
    eps_brake=1.0,
)


def _kin(laps=(3, 4), seconds=30.0, hz=50.0, discard=None, gps=False):
    """A stint's kinematics: each lap ``seconds`` long, cornering both ways,
    driving and braking, the speed moving with them."""
    t = np.arange(0.0, seconds * len(laps), 1.0 / hz) + 500.0
    lap = np.repeat(np.asarray(laps), int(seconds * hz))
    src = np.full(t.size, "gps_channel" if gps else "imu", dtype=object)
    if discard is not None:
        src[(t >= discard[0]) & (t < discard[1])] = "discard"
    return pd.DataFrame(
        {
            "t_session_s": t,
            "lap_num": lap,
            "speed_ms": 35.0 + 10.0 * np.sin(t / 5.0),
            "lat_g": 1.1 * np.sin(t / 3.0),
            "long_g": 0.6 * np.cos(t / 2.0) - 0.2,
            "lat_source": src,
        }
    )


class TestAgainstProductionsBins:
    def test_a_stints_bins_are_the_chains(self):
        kin = _kin(laps=(3,))
        series = ss.SessionSeries(kin=kin, tpms=None, surf=None)
        theirs = ss._bin_stint(series, [3], PHYSICAL_HEAT_INPUT, "car")
        t = kin["t_session_s"].to_numpy()
        ours = heat.bin_1s(
            t, kin["speed_ms"], kin["lat_g"], kin["long_g"], float(t[0]), float(t[0]) + 30.0
        )
        for name in ("lat_pos", "lat_neg", "long_pos", "long_neg"):
            assert getattr(ours, name) == pytest.approx(theirs[name], abs=1e-6)
        assert ours.v_ms == pytest.approx(theirs["v"], abs=1e-6)
        assert theirs["valid"].all()

    def test_the_chain_on_the_laps_production_fits(self, monkeypatch):
        kin = _kin(discard=(545.0, 548.0))
        monkeypatch.setattr(
            ss, "_session_timeseries", lambda root, sid: ss.SessionSeries(kin, None, None)
        )
        laps = pd.DataFrame({"session_id": "s", "lap_num": [3, 4], "car": "car"})
        got = heat_gate.chain_lap_heat("root", laps, {"car": SHARE}).set_index("lap_num")
        first = kin[kin["lap_num"] == 3]
        t = first["t_session_s"].to_numpy()
        want = heat.series_heat(
            t, first["speed_ms"], first["lat_g"], first["long_g"], SHARE, t[0], t[-1] + 0.02
        )
        assert got.loc[3, [f"q_{c}" for c in heat.CORNERS]].to_numpy(float) == pytest.approx(want)
        assert not got.loc[3, "hole"] and got.loc[3, "source"] == "imu"
        # half a second of phase and half a second of smoothing each move it a little
        assert got.loc[3, "phase_fl"] == pytest.approx(got.loc[3, "q_fl"], rel=0.05)
        assert got.loc[3, "smooth_fl"] < got.loc[3, "q_fl"]
        # the lap with a discarded window has no heat: an integral with a hole
        assert got.loc[4, "hole"]
        assert np.isnan(got.loc[4, [f"q_{c}" for c in heat.CORNERS]].to_numpy(float)).all()

    def test_a_car_without_shares_and_a_gps_lap(self, monkeypatch):
        kin = _kin(gps=True)
        monkeypatch.setattr(
            ss, "_session_timeseries", lambda root, sid: ss.SessionSeries(kin, None, None)
        )
        laps = pd.DataFrame({"session_id": "s", "lap_num": [3, 4], "car": "car"})
        assert heat_gate.chain_lap_heat("root", laps, {}).empty
        got = heat_gate.chain_lap_heat("root", laps, {"car": SHARE})
        assert set(got["source"]) == {"gps"}
