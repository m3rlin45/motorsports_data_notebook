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


class TestFrameFactor:
    def test_the_per_tyre_median_per_car_layout_and_source(self):
        rng = np.random.default_rng(1)
        rows = []
        for session, source, lift in (
            ("s1", "imu", 1.04),
            ("s2", "imu", 1.06),
            ("s3", "gps", 1.10),
        ):
            for lap in range(20):
                row = {"car": "FJ", "layout_id": "a", "source": source, "session_id": session}
                for k, c in enumerate(heat_gate.CORNERS):
                    base = 100.0 + 10.0 * k
                    row[f"t_{c}"] = base
                    row[f"q_{c}"] = base * lift * np.exp(rng.normal(0.0, 0.002))
                rows.append({**row, "lap_num": lap})
        factors, left = heat_gate.frame_factor(pd.DataFrame(rows))
        by = factors.set_index("source")
        assert by.loc["imu", "laps"] == 40 and by.loc["imu", "sessions"] == 2
        assert by.loc["imu", "f_fl"] == pytest.approx(1.05, abs=0.011)
        assert by.loc["gps", "f_rr"] == pytest.approx(1.10, abs=0.003)
        # the two accelerometer sessions sit a per cent either side of their cell
        imu = left[left["source"] == "imu"].iloc[0]
        assert imu["sessions"] == 2 and imu["median"] == pytest.approx(0.0095, abs=0.002)
        assert left[left["source"] == "gps"].iloc[0]["median"] < 0.002

    def test_laps_without_a_heat_on_either_side_are_left_out(self):
        row = {"car": "FJ", "layout_id": "a", "source": "imu", "session_id": "s", "lap_num": 1}
        for c in heat_gate.CORNERS:
            row[f"q_{c}"], row[f"t_{c}"] = np.nan, 100.0
        factors, left = heat_gate.frame_factor(pd.DataFrame([row]))
        assert factors.empty and left.empty


class TestBinsBeforeTheShares:
    """A lap is read and binned once; any fit's shares split the bins (#208)."""

    def test_the_sensor_bins_split_to_the_chains_heat(self, monkeypatch):
        kin = _kin(discard=(545.0, 548.0))
        monkeypatch.setattr(
            ss, "_session_timeseries", lambda root, sid: ss.SessionSeries(kin, None, None)
        )
        laps = pd.DataFrame({"session_id": "s", "lap_num": [3, 4], "car": "car"})
        bins = heat_gate.sensor_lap_bins("root", laps)
        assert set(bins) == {("s", 3), ("s", 4)}
        other = ShareParams(
            p_f=0.55,
            g_transfer_front=1.2,
            g_transfer_rear=1.2,
            beta0=math.log(0.7 / 0.3),
            beta1=0.0,
            eps_drive=1.0,
            eps_brake=1.0,
        )
        for share in (SHARE, other):
            want = heat_gate.chain_lap_heat("root", laps, {"car": share}).set_index("lap_num")
            for lap_num in (3, 4):
                own, source, hole = bins[("s", lap_num)]
                got = heat.lap_heat(own, share)
                np.testing.assert_array_equal(
                    got, want.loc[lap_num, [f"q_{c}" for c in heat.CORNERS]].to_numpy(float)
                )
                assert source == want.loc[lap_num, "source"] and hole == want.loc[lap_num, "hole"]
        # the two fits do not split the lap alike: the bins are not a heat
        a = heat.lap_heat(bins[("s", 3)][0], SHARE)
        b = heat.lap_heat(bins[("s", 3)][0], other)
        assert np.abs(np.log(a / b)).max() > 0.02

    def test_the_bins_need_no_shares_for_the_car(self, monkeypatch):
        kin = _kin(gps=True)
        monkeypatch.setattr(
            ss, "_session_timeseries", lambda root, sid: ss.SessionSeries(kin, None, None)
        )
        laps = pd.DataFrame({"session_id": "s", "lap_num": [3, 4], "car": "a car with no fit"})
        bins = heat_gate.sensor_lap_bins("root", laps)
        assert {source for _, source, _ in bins.values()} == {"gps"}
