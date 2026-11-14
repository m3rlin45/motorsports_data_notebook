"""One track for every car? (lap simulation B3)"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.heat_input import ShareParams
from motorsports_data_notebook.tire_model.lapsim import geometry as geo
from motorsports_data_notebook.tire_model.lapsim import holdout as ho
from motorsports_data_notebook.tire_model.lapsim import track as trk

from ._synthetic import LAYOUT_ID, write_dataset

SHARE = ShareParams(
    p_f=0.45,
    g_transfer_front=2.5,
    g_transfer_rear=2.5,
    beta0=math.log(0.6 / 0.4),
    beta1=0.0,
    eps_drive=1.0,
    eps_brake=1.0,
)


def _oval(n=400, ds=2.0):
    """A stadium's κ (right turns) and a speed trace slow in the corners."""
    s = np.arange(n) * ds
    kappa = np.where(((s > 100) & (s < 250)) | ((s > 500) & (s < 650)), 1 / 48.0, 0.0)
    speed = 30.0 - 10.0 * np.exp(-(((s - 175) / 80) ** 2)) - 10.0 * np.exp(-(((s - 575) / 80) ** 2))
    return kappa, speed, ds


def _write_model_json(root, cars=("car",)):
    model = {
        "heat_input": {
            "fitted_by_car": {c: {"g_transfer": 2.5} for c in cars},
            "car_facts_by_car": {c: {"p_f": 0.45, "brake_bias_front": 0.6} for c in cars},
        },
        "car_aliases": {"alias": cars[0]},
    }
    (root / "tire_model.json").write_text(json.dumps(model))


class TestHeatChain:
    def test_right_turns_load_the_left_tyres(self):
        kappa, speed, ds = _oval()
        fl, fr, rl, rr = ho.tyre_heat(SHARE, kappa, speed, ds)
        assert fl > fr and rl > rr
        mirrored = ho.tyre_heat(SHARE, -kappa, speed, ds)
        assert mirrored == pytest.approx([fr, fl, rr, rl])

    def test_longitudinal_g_is_the_speed_traces_own(self):
        ds = 2.0
        s = np.arange(500) * ds
        speed = 30.0 + 5.0 * np.sin(2 * math.pi * s / 1000.0)
        want = speed * 5.0 * (2 * math.pi / 1000.0) * np.cos(2 * math.pi * s / 1000.0) / 9.80665
        assert ho.longitudinal_g(speed, ds) == pytest.approx(want, abs=2e-4)

    def test_identical_profiles_do_not_differ(self):
        kappa, speed, ds = _oval()
        res = ho.compare_profiles(kappa, kappa, speed, ds, SHARE)
        assert res["dq_max"] == 0.0 and res["v2dk_rms_g"] == 0.0
        assert res["right"] == 0.0 and res["low"] == 0.0

    def test_a_curvature_scale_goes_into_heat(self):
        kappa, speed, ds = _oval()
        res = ho.compare_profiles(1.05 * kappa, kappa, speed, ds, SHARE)
        assert res["right"] == pytest.approx(math.log(1.05))
        # per tyre: lateral force and its load transfer both grow, braking does not
        assert 0.02 < res["dq_fl"] < 0.07
        assert res["dq_max"] == max(abs(res[f"dq_{c}"]) for c in ho.CORNERS)

    def test_low_curvature_share(self):
        kappa, speed, ds = _oval()
        gentle = kappa.copy()
        gentle[300:340] = 0.002
        res = ho.compare_profiles(gentle, kappa, speed, ds, SHARE)
        assert res["low"] > 0 and res["left"] == 0.0


class TestShares:
    def test_production_shares_and_stand_ins(self, tmp_path):
        _write_model_json(tmp_path, cars=("FJ", "Inferno 86"))
        shares = ho.production_shares(tmp_path)
        assert shares["FJ"].g_transfer_front == 2.5 and shares["FJ"].p_f == 0.45
        assert shares["alias"] is shares["FJ"]
        assert ho.share_for("SFJ", shares) is shares["FJ"]
        assert ho.share_for("EK9", shares) is shares["Inferno 86"]
        with pytest.raises(KeyError):
            ho.share_for("unknown car", shares)


def _verdict_frame(between, floor=0.006, vote=True):
    rows = [{"rung": 2, "dq_max": v, "vote": False} for v in np.linspace(0.0, floor / 0.9, 11)]
    rows += [{"rung": 1, "dq_max": 0.004, "vote": False}]
    rows += [{"rung": 5, "dq_max": between, "vote": vote}]
    return pd.DataFrame(rows)


class TestVerdict:
    @pytest.mark.parametrize(
        ("between", "want"),
        [
            (0.004, "one track (within the day-to-day floor)"),
            (0.015, "one track, bias carried to E1"),
            (0.03, "per-class tracks"),
            (0.06, "above the stop bound"),
        ],
    )
    def test_graded_rule(self, between, want):
        out = ho.layout_verdict(_verdict_frame(between))
        assert out["verdict"] == want
        assert out["floor_rung"] == 2 and out["floor_p90"] == pytest.approx(0.006)
        assert out["between_cars_max"] == between

    def test_no_vote(self):
        out = ho.layout_verdict(_verdict_frame(0.03, vote=False))
        assert out["verdict"] == "no vote" and out["voting"] == 0

    def test_floor_falls_back_to_rung_one(self):
        frame = _verdict_frame(0.003)
        out = ho.layout_verdict(frame[frame["rung"] != 2])
        assert out["floor_rung"] == 1 and out["floor_p90"] == pytest.approx(0.004)


def _two_car_dataset(tmp_path):
    geo._INDEX_CACHE.clear()
    sessions = {}
    for c, car in enumerate(("KK-F", "KK-SII")):
        for d in range(4):
            sessions[f"2026-01-1{d}_{car}"] = {
                "car": car,
                "date": f"2026-01-1{d}",
                "kwargs": {"n_laps": 9, "seed": 10 * c + d, "noise_m": 0.02},
            }
    write_dataset(tmp_path, sessions)
    _write_model_json(tmp_path, cars=("FJ",))
    model = json.loads((tmp_path / "tire_model.json").read_text())
    model["car_aliases"] = {"KK-F": "FJ", "KK-SII": "FJ"}
    (tmp_path / "tire_model.json").write_text(json.dumps(model))
    return tmp_path


class TestLadder:
    def test_rungs_votes_and_the_floor_on_one_track(self, tmp_path):
        root = _two_car_dataset(tmp_path)
        build = trk.build_track_model(root, LAYOUT_ID, inertial=False, n_boot=0)
        lad = ho.ladder(build, ho.production_shares(root), n_repeat=4)
        assert set(lad["rung"]) == {1, 2, 3, 4}  # one class: no rung 5
        assert (lad[lad["rung"] == 1].groupby("held_out").size() == 4).all()
        r4 = lad[lad["rung"] == 4].set_index("held_out")
        assert set(r4.index) == {"KK-F", "KK-SII"}
        assert r4.loc["KK-F", "donor"] == "KK-SII"
        assert r4["vote"].all() and not lad[lad["rung"] < 4]["vote"].any()
        assert (r4["days_held"] == 4).all() and (r4["laps_donor"] == 36).all()
        # the two cars drove the same circuit: every rung is noise
        assert lad["dq_max"].max() < 0.01
        verdict = ho.layout_verdict(lad)
        assert verdict["voting"] == 2 and verdict["verdict"].startswith("one track")

    def test_gate_and_saved_folds(self, tmp_path):
        root = _two_car_dataset(tmp_path / "data")
        report = ho.check_track_holdout(
            root, folds_dir=tmp_path / "folds", layouts=(LAYOUT_ID,), n_repeat=3
        )
        items = {i.name: i for i in report.items}
        assert items["Separability"].passed is True
        assert "one track" in items["Separability"].measured
        assert "Own reference lines" in items and "Ladder" in items
        saved = sorted(p.name for p in (tmp_path / "folds").glob("*.json"))
        assert f"{LAYOUT_ID}__without_KK-F__rung4.json" in saved
        assert f"{LAYOUT_ID}__without_KK-F__rung4__speed.json" in saved
        donor = trk.TrackModel.load(tmp_path / "folds" / f"{LAYOUT_ID}__without_KK-F__rung4.json")
        assert donor.meta["built_from"]["cars"] == ["KK-SII"]

    def test_per_lap_metric_is_near_zero_on_one_track(self, tmp_path):
        root = _two_car_dataset(tmp_path)
        build = trk.build_track_model(root, LAYOUT_ID, inertial=False, n_boot=0)
        lm = ho.per_lap_metric(build).set_index("car")
        assert set(lm.index) == {"KK-F", "KK-SII"}
        assert (lm["days"] == 4).all() and (lm["laps"] == 36).all()
        # a lap is noisier than a profile, so the ratio sits a little above
        # zero; it is the same against either profile
        assert lm["own_other_days"].abs().max() < 0.03
        assert lm["difference"].abs().max() < 0.005

    def test_lap_metric_reads_a_curvature_scale(self):
        kappa, speed, ds = _oval()
        laps = np.vstack([kappa, 1.1 * kappa])
        got = ho._lap_metric(laps, np.vstack([speed, speed]), kappa, ds)
        assert got[0] == pytest.approx(0.0, abs=1e-12)
        assert got[1] == pytest.approx(math.log(1.1), abs=1e-9)

    def test_strict_comparison_agrees_with_the_shared_projection(self, tmp_path):
        root = _two_car_dataset(tmp_path)
        build = trk.build_track_model(root, LAYOUT_ID, inertial=False, n_boot=0)
        a = [lap for lap in build.laps if lap.car == "KK-F"]
        b = [lap for lap in build.laps if lap.car == "KK-SII"]
        res, donor, speed = ho.strict_comparison(a, b, LAYOUT_ID, SHARE)
        assert res["dq_max"] < 0.01
        assert donor.meta["built_from"]["cars"] == ["KK-F"]
        assert len(speed) > 300 and speed.min() > 15.0
