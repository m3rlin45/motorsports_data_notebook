"""Tests for gate D2, the wet (#206)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import car_gate as cg
from motorsports_data_notebook.tire_model.lapsim import samples as smp
from motorsports_data_notebook.tire_model.lapsim import solver as sv
from motorsports_data_notebook.tire_model.lapsim import wet
from motorsports_data_notebook.tire_model.lapsim import wet_gate as wg

from .test_car_gate import CAR, _model, _oval_track


def _drawn(track):
    """``track`` with a reference line drawn from its curvature."""
    heading = np.concatenate([[0.0], np.cumsum(track.kappa_heat[:-1] * track.ds_m)])
    x = np.concatenate([[0.0], np.cumsum(np.cos(heading[:-1]) * track.ds_m)])
    y = np.concatenate([[0.0], np.cumsum(-np.sin(heading[:-1]) * track.ds_m)])
    return replace(track, x_m=x, y_m=y)


class TestWetLine:
    def test_the_wet_tracks_curvature_at_the_dry_tracks_nodes(self):
        dry = _drawn(_oval_track())
        tighter = replace(dry, kappa_heat=dry.kappa_heat * 1.06)  # the same road, read tighter
        np.testing.assert_allclose(wg.wet_line(dry, tighter), dry.kappa_heat * 1.06, atol=1e-9)

    def test_a_wet_track_on_other_nodes(self):
        dry = _drawn(_oval_track())
        keep = np.arange(0, dry.n, 2)  # every other node: 4 m apart
        coarse = replace(
            dry,
            ds_m=2 * dry.ds_m,
            kappa_heat=dry.kappa_heat[keep] * 1.06,
            x_m=dry.x_m[keep],
            y_m=dry.y_m[keep],
            alt_m=dry.alt_m[keep],
            grade=dry.grade[keep],
            n_laps=dry.n_laps[keep],
            kappa_se=dry.kappa_se[keep],
        )
        got = wg.wet_line(dry, coarse)
        assert got.shape == (dry.n,)
        steady = np.abs(np.gradient(dry.kappa_heat)) < 1e-9
        np.testing.assert_allclose(got[steady], dry.kappa_heat[steady] * 1.06, atol=2e-4)


class TestCounts:
    def test_the_count_a_coin_rarely_reaches(self):
        assert wg.sign_count(13) == 10 and wg.sign_count(9) == 8
        assert wg.sign_count(3) == 4  # three days cannot show it
        assert wg.sign_count(0) == 1

    def test_car_days_against_the_null(self):
        laps = pd.DataFrame(
            {
                "car": "car",
                "layout_id": "a",
                "date": ["d1", "d1", "d2", "d1", "d1", "d2"],
                "arm": ["A", "A", "A", "B", "B", "B"],
                "S1": 1.0,
                "S2": [0.5, -0.7, 0.2, 0.1, -0.3, 0.4],
                "S3": 0.0,
                "brake_g": [0.20, 0.30, 0.10, 0.15, 0.25, 0.12],
            }
        )
        days = wg.by_car_day(laps)
        assert len(days) == 4
        nearer, of, table = wg.against_null(days, "B", "brake_g")
        assert (nearer, of) == (1, 2) and list(table["nearer"]) == [True, False]
        assert wg.against_null(days, "B", "S2")[0] == 1  # sizes, not signs: 0.6 → 0.2; 0.2 → 0.4
        assert (
            wg.against_null(days, "C", "S3") == (0, 0) or wg.against_null(days, "C", "S3")[1] == 0
        )


def _wet_cell(track, logged_car, uses=(0.80, 0.81, 0.79, 0.80)):
    """Tables and rows of four wet laps on two days, driven by ``logged_car``
    in the wet."""
    keys, traces, rows = [], [], []
    for k, u in enumerate(uses):
        sim = sv.simulate_lap(track, logged_car, u, wet=True)
        key = {
            "session_id": f"s{k // 2}",
            "car": "car",
            "raw_car": "car",
            "date": f"2000-01-0{k // 2 + 1}",
            "layout_id": "oval",
            "basis": "wet",
            "lap_num": k,
        }
        keys.append({**key, "row": k})
        rows.append({**key, "lap_time_s": sim.lap_time_s})
        traces.append(sim.v_node_ms)
    none = pd.DataFrame()
    tables = smp.CarTables(
        sessions=none,
        throttle=none,
        coast=none,
        brake=none,
        apex=none,
        speed_laps=pd.DataFrame(keys),
        speed={"oval": np.vstack(traces)},
    )
    return tables, pd.DataFrame(rows)


class TestArms:
    def test_the_null_gives_a_lower_use_lap_back(self):
        track = _drawn(_oval_track())
        tables, rows = _wet_cell(track, replace(CAR, wet_scale=0.85))
        drive = cg.Drive(_model())
        laps, yards, terms = wg.run_wet(tables, rows, {"oval": track}, {}, "car", drive)
        assert set(laps["arm"]) == {"A"}  # no wet passes to read a term from, no wet track
        assert laps["S1"].max() < 2e-3 and laps["brake_g"].max() < 2e-3
        assert len(yards) and {"S1_same_day", "S1_other_day"} <= set(yards.columns)
        assert terms["brake_term"].isna().all() and len(terms) == 2

    def test_a_planted_wet_braking_term_is_found_by_its_arm(self, monkeypatch):
        track = _drawn(_oval_track())
        tables, rows = _wet_cell(track, replace(CAR, wet_scale=0.85, wet_brake=1.12))
        monkeypatch.setattr(
            wet, "wet_terms", lambda *a, **k: wet.WetTerms(cornering=0.85, braking=0.85 * 1.12)
        )
        tighter = replace(track, kappa_heat=track.kappa_heat * 1.05)
        laps, _, _ = wg.run_wet(
            tables, rows, {"oval": track}, {"oval": tighter}, "car", cg.Drive(_model())
        )
        assert set(laps["arm"]) == set(wg.ARMS)
        days = wg.by_car_day(laps)
        nearer, of, _ = wg.against_null(days, "B", "brake_g")
        assert (nearer, of) == (2, 2)
        by_arm = laps.groupby("arm")[["S1", "brake_g"]].median()
        assert by_arm.loc["B", "brake_g"] < 2e-3 < by_arm.loc["A", "brake_g"]
        # the laps were driven on the dry line with the term: a tighter line on top is further
        assert by_arm.loc["D", "S1"] > by_arm.loc["B", "S1"] + 0.05
        assert by_arm.loc["B", "S1"] < 2e-3
        got = wg._same_day(wg.ch.by_pace(laps[laps["arm"] == "B"], _keyed(laps)))
        assert {"passed_same", "scored_same", "passed"} <= set(got.columns)


def _keyed(laps):
    """Yardsticks that every lap meets on every measure."""
    own = laps[laps["arm"] == "A"][["car", "layout_id", "session_id", "lap_num", "date"]]
    out = own.assign(level=75.0)
    for k in wg.ch.YARDSTICK_MEASURES:
        out[f"{k}_other_day"] = 10.0
        out[f"{k}_same_day"] = 10.0
    return out
