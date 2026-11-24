"""Tests for gate F's folds and scores (#208)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import car_gate as cg
from motorsports_data_notebook.tire_model.lapsim import pace
from motorsports_data_notebook.tire_model.lapsim import pressure as pr
from motorsports_data_notebook.tire_model.lapsim import pressure_gate as pg

from .test_car_gate import _model, _oval_track
from .test_holdout import SHARE
from .test_pace import _ranged
from .test_pressure import TYRES, _fold_model, _laps, _prep


def _production(tmp_path, sessions, trained=None) -> pg.Production:
    frame = pd.DataFrame(sessions)
    frame["pooled"] = frame["car"].replace({"KK-F": "FJ", "KK-SII": "FJ"})
    return pg.Production(
        root=tmp_path,
        sessions=frame,
        raw_laps=pd.DataFrame(),
        trained=set(frame["session_id"]) if trained is None else set(trained),
        conditions={s: c for s, c in zip(frame["session_id"], frame["condition"])},
        targets={},
    )


def _session(sid, car, track, date, condition="dry", status="ok"):
    return {
        "session_id": sid,
        "car": car,
        "track_canonical": track,
        "date": date,
        "condition": condition,
        "status": status,
    }


class TestFolds:
    def test_a_fold_leaves_out_every_session_of_the_pooled_car_at_the_track(self, tmp_path):
        prod = _production(
            tmp_path,
            [
                _session("a", "KK-F", "tsukuba", "d1"),
                _session("b", "KK-SII", "tsukuba", "d1", condition="wet"),
                _session("c", "KK-SII", "tsukuba", "d2", status="error"),
                _session("d", "KK-SII", "suzuka", "d3"),
                _session("e", "Inferno 86", "tsukuba", "d1"),
            ],
        )
        # the sibling, the wet session and the one that cannot be scored all go
        assert pg.cell_sessions(prod, "FJ", "tsukuba") == ["a", "b", "c"]
        assert pg.cell_sessions(prod, "Inferno 86", "tsukuba") == ["e"]

    def test_the_cells_per_raw_car(self, tmp_path):
        prod = _production(
            tmp_path,
            [
                _session("a", "KK-F", "tsukuba", "d1"),
                _session("b", "KK-F", "tsukuba", "d1"),
                _session("c", "KK-F", "tsukuba", "d2", condition="wet"),
                _session("d", "KK-SII", "tsukuba", "d3"),
                _session("e", "KK-SII", "tsukuba", "d4"),
            ],
            trained=["a", "b", "c", "d"],
        )
        got = pg.cell_table(prod).set_index("raw_car")
        assert got.loc["KK-F", "left_out"] == 5 and got.loc["KK-SII", "left_out"] == 5
        assert got.loc["KK-F", "trained"] == 3 and got.loc["KK-F", "dry_sessions"] == 2
        assert got.loc["KK-F", "dry_days"] == 1 and got.loc["KK-F", "wet_damp_sessions"] == 1
        assert got.loc["KK-SII", "trained"] == 1  # "e" does not reach a fit

    def test_a_fold_is_fitted_once(self, tmp_path, monkeypatch):
        from motorsports_data_notebook.tire_model import validate

        fits: list[list[str]] = []

        def fit(root, ids):
            fits.append(sorted(ids))
            return {"value": 1.0 / 3.0, "left_out": sorted(ids)}

        monkeypatch.setattr(validate, "_fit_fold", fit)
        monkeypatch.setattr(pg, "_fit_signature", lambda root: "sig")
        first = pg.fold_model(tmp_path, ["b", "a"], tmp_path / "cache")
        again = pg.fold_model(tmp_path, ["a", "b"], tmp_path / "cache")
        assert fits == [["a", "b"]] and again == first
        assert first["value"] == 1.0 / 3.0  # the numbers come back as they were fitted
        other = pg.fold_model(tmp_path, ["a"], tmp_path / "cache")
        assert other["left_out"] == ["a"] and len(fits) == 2
        # another dataset or another fit's source is another fold
        monkeypatch.setattr(pg, "_fit_signature", lambda root: "changed")
        pg.fold_model(tmp_path, ["a", "b"], tmp_path / "cache")
        assert len(fits) == 3
        files = list((tmp_path / "cache" / "pressure_folds").glob("*.json"))
        assert len(files) == 3 and json.loads(files[0].read_text())
        # without a cache directory nothing is kept
        pg.fold_model(tmp_path, ["a", "b"], None)
        assert len(fits) == 4


class TestFedLaps:
    def _index(self):
        rows = []
        for sid, car, cond, times in (
            ("s1", "KK-F", "dry", [70.0, 61.0, 60.0, 62.0, 90.0]),  # 90 s is over 1.4 × 60 s
            ("s2", "KK-F", "wet", [70.0, 70.0, 70.0]),
            ("s3", "KK-SII", "dry", [60.0, 61.0]),  # two laps: no median to speak of
            ("s4", "KK-SII", "dry", [60.0, 61.0, 62.0, 5.0]),  # a 5 s fragment
        ):
            for k, t in enumerate(times):
                rows.append(
                    {
                        "session_id": sid,
                        "car": car,
                        "date": "2026-01-01",
                        "layout_id": "oval",
                        "condition": cond,
                        "lap_type": "full",
                        "stint_id": 0,
                        "lap_num": k + 1,
                        "lap_time_s": t,
                    }
                )
        rows.append({**rows[0], "lap_type": "out", "lap_num": 0, "lap_time_s": 61.5})
        return pd.DataFrame(rows)

    def test_a_dry_sessions_median_flying_lap(self):
        got = pg.fed_laps(self._index()).set_index("session_id")
        assert list(got.index) == ["s1", "s4"]
        assert got.loc["s1", "lap_time_s"] == 61.5 and got.loc["s1", "laps"] == 4
        assert got.loc["s4", "lap_time_s"] == 61.0 and got.loc["s4", "laps"] == 3


def _row(raw_car, date, session, stint, corner, arm, d, form="heat", pick=0, **over):
    return {
        "pooled": "FJ",
        "track": "t",
        "raw_car": raw_car,
        "date": date,
        "session_id": session,
        "stint_id": stint,
        "corner": corner,
        "n": 5,
        "form": form,
        "arm": arm,
        "pick": pick,
        "d_mbar": d,
        "gap": over.pop("gap", 0.0),
        "status": over.pop("status", pr.OK),
        "use": np.nan,
        **over,
    }


class TestScores:
    def _rows(self):
        rows = []
        # three days; another day is 4 mbar off on every one, the simulation 2, 3 and 9
        for date, sim in (("d1", 2.0), ("d2", -3.0), ("d3", 9.0)):
            for pick, off in enumerate((4.0, -6.0, 3.0)):
                rows.append(
                    _row("KK-F", date, f"s-{date}", 0, "fl", "other_day", off, pick=pick, gap=0.01)
                )
            rows.append(_row("KK-F", date, f"s-{date}", 0, "fl", "sim", sim))
        return pd.DataFrame(rows)

    def test_a_stint_is_the_median_over_an_arms_picks(self):
        got = pg.by_stint(self._rows())
        other = got[(got["arm"] == "other_day") & (got["date"] == "d1")].iloc[0]
        assert other["size"] == 4.0 and other["signed"] == 3.0 and other["picks"] == 3
        assert other["gap"] == pytest.approx(0.01)

    def test_a_cell_is_the_median_over_days_and_passes_beside_another_day(self):
        stints = pg.by_stint(self._rows())
        att = pg.attain(stints, "sim").iloc[0]
        assert att["size"] == 3.0 and att["signed"] == 2.0 and att["beside"] == 4.0
        assert att["days"] == 3 and att["scored"] and att["passed"]
        got = pg.tally(pg.attain(stints, "sim"))
        assert got == {
            "passed": 1,
            "scored": 1,
            "whole": 1,
            "groups": 1,
            "size": 3.0,
            "beside": 4.0,
        }

    def test_two_days_are_listed_and_not_scored(self):
        rows = self._rows()
        stints = pg.by_stint(rows[rows["date"] != "d3"])
        att = pg.attain(stints, "sim").iloc[0]
        assert att["days"] == 2 and not att["scored"] and not att["passed"]
        assert pg.tally(pg.attain(stints, "sim"))["scored"] == 0

    def test_a_day_without_a_pair_is_not_counted_for_either(self):
        rows = self._rows()
        lone = _row("KK-F", "d4", "s-d4", 0, "fl", "sim", 50.0)
        stints = pg.by_stint(pd.concat([rows, pd.DataFrame([lone])]))
        att = pg.attain(stints, "sim").iloc[0]
        assert att["days"] == 3 and att["size"] == 3.0  # the fourth day has no other day beside it

    def test_two_arms_are_compared_on_the_stints_both_have(self):
        """The simulation has no entry on the third day (no frame factor
        for that day's kind of source, say): the yardstick beside it is
        then of the first two days alone, not of all three."""
        rows = self._rows()
        rows.loc[rows["arm"] == "other_day", "d_mbar"] *= rows["date"].map(
            {"d1": 1.0, "d2": 1.0, "d3": 10.0}
        )
        rows = rows[~((rows["arm"] == "sim") & (rows["date"] == "d3"))]
        stints = pg.by_stint(rows)
        att = pg.attain(stints, "sim").iloc[0]
        assert att["days"] == 2 and att["beside_days"] == 2
        assert att["beside"] == 4.0  # d3's 40 mbar is not in it
        # the yardstick by itself is still of every day it has
        yard = pg.by_cell(stints)
        assert yard[yard["arm"] == "other_day"]["days"].iloc[0] == 3

    def test_a_lap_with_no_solution_is_a_miss_and_is_kept(self):
        rows = self._rows()
        rows.loc[(rows["arm"] == "sim") & (rows["date"] != "d1"), ["d_mbar", "status"]] = (
            np.nan,
            pr.NO_SOLUTION,
        )
        att = pg.attain(pg.by_stint(rows), "sim").iloc[0]
        assert np.isinf(att["size"]) and att["days"] == 3 and att["flagged"] == 2
        assert att["scored"] and not att["passed"]

    def test_the_four_tyres_of_a_cell_are_one_group(self):
        rows = self._rows()
        rear = rows.assign(corner="rl")
        rear.loc[rear["arm"] == "sim", "d_mbar"] = 20.0
        stints = pg.by_stint(pd.concat([rows, rear], ignore_index=True))
        got = pg.tally(pg.attain(stints, "sim"))
        assert got["passed"] == 1 and got["scored"] == 2
        assert got["whole"] == 0 and got["groups"] == 1

    def test_the_forms_and_the_raw_cars_are_apart(self):
        rows = self._rows()
        chain = rows.assign(form="chain", d_mbar=rows["d_mbar"] * 2.0)
        sibling = rows.assign(raw_car="KK-SII")
        stints = pg.by_stint(pd.concat([rows, chain, sibling], ignore_index=True))
        att = pg.attain(stints, "sim")
        assert len(att) == 3
        assert set(zip(att["raw_car"], att["form"], att["size"])) == {
            ("KK-F", "heat", 3.0),
            ("KK-F", "chain", 6.0),
            ("KK-SII", "heat", 3.0),
        }

    def test_nothing_scored(self):
        empty = pd.DataFrame()
        assert pg.by_stint(empty).empty and pg.by_cell(empty).empty
        assert pg.tally(empty)["scored"] == 0


# ------------------------------------------------- a fold, end to end


def _held(tmp_path, lap_times: dict[str, float]) -> pg.Held:
    """A fold of car "c" at track "new" (the fold model has it at "a" and
    "b"): one session a day, each lapping at its own pace."""
    frames = []
    for k, (date, pace_s) in enumerate(lap_times.items()):
        own = _laps(
            [pace_s + 2.0, pace_s + 1.0, pace_s + 0.5, pace_s, pace_s, pace_s, pace_s],
            session=f"s{k}",
            q=4.0 + 0.2 * k,
        )
        own["track_canonical"] = "new"
        own["t_cum_s"] = own["moving_s"].cumsum()
        for c in TYRES:
            own[f"t_gas_{c}_end"], own[f"tpms_temp_{c}_end"] = 45.0, 45.0
            own[f"t_anchor_{c}"], own[f"p_start_{c}"] = 5.0, 1.5
            own[f"tpms_press_{c}_end"] = 1.7
        own["date"], own["raw_car"], own["layout_id"] = date, "car", "oval"
        frames.append(own)
    full = pd.concat(frames, ignore_index=True)
    prep = _prep(full)
    prep.session_lap_time.update({f"s{k}": t for k, t in enumerate(lap_times.values())})
    raw = full[["session_id", "stint_id", "lap_num", "lap_time_s", "is_outlap"]]
    laps = pr.mark_laps(prep, raw)
    for name in ("date", "raw_car", "layout_id"):
        laps[name] = laps["session_id"].map(
            full.drop_duplicates("session_id").set_index("session_id")[name]
        )
    stints = pr.stints_of(prep, {("c", "DRY"): 1.8})
    entries, days = {}, []
    for sid, own in laps.groupby("session_id", sort=True):
        entries[str(sid)] = pr.day_entry(tmp_path, own, SHARE)
        days.append(
            {
                "session_id": str(sid),
                "raw_car": "car",
                "date": str(own["date"].iloc[0]),
                "layout_id": "oval",
                "powertrain": "engine",
                "source": "imu",
                "lap_time_s": entries[str(sid)].lap_time_s,
                "ordinary_laps": entries[str(sid)].n_laps,
            }
        )
    return pg.Held(
        tmp_path,
        "c",
        "new",
        _fold_model(),
        prep,
        SHARE,
        laps,
        stints,
        pd.DataFrame(days),
        entries,
        3,
    )


def _cell(lap_time_s: float) -> pg.SimCell:
    """The simulated side at "oval": a car whose fed use laps it in about
    ``lap_time_s``, and the same car with less power as the planted one."""
    track = _oval_track()
    track.meta["pit"] = {"s_join_m": 400.0, "v_join_ms": 15.0, "rolling_s": 20.0, "n": 12}
    model = _ranged(_model(), oval=(0.5, 1.0))
    table = pace.build_table(model, track)
    use = float(table.use_at(lap_time_s))
    drives = {"sim": cg.Drive(model), "low_power": cg.Drive(model), "other_car": None}
    return pg.SimCell(
        family="car",
        layout_id="oval",
        share=SHARE,
        drives=drives,
        fed={"sim": {"car": {"use": use}}, "low_power": {"car": {"use": use}}},
        factors={"imu": np.array([1.0, 1.01, 1.02, 1.03])},
        factor_laps={"imu": 40},
        neither=False,
        donors={"cars": [], "laps": 0, "days": 0},
        _sim=SimpleNamespace(tracks={"oval": track}),
        _raw_car="car",
    )


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fold")
    table = pace.build_table(_ranged(_model(), oval=(0.5, 1.0)), _oval_track())
    mid = float(table.lap_time_s[6])
    times = {"d1": mid, "d2": mid * 1.01, "d3": mid * 0.99, "d4": mid * 1.02}
    held = _held(tmp, times)
    rows, tpms = pg.score_cell(held, {"car": _cell(mid)})
    return held, rows, tpms, mid


class TestAFold:

    def test_every_arm_in_both_forms(self, scored):
        _, rows, _, _ = scored
        heat = set(rows.loc[rows["form"] == "heat", "arm"])
        chain = set(rows.loc[rows["form"] == "chain", "arm"])
        both = {"other_day", "ceiling", "fallback", "heat_up", "heat_down", "sim", "low_power"}
        assert both <= heat and both <= chain
        assert {"sim_direct", "sim_9", "other_layouts"} <= heat
        assert {"lap_up", "lap_down"} <= chain and "lap_up" not in heat
        assert set(rows["n"]) == set(pr.FLYING_LAPS) and set(rows["corner"]) == set(TYRES)
        assert rows["d_mbar"].notna().all()

    def test_another_day_is_three_nearest_sessions_or_every_other_day(self, scored):
        _, rows, _, _ = scored
        d1 = rows[(rows["session_id"] == "s0") & (rows["arm"] == "other_day") & (rows["n"] == 5)]
        assert d1[d1["form"] == "heat"]["pick"].nunique() == 3
        assert d1[d1["form"] == "chain"]["pick"].nunique() == 3
        # at a given lap time the pair's laps are the other day's own, a per cent or two away
        assert d1[d1["form"] == "heat"]["gap"].abs().between(0.005, 0.03).all()

    def test_a_planted_heat_error_is_seen_and_has_its_sign(self, scored):
        _, rows, _, _ = scored
        main = rows[(rows["n"] == 5) & (rows["form"] == "heat")]
        up = main[main["arm"] == "heat_up"]["d_mbar"]
        down = main[main["arm"] == "heat_down"]["d_mbar"]
        # more heat, a hotter tyre, a lower cold pressure for the same hot one
        assert (up < 0).all() and (down > 0).all()
        assert up.abs().median() == pytest.approx(down.abs().median(), rel=0.1)

    def test_the_simulation_supplies_its_own_lap_time_in_the_chain(self, scored):
        _, rows, _, mid = scored
        chain = rows[(rows["form"] == "chain") & (rows["n"] == 5)]
        sim = chain[(chain["arm"] == "sim") & (chain["session_id"] == "s0")]
        assert sim["gap"].abs().max() < 1e-6  # its fed use laps the oval at the day's pace
        up = chain[(chain["arm"] == "lap_up") & (chain["session_id"] == "s0")]
        assert up["gap"].iloc[0] == pytest.approx(np.log(1.0 + pg.LAP_TIME_PLANT), abs=1e-6)
        slow = chain[(chain["arm"] == "sim") & (chain["session_id"] == "s3")]
        assert slow["gap"].iloc[0] == pytest.approx(-np.log(1.02), abs=1e-6)
        assert (rows.loc[rows["arm"] == "sim", "status"] == pr.OK).all()

    def test_the_linear_read_of_the_knots_against_the_tables_own(self, scored):
        _, rows, _, _ = scored
        main = rows[(rows["n"] == 5) & (rows["form"] == "heat")].set_index(
            ["session_id", "stint_id", "corner"]
        )
        knots = main[main["arm"] == "sim"]["d_mbar"]
        direct = main[main["arm"] == "sim_direct"]["d_mbar"]
        assert (knots - direct).abs().max() < 2.0  # mbar

    def test_the_cells_and_the_score(self, scored):
        _, rows, _, _ = scored
        stints = pg.by_stint(rows)
        att = pg.attain(stints[(stints["n"] == 5) & (stints["form"] == "heat")], "sim")
        assert len(att) == 4 and (att["days"] == 4).all() and att["scored"].all()
        assert pg.tally(att)["scored"] == 4

    def test_pressure_against_the_tpms_at_the_fifth_flying_lap(self, scored):
        _, _, tpms, _ = scored
        assert set(tpms["arm"]) == {
            "sim",
            "ceiling",
            "other_layouts",
            "fallback",
            "own_heat",
            "heat_up",
            "heat_down",
        }
        # one row per day and tyre for each arm: the same rows under every entry
        counts = tpms.groupby("arm").size()
        assert (counts == 16).all()
        table = pg.tpms_table(tpms)
        assert set(table.columns) >= {"rows", "mae_mbar", "bias_mbar", "set_aside"}
        assert (table["set_aside"] == 0).all()

    def test_a_seen_cell_is_scored_with_the_fold_models_own_entry(self, tmp_path):
        held = _held(tmp_path, {"d1": 60.0, "d2": 61.0})
        held.track = "a"  # the fold model has an entry there
        held.stints["track_canonical"] = "a"
        held.prep.flying["track_canonical"] = "a"
        rows, tpms = pg.score_cell(held, {}, seen=True)
        assert set(rows["arm"]) == {"entry"} and set(rows["form"]) == set(pg.FORMS)
        assert set(tpms["arm"]) == {"entry"}
        chain = rows[rows["form"] == "chain"]
        assert chain["gap"].iloc[0] == pytest.approx(np.log(60.0 / 60.0), abs=1e-9)


class TestWetDays:
    def test_a_wet_day_is_set_against_a_run_of_the_same_day(self, tmp_path):
        held = _held(tmp_path, {"d1": 70.0, "d2": 71.0})
        # a second session on the first day, and every session wet
        twin = _held(tmp_path, {"d1": 70.5})
        for frame in (twin.laps, twin.stints):
            frame["session_id"] = "s9"
        twin.days["session_id"] = "s9"
        held.laps = pd.concat([held.laps, twin.laps], ignore_index=True)
        held.stints = pd.concat([held.stints, twin.stints], ignore_index=True)
        held.days = pd.concat([held.days, twin.days], ignore_index=True)
        held.entries["s9"] = twin.entries["s0"]
        held.days["condition"] = "wet"
        held.laps["condition"] = "wet"
        held.stints["condition"] = "wet"
        rows, tpms = pg.score_cell(held, {"car": _cell(70.0)}, wet=True)
        assert tpms.empty and set(rows["form"]) == {"heat"}
        pairs = rows[rows["arm"] == "other_day"]
        # only the two sessions of d1 have a run of the same day beside them
        assert set(pairs["session_id"]) == {"s0", "s9"}
        assert {"sim", "fallback", "heat_up", "heat_down"} <= set(rows["arm"])
        sim = rows[(rows["arm"] == "sim") & (rows["session_id"] == "s1")]
        assert sim["use"].between(0.4, 1.0).all()  # the dry table read at the wet lap time
        att = pg.attain(pg.by_stint(rows[rows["n"] == 5]), "sim")
        assert (att["days"] == 1).all() and not att["scored"].any()


class TestPooling:
    def test_weights_are_usable_laps_in_the_folds_fit(self, tmp_path):
        prod = _production(
            tmp_path,
            [
                _session("a", "KK-F", "tsukuba", "d1"),
                _session("b", "KK-SII", "tsukuba", "d1"),
                _session("c", "KK-SII", "suzuka", "d2"),
                _session("d", "KK-F", "fuji", "d3"),
                _session("e", "Inferno 86", "fuji", "d3"),
            ],
        )
        prod.raw_laps = pd.DataFrame(
            {
                "session_id": ["a"] * 5 + ["b"] * 4 + ["c"] * 6 + ["d"] * 3 + ["e"] * 9,
                "tire_usable": [True] * 26 + [False],
            }
        )
        # with Tsukuba left out the pooled car's fit has the KK-SII's Suzuka and the KK-F's Fuji
        assert pg.pool_weights(prod, "FJ", ["a", "b"]) == {"KK-F": 3.0, "KK-SII": 6.0}
        assert pg.pool_weights(prod, "Inferno 86", []) == {"Inferno 86": 8.0}

    def test_the_pooled_simulated_entry_of_two_raw_cars(self):
        table = pace.build_table(_ranged(_model(), oval=(0.5, 1.0)), _oval_track())
        t_fast, t_slow = float(table.lap_time_s[8]), float(table.lap_time_s[6])
        fast, slow = _cell(t_fast), _cell(t_slow)
        for cell, raw in ((fast, "one"), (slow, "two")):
            cell.fed = {"sim": {raw: cell.fed["sim"]["car"]}}
        day = {
            "session_id": "s0",
            "raw_car": "one",
            "date": "d1",
            "powertrain": "engine",
            "source": "imu",
            "lap_time_s": t_fast,
        }
        got = pg.pooled_sim_arms({"one": fast, "two": slow}, {"one": 1.0, "two": 1.0}, day)
        assert [a["form"] for a in got] == ["heat", "chain"]
        chain = got[1]["entry"]
        # the reference lap time is between the two cars' own
        assert chain.lap_time_s == pytest.approx(np.sqrt(t_fast * t_slow), rel=1e-6)
        assert chain.source == "sim_pooled" and len(chain.curve_t) == len(pg.POOL_GRID)
        # the day's own raw car alone under the name: the pooled entry is its own
        assert pg.pooled_sim_arms({"one": fast}, {"one": 1.0}, day) == []
        # another raw car alone in the fit: the day is read through that car's entry
        other = pg.pooled_sim_arms({"one": fast, "two": slow}, {"two": 1.0}, day)
        assert other[1]["entry"].lap_time_s == pytest.approx(t_slow, rel=1e-6)
        # a raw car without a fed use has no part: what is left is the day's own car's
        slow.fed = {"sim": {}}
        assert pg.pooled_sim_arms({"one": fast, "two": slow}, {"one": 1.0, "two": 1.0}, day) == []


class TestFirstVisits:
    def test_a_cells_first_day_with_history_elsewhere_before_it(self, tmp_path):
        prod = _production(
            tmp_path,
            [
                _session("a", "KK-SII", "tsukuba", "2026-01-01"),
                _session("b", "KK-SII", "suzuka", "2026-02-01"),
                _session("c", "KK-SII", "suzuka", "2026-03-01"),
            ],
        )
        days = pd.DataFrame(
            {
                "pooled": ["FJ", "FJ", "FJ"],
                "track": ["tsukuba", "suzuka", "suzuka"],
                "date": ["2026-01-01", "2026-02-01", "2026-03-01"],
            }
        )
        # Suzuka was first run after Tsukuba; Tsukuba had nothing before it
        assert pg.first_visits(prod, days) == [("FJ", "suzuka", "2026-02-01")]
        assert pg.first_visits(prod, days.iloc[:0]) == []
