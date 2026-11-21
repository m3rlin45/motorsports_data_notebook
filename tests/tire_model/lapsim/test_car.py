"""The car model from the sample tables (lap simulation C2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from motorsports_data_notebook.tire_model.lapsim import car as cm
from motorsports_data_notebook.tire_model.lapsim import geometry as geo
from motorsports_data_notebook.tire_model.lapsim import samples as smp

from ._synthetic import write_dataset, write_model_json

G = 9.80665
P_TRUE = 70.0
C_TRUE = 2.4e-4
GEARS = {3: 5.0, 4: 6.5, 5: 8.0}  # m/s per 1,000 rpm


def _throttle(layouts=None, *, p=P_TRUE, c=C_TRUE, k=0.0, days=None, car="car", n=80):
    """Full-throttle samples of a car with power ``p`` and drag ``c`` in
    standard air: ``layouts`` maps a layout to its top speed; ``days`` maps
    a date to ``(level, rho)``, the day's power level and the density its
    drag is met in."""
    layouts = layouts or {"long": 60.0, "short": 45.0}
    days = days or {"2026-01-10": (1.0, cm.RHO_REF)}
    rows = []
    for layout, v_top in layouts.items():
        for date, (level, rho) in days.items():
            for v in np.arange(27.5, v_top, 5.0):
                vs = v + np.linspace(-2.0, 2.0, n)
                for lat in (0.0, 0.8):
                    a = (p * level / vs - c * (rho / cm.RHO_REF) * vs**2) / G - k * lat**2
                    rows.append(
                        pd.DataFrame(
                            {
                                "session_id": f"{date}_{layout}",
                                "car": car,
                                "date": date,
                                "layout_id": layout,
                                "basis": "dry",
                                "lap_num": 1,
                                "v_ms": vs,
                                "a_g": a,
                                "lat_g": lat,
                                "straight": lat < smp.STRAIGHT_LAT_G,
                                "rho": rho,
                            }
                        )
                    )
    return pd.concat(rows, ignore_index=True)


def _curve(rpm):
    """The planted engine: 60 W/kg at 5,000 rpm rising to 80 at 7,000."""
    return 60.0 + 20.0 * (np.clip(rpm, 5000.0, 7000.0) - 5000.0) / 2000.0


def _geared(days=None, *, shift=7000.0, scale=1.0, car="car", layout="long", step=0.01):
    """Full-throttle straight-line samples of a car with the planted
    engine driven through :data:`GEARS`, shifting at ``shift``."""
    days = days or ["2026-01-10", "2026-01-11", "2026-01-12"]
    v = np.arange(25.0, 56.0, step)
    ratios = np.array(list(GEARS.values()))
    rpm_all = v[:, None] / ratios[None, :] * 1000.0
    ok = rpm_all <= shift
    idx = np.where(ok.any(axis=1), ok.argmax(axis=1), len(ratios) - 1)
    rpm = rpm_all[np.arange(len(v)), idx]
    a = (_curve(rpm) * scale / v - C_TRUE * v**2) / G
    frames = [
        pd.DataFrame(
            {
                "session_id": f"{d}_{layout}",
                "car": car,
                "date": d,
                "layout_id": layout,
                "basis": "dry",
                "lap_num": 1,
                "v_ms": v,
                "a_g": a,
                "lat_g": 0.0,
                "straight": True,
                "rpm": rpm,
                "gear": np.array(list(GEARS))[idx].astype(float),
            }
        )
        for d in days
    ]
    return pd.concat(frames, ignore_index=True)


class TestFamilies:
    def test_a_raw_car_and_a_date_name_a_powertrain(self):
        assert cm.family_of("KK-F-V2") == "KK-F" and cm.family_of("KK-SII") == "KK-SII"
        assert cm.powertrain_of("KK-F", "2026-11-13") == "KK-F"
        assert cm.powertrain_of("KK-F-V2", "2026-11-13") == "KK-F-V2"
        assert cm.powertrain_of("Inferno 86", "2026-04-06") == "first engine"
        assert cm.powertrain_of("Inferno 86", "2026-07-15") == "second engine"
        assert cm.powertrain_of("KK-SII", "2026-07-15") == cm.DEFAULT_ENGINE

    def test_as_family_keeps_the_raw_car(self):
        frame = pd.DataFrame({"car": ["KK-F-V2", "KK-F"], "date": ["2026-11-13", "2026-10-23"]})
        got = cm.as_family(frame)
        assert list(got["car"]) == ["KK-F", "KK-F"]
        assert list(got["raw_car"]) == ["KK-F-V2", "KK-F"]
        assert list(got["engine"]) == ["KK-F-V2", "KK-F"]

    def test_a_labelled_table_and_its_rows_are_not_labelled_again(self):
        frame = pd.DataFrame({"car": ["KK-F-V2", "KK-F"], "date": ["2026-11-13", "2026-10-23"]})
        got = cm.as_family(frame)
        assert cm.as_family(got) is got
        rows = got[got["raw_car"] == "KK-F"]
        assert cm.as_family(rows) is rows
        assert cm.as_family(frame) is not frame  # the raw table is not marked


class TestAir:
    def test_standard_air(self):
        std = cm.moist_air(cm.STD_T_C, cm.STD_DRY_HPA, 0.0)
        assert std["rho"] == pytest.approx(cm.RHO_REF) and std["cf"] == pytest.approx(1.0)

    def test_hot_thin_and_humid_air_costs_power(self):
        hot = cm.moist_air(35.0, 1000.0, 0.0)
        humid = cm.moist_air(35.0, 1000.0, 80.0)
        high = cm.moist_air(25.0, 920.0, 0.0)
        cold = cm.moist_air(5.0, 1013.0, 0.0)
        assert hot["cf"] > 1.0 and humid["cf"] > hot["cf"] and high["cf"] > 1.05
        assert cold["cf"] < 1.0 and cold["rho"] > cm.RHO_REF > high["rho"]
        assert humid["rho"] < hot["rho"]  # water vapour is lighter than air

    def test_from_the_hour_of_the_session(self, tmp_path):
        wx = pd.DataFrame(
            {
                "ts_utc": ["2026-01-10T01:00", "2026-01-10T02:00"],
                "temperature_2m": [10.0, 20.0],
                "relative_humidity_2m": [50.0, 50.0],
                "surface_pressure": [1013.25, 1000.0],
            }
        )
        (tmp_path / "weather_hourly" / "fuji").mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(wx), tmp_path / "weather_hourly/fuji/2026.parquet")
        sessions = pd.DataFrame(
            {
                "session_id": ["a", "b", "c"],
                "track_canonical": ["fuji", "fuji", "elsewhere"],
                "session_start_utc": pd.to_datetime(
                    ["2026-01-10T01:10", "2026-01-11T12:00", "2026-01-10T01:00"], utc=True
                ),
            }
        )
        air = cm.air_state(tmp_path, sessions)
        assert list(air.index) == ["a"]  # no hour within one; no weather
        want = cm.moist_air(10.0, 1013.25, 50.0)
        assert air.at["a", "rho"] == pytest.approx(want["rho"])
        assert air.at["a", "cf"] == pytest.approx(want["cf"])

    def test_with_air_defaults_to_standard(self):
        samples = pd.DataFrame({"session_id": ["a", "b"]})
        air = pd.DataFrame({"rho": [1.25], "cf": [0.95]}, index=["a"])
        got = cm.with_air(samples, air)
        assert list(got["rho"]) == pytest.approx([1.25, cm.RHO_REF])
        assert list(got["cf"]) == pytest.approx([0.95, 1.0])
        assert list(cm.with_air(samples, None)["cf"]) == [1.0, 1.0]


class TestDrag:
    def test_recovered_where_a_layout_reaches_50(self):
        bins = cm.solid_bins(_throttle())
        got = cm.fit_drag(bins)
        assert list(got["by_layout"]) == ["long"]  # the short layout tops out at 45 m/s
        assert got["c_drag"] == pytest.approx(C_TRUE, rel=0.02)
        assert got["by_layout"]["long"]["p_top"] == pytest.approx(P_TRUE, rel=0.02)
        assert got["by_layout"]["long"]["v_terminal_ms"] == pytest.approx(
            (P_TRUE / C_TRUE) ** (1 / 3), rel=0.02
        )
        assert got["source"] == "fitted (no prior)"

    def test_thin_air_is_brought_to_standard_density(self):
        thin = 0.92 * cm.RHO_REF
        bins = cm.solid_bins(_throttle(days={"2026-01-10": (1.0, thin)}))
        got = cm.fit_drag(bins)
        assert got["by_layout"]["long"]["rho"] == pytest.approx(thin)
        assert got["c_drag"] == pytest.approx(C_TRUE, rel=0.02)

    def test_held_within_the_priors_band(self):
        bins = cm.solid_bins(_throttle())
        assert cm.fit_drag(bins, 2.4e-4)["source"] == "fitted"
        low = cm.fit_drag(bins, 1.0e-4)
        assert low["c_drag"] == pytest.approx(1.3e-4)
        assert low["fitted"] == pytest.approx(C_TRUE, rel=0.02)
        assert "held" in low["source"]

    def test_no_layout_reaches(self):
        bins = cm.solid_bins(_throttle({"short": 45.0}))
        assert cm.fit_drag(bins, 2.4e-4)["c_drag"] == 2.4e-4
        assert np.isnan(cm.fit_drag(bins)["c_drag"])


class TestDragAcrossGears:
    def test_the_same_rpm_in_two_gears_gives_the_drag(self):
        got = cm.fit_drag_rpm(_geared())
        assert got["c_drag"] == pytest.approx(C_TRUE, rel=0.02)
        assert got["cells"] > 10 and got["n"] > 1000

    def test_a_days_power_level_does_not_bias_it(self):
        strong = _geared(["2026-01-11"], scale=1.08)
        both = pd.concat([_geared(["2026-01-10"]), strong], ignore_index=True)
        assert cm.fit_drag_rpm(both)["c_drag"] == pytest.approx(C_TRUE, rel=0.02)

    def test_it_follows_the_air(self):
        thin = _geared()
        thin["rho"] = 0.9 * cm.RHO_REF
        # the same accelerations met in thinner air: the drag at standard density is more
        assert cm.fit_drag_rpm(thin)["c_drag"] == pytest.approx(C_TRUE / 0.9, rel=0.02)

    def test_needs_rpm_and_two_gears(self):
        assert np.isnan(cm.fit_drag_rpm(_throttle())["c_drag"])
        one_gear = _geared()
        one_gear = one_gear[one_gear["gear"] == 3]
        assert np.isnan(cm.fit_drag_rpm(one_gear)["c_drag"])


class TestPower:
    def test_specific_power_and_layouts(self):
        bins = cm.solid_bins(_throttle())
        got = cm.specific_power(bins, C_TRUE)
        assert got["p_specific"] == pytest.approx(P_TRUE, rel=0.005)
        assert got["by_layout"]["short"]["p_wkg"] == pytest.approx(P_TRUE, rel=0.005)
        assert got["by_layout"]["long"]["matched"] == pytest.approx(1.0, abs=0.005)
        assert got["by_layout"]["long"]["bins"] == 7 and got["by_layout"]["short"]["bins"] == 4

    def test_power_is_brought_to_standard_air(self):
        # a hot day: the engine made 5 % less, and the factor gives it back
        th = _throttle(days={"2026-01-10": (1 / 1.05, cm.RHO_REF)})
        th["cf"] = 1.05
        got = cm.specific_power(cm.solid_bins(th), C_TRUE)
        assert got["p_specific"] == pytest.approx(P_TRUE, rel=0.005)

    def test_a_wrong_drag_shows_between_layouts(self):
        # with no drag the long layout, which spends its time at speed, reads low
        got = cm.specific_power(cm.solid_bins(_throttle()), 0.0)
        assert got["by_layout"]["long"]["p_wkg"] < 0.93 * got["by_layout"]["short"]["p_wkg"]

    def test_cornering_drag(self):
        th = _throttle(k=0.05)
        got = cm.corner_drag(th, cm.solid_bins(th))
        assert got["k_corner_drag"] == pytest.approx(0.05, abs=0.003)
        assert set(got["by_layout"]) == {"long", "short"}

    def test_day_levels_by_speed_bin_follow_the_planted_level(self):
        days = {f"2026-01-{10 + i}": (lvl, cm.RHO_REF) for i, lvl in enumerate((0.95, 1.0, 1.05))}
        th = _throttle(days=days)
        levels = cm.day_levels(th, cm.solid_bins(th), C_TRUE)
        got = levels.set_index(["date", "layout_id"])["level"]
        assert got[("2026-01-12", "long")] / got[("2026-01-11", "long")] == pytest.approx(
            1.05, rel=0.01
        )
        assert len(levels) == 6

    def test_what_the_air_correction_leaves(self):
        # the engine goes with the density and nothing corrects it: slope 1
        days = {
            f"2026-01-{10 + i}": (rho / cm.RHO_REF, cm.RHO_REF)
            for i, rho in enumerate((1.10, 1.15, 1.20, 1.25, 1.30, 1.18))
        }
        th = _throttle(days=days)
        for i, rho in enumerate((1.10, 1.15, 1.20, 1.25, 1.30, 1.18)):
            th.loc[th["date"] == f"2026-01-{10 + i}", "rho"] = rho
        # drag at standard density whatever the label says
        levels = cm.day_levels(th, cm.solid_bins(th), 0.0)
        dens = cm.density_slope(levels)
        assert dens["slope"] > 0.5 and dens["r2"] > 0.8
        assert np.isnan(cm.density_slope(levels.iloc[:3])["slope"])


class TestEngine:
    def test_the_curve_comes_back_against_rpm(self):
        curve = cm.engine_curve(_geared(), C_TRUE)
        assert curve["rpm"].min() == 5000.0 and curve["rpm"].max() >= 6750.0
        centre = curve["rpm"] + cm.RPM_BIN / 2.0
        assert curve["p_wkg"].to_numpy() == pytest.approx(_curve(centre), rel=0.01)
        assert (curve["n"] >= cm.CURVE_MIN_SAMPLES).all()

    def test_no_rpm_no_curve(self):
        assert cm.engine_curve(_throttle(), C_TRUE).empty
        assert cm.gearing(_throttle()) == {}
        assert cm.shift_rpm_by_day(_throttle()).empty

    def test_gears_and_the_days_shift_point(self):
        early, late = _geared(["2026-01-10"], shift=6800.0), _geared(["2026-01-11"], shift=7300.0)
        both = pd.concat([early, late], ignore_index=True)
        assert cm.gearing(both) == pytest.approx(GEARS)
        shifts = cm.shift_rpm_by_day(both)
        assert shifts["2026-01-10"] == pytest.approx(6800.0, abs=60.0)
        assert shifts["2026-01-11"] == pytest.approx(7300.0, abs=60.0)

    def test_every_day_at_once_is_each_day_alone(self):
        days = {
            "2026-01-10": _geared(["2026-01-10"], shift=6800.0),
            "2026-01-11": _geared(["2026-01-11"], shift=7300.0),
            "2026-01-12": _geared(["2026-01-12"], shift=7000.0, layout="short", step=0.02),
            "2026-01-13": _geared(["2026-01-13"]).drop(columns="gear"),  # no gear logged
            "2026-01-14": _geared(["2026-01-14"]).iloc[:100],  # too few samples
        }
        both = pd.concat(days.values(), ignore_index=True)
        shifts = cm.shift_rpm_by_day(both)
        assert list(shifts.index) == sorted(days)[:4] and shifts.name == "shift_rpm"
        for date in shifts.index:
            assert shifts[date] == cm.shift_rpm_by_day(days[date])[date]
        sets = cm.gear_sets_by_day(both)
        assert sorted(sets) == sorted(days)[:3] + ["2026-01-14"]
        for date in sets:
            assert sets[date] == cm.gear_sets_by_day(days[date])[date]
        # without a gear the day's shift point is the quantile of all its rpm
        bare = days["2026-01-13"]
        assert shifts["2026-01-13"] == pytest.approx(bare["rpm"].quantile(cm.SHIFT_QUANTILE))

    def test_each_layouts_power_against_the_curve_at_once(self):
        long, short = _geared(["2026-01-10"]), _geared(["2026-01-11"], scale=0.9, layout="short")
        slow = _geared(["2026-01-12"], layout="slow").assign(rpm=np.nan)  # no rpm there
        both = pd.concat([long, short, slow], ignore_index=True)
        curve = cm.engine_curve(long, C_TRUE)
        got = cm.power_scale_by(both, C_TRUE, curve)
        assert list(got) == ["long", "short", "slow"]
        assert got["long"] == cm.power_scale(long, C_TRUE, curve) == pytest.approx(1.0, abs=1e-3)
        assert got["short"] == cm.power_scale(short, C_TRUE, curve) == pytest.approx(0.9, abs=1e-3)
        assert np.isnan(got["slow"])
        assert cm.power_scale_by(both.iloc[:0], C_TRUE, curve) == {}

    def test_a_later_shift_is_the_same_engine_at_equal_rpm(self):
        early, late = _geared(["2026-01-10"], shift=6800.0), _geared(["2026-01-11"], shift=7300.0)
        both = pd.concat([early, late], ignore_index=True)
        bins = cm.solid_bins(both)
        by_rpm = cm.day_levels(both, bins, C_TRUE, cm.engine_curve(_geared(), C_TRUE))
        assert by_rpm["level"].to_numpy() == pytest.approx(1.0, abs=0.01)
        # at a speed where only the later shifter is still in third, it accelerates harder
        at = (both["v_ms"] > 34.5) & (both["v_ms"] < 36.0)
        a = both[at].groupby("date")["a_g"].median()
        assert a["2026-01-11"] > a["2026-01-10"] + 0.01

    def test_the_top_gear_is_not_a_shift_point(self):
        # fifth is run to 7,000 rpm at the end of the straight; the shifts are at 6,500
        day = _geared(["2026-01-10"], shift=6500.0)
        assert day[day["gear"] == 5]["rpm"].max() > 6900.0
        assert cm.shift_rpm_by_day(day)["2026-01-10"] == pytest.approx(6500.0, abs=60.0)
        # without a gear channel all that is left is the day's highest revs
        assert cm.shift_rpm_by_day(day.drop(columns="gear"))["2026-01-10"] > 6900.0

    def test_scale_against_another_powertrain(self):
        base = cm.engine_curve(_geared(), C_TRUE)
        assert cm.power_scale(_geared(scale=1.1), C_TRUE, base) == pytest.approx(1.1, abs=0.005)
        assert np.isnan(cm.power_scale(_throttle(), C_TRUE, base))


SHORT = {3: 4.5, 4: 5.5, 5: 6.5}  # a second gear set, for a shorter circuit


def _drag_strip(lost_s, laps=24, *, hz=25.0, guard_s=0.3, grade=0.0):
    """Laps of full throttle through :data:`GEARS` with ``lost_s`` seconds
    without drive at each upshift (shifting at 7,000 rpm), as the
    full-throttle table has them: no sample within ``guard_s`` of a gear
    change. ``lost_s`` is one time or one per gear shifted out of."""
    ratios, numbers = np.array(list(GEARS.values())), list(GEARS)
    lost = dict(zip(numbers, np.broadcast_to(lost_s, len(numbers) - 1)))
    dt, frames = 0.002, []
    for lap in range(laps):
        v, t, g, idle, rows, changes = 22.0 + 0.1 * lap, 0.0, 0, 0.0, [], []
        while v < 57.0 and t < 60.0:
            rpm = v / ratios[g] * 1000.0
            if idle <= 0.0 and g < len(ratios) - 1 and rpm >= 7000.0:
                idle, g = lost[numbers[g]], g + 1
                changes.append(t)
                rpm = v / ratios[g] * 1000.0
            engine = 0.0 if idle > 0.0 else float(_curve(rpm)) / v
            a = engine - C_TRUE * v * v - G * grade
            rows.append((t, v, (a + G * grade) / G, numbers[g], rpm, idle > 0.0))
            v, t, idle = v + a * dt, t + dt, idle - dt
        frame = pd.DataFrame(rows, columns=["t_s", "v_ms", "a_g", "gear", "rpm", "idle"])
        frame = frame.iloc[:: int(round(1.0 / hz / dt))]
        near = np.zeros(len(frame), dtype=bool)
        for at in changes:
            end = at + lost[numbers[0]] if not np.ndim(lost_s) else at + max(lost.values())
            near |= ((frame["t_s"] > at - guard_s) & (frame["t_s"] < end + guard_s)).to_numpy()
        frame = frame[~near & ~frame["idle"]].drop(columns="idle")
        frames.append(
            frame.assign(
                session_id="s", lap_num=lap, car="car", date="2026-01-10", layout_id="long",
                basis="dry", lat_g=0.0, grade=grade, straight=True, gear=frame["gear"].astype(float),
            )  # fmt: skip
        )
    return pd.concat(frames, ignore_index=True)


class TestShiftTimes:
    def test_the_time_without_drive_comes_back_per_gear(self):
        got = cm.shift_times(_drag_strip([0.4, 0.25]), C_TRUE)
        assert sorted(got) == [3, 4] and got[3]["n"] == got[4]["n"] == 24
        assert got[3]["shift_s"] == pytest.approx(0.4, abs=0.03)
        assert got[4]["shift_s"] == pytest.approx(0.25, abs=0.03)
        assert got[3]["gap_s"] == pytest.approx(0.4 + 0.6, abs=0.08)
        assert abs(got[3]["bias_s"]) < 0.03 and got[3]["short_ms"] > 0.5

    def test_a_shift_that_costs_nothing_reads_nothing(self):
        got = cm.shift_times(_drag_strip(0.0), C_TRUE)
        assert got[3]["shift_s"] < 0.03 and got[4]["shift_s"] < 0.03

    def test_uphill_it_is_the_same_time(self):
        got = cm.shift_times(_drag_strip(0.3, grade=0.03), C_TRUE)
        assert got[3]["shift_s"] == pytest.approx(0.3, abs=0.03)
        assert got[4]["shift_s"] == pytest.approx(0.3, abs=0.03)

    def test_a_wrong_drag_goes_to_the_bias_not_the_time(self):
        got = cm.shift_times(_drag_strip(0.3), 1.5 * C_TRUE)
        assert got[4]["shift_s"] == pytest.approx(0.3, abs=0.05)

    def test_too_few_upshifts_or_no_gear_is_no_time(self):
        assert cm.shift_times(_drag_strip(0.3, laps=10), C_TRUE) == {}
        strip = _drag_strip(0.3)
        assert cm.shift_times(strip.drop(columns="gear"), C_TRUE) == {}
        assert cm.shift_times(strip.iloc[:0], C_TRUE) == {}

    def test_the_fit_carries_the_times_and_a_borrowed_curve_borrows_them(self):
        tables = _tables(throttle=_drag_strip([0.4, 0.25]))
        model, work = cm.fit_car(tables, "car")
        spec = model.power["engines"][model.power["default_engine"]]
        assert spec["shift_s"] == {
            "3": pytest.approx(0.4, abs=0.04),
            "4": pytest.approx(0.25, abs=0.04),
        }
        assert sorted(work["engines"][model.power["default_engine"]]["shift_times"]) == [3, 4]


def _geared_with(gears, days, **kwargs):
    """:func:`_geared` driven through another gear set."""
    saved = dict(GEARS)
    GEARS.clear()
    GEARS.update(gears)
    try:
        return _geared(days, **kwargs)
    finally:
        GEARS.clear()
        GEARS.update(saved)


class TestInferredGear:
    def test_speed_over_rpm_falls_into_one_cluster_per_gear(self):
        day = _geared(["2026-01-10"])
        ratio = (day["v_ms"] / day["rpm"] * 1000.0).to_numpy()
        assert cm.ratio_clusters(ratio) == pytest.approx([5.0, 6.5, 8.0], rel=1e-3)
        noisy = ratio * (1.0 + np.random.default_rng(0).normal(0.0, 0.004, ratio.size))
        assert cm.ratio_clusters(noisy) == pytest.approx([5.0, 6.5, 8.0], rel=0.005)
        assert cm.ratio_clusters(ratio[:10]) == []

    def test_a_gear_channel_is_made_where_there_is_none(self):
        day = _geared(["2026-01-10"])
        truth = day["gear"].to_numpy()
        got = cm.with_gears(day.drop(columns="gear"))
        assert got["gear_inferred"].all()
        # numbered from 1 among the gears seen: third, fourth, fifth
        assert (got["gear"].to_numpy() == truth - 2).all()

    def test_a_logged_gear_is_kept(self):
        day = _geared(["2026-01-10"])
        got = cm.with_gears(day)
        assert not got["gear_inferred"].any() and (got["gear"] == day["gear"]).all()

    def test_a_sample_between_gears_has_none(self):
        day = _geared(["2026-01-10"]).drop(columns="gear")
        day.loc[day.index[:5], "rpm"] *= 1.15  # a slipping clutch
        got = cm.with_gears(day)
        assert got["gear"].iloc[:5].isna().all() and got["gear"].iloc[5:].notna().all()

    def test_a_day_without_its_lowest_gear_keeps_the_numbers(self):
        full = _geared(["2026-01-10"]).drop(columns="gear")
        part = _geared(["2026-01-11"]).drop(columns="gear")
        part = part[part["v_ms"] > 36.0]  # third gear never used that day
        got = cm.with_gears(pd.concat([full, part], ignore_index=True))
        sets = cm.gear_sets_by_day(got)
        assert sets["2026-01-10"] == pytest.approx({1: 5.0, 2: 6.5, 3: 8.0}, rel=1e-3)
        assert sets["2026-01-11"] == pytest.approx({2: 6.5, 3: 8.0}, rel=1e-3)

    def test_two_gear_sets_and_each_day_gets_all_of_its_own(self):
        long_days = ["2026-01-10", "2026-01-11"]
        short_days = ["2026-02-10", "2026-02-11", "2026-02-12"]
        part = _geared(["2026-01-12"])
        part = part[part["v_ms"] > 36.0]
        th = pd.concat(
            [_geared(long_days), part, _geared_with(SHORT, short_days)], ignore_index=True
        )
        got = cm.with_gears(th.drop(columns="gear"))
        sets, found = cm.complete_sets(cm.gear_sets_by_day(got))
        assert [f["days"] for f in found] == [3, 3]
        assert sets["2026-02-11"] == pytest.approx({1: 4.5, 2: 5.5, 3: 6.5}, rel=1e-3)
        # the day that never used its lowest gear still has it
        assert sets["2026-01-12"] == pytest.approx({1: 5.0, 2: 6.5, 3: 8.0}, rel=1e-3)
        assert sets["2026-01-12"] is sets["2026-01-10"]

    def test_a_car_without_a_gear_channel_is_fitted_through_its_gears(self):
        th = pd.concat(
            [_geared(), _geared_with(SHORT, ["2026-02-10", "2026-02-11"])], ignore_index=True
        )
        model, work = cm.fit_car(_tables(throttle=th.drop(columns="gear")), "car")
        engine = model.power["engines"][cm.DEFAULT_ENGINE]
        assert engine["gears_inferred"] and len(engine["gear_sets"]) == 2
        assert list(engine["gears"].values()) == pytest.approx([5.0, 6.5, 8.0], rel=1e-3)
        assert engine["shift_rpm"] == pytest.approx(7000.0, abs=60.0)
        assert [r for r, _ in engine["curve"]][0] == 5000.0
        # one engine, one drag, whichever set it was driven through
        assert model.power["c_drag"] == pytest.approx(C_TRUE, rel=0.02)
        day_sets = work["engines"][cm.DEFAULT_ENGINE]["gear_sets"]
        assert list(day_sets["2026-02-10"].values()) == pytest.approx([4.5, 5.5, 6.5], rel=1e-3)

    def test_a_powertrain_that_kept_the_gearbox_shares_its_gears(self, monkeypatch):
        monkeypatch.setitem(cm.FAMILY, "car-V2", "car")
        monkeypatch.setitem(
            cm.POWERTRAINS,
            "car",
            [{"name": "car", "raw_car": "car"}, {"name": "car-V2", "raw_car": "car-V2"}],
        )
        new = _geared(["2026-02-10"], scale=1.1, car="car-V2")
        new = new[new["v_ms"] > 36.0]  # its one day never used third
        th = pd.concat([_geared(), new], ignore_index=True)
        model, _ = cm.fit_car(_tables(throttle=th), "car")
        assert model.power["engines"]["car-V2"]["gears"] == model.power["engines"]["car"]["gears"]
        assert list(model.power["engines"]["car-V2"]["gears"]) == ["3", "4", "5"]


def _zones(share, n=600, *, mu_b=1.0, level=1.4, c=0.0, seed=0):
    """Braking zones of a car whose limit under cornering lies ``share``
    of the way from the diamond to the ellipse: each zone's peak is some
    way under the limit at its lateral, the best of them on it."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    limit = (1 - share) * (1 - x) + share * np.sqrt(1 - x**2)
    effort = 1.0 - 0.3 * rng.uniform(0.0, 1.0, n) ** 3  # the 90th percentile is on the limit
    effort = effort / np.quantile(effort, 0.9)
    v = rng.uniform(30.0, 50.0, n)
    return pd.DataFrame(
        {
            "layout_id": "long",
            "lat_peak_g": x * level * rng.choice([-1.0, 1.0], n),
            "decel_p95_g": limit * effort * mu_b + c * v**2 / G,
            "v_peak_ms": v,
        }
    )


class TestBrakingShape:
    @pytest.mark.parametrize("share", [0.0, 0.5, 0.8, 1.0])
    def test_the_share_of_the_way_to_the_ellipse_comes_back(self, share):
        got = cm.braking_shape(_zones(share, c=C_TRUE), C_TRUE, 1.0, 1.4)
        assert got["brake_ellipse"] == pytest.approx(share, abs=0.1)
        bands = got["bands"]
        assert list(bands["band"].round(1)) == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        assert (bands["zones"] >= cm.BRAKE_BAND_MIN_ZONES).all()

    def test_held_between_the_two_and_none_without_zones_in_the_bands(self):
        beyond = _zones(1.0).assign(decel_p95_g=lambda d: d["decel_p95_g"] * 1.5)
        assert cm.braking_shape(beyond, 0.0, 1.0, 1.4)["brake_ellipse"] == 1.0
        straight = _zones(0.5).assign(lat_peak_g=0.05)  # every stop in a straight line
        assert np.isnan(cm.braking_shape(straight, 0.0, 1.0, 1.4)["brake_ellipse"])
        assert np.isnan(cm.braking_shape(_zones(0.5, n=40), 0.0, 1.0, 1.4)["brake_ellipse"])
        assert np.isnan(cm.braking_shape(_zones(0.5), 0.0, float("nan"), 1.4)["brake_ellipse"])
        bare = _zones(0.5).drop(columns="lat_peak_g")
        assert np.isnan(cm.braking_shape(bare, 0.0, 1.0, 1.4)["brake_ellipse"])


class TestBrakeOnset:
    def _stops(self, n=60, lift=0.3, build=0.24, days=("2026-01-10",)):
        rng = np.random.default_rng(2)
        frames = [
            pd.DataFrame(
                {
                    "date": d,
                    "v_on_ms": rng.uniform(32.0, 50.0, n),
                    "lift_s": lift + 0.04 * k + rng.normal(0.0, 0.02, n),
                    "build_s": build + rng.normal(0.0, 0.02, n),
                }
            )
            for k, d in enumerate(days)
        ]
        return pd.concat(frames, ignore_index=True)

    def test_the_lift_and_half_the_build(self):
        got = cm.brake_onset(self._stops())
        assert got["lift_s"] == pytest.approx(0.3, abs=0.01)
        assert got["build_s"] == pytest.approx(0.24, abs=0.01)
        assert got["onset_s"] == pytest.approx(0.3 + 0.5 * 0.24 / 0.8, abs=0.015)
        assert got["zones"] == 60

    def test_only_the_stops_from_full_throttle_at_speed(self):
        stops = self._stops()
        slow = stops.assign(v_on_ms=20.0)
        coasting = stops.assign(lift_s=np.nan)
        both = pd.concat([stops, slow.assign(lift_s=0.9), coasting], ignore_index=True)
        assert cm.brake_onset(both)["onset_s"] == pytest.approx(cm.brake_onset(stops)["onset_s"])
        assert np.isnan(cm.brake_onset(stops.iloc[:10])["onset_s"])
        assert np.isnan(cm.brake_onset(stops.drop(columns="lift_s"))["onset_s"])
        # without a pressure channel the build counts as none
        bare = cm.brake_onset(stops.assign(build_s=np.nan))
        assert bare["onset_s"] == pytest.approx(bare["lift_s"])

    def test_every_day_at_once_is_each_day_alone(self):
        stops = self._stops(days=("2026-01-10", "2026-01-11", "2026-01-12"))
        few = self._stops(n=5, days=("2026-01-13",))
        days = cm.onset_by_day(pd.concat([stops, few], ignore_index=True))
        assert list(days["date"]) == ["2026-01-10", "2026-01-11", "2026-01-12"]
        for row in days.itertuples():
            alone = cm.brake_onset(stops[stops["date"] == row.date])
            assert row.onset_s == pytest.approx(alone["onset_s"], rel=1e-12)
        assert days["onset_s"].is_monotonic_increasing  # the later days lift later


class TestBrakingLevel:
    def test_from_the_zones_braked_in_a_straight_line(self):
        zones = _zones(0.6, n=1200, c=C_TRUE)
        every = cm.braking_level(zones, C_TRUE)
        line = cm.braking_level(zones, C_TRUE, 1.4)
        assert not every["straight"] and every["zones"] == 1200
        assert line["straight"] and line["zones"] == int((zones["lat_peak_g"].abs() < 0.28).sum())
        # the stops made while cornering pull the level over every zone down
        assert line["mu_b"] == pytest.approx(0.94, abs=0.05) and every["mu_b"] < line["mu_b"] - 0.03
        # and against the straight-line level the shape comes back
        shape = cm.braking_shape(zones, C_TRUE, line["mu_b"], 1.4)["brake_ellipse"]
        assert shape == pytest.approx(0.6 + 0.06 * 0.6, abs=0.1)

    def test_every_zone_where_too_few_are_straight_or_nothing_says(self):
        zones = _zones(0.6)
        cornering = zones[zones["lat_peak_g"].abs() > 0.3]
        got = cm.braking_level(cornering, 0.0, 1.4)
        assert not got["straight"] and got["zones"] == len(cornering)
        bare = zones.drop(columns="lat_peak_g")
        assert not cm.braking_level(bare, 0.0, 1.4)["straight"]
        assert not cm.braking_level(zones, 0.0, float("nan"))["straight"]
        assert cm.braking_level(zones.iloc[:0], 0.0, 1.4)["straight"] is False


class TestBraking:
    def test_peak_less_drag(self):
        brake = pd.DataFrame(
            {
                "layout_id": ["a"] * 50 + ["b"] * 50,
                "decel_p95_g": np.r_[np.linspace(0.5, 1.0, 50), np.linspace(0.4, 0.8, 50)],
                "v_peak_ms": 50.0,
            }
        )
        got = cm.braking_level(brake, C_TRUE)
        drag = C_TRUE * 50.0**2 / G
        assert got["by_layout"]["a"] == pytest.approx(0.95 - drag, abs=0.01)
        assert got["by_layout"]["b"] == pytest.approx(0.76 - drag, abs=0.01)
        assert got["by_layout"]["b"] < got["mu_b"] < got["by_layout"]["a"]


def _tables(car="car", throttle=None):
    th = _throttle(car=car, k=0.04) if throttle is None else throttle
    keys = {"session_id": "s", "car": car, "date": "2026-01-10", "lap_num": 1}
    brake = pd.DataFrame(
        {
            **keys,
            "layout_id": "long",
            "basis": "dry",
            "decel_p95_g": np.linspace(0.6, 1.0, 40),
            "v_peak_ms": 40.0,
            "v_on_ms": 50.0,
            "v_min_ms": 20.0,
        }
    )
    apex = []
    for basis, level in (("dry", 1.4), ("wet", 1.12)):
        for lap in range(10):
            apex.append(
                {
                    **keys,
                    "lap_num": lap,
                    "layout_id": "long",
                    "basis": basis,
                    "corner": 0,
                    "sign": 1,
                    "s_apex_m": 100.0,
                    "v_apex_ms": 25.0,
                    "lat_track_g": level,
                    "lat_doppler_g": level,
                    "lat_accel_g": np.nan,
                }
            )
    sessions = pd.DataFrame([{"session_id": "s", "car": car}])
    return smp.CarTables(sessions, th, th.iloc[:0], brake, pd.DataFrame(apex))


class TestFit:
    def test_model_from_tables_and_json(self, tmp_path):
        model, work = cm.fit_car(_tables(), "car")
        engine = model.power["engines"][cm.DEFAULT_ENGINE]
        assert model.power["default_engine"] == cm.DEFAULT_ENGINE
        assert engine["p_specific"] == pytest.approx(P_TRUE, rel=0.01)
        assert model.power["p_specific"] == engine["p_specific"]
        assert engine["curve"] is None and engine["gears"] is None and engine["shift_rpm"] is None
        assert model.power["c_drag"] == pytest.approx(C_TRUE, rel=0.02)
        assert model.power["rho_ref"] == pytest.approx(cm.RHO_REF, abs=1e-4)
        assert model.power["k_corner_drag"] == pytest.approx(0.04, abs=0.003)
        assert model.grip == {
            "mu_y": pytest.approx(1.4),
            "mu_b": pytest.approx(0.96 - C_TRUE * 1600 / G, abs=0.01),
            "n_brake": 1.0,
            "brake_ellipse": None,  # the zones here have no lateral
            "onset_s": None,  # nor how they began
        }
        assert model.wet["grip_scale"] == pytest.approx(0.8)
        assert model.support["straight"] == {"v_min": 25.0, "v_max": 60.0}
        assert model.ref_u == {} and "provisional" in model.meta["mu_y_source"]
        assert set(work) >= {"drag", "engines", "days", "density", "braking"}
        back = cm.CarModel.load(model.save(tmp_path / "car.json"))
        assert back.to_json() == model.to_json()

    def test_a_geared_engine_has_its_curve_gears_and_shift_point(self):
        model, _ = cm.fit_car(_tables(throttle=_geared()), "car")
        engine = model.power["engines"][cm.DEFAULT_ENGINE]
        assert [r for r, _ in engine["curve"]][0] == 5000.0
        assert engine["gears"] == {"3": 5.0, "4": 6.5, "5": 8.0}
        assert engine["shift_rpm"] == pytest.approx(7000.0, abs=60.0)

    def test_two_powertrains_share_the_chassis(self, monkeypatch):
        monkeypatch.setitem(
            cm.POWERTRAINS,
            "car",
            [{"name": "old", "until": "2026-01-31"}, {"name": "new", "from": "2026-02-01"}],
        )
        # the new engine: 10 % more power on one short day, too little for a curve of its own
        new = _geared(["2026-02-10"], scale=1.1, step=0.05)
        th = pd.concat([_geared(), new], ignore_index=True)
        model, work = cm.fit_car(_tables(throttle=th), "car")
        old_e, new_e = model.power["engines"]["old"], model.power["engines"]["new"]
        assert model.power["default_engine"] == "new"
        assert old_e["scale_against"] is None and old_e["curve_from"] is None
        assert new_e["scale"] == pytest.approx(1.1, abs=0.01) and new_e["scale_against"] == "old"
        assert new_e["curve_from"] == "old"
        ratio = np.array([b[1] / a[1] for a, b in zip(old_e["curve"], new_e["curve"])])
        assert ratio == pytest.approx(new_e["scale"], rel=1e-3)
        assert new_e["gears"] == old_e["gears"]
        # one drag, one braking peak, one lateral level for both
        assert model.power["c_drag"] == pytest.approx(C_TRUE, rel=0.01)
        assert "equal rpm" in work["drag"]["source"]
        # the fit at the top of the straight table takes the power for one
        # number over its bins, and this engine's is not: it reads 14 % low
        assert work["drag"]["top_of_table"] == pytest.approx(0.86 * C_TRUE, rel=0.03)
        assert set(work["days"]["engine"]) == {"old", "new"}
        assert work["days"]["level"].to_numpy() == pytest.approx(1.0, abs=0.02)

    def test_a_raw_car_of_the_family_is_a_powertrain(self, monkeypatch):
        monkeypatch.setitem(cm.FAMILY, "car-V2", "car")
        monkeypatch.setitem(
            cm.POWERTRAINS,
            "car",
            [{"name": "car", "raw_car": "car"}, {"name": "car-V2", "raw_car": "car-V2"}],
        )
        th = pd.concat(
            [_geared(), _geared(["2026-02-10"], scale=1.08, car="car-V2")], ignore_index=True
        )
        model, _ = cm.fit_car(_tables(throttle=th), "car")
        assert set(model.power["engines"]) == {"car", "car-V2"}
        assert model.power["engines"]["car-V2"]["scale"] == pytest.approx(1.08, abs=0.01)
        assert model.meta["fitted_from"]["raw_cars"] == ["car", "car-V2"]

    def test_a_car_without_a_throttle_still_has_grip(self):
        tables = _tables()
        tables.throttle = tables.throttle.iloc[:0]
        model, _ = cm.fit_car(tables, "car")
        assert model.power["p_specific"] is None and model.power["c_drag"] is None
        assert model.grip["mu_y"] == pytest.approx(1.4)
        assert model.grip["mu_b"] == pytest.approx(0.96, abs=0.01)  # no drag to take out

    def test_published_mass(self):
        model, _ = cm.fit_car(_tables("KK-SII"), "KK-SII")
        assert model.mass_kg == 495.0


class TestTableClauses:
    def test_report_on_a_synthetic_dataset(self, tmp_path, monkeypatch):
        geo._INDEX_CACHE.clear()
        sessions = {
            f"2026-01-1{i}_s{i}": {
                "car": "car",
                "date": f"2026-01-1{i}",
                "kwargs": {"n_laps": 4, "seed": i, "controls": True},
            }
            for i in range(3)
        }
        write_dataset(tmp_path, sessions)
        write_model_json(tmp_path)
        monkeypatch.setattr(smp, "MODEL_CARS", ("car",))
        tables = smp.car_tables(tmp_path, tracks_dir=tmp_path / "tracks")
        rep, fits = cm.table_clauses(tmp_path, tables)
        assert rep.gate == "C2" and set(fits) == {"car"}
        names = [it.name for it in rep.items]
        assert names[0] == "Power agrees between layouts"
        assert rep.items[0].passed is True  # one layout: its own value is the pooled one
        for want in ("The powertrains", "Power and shift point by day"):
            assert want in names
        assert fits["car"][0].grip["mu_b"] == pytest.approx(0.39, abs=0.05)
