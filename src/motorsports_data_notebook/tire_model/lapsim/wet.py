"""The wet as a condition on the car's limit (#206 D2; the driver
committee's clause 1.6).

A wet lap is solved to its own lap time, so the wet grip scale, the day's
water level and the driver's use are one product and no simulated lap
depends on how it is split. What the lap time does not absorb is that
**braking falls less than cornering** in the wet. This module reads it:

- the *wet cornering ratio*: at each of the car's dry grip-limited
  corners that has wet passes, the expected best of ``m`` wet passes over
  the expected best of the same ``m`` dry passes there (``m`` half the
  wet passes, at most 20, at least 3); the median over corners. One
  reading, the Doppler lateral in the tyres' frame over the load, with no
  noise correction: the statistic and ``m`` are the same on both sides.
- the *wet braking ratio*: the same on the straight-line stops' held
  deceleration less drag, per stopping place.
- the *wet braking term*: their quotient, which is what the solver takes
  (:attr:`~.solver.SolverCar.wet_brake`), on a wet lap only.

Each wet day's own cornering ratio is the day's water level as far as the
logs give it; it is reported and nothing is solved with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...tire_etl.kinematics import G_MS2
from . import limits as lim
from . import samples as smp

CORNER = lim.CORNER
#: The most passes the expected best is taken of, and the fewest.
M_MAX = lim.M_DEFAULT
M_MIN = 3


def cornering_passes(apex: pd.DataFrame) -> pd.DataFrame:
    """One car's passes at its dry grip-limited corners, dry and wet:
    ``layout_id, corner, date, basis`` and ``value`` (the Doppler lateral
    in the tyres' frame over the load)."""
    cols = [*CORNER, "date", "basis", "value"]
    need = {"lat_tyre_g", "load", "lat_doppler_g", "basis", "date", "car", *CORNER}
    if apex.empty or not need.issubset(apex.columns):
        return pd.DataFrame(columns=cols)
    summary = smp.apex_summary(apex[apex["basis"] == "dry"])
    limited = summary[summary["grip_limited"]][["car", *CORNER]]
    a = apex.merge(limited, on=["car", *CORNER])
    a = a[(a["lat_doppler_g"] > lim.MIN_READING_G) & (a["lat_tyre_g"] > 0.0) & (a["load"] > 0.0)]
    return a.assign(value=a["lat_tyre_g"] / a["load"])[cols].reset_index(drop=True)


def braking_stops(brake: pd.DataFrame, c_drag: float, lat_level: float) -> pd.DataFrame:
    """One car's straight-line stops, dry and wet, in the shape of
    :func:`cornering_passes`: ``corner`` is the stopping place and
    ``value`` the held deceleration from the GPS speed less the drag
    (:func:`~.limits.stops` has the rules; the places kept are those the
    dry stops brake at the limit)."""
    cols = [*CORNER, "date", "basis", "value"]
    need = {"decel_held_g", "v_held_ms", "lat_held_tyre_g", "held_s", "s_on_m", "basis"}
    need |= {"date", "layout_id"}
    ok_level = np.isfinite(lat_level) and lat_level > 0.0
    if brake.empty or not need.issubset(brake.columns) or not ok_level:
        return pd.DataFrame(columns=cols)
    b = brake[
        (brake["held_s"] >= lim.HELD_MIN_S)
        & (brake["lat_held_tyre_g"] < lim.STRAIGHT_SHARE * lat_level)
    ]
    drag = (c_drag if np.isfinite(c_drag) else 0.0) * b["v_held_ms"] ** 2 / G_MS2
    b = b.assign(value=b["decel_held_g"] - drag)
    b = b[b["value"] > lim.MIN_READING_G].copy()
    if b.empty:
        return pd.DataFrame(columns=cols)
    place = pd.Series(0, index=b.index)
    for _, grp in b.groupby("layout_id"):
        at = np.sort(grp["s_on_m"].unique())
        number = dict(zip(at, np.concatenate([[0], np.cumsum(np.diff(at) > lim.PLACE_GAP_M)])))
        place.loc[grp.index] = grp["s_on_m"].map(number).to_numpy()
    b["corner"] = place.astype(int)
    dry = b[b["basis"] == "dry"]
    high = dry.groupby(CORNER)["value"].quantile(0.9).rename("high").reset_index()
    high["top"] = high.groupby("layout_id")["high"].transform("max")
    kept = high[high["high"] >= lim.PLACE_LIMITED_SHARE * high["top"]][CORNER]
    return b.merge(kept, on=CORNER)[cols].reset_index(drop=True)


def best_ratio(passes: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """``(ratio, by_corner)``: per corner the expected best of ``m`` wet
    values over the expected best of ``m`` dry ones (``m`` half the wet
    values, :data:`M_MIN` to :data:`M_MAX`; a corner with fewer of either
    is left out), and the median of that over the corners (NaN without
    one). ``by_corner``: ``layout_id, corner, n_wet, n_dry, m, wet, dry,
    ratio``."""
    cols = [*CORNER, "n_wet", "n_dry", "m", "wet", "dry", "ratio"]
    rows = []
    for key, grp in passes.groupby(CORNER) if len(passes) else []:
        wet = grp.loc[grp["basis"] == "wet", "value"].to_numpy(float)
        dry = grp.loc[grp["basis"] == "dry", "value"].to_numpy(float)
        m = min(M_MAX, len(wet) // 2)
        if m < M_MIN or len(dry) < m:
            continue
        best_wet, best_dry = lim.expected_best(wet, m), lim.expected_best(dry, m)
        rows.append([*key, len(wet), len(dry), m, best_wet, best_dry, best_wet / best_dry])
    by_corner = pd.DataFrame(rows, columns=cols)
    ratio = float(by_corner["ratio"].median()) if len(by_corner) else float("nan")
    return ratio, by_corner


def day_levels(passes: pd.DataFrame) -> pd.DataFrame:
    """Each wet day's own ratio (:func:`best_ratio` with that day's wet
    passes alone): ``date, layout_id, ratio, corners, passes``. With
    cornering passes it is the day's water level as far as the logs give
    it."""
    rows = []
    wet = passes[passes["basis"] == "wet"] if len(passes) else passes
    dry = passes[passes["basis"] == "dry"] if len(passes) else passes
    for (date, layout), grp in wet.groupby(["date", "layout_id"]) if len(wet) else []:
        own = pd.concat([grp, dry[dry["layout_id"] == layout]], ignore_index=True)
        ratio, by_corner = best_ratio(own)
        rows.append(
            {
                "date": str(date),
                "layout_id": str(layout),
                "ratio": ratio,
                "corners": int(len(by_corner)),
                "passes": int(by_corner["n_wet"].sum()) if len(by_corner) else 0,
            }
        )
    return pd.DataFrame(rows, columns=["date", "layout_id", "ratio", "corners", "passes"])


@dataclass
class WetTerms:
    """A car's wet readings (module docstring). ``brake_term`` is
    ``braking / cornering``, NaN where either cannot be read."""

    cornering: float = float("nan")
    braking: float = float("nan")
    corners: pd.DataFrame = field(default_factory=pd.DataFrame)
    places: pd.DataFrame = field(default_factory=pd.DataFrame)
    days: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def brake_term(self) -> float:
        return self.braking / self.cornering

    def to_dict(self) -> dict:
        def _r(v: float) -> float | None:
            return round(float(v), 4) if np.isfinite(v) else None

        return {
            "cornering": _r(self.cornering),
            "braking": _r(self.braking),
            "brake_term": _r(self.brake_term),
            "corners": int(len(self.corners)),
            "places": int(len(self.places)),
            "wet_passes": int(self.corners["n_wet"].sum()) if len(self.corners) else 0,
            "wet_stops": int(self.places["n_wet"].sum()) if len(self.places) else 0,
        }


def wet_terms(
    apex: pd.DataFrame,
    brake: pd.DataFrame,
    c_drag: float,
    lat_level: float,
    *,
    without: str | None = None,
) -> WetTerms:
    """The wet readings of the one car whose tables ``apex`` and ``brake``
    are. ``without`` leaves a wet day out (a fold's reading)."""

    def _held_in(tbl: pd.DataFrame) -> pd.DataFrame:
        if without is None or tbl.empty:
            return tbl
        return tbl[~((tbl["basis"] == "wet") & (tbl["date"].astype(str) == str(without)))]

    corner = cornering_passes(_held_in(apex))
    stop = braking_stops(_held_in(brake), c_drag, lat_level)
    cornering, corners = best_ratio(corner)
    braking, places = best_ratio(stop)
    return WetTerms(cornering, braking, corners, places, day_levels(corner))


__all__ = [
    "M_MAX",
    "M_MIN",
    "WetTerms",
    "best_ratio",
    "braking_stops",
    "cornering_passes",
    "day_levels",
    "wet_terms",
]
