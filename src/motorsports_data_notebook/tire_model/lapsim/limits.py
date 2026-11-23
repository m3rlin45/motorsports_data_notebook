"""The car's cornering limit as its best moments (lap simulation, issue
#312; the driver committee's design, ``docs/lapsim_driver_model.md``
clause 1.1).

The true limit cannot be read from one driver's laps, so the car holds the
best it has been shown to do: per corner the *expected best of m passes*,
the median over the corners. It is a floor under the true limit (whatever
the driver leaves unused at their best moments is still missing from it).

- **A pass** is one dry build lap through one corner of the track's corner
  table (a row of the apex table) with both readings above
  :data:`MIN_READING_G`.
- **Two readings of a pass**, each the lateral the tyres hold over the
  load on them (the tyres' frame: ``lat_tyre_g`` and ``load`` of the apex
  table): the Doppler's, and the accelerometer's after it has been put on
  the Doppler's scale for its session (divided by the session's median of
  ``lat_accel_g / lat_tyre_g``; an accelerometer reads less than the tyres
  hold by the body's roll). The pass's value is their geometric mean.
- **Per corner**, every dry day pooled, the expected largest of m passes
  drawn from its n without replacement (:func:`expected_best`). A maximum
  or a 99th percentile rises as days are added and reads the sensor; a
  90th percentile reads the driver.
- **The sensors' noise.** The independent noise left in the two-reading
  mean has a quarter of the variance of the difference of the two
  readings' logs (taken within a corner and day). A corner's passes are
  drawn towards the corner's mean by ``σ_T / σ``, with ``σ`` the spread of
  the logs of its values and ``σ_T² = σ² −`` that noise, before the
  expected best is taken. An error the two readings share (the line, a
  kerb) is not removed.
- **The car's level** is the median over the grip-limited corners with
  :data:`SUPPORT_FACTOR` × m passes or more, every layout together. A car
  with no such corner (no accelerometer beside the Doppler, or too few
  passes) is read on the Doppler alone, uncorrected, and says so
  (``one_sensor``): that reading is high by the Doppler's own noise.

Nothing here fits anything or touches the solver; :mod:`.car_gate` puts
the level into the car.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import gammaln

from ...tire_etl.kinematics import G_MS2
from . import samples as smp

#: How rare a moment counts as the car's: the best of this many passes
#: (the owner's choice, 2026-11-23); the others are reported beside it.
M_DEFAULT = 20
M_REPORTED = (10, 20, 50)
#: A reading under this (g) is not a pass through a corner at speed.
MIN_READING_G = 0.1
#: A corner's day counts with this many passes or more.
CELL_MIN_PASSES = 5
#: A corner supports m with this many times m passes.
SUPPORT_FACTOR = 2
BOOT_RESAMPLES = 300
BOOT_SEED = 7

CORNER = ["layout_id", "corner"]
#: Stops that begin within this of each other along the track are at one
#: stopping place.
PLACE_GAP_M = 60.0
#: A stop's held part counts from this long.
HELD_MIN_S = 0.4
#: A stop is in a straight line with under this share of the car's
#: lateral level held sideways.
STRAIGHT_SHARE = 0.2
#: A stopping place is braked at the limit when its 90th percentile is
#: within this of the highest of the car's places at the layout (as a
#: corner is grip-limited, :data:`~.samples.GRIP_LIMITED_SHARE`).
PLACE_LIMITED_SHARE = smp.GRIP_LIMITED_SHARE


def expected_best(x: np.ndarray, m: int) -> float:
    """The expected largest of ``m`` values drawn without replacement from
    ``x``: ``Σ w_i·x_(i)`` over the sorted values, ``w_i = C(i−1, m−1) /
    C(n, m)`` the chance that the i-th smallest is the largest drawn. NaN
    with fewer than ``m`` values."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n < m or m < 1:
        return float("nan")
    i = np.arange(m - 1, n)
    # log C(i, m−1) − log C(n, m)
    logw = gammaln(i + 1) - gammaln(m) - gammaln(i - m + 2)
    logw -= gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1)
    return float(np.sum(np.exp(logw) * x[i]))


def best_of_normal(m: int) -> float:
    """The expected largest of ``m`` standard normal values: how many
    spreads above a corner's typical pass its expected best of m sits
    when the passes are bell-shaped."""
    z = np.linspace(-8.0, 8.0, 16001)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    cdf = np.cumsum(pdf) * (z[1] - z[0])
    return float(np.sum(z * m * np.clip(cdf, 0.0, 1.0) ** (m - 1) * pdf) * (z[1] - z[0]))


def passes(
    apex: pd.DataFrame, *, one_sensor: bool = False, cross: bool = False, track: bool = False
) -> pd.DataFrame:
    """The passes of one car's dry apex table as the statistic takes them:
    the rows at its grip-limited corners with the readings it needs,
    ``x1`` and ``x2`` the logs of the two readings over the load (``x2``
    NaN-free unless ``one_sensor``, where only the Doppler's is asked
    for), ``y`` the log of the pass's value, on days with
    :data:`CELL_MIN_PASSES` passes at the corner.

    A reading is the sensor's own largest value in the corner; with
    ``cross`` it is the sensor's value where the other one peaks
    (``lat_tyre_cross_g``, ``lat_accel_cross_g``): the first is raised by
    the sensor's own noise, the second is not and reads low where the
    peaks are apart (:func:`~.samples.apex_rows`). With ``track`` the
    one reading is the pass on the *track's* line (``lat_track_tyre_g``:
    the track's curvature at the apex at the lap's speed there), which
    is what the solver spends the limit on; it is a single reading."""
    need = {"lat_tyre_g", "load", "lat_doppler_g", "lat_accel_g", "basis", "session_id", "date"}
    if cross:
        need |= {"lat_tyre_cross_g", "lat_accel_cross_g"}
    if track:
        need |= {"lat_track_tyre_g"}
        one_sensor = True
    cols = [*CORNER, "date", "session_id", "x1", "x2", "y"]
    if apex.empty or not need.issubset(apex.columns) or (cross and one_sensor):
        return pd.DataFrame(columns=cols)
    a = apex[apex["basis"] == "dry"]
    summary = smp.apex_summary(a)
    limited = summary[summary["grip_limited"]][["car", *CORNER]]
    a = a.merge(limited, on=["car", *CORNER])
    ok = (a["lat_doppler_g"] > MIN_READING_G) & (a["lat_tyre_g"] > 0.0) & (a["load"] > 0.0)
    if not one_sensor:
        ok &= a["lat_accel_g"] > MIN_READING_G
    tyre, accel = (
        ("lat_tyre_cross_g", "lat_accel_cross_g") if cross else ("lat_tyre_g", "lat_accel_g")
    )
    if cross:
        ok &= (a[tyre] > MIN_READING_G) & (a[accel] > MIN_READING_G)
    if track:
        tyre = "lat_track_tyre_g"
        ok &= a[tyre] > MIN_READING_G
    a = a[ok].copy()
    if a.empty:
        return pd.DataFrame(columns=cols)
    a["x1"] = np.log(a[tyre] / a["load"])
    if one_sensor:
        a["x2"] = np.nan
        a["y"] = a["x1"]
    else:
        # the accelerometer on the Doppler's scale, by the session's own peaks
        scale = (a["lat_accel_g"] / a["lat_tyre_g"]).groupby(a["session_id"]).transform("median")
        a["x2"] = np.log(a[accel] / scale / a["load"])
        a["y"] = 0.5 * (a["x1"] + a["x2"])
    a = a[a.groupby([*CORNER, "date"])["y"].transform("size") >= CELL_MIN_PASSES]
    return a[cols].reset_index(drop=True)


def stops(
    brake: pd.DataFrame, c_drag: float, lat_level: float, *, one_sensor: bool = False
) -> pd.DataFrame:
    """The stops of one car's dry braking zones as the braking limit's
    statistic takes them, in the shape of :func:`passes` (``corner`` is
    the stopping place).

    A stop counts when its *held part* (the pressure at 80 % or more of
    the zone's highest; the tables' ``*held*`` columns) lasts
    :data:`HELD_MIN_S` and is in a straight line (:data:`STRAIGHT_SHARE`
    of ``lat_level``). Its two readings are the mean deceleration over
    the held part from the GPS speed and from the accelerometer (put on
    the GPS's scale per session), each less the drag at the held part's
    speed (``c_drag``). Stops that begin within :data:`PLACE_GAP_M` of
    each other are one place, and the places kept are those braked at the
    limit (:data:`PLACE_LIMITED_SHARE`)."""
    need = {"decel_held_g", "decel_held_accel_g", "v_held_ms", "lat_held_tyre_g", "held_s"}
    need |= {"s_on_m", "basis", "session_id", "date", "layout_id"}
    cols = [*CORNER, "date", "session_id", "x1", "x2", "y"]
    ok_level = np.isfinite(lat_level) and lat_level > 0.0
    if brake.empty or not need.issubset(brake.columns) or not ok_level:
        return pd.DataFrame(columns=cols)
    b = brake[brake["basis"] == "dry"]
    b = b[(b["held_s"] >= HELD_MIN_S) & (b["lat_held_tyre_g"] < STRAIGHT_SHARE * lat_level)]
    drag = (c_drag if np.isfinite(c_drag) else 0.0) * b["v_held_ms"] ** 2 / G_MS2
    b = b.assign(r1=b["decel_held_g"] - drag, r2=np.nan)
    if not one_sensor:
        b = b[b["decel_held_accel_g"].notna()]
        scale = (b["decel_held_accel_g"] / b["decel_held_g"]).groupby(b["session_id"])
        b = b.assign(r2=b["decel_held_accel_g"] / scale.transform("median") - drag.reindex(b.index))
        b = b[b["r2"] > MIN_READING_G]
    b = b[b["r1"] > MIN_READING_G].copy()
    if b.empty:
        return pd.DataFrame(columns=cols)
    place = pd.Series(0, index=b.index)
    for _, grp in b.groupby("layout_id"):
        at = np.sort(grp["s_on_m"].unique())
        number = dict(zip(at, np.concatenate([[0], np.cumsum(np.diff(at) > PLACE_GAP_M)])))
        place.loc[grp.index] = grp["s_on_m"].map(number).to_numpy()
    b["corner"] = place.astype(int)
    high = b.groupby(CORNER)["r1"].transform(lambda x: x.quantile(0.9))
    b = b[high >= PLACE_LIMITED_SHARE * high.groupby(b["layout_id"]).transform("max")].copy()
    b["x1"] = np.log(b["r1"])
    b["x2"] = np.nan if one_sensor else np.log(b["r2"])
    b["y"] = b["x1"] if one_sensor else 0.5 * (b["x1"] + b["x2"])
    b = b[b.groupby([*CORNER, "date"])["y"].transform("size") >= CELL_MIN_PASSES]
    return b[cols].reset_index(drop=True)


def _corner_stat(
    y: np.ndarray, ssd: float, dof: int, m: int, correct: bool
) -> tuple[float, float, float, float]:
    """``(raw, level, sd, sd_noise)`` of one corner's passes: ``y`` the
    logs of their values, ``ssd`` and ``dof`` the sum of squares of the
    two readings' difference about each day's mean and its degrees of
    freedom (NaN and 0 without two readings)."""
    n = len(y)
    sd = float(np.std(y, ddof=1)) if n > 1 else float("nan")
    level = raw = expected_best(np.exp(y), m)
    noise = float("nan")
    if correct and dof > 0 and np.isfinite(ssd) and n > 1:
        noise = float(np.sqrt(ssd / dof / 4.0))
        if sd > 0.0:
            keep = float(np.sqrt(max(sd * sd - noise * noise, 0.0))) / sd
            level = expected_best(np.exp(y.mean() + keep * (y - y.mean())), m)
    return raw, level, sd, noise


def _pieces(got: pd.DataFrame) -> dict[tuple, dict[str, tuple[np.ndarray, float]]]:
    """``corner -> day -> (y, ssd)``: the logs of a corner's values on a
    day and the sum of squares of the two readings' difference about that
    day's mean (NaN without two readings)."""
    out: dict[tuple, dict[str, tuple[np.ndarray, float]]] = {}
    for (layout, corner, date), grp in got.groupby([*CORNER, "date"], sort=True):
        ssd = float("nan")
        if grp["x2"].notna().all():
            d = (grp["x1"] - grp["x2"]).to_numpy(float)
            ssd = float(np.sum((d - d.mean()) ** 2))
        out.setdefault((layout, corner), {})[str(date)] = (grp["y"].to_numpy(float), ssd)
    return out


def _corner_rows(
    pieces: dict[tuple, dict[str, tuple[np.ndarray, float]]],
    m: int,
    correct: bool,
    times: dict[str, int] | None = None,
) -> list[list]:
    """Per corner ``[layout, corner, n, raw, level, sd, sd_noise,
    supports]``; ``times`` says how often each day counts (a resample of
    the days; None: once each)."""
    rows = []
    for key, days in pieces.items():
        ys, ssd, dof = [], 0.0, 0
        for date, (y, part) in days.items():
            k = 1 if times is None else times.get(date, 0)
            if k:
                ys.extend([y] * k)
                ssd += k * part
                dof += k * (len(y) - 1)
        if not ys:
            continue
        y_all = np.concatenate(ys)
        raw, level, sd, noise = _corner_stat(y_all, ssd, dof, m, correct)
        rows.append([*key, len(y_all), raw, level, sd, noise, len(y_all) >= SUPPORT_FACTOR * m])
    return rows


def corner_levels(got: pd.DataFrame, m: int, *, correct: bool = True) -> pd.DataFrame:
    """Per corner of ``got`` (:func:`passes`): ``n`` passes, ``raw`` (the
    expected best of m of the values as read), ``level`` (the same with
    the sensors' noise taken out, where there are two readings and
    ``correct``), ``sd`` (the spread of the logs), ``sd_noise`` (the
    independent noise left in the two-reading mean) and ``supports``
    (:data:`SUPPORT_FACTOR` × m passes or more)."""
    cols = [*CORNER, "n", "raw", "level", "sd", "sd_noise", "supports"]
    return pd.DataFrame(_corner_rows(_pieces(got), m, correct), columns=cols)


def _median_level(rows: list[list]) -> tuple[float, float]:
    """``(level, raw)``: the medians over the supporting corners."""
    own = [r for r in rows if r[-1]]
    if not own:
        return float("nan"), float("nan")
    return float(np.median([r[4] for r in own])), float(np.median([r[3] for r in own]))


@dataclass
class Limit:
    """A car's cornering limit (module docstring): ``level`` with the
    noise taken out and ``raw`` as read, both NaN when no corner supports
    ``m``; the supporting ``corners`` and their ``passes``; the
    ``layouts`` that have a supporting corner (the others borrow the
    car's level); ``one_sensor`` when it was read on the Doppler alone;
    ``lo`` and ``hi``, the 5 % and 95 % ends of a day bootstrap (NaN when
    none was asked for); ``by_corner``, the workings."""

    level: float
    raw: float
    m: int
    corners: int = 0
    passes: int = 0
    layouts: tuple[str, ...] = ()
    one_sensor: bool = False
    lo: float = float("nan")
    hi: float = float("nan")
    by_corner: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def found(self) -> bool:
        return bool(np.isfinite(self.level))

    def to_json(self) -> dict:
        def r(v: float) -> float | None:
            return round(float(v), 4) if np.isfinite(v) else None

        return {
            "level": r(self.level),
            "raw": r(self.raw),
            "m": self.m,
            "corners": self.corners,
            "passes": self.passes,
            "layouts": list(self.layouts),
            "one_sensor": self.one_sensor,
            "lo": r(self.lo),
            "hi": r(self.hi),
        }


def car_limit(
    apex: pd.DataFrame,
    m: int = M_DEFAULT,
    *,
    boot: int = 0,
    seed: int = BOOT_SEED,
    correct: bool = True,
    cross: bool = False,
    track: bool = False,
) -> Limit:
    """The cornering limit of the one car whose apex table ``apex`` is
    (:class:`Limit`). Read on two readings where a corner supports ``m``
    with them, else on the Doppler alone (``one_sensor``). ``boot`` asks
    for that many resamples of the car's days for ``lo`` and ``hi``.
    ``cross`` reads each sensor where the other peaks (:func:`passes`);
    it has no one-sensor form. ``track`` reads the passes on the track's
    line, a single reading."""
    return _limit(
        lambda single: passes(apex, one_sensor=single, cross=cross, track=track),
        m,
        singles=(True,) if track else (False,) if cross else (False, True),
        boot=boot,
        seed=seed,
        correct=correct,
    )


def braking_best(
    brake: pd.DataFrame,
    c_drag: float,
    lat_level: float,
    m: int = M_DEFAULT,
    *,
    boot: int = 0,
    seed: int = BOOT_SEED,
) -> Limit:
    """The best braking the car has been shown to hold (the driver
    committee's clause 1.2): :func:`car_limit`'s statistic on
    :func:`stops`, the expected best of ``m`` stops per stopping place,
    the median over the places with enough of them. ``corners`` and
    ``by_corner`` of the result are stopping places."""
    return _limit(
        lambda single: stops(brake, c_drag, lat_level, one_sensor=single),
        m,
        singles=(False, True),
        boot=boot,
        seed=seed,
        correct=True,
    )


def _limit(
    frame: Callable[[bool], pd.DataFrame],
    m: int,
    *,
    singles: tuple[bool, ...],
    boot: int,
    seed: int,
    correct: bool,
) -> Limit:
    """:class:`Limit` of the passes ``frame(one_sensor)`` gives, on two
    readings where a corner supports ``m`` with them, else on one."""
    cols = [*CORNER, "n", "raw", "level", "sd", "sd_noise", "supports"]
    for single in singles:
        got = frame(single)
        pieces = _pieces(got)
        rows = _corner_rows(pieces, m, correct and not single)
        level, raw = _median_level(rows)
        if np.isfinite(level):
            break
    else:
        return Limit(float("nan"), float("nan"), m, by_corner=pd.DataFrame(rows, columns=cols))
    corners = pd.DataFrame(rows, columns=cols)
    own = corners[corners["supports"]]
    out = Limit(
        level,
        raw,
        m,
        corners=int(len(own)),
        passes=int(own["n"].sum()),
        layouts=tuple(sorted(str(k) for k in own["layout_id"].unique())),
        one_sensor=single,
        by_corner=corners,
    )
    if boot > 0:
        rng = np.random.default_rng(seed)
        days = sorted({d for by_day in pieces.values() for d in by_day})
        vals = []
        for _ in range(boot):
            pick, count = np.unique(rng.choice(days, len(days), replace=True), return_counts=True)
            times = {str(d): int(c) for d, c in zip(pick, count)}
            vals.append(_median_level(_corner_rows(pieces, m, correct and not single, times))[0])
        if np.isfinite(vals).any():
            out.lo, out.hi = (float(v) for v in np.nanpercentile(vals, [5.0, 95.0]))
    return out


__all__ = [
    "BOOT_RESAMPLES",
    "CELL_MIN_PASSES",
    "Limit",
    "M_DEFAULT",
    "M_REPORTED",
    "MIN_READING_G",
    "SUPPORT_FACTOR",
    "best_of_normal",
    "braking_best",
    "car_limit",
    "corner_levels",
    "expected_best",
    "passes",
    "stops",
]
