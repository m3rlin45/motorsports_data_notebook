"""Inference: turn (track, car, lap, ambient, target hot pressures) → cold pressures.

Reads the JSON artifact written by :func:`build_warmup_table`, applies the
fallback chain documented in the plan (K[car, corner] → K[car] → global; track
⟨g²⟩ falls back to (track) mean → global), and
returns per-corner :class:`Prediction` records with full provenance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..tire_etl.paths import default_dataset_root
from .energy_balance import (
    warmup_two_stage_c,
    P_ATM_BAR,
    T_ZERO_C_TO_K,
    gay_lussac_cold_pressure_bar,
    t_effective_c,
    t_road_proxy_c,
    warmup_curve_c,
)

CORNERS = ("fl", "fr", "rl", "rr")

#: The entry tables per raw car beside the pooled ones (#383,
#: ``warmup_table.RAW_CAR_ENTRIES``): the same rows, ``car`` the car the
#: session names, for each raw car ``car_aliases`` pools under another name.
RAW_CAR_TABLES = {
    "g2_typ_by_track_car_cond": "g2_typ_by_track_raw_car_cond",
    "lap_time_typ_by_track_car_cond": "lap_time_typ_by_track_raw_car_cond",
    "outlap_typ_by_track_car_cond": "outlap_typ_by_track_raw_car_cond",
}

#: Where an entry row's numbers come from (its ``source``, absent meaning
#: logged): ``sim`` is a lap simulated on a committed track, whose road bank
#: is processed from GSI's elevation tiles, so a prediction that reads one
#: carries GSI's attribution (``data/lapsim/README.md``).
SIM = "sim"
LOGGED = "logged"

#: How a simulated pace curve answers the lap time a prediction is for
#: (``g2_vs_lap_time.support``; the lap simulation's ``pace`` statuses): ``ok``,
#: ``outside_support`` (answered, outside the car's logged range of pace
#: there) or ``no_solution`` (faster than the curve's fastest lap: answered
#: at it, the clamp).
PACE_OK = "ok"
PACE_OUTSIDE_SUPPORT = "outside_support"
PACE_NO_SOLUTION = "no_solution"


@dataclass(frozen=True)
class Prediction:
    corner: str
    cold_pressure_bar: float
    predicted_hot_temp_c: float
    target_hot_pressure_bar: float
    K_kelvin_per_g2: float
    tau_sec: float
    g2_typ: float
    lap_time_typ_s: float
    t_at_lap_n_s: float
    warmup_frac: float
    delta_t_inf_kelvin: float
    t_eff_c: float
    t_air_c: float
    t_road_c: float
    t_cold_c: float  # temperature the tire is at when you measure/set cold pressure
    t_start_c: float  # warmup-curve initial condition (== t_cold_c: the tire's temperature now)
    K_source_bucket: tuple[str, ...]
    K_from_prior: bool
    K_n_samples: int
    K_stderr: float
    tau_stderr: float
    # ---- Target-lap-time feature (schema v3); defaults preserve v2 behavior ----
    target_lap_time_s: float | None = None
    g2_scale: float = 1.0  # multiplier applied to g2_typ (1.0 when no target given)
    g2_pace_source: str | None = None  # "curve" | "exponent" | None (no target)
    # ---- Out-lap segment (v0.26): integrated first, from t_start_c ----
    outlap_time_s: float = 0.0
    outlap_g2: float = 0.0
    outlap_source: str | None = None
    t_after_outlap_c: float | None = None
    # ---- Provenance of the entry rows read (cutover): "sim" | "logged" ----
    heat_provenance: str = LOGGED
    lap_time_provenance: str = LOGGED
    outlap_provenance: str | None = None  # None: no out-lap row read
    uses_simulation: bool = False  # any of the three is "sim": show GSI's attribution
    pace_status: str | None = None  # a simulated curve's status at the lap time; else None


def _load_model(dataset_root: Path | None) -> dict[str, Any]:
    root = Path(dataset_root) if dataset_root else default_dataset_root()
    artifact = root / "tire_model.json"
    if not artifact.exists():
        raise FileNotFoundError(
            f"No tire_model.json at {artifact}. Run `just tire-build-warmup-table` first."
        )
    with artifact.open() as f:
        model: dict[str, Any] = json.load(f)
    return model


DEFAULT_CONDITION = "dry"

# Per-condition fallback order: when the requested condition has no fit,
# walk this list. Each list begins with the requested condition and walks
# through physically-closest neighbors before giving up.
_CONDITION_FALLBACK = {
    "dry": ("dry",),
    "damp": ("damp", "dry"),  # if no damp data, fall back to dry
    "wet": ("wet", "damp", "dry"),  # prefer damp over dry when wet data missing
}


def _condition_chain(condition: str) -> tuple[str, ...]:
    return _CONDITION_FALLBACK.get(condition, (condition, "dry"))


def fitted_car(model: dict[str, Any], car: str) -> str:
    """The car the model's constants were fitted as: a raw car pooled
    under another name (``car_aliases``, e.g. KK-F → FJ) resolves to it,
    any other name passes through."""
    return str(model.get("car_aliases", {}).get(car, car))


def _entry_row(
    model: dict[str, Any],
    table: str,
    track: str,
    car: str,
    condition: str,
    *,
    exact: bool = True,
) -> dict[str, Any] | None:
    """The row of entry table ``table`` that answers (track, car,
    condition): a raw car's own row (:data:`RAW_CAR_TABLES`) where it has
    one, else the fitted car's. ``car`` may be either name; an artifact
    without raw-car tables reads the fitted car's row, as before.

    A simulated row sits beside the logged one and answers only its own
    exact condition (schema §9), chosen by its ``source`` (absent:
    logged), never by its place in the table. At the ``exact`` step the
    order is raw sim, raw logged, pooled sim, pooled logged; at any other
    step (a condition fallback) only logged rows count, the raw car's
    before the pooled car's."""
    fit = fitted_car(model, car)
    names = [(RAW_CAR_TABLES[table], car)] if fit != car else []
    kinds = (SIM, LOGGED) if exact else (LOGGED,)
    for name, who in names + [(table, fit)]:
        rows: list[dict[str, Any]] = [
            r
            for r in model.get(name) or []
            if r["track_canonical"] == track and r["car"] == who and r["condition"] == condition
        ]
        for kind in kinds:
            for r in rows:
                if (SIM if _is_sim(r) else LOGGED) == kind:
                    return r
    return None


def _entry_rows_at_track(
    model: dict[str, Any], table: str, track: str, car: str
) -> list[dict[str, Any]]:
    """Every condition's logged row of ``table`` at ``track`` for the car
    whose entries answer there: the raw car's own where it has any, else
    the fitted car's (the ``track_car_pooled`` fallback; an average, so a
    simulated row is not counted, :func:`_logged`)."""
    fit = fitted_car(model, car)
    if fit != car:
        own = _logged(
            r
            for r in model.get(RAW_CAR_TABLES[table]) or []
            if r["track_canonical"] == track and r["car"] == car
        )
        if own:
            return own
    return _logged(
        r for r in model.get(table) or [] if r["track_canonical"] == track and r["car"] == fit
    )


def _is_sim(row: dict[str, Any]) -> bool:
    return str(row.get("source") or LOGGED) == SIM


def _logged(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The logged rows of ``rows``: what an averaged fallback step reads
    (the lead's ruling of 2026-11-27), so that a car with no simulated
    entry of its own reads what it read before the simulated rows came."""
    return [r for r in rows if not _is_sim(r)]


def _answering_rows(
    model: dict[str, Any], table: str, track: str, car: str, condition: str, *, global_step: bool
) -> tuple[list[dict[str, Any]], str]:
    """The rows of entry table ``table`` that answer (track, car,
    condition), and the step that found them: one row by the condition
    chain (``exact``, ``fallback(<cond>)``; a raw car's own first, and a
    simulated row only at the exact step, :func:`_entry_row`), else every
    condition's row at the track for the
    car (``track_car_pooled``), else the pooled rows at the track
    (``track_pooled``), else, with ``global_step``, every pooled row
    (``global``; none: ``global`` with no rows). The averaged steps read
    logged rows only, and a step the filter leaves empty is skipped. Every
    lookup walks this one chain, and a prediction's provenance is read from
    the rows it found."""
    for cond in _condition_chain(condition):
        r = _entry_row(model, table, track, car, cond, exact=cond == condition)
        if r is not None:
            return [r], "exact" if cond == condition else f"fallback({cond})"
    same_tc = _entry_rows_at_track(model, table, track, car)
    if same_tc:
        return same_tc, "track_car_pooled"
    rows = _logged(model.get(table) or [])
    same_t = [r for r in rows if r["track_canonical"] == track]
    if same_t:
        return same_t, "track_pooled"
    return (rows if global_step else []), "global"


#: The steps of :func:`_answering_rows` that answer with an average of rows.
_AVERAGED_STEPS = ("track_car_pooled", "track_pooled", "global")


def row_provenance(rows: Iterable[dict[str, Any]]) -> str:
    """``sim`` if any of ``rows`` is simulated (``source``), else ``logged``."""
    return SIM if any(_is_sim(r) for r in rows) else LOGGED


def _lookup_tau(
    model: dict[str, Any], car: str, corner: str, condition: str
) -> tuple[float, float, tuple[str, ...]]:
    """Return (τ_sec, stderr, source_bucket).

    Fallback walks the requested condition's neighbor chain, then (car, corner),
    then (car), then prior.
    """
    table = model["tau_sec_by_car_corner_cond"]
    for cond in _condition_chain(condition):
        for r in table:
            if r["car"] == car and r["corner"] == corner and r["condition"] == cond:
                return (
                    float(r["value_seconds"]),
                    float(r["stderr_seconds"]),
                    (car, corner, cond),
                )
    same_cc = [r for r in table if r["car"] == car and r["corner"] == corner]
    if same_cc:
        vals = [r["value_seconds"] for r in same_cc]
        return float(sum(vals) / len(vals)), 0.0, (car, corner)
    same_car = [r for r in table if r["car"] == car]
    if same_car:
        vals = [r["value_seconds"] for r in same_car]
        return float(sum(vals) / len(vals)), 0.0, (car,)
    return float(model["priors_when_no_fit"]["tau_sec_seconds"]), 0.0, ()


def _lookup_k(
    model: dict[str, Any], car: str, corner: str, condition: str
) -> tuple[float, float, int, bool, tuple[str, ...]]:
    """Return (K, stderr, n_samples, from_prior, source_bucket_tuple).

    Fallback chain (condition-aware):
      1. (car, corner, condition)
      2. (car, corner, "dry")
      3. (car, corner) — mean across whatever conditions exist
      4. (car) — mean across all (corner, condition)
      5. prior
    """
    table = model["K_buckets"]
    for cond in _condition_chain(condition):
        for r in table:
            k = r["key"]
            if k["car"] == car and k["corner"] == corner and k["condition"] == cond:
                return (
                    float(r["value_kelvin_per_g2"]),
                    float(r["stderr_kelvin_per_g2"]),
                    int(r["n_samples"]),
                    bool(r["from_prior"]),
                    (car, corner, cond),
                )
    same_cc = [r for r in table if r["key"]["car"] == car and r["key"]["corner"] == corner]
    if same_cc:
        vals = [r["value_kelvin_per_g2"] for r in same_cc]
        n = sum(int(r["n_samples"]) for r in same_cc)
        return float(sum(vals) / len(vals)), 0.0, n, False, (car, corner)
    same_car = [r for r in table if r["key"]["car"] == car]
    if same_car:
        vals = [r["value_kelvin_per_g2"] for r in same_car]
        n = sum(int(r["n_samples"]) for r in same_car)
        return float(sum(vals) / len(vals)), 0.0, n, False, (car,)
    prior_k = float(model["priors_when_no_fit"]["K_kelvin_per_g2"])
    return prior_k, 0.0, 0, True, ()


def _corner_value(entry: dict[str, Any], key: str, fallback_key: str, corner: str | None) -> float:
    """Schema v5 entries carry a per-corner dict next to the corner-blind value."""
    per = entry.get(key)
    if corner is not None and isinstance(per, dict) and corner in per:
        return float(per[corner])
    return float(entry[fallback_key])


def _lookup_g2(
    model: dict[str, Any], track: str, car: str, condition: str, corner: str | None = None
) -> tuple[float, int, str]:
    """Return (g2_typ, n_laps_used, source) — the corner's ``q_typ_by_corner``
    value on a schema-v5 artifact when ``corner`` is given.

    Fallback walks the condition chain ((wet) → damp → dry, (damp) → dry,
    (dry) → dry), then (track, car) pooled across conditions, then (track),
    then global (:func:`_answering_rows`). A raw car (``car_aliases``) reads
    its own entry before the pooled car's at each of the first two steps
    (:func:`_entry_row`); the track and global steps average the pooled rows.
    """
    rows, tag = _answering_rows(
        model, "g2_typ_by_track_car_cond", track, car, condition, global_step=True
    )
    if not rows:
        return 0.7, 0, "global"
    vals = [_corner_value(r, "q_typ_by_corner", "g2_typ", corner) for r in rows]
    if tag not in _AVERAGED_STEPS:
        return vals[0], int(rows[0]["n_laps_used"]), tag
    n = 0 if tag == "global" else sum(int(r["n_laps_used"]) for r in rows)
    return float(sum(vals) / len(vals)), n, tag


def _lookup_compound_k(
    model: dict[str, Any], car: str, compound: str, corner: str, condition: str
) -> tuple[float, float, int, tuple[str, ...]] | None:
    """Compound-specific K via the condition chain, or None if unavailable.

    Returns (value, stderr, n_laps, source_bucket_tuple).
    """
    table = model.get("K_by_car_compound_corner_cond", [])
    for cond in _condition_chain(condition):
        for r in table:
            if (
                r["car"] == car
                and r["compound"] == compound
                and r["corner"] == corner
                and r["condition"] == cond
            ):
                return (
                    float(r["value_kelvin_per_g2"]),
                    float(r.get("stderr_kelvin_per_g2", 0.0)),
                    int(r.get("n_laps", 0)),
                    (car, compound, corner, cond),
                )
    return None


def _interp_clamped(x: float, xs: list[float], ys: list[float]) -> float:
    """Piecewise-linear interpolation, clamped to the endpoints.

    Endpoint clamping is deliberate: beyond the fastest lap ever driven the
    curve has no support, so the prediction saturates at the fastest
    observed energy level rather than extrapolating. Must stay in lockstep
    with the C# and web-app implementations (pinned by the parity fixture).
    """
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def _lookup_g2_pace_curve(
    model: dict[str, Any], track: str, car: str, condition: str
) -> dict[str, Any] | None:
    """The bucket's simulated-lap ``g2_vs_lap_time`` curve, via the condition
    chain: the curve of the row :func:`_lookup_g2` reads (a raw car's own
    where it has an entry there)."""
    for cond in _condition_chain(condition):
        r = _entry_row(model, "g2_typ_by_track_car_cond", track, car, cond, exact=cond == condition)
        if r is not None:
            curve = r.get("g2_vs_lap_time")
            return curve if isinstance(curve, dict) else None
    return None


def _g2_pace_scale(
    model: dict[str, Any],
    track: str,
    car: str,
    condition: str,
    lap_time_typ_s: float,
    target_lap_time_s: float,
    corner: str | None = None,
) -> tuple[float, str]:
    """Multiplier on the typical heat input for a target lap time. Returns
    (scale, source).

    Preferred: ratio along the bucket's simulated-lap curve (``pace_sim``),
    the corner's own series when ``corner`` is given and the curve carries
    ``g2_by_corner``, else the corner-mean ``g2``; anchored at
    lap_time_typ_s so no-target and target==typical agree exactly.
    Fallback: the physics exponent. Clamped either way.
    """
    pace_cfg = model.get("g2_lap_time_model", {})
    clamp = pace_cfg.get("multiplier_clamp", {})
    lo = float(clamp.get("min", 0.4))
    hi = float(clamp.get("max", 2.5))

    curve = _lookup_g2_pace_curve(model, track, car, condition)
    if curve is not None:
        xs = [float(v) for v in curve["lap_time_s"]]
        series = curve["g2"]
        by_corner = curve.get("g2_by_corner")
        if corner is not None and isinstance(by_corner, dict) and corner in by_corner:
            series = by_corner[corner]
        ys = [float(v) for v in series]
        ref = _interp_clamped(lap_time_typ_s, xs, ys)
        if ref > 0:
            scale = _interp_clamped(target_lap_time_s, xs, ys) / ref
            return min(hi, max(lo, scale)), "curve"

    exponent = float(pace_cfg.get("default_exponent", 3.0))
    scale = (lap_time_typ_s / target_lap_time_s) ** exponent
    return min(hi, max(lo, scale)), "exponent"


def _lookup_outlap(
    model: dict[str, Any], track: str, car: str, condition: str, corner: str | None = None
) -> tuple[float, float, int, str] | None:
    """Typical out-lap ``(moving_s, g2, n, source)`` for the bucket, walking
    the condition chain, then (track, car) pooled, then (track) pooled.
    ``None`` when the artifact has no out-lap table or no data for the
    track (the out-lap is then treated as zero-length). A raw car reads
    its own out-lap first, as :func:`_lookup_g2` does."""
    rows, tag = _answering_rows(
        model, "outlap_typ_by_track_car_cond", track, car, condition, global_step=False
    )
    if not rows:
        return None
    if tag not in _AVERAGED_STEPS:
        r = rows[0]
        return (
            float(r["outlap_moving_s"]),
            _corner_value(r, "outlap_q_by_corner", "outlap_g2", corner),
            int(r["n_laps_used"]),
            tag,
        )
    n = sum(int(r["n_laps_used"]) for r in rows)
    return (
        float(sum(r["outlap_moving_s"] for r in rows) / len(rows)),
        float(
            sum(_corner_value(r, "outlap_q_by_corner", "outlap_g2", corner) for r in rows)
            / len(rows)
        ),
        n,
        tag,
    )


def _lookup_lap_time(
    model: dict[str, Any], track: str, car: str, condition: str
) -> tuple[float, int, str]:
    """Typical lap time ``(seconds, n, source)``, by the chain of
    :func:`_lookup_g2` (a raw car's own entry first); 90 s where nothing
    answers."""
    rows, tag = _answering_rows(
        model, "lap_time_typ_by_track_car_cond", track, car, condition, global_step=False
    )
    if not rows:
        return 90.0, 0, "global"
    if tag not in _AVERAGED_STEPS:
        return float(rows[0]["lap_time_typ_s"]), int(rows[0]["n_laps_used"]), tag
    vals = [r["lap_time_typ_s"] for r in rows]
    n = sum(int(r["n_laps_used"]) for r in rows)
    return float(sum(vals) / len(vals)), n, tag


def pace_status(
    model: dict[str, Any], track: str, car: str, condition: str, lap_time_s: float
) -> str | None:
    """How the simulated pace curve a prediction reads answers
    ``lap_time_s`` (the target, else the typical lap time): ``no_solution``
    under its fastest knot (answered there, the clamp), ``outside_support``
    outside its ``support`` (the lap times of the car's logged range of use
    at the layout) or past its slowest knot, else ``ok``; ``None`` where the
    row the pace scale reads is not simulated (or is an average of rows).
    The lap simulation's ``SimSource.status`` in lap-time terms. Display
    only: no number depends on it."""
    rows, tag = _answering_rows(
        model, "g2_typ_by_track_car_cond", track, car, condition, global_step=True
    )
    if tag in _AVERAGED_STEPS or row_provenance(rows) != SIM:
        return None
    curve = rows[0].get("g2_vs_lap_time")
    if not isinstance(curve, dict):
        return None
    xs = [float(v) for v in curve["lap_time_s"]]
    t = float(lap_time_s)
    if not t >= xs[0]:
        return PACE_NO_SOLUTION
    support = curve.get("support") or {}
    lo, hi = support.get("lap_time_min_s"), support.get("lap_time_max_s")
    if lo is None or hi is None or t > xs[-1] or not float(lo) <= t <= float(hi):
        return PACE_OUTSIDE_SUPPORT
    return PACE_OK


def predict_cold_pressure(
    *,
    track: str,
    car: str,
    lap_within_stint: int,
    target_hot_pressure_bar: dict[str, float],
    ambient_temp_c: float,
    cold_tire_temp_c: float | None = None,
    track_condition: str = "dry",
    track_temp_c: float | None = None,
    cloud_cover_pct: float | None = None,
    g2_typ_override: float | None = None,
    lap_time_typ_override_s: float | None = None,
    target_lap_time_s: float | None = None,
    compound: str | None = None,
    include_outlap: bool = True,
    outlap_time_s: float | None = None,
    dataset_root: Path | None = None,
    _model: dict[str, Any] | None = None,
) -> dict[str, Prediction]:
    """Predict per-corner cold pressure for a target lap.

    ``lap_within_stint`` counts flying laps: N = 1 is the first full lap
    after the out-lap. The out-lap (pit exit to the first start/finish
    crossing) is integrated first, from ``cold_tire_temp_c``, using the
    bucket's typical out-lap rolling time and g² (``include_outlap``;
    ``outlap_time_s`` overrides the duration). Pre-v0.26 artifacts carry no
    out-lap table and behave as before.

    Parameters
    ----------
    ambient_temp_c
        Air temperature for the upcoming session. Drives the T_road sun-proxy
        and the T_eff baseline used in the warmup curve.
    target_lap_time_s
        Optional pace target for the stint. When given, time-on-track uses
        it directly (``t = N * target_lap_time_s``) and ⟨g²⟩ is scaled by
        ``(lap_time_typ_s / target_lap_time_s) ** g2_lap_time_exponent``
        (clamped per the artifact's ``g2_lap_time_model.multiplier_clamp``)
        — running faster than the historical typical pace puts more energy
        into the tires, and vice versa. Omitted: v2 behavior.
    compound
        The tire on the car (canonical name, e.g. "A052", "RE-71RS") —
        every run has one compound on all four corners. When the artifact
        carries a fitted compound K for (car, compound, corner, condition)
        it replaces the pooled K; unknown compounds fall back to pooled.
    cold_tire_temp_c
        Optional temperature the tire is at right *now*, when you set the
        cold pressure and roll out. Defaults to ``ambient_temp_c`` when
        omitted. It is used twice, consistently: as the cold side of the
        Gay-Lussac inversion and as the initial condition of the warmup
        curve (``T(0) = T_start``). Tires still warm from the previous run
        reach their hot temperature sooner and end hotter; enter the TPMS
        reading from the pit lane to capture that.
    track_condition
        One of ``"dry"`` (default), ``"damp"`` (light drizzle, 0.1–1 mm/hr),
        or ``"wet"`` (≥ 1 mm/hr). Picks the condition-specific K, τ_sec,
        ⟨g²⟩, and lap_time_typ. Falls back to ``"dry"`` per-bucket when
        the requested condition has no data.
    """
    model = _model if _model is not None else _load_model(dataset_root)

    # Alias-pooled cars (e.g. KK-F / KK-SII -> FJ): the constants are the
    # label the model was fitted with; the entries (heat, pace curve, lap
    # time, out-lap) are the raw car's own where the artifact has them.
    entry_car = car
    car = fitted_car(model, car)

    cond = str(track_condition or "dry").lower()
    if cond not in {"dry", "damp", "wet"}:
        raise ValueError(f"track_condition must be one of dry/damp/wet; got {track_condition!r}")

    # Lookups (all condition-aware where applicable)
    lap_time_typ_s, lt_n, _lt_source = _lookup_lap_time(model, track, entry_car, cond)
    if lap_time_typ_override_s is not None:
        lap_time_typ_s = float(lap_time_typ_override_s)
    w_road = float(model["energy_balance"]["w_road"])
    sun_factor = float(model["energy_balance"]["t_road_proxy"].get("sun_factor_default", 1.0))
    delta_sun_max_c = float(model["energy_balance"]["t_road_proxy"].get("delta_sun_max_c", 10.0))

    # T_road sourcing
    if track_temp_c is not None:
        t_road_c = float(track_temp_c)
    else:
        t_road_c = t_road_proxy_c(
            t_air_c=ambient_temp_c,
            cloud_cover_pct=cloud_cover_pct,
            sun_factor=sun_factor,
            delta_sun_max_c=delta_sun_max_c,
        )
    t_eff_c = t_effective_c(t_air_c=ambient_temp_c, t_road_c=t_road_c, w_road=w_road)

    # "What's the tire at right now?" — defaults to T_air (a rested tire
    # equilibrates to the air, not to the sun-warmed asphalt). Used both as
    # the Gay-Lussac cold side and as the warmup curve's initial condition.
    t_cold_c = float(cold_tire_temp_c) if cold_tire_temp_c is not None else ambient_temp_c
    t_start_c = t_cold_c

    # Target-lap-time feature: pace sets both time-on-track and tire energy.
    # The heat multiplier is per corner (the simulated lap's force shares
    # load the corners differently); the clock is shared.
    g2_pace_source: str | None = None
    lap_time_for_clock_s = lap_time_typ_s
    target: float | None = None
    if target_lap_time_s is not None:
        target = float(target_lap_time_s)
        if target <= 0:
            raise ValueError(f"target_lap_time_s must be > 0; got {target_lap_time_s!r}")
        lap_time_for_clock_s = target

    # Time at end of flying lap N (the out-lap is a separate segment)
    t_at_lap_n_s = float(lap_within_stint) * lap_time_for_clock_s

    if outlap_time_s is not None and float(outlap_time_s) < 0:
        raise ValueError(f"outlap_time_s must be >= 0; got {outlap_time_s!r}")

    # Where the entry rows read come from (the overrides replace a value, not
    # the rows read): a simulated one carries GSI's attribution.
    heat_rows, _ = _answering_rows(
        model, "g2_typ_by_track_car_cond", track, entry_car, cond, global_step=True
    )
    lap_rows, _ = _answering_rows(
        model, "lap_time_typ_by_track_car_cond", track, entry_car, cond, global_step=False
    )
    out_rows = (
        _answering_rows(
            model, "outlap_typ_by_track_car_cond", track, entry_car, cond, global_step=False
        )[0]
        if include_outlap
        else []
    )
    heat_prov, lap_prov = row_provenance(heat_rows), row_provenance(lap_rows)
    out_prov = row_provenance(out_rows) if out_rows else None
    uses_sim = SIM in (heat_prov, lap_prov, out_prov)
    status = pace_status(model, track, entry_car, cond, lap_time_for_clock_s)

    out: dict[str, Prediction] = {}
    for corner in CORNERS:
        if corner not in target_hot_pressure_bar:
            raise KeyError(f"target_hot_pressure_bar missing corner {corner!r}")
        target_hot = float(target_hot_pressure_bar[corner])
        # Driving intensity for this corner (schema v5: per corner; older
        # artifacts carry one value for all four), scaled by the pace target.
        g2_typ, g2_n, _g2_source = _lookup_g2(model, track, entry_car, cond, corner)
        if g2_typ_override is not None:
            g2_typ = float(g2_typ_override)
        g2_scale = 1.0
        if target is not None:
            g2_scale, g2_pace_source = _g2_pace_scale(
                model, track, entry_car, cond, lap_time_typ_s, target, corner
            )
        g2_typ = g2_typ * g2_scale
        out_time_s = 0.0
        out_g2 = 0.0
        out_src: str | None = None
        if include_outlap:
            hit_out = _lookup_outlap(model, track, entry_car, cond, corner)
            if hit_out is not None:
                out_time_s, out_g2, _out_n, out_src = hit_out
            if outlap_time_s is not None:
                out_time_s = float(outlap_time_s)
                out_src = (out_src or "none") + "+override"
        K, K_stderr, K_n, K_from_prior, K_src = _lookup_k(model, car, corner, cond)
        if compound is not None:
            hit = _lookup_compound_k(model, car, compound, corner, cond)
            if hit is not None:
                K, K_stderr, K_n, K_src = hit[0], hit[1], hit[2], hit[3]
                K_from_prior = False
        tau_sec, tau_stderr, _tau_src = _lookup_tau(model, car, corner, cond)

        warmup_frac = 1.0 - math.exp(-t_at_lap_n_s / tau_sec) if tau_sec > 0 else 0.0
        delta_t_inf = K * g2_typ
        t_after_out_c, t_hot_c = warmup_two_stage_c(
            t_outlap_s=out_time_s,
            g2_outlap=out_g2,
            t_flying_s=t_at_lap_n_s,
            g2_flying=g2_typ,
            t_eff_c=t_eff_c,
            k_kelvin_per_g2=K,
            tau_sec=tau_sec,
            t_start_c=t_start_c,
        )
        cold = gay_lussac_cold_pressure_bar(
            target_hot_pressure_bar=target_hot,
            t_hot_c=t_hot_c,
            t_cold_c=t_cold_c,  # tire's actual current temp (default: T_air)
            p_atm_bar=P_ATM_BAR,
        )

        out[corner] = Prediction(
            corner=corner,
            cold_pressure_bar=cold,
            predicted_hot_temp_c=t_hot_c,
            target_hot_pressure_bar=target_hot,
            K_kelvin_per_g2=K,
            tau_sec=tau_sec,
            g2_typ=g2_typ,
            lap_time_typ_s=lap_time_typ_s,
            t_at_lap_n_s=t_at_lap_n_s,
            warmup_frac=warmup_frac,
            delta_t_inf_kelvin=delta_t_inf,
            t_eff_c=t_eff_c,
            t_air_c=ambient_temp_c,
            t_road_c=t_road_c,
            t_cold_c=t_cold_c,
            t_start_c=t_start_c,
            K_source_bucket=K_src,
            K_from_prior=K_from_prior,
            K_n_samples=K_n,
            K_stderr=K_stderr,
            tau_stderr=tau_stderr,
            target_lap_time_s=(float(target_lap_time_s) if target_lap_time_s is not None else None),
            g2_scale=g2_scale,
            g2_pace_source=g2_pace_source,
            outlap_time_s=out_time_s,
            outlap_g2=out_g2,
            outlap_source=out_src,
            t_after_outlap_c=t_after_out_c,
            heat_provenance=heat_prov,
            lap_time_provenance=lap_prov,
            outlap_provenance=out_prov,
            uses_simulation=uses_sim,
            pace_status=status,
        )
    return out


def predict_cold_pressure_symmetric(
    *,
    target_hot_pressure_bar: float,
    **kwargs: Any,
) -> dict[str, Prediction]:
    """Convenience: broadcast one hot-pressure target across all four corners."""
    per_corner = {c: float(target_hot_pressure_bar) for c in CORNERS}
    return predict_cold_pressure(target_hot_pressure_bar=per_corner, **kwargs)
