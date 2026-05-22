"""Inference: turn (track, car, lap, ambient, target hot pressures) → cold pressures.

Reads the JSON artifact written by :func:`build_warmup_table`, applies the
fallback chain documented in the plan (K[car, corner] → K[car] → global; track
falls back to ``c_track = 1.0``; ⟨g²⟩ falls back to (track) mean → global), and
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
    P_ATM_BAR,
    T_ZERO_C_TO_K,
    gay_lussac_cold_pressure_bar,
    t_effective_c,
    t_road_proxy_c,
    warmup_curve_c,
)

CORNERS = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class Prediction:
    corner: str
    cold_pressure_bar: float
    predicted_hot_temp_c: float
    target_hot_pressure_bar: float
    K_kelvin_per_g2: float
    tau_sec: float
    c_track: float
    g2_typ: float
    lap_time_typ_s: float
    t_at_lap_n_s: float
    warmup_frac: float
    delta_t_inf_kelvin: float
    t_eff_c: float
    t_air_c: float
    t_road_c: float
    t_cold_c: float  # temperature the tire is at when you measure/set cold pressure
    K_source_bucket: tuple[str, ...]
    K_from_prior: bool
    K_n_samples: int
    K_stderr: float
    tau_stderr: float
    c_track_stderr: float


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


def _lookup_c_track(model: dict[str, Any], track: str) -> tuple[float, float, bool]:
    """Return (c_track, stderr, from_prior). Track-only (condition-independent)."""
    for r in model["c_track_by_track"]:
        if r["track_canonical"] == track:
            return float(r["value"]), float(r["stderr"]), False
    prior_c = float(model["priors_when_no_fit"]["c_track"])
    return prior_c, 0.0, True


def _lookup_g2(
    model: dict[str, Any], track: str, car: str, condition: str
) -> tuple[float, int, str]:
    """Return (g2_typ, n_laps_used, source).

    Fallback walks the condition chain ((wet) → damp → dry, (damp) → dry,
    (dry) → dry), then (track, car) pooled across conditions, then (track),
    then global.
    """
    table = model["g2_typ_by_track_car_cond"]
    for cond in _condition_chain(condition):
        for r in table:
            if r["track_canonical"] == track and r["car"] == car and r["condition"] == cond:
                tag = "exact" if cond == condition else f"fallback({cond})"
                return float(r["g2_typ"]), int(r["n_laps_used"]), tag
    same_tc = [r for r in table if r["track_canonical"] == track and r["car"] == car]
    if same_tc:
        vals = [r["g2_typ"] for r in same_tc]
        n = sum(int(r["n_laps_used"]) for r in same_tc)
        return float(sum(vals) / len(vals)), n, "track_car_pooled"
    same_t = [r for r in table if r["track_canonical"] == track]
    if same_t:
        vals = [r["g2_typ"] for r in same_t]
        n = sum(int(r["n_laps_used"]) for r in same_t)
        return float(sum(vals) / len(vals)), n, "track_pooled"
    all_g2 = [r["g2_typ"] for r in table]
    if all_g2:
        return float(sum(all_g2) / len(all_g2)), 0, "global"
    return 0.7, 0, "global"


def _lookup_lap_time(
    model: dict[str, Any], track: str, car: str, condition: str
) -> tuple[float, int, str]:
    table = model["lap_time_typ_by_track_car_cond"]
    for cond in _condition_chain(condition):
        for r in table:
            if r["track_canonical"] == track and r["car"] == car and r["condition"] == cond:
                tag = "exact" if cond == condition else f"fallback({cond})"
                return float(r["lap_time_typ_s"]), int(r["n_laps_used"]), tag
    same_tc = [r for r in table if r["track_canonical"] == track and r["car"] == car]
    if same_tc:
        vals = [r["lap_time_typ_s"] for r in same_tc]
        n = sum(int(r["n_laps_used"]) for r in same_tc)
        return float(sum(vals) / len(vals)), n, "track_car_pooled"
    same_t = [r for r in table if r["track_canonical"] == track]
    if same_t:
        vals = [r["lap_time_typ_s"] for r in same_t]
        n = sum(int(r["n_laps_used"]) for r in same_t)
        return float(sum(vals) / len(vals)), n, "track_pooled"
    return 90.0, 0, "global"


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
    dataset_root: Path | None = None,
    _model: dict[str, Any] | None = None,
) -> dict[str, Prediction]:
    """Predict per-corner cold pressure for a target lap.

    Parameters
    ----------
    ambient_temp_c
        Air temperature for the upcoming session. Drives the T_road sun-proxy
        and the T_eff baseline used in the warmup curve.
    cold_tire_temp_c
        Optional temperature the tire is at right *now* when you'll measure
        or set cold pressure. Defaults to ``ambient_temp_c`` when omitted.
        Use this when the tire isn't at air temperature — e.g., sitting in
        a sun-warmed garage, set the night before in cooler air, or pre-heated.
        Affects the Gay-Lussac inversion only, not the warmup-curve target.
    track_condition
        One of ``"dry"`` (default), ``"damp"`` (light drizzle, 0.1–1 mm/hr),
        or ``"wet"`` (≥ 1 mm/hr). Picks the condition-specific K, τ_sec,
        ⟨g²⟩, and lap_time_typ. Falls back to ``"dry"`` per-bucket when
        the requested condition has no data.
    """
    model = _model if _model is not None else _load_model(dataset_root)

    cond = str(track_condition or "dry").lower()
    if cond not in {"dry", "damp", "wet"}:
        raise ValueError(f"track_condition must be one of dry/damp/wet; got {track_condition!r}")

    # Lookups (all condition-aware where applicable)
    g2_typ, g2_n, _g2_source = _lookup_g2(model, track, car, cond)
    if g2_typ_override is not None:
        g2_typ = float(g2_typ_override)
    lap_time_typ_s, lt_n, _lt_source = _lookup_lap_time(model, track, car, cond)
    if lap_time_typ_override_s is not None:
        lap_time_typ_s = float(lap_time_typ_override_s)
    c_track, c_track_stderr, _c_from_prior = _lookup_c_track(model, track)
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

    # Cold-side temperature for the Gay-Lussac inversion: defaults to T_air
    # but lets the caller pin a different "what's the tire currently at?"
    # value (e.g., garage temp ≠ track ambient).
    t_cold_c = float(cold_tire_temp_c) if cold_tire_temp_c is not None else ambient_temp_c

    # Time at end of lap N
    t_at_lap_n_s = float(lap_within_stint) * lap_time_typ_s

    out: dict[str, Prediction] = {}
    for corner in CORNERS:
        if corner not in target_hot_pressure_bar:
            raise KeyError(f"target_hot_pressure_bar missing corner {corner!r}")
        target_hot = float(target_hot_pressure_bar[corner])
        K, K_stderr, K_n, K_from_prior, K_src = _lookup_k(model, car, corner, cond)
        tau_sec, tau_stderr, _tau_src = _lookup_tau(model, car, corner, cond)

        warmup_frac = 1.0 - math.exp(-t_at_lap_n_s / tau_sec) if tau_sec > 0 else 0.0
        delta_t_inf = K * c_track * g2_typ
        t_hot_c = warmup_curve_c(
            t_seconds=t_at_lap_n_s,
            t_eff_c=t_eff_c,
            k_kelvin_per_g2=K,
            c_track=c_track,
            g2_typ=g2_typ,
            tau_sec=tau_sec,
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
            c_track=c_track,
            g2_typ=g2_typ,
            lap_time_typ_s=lap_time_typ_s,
            t_at_lap_n_s=t_at_lap_n_s,
            warmup_frac=warmup_frac,
            delta_t_inf_kelvin=delta_t_inf,
            t_eff_c=t_eff_c,
            t_air_c=ambient_temp_c,
            t_road_c=t_road_c,
            t_cold_c=t_cold_c,
            K_source_bucket=K_src,
            K_from_prior=K_from_prior,
            K_n_samples=K_n,
            K_stderr=K_stderr,
            tau_stderr=tau_stderr,
            c_track_stderr=c_track_stderr,
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
