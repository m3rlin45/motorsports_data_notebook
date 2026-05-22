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


def _lookup_tau(model: dict[str, Any], car: str, corner: str) -> tuple[float, float]:
    for r in model["tau_sec_by_car_corner"]:
        if r["car"] == car and r["corner"] == corner:
            return float(r["value_seconds"]), float(r["stderr_seconds"])
    # fallback to mean over (car) if any
    same_car = [r for r in model["tau_sec_by_car_corner"] if r["car"] == car]
    if same_car:
        vals = [r["value_seconds"] for r in same_car]
        return float(sum(vals) / len(vals)), 0.0
    return float(model["priors_when_no_fit"]["tau_sec_seconds"]), 0.0


def _lookup_k(
    model: dict[str, Any], car: str, corner: str
) -> tuple[float, float, int, bool, tuple[str, ...]]:
    """Return (K, stderr, n_samples, from_prior, source_bucket_tuple)."""
    for r in model["K_buckets"]:
        if r["key"]["car"] == car and r["key"]["corner"] == corner:
            return (
                float(r["value_kelvin_per_g2"]),
                float(r["stderr_kelvin_per_g2"]),
                int(r["n_samples"]),
                bool(r["from_prior"]),
                (car, corner),
            )
    # (car) fallback: average K over corners
    same_car = [r for r in model["K_buckets"] if r["key"]["car"] == car]
    if same_car:
        vals = [r["value_kelvin_per_g2"] for r in same_car]
        n = sum(int(r["n_samples"]) for r in same_car)
        return float(sum(vals) / len(vals)), 0.0, n, False, (car,)
    # global fallback
    prior_k = float(model["priors_when_no_fit"]["K_kelvin_per_g2"])
    return prior_k, 0.0, 0, True, ()


def _lookup_c_track(model: dict[str, Any], track: str) -> tuple[float, float, bool]:
    """Return (c_track, stderr, from_prior)."""
    for r in model["c_track_by_track"]:
        if r["track_canonical"] == track:
            return float(r["value"]), float(r["stderr"]), False
    prior_c = float(model["priors_when_no_fit"]["c_track"])
    return prior_c, 0.0, True


def _lookup_g2(model: dict[str, Any], track: str, car: str) -> tuple[float, int, str]:
    """Return (g2_typ, n_laps_used, source). Source is one of 'exact', 'track', 'global', 'override'."""
    for r in model["g2_typ_by_track_car"]:
        if r["track_canonical"] == track and r["car"] == car:
            return float(r["g2_typ"]), int(r["n_laps_used"]), "exact"
    same_track = [r for r in model["g2_typ_by_track_car"] if r["track_canonical"] == track]
    if same_track:
        vals = [r["g2_typ"] for r in same_track]
        n = sum(int(r["n_laps_used"]) for r in same_track)
        return float(sum(vals) / len(vals)), n, "track"
    all_g2 = [r["g2_typ"] for r in model["g2_typ_by_track_car"]]
    if all_g2:
        return float(sum(all_g2) / len(all_g2)), 0, "global"
    return 0.7, 0, "global"


def _lookup_lap_time(model: dict[str, Any], track: str, car: str) -> tuple[float, int, str]:
    for r in model["lap_time_typ_by_track_car"]:
        if r["track_canonical"] == track and r["car"] == car:
            return float(r["lap_time_typ_s"]), int(r["n_laps_used"]), "exact"
    same_track = [r for r in model["lap_time_typ_by_track_car"] if r["track_canonical"] == track]
    if same_track:
        vals = [r["lap_time_typ_s"] for r in same_track]
        n = sum(int(r["n_laps_used"]) for r in same_track)
        return float(sum(vals) / len(vals)), n, "track"
    return 90.0, 0, "global"


def predict_cold_pressure(
    *,
    track: str,
    car: str,
    lap_within_stint: int,
    target_hot_pressure_bar: dict[str, float],
    ambient_temp_c: float,
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
    track
        ``track_canonical`` string (e.g. ``"tsukuba_2000"``).
    car
        Car string as it appears in ``sessions/*.parquet`` (e.g. ``"KK-SII"``).
    lap_within_stint
        0-indexed lap number within a stint. ``5`` means "5 laps in" — the
        prediction is at the END of that lap.
    target_hot_pressure_bar
        Dict keyed by corner ``{"fl", "fr", "rl", "rr"}``, values in bar gauge.
    ambient_temp_c
        Outside air temperature in °C (T_air for Gay-Lussac).
    track_temp_c
        Optional measured track surface temperature in °C. If ``None``, the
        proxy ``T_air + Δ_sun · (1 - cloud_cover/100)`` is used (see
        :func:`energy_balance.t_road_proxy_c`).
    cloud_cover_pct
        0..100, used only if ``track_temp_c is None``. ``None`` ⇒ no sun
        offset (T_road = T_air).
    g2_typ_override
        Override the looked-up ⟨g²⟩ for the (track, car) bucket. Useful for
        brand-new tracks.
    lap_time_typ_override_s
        Override the looked-up median lap time. Useful when the user wants a
        race-pace prediction that differs from session median.
    """
    model = _model if _model is not None else _load_model(dataset_root)

    # Lookups
    g2_typ, g2_n, _g2_source = _lookup_g2(model, track, car)
    if g2_typ_override is not None:
        g2_typ = float(g2_typ_override)
    lap_time_typ_s, lt_n, _lt_source = _lookup_lap_time(model, track, car)
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

    # Time at end of lap N
    t_at_lap_n_s = float(lap_within_stint) * lap_time_typ_s

    out: dict[str, Prediction] = {}
    for corner in CORNERS:
        if corner not in target_hot_pressure_bar:
            raise KeyError(f"target_hot_pressure_bar missing corner {corner!r}")
        target_hot = float(target_hot_pressure_bar[corner])
        K, K_stderr, K_n, K_from_prior, K_src = _lookup_k(model, car, corner)
        tau_sec, tau_stderr = _lookup_tau(model, car, corner)

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
            t_cold_c=ambient_temp_c,  # cold uses T_air only
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
