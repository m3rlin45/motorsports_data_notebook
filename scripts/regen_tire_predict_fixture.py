#!/usr/bin/env python3
"""Regenerate tire_pressure_calculator/Tests/Fixtures/python_predictions.json
against the current tire model (data/tire_dataset/tire_model.json).

Run after retraining the tire model to keep the C# parity test in sync:

    uv run scripts/regen_tire_predict_fixture.py

The case list mirrors the scenarios the C# `Predict_MatchesPythonOutputOnVendoredFixture`
test cares about: dry/damp/wet at multiple lap counts, a different track+car pair,
and an explicit cold-tire-temp override. Each case also pins the provenance of
the entry rows read (``sim`` / ``logged``), whether GSI's attribution shows, and
a simulated pace curve's status, which both ports must reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path

from typing import Callable

from motorsports_data_notebook.tire_model.predict import (
    CORNERS,
    _answering_rows,
    predict_cold_pressure,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT / "tire_pressure_calculator" / "Tests" / "Fixtures" / "python_predictions.json"
)

# Each case is the input dict that lands in the fixture's "inputs" block
# (target_hot_pressure_bar appears here as a scalar; predict_cold_pressure
# expects a per-corner dict, so we broadcast below).
CASES: list[tuple[str, dict]] = [
    (
        "tsukuba_KKSII_dry_lap5",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 18.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    (
        "tsukuba_KKSII_dry_lap10_cold15",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 10,
            "track_condition": "dry",
            "ambient_temp_c": 15.0,
            "cloud_cover_pct": 100.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    (
        "tsukuba_KKSII_damp_lap10",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 10,
            "track_condition": "damp",
            "ambient_temp_c": 15.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    (
        "tsukuba_KKSII_wet_falls_back",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 10,
            "track_condition": "wet",
            "ambient_temp_c": 15.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    (
        "sodegaura_Inferno_dry_lap5",
        {
            "track": "sodegaura",
            "car": "Inferno 86",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 22.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    (
        "tsukuba_KKSII_warmtire",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 10,
            "track_condition": "dry",
            "ambient_temp_c": 15.0,
            "cold_tire_temp_c": 22.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    # Target-lap-time feature: fast target on a curve-covered bucket, slow
    # target on the same bucket, a third curve-covered bucket (the
    # simulated-lap curves are per corner, so each corner's g2_scale is
    # pinned), and an unknown track to pin the default-exponent fallback.
    (
        "tsukuba_KKSII_dry_lap5_target60",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 18.0,
            "target_hot_pressure_bar": 1.7,
            "target_lap_time_s": 60.0,
        },
    ),
    (
        "tsukuba_KKSII_dry_lap5_target68",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 18.0,
            "target_hot_pressure_bar": 1.7,
            "target_lap_time_s": 68.0,
        },
    ),
    (
        "motegi_Inferno_dry_target_per_corner_curve",
        {
            "track": "motegi",
            "car": "Inferno 86",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 20.0,
            "target_hot_pressure_bar": 1.7,
            "target_lap_time_s": 120.0,
        },
    ),
    (
        "unknown_track_FJ_dry_target_exponent_fallback",
        {
            "track": "no_such_circuit",
            "car": "FJ",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 20.0,
            "target_hot_pressure_bar": 1.7,
            "target_lap_time_s": 80.0,
        },
    ),
    # Compound-aware K: the fitted compound replaces the pooled K on every
    # corner (one tire per run), an unknown compound falls back to pooled,
    # and the KK-SII's WET compound rides the wet condition chain.
    (
        "sodegaura_Inferno_dry_A050",
        {
            "track": "sodegaura",
            "car": "Inferno 86",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 22.0,
            "target_hot_pressure_bar": 1.7,
            "compound": "A050",
        },
    ),
    (
        "sodegaura_Inferno_dry_A052",
        {
            "track": "sodegaura",
            "car": "Inferno 86",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 22.0,
            "target_hot_pressure_bar": 1.7,
            "compound": "A052",
        },
    ),
    (
        "sodegaura_Inferno_unknown_compound_pooled",
        {
            "track": "sodegaura",
            "car": "Inferno 86",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 22.0,
            "target_hot_pressure_bar": 1.7,
            "compound": "SLICKS9000",
        },
    ),
    (
        "tsukuba_KKSII_wet_compound_WET",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 10,
            "track_condition": "wet",
            "ambient_temp_c": 15.0,
            "target_hot_pressure_bar": 1.7,
            "compound": "WET",
        },
    ),
    # Car aliasing: KK-F and KK-SII are pooled as "FJ" at fit time; both raw
    # names and the pooled label must resolve to the same FJ buckets.
    (
        "tsukuba_KKF_alias_dry_lap5",
        {
            "track": "tsukuba_2000",
            "car": "KK-F",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 18.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    (
        "tsukuba_FJ_pooled_dry_lap5",
        {
            "track": "tsukuba_2000",
            "car": "FJ",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 18.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    # Simulated entries (the cutover): a raw car's simulated dry cell with
    # the logged out-lap beside it, read at a target and with none; the
    # pace flags under the fastest knot and outside the logged pace; cells
    # only the simulation covers (the Inferno 86 at Suzuka, a formula car at
    # Motegi); a raw car's damp and wet (simulated or logged as
    # SIM_CONDITIONS has it, pinned to entry_coverage either way).
    (
        "tsukuba_KKSII_dry_sim_lap5",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 18.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    (
        "tsukuba_KKSII_dry_sim_target_in_support",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 18.0,
            "target_hot_pressure_bar": 1.7,
            "target_lap_time_s": None,  # set from the artifact (TARGETS)
        },
    ),
    (
        "tsukuba_KKSII_dry_sim_target_under_fastest_knot",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 18.0,
            "target_hot_pressure_bar": 1.7,
            "target_lap_time_s": None,  # set from the artifact (TARGETS)
        },
    ),
    (
        "tsukuba_KKSII_dry_sim_target_slower_than_support",
        {
            "track": "tsukuba_2000",
            "car": "KK-SII",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 18.0,
            "target_hot_pressure_bar": 1.7,
            "target_lap_time_s": None,  # set from the artifact (TARGETS)
        },
    ),
    (
        "suzuka_Inferno_dry_sim_only",
        {
            "track": "suzuka",
            "car": "Inferno 86",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 20.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    (
        "motegi_KKF_dry_sim_only",
        {
            "track": "motegi",
            "car": "KK-F",
            "lap_within_stint": 5,
            "track_condition": "dry",
            "ambient_temp_c": 20.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    (
        "tsukuba_KKF_damp_lap10",
        {
            "track": "tsukuba_2000",
            "car": "KK-F",
            "lap_within_stint": 10,
            "track_condition": "damp",
            "ambient_temp_c": 15.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
    (
        "fuji_KKFV2_wet_lap10",
        {
            "track": "fuji",
            "car": "KK-F-V2",
            "lap_within_stint": 10,
            "track_condition": "wet",
            "ambient_temp_c": 15.0,
            "target_hot_pressure_bar": 1.7,
        },
    ),
]

#: Target lap times read off the artifact at regeneration, so that the
#: pace-flag cases stay where they belong whatever the export's knots and
#: support (narrow: a few seconds): from the simulated curve of the case's
#: cell, under its fastest knot, the middle of its support, past its slow end.
TARGETS: dict[str, Callable[[dict], float]] = {
    "tsukuba_KKSII_dry_sim_target_in_support": lambda c: (
        c["support"]["lap_time_min_s"] + c["support"]["lap_time_max_s"]
    )
    / 2,
    "tsukuba_KKSII_dry_sim_target_under_fastest_knot": lambda c: c["lap_time_s"][0] - 10.0,
    "tsukuba_KKSII_dry_sim_target_slower_than_support": lambda c: (
        c["support"]["lap_time_max_s"] + 10.0
    ),
}

#: What a case is there to pin, checked at regeneration so that a case
#: whose label no longer says what the artifact gives fails loudly
#: (``uses_simulation``, ``pace_status``, the out-lap's provenance).
#: Every case is also held to the artifact's ``entry_coverage`` (``_check``).
EXPECT: dict[str, dict] = {
    "tsukuba_FJ_pooled_dry_lap5": {"uses_simulation": False, "pace_status": None},
    "unknown_track_FJ_dry_target_exponent_fallback": {"uses_simulation": False},
    "tsukuba_KKSII_dry_sim_lap5": {"uses_simulation": True, "outlap": "logged"},
    "tsukuba_KKSII_dry_sim_target_in_support": {"uses_simulation": True, "pace_status": "ok"},
    "tsukuba_KKSII_dry_sim_target_under_fastest_knot": {"pace_status": "no_solution"},
    "tsukuba_KKSII_dry_sim_target_slower_than_support": {"pace_status": "outside_support"},
    "suzuka_Inferno_dry_sim_only": {"uses_simulation": True, "outlap": "sim"},
    "motegi_KKF_dry_sim_only": {"uses_simulation": True, "outlap": "sim"},
}


def _run_case(label: str, inputs: dict) -> dict:
    scalar_target = inputs["target_hot_pressure_bar"]
    kwargs = {k: v for k, v in inputs.items() if k != "target_hot_pressure_bar"}
    kwargs["target_hot_pressure_bar"] = {c: scalar_target for c in CORNERS}
    predictions = predict_cold_pressure(**kwargs)
    any_p = next(iter(predictions.values()))
    return {
        "label": label,
        "inputs": inputs,
        "g2_pace_source": any_p.g2_pace_source,
        # Where the entry rows read come from, shared by the corners: the
        # ports show GSI's attribution and the pace flag from these.
        "provenance": {
            "heat": any_p.heat_provenance,
            "lap_time": any_p.lap_time_provenance,
            "outlap": any_p.outlap_provenance,
        },
        "uses_simulation": any_p.uses_simulation,
        "pace_status": any_p.pace_status,
        "corners": {
            c: {
                "cold_pressure_bar": predictions[c].cold_pressure_bar,
                "predicted_hot_temp_c": predictions[c].predicted_hot_temp_c,
                "K_source_bucket": list(predictions[c].K_source_bucket),
                # Per corner: the simulated-lap pace curve loads the corners differently.
                "g2_scale": predictions[c].g2_scale,
            }
            for c in CORNERS
        },
    }


def _check(case: dict, coverage: dict[tuple[str, str, str], dict]) -> None:
    """Fail where a case no longer pins what it is there for (:data:`EXPECT`),
    or where its provenance is not what the artifact's ``entry_coverage``
    says the lookups answer (an averaged answer is logged)."""
    label = case["label"]
    got = {
        "uses_simulation": case["uses_simulation"],
        "pace_status": case["pace_status"],
        "outlap": case["provenance"]["outlap"],
    }
    for key, want in EXPECT.get(label, {}).items():
        if got[key] != want:
            raise SystemExit(f"{label}: {key} is {got[key]!r}, the case is there for {want!r}")
    inputs = case["inputs"]
    cell = coverage.get((inputs["track"], inputs["car"], inputs["track_condition"]))
    if cell is None:
        return
    for key in ("heat", "lap_time", "outlap"):
        want = {"averaged": "logged", "none": None if key == "outlap" else "logged"}.get(
            cell[key], cell[key]
        )
        if case["provenance"][key] != want:
            raise SystemExit(
                f"{label}: {key} provenance {case['provenance'][key]!r}, "
                f"entry_coverage says {cell[key]!r}"
            )


def _with_target(model: dict, label: str, inputs: dict) -> dict:
    """``inputs`` with the target lap time :data:`TARGETS` reads off the
    simulated curve of the case's cell (rounded to 0.1 s, as the
    calculators type it)."""
    if label not in TARGETS:
        return inputs
    rows, _step = _answering_rows(
        model,
        "g2_typ_by_track_car_cond",
        inputs["track"],
        inputs["car"],
        inputs["track_condition"],
        global_step=True,
    )
    curve = rows[0].get("g2_vs_lap_time") if rows else None
    if not rows or rows[0].get("source") != "sim" or not curve or "support" not in curve:
        raise SystemExit(f"{label}: the cell has no simulated curve with a support to read")
    return inputs | {"target_lap_time_s": round(TARGETS[label](curve), 1)}


def main() -> None:
    model = json.loads((REPO_ROOT / "data" / "tire_dataset" / "tire_model.json").read_text())
    coverage = {
        (c["track_canonical"], c["car"], c["condition"]): c
        for c in model.get("entry_coverage") or []
    }
    cases_out = [_run_case(label, _with_target(model, label, inputs)) for label, inputs in CASES]
    for case in cases_out:
        _check(case, coverage)
    FIXTURE_PATH.write_text(json.dumps(cases_out, indent=2) + "\n")
    print(f"Wrote {len(cases_out)} cases to {FIXTURE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
