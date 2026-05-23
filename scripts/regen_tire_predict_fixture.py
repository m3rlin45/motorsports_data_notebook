#!/usr/bin/env python3
"""Regenerate tire_pressure_calculator/Tests/Fixtures/python_predictions.json
against the current tire model (data/tire_dataset/tire_model.json).

Run after retraining the tire model to keep the C# parity test in sync:

    uv run scripts/regen_tire_predict_fixture.py

The case list mirrors the scenarios the C# `Predict_MatchesPythonOutputOnVendoredFixture`
test cares about: dry/damp/wet at multiple lap counts, a different track+car pair,
and an explicit cold-tire-temp override.
"""

from __future__ import annotations

import json
from pathlib import Path

from motorsports_data_notebook.tire_model.predict import CORNERS, predict_cold_pressure

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
]


def _run_case(label: str, inputs: dict) -> dict:
    scalar_target = inputs["target_hot_pressure_bar"]
    kwargs = {k: v for k, v in inputs.items() if k != "target_hot_pressure_bar"}
    kwargs["target_hot_pressure_bar"] = {c: scalar_target for c in CORNERS}
    predictions = predict_cold_pressure(**kwargs)
    return {
        "label": label,
        "inputs": inputs,
        "corners": {
            c: {
                "cold_pressure_bar": predictions[c].cold_pressure_bar,
                "predicted_hot_temp_c": predictions[c].predicted_hot_temp_c,
                "K_source_bucket": list(predictions[c].K_source_bucket),
            }
            for c in CORNERS
        },
    }


def main() -> None:
    cases_out = [_run_case(label, inputs) for label, inputs in CASES]
    FIXTURE_PATH.write_text(json.dumps(cases_out, indent=2) + "\n")
    print(f"Wrote {len(cases_out)} cases to {FIXTURE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
