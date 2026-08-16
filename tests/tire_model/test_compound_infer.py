"""Tests for the joint (multi-task, partially supervised) compound EM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.compound_infer import (
    apply_condition_seeds,
    fit_compounds_em,
)

CORNERS = ("fl", "fr", "rl", "rr")


@dataclass
class _FP:
    value: float


TAU = {("ToyCar", c, "dry"): _FP(200.0) for c in CORNERS}
C_TRACK = {"track_a": _FP(1.0)}


def _session_laps(sid: str, k_true: float, n_laps: int = 12, noise: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(sid)) % 2**32)
    t = np.linspace(120, 120 * n_laps, n_laps)
    g2 = rng.uniform(0.5, 1.1, n_laps)
    frac = 1 - np.exp(-t / 200.0)
    rows = {
        "session_id": [sid] * n_laps,
        "car": ["ToyCar"] * n_laps,
        "condition": ["dry"] * n_laps,
        "track_canonical": ["track_a"] * n_laps,
        "t_cum_s": t,
        "on_track_s": np.full(n_laps, 100.0),
        "heat_proxy": g2 * 100.0,
    }
    for c in CORNERS:
        rows[f"delta_t_{c}"] = k_true * g2 * frac + rng.normal(0, noise, n_laps)
    return pd.DataFrame(rows)


def _labels(entries: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "session_id": sid,
                "compound_front": comp,
                "compound_rear": comp,
                "source": "sidecar",
            }
            for sid, comp in entries.items()
        ]
    )


def _mixture_frame() -> pd.DataFrame:
    parts = [
        _session_laps("soft1", 30.0),
        _session_laps("soft2", 30.0),
        _session_laps("hard1", 60.0),
        _session_laps("hard2", 60.0),
        _session_laps("mystery_soft", 30.0),
        _session_laps("mystery_hard", 60.0),
        _session_laps("ambiguous", 45.0),
    ]
    return pd.concat(parts, ignore_index=True)


class TestFitCompoundsEM:
    def test_recovers_unlabeled_assignments_and_k(self):
        laps = _mixture_frame()
        labels = _labels({"soft1": "SOFT", "soft2": "SOFT", "hard1": "HARD", "hard2": "HARD"})
        k, assignments = fit_compounds_em(laps, labels, TAU, C_TRACK)

        by_unit = {(a.session_id, a.axle): a for a in assignments}
        assert by_unit[("mystery_soft", "front")].compound == "SOFT"
        assert by_unit[("mystery_soft", "front")].responsibility > 0.95
        assert by_unit[("mystery_hard", "rear")].compound == "HARD"
        assert by_unit[("mystery_hard", "rear")].responsibility > 0.95
        # The mid-K session fits neither compound: the outlier gate must
        # leave it unlabeled rather than absorb it into a cluster.
        assert ("ambiguous", "front") not in by_unit

        assert k[("ToyCar", "SOFT", "fl", "dry")][0] == pytest.approx(30.0, abs=2.0)
        assert k[("ToyCar", "HARD", "fl", "dry")][0] == pytest.approx(60.0, abs=2.0)

    def test_pinned_labels_never_flip(self):
        laps = _mixture_frame()
        # Deliberately mislabel a hard session as SOFT: it must stay pinned.
        labels = _labels(
            {
                "soft1": "SOFT",
                "soft2": "SOFT",
                "hard1": "HARD",
                "hard2": "HARD",
                "mystery_hard": "SOFT",
            }
        )
        _, assignments = fit_compounds_em(laps, labels, TAU, C_TRACK)
        by_unit = {(a.session_id, a.axle): a for a in assignments}
        a = by_unit[("mystery_hard", "front")]
        assert a.pinned and a.compound == "SOFT" and a.responsibility == 1.0

    def test_single_compound_car_fits_without_latents(self):
        laps = pd.concat(
            [_session_laps("s1", 40.0), _session_laps("s2", 40.0), _session_laps("un", 40.0)],
            ignore_index=True,
        )
        labels = _labels({"s1": "ONLY", "s2": "ONLY"})
        k, assignments = fit_compounds_em(laps, labels, TAU, C_TRACK)
        assert k[("ToyCar", "ONLY", "fl", "dry")][0] == pytest.approx(40.0, abs=2.0)
        # The unlabeled session contributes nothing and gets no assignment.
        assert all(a.session_id != "un" for a in assignments)

    def test_no_labels_no_output(self):
        laps = _mixture_frame()
        k, assignments = fit_compounds_em(laps, laps.iloc[0:0], TAU, C_TRACK)
        assert k == {} and assignments == []


class TestApplyConditionSeeds:
    def test_seeds_uniform_condition_sessions_only(self):
        laps = pd.DataFrame(
            {
                "session_id": ["dry1", "dry1", "wet1", "mixed", "mixed"],
                "car": ["KK-SII"] * 5,
                "condition": ["dry", "dry", "wet", "dry", "wet"],
            }
        )
        labels = pd.DataFrame(columns=["session_id", "compound_front", "compound_rear", "source"])
        out = apply_condition_seeds(labels, laps, {"KK-SII": {"dry": "DRY", "wet": "WET"}})
        got = dict(zip(out.session_id, out.compound_front))
        assert got == {"dry1": "DRY", "wet1": "WET"}  # mixed stays unlabeled

    def test_existing_labels_win(self):
        laps = pd.DataFrame({"session_id": ["s1"], "car": ["KK-SII"], "condition": ["dry"]})
        labels = pd.DataFrame(
            [
                {
                    "session_id": "s1",
                    "compound_front": "WET",
                    "compound_rear": "WET",
                    "source": "sidecar",
                }
            ]
        )
        out = apply_condition_seeds(labels, laps, {"KK-SII": {"dry": "DRY"}})
        assert len(out) == 1 and out.iloc[0].compound_front == "WET"
