"""Tests for compound label loading + normalization (tire_model.compounds)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from motorsports_data_notebook.tire_model.compounds import (
    load_compound_labels,
    normalize_compound,
)


class TestNormalizeCompound:
    def test_canonical_variants(self):
        assert normalize_compound("A052") == "A052"
        assert normalize_compound("052") == "A052"
        assert normalize_compound("a 052") == "A052"
        assert normalize_compound("RE-71RS") == "RE-71RS"
        assert normalize_compound("71RS") == "RE-71RS"
        assert normalize_compound("RE-71 RS") == "RE-71RS"
        assert normalize_compound("Nankang NS-2R Used") == "NS-2R"
        assert normalize_compound("A050_22") == "A050"

    def test_set_labels_pass_through(self):
        assert normalize_compound("SET:black") == "SET:black"
        assert normalize_compound("SET:Silver") == "SET:silver"

    def test_junk_returns_none(self):
        for junk in ("15", "2/2", "14c", "silver worn", None, "", 22):
            assert normalize_compound(junk) is None


def _write_sidecar(root: Path, text: str) -> None:
    (root / "tire_compounds.yaml").write_text(text)


def _write_matches(root: Path, rows: list[dict]) -> None:
    base = {
        "session_id": "",
        "tire_compound_fl": None,
        "tire_compound_rl": None,
        "match_confidence": 1.0,
    }
    table = pa.Table.from_pylist([{**base, **r} for r in rows])
    pq.write_table(table, root / "notes_matches.parquet")


class TestLoadCompoundLabels:
    def test_sidecar_wins_over_notes(self, tmp_path: Path):
        _write_sidecar(tmp_path, "sessions:\n  - session_id: s1\n    front: A052\n    rear: A052\n")
        _write_matches(
            tmp_path,
            [
                {"session_id": "s1", "tire_compound_fl": "RE-71RS", "tire_compound_rl": "RE-71RS"},
                {"session_id": "s2", "tire_compound_fl": "71RS", "tire_compound_rl": "71RS"},
            ],
        )
        out = load_compound_labels(tmp_path).set_index("session_id")
        assert out.loc["s1", "compound_front"] == "A052"
        assert out.loc["s1", "source"] == "sidecar"
        assert out.loc["s2", "compound_front"] == "RE-71RS"
        assert out.loc["s2", "source"] == "notes"

    def test_wheel_sets_resolve_set_labels(self, tmp_path: Path):
        _write_sidecar(tmp_path, "wheel_sets:\n  black: RE-71RS\n  silver: A052\nsessions: []\n")
        _write_matches(
            tmp_path,
            [
                {
                    "session_id": "s1",
                    "tire_compound_fl": "SET:silver",
                    "tire_compound_rl": "SET:black",
                },
                {
                    "session_id": "s2",
                    "tire_compound_fl": "SET:silver-blue",
                    "tire_compound_rl": None,
                },
            ],
        )
        out = load_compound_labels(tmp_path).set_index("session_id")
        assert out.loc["s1", "compound_front"] == "A052"
        assert out.loc["s1", "compound_rear"] == "RE-71RS"
        # Unmapped set -> no label at all for that session
        assert "s2" not in out.index

    def test_highest_confidence_note_wins(self, tmp_path: Path):
        _write_matches(
            tmp_path,
            [
                {"session_id": "s1", "tire_compound_fl": "A052", "match_confidence": 0.6},
                {"session_id": "s1", "tire_compound_fl": "71RS", "match_confidence": 0.95},
            ],
        )
        out = load_compound_labels(tmp_path).set_index("session_id")
        assert out.loc["s1", "compound_front"] == "RE-71RS"

    def test_missing_files(self, tmp_path: Path):
        out = load_compound_labels(tmp_path)
        assert out.empty
