"""Tests for notes_parser with mocked `claude -p` subprocess."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motorsports_data_notebook.tire_etl import notes_parser


def _canned_claude_output() -> str:
    return json.dumps(
        {
            "file_date": "2026-04-04",
            "track": "Tsukuba",
            "car": "Inferno 86",
            "sessions": [
                {
                    "session_index": 1,
                    "start_time_local": "09:30",
                    "weather_text": "clear, 14C",
                    "ambient_temp_c": 14.0,
                    "track_condition": "dry",
                    "cold_pressure_bar": {
                        "fl": 1.80,
                        "fr": 1.80,
                        "rl": 1.90,
                        "rr": 1.90,
                    },
                    "tire_compound": {
                        "fl": "22",
                        "fr": "22",
                        "rl": "21",
                        "rr": "21",
                    },
                    "setup_changes": [],
                    "incidents": [],
                    "notes": "",
                }
            ],
        }
    )


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Path:
    (tmp_path / "notes_extracted").mkdir(parents=True)
    return tmp_path


def test_parse_and_cache(monkeypatch: pytest.MonkeyPatch, tmp_dataset: Path) -> None:
    notes_file = tmp_dataset / "2026-04-04 Tsukuba.txt"
    notes_file.write_text("09:30 Session\nTires 1.8 all around\n")

    calls = {"n": 0}

    def fake_invoke(
        notes_text: str, system_prompt: str, model: str, timeout_s: float = 300.0
    ) -> str:
        calls["n"] += 1
        return _canned_claude_output()

    monkeypatch.setattr(notes_parser, "_invoke_claude_p", fake_invoke)

    pn = notes_parser.parse_notes_file(notes_file, dataset_root=tmp_dataset)
    assert pn.data.file_date == "2026-04-04"
    assert pn.data.sessions[0].cold_pressure_bar.fl == 1.80
    assert calls["n"] == 1

    # Second call: cache hit, no subprocess.
    pn2 = notes_parser.parse_notes_file(notes_file, dataset_root=tmp_dataset)
    assert calls["n"] == 1
    assert pn2.data.sessions[0].tire_compound.fl == "22"


def test_force_reruns_even_with_cache(monkeypatch: pytest.MonkeyPatch, tmp_dataset: Path) -> None:
    notes_file = tmp_dataset / "2026-04-04 Tsukuba.txt"
    notes_file.write_text("foo")

    calls = {"n": 0}

    def fake_invoke(*a, **kw) -> str:
        calls["n"] += 1
        return _canned_claude_output()

    monkeypatch.setattr(notes_parser, "_invoke_claude_p", fake_invoke)

    notes_parser.parse_notes_file(notes_file, dataset_root=tmp_dataset)
    notes_parser.parse_notes_file(notes_file, dataset_root=tmp_dataset, force=True)
    assert calls["n"] == 2


def test_strip_code_fences_around_json(monkeypatch: pytest.MonkeyPatch, tmp_dataset: Path) -> None:
    notes_file = tmp_dataset / "f.txt"
    notes_file.write_text("x")

    def fake_invoke(*a, **kw) -> str:
        return "```json\n" + _canned_claude_output() + "\n```"

    monkeypatch.setattr(notes_parser, "_invoke_claude_p", fake_invoke)
    pn = notes_parser.parse_notes_file(notes_file, dataset_root=tmp_dataset)
    assert pn.data.track == "Tsukuba"


def test_invalid_json_raises(monkeypatch: pytest.MonkeyPatch, tmp_dataset: Path) -> None:
    notes_file = tmp_dataset / "f.txt"
    notes_file.write_text("x")

    monkeypatch.setattr(notes_parser, "_invoke_claude_p", lambda *a, **kw: "not json at all")
    with pytest.raises(ValueError, match="did not return valid JSON"):
        notes_parser.parse_notes_file(notes_file, dataset_root=tmp_dataset)
