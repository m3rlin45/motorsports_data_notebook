"""Tests for the per-day notes + comments parse with a mocked `claude -p`."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from motorsports_data_notebook.tire_etl import session_notes as sn


def _sessions_parquet(root: Path, rows: list[dict]) -> None:
    (root / "sessions").mkdir(exist_ok=True)
    table = pa.table(
        {
            "session_id": [r["session_id"] for r in rows],
            "date": pa.array([r["date"] for r in rows], type=pa.date32()),
            "session_start_utc": pa.array(
                [r["start"] for r in rows], type=pa.timestamp("us", tz="UTC")
            ),
            "car": [r.get("car", "KK-F") for r in rows],
            "track_canonical": [r.get("track", "tsukuba_2000") for r in rows],
            "n_laps": pa.array([r.get("n_laps", 6) for r in rows], type=pa.int32()),
            "session_type": [r.get("session_type", "Car Race") for r in rows],
            "comment": [r.get("comment") for r in rows],
        }
    )
    pq.write_table(table, root / "sessions" / "2026-10.parquet")


def _day_rows() -> list[dict]:
    return [
        {
            "session_id": "s1",
            "date": date(2026, 10, 25),
            "start": datetime(2026, 10, 25, 8, 19, tzinfo=timezone.utc),
            "comment": "qually, slick tires, damp track",
        },
        {
            "session_id": "s2",
            "date": date(2026, 10, 25),
            "start": datetime(2026, 10, 25, 8, 37, tzinfo=timezone.utc),
            "comment": "Qually, wet tires, damp track",
        },
        {
            "session_id": "s3",
            "date": date(2026, 10, 24),
            "start": datetime(2026, 10, 24, 3, 0, tzinfo=timezone.utc),
            "comment": None,
        },
    ]


def _canned(session_ids: list[str]) -> str:
    recs = []
    for sid in session_ids:
        recs.append(
            {
                "session_id": sid,
                "note_file": "2026-10-25-Tsukuba.txt",
                "note_session_index": 1 if sid == "s1" else None,
                "match_confidence": 1.0 if sid == "s1" else 0.0,
                "weather_text": None,
                "ambient_temp_c": 17.0 if sid == "s1" else None,
                "track_condition": "damp",
                "cold_pressure_bar": (
                    {"fl": 1.3, "fr": 1.3, "rl": 1.35, "rr": 1.35} if sid == "s1" else {}
                ),
                "tire_compound": (
                    {"fl": "DRY", "fr": "DRY", "rl": "DRY", "rr": "DRY"}
                    if sid == "s1"
                    else {"fl": "WET", "fr": "WET", "rl": "WET", "rr": "WET"}
                ),
                "new_tires": None,
                "session_kind": "qualifying",
                "setup_changes": [],
                "incidents": [],
                "notes": "",
            }
        )
    return json.dumps({"sessions": recs})


def test_infer_date_track_from_filename() -> None:
    assert sn.infer_date_track_from_filename(Path("/n/2026-04-04 Tsukuba.txt")) == (
        date(2026, 4, 4),
        "tsukuba_2000",
    )
    assert sn.infer_date_track_from_filename(Path("/n/Tsukuba KKSII 2026-04-04.txt")) == (
        date(2026, 4, 4),
        "tsukuba_2000",
    )
    d, t = sn.infer_date_track_from_filename(Path("/n/2025-10-05-Sodegaura Marutai.txt"))
    assert (d, t) == (date(2025, 10, 5), "sodegaura")
    assert sn.infer_date_track_from_filename(Path("/n/random_notes.txt")) == (None, None)


def test_day_input_lists_notes_and_sessions_in_local_time(tmp_path: Path) -> None:
    _sessions_parquet(tmp_path, _day_rows())
    by_day = sn.logged_sessions_by_day(tmp_path)
    assert sorted(by_day) == [date(2026, 10, 24), date(2026, 10, 25)]
    day = by_day[date(2026, 10, 25)]
    assert [s.session_id for s in day] == ["s1", "s2"]
    assert day[0].start_local == "17:19"  # JST
    notes = [sn.NotesFile(path=Path("/n/2026-10-25-Tsukuba.txt"), text="17:15 quali\nTires 1.3")]
    text = sn.build_day_input(date(2026, 10, 25), notes, day)
    assert 'RUN NOTES FILE "2026-10-25-Tsukuba.txt":' in text
    assert "Tires 1.3" in text
    assert '1. session_id=s1 start=17:19 car="KK-F"' in text
    assert 'comment="qually, slick tires, damp track"' in text
    assert "RUN NOTES: none" in sn.build_day_input(date(2026, 10, 25), [], day)


def test_run_enrich_notes_parses_caches_and_writes_table(tmp_path: Path, monkeypatch) -> None:
    _sessions_parquet(tmp_path, _day_rows())
    notes_root = tmp_path / "notes"
    notes_root.mkdir()
    (notes_root / "2026-10-25-Tsukuba.txt").write_text("17:15 quali\nTires 1.3 / 1.35 rear")

    calls: list[str] = []

    def fake_invoke(user_text: str, system_prompt: str, model: str, timeout_s: float = 0) -> str:
        calls.append(user_text)
        ids = [
            line.split("session_id=")[1].split()[0]
            for line in user_text.splitlines()
            if "session_id=" in line
        ]
        # Wrap in fences once to exercise the stripper; return an extra unknown id too.
        body = json.loads(_canned(ids))
        body["sessions"].append({"session_id": "ghost"})
        return "```json\n" + json.dumps(body) + "\n```"

    monkeypatch.setattr(sn, "_invoke_claude_p", fake_invoke)
    counts = sn.run_enrich_notes(notes_root=notes_root, dataset_root=tmp_path, model="m")
    # 10-25 has notes + comments; 10-24 has neither -> only one day parsed.
    assert counts == {"days": 1, "cached": 0, "parsed": 1, "errors": 0, "rows": 2}
    assert len(calls) == 1
    cache = json.loads((tmp_path / "session_notes" / "2026-10-25.json").read_text())
    assert cache["_model"] == "m"
    assert [r["session_id"] for r in cache["data"]["sessions"]] == ["s1", "s2"]

    t = pq.read_table(tmp_path / "notes_matches.parquet").to_pandas().set_index("session_id")
    assert t.loc["s1", "tire_compound_fl"] == "DRY"
    assert t.loc["s2", "tire_compound_rl"] == "WET"
    assert t.loc["s1", "note_session_index"] == 1
    assert t.loc["s1", "source_file"].endswith("2026-10-25-Tsukuba.txt")
    assert t.loc["s1", "cold_pressure_bar_rl"] == pytest.approx(1.35)
    assert t.loc["s2", "comment"] == "Qually, wet tires, damp track"
    assert t.loc["s2", "session_kind"] == "qualifying"

    # Second run: everything cached, no model call, same table.
    counts2 = sn.run_enrich_notes(notes_root=notes_root, dataset_root=tmp_path, model="m")
    assert counts2["cached"] == 1 and counts2["parsed"] == 0 and len(calls) == 1
    # A changed comment invalidates that day's cache.
    rows = _day_rows()
    rows[1]["comment"] = "Qually, wet tires, heavy rain"
    _sessions_parquet(tmp_path, rows)
    counts3 = sn.run_enrich_notes(notes_root=notes_root, dataset_root=tmp_path, model="m")
    assert counts3["parsed"] == 1 and len(calls) == 2
    # --force re-parses too.
    sn.run_enrich_notes(notes_root=notes_root, dataset_root=tmp_path, model="m", force=True)
    assert len(calls) == 3


def test_model_omissions_are_filled_and_empty_records_dropped(tmp_path: Path, monkeypatch) -> None:
    rows = _day_rows()[:2]
    rows[1]["comment"] = None
    _sessions_parquet(tmp_path, rows)

    def fake_invoke(user_text: str, system_prompt: str, model: str, timeout_s: float = 0) -> str:
        return _canned(["s1"])  # s2 omitted by the model

    monkeypatch.setattr(sn, "_invoke_claude_p", fake_invoke)
    counts = sn.run_enrich_notes(notes_root=tmp_path / "none", dataset_root=tmp_path, model="m")
    assert counts["rows"] == 1  # s2 came back empty and is not a row
    cache = json.loads((tmp_path / "session_notes" / "2026-10-25.json").read_text())
    assert [r["session_id"] for r in cache["data"]["sessions"]] == ["s1", "s2"]


def test_invalid_json_is_an_error_not_a_crash(tmp_path: Path, monkeypatch) -> None:
    _sessions_parquet(tmp_path, _day_rows()[:1])
    monkeypatch.setattr(sn, "_invoke_claude_p", lambda *a, **k: "not json")
    counts = sn.run_enrich_notes(notes_root=tmp_path / "none", dataset_root=tmp_path, model="m")
    assert counts["errors"] == 1 and counts["rows"] == 0
    assert not (tmp_path / "session_notes" / "2026-10-25.json").exists()
