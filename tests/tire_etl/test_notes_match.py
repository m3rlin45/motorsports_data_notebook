"""Tests for matching parsed notes to sessions."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa

from motorsports_data_notebook.tire_etl.notes_match import (
    infer_date_track_from_filename,
    match_notes_to_sessions,
)
from motorsports_data_notebook.tire_etl.notes_parser import (
    CornerStr,
    CornerValue,
    NoteSession,
    NotesData,
    ParsedNotes,
)


def _sessions(rows: list[dict]) -> pa.Table:
    cols: dict[str, list] = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k in cols:
            cols[k].append(r.get(k))
    return pa.table(
        {
            "session_id": pa.array(cols["session_id"], type=pa.string()),
            "date": pa.array(cols["date"], type=pa.date32()),
            "track_canonical": pa.array(cols["track_canonical"], type=pa.string()),
            "session_start_utc": pa.array(
                cols["session_start_utc"], type=pa.timestamp("us", tz="UTC")
            ),
        }
    )


def _make_parsed(file_date: str, track: str, sessions: list[NoteSession]) -> ParsedNotes:
    data = NotesData(file_date=file_date, track=track, sessions=sessions)
    return ParsedNotes(
        source_file=Path("/tmp/fake.txt"),
        source_sha1="abc",
        prompt_sha1="def",
        model="claude-opus-4-7",
        extracted_at="2026-04-04T00:00:00Z",
        data=data,
    )


def test_unique_match_same_day_same_track() -> None:
    sess = _sessions(
        [
            {
                "session_id": "s1",
                "date": date(2026, 4, 4),
                "track_canonical": "tsukuba_2000",
                "session_start_utc": datetime(2026, 4, 4, 0, 30, tzinfo=timezone.utc),
            }
        ]
    )
    ns = NoteSession(
        session_index=1,
        start_time_local="09:30",
        track_condition="dry",
        cold_pressure_bar=CornerValue(fl=1.80, fr=1.80, rl=1.90, rr=1.90),
        tire_compound=CornerStr(fl="22", fr="22", rl="21", rr="21"),
    )
    parsed = [_make_parsed("2026-04-04", "Tsukuba", [ns])]
    matches = match_notes_to_sessions(sess, parsed)
    assert len(matches) == 1
    assert matches[0].session_id == "s1"
    assert matches[0].match_confidence == 1.0
    assert matches[0].cold_pressure_bar_fl == 1.80


def test_no_match_when_tracks_differ() -> None:
    sess = _sessions(
        [
            {
                "session_id": "s1",
                "date": date(2026, 4, 4),
                "track_canonical": "fuji",
                "session_start_utc": datetime(2026, 4, 4, 0, 30, tzinfo=timezone.utc),
            }
        ]
    )
    ns = NoteSession(session_index=1, track_condition="dry")
    parsed = [_make_parsed("2026-04-04", "Tsukuba", [ns])]
    matches = match_notes_to_sessions(sess, parsed)
    assert matches == []


def test_infer_date_track_from_filename_standard() -> None:
    d, t = infer_date_track_from_filename(Path("/tmp/2026-04-04 Tsukuba.txt"))
    assert d == date(2026, 4, 4)
    assert t == "tsukuba_2000"


def test_infer_date_track_from_filename_with_car_token() -> None:
    d, t = infer_date_track_from_filename(Path("/tmp/Tsukuba KKSII 2026-04-04.txt"))
    assert d == date(2026, 4, 4)
    assert t == "tsukuba_2000"


def test_infer_date_track_from_filename_no_match() -> None:
    d, t = infer_date_track_from_filename(Path("/tmp/random_notes.txt"))
    assert d is None
    assert t is None


def test_filename_fallback_enables_match_when_body_has_null_metadata() -> None:
    sess = _sessions(
        [
            {
                "session_id": "s1",
                "date": date(2026, 4, 4),
                "track_canonical": "tsukuba_2000",
                "session_start_utc": datetime(2026, 4, 4, 0, 30, tzinfo=timezone.utc),
            }
        ]
    )
    ns = NoteSession(session_index=1, track_condition="dry")
    # Notes data has null file_date / track — forces fallback to filename.
    data = NotesData(file_date=None, track=None, sessions=[ns])
    pn = ParsedNotes(
        source_file=Path("/tmp/Tsukuba KKSII 2026-04-04.txt"),
        source_sha1="abc",
        prompt_sha1="def",
        model="claude-opus-4-7",
        extracted_at="2026-04-04T00:00:00Z",
        data=data,
    )
    matches = match_notes_to_sessions(sess, [pn])
    assert len(matches) == 1
    assert matches[0].session_id == "s1"


def test_greedy_time_matching_with_multiple_sessions() -> None:
    # Two sessions same day, same track — notes have two entries with times.
    sess = _sessions(
        [
            {
                "session_id": "morning",
                "date": date(2026, 4, 4),
                "track_canonical": "tsukuba_2000",
                "session_start_utc": datetime(2026, 4, 4, 0, 30, tzinfo=timezone.utc),  # 09:30 JST
            },
            {
                "session_id": "afternoon",
                "date": date(2026, 4, 4),
                "track_canonical": "tsukuba_2000",
                "session_start_utc": datetime(2026, 4, 4, 5, 0, tzinfo=timezone.utc),  # 14:00 JST
            },
        ]
    )
    parsed = [
        _make_parsed(
            "2026-04-04",
            "Tsukuba",
            [
                NoteSession(session_index=1, start_time_local="09:30", track_condition="dry"),
                NoteSession(session_index=2, start_time_local="14:00", track_condition="dry"),
            ],
        )
    ]
    matches = match_notes_to_sessions(sess, parsed)
    by_sid = {m.session_id: m for m in matches}
    assert by_sid["morning"].note_session_index == 1
    assert by_sid["afternoon"].note_session_index == 2


def test_filename_date_outranks_hallucinated_body_date() -> None:
    """The LLM has been observed inventing file_date for undated note
    bodies, which voided every match for the file. A date in the filename
    is authoritative."""
    sess = _sessions(
        [
            {
                "session_id": "s1",
                "date": date(2026, 1, 12),
                "track_canonical": "sodegaura",
                "session_start_utc": datetime(2026, 1, 12, 4, 0, tzinfo=timezone.utc),
            }
        ]
    )
    ns = NoteSession(session_index=1, track_condition="dry")
    data = NotesData(file_date="2024-04-28", track="Sodegaura", sessions=[ns])
    pn = ParsedNotes(
        source_file=Path("/tmp/2026-01-12 Sodegaura.txt"),
        source_sha1="abc",
        prompt_sha1="def",
        model="claude-opus-4-7",
        extracted_at="2026-04-04T00:00:00Z",
        data=data,
    )
    matches = match_notes_to_sessions(sess, [pn])
    assert len(matches) == 1
    assert matches[0].session_id == "s1"
