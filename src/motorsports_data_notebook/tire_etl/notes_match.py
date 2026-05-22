"""Match parsed run-notes to telemetry sessions by date + track (+ time).

Produces a table keyed on ``session_id`` with the source note file and the
session index within that file. Ambiguous matches (multiple candidate note
sessions for one telemetry session) receive a lower ``match_confidence``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as _date, datetime, timezone
from pathlib import Path

import pyarrow as pa

from .notes_parser import NoteSession, ParsedNotes
from .tracks import get_track, known_canonicals, normalize_track_name

_FILENAME_DATE_RE = re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})")


def infer_date_track_from_filename(path: Path) -> tuple[_date | None, str | None]:
    """Best-effort recovery of (date, track_canonical) from a notes filename.

    The LLM can't always find a date in the body of a notes file when it's
    only present in the filename (e.g. ``2026-04-04 Tsukuba.txt``). This
    helper fills that gap so matching still works.
    """
    stem = path.stem
    d: _date | None = None
    m = _FILENAME_DATE_RE.search(stem)
    if m:
        try:
            d = _date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        except ValueError:
            d = None
    lower = stem.lower()
    track: str | None = None
    for canonical in known_canonicals():
        # Strip the suffix ("_2000" etc.) and match the leading token.
        head = canonical.split("_")[0]
        if head in lower:
            track = canonical
            break
    # Alias check (e.g. "kksii" doesn't map to a track token directly).
    if track is None:
        for tok in ("tsukuba", "sodegaura", "fuji", "motegi", "marutai"):
            if tok in lower:
                track = normalize_track_name(tok)
                break
    return d, track


@dataclass(frozen=True)
class NoteMatch:
    session_id: str
    source_file: str
    source_sha1: str
    note_session_index: int
    match_confidence: float
    cold_pressure_bar_fl: float | None
    cold_pressure_bar_fr: float | None
    cold_pressure_bar_rl: float | None
    cold_pressure_bar_rr: float | None
    tire_compound_fl: str | None
    tire_compound_fr: str | None
    tire_compound_rl: str | None
    tire_compound_rr: str | None
    track_condition: str
    ambient_temp_c: float | None
    weather_text: str | None


def _parse_local_time(hhmm: str | None) -> tuple[int, int] | None:
    if not hhmm:
        return None
    try:
        h, m = hhmm.split(":")
        return int(h), int(m)
    except (ValueError, AttributeError):
        return None


def _session_utc_to_local_minutes(
    session_start_utc: datetime, track_canonical: str | None
) -> int | None:
    if track_canonical is None:
        return None
    ti = get_track(track_canonical)
    if ti is None:
        return None
    local = session_start_utc.astimezone(ti.tzinfo())
    return local.hour * 60 + local.minute


def match_notes_to_sessions(
    sessions: pa.Table,
    parsed_notes: list[ParsedNotes],
    *,
    date_tolerance_days: int = 1,
    time_tolerance_minutes: int = 90,
) -> list[NoteMatch]:
    """Return per-session match records for every resolvable session."""
    matches: list[NoteMatch] = []
    if len(sessions) == 0 or not parsed_notes:
        return matches

    # Index notes by (date, track_canonical)
    note_entries: list[tuple[_date, str | None, ParsedNotes, NoteSession]] = []
    for pn in parsed_notes:
        file_date = pn.data.file_date
        try:
            d = datetime.strptime(file_date, "%Y-%m-%d").date() if file_date else None
        except ValueError:
            d = None
        note_track_can = normalize_track_name(pn.data.track or "")
        # Fall back to the notes filename for date/track when the LLM didn't
        # extract them from the body.
        if d is None or note_track_can is None:
            fb_date, fb_track = infer_date_track_from_filename(pn.source_file)
            if d is None:
                d = fb_date
            if note_track_can is None:
                note_track_can = fb_track
        if d is None:
            continue
        for ns in pn.data.sessions:
            note_entries.append((d, note_track_can, pn, ns))

    sess_ids = sessions.column("session_id").to_pylist()
    sess_dates = sessions.column("date").to_pylist()
    sess_tracks = sessions.column("track_canonical").to_pylist()
    sess_utc = sessions.column("session_start_utc").to_pylist()

    for i, sess_id in enumerate(sess_ids):
        sess_date: _date = sess_dates[i]
        track_can: str | None = sess_tracks[i]
        # Candidate notes: within date window + same track.
        candidates = [
            entry
            for entry in note_entries
            if abs((entry[0] - sess_date).days) <= date_tolerance_days
            and (track_can is None or entry[1] == track_can)
        ]
        if not candidates:
            continue

        # If only one candidate: high confidence.
        if len(candidates) == 1:
            _, _, pn, ns = candidates[0]
            matches.append(_make_match(sess_id, pn, ns, confidence=1.0 if track_can else 0.9))
            continue

        # Multiple candidates: prefer time-of-day match.
        sess_local_min = _session_utc_to_local_minutes(
            sess_utc[i].replace(tzinfo=timezone.utc) if sess_utc[i].tzinfo is None else sess_utc[i],
            track_can,
        )
        best = None
        best_delta = None
        for entry in candidates:
            _, _, pn, ns = entry
            lt = _parse_local_time(ns.start_time_local)
            if lt is None or sess_local_min is None:
                continue
            nmin = lt[0] * 60 + lt[1]
            delta = abs(nmin - sess_local_min)
            if delta <= time_tolerance_minutes and (best_delta is None or delta < best_delta):
                best = entry
                best_delta = delta
        if best is not None:
            _, _, pn, ns = best
            conf = max(0.5, 1.0 - (best_delta or 0) / (time_tolerance_minutes * 2))
            matches.append(_make_match(sess_id, pn, ns, confidence=conf))
        else:
            # All candidates equally plausible — record first with low confidence.
            _, _, pn, ns = candidates[0]
            matches.append(_make_match(sess_id, pn, ns, confidence=0.3))

    return matches


def _make_match(
    session_id: str, pn: ParsedNotes, ns: NoteSession, *, confidence: float
) -> NoteMatch:
    return NoteMatch(
        session_id=session_id,
        source_file=str(pn.source_file),
        source_sha1=pn.source_sha1,
        note_session_index=ns.session_index,
        match_confidence=float(confidence),
        cold_pressure_bar_fl=ns.cold_pressure_bar.fl,
        cold_pressure_bar_fr=ns.cold_pressure_bar.fr,
        cold_pressure_bar_rl=ns.cold_pressure_bar.rl,
        cold_pressure_bar_rr=ns.cold_pressure_bar.rr,
        tire_compound_fl=ns.tire_compound.fl,
        tire_compound_fr=ns.tire_compound.fr,
        tire_compound_rl=ns.tire_compound.rl,
        tire_compound_rr=ns.tire_compound.rr,
        track_condition=ns.track_condition,
        ambient_temp_c=ns.ambient_temp_c,
        weather_text=ns.weather_text,
    )


def matches_to_table(matches: list[NoteMatch]) -> pa.Table:
    if not matches:
        return pa.table(
            {
                "session_id": pa.array([], type=pa.string()),
                "source_file": pa.array([], type=pa.string()),
                "source_sha1": pa.array([], type=pa.string()),
                "note_session_index": pa.array([], type=pa.int32()),
                "match_confidence": pa.array([], type=pa.float32()),
                "cold_pressure_bar_fl": pa.array([], type=pa.float32()),
                "cold_pressure_bar_fr": pa.array([], type=pa.float32()),
                "cold_pressure_bar_rl": pa.array([], type=pa.float32()),
                "cold_pressure_bar_rr": pa.array([], type=pa.float32()),
                "tire_compound_fl": pa.array([], type=pa.string()),
                "tire_compound_fr": pa.array([], type=pa.string()),
                "tire_compound_rl": pa.array([], type=pa.string()),
                "tire_compound_rr": pa.array([], type=pa.string()),
                "track_condition": pa.array([], type=pa.string()),
                "ambient_temp_c": pa.array([], type=pa.float32()),
                "weather_text": pa.array([], type=pa.string()),
            }
        )
    cols: dict[str, list] = {k: [] for k in NoteMatch.__annotations__}
    for m in matches:
        for k in cols:
            cols[k].append(getattr(m, k))
    return pa.table(cols)


def write_matches(dataset_root: Path, matches: list[NoteMatch]) -> None:
    """Write the matches table to dataset_root/notes_matches.parquet."""
    import pyarrow.parquet as pq

    path = dataset_root / "notes_matches.parquet"
    table = matches_to_table(matches)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=3)
    tmp.replace(path)
