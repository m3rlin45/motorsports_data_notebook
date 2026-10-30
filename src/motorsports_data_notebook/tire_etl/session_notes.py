"""Per-track-day extraction of run notes + logger file comments via ``claude -p``.

One model call per track day receives both human inputs for the day — the
free-form run-notes ``.txt`` file(s) from the OneDrive notes folder and the
short "Long Comment" the driver typed into each AIM file (``sessions.comment``,
dataset schema 5) — together with the list of logged sessions (id, local
start time, car, track, laps). It returns one structured record per logged
session: which notes block describes it, cold pressures, tire compound,
track condition, setup changes, incidents. The model does the matching,
which used to be a separate time-window step.

The call is cached per day in ``data/tire_dataset/session_notes/<date>.json``
(committed), keyed by the hashes of the prompt, the model, the notes text
and the session listing: an unchanged day never re-invokes the model, and a
day gains a re-parse when a session is added or a note edited. The flat
table ``notes_matches.parquet`` is rebuilt from the caches every run; it is
what the compound labelling and the notes validation read.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, ValidationError

from .paths import DEFAULT_NOTES_ROOT, default_dataset_root, session_notes_dir, sessions_dir
from .tracks import get_track, known_canonicals

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"
_PROMPT_RESOURCE = ("motorsports_data_notebook.tire_etl", "session_notes_prompt.txt")
_FILENAME_DATE_RE = re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})")
_FALLBACK_TZ = ZoneInfo("Asia/Tokyo")
MATCHES_FILE = "notes_matches.parquet"


# ---------------------------------------------------------------------------
# Schema — mirrors the prompt exactly so validation is meaningful.
# ---------------------------------------------------------------------------


class CornerValue(BaseModel):
    fl: float | None = None
    fr: float | None = None
    rl: float | None = None
    rr: float | None = None


class CornerStr(BaseModel):
    fl: str | None = None
    fr: str | None = None
    rl: str | None = None
    rr: str | None = None


class SessionRecord(BaseModel):
    session_id: str
    note_file: str | None = None
    note_session_index: int | None = None
    match_confidence: float = 0.0
    weather_text: str | None = None
    ambient_temp_c: float | None = None
    track_condition: Literal["dry", "damp", "wet", "snow", "unknown"] = "unknown"
    cold_pressure_bar: CornerValue = Field(default_factory=CornerValue)
    tire_compound: CornerStr = Field(default_factory=CornerStr)
    new_tires: bool | None = None
    session_kind: Literal["practice", "qualifying", "race"] | None = None
    setup_changes: list[str] = Field(default_factory=list)
    incidents: list[str] = Field(default_factory=list)
    notes: str = ""


class DayData(BaseModel):
    sessions: list[SessionRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Inputs: notes files by day, logged sessions by day
# ---------------------------------------------------------------------------


def infer_date_track_from_filename(path: Path) -> tuple[_date | None, str | None]:
    """Best-effort ``(date, track_canonical)`` from a notes filename.

    Note files are named for the day ("2026-04-04 Tsukuba.txt",
    "Tsukuba KKSII 2026-04-04.txt"); the date in the name is authoritative.
    A filename can contain several track-ish tokens ("2025-10-05-Sodegaura
    Marutai"); the leftmost match wins.
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
    best_pos: int | None = None
    track: str | None = None
    for canonical in known_canonicals():
        head = canonical.split("_")[0]
        pos = lower.find(head)
        if pos >= 0 and (best_pos is None or pos < best_pos):
            best_pos = pos
            track = canonical
    return d, track


@dataclass(frozen=True)
class NotesFile:
    path: Path
    text: str

    @property
    def sha1(self) -> str:
        return _sha1(self.text)


@dataclass(frozen=True)
class LoggedSession:
    session_id: str
    date: _date
    start_local: str  # "HH:MM"
    car: str | None
    track_canonical: str | None
    n_laps: int
    session_type: str | None
    comment: str | None

    def listing_line(self, index: int) -> str:
        comment = (self.comment or "").replace("\n", " / ").strip()
        return (
            f"{index}. session_id={self.session_id} start={self.start_local} "
            f'car="{self.car or "?"}" track={self.track_canonical or "unknown"} '
            f'laps={self.n_laps} type="{self.session_type or ""}" comment="{comment}"'
        )


def notes_files_by_day(notes_root: Path) -> dict[_date, list[NotesFile]]:
    out: dict[_date, list[NotesFile]] = {}
    if not notes_root.exists():
        logger.warning("notes root %s does not exist", notes_root)
        return out
    for path in sorted(notes_root.glob("*.txt")):
        d, _ = infer_date_track_from_filename(path)
        if d is None:
            logger.warning("notes file %s has no date in its name — skipped", path.name)
            continue
        out.setdefault(d, []).append(NotesFile(path=path, text=path.read_text(encoding="utf-8")))
    return out


def _local_hhmm(start_utc: datetime | None, track_canonical: str | None) -> str:
    if start_utc is None:
        return "??:??"
    tz = _FALLBACK_TZ
    if track_canonical is not None:
        ti = get_track(track_canonical)
        if ti is not None:
            tz = ti.tzinfo()
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    return start_utc.astimezone(tz).strftime("%H:%M")


def logged_sessions_by_day(dataset_root: Path) -> dict[_date, list[LoggedSession]]:
    out: dict[_date, list[LoggedSession]] = {}
    files = sorted(sessions_dir(dataset_root).glob("*.parquet"))
    if not files:
        return out
    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables, promote_options="default")
    names = table.column_names
    cols = {
        n: table.column(n).to_pylist() if n in names else [None] * len(table)
        for n in (
            "session_id",
            "date",
            "session_start_utc",
            "car",
            "track_canonical",
            "n_laps",
            "session_type",
            "comment",
        )
    }
    rows = []
    for i in range(len(table)):
        d = cols["date"][i]
        if d is None:
            continue
        rows.append(
            LoggedSession(
                session_id=str(cols["session_id"][i]),
                date=d,
                start_local=_local_hhmm(cols["session_start_utc"][i], cols["track_canonical"][i]),
                car=cols["car"][i],
                track_canonical=cols["track_canonical"][i],
                n_laps=int(cols["n_laps"][i] or 0),
                session_type=cols["session_type"][i],
                comment=cols["comment"][i] if isinstance(cols["comment"][i], str) else None,
            )
        )
    for s in sorted(rows, key=lambda r: (r.date, r.start_local, r.session_id)):
        out.setdefault(s.date, []).append(s)
    return out


# ---------------------------------------------------------------------------
# The model call and its cache
# ---------------------------------------------------------------------------


def _load_system_prompt() -> str:
    package, resource = _PROMPT_RESOURCE
    return importlib.resources.files(package).joinpath(resource).read_text()


def _sha1(data: str) -> str:
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def build_day_input(day: _date, notes: list[NotesFile], sessions: list[LoggedSession]) -> str:
    parts = [f"DATE: {day.isoformat()} (times are local to the track)", ""]
    if notes:
        for nf in notes:
            parts += [f'RUN NOTES FILE "{nf.path.name}":', nf.text.strip(), ""]
    else:
        parts += ["RUN NOTES: none for this day.", ""]
    parts.append("LOGGED SESSIONS:")
    parts += [s.listing_line(i) for i, s in enumerate(sessions, start=1)]
    return "\n".join(parts) + "\n"


def _invoke_claude_p(
    user_text: str, system_prompt: str, model: str, timeout_s: float = 300.0
) -> str:
    """Call ``claude -p`` (non-interactive print mode) and return raw stdout."""
    combined = f"{system_prompt}\n\n---\n{user_text}"
    try:
        proc = subprocess.run(
            ["claude", "-p", "--model", model, combined],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "`claude` CLI not found on PATH. Install Claude Code: https://claude.com/claude-code"
        ) from e
    return proc.stdout.strip()


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json fences despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def _validate_day(payload: object, sessions: list[LoggedSession]) -> DayData:
    """Validate the model output and keep exactly one record per logged session."""
    data = DayData.model_validate(payload)
    wanted = {s.session_id for s in sessions}
    by_id: dict[str, SessionRecord] = {}
    for rec in data.sessions:
        if rec.session_id in wanted and rec.session_id not in by_id:
            by_id[rec.session_id] = rec
    dropped = len(data.sessions) - len(by_id)
    if dropped:
        logger.warning("dropped %d record(s) with unknown or duplicate session_id", dropped)
    missing = [s.session_id for s in sessions if s.session_id not in by_id]
    for sid in missing:
        by_id[sid] = SessionRecord(session_id=sid)
    if missing:
        logger.warning(
            "model omitted %d logged session(s); filled with empty records", len(missing)
        )
    return DayData(sessions=[by_id[s.session_id] for s in sessions])


@dataclass(frozen=True)
class ParsedDay:
    day: _date
    data: DayData
    from_cache: bool


def _cache_path(dataset_root: Path, day: _date) -> Path:
    return session_notes_dir(dataset_root) / f"{day.isoformat()}.json"


def parse_day(
    day: _date,
    notes: list[NotesFile],
    sessions: list[LoggedSession],
    *,
    dataset_root: Path,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    prompt_text: str | None = None,
) -> ParsedDay:
    """Return the validated, cached parse of one track day."""
    prompt_text = prompt_text if prompt_text is not None else _load_system_prompt()
    user_text = build_day_input(day, notes, sessions)
    cache_key = f"{_sha1(prompt_text)}|{model}|{_sha1(user_text)}"
    cache_path = _cache_path(dataset_root, day)
    if cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("_cache_key") == cache_key:
                data = _validate_day(cached["data"], sessions)
                return ParsedDay(day=day, data=data, from_cache=True)
        except (json.JSONDecodeError, ValidationError, KeyError):
            logger.warning("cache invalid for %s — re-parsing", cache_path.name)

    raw = _invoke_claude_p(user_text, prompt_text, model)
    cleaned = _strip_code_fences(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"claude -p did not return valid JSON for {day}: {e}\n{cleaned[:500]}"
        ) from e
    data = _validate_day(payload, sessions)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "_cache_key": cache_key,
        "_model": model,
        "_prompt_sha1": _sha1(prompt_text),
        "_extracted_at": datetime.now(timezone.utc).isoformat(),
        "_notes_files": [{"file": str(nf.path), "sha1": nf.sha1} for nf in notes],
        "_sessions": [s.session_id for s in sessions],
        "input": user_text,
        "data": data.model_dump(),
    }
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(cache_path)
    return ParsedDay(day=day, data=data, from_cache=False)


# ---------------------------------------------------------------------------
# The flat table
# ---------------------------------------------------------------------------

_MATCH_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("source_file", pa.string()),
        ("source_sha1", pa.string()),
        ("note_session_index", pa.int32()),
        ("match_confidence", pa.float32()),
        ("cold_pressure_bar_fl", pa.float32()),
        ("cold_pressure_bar_fr", pa.float32()),
        ("cold_pressure_bar_rl", pa.float32()),
        ("cold_pressure_bar_rr", pa.float32()),
        ("tire_compound_fl", pa.string()),
        ("tire_compound_fr", pa.string()),
        ("tire_compound_rl", pa.string()),
        ("tire_compound_rr", pa.string()),
        ("track_condition", pa.string()),
        ("ambient_temp_c", pa.float32()),
        ("weather_text", pa.string()),
        ("new_tires", pa.bool_()),
        ("session_kind", pa.string()),
        ("setup_changes", pa.string()),
        ("incidents", pa.string()),
        ("notes", pa.string()),
        ("comment", pa.string()),
    ]
)


def _has_content(rec: SessionRecord) -> bool:
    cp, tc = rec.cold_pressure_bar, rec.tire_compound
    return any(
        [
            rec.note_session_index is not None,
            rec.weather_text,
            rec.ambient_temp_c is not None,
            rec.track_condition != "unknown",
            any(v is not None for v in (cp.fl, cp.fr, cp.rl, cp.rr)),
            any(v for v in (tc.fl, tc.fr, tc.rl, tc.rr)),
            rec.new_tires is not None,
            rec.session_kind is not None,
            rec.setup_changes,
            rec.incidents,
            rec.notes,
        ]
    )


def days_to_table(
    parsed: list[ParsedDay], inputs: dict[_date, tuple[list[NotesFile], list[LoggedSession]]]
) -> pa.Table:
    """One row per logged session that the day's inputs said anything about."""
    rows: list[dict] = []
    for pd_ in parsed:
        notes, sessions = inputs[pd_.day]
        by_name = {nf.path.name: nf for nf in notes}
        comments = {s.session_id: s.comment for s in sessions}
        for rec in pd_.data.sessions:
            if not _has_content(rec):
                continue
            nf = by_name.get(rec.note_file or "") if rec.note_session_index is not None else None
            if nf is None and rec.note_session_index is not None and len(notes) == 1:
                nf = notes[0]
            rows.append(
                {
                    "session_id": rec.session_id,
                    "source_file": str(nf.path) if nf else None,
                    "source_sha1": nf.sha1 if nf else None,
                    "note_session_index": rec.note_session_index,
                    "match_confidence": rec.match_confidence,
                    "cold_pressure_bar_fl": rec.cold_pressure_bar.fl,
                    "cold_pressure_bar_fr": rec.cold_pressure_bar.fr,
                    "cold_pressure_bar_rl": rec.cold_pressure_bar.rl,
                    "cold_pressure_bar_rr": rec.cold_pressure_bar.rr,
                    "tire_compound_fl": rec.tire_compound.fl,
                    "tire_compound_fr": rec.tire_compound.fr,
                    "tire_compound_rl": rec.tire_compound.rl,
                    "tire_compound_rr": rec.tire_compound.rr,
                    "track_condition": rec.track_condition,
                    "ambient_temp_c": rec.ambient_temp_c,
                    "weather_text": rec.weather_text,
                    "new_tires": rec.new_tires,
                    "session_kind": rec.session_kind,
                    "setup_changes": "; ".join(rec.setup_changes) or None,
                    "incidents": "; ".join(rec.incidents) or None,
                    "notes": rec.notes or None,
                    "comment": comments.get(rec.session_id),
                }
            )
    rows.sort(key=lambda r: r["session_id"])
    return pa.Table.from_pylist(rows, schema=_MATCH_SCHEMA)


def write_matches(dataset_root: Path, table: pa.Table) -> Path:
    path = dataset_root / MATCHES_FILE
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=3)
    tmp.replace(path)
    return path


def run_enrich_notes(
    *,
    notes_root: Path | None = None,
    dataset_root: Path | None = None,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    workers: int = 4,
) -> dict[str, int]:
    """Parse every track day that has run notes or a session comment; rebuild the table."""
    if notes_root is None:
        notes_root = DEFAULT_NOTES_ROOT
    if dataset_root is None:
        dataset_root = default_dataset_root()
    prompt_text = _load_system_prompt()
    notes = notes_files_by_day(notes_root)
    sessions = logged_sessions_by_day(dataset_root)

    inputs: dict[_date, tuple[list[NotesFile], list[LoggedSession]]] = {}
    for day, sess in sessions.items():
        day_notes = notes.get(day, [])
        if day_notes or any(s.comment for s in sess):
            inputs[day] = (day_notes, sess)
    for day in notes:
        if day not in sessions:
            logger.warning("notes for %s but no logged session that day", day)

    counts = {"days": len(inputs), "cached": 0, "parsed": 0, "errors": 0, "rows": 0}

    def _one(day: _date) -> ParsedDay | None:
        day_notes, sess = inputs[day]
        try:
            return parse_day(
                day,
                day_notes,
                sess,
                dataset_root=dataset_root,
                model=model,
                force=force,
                prompt_text=prompt_text,
            )
        except Exception:
            logger.exception("failed to parse %s", day)
            return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(_one, sorted(inputs)))
    parsed = [r for r in results if r is not None]
    counts["errors"] = len(results) - len(parsed)
    counts["cached"] = sum(1 for r in parsed if r.from_cache)
    counts["parsed"] = sum(1 for r in parsed if not r.from_cache)

    # Stale caches (a day that no longer has inputs) are removed so the
    # committed directory mirrors the inputs.
    for stale in sorted(session_notes_dir(dataset_root).glob("*.json")):
        try:
            d = _date.fromisoformat(stale.stem)
        except ValueError:
            continue
        if d not in inputs:
            stale.unlink()

    table = days_to_table(parsed, inputs)
    write_matches(dataset_root, table)
    counts["rows"] = len(table)
    return counts
