"""Parse free-form run notes via ``claude -p``.

The LLM call is cached to disk (and committed) keyed by content hashes:
unchanged notes never re-invoke the model. The committed JSON IS the cache.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .paths import DEFAULT_NOTES_ROOT, default_dataset_root, notes_extracted_dir

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-7"
_PROMPT_RESOURCE = ("motorsports_data_notebook.tire_etl", "notes_system_prompt.txt")


# ---------------------------------------------------------------------------
# Pydantic schema — mirrors the prompt exactly so validation is meaningful.
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


class NoteSession(BaseModel):
    session_index: int
    start_time_local: str | None = None
    weather_text: str | None = None
    ambient_temp_c: float | None = None
    track_condition: Literal["dry", "damp", "wet", "snow", "unknown"] = "unknown"
    cold_pressure_bar: CornerValue = Field(default_factory=CornerValue)
    tire_compound: CornerStr = Field(default_factory=CornerStr)
    setup_changes: list[str] = Field(default_factory=list)
    incidents: list[str] = Field(default_factory=list)
    notes: str = ""


class NotesData(BaseModel):
    file_date: str | None = None
    track: str | None = None
    car: str | None = None
    sessions: list[NoteSession] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache + claude invocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedNotes:
    source_file: Path
    source_sha1: str
    prompt_sha1: str
    model: str
    extracted_at: str
    data: NotesData


def _load_system_prompt() -> str:
    package, resource = _PROMPT_RESOURCE
    return importlib.resources.files(package).joinpath(resource).read_text()


def _sha1(data: str) -> str:
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def _invoke_claude_p(
    notes_text: str,
    system_prompt: str,
    model: str,
    timeout_s: float = 300.0,
) -> str:
    """Call ``claude -p`` and return raw stdout.

    Uses the non-interactive print mode. We pass the notes text as the user
    prompt (suffixed so the system prompt leads), and request text output
    (which the prompt instructs to be pure JSON).
    """
    combined = f"{system_prompt}\n\n---\nNOTES FILE CONTENTS:\n{notes_text}\n"
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
        # Drop leading fence (optionally with language) and trailing fence.
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def parse_notes_file(
    path: Path,
    *,
    dataset_root: Path | None = None,
    model: str = DEFAULT_MODEL,
    force: bool = False,
) -> ParsedNotes:
    """Return validated, cached parse of a single notes .txt file."""
    if dataset_root is None:
        dataset_root = default_dataset_root()
    notes_text = path.read_text(encoding="utf-8", errors="replace")
    source_sha1 = _sha1(notes_text)
    prompt_text = _load_system_prompt()
    prompt_sha1 = _sha1(prompt_text)
    cache_key = f"{source_sha1}|{prompt_sha1}|{model}"

    cache_path = notes_extracted_dir(dataset_root) / f"{path.stem}.json"
    if cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("_cache_key") == cache_key:
                data = NotesData.model_validate(cached["data"])
                return ParsedNotes(
                    source_file=path,
                    source_sha1=source_sha1,
                    prompt_sha1=prompt_sha1,
                    model=model,
                    extracted_at=cached.get("_extracted_at", ""),
                    data=data,
                )
        except (json.JSONDecodeError, ValidationError, KeyError):
            logger.warning("cache invalid for %s — re-parsing", path.name)

    raw = _invoke_claude_p(notes_text, prompt_text, model)
    cleaned = _strip_code_fences(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"claude -p did not return valid JSON for {path.name}: {e}\n{cleaned[:500]}"
        ) from e

    data = NotesData.model_validate(payload)
    extracted_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "_cache_key": cache_key,
        "_source_file": str(path),
        "_source_sha1": source_sha1,
        "_prompt_sha1": prompt_sha1,
        "_model": model,
        "_extracted_at": extracted_at,
        "data": data.model_dump(),
    }
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(cache_path)

    return ParsedNotes(
        source_file=path,
        source_sha1=source_sha1,
        prompt_sha1=prompt_sha1,
        model=model,
        extracted_at=extracted_at,
        data=data,
    )


def run_enrich_notes(
    *,
    notes_root: Path | None = None,
    dataset_root: Path | None = None,
    model: str = DEFAULT_MODEL,
    force: bool = False,
) -> dict[str, int]:
    """Parse every .txt file under ``notes_root``."""
    if notes_root is None:
        notes_root = DEFAULT_NOTES_ROOT
    if dataset_root is None:
        dataset_root = default_dataset_root()
    counts = {"scanned": 0, "cached": 0, "parsed": 0, "errors": 0}
    if not notes_root.exists():
        logger.warning("notes root %s does not exist", notes_root)
        return counts

    for path in sorted(notes_root.glob("*.txt")):
        counts["scanned"] += 1
        try:
            before = (notes_extracted_dir(dataset_root) / f"{path.stem}.json").exists()
            parse_notes_file(path, dataset_root=dataset_root, model=model, force=force)
            # Classify as cache hit if file existed AND we're not forcing.
            if before and not force:
                counts["cached"] += 1
            else:
                counts["parsed"] += 1
        except Exception:
            logger.exception("failed to parse %s", path.name)
            counts["errors"] += 1
    return counts
