"""Discover AIM session files and parse their filenames."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

from .tracks import normalize_track_name

# Filename examples:
#   CMD_Inferno 86_Tsukuba_Car_Generic testing_a_0192.xrk
#   CMD + Maruyama_Inferno 86_Sodegaura_a_0045.xrk
#   CMD_Inferno 86_Sodegaura_Qualifying testing_a_0226.xrk
#
# Be liberal: split on underscores, then heuristically assign. The suffix
# "_{letter}_{4-digit-run}.xrk" is the most reliable anchor — strip it first.
_SUFFIX_RE = re.compile(r"_([a-zA-Z])_(\d{3,5})\.xr[kz]$")


@dataclass(frozen=True)
class SessionCandidate:
    path: Path
    date: _date
    driver: str | None
    car: str | None
    track_raw: str | None
    track_canonical: str | None
    session_type: str | None
    run_num: int | None


def parse_filename(path: Path, date_hint: _date | None = None) -> SessionCandidate:
    """Parse an AIM filename into structured fields.

    The directory one level up is expected to be a ``YYYY-MM-DD`` folder; if so,
    we use it as the session date. Fields inside the filename are best-effort.
    """
    stem = path.name
    m = _SUFFIX_RE.search(stem)
    run_num: int | None = None
    if m:
        run_num = int(m.group(2))
        body = stem[: m.start()]
    else:
        body = path.stem

    parts = [p.strip() for p in body.split("_") if p.strip()]

    # Infer date: prefer parent dir name if it matches YYYY-MM-DD, else hint,
    # else fall back to today's date (should not happen in practice since the
    # AIM tree is date-partitioned).
    session_date = date_hint
    parent_name = path.parent.name
    try:
        y, mo, d = parent_name.split("-")
        session_date = _date(int(y), int(mo), int(d))
    except Exception:
        pass
    if session_date is None:
        session_date = _date(1970, 1, 1)

    # Positional heuristic: driver_car_track_[sessiontype...]
    driver = parts[0] if len(parts) >= 1 else None
    car = parts[1] if len(parts) >= 2 else None
    track_raw = parts[2] if len(parts) >= 3 else None
    session_type = " ".join(parts[3:]) if len(parts) > 3 else None

    track_canonical = normalize_track_name(track_raw) if track_raw else None

    return SessionCandidate(
        path=path,
        date=session_date,
        driver=driver,
        car=car,
        track_raw=track_raw,
        track_canonical=track_canonical,
        session_type=session_type,
        run_num=run_num,
    )


def scan_aim_tree(
    aim_root: Path,
    *,
    since: _date | None = None,
    only_car: str | None = None,
) -> list[SessionCandidate]:
    """Walk ``aim_root`` and return parsed session candidates.

    Prefers ``.xrk`` over ``.xrz`` when both exist for the same session (xrk
    is the loss-less raw format; xrz is compressed).
    """
    if not aim_root.exists():
        return []

    candidates: dict[Path, SessionCandidate] = {}
    # Two passes: collect xrk first, then xrz only where no xrk exists.
    for ext in (".xrk", ".xrz"):
        for p in sorted(aim_root.rglob(f"*{ext}")):
            # Skip drive-download / drive-upload temp areas
            if any(seg.startswith(".tmp.") for seg in p.parts):
                continue
            # If xrk already picked this session, skip the xrz.
            if ext == ".xrz":
                xrk = p.with_suffix(".xrk")
                if xrk in candidates:
                    continue
            cand = parse_filename(p)
            if since is not None and cand.date < since:
                continue
            if only_car is not None and (cand.car or "") != only_car:
                continue
            candidates[p] = cand

    return list(candidates.values())


def group_split_sessions(
    candidates: list[SessionCandidate],
    *,
    max_run_num_gap: int = 1,
) -> list[list[SessionCandidate]]:
    """Group candidates that belong to the same physical on-track session.

    The AIM logger creates a new file whenever the car restarts (e.g. after
    a spin or shutdown). Candidates that share the same
    ``(date, driver, car, track_canonical, session_type)`` and have
    consecutive ``run_num`` values within ``max_run_num_gap`` are treated as
    one logical session; their files should be loaded together and merged
    before downstream processing so the warm-up curve sees one continuous
    timeseries.

    The default ``max_run_num_gap=1`` means only strictly-consecutive
    ``run_num`` values group. Set higher to tolerate small gaps when the
    logger skipped numbers for unrelated activity in between.

    Returns a list of groups (each itself a list of SessionCandidate),
    each group sorted by ``run_num``. Singleton groups (one candidate) are
    included so callers can iterate uniformly.
    """
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda c: (
            c.date,
            c.driver or "",
            c.car or "",
            c.track_canonical or "",
            c.session_type or "",
            c.run_num if c.run_num is not None else -1,
        ),
    )
    groups: list[list[SessionCandidate]] = [[ordered[0]]]
    for cand in ordered[1:]:
        prev = groups[-1][-1]
        same_bucket = (
            prev.date == cand.date
            and prev.driver == cand.driver
            and prev.car == cand.car
            and prev.track_canonical == cand.track_canonical
            and prev.session_type == cand.session_type
        )
        sequential = (
            prev.run_num is not None
            and cand.run_num is not None
            and 0 < (cand.run_num - prev.run_num) <= max_run_num_gap
        )
        if same_bucket and sequential:
            groups[-1].append(cand)
        else:
            groups.append([cand])
    return groups
