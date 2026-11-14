"""Human-curated corrections to what a recording says about itself.

A logger file names its car twice — the Race Studio ``Vehicle`` metadata
and the file-name token Race Studio derives from it — and both come from a
setting the driver changes by hand on the dash. When that setting was left
on the previous car, the file is wrong about which car it recorded, and
with it the session's id, its vehicle profile (channel names) and every
per-car pooling downstream. ``data/tire_dataset/session_overrides.yaml``
lists those files with the car they really recorded and the evidence; the
extractor applies it before anything else reads the car.

The file is keyed by recording, not by session: a session id contains the
car and so changes when the correction is applied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from .discovery import SessionCandidate

OVERRIDES_FILE = "session_overrides.yaml"

_REQUIRED = ("date", "file", "car", "reason")

OverrideKey = tuple[str, str]


@dataclass(frozen=True)
class FileOverride:
    """One corrected recording.

    Attributes
    ----------
    date : str
        The recording's track-day folder, ``YYYY-MM-DD``.
    file : str
        The file name without its extension, as Race Studio wrote it.
    car : str
        The car that was really recorded, spelled as its other sessions are.
    reason : str
        The evidence for the correction.
    """

    date: str
    file: str
    car: str
    reason: str

    @property
    def key(self) -> OverrideKey:
        return override_key(self.date, self.file)


def override_key(date: str, stem: str) -> OverrideKey:
    """Identity of a recording: its day folder and its stem, letter case
    ignored (the AIM tree lives on a Windows volume and the same recording
    exists as ``.xrk`` and ``.xrz``)."""
    return (str(date), stem.casefold())


def load_overrides(dataset_root: Path) -> dict[OverrideKey, FileOverride]:
    """Read ``session_overrides.yaml`` under ``dataset_root``.

    Parameters
    ----------
    dataset_root : Path
        The dataset directory (``data/tire_dataset``).

    Returns
    -------
    dict[OverrideKey, FileOverride]
        The corrections keyed by :func:`override_key`; empty when the file
        does not exist.

    Raises
    ------
    ValueError
        On an entry with a missing, empty or unknown field, an extension in
        ``file``, or two entries for one recording. A correction that is
        silently skipped would leave the wrong car in the dataset.
    """
    path = Path(dataset_root) / OVERRIDES_FILE
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping with a 'files' list")
    unknown_top = set(data) - {"files"}
    if unknown_top:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(unknown_top)}")
    entries = data.get("files") or []
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'files' must be a list")
    out: dict[OverrideKey, FileOverride] = {}
    for i, entry in enumerate(entries):
        where = f"{path}: files[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: expected a mapping")
        unknown = set(entry) - set(_REQUIRED)
        if unknown:
            raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}")
        values = {k: str(entry[k]).strip() if entry.get(k) is not None else "" for k in _REQUIRED}
        missing = [k for k in _REQUIRED if not values[k]]
        if missing:
            raise ValueError(f"{where}: missing {missing}")
        if values["file"].lower().endswith((".xrk", ".xrz")):
            raise ValueError(f"{where}: 'file' is the name without its extension")
        override = FileOverride(**values)
        if override.key in out:
            raise ValueError(f"{where}: second entry for {override.date}/{override.file}")
        out[override.key] = override
    return out


def override_for(
    overrides: Mapping[OverrideKey, FileOverride] | None, path: Path
) -> FileOverride | None:
    """The correction for the recording at ``path`` (``…/YYYY-MM-DD/<file>``), if any."""
    if not overrides:
        return None
    return overrides.get(override_key(path.parent.name, path.stem))


def apply_override(
    cand: SessionCandidate, overrides: Mapping[OverrideKey, FileOverride] | None
) -> SessionCandidate:
    """``cand`` with its car replaced when its file is listed; unchanged otherwise."""
    override = override_for(overrides, cand.path)
    if override is None:
        return cand
    return replace(cand, car=override.car)
