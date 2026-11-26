"""Which frame the pressure model's heat input is in (#376).

``tyres`` (the default since the owner's promotion of 2026-11-26) rebuilds
the lateral and longitudinal accelerations from each session's GPS laid on
the committed track, in the frame a simulated lap is in
(:mod:`.lapsim.tyres_frame`): what the tyres hold on the road's bank and
``dV/dt`` with the grade. A model fitted on it is served a simulated lap
with a frame factor of 1, and no sensor verdict is read anywhere.
``sensor`` is the heat the model was fitted on before: the accelerations
the sensor verdict chose for each 20 s window, the accelerometer or a
GPS-derived fallback (:func:`~..tire_etl.kinematics.load_kinematics`); it
is kept for comparison (``--heat-frame sensor``), and only there does a
frame factor exist.

The frame is a query-time choice, never the extractor's. Every loader of
the fit's kinematics goes through :func:`load_heat_kinematics`, which reads
the frame set by :func:`using` (``tyres`` outside any); a function that
takes ``heat_frame=None`` means the frame in force (:func:`resolve`). A
fitted model records it as ``heat_input.frame`` (absent: ``sensor``, a
model fitted before the option existed), and the scorers set it from the
model they are given (:func:`frame_of`). The lap-sim gates other than the
standing holdout read heat in the frame of the committed
``tire_model.json`` (:func:`model_frame`), the model they compare with.

Where the tyres' frame cannot be built as such, it says so and falls back
(:func:`fallbacks`; a warning once per layout or session, and the fitted
model's ``heat_input.frame_fallbacks``):

- a layout with no committed track in ``data/lapsim/tracks/`` (a new
  circuit, before gate B2 has built and committed its track): its sessions
  are in the ground frame, bank and grade 0;
- a session with no GPS geometry (no layout, a layout the lap simulation's
  geometry does not know, or no GPS family): it has no heat in this frame
  and is left out of the fit. (The fit already leaves out a session with no
  track, :func:`~.warmup_table._load_filtered_laps`.)

A dataset root with no sessions table cannot be placed at all: that is a
:class:`HeatFrameError`, never a silent skip.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

    from .lapsim.track import TrackModel

SENSOR = "sensor"
TYRES = "tyres"
FRAMES = (SENSOR, TYRES)

logger = logging.getLogger(__name__)

_STATE: dict[str, Any] = {"frame": TYRES, "tracks_dir": None}
# per (dataset root, tracks dir): session -> layout, and layout -> track (or None)
_LAYOUTS: dict[str, dict[str, str]] = {}
_TRACKS: dict[tuple[str, str], "TrackModel | None"] = {}
# what fell back, this process: layout -> sessions in the ground frame; session -> why left out
_GROUND: dict[str, set[str]] = {}
_LEFT_OUT: dict[str, str] = {}


def fallbacks(session_ids: set[str] | None = None) -> dict[str, Any]:
    """What the tyres' frame fell back on in this process, for the
    sessions ``session_ids`` (all when None): ``ground_frame_layouts``
    (layout -> its sessions put in the ground frame, no committed track)
    and ``left_out_sessions`` (session -> why it has no heat in this
    frame)."""
    keep = (lambda s: True) if session_ids is None else (lambda s: s in session_ids)
    ground = {lay: sorted(s for s in sids if keep(s)) for lay, sids in _GROUND.items()}
    return {
        "ground_frame_layouts": {lay: sids for lay, sids in sorted(ground.items()) if sids},
        "left_out_sessions": {s: why for s, why in sorted(_LEFT_OUT.items()) if keep(s)},
    }


class HeatFrameError(RuntimeError):
    """The tyres' frame cannot be built for a dataset at all (it has no
    sessions table to place a session on its layout)."""


def _left_out(session_id: str, why: str) -> None:
    if session_id not in _LEFT_OUT:
        _LEFT_OUT[session_id] = why
        logger.warning("tyres' frame: %s has no heat in this frame (%s): left out", session_id, why)


def check(frame: str) -> str:
    """``frame`` itself; ``ValueError`` when it is not one of :data:`FRAMES`."""
    if frame not in FRAMES:
        raise ValueError(f"heat frame must be one of {FRAMES}; got {frame!r}")
    return frame


def current() -> str:
    """The frame the kinematics loaders hand out now."""
    return str(_STATE["frame"])


def resolve(frame: str | None) -> str:
    """``frame``, or the frame in force when it is None (checked)."""
    return check(frame if frame is not None else current())


def frame_of(model: dict[str, Any]) -> str:
    """The frame a fitted model's heat is in (``heat_input.frame``; a model
    without one is ``sensor``)."""
    hi = model.get("heat_input") or {}
    return check(str(hi.get("frame", SENSOR)))


def model_frame(root: Path) -> str:
    """The frame of the committed model at ``root`` (``tire_model.json``),
    or the frame in force where there is none: the frame a gate that
    compares with production reads heat in."""
    import json

    path = Path(root) / "tire_model.json"
    if not path.exists():
        return current()
    return frame_of(json.loads(path.read_text()))


def default_tracks_dir() -> Path:
    """The committed tracks (``data/lapsim/tracks``)."""
    from .lapsim.cli import default_tracks

    return default_tracks()


@contextmanager
def using(frame: str, tracks_dir: Path | None = None) -> Iterator[str]:
    """Hand out heat in ``frame`` inside the block (the tracks from
    ``tracks_dir``, the committed ones by default), and restore the frame
    in force before it on the way out."""
    check(frame)
    before = dict(_STATE)
    _STATE["frame"] = frame
    if tracks_dir is not None:
        _STATE["tracks_dir"] = Path(tracks_dir)
    try:
        yield frame
    finally:
        _STATE.clear()
        _STATE.update(before)


def _tracks_dir() -> Path:
    got = _STATE.get("tracks_dir")
    return Path(got) if got is not None else default_tracks_dir()


def session_layout(root: Path, session_id: str) -> str | None:
    """The layout a session was driven on (the ``sessions`` table)."""
    key = str(Path(root).resolve())
    if key not in _LAYOUTS:
        from .lapsim import geometry as geo

        import pandas as pd

        try:
            ses = geo.load_sessions(Path(root))
        except FileNotFoundError as exc:
            raise HeatFrameError(
                f"the tyres' frame places each session on its layout from the sessions table, "
                f"and {root} has none: fit in the sensor's frame instead (`--heat-frame sensor`, "
                f"`heat_frame.using('sensor')`)"
            ) from exc
        _LAYOUTS[key] = {
            str(s): str(lay)
            for s, lay in zip(ses["session_id"], ses["layout_id"])
            if not pd.isna(lay) and str(lay)
        }
    return _LAYOUTS[key].get(str(session_id))


def layout_track(layout_id: str, tracks_dir: Path | None = None) -> "TrackModel | None":
    """The committed dry track of ``layout_id`` (``<tracks_dir>/<layout>.json``),
    read once; None where there is none. Never built here: a fit must not
    write a track."""
    from .lapsim.track import TrackModel

    base = Path(tracks_dir) if tracks_dir is not None else _tracks_dir()
    key = (str(base.resolve()), str(layout_id))
    if key not in _TRACKS:
        path = base / f"{layout_id}.json"
        _TRACKS[key] = TrackModel.load(path) if path.exists() else None
        if _TRACKS[key] is None:
            logger.warning(
                "tyres' frame: no committed track for layout %s in %s: its sessions are in the "
                "ground frame (bank and grade 0). Build it with `just tire-lapsim "
                "check-track-truth` (gate B2) and commit it to data/lapsim/tracks/.",
                layout_id,
                base,
            )
    return _TRACKS[key]


def load_heat_kinematics(
    root: Path, session_id: str, frame: str | None = None
) -> "pa.Table | None":
    """The session's kinematics as the fit reads them, in ``frame`` (the
    one in force when None): ``load_kinematics`` itself for ``sensor``;
    for ``tyres`` the same clock, laps and speed with both accelerations
    rebuilt from GPS and the track (:func:`.lapsim.tyres_frame.tyres_kinematics`).
    None when the session has no usable series in that frame."""
    from ..tire_etl.kinematics import load_kinematics

    frame = resolve(frame)
    kin = load_kinematics(Path(root), session_id)
    if kin is None or frame == SENSOR:
        return kin
    from .lapsim.tyres_frame import tyres_kinematics

    from .lapsim.geometry import LAYOUTS

    sid = str(session_id)
    layout = session_layout(root, sid)
    if layout is None:
        _left_out(sid, "no layout in the sessions table")
        return None
    if layout not in LAYOUTS:
        _left_out(sid, f"layout {layout} is not in the lap simulation's geometry")
        return None
    track = layout_track(layout)
    out = tyres_kinematics(Path(root), sid, kin, layout, track)
    if out is None:
        _left_out(sid, "no GPS geometry")
    elif track is None:
        _GROUND.setdefault(layout, set()).add(sid)
    return out
