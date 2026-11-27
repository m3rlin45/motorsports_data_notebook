"""The committed export of the simulated laps (the cutover; #208's adoption).

The pressure model's simulated entries are made at every retrain from the
laps in ``data/lapsim/exports/`` (:mod:`..sim_entries`): per raw car and
committed track, the pace table of the raw car's latest powertrain as the
heat chain bins it (:class:`~.export.TableExport`), with the raw car's fed
use (the use of a session's median flying lap, :func:`~.car_gate.fed_use`:
the lap an entry stands for). A heat is of one fit's shares, so the export
holds bins and the retrain splits them through its own; nothing in it
moves when the pressure model is refitted.

It is made as the standing holdout's ``served`` arm makes a cell's
(:func:`~.pressure_gate.held_out_drive`, here with nothing held out): the
car fitted on all its laps, its limit, the driver's braking use and the
typical use; the fed use per raw car, the layout's own where the raw car
has run it, else the median over its layouts. ``lapsim export-entries``
writes it, after the gates it rests on (B2's tracks, C2's car fit) on the
same data.

The tracks carry the road's bank from GSI's elevation tiles, so the export
and every simulated entry made from it carry GSI's attribution
(``data/lapsim/README.md``; :data:`ATTRIBUTION`).
"""

from __future__ import annotations

import json
import logging
from datetime import date as _date
from pathlib import Path
from typing import Any

from ...tire_etl.paths import repo_root
from . import car as cm
from . import car_gate as cg
from . import pace
from .export import TableExport

logger = logging.getLogger(__name__)

#: The attribution GSI's terms ask for, of anything computed from a lap
#: driven on the banked tracks (``data/lapsim/README.md``).
ATTRIBUTION = {
    "ja": (
        "出典：国土地理院ウェブサイト（地理院タイル：https://maps.gsi.go.jp/development/ichiran.html）"
        "「地理院タイル（標高タイル（基盤地図情報数値標高モデル））を加工して作成」"
    ),
    "en": (
        "Source: GSI Tiles (elevation tiles, Fundamental Geospatial Data digital elevation "
        "model), Geospatial Information Authority of Japan; processed by this project."
    ),
}
INDEX = "index.json"
#: The export's numbers are committed to this many significant digits (the
#: lead, 2026-11-27: it halves the files and moves no arm by 0.001 mbar).
DIGITS = 7


def default_exports() -> Path:
    """Where the export is committed (``data/lapsim/exports``)."""
    return repo_root() / "data" / "lapsim" / "exports"


def file_name(raw_car: str, layout_id: str) -> str:
    return f"{raw_car.replace(' ', '_')}__{layout_id}.json"


def latest_powertrain(raw_car: str, today: str) -> str:
    """The powertrain a raw car has now (the one an entry is served for:
    the calculators have no date)."""
    return cm.powertrain_of(raw_car, today)


def build_exports(
    sim: Any,
    tracks_dir: Path,
    *,
    today: str | None = None,
    made: dict | None = None,
) -> tuple[dict[tuple[str, str], TableExport], list[dict]]:
    """Every raw car's export at every committed track (``tracks_dir``'s
    ``<layout>.json``): ``(exports, notes)``, ``notes`` saying per raw car
    and layout what was made or why not. ``sim`` is
    :func:`~.pressure_gate.load_sim`'s bundle."""
    from .pressure_gate import held_out_drive

    today = today or _date.today().isoformat()
    layouts = sorted(k for k in sim.tracks if (Path(tracks_dir) / f"{k}.json").exists())
    out: dict[tuple[str, str], TableExport] = {}
    notes: list[dict] = []
    for family in sorted(sim.full):
        drive, fed, rows = held_out_drive(sim, family, set())
        if drive is None:
            notes.append({"car": family, "why": "no car model"})
            continue
        # the wet laps' range too: the slow end a wet target is flagged against
        ranged = cm.CarModel.from_json(drive.model.to_json())
        ranged.meta = {
            **drive.model.meta,
            "use_range_wet": cg.use_range(drive, sim.tracks, rows, family, basis="wet"),
        }
        model = ranged
        engines = model.power.get("engines") or {}
        for raw_car in sorted(fed):
            engine = latest_powertrain(raw_car, today)
            if engines and engine not in engines:
                notes.append({"car": raw_car, "powertrain": engine, "why": "#378: never logged"})
                continue
            own = fed[raw_car]
            for layout in layouts:
                lay = own["layouts"].get(layout)
                table = pace.build_table(
                    model, sim.tracks[layout], powertrain=engine if engines else None
                )
                exp = table.laps_export(made=made)
                exp.raw_car = raw_car
                exp.fed = {
                    "use": float((lay or own)["use"]),
                    "from": (
                        "the raw car's sessions at this layout"
                        if lay
                        else "the median over the raw car's layouts (it has not run this one)"
                    ),
                    "sessions": int(lay["sessions"]) if lay else 0,
                    "layouts": sorted(own["layouts"]),
                }
                exp.assumes = {**exp.assumes, "powertrain": engine if engines else None}
                out[(raw_car, layout)] = exp.rounded(DIGITS)
                notes.append({"car": raw_car, "layout_id": layout, "made": True})
    return out, notes


def write_exports(
    exports: dict[tuple[str, str], TableExport], out_dir: Path, notes: list[dict], made: dict
) -> Path:
    """Write every export and an index (what is there, when and at which
    commit it was made, GSI's attribution, and the notes); files of an
    earlier export that are not in this one are removed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = set()
    for (raw_car, layout), exp in sorted(exports.items()):
        name = file_name(raw_car, layout)
        exp.save(out_dir / name)
        names.add(name)
    for old in out_dir.glob("*__*.json"):
        if old.name not in names:
            old.unlink()
    index = {
        "made": made,
        "attribution": ATTRIBUTION,
        "files": [
            {"raw_car": raw_car, "layout_id": layout, "file": file_name(raw_car, layout)}
            for raw_car, layout in sorted(exports)
        ],
        "notes": notes,
    }
    path = out_dir / INDEX
    path.write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")
    return path


def load_exports(out_dir: Path) -> tuple[list[TableExport], dict]:
    """The committed export: ``(exports, index)``; nothing where there is
    no index."""
    path = Path(out_dir) / INDEX
    if not path.exists():
        return [], {}
    index = json.loads(path.read_text())
    return [TableExport.load(Path(out_dir) / f["file"]) for f in index["files"]], index


def _made(root: Path) -> dict:
    """When, from which dataset and at which commit the export is made."""
    import subprocess
    from datetime import datetime, timezone

    commit = ""
    try:
        got = subprocess.run(
            ["sl", "log", "-r", ".", "-T", "{node}"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        commit = got.stdout.strip()
    except OSError:
        commit = ""
    manifest = Path(root) / "MANIFEST.jsonl"
    return {
        "at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "commit": commit,
        "dataset_sessions": (
            sum(1 for _ in manifest.open()) if manifest.exists() else 0  # one line a recording
        ),
    }


def export_entries(
    root: Path,
    *,
    cache_dir: Path | None,
    tracks_dir: Path,
    folds_dir: Path | None,
    out_dir: Path | None = None,
) -> Any:
    """``lapsim export-entries``: build and write the committed export
    (module docstring); the report lists what was made, the fed uses and
    the files' sizes."""
    import pandas as pd

    from .gates import GateItem, GateReport, md_table
    from .pressure_gate import load_sim

    out_dir = Path(out_dir) if out_dir is not None else default_exports()
    rep = GateReport("EX", "the committed export of the simulated laps (`lapsim export-entries`)")
    sim = load_sim(root, cache_dir=cache_dir, tracks_dir=tracks_dir, folds_dir=folds_dir)
    made = _made(root)
    exports, notes = build_exports(sim, tracks_dir, made=made)
    index = write_exports(exports, out_dir, notes, made)
    rows: list[dict[str, Any]] = []
    for (raw_car, layout), exp in sorted(exports.items()):
        f = out_dir / file_name(raw_car, layout)
        rows.append(
            {
                "raw_car": raw_car,
                "layout_id": layout,
                "powertrain": exp.assumes.get("powertrain"),
                "fed_use": round(float(exp.fed["use"]), 4),
                "fed_lap_s": round(exp.lap_time_at(float(exp.fed["use"])), 2),
                "fed_from": "layout" if exp.fed.get("sessions") else "car",
                "knots": len(exp.use),
                "kb": round(f.stat().st_size / 1024.0, 1),
            }
        )
    total = sum(float(r["kb"]) for r in rows)
    rep.items.append(
        GateItem(
            "exports written",
            None,
            f"{len(rows)} files, {total:.0f} kB, in {out_dir} (index {index.name})",
            "reported",
        )
    )
    missing = [n for n in notes if not n.get("made")]
    rep.notes.append("**Made.**\n\n" + md_table(pd.DataFrame(rows)))
    if missing:
        rep.notes.append("**Not made.**\n\n" + md_table(pd.DataFrame(missing)))
    return rep
