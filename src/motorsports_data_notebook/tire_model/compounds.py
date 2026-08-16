"""Per-session tire-compound labels for model training and prediction.

Two sources, in priority order:

1. ``data/tire_dataset/tire_compounds.yaml`` — the human-curated sidecar
   (per-session, per-axle). Authoritative wherever it has a value.
2. ``notes_matches.parquet`` — compounds extracted from run notes by the
   LLM stage, normalized here. Wheel-set labels (``SET:<name>``) resolve
   through the sidecar's optional ``wheel_sets`` mapping (e.g. the 86's
   black wheels carried the RE-71RS set through 2025).

Sessions with no label from either source stay unlabeled and train/predict
on the pooled per-car parameters exactly as before.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

COMPOUNDS_FILE = "tire_compounds.yaml"

# Canonical spellings for the tires this team runs. Extend as new tires
# appear in the notes.
_CANONICAL = ("A052", "RE-71RS", "A050", "NS-2R", "AD09", "CR-S", "ZIII")


def normalize_compound(raw: object) -> str | None:
    """Map a free-text compound mention to its canonical name, or None.

    ``SET:<name>`` wheel-set labels are returned verbatim (upper-cased name)
    so the caller can resolve them via the sidecar's wheel_sets mapping.
    """
    if raw is None or not isinstance(raw, str) or not raw.strip():
        return None
    s = re.sub(r"[\s\-_]", "", raw.upper())
    if s.startswith("SET:"):
        return "SET:" + s[4:].lower()
    if "A052" in s or s == "052":
        return "A052"
    if "71RS" in s or "RE71" in s:
        return "RE-71RS"
    if "NS2R" in s:
        return "NS-2R"
    if "A050" in s or s == "050":
        return "A050"
    if "AD09" in s:
        return "AD09"
    if "CRS" in s:
        return "CR-S"
    if "ZIII" in s or s == "Z3":
        return "ZIII"
    return None


def load_compound_labels(dataset_root: Path) -> pd.DataFrame:
    """Return per-session compound labels: columns
    ``session_id, compound_front, compound_rear, source``.

    Sidecar rows win over notes-derived rows; either axle may be null.
    """
    import yaml  # local import to keep top-of-module deps minimal

    frames: list[pd.DataFrame] = []
    wheel_sets: dict[str, str] = {}

    sidecar_path = dataset_root / COMPOUNDS_FILE
    if sidecar_path.exists():
        doc = yaml.safe_load(sidecar_path.read_text(encoding="utf-8")) or {}
        for name, compound in (doc.get("wheel_sets") or {}).items():
            canon = normalize_compound(compound)
            if canon and not canon.startswith("SET:"):
                wheel_sets[str(name).lower()] = canon
        rows = []
        for entry in doc.get("sessions") or []:
            front = normalize_compound(entry.get("front"))
            rear = normalize_compound(entry.get("rear"))
            if front is None and rear is None:
                continue
            rows.append(
                {
                    "session_id": entry["session_id"],
                    "compound_front": front,
                    "compound_rear": rear,
                    "source": "sidecar",
                }
            )
        if rows:
            frames.append(pd.DataFrame(rows))

    matches_path = dataset_root / "notes_matches.parquet"
    if matches_path.exists():
        m = pq.read_table(
            matches_path,
            columns=[
                "session_id",
                "tire_compound_fl",
                "tire_compound_rl",
                "match_confidence",
            ],
        ).to_pandas()

        def _resolve(raw: object) -> str | None:
            canon = normalize_compound(raw)
            if canon is None:
                return None
            if canon.startswith("SET:"):
                return wheel_sets.get(canon[4:])
            return canon

        m["compound_front"] = m["tire_compound_fl"].map(_resolve)
        m["compound_rear"] = m["tire_compound_rl"].map(_resolve)
        m = m[(m["compound_front"].notna()) | (m["compound_rear"].notna())]
        if len(m) > 0:
            # One note-session may match several telemetry sessions; keep the
            # highest-confidence row per session.
            m = (
                m.sort_values("match_confidence", ascending=False)
                .drop_duplicates("session_id")
                .assign(source="notes")
            )
            frames.append(m[["session_id", "compound_front", "compound_rear", "source"]])

    if not frames:
        return pd.DataFrame(columns=["session_id", "compound_front", "compound_rear", "source"])

    out = pd.concat(frames, ignore_index=True)
    # Sidecar first (it was appended first) — drop notes rows it covers.
    out = out.drop_duplicates("session_id", keep="first").reset_index(drop=True)
    return out
