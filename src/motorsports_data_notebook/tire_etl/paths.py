"""Path resolution for tire ETL inputs and outputs."""

from __future__ import annotations

from pathlib import Path

DEFAULT_AIM_ROOT = Path("/mnt/c/AIM_SPORT/RaceStudio3/user/data")
DEFAULT_NOTES_ROOT = Path("/mnt/c/Users/m3rli/OneDrive/Documents/Car Running Notes")


def repo_root() -> Path:
    """Return the repository root (three parents up from this file)."""
    # src/motorsports_data_notebook/tire_etl/paths.py -> repo root
    return Path(__file__).resolve().parents[3]


def default_dataset_root() -> Path:
    """Committed dataset directory inside the repo."""
    return repo_root() / "data" / "tire_dataset"


def sessions_dir(dataset_root: Path) -> Path:
    return dataset_root / "sessions"


def laps_dir(dataset_root: Path) -> Path:
    return dataset_root / "laps"


def timeseries_dir(dataset_root: Path) -> Path:
    return dataset_root / "timeseries"


def notes_extracted_dir(dataset_root: Path) -> Path:
    return dataset_root / "notes_extracted"


def weather_dir(dataset_root: Path) -> Path:
    return dataset_root / "weather_hourly"


def manifest_path(dataset_root: Path) -> Path:
    return dataset_root / "MANIFEST.jsonl"


def schema_version_path(dataset_root: Path) -> Path:
    return dataset_root / "schema_version.txt"
