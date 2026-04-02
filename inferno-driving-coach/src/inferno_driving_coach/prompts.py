"""MCP prompt definitions for inferno-driving-coach."""

from __future__ import annotations

import importlib.resources

from . import mcp


def _load_prompt_data(filename: str) -> str:
    """Load a prompt template from the data directory."""
    ref = importlib.resources.files("inferno_driving_coach.data").joinpath(filename)
    return ref.read_text(encoding="utf-8")


@mcp.prompt()
def session_analysis(file_paths: str) -> str:
    """Analyze a motorsports telemetry session.

    Runs full session analysis on telemetry files, identifies improvement
    areas, sets KPIs, and generates a coaching report.

    Parameters
    ----------
    file_paths : str
        Space-separated paths to telemetry files (XRK/XRZ/IBT).
    """
    template = _load_prompt_data("prompt_session_analysis.md")
    return template.replace("{{file_paths}}", file_paths)


@mcp.prompt()
def session_day_review(directory_path: str) -> str:
    """Analyze all sessions from a track day.

    Discovers all telemetry files in the directory, analyzes each session
    sequentially, tracks progress across sessions, and generates a day summary.

    Parameters
    ----------
    directory_path : str
        Path to directory containing telemetry files.
    """
    template = _load_prompt_data("prompt_day_review.md")
    return template.replace("{{directory_path}}", directory_path)
