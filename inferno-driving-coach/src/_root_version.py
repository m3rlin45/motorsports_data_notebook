"""Build-time helper: exposes the project version from the root pyproject.toml.

Setuptools resolves `tool.setuptools.dynamic.version = {attr = "_root_version.VERSION"}`
by importing this module, which reads `../../pyproject.toml` so every release
artifact ships at the same version with no per-file bump.

This module is intentionally outside any package and is excluded from the wheel.
"""

from __future__ import annotations

import pathlib
import tomllib

_ROOT_PYPROJECT = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
VERSION: str = tomllib.loads(_ROOT_PYPROJECT.read_text())["project"]["version"]
