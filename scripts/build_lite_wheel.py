#!/usr/bin/env python3
"""
Strip all dependencies from a wheel for JupyterLite.

Used for external wheels (e.g. libxrk) whose native dependencies are already
provided by the Pyodide runtime and would fail to resolve via micropip.
"""

import re
import sys
import zipfile
from pathlib import Path


def strip_dependencies_from_wheel(input_wheel: Path, output_dir: Path) -> Path:
    """Remove all Requires-Dist and Provides-Extra from a wheel's METADATA."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wheel = output_dir / input_wheel.name

    with zipfile.ZipFile(input_wheel, "r") as zin:
        with zipfile.ZipFile(output_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.namelist():
                content = zin.read(item)

                if item.endswith("METADATA"):
                    text = content.decode("utf-8")
                    text = re.sub(r"^Requires-Dist:.*\n?", "", text, flags=re.MULTILINE)
                    text = re.sub(r"^Provides-Extra:.*\n?", "", text, flags=re.MULTILINE)
                    content = text.encode("utf-8")

                zout.writestr(item, content)

    return output_wheel


def main():
    if len(sys.argv) < 2:
        print("Usage: build_lite_wheel.py <wheel_path>")
        return 1

    wheel = Path(sys.argv[1])
    if not wheel.exists():
        print(f"Error: Wheel not found: {wheel}")
        return 1

    output = strip_dependencies_from_wheel(wheel, Path("pypi"))
    print(f"Created {output} (dependencies stripped)")
    return 0


if __name__ == "__main__":
    exit(main())
