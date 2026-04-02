#!/usr/bin/env python3
"""
Download Pyodide built-in packages for the piplite index.

Some packages (like pyarrow) are compiled C extensions available in the Pyodide
distribution but not installable via micropip from PyPI. This script downloads
their wheels from the Pyodide CDN so piplite can serve them locally.
"""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

# Packages to download from the Pyodide CDN for the piplite index
PYODIDE_PACKAGES = ["pyarrow", "pyodide-unix-timezones"]


def main():
    pypi_dir = Path("pypi")
    pypi_dir.mkdir(parents=True, exist_ok=True)

    version = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('pyodide-build'))",
        ],
        text=True,
    ).strip()
    base_url = f"https://cdn.jsdelivr.net/pyodide/v{version}/full"

    lock_url = f"{base_url}/pyodide-lock.json"
    print(f"Fetching lock file from {lock_url}")
    with urllib.request.urlopen(lock_url) as resp:
        lock = json.loads(resp.read())

    for pkg_name in PYODIDE_PACKAGES:
        pkg = lock["packages"].get(pkg_name)
        if not pkg:
            print(f"Warning: {pkg_name} not found in Pyodide {version}")
            continue

        filename = pkg["file_name"]
        url = f"{base_url}/{filename}"
        dest = pypi_dir / filename

        if dest.exists():
            print(f"Already exists: {dest}")
            continue

        print(f"Downloading {filename}")
        urllib.request.urlretrieve(url, dest)

    print("Done")


if __name__ == "__main__":
    main()
