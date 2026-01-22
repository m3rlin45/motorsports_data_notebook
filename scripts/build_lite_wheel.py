#!/usr/bin/env python3
"""
Build a wheel without dependencies for JupyterLite.

This creates a copy of the wheel with all Requires-Dist entries removed,
so it can be installed in JupyterLite without dependency conflicts.
"""
import re
import zipfile
from pathlib import Path


def strip_dependencies_from_wheel(input_wheel: Path, output_dir: Path) -> Path:
    """Remove all Requires-Dist from a wheel's METADATA."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wheel = output_dir / input_wheel.name

    with zipfile.ZipFile(input_wheel, "r") as zin:
        with zipfile.ZipFile(output_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.namelist():
                content = zin.read(item)

                # Modify METADATA to remove dependencies
                if item.endswith("METADATA"):
                    text = content.decode("utf-8")
                    # Remove all Requires-Dist lines
                    text = re.sub(r"^Requires-Dist:.*\n?", "", text, flags=re.MULTILINE)
                    # Remove Provides-Extra lines too
                    text = re.sub(r"^Provides-Extra:.*\n?", "", text, flags=re.MULTILINE)
                    content = text.encode("utf-8")

                zout.writestr(item, content)

    return output_wheel


def main():
    dist_dir = Path("dist")
    pypi_dir = Path("pypi")

    # Find the wheel
    wheels = list(dist_dir.glob("motorsports_data_notebook*.whl"))
    if not wheels:
        print("Error: No wheel found in dist/. Run 'poetry build -f wheel' first.")
        return 1

    wheel = wheels[0]
    output = strip_dependencies_from_wheel(wheel, pypi_dir)
    print(f"Created {output} (dependencies stripped for JupyterLite)")
    return 0


if __name__ == "__main__":
    exit(main())
