#!/usr/bin/env python3
"""Check that a NativeAOT publish of the analyzer contains exactly the files it should.

    python3 desktop_app_cs/tools/check_publish.py <publish dir> <rid>

The executable carries the .NET runtime, the app and the libxrk/libibt parsers
(static libraries, see Directory.Build.targets). The only files allowed beside
it are the graphics libraries Avalonia loads at run time, which ship as shared
libraries only, and debug symbols. Anything else (a parser DLL, a `runtimes/`
folder, a stray data file) fails the check, as does a missing file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Runtime identifier -> (executable, graphics libraries shipped beside it)
EXPECTED = {
    "win-x64": (
        "InfernoAnalyzer.exe",
        {"libSkiaSharp.dll", "libHarfBuzzSharp.dll", "av_libglesv2.dll"},
    ),
    "linux-x64": ("InfernoAnalyzer", {"libSkiaSharp.so", "libHarfBuzzSharp.so"}),
}
SYMBOL_SUFFIXES = {".pdb", ".dbg"}


def check(publish: Path, rid: str) -> list[str]:
    """Return the problems found in `publish` for runtime `rid` (empty when it is as expected)."""
    exe, graphics = EXPECTED[rid]
    files = {p.relative_to(publish).as_posix() for p in publish.rglob("*") if p.is_file()}
    shipped = {f for f in files if Path(f).suffix not in SYMBOL_SUFFIXES}
    problems = [f"missing: {name}" for name in sorted(({exe} | graphics) - shipped)]
    problems += [f"unexpected: {name}" for name in sorted(shipped - {exe} - graphics)]
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[2] not in EXPECTED:
        print(
            f"usage: check_publish.py <publish dir> <{'|'.join(sorted(EXPECTED))}>", file=sys.stderr
        )
        return 2
    publish = Path(argv[1])
    for path in sorted(publish.rglob("*")):
        if path.is_file():
            print(f"{path.stat().st_size:>12,}  {path.relative_to(publish).as_posix()}")
    problems = check(publish, argv[2])
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
