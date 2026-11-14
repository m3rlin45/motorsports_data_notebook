#!/usr/bin/env python3
"""Check that a NativeAOT publish of the analyzer contains exactly the files it should.

    python3 desktop_app_cs/tools/check_publish.py <publish dir> <rid> [--graphics-beside]

The executable carries the .NET runtime, the app and the libxrk/libibt parsers
(static libraries, see Directory.Build.targets). A self-extracting publish
(-p:InfernoSelfExtract=true, the default expectation here) also carries the
graphics libraries Avalonia loads at run time, so the executable is the only
file; with --graphics-beside those libraries are expected as files next to it.
Debug symbols are ignored. Anything else (a parser DLL, a `runtimes/` folder,
a stray data file) fails the check, as does a missing file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Runtime identifier -> (executable, graphics libraries Avalonia loads at run time)
EXPECTED = {
    "win-x64": (
        "InfernoAnalyzer.exe",
        {"libSkiaSharp.dll", "libHarfBuzzSharp.dll", "av_libglesv2.dll"},
    ),
    "linux-x64": ("InfernoAnalyzer", {"libSkiaSharp.so", "libHarfBuzzSharp.so"}),
}
SYMBOL_SUFFIXES = {".pdb", ".dbg"}


def check(publish: Path, rid: str, graphics_beside: bool) -> list[str]:
    """Return the problems found in `publish` for runtime `rid` (empty when it is as expected)."""
    exe, graphics = EXPECTED[rid]
    wanted = {exe} | (graphics if graphics_beside else set())
    files = {p.relative_to(publish).as_posix() for p in publish.rglob("*") if p.is_file()}
    shipped = {f for f in files if Path(f).suffix not in SYMBOL_SUFFIXES}
    problems = [f"missing: {name}" for name in sorted(wanted - shipped)]
    problems += [f"unexpected: {name}" for name in sorted(shipped - wanted)]
    return problems


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if a != "--graphics-beside"]
    if len(args) != 2 or args[1] not in EXPECTED:
        rids = "|".join(sorted(EXPECTED))
        print(
            f"usage: check_publish.py <publish dir> <{rids}> [--graphics-beside]", file=sys.stderr
        )
        return 2
    publish = Path(args[0])
    for path in sorted(publish.rglob("*")):
        if path.is_file():
            print(f"{path.stat().st_size:>12,}  {path.relative_to(publish).as_posix()}")
    problems = check(publish, args[1], "--graphics-beside" in argv)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
