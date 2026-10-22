#!/usr/bin/env python3
"""Build the libxrk/libibt FFI libraries the C# analyzer loads.

Fetches each parser repository at its pinned commit (a plain tarball from
GitHub, no git needed) into `desktop_app_cs/native/src/`, builds the FFI
crate with cargo for each requested runtime, and copies the shared library
to `desktop_app_cs/native/runtimes/<rid>/native/` where the C# projects pick
it up (see Core/InfernoAnalyzer.Core.csproj and Core/Io/NativeLoader.cs).

    python3 desktop_app_cs/tools/build_native.py                  # host runtime only
    python3 desktop_app_cs/tools/build_native.py --rids linux-x64 win-x64
    python3 desktop_app_cs/tools/build_native.py --source-libxrk ~/src/libxrk  # a local checkout

Cross-compiling win-x64 from Linux needs the `x86_64-pc-windows-gnu` rustup
target and the mingw-w64 linker (`gcc-mingw-w64-x86-64`). Everything else
is stdlib + cargo.
"""

from __future__ import annotations

import argparse
import io
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "desktop_app_cs" / "native"

# Pinned parser commits: the ones carrying the FFI crates.
PARSERS = {
    "libxrk": {"repo": "m3rlin45/libxrk", "ref": "015ad78a5e61ae96b32aa45b7a785633af48a670", "crate": "libxrk-ffi", "lib": "xrk_ffi"},
    "libibt": {"repo": "m3rlin45/libibt", "ref": "6e8fb20c980be4be34589f19820daad7ac43eea7", "crate": "libibt-ffi", "lib": "ibt_ffi"},
}

# .NET runtime identifier -> (rust target triple, shared-library file name pattern)
RIDS = {
    "linux-x64": ("x86_64-unknown-linux-gnu", "lib{lib}.so"),
    "linux-arm64": ("aarch64-unknown-linux-gnu", "lib{lib}.so"),
    "win-x64": ("x86_64-pc-windows-gnu", "{lib}.dll"),
    "osx-x64": ("x86_64-apple-darwin", "lib{lib}.dylib"),
    "osx-arm64": ("aarch64-apple-darwin", "lib{lib}.dylib"),
}


def host_rid() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    os_name = {"Linux": "linux", "Windows": "win", "Darwin": "osx"}.get(system)
    if os_name is None:
        raise SystemExit(f"unsupported host OS: {system}")
    return f"{os_name}-{arch}"


def fetch_source(name: str, spec: dict) -> Path:
    """The parser checkout at its pinned ref (downloaded once)."""
    dest = NATIVE / "src" / f"{name}-{spec['ref'][:12]}"
    if (dest / "Cargo.toml").exists():
        return dest
    url = f"https://codeload.github.com/{spec['repo']}/tar.gz/{spec['ref']}"
    print(f"fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https host
        data = response.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(tmp, filter="data")
    (inner,) = [p for p in tmp.iterdir() if p.is_dir()]
    shutil.rmtree(dest, ignore_errors=True)
    inner.rename(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def build(name: str, spec: dict, source: Path, rid: str, out_root: Path) -> Path:
    triple, pattern = RIDS[rid]
    cmd = ["cargo", "build", "--release", "-p", spec["crate"]]
    target_dir = source / "target" / "release"
    # The host runtime builds with the default toolchain target (MSVC on a
    # Windows host); any other runtime is a cross build for the listed triple.
    if rid != host_rid():
        cmd += ["--target", triple]
        target_dir = source / "target" / triple / "release"
    print(f"[{name}] {' '.join(cmd)}  (in {source})", file=sys.stderr)
    subprocess.run(cmd, cwd=source, check=True)
    built = target_dir / pattern.format(lib=spec["lib"])
    if not built.exists():
        raise SystemExit(f"[{name}] expected {built} after the build")
    dest_dir = out_root / rid / "native"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / built.name
    shutil.copy2(built, dest)
    print(f"[{name}] -> {dest} ({dest.stat().st_size // 1024} KiB)", file=sys.stderr)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rids", nargs="+", default=[host_rid()], choices=sorted(RIDS), help="runtimes to build (default: host)")
    parser.add_argument("--only", nargs="+", choices=sorted(PARSERS), help="parsers to build (default: all)")
    parser.add_argument("--out", type=Path, default=NATIVE / "runtimes")
    for name in PARSERS:
        parser.add_argument(f"--source-{name}", type=Path, help=f"local {name} checkout to build instead of the pinned ref")
    args = parser.parse_args(argv)

    for name, spec in PARSERS.items():
        if args.only and name not in args.only:
            continue
        source = getattr(args, f"source_{name}") or fetch_source(name, spec)
        for rid in args.rids:
            build(name, spec, Path(source).resolve(), rid, args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
