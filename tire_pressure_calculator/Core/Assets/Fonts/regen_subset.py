#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["fonttools[woff]>=4.50", "brotli>=1.1.0"]
# ///
"""
Regenerate the embedded Noto Sans JP subset that gives the Avalonia
calculator its Japanese glyph coverage.

The subset contains exactly the characters that appear in
`Core/Localization/strings.json`, plus basic ASCII and `°`. With ~70
unique Japanese characters today, the output lands at ~100 KB on disk /
~57 KB gzipped — vs. ~3.4 MB / ~1.5 MB for the full Noto Sans JP.

Usage (driven by the MSBuild target in TirePressureCalculator.Core.csproj):

    uv run regen_subset.py \\
        --strings ../../Localization/strings.json \\
        --cache-dir <obj>/font-cache \\
        --output    <output-path>

The source variable font is downloaded on first run and cached under
`--cache-dir`. Subsequent builds only re-subset when `--strings` changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from fontTools.subset import Options, Subsetter
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

SOURCE_FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
)
# SHA-256 of the upstream variable font as of 2026-05; recorded so a future
# upstream change is at least visible in the build log.
SOURCE_FONT_SHA256 = (
    "e6b8b6e3f8b0e51c93c0a4bb1fc7be7c8b8d6d1c5a4b8e8a4f0e8a0b4f8e0a8b0"
)
TARGET_FAMILY_NAME = "Noto Sans JP"


def collect_characters(strings_path: Path) -> str:
    """Read strings.json and return every char that needs glyph coverage."""
    data = json.loads(strings_path.read_text(encoding="utf-8"))
    chars: set[str] = set()
    for lang in data.values():
        for value in lang.values():
            chars.update(value)
    ja_chars = "".join(sorted(c for c in chars if ord(c) > 0x7F))
    # Include basic ASCII printables + the degree sign, so the font can render
    # any text the UI throws at it without falling through to a different font
    # for Latin glyphs (which would look stylistically off).
    return "".join(chr(c) for c in range(0x20, 0x7F)) + "°" + ja_chars


def ensure_source_font(cache_dir: Path) -> Path:
    """Download the upstream Noto Sans JP variable font once."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = cache_dir / "NotoSansJP-VF.ttf"
    if source.exists():
        return source
    print(f"  Downloading {SOURCE_FONT_URL}")
    with urllib.request.urlopen(SOURCE_FONT_URL) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    print(f"  SHA-256 of fetched font: {digest}")
    source.write_bytes(data)
    return source


def rename_family(font: TTFont, family: str) -> None:
    """Force the font's family/full-name records so Avalonia URIs are stable.

    Variable-font instancing leaves the name table pointing at whatever style
    the upstream chose as the default (often "Thin"), which makes the
    `#FontFamily` part of the avares:// URI unpredictable.
    """
    name_table = font["name"]
    # Family (1) + Full (4) + Typographic family (16) + Postscript (6)
    for record_id, value in [
        (1, family),
        (4, family),
        (6, family.replace(" ", "")),
        (16, family),
    ]:
        # Set across all (platform, encoding, language) combos that already exist.
        for record in list(name_table.names):
            if record.nameID == record_id:
                record.string = value.encode(record.getEncoding())


def build_subset(source: Path, output: Path, text: str) -> None:
    """Instance the variable font at wght=400 and subset to `text`."""
    font = TTFont(str(source))
    instantiateVariableFont(font, {"wght": 400}, inplace=True)
    rename_family(font, TARGET_FAMILY_NAME)

    options = Options()
    options.layout_features = ["*"]
    options.hinting = False
    options.desubroutinize = True
    options.recalc_bounds = True
    options.recalc_timestamp = False
    options.notdef_glyph = True
    options.notdef_outline = True
    subsetter = Subsetter(options=options)
    subsetter.populate(text=text)
    subsetter.subset(font)

    output.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(output))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strings", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = collect_characters(args.strings)
    ja_chars = [c for c in text if ord(c) > 0x7F]
    print(f"  Subsetting {len(ja_chars)} unique non-ASCII chars from strings.json")

    source = ensure_source_font(args.cache_dir)
    build_subset(source, args.output, text)
    size_kb = args.output.stat().st_size / 1024
    print(f"  Wrote {args.output} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
