"""Tests for track name normalization."""

from __future__ import annotations

from motorsports_data_notebook.tire_etl.tracks import (
    get_track,
    known_canonicals,
    normalize_track_name,
)


def test_tsukuba_aliases() -> None:
    assert normalize_track_name("Tsukuba") == "tsukuba_2000"
    assert normalize_track_name("tsukuba") == "tsukuba_2000"
    assert normalize_track_name("Tsukuba 2000") == "tsukuba_2000"


def test_unknown_track_returns_none() -> None:
    assert normalize_track_name("Atlanta") is None


def test_fuji_layout_variants_all_collapse_to_fuji() -> None:
    """AIM filenames can carry "Fuji GP" (main grand-prix layout) or
    "Fuji Short" (shortened layout); both are the same physical venue for
    weather purposes."""
    assert normalize_track_name("Fuji") == "fuji"
    assert normalize_track_name("Fuji GP") == "fuji"
    assert normalize_track_name("Fuji Short") == "fuji"
    assert normalize_track_name("Fuji Speedway") == "fuji"


def test_get_track_returns_info() -> None:
    ti = get_track("fuji")
    assert ti is not None
    assert ti.display == "Fuji Speedway"
    assert 35 < ti.lat < 36


def test_known_canonicals_covers_expected() -> None:
    assert set(known_canonicals()) == {
        "tsukuba_2000",
        "sodegaura",
        "fuji",
        "motegi",
        "marutai",
        "suzuka",
        "minami",
    }


def test_suzuka_and_layout_aliases() -> None:
    assert normalize_track_name("Suzuka") == "suzuka"
    assert normalize_track_name("Suzuka Car") == "suzuka"


def test_motegi_east_collapses_to_motegi() -> None:
    """Motegi East is a layout of the same venue as Motegi; lat/lon identical."""
    assert normalize_track_name("Motegi East") == "motegi"
    assert normalize_track_name("Motegi") == "motegi"


def test_fuji_gp_sh_alias() -> None:
    """Fuji GP Sh is the shortened GP layout; same venue as fuji."""
    assert normalize_track_name("Fuji GP Sh") == "fuji"


def test_dash_variant_aliases_pool_to_venue() -> None:
    """Dash variants (Venue metadata) collapse to the venue; layouts.py decides
    which layout was actually driven from GPS."""
    assert normalize_track_name("Tsukuba_Car") == "tsukuba_2000"
    assert normalize_track_name("Tsukuba_Bike") == "tsukuba_2000"
    assert normalize_track_name("Suzuka West") == "suzuka"
    assert normalize_track_name("Suzuka East") == "suzuka"
    assert normalize_track_name("MINAMI_CHIBA_1") == "minami"


def test_minami_is_minami_chiba() -> None:
    ti = get_track("minami")
    assert ti is not None
    assert 35.4 < ti.lat < 35.6
    assert 140.2 < ti.lon < 140.3
