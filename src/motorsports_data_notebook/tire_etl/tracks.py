"""Track registry: canonical names, coordinates, timezones.

Maintained by hand — when a new track appears in filenames, add it here with
approximate lat/lon (main straight is fine) and the IANA timezone. Coordinates
are only used to query Open-Meteo historical weather, so 3 decimal places is
plenty.
"""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class TrackInfo:
    canonical: str  # stable ID for joins and directory names
    display: str  # human-readable name
    lat: float
    lon: float
    timezone: str  # IANA TZ; consumers wrap in ZoneInfo

    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


# Canonical list. Keys are the lowercased track name tokens as they appear in
# AIM filenames (e.g. "Tsukuba"). Alias map below handles common variants.
_TRACKS: dict[str, TrackInfo] = {
    "tsukuba_2000": TrackInfo(
        canonical="tsukuba_2000",
        display="Tsukuba Circuit (2000)",
        lat=36.170,
        lon=140.218,
        timezone="Asia/Tokyo",
    ),
    "sodegaura": TrackInfo(
        canonical="sodegaura",
        display="Sodegaura Forest Raceway",
        lat=35.412,
        lon=139.991,
        timezone="Asia/Tokyo",
    ),
    "fuji": TrackInfo(
        canonical="fuji",
        display="Fuji Speedway",
        lat=35.372,
        lon=138.927,
        timezone="Asia/Tokyo",
    ),
    "motegi": TrackInfo(
        canonical="motegi",
        display="Mobility Resort Motegi",
        lat=36.533,
        lon=140.228,
        timezone="Asia/Tokyo",
    ),
    "marutai": TrackInfo(
        canonical="marutai",
        display="Marutai Circuit",
        lat=35.290,
        lon=140.140,
        timezone="Asia/Tokyo",
    ),
    "suzuka": TrackInfo(
        canonical="suzuka",
        display="Suzuka Circuit",
        lat=34.844,
        lon=136.537,
        timezone="Asia/Tokyo",
    ),
    "minami": TrackInfo(
        # "MINAMI" in AIM filenames — appears to be a Japanese regional course.
        # Coordinates approximate; correct later if weather joins look off.
        canonical="minami",
        display="Minami (TBD)",
        lat=36.170,
        lon=140.218,
        timezone="Asia/Tokyo",
    ),
}

# Map arbitrary filename tokens (after lowercasing + stripping) to canonical IDs.
_ALIASES: dict[str, str] = {
    "tsukuba": "tsukuba_2000",
    "tsukuba2000": "tsukuba_2000",
    "tc2000": "tsukuba_2000",
    "sodegaura": "sodegaura",
    "fuji": "fuji",
    "fujispeedway": "fuji",
    "fujigp": "fuji",  # AIM files encode the GP layout as "Fuji GP" in the filename
    "fujishort": "fuji",
    "fujigpsh": "fuji",  # "Fuji GP Sh" (short/shortened GP layout variant)
    "motegi": "motegi",
    "motegieast": "motegi",
    "marutai": "marutai",
    "suzuka": "suzuka",
    "suzukacar": "suzuka",
    "minami": "minami",
}


def normalize_track_name(raw: str) -> str | None:
    """Map a raw filename track token to a canonical track ID.

    Returns None if the token is unknown. Callers should then record the raw
    value in the ``track`` column and leave ``track_canonical`` as null.
    """
    key = "".join(c.lower() for c in raw if c.isalnum())
    return _ALIASES.get(key)


def get_track(canonical: str) -> TrackInfo | None:
    return _TRACKS.get(canonical)


def known_canonicals() -> list[str]:
    return sorted(_TRACKS.keys())
