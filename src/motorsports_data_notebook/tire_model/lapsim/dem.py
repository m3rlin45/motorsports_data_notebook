"""Ground elevation from the GSI elevation tiles (the check of gate B2).

The Geospatial Information Authority of Japan serves its elevation models
as text tiles of 256 × 256 values (``https://maps.gsi.go.jp/development/
demtile.html``): the 5 m airborne-laser model ``dem5a`` at zoom 15, with
the photogrammetric ``dem5b`` / ``dem5c`` and the 10 m ``dem`` (zoom 14)
behind it where the laser model has no cover. Tiles are fetched once and
kept in ``workspace/lapsim/gsi/``.

A track's ``alt_m`` and ``grade`` come from the logs and the model only
checks them. The road's bank is the one thing read from the model itself
(:mod:`.bank`, #299): the logs do not give it. It is kept with the track,
and the tracks are committed with it in ``data/lapsim/tracks/`` (owner,
2026-11-24, #208: "commit the bank track info, add attribution"). The
attribution GSI's content terms ask for, the source and a note that the
content was processed, is in ``data/lapsim/README.md``:
出典：国土地理院ウェブサイト, 「地理院タイル（標高タイル（基盤地図情報数値標高
モデル））を加工して作成」. Read for that ruling: the Survey Act Q&A (Q1-16:
processing without reproduction needs no approval; Q2-4: a map made from
the elevation model does) and the content terms, section 1; neither
addresses the numeric outputs of a simulation, and the owner ruled that
attribution is enough. The raw tiles themselves are not committed.

The model is bare earth, so a bridge deck reads the ground beneath it.
Heights are above mean sea level where GPS altitude is above the
ellipsoid: the two differ by a constant over a circuit, which neither a
grade nor a relief sees.
"""

from __future__ import annotations

import logging
import math
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

GSI_URL = "https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.txt"
#: Layers in order of preference: ``(name, zoom)``.
LAYERS: tuple[tuple[str, int], ...] = (("dem5a", 15), ("dem5b", 15), ("dem5c", 15), ("dem", 14))
TILE = 256


def tile_coordinates(lat: np.ndarray, lon: np.ndarray, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractional web-mercator tile coordinates ``(x, y)`` at ``zoom``."""
    n = 2.0**zoom
    x = (np.asarray(lon, float) + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(np.asarray(lat, float)))) / math.pi) / 2.0 * n
    return x, y


def parse_tile(text: str) -> np.ndarray:
    """A text tile as a 256 × 256 array; ``e`` (no data) becomes NaN."""
    rows = [r for r in text.strip().splitlines() if r]
    out = np.full((TILE, TILE), np.nan)
    for i, row in enumerate(rows[:TILE]):
        vals = row.split(",")[:TILE]
        out[i, : len(vals)] = [np.nan if v.strip() in ("e", "") else float(v) for v in vals]
    return out


class GsiDem:
    """Elevation at points from cached GSI tiles; ``fetch=False`` reads the
    cache only (a tile not in it counts as having no data)."""

    def __init__(self, cache_dir: Path, *, fetch: bool = True, timeout_s: float = 30.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.fetch = fetch
        self.timeout_s = timeout_s
        self._tiles: dict[tuple[str, int, int, int], np.ndarray | None] = {}

    def _path(self, layer: str, z: int, x: int, y: int) -> Path:
        return self.cache_dir / layer / str(z) / str(x) / f"{y}.txt"

    def tile(self, layer: str, z: int, x: int, y: int) -> np.ndarray | None:
        """The tile's values, or None where the layer has no such tile."""
        key = (layer, z, x, y)
        if key in self._tiles:
            return self._tiles[key]
        path = self._path(layer, z, x, y)
        text: str | None = None
        if path.exists():
            text = path.read_text()
        elif self.fetch:
            url = GSI_URL.format(layer=layer, z=z, x=x, y=y)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    text = resp.read().decode()
            except urllib.error.HTTPError as err:
                if err.code != 404:
                    raise
                text = ""  # the layer has no tile here; remember that
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        arr = parse_tile(text) if text else None
        self._tiles[key] = arr
        return arr

    def _sample(self, layer: str, zoom: int, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        x, y = tile_coordinates(lat, lon, zoom)
        out = np.full(len(x), np.nan)
        # a tile value belongs to the centre of its pixel
        px, py = x * TILE - 0.5, y * TILE - 0.5
        tx, ty = np.floor(x).astype(int), np.floor(y).astype(int)
        for txi, tyi in {(int(a), int(b)) for a, b in zip(tx, ty)}:
            arr = self.tile(layer, zoom, txi, tyi)
            if arr is None:
                continue
            m = (tx == txi) & (ty == tyi)
            u = np.clip(px[m] - txi * TILE, 0.0, TILE - 1.0)
            v = np.clip(py[m] - tyi * TILE, 0.0, TILE - 1.0)
            u0 = np.minimum(np.floor(u).astype(int), TILE - 2)
            v0 = np.minimum(np.floor(v).astype(int), TILE - 2)
            fu, fv = u - u0, v - v0
            out[m] = (
                arr[v0, u0] * (1 - fu) * (1 - fv)
                + arr[v0, u0 + 1] * fu * (1 - fv)
                + arr[v0 + 1, u0] * (1 - fu) * fv
                + arr[v0 + 1, u0 + 1] * fu * fv
            )
        return out

    def elevation(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(height_m, layer)`` at each point: bilinear in the first layer
        of :data:`LAYERS` that has data there; NaN and ``""`` where none
        has."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        height = np.full(lat.shape, np.nan)
        used = np.full(lat.shape, "", dtype=object)
        for layer, zoom in LAYERS:
            todo = ~np.isfinite(height)
            if not todo.any():
                break
            got = self._sample(layer, zoom, lat[todo], lon[todo])
            idx = np.flatnonzero(todo)[np.isfinite(got)]
            height[idx] = got[np.isfinite(got)]
            used[idx] = layer
        return height, used
