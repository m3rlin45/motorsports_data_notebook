"""GSI elevation tiles (the altitude check of gate B2), without the network."""

from __future__ import annotations

import numpy as np
import pytest

from motorsports_data_notebook.tire_model.lapsim import dem as demmod
from motorsports_data_notebook.tire_model.lapsim import gates
from motorsports_data_notebook.tire_model.lapsim import geometry as geo

from motorsports_data_notebook.tire_model.lapsim import track as trk

from ._synthetic import LAYOUT_ID, RADIUS_M, STRAIGHT_M, write_dataset


def _write_tile(cache, layer, zoom, x, y, values):
    path = cache / layer / str(zoom) / str(x) / f"{y}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(",".join(v for v in row) for row in values))


def _plane_tile(x0, y0):
    """A tile whose value at pixel (col, row) is 100 + col + 2·row, with
    one pixel of no data."""
    rows = [[f"{100 + c + 2 * r:.2f}" for c in range(256)] for r in range(256)]
    rows[y0][x0] = "e"
    return rows


class TestTiles:
    def test_tile_coordinates(self):
        x, y = demmod.tile_coordinates(np.array([0.0, 35.37218]), np.array([0.0, 138.92709]), 15)
        assert x[0] == pytest.approx(16384.0) and y[0] == pytest.approx(16384.0)
        assert int(x[1]) == 29029 and int(y[1]) == 12937

    def test_parse_tile(self):
        arr = demmod.parse_tile("1.5,e,3\n4,5,6\n")
        assert arr.shape == (256, 256)
        assert arr[0, 0] == 1.5 and np.isnan(arr[0, 1]) and arr[1, 2] == 6.0
        assert np.isnan(arr[5, 5])

    def test_bilinear_inside_a_cached_tile(self, tmp_path):
        _write_tile(tmp_path, "dem5a", 15, 29028, 12928, _plane_tile(200, 200))
        dem = demmod.GsiDem(tmp_path, fetch=False)
        # the point at pixel centre (10, 20) and one a quarter pixel on
        n = 2.0**15

        def _at(px, py):
            x, y = (29028 + (px + 0.5) / 256) / n, (12928 + (py + 0.5) / 256) / n
            lon = x * 360.0 - 180.0
            lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y))))
            return lat, lon

        lat, lon = zip(_at(10, 20), _at(10.25, 20.5), _at(200, 200))
        height, layer = dem.elevation(np.array(lat), np.array(lon))
        assert height[0] == pytest.approx(100 + 10 + 40)
        assert height[1] == pytest.approx(100 + 10.25 + 41.0)
        assert layer[0] == "dem5a"
        # no data in the laser model there and no other layer cached
        assert np.isnan(height[2]) and layer[2] == ""

    def test_falls_back_to_the_next_layer(self, tmp_path):
        _write_tile(tmp_path, "dem5a", 15, 29028, 12928, _plane_tile(10, 20))
        coarse = [["7.00"] * 256 for _ in range(256)]
        _write_tile(tmp_path, "dem", 14, 14514, 6464, coarse)
        dem = demmod.GsiDem(tmp_path, fetch=False)
        n = 2.0**15
        x, y = (29028 + 10.5 / 256) / n, (12928 + 20.5 / 256) / n
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y))))
        height, layer = dem.elevation(np.array([lat]), np.array([x * 360.0 - 180.0]))
        assert height[0] == pytest.approx(7.0) and layer[0] == "dem"

    def test_offline_and_uncached_is_no_data(self, tmp_path):
        dem = demmod.GsiDem(tmp_path, fetch=False)
        height, layer = dem.elevation(np.array([35.0]), np.array([139.0]))
        assert np.isnan(height[0]) and layer[0] == ""
        assert not any(tmp_path.iterdir())


class _FlatGround:
    """Ground at 55 m everywhere but a 6 m dip 80 m long: a bridge."""

    def __init__(self, gate, dip_at=None):
        self.gate, self.dip_at = gate, dip_at

    def elevation(self, lat, lon):
        xy = geo.local_xy(lat, lon, self.gate.lat, self.gate.lon)
        height = np.full(len(lat), 55.0)
        if self.dip_at is not None:
            height[np.hypot(xy[:, 0] - self.dip_at[0], xy[:, 1] - self.dip_at[1]) < 40.0] -= 6.0
        return height, np.full(len(lat), "dem5a", dtype=object)


class _TiltedGround:
    """Ground rising to the east by ``slope``."""

    def __init__(self, gate, slope):
        self.gate, self.slope = gate, slope

    def elevation(self, lat, lon):
        xy = geo.local_xy(lat, lon, self.gate.lat, self.gate.lon)
        return 55.0 + self.slope * xy[:, 0], np.full(len(lat), "dem5a", dtype=object)


class TestAltitudeCheck:
    @pytest.fixture()
    def root(self, tmp_path):
        geo._INDEX_CACHE.clear()
        write_dataset(
            tmp_path,
            {
                f"2026-01-1{i}_s{i}": {
                    "car": "car",
                    "date": f"2026-01-1{i}",
                    "kwargs": {"n_laps": 5, "seed": i, "noise_m": 0.02},
                }
                for i in range(3)
            },
        )
        return tmp_path

    def test_flat_track_on_flat_ground(self, root):
        ground = _FlatGround(geo.layout_gate(LAYOUT_ID))
        report = gates.check_track_truth(root, layouts=(LAYOUT_ID,), dem=ground)
        item = {i.name: i for i in report.items}["Altitude against the GSI DEM"]
        assert item.passed is True
        assert "0 bridge notches" in item.measured

    def test_a_bridge_is_masked_and_listed(self, root):
        # the middle of the east straight: x = 100 m, y = 0
        ground = _FlatGround(geo.layout_gate(LAYOUT_ID), dip_at=(100.0, 0.0))
        report = gates.check_track_truth(root, layouts=(LAYOUT_ID,), dem=ground)
        item = {i.name: i for i in report.items}["Altitude against the GSI DEM"]
        assert item.passed is True
        assert "1 bridge notches" in item.measured
        assert "-6" in item.detail

    def test_without_a_model_the_check_is_not_measured(self, root, tmp_path):
        report = gates.check_track_truth(root, layouts=(LAYOUT_ID,), tracks_dir=tmp_path / "tracks")
        items = {i.name: i for i in report.items}
        item = items["Altitude against the GSI DEM"]
        assert item.passed is None and item.measured == "not measured"
        assert items["Bank from the GSI DEM"].measured == "not measured"
        track = trk.TrackModel.load(tmp_path / "tracks" / f"{LAYOUT_ID}.json")
        assert len(track.bank) == 0 and "bank" not in track.meta

    def test_the_track_gets_its_bank_from_the_model(self, root, tmp_path):
        ground = _TiltedGround(geo.layout_gate(LAYOUT_ID), 0.04)
        report = gates.check_track_truth(
            root, layouts=(LAYOUT_ID,), dem=ground, tracks_dir=tmp_path / "tracks"
        )
        item = {i.name: i for i in report.items}["Bank from the GSI DEM"]
        assert item.passed is None  # reported
        assert "0 nodes filled" in item.measured and "straight_sd" in item.detail
        track = trk.TrackModel.load(tmp_path / "tracks" / f"{LAYOUT_ID}.json")
        assert len(track.bank) == track.n and (track.bank_flag == 0).all()
        # up the first straight the car goes north and the low side is on its
        # left; down the other it goes south and the high side is
        north = int(round(30.0 / track.ds_m))
        south = int(round((STRAIGHT_M / 2 + np.pi * RADIUS_M + 100.0) / track.ds_m))
        assert track.bank[north] == pytest.approx(-np.arctan(0.04), abs=2e-3)
        assert track.bank[south] == pytest.approx(np.arctan(0.04), abs=2e-3)
        assert track.meta["bank"]["nodes"] == track.n
        # the stadium turns right: at the first corner's apex the car heads
        # east, up the slope, and the road does not lean
        apex = int(round((STRAIGHT_M / 2 + np.pi * RADIUS_M / 2) / track.ds_m))
        assert abs(track.bank[apex]) < 3e-3

    def test_local_lat_lon_inverts_local_xy(self):
        gate = geo.layout_gate(LAYOUT_ID)
        x, y = np.array([0.0, 250.0, -900.0]), np.array([0.0, -400.0, 1200.0])
        lat, lon = geo.local_lat_lon(x, y, gate.lat, gate.lon)
        back = geo.local_xy(lat, lon, gate.lat, gate.lon)
        assert back[:, 0] == pytest.approx(x, abs=1e-6)
        assert back[:, 1] == pytest.approx(y, abs=1e-6)
