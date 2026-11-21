"""The road's bank from an elevation model (stand-in grounds, no tiles)."""

from __future__ import annotations

import numpy as np
import pytest

from motorsports_data_notebook.tire_model.lapsim import bank as bankmod
from motorsports_data_notebook.tire_model.lapsim import geometry as geo
from motorsports_data_notebook.tire_model.lapsim import track as trk

GATE = {"lat": 35.0, "lon": 139.0}
DS = 2.0


def _circle(radius: float = 100.0, *, right: bool = True, alt: float = 55.0) -> trk.TrackModel:
    """A circular track of ``radius`` round the gate's origin, driven
    clockwise (a right turn all the way) or the other way."""
    n = int(round(2 * np.pi * radius / DS))
    a = np.arange(n) * (2 * np.pi / n) * (1.0 if right else -1.0)
    # at angle a from north, going clockwise: east = R sin a, north = R cos a
    x, y = radius * np.sin(a), radius * np.cos(a)
    return trk.TrackModel(
        layout_key="circle",
        track_canonical="Circle",
        length_m=n * DS,
        ds_m=DS,
        kappa_heat=np.full(n, (1.0 if right else -1.0) / radius),
        x_m=x,
        y_m=y,
        alt_m=np.full(n, alt),
        grade=np.zeros(n),
        n_laps=np.full(n, 10),
        kappa_se=np.zeros(n),
        gate=dict(GATE),
        kappa_v=np.zeros(n),
    )


class _Ground:
    """Ground whose height is a function of the metres east and north of
    the gate."""

    def __init__(self, height, layer="dem5a"):
        self.height, self.layer = height, layer

    def elevation(self, lat, lon):
        xy = geo.local_xy(lat, lon, GATE["lat"], GATE["lon"])
        return self.height(xy[:, 0], xy[:, 1]), np.full(len(lat), self.layer, dtype=object)


class TestRoadBank:
    def test_a_bowl_banks_every_node_into_the_turn(self):
        # ground rising 5 % with the distance from the centre: the outside is higher
        ground = _Ground(lambda x, y: 55.0 + 0.05 * np.hypot(x, y))
        for right in (True, False):
            track = _circle(right=right)
            got = bankmod.road_bank(track, ground)
            assert got is not None and (got.flag == 0).all()
            into = bankmod.into_turn(got.left_rad, track.kappa_heat)
            assert into == pytest.approx(np.arctan(0.05), abs=2e-4)
            # the left side is the outside of a right turn, the inside of a left one
            assert np.sign(got.left_rad[0]) == (1.0 if right else -1.0)
            assert got.info["into_corners_mean_rad"] == pytest.approx(np.arctan(0.05), abs=2e-4)
            assert got.info["layers"] == {"dem5a": 1.0}

    def test_a_tilted_plane_is_read_by_the_direction_of_travel(self):
        # ground rising 3 % to the east: left of a car going north is west, the low side
        track = _circle()
        got = bankmod.road_bank(track, _Ground(lambda x, y: 55.0 + 0.03 * x))
        assert got is not None
        north = int(np.argmin(np.abs(track.x_m + 100.0)))  # the westmost node, going north
        south = int(np.argmax(track.x_m))  # the eastmost, going south
        assert got.left_rad[north] == pytest.approx(-np.arctan(0.03), abs=3e-4)
        assert got.left_rad[south] == pytest.approx(np.arctan(0.03), abs=3e-4)
        # going east or west the slope is along the road, not across it
        assert abs(got.left_rad[0]) < 3e-4

    def test_no_data_is_no_bank(self):
        track = _circle()
        assert bankmod.road_bank(track, _Ground(lambda x, y: np.full(len(x), np.nan))) is None
        assert bankmod.attach(track, None) is None
        assert len(track.bank) == 0

    def test_a_wall_beside_the_road_is_flagged_and_filled(self):
        # a 3 m step two metres outside the line on one quarter of the lap
        def height(x, y):
            r = np.hypot(x, y)
            return 55.0 + 0.02 * r + np.where((r > 102.0) & (x > 0) & (y > 0), 3.0, 0.0)

        track = _circle()
        got = bankmod.road_bank(track, _Ground(height))
        assert got is not None
        quarter = (track.x_m > 5.0) & (track.y_m > 5.0)
        assert (got.flag[quarter] == 3).all() and (got.flag[~quarter] == 0).mean() > 0.95
        assert got.info["not_plane"] == int((got.flag == 3).sum()) > 0
        # filled from the nodes either side, which read the bowl
        assert got.left_rad == pytest.approx(np.arctan(0.02), abs=5e-4)

    def test_a_bridge_is_flagged(self):
        # the bare earth is 6 m under the road for 80 m: the road crosses a cutting
        def height(x, y):
            r = np.hypot(x, y)
            return 55.0 + 0.02 * (r - 100.0) - np.where(np.hypot(x - 100.0, y) < 40.0, 6.0, 0.0)

        track = _circle()
        got = bankmod.road_bank(track, _Ground(height))
        assert got is not None
        at = int(np.argmax(track.x_m))
        assert got.flag[at] == 2 and got.info["bridge"] >= 40
        assert got.left_rad[at] == pytest.approx(np.arctan(0.02), abs=1e-3)

    def test_notch_zones_names_the_stretch(self):
        n = 400
        ground = np.full(n, 50.0)
        ground[100:130] -= 5.0
        zones, departure = bankmod.notch_zones(ground, np.full(n, 50.0), DS)
        assert len(zones) == 1
        first, last = zones[0]
        assert first < 100 and last > 129
        assert departure[115] == pytest.approx(-5.0) and departure[300] == pytest.approx(0.0)
        assert list(bankmod.zone_nodes((398, 1), n)) == [398, 399, 0, 1]


class TestTyreLateral:
    def test_flat_banked_and_mirrored(self):
        kappa, speed = np.array([0.02, 0.0, -0.02]), np.array([20.0, 40.0, 20.0])
        g = 9.80665
        flat = bankmod.tyre_lateral_g(kappa, speed)
        assert flat == pytest.approx([400 * 0.02 / g, 0.0, -400 * 0.02 / g])
        # into each turn by 0.05: the left side higher in the right turn, lower in the left
        left = np.array([0.05, 0.03, -0.05])
        got = bankmod.tyre_lateral_g(kappa, speed, left)
        assert got[0] == pytest.approx(flat[0] * np.cos(0.05) - np.sin(0.05))
        assert got[1] == pytest.approx(-np.sin(0.03))  # held against the cross-slope
        assert abs(got[0]) < abs(flat[0]) and abs(got[2]) < abs(flat[2])
        assert bankmod.tyre_lateral_g(-kappa, speed, -left) == pytest.approx(-got)
        assert np.array_equal(bankmod.tyre_lateral_g(kappa, speed, np.zeros(3)), flat)


class TestOnTheTrack:
    def test_attach_and_round_trip(self, tmp_path):
        track = _circle()
        got = bankmod.attach(track, _Ground(lambda x, y: 55.0 + 0.05 * np.hypot(x, y)))
        assert got is not None
        assert track.bank == pytest.approx(got.left_rad) and len(track.bank_flag) == track.n
        assert track.meta["bank"]["nodes"] == track.n
        # a circle has no straight: the scatter is not a number and is stored as none
        assert track.meta["bank"]["straight_sd_rad"] is None
        back = trk.TrackModel.load(track.save(tmp_path / "circle.json"))
        assert back.bank == pytest.approx(track.bank, abs=1e-5)
        assert (back.bank_flag == track.bank_flag).all()
        assert back.meta["bank"]["into_corners_mean_rad"] == pytest.approx(
            np.arctan(0.05), abs=1e-3
        )

    def test_a_track_without_a_bank_round_trips_without_one(self, tmp_path):
        track = _circle()
        data = track.to_json()
        assert "bank_rad" not in data
        back = trk.TrackModel.load(track.save(tmp_path / "circle.json"))
        assert len(back.bank) == 0 and len(back.bank_flag) == 0
