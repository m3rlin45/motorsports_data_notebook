"""Tests for GPS-based layout detection + lap re-splitting (tire_etl.layouts).

Synthetic trajectories are built around the real Suzuka gate coordinates so
the registry itself is exercised: a loop that passes through both the main
and the West-course start/finish is a full-circuit lap; a loop that only
passes the West line is a West-course lap.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest

from motorsports_data_notebook.tire_etl.layouts import (
    LAP_SOURCE_LOGGER,
    LAP_SOURCE_RESPLIT,
    LAP_SOURCE_UNVERIFIED,
    TRACK_RES_DECLARED,
    TRACK_RES_GPS,
    TRACK_RES_GPS_OVERRIDE,
    VENUES,
    GpsTrace,
    apply_layout_correction,
    detect_layout,
    find_venue,
    gate_crossings,
    gps_trace_from_log,
    normalize_layout_name,
    resplit_laps,
)

SUZUKA = VENUES["suzuka"]
MAIN = SUZUKA.gates["sf"]
WEST = SUZUKA.gates["west_sf"]

_EARTH_R = 6_371_000.0
_HZ = 25
_SPEED_MS = 50.0


def _offset(lat: float, lon: float, dx_m: float, dy_m: float) -> tuple[float, float]:
    k = math.pi / 180.0
    return lat + dy_m / (_EARTH_R * k), lon + dx_m / (_EARTH_R * k * math.cos(lat * k))


def _loop_through(
    points: list[tuple[float, float]], *, bulge_m: float
) -> list[tuple[float, float]]:
    """Closed loop: straight line through ``points`` out, arc offset north back."""
    a, b = points[0], points[-1]
    n_leg = 300
    out = [(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) for t in np.linspace(0, 1, n_leg)]
    back = []
    for t in np.linspace(0, 1, n_leg):
        lat = b[0] + (a[0] - b[0]) * t
        lon = b[1] + (a[1] - b[1]) * t
        back.append(_offset(lat, lon, 0.0, bulge_m * math.sin(math.pi * t)))
    return out + back[1:-1]


def _resample_const_speed(loop: list[tuple[float, float]], n_laps: int) -> tuple[np.ndarray, ...]:
    """Walk the loop ``n_laps`` times at constant speed, 25 Hz samples."""
    pts = np.array(loop * n_laps + [loop[0]])
    k = math.pi / 180.0
    lat0 = pts[0, 0]
    xy = np.column_stack(
        [
            (pts[:, 1] - pts[0, 1]) * k * _EARTH_R * math.cos(lat0 * k),
            (pts[:, 0] - lat0) * k * _EARTH_R,
        ]
    )
    seg = np.hypot(*np.diff(xy, axis=0).T)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = dist[-1]
    t = np.arange(0.0, total / _SPEED_MS, 1.0 / _HZ)
    d = t * _SPEED_MS
    lat = np.interp(d, dist, pts[:, 0])
    lon = np.interp(d, dist, pts[:, 1])
    return (t * 1000.0).astype(np.int64), lat, lon


class _FakeLog:
    def __init__(self, t_ms, lat, lon, laps: pa.Table | None = None, metadata=None) -> None:
        self.channels = {
            "GPS Latitude": pa.table({"timecodes": t_ms, "GPS Latitude": lat}),
            "GPS Longitude": pa.table({"timecodes": t_ms, "GPS Longitude": lon}),
        }
        self.laps = laps if laps is not None else _laps_table([])
        self.metadata = metadata or {}
        self.file_name = "fake.xrk"


def _laps_table(bounds_ms: list[tuple[int, int]]) -> pa.Table:
    return pa.table(
        {
            "num": pa.array(list(range(1, len(bounds_ms) + 1)), type=pa.int64()),
            "start_time": pa.array([s for s, _ in bounds_ms], type=pa.int64()),
            "end_time": pa.array([e for _, e in bounds_ms], type=pa.int64()),
            "lap_type": pa.array(["full"] * len(bounds_ms)),
        }
    )


def _full_course(n_laps: int = 5):
    loop = _loop_through([(MAIN.lat, MAIN.lon), (WEST.lat, WEST.lon)], bulge_m=400.0)
    return _resample_const_speed(loop, n_laps)


def _west_course(n_laps: int = 5):
    # A short loop through the West S/F that never comes within 15 m of the main line.
    start = _offset(WEST.lat, WEST.lon, -400.0, 0.0)
    end = _offset(WEST.lat, WEST.lon, 400.0, 0.0)
    loop = _loop_through([start, (WEST.lat, WEST.lon), end], bulge_m=250.0)
    return _resample_const_speed(loop, n_laps)


def _logger_laps_at(trace_t: np.ndarray, lat, lon, gate) -> pa.Table:
    """Logger-style laps whose boundaries are the crossings of ``gate``."""
    times = gate_crossings(GpsTrace(trace_t, lat, lon), gate)
    bounds = [(int(round(a)), int(round(b))) for a, b in zip(times[:-1], times[1:])]
    return _laps_table(bounds)


# --------------------------------------------------------------------------


def test_layout_aliases() -> None:
    assert normalize_layout_name("Suzuka West") == "suzuka_west"
    assert normalize_layout_name("Suzuka Car") == "suzuka_full"
    assert normalize_layout_name("Tsukuba_Bike") == "tsukuba_2000"
    assert normalize_layout_name("Tsukuba_Car") == "tsukuba_2000"
    assert normalize_layout_name("Motegi East") == "motegi_east"
    assert normalize_layout_name("Fuji GP Sh") == "fuji_short"
    assert normalize_layout_name("Race") is None
    assert normalize_layout_name(None) is None


def test_registry_is_consistent() -> None:
    for venue in VENUES.values():
        ids = {lay.layout_id for lay in venue.layouts}
        assert venue.default_layout in ids
        for lay in venue.layouts:
            assert lay.track_canonical == venue.venue_id
            for g in lay.gates:
                assert g in venue.gates


def test_find_venue_and_gate_crossings_full_course() -> None:
    t, lat, lon = _full_course(5)
    trace = GpsTrace(t, lat, lon)
    assert find_venue(trace) is SUZUKA
    # 5 loops that start and end *on* the main gate: the endpoints themselves
    # are not crossings, so expect n_laps - 1 .. n_laps.
    assert gate_crossings(trace, MAIN).size in (4, 5)
    assert gate_crossings(trace, WEST).size in (4, 5)


def test_full_course_logged_as_west_is_overridden_and_resplit() -> None:
    t, lat, lon = _full_course(6)
    laps = _logger_laps_at(t, lat, lon, WEST)  # beacon on the West line
    log = _FakeLog(t, lat, lon, laps)
    det = detect_layout(log, declared_track_raw="Suzuka West")
    assert det.venue_id == "suzuka"
    assert det.layout_id == "suzuka_full"
    assert det.track_canonical == "suzuka"
    assert det.declared_layout_id == "suzuka_west"
    assert det.track_resolution == TRACK_RES_GPS_OVERRIDE
    assert det.beacon_gate == "west_sf"
    assert det.lap_source == LAP_SOURCE_RESPLIT
    assert det.resplit_laps is not None
    n = len(det.resplit_laps)
    assert n >= 4
    dur = (
        det.resplit_laps.column("end_time").to_numpy()
        - det.resplit_laps.column("start_time").to_numpy()
    ) / 1000.0
    assert np.allclose(dur, dur[0], rtol=0.02)
    # Boundaries now sit on the main line, not the West one.
    starts = det.resplit_laps.column("start_time").to_numpy()
    main_times = gate_crossings(GpsTrace(t, lat, lon), MAIN)
    for s in starts:
        assert np.min(np.abs(main_times - s)) < 100

    apply_layout_correction(log, det)
    assert len(log.laps) == n
    assert "lap_time" in log.laps.column_names


def test_full_course_logged_correctly_keeps_logger_laps() -> None:
    t, lat, lon = _full_course(4)
    laps = _logger_laps_at(t, lat, lon, MAIN)
    log = _FakeLog(t, lat, lon, laps)
    det = detect_layout(log, declared_track_raw="Suzuka Car")
    assert det.layout_id == "suzuka_full"
    assert det.track_resolution == TRACK_RES_DECLARED
    assert det.beacon_gate == "sf"
    assert det.lap_source == LAP_SOURCE_LOGGER
    assert det.resplit_laps is None
    before = log.laps
    apply_layout_correction(log, det)
    assert log.laps is before


def test_genuine_west_course_is_kept_as_west() -> None:
    t, lat, lon = _west_course(5)
    laps = _logger_laps_at(t, lat, lon, WEST)
    log = _FakeLog(t, lat, lon, laps)
    det = detect_layout(log, declared_track_raw="Suzuka West")
    assert det.gate_crossings["sf"] == 0
    assert det.layout_id == "suzuka_west"
    assert det.track_canonical == "suzuka"  # pooled venue bucket
    assert det.track_resolution == TRACK_RES_DECLARED
    assert det.lap_source == LAP_SOURCE_LOGGER


def test_unknown_filename_token_resolves_from_venue_metadata_then_gps() -> None:
    t, lat, lon = _full_course(3)
    laps = _logger_laps_at(t, lat, lon, MAIN)
    log = _FakeLog(t, lat, lon, laps, metadata={"Venue": "Suzuka Car"})
    det = detect_layout(log, declared_track_raw="Race", venue_meta="Suzuka Car")
    assert det.declared_track_canonical == "suzuka"
    assert det.track_resolution == TRACK_RES_DECLARED
    assert det.track_canonical == "suzuka"

    # No usable declaration anywhere: GPS supplies the track.
    det2 = detect_layout(log, declared_track_raw="Generic testing", venue_meta="")
    assert det2.declared_track_canonical is None
    assert det2.track_resolution == TRACK_RES_GPS
    assert det2.track_canonical == "suzuka"


def test_no_gps_falls_back_to_declaration_unverified() -> None:
    log = _FakeLog(np.array([0, 40], dtype=np.int64), np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    log.channels = {}
    det = detect_layout(log, declared_track_raw="Tsukuba")
    assert det.venue_id is None
    assert det.track_canonical == "tsukuba_2000"
    assert det.layout_id == "tsukuba_2000"
    assert det.lap_source == LAP_SOURCE_UNVERIFIED
    assert det.track_resolution == TRACK_RES_DECLARED


def test_unregistered_location_keeps_declaration() -> None:
    # Central Tokyo road drive: no venue within range.
    t = np.arange(0, 400 * 40, 40, dtype=np.int64)
    lat = np.linspace(35.690, 35.700, len(t))
    lon = np.linspace(139.760, 139.770, len(t))
    det = detect_layout(_FakeLog(t, lat, lon), declared_track_raw="Generic testing")
    assert det.venue_id is None
    assert det.track_canonical is None
    assert det.lap_source == LAP_SOURCE_UNVERIFIED


def test_gps_glitches_are_dropped_from_trace() -> None:
    t, lat, lon = _full_course(3)
    lat_g = lat.copy()
    lon_g = lon.copy()
    glitch = slice(100, 110)
    lat_g[glitch] = 37.38
    lon_g[glitch] = 168.49
    lat_g[200] = 0.0
    lon_g[200] = 0.0
    trace = gps_trace_from_log(_FakeLog(t, lat_g, lon_g))
    assert trace is not None
    assert len(trace) == len(t) - 11
    assert np.all(np.abs(trace.lon - WEST.lon) < 0.05)


def test_resplit_drops_pit_stop_and_double_crossings() -> None:
    t, lat, lon = _full_course(6)
    # Insert a 20-minute stop right after the second main-line crossing by
    # shifting all later timestamps.
    main_times = gate_crossings(GpsTrace(t, lat, lon), MAIN)
    cut = int(np.searchsorted(t, main_times[1] + 500))
    t2 = t.copy()
    t2[cut:] += 20 * 60 * 1000
    laps = resplit_laps(GpsTrace(t2, lat, lon), MAIN)
    dur = (laps.column("end_time").to_numpy() - laps.column("start_time").to_numpy()) / 1000.0
    assert len(laps) >= 3
    assert dur.max() < 2.0 * np.median(dur)  # the 20-minute "lap" is gone
    assert laps.column("num").to_pylist() == list(range(1, len(laps) + 1))


def test_resplit_uses_logger_lap_time_as_reference_when_few_laps() -> None:
    """Two intervals (one real lap, one 20-minute pit stop) have a useless
    median; the logger's own lap duration disambiguates."""
    t, lat, lon = _full_course(3)
    main_times = gate_crossings(GpsTrace(t, lat, lon), MAIN)
    lap_s = float(main_times[1] - main_times[0]) / 1000.0
    cut = int(np.searchsorted(t, main_times[1] + 500))
    t2 = t.copy()
    t2[cut:] += 20 * 60 * 1000
    laps = resplit_laps(GpsTrace(t2, lat, lon), MAIN, ref_lap_s=lap_s)
    dur = (laps.column("end_time").to_numpy() - laps.column("start_time").to_numpy()) / 1000.0
    assert len(laps) == 1
    assert dur[0] == pytest.approx(lap_s, rel=0.02)


def test_resplit_with_too_few_crossings_is_empty() -> None:
    t, lat, lon = _west_course(3)
    assert len(resplit_laps(GpsTrace(t, lat, lon), MAIN)) == 0


@pytest.mark.parametrize("venue_id", sorted(VENUES))
def test_venue_reference_is_within_track_registry_radius(venue_id: str) -> None:
    """Every venue reference must sit near the tracks.py coordinates used for weather."""
    from motorsports_data_notebook.tire_etl.tracks import get_track

    ti = get_track(venue_id)
    assert ti is not None
    ref = VENUES[venue_id].reference()
    assert abs(ti.lat - ref.lat) < 0.1
    assert abs(ti.lon - ref.lon) < 0.1


def test_wrong_beacon_fragment_without_full_lap_loses_its_laps() -> None:
    """A one-lap fragment split on the West line while the trace shows the
    full course (but no completed main-line lap) must not keep that lap."""
    t, lat, lon = _full_course(2)
    # Keep only the first ~1.3 loops: one West crossing, zero complete main laps
    # after the start, and a logger "lap" bounded by the start and the West gate.
    west_times = gate_crossings(GpsTrace(t, lat, lon), WEST)
    cut = int(np.searchsorted(t, west_times[0] + 15_000))
    t, lat, lon = t[:cut], lat[:cut], lon[:cut]
    laps = _laps_table([(int(t[0]), int(round(west_times[0])))])
    log = _FakeLog(t, lat, lon, laps)
    det = detect_layout(log, declared_track_raw="Suzuka West")
    assert det.beacon_gate != det.layout_id and det.lap_source == LAP_SOURCE_RESPLIT
    assert det.resplit_laps is not None and len(det.resplit_laps) == 0
    apply_layout_correction(log, det)
    assert len(log.laps) == 0


def test_split_on_layout_separates_differing_fragments(tmp_path) -> None:
    from pathlib import Path

    from motorsports_data_notebook.tire_etl.extract import _split_on_layout
    from motorsports_data_notebook.tire_etl.layouts import LayoutDetection

    def det(layout):
        return LayoutDetection(
            venue_id="suzuka",
            layout_id=layout,
            track_canonical="suzuka",
            declared_track_canonical="suzuka",
            declared_layout_id=None,
            venue_meta=None,
            track_resolution="declared",
            lap_source="logger",
            beacon_gate=None,
            beacon_dist_m=None,
        )

    a, b, c = (Path(tmp_path / f"{n}.xrk") for n in ("a", "b", "c"))
    dets = {str(a): det("suzuka_west"), str(b): det("suzuka_full"), str(c): det("suzuka_full")}
    split = _split_on_layout([([a, b, c], ["la", "lb", "lc"], [0, 1000, 2500])], dets)
    assert [sp for sp, _, _ in split] == [[a], [b, c]]
    assert [so for _, _, so in split] == [[0], [0, 1500]]
    # Homogeneous groups pass through untouched.
    same = {str(a): det("suzuka_full"), str(b): det("suzuka_full")}
    assert _split_on_layout([([a, b], ["la", "lb"], [0, 10])], same) == [
        ([a, b], ["la", "lb"], [0, 10])
    ]
