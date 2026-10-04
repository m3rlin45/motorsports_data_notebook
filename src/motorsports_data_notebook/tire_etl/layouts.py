"""GPS-based circuit layout detection and lap re-splitting.

Why this exists
---------------
The AIM logger splits laps with a virtual beacon taken from whichever track
*variant* the driver selected on the dash. When the selection is wrong (the
car runs the full Suzuka circuit with "Suzuka West" selected; "Tsukuba_Bike"
selected in a car session) three things go wrong downstream:

1. The filename / ``Venue`` metadata carries the wrong variant, so the
   filename-based track resolution either maps to the wrong canonical track
   or to none at all (and the session silently drops out of the model).
2. Lap boundaries sit at the *variant's* start/finish line, which can be
   half a lap away from the real one. Per-lap aggregates are then phase
   shifted and the out-lap / in-lap bookkeeping is off.
3. Nothing records that this happened, so it cannot be audited.

This module resolves all three from the GPS trace itself:

- Every known venue has a set of **gates** — points on the start/finish line
  of each layout the venue offers. A **layout** is the set of gates a lap of
  that layout crosses (the first gate is its start/finish, used to split
  laps). E.g. Suzuka full course crosses both the main and the West-course
  S/F; the West course crosses only the West one.
- :func:`detect_layout` finds the venue from the GPS centroid, counts
  crossings of every gate, picks the most specific layout whose gates were
  all crossed, and checks where the logger's lap beacon actually sits.
- When the beacon is not on the resolved layout's S/F, :func:`resplit_laps`
  rebuilds the laps table from GPS crossings of the correct gate.

Gate coordinates were measured from the committed dataset (median GPS
position at the logger's lap starts, per declared variant). Add a venue by
measuring the S/F point of each layout the same way.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

from .tracks import normalize_track_name

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """A point on a start/finish line. Crossing direction is inferred from data."""

    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class Layout:
    """One drivable layout of a venue.

    ``gates`` lists the gates a lap of this layout crosses exactly once; the
    first entry is the layout's own start/finish and is the gate laps are
    split on. ``track_canonical`` is the model bucket the layout feeds —
    variants of one venue are pooled under the venue's canonical track id
    (matching the long-standing Fuji GP/Short and Motegi/East pooling).
    """

    layout_id: str
    track_canonical: str
    gates: tuple[str, ...]
    display: str

    @property
    def sf_gate(self) -> str:
        return self.gates[0]


@dataclass(frozen=True)
class Venue:
    venue_id: str  # == the pooled track_canonical
    gates: dict[str, Gate]
    layouts: tuple[Layout, ...]
    default_layout: str

    def layout(self, layout_id: str) -> Layout:
        for lay in self.layouts:
            if lay.layout_id == layout_id:
                return lay
        raise KeyError(layout_id)

    def reference(self) -> Gate:
        return self.gates[self.layout(self.default_layout).sf_gate]


VENUES: dict[str, Venue] = {
    "tsukuba_2000": Venue(
        venue_id="tsukuba_2000",
        gates={"sf": Gate("sf", 36.15010, 139.91944)},
        layouts=(
            # The "Tsukuba_Car" and "Tsukuba_Bike" dash variants share the
            # same tarmac and (within ~2 m) the same S/F line, so they are
            # one layout here; the declared variant is still recorded.
            Layout("tsukuba_2000", "tsukuba_2000", ("sf",), "Tsukuba Circuit 2000"),
        ),
        default_layout="tsukuba_2000",
    ),
    "sodegaura": Venue(
        venue_id="sodegaura",
        gates={"sf": Gate("sf", 35.39513, 140.08808)},
        layouts=(Layout("sodegaura", "sodegaura", ("sf",), "Sodegaura Forest Raceway"),),
        default_layout="sodegaura",
    ),
    "fuji": Venue(
        venue_id="fuji",
        gates={"sf": Gate("sf", 35.37218, 138.92709)},
        layouts=(
            # GP and Short share the main-straight S/F; geometry alone cannot
            # separate them, so the declared variant breaks the tie.
            Layout("fuji_gp", "fuji", ("sf",), "Fuji Speedway GP"),
            Layout("fuji_short", "fuji", ("sf",), "Fuji Speedway Short"),
        ),
        default_layout="fuji_gp",
    ),
    "motegi": Venue(
        venue_id="motegi",
        gates={
            "sf": Gate("sf", 36.53299, 140.22668),
            "east_sf": Gate("east_sf", 36.53181, 140.23661),
        },
        layouts=(
            Layout("motegi_full", "motegi", ("sf", "east_sf"), "Mobility Resort Motegi"),
            Layout("motegi_east", "motegi", ("east_sf",), "Motegi East Course"),
        ),
        default_layout="motegi_full",
    ),
    "suzuka": Venue(
        venue_id="suzuka",
        gates={
            "sf": Gate("sf", 34.84501, 136.53866),
            "west_sf": Gate("west_sf", 34.84538, 136.52586),
        },
        layouts=(
            Layout("suzuka_full", "suzuka", ("sf", "west_sf"), "Suzuka Circuit (full)"),
            Layout("suzuka_west", "suzuka", ("west_sf",), "Suzuka West Course"),
            Layout("suzuka_east", "suzuka", ("sf",), "Suzuka East Course"),
        ),
        default_layout="suzuka_full",
    ),
    "minami": Venue(
        venue_id="minami",
        gates={"sf": Gate("sf", 35.48826, 140.25061)},
        layouts=(Layout("minami", "minami", ("sf",), "Minami Chiba Circuit"),),
        default_layout="minami",
    ),
}

# Dash / filename variant strings -> layout id. Keys are lowercased alnum.
_LAYOUT_ALIASES: dict[str, str] = {
    "tsukuba": "tsukuba_2000",
    "tsukuba2000": "tsukuba_2000",
    "tc2000": "tsukuba_2000",
    "tsukubacar": "tsukuba_2000",
    "tsukubabike": "tsukuba_2000",
    "sodegaura": "sodegaura",
    "fuji": "fuji_gp",
    "fujigp": "fuji_gp",
    "fujispeedway": "fuji_gp",
    "fujishort": "fuji_short",
    "fujigpsh": "fuji_short",
    "motegi": "motegi_full",
    "motegieast": "motegi_east",
    "suzuka": "suzuka_full",
    "suzukacar": "suzuka_full",
    "suzukawest": "suzuka_west",
    "suzukaeast": "suzuka_east",
    "minami": "minami",
}


def normalize_layout_name(raw: str | None) -> str | None:
    """Map a dash/filename variant string (``"Suzuka West"``) to a layout id."""
    if not raw:
        return None
    key = "".join(c.lower() for c in raw if c.isalnum())
    return _LAYOUT_ALIASES.get(key)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

_EARTH_R_M = 6_371_000.0
VENUE_RADIUS_M = 15_000.0  # GPS centroid must be within this of a venue reference
GATE_HALF_WIDTH_M = 15.0  # lateral tolerance for a trajectory segment to count as crossing
GATE_HEADING_RADIUS_M = 12.0  # samples this close to the gate define its crossing heading
BEACON_TOL_M = 40.0  # logger lap starts within this of a gate => beacon is on that gate
MIN_CROSS_SPEED_MS = 5.0  # ignore crossings slower than this (parked / pushed cars)
MIN_GPS_SAMPLES = 200


def _local_xy(lat: np.ndarray, lon: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
    """Equirectangular projection to metres around ``(ref_lat, ref_lon)``; shape (n, 2)."""
    k = math.pi / 180.0
    x = (lon - ref_lon) * k * _EARTH_R_M * math.cos(ref_lat * k)
    y = (lat - ref_lat) * k * _EARTH_R_M
    return np.column_stack([x, y])


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    k = math.pi / 180.0
    dlat = (lat2 - lat1) * k
    dlon = (lon2 - lon1) * k
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * k) * math.cos(lat2 * k) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_R_M * math.asin(math.sqrt(a))


@dataclass
class GpsTrace:
    t_ms: np.ndarray  # int64, strictly increasing
    lat: np.ndarray
    lon: np.ndarray

    def __len__(self) -> int:
        return int(len(self.t_ms))


def gps_trace_from_log(
    log,
    *,
    lat_channel: str = "GPS Latitude",
    lon_channel: str = "GPS Longitude",
) -> GpsTrace | None:
    """Pull a cleaned (lat, lon, t) trace out of a LogFile-like object.

    Drops zero / non-finite fixes and anything farther than ``VENUE_RADIUS_M``
    from the trace's median position (GPS glitches have been seen jumping
    thousands of km mid-session). Returns None when there is no usable GPS.
    """
    chans = log.channels
    if lat_channel not in chans or lon_channel not in chans:
        return None
    lat_tbl = chans[lat_channel]
    lon_tbl = chans[lon_channel]
    t = lat_tbl.column("timecodes").to_numpy().astype(np.int64)
    lat = lat_tbl.column(lat_channel).to_numpy(zero_copy_only=False).astype(np.float64)
    lon_t = lon_tbl.column("timecodes").to_numpy().astype(np.int64)
    lon = lon_tbl.column(lon_channel).to_numpy(zero_copy_only=False).astype(np.float64)
    if len(t) != len(lon_t) or not np.array_equal(t, lon_t):
        # Align longitude onto the latitude timebase (nearest-previous sample).
        idx = np.searchsorted(lon_t, t, side="right") - 1
        idx = np.clip(idx, 0, len(lon) - 1)
        lon = lon[idx]
    ok = np.isfinite(lat) & np.isfinite(lon) & (lat != 0.0) & (lon != 0.0)
    ok &= (np.abs(lat) <= 90.0) & (np.abs(lon) <= 180.0)
    if ok.sum() < MIN_GPS_SAMPLES:
        return None
    med_lat = float(np.median(lat[ok]))
    med_lon = float(np.median(lon[ok]))
    xy = _local_xy(lat, lon, med_lat, med_lon)
    ok &= np.hypot(xy[:, 0], xy[:, 1]) <= VENUE_RADIUS_M
    if ok.sum() < MIN_GPS_SAMPLES:
        return None
    t, lat, lon = t[ok], lat[ok], lon[ok]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return GpsTrace(t_ms=t[keep], lat=lat[keep], lon=lon[keep])


def find_venue(trace: GpsTrace) -> Venue | None:
    """Nearest registered venue to the trace's median position, within VENUE_RADIUS_M."""
    med_lat = float(np.median(trace.lat))
    med_lon = float(np.median(trace.lon))
    best: tuple[float, Venue] | None = None
    for venue in VENUES.values():
        ref = venue.reference()
        d = _haversine_m(med_lat, med_lon, ref.lat, ref.lon)
        if d <= VENUE_RADIUS_M and (best is None or d < best[0]):
            best = (d, venue)
    return best[1] if best else None


def gate_crossings(
    trace: GpsTrace,
    gate: Gate,
    *,
    half_width_m: float = GATE_HALF_WIDTH_M,
    min_speed_ms: float = MIN_CROSS_SPEED_MS,
) -> np.ndarray:
    """Return crossing times (ms, float) of ``gate`` by ``trace``.

    The gate's line is perpendicular to the dominant travel direction of the
    samples that pass within ``GATE_HEADING_RADIUS_M`` of the gate point —
    so a gate is a point plus a data-derived heading, and a crossing is a
    trajectory segment that goes from behind the line to in front of it
    within ``half_width_m`` laterally, travelling at least ``min_speed_ms``.
    """
    xy = _local_xy(trace.lat, trace.lon, gate.lat, gate.lon)
    if len(xy) < 2:
        return np.empty(0, dtype=np.float64)
    seg = np.diff(xy, axis=0)
    dt_s = np.diff(trace.t_ms).astype(np.float64) / 1000.0
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt_s > 0, seg_len / dt_s, 0.0)

    near = np.hypot(xy[:-1, 0], xy[:-1, 1]) <= GATE_HEADING_RADIUS_M
    near &= seg_len > 0.1
    if near.sum() < 3:
        return np.empty(0, dtype=np.float64)
    # Dominant heading: sum of unit direction vectors of nearby fast segments.
    unit = seg[near] / seg_len[near][:, None]
    heading = unit.sum(axis=0)
    norm = float(np.hypot(*heading))
    if norm < 1e-6:
        return np.empty(0, dtype=np.float64)
    n = heading / norm  # along-track unit vector
    lateral = np.array([-n[1], n[0]])

    s = xy @ n  # signed distance ahead of the gate line
    cross = (s[:-1] < 0.0) & (s[1:] >= 0.0)
    cross &= speed >= min_speed_ms
    idx = np.nonzero(cross)[0]
    if idx.size == 0:
        return np.empty(0, dtype=np.float64)
    frac = -s[idx] / (s[idx + 1] - s[idx])
    pts = xy[idx] + seg[idx] * frac[:, None]
    lat_off = np.abs(pts @ lateral)
    good = lat_off <= half_width_m
    idx, frac = idx[good], frac[good]
    t0 = trace.t_ms[idx].astype(np.float64)
    t1 = trace.t_ms[idx + 1].astype(np.float64)
    return np.asarray(t0 + frac * (t1 - t0), dtype=np.float64)


def nearest_gate_to_points(
    venue: Venue, lat: np.ndarray, lon: np.ndarray, *, tol_m: float = BEACON_TOL_M
) -> tuple[str | None, float]:
    """Gate whose median distance to the given points is smallest (and within tol)."""
    best_name: str | None = None
    best_d = float("inf")
    for name, gate in venue.gates.items():
        xy = _local_xy(lat, lon, gate.lat, gate.lon)
        d = float(np.median(np.hypot(xy[:, 0], xy[:, 1])))
        if d < best_d:
            best_name, best_d = name, d
    if best_d > tol_m:
        return None, best_d
    return best_name, best_d


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

LAP_SOURCE_LOGGER = "logger"  # logger beacon sits on the resolved layout's S/F
LAP_SOURCE_RESPLIT = "gps_resplit"  # laps rebuilt from GPS crossings of the correct S/F
LAP_SOURCE_UNVERIFIED = "logger_unverified"  # no usable GPS / unknown venue; logger laps kept

TRACK_RES_DECLARED = "declared"  # GPS agrees with filename/venue (or no GPS to check)
TRACK_RES_GPS = "gps"  # filename/venue gave nothing; GPS supplied the track
TRACK_RES_GPS_OVERRIDE = "gps_override"  # GPS layout contradicts the declared variant


@dataclass
class LayoutDetection:
    venue_id: str | None
    layout_id: str | None
    track_canonical: str | None
    declared_track_canonical: str | None
    declared_layout_id: str | None
    venue_meta: str | None
    track_resolution: str
    lap_source: str
    beacon_gate: str | None
    beacon_dist_m: float | None
    gate_crossings: dict[str, int] = field(default_factory=dict)
    n_laps_logger: int = 0
    resplit_laps: pa.Table | None = None
    note: str = ""


def _full_laps(laps: pa.Table) -> pa.Table:
    """The logger's full laps: out/in-typed segments start at the recording
    start / end at its end, not at a beacon crossing, so they must not take
    part in beacon or lap-time reasoning."""
    if "lap_type" in laps.column_names and len(laps) > 0:
        import pyarrow.compute as pc

        return laps.filter(pc.equal(laps.column("lap_type"), "full"))
    return laps


def _logger_lap_start_positions(log, trace: GpsTrace) -> tuple[np.ndarray, np.ndarray]:
    laps = _full_laps(log.laps)
    if len(laps) == 0:
        return np.empty(0), np.empty(0)
    starts = laps.column("start_time").to_numpy().astype(np.int64)
    idx = np.clip(np.searchsorted(trace.t_ms, starts), 0, len(trace) - 1)
    # Only trust starts that have a GPS sample within 2 s.
    close = np.abs(trace.t_ms[idx] - starts) <= 2000
    return trace.lat[idx[close]], trace.lon[idx[close]]


def _pick_layout(
    venue: Venue, crossings: dict[str, int], declared_layout_id: str | None
) -> tuple[Layout | None, str]:
    """Most specific layout whose gates were all crossed; declared breaks ties."""
    matching = [lay for lay in venue.layouts if all(crossings.get(g, 0) >= 1 for g in lay.gates)]
    if not matching:
        return None, "no layout matched the gate crossings"
    best_n = max(len(lay.gates) for lay in matching)
    best = [lay for lay in matching if len(lay.gates) == best_n]
    if len(best) == 1:
        return best[0], "unique gate-set match"
    for lay in best:
        if lay.layout_id == declared_layout_id:
            return lay, "gate-set tie broken by declared variant"
    for lay in best:
        if lay.layout_id == venue.default_layout:
            return lay, "gate-set tie broken by venue default"
    return best[0], "gate-set tie broken by registry order"


def resplit_laps(
    trace: GpsTrace,
    gate: Gate,
    *,
    ref_lap_s: float | None = None,
    min_frac: float = 0.5,
    max_frac: float = 2.0,
    min_edge_s: float = 10.0,
) -> pa.Table:
    """Build a laps table from consecutive GPS crossings of ``gate``.

    Intervals shorter than ``min_frac`` × reference or longer than
    ``max_frac`` × reference are dropped (spurious double crossings, pit
    stops), which leaves a gap in the timeline exactly where the logger would
    have typed an in/out lap — so downstream stint detection behaves the same
    way. The reference lap time is ``ref_lap_s`` when given (the logger's own
    median lap duration is the natural choice: a beacon on the wrong gate
    still measures full laps), else the median interval. Laps are renumbered
    1..N and typed ``"full"``; the segment from the first GPS sample to the
    first crossing is lap 0 typed ``"out"`` and the segment after the last
    crossing is typed ``"in"``, mirroring what the logger emits, when each is
    at least ``min_edge_s`` long.
    """
    times = gate_crossings(trace, gate)
    if times.size < 2:
        return pa.table(
            {
                "num": pa.array([], type=pa.int64()),
                "start_time": pa.array([], type=pa.int64()),
                "end_time": pa.array([], type=pa.int64()),
                "lap_type": pa.array([], type=pa.string()),
            }
        )
    starts = np.round(times[:-1]).astype(np.int64)
    ends = np.round(times[1:]).astype(np.int64)
    dur = (ends - starts).astype(np.float64)
    ref = ref_lap_s * 1000.0 if ref_lap_s and ref_lap_s > 0 else float(np.median(dur))
    keep = (dur >= min_frac * ref) & (dur <= max_frac * ref) & (dur > 0)
    starts, ends = starts[keep], ends[keep]
    nums = list(np.arange(1, len(starts) + 1, dtype=np.int64))
    s_list = list(starts)
    e_list = list(ends)
    types = ["full"] * len(starts)
    t0 = int(trace.t_ms[0])
    t1 = int(trace.t_ms[-1])
    first_cross = int(np.round(times[0]))
    last_cross = int(np.round(times[-1]))
    if first_cross - t0 >= min_edge_s * 1000.0:
        nums.insert(0, np.int64(0))
        s_list.insert(0, np.int64(t0))
        e_list.insert(0, np.int64(first_cross))
        types.insert(0, "out")
    if t1 - last_cross >= min_edge_s * 1000.0:
        nums.append(np.int64(len(starts) + 1))
        s_list.append(np.int64(last_cross))
        e_list.append(np.int64(t1))
        types.append("in")
    return pa.table(
        {
            "num": pa.array(np.asarray(nums, dtype=np.int64)),
            "start_time": pa.array(np.asarray(s_list, dtype=np.int64)),
            "end_time": pa.array(np.asarray(e_list, dtype=np.int64)),
            "lap_type": pa.array(types),
        }
    )


def detect_layout(
    log,
    *,
    declared_track_raw: str | None,
    venue_meta: str | None = None,
    lat_channel: str = "GPS Latitude",
    lon_channel: str = "GPS Longitude",
) -> LayoutDetection:
    """Resolve the layout actually driven and decide whether laps need re-splitting.

    ``declared_track_raw`` is the filename track token; ``venue_meta`` the
    logger's ``Venue`` metadata (used when the filename token is unknown,
    e.g. ``CMD_KK-F_Race_a_0338.xrk``). GPS is the final arbiter for the
    track and the lap boundaries; the declared variant only breaks ties
    between layouts that share all their gates (Fuji GP vs Short).
    """
    declared_raw = declared_track_raw
    declared_canonical = normalize_track_name(declared_raw) if declared_raw else None
    declared_layout = normalize_layout_name(declared_raw)
    if declared_canonical is None and venue_meta:
        declared_canonical = normalize_track_name(venue_meta)
        declared_layout = normalize_layout_name(venue_meta)
        declared_raw = venue_meta
    n_logger = int(len(_full_laps(log.laps)))

    trace = gps_trace_from_log(log, lat_channel=lat_channel, lon_channel=lon_channel)
    if trace is None:
        return LayoutDetection(
            venue_id=None,
            layout_id=declared_layout,
            track_canonical=declared_canonical,
            declared_track_canonical=declared_canonical,
            declared_layout_id=declared_layout,
            venue_meta=venue_meta,
            track_resolution=TRACK_RES_DECLARED,
            lap_source=LAP_SOURCE_UNVERIFIED,
            beacon_gate=None,
            beacon_dist_m=None,
            n_laps_logger=n_logger,
            note="no usable GPS trace",
        )

    venue = find_venue(trace)
    if venue is None:
        return LayoutDetection(
            venue_id=None,
            layout_id=declared_layout,
            track_canonical=declared_canonical,
            declared_track_canonical=declared_canonical,
            declared_layout_id=declared_layout,
            venue_meta=venue_meta,
            track_resolution=TRACK_RES_DECLARED,
            lap_source=LAP_SOURCE_UNVERIFIED,
            beacon_gate=None,
            beacon_dist_m=None,
            n_laps_logger=n_logger,
            note="GPS centroid not near any registered venue",
        )

    crossings = {name: int(gate_crossings(trace, g).size) for name, g in venue.gates.items()}
    layout, why = _pick_layout(venue, crossings, declared_layout)
    if layout is None:
        # Never completed a lap through any gate: trust the declaration if it
        # belongs to this venue, else the venue default.
        if declared_layout and any(l.layout_id == declared_layout for l in venue.layouts):
            layout = venue.layout(declared_layout)
        else:
            layout = venue.layout(venue.default_layout)

    b_lat, b_lon = _logger_lap_start_positions(log, trace)
    beacon_gate: str | None = None
    beacon_dist: float | None = None
    if b_lat.size:
        beacon_gate, beacon_dist = nearest_gate_to_points(venue, b_lat, b_lon)

    if declared_canonical is None:
        resolution = TRACK_RES_GPS
    elif declared_canonical != layout.track_canonical or (
        declared_layout is not None and declared_layout != layout.layout_id
    ):
        resolution = TRACK_RES_GPS_OVERRIDE
    else:
        resolution = TRACK_RES_DECLARED

    resplit: pa.Table | None = None
    if n_logger > 0 and beacon_gate == layout.sf_gate:
        lap_source = LAP_SOURCE_LOGGER
    else:
        ref_lap_s: float | None = None
        if n_logger > 0:
            full = _full_laps(log.laps)
            ld = full.column("end_time").to_numpy().astype(np.float64) - full.column(
                "start_time"
            ).to_numpy().astype(np.float64)
            ref_lap_s = float(np.median(ld)) / 1000.0
        candidate = resplit_laps(trace, venue.gates[layout.sf_gate], ref_lap_s=ref_lap_s)
        if len(candidate) > 0:
            resplit = candidate
            lap_source = LAP_SOURCE_RESPLIT
        elif n_logger > 0 and beacon_gate is not None and beacon_gate != layout.sf_gate:
            # The logger's laps are split on another layout's line and the car
            # never completed a lap through this layout's S/F: those are not
            # laps of the resolved layout (typically a one-lap fragment), so
            # the session ends up with no laps rather than mislabelled ones.
            resplit = candidate
            lap_source = LAP_SOURCE_RESPLIT
            why += "; beacon on wrong gate and no full lap at S/F, logger laps dropped"
        elif n_logger > 0:
            lap_source = LAP_SOURCE_UNVERIFIED
            why += "; S/F resplit found no laps, kept logger laps"
        else:
            lap_source = LAP_SOURCE_UNVERIFIED
            why += "; no logger laps and no S/F crossings"

    return LayoutDetection(
        venue_id=venue.venue_id,
        layout_id=layout.layout_id,
        track_canonical=layout.track_canonical,
        declared_track_canonical=declared_canonical,
        declared_layout_id=declared_layout,
        venue_meta=venue_meta,
        track_resolution=resolution,
        lap_source=lap_source,
        beacon_gate=beacon_gate,
        beacon_dist_m=beacon_dist,
        gate_crossings=crossings,
        n_laps_logger=n_logger,
        resplit_laps=resplit,
        note=why,
    )


def apply_layout_correction(log, det: LayoutDetection) -> None:
    """Replace ``log.laps`` with the GPS re-split table when the detection asks for it.

    Mutates ``log`` in place and recomputes the derived ``lap_time`` column
    the loader adds. The per-lap ``distance_m`` channel (if present) is
    dropped because it was integrated over the old lap boundaries; the ETL
    does not consume it.
    """
    if det.lap_source != LAP_SOURCE_RESPLIT or det.resplit_laps is None:
        return
    from .._util import _add_lap_time

    log.laps = det.resplit_laps
    _add_lap_time(log)
    if isinstance(getattr(log, "channels", None), dict):
        log.channels.pop("distance_m", None)
    logger.info(
        "%s: re-split %d logger laps (beacon on %s) into %d laps at %s S/F (%s)",
        getattr(log, "file_name", "<log>"),
        det.n_laps_logger,
        det.beacon_gate,
        len(det.resplit_laps),
        det.layout_id,
        det.note,
    )
