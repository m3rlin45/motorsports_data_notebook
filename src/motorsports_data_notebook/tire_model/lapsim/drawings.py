"""The operators' published geometry, transcribed (gate B2, issue #204).

Each circuit's corners in lap order from the start/finish line, as the
operator prints them: direction as driven, centre-line radius, and where
the document gives them the arc length, the station and the track width.
Transcribed on 2026-11-15 from the documents named in ``source``; every
radius was read from the document itself. Directions are printed for some
corners and otherwise read off the plan (noted per circuit).

What a driven line can be held to:

- **The order of lefts and rights.** Consecutive elements turning the
  same way are one group; the groups alternate, and the profile's corner
  table must alternate the same way (:func:`turn_sequence`).
- **Net turning between long straights** (:func:`straight_to_straight`),
  where the document gives arc lengths. On a long straight the car is
  aligned with the track whatever line it took through the corners, so the
  heading change from one long straight to the next is the drawing's. It
  is not the same corner by corner: through two corners close together
  the driven line cuts across the short straight between them.
- **A bound on the apex.** A centre-line radius is not the driven radius
  (the line is straighter), but the car cannot turn tighter than the
  inside edge, ``R − w/2``.

Fuji is absent: its operator publishes no radii for the current surface.
"""

from __future__ import annotations

import math

#: A straight at least this long separates two independent pieces of the
#: lap. At Sodegaura the 92 m and 94 m straights either side of the 100R
#: kink are not straight on the driven line (the car is still unwinding
#: from the hairpin before it): cut there, the two pieces differ from the
#: drawing by +9.5° and −9.6°, which cancel.
LONG_STRAIGHT_M = 100.0

# (direction, radius_m, arc_m or None, station_from_m or None, width_m or None)
_SODEGAURA = [
    ("R", 40, 65.0, 162.0, None),
    ("R", 95, 94.7, 383.4, None),
    ("R", 70, 84.2, 697.1, None),
    ("R", 25, 55.8, 824.1, None),
    ("R", 100, 11.9, 974.3, None),
    ("L", 100, 49.8, 1078.3, None),
    ("L", 70, 75.0, 1152.7, None),
    ("L", 60, 87.0, 1237.7, None),
    ("R", 100, 45.5, 1434.3, None),
    ("L", 55, 158.8, 1542.7, None),
    ("R", 25, 78.5, 1781.5, None),
    ("R", 220, 48.0, 1918.5, None),
    ("R", 94, 28.3, 2013.5, None),
    ("R", 50, 95.0, 2095.7, None),
]
# Width is the widest section the table lists for the corner.
_TSUKUBA = [
    ("R", 55, None, None, 15.05),
    ("R", 35, None, None, 10.10),
    ("L", 105, None, None, 10.45),
    ("R", 75, None, None, 10.35),
    ("L", 25, None, None, 13.25),
    ("R", 35, None, None, 13.40),
    ("L", 80, None, None, 14.25),
    ("L", 80, None, None, 10.25),
    ("R", 25, None, None, 13.00),
    ("R", 105, None, None, 10.40),
    ("R", 100, None, None, 10.95),
    ("R", 90, None, None, 17.20),
]
_SUZUKA = [
    ("R", 100), ("R", 60), ("L", 70), ("R", 70), ("L", 75), ("R", 85), ("R", 70),
    ("L", 77.5), ("L", 210), ("L", 130), ("L", 200), ("R", 15), ("R", 25), ("R", 110),
    ("L", 20), ("R", 200), ("R", 120), ("R", 300), ("R", 250), ("L", 60), ("L", 200),
    ("L", 60), ("L", 100), ("L", 200), ("L", 1000), ("L", 85), ("L", 340), ("R", 250),
    ("R", 9.5), ("L", 20), ("R", 115), ("R", 250),
]  # fmt: skip
# None: the plan shows the corner without a radius.
_MOTEGI = [
    ("R", 45), ("R", 45), ("L", 30), ("L", 70), ("R", 30), ("R", 130), ("L", 60),
    ("R", 70), ("L", 30), ("R", None), ("R", 25), ("L", None), ("R", 30),
]  # fmt: skip


def _elements(rows: list[tuple]) -> list[dict]:
    out = []
    for row in rows:
        direction, radius = row[0], row[1]
        arc, station, width = (list(row[2:]) + [None, None, None])[:3]
        out.append(
            {
                "sign": 1 if direction == "R" else -1,
                "radius_m": radius,
                "arc_m": arc,
                "station_m": station,
                "width_m": width,
            }
        )
    return out


DRAWINGS: dict[str, dict] = {
    "sodegaura": {
        "source": "https://www.sodegaura-forest-raceway.com/course/a01-model.pdf (land-use "
        "planning drawing, 1/1000, a pre-construction sheet) and .../course/index.html",
        "length_m": 2436.2,
        "width_m": (15.0, 18.0),
        "directions": "read off the plan; the arcs then close to 360.0 degrees",
        "elements": _elements(_SODEGAURA),
    },
    "tsukuba_2000": {
        "source": "https://www.tsukuba-circuit.jp/guide/course-spec.html (table "
        "img/cource-spec_03.svg, drawing img/course-spec_2000_01.pdf)",
        "length_m": 2045.0,
        "width_m": (10.1, 19.75),
        "directions": "read off the plan; the operator's text gives Dunlop right and 80R left. "
        "The text also calls the second 80R '170R', against the table and the drawing",
        "elements": _elements(_TSUKUBA),
    },
    "suzuka_full": {
        "source": "https://www.suzukacircuit.jp/course_s/pdf/suzuka_courseguide.pdf (curve "
        "radius strip; the guide says shape and dimensions may differ from reality)",
        "length_m": 5807.0,
        "dimensions_disclaimed": True,
        "width_m": (10.0, 16.0),
        "directions": "radius lines on the plan and the guide's text; R120, R300 and the R250 "
        "before the chicane only from the lap's zero net turning",
        "elements": _elements(_SUZUKA),
    },
    "motegi_full": {
        "source": "https://www.mr-motegi.jp/course_m/pdf/racing_course.pdf",
        "length_m": 4801.379,
        "width_m": (12.0, 15.0),
        "directions": "read off the plan; it prints 8 right and 6 left, which needs the bend "
        "after the second underpass counted as a left (no radius printed)",
        "elements": _elements(_MOTEGI),
    },
}


def turn_sequence(signs: list[int]) -> list[int]:
    """``signs`` with every run of equal values collapsed to one: the
    order in which a lap alternates between right (+1) and left (−1)."""
    out: list[int] = []
    for s in signs:
        if not out or out[-1] != s:
            out.append(int(s))
    return out


def groups(elements: list[dict]) -> list[list[dict]]:
    """Consecutive elements turning the same way, as lists."""
    out: list[list[dict]] = []
    for el in elements:
        if out and out[-1][0]["sign"] == el["sign"]:
            out[-1].append(el)
        else:
            out.append([el])
    return out


def straight_to_straight(drawing: dict) -> list[dict]:
    """The lap cut at the middle of every straight of
    :data:`LONG_STRAIGHT_M` or more: ``station_from_m, station_to_m`` (the
    last piece runs through the start line, so its end exceeds the lap
    length) and ``turn_deg``, the drawing's net turning in the piece
    (right positive). Empty unless every element has an arc length and a
    station."""
    els = drawing["elements"]
    if any(e["arc_m"] is None or e["station_m"] is None for e in els):
        return []
    length = drawing["length_m"]
    cuts: list[float] = []
    for i, el in enumerate(els):
        end = el["station_m"] + el["arc_m"]
        nxt = els[(i + 1) % len(els)]["station_m"] + (length if i + 1 == len(els) else 0.0)
        if nxt - end >= LONG_STRAIGHT_M:
            cuts.append(0.5 * (end + nxt))
    out = []
    for a, b in zip(cuts, cuts[1:] + [cuts[0] + length]):
        turn = sum(
            e["sign"] * math.degrees(e["arc_m"] / e["radius_m"])
            for e in els
            if a <= e["station_m"] < b or a <= e["station_m"] + length < b
        )
        out.append({"station_from_m": a, "station_to_m": b, "turn_deg": turn})
    return out


def inside_edge_radius(group: list[dict], default_width_m: float) -> float | None:
    """The tightest radius a car could drive in a group of elements: the
    smallest ``R − w/2`` over those with a printed radius (``w`` the
    element's own width where printed, else ``default_width_m``); None when
    no element has a radius."""
    vals = [
        e["radius_m"] - 0.5 * (e["width_m"] or default_width_m)
        for e in group
        if e["radius_m"] is not None
    ]
    return min(vals) if vals else None
