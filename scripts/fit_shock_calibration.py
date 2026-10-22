#!/usr/bin/env python
"""Fit stroke-sensor calibration tables and print profile motion ratios.

Measure one corner per axle with the spring and anti-roll bar removed: move the
wheel through its travel in steps and record, at each step, the wheel position
(mm, any fixed reference), the damper length (mm) and the stroke-sensor reading
shown by the logger (mm). Save it as a CSV with columns
``wheel_mm,damper_mm,sensor_mm`` and run::

    uv run python scripts/fit_shock_calibration.py front.csv rear.csv --exclude-rear 1 2

Straight lines are fitted by least squares; only slopes matter, so the zero
references do not, and the pots only need to be zeroed on the car.

The suspension analysis uses the POT/WHEEL motion ratio only: it divides the
logged stroke-sensor channel by ``1 / wheel_per_sensor`` to get wheel travel.
Paste the printed ``shock_calibration`` block into the vehicle profile
(``builtin_profiles.yaml``); ``motion_ratios`` derive from it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def read_table(path: Path) -> np.ndarray:
    """Read a wheel_mm,damper_mm,sensor_mm CSV (header optional) into an (n, 3) array."""
    rows: list[list[float]] = []
    with open(path, newline="") as f:
        for raw in csv.reader(f):
            cells = [c.strip() for c in raw if c.strip()]
            if len(cells) < 3:
                continue
            try:
                rows.append([float(c) for c in cells[:3]])
            except ValueError:
                continue  # header or junk line
    if len(rows) < 3:
        raise SystemExit(f"{path}: need at least 3 numeric rows, got {len(rows)}")
    return np.asarray(rows)


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) of the least-squares line y = slope*x + intercept."""
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), r2


def fit_axle(name: str, table: np.ndarray, exclude: list[int]) -> dict:
    """Print the fits for one axle and return the profile ``shock_calibration`` entry."""
    mask = np.ones(len(table), dtype=bool)
    for i in exclude:
        if not 0 <= i < len(table):
            raise SystemExit(f"{name}: exclude index {i} out of range 0..{len(table) - 1}")
        mask[i] = False
    if mask.sum() < 3:
        raise SystemExit(f"{name}: fewer than 3 points left after exclusions")
    wheel, damper, sensor = table[mask].T

    kw, _, r2w = linear_fit(sensor, wheel)
    kd, _, r2d = linear_fit(sensor, damper)
    km, _, r2m = linear_fit(wheel, damper)
    wheel_per_sensor = abs(kw)
    damper_per_sensor = abs(kd)

    print(f"{name} ({int(mask.sum())}/{len(table)} points used)")
    print(f"  wheel  = {wheel_per_sensor:.4f} * sensor   R^2 = {r2w:.4f}")
    print(f"  damper = {damper_per_sensor:.4f} * sensor   R^2 = {r2d:.4f}")
    print(f"  damper = {abs(km):.4f} * wheel    R^2 = {r2m:.4f}   (damper/wheel motion ratio)")
    print(f"  -> analysis motion_ratio (pot/wheel) = {1.0 / wheel_per_sensor:.3f}")
    if (~mask).any():
        resid_w = table[:, 0] - np.polyval(np.polyfit(sensor, wheel, 1), table[:, 2])
        for row_idx in np.where(~mask)[0]:
            print(f"  excluded row {row_idx}: wheel residual {resid_w[row_idx]:+.1f} mm")

    entry = {
        "wheel_per_sensor": round(wheel_per_sensor, 4),
        "damper_per_sensor": round(damper_per_sensor, 4),
        "damper_per_wheel": round(abs(km), 4),
        "wheel_r2": round(r2w, 4),
        "damper_r2": round(r2d, 4),
        "damper_wheel_r2": round(r2m, 4),
        "points": [[float(v) for v in table_row] for table_row in table],
    }
    if exclude:
        entry["excluded_points"] = sorted(set(exclude))
    return entry


def _yaml_axle(name: str, entry: dict, indent: str = "      ") -> str:
    lines = [f"{indent}{name}:"]
    for key in (
        "wheel_per_sensor",
        "damper_per_sensor",
        "damper_per_wheel",
        "wheel_r2",
        "damper_r2",
        "damper_wheel_r2",
    ):
        lines.append(f"{indent}  {key}: {entry[key]}")
    lines.append(f"{indent}  points:")
    for row in entry["points"]:
        lines.append(f"{indent}    - [{row[0]:g}, {row[1]:g}, {row[2]:g}]")
    if entry.get("excluded_points"):
        lines.append(f"{indent}  excluded_points: {entry['excluded_points']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("front", type=Path, help="front axle CSV (wheel_mm,damper_mm,sensor_mm)")
    parser.add_argument("rear", type=Path, nargs="?", help="rear axle CSV (optional)")
    parser.add_argument(
        "--exclude-front", type=int, nargs="*", default=[], help="0-based rows to drop"
    )
    parser.add_argument(
        "--exclude-rear", type=int, nargs="*", default=[], help="0-based rows to drop"
    )
    args = parser.parse_args(argv)

    front = fit_axle("front", read_table(args.front), args.exclude_front)
    rear = fit_axle("rear", read_table(args.rear), args.exclude_rear) if args.rear else None

    print("\n# Paste into the vehicle profile (motion_ratios derive from it):")
    print("    shock_calibration:")
    print(_yaml_axle("front", front))
    if rear is not None:
        print(_yaml_axle("rear", rear))
    return 0


if __name__ == "__main__":
    sys.exit(main())
