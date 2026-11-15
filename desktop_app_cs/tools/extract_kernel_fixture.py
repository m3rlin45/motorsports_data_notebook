"""Dump a Python analysis kernel's inputs and outputs for the C# port's parity tests.

Run from the repo root (Rust parser backend, as the C# port binds)::

    LIBXRK_BACKEND=rust uv run python desktop_app_cs/tools/extract_kernel_fixture.py --module corners

Each module writes ``desktop_app_cs/Tests/Fixtures/<module>_86.json`` from the
Fuji 86 sample (laps 14 and 12 unless the module says otherwise). Inputs are
dumped verbatim so the C# kernel can be driven with exactly the arrays Python
saw; outputs are dumped in full where small and as summaries where large.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

SAMPLE = Path("workspace_template/data/CMD_Inferno 86_Fuji GP Sh_Generic testing_a_2248.xrz")
FIXTURES = Path("desktop_app_cs/Tests/Fixtures")
LAPS = [14, 12]


def summary(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "length": int(len(v)),
        "first": float(v[0]),
        "last": float(v[-1]),
        "sum": float(np.sum(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def corners_fixture(log, profile) -> dict:
    from motorsports_data_notebook import corners as C

    lat_name = profile.channel_names["gps_latitude"]
    lon_name = profile.channel_names["gps_longitude"]
    laps = {}
    for num in LAPS:
        lap = log.filter_by_lap(num)
        lat = lap.channels[lat_name].column(lat_name).to_numpy()
        lon = lap.channels[lon_name].column(lon_name).to_numpy()
        x, y = C.gps_to_local_xy(lat, lon)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        distance = np.cumsum(np.sqrt(dx**2 + dy**2))
        curvature, signed, radius = C.compute_curvature(x, y)
        detected = C.identify_corners(lat, lon)
        laps[str(num)] = {
            "lat": lat.tolist(),
            "lon": lon.tolist(),
            "x": summary(x),
            "y": summary(y),
            "path_distance": summary(distance),
            "curvature": summary(curvature),
            "signed_curvature": summary(signed),
            "radius": summary(radius),
            "corners": [dataclasses.asdict(c) for c in detected],
        }
    return {"channels": {"lat": lat_name, "lon": lon_name}, "laps": laps}


def _none_safe(d: dict) -> dict:
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in d.items()}


def zones_fixture(log, profile) -> dict:
    """Zone detection, segments, segment stats and G utilization over laps 14 and 12."""
    import dataclasses

    from motorsports_data_notebook import corners as C
    from motorsports_data_notebook import zones as Z

    names = profile.channel_names
    throttle, brake, speed = names["throttle"], names["brake"], names["gps_speed"]
    lat, lon, lateral = names["gps_latitude"], names["gps_longitude"], names["lateral_g"]
    inline = names.get("inline_g", "")
    has_inline = inline in log.channels

    laps_out = {}
    arrays = {}
    for num in LAPS:
        lap_row = log.laps.to_pandas().set_index("num").loc[num]
        wanted = list(
            dict.fromkeys(
                ["distance_m", brake, throttle, speed, "speed_kmh", lateral]
                + ([inline] if has_inline else [])
            )
        )
        aligned = (
            log.filter_by_lap(num)
            .select_channels(wanted)
            .resample_to_channel("distance_m")
            .channels
        )
        col = (
            lambda n: aligned[n].column(n).to_numpy(zero_copy_only=False).astype(np.float64)
        )  # noqa: E731
        arrays[num] = {
            "distance": col("distance_m"),
            "brake": col(brake),
            "throttle": col(throttle),
            "speed_ms": col(speed),
            "speed_kmh": col("speed_kmh"),
            "lateral_g": col(lateral),
            "inline_g": col(inline) if has_inline else None,
            "lap_time_s": float(lap_row["lap_time"].total_seconds()),
        }
        braking, accel = Z.identify_zones_single_lap(
            arrays[num]["distance"],
            arrays[num]["brake"],
            arrays[num]["throttle"],
            arrays[num]["speed_ms"],
        )
        laps_out[str(num)] = {
            "lap_time_s": arrays[num]["lap_time_s"],
            "inputs": {k: v.tolist() for k, v in arrays[num].items() if isinstance(v, np.ndarray)},
            "braking_zones": braking,
            "accel_zones": accel,
        }

    order = [arrays[n] for n in LAPS]
    braking_avg, accel_avg = Z.detect_zones_from_arrays(
        [a["distance"] for a in order],
        [a["brake"] for a in order],
        [a["throttle"] for a in order],
        [a["speed_ms"] for a in order],
    )
    ref_lap = log.filter_by_lap(LAPS[0])
    corners = C.identify_corners(
        ref_lap.channels[lat].column(lat).to_numpy(), ref_lap.channels[lon].column(lon).to_numpy()
    )
    track_length = float(order[0]["distance"].max())
    segments = Z.create_track_segments(corners, braking_avg, accel_avg, track_length)
    stats = Z.compute_segment_stats_from_arrays(
        [a["distance"] for a in order],
        [a["speed_kmh"] for a in order],
        [a["brake"] for a in order],
        [a["throttle"] for a in order],
        LAPS,
        [a["lap_time_s"] for a in order],
        segments,
    )
    g_util = Z.compute_g_utilization(
        [a["distance"] for a in order],
        [a["speed_kmh"] for a in order],
        [a["lateral_g"] for a in order],
        [a["inline_g"] for a in order] if has_inline else None,
        LAPS,
        segments,
        corners,
    )
    g_util_derived = Z.compute_g_utilization(
        [a["distance"] for a in order],
        [a["speed_kmh"] for a in order],
        [a["lateral_g"] for a in order],
        None,
        LAPS,
        segments,
        corners,
    )
    return {
        "channels": {
            "throttle": throttle,
            "brake": brake,
            "speed": speed,
            "lateral_g": lateral,
            "inline_g": inline if has_inline else None,
        },
        "laps": laps_out,
        "braking_zones": braking_avg,
        "accel_zones": accel_avg,
        "track_length": track_length,
        "corners": [dataclasses.asdict(c) for c in corners],
        "segments": [dataclasses.asdict(s) for s in segments],
        "segment_stats": [_none_safe(r) for r in stats.to_dict(orient="records")],
        "g_utilization": [_none_safe(r) for r in g_util.to_dict(orient="records")],
        "g_utilization_derived_inline": [
            _none_safe(r) for r in g_util_derived.to_dict(orient="records")
        ],
    }


def suspension_fixture(log, profile) -> dict:
    """Velocity histograms: single lap for each of LAPS and the multi-lap aggregate."""
    import dataclasses

    from motorsports_data_notebook import suspension as S

    names = {k: profile.channel_names[k] for k in ("shock_fl", "shock_fr", "shock_rl", "shock_rr")}

    def corner(c: S.CornerVelocityData) -> dict:
        d = dataclasses.asdict(c)
        d["velocity"] = summary(c.velocity)
        d["bin_edges"] = c.bin_edges.tolist()
        d["bin_centers"] = c.bin_centers.tolist()
        d["histogram"] = c.histogram.tolist()
        return d

    def result(r: S.VelocityHistogramResult) -> dict:
        out = {
            k: corner(getattr(r, k))
            for k in ("front_left", "front_right", "rear_left", "rear_right")
        }
        out["symmetry"] = S.format_symmetry_table(r).to_dict(orient="records")
        out["comparison"] = S.format_comparison_table(r).to_dict(orient="records")
        return out

    mr = profile.motion_ratios
    single = {
        str(num): result(
            S.analyze_suspension_velocity(log.filter_by_lap(num), names, motion_ratios=mr)
        )
        for num in LAPS
    }
    multi = result(S.analyze_suspension_velocity_multi_lap(log, LAPS, names, motion_ratios=mr))
    # Exact-math checks for the velocity derivation on a short slice of lap 14's FL channel.
    lap = (
        log.filter_by_lap(LAPS[0])
        .select_channels(list(names.values()))
        .resample_to_channel(names["shock_fl"])
        .channels
    )
    fl = lap[names["shock_fl"]]
    disp = fl.column(names["shock_fl"]).to_numpy(zero_copy_only=False).astype(np.float64)[:64]
    tc = fl.column("timecodes").to_numpy()[:64]
    return {
        "channels": names,
        "motion_ratios": dataclasses.asdict(mr),
        "single_lap": single,
        "multi_lap": multi,
        "velocity_slice": {
            "displacement": disp.tolist(),
            "timecodes": tc.tolist(),
            "raw_velocity": S.compute_shock_velocity(disp, tc, smoothing_window=1).tolist(),
            "smoothed_velocity": S.compute_shock_velocity(disp, tc, smoothing_window=5).tolist(),
        },
    }


def roll_gradient_fixture(log, profile) -> dict:
    """Roll gradient per lap and multi-lap; inputs come from the C# loader, outputs dumped."""
    from motorsports_data_notebook import roll_gradient as R

    keys = [
        "shock_fl",
        "shock_fr",
        "shock_rl",
        "shock_rr",
        "lateral_g",
        "gps_latitude",
        "gps_longitude",
    ]
    names = {k: profile.channel_names[k] for k in keys}
    mr, tw = profile.motion_ratios, profile.track_widths
    assert tw is not None

    def axle(a: R.AxleRollData) -> dict:
        return {
            "axle": a.axle,
            "lateral_g": summary(a.lateral_g),
            "roll_deg": summary(a.roll_deg),
            "gradient_deg_per_g": a.gradient_deg_per_g,
            "intercept_deg": a.intercept_deg,
            "r2": a.r2,
            "bin_centers_g": a.bin_centers_g.tolist(),
            "bin_mean_deg": a.bin_mean_deg.tolist(),
            "bin_counts": a.bin_counts.tolist(),
            "n_samples": a.n_samples,
        }

    def result(r: R.RollGradientResult) -> dict:
        return {
            "front": axle(r.front),
            "rear": axle(r.rear),
            "lateral_g_unit": r.lateral_g_unit,
            "roll_ratio": r.roll_ratio,
            "roll_ratio_intercept_deg": r.roll_ratio_intercept_deg,
            "roll_ratio_r2": r.roll_ratio_r2,
            "ratio_bin_centers_deg": r.ratio_bin_centers_deg.tolist(),
            "ratio_bin_mean_deg": r.ratio_bin_mean_deg.tolist(),
            "ratio_bin_counts": r.ratio_bin_counts.tolist(),
            "x_m": summary(r.x_m) if r.has_track_map else None,
            "y_m": summary(r.y_m) if r.has_track_map else None,
            "gradient_ratio": r.gradient_ratio,
            "mean_gradient_deg_per_g": r.mean_gradient_deg_per_g,
            "signs_consistent": r.signs_consistent,
        }

    single = {
        str(n): result(R.analyze_roll_gradient(log.filter_by_lap(n), names, mr, tw)) for n in LAPS
    }
    multi = result(R.analyze_roll_gradient_multi_lap(log, LAPS, names, mr, tw))
    rng = np.random.default_rng(7)
    g = rng.normal(0, 0.8, 500)
    roll = 1.7 * g + 0.2 + rng.normal(0, 0.1, 500)
    fit = R.fit_roll_gradient(g, roll, min_lateral_g=0.2)
    centers, means, counts = R.bin_roll_by_lateral_g(g, roll, 0.25, 3)
    return {
        "channels": names,
        "motion_ratios": {
            "front_left": mr.front_left,
            "front_right": mr.front_right,
            "rear_left": mr.rear_left,
            "rear_right": mr.rear_right,
        },
        "track_widths": {"front_mm": tw.front_mm, "rear_mm": tw.rear_mm},
        "single_lap": single,
        "multi_lap": multi,
        "synthetic": {
            "lateral_g": g.tolist(),
            "roll_deg": roll.tolist(),
            "fit": list(fit),
            "bin_centers": centers.tolist(),
            "bin_means": means.tolist(),
            "bin_counts": counts.tolist(),
            "smoothed_even": R._smooth(roll[:12], 4).tolist(),
        },
    }


def driver_analysis_fixture(log, profile) -> dict:
    """Throttle acceptance per corner (lap-14 corners) for each lap, on the zones arrays."""
    from motorsports_data_notebook import corners as C
    from motorsports_data_notebook import driver_analysis as D

    names = profile.channel_names
    throttle, lateral, lat, lon = (
        names["throttle"],
        names["lateral_g"],
        names["gps_latitude"],
        names["gps_longitude"],
    )
    ref = log.filter_by_lap(LAPS[0])
    corners = C.identify_corners(
        ref.channels[lat].column(lat).to_numpy(), ref.channels[lon].column(lon).to_numpy()
    )
    laps = {}
    for num in LAPS:
        aligned = (
            log.filter_by_lap(num)
            .select_channels(["distance_m", throttle, lateral])
            .resample_to_channel("distance_m")
            .channels
        )
        distance = aligned["distance_m"].column("distance_m").to_numpy()
        timecodes = aligned["distance_m"].column("timecodes").to_numpy()
        thr = aligned[throttle].column(throttle).to_numpy(zero_copy_only=False).astype(np.float64)
        latg = aligned[lateral].column(lateral).to_numpy(zero_copy_only=False).astype(np.float64)
        smoothed, eff = D.prepare_throttle_acceptance(thr, latg, throttle_threshold=98.0)
        results = []
        for c in corners:
            r = D.find_throttle_acceptance_from_arrays(
                distance,
                timecodes,
                thr,
                latg,
                c,
                smoothed_lateral_g=smoothed,
                effective_threshold=eff,
            )
            results.append(r)
        laps[str(num)] = {
            "inputs": {
                "distance": distance.tolist(),
                "timecodes": timecodes.tolist(),
                "throttle": thr.tolist(),
                "lateral_g": latg.tolist(),
            },
            "effective_threshold": float(eff),
            "smoothed_lateral_g": summary(smoothed),
            "throttle_acceptance": results,
        }
    import dataclasses

    return {
        "channels": {"throttle": throttle, "lateral_g": lateral},
        "corners": [dataclasses.asdict(c) for c in corners],
        "laps": laps,
    }


def tire_grip_fixture(log, profile) -> dict:
    """Tire grip per lap and multi-lap, pressure and temperature modes; inputs from the C# loader."""
    from motorsports_data_notebook import tire_grip as T

    keys = [
        "lateral_g",
        "inline_g",
        "tpms_press_fl",
        "tpms_press_fr",
        "tpms_press_rl",
        "tpms_press_rr",
        "tpms_temp_fl",
        "tpms_temp_fr",
        "tpms_temp_rl",
        "tpms_temp_rr",
    ]
    names = {k: profile.channel_names[k] for k in keys}

    def corner(c: T.CornerTireGripData) -> dict:
        return {
            "corner_name": c.corner_name,
            "total_g": summary(c.total_g),
            "tire_metric": summary(c.tire_metric),
            "mean_g": c.mean_g,
            "mean_metric": c.mean_metric,
            "std_g": c.std_g,
            "std_metric": c.std_metric,
            "bucket_centers": c.bucket_centers.tolist(),
            "bucket_values": c.bucket_values.tolist(),
            "bucket_counts": c.bucket_counts.tolist(),
        }

    def result(r: T.TireGripResult) -> dict:
        out = {
            k: corner(getattr(r, k))
            for k in ("front_left", "front_right", "rear_left", "rear_right")
        }
        out.update(metric_mode=r.metric_mode, metric_unit=r.metric_unit, accel_unit=r.accel_unit)
        return out

    out = {"channels": names, "modes": {}}
    for mode in ("pressure", "temperature"):
        out["modes"][mode] = {
            "single_lap": {
                str(n): result(T.analyze_tire_grip(log.filter_by_lap(n), names, mode)) for n in LAPS
            },
            "multi_lap": result(T.analyze_tire_grip_multi_lap(log, LAPS, names, mode)),
        }
    rng = np.random.default_rng(3)
    g = np.abs(rng.normal(1.0, 0.3, 400))
    p = rng.uniform(28, 34, 400)
    out["synthetic"] = {
        "total_g": g.tolist(),
        "metric": p.tolist(),
        "corner": corner(T._compute_corner(g, p, "FL")),
    }
    return out


MODULES = {
    "corners": corners_fixture,
    "driver_analysis": driver_analysis_fixture,
    "tire_grip": tire_grip_fixture,
    "zones": zones_fixture,
    "suspension": suspension_fixture,
    "roll_gradient": roll_gradient_fixture,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--module", required=True, choices=sorted(MODULES))
    args = parser.parse_args(argv)
    if os.environ.get("LIBXRK_BACKEND") != "rust":
        print("set LIBXRK_BACKEND=rust: the C# port binds the Rust parser", file=sys.stderr)
        return 2

    from motorsports_data_notebook import profiles as P
    from motorsports_data_notebook._util import load_session

    P._get_user_profiles_path = lambda: Path("/nonexistent/profiles.yaml")  # builtins only
    log = load_session(str(SAMPLE))
    profile = P.get_profile_for_log(log)
    assert profile is not None

    fixture = {"sample": SAMPLE.name, "laps": LAPS, **MODULES[args.module](log, profile)}
    out = FIXTURES / f"{args.module}_86.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture) + "\n")
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
