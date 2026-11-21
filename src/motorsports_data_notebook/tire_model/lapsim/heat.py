"""Production's heat chain on one lap (lap simulation, issues #205, #206).

The pressure model's heat input is ``heat_input.share_heat_rate`` on 1 s
means of the four non-negative parts of the accelerations (right turns,
left turns, drive, braking) and of speed (``statespace._bin_stint`` and
``statespace.lap_heat_share``). The positive parts are taken sample by
sample, *before* the 1 s means: 15–20 % of seconds hold both drive and
braking, and taking them after the mean loses 0.9–1.8 % of the heat.

:func:`bin_1s` and :func:`lap_heat` are that chain for one lap of any
series (a logged lap, the same lap in the ground frame, a simulated one),
with the seconds counted from the lap's start so that two series of one lap
are binned alike: the phase of the bins moves a lap's heat by 0.5–1.8 %.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..heat_input import DEFAULT_G_CLIP, ShareParams, share_heat_rate

CORNERS = ("fl", "fr", "rl", "rr")
#: A lap whose samples leave more than this share of its seconds empty has
#: no heat: an integral with a hole is not a smaller integral.
COVERAGE_MIN = 0.98


@dataclass(frozen=True)
class LapBins:
    """One lap as 1 s means, the seconds counted from the lap's start."""

    #: Seconds of the lap each bin stands for: 1, less in the last bin, 0
    #: where the bin holds no sample.
    dur_s: np.ndarray
    v_ms: np.ndarray
    lat_pos: np.ndarray  # right turns (they load the left tyres)
    lat_neg: np.ndarray
    long_pos: np.ndarray  # drive
    long_neg: np.ndarray  # braking
    lap_time_s: float

    @property
    def coverage(self) -> float:
        """Share of the lap's time that lies in a bin with samples."""
        return float(self.dur_s.sum() / self.lap_time_s) if self.lap_time_s > 0 else 0.0


def bin_1s(
    t: np.ndarray,
    v: np.ndarray,
    lat_g: np.ndarray,
    long_g: np.ndarray,
    t_start: float,
    t_end: float,
    *,
    g_clip: float = DEFAULT_G_CLIP,
) -> LapBins:
    """The samples with ``t_start ≤ t < t_end`` as 1 s means.

    Each bin is the plain mean of its samples, as production takes it (a
    series on an irregular clock is not re-weighted by its time steps). A
    sample with a non-finite speed or acceleration is left out; it is not
    counted as zero. Accelerations are clipped to ``± g_clip``.
    """
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    lat = np.asarray(lat_g, float)
    lng = np.asarray(long_g, float)
    lap_time = float(t_end - t_start)
    n = max(int(np.ceil(lap_time - 1e-9)), 1)
    keep = (t >= t_start) & (t < t_end) & np.isfinite(v) & np.isfinite(lat) & np.isfinite(lng)
    b = np.clip(np.floor(t[keep] - t_start).astype(np.int64), 0, n - 1)
    lat = np.clip(lat[keep], -g_clip, g_clip)
    lng = np.clip(lng[keep], -g_clip, g_clip)
    cnt = np.bincount(b, minlength=n).astype(float)

    def _mean(x: np.ndarray) -> np.ndarray:
        return np.asarray(np.bincount(b, weights=x, minlength=n) / np.maximum(cnt, 1.0))

    dur = np.ones(n)
    dur[-1] = lap_time - (n - 1)
    return LapBins(
        dur_s=np.where(cnt > 0, dur, 0.0),
        v_ms=_mean(v[keep]),
        lat_pos=_mean(np.maximum(lat, 0.0)),
        lat_neg=_mean(np.maximum(-lat, 0.0)),
        long_pos=_mean(np.maximum(lng, 0.0)),
        long_neg=_mean(np.maximum(-lng, 0.0)),
        lap_time_s=lap_time,
    )


def lap_heat(bins: LapBins, share: ShareParams) -> np.ndarray:
    """Lap-mean heat rate per tyre (``fl, fr, rl, rr``; G·m/s): the share
    model on the bins, weighted by what each bin stands for. NaN when the
    bins cover less than :data:`COVERAGE_MIN` of the lap."""
    if bins.coverage < COVERAGE_MIN:
        return np.full(len(CORNERS), np.nan)
    total = float(bins.dur_s.sum())
    return np.array(
        [
            float(
                np.sum(
                    share_heat_rate(
                        share,
                        bins.lat_pos,
                        bins.lat_neg,
                        bins.long_pos,
                        bins.long_neg,
                        bins.v_ms,
                        c,
                    )
                    * bins.dur_s
                )
                / total
            )
            for c in CORNERS
        ]
    )


def series_heat(
    t: np.ndarray,
    v: np.ndarray,
    lat_g: np.ndarray,
    long_g: np.ndarray,
    share: ShareParams,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """:func:`lap_heat` of :func:`bin_1s` of a series."""
    return lap_heat(bin_1s(t, v, lat_g, long_g, t_start, t_end), share)


__all__ = ["CORNERS", "COVERAGE_MIN", "LapBins", "bin_1s", "lap_heat", "series_heat"]
