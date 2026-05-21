"""Group sequential laps into stints based on off-track gaps."""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def assign_stint_ids(
    laps: pa.Table,
    *,
    gap_s: float = 600.0,
) -> pa.Array:
    """Return an int16 array of stint IDs, one per row of ``laps``.

    A new stint starts whenever ``start_time[i] - end_time[i-1]`` exceeds
    ``gap_s`` seconds (pit-and-wait between sessions). Stint IDs start at 1
    and are strictly monotonic.

    ``laps`` is expected to have ``start_time`` and ``end_time`` columns in
    milliseconds (AIM convention); the table must already be sorted by
    ``start_time``.
    """
    n = len(laps)
    if n == 0:
        return pa.array([], type=pa.int16())

    starts = laps.column("start_time").to_numpy().astype(np.int64)
    ends = laps.column("end_time").to_numpy().astype(np.int64)

    gap_ms = int(gap_s * 1000.0)
    stint = np.ones(n, dtype=np.int16)
    for i in range(1, n):
        if starts[i] - ends[i - 1] > gap_ms:
            stint[i] = stint[i - 1] + 1
        else:
            stint[i] = stint[i - 1]
    return pa.array(stint, type=pa.int16())
