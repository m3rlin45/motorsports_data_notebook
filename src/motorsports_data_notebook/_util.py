"""Internal utilities shared across modules."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

# IBT file magic bytes: little-endian uint32 with value 2
_IBT_MAGIC = b"\x02\x00\x00\x00"


def infer_channel_scale(data: np.ndarray) -> float:
    """Infer whether channel data uses 0-1 or 0-100 scale.

    Parameters
    ----------
    data : np.ndarray
        Channel data array (e.g., throttle or brake values).

    Returns
    -------
    float
        ``1.0`` for 0-1 scale data, ``100.0`` for 0-100 scale data.
    """
    max_val = float(np.max(np.abs(data)))
    return 1.0 if max_val <= 1.5 else 100.0


def detect_file_type(file_data: Union[str, bytes]) -> Literal["aim", "ibt"]:
    """Detect telemetry file type from path or bytes.

    Parameters
    ----------
    file_data : str or bytes
        File path string or raw file bytes.

    Returns
    -------
    Literal["aim", "ibt"]
        ``"aim"`` for XRK/XRZ files, ``"ibt"`` for iRacing IBT files.
    """
    if isinstance(file_data, str):
        lower = file_data.lower()
        if lower.endswith(".ibt"):
            return "ibt"
        return "aim"

    # bytes: check IBT magic header
    if isinstance(file_data, (bytes, bytearray)) and len(file_data) >= 4:
        if file_data[:4] == _IBT_MAGIC:
            return "ibt"
    return "aim"


def validate_channel_names(channel_names: dict, required_keys: list[str], func_name: str) -> None:
    """Validate that required keys are present in channel_names dict.

    Parameters
    ----------
    channel_names : dict
        Channel name mapping from canonical names to actual channel names.
    required_keys : list[str]
        List of required keys that must be present.
    func_name : str
        Name of the calling function (for error messages).

    Raises
    ------
    KeyError
        If any required key is missing from channel_names.
    """
    missing = [key for key in required_keys if key not in channel_names]
    if missing:
        raise KeyError(
            f"{func_name}() requires channel_names to have keys: {required_keys}. "
            f"Missing: {missing}"
        )


def clean_laps_table(laps_table: pa.Table, channels: dict) -> pa.Table:
    """Remove incomplete laps and deduplicate lap numbers.

    Detects partial laps (out-laps and in-laps) using LapDistPct if available
    (iRacing), falling back to duration-based filtering for AIM data.

    iRacing IBT files can contain:
    - Pit markers: ~0.016s single-sample entries at every pit in/out
    - Partial laps: short fragments from aborted laps or session resets
    - Duplicate lap numbers: lap counter resets on each pit stop

    Parameters
    ----------
    laps_table : pa.Table
        Raw laps table with 'num', 'start_time', 'end_time' columns.
    channels : dict
        Channel data dict from the LogFile (used to access LapDistPct).

    Returns
    -------
    pa.Table
        Cleaned laps table with incomplete laps removed and unique lap numbers.
    """
    if len(laps_table) == 0:
        return laps_table

    start_times = laps_table.column("start_time").to_numpy()
    end_times = laps_table.column("end_time").to_numpy()
    durations_ms = end_times - start_times
    lap_nums = laps_table.column("num").to_pylist()
    n = len(lap_nums)

    valid = np.ones(n, dtype=bool)

    # Step 1: Remove the out-lap (lap 0) — always the first lap in both formats
    for i in range(n):
        if lap_nums[i] == 0:
            valid[i] = False

    # Step 2: Detect incomplete laps
    if "LapDistPct" in channels:
        # iRacing: remove pit markers (< 10s) and use track position to detect
        # partial laps (out-laps from pit, in-laps that didn't reach S/F)
        valid &= durations_ms >= 10_000

        ldp_table = channels["LapDistPct"]
        ldp_vals = ldp_table.column("LapDistPct").to_numpy(zero_copy_only=False)
        ldp_tc = ldp_table.column("timecodes").to_numpy()

        for i in range(n):
            if not valid[i]:
                continue
            mask = (ldp_tc >= start_times[i]) & (ldp_tc < end_times[i])
            lap_ldp = ldp_vals[mask]
            if len(lap_ldp) < 2:
                valid[i] = False
                continue
            start_pct = float(lap_ldp[0])
            end_pct = float(lap_ldp[-1])
            # Out-lap: didn't start near S/F line (> 2% into track)
            if 0.02 < start_pct < 0.98:
                valid[i] = False
            # In-lap: didn't reach S/F line (< 98% around track)
            if 0.02 < end_pct < 0.98:
                valid[i] = False

    # Step 3: Deduplicate — for each lap number, keep only the longest entry
    best_by_num: dict[int, tuple[int, float]] = {}  # num -> (row_index, duration)
    for i in range(n):
        if not valid[i]:
            continue
        num = lap_nums[i]
        dur = float(durations_ms[i])
        if num not in best_by_num or dur > best_by_num[num][1]:
            best_by_num[num] = (i, dur)

    keep_indices = sorted(idx for idx, _ in best_by_num.values())

    if not keep_indices:
        return laps_table  # Don't filter if it would remove everything

    return laps_table.take(keep_indices)
