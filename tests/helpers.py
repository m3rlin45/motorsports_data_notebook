"""Shared test fixtures and helpers.

Provides a unified MockLogFile and factory functions used across test modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa


@dataclass
class MockLogFile:
    """Mock LogFile for testing.

    Supports method chaining (filter_by_lap, select_channels, resample_to_channel)
    and exposes channels, laps, metadata, and file_name attributes.
    """

    channels: dict[str, pa.Table] = field(default_factory=dict)
    laps: pa.Table | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    file_name: str = "test.xrk"

    def __post_init__(self):
        if self.laps is None:
            self.laps = pa.table({"num": [], "start_time": [], "end_time": []})

    def filter_by_lap(self, lap_num: int):
        """Mock filter_by_lap returning self for method chaining."""
        return self

    def select_channels(self, channel_names: list):
        """Mock select_channels returning self for method chaining."""
        return self

    def resample_to_channel(self, reference_channel: str):
        """Mock resample_to_channel returning self for method chaining."""
        return self


def make_channel_table(timecodes: np.ndarray, name: str, values: np.ndarray) -> pa.Table:
    """Create a PyArrow table with timecodes and a named column."""
    return pa.table(
        {
            "timecodes": pa.array(timecodes, type=pa.int64()),
            name: pa.array(values),
        }
    )


def make_table_with_unit(
    timecodes: np.ndarray, name: str, values: np.ndarray, unit: str
) -> pa.Table:
    """Create a PyArrow table with unit metadata on the value field."""
    schema = pa.schema(
        [
            pa.field("timecodes", pa.int64()),
            pa.field(name, pa.float64(), metadata={"units": unit}),
        ]
    )
    return pa.table({"timecodes": timecodes, name: values}, schema=schema)
