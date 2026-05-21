"""Tire ETL pipeline.

Batch-processes AIM telemetry sessions plus hand-written run notes plus
historical weather into a queryable Parquet dataset under ``data/tire_dataset/``.

This module is populated incrementally as the stack lands:
- scaffold: paths, tracks, discovery
- transforms: aggregates, filters, stints
- extract + dataset
- notes parsing
- weather
- CLI
"""

from __future__ import annotations

EXTRACTOR_VERSION = "0.2.0"

__all__ = ["EXTRACTOR_VERSION"]
