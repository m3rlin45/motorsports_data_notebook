"""Cold tire pressure predictor — physically-based v0.

Stage-by-stage public API (re-exports populated as later commits land
warmup_table.build_warmup_table and predict.predict_cold_pressure).
"""

from __future__ import annotations

from .energy_balance import (
    gay_lussac_cold_pressure_bar,
    t_effective_c,
    t_road_proxy_c,
    warmup_curve_c,
)

__all__ = [
    "gay_lussac_cold_pressure_bar",
    "t_effective_c",
    "t_road_proxy_c",
    "warmup_curve_c",
]
