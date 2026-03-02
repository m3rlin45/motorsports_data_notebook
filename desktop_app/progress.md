# Driver Consistency Performance Optimization Progress

## Baseline
- **Average time:** 2.132s (9 corners x 14 laps)
- **Best time:** 2.112s
- **filter_by_lap calls:** ~140 (126 inner loop + 14 other)
- **Dominant cost:** pyarrow compute in filter_by_time_range

## Step 1: Reorder iteration (lap-first, corner-second)
- **Average time:** 0.804s
- **Best time:** 0.782s
- **Improvement:** -62.3% avg (2.65x speedup)
- **filter_by_lap calls:** ~28 (14 inner loop + 14 other)
- **Output:** 118 TAs, 126 traces (identical to baseline)

## Step 2: Pre-group stats_df and segments — SKIPPED
- Tested: 0.815s avg (within noise of Step 1)
- Reverted: negligible gain, adds complexity, doesn't block later steps

## Step 3: Eliminate DataFrames (numpy arrays)
- **Average time:** 0.814s
- **Best time:** 0.789s
- **Improvement:** within noise of Step 1 (-0.5%)
- **Output:** 118 TAs, 126 traces (identical)
- **Notes:** Full-lap DataFrame eliminated; per-corner slicing uses np.searchsorted.
  Still builds small DataFrame per corner for find_throttle_acceptance — removes
  that overhead once TA is inlined in Step 4.

## Step 4: Inline find_throttle_acceptance
- **Average time:** 0.664s
- **Best time:** 0.643s
- **Improvement:** -18.4% from Step 3, -68.9% from baseline (3.21x speedup)
- **Output:** 118 TAs, 126 traces (identical)
- **Notes:** Lateral G smoothing + threshold adaptation computed ONCE per lap
  instead of per (corner x lap). All DataFrames eliminated from hot path.
  `find_throttle_acceptance` import removed (logic inlined with numpy arrays).

## Step 5: Padded 2D vectorized computation — SKIPPED
- Tested: 0.668s avg (within noise of Step 4)
- Reverted: no measurable gain at 9 corners, significantly more complex code

## Final Profile (after Step 4)

| Component | Time | % | Calls |
|-----------|------|---|-------|
| filter_by_lap (total) | 0.51s | 61% | 43 |
| compute_segment_stats | 0.45s | 53% | 1 (includes 13 filter_by_lap) |
| detect_zones_averaged | 0.19s | 23% | 1 (includes 13 filter_by_lap) |
| Our per-corner loop | <0.01s | <1% | — |
| pyarrow compute (self) | 0.42s | 50% | 13,287 |

**Remaining bottleneck is entirely in zones.py** (compute_segment_stats + detect_zones_averaged),
not in driver_consistency.py. Further optimization requires changes to those upstream functions.

## Phase 2: Single-pass extraction + array-based APIs
- **Average time:** 0.248s
- **Best time:** 0.244s
- **Improvement:** -62.7% from Phase 1, -88.4% from baseline (8.60x speedup)
- **filter_by_lap calls:** 15 (1 GPS + 14 extraction loop)
- **Output:** 118 TAs, 126 traces (identical)
- **Changes:**
  - New `detect_zones_from_arrays()` in zones.py — accepts pre-extracted numpy arrays
  - New `compute_segment_stats_from_arrays()` in zones.py — numpy helpers replace DataFrame ops
  - New `prepare_throttle_acceptance()` + `find_throttle_acceptance_from_arrays()` in driver_analysis.py
  - Single extraction loop in driver_consistency.py caches all arrays, shared across zones/stats/TA
  - Existing LogFile-based functions refactored to thin wrappers (backward compatible)
  - Inlined TA logic replaced with standalone function calls

## Summary

| Step | Avg Time | Speedup | Status |
|------|----------|---------|--------|
| Baseline | 2.132s | 1.0x | — |
| Step 1: Lap-first iteration | 0.804s | 2.65x | Kept |
| Step 2: Pre-group stats | 0.815s | — | Skipped |
| Step 3: Numpy arrays | 0.814s | — | Kept (enables Step 4) |
| Step 4: Inline TA | 0.660s | 3.23x | Kept |
| Step 5: 2D vectorized | 0.668s | — | Skipped |
| Phase 2: Array-based APIs | 0.248s | 8.60x | Kept |
