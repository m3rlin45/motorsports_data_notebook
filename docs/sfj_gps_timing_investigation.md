# SFJ GPS Timing Misalignment Investigation

**Date:** January 25, 2026  
**Files Analyzed:**
- `CMD_SFJ_Fuji GP Sh_Generic testing_a_0101.xrk`
- `CMD_SFJ_Suzuka Car_Generic testing_a_0090.xrk`

## Summary

The BRK and ACCEL channels appeared misaligned with GPS traces. Investigation revealed **two different issues** affecting the SFJ files:

1. **Fuji file**: 65533ms timestamp discontinuity in GPS channels at ~4 seconds (critical bug)
2. **Suzuka file**: Higher GPS noise (~2x) but no major timing issues

## File Comparison

| File | Issue | GPS Noise (std) | Correlation (raw) | Correlation (fixed/smoothed) |
|------|-------|-----------------|-------------------|------------------------------|
| Fuji | 65533ms gap at 4s | 0.084 m/s | 0.08 ⚠️ | 0.82 ✓ |
| Suzuka | Higher GPS noise | 0.162 m/s | 0.76 | 0.81 ✓ |

## Key Finding - Fuji File

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| GPS-derived vs InlineAcc correlation | 0.08 | 0.82 |
| Smoothed correlation | 0.09 | 0.95 |

## Root Cause

At index 87 (timestamp ~3986ms), there is a **65533ms gap** in the GPS timestamps:

```
Before gap: time=3986ms, speed=1.80 m/s
After gap:  time=69519ms, speed=1.96 m/s
Actual gap: 65533ms (should be ~40ms)
```

This affects all GPS-related channels:
- `GPS Speed`
- `GPS Latitude`
- `GPS Longitude`
- `GPS Altitude`

All other channels (BRK, InlineAcc, ACCEL, steering, etc.) have consistent timing with no gaps.

## Investigation Process

### Step 1: Verify Channel Alignment

First verified that non-GPS channels are properly aligned:

| Comparison | Correlation | Expected |
|------------|-------------|----------|
| BRK vs InlineAcc | -0.876 | Negative ✓ |
| Throttle vs InlineAcc | +0.647 | Positive ✓ |
| GPS-derived vs InlineAcc | +0.090 | Should be high ✗ |

**Conclusion:** BRK and InlineAcc are perfectly aligned. The issue is specifically with GPS.

### Step 2: Check Time Ranges

```
GPS Speed:  506ms - 1,198,710ms (extends 66s beyond other channels)
InlineAcc:  0ms   - 1,132,780ms
BRK:        462ms - 1,133,262ms
```

GPS data extends ~66 seconds longer than other channels, matching the spurious gap.

### Step 3: Identify the Gap

```python
gps_dt = np.diff(gps_time)
gap_idx = np.argmax(gps_dt)  # Index 87
gap_size = gps_dt[gap_idx]   # 65533ms
```

The gap value (65533 ≈ 0xFFED) suggests possible integer overflow or parsing error.

### Step 4: Verify Fix

Applied correction by subtracting (65533 - 40) = 65493ms from all GPS timestamps after the gap:

**Braking Event Validation:**

| Event | GPS Δv (fixed) | InlineAcc ∫a·dt | Match |
|-------|----------------|-----------------|-------|
| 89.4-91.0s | -10.0 m/s | -11.1 m/s | ✓ |
| 90.4-91.0s | -2.7 m/s | -2.9 m/s | ✓ |
| 103.0-104.5s | -7.6 m/s | -7.5 m/s | ✓ |

## Channel Characteristics

| Channel | Sample Rate | Start Time | End Time |
|---------|-------------|------------|----------|
| InlineAcc | 50 Hz (20ms) | 0ms | 1,132,780ms |
| BRK | 50 Hz (20ms) | 462ms | 1,133,262ms |
| GPS Speed | 25 Hz (40ms) | 506ms | 1,198,710ms* |
| Throttle | ~10 Hz (98ms) | 475ms | 1,133,173ms |
| steering | ~44 Hz (22ms) | 462ms | 1,133,262ms |

*Contains 65533ms spurious gap

## Possible Causes

1. **libxrk parsing bug** - GPS timestamps may be parsed incorrectly from the binary format
2. **Data corruption** - The original .xrk file may have corrupted GPS timestamp data
3. **AiM logger behavior** - GPS module may have had a timing glitch during recording
4. **Integer overflow** - The gap value (65533 ≈ 2^16 - 3) suggests a 16-bit overflow issue

## Recommended Next Steps

1. **Check other files** - Verify if this issue affects the Inferno 86 sample file
2. **Examine libxrk source** - Look at how GPS timestamps are parsed in `aim_xrk.pyx`
3. **Implement workaround** - Add gap detection/correction to the data loading pipeline
4. **Report upstream** - If this is a libxrk bug, document for future fix

## Scripts Created

| Script | Purpose |
|--------|---------|
| `debug_sfj_channels.py` | List all channels with sample rates |
| `debug_sfj_acceleration.py` | Compare GPS-derived vs InlineAcc |
| `debug_sfj_crosscorr.py` | Cross-correlation analysis |
| `debug_sfj_brk_acc.py` | Verify BRK/Throttle vs InlineAcc alignment |
| `debug_sfj_gps_speed.py` | GPS speed signal quality analysis |
| `debug_sfj_timing.py` | Detailed timing analysis |
| `debug_sfj_fix_timing.py` | Verify timing fix improves correlation |
| `debug_sfj_braking.py` | Braking event visualization |

## Workaround Code

```python
def fix_gps_timing(gps_time: np.ndarray, expected_dt: float = 40.0) -> np.ndarray:
    """Detect and correct large timing gaps in GPS data."""
    gps_time_fixed = gps_time.copy()
    dt = np.diff(gps_time)
    
    # Find gaps significantly larger than expected
    gap_threshold = expected_dt * 10  # 400ms
    gap_indices = np.where(dt > gap_threshold)[0]
    
    for gap_idx in gap_indices:
        gap_size = dt[gap_idx]
        correction = gap_size - expected_dt
        gps_time_fixed[gap_idx + 1:] -= correction
        
    return gps_time_fixed
```

---

## Suzuka File Analysis

### Overview

The Suzuka file (`CMD_SFJ_Suzuka Car_Generic testing_a_0090.xrk`) has **no critical timing bug** but exhibits different characteristics.

### Channel Time Ranges

| Channel | Start Time | End Time | Samples |
|---------|------------|----------|---------|
| GPS Speed | 131ms | 1,888,839ms | 45,909 |
| InlineAcc | 0ms | 1,823,180ms | 91,160 |
| BRK | 161ms | 1,823,361ms | 91,161 |

### GPS Timing

- **Gap detected**: 52376ms at 1456.58s (index 36410)
- This gap is near the **end of the session** (~24 minutes into a 30-minute session)
- Likely represents end of recording / GPS signal loss, not a parsing bug

### GPS Signal Quality

| Metric | Suzuka | Fuji |
|--------|--------|------|
| GPS noise (std) | 0.162 m/s | 0.084 m/s |
| GPS-derived accel range (1-99%) | -1.00 to 0.56 G | -1.19 to 0.57 G |
| InlineAcc range (1-99%) | -1.10 to 0.73 G | -1.20 to 0.62 G |

The Suzuka GPS has **~2x higher noise** than Fuji, likely due to:
- Different GPS reception conditions at Suzuka circuit
- More trees/buildings causing multipath interference
- Different satellite geometry on recording day

### Correlation by Time Window

| Time Window | Correlation | Notes |
|-------------|-------------|-------|
| 0-300s | 0.61 | Lower (pit lane / stationary periods) |
| 300-600s | 0.77 | Good |
| 600-900s | 0.81 | Good |
| 900-1200s | 0.80 | Good |
| 1200-1500s | 0.76 | Good |
| 1500-1800s | -0.10 | Broken (after 52s gap) |

### Suzuka Conclusions

1. ✅ No major timing bug like Fuji file
2. ⚠️ 52-second gap at end is legitimate (end of session)
3. ⚠️ GPS signal is noisier (~2x noise std)
4. ✅ Correlation is acceptable (0.76-0.81) when smoothed
5. ⚠️ BRK vs InlineAcc correlation (-0.68) is lower than Fuji (-0.88)

### Additional Scripts for Suzuka

| Script | Purpose |
|--------|---------|
| `check_sfj_files.py` | Compare both SFJ files |
| `debug_suzuka.py` | Initial Suzuka analysis |
| `debug_suzuka_early.py` | Check early timing issues |
| `debug_suzuka_noise.py` | GPS noise comparison |
