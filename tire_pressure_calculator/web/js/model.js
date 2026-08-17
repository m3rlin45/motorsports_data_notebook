// Pure model logic for the tire pressure calculator web app.
//
// JS port of the C# Core modeling layer, which is itself a port of the
// Python source of truth:
//   - EnergyBalance:    src/motorsports_data_notebook/tire_model/energy_balance.py
//   - TireModel/lookups + predictCorner: src/motorsports_data_notebook/tire_model/predict.py
//
// Any change here MUST be mirrored in the Python module and the C# Core
// (Core/Services/Modeling/*), and is pinned against the shared fixture at
// Tests/Fixtures/python_predictions.json by web/tests/model.test.mjs.

export const P_ATM_BAR = 1.0;
export const T_ZERO_C_TO_K = 273.15;

// v3 adds the target-lap-time feature; v2 artifacts still load (the pace
// scaling then always uses the exponent fallback defaults).
export const SUPPORTED_SCHEMA_VERSION = 3;
export const MIN_SUPPORTED_SCHEMA_VERSION = 2;

// C# Math.Round uses banker's rounding (half to even); mirror it so the
// web app displays the exact same values as the desktop/Android heads.
export function roundTo(value, digits) {
  const factor = 10 ** digits;
  const x = value * factor;
  const floor = Math.floor(x);
  if (Math.abs(x - floor - 0.5) < Number.EPSILON) {
    return (floor % 2 === 0 ? floor : floor + 1) / factor;
  }
  return Math.round(x) / factor;
}

export function tEffectiveC(tAirC, tRoadC, wRoad) {
  if (wRoad < 0 || wRoad > 1) throw new RangeError(`w_road must be in [0, 1]; got ${wRoad}`);
  return (1 - wRoad) * tAirC + wRoad * tRoadC;
}

export function warmupCurveC(tSeconds, tEffC, kKelvinPerG2, cTrack, g2Typ, tauSec) {
  if (tauSec <= 0) throw new RangeError(`tau_sec must be > 0; got ${tauSec}`);
  if (tSeconds < 0) throw new RangeError(`t_seconds must be >= 0; got ${tSeconds}`);
  const warmupFrac = 1 - Math.exp(-tSeconds / tauSec);
  const deltaTInf = kKelvinPerG2 * cTrack * g2Typ;
  return tEffC + deltaTInf * warmupFrac;
}

// Invert Gay-Lussac: cold gauge pressure from target hot gauge pressure +
// hot/cold temperatures (gauge <-> absolute via +pAtm).
export function gayLussacColdPressureBar(targetHotPressureBar, tHotC, tColdC, pAtmBar = P_ATM_BAR) {
  const tColdK = tColdC + T_ZERO_C_TO_K;
  const tHotK = tHotC + T_ZERO_C_TO_K;
  if (tHotK <= 0 || tColdK <= 0) {
    throw new RangeError(`Absolute temperatures must be positive (got T_hot_K=${tHotK}, T_cold_K=${tColdK})`);
  }
  const pHotAbs = targetHotPressureBar + pAtmBar;
  const pColdAbs = pHotAbs * (tColdK / tHotK);
  return pColdAbs - pAtmBar;
}

// Track-surface temperature proxy from air temp + cloud cover.
// 0 % cloud cover => T_air + deltaSunMaxC; 100 % => T_air.
export function tRoadProxyC(tAirC, cloudCoverPct, sunFactor = 1.0, deltaSunMaxC = 10.0) {
  if (cloudCoverPct === null || cloudCoverPct === undefined) return tAirC;
  const clamped = Math.max(0, Math.min(100, cloudCoverPct));
  return tAirC + deltaSunMaxC * (1 - clamped / 100) * sunFactor;
}

// ---- Manual-mode helpers (port of TireCornerViewModel math) ----

// Percent adjustment applied in Kelvin space, rounded to 0.1 degC.
export function adjustedHotTempC(targetHotTempC, adjustPercent) {
  const baseK = targetHotTempC + T_ZERO_C_TO_K;
  const adjustedK = baseK * (1 + adjustPercent / 100);
  return roundTo(adjustedK - T_ZERO_C_TO_K, 1);
}

// Displayed "Set Cold" value: Gay-Lussac inversion rounded to 3 decimals,
// with the same non-physical-temperature guard as the corner view model.
export function cornerColdPressureBar(targetHotPressureBar, effectiveHotTempC, currentTempC) {
  if (effectiveHotTempC + T_ZERO_C_TO_K <= 0) return 0;
  if (currentTempC + T_ZERO_C_TO_K <= 0) return 0;
  return roundTo(gayLussacColdPressureBar(targetHotPressureBar, effectiveHotTempC, currentTempC), 3);
}

// ---- Condition fallback chain (mirrors predict.py:_CONDITION_FALLBACK) ----

const CONDITION_CHAIN = {
  dry: ['dry'],
  damp: ['damp', 'dry'],
  wet: ['wet', 'damp', 'dry'],
};

export function conditionChain(condition) {
  return CONDITION_CHAIN[condition] ?? [condition, 'dry'];
}

const average = (rows, pick) => rows.reduce((s, r) => s + pick(r), 0) / rows.length;
const sum = (rows, pick) => rows.reduce((s, r) => s + pick(r), 0);

// Piecewise-linear interpolation clamped to the endpoints. Must stay in
// lockstep with the Python and C# implementations (pinned by the parity
// fixture).
export function interpClamped(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) {
      const w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + w * (ys[i] - ys[i - 1]);
    }
  }
  return ys[ys.length - 1];
}

// In-memory wrapper around the parsed tire_model.json (schema v3; v2
// artifacts still load). Adds the lookup helpers + fallback chains that
// mirror the Python predictor.
export class TireModel {
  constructor(dto) {
    if (dto.schema_version < MIN_SUPPORTED_SCHEMA_VERSION
        || dto.schema_version > SUPPORTED_SCHEMA_VERSION) {
      throw new Error(
        `Unsupported tire_model.json schema_version ${dto.schema_version}; ` +
        `expected ${MIN_SUPPORTED_SCHEMA_VERSION}-${SUPPORTED_SCHEMA_VERSION}. ` +
        'Rebuild with `just tire-build-warmup-table`.');
    }
    this.dto = dto;
  }

  get availableCars() {
    return [...new Set(this.dto.tau_sec_by_car_corner_cond.map((r) => r.car))].sort();
  }

  get availableTracks() {
    // Every track with observed data, not just those with a fitted
    // c_track — thin tracks (e.g. Motegi's 18 Inferno laps) predict via
    // the c_track prior until enough laps accumulate.
    const fitted = this.dto.c_track_by_track.map((r) => r.track_canonical);
    const observed = this.dto.g2_typ_by_track_car_cond.map((r) => r.track_canonical);
    return [...new Set([...fitted, ...observed])].sort();
  }

  get availableConditions() { return this.dto.conditions.values; }
  get defaultCondition() { return this.dto.conditions.default; }
  get wRoad() { return this.dto.energy_balance.w_road; }
  get sunFactorDefault() { return this.dto.energy_balance.t_road_proxy.sun_factor_default; }
  get deltaSunMaxC() { return this.dto.energy_balance.t_road_proxy.delta_sun_max_c; }
  get pAtmBar() { return this.dto.gay_lussac.p_atm_bar; }

  lookupTau(car, corner, condition) {
    const rows = this.dto.tau_sec_by_car_corner_cond;
    for (const cond of conditionChain(condition)) {
      const hit = rows.find((r) => r.car === car && r.corner === corner && r.condition === cond);
      if (hit) {
        return {
          valueSeconds: hit.value_seconds, stderrSeconds: hit.stderr_seconds,
          sourceBucket: `(${car}, ${corner}, ${cond})`, fromPrior: hit.from_prior,
        };
      }
    }
    const sameCC = rows.filter((r) => r.car === car && r.corner === corner);
    if (sameCC.length > 0) {
      return {
        valueSeconds: average(sameCC, (r) => r.value_seconds), stderrSeconds: 0,
        sourceBucket: `(${car}, ${corner})`, fromPrior: false,
      };
    }
    const sameCar = rows.filter((r) => r.car === car);
    if (sameCar.length > 0) {
      return {
        valueSeconds: average(sameCar, (r) => r.value_seconds), stderrSeconds: 0,
        sourceBucket: `(${car})`, fromPrior: false,
      };
    }
    return {
      valueSeconds: this.dto.priors_when_no_fit.tau_sec_seconds, stderrSeconds: 0,
      sourceBucket: '(prior)', fromPrior: true,
    };
  }

  lookupK(car, corner, condition) {
    const rows = this.dto.K_buckets;
    for (const cond of conditionChain(condition)) {
      const hit = rows.find(
        (r) => r.key.car === car && r.key.corner === corner && r.key.condition === cond);
      if (hit) {
        return {
          valueKelvinPerG2: hit.value_kelvin_per_g2, stderrKelvinPerG2: hit.stderr_kelvin_per_g2,
          nSamples: hit.n_samples,
          sourceBucket: `(${car}, ${corner}, ${cond})`, fromPrior: hit.from_prior,
        };
      }
    }
    const sameCC = rows.filter((r) => r.key.car === car && r.key.corner === corner);
    if (sameCC.length > 0) {
      return {
        valueKelvinPerG2: average(sameCC, (r) => r.value_kelvin_per_g2), stderrKelvinPerG2: 0,
        nSamples: sum(sameCC, (r) => r.n_samples),
        sourceBucket: `(${car}, ${corner})`, fromPrior: false,
      };
    }
    const sameCar = rows.filter((r) => r.key.car === car);
    if (sameCar.length > 0) {
      return {
        valueKelvinPerG2: average(sameCar, (r) => r.value_kelvin_per_g2), stderrKelvinPerG2: 0,
        nSamples: sum(sameCar, (r) => r.n_samples),
        sourceBucket: `(${car})`, fromPrior: false,
      };
    }
    return {
      valueKelvinPerG2: this.dto.priors_when_no_fit.K_kelvin_per_g2, stderrKelvinPerG2: 0,
      nSamples: 0, sourceBucket: '(prior)', fromPrior: true,
    };
  }

  lookupCTrack(track) {
    const hit = this.dto.c_track_by_track.find((r) => r.track_canonical === track);
    if (hit) return { value: hit.value, stderr: hit.stderr, fromPrior: false };
    return { value: this.dto.priors_when_no_fit.c_track, stderr: 0, fromPrior: true };
  }

  lookupG2(track, car, condition) {
    const rows = this.dto.g2_typ_by_track_car_cond;
    for (const cond of conditionChain(condition)) {
      const hit = rows.find(
        (r) => r.track_canonical === track && r.car === car && r.condition === cond);
      if (hit) {
        const source = cond === condition ? 'exact' : `fallback(${cond})`;
        return { value: hit.g2_typ, nLapsUsed: hit.n_laps_used, source };
      }
    }
    const sameTC = rows.filter((r) => r.track_canonical === track && r.car === car);
    if (sameTC.length > 0) {
      return {
        value: average(sameTC, (r) => r.g2_typ),
        nLapsUsed: sum(sameTC, (r) => r.n_laps_used), source: 'track_car_pooled',
      };
    }
    const sameT = rows.filter((r) => r.track_canonical === track);
    if (sameT.length > 0) {
      return {
        value: average(sameT, (r) => r.g2_typ),
        nLapsUsed: sum(sameT, (r) => r.n_laps_used), source: 'track_pooled',
      };
    }
    if (rows.length > 0) {
      return { value: average(rows, (r) => r.g2_typ), nLapsUsed: 0, source: 'global' };
    }
    return { value: 0.7, nLapsUsed: 0, source: 'global' };
  }

  // ---- Compound-aware K (decomposed c_track × base × multiplier) ----

  // Distinct compounds fitted for a car, for UI enumeration.
  availableCompounds(car) {
    const rows = this.dto.K_by_car_compound_corner_cond ?? [];
    return [...new Set(rows.filter((r) => r.car === car).map((r) => r.compound))].sort();
  }

  // Compound-specific K via the condition chain, or null when the artifact
  // has no fitted bucket (caller falls back to the pooled K).
  lookupCompoundK(car, compound, corner, condition) {
    const rows = this.dto.K_by_car_compound_corner_cond ?? [];
    for (const cond of conditionChain(condition)) {
      const hit = rows.find(
        (r) => r.car === car && r.compound === compound
          && r.corner === corner && r.condition === cond);
      if (hit) {
        return {
          valueKelvinPerG2: hit.value_kelvin_per_g2,
          stderrKelvinPerG2: hit.stderr_kelvin_per_g2 ?? 0,
          nSamples: hit.n_laps ?? 0,
          sourceBucket: `(${car}, ${compound}, ${corner}, ${cond})`,
          fromPrior: false,
        };
      }
    }
    return null;
  }

  // Steady-state hot temp / hot pressure medians for UI prefills, via the
  // condition chain; null when the artifact has no entry (caller keeps its
  // static defaults).
  lookupCornerDefaults(car, corner, condition) {
    const rows = this.dto.corner_defaults_by_car_corner_cond ?? [];
    for (const cond of conditionChain(condition)) {
      const hit = rows.find(
        (r) => r.car === car && r.corner === corner && r.condition === cond);
      if (hit) {
        return {
          hotTempC: hit.hot_temp_c,
          hotPressureBar: hit.hot_pressure_bar,
          nLapsUsed: hit.n_laps_used ?? 0,
          source: cond === condition ? 'exact' : `fallback(${cond})`,
        };
      }
    }
    return null;
  }

  // ---- Target-lap-time pace scaling (schema v3) ----

  lookupG2PaceCurve(track, car, condition) {
    const rows = this.dto.g2_typ_by_track_car_cond;
    for (const cond of conditionChain(condition)) {
      const hit = rows.find(
        (r) => r.track_canonical === track && r.car === car && r.condition === cond);
      if (hit) return hit.g2_vs_lap_time ?? null;
    }
    return null;
  }

  // Multiplier on g2_typ for a target lap time; mirrors the Python
  // predictor's _g2_pace_scale. Preferred: ratio along the bucket's
  // sector-fit curve (anchored at lap_time_typ so target === typical
  // scales by exactly 1). Fallback: the pooled sector exponent.
  // Clamped either way.
  g2PaceScale(track, car, condition, lapTimeTypS, targetLapTimeS) {
    const paceCfg = this.dto.g2_lap_time_model ?? {};
    const lo = paceCfg.multiplier_clamp?.min ?? 0.4;
    const hi = paceCfg.multiplier_clamp?.max ?? 2.5;

    const curve = this.lookupG2PaceCurve(track, car, condition);
    if (curve) {
      const ref = interpClamped(lapTimeTypS, curve.lap_time_s, curve.g2);
      if (ref > 0) {
        const scale = interpClamped(targetLapTimeS, curve.lap_time_s, curve.g2) / ref;
        return { scale: Math.min(hi, Math.max(lo, scale)), source: 'curve' };
      }
    }
    const exponent = paceCfg.default_exponent ?? 3.0;
    const scale = (lapTimeTypS / targetLapTimeS) ** exponent;
    return { scale: Math.min(hi, Math.max(lo, scale)), source: 'exponent' };
  }

  lookupLapTime(track, car, condition) {
    const rows = this.dto.lap_time_typ_by_track_car_cond;
    for (const cond of conditionChain(condition)) {
      const hit = rows.find(
        (r) => r.track_canonical === track && r.car === car && r.condition === cond);
      if (hit) {
        const source = cond === condition ? 'exact' : `fallback(${cond})`;
        return { valueSeconds: hit.lap_time_typ_s, nLapsUsed: hit.n_laps_used, source };
      }
    }
    const sameTC = rows.filter((r) => r.track_canonical === track && r.car === car);
    if (sameTC.length > 0) {
      return {
        valueSeconds: average(sameTC, (r) => r.lap_time_typ_s),
        nLapsUsed: sum(sameTC, (r) => r.n_laps_used), source: 'track_car_pooled',
      };
    }
    const sameT = rows.filter((r) => r.track_canonical === track);
    if (sameT.length > 0) {
      return {
        valueSeconds: average(sameT, (r) => r.lap_time_typ_s),
        nLapsUsed: sum(sameT, (r) => r.n_laps_used), source: 'track_pooled',
      };
    }
    return { valueSeconds: 90.0, nLapsUsed: 0, source: 'global' };
  }
}

// Per-corner cold-pressure prediction (port of CircuitPredictor.Predict /
// predict.py:predict_cold_pressure). Returns the recommended cold pressure
// plus the intermediate quantities.
export function predictCorner(model, {
  track, car, condition, lapWithinStint, ambientTempC,
  trackTempC = null, cloudCoverPct = null, corner,
  targetHotPressureBar, coldTireTempC = null, targetLapTimeS = null,
  compound = null,
}) {
  const cond = condition.toLowerCase();
  if (cond !== 'dry' && cond !== 'damp' && cond !== 'wet') {
    throw new RangeError(`track_condition must be dry/damp/wet; got '${condition}'`);
  }

  // One tire on all four corners — the compound applies to every corner.
  let k = model.lookupK(car, corner, cond);
  if (compound) {
    const hit = model.lookupCompoundK(car, compound, corner, cond);
    if (hit) k = hit;
  }
  const tau = model.lookupTau(car, corner, cond);
  const c = model.lookupCTrack(track);
  const g2 = model.lookupG2(track, car, cond);
  const lap = model.lookupLapTime(track, car, cond);

  // T_road: user-supplied -> sun-cover proxy -> fall back to T_air.
  const tRoadC = trackTempC ?? tRoadProxyC(
    ambientTempC, cloudCoverPct, model.sunFactorDefault, model.deltaSunMaxC);
  const tEffC = tEffectiveC(ambientTempC, tRoadC, model.wRoad);

  const tColdC = coldTireTempC ?? ambientTempC;

  // Target-lap-time feature: pace sets both time-on-track and tire energy.
  let g2Scale = 1.0;
  let g2PaceSource = null;
  let lapTimeForClockS = lap.valueSeconds;
  let g2Value = g2.value;
  if (targetLapTimeS !== null && targetLapTimeS !== undefined) {
    if (!(targetLapTimeS > 0)) {
      throw new RangeError(`target lap time must be > 0; got ${targetLapTimeS}`);
    }
    const pace = model.g2PaceScale(track, car, cond, lap.valueSeconds, targetLapTimeS);
    g2Scale = pace.scale;
    g2PaceSource = pace.source;
    g2Value *= g2Scale;
    lapTimeForClockS = targetLapTimeS;
  }

  const tAtLapNs = lapWithinStint * lapTimeForClockS;
  const warmupFrac = tau.valueSeconds > 0 ? 1 - Math.exp(-tAtLapNs / tau.valueSeconds) : 0;
  const deltaTInf = k.valueKelvinPerG2 * c.value * g2Value;
  const tHotC = warmupCurveC(tAtLapNs, tEffC, k.valueKelvinPerG2, c.value, g2Value, tau.valueSeconds);

  const coldPressureBar = gayLussacColdPressureBar(
    targetHotPressureBar, tHotC, tColdC, model.pAtmBar);

  return {
    corner,
    coldPressureBar,
    predictedHotTempC: tHotC,
    targetHotPressureBar,
    kKelvinPerG2: k.valueKelvinPerG2,
    tauSec: tau.valueSeconds,
    cTrack: c.value,
    g2Typ: g2Value,
    lapTimeTypS: lap.valueSeconds,
    tAtLapNs,
    warmupFrac,
    deltaTInfKelvin: deltaTInf,
    tEffC,
    tAirC: ambientTempC,
    tRoadC,
    tColdC,
    kSourceBucket: k.sourceBucket,
    kFromPrior: k.fromPrior,
    kNSamples: k.nSamples,
    targetLapTimeS,
    g2Scale,
    g2PaceSource,
  };
}
