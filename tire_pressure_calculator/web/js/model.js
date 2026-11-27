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

// v5 drops the per-track constant c_track (the circuit enters only through
// its per-corner heat input q_typ_by_corner / outlap_q_by_corner) and adds
// an informational `heat_input` block (ignored here). Older artifacts
// carried K values fitted against c_track and are no longer loadable.
export const SUPPORTED_SCHEMA_VERSION = 5;

// The entry tables per raw car beside the pooled ones (#383; additive in
// schema 5): the same rows, `car` the car the session names, for each raw
// car `car_aliases` pools under another name. Mirrors predict.py.
export const RAW_CAR_TABLES = {
  g2_typ_by_track_car_cond: 'g2_typ_by_track_raw_car_cond',
  lap_time_typ_by_track_car_cond: 'lap_time_typ_by_track_raw_car_cond',
  outlap_typ_by_track_car_cond: 'outlap_typ_by_track_raw_car_cond',
};
export const MIN_SUPPORTED_SCHEMA_VERSION = 5;

// Where an entry row's numbers come from (its `source`, absent meaning
// logged): 'sim' is a lap simulated on a committed track, whose road bank is
// processed from GSI's elevation tiles, so a prediction that reads one shows
// GSI's attribution. Mirrors predict.py.
export const SIM = 'sim';
export const LOGGED = 'logged';

// How a simulated pace curve answers the prediction's lap time (its
// `support`): answered either way, flagged unless 'ok'. Mirrors predict.py.
export const PACE_OK = 'ok';
export const PACE_OUTSIDE_SUPPORT = 'outside_support';
export const PACE_NO_SOLUTION = 'no_solution';

// The steps of answeringRows that answer with an average of rows.
const AVERAGED_STEPS = ['track_car_pooled', 'track_pooled', 'global'];

const isSim = (r) => (r.source ?? LOGGED) === SIM;

// The logged rows: what an averaged fallback step reads (the lead's ruling
// of 2026-11-27), so that a car with no simulated entry of its own reads
// what it read before the simulated rows came. Mirrors predict.py.
const logged = (rows) => rows.filter((r) => !isSim(r));

// 'sim' if any of the rows is simulated, else 'logged'.
export function rowProvenance(rows) {
  return rows.some(isSim) ? SIM : LOGGED;
}

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

// Closed-form warmup from the tire's starting temperature tStartC (default:
// in equilibrium at tEffC). The start excess decays with the same tau:
//   T(t) = T_eff + K*q*(1 - e^{-t/tau}) + (T_start - T_eff)*e^{-t/tau}
// Schema v5 carries no per-track constant: the circuit enters only through
// its per-corner heat input q.
export function warmupCurveC(tSeconds, tEffC, kKelvinPerG2, g2Typ, tauSec, tStartC = null) {
  if (tauSec <= 0) throw new RangeError(`tau_sec must be > 0; got ${tauSec}`);
  if (tSeconds < 0) throw new RangeError(`t_seconds must be >= 0; got ${tSeconds}`);
  const decay = Math.exp(-tSeconds / tauSec);
  const warmupFrac = 1 - decay;
  const deltaTInf = kKelvinPerG2 * g2Typ;
  const start = tStartC ?? tEffC;
  return tEffC + deltaTInf * warmupFrac + (start - tEffC) * decay;
}

// Warm-up from pit exit: the out-lap (gentler, its own rolling time) from
// tStartC, then the flying laps from the temperature at the end of the
// out-lap. Exact for piecewise-constant g². Returns [tAfterOutlap, tHot].
export function warmupTwoStageC(
  tOutlapS, g2Outlap, tFlyingS, g2Flying, tEffC, kKelvinPerG2, tauSec, tStartC) {
  const tAfterOut = warmupCurveC(
    Math.max(0, tOutlapS), tEffC, kKelvinPerG2, g2Outlap, tauSec, tStartC);
  const tHot = warmupCurveC(
    Math.max(0, tFlyingS), tEffC, kKelvinPerG2, g2Flying, tauSec, tAfterOut);
  return [tAfterOut, tHot];
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

// Per-corner value from a v5 {fl, fr, rl, rr} map, else the entry's corner
// mean. A null corner (or a map without that corner) keeps the mean, so
// pre-v5 artifacts produce identical numbers.
const perCornerOr = (byCorner, corner, mean) => {
  if (corner === null || corner === undefined || !byCorner) return mean;
  const v = byCorner[corner];
  return typeof v === 'number' ? v : mean;
};

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

  // The cars the model's constants were fitted as (FJ, Inferno 86).
  get fittedCars() {
    return [...new Set(this.dto.tau_sec_by_car_corner_cond.map((r) => r.car))].sort();
  }

  isFittedCar(car) { return this.fittedCars.includes(car); }

  // The cars offered for selection: the fitted cars, with a pooled car
  // whose raw cars have their own entries in the artifact (#383) listed as
  // those raw cars (KK-SII, KK-F, KK-F-V2 in place of FJ). An artifact
  // without raw-car rows offers the fitted cars, as before.
  get availableCars() {
    const fitted = this.fittedCars;
    const aliases = this.dto.car_aliases ?? {};
    const split = new Set((this.dto[RAW_CAR_TABLES.g2_typ_by_track_car_cond] ?? [])
      .map((r) => r.car)
      .filter((raw) => this.resolveCar(raw) !== raw && fitted.includes(this.resolveCar(raw)))
      .map((raw) => this.resolveCar(raw)));
    const raws = Object.keys(aliases).filter((raw) => aliases[raw] !== raw && split.has(aliases[raw]));
    return [...fitted.filter((c) => !split.has(c)), ...raws].sort();
  }

  // The car to select for a saved one: kept where it is offered; a fitted
  // (pooled) name saved before its raw cars were listed keeps working on
  // the pooled entries; a raw name the artifact does not list resolves to
  // its pooled car, as before. null for a car the model does not know.
  selectableCar(saved) {
    if (!saved) return null;
    const offered = this.availableCars;
    if (offered.includes(saved) || this.isFittedCar(saved)) return saved;
    const pooled = this.resolveCar(saved);
    return offered.includes(pooled) || this.isFittedCar(pooled) ? pooled : null;
  }

  // Newest session that fed the fit, for the footer: the display-ready
  // track-local start time ("2026-10-02 14:23 JST") when the artifact has
  // it, else its date, else (older artifacts) the fit timestamp's date.
  get dataThrough() {
    const local = this.dto.data_through_local;
    if (typeof local === 'string' && local) return local;
    const d = this.dto.data_through_date;
    if (typeof d === 'string' && d) return d.slice(0, 10);
    const iso = this.dto.fit_at_utc;
    return typeof iso === 'string' && iso ? iso.slice(0, 10) : null;
  }

  // Resolve an alias-pooled car name (e.g. KK-F / KK-SII -> FJ) onto the
  // label the model was fitted with. Unknown cars pass through.
  resolveCar(car) {
    const aliases = this.dto.car_aliases;
    return (aliases && Object.prototype.hasOwnProperty.call(aliases, car)) ? aliases[car] : car;
  }

  // The row of an entry table (g2 / lap time / out-lap) that answers
  // (track, car, condition) at one step of the condition chain: a raw car's
  // own row in the table's raw-car sibling (`*_by_track_raw_car_cond`, #383)
  // where it has one, else the fitted car's. A simulated row sits beside the
  // logged one and answers only its own exact condition (schema §9), chosen
  // by its `source` (absent: logged), never by its place in the table: at
  // the `exact` step the order is raw sim, raw logged, pooled sim, pooled
  // logged; at any other step (a condition fallback) only logged rows count,
  // the raw car's before the pooled car's. Mirrors predict.py's _entry_row.
  entryRow(table, track, car, condition, { exact = true } = {}) {
    const fit = this.resolveCar(car);
    const tables = [...(fit !== car ? [[RAW_CAR_TABLES[table], car]] : []), [table, fit]];
    const kinds = exact ? [SIM, LOGGED] : [LOGGED];
    for (const [name, who] of tables) {
      const rows = (this.dto[name] ?? []).filter(
        (r) => r.track_canonical === track && r.car === who && r.condition === condition);
      for (const kind of kinds) {
        const hit = rows.find((r) => (isSim(r) ? SIM : LOGGED) === kind);
        if (hit) return hit;
      }
    }
    return null;
  }

  // Every condition's logged row at `track` for the car whose entries
  // answer there: the raw car's own where it has any, else the fitted car's
  // (the track_car_pooled fallback; an average, so a simulated row is not
  // counted).
  entryRowsAtTrack(table, track, car) {
    const fit = this.resolveCar(car);
    if (fit !== car) {
      const own = logged((this.dto[RAW_CAR_TABLES[table]] ?? []).filter(
        (r) => r.track_canonical === track && r.car === car));
      if (own.length > 0) return own;
    }
    return logged((this.dto[table] ?? []).filter((r) => r.track_canonical === track && r.car === fit));
  }

  // The rows of an entry table that answer (track, car, condition), and the
  // step that found them: one row by the condition chain ('exact',
  // 'fallback(<cond>)'; a raw car's own first; a simulated row only at the
  // exact condition), else every condition's row
  // at the track for the car ('track_car_pooled'), else the pooled rows at
  // the track ('track_pooled'), else, with globalStep, every pooled row
  // ('global'; none: 'global' with no rows). The averaged steps read logged
  // rows only, and a step the filter leaves empty is skipped. Every lookup
  // walks this one chain, and a prediction's provenance is read from the
  // rows it found. Mirrors predict.py's _answering_rows.
  answeringRows(table, track, car, condition, { globalStep }) {
    for (const cond of conditionChain(condition)) {
      const exact = cond === condition;
      const hit = this.entryRow(table, track, car, cond, { exact });
      if (hit) return { rows: [hit], source: exact ? 'exact' : `fallback(${cond})` };
    }
    const sameTC = this.entryRowsAtTrack(table, track, car);
    if (sameTC.length > 0) return { rows: sameTC, source: 'track_car_pooled' };
    const rows = logged(this.dto[table] ?? []);
    const sameT = rows.filter((r) => r.track_canonical === track);
    if (sameT.length > 0) return { rows: sameT, source: 'track_pooled' };
    return { rows: globalStep ? rows : [], source: 'global' };
  }

  get availableTracks() {
    // Every track with an entry (a typical heat input), a raw car's own
    // included (a track only the simulation covers for it).
    const observed = [
      ...this.dto.g2_typ_by_track_car_cond,
      ...(this.dto[RAW_CAR_TABLES.g2_typ_by_track_car_cond] ?? []),
    ].map((r) => r.track_canonical);
    return [...new Set(observed)].sort();
  }

  get availableConditions() { return this.dto.conditions.values; }
  get defaultCondition() { return this.dto.conditions.default; }
  get wRoad() { return this.dto.energy_balance.w_road; }
  get sunFactorDefault() { return this.dto.energy_balance.t_road_proxy.sun_factor_default; }
  get deltaSunMaxC() { return this.dto.energy_balance.t_road_proxy.delta_sun_max_c; }
  get pAtmBar() { return this.dto.gay_lussac.p_atm_bar; }

  // The constants (tau, K, compound K, corner defaults, hot-pressure
  // targets) are the fitted car's: a raw name resolves through car_aliases.
  lookupTau(car, corner, condition) {
    car = this.resolveCar(car);
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
    car = this.resolveCar(car);
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

  // Typical flying-lap heat input (g² in v2-v4; the per-corner q_typ in v5
  // when `corner` is given and the entry carries q_typ_by_corner). Pooled
  // fallbacks average the same per-corner value over the pooled rows.
  lookupG2(track, car, condition, corner = null) {
    const value = (r) => perCornerOr(r.q_typ_by_corner, corner, r.g2_typ);
    const { rows, source } = this.answeringRows(
      'g2_typ_by_track_car_cond', track, car, condition, { globalStep: true });
    if (rows.length === 0) return { value: 0.7, nLapsUsed: 0, source: 'global' };
    if (!AVERAGED_STEPS.includes(source)) {
      return { value: value(rows[0]), nLapsUsed: rows[0].n_laps_used, source };
    }
    return {
      value: average(rows, value),
      nLapsUsed: source === 'global' ? 0 : sum(rows, (r) => r.n_laps_used), source,
    };
  }

  // ---- Compound-aware K (base × compound multiplier) ----

  // Distinct compounds fitted for a car, for UI enumeration.
  availableCompounds(car) {
    car = this.resolveCar(car);
    const rows = this.dto.K_by_car_compound_corner_cond ?? [];
    return [...new Set(rows.filter((r) => r.car === car).map((r) => r.compound))].sort();
  }

  // Compound-specific K via the condition chain, or null when the artifact
  // has no fitted bucket (caller falls back to the pooled K).
  lookupCompoundK(car, compound, corner, condition) {
    car = this.resolveCar(car);
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
    car = this.resolveCar(car);
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

  // Team hot-pressure target for (car, compound) from
  // hot_pressure_targets.yaml (bar gauge), or null when the artifact has
  // none for that pair (caller falls back to the measured median).
  lookupTargetHotPressure(car, compound) {
    if (!compound) return null;
    car = this.resolveCar(car);
    const rows = this.dto.target_hot_pressure_by_car_compound ?? [];
    const hit = rows.find((r) => r.car === car && r.compound === compound);
    return hit ? { targetHotPressureBar: hit.target_hot_pressure_bar } : null;
  }

  // ---- Target-lap-time pace scaling (schema v3) ----

  // The pace curve of the row lookupG2 reads (a condition-chain hit; an
  // average of rows has none).
  lookupG2PaceCurve(track, car, condition) {
    const { rows, source } = this.answeringRows(
      'g2_typ_by_track_car_cond', track, car, condition, { globalStep: true });
    if (AVERAGED_STEPS.includes(source) || rows.length === 0) return null;
    return rows[0].g2_vs_lap_time ?? null;
  }

  // Multiplier on the typical heat input for a target lap time; mirrors
  // the Python predictor's _g2_pace_scale. Preferred: ratio along the
  // bucket's simulated-lap curve — the corner's own series when `corner`
  // is given and the curve carries g2_by_corner, else the corner mean
  // (anchored at lap_time_typ so target === typical scales by exactly 1).
  // Fallback: the physics exponent. Clamped either way.
  g2PaceScale(track, car, condition, lapTimeTypS, targetLapTimeS, corner = null) {
    const paceCfg = this.dto.g2_lap_time_model ?? {};
    const lo = paceCfg.multiplier_clamp?.min ?? 0.4;
    const hi = paceCfg.multiplier_clamp?.max ?? 2.5;

    const curve = this.lookupG2PaceCurve(track, car, condition);
    if (curve) {
      const byCorner = curve.g2_by_corner;
      const series = (corner && byCorner && Array.isArray(byCorner[corner]))
        ? byCorner[corner] : curve.g2;
      const ref = interpClamped(lapTimeTypS, curve.lap_time_s, series);
      if (ref > 0) {
        const scale = interpClamped(targetLapTimeS, curve.lap_time_s, series) / ref;
        return { scale: Math.min(hi, Math.max(lo, scale)), source: 'curve' };
      }
    }
    const exponent = paceCfg.default_exponent ?? 3.0;
    const scale = (lapTimeTypS / targetLapTimeS) ** exponent;
    return { scale: Math.min(hi, Math.max(lo, scale)), source: 'exponent' };
  }

  // Typical out-lap (pit exit to the first start/finish crossing): rolling
  // seconds and g², integrated first from the typed pit-exit temperature.
  // null when the artifact predates the table or has nothing for the track
  // (the out-lap is then zero-length, the pre-v0.26 behaviour). With a
  // `corner`, a v5 outlap_q_by_corner entry supplies that corner's heat
  // input instead of outlap_g2.
  lookupOutlap(track, car, condition, corner = null) {
    const g2 = (r) => perCornerOr(r.outlap_q_by_corner, corner, r.outlap_g2);
    const { rows, source } = this.answeringRows(
      'outlap_typ_by_track_car_cond', track, car, condition, { globalStep: false });
    if (rows.length === 0) return null;
    if (!AVERAGED_STEPS.includes(source)) {
      const hit = rows[0];
      return { movingS: hit.outlap_moving_s, g2: g2(hit), nLapsUsed: hit.n_laps_used, source };
    }
    return {
      movingS: average(rows, (r) => r.outlap_moving_s), g2: average(rows, g2),
      nLapsUsed: sum(rows, (r) => r.n_laps_used), source,
    };
  }

  lookupLapTime(track, car, condition) {
    const { rows, source } = this.answeringRows(
      'lap_time_typ_by_track_car_cond', track, car, condition, { globalStep: false });
    if (rows.length === 0) return { valueSeconds: 90.0, nLapsUsed: 0, source: 'global' };
    if (!AVERAGED_STEPS.includes(source)) {
      return { valueSeconds: rows[0].lap_time_typ_s, nLapsUsed: rows[0].n_laps_used, source };
    }
    return {
      valueSeconds: average(rows, (r) => r.lap_time_typ_s),
      nLapsUsed: sum(rows, (r) => r.n_laps_used), source,
    };
  }

  // How the simulated pace curve a prediction reads answers lapTimeS (the
  // target, else the typical lap time): 'no_solution' under its fastest
  // knot (answered there, the clamp), 'outside_support' outside its
  // `support` (the lap times of the car's logged range of use at the
  // layout) or past its slowest knot, else 'ok'; null where the row the pace
  // scale reads is not simulated (or is an average of rows). Display only.
  // Mirrors predict.py's pace_status.
  paceStatus(track, car, condition, lapTimeS) {
    const { rows, source } = this.answeringRows(
      'g2_typ_by_track_car_cond', track, car, condition, { globalStep: true });
    if (AVERAGED_STEPS.includes(source) || rowProvenance(rows) !== SIM) return null;
    const curve = rows[0].g2_vs_lap_time;
    if (!curve) return null;
    const xs = curve.lap_time_s;
    if (!(lapTimeS >= xs[0])) return PACE_NO_SOLUTION;
    const lo = curve.support?.lap_time_min_s;
    const hi = curve.support?.lap_time_max_s;
    if (lo === undefined || lo === null || hi === undefined || hi === null
        || lapTimeS > xs[xs.length - 1] || !(lo <= lapTimeS && lapTimeS <= hi)) {
      return PACE_OUTSIDE_SUPPORT;
    }
    return PACE_OK;
  }

  // A pooled car selected where its raw cars are offered in its place (a
  // saved "FJ" from before #383): it reads the pooled, logged entries, and
  // the UI hints to pick the raw car for its own.
  isPooledSelection(car) {
    return !!car && this.isFittedCar(car) && !this.availableCars.includes(car);
  }

}

// Per-corner cold-pressure prediction (port of CircuitPredictor.Predict /
// predict.py:predict_cold_pressure). Returns the recommended cold pressure
// plus the intermediate quantities.
export function predictCorner(model, {
  track, car, condition, lapWithinStint, ambientTempC,
  trackTempC = null, cloudCoverPct = null, corner,
  targetHotPressureBar, coldTireTempC = null, targetLapTimeS = null,
  compound = null, includeOutlap = true, outlapTimeS = null,
}) {
  const cond = condition.toLowerCase();
  if (cond !== 'dry' && cond !== 'damp' && cond !== 'wet') {
    throw new RangeError(`track_condition must be dry/damp/wet; got '${condition}'`);
  }

  // The constants are the car the model was fitted as; the entries (heat,
  // pace curve, lap time, out-lap) are a raw car's own where the artifact
  // has them (#383).
  const entryCar = car;
  car = model.resolveCar(car);

  // One tire on all four corners — the compound applies to every corner.
  let k = model.lookupK(car, corner, cond);
  if (compound) {
    const hit = model.lookupCompoundK(car, compound, corner, cond);
    if (hit) k = hit;
  }
  const tau = model.lookupTau(car, corner, cond);
  // The circuit enters only through its per-corner heat input (v5 has no
  // track constant); entries without the map fall back to the mean.
  const g2 = model.lookupG2(track, entryCar, cond, corner);
  const lap = model.lookupLapTime(track, entryCar, cond);

  // T_road: user-supplied -> sun-cover proxy -> fall back to T_air.
  const tRoadC = trackTempC ?? tRoadProxyC(
    ambientTempC, cloudCoverPct, model.sunFactorDefault, model.deltaSunMaxC);
  const tEffC = tEffectiveC(ambientTempC, tRoadC, model.wRoad);

  // "What's the tire at right now?" -- Gay-Lussac cold side AND the warmup
  // curve's initial condition (a rested tire sits at T_air, not T_eff).
  const tColdC = coldTireTempC ?? ambientTempC;
  const tStartC = tColdC;

  // Target-lap-time feature: pace sets both time-on-track and tire energy
  // (the heat multiplier is per corner: the simulated lap loads them differently).
  let g2Scale = 1.0;
  let g2PaceSource = null;
  let lapTimeForClockS = lap.valueSeconds;
  let g2Value = g2.value;
  if (targetLapTimeS !== null && targetLapTimeS !== undefined) {
    if (!(targetLapTimeS > 0)) {
      throw new RangeError(`target lap time must be > 0; got ${targetLapTimeS}`);
    }
    const pace = model.g2PaceScale(track, entryCar, cond, lap.valueSeconds, targetLapTimeS, corner);
    g2Scale = pace.scale;
    g2PaceSource = pace.source;
    g2Value *= g2Scale;
    lapTimeForClockS = targetLapTimeS;
  }

  // Time at the end of flying lap N; the out-lap is a separate segment.
  const tAtLapNs = lapWithinStint * lapTimeForClockS;
  let outlapS = 0.0;
  let outlapG2 = 0.0;
  let outlapSource = null;
  if (includeOutlap) {
    const o = model.lookupOutlap(track, entryCar, cond, corner);
    if (o) { outlapS = o.movingS; outlapG2 = o.g2; outlapSource = o.source; }
    if (outlapTimeS !== null && outlapTimeS !== undefined) {
      if (!(outlapTimeS >= 0)) throw new RangeError(`outlap time must be >= 0; got ${outlapTimeS}`);
      outlapS = outlapTimeS;
      outlapSource = `${outlapSource ?? 'none'}+override`;
    }
  }
  // Where the entry rows read come from: a simulated one carries GSI's
  // attribution (mirrors predict.py).
  const heatProvenance = rowProvenance(model.answeringRows(
    'g2_typ_by_track_car_cond', track, entryCar, cond, { globalStep: true }).rows);
  const lapTimeProvenance = rowProvenance(model.answeringRows(
    'lap_time_typ_by_track_car_cond', track, entryCar, cond, { globalStep: false }).rows);
  const outRows = includeOutlap ? model.answeringRows(
    'outlap_typ_by_track_car_cond', track, entryCar, cond, { globalStep: false }).rows : [];
  const outlapProvenance = outRows.length > 0 ? rowProvenance(outRows) : null;
  const usesSimulation = [heatProvenance, lapTimeProvenance, outlapProvenance].includes(SIM);
  const paceStatus = model.paceStatus(track, entryCar, cond, lapTimeForClockS);

  const warmupFrac = tau.valueSeconds > 0 ? 1 - Math.exp(-tAtLapNs / tau.valueSeconds) : 0;
  const deltaTInf = k.valueKelvinPerG2 * g2Value;
  const [tAfterOutlapC, tHotC] = warmupTwoStageC(
    outlapS, outlapG2, tAtLapNs, g2Value, tEffC, k.valueKelvinPerG2,
    tau.valueSeconds, tStartC);

  const coldPressureBar = gayLussacColdPressureBar(
    targetHotPressureBar, tHotC, tColdC, model.pAtmBar);

  return {
    corner,
    coldPressureBar,
    predictedHotTempC: tHotC,
    targetHotPressureBar,
    kKelvinPerG2: k.valueKelvinPerG2,
    tauSec: tau.valueSeconds,
    g2Typ: g2Value,
    lapTimeTypS: lap.valueSeconds,
    tAtLapNs,
    warmupFrac,
    deltaTInfKelvin: deltaTInf,
    tEffC,
    tAirC: ambientTempC,
    tRoadC,
    tColdC,
    tStartC,
    kSourceBucket: k.sourceBucket,
    kFromPrior: k.fromPrior,
    kNSamples: k.nSamples,
    targetLapTimeS,
    g2Scale,
    g2PaceSource,
    outlapTimeS: outlapS,
    outlapG2,
    outlapSource,
    tAfterOutlapC,
    heatProvenance,
    lapTimeProvenance,
    outlapProvenance,
    usesSimulation,
    paceStatus,
  };
}

// ---- Stint trajectory (Preview) ----

// Temperature and gauge pressure of one corner over a stint, from pit
// exit at the recommended cold pressure: what the gauge would read as
// the car runs. The temperature follows the same two-stage warm-up as
// predictCorner (out-lap, then flying laps at the target pace); the
// pressure follows it by Gay-Lussac from the cold set point, so it
// reaches the target hot pressure exactly at the end of the target lap
// and keeps rising past it as the tire keeps warming.
//
// `horizonLaps` flying laps are covered (default: five times the target
// lap, at least ten laps); `samplesPerLap` points per lap plus every lap
// boundary. Returns the underlying prediction, the clock lap time, the
// time at the end of each lap (index 0 = end of the out-lap), the target
// lap's time, and the sampled points.
export function predictCornerTrajectory(model, inputs, { horizonLaps = null, samplesPerLap = 20 } = {}) {
  const p = predictCorner(model, inputs);
  const lapS = p.targetLapTimeS ?? p.lapTimeTypS;
  const laps = horizonLaps ?? Math.max(10, inputs.lapWithinStint * 5);
  const totalS = p.outlapTimeS + laps * lapS;
  const pColdAbs = p.coldPressureBar + model.pAtmBar;
  const tColdK = p.tColdC + T_ZERO_C_TO_K;

  const tempAt = (t) => (t <= p.outlapTimeS
    ? warmupCurveC(t, p.tEffC, p.kKelvinPerG2, p.outlapG2, p.tauSec, p.tStartC)
    : warmupCurveC(t - p.outlapTimeS, p.tEffC, p.kKelvinPerG2, p.g2Typ, p.tauSec, p.tAfterOutlapC));
  const pressureAt = (tempC) => pColdAbs * (tempC + T_ZERO_C_TO_K) / tColdK - model.pAtmBar;

  const lapEndsS = [];
  for (let k = 0; k <= laps; k++) lapEndsS.push(p.outlapTimeS + k * lapS);

  const times = new Set([0, p.outlapTimeS, ...lapEndsS]);
  const step = lapS / samplesPerLap;
  for (let t = 0; t < totalS; t += step) times.add(t);
  const points = [...times].filter((t) => t <= totalS).sort((a, b) => a - b).map((t) => {
    const tempC = tempAt(t);
    return { t, tempC, pressureBar: pressureAt(tempC) };
  });

  return {
    prediction: p,
    lapTimeS: lapS,
    outlapS: p.outlapTimeS,
    lapEndsS,
    targetTimeS: p.outlapTimeS + inputs.lapWithinStint * lapS,
    totalS,
    points,
  };
}
