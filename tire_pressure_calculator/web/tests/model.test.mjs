// Model-logic tests for the web calculator. Run with:
//   node --test tire_pressure_calculator/web/tests/
//
// The parity suite pins the JS port against the same Python-generated
// fixture the C# tests use (Tests/Fixtures/python_predictions.json),
// evaluated against the committed production model artifact.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  TireModel, predictCorner, conditionChain,
  gayLussacColdPressureBar, tRoadProxyC, tEffectiveC, warmupCurveC,
  adjustedHotTempC, cornerColdPressureBar, roundTo,
} from '../js/model.js';

const readJson = (relPath) =>
  JSON.parse(readFileSync(new URL(relPath, import.meta.url), 'utf8'));

const modelDto = readJson('../../../data/tire_dataset/tire_model.json');
const fixture = readJson('../../Tests/Fixtures/python_predictions.json');

test('parity with Python predictor fixture', () => {
  const model = new TireModel(modelDto);
  for (const testCase of fixture) {
    const inputs = testCase.inputs;
    for (const [corner, expected] of Object.entries(testCase.corners)) {
      const prediction = predictCorner(model, {
        track: inputs.track,
        car: inputs.car,
        condition: inputs.track_condition,
        lapWithinStint: inputs.lap_within_stint,
        ambientTempC: inputs.ambient_temp_c,
        cloudCoverPct: inputs.cloud_cover_pct ?? null,
        corner,
        targetHotPressureBar: inputs.target_hot_pressure_bar,
        coldTireTempC: inputs.cold_tire_temp_c ?? null,
        targetLapTimeS: inputs.target_lap_time_s ?? null,
        compound: inputs.compound ?? null,
      });
      const label = `${testCase.label}/${corner}`;
      assert.ok(
        Math.abs(prediction.coldPressureBar - expected.cold_pressure_bar) < 1e-3,
        `${label}: cold ${prediction.coldPressureBar} != ${expected.cold_pressure_bar}`);
      assert.ok(
        Math.abs(prediction.predictedHotTempC - expected.predicted_hot_temp_c) < 1e-2,
        `${label}: hot ${prediction.predictedHotTempC} != ${expected.predicted_hot_temp_c}`);
      assert.equal(
        prediction.kSourceBucket,
        `(${expected.K_source_bucket.join(', ')})`,
        `${label}: K source bucket`);
      if (testCase.g2_scale !== undefined) {
        assert.ok(
          Math.abs(prediction.g2Scale - testCase.g2_scale) < 1e-9,
          `${label}: g2 scale ${prediction.g2Scale} != ${testCase.g2_scale}`);
        assert.equal(prediction.g2PaceSource, testCase.g2_pace_source, `${label}: pace source`);
      }
    }
  }
});

test('rejects unsupported schema versions', () => {
  // v4 and below carried K values fitted against the per-track c_track,
  // which v5 dropped; only v5 loads.
  for (const v of [1, 2, 3, 4, 6]) {
    assert.throws(() => new TireModel({ ...modelDto, schema_version: v }), /schema_version/, `schema v${v}`);
  }
  assert.doesNotThrow(() => new TireModel({ ...modelDto, schema_version: 5 }));
});

test('legacy c_track keys in a v5 artifact are ignored', () => {
  // An artifact written by an older trainer may still carry the keys (all
  // 1.0); the model never reads them.
  const withLegacy = {
    ...modelDto, schema_version: 5,
    priors_when_no_fit: { ...modelDto.priors_when_no_fit, c_track: 1.0 },
    c_track_by_track: [{ track_canonical: 'synth', value: 1.0, stderr: 0, n_buckets_used: 4, anchor: true }],
  };
  const model = new TireModel(withLegacy);
  assert.ok(!model.availableTracks.includes('synth'), 'c_track_by_track no longer feeds the track list');
  assert.equal(typeof model.lookupCTrack, 'undefined');
});

// Tiny synthetic v5 artifact: one car, one track, with per-corner heat
// inputs on the exact (track, car, dry) entries. Every corner shares the
// same K/tau so any hot-temp difference comes from q_typ_by_corner /
// outlap_q_by_corner alone.
function syntheticV5Model({ withPerCorner = true } = {}) {
  const corners = ['fl', 'fr', 'rl', 'rr'];
  const g2Entry = {
    track_canonical: 'synth', car: 'Car', condition: 'dry', g2_typ: 1.0, n_laps_used: 10,
    g2_vs_lap_time: { lap_time_s: [50, 60, 70], g2: [1.4, 1.0, 0.7], n_laps: 10 },
  };
  const outlapEntry = {
    track_canonical: 'synth', car: 'Car', condition: 'dry',
    outlap_moving_s: 90, outlap_g2: 0.4, n_laps_used: 10,
  };
  if (withPerCorner) {
    g2Entry.q_typ_by_corner = { fl: 1.3, fr: 0.9, rl: 1.1, rr: 0.7 };
    outlapEntry.outlap_q_by_corner = { fl: 0.6, fr: 0.3, rl: 0.5, rr: 0.2 };
  }
  return {
    schema_version: 5,
    fit_at_utc: '2026-10-08T00:00:00Z',
    model_form: 'synthetic',
    gay_lussac: { p_atm_bar: 1.0, t_zero_c_to_k: 273.15, t_cold_uses: 'T_air' },
    energy_balance: {
      w_road: 0.2, w_road_fitted: false,
      t_road_proxy: { formula: 'x', delta_sun_max_c: 10.0, sun_factor_default: 1.0 },
    },
    conditions: { values: ['dry', 'damp', 'wet'], default: 'dry' },
    corners,
    min_samples_per_bucket: 5,
    priors_when_no_fit: { tau_sec_seconds: 240.0, K_kelvin_per_g2: 60.0 },
    tau_sec_by_car_corner_cond: corners.map((corner) => ({
      car: 'Car', corner, condition: 'dry', value_seconds: 300.0, stderr_seconds: 0,
      n_samples_used: 10, from_prior: false,
    })),
    K_buckets: corners.map((corner) => ({
      key: { car: 'Car', corner, condition: 'dry' }, value_kelvin_per_g2: 40.0,
      stderr_kelvin_per_g2: 0, n_samples: 10, from_prior: false, from_single_track: false,
    })),
    g2_typ_by_track_car_cond: [g2Entry],
    lap_time_typ_by_track_car_cond: [
      { track_canonical: 'synth', car: 'Car', condition: 'dry', lap_time_typ_s: 60.0, n_laps_used: 10 },
    ],
    outlap_typ_by_track_car_cond: [outlapEntry],
    g2_lap_time_model: {
      method: 'sector_curve', formula: 'x', default_exponent: 3.0,
      multiplier_clamp: { min: 0.4, max: 2.5 },
    },
    // Informational v5 block; the predictor must tolerate it untouched.
    heat_input: {
      form: 'q = ...', v_ref_ms: 30.0,
      geometry_by_car: { Car: { wheelbase_m: 2.5 } },
      drive_by_car: { Car: 'rwd' }, brake_by_car: { Car: { front_bias: 0.6 } },
    },
  };
}

test('schema v5: per-corner q_typ / outlap_q drive the corner prediction', () => {
  const model = new TireModel(syntheticV5Model());
  const args = {
    track: 'synth', car: 'Car', condition: 'dry', lapWithinStint: 5,
    ambientTempC: 20, cloudCoverPct: 100, targetHotPressureBar: 1.8,
  };
  const fl = predictCorner(model, { ...args, corner: 'fl' });
  const rr = predictCorner(model, { ...args, corner: 'rr' });
  // Same K and tau: only the per-corner heat input differs.
  assert.equal(fl.kKelvinPerG2, rr.kKelvinPerG2);
  assert.equal(fl.tauSec, rr.tauSec);
  assert.equal(fl.g2Typ, 1.3);
  assert.equal(rr.g2Typ, 0.7);
  assert.equal(fl.outlapG2, 0.6);
  assert.equal(rr.outlapG2, 0.2);
  assert.ok(fl.predictedHotTempC > rr.predictedHotTempC, 'FL (hotter corner) ends hotter');
  assert.ok(fl.coldPressureBar < rr.coldPressureBar);

  // The lookups themselves resolve the corner, and keep the mean without one.
  assert.equal(model.lookupG2('synth', 'Car', 'dry', 'rl').value, 1.1);
  assert.equal(model.lookupG2('synth', 'Car', 'dry').value, 1.0);
  assert.equal(model.lookupOutlap('synth', 'Car', 'dry', 'rl').g2, 0.5);
  assert.equal(model.lookupOutlap('synth', 'Car', 'dry').g2, 0.4);
  // Pooled fallbacks (condition chain exhausted) average the per-corner value.
  assert.equal(model.lookupG2('synth', 'Car', 'wet', 'fr').value, 0.9);
  assert.equal(model.lookupG2('synth', 'Car', 'wet', 'fr').source, 'fallback(dry)');

  // The pace multiplier applies to the per-corner value exactly as to g2_typ.
  const flFast = predictCorner(model, { ...args, corner: 'fl', targetLapTimeS: 55 });
  assert.equal(flFast.g2PaceSource, 'curve');
  assert.ok(Math.abs(flFast.g2Typ - 1.3 * flFast.g2Scale) < 1e-12);
  assert.ok(flFast.g2Scale > 1);
});

test('schema v5: entries without per-corner fields fall back to g2_typ / outlap_g2', () => {
  const model = new TireModel(syntheticV5Model({ withPerCorner: false }));
  const args = {
    track: 'synth', car: 'Car', condition: 'dry', lapWithinStint: 5,
    ambientTempC: 20, cloudCoverPct: 100, targetHotPressureBar: 1.8,
  };
  const fl = predictCorner(model, { ...args, corner: 'fl' });
  const rr = predictCorner(model, { ...args, corner: 'rr' });
  assert.equal(fl.g2Typ, 1.0);
  assert.equal(rr.g2Typ, 1.0);
  assert.equal(fl.outlapG2, 0.4);
  assert.equal(fl.predictedHotTempC, rr.predictedHotTempC);
  assert.equal(fl.coldPressureBar, rr.coldPressureBar);
  // Same numbers as the corner-less lookup (the pre-v5 call shape).
  assert.equal(model.lookupG2('synth', 'Car', 'dry', 'fl').value,
    model.lookupG2('synth', 'Car', 'dry').value);
  // A map missing the asked corner also keeps the mean.
  const partial = syntheticV5Model();
  partial.g2_typ_by_track_car_cond[0].q_typ_by_corner = { fl: 1.3 };
  assert.equal(new TireModel(partial).lookupG2('synth', 'Car', 'dry', 'rr').value, 1.0);
});

test('bundled artifact: corner-aware lookups match the corner-less ones when no per-corner fields', () => {
  // The committed artifact may or may not carry the v5 fields; when it does
  // not, passing a corner must be a no-op (pre-v5 numbers are preserved).
  const model = new TireModel(modelDto);
  const hasPerCorner = modelDto.g2_typ_by_track_car_cond.some((r) => r.q_typ_by_corner);
  if (hasPerCorner) return;
  for (const track of model.availableTracks) {
    for (const car of model.availableCars) {
      for (const corner of ['fl', 'fr', 'rl', 'rr']) {
        assert.equal(model.lookupG2(track, car, 'dry', corner).value,
          model.lookupG2(track, car, 'dry').value, `${track}/${car}/${corner}`);
        assert.deepEqual(model.lookupOutlap(track, car, 'dry', corner),
          model.lookupOutlap(track, car, 'dry'), `${track}/${car}/${corner} outlap`);
      }
    }
  }
});

test('available tracks and cars are sorted and non-empty', () => {
  const model = new TireModel(modelDto);
  assert.ok(model.availableTracks.length > 0);
  assert.ok(model.availableCars.length > 0);
  assert.deepEqual(model.availableTracks, [...model.availableTracks].sort());
  assert.deepEqual(model.availableCars, [...model.availableCars].sort());
});

test('condition fallback chain mirrors predict.py', () => {
  assert.deepEqual(conditionChain('dry'), ['dry']);
  assert.deepEqual(conditionChain('damp'), ['damp', 'dry']);
  assert.deepEqual(conditionChain('wet'), ['wet', 'damp', 'dry']);
  assert.deepEqual(conditionChain('snow'), ['snow', 'dry']);
});

test('predictCorner rejects unknown conditions', () => {
  const model = new TireModel(modelDto);
  assert.throws(() => predictCorner(model, {
    track: model.availableTracks[0], car: model.availableCars[0],
    condition: 'snow', lapWithinStint: 5, ambientTempC: 20,
    corner: 'fl', targetHotPressureBar: 1.8,
  }), /dry\/damp\/wet/);
});

test('unknown track/car fall back to priors and pooled values', () => {
  const model = new TireModel(modelDto);
  const k = model.lookupK('NoSuchCar', 'fl', 'dry');
  assert.equal(k.sourceBucket, '(prior)');
  assert.equal(k.valueKelvinPerG2, modelDto.priors_when_no_fit.K_kelvin_per_g2);
  assert.ok(k.fromPrior);

  const tau = model.lookupTau('NoSuchCar', 'fl', 'dry');
  assert.equal(tau.sourceBucket, '(prior)');
  assert.equal(tau.valueSeconds, modelDto.priors_when_no_fit.tau_sec_seconds);

  const g2 = model.lookupG2('no_such_track', 'NoSuchCar', 'dry');
  assert.equal(g2.source, 'global');
  const lap = model.lookupLapTime('no_such_track', 'NoSuchCar', 'dry');
  assert.equal(lap.source, 'global');
});

test('gay-lussac inversion round-trips', () => {
  // Set cold at 20 °C so the tire reads 1.8 bar gauge at 80 °C.
  const cold = gayLussacColdPressureBar(1.8, 80, 20);
  const tColdK = 20 + 273.15;
  const tHotK = 80 + 273.15;
  const hotAbs = (cold + 1.0) * (tHotK / tColdK);
  assert.ok(Math.abs(hotAbs - 1.0 - 1.8) < 1e-12);
  // Equal temperatures -> cold equals target.
  assert.ok(Math.abs(gayLussacColdPressureBar(1.8, 25, 25) - 1.8) < 1e-12);
  assert.throws(() => gayLussacColdPressureBar(1.8, -300, 20), RangeError);
});

test('t_road proxy clamps cloud cover and passes null through', () => {
  assert.equal(tRoadProxyC(20, null), 20);
  assert.equal(tRoadProxyC(20, 0), 30);   // full sun: +delta_sun_max_c
  assert.equal(tRoadProxyC(20, 100), 20); // overcast: T_air
  assert.equal(tRoadProxyC(20, 150), 20); // clamped
  assert.equal(tRoadProxyC(20, -10), 30); // clamped
});

test('effective temperature blends air and road by w_road', () => {
  assert.equal(tEffectiveC(10, 30, 0.2), 14);
  assert.throws(() => tEffectiveC(10, 30, 1.5), RangeError);
});

test('warmup curve starts at T_eff and saturates at T_eff + K*q', () => {
  const tEff = 15, k = 60, g2 = 0.7, tau = 240;
  assert.ok(Math.abs(warmupCurveC(0, tEff, k, g2, tau) - tEff) < 1e-12);
  const nearInf = warmupCurveC(tau * 50, tEff, k, g2, tau);
  assert.ok(Math.abs(nearInf - (tEff + k * g2)) < 1e-6);
  assert.throws(() => warmupCurveC(10, tEff, k, g2, 0), RangeError);
});

test('warmup curve starts at the given tire temperature and forgets it with tau', () => {
  const tEff = 15, k = 60, g2 = 0.7, tau = 240, tStart = 45;
  assert.ok(Math.abs(warmupCurveC(0, tEff, k, g2, tau, tStart) - tStart) < 1e-12);
  const asymptote = tEff + k * g2;
  const atTau = warmupCurveC(tau, tEff, k, g2, tau, tStart);
  assert.ok(Math.abs(atTau - (asymptote + (tStart - asymptote) * Math.exp(-1))) < 1e-9);
  assert.ok(Math.abs(warmupCurveC(tau * 50, tEff, k, g2, tau, tStart) - asymptote) < 1e-6);
  // null start reproduces the T_eff start.
  assert.equal(warmupCurveC(100, tEff, k, g2, tau, null), warmupCurveC(100, tEff, k, g2, tau));
});

test('current tire temp moves the predicted hot temperature by the decayed start excess', () => {
  const model = new TireModel(modelDto);
  const common = {
    track: 'tsukuba_2000', car: 'KK-SII', condition: 'dry', lapWithinStint: 5,
    ambientTempC: 15.0, cloudCoverPct: 100.0, corner: 'fl', targetHotPressureBar: 1.7,
  };
  const base = predictCorner(model, common);
  const warm = predictCorner(model, { ...common, coldTireTempC: 25.0 });
  assert.equal(base.tStartC, 15.0);
  assert.equal(warm.tStartC, 25.0);
  // The start excess decays over the out-lap segment plus the flying laps.
  const decay = Math.exp(-(base.outlapTimeS + base.tAtLapNs) / base.tauSec);
  assert.ok(Math.abs((warm.predictedHotTempC - base.predictedHotTempC) - 10 * decay) < 1e-9);
  assert.ok(warm.coldPressureBar > base.coldPressureBar);
});

test('manual-mode adjusted hot temp matches TireCornerViewModel', () => {
  // Adjustment is applied in Kelvin space, rounded to 0.1 °C.
  assert.equal(adjustedHotTempC(80, 0), 80);
  const expected = (80 + 273.15) * 1.05 - 273.15;
  assert.ok(Math.abs(adjustedHotTempC(80, 5) - roundTo(expected, 1)) < 1e-12);
});

test('manual-mode cold pressure matches known values', () => {
  // 20 °C current, 80 °C hot, 1.80 bar target — the app's default state.
  const expected = (1.8 + 1.0) * ((20 + 273.15) / (80 + 273.15)) - 1.0;
  assert.equal(cornerColdPressureBar(1.8, 80, 20), roundTo(expected, 3));
  // Non-physical temperatures guard to 0 instead of throwing.
  assert.equal(cornerColdPressureBar(1.8, -280, 20), 0);
});

test('roundTo uses banker\'s rounding like C# Math.Round', () => {
  assert.equal(roundTo(0.5, 0), 0);
  assert.equal(roundTo(1.5, 0), 2);
  assert.equal(roundTo(2.5, 0), 2);
  assert.equal(roundTo(-2.5, 0), -2);
  // Values not exactly representable round by their true double value,
  // same as .NET Core's Math.Round.
  assert.equal(roundTo(1.2345, 3), 1.234); // stored as 1.23449999…
  assert.equal(roundTo(80.05, 1), 80);     // stored as 80.04999…
  assert.equal(roundTo(80.15, 1), 80.2);   // stored as 80.15000000000000568…
});

test('interpClamped: linear inside, clamped outside', async () => {
  const { interpClamped } = await import('../js/model.js');
  const xs = [55, 60, 65];
  const ys = [1.2, 0.9, 0.6];
  assert.equal(interpClamped(60, xs, ys), 0.9);
  assert.ok(Math.abs(interpClamped(57.5, xs, ys) - 1.05) < 1e-12);
  assert.equal(interpClamped(40, xs, ys), 1.2);  // clamp low
  assert.equal(interpClamped(80, xs, ys), 0.6);  // clamp high
});

test('car aliases resolve raw names to the pooled label', () => {
  const model = new TireModel(modelDto);
  assert.equal(model.resolveCar('KK-F'), 'FJ');
  assert.equal(model.resolveCar('KK-SII'), 'FJ');
  assert.equal(model.resolveCar('FJ'), 'FJ');
  assert.equal(model.resolveCar('Inferno 86'), 'Inferno 86');
});

test('g2PaceScale: curve ratio anchored at typical, exponent fallback, clamps', () => {
  const model = new TireModel(modelDto);
  const track = model.availableTracks.find(
    (t) => model.lookupG2PaceCurve(t, 'FJ', 'dry') !== null);
  assert.ok(track, 'expected at least one curve-covered FJ bucket');
  const lapTyp = model.lookupLapTime(track, 'FJ', 'dry').valueSeconds;
  const atTyp = model.g2PaceScale(track, 'FJ', 'dry', lapTyp, lapTyp);
  assert.equal(atTyp.source, 'curve');
  assert.ok(Math.abs(atTyp.scale - 1.0) < 1e-12);
  const faster = model.g2PaceScale(track, 'FJ', 'dry', lapTyp, lapTyp * 0.95);
  assert.ok(faster.scale > 1.0);
  // Unknown bucket -> exponent fallback, extreme target hits the clamp
  const fb = model.g2PaceScale('no_such_track', 'NoSuchCar', 'dry', 100, 10);
  assert.equal(fb.source, 'exponent');
  assert.equal(fb.scale, modelDto.g2_lap_time_model.multiplier_clamp.max);
});

test('target lap time drives time-on-track and rejects non-positive values', () => {
  const model = new TireModel(modelDto);
  const args = {
    track: model.availableTracks[0], car: model.availableCars[0],
    condition: 'dry', lapWithinStint: 5, ambientTempC: 20,
    corner: 'fl', targetHotPressureBar: 1.8,
  };
  const withTarget = predictCorner(model, { ...args, targetLapTimeS: 61.5 });
  assert.ok(Math.abs(withTarget.tAtLapNs - 5 * 61.5) < 1e-9);
  const without = predictCorner(model, args);
  assert.equal(without.g2Scale, 1.0);
  assert.equal(without.g2PaceSource, null);
  assert.throws(() => predictCorner(model, { ...args, targetLapTimeS: 0 }), RangeError);
});

test('compound K overrides pooled K per axle', () => {
  const model = new TireModel(modelDto);
  const compounds = model.availableCompounds('Inferno 86');
  assert.ok(compounds.includes('A052') && compounds.includes('RE-71RS'));
  const args = {
    track: 'sodegaura', car: 'Inferno 86', condition: 'dry',
    lapWithinStint: 5, ambientTempC: 22, corner: 'rr', targetHotPressureBar: 1.9,
  };
  const pooled = predictCorner(model, args);
  const a052 = predictCorner(model, { ...args, compound: 'A052' });
  const rs71 = predictCorner(model, { ...args, compound: 'RE-71RS' });
  assert.notEqual(a052.kKelvinPerG2, pooled.kKelvinPerG2);
  assert.ok(rs71.kKelvinPerG2 > a052.kKelvinPerG2, '71RS hotter than A052');
  assert.ok(a052.predictedHotTempC < rs71.predictedHotTempC);
  assert.ok(a052.coldPressureBar > rs71.coldPressureBar);
  // One tire on all four corners: a front corner moves too.
  const front = predictCorner(model, { ...args, corner: 'fl', compound: 'A052' });
  assert.notEqual(front.kKelvinPerG2, predictCorner(model, { ...args, corner: 'fl' }).kKelvinPerG2);
  // Unknown compound falls back to pooled.
  const unknown = predictCorner(model, { ...args, compound: 'SLICKS9000' });
  assert.equal(unknown.kKelvinPerG2, pooled.kKelvinPerG2);
});

test('corner defaults: per-car steady-state medians with condition fallback', () => {
  const model = new TireModel(modelDto);
  const kk = model.lookupCornerDefaults('FJ', 'fl', 'dry');
  const inferno = model.lookupCornerDefaults('Inferno 86', 'fl', 'dry');
  assert.ok(kk && inferno, 'both cars carry dry FL defaults');
  assert.ok(inferno.hotTempC > kk.hotTempC, 'Inferno runs hotter than FJ');
  assert.ok(kk.hotTempC > 20 && kk.hotTempC < 100);
  assert.ok(kk.hotPressureBar > 1.0 && kk.hotPressureBar < 3.0);
  assert.equal(kk.source, 'exact');
  // Wet FJ data exists and is cooler than dry.
  const wet = model.lookupCornerDefaults('FJ', 'fl', 'wet');
  assert.ok(wet.hotTempC < kk.hotTempC);
  // Inferno has no wet laps: the condition chain falls back toward dry.
  const infernoWet = model.lookupCornerDefaults('Inferno 86', 'fl', 'wet');
  // Inferno wet crossed the 5-lap minimum (6 laps), so it may now be exact.
  assert.ok(infernoWet && (infernoWet.source === 'exact' || infernoWet.source.startsWith('fallback(')));
  // Unknown car -> null (caller keeps its static defaults).
  assert.equal(model.lookupCornerDefaults('NoSuchCar', 'fl', 'dry'), null);
});

test('dataThrough prefers the local time, then the date, then the fit date', () => {
  const model = new TireModel(modelDto);
  assert.equal(model.dataThrough, modelDto.data_through_local);
  assert.match(model.dataThrough, /^\d{4}-\d{2}-\d{2} \d{2}:\d{2} /);
  const dateOnly = new TireModel({ ...modelDto, data_through_local: undefined });
  assert.equal(dateOnly.dataThrough, modelDto.data_through_date);
  const legacy = new TireModel({ ...modelDto, data_through_local: undefined, data_through_date: undefined });
  assert.equal(legacy.dataThrough, modelDto.fit_at_utc.slice(0, 10));
  const bare = new TireModel({ ...legacy.dto, fit_at_utc: undefined });
  assert.equal(bare.dataThrough, null);
});
