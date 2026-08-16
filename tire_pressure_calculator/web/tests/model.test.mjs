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
    }
  }
});

test('rejects unsupported schema versions', () => {
  assert.throws(() => new TireModel({ ...modelDto, schema_version: 1 }), /schema_version/);
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

  const c = model.lookupCTrack('no_such_track');
  assert.ok(c.fromPrior);
  assert.equal(c.value, modelDto.priors_when_no_fit.c_track);

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

test('warmup curve starts at T_eff and saturates at T_eff + K*c*g2', () => {
  const tEff = 15, k = 60, c = 1.0, g2 = 0.7, tau = 240;
  assert.ok(Math.abs(warmupCurveC(0, tEff, k, c, g2, tau) - tEff) < 1e-12);
  const nearInf = warmupCurveC(tau * 50, tEff, k, c, g2, tau);
  assert.ok(Math.abs(nearInf - (tEff + k * c * g2)) < 1e-6);
  assert.throws(() => warmupCurveC(10, tEff, k, c, g2, 0), RangeError);
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
