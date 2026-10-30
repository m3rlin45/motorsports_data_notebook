// Layout tests for the Preview chart builders (pure SVG markup, no DOM).

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { TireModel, predictCornerTrajectory } from '../js/model.js';
import {
  renderPressureChart, chartLayout, pressureRange, tickStep, AXIS_WIDTH, VISIBLE_LAP_FACTOR,
} from '../js/chart.js';

const modelDto = JSON.parse(readFileSync(
  new URL('../../../data/tire_dataset/tire_model.json', import.meta.url), 'utf8'));
const model = new TireModel(modelDto);

const trajectoryFor = (lapWithinStint, extra = {}) => predictCornerTrajectory(model, {
  track: model.availableTracks[0], car: model.availableCars[0], condition: 'dry',
  lapWithinStint, ambientTempC: 20, cloudCoverPct: 50, corner: 'fl',
  targetHotPressureBar: 1.8, coldTireTempC: 20, ...extra,
});

test('viewport shows the stint up to 1.5 × the target lap; the plot scrolls to the horizon', () => {
  const traj = trajectoryFor(6);
  const L = chartLayout(traj, { viewportWidth: 300, height: 160 });
  const visibleInner = 300 - L.margin.left - L.margin.right;
  assert.ok(Math.abs(L.pxPerS * traj.targetTimeS * VISIBLE_LAP_FACTOR - visibleInner) < 1e-6);
  // The plot reaches at least five times the target lap, so it scrolls
  // well past the 1.5 × window.
  assert.ok(traj.totalS >= traj.outlapS + 5 * 6 * traj.lapTimeS - 1e-9);
  assert.ok(L.plotWidth > 300 * 2.5, `plot wider than the viewport (${L.plotWidth})`);
  assert.ok(Math.abs(L.plotWidth - (L.margin.left + traj.totalS * L.pxPerS + L.margin.right)) < 1);
});

test('chart markup: target line, lap ticks, target lap marker, curve', () => {
  const traj = trajectoryFor(6);
  const out = renderPressureChart(traj, {
    targetBar: 1.8, lapWithinStint: 6, viewportWidth: 300, height: 160,
    labels: { lap: 'Lap', out: 'Out' },
  });
  assert.match(out.axisSvg, new RegExp(`width="${AXIS_WIDTH}"`));
  assert.match(out.axisSvg, /chart-target-label[^>]*>1\.80</);
  assert.match(out.plotSvg, /class="chart-target"/);
  assert.match(out.plotSvg, /class="chart-target-lap"/);
  assert.match(out.plotSvg, /class="chart-curve" d="M/);
  assert.match(out.plotSvg, /class="chart-target-dot"/);
  const lapTicks = out.plotSvg.match(/class="chart-lap-tick/g) ?? [];
  assert.equal(lapTicks.length, traj.lapEndsS.length - (traj.outlapS > 0 ? 0 : 1));
  // The target lap is always labelled, in the target colour.
  assert.match(out.plotSvg, /chart-target-label[^>]*>6</);
  if (traj.outlapS > 0) assert.match(out.plotSvg, />Out</);
  assert.match(out.plotSvg, />Lap</);
  const [lo, hi] = out.yRange;
  assert.ok(lo < traj.prediction.coldPressureBar && hi > 1.8);
});

test('lap labels thin out for long stints', () => {
  const traj = trajectoryFor(40);
  const out = renderPressureChart(traj, {
    targetBar: 1.8, lapWithinStint: 40, viewportWidth: 240, height: 160,
  });
  const labels = [...out.plotSvg.matchAll(/chart-tick[^>]*y="\d+(?:\.\d+)?" text-anchor="middle">(\d+)</g)]
    .map((m) => Number(m[1]));
  assert.ok(labels.includes(40));
  assert.ok(labels.length < 60, `labels thinned (${labels.length})`);
  const step = labels[1] - labels[0];
  assert.ok(step >= 2, `label step ${step}`);
});

test('pressureRange pads around data and target; tickStep spaces labels', () => {
  const traj = trajectoryFor(5);
  const [lo, hi] = pressureRange(traj, 1.8);
  const pMin = Math.min(...traj.points.map((p) => p.pressureBar));
  const pMax = Math.max(1.8, ...traj.points.map((p) => p.pressureBar));
  assert.ok(lo <= pMin && lo > pMin - 0.11);
  assert.ok(hi >= pMax && hi < pMax + 0.11);
  assert.equal(tickStep(1.0, 100, [0.05, 0.1, 0.2, 0.5], 22), 0.5);
  assert.equal(tickStep(1.0, 1000, [0.05, 0.1, 0.2, 0.5], 22), 0.05);
  assert.equal(tickStep(1, 10, [1, 2, 5, 10], 28), 5);
});
