// Pressure-over-stint chart for Preview mode: pure SVG markup builders, no
// DOM access, so the layout math is testable under node.
//
// A chart is two SVGs side by side: a fixed y axis (pressure in bar, the
// target tick in accent) and a scrollable plot whose width covers the
// whole trajectory (five times the target lap) while the viewport shows
// the stint up to 1.5 × the target lap. Lap boundaries are ticked along the bottom.

export const AXIS_WIDTH = 40;
const MARGIN = { top: 10, right: 10, bottom: 20, left: 4 };
export const VISIBLE_LAP_FACTOR = 1.5;

const fmt = (v, d) => v.toFixed(d);

// Y range: pad the data to the next 0.05 bar either side.
export function pressureRange(trajectory, targetBar) {
  let lo = targetBar;
  let hi = targetBar;
  for (const pt of trajectory.points) {
    lo = Math.min(lo, pt.pressureBar);
    hi = Math.max(hi, pt.pressureBar);
  }
  lo = Math.floor(lo * 20) / 20 - 0.05;
  hi = Math.ceil(hi * 20) / 20 + 0.05;
  if (hi - lo < 0.2) hi = lo + 0.2;
  return [lo, hi];
}

// Tick step so that labels sit at least `minPx` apart.
export function tickStep(rangeValue, rangePx, candidates, minPx) {
  for (const c of candidates) {
    if ((c / rangeValue) * rangePx >= minPx) return c;
  }
  return candidates[candidates.length - 1];
}

// Geometry shared by the axis and the plot. `viewportWidth` is the
// scroll container's inner width; the plot is wider by total / visible.
export function chartLayout(trajectory, { viewportWidth, height, visibleFactor = VISIBLE_LAP_FACTOR }) {
  const visibleS = Math.max(1, trajectory.targetTimeS * visibleFactor);
  const innerVisible = Math.max(1, viewportWidth - MARGIN.left - MARGIN.right);
  const pxPerS = innerVisible / visibleS;
  const plotWidth = Math.ceil(MARGIN.left + trajectory.totalS * pxPerS + MARGIN.right);
  const innerH = Math.max(1, height - MARGIN.top - MARGIN.bottom);
  return { pxPerS, plotWidth, height, innerH, margin: MARGIN };
}

export function renderPressureChart(trajectory, {
  targetBar, lapWithinStint, viewportWidth, height, labels = { lap: 'Lap', out: 'Out' },
}) {
  const L = chartLayout(trajectory, { viewportWidth, height });
  const [yLo, yHi] = pressureRange(trajectory, targetBar);
  const x = (t) => L.margin.left + t * L.pxPerS;
  const y = (bar) => L.margin.top + (yHi - bar) / (yHi - yLo) * L.innerH;

  // ---- y axis ----
  const yStep = tickStep(yHi - yLo, L.innerH, [0.05, 0.1, 0.2, 0.25, 0.5, 1], 22);
  const yTicks = [];
  for (let v = Math.ceil(yLo / yStep) * yStep; v <= yHi + 1e-9; v += yStep) {
    yTicks.push(Math.round(v * 1000) / 1000);
  }
  const axis = [];
  axis.push(`<line class="chart-axis-line" x1="${AXIS_WIDTH - 1}" y1="${L.margin.top}" x2="${AXIS_WIDTH - 1}" y2="${L.margin.top + L.innerH}"/>`);
  for (const v of yTicks) {
    // Skip a tick that would collide with the target label.
    if (Math.abs(y(v) - y(targetBar)) < 10) continue;
    axis.push(`<text class="chart-tick" x="${AXIS_WIDTH - 5}" y="${fmt(y(v), 1)}" text-anchor="end" dominant-baseline="middle">${fmt(v, 2)}</text>`);
  }
  axis.push(`<text class="chart-tick chart-target-label" x="${AXIS_WIDTH - 5}" y="${fmt(y(targetBar), 1)}" text-anchor="end" dominant-baseline="middle">${fmt(targetBar, 2)}</text>`);
  const axisSvg = `<svg class="chart-axis" width="${AXIS_WIDTH}" height="${L.height}" viewBox="0 0 ${AXIS_WIDTH} ${L.height}" aria-hidden="true">${axis.join('')}</svg>`;

  // ---- plot ----
  const plot = [];
  const yBottom = L.margin.top + L.innerH;
  for (const v of yTicks) {
    plot.push(`<line class="chart-grid" x1="${L.margin.left}" y1="${fmt(y(v), 1)}" x2="${L.plotWidth - L.margin.right}" y2="${fmt(y(v), 1)}"/>`);
  }
  // Out-lap band.
  if (trajectory.outlapS > 0) {
    plot.push(`<rect class="chart-outlap" x="${L.margin.left}" y="${L.margin.top}" width="${fmt(trajectory.outlapS * L.pxPerS, 1)}" height="${L.innerH}"/>`);
  }
  // Lap ticks: a mark at every lap end, a number every `labelEvery` laps.
  const lapPx = trajectory.lapTimeS * L.pxPerS;
  const labelEvery = tickStep(1, lapPx, [1, 2, 5, 10, 20], 28);
  trajectory.lapEndsS.forEach((tEnd, k) => {
    if (k === 0 && trajectory.outlapS <= 0) return;
    const major = k === 0 || k % labelEvery === 0 || k === lapWithinStint;
    plot.push(`<line class="chart-lap-tick${major ? ' major' : ''}" x1="${fmt(x(tEnd), 1)}" y1="${yBottom}" x2="${fmt(x(tEnd), 1)}" y2="${yBottom + (major ? 6 : 3)}"/>`);
    if (major) {
      const text = k === 0 ? labels.out : String(k);
      plot.push(`<text class="chart-tick${k === lapWithinStint ? ' chart-target-label' : ''}" x="${fmt(x(tEnd), 1)}" y="${yBottom + 16}" text-anchor="middle">${text}</text>`);
    }
  });
  plot.push(`<line class="chart-axis-line" x1="${L.margin.left}" y1="${yBottom}" x2="${L.plotWidth - L.margin.right}" y2="${yBottom}"/>`);
  // Target pressure (dashed) and target lap (dotted).
  plot.push(`<line class="chart-target" x1="${L.margin.left}" y1="${fmt(y(targetBar), 1)}" x2="${L.plotWidth - L.margin.right}" y2="${fmt(y(targetBar), 1)}"/>`);
  plot.push(`<line class="chart-target-lap" x1="${fmt(x(trajectory.targetTimeS), 1)}" y1="${L.margin.top}" x2="${fmt(x(trajectory.targetTimeS), 1)}" y2="${yBottom}"/>`);
  // The curve.
  const d = trajectory.points.map((pt, i) =>
    `${i === 0 ? 'M' : 'L'}${fmt(x(pt.t), 1)} ${fmt(y(pt.pressureBar), 1)}`).join('');
  plot.push(`<path class="chart-curve" d="${d}"/>`);
  plot.push(`<circle class="chart-target-dot" cx="${fmt(x(trajectory.targetTimeS), 1)}" cy="${fmt(y(targetBar), 1)}" r="3.5"/>`);
  // Axis caption at the far right so it never covers the stint.
  plot.push(`<text class="chart-tick chart-caption" x="${L.plotWidth - L.margin.right}" y="${L.margin.top + 10}" text-anchor="end">${labels.lap}</text>`);
  const plotSvg = `<svg class="chart-plot" width="${L.plotWidth}" height="${L.height}" viewBox="0 0 ${L.plotWidth} ${L.height}" aria-hidden="true">${plot.join('')}</svg>`;

  return { axisSvg, plotSvg, plotWidth: L.plotWidth, yRange: [yLo, yHi] };
}
