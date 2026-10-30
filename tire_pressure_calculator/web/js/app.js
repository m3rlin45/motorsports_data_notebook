// UI wiring for the tire pressure calculator web app. All model math lives
// in model.js; this file owns DOM, settings persistence, and localization.

import {
  TireModel, predictCorner, predictCornerTrajectory, adjustedHotTempC, cornerColdPressureBar,
} from './model.js';
import { renderPressureChart } from './chart.js';
import { STRINGS, resolveLanguage } from './strings.js';

// Same key AND JSON shape as the .NET heads (C# AppSettings serialized with
// PascalCase properties, Mode as 0/1) so settings saved by the old WASM
// build carry over, and stay portable with Desktop/Android exports.
const STORAGE_KEY = 'tire-pressure-calculator/settings';

// Stamped by the deploy workflow (.github/workflows/build-tire-pressure-web.yml)
// from pyproject.toml; the repo copy keeps the placeholder, which renders
// as "dev" locally.
const APP_VERSION = '__VERSION__';

const MODE_MANUAL = 0;
const MODE_PREDICTION = 1;

const CORNERS = [
  { id: 'fl', label: 'FL', settingsKey: 'FrontLeft' },
  { id: 'fr', label: 'FR', settingsKey: 'FrontRight' },
  { id: 'rl', label: 'RL', settingsKey: 'RearLeft' },
  { id: 'rr', label: 'RR', settingsKey: 'RearRight' },
];

const CONDITIONS = [
  { value: 'dry', key: 'ConditionDry' },
  { value: 'damp', key: 'ConditionDamp' },
  { value: 'wet', key: 'ConditionWet' },
];

// ---- Settings ----

function defaultCorner() {
  return { CurrentTemp: 20.0, TargetHotTemp: 80.0, TargetHotPressure: 1.8 };
}

function defaultSettings() {
  return {
    FrontLeft: defaultCorner(),
    FrontRight: defaultCorner(),
    RearLeft: defaultCorner(),
    RearRight: defaultCorner(),
    TempAdjustPercent: 0,
    Mode: MODE_PREDICTION, // falls back to manual when the model cannot be loaded
    Prediction: {
      Track: null,
      Car: null,
      Condition: 'dry',
      LapWithinStint: 5,
      AmbientTempC: 20.0,
      TrackTempC: null,
      CloudCoverPct: 50,
      TargetLapTimeS: null, // snapped to the selection's typical lap once the model loads
      Compound: null,
    },
    UiLanguage: 'auto',
    // Corner targets remembered per configuration ("car|compound|condition"
    // -> { fl: { TargetHotTemp, TargetHotPressure }, ... }). A configuration
    // is prefilled from the model / team table the first time it is opened
    // and after Reset; otherwise the targets the user left for it come back.
    TargetMemory: {},
  };
}

// True when settings came from storage (vs factory defaults). First run
// prefills the corner targets from the model; a returning user's values
// for the active configuration are left alone.
let hadSavedSettings = false;

function loadSettings() {
  const defaults = defaultSettings();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaults;
    hadSavedSettings = true;
    const parsed = JSON.parse(raw);
    // Shallow-merge each section over the defaults so partial/older payloads
    // still produce a complete settings object.
    for (const c of CORNERS) {
      Object.assign(defaults[c.settingsKey], parsed[c.settingsKey] ?? {});
    }
    Object.assign(defaults.Prediction, parsed.Prediction ?? {});
    if (typeof parsed.TempAdjustPercent === 'number') defaults.TempAdjustPercent = parsed.TempAdjustPercent;
    if (parsed.Mode === MODE_MANUAL || parsed.Mode === MODE_PREDICTION) defaults.Mode = parsed.Mode;
    if (typeof parsed.UiLanguage === 'string') defaults.UiLanguage = parsed.UiLanguage;
    if (parsed.TargetMemory && typeof parsed.TargetMemory === 'object') {
      defaults.TargetMemory = parsed.TargetMemory;
    }
    return defaults;
  } catch {
    return defaults;
  }
}

function saveSettings() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings, null, 2));
  } catch { /* private-mode storage failures are non-fatal */ }
}

// ---- State ----

const settings = loadSettings();
let model = null; // TireModel | null
let modelLoadSettled = false; // suppress the "unavailable" warning while fetching
let lang = resolveLanguage(settings.UiLanguage, navigator.language);
const predictions = { fl: null, fr: null, rl: null, rr: null }; // predicted hot °C
// Preview: the corner cards show pressure-over-stint charts instead of
// their fields. A view state, not a setting (never persisted); only
// meaningful in Circuit Prediction mode.
let preview = false;

const t = (key) => STRINGS[lang][key] ?? STRINGS.en[key] ?? key;

const isPredictionMode = () => settings.Mode === MODE_PREDICTION && model !== null;
const isPreview = () => preview && isPredictionMode();

// ---- DOM handles ----

const $ = (id) => document.getElementById(id);
const els = {
  modeToggle: $('modeToggle'), modeContent: $('modeContent'),
  modelUnavailable: $('modelUnavailable'), previewBtn: $('previewBtn'),
  langLabel: $('langLabel'), langSelect: $('langSelect'),
  cornerGrid: $('cornerGrid'),
  resetBtn: $('resetBtn'),
  adjustGroup: $('adjustGroup'), adjustLabel: $('adjustLabel'), adjustSlider: $('adjustSlider'),
  predPanel: $('predPanel'),
  trackLabel: $('trackLabel'), trackSelect: $('trackSelect'),
  carLabel: $('carLabel'), carSelect: $('carSelect'),
  conditionLabel: $('conditionLabel'), conditionSelect: $('conditionSelect'),
  lapLabel: $('lapLabel'), lapInput: $('lapInput'),
  ambientLabel: $('ambientLabel'), ambientInput: $('ambientInput'),
  cloudLabel: $('cloudLabel'), cloudInput: $('cloudInput'),
  targetLapLabel: $('targetLapLabel'), targetLapInput: $('targetLapInput'),
  compoundLabel: $('compoundLabel'), compoundSelect: $('compoundSelect'),
  lastUpdated: $('lastUpdated'),
};

// "Model data: 2026-10-02 14:23 JST · v1.1.1": the newest session in the
// model and the deployed app version, so a cached PWA can be told apart
// from the latest deploy at a glance.
function appVersionLabel() {
  return APP_VERSION.startsWith('__') ? 'dev' : `v${APP_VERSION}`;
}

function renderLastUpdated() {
  const data = t('LastUpdatedFormat').replace('{0}', model?.dataThrough ?? '—');
  els.lastUpdated.textContent = `${data} · ${appVersionLabel()}`;
}

// ---- Corner cards ----

function buildCornerCards() {
  for (const c of CORNERS) {
    const section = document.createElement('section');
    section.className = 'corner';
    section.dataset.corner = c.id;
    section.innerHTML = `
      <h2>${c.label}</h2>
      <div class="fields">
        <div class="field f-current">
          <label data-str="CurrentTemp"></label>
          <div class="box box-current"><input class="in-current" inputmode="decimal" autocomplete="off" enterkeyhint="next"></div>
        </div>
        <div class="field f-target manual-only">
          <label data-str="TargetTempCorr"></label>
          <div class="box box-target"><input class="in-target" inputmode="decimal" autocomplete="off" enterkeyhint="next"></div>
        </div>
        <div class="field f-pred pred-only">
          <label data-str="PredictedHotTemp"></label>
          <div class="box box-pred out-pred">—</div>
        </div>
        <div class="field f-bar">
          <label data-str="TargetBar"></label>
          <div class="box box-target"><input class="in-bar" inputmode="decimal" autocomplete="off" enterkeyhint="done"></div>
        </div>
      </div>
      <div class="corner-chart preview-only">
        <div class="chart-axis-host"></div>
        <div class="chart-scroll"><div class="chart-plot-host"></div></div>
      </div>
      <hr>
      <div class="setcold">
        <span class="setcold-label" data-str="SetCold"></span>
        <span class="setcold-value">—</span>
      </div>`;
    els.cornerGrid.appendChild(section);
    chartResizeObserver?.observe(section.querySelector('.chart-scroll'));

    const cs = settings[c.settingsKey];
    bindNumberInput(section.querySelector('.in-current'), {
      min: -40, max: 60, decimals: 1,
      get: () => cs.CurrentTemp,
      set: (v) => { cs.CurrentTemp = v; },
    });
    bindNumberInput(section.querySelector('.in-target'), {
      min: 0, max: 200, decimals: 1,
      get: () => cs.TargetHotTemp,
      set: (v) => { cs.TargetHotTemp = v; stashTargets(); },
      // Unfocused, an active temp adjustment shows "80.0 (84.2)".
      display: () => settings.TempAdjustPercent !== 0
        ? `${cs.TargetHotTemp.toFixed(1)} (${adjustedHotTempC(cs.TargetHotTemp, settings.TempAdjustPercent).toFixed(1)})`
        : cs.TargetHotTemp.toFixed(1),
    });
    bindNumberInput(section.querySelector('.in-bar'), {
      min: 0, max: 5, decimals: 3,
      get: () => cs.TargetHotPressure,
      set: (v) => { cs.TargetHotPressure = v; stashTargets(); },
    });
  }
}

// Numeric text input: shows a formatted value when idle, raw editable value
// on focus, clamps + persists on commit (blur or Enter).
function bindNumberInput(input, { min, max, decimals, get, set, display = null }) {
  input._refresh = () => {
    if (document.activeElement === input) return;
    input.value = display ? display() : get().toFixed(decimals);
  };
  input.addEventListener('focus', () => {
    input.value = get().toFixed(decimals);
    input.select();
  });
  input.addEventListener('blur', () => {
    const v = parseFloat(input.value.trim().replace(',', '.'));
    if (Number.isFinite(v)) {
      set(Math.min(max, Math.max(min, v)));
      saveSettings();
    }
    render();
  });
}

// Enter advances focus to the next field (what every form does); on the
// last field it just commits by dropping focus.
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter') return;
  const el = e.target;
  if (!(el instanceof HTMLInputElement) && !(el instanceof HTMLSelectElement)) return;
  const focusables = [...document.querySelectorAll('input, select')]
    .filter((f) => f.offsetParent !== null && !f.disabled);
  const idx = focusables.indexOf(el);
  if (idx >= 0 && idx < focusables.length - 1 && el instanceof HTMLInputElement
      && el.type !== 'range') {
    e.preventDefault();
    focusables[idx + 1].focus();
    if (focusables[idx + 1] instanceof HTMLInputElement) focusables[idx + 1].select();
  } else {
    el.blur();
  }
});

// ---- Prediction ----

// Inputs for predictCorner / predictCornerTrajectory for one corner, or
// null when the prediction cannot run.
function cornerInputs(c) {
  const p = settings.Prediction;
  if (!isPredictionMode() || !p.Track || !p.Car) return null;
  return {
    track: p.Track,
    car: p.Car,
    condition: p.Condition,
    lapWithinStint: p.LapWithinStint,
    ambientTempC: p.AmbientTempC,
    trackTempC: p.TrackTempC,
    cloudCoverPct: p.CloudCoverPct,
    corner: c.id,
    targetHotPressureBar: settings[c.settingsKey].TargetHotPressure,
    coldTireTempC: settings[c.settingsKey].CurrentTemp,
    targetLapTimeS: p.TargetLapTimeS,
    compound: p.Compound,
  };
}

function refreshPredictions() {
  for (const c of CORNERS) {
    predictions[c.id] = null;
    const inputs = cornerInputs(c);
    if (!inputs) continue;
    try {
      const result = predictCorner(model, inputs);
      predictions[c.id] = result.predictedHotTempC;
    } catch {
      predictions[c.id] = null;
    }
  }
}

// ---- Preview charts ----

// Charts are sized from their scroll container, so they follow the
// viewport (and the grid's row height) instead of a fixed pixel size.
const chartResizeObserver = typeof ResizeObserver === 'function'
  ? new ResizeObserver(() => { if (isPreview()) renderCharts(); })
  : null;

function renderCharts() {
  for (const c of CORNERS) {
    const card = els.cornerGrid.querySelector(`[data-corner="${c.id}"]`);
    const scroll = card.querySelector('.chart-scroll');
    const axisHost = card.querySelector('.chart-axis-host');
    const plotHost = card.querySelector('.chart-plot-host');
    const inputs = cornerInputs(c);
    const width = scroll.clientWidth;
    const height = scroll.clientHeight;
    if (!inputs || width < 20 || height < 20) {
      axisHost.innerHTML = '';
      plotHost.innerHTML = '';
      continue;
    }
    try {
      const trajectory = predictCornerTrajectory(model, inputs);
      const { axisSvg, plotSvg } = renderPressureChart(trajectory, {
        targetBar: inputs.targetHotPressureBar,
        lapWithinStint: inputs.lapWithinStint,
        viewportWidth: width,
        height,
        labels: { lap: t('ChartLapAxis'), out: t('ChartOutLap') },
      });
      axisHost.innerHTML = axisSvg;
      plotHost.innerHTML = plotSvg;
    } catch {
      axisHost.innerHTML = '';
      plotHost.innerHTML = '';
    }
  }
}

// ---- Rendering ----

function applyStrings() {
  document.documentElement.lang = lang;
  for (const el of document.querySelectorAll('[data-str]')) {
    el.textContent = t(el.dataset.str);
  }
  els.modelUnavailable.textContent = t('ModelUnavailable').trim();
  els.langLabel.textContent = t('LanguageLabel');
  els.resetBtn.textContent = t('ResetButton');
  els.trackLabel.textContent = t('Track');
  els.carLabel.textContent = t('Car');
  els.conditionLabel.textContent = t('Condition');
  els.lapLabel.textContent = t('LapWithinStint');
  els.ambientLabel.textContent = t('Ambient');
  els.cloudLabel.textContent = t('CloudCover');
  els.targetLapLabel.textContent = t('TargetLapTime');
  els.compoundLabel.textContent = t('Compound');
  fillCompoundSelects();
  renderLastUpdated();

  fillSelect(els.langSelect, [
    { value: 'auto', label: t('LanguageAuto') },
    { value: 'en', label: t('LanguageEnglish') },
    { value: 'ja', label: t('LanguageJapanese') },
  ], settings.UiLanguage);
  fillSelect(els.conditionSelect,
    CONDITIONS.map((c) => ({ value: c.value, label: t(c.key) })),
    settings.Prediction.Condition);
}

// Lap times read as minutes:seconds — "1:05.2" — the way everyone thinks
// about them. Sub-minute laps display as plain seconds ("58.4"). Input
// accepts either form.
function formatLapTime(s) {
  if (!Number.isFinite(s) || s <= 0) return '';
  if (s < 60) return s.toFixed(1);
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `${m}:${sec.toFixed(1).padStart(4, '0')}`;
}

function parseLapTime(raw) {
  const t = raw.trim();
  if (!t) return null;
  if (t.includes(':')) {
    const [m, s] = t.split(':');
    const v = parseInt(m, 10) * 60 + parseFloat(s.replace(',', '.'));
    return Number.isFinite(v) ? v : null;
  }
  const v = parseFloat(t.replace(',', '.'));
  return Number.isFinite(v) ? v : null;
}

// Target lap time always carries a value: the typical lap for the current
// (track, car, condition). Snaps on selection change; edit it per session.
function snapTargetLap() {
  const p = settings.Prediction;
  if (!model || !p.Track || !p.Car) return;
  const typ = model.lookupLapTime(p.Track, p.Car, p.Condition).valueSeconds;
  if (Number.isFinite(typ) && typ > 0) p.TargetLapTimeS = Math.round(typ * 10) / 10;
}

// ---- Corner targets per configuration ----

// The configuration whose targets the corner cards currently show.
let activeTargetKey = null;

function targetKey() {
  const p = settings.Prediction;
  return p.Car ? `${p.Car}|${p.Compound ?? ''}|${p.Condition}` : null;
}

// Remember the corner cards' targets for the active configuration.
function stashTargets() {
  if (!activeTargetKey) return;
  const mem = {};
  for (const c of CORNERS) {
    const cs = settings[c.settingsKey];
    mem[c.id] = { TargetHotTemp: cs.TargetHotTemp, TargetHotPressure: cs.TargetHotPressure };
  }
  settings.TargetMemory[activeTargetKey] = mem;
}

// Called after any car / compound / condition change (and at init): stash
// the targets of the configuration being left, then show the new one's
// remembered targets, or its defaults when it is opened for the first
// time. `keepCurrent` (init with saved settings from before the memory
// existed) adopts the fields as they are instead of prefilling.
function loadTargetsForSelection({ keepCurrent = false } = {}) {
  stashTargets();
  activeTargetKey = targetKey();
  if (!activeTargetKey || !model) return;
  const mem = settings.TargetMemory[activeTargetKey];
  if (mem) {
    for (const c of CORNERS) {
      const m = mem[c.id];
      if (!m) continue;
      const cs = settings[c.settingsKey];
      if (Number.isFinite(m.TargetHotTemp)) cs.TargetHotTemp = m.TargetHotTemp;
      if (Number.isFinite(m.TargetHotPressure)) cs.TargetHotPressure = m.TargetHotPressure;
    }
  } else if (!keepCurrent) {
    snapCornerTargets();
  }
  stashTargets();
}

// Defaults for a configuration: the target hot temp is the model's
// steady-state median for the selected (car, condition); the target hot
// pressure the team's target for the selected (car, compound), else that
// median. Buckets missing from the artifact keep whatever the fields hold.
function snapCornerTargets() {
  const p = settings.Prediction;
  if (!model || !p.Car) return;
  const target = model.lookupTargetHotPressure(p.Car, p.Compound);
  for (const c of CORNERS) {
    const d = model.lookupCornerDefaults(p.Car, c.id, p.Condition);
    const cs = settings[c.settingsKey];
    if (d) cs.TargetHotTemp = Math.round(d.hotTempC * 10) / 10;
    if (target) cs.TargetHotPressure = Math.round(target.targetHotPressureBar * 100) / 100;
    else if (d) cs.TargetHotPressure = Math.round(d.hotPressureBar * 100) / 100;
  }
}

// Compound choices depend on the selected car. The choice is FORCED —
// every run has exactly one tire on all four corners, so there is no
// pooled "default" option; an unset selection snaps to the first compound.
function fillCompoundSelects() {
  const car = settings.Prediction.Car;
  const compounds = model && car ? model.availableCompounds(car) : [];
  if (compounds.length > 0
      && (!settings.Prediction.Compound || !compounds.includes(settings.Prediction.Compound))) {
    settings.Prediction.Compound = compounds[0];
  }
  fillSelect(els.compoundSelect,
    compounds.map((v) => ({ value: v, label: v })), settings.Prediction.Compound ?? '');
}

function fillSelect(select, options, selectedValue) {
  select.innerHTML = '';
  for (const o of options) {
    const opt = document.createElement('option');
    opt.value = o.value;
    opt.textContent = o.label;
    select.appendChild(opt);
  }
  select.value = selectedValue ?? '';
}

function render() {
  refreshPredictions();

  const prediction = isPredictionMode();
  document.body.classList.toggle('mode-manual', !prediction);
  document.body.classList.toggle('mode-prediction', prediction);

  els.modeToggle.setAttribute('aria-checked', String(prediction));
  els.modeToggle.disabled = model === null;
  els.modeContent.textContent = prediction ? t('ModePrediction') : t('ModeManual');
  els.modelUnavailable.hidden = model !== null || !modelLoadSettled;

  // Preview swaps the corner fields for charts and hides the setup
  // (inputs panel + bottom bar) until the user comes back to it.
  const inPreview = isPreview();
  document.body.classList.toggle('preview', inPreview);
  els.previewBtn.hidden = !prediction;
  els.previewBtn.textContent = inPreview ? t('SetupButton') : t('PreviewButton');
  els.previewBtn.setAttribute('aria-pressed', String(inPreview));

  // Bottom bar: temp adjust only in manual mode.
  els.adjustGroup.hidden = prediction;
  els.adjustSlider.value = String(settings.TempAdjustPercent);
  els.adjustLabel.textContent = settings.TempAdjustPercent === 0
    ? t('TempAdjustZero')
    : t('TempAdjustFormat').replace('{0}',
        (settings.TempAdjustPercent > 0 ? '+' : '') + settings.TempAdjustPercent.toFixed(1));

  // Prediction panel.
  els.predPanel.hidden = !prediction;
  const p = settings.Prediction;
  els.trackSelect.value = p.Track ?? '';
  els.carSelect.value = p.Car ?? '';
  els.conditionSelect.value = p.Condition;
  els.lapInput.value = String(p.LapWithinStint);
  els.ambientInput.value = p.AmbientTempC.toFixed(1);
  els.cloudInput.value = p.CloudCoverPct === null ? '' : String(p.CloudCoverPct);
  if (document.activeElement !== els.targetLapInput) {
    els.targetLapInput.value = formatLapTime(p.TargetLapTimeS);
  }
  els.compoundSelect.value = p.Compound ?? '';

  // Corner cards.
  for (const c of CORNERS) {
    const cs = settings[c.settingsKey];
    const card = els.cornerGrid.querySelector(`[data-corner="${c.id}"]`);
    for (const input of card.querySelectorAll('input')) input._refresh?.();

    const predictedHot = predictions[c.id];
    card.querySelector('.out-pred').textContent =
      predictedHot === null ? '—' : predictedHot.toFixed(1);

    // Effective hot temp: prediction when available, else the (possibly
    // percent-adjusted) manual target.
    const effectiveHot = predictedHot ?? adjustedHotTempC(cs.TargetHotTemp, settings.TempAdjustPercent);
    let cold = 0;
    try {
      cold = cornerColdPressureBar(cs.TargetHotPressure, effectiveHot, cs.CurrentTemp);
    } catch { /* out-of-range temps -> show 0 */ }
    card.querySelector('.setcold-value').textContent = `${cold.toFixed(3)} bar`;
  }

  if (inPreview) renderCharts();
}

// ---- Event wiring ----

function wireEvents() {
  els.modeToggle.addEventListener('click', () => {
    if (model === null) return;
    settings.Mode = isPredictionMode() ? MODE_MANUAL : MODE_PREDICTION;
    preview = false; // Preview belongs to Circuit Prediction; a mode change is a setup action
    saveSettings();
    render();
  });

  els.previewBtn.addEventListener('click', () => {
    preview = !preview;
    render();
    // Entering Preview always starts at pit exit, whatever was scrolled
    // to last time (after render, once the containers are displayed —
    // a hidden element's scroll offset cannot be set).
    if (preview) {
      for (const scroll of els.cornerGrid.querySelectorAll('.chart-scroll')) scroll.scrollLeft = 0;
    }
  });

  els.langSelect.addEventListener('change', () => {
    settings.UiLanguage = els.langSelect.value;
    lang = resolveLanguage(settings.UiLanguage, navigator.language);
    saveSettings();
    applyStrings();
    render();
  });

  els.resetBtn.addEventListener('click', () => {
    for (const c of CORNERS) Object.assign(settings[c.settingsKey], defaultCorner());
    settings.TempAdjustPercent = 0;
    settings.Prediction = {
      ...settings.Prediction,
      Track: model?.availableTracks[0] ?? null,
      Car: model?.availableCars[0] ?? null,
      Condition: 'dry',
      LapWithinStint: 5,
      AmbientTempC: 20.0,
      TrackTempC: null,
      CloudCoverPct: 50,
      TargetLapTimeS: null, // snapped to the selection's typical lap once the model loads
      Compound: null,
    };
    settings.TargetMemory = {}; // every configuration is prefilled afresh
    activeTargetKey = null;
    fillCompoundSelects(); // forced choice: first compound of the reset car, before the snap
    snapTargetLap();
    loadTargetsForSelection();
    saveSettings();
    applyStrings(); // condition + compound selections changed
    render();
  });

  els.adjustSlider.addEventListener('input', () => {
    settings.TempAdjustPercent = parseFloat(els.adjustSlider.value);
    saveSettings();
    render();
  });

  els.trackSelect.addEventListener('change', () => {
    settings.Prediction.Track = els.trackSelect.value || null;
    snapTargetLap();
    saveSettings();
    render();
  });

  els.carSelect.addEventListener('change', () => {
    settings.Prediction.Car = els.carSelect.value || null;
    // Compounds are per-car: snap to the new car's first compound.
    settings.Prediction.Compound = null;
    fillCompoundSelects();
    snapTargetLap();
    loadTargetsForSelection();
    saveSettings();
    render();
  });

  els.compoundSelect.addEventListener('change', () => {
    settings.Prediction.Compound = els.compoundSelect.value || null;
    loadTargetsForSelection();
    saveSettings();
    render();
  });

  els.conditionSelect.addEventListener('change', () => {
    settings.Prediction.Condition = els.conditionSelect.value;
    snapTargetLap();
    loadTargetsForSelection();
    saveSettings();
    render();
  });

  els.lapInput.addEventListener('change', () => {
    const v = Math.round(parseFloat(els.lapInput.value));
    if (Number.isFinite(v)) settings.Prediction.LapWithinStint = Math.min(60, Math.max(1, v));
    saveSettings();
    render();
  });

  els.ambientInput.addEventListener('change', () => {
    const v = parseFloat(els.ambientInput.value);
    if (Number.isFinite(v)) settings.Prediction.AmbientTempC = Math.min(50, Math.max(-20, v));
    saveSettings();
    render();
  });

  els.cloudInput.addEventListener('change', () => {
    const raw = els.cloudInput.value.trim();
    const v = parseFloat(raw);
    if (raw === '' || !Number.isFinite(v)) {
      settings.Prediction.CloudCoverPct = 50; // neutral sky, never blank
    } else {
      settings.Prediction.CloudCoverPct = Math.min(100, Math.max(0, v));
    }
    saveSettings();
    render();
  });

  els.targetLapInput.addEventListener('change', () => {
    const v = parseLapTime(els.targetLapInput.value);
    if (v === null) {
      snapTargetLap(); // blank/invalid falls back to the selection's typical lap
    } else {
      settings.Prediction.TargetLapTimeS = Math.min(900, Math.max(20, v));
    }
    saveSettings();
    render();
  });
}

// ---- Model loading ----

// The deploy workflow copies data/tire_dataset/tire_model.json next to
// index.html; the second path makes `just tire-web-serve` (repo root over
// plain http.server) work without an assembly step.
async function loadModel() {
  for (const url of ['./tire_model.json', '../../data/tire_dataset/tire_model.json']) {
    try {
      const response = await fetch(url);
      if (!response.ok) continue;
      return new TireModel(await response.json());
    } catch { /* try next path */ }
  }
  return null;
}

// ---- Init ----

async function init() {
  buildCornerCards();
  wireEvents();
  applyStrings();
  render(); // manual-mode UI is usable immediately while the model loads

  model = await loadModel();
  modelLoadSettled = true;
  renderLastUpdated();

  if (model !== null) {
    const p = settings.Prediction;
    if (!p.Track || !model.availableTracks.includes(p.Track)) p.Track = model.availableTracks[0] ?? null;
    // Resolve alias-pooled car names so a car saved before pooling
    // (e.g. "KK-SII" -> "FJ") lands on its pooled entry, not the default.
    if (p.Car) p.Car = model.resolveCar(p.Car);
    if (!p.Car || !model.availableCars.includes(p.Car)) p.Car = model.availableCars[0] ?? null;
    fillSelect(els.trackSelect,
      model.availableTracks.map((v) => ({ value: v, label: v })), p.Track);
    fillSelect(els.carSelect,
      model.availableCars.map((v) => ({ value: v, label: v })), p.Car);
    fillCompoundSelects();
    if (p.TargetLapTimeS === null) snapTargetLap();
    if (p.CloudCoverPct === null) p.CloudCoverPct = 50;
    // First run prefills; saved settings keep the active configuration's
    // fields (restored from memory when it has an entry).
    loadTargetsForSelection({ keepCurrent: hadSavedSettings });
    saveSettings();
  } else {
    settings.Mode = MODE_MANUAL;
  }
  render();
}

init();

// Offline support is progressive — registration failure (old browser,
// file:// serve) just means the app needs a network connection.
//
// Updates: the deploy stamps a new cache name into sw.js and the worker
// claims the page as soon as its precache is complete (see sw.js), so a
// client on an old build reloads once onto the new files the first time
// it is online and the browser sees the new worker. `updateViaCache:
// 'none'` keeps the HTTP cache from hiding a new sw.js, and the explicit
// update() on launch and on every return to the foreground covers an
// installed PWA that never navigates. Offline, update() fails silently
// and the current worker keeps serving. Settings live in localStorage,
// so the reload loses nothing.
if ('serviceWorker' in navigator) {
  // A first install also claims the page (controllerchange with no previous
  // controller); only a change of controller is an update worth a reload.
  let hadController = Boolean(navigator.serviceWorker.controller);
  let reloading = false;
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!hadController) { hadController = true; return; }
    if (reloading) return;
    reloading = true;
    window.location.reload();
  });
  navigator.serviceWorker.register('./sw.js', { updateViaCache: 'none' })
    .then((reg) => {
      const check = () => { reg.update().catch(() => {}); };
      check();
      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') check();
      });
    })
    .catch(() => {});
}
