// UI wiring for the tire pressure calculator web app. All model math lives
// in model.js; this file owns DOM, settings persistence, and localization.

import {
  TireModel, predictCorner, adjustedHotTempC, cornerColdPressureBar,
} from './model.js';
import { STRINGS, resolveLanguage } from './strings.js';

// Same key AND JSON shape as the .NET heads (C# AppSettings serialized with
// PascalCase properties, Mode as 0/1) so settings saved by the old WASM
// build carry over, and stay portable with Desktop/Android exports.
const STORAGE_KEY = 'tire-pressure-calculator/settings';

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
    Mode: MODE_MANUAL,
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
  };
}

// True when settings came from storage (vs factory defaults). First run
// prefills the corner targets from the model; a returning user's tuned
// values are left alone until they change car/condition or hit Reset.
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

const t = (key) => STRINGS[lang][key] ?? STRINGS.en[key] ?? key;

const isPredictionMode = () => settings.Mode === MODE_PREDICTION && model !== null;

// ---- DOM handles ----

const $ = (id) => document.getElementById(id);
const els = {
  modeLabel: $('modeLabel'), modeToggle: $('modeToggle'), modeContent: $('modeContent'),
  modelUnavailable: $('modelUnavailable'),
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
};

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
      <hr>
      <div class="setcold">
        <span class="setcold-label" data-str="SetCold"></span>
        <span class="setcold-value">—</span>
      </div>`;
    els.cornerGrid.appendChild(section);

    const cs = settings[c.settingsKey];
    bindNumberInput(section.querySelector('.in-current'), {
      min: -40, max: 60, decimals: 1,
      get: () => cs.CurrentTemp,
      set: (v) => { cs.CurrentTemp = v; },
    });
    bindNumberInput(section.querySelector('.in-target'), {
      min: 0, max: 200, decimals: 1,
      get: () => cs.TargetHotTemp,
      set: (v) => { cs.TargetHotTemp = v; },
      // Unfocused, an active temp adjustment shows "80.0 (84.2)".
      display: () => settings.TempAdjustPercent !== 0
        ? `${cs.TargetHotTemp.toFixed(1)} (${adjustedHotTempC(cs.TargetHotTemp, settings.TempAdjustPercent).toFixed(1)})`
        : cs.TargetHotTemp.toFixed(1),
    });
    bindNumberInput(section.querySelector('.in-bar'), {
      min: 0, max: 5, decimals: 3,
      get: () => cs.TargetHotPressure,
      set: (v) => { cs.TargetHotPressure = v; },
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

function refreshPredictions() {
  const p = settings.Prediction;
  for (const c of CORNERS) {
    predictions[c.id] = null;
    if (!isPredictionMode() || !p.Track || !p.Car) continue;
    try {
      const result = predictCorner(model, {
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
      });
      predictions[c.id] = result.predictedHotTempC;
    } catch {
      predictions[c.id] = null;
    }
  }
}

// ---- Rendering ----

function applyStrings() {
  document.documentElement.lang = lang;
  for (const el of document.querySelectorAll('[data-str]')) {
    el.textContent = t(el.dataset.str);
  }
  els.modeLabel.textContent = t('ModeLabel');
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

// Corner-card prefills follow the car: the target hot temp and hot
// pressure snap to the model's steady-state medians for the selected
// (car, condition). Buckets missing from the artifact keep whatever the
// fields currently hold.
function snapCornerTargets() {
  const p = settings.Prediction;
  if (!model || !p.Car) return;
  for (const c of CORNERS) {
    const d = model.lookupCornerDefaults(p.Car, c.id, p.Condition);
    if (!d) continue;
    const cs = settings[c.settingsKey];
    cs.TargetHotTemp = Math.round(d.hotTempC * 10) / 10;
    cs.TargetHotPressure = Math.round(d.hotPressureBar * 100) / 100;
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
}

// ---- Event wiring ----

function wireEvents() {
  els.modeToggle.addEventListener('click', () => {
    if (model === null) return;
    settings.Mode = isPredictionMode() ? MODE_MANUAL : MODE_PREDICTION;
    saveSettings();
    render();
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
    snapTargetLap();
    snapCornerTargets();
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
    snapCornerTargets();
    saveSettings();
    render();
  });

  els.compoundSelect.addEventListener('change', () => {
    settings.Prediction.Compound = els.compoundSelect.value || null;
    saveSettings();
    render();
  });

  els.conditionSelect.addEventListener('change', () => {
    settings.Prediction.Condition = els.conditionSelect.value;
    snapTargetLap();
    snapCornerTargets();
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
    if (!hadSavedSettings) snapCornerTargets();
  } else {
    settings.Mode = MODE_MANUAL;
  }
  render();
}

init();

// Offline support is progressive — registration failure (old browser,
// file:// serve) just means the app needs a network connection.
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js').catch(() => {});
}
