# Cold tire pressure model — v0.3 (rain-aware)

A physically-based predictor that takes
**(track, car, target lap within stint, target hot pressure per corner, expected ambient temp)**
and returns the cold pressure to set, per corner.

> Quickstart: `just tire-build-warmup-table && just tire-predict --track tsukuba_2000 --car KK-SII --lap 5 --ambient 18 --hot-all 1.95`

## 1. Why this is the v0

The plan was deliberately conservative: get a working end-to-end predictor with
honest accuracy numbers we can iterate on, not a full Bayesian hierarchical
model. The constraints we adopted:

- **Physically based.** Every fitted parameter is a real thermal quantity the
  user can sanity-check.
- **Few free parameters.** ~20 across the whole dataset, not hundreds.
- **Per-corner output.** Front/rear and left/right tires are different physical
  objects, so per-corner cold pressures fall out naturally.
- **Track-independent fit.** A car's thermal parameters are properties of the
  car, not the venue. The track effect enters through data (⟨g²⟩, c_track),
  not through track-specific fitted coefficients.
- **Tire compound omitted.** Notes-derived compound coverage is only 49 of
  142 ok sessions (34%) and strings are messy. Including it as a model
  dimension makes K + c_track + compound mutually unidentifiable in the
  current data. Deferred until coverage improves.
- **Rain awareness via the weather data, not the run-notes.** v0.3 adds
  `condition ∈ {dry, damp, wet}` as a model dimension, classified from
  Open-Meteo's `precipitation` field (mm/hr). 75% of sessions have weather
  coverage vs only ~25% with notes-derived condition. See §2.9.

## 2. Modeling approach

### 2.1 Energy balance on a lumped tire mass

The model is one ODE — heat flux balance on the tire treated as a single
thermal mass:

```
                                              ┌──────────────┐   ┌──────────────┐
   m·c · dT/dt   =   c_track · α · g²(t)   −   │ h_air · (T   │ + │ h_road ·     │
                                               │  − T_air )   │   │ (T − T_road) │
   ───────────       ──────────────────        └──────────────┘   └──────────────┘
   energy stored     energy IN                       energy OUT
   per second        per second                      per second
   (W = J/s)         (friction work + hysteresis)    (convection to air at the
                                                      tire's top/sides + conduction
                                                      to the track at the patch)
```

**Energy IN.** Friction work at the contact patch scales with squared total
acceleration `g²(t) = lat_g(t)² + long_g(t)²` (cornering + braking) times a
per-track surface factor `c_track` (asphalt grip, roughness — what's left over
after accounting for ⟨g²⟩) times a coefficient `α` that absorbs friction
coefficient, contact-patch geometry, brake-disc-to-tire heat coupling, and
compound hysteresis.

**Energy OUT.** Two parallel paths — convection to ambient air at the tire's
top/sides, and conduction to the track surface at the contact patch. These
have different reference temperatures: hot asphalt cools the tire less than
cold air does. We collect them into one effective ambient:

```
T_eff = (1 − w_road) · T_air + w_road · T_road
```

where `w_road = h_road / (h_air + h_road)` is the fraction of energy-OUT going
to the track. **v0 fixes `w_road = 0.2`** based on the physical prior that
convection-to-air dominates conduction-to-road at race speeds (fast airflow
over the tire; small contact-patch area relative to tire surface area). Fitting
`w_road` is deferred to v1 — at 34% c_track-known × ~5% T_road-measured, the
joint identifiability is poor.

**T_road sourcing** at inference (priority order): user-supplied → AIM logger
channel (rare, <5%) → proxy `T_road = T_air + 10 · (1 − cloud_cover/100) ·
sun_factor` from Open-Meteo cloud cover → fall back to `T_road = T_air`.

### 2.2 Closed-form solution

Approximating `g²(t)` by its session average `⟨g²⟩` (good within a stint for
a consistent driver), the linear ODE has a closed-form solution starting
from `T_0 = T_eff`:

```
T_hot(t) − T_eff  =  K · c_track · ⟨g²⟩ · (1 − exp(−t / τ_sec))
```

with:

- `K = α / (h_air + h_road)` — units of K per G². The **warmup gain**:
  steady-state Kelvin per unit of average G². Property of `(car, corner)`.
- `τ_sec = m·c / (h_air + h_road)` — units of seconds. The **thermal time
  constant**: how fast the tire approaches its steady state. Property of
  `(car, corner)`.
- `t` = on-track seconds since stint start.

### 2.3 Why on-track seconds, not laps

A Tsukuba lap takes ~60 s; a Fuji lap takes ~115 s. Fitting τ in laps would
mix tire physics with circuit geometry — "3 laps to warm up" means different
things at different tracks. **Fitting τ in seconds gives a quantity that is
truly a property of the wheel/tire/hub system**, transferable across tracks.

At inference time we convert lap N → seconds via the bucket's median lap
time: `t_at_lap_N = N · lap_time_typ_s[track, car]`. Users can override with
`--lap-time-s` (e.g. for race-pace estimates that differ from session
median).

### 2.4 Gay-Lussac inversion

Given the predicted hot temperature, invert Gay-Lussac's Law at constant
volume (P/T = const, absolute units):

```
T_cold_K  = T_air + 273.15          # cold tires equilibrate to AIR (not T_eff)
T_hot_K   = T_hot + 273.15
P_hot_abs = target_hot_pressure_bar + 1.0
P_cold    = P_hot_abs · (T_cold_K / T_hot_K) − 1.0
```

Matches the C# tire-pressure calculator's convention exactly
(`tire_pressure_calculator/Core/ViewModels/TireCornerViewModel.cs:77-89`)
so round-trip is bit-identical within float rounding. **Note `T_cold` uses
`T_air` only, not the road-blended `T_eff`** — cold tires sitting in the
pits aren't being cooled by hot asphalt; they equilibrate to whatever air
they're sitting in.

### 2.5 Parameter pooling

| Param            | Physics                                  | Pooled over                       | Count   | Notes |
|------------------|------------------------------------------|-----------------------------------|---------|---|
| `K`              | `α / (h_air + h_road)`                   | `(car, corner)`                   | 8       | Energy-IN / Energy-OUT gain |
| `τ_sec`          | `m·c / (h_air + h_road)`                 | `(car, corner)`                   | 8       | Thermal time constant |
| `c_track`        | per-track surface scalar                 | `(track)`                         | ~3–4    | Tsukuba anchored at 1.0 |
| `w_road`         | `h_road / (h_air + h_road)`              | **fixed at 0.2 in v0**            | 0       | Deferred |
| `⟨g²⟩`           | `median(heat_proxy / on_track_s)`        | `(track, car)` — lookup, not fit  | ~6      | From data |
| `lap_time_typ_s` | `median(on_track_s)`                     | `(track, car)` — lookup, not fit  | ~6      | From data |
| `T_road`         | logger / weather + sun proxy             | per-session                       | 0       | From data |

**Total fitted: ~20 parameters** across the entire dataset (8 K + 8 τ_sec +
~4 c_track). Compare to a per-(track, car, lap) regression approach which
would have hundreds. The energy-balance framing means the **track-aggressiveness
signal is captured by ⟨g²⟩ data, not by a fitted constant** — that's why the
model is track-independent at fit time and only enters the prediction via
data lookups.

### 2.6 Fitting procedure

Two-pass non-linear least squares using `scipy.optimize.curve_fit` against
per-lap aggregates from `laps.parquet`:

1. **Pass 1 — `τ_sec[car, corner]` + per-bucket gains.** For each (car, corner),
   select that car's (track) buckets with ≥ 30 lap samples. Fit jointly across
   them: `δT_i = gain_{bucket(i)} · (1 − exp(−t_i / τ_sec))` with a shared
   `τ_sec[car, corner]` and bucket-specific `gain_b = K · c_track · ⟨g²⟩`.
   KK-SII FL τ is fit jointly from Tsukuba (326 laps) + Fuji (71 laps) data —
   precisely the cross-circuit shrinkage we want.

2. **Pass 2 — factor `gain_b` into `K[car, corner] × c_track[track]`.** Divide
   out ⟨g²⟩ (a lookup) and use alternating least squares in log-space, with
   `c_track[tsukuba_2000] ≡ 1.0` anchored for identifiability. Standard errors
   propagate from Pass 1's bucket-gain stderrs.

If a (car, corner) bucket has fewer than `MIN_LAPS_FOR_TAU_FIT = 30` lap
samples, the fit returns the prior `τ_sec = 240 s, K = 60 K/G²` with
`from_prior: true` flagged in the artifact.

### 2.7 Artifact: `tire_model.json`

The model serializes to a versioned JSON committed under
`data/tire_dataset/tire_model.json`. Six top-level tables, one per physical
quantity in the energy balance:

- `tau_sec_by_car_corner` — 8 entries with stderrs
- `K_buckets` — 8 entries with stderrs and `from_single_track` flags
- `c_track_by_track` — per-track surface scalars, Tsukuba marked as anchor
- `g2_typ_by_track_car` — lookup
- `lap_time_typ_by_track_car` — lookup
- `energy_balance` — the `w_road` config + T_road proxy formula
- `sensor_blacklist_applied` — audit trail of masked (session, corner) pairs
- `fallback_order_for_K` — declarative fallback chain

Typical file size: 6–12 KB. Diff-friendly — when new sessions land, the
artifact updates in lockstep and the JSON diff shows reviewers exactly which
buckets gained samples or changed coefficients. The C# calculator (future
plan) reads the same file.

### 2.10 Target lap time (schema v3)

Tire energy scales strongly with pace: within a (track, car, condition)
bucket, `log(g²_lap)` vs `log(lap_time)` is close to a power law (slopes
−2.4…−3.6, |r| 0.8–0.97 on the 2026-08 dataset; pure v²-scaling physics
would give −4). Schema v3 exposes that as an optional prediction input: a
**target lap time** sets both the time-on-track clock (`t = N × target`)
and a multiplier on ⟨g²⟩.

The pace→energy mapping is fitted **sector-wise** (`tire_model/sectors.py`)
so one bad turn on an otherwise aggressive lap can't skew it: each lap is
split into 3 distance-based sectors from the timeseries (same
`(lat² + long²)·dt` integrand as `heat_proxy`), and for each curve sample
at total time T the 15 nearest laps by lap time contribute *median* sector
times and median sector g² (rescaled to sum to T, recombined as
`g²(T) = Σ g²_s·t_s / T`). The artifact stores the result as a small
piecewise-linear `g2_vs_lap_time` curve per ⟨g²⟩ entry; prediction scales
`g2_typ` by `curve(target)/curve(lap_time_typ)` so an omitted target (or
target == typical pace) reproduces v2 behavior exactly. Buckets without a
curve fall back to the pooled sector-fit exponent
(`g2_lap_time_model.default_exponent`); the multiplier is clamped to
`multiplier_clamp` either way, and interpolation clamps at the curve
endpoints (no extrapolation beyond the fastest pace ever driven).

Held-out CV evidence (predicting with only the lap's time instead of its
measured g²): on the KK-SII an accurate target recovers ≈half the gap
between the pooled-⟨g²⟩ prediction and the measured-g² oracle
(−0.35…−0.5 °C MAE per corner). On the Inferno 86 pace-conditioning
currently *hurts* — that car's heat does not track measured g²
proportionally (even the oracle underperforms a constant), so leave the
target blank there until the per-car g² sensitivity is modeled.

### 2.11 Tire compound (per-axle K overrides)

The Inferno 86 alternates between tire sets (A052 and RE-71RS through
2025-2026, sometimes different compounds per axle in the same session),
and the two heat very differently: session-median implied K separates by
~45% on the rears (A052 ≈ 39-43, RE-71RS ≈ 59-61 K/G² dry) with
within-compound spread of ±2-4 vs ±10 for the pooled mix. The pooled K
splits the difference and mis-predicts both — this was the dominant
source of the car's dry-rear MAE.

Labels come from ``data/tire_dataset/tire_compounds.yaml`` (human-curated
per-session, per-axle; authoritative) with the notes-extraction compounds
as fallback; wheel-set names from the notes ("Black wheels") resolve
through the sidecar's ``wheel_sets`` mapping. Labeled sessions get a
per-(car, compound, corner, condition) K fitted by closed-form weighted
least squares with the pooled τ and c_track held fixed
(``ΔT = K · g²·c_track·warmup_frac``), which stays stable on sparse
buckets; ``MIN_LAPS_FOR_COMPOUND_K = 10``. The artifact carries these in
``K_by_car_compound_corner_cond`` (additive to schema v3 — consumers
without compound support ignore the table).

Prediction: ``predict_cold_pressure(compound_front=..., compound_rear=...)``
(CLI ``--compound`` / ``--compound-front`` / ``--compound-rear``) swaps in
the compound K per axle when a fitted bucket exists (condition chain
applies); unknown compounds and unlabeled cars keep the pooled K. Held-out
CV on the labeled Inferno sessions (same folds, pooled vs compound K):
FL 6.80→4.95, FR 5.39→4.28, RL 5.99→4.49, RR 4.99→4.51 °C MAE, with the
per-corner bias collapsing (RL +3.50→0.00).

### 2.9 Track condition (rain)

v0.3 adds a `condition` dimension to K, τ_sec, ⟨g²⟩, and lap_time_typ. The
condition is **derived from weather data**, not from run-notes:

- `dry`: precipitation < 0.1 mm/hr
- `damp`: 0.1 ≤ precipitation < 1.0 mm/hr
- `wet`: precipitation ≥ 1.0 mm/hr
- `unknown`: no weather data (excluded from training)

Physically, more cooling (rain on the tire) means **larger `h_air + h_road`**,
so both `K = α/(h_air+h_road)` and `τ = m·c/(h_air+h_road)` should **drop**
in damp/wet — the tire reaches equilibrium faster *and* at a lower steady
state. ⟨g²⟩ also drops naturally because drivers go slower in rain (it's a
data lookup, not a fitted param).

`c_track` stays per-track (no condition dimension) — surface character is a
property of the venue. The condition's effect lives in K and ⟨g²⟩.

**Inference-time fallback chain** (when the requested condition has no fit):

```
wet  → damp → dry      (physically closest neighbors)
damp → dry
dry  → dry
```

**Fit dataset breakdown** (with the v0.3 weather classification):

| Condition | Sessions | Usable laps | Notes |
|---|---|---|---|
| dry | 69 | ~677 | both cars, all 4 tracks |
| damp | 30 | ~212 | both cars, all 4 tracks |
| wet | 8 | ~34 | mostly KK-SII at Fuji 2026-02-25 |
| unknown | 46 | excluded | no weather data |

**Physical-prior clips** to guard against pathological sparse-data fits:

- `τ_sec[damp/wet]` is clipped to ≤ 1.5 × `τ_sec[dry]` (faster cooling)
- `K[damp/wet]` is clipped to ≤ 1.2 × `K[dry]` (less heat per G²)

Clipped parameters are flagged with `from_prior=True` in the artifact.
Without these clips, Inferno 86 damp K and τ were 1.5-3× the dry values —
the optimizer compensating for warmup-incomplete short stints rather than
real thermal physics. The 1.2× / 1.5× ratios are physical priors; the
exact thresholds are empirical and easy to tune.

**Known limitation**: KK-SII wet predictions are conservative (the model
under-predicts hot temp slightly, so cold pressure recommendations skew
higher than strictly needed). The safer error direction.

### 2.8 Sensor blacklist (human-curated)

`just tire-sensor-audit` auto-detects (session, corner) channels whose TPMS
temperature has std < 1.0 °C across ≥ 4 usable laps (i.e. the sensor looks
stuck) and presents them for human review. **No auto-masking.** Confirmed
broken entries get added to a committed YAML file
(`data/tire_dataset/sensor_blacklist.yaml`); the build pipeline reads that
file and masks those channels from training and held-out evaluation alike.

The v0 dataset has 8 stuck-at-X channels confirmed via this workflow:

| session | car | track | date | corner | stuck at |
|---|---|---|---|---|---|
| `01811fbc44ee4dcb` | Inferno 86 | fuji | 2025-07-07 | RL | 28 °C |
| `1cd906c9ecf27b24` | KK-SII | fuji | 2026-02-25 | RR | 12 °C |
| `034b6b78a440fe3a` | KK-SII | fuji | 2026-02-26 | RR | 13 °C |
| `1efdf17c42265ba9` | KK-SII | fuji | 2026-02-26 | RR | 13 °C |
| `bbe7b51bd428f5a0` | KK-SII | fuji | 2026-02-26 | RR | 13 °C |
| `d3bf612415fb31ba` | KK-SII | fuji | 2026-02-26 | RR | 14 °C |
| `42a95638e6bd528b` | KK-SII | tsukuba | 2026-03-22 | RL | 41 °C |
| `cd9aaae0b59ff389` | KK-SII | tsukuba | 2026-03-22 | RL | 41 °C |

KK-SII RR appears to have had **a single bad sensor that ran across 5
consecutive Fuji sessions on 2026-02-25/26**, and KK-SII RL had a similar
recurring failure across **2 Tsukuba sessions on 2026-03-22**.

## 3. Test methodology

### 3.1 Two distinct validations

- **`tire-predict-validate`** — compares predicted cold pressures against
  notes-recorded cold pressures from `notes_matches.parquet` (79 sessions
  with logged cold pressures from run notes). Uses the production
  (full-data) model. This is a **consistency check**, not a held-out test —
  the model was trained on these sessions too.
- **`tire-predict-holdout`** — the honest generalization measure. Excludes
  N=2 sessions per (track, car) bucket from training, predicts per-lap
  T_hot for those sessions, reports per-corner residuals. **No training on
  the test set.**

### 3.2 Held-out test design

- **Bucket selection.** Only (track, car) buckets with ≥ 10 sessions are
  eligible (Tsukuba KK-SII: 34, Sodegaura Inferno 86: 30, Fuji Inferno 86:
  21, Fuji KK-SII: 11). Tsukuba Inferno 86 (7 sessions) and Motegi Inferno
  86 (4 sessions) are too sparse to safely exclude from.
- **Session picking.** For each eligible bucket, the first 2 session_ids
  in sort order are held out. Deterministic, reproducible.
- **Lap filtering.** Out-laps (`lap_within_stint == 0`) are excluded since
  they're cold-start points, not warmup-curve observations. Same convention
  as training.
- **Blacklist applied.** Confirmed broken sensors are masked in both
  training and evaluation — we don't grade the model against channels we
  already know are broken.
- **Metric.** Per-corner per-lap **T_hot residual** (predicted minus
  observed end-of-lap TPMS temperature). MAE, RMSE, and mean signed bias
  reported per corner. T_hot is the right metric because everything
  downstream (Gay-Lussac, cold pressure) is a deterministic transform of
  it — predict T_hot well and the cold pressure is right.

### 3.3 What MAE in T_hot translates to in cold pressure

For target hot 1.9 bar (gauge), air 18 °C, T_hot ≈ 50–60 °C, the Gay-Lussac
inversion has

```
|∂P_cold / ∂T_hot|  ≈  (P_hot + 1) · T_cold_K / T_hot_K²  ≈  0.025 bar / °C
```

so an MAE of 2 °C in T_hot → roughly 0.05 bar in cold pressure (~0.7 psi).
An MAE of 4 °C → ~0.10 bar (~1.5 psi).

## 4. Results

### 4.1 Headline (held-out, pooled, 236 (lap × corner) points)

| Corner | MAE | RMSE | mean bias | n |
|---|---|---|---|---|
| FL | 3.01 °C | 3.72 °C | +0.09 °C | 61 |
| FR | 2.99 °C | 3.71 °C | +0.19 °C | 61 |
| RL | 2.77 °C | 3.98 °C | −0.51 °C | 53 |
| RR | 2.65 °C | 3.49 °C | −0.84 °C | 61 |

Pooled MAE ~3 °C ↔ ~±0.075 bar cold-pressure precision per corner.

### 4.2 Per-car breakdown — the cars are in very different regimes

#### KK-SII (excellent)

| Corner | MAE | RMSE | mean bias | n |
|---|---|---|---|---|
| FL | **1.48 °C** | 1.81 °C | +1.18 °C | 30 |
| FR | **1.94 °C** | 2.31 °C | +1.92 °C | 30 |
| RL | **0.90 °C** | 1.30 °C | +0.53 °C | 30 |
| RR | **1.07 °C** | 1.36 °C | +0.07 °C | 30 |

MAE 0.9–1.9 °C ↔ ~±0.03 bar cold-pressure precision. The model is
essentially right for this car. Small positive bias (+0.07 to +1.92 °C)
means we slightly under-predict warmup — likely the 2026-02-25 cold-day
session whose three borderline corners survived the blacklist as
"plausibly real" pulled mean K down a touch.

#### Inferno 86 (acceptable but markedly worse)

| Corner | MAE | RMSE | mean bias | n |
|---|---|---|---|---|
| FL | 4.49 °C | 4.91 °C | −0.97 °C | 31 |
| FR | 4.00 °C | 4.69 °C | −1.48 °C | 31 |
| RL | 5.20 °C | 5.86 °C | −1.86 °C | 23 |
| RR | 4.17 °C | 4.71 °C | −1.72 °C | 31 |

MAE 4–5 °C ↔ ~±0.10 bar cold-pressure precision. The systematic **negative
bias on every corner (−1.0 to −1.9 °C)** is the telling signal: we're
consistently over-predicting T_hot for held-out Inferno 86 sessions.
Hypothesized causes:

- Inferno 86 trains on data from 4 tracks vs KK-SII's 2 — wider variance
  across driving styles + track surfaces, more for c_track to absorb.
- Several Inferno 86 sessions have very short stint counts; the K · c_track
  decomposition is poorly constrained in those buckets.

### 4.3 Fitted parameter values (sanity check against physics)

```
=== τ_sec by (car, corner) ===                  === K by (car, corner) ===
     Inferno 86 fl  τ=249 ± 14 s                     Inferno 86 fl  K=70.4 ± 2.85
     Inferno 86 fr  τ=229 ± 12 s                     Inferno 86 fr  K=61.7 ± 1.10
     Inferno 86 rl  τ=300 ± 19 s                     Inferno 86 rl  K=64.3 ± 1.71
     Inferno 86 rr  τ=275 ± 16 s                     Inferno 86 rr  K=58.5 ± 1.60
         KK-SII fl  τ=211 ± 14 s                         KK-SII fl  K=29.2 ± 0.04
         KK-SII fr  τ=187 ± 14 s                         KK-SII fr  K=23.0 ± 1.11
         KK-SII rl  τ=220 ± 20 s                         KK-SII rl  K=24.9 ± 0.08
         KK-SII rr  τ=231 ± 16 s                         KK-SII rr  K=25.6 ± 0.00

=== c_track ===
         tsukuba_2000  c=1.000 (anchor)
             sodegaura c=0.968 ± 0.019
                  fuji c=1.273 ± 0.037
```

- **τ in 187–300 s** range — literature-typical for racing tires; rears
  consistently longer than fronts (more thermal mass, less brake heat),
  which matches physical intuition.
- **K in 23–70 K/G²** — Inferno 86 generates roughly 2.5× the steady-state
  ΔT per unit G² as KK-SII. Makes sense given the Inferno 86 is heavier
  with more downforce and runs slicks; KK-SII is a lighter formula car
  with a different compound family.
- **c_track[fuji] = 1.27** — Fuji's faster, wider corners generate more
  load per unit of measured G² (sustained high-G sweepers vs Tsukuba's
  short heavy hits). The model is capturing this.

### 4.4 Effect of the sensor blacklist

Before applying the 8-entry blacklist, the held-out validation produced:

| Corner | MAE pre-blacklist | MAE post-blacklist | Improvement |
|---|---|---|---|
| FL | 4.00 °C | 3.01 °C | 25% |
| FR | 4.21 °C | 2.99 °C | 29% |
| RL | **7.57 °C** | **2.77 °C** | **63%** |
| RR | **7.72 °C** | **2.65 °C** | **66%** |

The dramatic 60+% improvement on RL/RR came from removing the stuck-at-X
training data that was pulling K down. Most starkly: **KK-SII RR went from
`K = 13 ± 8.77` (fit on broken stuck-at-13 data) to `K = 25.6` (sensible,
matching the other corners)**.

## 5. Limitations + ideas for next steps

In rough priority order:

### v0.1 — same architecture, better data hygiene

1. **Tighten the borderline blacklist.** The audit surfaced 6 borderline
   candidates (std 0.4–0.9 °C) we left in. Some, like the 2026-02-25
   cold-day FL/FR/RL on `1cd906c9ecf27b24`, might genuinely be valid
   short-warmup data; others might be intermittent sensor issues. Manual
   inspection of the per-lap traces would settle it.
2. **Track-canonical cleanup.** 46 Inferno 86 sessions have
   `track_canonical = None` because the filename track string didn't
   normalize. Recovering even half of those would meaningfully shrink the
   Inferno 86 held-out variance.
3. **Per-(track, car) breakdown of the held-out report.** Right now we see
   per-car residuals; splitting further by track would tell us whether
   Inferno 86's worse fit is uniformly bad or concentrated at one venue
   (e.g., Sodegaura has 232 usable laps but might be biasing).

### v0.2 — model refinements that don't change the architecture

4. **Validate against notes-recorded cold pressures.** `tire-predict-validate`
   exists but I haven't reported its number in this v0 — it would tell us
   whether the T_hot fit translates to real-world cold-pressure accuracy as
   the Gay-Lussac sensitivity analysis predicts.
5. **Fit `w_road`.** Currently fixed at 0.2 from a physical prior. Even with
   sparse logger-T_road coverage, a single global `w_road` could be fit
   jointly with K + c_track + τ — the worst that happens is the optimizer
   stays close to 0.2 and we learn nothing, but we get an honest standard
   error on it.
6. **T_road sun-factor refinement.** Currently 1.0 globally. A
   latitude/time-of-day-aware factor would matter for Japan summer/winter
   contrast — a simple lookup by month + venue lat would do it.

### v1 — bigger architecture changes

7. **Within-lap fitting.** Use the per-sample `timeseries/*.parquet` data
   (already committed) to fit the discretized ODE step-by-step instead of
   the per-lap aggregate closed-form. The same K and τ values come out but
   with much smaller stderr because we use ~thousands of samples per
   session instead of one number per lap. Also opens the door to within-lap
   T_hot predictions ("what's my FL temp at t=180 s into lap 3?"). The
   discretized recurrence is already implemented and tested in
   `energy_balance.py`.
8. **Hierarchical / Bayesian partial pooling.** Sparse (car, corner) buckets
   would benefit from shrinking toward a global mean — Motegi Inferno 86
   has only 18 usable laps. NumPyro Stage-2 partial pooling over scipy
   Stage-1 per-bucket fits is the canonical move. Probably worth doing
   once we have ≥ 3 cars in the dataset; with 2 cars, the Stage-2 prior is
   too narrow to learn much.
9. **Re-introduce tire compound.** When notes have ≥ 80% compound coverage
   with consistent normalization (or AIM logger reads compound RFID
   directly), and we have multiple compounds at the same (car, track) for
   identifiability, compound can come back as a `K[car, compound, corner]`
   dimension. v0 explicitly omits it.

### v2 — calculator integration

10. **C# calculator integration.** The committed `tire_model.json` is the
    integration hand-off. A follow-up plan adds:
    1. A versioned JSON loader on the C# side, sanity-checking
       `schema_version` and the Gay-Lussac block matches the calculator's
       own constants.
    2. UI inputs for track + car + lap + cloud cover.
    3. Auto-fill of the per-corner cold-pressure fields from the model
       prediction (still overridable).

## 6. CLI reference

```bash
# Detect candidate broken TPMS sensors (low-variance channels). Human-curated.
just tire-sensor-audit

# Fit the energy-balance model and write both artifacts
just tire-build-warmup-table

# Predict per-corner cold pressures for a target lap
just tire-predict --track tsukuba_2000 --car KK-SII --lap 5 --ambient 18 --hot-all 1.95
# (per-corner: --hot-fl 1.95 --hot-fr 1.95 --hot-rl 1.90 --hot-rr 1.90)
# (optional: --track-temp 35 --cloud-cover 30 --g2-typ 0.85 --lap-time-s 70)

# Predict per-corner cold pressures, with rain condition
just tire-predict --track tsukuba_2000 --car KK-SII --lap 5 --ambient 15 \
                  --condition damp --hot-all 1.5

# Predict with a target lap time (scales tire energy via the sector-fit
# g² vs lap-time curve and sets time-on-track t = N × target)
just tire-predict --track tsukuba_2000 --car KK-SII --lap 5 --ambient 15 \
                  --hot-all 1.7 --target-lap-time 60

# Predict with the tire compound (per-axle K override; --compound-front /
# --compound-rear for split setups)
just tire-predict --track sodegaura --car "Inferno 86" --lap 5 --ambient 22 \
                  --hot-all 1.9 --compound A052

# Held-out validation (train without N sessions per bucket, report per-corner T_hot residuals)
just tire-predict-holdout
# (--n-per-bucket 2 --min-bucket-size 10)

# Validate against notes-recorded cold pressures (consistency check, not held-out)
just tire-predict-validate
```

## 7. Files of interest

| Path | What |
|---|---|
| `src/motorsports_data_notebook/tire_model/energy_balance.py` | Pure physics functions (T_eff, warmup_curve, Gay-Lussac, T_road proxy, discretized recurrence) |
| `src/motorsports_data_notebook/tire_model/warmup_table.py` | Data prep + two-pass scipy fit + JSON serializer + sensor audit + blacklist |
| `src/motorsports_data_notebook/tire_model/predict.py` | `predict_cold_pressure(...)` and the fallback chain for K / τ / c_track / ⟨g²⟩ |
| `src/motorsports_data_notebook/tire_model/validate.py` | `tire-predict-validate` (notes-recorded ground truth) and `tire-predict-holdout` (held-out generalization test) |
| `data/tire_dataset/tire_model.json` | The committed fitted artifact (~6–12 KB, diff-friendly) |
| `data/tire_dataset/sensor_blacklist.yaml` | Human-curated list of broken (session, corner) channels |
| `tests/tire_model/test_energy_balance.py` | Physics functions in isolation |
| `tests/tire_model/test_warmup_table.py` | Synthetic-data round-trip: known K, τ, c_track → fit → recover |
| `tests/tire_model/test_predict.py` | Fallback chain hits every level with mocked artifacts |
