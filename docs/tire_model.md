# Cold tire pressure model — compound-aware (schema v3)

A physically-based predictor that takes
**(track, car, tire compound, target lap within stint, target lap time, target hot pressure per corner, expected ambient temp)**
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
from the tire's actual temperature `T_start` at roll-out:

```
T_hot(t) − T_eff  =  K · c_track · ⟨g²⟩ · (1 − exp(−t / τ_sec))
                     + (T_start − T_eff) · exp(−t / τ_sec)
```

The second term is the decaying memory of where the tire started. v0 assumed
`T_start = T_eff` (a cold tire in equilibrium with its surroundings). That is
only true for the first run of a day: at the start of the first full lap the
TPMS reads a median +11 °C above T_eff on the FJ and +18 °C on the Inferno 86
across all dry stints (the previous run's heat is still in the carcass, plus
whatever the out-lap added). Fitting the v0 form to warm starts forced τ short
(the optimizer explains a warm start as a fast warm-up) and left a bias that
depended on how long the car had been parked. **The fit now anchors every
stint on its first finite TPMS reading, and the predictor starts the curve
from the entered current tire temperature** (§2.6, §2.4).

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

The calculators' per-corner **"Current °C"** field (`cold_tire_temp_c` in the
Python API) overrides `T_cold`, and the same number is the warmup curve's
`T_start`: it answers one question, "what is this tire at right now?", and is
used consistently on both sides. Entering the pit-lane TPMS reading when the
tires are still warm from the previous run both raises the predicted hot
temperature (less warm-up left to do) and lowers the cold pressure to set. When
the field is left blank the predictor uses `T_air` for both.

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

0. **Stint anchor.** For each (session, stint, corner) the first finite TPMS
   reading becomes the initial condition `(t_a, T_start)`: the first full
   lap's start reading (t_a = 0) when it exists, otherwise that lap's end
   reading, and so on (TPMS channels are NaN for the first samples of nearly
   every session). Laps at or before the anchor are not scored; the first
   full lap *is* scored when the anchor sits at its start.

1. **Pass 1 — `τ_sec[car, corner]` + per-bucket gains.** For each (car, corner),
   select that car's (track) buckets with ≥ 30 lap samples. Fit jointly across
   them: `δT_i = gain_{bucket(i)} · g²_i · (1 − exp(−Δt_i / τ_sec)) + (T_start − T_eff) · exp(−Δt_i / τ_sec)`
   with `Δt_i = t_i − t_a`, per-lap `g²_i = heat_proxy_i / on_track_s_i`, a
   shared `τ_sec[car, corner]` and bucket-specific `gain_b = K · c_track`.
   KK-SII FL τ is fit jointly from Tsukuba + Fuji + Suzuka data —
   precisely the cross-circuit shrinkage we want. The anchor term is known
   per lap, so it adds no parameters; it roughly doubles the fitted τ
   (FJ 215–247 s → 408–571 s, Inferno 86 248–312 s → 432–552 s) and raises K
   by 5–15 %, because warm starts no longer masquerade as fast warm-ups.

2. **Pass 2 — factor `gain_b` into `K[car, corner] × c_track[track]`.** Divide
   out ⟨g²⟩ (a lookup) and use alternating least squares in log-space, with
   `c_track[tsukuba_2000] ≡ 1.0` anchored for identifiability. Standard errors
   propagate from Pass 1's bucket-gain stderrs.

If a (car, corner) bucket has fewer than `MIN_LAPS_FOR_TAU_FIT = 30` lap
samples, the fit returns the prior `τ_sec = 240 s, K = 60 K/G²` with
`from_prior: true` flagged in the artifact.

### 2.7 Artifact: `tire_model.json`

The model serializes to a versioned JSON committed under
`data/tire_dataset/tire_model.json` (`schema_version: 3`). Top-level keys:

Fitted tables:

- `tau_sec_by_car_corner_cond` — warmup time constants with stderrs
- `K_buckets` — pooled K per (car, corner, condition) with stderrs and
  `from_single_track` flags
- `K_by_car_compound_corner_cond` — compound-specific K (decomposed
  c_track × base × multiplier products, ready to use; see §2.11)
- `K_compound_multipliers` — the fitted per-(car, compound) ratios, for audit
- `c_track_by_track` — per-track surface scalars, Tsukuba marked as anchor
- `g2_typ_by_track_car_cond` — ⟨g²⟩ lookup; each entry may carry a
  `g2_vs_lap_time` piecewise-linear curve (see §2.10)
- `lap_time_typ_by_track_car_cond` — typical lap time lookup
- `corner_defaults_by_car_corner_cond` — steady-state median hot temp +
  hot pressure, used by the calculators to prefill the corner-card targets
- `rain_thermal` — how rain τ/K are fitted (τ bound, session minimum; §2.9)

Config + provenance:

- `car_aliases` — raw car label → pooled fit label (e.g. `KK-F` / `KK-SII`
  → `FJ`: near-identical FJ-series machines on the same tires train one
  car). All predictors resolve an input car through this map before any
  lookup, so raw names keep working; the fitted tables carry only pooled
  labels.
- `g2_lap_time_model` — pace-scaling method, default exponent, multiplier clamp
- `energy_balance` — the `w_road` config + T_road proxy formula
- `gay_lussac`, `conditions`, `corners`, `priors_when_no_fit`,
  `min_samples_per_bucket` — constants the consumers sanity-check against
- `fallback_order_for_K`, `fallback_order_for_condition_lookups` —
  declarative fallback chains
- `sensor_blacklist_applied` — audit trail of masked (session, corner) pairs
- `schema_version`, `fit_at_utc`, `model_form`

Typical file size: tens of KB. Diff-friendly — when new sessions land, the
artifact updates in lockstep and the JSON diff shows reviewers exactly which
buckets gained samples or changed coefficients. All three calculators read
this same file: the Python CLI, the C# Avalonia heads (embedded at build),
and the static web app (fetched at deploy).

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

### 2.11 Tire compound (one tire per run)

The Inferno 86 alternates between tire sets (A050, A052, and RE-71RS
through 2025-2026), and they heat very differently: session-median
implied K separates by ~45% on the rears (A052 ≈ 39-43, RE-71RS ≈ 59-61
K/G² dry) with within-compound spread of ±2-4 vs ±10 for the pooled mix.
The pooled K splits the difference and mis-predicts both — this was the
dominant source of the car's dry-rear MAE.

**One tire runs on all four corners, always** — there is no per-axle
splitting anywhere in the pipeline or the calculators, and no pooled
"default" choice in the UI: compound selection is forced per run.

Labels come from ``data/tire_dataset/tire_compounds.yaml`` (human-curated,
one ``compound:`` per session; authoritative) with the notes-extraction
compounds as fallback; wheel-set names from the notes ("Black wheels")
resolve through the sidecar's ``wheel_sets`` mapping. The artifact carries
the fitted per-(car, compound, corner, condition) K products in
``K_by_car_compound_corner_cond`` (additive to schema v3 — consumers
without compound support ignore the table).

Prediction: ``predict_cold_pressure(compound=...)`` (CLI ``--compound``)
swaps in the compound K on every corner when a fitted bucket exists
(condition chain applies); unknown compounds and unlabeled cars keep the
pooled K.

**Decomposed K with partial supervision**
(``tire_model/compound_infer.py``): the compound K is not fitted as free
buckets but decomposed as

    K_effective = c_track[track] · K_base[car, corner, condition] · m[car, compound]

so every lap of every tire informs the car's base K, and each compound is
one scalar multiplier shared across corners and conditions (fitted
Inferno ratios: A050 ×1.18, RE-71RS ×1.05, A052 ×0.86; identifiability:
lap-weighted geometric mean of m per car is 1, making K_base the
"average tire"). Labels are sparse, so the fit is a multi-task objective
— the compound-assignment task is supervised where labels exist and
latent elsewhere, sharing (K_base, m) with the temperature-regression
task. Solved by EM: labeled/seeded sessions pinned one-hot, free
(session, axle) units get posteriors from their lap residuals, and the
M-step alternates least squares on (K_base, m). Selection is FORCED —
every unit in a participating car carries an assignment — with two guard
rails: posteriors tempered to ``SESSION_EFF_SAMPLES`` effective
observations (laps within a session are correlated), and robust
down-weighting by best-fit χ²/lap so a session matching no known tire is
still assigned (flagged ``poor_fit``) but cannot drag a cluster toward
itself. Weather-driven tire choices use
declarative ``condition_seeds`` in the sidecar (KK-SII: all-dry session ⇒
DRY tires, all-wet ⇒ WET; damp/mixed left to the classifier — the EM
independently recovers the known 2026-04-04 rain day as WET at ≥0.998).
Soft assignments are training-only; held-out evaluation uses human/seed
labels exclusively. ``just tire-model infer-compounds`` audits every
latent assignment for human review.

Held-out CV, fleet pooled MAE at v0.20: FL 4.22 / FR 4.21 / RL 4.61 /
RR 4.18 °C with pooled bias within ±0.75 °C — versus FL 4.95 / FR 4.01 /
RL 5.63 / RR 4.70 before the compound era (v0.17), with the Inferno's
rear bias collapsing from ≈−5 °C to −0.7/−1.5 °C.

### 2.9 Track condition (rain)

v0.3 added a `condition` dimension derived from weather data, not run-notes.
Since 2026-10 the category is the **expected wetness of the track surface**,
not the rain rate of one hour (`tire_model/wetness.py`):

- Open-Meteo's hourly `precipitation` is the sum of the *preceding* hour, so
  the old "precipitation of the hour containing the session start" rule
  attributed showers that had ended before roll-out to the run (Tsukuba
  2025-07-26: 0.6 mm fell 13:00–14:00 JST, the car rolled out at 14:08 on a
  38 °C clear afternoon and ran 46 laps with the fronts at 90 °C; the rule
  said damp). It also called a trace of grid-cell drizzle "damp" regardless
  of whether the asphalt could possibly be wet.
- Instead we integrate a surface water balance over the hourly series:
  `d(t+1h) = clamp(d(t) + P − E, 0, 1 mm)`, with `E` a bulk evaporation rate
  from the vapour-pressure deficit between the sun-warmed surface
  (`T_air + 10·(1 − cloud/100)`, the same proxy as T_road) and the air, times
  a Penman-style wind factor. The clamp says a track holds about a
  millimetre before the rest runs off, and that a dry track has forgotten
  earlier rain. 0.6 mm evaporates in ~20 min at 38 °C / 36 % RH and takes
  hours at 10 °C under cloud.
- A session's class comes from the maximum film depth over the hours that
  overlap the run: `dry` < 0.05 mm ≤ `damp` < 0.5 mm ≤ `wet`; `unknown`
  when the weather series does not cover the session (excluded from
  training). The per-session depths are kept on the laps frame as
  `track_wetness_start_mm` / `track_wetness_run_max_mm` for a future
  continuous rain-cooling factor.
- The evaporation coefficient (0.2 mm/h per kPa) and storage depth are
  physical priors. Against the dry model's per-session residuals the water
  balance separates sessions in the physically expected direction
  (film present → over-predicted by ≈ +1 °C, dry → ≈ −1 °C) and explains
  about twice the session-level variance of the old labels, which is still
  little: most session-level error is compound and track structure, not
  rain. Weather explains little either way; the point of the change is to
  stop putting dry sessions in the rain buckets.

The condition's definition for the *user* is unchanged: at prediction time
they say how wet the track is.

**Where the condition enters.** ⟨g²⟩, lap_time_typ and the corner defaults
are per condition (drivers go slower and pull less g in the rain; those are
data lookups, not fitted parameters). **τ and K are fitted per condition**,
with the one bound the physics supports placed inside the Pass 1 fit:
rain only adds cooling (evaporation off the tread, conduction into a wet,
cold surface), so `τ_rain ≤ τ_dry` for any tire; `K = α/h` is left free
because a rain compound has its own hysteresis α. A rain (car, track)
bucket is fitted on its own only when it has ≥ 3 sessions and ≥ 30 laps
(`MIN_SESSIONS_FOR_RAIN_FIT`, `MIN_LAPS_FOR_TAU_FIT`); otherwise the
predictors' fallback chain resolves it to dry:

```
wet  → damp → dry      (physically closest neighbors)
damp → dry
dry  → dry
```

**History of this choice (2026-10).** With the old start-hour rain labels
the rain buckets were a mix of dry and wet days, an independent rain fit
predicted held-out rain sessions *worse* than reusing the dry parameters,
and v0.3's post-hoc clips (1.5× τ, 1.2× K) replaced one parameter after
the fit without refitting the other. Rain inherited dry for one release.
Once the condition came from the surface water balance above, the wet
class became coherent and the independent fit won on 21 held-out rain
sessions (wet bias +3.9 °C → −0.4 °C; with calculator inputs wet MAE
4.6 → 3.8 °C, FJ 4.5 → 2.8). The fitted wet gains are 0.6–0.95× dry with
τ at the dry bound — less heat per unit g² in the rain, as expected. The
clips are gone; the τ bound inside the fit replaces them.

**Fit dataset breakdown** (wetness classification, dataset through
2026-10-02):

| Condition | Sessions (FJ / Inferno 86) | Usable laps | Notes |
|---|---|---|---|
| dry | 53 / 46 | 1096 | all tracks |
| damp | 6 / 1 | 109 | a transition state: partial film, mostly early morning after overnight rain |
| wet | 14 / 8 | 212 | saturated film; Tsukuba 2025-10-25, 2025-12-14, 2026-04-04, 2026-09-26, Fuji 2026-02-25, Sodegaura 2026-02-07 |
| unknown | — | excluded | no weather coverage |

**Known limitation**: damp is thin (7 sessions) and the Inferno's wet
buckets are small (6 sessions); where a bucket falls back to dry the
prediction over-reads the hot temperature in the rain by a few degrees,
which skews the cold-pressure recommendation slightly low.

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

- **Bucket selection, stratified by condition.** Buckets are
  (track, car, condition). Dry buckets need ≥ 10 sessions to be eligible;
  damp and wet buckets are far smaller and need ≥ 3, otherwise no rain
  session would ever be held out and the rain numbers would be whatever
  happened to fall into the dry slices (before 2026-10 that was 3 damp
  sessions out of 30 held out). Sessions with unknown condition (no weather)
  are never held out; they are excluded from training too.
- **Session picking.** Within each eligible bucket the session_ids are
  sorted and `--n-folds` disjoint slices of `--n-per-bucket` are swept
  through them, so k-fold CV evaluates every session once until a bucket
  runs out. Deterministic, reproducible. The per-fold line prints how many
  dry / damp / wet sessions are held out.
- **Lap filtering.** The first full lap of each stint
  (`lap_within_stint == 0`) is not scored, so numbers stay comparable with
  the v0 reports; the stint's first finite TPMS reading is the initial
  condition for every scored lap — the same information a driver supplies
  as "Current °C" at roll-out. Laps at or before a corner's anchor are
  skipped.
- **Blacklist applied.** Confirmed broken sensors are masked in both
  training and evaluation — we don't grade the model against channels we
  already know are broken.
- **Inputs: calculator or oracle.** By default (`--inputs calculator`) each
  held-out lap is predicted from what the calculator has: the fold model's
  ⟨g²⟩ scaled along the pace curve at the session's median lap time (the
  target a driver would enter), the clock `N × lap time`, and the start
  temperature the driver types in, stood in by the stint's first TPMS
  reading (the start fields are always filled in practice; leaving them
  blank was measured at +1.2 °C MAE and dropped as an option).
  `--inputs oracle` instead feeds the lap's own
  measured g², its actual cumulative on-track time and the measured anchor:
  that is the thermal model's accuracy given the real driving, an upper
  bound on what the calculator can deliver. Numbers quoted before 2026-10
  in this document are oracle numbers.
- **Metric.** Per-corner per-lap **T_hot residual** (predicted minus
  observed end-of-lap TPMS temperature). MAE, RMSE, and mean signed bias
  reported per corner. T_hot is the right metric because everything
  downstream (Gay-Lussac, cold pressure) is a deterministic transform of
  it — predict T_hot well and the cold pressure is right.

### 3.3 Pressure is what matters: the hot-pressure residual

The holdout also scores every lap in the pressure domain, which is what the
driver actually gets. From the pressure and temperature at the stint anchor
it pushes the predicted hot temperature through the same constant-volume
step the calculators use and compares the implied hot pressure with the
TPMS hot pressure, in bar. Two columns are printed: as the calculators
compute it today (γ = 1) and with a per-car gain γ in `P_abs ∝ T_abs^γ`
fitted on the training sessions. γ is below 1 on both cars (TPMS reads the
valve, which runs warmer than the mean cavity gas), so the γ = 1 step
over-states the pressure rise and the per-car γ removes a systematic bias.
Fitting γ into the calculators is the natural next step; the holdout
already reports what it would buy.

### 3.4 What MAE in T_hot translates to in cold pressure

For target hot 1.9 bar (gauge), air 18 °C, T_hot ≈ 50–60 °C, the Gay-Lussac
inversion has

```
|∂P_cold / ∂T_hot|  ≈  (P_hot + 1) · T_cold_K / T_hot_K²  ≈  0.025 bar / °C
```

so an MAE of 2 °C in T_hot → roughly 0.05 bar in cold pressure (~0.7 psi).
An MAE of 4 °C → ~0.10 bar (~1.5 psi).

## 4. Results

### 4.0 Effect of the measured initial condition (3-fold CV, 1447 (lap × corner) points, dataset through 2026-10-02)

Same folds, same laps, same code apart from the stint anchor
(`just tire-predict-holdout --n-folds 3`):

| Corner | v0 form (T_start = T_eff) MAE / bias | anchored MAE / bias |
|---|---|---|
| FL | 4.68 / −0.47 °C | **4.16** / −0.23 °C |
| FR | 4.34 / −1.92 °C | **3.99** / −1.92 °C |
| RL | 4.82 / −2.76 °C | **4.05** / −2.22 °C |
| RR | 4.00 / −2.42 °C | **3.36** / −1.81 °C |

Per car: FJ 3.59 / 3.66 / 3.39 / 2.84 → 2.95 / 3.16 / 2.78 / 2.00 °C;
Inferno 86 5.76 / 5.14 / 6.60 / 5.38 → 5.37 / 4.98 / 5.63 / 4.98 °C. Damp
sessions improve most (FL 7.7 → 5.8 °C). The remaining Inferno error is
concentrated in a few long Sodegaura/Fuji sessions that both forms
over-predict by 5–7 °C. Note that the anchor in this evaluation is the TPMS
reading at the start of the first full lap, i.e. after the out-lap; a
pit-lane reading entered by the driver carries a little less information.

A state-space refit of the same energy balance on the 1 Hz timeseries
(speed-dependent cooling, g²·v drive, rolling term, load-transfer corner
split, dropping c_track) was evaluated alongside this change; only the
measured initial condition and a per-car pressure–temperature gain survived
held-out testing, so the other ideas were not adopted.

### 4.1 Headline (v0.20 held-out, pooled, 236 (lap × corner) points — pre-anchor)

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
   the per-lap aggregate closed-form. Tested offline (2026-10): under the
   same assumptions it does **not** beat the per-lap closed form at
   predicting end-of-lap temperatures, so it is only worth doing for
   within-lap T_hot predictions ("what's my FL temp at t=180 s into lap
   3?"). The discretized recurrence is already implemented and tested in
   `energy_balance.py`.
8. **Hierarchical / Bayesian partial pooling.** Sparse (car, corner) buckets
   would benefit from shrinking toward a global mean — Motegi Inferno 86
   has only 18 usable laps. NumPyro Stage-2 partial pooling over scipy
   Stage-1 per-bucket fits is the canonical move. Probably worth doing
   once we have ≥ 3 cars in the dataset; with 2 cars, the Stage-2 prior is
   too narrow to learn much.
9. ~~**Re-introduce tire compound.**~~ **Done** (v0.19–v0.23): compound is
   a fitted dimension via the decomposed `c_track × K_base × m[car,
   compound]` with partial supervision — see §2.11.
10. **Compound-conditioned pace curves.** The `g2_vs_lap_time` curves pool
    all compounds per (track, car, condition); once labels are denser they
    could split per compound.

### v2 — calculator integration (done)

The committed `tire_model.json` was the integration hand-off, and all
three planned pieces shipped: a versioned loader on the C# side
(`Core/Services/Modeling/TireModelLoader.cs`, plus the same in
`web/js/model.js`), prediction-panel inputs (track → car → tire →
condition → lap → ambient → cloud → target lap time), and model-driven
prefill of the corner-card targets from
`corner_defaults_by_car_corner_cond` (still overridable).

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

# Predict with the tire compound (one tire on all four corners)
just tire-predict --track sodegaura --car "Inferno 86" --lap 5 --ambient 22 \
                  --hot-all 1.9 --compound A052

# Held-out validation (train without N sessions per bucket, report per-corner T_hot residuals)
just tire-predict-holdout
# (--n-per-bucket 2 --min-bucket-size 10 --n-folds 3 --inputs calculator|oracle)

# Validate against notes-recorded cold pressures (consistency check, not held-out)
just tire-predict-validate
```

### 6.1 Circuit registry and variant reconciliation

The AIM dash's track selection decides where the lap beacon sits and what the
filename / `Venue` metadata say — and it is often the wrong *variant* of the right
venue (full Suzuka laps logged as "Suzuka West", full Motegi laps as "Motegi
East"), or no track at all ("Generic testing"). `tire_etl/layouts.py` is the
canonical circuit registry: per venue, the start/finish **gate** of every layout
(measured from GPS) and the set of gates each layout crosses per lap. At extract
time the GPS trace picks the layout actually driven (most specific layout whose
gates were all crossed), `track_canonical` is coerced to the venue's pooled id,
and when the logger's beacon was on another layout's line the laps are re-split
at the correct start/finish. Layouts of one venue still pool into one model
bucket (Fuji GP/Short, Motegi/East, Suzuka full/West) — `layout_id` is recorded
per session so that pooling can be revisited.

```bash
just tire-track-audit [--since YYYY-MM-DD]   # every session reconciled from GPS, or unresolved
```

## 7. Files of interest

| Path | What |
|---|---|
| `src/motorsports_data_notebook/tire_model/energy_balance.py` | Pure physics functions (T_eff, warmup_curve, Gay-Lussac, T_road proxy, discretized recurrence) |
| `src/motorsports_data_notebook/tire_model/warmup_table.py` | Data prep + two-pass scipy fit + JSON serializer + sensor audit + blacklist |
| `src/motorsports_data_notebook/tire_model/sectors.py` | Sector-wise pace model: per-lap sector split + kNN-median `g2_vs_lap_time` curves |
| `src/motorsports_data_notebook/tire_model/compounds.py` | Compound label loading (sidecar + notes fallback, wheel-set mapping, condition seeds) |
| `src/motorsports_data_notebook/tire_model/compound_infer.py` | Decomposed compound K: EM with partial supervision, forced selection |
| `src/motorsports_data_notebook/tire_model/predict.py` | `predict_cold_pressure(...)` and the fallback chain for K / τ / c_track / ⟨g²⟩ |
| `src/motorsports_data_notebook/tire_model/validate.py` | `tire-predict-validate` (notes-recorded ground truth), `tire-predict-holdout` (held-out generalization test, temperature and pressure domains) |
| `data/tire_dataset/tire_model.json` | The committed fitted artifact (diff-friendly) |
| `data/tire_dataset/tire_compounds.yaml` | Human-curated compound history: per-session `compound:`, `wheel_sets`, `condition_seeds` |
| `data/tire_dataset/sensor_blacklist.yaml` | Human-curated list of broken (session, corner) channels |
| `scripts/regen_tire_predict_fixture.py` | Regenerates the Python-parity fixture pinned by the C# and web test suites |
| `tests/tire_model/test_energy_balance.py` | Physics functions in isolation |
| `tests/tire_model/test_warmup_table.py` | Synthetic-data round-trip: known K, τ, c_track → fit → recover |
| `tests/tire_model/test_compound_infer.py` | EM recovery on synthetic mixtures: pinned labels, latent posteriors, forced selection |
| `tests/tire_model/test_predict.py` | Fallback chain hits every level with mocked artifacts |
