# Cold tire pressure model — physical heat input, no track constants (schema v5)

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
  car, not the venue. The track effect enters through data (the per-corner ⟨q⟩),
  not through track-specific fitted coefficients.
- **Tire compound omitted.** Notes-derived compound coverage is only 49 of
  142 ok sessions (34%) and strings are messy. Including it as a model
  dimension makes K + compound mutually unidentifiable in the
  current data. Deferred until coverage improves.
- **Rain awareness via the weather data, not the run-notes.** v0.3 adds
  `condition ∈ {dry, damp, wet}` as a model dimension, classified from
  Open-Meteo's `precipitation` field (mm/hr). 75% of sessions have weather
  coverage vs only ~25% with notes-derived condition. See §2.9.

## 2.0 The schema-v5 model in one page (2026-10-09)

Per corner ``i`` of a car, on axle ``A(i)``, with ``T_i`` the corner's
cavity-gas temperature (°C, the quantity the TPMS pressure reads, §2.2a)
corrected for the reading at speed (§2.2a′), fitted **per second** on the
1 Hz stint series (§2.6a)::

    dT_i/dt = a · q_i(t) − b_A · (T_i − T_eff)                 [K/s]

    q_i = V · √( F_y,i² + F_x,i² )                              [G·m/s;  V in m/s]

    lateral:  F_y,i = p_A · λ_i · |lat|
              λ_i = ½ (1 + tanh(|lat| / g_transfer))   outer tyre of the axle (the side the turn loads)
              λ_i = ½ (1 − tanh(|lat| / g_transfer))   inner tyre
              p_front = p_f,  p_rear = 1 − p_f
    braking:  F_x,i = ½ · β_A · |long|                  (long < 0;  β_front = brake bias, β_rear = 1 − β_front)
    accel:    F_x,i = ½ · long                          (long > 0;  driven axle only)

**Heat input: force × slip fraction × speed.** A race tyre runs near its
optimum slip, a kinematic fraction that does not grow with force, so the
power dissipated at the patch is the force the corner produces times that
fraction times the rolling speed. ``F_y,i`` and ``F_x,i`` are the corner's
force components as fractions of car weight (in G), built from the car's
measured accelerations by **bounded shares**: fractions in [0, 1] that
conserve across the axle and across the car (no corner can receive more
than the whole force) and saturate as a tyre unloads. Lateral and
longitudinal combine in quadrature because they are the two components of
one force vector at the contact patch, and a unit of force heats the same
whatever its direction: there are no fitted brake or drive efficiencies
(§2.0c).

**Energy out.** ``T_eff = 0.8·T_air + 0.2·T_road``: the tyre loses heat to
the air over most of its surface and to the asphalt at the patch. Cooling
is per axle because the rears sit in the car's wake.

**No track constants.** The circuit enters only through data: the typical
per-corner intensity ``q_typ_by_corner`` per (track, car, condition) and
the out-lap intensity ``outlap_q_by_corner``, computed from the fitted
stints' own laps, so direction and corner mix are in the numbers (Tsukuba
gives the FJ 8.5 G·m/s on the front-left against 6.6 on the front-right;
Suzuka's figure-eight is balanced at 8.0 / 8.0).

### 2.0a The constants: units and what they are

| Constant | Per | Units | Fitted (FJ / Inferno 86) | What it is |
|---|---|---|---|---|
| ``a`` | car, condition | K/s per (G·m/s) | 0.0134 / 0.0182 | **Heating rate**: heat input per unit intensity over the effective heat capacity of the carcass + gas the pressure reads. At 1 G of force and 30 m/s (``q`` = 30) the FJ's gas warms ≈ 0.40 K/s, the Inferno's ≈ 0.55 K/s, with cooling switched off. |
| ``b_A`` (reported as ``τ_A = 1/b_A``) | car, axle, condition | 1/s (τ in s) | FJ front 433 s / rear 282 s; Inferno 573 / 560 s | **Cooling rate**: conductance to air and road over the same heat capacity; τ is the time to close 63 % of the gap to the steady state. The open-wheel FJ's exposed rears cool 1.5× faster than its fronts; the Inferno's axles match. |
| ``K_A = a / b_A`` | car, corner (derived) | K per (G·m/s) | FJ 5.8 / 3.8 (f/r); Inferno 10.4 / 10.2 | **Steady-state gain**: the equilibrium rise above ``T_eff`` per unit intensity; what the calculators multiply ``q_typ`` by. |
| ``g_transfer`` | car | G | 2.43 / 3.13 | **Lateral transfer scale**: the lateral g at which the outer tyre carries ≈ 88 % of its axle's load. The rigid-body lift acceleration ``track/(2·CG height)`` is its geometric floor: ≈ 2.4 G for the FJ, which the fit reproduces; the Inferno's 3.1 G against a geometric 1.6 G says tyre load sensitivity and the roll-stiffness split soften the transfer the heat sees. |
| ``κ`` | car | 1/(m/s)² | 2.0 × 10⁻⁶ / 1.0 × 10⁻⁵ | **Speed–pressure constant**: ``P_read = P_gas/(1 + κV²)``, the fractional drop of the TPMS reading per squared speed from centrifugal tyre growth and the sensor's g-load; 0.6 % and 3 % at 200 km/h. |
| ``p_f`` | car | fraction | 0.42 / 0.53 | Static front weight share. | 
| ``β_front`` | car | fraction | 0.58 / 0.65 | Front brake bias. |
| driven axle | car | | rear / rear | By drivetrain. |
| ``w_road`` | all | fraction | 0.2 | Road share of the cooling sink. |
| compound multiplier ``m`` | car, compound | dimensionless | FJ DRY 1.09, WET 0.74; A050 0.98, A052 0.98, RE-71RS 1.07 | Ratio of a compound's ``K`` to the car's base ``K``. |
| ``q_typ_by_corner`` | track, car, condition, corner | G·m/s | e.g. Tsukuba FJ FL 8.5, FR 6.6 | Not fitted: the median per-lap intensity of each corner on that circuit. |

Four fitted constants per car (``a``, ``b`` front and rear, ``g_transfer``)
plus κ; three car facts; nothing per track. The last three rows are car
facts and data, not fitted.

### 2.0b What is deliberately not in the model, and why

Each of these was built, run on session folds and leave-one-track-out,
and rejected (details in §2.6b–c and the history below):

- **A per-track constant** (``c_track``). With the sliding-power input the
  fitted factors collapsed to 1 ± 7 % and pinning them cost nothing on
  known circuits while improving unseen circuits by 18 %. A constant that
  can only be fitted after visiting a track is not a constant.
- **Longitudinal transfer of the lateral share** (fronts take more lateral
  force under braking, rears under acceleration). Physically real, but the
  gas temperature cannot see it: fitted, the Inferno's scale ran to its
  bound (no effect) and the FJ's to an unphysical 0.67 G while trading off
  against the drive and brake efficiencies; fixed at the geometry it cost
  0.0004 bar. The combined-slip transient lasts seconds and the carcass
  node averages it away. Left out rather than carried as a dead parameter.
- **A reference speed** in ``q``: a pure normalisation; removed, ``q`` is
  in G·m/s.
- **Fitted brake and drive heat efficiencies** (how much heat a unit of
  braking or drive force makes relative to cornering force). Fitted per
  car they came out at 2.0–2.3 for braking and 0.02–0.6 for drive and
  improved known circuits by 0.6 %, but predicted an unseen circuit 6.4 %
  *worse* than efficiencies of 1 (§2.0c). A per-car constant that only
  helps on the circuits it was fitted to is a track constant hiding in the
  car: it was absorbing the training circuits' braking-zone mix. With them
  gone the FJ's ``g_transfer`` lands on its geometric value. Removed
  2026-10-09; a unit of force heats the same in every direction.
- **g² as the heat input.** The small-slip brush model (slip ∝ force)
  double-counts the force in the racing regime; |g|·V scored better,
  tighter per-track ratios and a wet/dry ratio of 1.
- **Speed-weighted and slip-activated inputs on the carcass node**
  (ChassisSim form, power laws), **speed-dependent cooling**, and the
  **two-node tread model** for pressure: either worse, or invisible to the
  gas, or both (§2.6b–c).
- **Set pressure as a heat factor.** Strong in the data (−7.5 %) but failed
  the instantaneous-pressure and cross-car controls; it may be the driver
  anticipating an unmeasured tyre state. Kept as a harness option only.
- **Per-corner free gains and free force shares.** Each buys 1–4 % on
  known circuits by letting constants float (per-corner gains encode the
  training circuits' direction mix; seven free shares are degenerate in
  brake bias vs brake efficiency and unstable under leave-one-track-out).

Held-out results for the promoted model are in §4.0b.

### 2.0c Ablation of the fitted constants (2026-10-09)

One constant removed per variant (replaced by its neutral value: 1 for a
rate, 0 for κ, ∞ for the transfer scale), everything else refitted;
held-out lap-end pressure MAE on 5 session folds and on
leave-one-track-out (unseen circuit), harness inputs. Ranked by the
damage its removal does:

| Removed | Session folds (bar) | Δ | Unseen circuit (bar) | Δ |
|---|---|---|---|---|
| none (production) | 0.0513 | | 0.0595 | |
| ``a`` = 1 | 3.05 | ×60 | 3.23 | ×54 |
| ``b`` = 1 (τ = 1 s) | 0.283 | ×6 | 0.299 | ×5 |
| ``g_transfer`` → ∞ (½/½ axle split) | 0.0526 | +2.5 % | 0.0620 | +4.2 % |
| κ = 0 | 0.0525 | +2.3 % | 0.0587 | −1.3 % |

Reading: the energy balance itself (``a``, ``b``) is what the accuracy
rests on. With ``a`` pinned the cooling rate would need a time constant of
seconds to compensate, below the fitter's 30 s floor, and the prediction
runs away; with ``b`` pinned the tyre tracks its steady state instantly
and the warm-up dynamics are gone. ``g_transfer`` is worth 2.5 % on known
circuits and 4.2 % on an unseen one: it transfers. κ is worth 2.3 % on
known circuits and nothing on an unseen one; it is kept because it is a
measured instrument effect (§2.2a′) and the production holdout is scored
on the corrected reading.

**The study that removed the efficiencies.** The same ablation run on the
previous production model (two fitted heat efficiencies ``ε_brake``,
``ε_drive`` per car in addition to the above; session folds 0.0510,
unseen 0.0636): ``ε_brake`` = 1 scored 0.0505 / 0.0587 (−1.0 % / −7.7 %),
``ε_drive`` = 1 scored 0.0516 / 0.0638, both = 1 scored 0.0513 / 0.0595.
The fitted efficiencies were per car, not per circuit, yet only helped on
the circuits they were fitted to: a track constant hiding in the car.
Removing both cost 0.6 % on known circuits, gained 6.4 % on an unseen
circuit, and moved the FJ's ``g_transfer`` from 1.95 G to its geometric
2.43 G. Removed.

### 2.0d The cooling side: what was tested (2026-10-09)

The cooling term is one conductance per axle to one sink,
``b_A·(T − T_eff)`` with ``T_eff = 0.8·T_air + 0.2·T_road`` hand-set. It
lumps convection off the tread, conduction into the asphalt and the bead →
rim → hub path; the two-node experiment (§2.6c) puts the tread's own time
constant at ≈ 15 s with the carcass holding 8× its capacity, so the
minutes-long ``τ`` the pressure sees is the carcass + rim reservoir. Each
candidate below was fitted on the production model (harness options
``--fit-w-road``, ``--cool-mode speed|speed_axle|speed_car|power``),
scored on 5 session folds and leave-one-track-out:

| Variant | Session folds (bar) | Unseen circuit (bar) | Fitted (full; fold range) |
|---|---|---|---|
| production (linear, ``w_road`` 0.2) | 0.0513 | 0.0595 | |
| sink weight ``w_road`` fitted | 0.0513 | 0.0600 | 0.10 (0.07–0.14) |
| speed convection ``b·(1+βV/30)/(1+β)``, one β | 0.0513 | 0.0607 | 0 (0–0) |
| speed convection, β per axle | 0.0508 | 0.0624 | front 0, rear 1.2 (0.5–2.0) |
| speed convection, β per car | 0.0514 | 0.0647 | FJ 0.05, Inferno 0 |
| **nonlinear ``b·(ΔT/20 K)^(n−1)``** | **0.0509** | **0.0592** | **n = 1.38 (1.36–1.40)** |
| nonlinear + sink weight | 0.0509 | 0.0595 | n 1.41, ``w_road`` 0.02 |

- **Speed-dependent convection is rejected again**, now on the current
  model: the shared slope sits at its zero bound in every fold, and the
  per-axle and per-car versions are unstable and worse on an unseen
  circuit. The reservoir the pressure sees does not feel the airstream.
- **The sink weight is not identifiable.** The track temperature in the
  dataset is not a measurement: it is ``T_air + 10 K·(1 − cloud/100)``
  from the hourly weather model, so the air/track ratio ``w`` is really
  the coefficient of a clear-sky solar term. Fitted one shared value it is
  0.10 (folds 0.07–0.14); per car FJ 0.15, Inferno 0.00 (folds 0–0.14);
  per axle front 0.00, rear 0.26 (folds 0.14–0.34); with the nonlinear
  cooling law 0.02. Session-fold MAE is 0.0513–0.0514 bar and
  unseen-circuit MAE 0.0595–0.0600 bar in every case, against 0.0513 /
  0.0595 at the hand-set 0.2. The sink is the air temperature to within a
  couple of kelvin and the data cannot say more; values pinned at the
  zero bound are the fitter finding nothing. Left at 0.2. A measured track
  temperature (an IR road sensor, or the tyre IR sensors' ambient reading)
  is what would make this ratio fittable.
- **A solar-geometry sink** (``solar.py``, harness ``--fit-sun``): the
  flat 10 K clear-sky offset has no sun height in it, so the proxy is
  replaced by ``T_eff = T_air + c_sun · sin(elevation)⁺ · (1 − 0.75·N^3.4)``
  with the elevation from the session start time and the track's
  coordinates and ``N`` the hourly cloud fraction (Kasten–Czeplak). The
  index spans 0–0.96 over the fitted stints (median 0.23; half the
  sessions are overcast) and is only 74 % correlated with the flat proxy.
  Fitted shared, ``c_sun`` = 4.0 K (folds 2.9–4.7): session folds 0.0511
  bar (−0.4 %), unseen circuit 0.0591 (−0.7 %), against 0.0513 / 0.0599
  with the sink at the air temperature alone and 0.0513 / 0.0595 with the
  flat proxy. Per axle (front 6.0 K, rear 2.2 K, rear folds 0.8–4.3) buys
  nothing. The value is physically coherent: an asphalt 20 K above air at
  overhead sun seen through the hand-set 0.2 patch weight is the same 4 K.
  So the gas sink does respond to sun and cloud, by at most ≈ 4 K between
  a dark overcast day and clear summer noon (≈ 0.035 bar on the hot
  reading), and that is the whole size of the effect; the flat proxy
  misses it by up to 2 K on a clear noon and 1 K on a clear winter
  morning. Not promoted yet: the calculators would need the date, time and
  track coordinates to evaluate the sun, in all three ports.
- **Is the cloud cover worth anything?** Sun height alone, with the cloud
  attenuation removed (``--fit-sun clear``, ``c`` = 4.9 K), scores 0.0522
  / 0.0617: *worse* than the air temperature alone (0.0513 / 0.0599),
  because half the sessions are overcast and a clear-sky offset is wrong
  on every one of them. With the cloud attenuation the same term scores
  0.0511 / 0.0591. So the hourly cloud cover is the informative half of
  the solar term: it is what says whether the sun reaches the asphalt. On
  its own (the flat proxy) it is worth 0.7 % on an unseen circuit; paired
  with the sun height, 1.3 %. The cloud cover also feeds the surface
  wetness model that classifies sessions dry / damp / wet (``wetness.py``),
  which is a separate and larger use.
- **A cooling exponent of 1.38 is the one cooling change that helps on
  both known and unseen circuits**, by under 1 %, and it is the most
  stable constant in the model (±0.02 across folds). Heat loss growing as
  ΔT^1.25–1.33 is the natural-convection law (laminar to turbulent), and
  taken with the zero speed slope it paints one picture: the carcass + rim
  reservoir sheds its heat through the wheel's inner surfaces, where the
  air is slow, not off the tread into the airstream. With it, ``a`` falls
  15 % and the 20 K-reference ``τ`` lengthens to 627/405 s (FJ) and
  924/909 s (Inferno), the effective rate at a 40 K gap being 1.3× that.
  Not promoted: the gain is below 1 %, and the steady state the
  calculators use would change from ``T_eff + K·q`` to
  ``T_eff + 20·(K·q/20)^(1/n)`` in all three ports.

A direct check of the same law (``(a·q − dT/dt)/ΔT`` on 31 s windows,
production constants) shows the implied rate 2–3× the fitted one at gaps
under 10 K and 0.9× above 40 K. That is the opposite sign to ``n`` > 1
and is the tread → carcass lag at the start of a stint (heat arrives
later than ``q`` says), not the cooling law; the fit, which integrates
through the lag, resolves it the other way. The same windows show cooling
3–5× slower below 15 m/s, but those are < 1 % of the seconds (pit lane),
which is why the fitted speed slope cannot see them.

### 1.0a Raw dataset, training-time cleaning (schema 4, 2026-10-12)

The extractor writes what the logger recorded (downsampled to the TPMS
module's 1.25 Hz clock until schema 6; verbatim at every channel's native
rate since schema 7, §1.0d), with laps and stints split, the circuit
resolved from GPS, channels selected and renamed, and a raw per-lap summary. Nothing in
`data/tire_dataset/` is masked, filtered or judged. Every rule about which
samples and laps the model learns from lives in
`src/motorsports_data_notebook/tire_model/cleaning.py` and runs when the
model loads the data (`warmup_table._load_filtered_laps`, the holdout and
notes-validation loaders, `statespace._session_timeseries`):

1. **Stale TPMS readings** (`cleaning.mask_stale_tpms`, the one entry
   point for both consumers of the streams: the lap aggregates and the
   per-second fit's loader; §1.0a′).
   - *At the start of the recording* (`mask_stale_session_prefix`). A
     module asleep at roll-out repeats its stored value bit-exactly until
     it wakes; the first change on either of a corner's two streams proves
     the module awake, and everything before it is masked on both streams.
   - *At the start of every stint* (`mask_stale_stint_starts`, #371,
     2026-11-26). On the stint's first lap, a first pressure change of 3
     counts or more within 60 s of the first sample masks every earlier
     sample of the lap on both streams; the pressure is unmasked at its own
     first change, not at the temperature's.
2. **Per-lap TPMS summaries rebuilt** from the masked streams, so an
   out-lap's `tpms_*_start` — the stint anchor — is the first live reading.
3. **Lap usability** (`tire_usable` / `exclude_reason`): out-laps need
   60 s of rolling time, in-laps are excluded, laps shorter than 30 s or
   slower than 1.4× the stint's best are dropped, and a lap needs a TPMS
   pressure range above 0.003 bar on at least one corner. The stint's best
   is of its flying laps only, out-laps and in-laps left out (#391,
   2026-11-26). A short partial in-lap had been the "best" of 15 stints in
   the dataset. In the TPMS sessions it dropped all 13 flying laps of the
   Inferno 86's a0837 (Sodegaura, 2025-07-24: a 54.6 s in-lap against 79 s
   laps), and on 2026-11-25 all four of the KK-F-V2's a0436 (72.4 s against
   113.7 s).
4. **A corner losing gas** (`cleaning.drifting_tpms`, #373, 2026-11-26;
   §1.0a″): a tyre whose amount of gas falls against its siblings' through
   a session (a leak, or a transducer that drifts) has its TPMS columns
   masked for that session, at training and scoring alike.

A cleaning change is therefore a code change plus a retrain; the dataset
is re-extracted only when the extractor's own output (lap splitting,
channels, storage layout) changes. `just tire-clean-laps` materialises the
cleaned table for DuckDB queries. Before schema 4 the extractor applied
steps 1 and 3 and committed the result, which is why the 2026-10 anchor
fix first looked like a dataset rebuild.

### 1.0a′ The stale packet at a stint's start (#371, 2026-11-26)

**The mechanism** (measured on the raw streams of the 128 held-out sessions
by the 2026-11 committee). The sensors go silent at rest: during 1–5 min
stops a pressure change is logged in 0–12 % of stops, though the tyres are
cooling. They wake once the car passes about 20–30 km/h; the impossible
jumps (85 mbar or more within seconds) arrive a median 7 s and 36 m after
roll-out, and at the start of a recording a corner's first change comes a
median 9 s and a p99 of 131 s after the car first passes 20 km/h. Until then
the logger keeps the last reading it had: every value change falls on a
multiple of 0.8 s, a 1.25 Hz CAN value sampled and held (the KK-SII's kit
is named as the Trailbrake TPMS in `builtin_profiles.yaml`; the Inferno
86's is not named but behaves the same). After a stop in the middle of a
session the next stint therefore starts on the hot pressure of the pit
entry (in 8 of 39 held-out cases exactly the previous stint's last
reading), and the session-start rule, which ran once per recording, took it
as the stint's anchor and over-predicted the stint from lap 1. It also
unmasked both streams at the first change of either, and the two do not
always refresh together: on the Inferno 86 about one pressure reveal in six
came after the temperature had refreshed, a median 27 s later (a1603 FR held
1.41 bar until 43.6 s while its temperature updated at 22 s). **One count of
the pressure reading is 31.3 ± 0.2 mbar**, logged to 0.01 bar, so it shows
as a step of 0.03 or 0.04 bar, on every kit in the dataset (not the 32.6
mbar of some of the committee's notes).

**The rule** (rule A, pre-registered by the committee; the owner's choice,
2026-11-26: "3 step rule is ok, go with that"). After the session-start
rule, per stint and corner, on the stint's first lap (the lowest `lap_num`
with its `stint_id`): take the corner's finite pressure samples in time
order and the first that differs from the first. If it differs by 3 counts
or more (`TPMS_PRESSURE_STEP_BAR` × 2.5 = 78 mbar, between the logged
two-count steps of 0.06–0.07 bar and the three-count ones of 0.09–0.10)
within 60 s of the first sample, every earlier sample of the lap is masked
on both streams, so the pressure is unmasked at its own first change.
Three counts need a true change of more than two, about 65 mbar or 8 K of
gas within a minute, which the single-node model does not allow (τ 250–640
s); two counts a hot restart can make. This implementation masks the same
57 stint × tyres in 30 sessions as the committee's patch (193 at 2 counts,
37 at 4), 47 of them on the Inferno 86: 22 start a stint after a stop, and
35 are on a session's first lap, which the session-start rule had let
through (16 of them because the temperature refreshed first).

**The first-lap anchor's time** (`warmup_table._compute_stint_anchor`,
changed with the mask). When a stint has no out-lap reading, the anchor
falls back to the first lap with a reading, and its start reading was
stamped at that lap's start; a reading first valid mid-lap had its value at
the wrong time, by up to a lap. It is now stamped as a pit exit's is: at its
own first-valid time on the rolling clock, less any standstill, from the
lap's start. A correctness fix, kept on principle (its measured effect is
below).

**Measured in the tyres' frame** (2026-11-27). On #376 and its refit, the
standing holdout's 15 folds were refitted with the same fold ids, #388's
count of flying laps, and the same 5,751 rows. MAE in mbar, #376 → rule A;
the SE is session-clustered / date-clustered:
- current calculator 43.60 → 37.53 (paired −6.07, SE 2.15 / 1.71; per
  session −4.80 ± 1.31 by date; nearer on 87 of 128 sessions);
- pace table 43.76 → 37.57 (−6.18 ± 2.13 / 1.68);
- direct 43.84 → 37.68 (−6.16).

By car, current calculator:
- FJ 35.18 → 31.19, Inferno 86 54.99 → 46.11;
- raw cars: KK-SII 31.34 → 29.68, KK-F 42.57 → 34.08.

By condition:
- dry 43.19 → 36.67 and damp 75.1 → 44.4;
- wet 37.73 → 40.33 loses (+2.60 ± 1.20 / 1.22), as it did in the sensor's
  frame.

The table's gap to the current calculator goes from +0.16 to +0.04. The
numbers below are the sensor frame's, measured before #376 and before #388.

**Measured** (the standing holdout, `check-pressure-holdout`, every fold
refitted; 15 folds, 128 sessions, the same fold ids in every run; MAE of the
hot pressure in mbar; the change against before #371 is the paired per-row
change with a date-clustered SE and a session-bootstrap 95 % interval; MDE
≈ 2.8 × 0.06 × the rms change of the predictions):

| | rows | current | pace table | direct | FJ (current) | Inferno 86 (current) |
|---|---|---|---|---|---|---|
| before #371 | 5,751 | 43.71 | 44.64 | 44.74 | 35.29 | 55.11 |
| rule A, anchor stamp as before | 5,751 | 37.47 | 38.38 | 38.50 | 31.20 | 45.96 |
| **rule A with the stamp fix (as committed)** | 5,751 | **37.48** | **38.40** | **38.52** | 31.18 | 46.00 |
| either-stream rule with the stamp fix | 5,751 | 40.66 | 41.51 | 41.60 | 31.14 | 53.53 |
| rule C with the stamp fix | 5,702 | 37.86 | 38.48 | 38.58 | 31.52 | 46.32 |

- Before → rule A as committed, on the same 5,751 rows: current −6.23 mbar
  (SE 1.73, bootstrap −10.45 to −2.34, MDE 6.2; better on 78 of 128
  sessions), pace table −6.24 (SE 1.73). Bias +7.0 → +3.8 mbar. By raw car:
  KK-SII 31.76 → 30.03, KK-F 42.06 → 33.38, Inferno 86 55.11 → 46.00. Damp
  rows 73.2 → 40.4, dry 43.4 → 36.8; wet rows lose, 37.7 → 40.1.
  `tire-predict-holdout --n-folds 20` gives the calculator arm's rows and
  per-corner MAE and bias exactly (checked on the either-stream code below;
  the scoring is the same code).
- The stamp fix alone: +0.00 ± 0.02 mbar (pace table +0.02 ± 0.03),
  predictions moved by 0.6 mbar rms: under rule A few stints' first live
  reading falls past the out-lap.
- The simulation-based calculator's gap to the current one: +0.93 before,
  +0.93 after.
- Within-stint slope of a tyre's hot pressure (laps ≥ 3) on its pit-exit
  deviation from its siblings: 0.32 before, 0.42 after (112 stints; the
  committee had predicted 0.7 or more).
- The fitted constants, fold by fold (median over the 15 folds, before →
  after): the FJ's speed-pressure κ 5.7e-7 → 5.0e-7, the Inferno 86's
  1.03e-5 → 9.0e-6; g_transfer 2.49 → 2.55 (FJ) and 3.89 → 4.92 (Inferno 86).
- Most of the gain exists because the holdout takes the TPMS's first
  reading as the pressure the driver typed; in use the driver sets the
  pressure with a gauge. The part from the refitted constants, which
  carries to real use, was −0.95 (not significant) of the −6.0 on the
  committee's rows, which also found that the refit takes out most of a
  bias: the constants had been fitted on stale anchors (the session's own
  heat under-predicted by 7.7 mbar on masked anchors, 2.8 after the
  refit).

**Two other rules measured beside it**, built on the owner's account of the
sensors (a sensor reports the last reading it had until it wakes; until a
value has changed once it is stale, likely too high):
- *Rule C*: per stint and corner, both streams masked before the later of
  the two streams' first changes, of any size, with no timer; a stream that
  never changes in the stint drops the stint × tyre. 38.25 with the old
  anchor stamp (+0.68 ± 1.09 against rule A on the rows both score, all of
  it in the 69 held-out stint × tyres whose out-lap it masked entirely),
  37.86 with the stamp fix. It drops 188 stint × tyres and holds an anchor
  for a temperature that ticks over by a single count long after the
  pressure (198 of 748 stint × tyres, a median 68 s, a 1 °C step in 193 of
  them).
- *The either-stream rule* ("if either changes, it is live"): per stint and
  corner, both streams masked before the first change of either; 40.66 with
  the stamp fix (−3.05 against before, SE 1.28; +3.19 against rule A, SE
  1.49). Its known exception is a pressure still held when the temperature
  refreshes, which it keeps: 22 stint × tyres in the dataset have a first
  pressure change of 3 counts or more after the temperature's (21 on the
  Inferno 86), 15 of them held out (14 on the Inferno 86, 226 rows), and
  their rows score 123.0 mbar under it against 62.0 under rule A (120.4
  before), 2.4 of its 3.2 mbar deficit to rule A. Under it, and under it
  only, the FJ's speed-pressure κ collapses in every fold (a median 5.7e-7
  before, 9e-10 under it; rule A's and rule C's full-data fits keep 5.4e-7),
  which is not diagnosed.
The owner chose rule A on these numbers.

**a4512 RR** (Sodegaura, wet, 2026-02-07; not among the held-out sessions),
the committee's one misfire, which rule A keeps and the owner accepted. The
pressure holds 1.88 bar from the first sample, the temperature refreshes at
8.3 s, the pressure reads 1.79 at 19.5 s (three counts) and climbs back to
1.82–1.85. Rule A anchors the stint at 1.79 bar at 19.5 s. Scored out of
sample (a fold holding out a4512 alone, refitted), RR's MAE is 15.8 mbar
before #371 (anchored at 1.88 bar from 8.3 s), 91.8 under rule A as
committed (91.7 with the old stamp), 73.3 / 71.8 under rule C without and
with the stamp fix, and 15.5 under the either-stream rule. The masking moves
the anchor's time with its value, so it is not the committee's value-only
repair (16 → 86 there), but the 1.88 bar reading was the better anchor.

**What it moves.** The frozen streams (a corner flat for a whole session,
among them 15 sessions flat on all four corners: the KK-SII on sensorless
wheels on 2026-08-09, a0216/a0219/a0222; eight KK-SII sessions at Suzuka on
2026-09-21 and three on 2026-10-03/04; the Inferno 86's 2025-07-22 test)
were masked whole by the session-start rule before #371 and still are: they
are out of training and scoring, and the blacklist is not needed for them.

### 1.0a″ A corner losing gas (#373, 2026-11-26)

The Inferno 86's front-left TPMS loses gas, or its transducer drifts,
through both Sodegaura sessions of 2025-10-05 (a1599, a1603): at a steady
70 °C its pressure falls from 1.454 to 1.395 bar over a1599 while the front
right holds 1.477–1.480, and it starts a1603 at 1.16 bar and 41 °C where
the front right reads 1.22 bar at 44 °C (the notes say all four were set to
1.4 bar cold).
No prediction-time input can know it, so the rows are dropped at cleaning,
not modelled.

**The screen** (`cleaning.gas_drift_screen`). The amount of gas in a tyre
goes as `(P + 1.013) / (T + 273.15)`; on steady laps it changes only by a
leak or a drifting transducer, and the session's other tyres share whatever
the sensors have in common. Per session, over its usable laps on which all
four tyres read a finite mean and a temperature range within the lap of
1 °C or less: each tyre's log amount of gas, less the mean of the other
three tyres', fitted linearly against the session's on-track clock, times
the span: the tyre's change over the session in %. A session is judged on 8
steady laps or more over 15 min or more (the committee's screen, except
that the clock counts every lap of the session, not only the usable ones).

**The threshold, fixed before scoring from the sound tyres** (the issue's
−4 % was set after seeing the values). Over every screened tyre-session of
the dataset, the centre is the median change and the scale 1.4826 × the
median absolute deviation (robust, so a leaking tyre does not widen it). A
tyre is flagged when its change is below the centre less z scales, z being
the one-sided Bonferroni quantile at which a sound tyre anywhere among the
N screened tyre-sessions is flagged by chance with probability 5 % at most
(`DRIFT_FALSE_FLAG_RATE`; z = Φ⁻¹(1 − 0.05 / N)), and when its reference is
sound: each of the other three tyres' change against the remaining two lies
within z scales of the centre. Only a loss is flagged. On the dataset of
2026-11-26: 45 sessions screened, N = 180, centre +0.02 %, scale 0.80 %,
z = 3.45, threshold −2.73 %. Flagged: a1603 FL (−4.28 %, its siblings within
0.24 % of each other) and a1599 FL (−3.20 %, within 0.24 %). The next three
are KK-SII tyres at Tsukuba, at −2.69 % (a0082 RR, 0.04 % short of the
threshold, 9 steady laps), −2.24 % and −2.10 %.

**No sound tyre flagged.** Both flags are the front left of 2025-10-05,
whose loss is in its own stream (above; its gas amount at steady
temperature −2.7 % over a1599 where the front right's is +0.2 %). One
warning on the screen's reach: with only 8 steady laps a session is near
the edge of what it can judge. a1329 FR (2025-08-16, Sodegaura, 8 steady
laps over 22.5 min) reads −1.85 % on the cleaning as committed, but −3.75 %
(and flagged) under the either-stream stale mask of §1.0a′, where its first
lap's readings change; its own stream shows no loss (it holds 1.55–1.56 bar
at 76–77 °C over the last six laps, and its gas amount at the end of the
session is 0.4 % above the session before's).

**Measured** (the standing holdout refitted on top of §1.0a′'s rule A as
committed, the same 15 folds and 128 sessions). The flag takes out 100 lap ×
tyre rows (a1599 FL 56, a1603 FL 44) that score 107.2 mbar (bias +103.4)
for the current calculator:

| hot-pressure MAE, mbar | rows | current | pace table | direct | Inferno 86 (current) |
|---|---|---|---|---|---|
| rule A (§1.0a′) | 5,751 | 37.48 | 38.40 | 38.52 | 46.00 |
| rule A and the drift flag | 5,651 | 36.12 | 37.10 | 37.22 | 43.10 |

On the 5,651 rows both score, the refit without the two corners moves the
current calculator by −0.12 mbar (date-clustered SE 0.25, bootstrap −0.56 to
+0.28; the pace table −0.14 ± 0.27): the pooled gain is the rows left out,
not a better model of the rest. The gap of the simulation-based calculator
to the current one is unchanged (+1.00 → +0.98 on those rows). The FJ's
rows are untouched (31.18).

### 1.0b Layouts by measured lap length, one row per recording (schema 5, 2026-10-30)

Every "Fuji GP Sh" session in the dataset (19 sessions, Inferno 86 and
SFJ, 2025-07 to 2025-12) lapped the 4.56 km GP course: the GPS path per
lap is 4,499–4,528 m and lap times are 117–131 s, where the real short
course is about 1.5 km. The registry listed GP and Short with the same
start/finish gate and let the dash variant name break the tie, so the
sessions carried `layout_id = fuji_short` (#209). Both pool into
`track_canonical = fuji`, which is why the model never noticed. Since
extractor 0.12.0 a layout can declare a nominal `lap_length_m`, and the
detector measures the median GPS path length between consecutive S/F
crossings and picks the layout whose nominal length is nearest; the dash
name only breaks ties the geometry cannot. The measurement is recorded
per session as `lap_length_m` (2,083 m at Tsukuba, 2,391 m at Sodegaura,
4,514 m at Fuji, 4,753 m at Motegi, 5,760 m at Suzuka), so a future
mislabelled variant is visible in `just tire-track-audit` without
re-deriving it.

The same bump makes discovery case-insensitive: the AIM tree is a
Windows volume, and a recording that arrives as a lowercase `.xrk` next
to the uppercase `.xrz` of the same run (2026-10-24, KK-F Tsukuba
0370–0372) was being extracted twice as two sessions and a second car
named `kk-f`. One `.xrk` now wins per stem regardless of case, the
manifest row of the superseded spelling is dropped (and its session
pruned), and the car name takes the logger's `Vehicle` spelling when
the filename token differs only in case.

### 1.0c Session comments and one notes parse per track day (schema 6, 2026-10-30)

Extractor 0.13.0 stores the RaceStudio "Long Comment" of each file as
`sessions.comment` (a merged session joins its files' distinct comments).
The driver records the tire fitted and the conditions there ("re-71rs",
"qually, slick tires, damp track"), which the old pipeline never saw. The
run-notes pipeline is one `claude -p` call per track day over the day's
notes files and every logged session with its comment, cached in
`session_notes/<date>.json`; `notes_matches.parquet` is rebuilt from the
caches, and the compound labelling reads the comments.

### 1.0d Native-rate storage and IMU validation (schemas 7–9, 2026-10-30/31)

Extractor 0.14.0 stores every selected channel **verbatim at its native
rate**: a session file holds one row per raw sample and one row group per
raw *timebase* (channels the logger sampled on one timecode array: GPS
25 Hz, IMU accelerometer 50 Hz — the KK-SII dash since 2026-03 logs it as
pairs 2 ms apart on 50 Hz frames, 100 Hz in bursts — gyro 50 Hz, each TPMS
corner on its own 1.25 Hz clock, the IR zones 1–4 and 5–8 of a corner on
two 10 Hz clocks), with a channel column null outside its timebase.
Nothing is resampled or aligned in the extractor; the loaders merge at
query time (`tire_etl/dataset.load_group`, `kinematics.load_kinematics`)
and the 1 Hz binning of §2.6a maps the TPMS and IR samples onto the
rolling clock defined by the kinematics. New channels: IMU vertical g, the
gyro rates, the logger's GPS-derived g and yaw rate, altitude. A survey of
one file per car showed 56 channels stored losslessly (row counts, rates
and values equal to the raw file after the float32 cast; latitude and
longitude stay float64, since float32 quantises them to about a metre and
destroys curvature). The per-lap TPMS start/end/mean are unchanged on the
reference corner and within one quantisation step (0.03 bar / 1 °C) on the
corners that used to be interpolated onto it. Merged multi-file sessions
now carry their later files on the merged clock (`t_lap_s` used to go
negative there by the restart offset).

The IMU is **validated against GPS position + speed** on each session's
fastest full lap: curvature of the ~10 m smoothed local-XY path on a 1 m
arc-length grid gives `a_lat = V²κ` and `ω = Vκ` (right-positive),
`a_long = dV/dt` over 0.5 s; fixes a receiver reports while still
acquiring (anywhere on the globe at zero speed for the first seconds of a
file — one SFJ session put its path length at a billion metres and the
extractor out of memory) are dropped when farther than 20 km from the
session's median position or an isolated jump of over 100 m, and
interpolated across. Each candidate axis must have samples
(coverage ≥ 90 %), not be flat (std ≥ 0.1 g / 5 °/s), be in family (OLS
slope ∈ [0.85, 1.15], corr ≥ 0.9 — ≥ 0.85 longitudinal, dV/dt from 25 Hz
GPS being the noisier reference — |offset| ≤ 0.4 g / 10 °/s) and not be
twisted (|corr| with the other axis' reference ≤ 0.5). One verdict per
session for the accelerometer (`accel_source`: both axes together) and one
for the gyro (`gyro_source`), fallback IMU → the logger's GPS-derived
channels, validated the same way → series derived from position; every
number is on the session row (`kin_*`) and in the file metadata, and the
laps table's dynamics (`heat_proxy*`, `lat_g_peak`) come from the chosen
series. Raw channels stay untouched. On the survey files: the Inferno 86
IMU passes (lateral slope 0.98, corr 0.996; longitudinal 0.995 / 0.97);
the KK-SII 2026-10-04 Suzuka race has its IMU axes swapped (lateral
correlates 0.97 with the longitudinal reference) and the default-profile
KK-F reads lateral at 0.42× — both fall back to the logger's GPS channels
(slopes 0.96–1.02); the EK9's IMU is rotated about 30° in yaw (lateral
slope 0.87 ≈ cos 30°, longitudinal cross-correlation 0.63) and falls back
too. The AIM gyro reads about 0.87× `Vκ` on the Inferno 86 and SFJ and
≈ 1.0× on the KK cars and EK9. `just tire-kinematics-audit` lists every
session with the numbers and the A2 gate on the chosen series.

The audit over the whole dataset (281 sessions with ≥ 3 laps, 2026-10-30)
explains every verdict. Inferno 86 (113 sessions): IMU on 103, lateral
slope 0.95/0.98/1.01 (10th/50th/90th percentile) at corr ≥ 0.98; six
sessions with a dead or absent IMU use position, four with longitudinal
corr 0.81–0.84 use the GPS channels. SFJ (22): the IMU lateral is healthy
(slope 0.96–0.97) but its longitudinal corr sits at 0.76–0.85, so 11
sessions fall to the GPS channels on that axis' threshold. EK9 (24):
IMU on 15 with lateral slope 0.92–0.98 (the ~30° yaw rotation), nine on
the GPS channels. KK-F (42): the 18 `default`-profile sessions read
lateral at 0.43–0.55× and use the GPS channels; under the KK-SII profile
the IMU passes on 16, is absent from the file on 7 and twisted on one.
(Since extractor 0.23.0 those sessions carry the profiles
`KK-F (data logger)` and `KK-F`; the verdicts did not change, §1.0f.)
KK-SII (72): the IMU is correctly oriented only on 2026-02-25/26 (5
sessions); from 2026-03-18 through 2026-10-04 its axes are **swapped**
(lateral follows the longitudinal reference at corr 0.97, 44 sessions),
and on 16 sessions between 2026-02-18 and 2026-05-24 the IMU and gyro
streams stop about six minutes into the file (no samples on the reference
lap); the IMU channel is absent from 8 files. All of those use the logger's
GPS channels (chosen lateral slope 0.97–0.99, corr 0.96–0.98). On the gate
bands of the chosen series, 186 of 309 sessions are outside at least one:
the braking-only longitudinal regression (reference < −0.2 g, a
range-restricted 25 Hz dV/dt) has corr 0.70/0.89/0.95 on the Inferno 86,
0.72/0.83/0.88 on the SFJ, 0.74/0.79/0.84 on the EK9 and slope up to 1.2
under braking (pitch adds a gravity component to the IMU), and the Inferno
86 and SFJ gyro read 0.85–0.88× the GPS yaw rate (the Inferno 86 from
2025-06-09 on; 1.01–1.02× before). Lateral is inside its band everywhere
but three Inferno 86 sessions at 0.88–0.92. Those two bands are the
reference's, not the sensors'. Extractor 0.17.0 (2026-10-31) addressed
the first: the candidate is now smoothed over the reference's own 0.5 s
window before the regression (the 50 Hz IMU's vibration alone had pulled
healthy longitudinal axes under corr 0.85 — SFJ 11/22 → 22/22, Inferno 86
103/112 → 108/112 pass; a ±0.4 s lag search was tried and found nothing,
the best lags being one GPS sample), and the gyro must read within
0.9–1.1× the position yaw rate — the logger's own GPS yaw channel reads
0.99–1.00× the position reference on every Inferno 86 session before and
after 2025-06-09 while the gyro went 1.02× → 0.87×, so the scale change is
the gyro's and those 73 Inferno 86 and 25 SFJ sessions take the position
yaw. Sessions inside every gate band went from 101 to 229 of 280. What
remains outside is physics rather than sensor health: 27 braking-only
longitudinal slopes of 1.15–1.21 (pitch puts a gravity component into the
IMU under braking) and 20 braking-only correlations of 0.80–0.85 on the
range-restricted subset, and the lateral slopes of 0.92–0.95 on the EK9,
KK-F, SFJ and 2025 Inferno 86 IMUs that the gate's [0.95, 1.1]
expectation excludes: a chassis-mounted accelerometer reads
a·cos φ − g·sin φ under body roll φ, a 2–4° roll costing 3–7 % of the
lateral g, so 0.92–0.97 is a healthy sensor and the verdict's 0.85–1.15
band is the right one. Five early-2025 Inferno 86 sessions with lateral
0.88–0.92 and longitudinal corr 0.4–0.9 (2025-03-16, 03-29, 05-17) look
like a loose mount and keep the IMU only because lateral still correlates
at 0.98.

Extractor 0.18.0 adds the `imu_swapped` verdict. The KK-SII dash has
logged the car's longitudinal acceleration as `LateralAcc` and the
lateral one as `InlineAcc` since 2026-03-18 (both signs intact, slopes
0.92–0.97 against the exchanged references, vertical still −0.9 to −1.0 g,
gyro unaffected: a channel assignment, not a rotation). An IMU that fails
its own axes as twisted is scored once more with the two channels
exchanged; when both pass, the verdict is `imu_swapped` and
`load_kinematics` hands the columns out exchanged — the stored channels
stay as logged. That recovers 46 sessions (45 KK-SII, one KK-F of 2026-10-02; 425 laps,
all with TPMS) from the 25 Hz GPS-derived channels to the 100 Hz
accelerometer — exchanged, the channels regress at lateral slope
0.92–0.94 (corr 0.99) and longitudinal 0.95–0.97 (corr 0.99) — and their
heat proxy drops to 0.87× of what the noisier GPS derivatives gave, the
lateral peaks by 0.6 g. A rotated mount (the
EK9, the Inferno 86 in March 2025) keeps its cross-correlation through the
exchange and is not recovered; a horizontal-magnitude check
(√(lat² + long²), invariant to any rotation) confirms those sensors are
sound and the KK-F `default`-profile units are not (magnitude 0.44× with
corr 0.6) — a fitted-angle correction would recover the former 13
sessions and is left as a decision.

Storage (schema 8, extractor 0.15.0, 2026-10-31): the AIM file logs the
IMU, the gyro and the CAN brake / steer / throttle as float16 and libxrk
widens them to float32 (libxrk #115); every channel is now stored at the
narrowest float width that holds its samples exactly (float16 for those,
float32 for the analog inputs, IR zones and TPMS pressures, float64 for
positions), so the dtype documents the source precision — the
accelerometer carries a 10-bit mantissa, about 0.001 g near 1 g. Parquet's
default dictionary attempt on a float column was defeating zstd; the
float32/float64 columns are byte-stream-split under zstd 9 and the same
session file went from 20.6 MB to 8.7 MB with identical values. Schema 9
(extractor 0.16.0) replaces the path-and-mtime hash that keyed every table
with a readable, chronologically sortable id —
`2026-10-24_122053_KK-F_tsukuba_a0370-0372`: date, the logger's wall-clock
start, car, canonical track, run numbers — so a directory listing reads
as a session list and a rewritten file re-extracts under the same id.

### 1.0e Per-window sensor source, training on windows (extractor 0.21.0, 2026-11-11)

**Decision.** The choice between the IMU and the GPS-derived accelerations
is made per 20 s window of a run, not once per session, and a window with
no usable sensor is discarded: the model trains on windows rather than on
whole sessions. This replaces §1.0d's "one verdict per session" as the unit
of sensor selection. The session verdict remains, with a narrower job: it
says whether the session's IMU is sound and in which orientation.

**Why.** One verdict per session, scored on the fastest full lap, had two
failure modes, both on the KK-SII, whose IMU and gyro streams stop part-way
through the file on 23 sessions between February and May 2026 (anywhere
from 24 s to 22 minutes in). When the stream stopped *before* the fastest
lap the whole session went to the logger's 25 Hz GPS-derived channels (20
sessions), including the laps the IMU had recorded. When it stopped
*after* it (3 sessions, all with TPMS) the session kept `imu_swapped` and
nothing covered the rest of the run: the later laps had NaN dynamics, and
the lap the stream stopped in carried a truncated heat integral that looked
like a number (29.6 on a 61.5 s lap whose neighbours read 60).

**What is scored.** The session is cut into fixed 20 s bins of session time
and every window gets one source, stored in the timeseries file's metadata
(`kinematics.windows`, one code per window) and counted on the session row
(`kin_win_imu`, `kin_win_gps`, `kin_win_discard`, `kin_win_mismatch`):

| Source | When |
|---|---|
| `imu` | the session's IMU is sound and has samples within 0.5 s of at least 90 % of the window's GPS samples, and is not frozen |
| `gps` | otherwise, when the GPS fallback (`kin_gps_fallback`: the logger's GPS channels where they validated, else the position-derived series) covers the window with no value over 3 g |
| `discard` | neither |

Three rules complete it:

- *The IMU is scored on a lap it covers.* When its stream had stopped
  before the fastest full lap, the regression of §1.0d runs on the fastest
  full lap it does cover (`kin_imu_ref_lap`); the windows then use it for
  the part of the run it recorded.
- *Frozen.* A window where the IMU's lateral channel is flat (std under
  0.1 g) while the position reference shows cornering (std ≥ 0.3 g) at a
  mean speed of 15 m/s or more goes to the GPS. The speed floor keeps a
  parked car under a wandering GPS fix out of the rule.
- *Mismatch cap.* A window where a live IMU correlates below 0.5 with a
  sane, excited position reference (std ≥ 0.3 g lateral, ≥ 0.2 g
  longitudinal, no value over 3 g, at least 5 s moving) is a mismatch. Up
  to three per session are tolerated and the IMU is still used there; more
  than three and the IMU is not trusted for the session (`kin_note =
  imu_window_mismatch`).

GPS spikes are judged on the series smoothed over the reference's 0.5 s
window, while the car is moving: one noisy 25 Hz sample of the logger's
GPS-derived g is not a spike, half a second of 5 g is. Judged on raw
samples the same rule discarded three times as many windows.

**What the measurements said** (schema-9 dataset, 20,349 windows):

- *A mismatch is the GPS's fault.* On sessions whose IMU is otherwise
  sound, every mismatched window inspected was the reference failing, not
  the sensor: a receiver still acquiring at the start of a file (a 166 km
  "path" in ten seconds and a 200 g reference with the throttle closed and
  the IMU reading a parked car), position spikes mid-session, a spin. A
  mismatch therefore never switches a window to GPS; it only counts toward
  the cap. With non-overlapping windows, 203 of 222 trusted sessions have
  no mismatch, 14 have one, 2 have two — hence a cap of three.
- *One regression over the whole session does not work.* A few corrupt
  fixes put the position reference at thousands of g (10⁹ g on one EK9
  Motegi session) and drive the slope and correlation to zero; it flipped
  31 accelerometer verdicts, most of them healthy sensors. The reference is
  only usable lap by lap or window by window, with its impossible values
  excluded.
- *Healthy sensors track GPS closely in a window.* On windows with a sane
  reference the median lateral correlation is 0.99 and the 1st percentile
  0.90 once the reference shows 0.3 g of excitation; below that the
  correlation measures noise.
- *No frozen IMU exists in the data so far.* Every flat-IMU window on a
  trusted session was a parked car under a wandering fix. The rule is there
  for the day one appears; the observed fault is the stream stopping.
- *Rejected on the way:* requiring the IMU to cover 98 % of the whole
  session (it sends a session to GPS for a stream that dies in its last
  minute and throws away the laps the IMU did record), and de-spiking the
  GPS sample by sample instead of discarding (it would leave no holes, but
  it invents data where the sensor had none; the loss from discarding is
  small enough to take).

**What the consumers do with it.** `kinematics.load_kinematics` hands out
each window from its source: the IMU's own samples, the fallback on the GPS
clock, or — for a discarded window — the GPS-clock rows with NaN
accelerations and `lat_source = "discard"`, kept so that anything
integrating over time sees the hole instead of stepping across it. In the
stored file nothing is removed: a discarded window keeps every raw sample
(the spike that caused it included), and since extractor 0.22.0 its rows
carry the mark as a column — `discard`, a sparse boolean key column, true
on each row of any timebase whose session time falls in the window and
null everywhere else — so a query can filter on it (`WHERE discard` in
`just tire-query`). A discarded window is a hole in the heat input, never
a value:

- the laps table's `heat_proxy*`, `lat_g_peak` and `long_g_peak_brake` are
  NaN on a lap that holds a discarded window;
- the per-second fit (§2.6a) does not score the seconds of a hole and
  restarts the simulation after it from the measured state — the mean
  pressure-implied gas temperature of the first 10 scored seconds, on the
  same pit-exit gas-law scale and with the same speed correction as every
  other observation. A stretch between holes with fewer than 60 scored
  seconds is dropped. On a synthetic stint with holes the fit recovers
  τ = 302 s against a true 300; with the restart switched off, and the
  hole's heat read as zero, it returns 73 s;
- the pace check (§2.10) leaves out a lap with a hole.

GPS-derived heat reads about 15 % above IMU heat on the same car (noise in
the derivatives), so a run that changes source part-way has a step in its
heat input. That is accepted: it was already there between sessions, and
the alternative was no dynamics at all for those laps.

**Where the committed dataset and model stand** (schema 9, extractor
0.21.0; every number run on 2026-11-11). 309 sessions and 3,555 laps — the
readable ids exposed one session downloaded twice (Sodegaura 2025-04-18,
run a_0258), so one fewer than under the hash — in 307 timeseries files,
574 MB. Accelerometer verdicts: `imu` 178, `imu_swapped` 54, `gps_channel`
59, `gps_position` 16; gyro: `imu` 170, `gps_position` 137; two sessions
have no samples inside a lap.

Of the 20,349 windows, 14,927 (73.4 %) use the IMU, 5,379 (26.4 %) the GPS
fallback and 43 (0.21 %) are discarded, in 11 sessions. Fourteen sessions,
all KK-SII, mix sources: the IMU while its stream lasts, the GPS after. On
13 sessions the IMU was scored on a lap other than the fastest. 21
sessions have one mismatched window and none has more, so the cap has not
fired yet. Of the 43 discarded windows, 31 fall in six sessions with TPMS:
20 in three road drives with no resolved track, and 11 on circuit sessions
the model trains on, all KK-SII at Tsukuba after the IMU had stopped
(2026-03-22 `a0066` and 2026-03-28 `a0080` one each; 2026-05-23
`a0166-0167` nine — that session's GPS is noisy enough that neither its GPS
channels nor its IMU validate against it). 29 of the 3,555 laps (19 of the
2,978 full laps) have no heat summary. Of the 280 sessions with ≥ 3 laps,
229 are inside every gate band of `just tire-kinematics-audit`; 43 of the
51 outside fail only the braking-only longitudinal regression.

The model refitted at each step, FL/FR/RL/RR:

| | schema 6 | schema 9, one source per session (0.20.0) | schema 9, per window (0.21.0) |
|---|---|---|---|
| 5-fold calculator holdout, hot-pressure MAE (bar) | 0.048/0.047/0.044/0.038 | 0.042/0.040/0.049/0.038 | 0.042/0.040/0.049/0.038 |
| 5-fold oracle holdout (bar) | 0.048/0.047/0.046/0.041 | 0.044/0.042/0.049/0.041 | 0.044/0.041/0.049/0.041 |
| Notes validation, 12 sessions (bar) | 0.101/0.085/0.060/0.069 | 0.101/0.085/0.060/0.069 | 0.101/0.085/0.060/0.069 |
| Pace check slope | 1.01/1.07/1.08/1.18 | 0.87/0.90/0.91/0.95 | 0.87/0.95/0.85/0.94 |
| Pace check bias (log) | | +0.004/+0.005/+0.005/+0.006 | +0.004/+0.004/+0.005/+0.005 |

- The folds are drawn in session-id order within a (track, car, condition)
  bucket, so the readable ids changed the partition between the first two
  columns as well as the inputs; they are not the same split. The last two
  are.
- RL is the one corner worse than at schema 6, and all of it is the damp
  bucket: 58 RL points at 0.132 bar against 0.040 dry and 0.046 wet; over
  the dry and wet points together the calculator holdout reads
  0.039/0.041/0.042/0.038.
- Per-window sources left the pressure accuracy where it was. One constant
  moved by more than 15 %: the FJ speed-pressure κ, 4.4e-7 → 2.2e-7. It had
  already halved from 8.7e-7 when the `imu_swapped` verdict moved 46 KK-SII
  sessions from GPS-derived to IMU heat, and it halved again as ten more
  sessions got their IMU back for part of the run — which suggests that on
  this car κ is absorbing the scale of the heat input rather than
  measuring an effect of speed on the pressure reading.
- The pace check compares a session's fast laps with its typical laps, a
  heat ratio of about 6 %. Across a change of sensor that ratio is swamped
  by the 15 % step between GPS-derived and IMU heat: run over every lap of
  the mixed sessions it read 0.75/0.85/0.75/0.84 (FJ 0.45/0.43/0.55/0.58),
  an artefact of comparing unlike laps. It now compares only laps of a
  session's most common single source (34 sessions instead of 37). Per
  car: FJ 0.75/0.74/0.75/0.76, Inferno 86 0.91/0.98/0.90/0.98. The slopes
  under 0.9 remain lap-sim segment E's question (#207).

### 1.0f Profiles by vehicle, the KK-F's controls, drivetrain channels, car corrections (extractor 0.23.0, 2026-11-14)

**Decisions.**

1. The extractor resolves the vehicle profile by logger ID **plus vehicle
   name** (`extract.session_channel_names` →
   `profiles.get_profile_for_log`), as every other consumer does. It used
   the logger ID alone, which on the SW4 wheel shared by the KK-SII, the
   KK-F and the KK-F-V2 is always the KK-SII's profile.
2. The KK-F's two pre-SW4 loggers are mapped: 6521016 (device name
   "Inferno KK-F", 2026-05-31 to 2026-06-15) and 6520348 (the "Inferno
   Solo2DL", 2026-06-21 to 2026-07-04) to a new profile `kk_f_dl`. 6520348
   also recorded the 718 Cayman, so it is mapped `by_vehicle` only: another
   car on it gets no profile instead of the KK-F's.
3. New stored channels, verbatim at their native rate like every other:
   `throttle_deg` in `controls` (the ECU's throttle angle), and a
   `drivetrain` family with `engine_rpm`, `gear` and
   `wheel_speed_{fl,fr,rl,rr}_kmh`. New profile keys `throttle_angle`,
   `rpm`, `gear`, `wheel_speed_*`.
4. The car a recording names can be corrected in
   `data/tire_dataset/session_overrides.yaml` (`tire_etl/overrides.py`),
   keyed by file. The extractor applies it before grouping, so the session
   id, the profile and `sessions.car` follow the corrected car;
   `sessions.car_declared` keeps what the file says, and the manifest row
   carries `car_override` so that editing the file re-extracts the
   recording without `--force`.

**Why** (#254). Of the KK-F's 44 sessions, 25 were extracted with the
KK-SII's channel names and 19 with the defaults, so its throttle, brake
and steering were stored on one session — and that one by accident (below).
The car model of the lap simulation (#205) selects straight-line samples by
full throttle and braking zones by brake pressure, and needs the gear to
read the full-throttle acceleration.

**What the KK-F logs**, surveyed over all 62 of its raw files (57 as the
KK-F, five as the KK-F-V2; `MYST` ECU stream on all three loggers):

| Signal | Channel | Where | Notes |
|---|---|---|---|
| Throttle angle | `TA` (deg) | every file | 5.0° closed, 84.0° open, the same on the three loggers; 50 Hz on the data loggers, 10 Hz on the SW4 |
| Throttle percentage | `Throttle_PCT` | SW4 from 2026-08-23 | logged on change; not a linear function of `TA` (residual 7–10 % rms on two files), so both are stored |
| Brake pressure | `P_Brake` (data loggers), `P_Brake_Actual` (SW4) | every file | positive; logged on change with a 2 s heartbeat. Mean GPS longitudinal g while it reads over 10: −0.38 to −0.99 on the 47 sessions. No negative sample; 74 `-inf` samples in all (three per data-logger file, one on the SW4), stored as logged |
| ECU brake count | `Pbrake` / `Pbrake_Value` | every file | the same signal negated, at a regular 10 Hz. Not stored |
| RPM | `ENGRPM` | every file | |
| Gear | `GP` | every file | 0–7, and 100 on about 0.3 % of samples. The SW4's own `Gear` disagrees with it and is not stored |
| Wheel speed | `SPD_FL` (km/h) | every file | one pickup, front left. 1.00–1.07 × GPS speed on part of each file, spikes to 3,238 km/h on the rest (15–84 % of the samples above 36 km/h on five files) |
| Steering | `Steering` (data loggers, from 2026-06-15), `steering` (SW4) | 39 of 47 sessions | dropouts to −1,600° on the data loggers; a floor of −101.9° from 2026-08-23 to 09-25 and within 0–0.5° from 2026-09-26 on the SW4 |

Two things a consumer has to know, both answers to #254's open questions:

- **The brake scale differs between the loggers.** `P_Brake` on the data
  loggers is the ECU count itself (its peak equals the peak of `Pbrake`
  negated on all 24 files; daily peaks 48–63). `P_Brake_Actual` on the SW4
  is 0.625 × that count (the ratio of the two peaks is 0.625 on 36 of 38
  files, and 0.58 and 0.61 on the two where the change-logged channel
  missed the peak; daily peaks 22–39). For the same pedal the data loggers
  read 1.6 × the SW4. Both are stored as logged in `brake_bar`; comparing brake
  pressure across 2026-07/08 needs the factor.
- **Brake pressure and throttle percentage are not on a regular clock.**
  A value holds until the next sample. The laps table's `brake_mean` /
  `throttle_mean` are plain sample means and so weight the moments these
  channels change; they are descriptive columns that nothing in the model
  reads. They ignore non-finite samples since this version.

The broken steering and wheel-speed channels are stored too: cleaning
happens when a consumer loads the data, not in the extractor.

**RPM, gear and wheel speeds on the other cars.** Inferno 86: `RPM`,
`Gear` and four `WheelSpd*` on all 122 sessions. KK-SII and SFJ: `RPM`
only (79 and 28 sessions). EK9 and RX8: none. The 718 Cayman's one session
has no profile; its `RPM` is all zeros and its `WHEEL_SP_*` channels are
not stored.

**Car corrections.** The SW4 wheel moves between the cars: the dash config
is loaded for the car, and the vehicle name is changed by hand. Twice the
name was left on the previous car. The channel set is the config, and the
config is the car's wiring (the KK-SII has analog `F_BRK` / `R_BRK` /
`Throttle` / `RPM` / `Steering` and no ECU stream; the KK-F has the ECU
stream and none of those), so a file with real signals on one car's
channels recorded that car:

| Recording | Named | Recorded | Evidence |
|---|---|---|---|
| 2026-08-23 `a_0252`, Tsukuba, 19 laps | KK-SII | KK-F | the KK-F's ECU stream, none of the KK-SII's analog channels; no KK-SII session that day; the day's next file has the same channel set and is named KK-F |
| 2026-10-02 `a_0350` + `a_0351`, Suzuka, 9 laps | KK-F | KK-SII | the KK-SII's analog channels with real signals, no ECU stream; the accelerometer has the KK-SII's exchanged axes (`imu_swapped`); first session of the KK-SII's Suzuka weekend |

Before the correction, resolving by vehicle name took the controls away
from the second one (it had been the "one KK-F session with throttle
stored") and did not give them to the first. After it, the KK-F has 44
sessions and the KK-SII 79, as before, with two swapped.

**What changed in the dataset** (re-extract of all 411 files; the 309
sessions that existed keep every session-row value, including every
`kin_*` number and both sensor verdicts, apart from `profile_name`,
`car` / `session_id` on the two corrected ones, and the new
`car_declared`):

| | Before | After |
|---|---|---|
| KK-F sessions with throttle stored | 1 of 44 (`throttle_pct`) | 44 of 44 (`throttle_deg`), 23 of 25 SW4 sessions also `throttle_pct` |
| KK-F sessions with brake pressure stored | 1 of 44 | 44 of 44 |
| KK-F sessions with steering stored | 1 of 44 | 36 of 44 |
| KK-SII sessions with throttle and brake stored | 78 of 79 | 79 of 79 |
| Sessions with RPM / gear / wheel speeds | 0 | 277 / 169 / 169 of 310 |
| Timeseries rows, files | 78.6 M, 307 | 97.8 M, 310 (drivetrain 15.8 M) |
| Timeseries directory | 574 MB | 634 MB |

Laps table: the KK-F's `brake_mean` / `brake_max` are now filled on its
593 laps and `throttle_mean` on 338; no dynamics column changes on any car.
`on_track_s` is the span of a lap's samples, so it grows where a newly
stored channel has a sample nearer the lap's edge: on 960 of 3,555 laps,
by under 75 ms on all but two (2.7 s on two KK-F laps of 2026-08-11 whose
other channels start late).

The 19 KK-F sessions on the data loggers keep their `gps_channel` verdict
under the right profile: the accelerometer channels have the same names in
both, and the logger's own accelerometer reads lateral at 0.25–0.62 × the
GPS value on them.

**Effect on the model.** The control and drivetrain channels are not
inputs, and the tyre model pools the three KK cars as one (`FJ`), so
neither the profiles nor the car corrections should move it. Refitted on
the re-extracted dataset without the new day, it does not: FJ
speed-pressure κ 2.1829e-7 against 2.1829e-7, 5-fold calculator holdout
0.043/0.040/0.049/0.038 bar (FL/FR/RL/RR) against 0.042/0.040/0.049/0.038
on the committed dataset (the folds are drawn in session-id order, and two
ids changed).

**The new day** (2026-11-13, Tsukuba, the KK-F-V2's first: 3 sessions, 45
laps, 5 files; no run notes, one file comment) is what moves the refit:

| FL/FR/RL/RR, hot-pressure MAE in bar | Committed (through 2026-10-25) | This refit (through 2026-11-13) |
|---|---|---|
| Calculator holdout, pooled | 0.042/0.040/0.049/0.038 | 0.041/0.041/0.050/0.038 |
| FJ | 0.035/0.032/0.050/0.034 | 0.034/0.033/0.051/0.034 |
| Inferno 86 | 0.058/0.061/0.046/0.046 | 0.058/0.061/0.046/0.046 |
| dry | 0.040/0.046/0.040/0.037 | 0.041/0.045/0.040/0.038 |
| wet | 0.037/0.029/0.046/0.041 | 0.037/0.029/0.046/0.041 |
| damp (50–58 points) | 0.086/0.039/0.132/0.025 | 0.061/0.055/0.149/0.026 |
| Oracle holdout, pooled | 0.044/0.041/0.049/0.041 | 0.043/0.042/0.050/0.041 |
| Notes validation (12 sessions) | 0.101/0.085/0.060/0.069 | 0.101/0.085/0.060/0.069 |
| Pace check slope / bias | 0.87/0.95/0.85/0.94, +0.004 to +0.005 (34 sessions) | 0.88/0.95/0.86/0.94, +0.003 to +0.005 (35 sessions) |

- One constant moves by more than 3.5 %: **the FJ speed-pressure κ,
  2.18e-7 → 5.88e-7**, on three sessions out of 178. It is the third time
  this constant has moved by a factor of two or more on a small change of
  the data (§1.0e); here the car is the KK-F with more power. The rest of
  the FJ's constants move by under 3.5 % (`g_transfer` 2.40 → 2.49, K
  +0.5 to +2.8 %, τ under 1 %); the Inferno 86's by under 0.1 %.
- The pooled accuracy is where it was. The damp bucket, which the thin
  FJ damp data carries, moves with κ: front-left better by 0.025 bar,
  front-right and rear-left worse by 0.016 and 0.017.

## 2. Modeling approach

### 2.1 Energy balance on a lumped tire mass

The model is one ODE — heat flux balance on the tire treated as a single
thermal mass:

```
                                              ┌──────────────┐   ┌──────────────┐
   m·c · dT/dt   =   α · q(t)             −   │ h_air · (T   │ + │ h_road ·     │
                                               │  − T_air )   │   │ (T − T_road) │
   ───────────       ──────────────────        └──────────────┘   └──────────────┘
   energy stored     energy IN                       energy OUT
   per second        per second                      per second
   (W = J/s)         (friction work + hysteresis)    (convection to air at the
                                                      tire's top/sides + conduction
                                                      to the track at the patch)
```

**Energy IN.** Friction work at the contact patch: force × slip velocity.
Since schema v5 the driving intensity is `q_i = |g|·V/V_ref` with the
per-corner weight-transfer / drive / brake split of §2.0 (the v0 proxy was
`g² = lat_g² + long_g²` with a per-track factor; the track factor is gone —
the circuit enters only through its measured per-corner ⟨q⟩). The
coefficient `α` absorbs friction coefficient, contact-patch geometry and
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
`w_road` is deferred to v1 — at ~5% T_road-measured, the
joint identifiability is poor.

**T_road sourcing** at inference (priority order): user-supplied → AIM logger
channel (rare, <5%) → proxy `T_road = T_air + 10 · (1 − cloud_cover/100) ·
sun_factor` from Open-Meteo cloud cover → fall back to `T_road = T_air`.

### 2.2 Closed-form solution

Approximating `g²(t)` by its session average `⟨g²⟩` (good within a stint for
a consistent driver), the linear ODE has a closed-form solution starting
from the tire's actual temperature `T_start` at roll-out:

```
T_hot(t) − T_eff  =  K · ⟨q⟩ · (1 − exp(−t / τ_sec))
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

### 2.2a The observable: pressure-implied gas temperature (2026-10)

The equations above are written in "tire temperature", but what the model
is fitted to and predicts is the **cavity-gas temperature implied by the
TPMS pressure**, converted with the constant-volume gas law from the
stint's pit-exit reading:

```
T_gas_K  =  T_start_K · P_abs / P_start_abs
```

Two reasons. The pressure channel responds to the tread within seconds,
while the valve-mounted TPMS temperature sensor lags the gas by a median
125–140 s (a first-order lag of ≈ 180 s at 1 Hz on 83 stints): from pit
exit the gas is +3.8 °C at the end of flying lap 1 and +11 °C at lap 3
where the TPMS shows +1 and +5. Fitted to the TPMS temperature the model
absorbed that lag into τ (≈ 650 s) and over-predicted the first three laps
by +2.7 °C. And pressure is what the driver sets and what the calculator
outputs: with the gas temperature as the state, the hot pressure is exactly
`P_start_abs · T_hot_K / T_start_K`, with no empirical gain to correct.

The TPMS temperature is used for one thing only: the **initial condition**.
At pit exit the tire has rested, the separate measurement the driver types
and the gas agree, and that reading together with the set pressure defines
the anchor state. The predicted "hot temperature" the calculators display
is therefore a gas temperature and reads lower than the dash early in a
stint; the hot *pressure* is the number to trust. Pressure is reported in
steps of 31.3 mbar, logged as 0.03 or 0.04 bar (≈ 3.6 K of gas temperature
at 2.5 bar absolute), so the per-lap target is noisier than the TPMS
temperature; the stale-TPMS masks (§1.0a, §1.0a′) keep the anchor clean: a
module that fell asleep hot on an earlier run reports that stored value
until it wakes, and taking it as the anchor put the Sodegaura 2026-10-10
rear axle 0.4–0.5 bar high, +43 °C on every lap of that stint. With
native-rate storage the first unmasked sample is the module's first live
packet.
Since dataset schema 4 all of this runs at training time
(`tire_model/cleaning.py`, §1.0a), not in the extractor.

### 2.2a′ The reading at speed: the speed-pressure correction (2026-10-08)

The TPMS pressure read while moving sits *below* the cavity gas-law
pressure, by an amount that grows with speed²: the tyre grows under
centrifugal load (and the valve-mounted sensor sees the same ∝ V²
acceleration), so the cavity volume depends on speed:

```
P_read · V₀·(1 + κ·v²) = n·R·T        →   P_read = P_gas / (1 + κ·v²)
```

This was found from the within-stint residual of the per-second fit: with
stint offsets and slow drift removed, the Inferno's gas residual rises
monotonically from −2 °C below 60 km/h to +4 °C at 200 km/h, with no lag —
a carcass with a four-minute time constant cannot do that, and it is the
same on every track (κ 6.9–8.0 × 10⁻⁶ at Fuji, Sodegaura and Motegi), every
corner (5.7–7.4 × 10⁻⁶) and across sessions (IQR 5.8–9.2 × 10⁻⁶). The
per-second fit now carries **one κ per car** (`statespace`, ``v_corr``,
fold-stable to ±5 % on the Inferno), and every lap-end observable is put on
the gas-law scale with ``energy_balance.gas_temperature_at_speed_c`` using
the lap-end speed and the anchor speed from the timeseries
(``warmup_table.apply_speed_correction``; the compound EM and the holdout
use it). Fitted: FJ 1.5 × 10⁻⁶, Inferno 86 1.0 × 10⁻⁵ per (m/s)² — 0.5 %
and 3 % of pressure at 200 km/h; the stiff-belted slick grows far less
than the tall street tyre. The artifact carries it under
``energy_balance.speed_pressure`` (additive, schema unchanged).

**What it changes.** The calculators' hot pressure is the standstill
gas-law value, as before; a dash reading at speed is lower by the factor.
In the oracle harness the lap-end pressure MAE went 0.0550 → 0.0527 bar
and the lap-end bias 0.0225 → 0.0100 bar; the Inferno's per-stint gain
spread went 0.41 → 0.34 and its per-track gain ratios moved to within ±9 %
of 1. Speed-dependent *cooling* on the carcass node is rejected a second
time once κ is in (β unstable 1.4–3.6, track factors drift up).

### 2.2a″ κ read within the lap (#417; an option, default off)

**What was wrong.** The FJ's fitted κ fell from 8.10e-7 to 1.22e-7 when the
KK-F-V2's Fuji day of 2026-11-25 entered the fit, and across the standing
holdout's 15 folds it runs from 3e-12 to 3.1e-7. It is not the solver
(starts at 1e-9, 8e-7 and 2e-6 end at the same value) and not a session:
each of the day's nine sessions alone, added to the earlier data, leaves κ
between 4.8e-7 and 8.7e-7. Leaving out any one of the 24 dates moves it
between 0 and 8.1e-7: five dates put it at zero, fourteen under 1e-7.

**Why.** In the fit κ acts through `v² − v_anchor²`. That has a part
within the lap (`v²` less its lap mean) and a level (the lap mean less the
anchor), and the level is the larger regressor. Fitted as two parameters
on the FJ's dry stints:

| | all data | without the Fuji day | leaving one date out | one raw car × track at a time |
|---|---|---|---|---|
| within the lap | 1.54e-6 | 1.57e-6 | 1.50 to 1.59e-6 | 0.97 to 1.96e-6 |
| level | −1.74e-6 | −1.6e-7 | −2.7e-6 to −1.6e-7 | −2.0e-6 to +3.6e-6 |

The level is where every stint's offset sits (tens of mbar; §2.2, #421),
against the 4 to 14 mbar this car's κ is worth, so the single fitted κ is
a compromise between the reading's dependence on speed and whichever
circuits are over- or under-predicted. The two fast circuits have the
largest lever (lap-mean `v²` about 1,550 at Fuji and 1,600 at Suzuka
against 1,100 at Tsukuba) and pull opposite ways: Fuji is under-predicted
by 30 to 65 mbar (#394) and wants κ at zero, Suzuka wants it high. The
sum of squares at κ = 8e-7 less that at κ = 0, everything else refitted,
is +8,245 on the KK-F-V2's Fuji day and +7,343 on the KK-SII's, −4,988 at
Suzuka, −5,740 on the KK-F at Tsukuba and +2,643 on the KK-SII there.
Before the Fuji day the balance sat at 8.1e-7; the day moved it to
1.2e-7. The objective is flat: 2,987,789 at the optimum, +144 at κ = 0,
+4,449 at 8e-7. On the Inferno 86 the two parts agree (8.8e-6 and 1.0e-5)
and the fitted κ is stable (8.9e-6 to 1.0e-5 leaving one date out): its
effect is five times larger.

**A direct reading.** Within a lap the gas hardly changes, so `ln P`
against `v²` there is κ with no thermal model behind it
(`statespace.within_lap_kappa`): per dry stint, tyre and lap of 40 s or
more on the fit's 1 s clock, a quadratic trend taken out of every series,
`ln P_abs` against `v²` half a second earlier (the held 1.25 Hz packet's
mean age), the single node's ripple from the corner-blind heat rate, and
the tyre's lateral and longitudinal load. One least-squares fit per car;
the standard error is a jackknife over sessions.

| | Inferno 86 | FJ |
|---|---|---|
| read within the lap | 9.60e-6 ± 0.43e-6 | 1.86e-6 ± 0.07e-6 |
| fitted (v1.4.0) | 9.51e-6 | 1.22e-7 |
| leaving one date out | 9.32 to 9.82e-6 | 1.81 to 1.91e-6 |
| detrend order 1 to 3, with and without the ripple and the loads | 9.0 to 9.8e-6 | 1.60 to 1.94e-6 |
| damp stints | 6.9e-6 (one session) | 1.61e-6 ± 0.40e-6 |
| wet stints | 8.03e-6 ± 0.26e-6 | 1.62e-6 ± 0.16e-6 |

- The estimator gives the Inferno 86's fitted value back, which is what
  says it can be trusted on the FJ.
- **Per raw car and circuit** (the same tyres and sensors on the three
  formula cars):

  | | Tsukuba | Suzuka | Fuji | every circuit |
  |---|---|---|---|---|
  | KK-SII | 1.58e-6 ± 0.08e-6 (30 sessions) | 2.01e-6 ± 0.13e-6 (14) | 1.17e-6 ± 0.22e-6 (2) | 1.79e-6 ± 0.09e-6 |
  | KK-F | 2.21e-6 ± 0.16e-6 (11) | | | 2.21e-6 ± 0.16e-6 |
  | KK-F-V2 | 1.83e-6 ± 0.41e-6 (3) | | 1.67e-6 ± 0.11e-6 (8) | 1.73e-6 ± 0.10e-6 |

  At Tsukuba, the one circuit they share, the KK-F reads 0.63e-6 above
  the KK-SII (3.4 standard errors), so the raw cars do not read alike;
  but the KK-SII alone reads 1.58e-6 at Tsukuba and 2.01e-6 at Suzuka
  (2.9 standard errors), so the reading moves as much with the circuit as
  with the car, and what it follows is not known (the pressures run, the
  kerbs, what is left of the ripple and the load). The whole spread is
  ± 20 % of the car's value, 2 to 4 mbar at the end of a straight, where
  the fitted value is off by a factor of 15. A κ per raw car is not
  built: it is not what made the fitted value move, and one car's
  circuits differ as much as the cars.
- The wet stints read about 15 % lower than the dry on both cars (not
  explained; the option reads the dry stints, as the fit does).
- The four tyres read differently until the load is a regressor (FJ
  2.48 / 0.93 / 2.35 / 1.32e-6 without it, 1.90 / 1.52 / 1.76 / 1.86e-6
  with it): the outer tyre in a corner reads higher by about 0.55 K of
  gas temperature per g on the FJ and 1.2 to 2.3 K per g on the
  Inferno 86, a loaded tyre's cavity being smaller. It averages out over
  the car and is not modelled.
- This is the 1.5e-6 this section first reported for the FJ (2026-10-08).

**The option.** `kappa_source="within_lap"` (`fit_physical`,
`build_warmup_table`, `--kappa-source within_lap` on
`tire-build-warmup-table` and `tire-predict-holdout`; default `fit`,
`statespace.DEFAULT_KAPPA_SOURCE`): κ is read as above from the training
set's dry stints and held in the per-second fit, dry and rain, for every
car with 200 (lap, tyre) segments behind the reading; another car keeps a
fitted κ. The model then records the reading under
`energy_balance.speed_pressure.within_lap`. With the option off a fold's
model is the same bit for bit (checked on folds 0 and 1 against the code
before the option; only `fit_at_utc` differs).

**The gate**, written before any fold of the option was fitted
(`gate.md` in the worker's notes; 15 folds pinned by master's picker,
129 sessions, 5,703 rows, calculator inputs; a paired difference is per
held-out session, mean |residual| of the option less master's, ± the SE
clustered by date):

| check | master | option | |
|---|---|---|---|
| κ_FJ across the folds, sd / mean (pass: at most 10 %) | 77 % (3e-12 to 3.1e-7) | 0.8 % (1.83 to 1.88e-6) | pass |
| κ of the Inferno 86 across the folds, sd / mean (no larger) | 3.4 % | 1.2 % | pass |
| hot-pressure MAE, pooled (pass: paired at most +0.5 mbar) | 36.19 | 36.36, paired +0.19 ± 0.11 | pass |

| rows | MAE, mbar | bias, mbar | paired, mbar |
|---|---|---|---|
| FJ (75 sessions) | 31.14 → 31.52 | +7.2 → +7.2 | +0.43 ± 0.17 |
| Inferno 86 (54) | 43.16 → 43.04 | +0.5 → +0.5 | −0.15 ± 0.06 |
| dry (101) | 34.99 → 35.00 | +3.8 → +3.8 | +0.01 ± 0.08 |
| damp (4) | 42.83 → 44.50 | +21.7 → +24.3 | +1.56 ± 0.76 |
| wet (24) | 40.76 → 41.40 | +2.9 → +2.5 | +0.73 ± 0.37 |
| Tsukuba (63) | 32.42 → 32.63 | +8.7 → +8.9 | +0.20 ± 0.12 |
| Fuji (27) | 42.76 → 42.94 | −19.2 → −19.8 | +0.17 ± 0.44 |
| Fuji, FJ (5, wet, one date) | 40.25 → 43.30 | −27.0 → −31.6 | +2.26 |
| Fuji, Inferno 86 (22) | 43.15 → 42.88 | −18.0 → −18.0 | −0.30 ± 0.11 |
| Suzuka, FJ (13) | 34.66 → 35.40 | +5.8 → +6.1 | +0.68 ± 0.30 |

- The standing holdout holds out no dry Fuji session of the FJ (the
  KK-F-V2's bucket has nine). The two days that are out of sample, by the
  holdout's scoring: 2026-11-25 with both models fitted without the date
  (168 rows) 53.5 → 56.4 mbar, bias −50.8 → −54.1, and the same with the
  laps' own heat (53.7 → 56.6); 2026-11-27 on the all-data models with
  the simulated entries placed as the shipped artifact has them (68 rows)
  46.2 → 49.0, bias −20.1 → −19.9 (logged entries only 45.6 → 48.4; own
  heat 54.4 → 56.2).
- So the option costs 2 to 3 mbar on the FJ at Fuji. Elsewhere on the
  FJ the dry Tsukuba rows are level (28.34 → 28.39, +0.01 ± 0.12) and
  Suzuka (+0.68 ± 0.30), the damp and the wet rows are worse; the
  Inferno 86 is 0.15 better. Fuji was expected and written into the gate
  beforehand: the lap-end reading there is taken at the fastest point of
  the lap, a larger κ lowers the predicted reading there, and the fitted
  κ of about zero had been hiding that much of #394's under-prediction.
- **What was not run.** The option is scored on the calculator fed from
  logged entries (`validate`'s folds). It is not plumbed into
  `lapsim/pressure_gate.py`, so the folds' served arm with simulated
  entries and `just tire-sim-compare` have not seen it (the 2026-11-27
  figure above is the one place a simulated entry was scored).
- **What is assumed.** That the slope within a lap is also the offset
  between standstill and speed is confirmed on the Inferno 86, where the
  level part of the fit agrees with it, and assumed on the FJ, where the
  level part cannot be read.
- κ is not what moves the other constants: with κ held at the reading
  K_f still goes 6.29 → 6.65 and τ_f 453 → 501 s when the Fuji day enters
  (fitted κ: 6.23 → 6.53, 457 → 507). Across the folds K, τ and
  `g_transfer` vary as before (1.1, 1.7 and 2.7 % on the FJ).
- What a retrain with the option would change: κ_FJ 1.22e-7 → 1.86e-6,
  κ of the Inferno 86 9.51e-6 → 9.60e-6; the FJ's dry K +1.8 % (6.53 →
  6.66 front, 4.20 → 4.27 rear), its wet K +3.3 %, its compound K +3 %
  (DRY, dry) to +9 % (DRY, wet, rear), τ −1.2 %, `g_transfer` 2.74 → 2.80;
  the Inferno 86's constants under 0.15 %. On the parity fixture's 24 cases the
  recommended cold pressure falls by 4.6 mbar on the FJ (3.0 to 5.3; 9.0
  with the WET compound) and by 0.2 mbar on the Inferno 86. The
  calculators' hot pressure is the standstill gas-law value; with this κ a
  dash reading at 200 km/h is 0.6 % under it on the FJ (15 mbar), with the
  shipped κ 0.04 %. Whether the lower cold pressure is wanted depends on
  what the team's hot target is read from, the dash at speed or a gauge at
  rest, which is the owner's to say.

Not promoted: the default stays `fit` until the owner decides.

### 2.2b The out-lap (schema v4)

The stint does not start at the first flying lap. It starts when the car
leaves the pits: the driver has just set the cold pressures and read the
TPMS, then drives an out-lap that is slower and gentler than a flying lap
(g² about 0.3–0.6 versus 0.6–1.0) and whose length is not a lap time (pit
exit to the first start/finish crossing, often with a grid or pit-lane wait
first). Since schema v3 of the dataset the ETL keeps that out-lap
(`lap_type = "out"`, lap 0) and the model treats it as its own segment:

- **Fit.** The stint anchor is the out-lap's first valid TPMS reading — the
  pit-exit temperature and the cold pressure actually set — placed on a
  *rolling-time* clock (`moving_s`, time above 5 km/h; the grid wait is not
  warm-up time). The out-lap is an ordinary Pass 1 observation with its own
  measured g² and rolling time. Stints whose recording started mid-track
  (no pit exit) keep the previous first-lap anchor (`anchor_kind`).
- **Predict.** `outlap_typ_by_track_car_cond` carries the bucket's median
  out-lap rolling time and g². The predictor integrates that segment first
  from the typed current tire temperature, then `N` flying laps at the
  pace-scaled ⟨g²⟩ from the temperature at the end of the out-lap
  (`energy_balance.warmup_two_stage_c`). `lap_within_stint = N` therefore
  means the N-th flying lap, as before. `--no-outlap` and `--outlap-time-s`
  override it; artifacts without the table behave as pre-v0.26.
- **Evaluate.** The holdout's calculator inputs start from the pit-exit
  reading and include the bucket out-lap, and the pressure-domain residual
  is measured from the cold pressure actually set.

Before this the loader dropped the real out-lap and the ETL mis-flagged the
first *flying* lap of each stint as the out-lap and excluded it, so the model
never saw either, the stint clock started one lap late, and the typed pit
temperature was applied as if it were the temperature after the out-lap.

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
| `K = a/b`        | `α / (h_air + h_road)`                   | `(car, condition)` gain, `(car, axle, condition)` cooling | 2 + 4 per car | Energy-IN / Energy-OUT gain, emitted per corner |
| `τ_sec = 1/b`    | `m·c / (h_air + h_road)`                 | `(car, axle, condition)`          | 4 per car | Thermal time constant |
| `d`, `c_b`, `κ`  | drive-slip, brake-disc, speed–pressure   | `(car)`                           | 3 per car | §2.0a |
| `w_road`         | `h_road / (h_air + h_road)`              | **fixed at 0.2 in v0**            | 0       | Deferred |
| `⟨g²⟩`           | `median(heat_proxy / on_track_s)`        | `(track, car)` — lookup, not fit  | ~6      | From data |
| `lap_time_typ_s` | `median(on_track_s)`                     | `(track, car)` — lookup, not fit  | ~6      | From data |
| `T_road`         | logger / weather + sun proxy             | per-session                       | 0       | From data |

**Total fitted: ~9 constants per car** and **no track constants**. Compare
to a per-(track, car, lap) regression approach which would have hundreds.
The energy-balance framing means the **track signal is captured by the
per-corner ⟨q⟩ data, not by a fitted constant** — that's why the model is
track-independent at fit time and transfers to circuits it has never seen.

### 2.5a Logged entries per raw car (#383, measured 2026-11-26)

The FJ pools the KK-SII, the KK-F and the KK-F-V2 (`CAR_FIT_ALIASES`) for
every fitted quantity. The *entries* (the typical heat per tyre and its pace
curve, the typical lap time, the out-lap) were pooled too, so where one raw
car seldom runs a (track, condition) its entry rests on the other car's laps.
In the committee's thinning draws (#370) a logged entry resting on one
session of the other raw car was 1.65 ± 0.60 mbar per session worse than the
simulation, against 0.36 for one session of the same car on another date.

**What is built.** `warmup_table.RAW_CAR_ENTRIES` writes the three entry
tables again per raw car (`*_by_track_raw_car_cond`, §2.7), from the pooled
entries' own builders on each raw car's laps alone (the pace curve through
the pooled car's shares). A raw car has a row only where it has its own heat
at the cell; its lap time, curve and out-lap come with it. The lookups
(`predict._entry_row`, the web and C# ports alike) walk the condition chain
and at each step read the raw car's own row before the pooled car's, so the
condition comes first: a KK-F damp session with no KK-F damp entry reads the
FJ damp entry, not the KK-F dry one. Each table walks the chain alone, so an
out-lap can come from another step than the heat. The `track_car_pooled`
step reads the raw car's rows at the track first; the track and global
fallbacks average the pooled rows only. K, tau, kappa, the shares, the
compound K and the corner defaults stay pooled. An artifact without the
tables reads exactly as before (the committed one: every raw name gives its
pooled car's prediction, `test_predict`). The standing holdout scores each
held-out lap by its raw car (`validate._score_fold_laps`); a simulated entry
placed for a pooled car answers its raw cars too (`pressure.place` takes
their rows out of the cell), so the simulated arms are unchanged (fold 0:
`table`, `direct` and `direct_setup` row for row). Gate F's recommendations
still read the pooled entries.

**The gate, written before scoring** (in the commit): on the standing
holdout's 15 folds, one fit per fold, arm A with the pooled entries only
(today) and arm B with the raw cars' own first. Hard checks: A reproduces
the standing holdout, the unexposed sessions are identical, the simulated
arms unshadowed. Adopt if B ≤ A pooled, the 57 exposed sessions' mean
per-session B − A ≤ 0, the KK-F's damp sessions at Tsukuba (the cell that
lost 22 mbar when the KK-F was fitted apart, constants and all) no worse than
+2 mbar, and no exposed raw car × condition cell worse than +3 mbar.

**Exposure.** 57 of the 128 held-out sessions answer from a bucket holding
another raw car in training, all at Tsukuba; the other 71 (the Inferno 86,
the KK-SII at Suzuka and Fuji) are identical in A and B.

**Result** (hot-pressure MAE, mbar; ± the date-clustered SE of the paired
difference; MDE = 2.8 × 0.06 × the rms change in predictions, #386):

| set | sessions | MAE A | MAE B | B − A | rms ΔP | MDE |
|---|---|---|---|---|---|---|
| pooled | 128 | 43.71 | 43.53 | −0.18 ± 0.15 | 2.4 | 0.41 |
| FJ | 75 | 35.29 | 34.98 | −0.31 ± 0.25 | 3.2 | 0.54 |
| exposed | 57 | 35.23 | 34.87 | −0.36 ± 0.29 | 3.5 | 0.58 |
| KK-SII | 53 | 31.76 | 31.22 | −0.54 ± 0.34 | 3.5 | |
| KK-F | 22 | 42.06 | 42.19 | +0.13 ± 0.30 | 2.7 | |
| KK-SII Tsukuba dry | 29 | 30.73 | 29.72 | −1.01 ± 0.24 | 3.6 | |
| KK-SII Tsukuba wet | 6 | 31.03 | 32.27 | +1.24 ± 1.82 | 5.1 | |
| KK-F Tsukuba dry | 11 | 34.94 | 35.32 | +0.38 ± 0.38 | 2.8 | |
| KK-F Tsukuba wet | 7 | 32.84 | 32.43 | −0.41 ± 0.74 | 3.5 | |
| KK-F Tsukuba damp | 4 | 73.16 | 73.13 | −0.03 ± 0.03 | 1.0 | |

- Every clause of the gate passes: B better on 34 of the 57 exposed
  sessions and worse on 23; the KK-F's damp cell does not move (its pooled
  damp entry was 60 of 64 laps the KK-F's own); the worst cell is +1.2.
  The size is the committee's reasoned −0.1 and under the MDE: the holdout
  does not resolve it.
- By #386's reading: first-order term −0.25 (between cell × tyre −0.71,
  within +0.46), curvature +0.07. A random-sign placebo of the same size
  (the change flipped per fold × raw car × track × condition × tyre, 119
  units, 4,000 draws) averages +0.07 (sd 0.10); the observed −0.18 is below
  99.6 % of the draws. The gain is the between-cell term, in the direction
  the mechanism predicts (the KK-SII's predictions fall, the KK-F's rise,
  each toward its own car), which is also the direction of the cell biases
  (+15 and −12 mbar), so the TPMS cannot tell a better heat from a bias
  moved the right way.
- Leakage (#386: most held-out sessions have a same-day session in
  training): on the 13 exposed sessions whose raw car has no same-date
  session in the training cell, −0.70 ± 0.31; on the 44 that have one,
  −0.23 ± 0.38.
- The fits are those of the standing holdout: A reproduces its rows (folds
  0–1 to 1e-15 bar, folds 2–14 within 0.04 mbar, which is the fit's thread
  count, not this change: the branch's fit at two threads is the cached
  fold's, and master's fit path at eight is the branch's).

**Decision.** On by the owner (2026-11-26: "ok, do it with the retrain"),
by the gate written beforehand, which passed; the gain is under what the
holdout resolves. Until a retrain writes the tables nothing changes. After
one, the raw names read their own entries in every predictor (the Python
CLI, the web and C# ports, the parity fixture's KK-SII cases), and the
calculators' car picker lists the raw cars (Inferno 86, KK-F, KK-F-V2,
KK-SII) in place of the FJ; a saved FJ keeps working on the pooled entries
and stays in the list while it is selected; an artifact without the tables
lists the pooled cars as before. The constants and the prefills (compounds,
corner defaults, hot-pressure targets) resolve a raw name to the FJ.

**The guard** (rule (b), chosen 2026-11-26): a raw car's own entry is
written only where it rests on at least two training sessions and has a
pace curve (`warmup_table.RAW_CAR_MIN_SESSIONS`, `RAW_CAR_NEEDS_CURVE`);
elsewhere the raw car reads the pooled entry. The gate above was scored
with the guard off, and the guard changes none of its rows: in every one
of the 15 fold models the raw entries without a curve are the KK-SII's damp
ones at Suzuka and at Tsukuba, and taking them out leaves every lookup of
every held-out session (heat and out-lap per tyre, lap time, curve)
unchanged; the raw entries the held-out sessions read rest on two
training sessions (the KK-F's damp at Tsukuba) or more, each with a curve.
In the full fit on the committed data it drops the KK-SII's damp entry at
Tsukuba (one session, 4 laps, no pace curve; the pooled one
holds 5 sessions, 60 of its 64 laps the KK-F's), so that cell reads the
pooled entry. The KK-F-V2's dry entry there (3 sessions, 38 laps) was not
measured by the holdout; it is written if it has a curve.

### 2.6 Fitting procedure

One pass, `statespace.fit_physical` (§2.0, §2.6a): the per-second fit of the
physical model on every stint, dry cells first (with `κ`, `d`, `c_b`), then
the rain cells with `b_rain ≥ b_dry` bounded and the per-car constants held.
There is no second pass: with no track constants the fitted `K = a/b_axle`
is the model, and the per-corner ⟨q⟩ lookups are computed from the fitted
stints' own per-lap heat components (`statespace.lap_heat_components`).
The earlier two-pass procedure (a per-lap closed-form Pass 1 and a Pass 2
that factored per-track gains into `K × c_track`) was removed with schema
v5.

### 2.6a Pass 1 at 1 Hz: the per-second fit (default since 2026-10)

The per-lap closed form sees one pressure sample per lap. The TPMS reports
pressure in 0.03 bar steps, so a lap-end sample is uncertain by ±0.015 bar
(≈ 2 K of gas temperature) and the pit-exit sample the whole stint is
referenced to is uncertain by the same amount; on the Inferno 86 that
quantisation was the dominant residual once the observable became the gas
temperature (§2.2a). At 1 Hz a rising pressure crosses a step every 10–30 s
early in a stint and the *time* of each crossing locates the pressure to a
fraction of a step, and the measured g²(t) through the lap identifies τ and
K from each lap's shape rather than from lap-end levels alone.

`statespace.fit_physical` (via `fit_cells`) integrates the energy balance
exactly with the inputs held constant over each 1 s step of the stint's
rolling clock (standstill excluded; a lap the usability filters dropped
mid-stint still heats the tire and advances the clock, it is only not
scored):

    dT_i/dt = a · q_i(t) − b_axle(i) · (T_i − T_eff),   T(t_anchor) = T_start
    K = a / b_axle,  τ = 1 / b_axle

The observation is the pressure-implied gas temperature from the pit-exit
anchor for every second with a finite pressure at or after the anchor (a
(stint, corner) needs ≥ 60 scored seconds). `a` is fitted per (car,
condition), `b` per (car, axle, condition), `d`, `c_b` and `κ` per car, by
bounded least squares in log space (`τ` 30–3000 s); dry first, then the
rain conditions with `b_rain ≥ b_dry` (τ_rain ≤ τ_dry) as a bound and the
per-car constants fixed, a rain (car, condition) cell needing ≥ 3 sessions.
The whole two-car fit takes under a minute (vectorised cumulative-sum
recurrence, 450 k residuals). Parameter recovery on synthetic stints, with
and without the 0.03 bar quantisation and with the speed-pressure effect, is
unit-tested.

History: the per-second fit replaced the per-lap closed form in 2026-10
(τ 5–15 % shorter, K 5–15 % higher, the τ·K product unchanged; held-out
results in §4.0a); the per-track constant it still carried then was
removed with schema v5 (§2.0).

### 2.6b Heat-input and cooling experiments (2026-10-07, `tire_model/heat_experiment.py`)

The per-second fit makes alternative physics cheap to test: a candidate
heat input or cooling law is dropped into the same 1 Hz integration and
scored on the same held-out stints. The harness fits ``(a, b)`` on 5-fold
session holdouts (the ``tire-predict-holdout`` buckets) or
leave-one-track-out, simulates every held-out stint **from its own measured
trace and pit-exit anchor** (oracle inputs — the thermal model's accuracy,
not the calculator's), and scores the lap-end residual in the pressure
domain. It also reports how *constant the constants are*: the IQR/median
of per-stint implied gains, the per-track median gain ratio (1.0 when the
heat input explains the track) and the wet/dry gain ratio. (The runs below
pre-date the removal of the track constant; their ``c_track`` columns show
what a fitted per-track factor did under each heat input.)

```
uv run python -m motorsports_data_notebook.tire_model.heat_experiment \
    --form g2 --form sliding --form "sliding:glim=1.15" --n-folds 5
```

**Heat-input forms** (`tire_model/heat_input.py`). The ChassisSim
"tyre model from nothing" writes the heat input as sliding power,
``√((F_y·α·V)² + (F_x·SR·V)²)``; with one slip proxy for both axes that is
``V · g · s(g)`` with ``g`` the friction-circle total g. Without slip
channels ``s(g)`` was taken as the inverse of a ``tanh`` force saturation,
``g_lim·atanh(g/g_lim)`` (linear well below the car's grip limit, steep at
it), with ``g_lim`` from the car's p99.5 of total g.

| heat input                    | held-out P MAE (bar) | c_track fuji / sodegaura | wet/dry gain (Inferno) |
|-------------------------------|---------------------:|--------------------------|-----------------------:|
| ``g²`` (production)           | **0.0550**           | 1.04 / 0.86              | 1.05 |
| ``V·g²`` (sliding, linear slip)| 0.0567              | 0.93 / 0.83              | 1.23 |
| ``V·g·s(g)``, g_lim = p99.5   | 0.0593               | 0.88 / 0.83              | 1.67 |
| ``V·g·s(g)``, g_lim = 1.3×    | 0.0575               | 0.91 / 0.83              | 1.38 |
| ``V·g² + 0.2·V`` (rolling)    | 0.0557               | 0.90 / 0.84              | 1.02 |
| ``V·g^2.5``                   | 0.0582               | 0.91 / 0.83              | 1.53 |
| ``V·g³``                      | 0.0598               | 0.90 / 0.83              | 1.84 |

Every speed- or slip-weighted form is less accurate than ``g²``, moves the
track factors further from 1 and (for the activation) makes the wet gain
exceed the dry gain, which is unphysical. At this node the heat grows no
faster than ``g²`` with pace. The speed and load² hidden in α (see the
derivation: sliding power ≈ ``W²·g²·V/C_α``) are evidently stable enough
within a car to stay in the constant. **``g²`` stays.**

**Speed-dependent convection** (``b(t) = b·(1 + β·V/30)/(1 + β)``,
``--cool-mode speed``): β = 0.50 pooled with folds 0.27–0.81, β = 0 on the
FJ alone; held-out 0.0548 vs 0.0550. Not supported at the gas node.

**Where the error actually is.** 73 % of the held-out MAE is a
per-session offset (the per-stint implied gains have an IQR/median of
0.28 on the FJ and 0.41 on the Inferno). Weather explains none of it
(|r| ≤ 0.15 against T_eff, cloud, sun, wind, humidity, hour). The bias is
+0.025 bar from lap 1 and flat thereafter, and in-sample it is +0.015 bar
at lap ends against +0.005 over all seconds, concentrated in the first and
last tenth of the Inferno's laps (the straight, g² ≈ 0.14): the start/finish
reading sits below what a six-minute single node predicts. That is the
tread-lag item (a fast tread node feeding the gas), not a heat-input
question.

**Set pressure.** The one session-level variable that explains the offset
is the pit-exit set pressure: r = 0.53 (FJ) and, with compound dummies,
a slope of ≈ +0.15–0.17 bar of residual per bar of set pressure on both
cars; the correlation holds *within* every track (0.35–0.69). Fitted as a
heat-input factor ``(P_ref / P_set)^n`` with ``P_ref`` = 2.5 bar abs
(``--p-mode anchor``), n = 1.50 pooled (folds 1.37–1.54), held-out
0.0509 bar (−7.5 %), FJ 0.0454 → 0.0429, Inferno 0.0660 → 0.0602, per-stint
gain spread 0.28 → 0.24 (FJ). Per car: FJ n = 3.06 (folds 2.75–3.4),
Inferno n = 1.39 (1.29–1.42), 1.95 with compound cells split out.

Three controls keep this out of the production model for now:

- *Instantaneous* pressure carries no effect (``--p-mode instant``:
  n = 0.27, folds 0.0–0.57; 0.0543 bar). Deflection physics would act on
  the running pressure. A reconciliation exists — rubber modulus and gas
  pressure both scale with absolute temperature, so ``P/E`` and the
  deflection are invariant within a stint and set by the cold inflation
  (the gas *density*) — but it is a hypothesis, not a measurement.
- *Within a day*, a run-to-run change of set pressure moves the residual
  with the same slope on the FJ (0.166 bar/bar, 36 pairs, r = 0.55) but not
  at all on the Inferno (0.016 bar/bar, 34 pairs, r = 0.08) whose
  between-session exponent is nevertheless 1.4–1.9. On the Inferno the
  signal is a session-level confound (compound: A050 runs at 1.2–1.4 bar,
  A052/RE-71RS at 1.5–2.2; unlabeled sessions stay pooled).
- The FJ's set pressure also tracks the weather (r = −0.55 vs T_eff) and
  the date (−0.60): drivers set higher cold pressures when they expect
  less warm-up. The partial correlation given T_eff, T_start and g² is
  still 0.63, so anticipation of the *weather* is not the explanation, but
  anticipation of an unmeasured tyre state (wear, heat cycles) cannot be
  excluded with this dataset, and n = 3 is larger than deflection physics
  alone predicts.

The harness keeps the term available (``fit_cells(..., p_mode="anchor")``
with a global exponent and a synthetic-recovery test) so it can be
promoted once a physical variable behind it is identified or a controlled
pressure sweep on one tyre set is run.

### 2.6c Two-node tread + carcass model on the IR tread sensors (2026-10-08, `tire_model/twonode.py`)

The Inferno carries an 8-zone IR tread-temperature array per corner on 68
sessions (the KK-F's 8 sessions have no usable IR: unplugged −200 sentinel
or a covered sensor). A two-node model puts the ChassisSim heat input on a
tread node that the IR observes and a carcass node that the gas observes::

    dT_s/dt = a_s·c_track·q − b_sa·(T_s − T_eff) − b_sc·(T_s − T_c)
    dT_c/dt = b_cs·(T_s − T_c) − b_c·(T_c − T_eff)

fitted jointly on the IR mean-of-zones and the pressure-implied gas
temperature (σ 4 K / 2 K), integrated exactly per second with the 2×2
matrix exponential — loop-free via the modal decomposition, with a Picard
iteration when the surface cooling varies with speed.

**IR gating** (`twonode.ir_channel_ok`, per stint × corner): ≥ 300 moving
seconds, median > 25 °C, max < 150 °C, p95−p5 swing ≥ 15 °C, 1 s step std
≥ 0.8 °C, median across-zone range ≥ 5 °C (a tread has a lateral profile,
bodywork does not), correlation ≥ 0.4 with the 10 s mean of g², and median
tread − gas ≥ −3 °C (a running tread cannot sit below the gas). 156 of 224
Inferno dry channels pass (42 sessions); the rejects are short stints,
sensors reading below the gas, and near-ambient flat channels.

**Result** (5-fold, Inferno dry, lap-end pressure, oracle inputs, κ not
yet applied):

| model | P MAE (bar) | tread MAE | constants |
|---|---:|---:|---|
| single node (gas only) | 0.0673 | — | τ 440–660 s |
| two node, 5 constants per corner | 0.0682 | 7.1 °C | τ_s 9–88 s, C_c/C_s 1.3–11: under-identified |
| two node, constants shared across corners | 0.0674 | 7.5 °C | τ_s 61 s (53–87), τ_c 249 s (242–261), C_c/C_s 2.0 (1.3–2.5) |
| + speed-dependent tread cooling | 0.0648 | 7.1 °C | τ_s 90 s, β 1.2 (0.7–1.5), C_c/C_s 1.2 (0.9–1.4) |

Sharing the thermal constants across the four corners of one tyre makes
them fold-stable; the carcass loses heat only through the tread (``b_c``
pins to its lower bound in every variant, so the carcass node can drop a
constant). The tread node reproduces the within-lap IR swing on long
stints but not the deep cooling on Fuji's straight, and the **gas-side
accuracy barely moves**: the +0.024 bar lap-end bias survives every
variant, which is what led to the speed-pressure correction above
(§2.2a′) — the within-lap gas swing is mechanical, not thermal. The
two-node fit should be re-run with κ applied to the gas observable before
drawing conclusions about the tread constants.

### 2.7 Artifact: `tire_model.json`

The model serializes to a versioned JSON committed under
`data/tire_dataset/tire_model.json` (`schema_version: 5`). Top-level keys:

Fitted tables:

- `tau_sec_by_car_corner_cond` — warmup time constants with stderrs
- `K_buckets` — pooled K per (car, corner, condition) with stderrs and
  `from_single_track` flags
- `K_by_car_compound_corner_cond` — compound-specific K (decomposed
  base × multiplier products, ready to use; see §2.11)
- `K_compound_multipliers` — the fitted per-(car, compound) ratios, for audit
- `heat_input` — the schema-v5 heat input: form, `v_ref_ms`, `force_exp`,
  `speed_exp`, `geometry_by_car`, `drive_by_car`, `brake_by_car`
- `g2_typ_by_track_car_cond` — ⟨q⟩ lookup; each entry carries the
  per-corner `q_typ_by_corner` (schema v5) and may carry a
  `g2_vs_lap_time` piecewise-linear curve with per-corner series
  `g2_by_corner` (the simulated faster lap, §2.10)
- `lap_time_typ_by_track_car_cond` — typical lap time lookup
- `corner_defaults_by_car_corner_cond` — steady-state median hot temp +
  hot pressure, used by the calculators to prefill the corner-card targets
- `outlap_typ_by_track_car_cond` — typical out-lap rolling time and g² per
  bucket (schema v4, §2.2b); consumers without it treat the out-lap as
  zero-length
- `g2_typ_by_track_raw_car_cond`, `lap_time_typ_by_track_raw_car_cond`,
  `outlap_typ_by_track_raw_car_cond` — the same three entries per *raw*
  car, for each car `car_aliases` pools under another name (additive in
  schema 5, `warmup_table.RAW_CAR_ENTRIES`, §2.5a): the same row shape with
  `car` the car the session names. A raw car has a row only where it has
  its own heat at the (track, condition) on at least two training sessions
  with a pace curve (the guard), and its lap time and pace curve come with
  it. Consumers that do not know the tables read the pooled
  entries as before
- **Simulated rows** (the cutover; additive in schema 5;
  `sim_entries.place_simulated`): rows of the three entry tables and their
  raw-car siblings that carry `source: "sim"`. A row without `source` is
  logged. A simulated row comes from a lap driven on a committed track
  (`data/lapsim/tracks/`, whose road bank is processed from GSI's elevation
  tiles), and any prediction that reads one shows GSI's attribution.
  - **Where they sit:** the FJ's raw cars' rows go in the raw-car tables;
    the Inferno 86's go in the pooled tables. The pooled FJ rows stay
    logged. A simulated row is added beside the logged row of the same
    key, never in its place.
  - **Shape:** the row shape is unchanged. `n_laps_used` is 0. A simulated
    heat row's `g2_vs_lap_time` is the pace table's knots: any number of
    them, rising lap times, `method: "sim"`, read linearly as every curve
    is. It carries `support: {lap_time_min_s, lap_time_max_s}`: the lap
    times of the car's logged range of use at the layout, or of its range
    over its layouts where it has not run this one. A wet or damp row's slow
    end is the wet laps'.
  - **Wet and damp:** a wet or damp cell carries a simulated row only where
    the model has a logged lap time for that exact condition (rule (c),
    §2.12 "The cutover"). Elsewhere it reads the logged chain.
  - **How they are read** (schema §9 of the cutover; `predict._entry_row`,
    `_answering_rows`, mirrored in both ports):
    - A simulated row answers only its own exact condition, chosen by its
      `source`, never by its place in the table. The order at the exact
      step is raw sim, raw logged, pooled sim, pooled logged.
    - A condition fallback (wet → damp → dry) reads logged rows only, the
      raw car's before the pooled car's.
    - The averaged steps (`track_car_pooled`, `track_pooled`, `global`)
      average logged rows only, and a step with none is skipped. So a car
      with no simulated entry of its own reads what it read before.
  - **What a prediction reports:** `predict_cold_pressure` (and both ports)
    reports `heat_provenance`, `lap_time_provenance`, `outlap_provenance`
    (`sim` / `logged`) and `uses_simulation`. It also reports
    `pace_status` for the lap time, read from a simulated curve only:
    - `no_solution` under its fastest knot (answered there, by the clamp);
    - `outside_support` outside `support`, or past the slowest knot;
    - `ok` otherwise.

    Both are display only: the calculators show the GSI line and a short
    flag. No number depends on them.
- `entry_coverage` — one row per cell the calculators offer (every track
  with a row × every car the picker lists × dry, damp, wet), with
  `layout_id`. For each of `heat`, `lap_time`, `outlap` and `curve` it gives
  the kind of row the lookup answers with: `sim`, `logged`, `averaged` (a
  track or global mean) or `none`. It also carries `sim_only`, and a
  `reason` where nothing answers `sim`. The ports ignore it at run time,
  and their tests hold their own lookups to it.
- `simulation` — the cutover's provenance: `placed`, `export` (when, from
  which commit and dataset), `conditions`, `rules`, `cells`,
  `logged_cells_moved` (empty by construction) and `attribution`. The last
  is GSI's wording in `ja` and `en`, which the tests hold equal to the
  calculators' `GsiAttribution` string.
- `rain_thermal` — how rain τ/K are fitted (τ bound, session minimum; §2.9)

Config + provenance:

- `car_aliases` — raw car label → pooled fit label (e.g. `KK-F` / `KK-F-V2`
  / `KK-SII` → `FJ`: near-identical FJ-series machines on the same tires
  train one car; the KK-F-V2 is the KK-F with a new intake). Every
  constant is looked up by the pooled label, so raw names keep working;
  the entries (heat, pace curve, lap time, out-lap) are read for a raw car
  from its own rows first, at each step of the condition chain, then from
  the pooled car's (§2.5a). The track and global fallbacks average the
  pooled tables' logged rows only.
- `g2_lap_time_model` — pace-scaling method (`uniform_scaled_lap_shares`), physics fallback exponent, multiplier clamp
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

### 2.10 Target lap time: the simulated faster lap

A **target lap time** is an optional prediction input that sets both the
time-on-track clock (`t = N × target`) and a multiplier on each corner's
typical heat input ⟨q⟩. The multiplier answers a counterfactual — the same
car on the same circuit lapping at a different pace — and since 2026-10-30
it comes from the logged laps themselves, simulated per second through the
production heat equations (`tire_model/pace_sim.py`), with nothing fitted:

1. For each (track, car, condition) bucket take the 15 training laps
   nearest its typical lap time, sample by sample (speed `V`, lateral g,
   longitudinal g, duration `dt`).
2. Scale the whole lap uniformly to the target lap time with the speed
   multiplier `m = T_lap / T_target`: `V' = m·V`, `lat' = m²·lat` (same line,
   same corner radius), longitudinal g unchanged (braking and traction at
   their limits), `dt' = dt / m`. The lap time is then exactly `T_lap / m`.
3. Push the rescaled samples through the per-corner force shares of the
   schema-v5 heat input (`q_i = V·|F_i|`, §2.0) and take each corner's
   lap-mean heat relative to the unscaled lap.

The median over the reference laps, on a grid of 9 lap times from 0.75 to
1.5 × typical, is stored as the bucket's `g2_vs_lap_time` curve: `g2` is the
corner mean (for consumers without per-corner curves), `g2_by_corner` one
series per corner. Prediction scales each corner's ⟨q⟩ by
`curve_i(target) / curve_i(lap_time_typ)`, so an omitted target (or target
== typical pace) changes nothing; the multiplier is clamped to
`multiplier_clamp` (0.4–2.5). A bucket with fewer than 5 simulable laps
falls back to the physics exponent `default_exponent = 3`
(`q = V·|F|` with `F ∝ V²`). Because the extra lateral force lands mostly
on the loaded outer tyre, the left tyres respond more than the right on
these clockwise circuits (FL ≈ 2.3× vs RR ≈ 1.9× at 0.75 × typical).

**Why this transform.** `just tire-pace-check` is the within-session test
of the simulation itself: per session, how much hotter the fastest clean
laps (1.5–6 % under the session's median) ran each corner than the median
laps — observed from the per-second heat model's own per-lap intensities —
against what the simulation predicts for that lap-time gap. On the
2026-10 dataset (37 sessions) uniform scaling is unbiased on all four
corners (|bias| ≤ 0.005 in log ratio, slope of observed on predicted ≈ 1.0).
The alternatives were rejected on the same test: scaling only the
corner-limited samples over-predicts the heat by 1–3 % (a faster lap is
faster everywhere); a dynamically consistent point-mass lap with braking
and traction held at the logged lap's limits over-predicts by 1–2 % (a
median lap is not at those limits); the previous sector kNN-median curve
(pooled per-lap laps, where slow laps are slow for non-cornering reasons)
reproduced only 0.6 of the observed slope and was clamped at the fastest
lap ever driven; a fitted lap-time exponent of 4 — the between-session
holdout optimum — over-predicts by 4 %, i.e. it absorbed day effects
(cold track, damp patches) rather than pace. On the 5-fold calculator
holdout the simulated lap scores the same as the old curve
(0.048/0.047/0.044/0.038 bar hot-pressure MAE vs 0.049/0.047/0.045/0.038)
while being several times more responsive: at Fuji in the FJ a 5 % faster
target now moves the predicted hot temperature by ≈ 5 °C instead of 0.4 °C.
The residual held-out bias still leans with session pace (slower sessions
over-predicted); together with the within-session result that points at
day conditions the model does not yet carry, not at the pace physics.

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

    K_effective = K_base[car, corner, condition] · m[car, compound]

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
A seed belongs to the tyres a car runs, so the sidecar lists it per raw
car and it is matched on the car the session names (``raw_car``), not on
the fit's pooled car (#393, 2026-11-26): the KK-SII, the KK-F and the
KK-F-V2 run the same FJ slick and rain tyre and carry the same seed. Until
then the KK-SII's seed reached the KK-F and the KK-F-V2 only through
``CAR_FIT_ALIASES``, which the sidecar's comment denied ("the KK-F is
deliberately not seeded"), ``infer-compounds`` never applied (its seeds
were not aliased, so no FJ session was seeded there) and ``--split KK-F
KK-F-V2`` silently dropped for the split cars (that experiment's KK-F ran
with no DRY/WET seed). Every one of the 20 seeded KK-F-family sessions (29
with 2026-11-25's) has the seed its pace supports (dry at 0.98–1.03 × the
car's dry pace, wet at 1.09–1.12); the session the comment worried about (a0381, slicks on a track the wetness model calls wet) carries a
notes label, and a written label beats a seed.
Soft assignments are training-only; held-out evaluation uses human/seed
labels exclusively. ``just tire-model infer-compounds`` audits every
latent assignment for human review.

**Set-pressure prior (2026-10).** Compounds are run at characteristic cold
pressures (Inferno 86: A050 at 1.24–1.41 bar, A052 and RE-71RS at 1.5–2.7),
which the thermal likelihood cannot see. The EM therefore multiplies the
E-step by a per-(car, compound) Gaussian on the session's pit-exit pressure,
fitted from the labeled sessions when every compound of the car has ≥ 3 of
them (`PRESSURE_PRIOR_MIN_SESSIONS`, σ floored at 0.12 bar). This settles
the A050 sessions outright; A052 versus RE-71RS overlap in pressure and stay
a thermal question.

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

**The owner's word over the archive (#395, 2026-11-26).** The archive is a
reanalysis grid, and at the damp boundary its label is no better than a guess.
The water balance called the KK-F-V2's dry Fuji afternoon of 2026-11-25 damp
(a0434–a0441: 0.2–0.5 mm/h at 12–15 JST, a film of 0.07–0.24 mm), on the
day's fastest laps; the owner, who was there, says the track was dry all day.
Against the run notes, on 187 sessions with the 2026-11-25 ones:

- where the notes give dry or wet, the archive agrees (2 of 2, 13 of 13);
- all 12 disagreements are sessions the notes call damp, which the archive
  labels dry (6: Fuji 2025-11-01, Sodegaura 2026-08-15 "rained part way
  through") or wet (6, at a film of 0.58 mm or more);
- the archive's damp has one written confirmation (a0385, "slightly damp,
  slicks", the KK-F at Tsukuba on 2026-10-25) and now five refutations.

Lap pace does not tell dry from damp either: 8 of the 13 noted-damp sessions
are within 2 % of the car's dry pace at the layout (a0385 at 0.996 of it, the
fastest of its day). So no rule on pace, on notes alone or on the film
threshold was taken; each would relabel sessions already in training whose
truth nobody has written down, and is the owner's call:

- `DAMP_FILM_MM` raised to 0.25–0.30 separates 2026-11-25 (≤ 0.242 mm) from
  a0385 (0.314 mm) and turns 5–6 other unverified sessions dry;
- "damp needs a note" turns 10 other unnoted damp sessions dry (about 30 of
  187 sessions have a written condition at all);
- "notes outrank the archive" relabels 12 sessions, and two noted "snow"
  have no category.

Instead `data/tire_dataset/condition_overrides.yaml` holds the owner's word
per track day (`track`, `date`, `condition`, `evidence`).
`wetness.apply_condition_overrides` puts it over the computed label wherever
the label is made: `warmup_table._attach_weather` (the fit, the holdout's
folds and their scoring, every caller passing
`wetness.load_condition_overrides` of its root) and the lap simulation's
`geometry.session_conditions` (its lap index is keyed on the file's bytes, and
the fold cache lists it in `pressure_gate._FIT_SIDECARS`). The film depths
stay as computed. Its one entry, Fuji 2026-11-25 dry, relabels exactly the
five sessions (damp → dry) and nothing else; none of them is in the dataset
before the 2026-11-25 refresh, so the standing holdout on the committed data
is unchanged by construction.


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

### 2.12 Lap simulation: the track model and the car model (issues #202, #204, #205)

Exploration code in `tire_model/lapsim/`; nothing here is read by the
production model. Reports and caches go to `workspace/lapsim/` (not
committed), so the gate numbers are recorded here. Commands:
`just tire-lapsim <command>`, under `ulimit -v 10000000`.

#### B1. Lap geometry on one clock (`geometry.py`, gate `check-geometry`, 2026-11-15)

What a track is built from is computed per session at query time; the
extractor and the dataset are untouched.

- **Course over ground from the stored GPS yaw rate.** `gps_yaw_rate_dps`
  is libxrk's one-sample difference of the Doppler course `atan2(vE, vN)`
  over the raw timecode step, so `Σ yaw·Δt` gives the course back exactly.
  An irregular step (35–120 ms) is summed as logged, since the stored rate
  was divided by that same step. The sum is cut only at a non-finite rate, a
  non-positive step or a gap over 0.12 s; each piece's constant is set from
  the heading of the position chords, which makes the course absolute.
- **Lag.** The receiver's velocity solution trails its position solution.
  `course − chord heading = −τ·ω`, and the slope on the yaw rate is the
  session's `τ`. Speed, course and `gps_lat_g` are handed out at `t + τ`, on
  the position clock; segments C–F take V from this loader.
- **A local frame on the WGS84 ellipsoid.** The extractor's `_local_xy`
  uses one sphere radius: at 36° N that reads north-south distances 0.21 %
  long and east-west ones 0.23 % short. A curvature profile cannot carry
  an anisotropy, so `geometry.local_xy` uses the ellipsoid's two radii of
  curvature. The origin is the layout's start/finish gate, so every session
  of a venue shares the frame. `_local_xy` and `position_kinematics` are
  left as they are (they are the reference the IMU verdicts were scored
  against; the 0.2 % is far inside those bands).
- **Laps keep `lap_num`.** The gate crossing only places `s = 0`: a lap is
  the crossing interval that coincides with the dataset lap of that number
  (common time ≥ 98 % of the longer of the two).
- **Extrapolated crossings.** The logger closes a file on its own beacon,
  3–5 m before the registered gate point, so the last lap of a file has no
  fix beyond the line and no crossing. Where a recording stops within 10 m
  before the line (or starts within 10 m after it), the crossing is put at
  `t + distance / speed`. This was the unexplained "one lap in each of 77
  sessions": 102 full laps at a normal lap time, in 87 sessions, had no
  interval; 100 now match and 76 of them are clean. Such a lap's trace
  stops at the last fix.
- **Position slews are flagged, not repaired**: a lap where
  `|V²κ_position − gps_lat_g|` exceeds 1.5 g (both over 0.5 s, on the
  position clock, κ from the extractor's `position_kinematics`).
- **Gyro scale** per lap: the gyro's integrated yaw regressed on the
  course with a free zero offset (`∫ω dt = a·course + b·t + c`). A lap's
  own closure would read the zero offset as scale.
- **Clean lap** as defined on #202; the session best is the fastest lap
  that passes every test but the pace one. **Build lap**: clean, not
  flagged, from a session whose course and position heading agree within
  0.5° once the lag is removed, with a session best within 1.05 × the
  car's third-fastest session best at the layout, and a lap time within
  1.02 × the session best. `geometry.clean_lap_index(root)` returns every
  lap with both verdicts and their reasons.
- **The build does not use the wet label** (decided by the owner,
  2026-11-15: "the track geometry doesn't change in the wet"). #205's
  review had added a veto on sessions the wetness model labels wet. The
  road is the same wet or dry; a session that really was wet is slow and
  falls to the pace rule; and the label, which comes from the hourly
  weather and not from the laps, was also removing sessions that had
  dried: 11 of the Inferno 86's 14 Tsukuba sessions carried it, three of
  them with bests of 69.8–72.8 s against 69.4 s for its second-fastest dry
  session. It is also the design review's own finding 10, that a rule may
  not depend on a label unknown on 152 sessions. Without the veto the
  build gains 30 laps and 3 days (measured on the dataset through
  2026-11-13: 1,257 laps in 208 sessions with it, 1,287 in 215 without), and
  the Inferno 86 has five sessions on four days at Tsukuba where it had
  three on three. The KK-SII's first Fuji day (2026-02-25) stays out on
  pace alone (bests of 133–140 s against 117 s the next day).

Two more things differ from the issue as written, both forced by
measurement:

1. **The closure statistic.** A lap's raw course change is not a whole
   number of turns: the car crosses the line at an angle that differs by a
   degree or two from lap to lap (Tsukuba: sd 0.5 % of a turn). Referred
   to the position heading within 0.5 s of each crossing, the closure is
   ±0.0002 turns (P10–P90), which is what the review measured.
2. **The receiver rule.** The issue left open whether logger 6520348's
   sessions enter a build. The scatter bound is applied to every session
   instead of to a logger: 5 sessions with clean laps are over 0.5° and
   are left out (65 laps), two of 6520348's six KK-F sessions among them.

Gate B1 on the dataset through 2026-11-14 (3,631 laps, 313 sessions;
2,750 clean laps in 282 sessions; 1,296 build laps in 218 sessions on 73
days):

| Check | Threshold | Measured | |
|---|---|---|---|
| Doppler course closes | closure ≤ 0.002 turns on ≥ 99 % of build laps | 100 %; P10–P90 ±0.0002; one clean lap outside (0.0023, not a build lap) | pass |
| Lag removed | scatter ≤ 0.5° after the shift | 0.18–0.33° median per logger (as logged 0.65–1.8°), 0.50° on logger 6520348's KK-F sessions (2.8°); 5 of 282 sessions over, left out | reported |
| Path length = ∫V dt | within 0.3 % on every build lap | 96.7 % of laps; median +0.06 %, range −0.27 to +0.44 % | **fail**, cause below |
| Lap length = session row | within 0.5 % | 284 of 284 sessions | pass |
| Lap length spread | P90 − P10 ≤ 1 % per layout | 0.45–0.97 % | pass |
| Layout is the declared one | median within 2 % of the official length | Fuji −1.11 %, Motegi −0.99 %, Sodegaura −1.95 %, Suzuka −0.77 %, Tsukuba +1.75 % | pass |
| No unflagged clean lap over 3 g | position and Doppler | 0 laps | pass |
| Flagged laps | ≤ 2 % overall, cells over 5 % explained | 11 of 2,750 (0.40 %) | pass |
| Laps map to crossing intervals | every full lap, or a cause | 32 without one: 30 hold a stop or the start of the recording, 1 fragment, 1 session that never crosses the gate twice | pass |
| Gyro scale | P10–P90 ≤ 1 % per car and logger | 0.28–0.87 % | pass |

Lag by logger (median, ms): SW4 8401203 65–79 (KK-F, KK-F-V2, KK-SII);
6701209 78 and 6519522 144 (the Inferno 86's two loggers); 15074 101 (EK9)
and 114 (RX8); 6603435 105 (SFJ); 6521016 126 (KK-F); 6520348 177 (KK-F,
167–223) and 438 (718 Cayman, one session, scatter 1.8°).

Gyro scale (median): EK9 0.996, RX8 0.998, Inferno 86 1.025 on logger
6519522 (to 2025-05-24) and 0.873 on 6701209 (from 2025-06-09), SFJ 0.857,
KK-F 1.025–1.031 by logger, KK-F-V2 1.024, KK-SII 1.028.

**The failed check.** At Motegi the XY path of a build lap is 0.20–0.28 %
longer than ∫V dt (39 of the 43 laps outside the bound are there). 6.0 m
of the 11.9 m sit in two 250 m stretches, at s ≈ 1,750–2,000 m and
4,000–4,250 m: the two underpasses, where the position solution wanders
the same way every lap. The rest is spread over the lap, under 1 m in any
250 m, as it is elsewhere: the layout medians are −0.04 to +0.03 % at Fuji,
0.02–0.04 % at Suzuka, 0.05–0.09 % at Sodegaura and −0.01 to +0.15 % at
Tsukuba, where the Inferno 86 reads +0.15 % and the formula cars +0.06 %
or less. The cause of that remainder was not pinned down; noise in the
fixes lengthens a path and is the likely one. The Doppler course and speed
are not what is wrong at Motegi, so the consequence is for B2:
arc length under the underpasses must not come from the XY path, and the
profile there is checked against the inertial sensors.

**Flagged laps.** The three flagged laps at Motegi all have their spike at
s ≈ 4,140–4,175 m (the second underpass), which is why two cells of 18
laps read 5.6 %; the one at Suzuka is at the crossover (s ≈ 2,490 m). The
others are single events on the KK-SII (7 laps, up to 12 g).

**Build population** (laps / sessions / days; a cell with fewer than
three days feeds the profile and has no vote in B3):

| Layout | Cars |
|---|---|
| Fuji GP | Inferno 86 111 / 23 / 10, KK-F 57 / 6 / 2, KK-SII 25 / 4 / 1, SFJ 23 / 6 / 2 |
| Motegi | EK9 103 / 19 / 4, Inferno 86 15 / 3 / 2, SFJ 11 / 6 / 1 |
| Sodegaura | Inferno 86 355 / 54 / 16, RX8 13 / 7 / 2 |
| Suzuka | KK-SII 106 / 25 / 7, KK-F 41 / 7 / 2, SFJ 17 / 6 / 2 |
| Tsukuba 2000 | KK-SII 193 / 28 / 12, KK-F 166 / 16 / 8, Inferno 86 37 / 5 / 4, KK-F-V2 23 / 3 / 1 |

Of the clean laps left out: 952 are slower than 1.02 × their session's
best, 426 are from sessions off the car's reference pace (the wet ones
among them), 65 from the five sessions over the receiver bound and 11 are
flagged.

#### B2. The track profile (`track.py`, gate `check-track-truth`, 2026-11-15)

`build_track_model(root, layout_key, cars=None, exclude_session_ids=())`
builds a layout's track from its build laps; `TrackModel` holds
`kappa_heat` (1/m, positive in a right turn), the line, altitude, grade,
the per-bin lap count and κ standard error (bootstrap over days), the
corner table and the fills, on a closed grid of about 2 m, and round-trips
through JSON (`data/lapsim/tracks/<layout>.json`, committed since
2026-11-24; before that `workspace/lapsim/tracks/`).

- **Projection within a window.** A lap's sample is matched only to the
  piece of the reference line within 80 m of where the lap's own distance
  puts it, so the two branches of Suzuka's crossover are never confused.
- **The seam is open in the first pass.** The first reference is the
  fastest build lap, and a lap ends where it crossed the line a lap later:
  at Suzuka that was 9.7 m to the side of where it started (the straight
  is wide; 27 % of session-best laps there finish over a metre further
  right than they began, and 25 % further left). Smoothing the laps'
  offsets across that step left a kink at the start line that did not
  converge (closing gap 9.5 m, a further re-projection moved V²κ by
  0.03 g). The first re-referencing therefore treats the reference as an
  open line and joins its two ends on the median line; a second one is
  done on the closed line. A lap's bins before its own first sample are
  read from its other end, one lap on.
- **κ from the per-bin median of the Doppler course**, as the difference
  over 10 m; the line is the integral of the course and nothing is
  differentiated from it.
- **The accelerometer trails GPS position**, by 122–126 ms on the SW4
  (KK-SII, KK-F, KK-F-V2; 117–146 ms across sessions), 49 and 61 ms on the
  Inferno 86's two loggers and 30–34 ms on the EK9, RX8 and SFJ (medians
  over sessions;
  `track.accel_delay`, the shift that best correlates the accelerometer
  with the Doppler lateral g on the position clock). Uncorrected, the
  track and the accelerometer disagree by ±0.3 g through Suzuka's S-curves
  and chicane with the sign of the transition, and the comparison fails
  (0.072 g rms at Tsukuba, 0.081 at Suzuka). The inertial profile is put
  on the position clock before it is compared. The extractor's audit
  slopes were fitted without this shift; #255 and #205's frame record
  should take it up.
- **Fixed-place errors.** Where the 20 m mean of `V²(κ − κ_inertial)`
  exceeds 0.25 g the stretch is replaced from the accelerometer profile,
  scaled so the course change across it stays the Doppler one
  (`fill_from_inertial`). **No stretch reaches the threshold on the
  current dataset**: the largest is −0.21 g at Motegi's second underpass
  (s ≈ 4,150 m; −0.43 g on the Inferno 86's 15 laps alone), then −0.17 g in
  Suzuka's S-curves. The mechanism is in place and tested; the profiles
  carry no fill.

Three statistics are computed differently from the issue's wording, each
because the first form measured something else:

1. *Per car against the inertial profile*: a car's own Doppler profile
   (the median course of its laps) against its own accelerometer, gated
   where it has 30 IMU-sourced laps. Against the pooled track the
   difference also holds the car's line, which is B3's subject, and a
   median of 11–23 laps of accelerometer is too rough to gate on (0.05 to
   0.10 g rms on those cells).
2. *Apexes not flattened*: the peak of the course differenced over one
   bin is no reference, since the largest of a corner's noisy bins reads
   2–5 % high (the first form gave a mean of 0.023–0.042 against the 0.03
   bound, every corner negative). The loss is estimated from the profile
   over twice the window: a boxcar lowers a peak by an amount that goes
   with its width squared, so a third of the difference between the two
   profiles is what the 10 m window costs.
3. *Apex against the accelerometer*: the accelerometer profile is brought
   to the track's 10 m bandwidth before its peak is taken.

Gate B2 on the five layouts (Tsukuba 419 laps / 24 days, Fuji 216 / 15,
Suzuka 164 / 11, Motegi 129 / 6, Sodegaura 368 / 17):

| Check | Threshold | Measured | |
|---|---|---|---|
| Profile closes | ∮κ ds within 0.002 turns of n; line within 5 m | closure ≤ 0.0001 turns; line gap 1.0–2.5 m; n = +1, Suzuka 0 | pass |
| Re-referencing converged | one more moves V²κ ≤ 0.01 g rms | 0.0003–0.0013 g | pass |
| Against the inertial profile | rms ≤ 0.06 g; no 20 m over 0.25 g | per layout 0.027 (Sodegaura), 0.039, 0.042, 0.050, 0.055 g (Motegi); per car 0.027–0.059 g on 7 cells | pass |
| Apexes not flattened | loss to the window: mean ≤ 0.03, any corner ≤ 0.06 | mean 0.006–0.008, worst corner 0.029 | pass |
| Apex κ against the accelerometer | mean log ratio within ± 0.05 | −0.002 to −0.011; Motegi −0.025 | pass |
| Grade between cars | ≤ 1 % rms | 0.18–0.63 % | pass |
| Altitude against the GSI elevation model | grade over 50 m within 1 % rms; relief within 3 m | grade 0.17–0.36 % rms; relief within 1.0 m | pass |
| Against the operators' drawings | left/right order exact; turning between long straights within 5°; no apex under 0.95 × the inside edge | order matches at 4 of 4; turning within 3.3° (Sodegaura); tightest apex 0.98 × | pass |
| Peak retention | reported | profile apex over the laps' own: median 0.93–0.96 per layout, lowest corner 0.88 | |
| Share of lateral heat under 0.004 /m | reported | Tsukuba 0.093, Sodegaura 0.105, Motegi 0.109, Fuji 0.139, Suzuka 0.229; day-split floor 0.0004–0.010 | |
| Right against left at the laps' own speed | reported against 0.02 | 15 of 16 car × layout rows within 0.013; KK-F-V2 at Tsukuba 0.026 (23 laps, one day) | |

Lengths of the median line: Tsukuba 2,078.1 m, Fuji 4,506.1, Suzuka
5,755.3, Motegi 4,751.4, Sodegaura 2,387.1. Corners found (stretches over
0.002 /m turning one way by 10° or more with an apex over 0.004 /m): 8,
10, 16, 11, 8.

**Altitude against the ground survey** (`dem.py`). The GSI 5 m laser
model (`dem5a` text tiles, zoom 15) is sampled along each track's own
line, smoothed and differenced exactly as the logged altitude is. Relief,
logged against surveyed: Tsukuba 5.2 / 5.0 m, Sodegaura 13.9 / 13.9,
Motegi 31.1 / 30.4 (the operator publishes 30.4), Fuji 36.7 / 37.1, Suzuka
39.4 / 40.3. Steepest grades at Fuji: +8.1 and −9.5 % logged, +8.3 and
−9.4 % surveyed. The model is bare earth, so a bridge deck reads the
ground under it: a stretch where the survey leaves the logged altitude by
more than 2 m against its own 300 m median is masked and listed. That
finds four at Suzuka (s ≈ 320–390 m, 1,950–2,040 m, 4,840–4,940 m, the
crossover, at −6.2 m, and 5,270–5,350 m) and none elsewhere. The 16 tiles
are cached in `workspace/lapsim/gsi/` and nothing derived from them is
committed; `--dem-offline` and `--no-dem` run the gate without the
network.

**Against the operators' drawings** (`drawings.py`, the transcription
with its sources). Sodegaura's planning drawing gives every element's
radius, arc length and station; Tsukuba's table gives radius, width and
cant per section; Suzuka's guide a radius strip (the guide says shape and
dimensions may differ from reality); Motegi's plan radii as labels. Fuji
publishes none for the current surface. The transcription corrected one
figure the design review carried: Sodegaura's tenth element is printed
R=55 m, not 65; with 55 the drawing's arcs close to 360.0°.

What a driven line can be held to, and what it reads:

- *The order of lefts and rights* (elements turning the same way are one
  group): the corner table alternates exactly as the drawing does at all
  four circuits (5 groups at Sodegaura, 7 at Tsukuba, 13 at Suzuka, 9 at
  Motegi, where the count needs the unlabelled left after the second
  underpass).
- *Net turning between long straights*, at Sodegaura, where arc lengths
  are published. On a long straight the car is aligned with the track
  whatever line it took, so the heading change from one to the next is
  the drawing's:

  | Between stations | Drawing | Profile | Difference |
  |---|---|---|---|
  | 305–588 m (the 95R) | +57.1° | +54.0° | −3.1° |
  | 588–1,380 m (70R, 25R, kink, the three lefts) | +30.6° | +27.9° | −2.7° |
  | 1,380–2,394 m (100R to the last corner) | +179.2° | +181.7° | +2.5° |
  | 2,394 m through the line to 305 m (the 40R) | +93.1° | +96.4° | +3.3° |

  All within the 5° bound. The size is what geometry gives: a car that
  crosses a 150 m straight from one side to the other is 3° off the
  track's axis. A straight has to be 100 m long to count: cut at the 92 m
  and 94 m straights either side of the 100R kink, the two pieces differ
  from the drawing by +9.5° and −9.6°, which cancel, because the car is
  still unwinding from the hairpin there. It is not a corner-by-corner
  test for the same reason.
- *No apex tighter than the inside edge*, R − w/2, with 5 % allowed. The
  tightest case is Tsukuba's first corner: 29.2 m driven against 29.9 m
  for the 35R at its printed 10.1 m width (0.98). Everywhere else at
  Tsukuba, Sodegaura and Motegi the driven apex is 1.05 to 2.4 times the
  inside-edge radius: the driven line is straighter than the centre line,
  most of all through the fast bends (Tsukuba's Dunlop, printed 35R, is
  driven at 62 m). Suzuka's guide is not gated, as it disclaims its
  dimensions; its last corner reads 95 m driven against 107 m (0.89).

Gate result: pass (order exact at 4 of 4; turning within 3.3°; no gated
apex under 0.98).

Motegi's profile per year was asked for (resurfaced for 2026): every
Motegi build lap is from 2025, so there is nothing to compare.

**Asked of the track by #206's review, now carried by it:**

- *Vertical curvature* (`TrackModel.kappa_v`, d(grade)/ds over 50 m:
  positive in a compression, negative over a crest; the normal load goes
  with 1 + V²κ_v/g). On the median speed trace it runs from −0.03 to
  −0.10 g over the crests and up to +0.08 to +0.15 g in the compressions
  (largest: Suzuka +0.15, Tsukuba +0.15, Motegi +0.12). Against the same
  quantity from the elevation survey it differs by 0.005–0.010 g rms on
  four layouts; at Suzuka by 0.068 g, around its masked bridges.
- *Pit exit* (`pit.py`, in each track's `meta.pit`). An out-lap from the
  pit is laid on the reference line by counting distance back from the
  line it ends on; the join is where it comes inside the band the build
  laps occupy (their 1st–99th percentile of offset, plus a metre) and
  stays there for 150 m. Medians over the latest out-laps:

  | Layout | Joins at s (P10–P90) | Rolling by then (P10–P90) | Out-laps |
  |---|---|---|---|
  | Tsukuba | 215 m (200–273) | 16 s (7–172) | 60 |
  | Motegi | 240 m (212–646) | 29 s (10–57) | 29 |
  | Sodegaura | 328 m (136–635) | 40 s (17–90) | 60 |
  | Suzuka | 641 m (494–802) | 26 s (9–76) | 34 |
  | Fuji | 752 m (369–872) | 44 s (17–93) | 48 |

  #206's review, with its own method, had 200–225 m, 234 m, about 340 m,
  590 m and 730–760 m. The medians agree except Suzuka (641 against 590);
  the wide P10–P90 ranges are out-laps that rejoin the driven line late or
  early, and rolling times that include waiting at the pit exit.

#### B2a. A wet and a dry version of each track (2026-11-15)

Decided by the owner: the road does not change in the wet, but wet days
are a separate basis, with their own meaning of fast; where a layout has
the wet laps it keeps a wet track beside the dry one, every car pooled as
in the dry one.

**Wet basis** (`geometry.classify_laps`, columns `in_build_wet`,
`wet_build_reason`, `wet_reference_pace_s`): a session is wet when the
wetness model labels it wet *and* its best is off the car's dry pace (over
1.05 × the dry reference, and within 1.3 × of it: slower than that is not
a wet session driven for time). Among those the same two pace rules apply
on their own scale: a session best within 1.05 × the car's third-fastest
wet session best at the layout, and a lap within 1.02 × its session's
best. The two bases never share a lap. `build_track_model(..., basis="wet")`
builds from it; the track's `meta.basis` says which it is, and the wet one
is saved as `<layout>__wet.json`.

**Why it is worth a second track.** A car's fast wet laps against its own
dry build laps, on the dry build's reference line:

| Layout | Car | Wet laps (sessions / days) | Wet / dry reference lap | Line against the dry line, rms / max | Apex κ, mean | Corners tighter / wider by 2 % | Heat at equal speed |
|---|---|---|---|---|---|---|---|
| Tsukuba | Inferno 86 | 41 (8 / 4) | 76.0 / 69.8 s | 1.5 / 3.3 m | +4 % | 4 / 2 of 8 | 0.019 |
| Tsukuba | KK-F | 33 (8 / 4) | 65.0 / 59.5 s | 1.0 / 2.5 m | +4 % | 5 / 2 of 8 | 0.015 |
| Tsukuba | KK-SII | 17 (4 / 1) | 67.3 / 60.4 s | 2.7 / 4.9 m | +9 % | 7 / 0 of 8 | 0.007 |
| Fuji | KK-SII | 14 (5 / 1) | 133.7 / 117.6 s | 1.7 / 4.3 m | +7 % | 7 / 3 of 11 | 0.011 |
| Sodegaura | Inferno 86 | 23 (5 / 2) | 88.9 / 77.1 s | 1.9 / 4.1 m | +9 % | 8 / 0 of 8 | 0.009 |

The wet line lies 1–3 m from the dry one and its apexes are 4–9 % tighter
on average; at equal speed that is 0.007–0.019 in per-tyre heat, the size
of the difference between cars at Tsukuba and one to three times the
day-to-day floor. Fast wet laps therefore do not belong in the dry track
(the pace rule keeps them out) and are a track of their own.

**What exists today.** A wet track is built from 30 wet-basis laps on 3
days:

| Layout | Wet laps | Sessions | Days | Cars | Wet track |
|---|---|---|---|---|---|
| Tsukuba | 91 | 20 | 9 | Inferno 86, KK-F, KK-SII | built |
| Sodegaura | 23 | 5 | 2 | Inferno 86 | not yet |
| Fuji | 20 | 6 | 2 | KK-F, KK-SII | not yet |
| Motegi | 1 | 1 | 1 | EK9 | not yet |
| Suzuka | 0 | | | | not yet |

Tsukuba's wet track closes like a dry one (0.0001 turns, line gap 1.0 m)
and has the same length (2,078 m). Pooled over the three cars it lies
0.7 m rms (1.5 m at most) from the dry track with apexes 2 % tighter on
average (−4 to +9 % by corner), 0.020 in per-tyre heat at the dry speed
trace. The pooled figures are smaller than the per-car ones above because
the cars' wet lines differ from each other as well. The gate checks that a
wet track closes; the comparison with the dry track is reported.

Known limit of the dry reference pace: it is the third-fastest session
whatever its label, so a car with only two dry sessions at a layout and
several wet ones would take a wet session as its dry reference. No cell
is in that state today (the reference is within 2.4 % of the fastest
session everywhere except the RX8 at Sodegaura, 7 %, and the Inferno 86
at Tsukuba, 5.3 %).

#### B2b. The road's bank (`bank.py`, 2026-11-17)

Built on the owner's decisions of 2026-11-17, after the review's third
round (C5, item 11): the bank goes into the track model, into the lap
wherever a track has one, and into the simulated heat. Issue #299, which
asked for a list of corners where every car's lateral runs high, now
asks for this instead.

**How it is read** (`bank.road_bank`). At each 2 m node: the line's
direction over ± 6 m, the ground's height at 1 m steps out to 5 m either
side of it (the GSI 5 m laser mesh, bilinear), and a least-squares line
through the eleven heights. Its slope is the cross-slope, positive where
the left side of the direction of travel is higher. A node is flagged and
filled along the track from its neighbours where the model has no data,
where gate B2's altitude check finds a bridge (`bank.notch_zones`, the
same stretches as above), or where the section is not a plane (residual
over 0.15 m rms or slope over 0.12: a wall, an earth bank). Then a mean
over ± 10 m along the track; the angle is the slope's arctangent. Gate B2
attaches it to each track it builds (`TrackModel.bank`, `bank_flag`,
`meta["bank"]`; the wet track gets its own on its own line) and reports
it. With `--no-dem`, or where the tiles do not reach, a track has no
bank and is driven flat. The bank is derived from GSI's tiles. Until
2026-11-24 it stayed in `workspace/` with them and was not committed;
since the owner's ruling of that day (#208) the tracks are committed with
their bank in `data/lapsim/tracks/`, with the attribution GSI's content
terms ask for in `data/lapsim/README.md` (出典：国土地理院ウェブサイト;
「地理院タイル（標高タイル（基盤地図情報数値標高モデル））を加工して作成」). The
tiles themselves stay in `workspace/lapsim/gsi/`.

| layout | nodes filled (bridge / not a plane) | scatter on the straights (rad) | mean into the corners (rad) | sd over the corners | largest |
|---|---|---|---|---|---|
| Tsukuba | 0 | 0.0033 | 0.035 | 0.024 | 0.085 |
| Tsukuba, wet line | 0 | 0.0031 | 0.036 | 0.024 | 0.090 |
| Fuji | 0 | 0.0038 | 0.035 | 0.021 | 0.079 |
| Motegi | 6 (0 / 6) | 0.0101 | 0.031 | 0.016 | 0.104 |
| Sodegaura | 0 | 0.0026 | 0.011 | 0.008 | 0.045 |
| Suzuka | 202 (175 / 27) | 0.0188 | 0.037 | 0.020 | 0.098 |

The scatter on the straights (about each straight's own mean) is an
upper bound of the noise: a straight's cross-slope also changes along
it. The two halves of a section differ in slope by 0.016–0.049 at a
tenth of the nodes, which is the crown and the kerbs a 5 m mesh cannot
give. These are the reviewer's numbers to the fourth decimal: the built
version is the prototype's method.

**Against the accelerometers** (gate B2, reported). Per car on its own
laps, the ground-frame lateral less the accelerometer's is the bank the
sensor sees. The accelerometer is already divided by its session's audit
slope, which takes out whatever goes with the lateral, the bank's habit
of leaning into the turns with it; so the fit (`slope × bank + per_g ×
lateral + constant`, `gates.bank_against_sensor`) reads the bank only
where it does not follow the lateral. Cars with 30 IMU laps or more:

| layout | car | IMU laps | slope | partial r |
|---|---|---|---|---|
| Fuji | Inferno 86 | 111 | 0.94 | 0.63 |
| Tsukuba | Inferno 86 | 37 | 1.00 | 0.39 |
| Tsukuba | KK-F | 93 | 1.09 | 0.52 |
| Tsukuba | KK-SII | 92 | 1.45 | 0.44 |
| Suzuka | KK-SII | 106 | 0.90 | 0.46 |
| Sodegaura | Inferno 86 | 336 | 0.71 | 0.29 |
| Motegi | EK9 | 30 | 0.60 | 0.21 |

Where the bank is large and the laps are many the slope is 0.9–1.1; at
Sodegaura the bank is too small to read (sd 0.008 rad) and the KK-SII at
Tsukuba reads high. The review's own check, on the laps' own course rate
and the raw accelerometer, gave 0.92–0.99 with r = 0.62–0.80 at Fuji and
Tsukuba; this one is cruder (the track's curvature for every lap, and
the audit slope's share gone) and agrees with it.

**In the lap** (`solver._road`; on wherever a track has a bank, by the
owner's decision of 2026-11-17). With b the lean into the turn (the bank
with the curvature's sign):

- the tyres hold `V²|κ|·cos b − g·sin b` sideways;
- the load is `g·cos θ·cos b + V²(κ_v + |κ|·sin b)`;
- so the cornering limit is `V²(|κ|·cos b − μ(κ_v + |κ|·sin b)) =
  g(μ·cos θ·cos b + sin b)`, and on a straight the tyres hold the car
  against the cross-slope.

The bank is folded into the track's arrays before either form of the
solver sees them (`|κ|·cos b`, `κ_v + |κ|·sin b`, `cos θ·cos b`, and one
new array `g·sin b`), so the passes changed in three statements and a
flat road is the old arithmetic: before it was switched on, with every
road flat, the node speeds, u and lap times of every track and car in
both forms were the same to the last bit as before the change (72
cases), and the C2 and C3 reports and the car JSONs came out unchanged.
Sines and cosines are taken once in numpy, outside the passes, so the
two forms stay identical with a bank too (tested to the last bit).
`Profile.from_track(track, bank=False)` and `Drive.bank = False` take a
road as flat; that is a comparison the gate reports, not a setting.

**In the heat** (`bank.tyre_lateral_g`, `holdout.tyre_heat(..., bank)`).
The simulated heat took the ground-frame lateral `V²κ` whatever the
road. On a banked road it now takes what the tyres hold, `V²κ·cos b / g
− sin b` with b the angle by which the left side is higher: what an
accelerometer on the road's plane reads, which is what production's
chain is given by a car with a sound one. `SimLap.lat_tyre_g` is the
same formula at a step's two nodes; `lat_g` stays the ground-frame
lateral.

**The first design left the term off**, by the reviewer's rule set
before its run (C5, item 11) and because S1 did not improve. The owner
asked why, and the answer did not hold: the bank is measured and read
right, a term that is left out is paid for by whatever is fitted next
(the driver model would have taken Tsukuba's and Fuji's bank for the
driver's use of the grip), S1 is the score the review showed does not
carry the tyre model's need, and on the measure that does the bank
wins. So it is on and there is no default to set.

**Gate C3 with it on** (verdict as before: Inferno 86 inside on 29 of
31 days and 4 of 4 layouts, KK-SII 19 of 19, KK-F 13 of 13; the low-power
control is told on 13 of the Inferno 86's days where it was 15). By held-out
layout, the car and `mu_y` set without the layout each way, heat as the
largest over the four tyres of the absolute median log ratio of
simulated to logged heat (tyres' frame with the bank, ground frame
without):

| car | layout | S1 with / without (m/s) | u with / without | heat, own lap time | heat, pace carried |
|---|---|---|---|---|---|
| Inferno 86 | Fuji | 1.20 / 1.33 | 0.970 / 1.032 | 1.1 / 1.2 % | 1.5 / 6.9 % |
| Inferno 86 | Motegi | 1.38 / 1.48 | 0.993 / 1.031 | 2.3 / 2.6 % | 4.9 / 8.0 % |
| Inferno 86 | Sodegaura | 0.98 / 0.95 | 0.949 / 0.896 | 0.8 / 0.5 % | 1.2 / 6.5 % |
| Inferno 86 | Tsukuba | 1.03 / 1.11 | 0.884 / 0.909 | 1.8 / 0.9 % | 9.5 / 7.3 % |
| KK-F | Fuji | 1.27 / 1.28 | 0.930 / 0.913 | 3.1 / 3.1 % | 1.1 / 2.5 % |
| KK-F | Suzuka | 1.41 / 1.41 | 0.971 / 0.950 | 2.0 / 2.4 % | 3.6 / 2.5 % |
| KK-F | Tsukuba | 1.16 / 1.11 | 0.968 / 1.001 | 2.7 / 1.5 % | 4.2 / 6.4 % |
| KK-SII | Fuji | 1.43 / 1.50 | 0.907 / 0.891 | 4.2 / 4.2 % | 2.6 / 3.8 % |
| KK-SII | Suzuka | 1.46 / 1.45 | 1.007 / 0.987 | 1.9 / 2.5 % | 6.4 / 5.4 % |
| KK-SII | Tsukuba | 1.23 / 1.19 | 0.941 / 0.970 | 3.6 / 2.5 % | 3.9 / 3.6 % |

- **Heat at a layout the car has not run, the pace carried from its
  other layouts: 3.7 % with the bank against 5.9 % without** (median of
  the ten folds; lower in six). It is the review's number, now the
  gate's. At the lap's own lap time it is 2.1 % against 2.4 %, with
  Tsukuba worse on each car there too (0.9 → 1.8, 1.5 → 2.7 and
  2.5 → 3.6 %).
- **The pace level carries better to seven of ten layouts, and worse to
  Tsukuba for all three cars.** `|u − 1|` falls at every layout but
  Tsukuba (mean over the seven 0.061 → 0.041) and rises at Tsukuba on
  each car (0.091 → 0.116, 0.001 → 0.032, 0.030 → 0.059). With the bank,
  a car set elsewhere has grip to spare at Tsukuba: the cars do not use
  Tsukuba's bank as the model gives it (the hairpins overshoot and the
  fast corners fall behind), which is the car or the driver and not the
  road.
- **S1 is lower in five of ten cells and the apex speed nearer in
  three.** The apexes come out slower with the bank (S3 is lower in nine
  of ten cells; Suzuka −0.45 → −0.73 and −0.46 → −0.69): `mu_y` by the
  convention falls when the road carries part of the cornering, and the
  corners without much bank lose it. Over the day folds the Inferno 86's
  median S1 goes from 1.11 to 1.10 m/s, the KK-F's from 1.23 to 1.28 and
  the KK-SII's from 1.29 to 1.33.

**What it moves in the car** (gate C2). `mu_y` by its convention is
1.128 / 1.270 / 1.334 (Inferno 86 / KK-SII / KK-F) where it was 1.190 /
1.433 / 1.464, which is 16 / 24 / 21 % under the apex table's level
where it was 11–14 %. The apex table is in the ground frame, where the
road's help is counted as the car's grip: on simulated laps of a known
car it now returns a level 8.5 / 13.7 / 15.0 % over the planted `mu_y`
(2.3–2.8 % on flat roads). The spread of the reference u between
layouts is 14.2 / 7.4 / 3.4 % where it was 8.3 / 7.3 / 6.2 %; the
Inferno 86's is Tsukuba. `mu_b` is a measurement and does not move, so
the solver's `mu_b / mu_y` is 0.85 / 0.95 / 0.93 where it was 0.80 /
0.85 / 0.85, and the logs hold 0.64–0.65 (C5, item 2, in the ground
frame): the simulated car brakes harder against its cornering than
before.

**Not done.**
- The car's sample tables are still in the ground frame: the apex
  table's level, the lateral the braking shape is read against and the
  lateral of the cornering drag all count the road's help as the car's.
  They go to the tyres' frame with the tables' next version, which the
  driver model needs anyway (#312).
- Braking against cornering (above) and Tsukuba are the driver
  model's.
- The Fuji survey (#251) is the truth the 0.94 is to be held against.
- A bank from every sound-IMU session (six sessions agree to
  0.014–0.020 rad per node) would cover the crossover at Suzuka and
  tracks outside Japan.

#### B3. One track for every car (`holdout.py`, gate `check-track-holdout`, 2026-11-15)

The statistic is profile against profile: the curvature from one group of
laps against the curvature from another, both driven with the held-out
group's median speed trace through production's `share_heat_rate` (lateral
g from V²κ, longitudinal g from the speed trace, the held-out car's fitted
force shares), giving the log ratio of lap-mean heat per tyre, Δq. A gate
reads the largest of the four in magnitude. Cars without fitted shares are
read through a stand-in (SFJ through the FJ's, EK9 and RX8 through the
Inferno 86's): the statistic needs a lens, not that car's heat model.

The ladder puts a difference between cars beside the differences that
have nothing to do with the car: (1) random halves of a car's laps within
each day, (2) random halves of its days, (3) its early days against its
late days, (4) KK-F against KK-SII, (5) each class against every other car
at the layout. A comparison between cars votes when the held-out group has
three days and the donor laps three days and thirty laps.

Everything is computed on one projection (subsets of the build laps on
the build's reference line). The voting comparisons were also rebuilt with
each side on its own reference line: the statistic moved by at most 0.003.

Largest per-tyre |Δq|, by layout:

| Layout | Laps in a day (P90) | Days (P90) | Early / late days | KK-F / KK-SII | Class against the rest | Verdict |
|---|---|---|---|---|---|---|
| Tsukuba | 0.005 | 0.008 | 0.004–0.007 | **0.016–0.017** | **0.008–0.010** (FJ, Inferno 86) | one track; bias carried to E1 |
| Fuji | 0.005 | 0.006 | 0.003 | 0.011–0.012 | **0.006–0.007** (FJ, Inferno 86); 0.007 (SFJ) | one track; bias carried to E1 |
| Suzuka | 0.005 | 0.003 | 0.003 | 0.004–0.005 | 0.004–0.005 (FJ, SFJ) | no vote |
| Sodegaura | 0.011 | 0.006 | 0.002 | | 0.008–0.013 (Inferno 86, RX8) | no vote |
| Motegi | 0.009 | 0.012 | 0.009 | | 0.023–0.028 (EK9, Inferno 86, SFJ) | no vote |

Bold figures vote. **Gate: pass.** No layout is above the 0.05 stop
bound; the two layouts that can vote are under the 0.02 bound for one
shared track. Read plainly:

- **The day-to-day floor is 0.003–0.012 by layout**, 0.006 and 0.008 at
  the two that vote. The same car on different days differs by that much
  in per-tyre heat through the track profile alone.
- **At Fuji the cars agree at the floor** (0.006 and 0.007 against 0.006),
  on the two voting comparisons there (the FJ class and the Inferno 86,
  each against every other car).
- **At Tsukuba the two formula cars differ from each other by twice the
  floor (0.017 against 0.008), and from the Inferno 86 by about the floor
  (0.008–0.010).** A same-class difference larger than the other-class one
  is not a car effect in any simple sense; the review expected the two to
  be the same size and inseparable from era. The Inferno 86 has 37 laps on
  four days there. The bias, up to 0.017, is carried into E1's 0.03 budget.
  With each side on its own reference line the KK-F / KK-SII figure is
  0.013–0.014.
- **Three layouts cannot vote.** Suzuka's non-voting comparisons sit at
  its floor. Sodegaura's are at twice the floor on an RX8 with 13 laps.
  Motegi's read 0.023–0.028, above the 0.02 bound, with donors of 26 laps
  or held-out groups of 11–15: #207's review measured that a track from
  ten laps on two days shifts per-tyre heat by 2.6 %, so this is the size
  thin data gives and it is not evidence about Motegi either way.

Beside the ladder (not gated):

- *The line moves with pace.* A car's laps at 1.04–1.12 × its session best
  against its own build laps: 0.003–0.018 (largest: KK-F at Tsukuba 0.018,
  then 0.007–0.009 on four cells), with less lateral heat in both right
  and left turns in six of the seven cells (−0.005 to −0.017). This goes
  to D2.
- *Sessions labelled wet.* The build does not use the label, so this is
  measured inside the build: a car's build laps from sessions the wetness
  model labels wet against its other build laps. Three cells have twelve
  laps on each side (KK-SII at Tsukuba, EK9 at Motegi, Inferno 86 at
  Sodegaura): 0.006–0.010, at or near the day-to-day floor.
- The donor tracks of the voting comparisons and the held-out groups'
  speed traces are saved in `workspace/lapsim/tracks/folds/` for C3 and D1.
- *The per-lap metric of the first revision*, both sides smoothed over
  0.5 s: the log ratio of lap-mean |V²κ|, a lap against a profile at the
  lap's own speed. Against the profile of the car's own other days it
  reads +0.005 to +0.018 on twelve cells (a lap is noisier than a
  profile; the RX8's 13 laps read −0.013). Against every other car's
  profile, less that floor: KK-F at Tsukuba +0.018 and KK-SII there
  −0.015, EK9 at Motegi −0.018, Inferno 86 at Motegi +0.037 on 15 laps,
  and within ±0.009 on the other nine. It tells the same story as the
  ladder and shows no bias the profiles share.

#### C0. The session frame record (`frame.py`, `heat.py`, gate `check-frame`, 2026-11-15)

Segment C (#205) is the car model. Its first piece needs no car: it
measures what a session's accelerometer reads of a motion known in the
ground frame, which is where the track and any simulated lap live.

- **Ground frame**, on the position clock: lateral `V·Δψ/Δt` from the
  Doppler course, longitudinal `ΔV/Δt`, both central differences over
  0.5 s (`frame.ground_series`).
- **The record**, per session, from its clean laps: the accelerometer's
  lateral slope and zero against the ground-frame lateral (a line fitted
  with 4-sd clipping), and its longitudinal zero, the median over the laps
  of lap-mean accelerometer minus lap-mean `dV/dt`. The accelerometer is
  first averaged over the same 0.5 s and moved onto the position clock by
  its measured delay. A longitudinal slope is measured and not applied.
  Nothing is fitted per car.
- **A GPS-sourced lap has no record.** Its logged series low-passed over
  0.5 s is the ground frame: 0.001–0.003 in heat on 61 cells
  (`gps_channel`), 0.010 on the two `gps_position` cells.
- **Heat is production's chain** (`heat.bin_1s`, `heat.lap_heat`): the
  share model on 1 s means of the four positive parts and of speed, the
  seconds counted from the lap's start so that two series of one lap are
  binned alike. The issue's text had `pace_sim.corner_heat_means`; #206's
  finding 11 replaced it. The same two functions serve D1.
- **The record is a measurement, not something applied downstream.** After
  the reviews of D and E a simulated lap is compared with an observed one
  in the ground frame on both sides. What C0 leaves behind is the size of
  the sensor frame per session, for #255.

Gate C0 on the dataset through 2026-11-14: 282 sessions with clean laps,
223 of them with a fitted record (the other 59 have no accelerometer
laps); 2,750 clean laps, 9 of them left out because they change source.
Scored per (session, source) with three or more laps: 249 cells, 151 of
them in the 147 sessions with TPMS (1,699 laps). The statistic is the log
ratio of lap-mean heat per tyre, the lap's ground-frame series through its
session's record over the series as logged, averaged over the session's
laps; a session reads its worst tyre.

| Check | Threshold | Measured | |
|---|---|---|---|
| Worst tyre per session, sessions with TPMS | P90 ≤ 0.03 per car | Inferno 86 0.028, KK-SII 0.017, KK-F 0.009 (57, 69 and 22 cells); with no record 0.085, 0.086, 0.087 | pass |
| Per-tyre mean (fl / fr / rl / rr) | within ± 0.02 per car | Inferno 86 −0.014 / −0.012 / −0.005 / 0.000; KK-SII −0.006 / −0.007 / +0.003 / +0.003; KK-F −0.001 / +0.003 / +0.002 / +0.004 | pass |
| Sessions without TPMS | reported | 98 cells, P90 0.053; accelerometer cells: SFJ 0.094, Inferno 86 0.041, EK9 0.020, RX8 0.013 | reported |
| The clocks | reported | −0.005 to +0.004 per cell and tyre | reported |
| Accelerometer noise | reported | +0.005 to +0.014 by tyre on every logger but the SFJ's, +0.036 to +0.056 there | reported |

Four cells with TPMS are over 0.03, none over 0.036. The KK-F-V2's three
sessions read 0.006–0.008 and are listed (under ten cells).

What the record's parameters are:

- **Lateral slope, by layout** (median over sessions): Inferno 86 Sodegaura
  1.013, Motegi 0.994, Fuji 0.990, Tsukuba 0.949; KK-SII Suzuka 0.955,
  Tsukuba 0.951, Fuji 0.939; KK-F and KK-F-V2 Tsukuba 0.943 and 0.942; EK9
  Motegi 0.955; SFJ 0.967–0.979; RX8 0.997. The design review read
  1.015 / 0.991 / 0.955 for the Inferno 86: the slope belongs to the layout
  as much as to the car, which is why a per-car `sensor_scale` could not
  carry it.
- **Longitudinal zero, by day**: Inferno 86 −0.019 to +0.114 g over 37
  days from 2025-03-24 (sd 0.024 g between days, 0.003 g between the
  sessions of a day), and −0.17 g on its first two days (2025-03-08 and
  03-16: the logger sat about 10° nose-down); KK-SII −0.074 to +0.025 g
  over 20 days; KK-F +0.006 to +0.046 g over 8; EK9 0.000 and +0.090 g on
  its two days. On the Inferno 86 a zero of +0.01 g is +0.9 % of rear heat
  and −0.8 % of front heat as production computes it, so the day-to-day sd
  is about 2 % of heat on either axle from the mount alone (#255).
- **Longitudinal slope** (not applied): 0.93–1.02 by car and layout.
- **Accelerometer delay** behind GPS position: 122–126 ms on the SW4, 50
  and 64 ms on the Inferno 86's two loggers, 30–34 ms on the EK9, RX8 and
  SFJ, as B2 found.

Two things the gate turned up:

1. **The clocks cost nothing in heat.** Production bins the accelerometer
   on its own clock against a speed that is itself late; putting all three
   on the position clock moves a cell's heat by at most 0.005. The delays
   matter for a sample-by-sample comparison (B2), not for a lap mean.
2. **Where the record misses, it is vibration.** Heat from the
   accelerometer as logged is 0.5–1.4 % above heat from the same series
   low-passed over 0.5 s on seven of the eight car and logger pairs (#206's
   finding 13 measured 0.7–1.3 %), and 3.6–5.6 % on the SFJ's logger, whose
   lateral residual is 0.064 g against 0.023–0.043 g elsewhere. The record
   reads low there by the same amount on all four tyres (−0.053 to
   −0.058). A slope and a zero cannot carry content above the ground
   frame's bandwidth. The SFJ has no TPMS, so nothing production fits is
   touched; it matters if SFJ laps are ever used as logged heat.

#### C1. The car's sample tables (`samples.py`, gate `check-envelope-samples`, 2026-11-15)

What each car did on its build laps, as tables to read; C2 takes the car
model's numbers from them. Every sample is in the ground frame on the
position clock and is projected onto the layout's track, so the track's
grade comes out of every longitudinal value. The tables are cached in
`workspace/lapsim/cache/car_tables/` and rebuilt when the dataset's files
or a track change. (The issue names `car.py`; the tables are `samples.py`
and `car.py` is kept for the model.)

- **Populations.** Dry: B's build laps. Wet: B's wet build laps (sessions
  labelled wet and off the dry pace, judged against each other). The
  issue's text defined wet by the lateral level itself, under 0.90 of the
  dry median; since B the wet basis exists on pace, and the level is what
  the table measures.
- **One rule for every throttle and brake channel.** The throttle is read
  relative to the session's own closed and wide-open readings (1st and
  99th percentile while moving); full is the top 5 % of that span. That
  is 95 % on the Inferno 86's pedal, 93.6 % on the KK-SII's (it reads
  −0.6 closed and 98.5 open), 80.1° on the KK-F's throttle angle (5.04°
  closed, 84.0° open on all three loggers) and 50.8 % on the SFJ's
  channel, which tops out at 53.5 %. A brake is on above 5 % of the
  session's hardest application (99.5th percentile): 2.9–3.5 bar on the
  Inferno 86, 1.4 bar on the KK-SII, 1.2 bar on the KK-F's SW4 sessions
  and 1.8–1.9 in the ECU's counts on its two data loggers, so the 1.6 ×
  between those scales needs no handling. Channels logged on change hold
  their last value, for 2.5 s at most.
- **Full-throttle table.** Fixes where the throttle has been full for
  0.8 s and stays full for the 0.25 s the derivative still looks ahead,
  with no gear change within 0.3 s where a gear is stored (the KK-F's
  flat shifts; on the 0.5 s derivative a flat upshift is the step from
  one gear's level to the next, 0.25 to 0.21 g at Fuji, with a dip of
  about 0.01 g). Value:
  `dV/dt + g·grade`. Samples under 0.15 g of lateral are the straight
  table; the others are kept for the cost of cornering at full throttle.
- **Coast table.** Throttle closed and brakes off for 0.3 s, in a
  straight line: the lift-off deceleration. Needs both channels.
- **Braking zones.** One row per application (releases under 0.2 s
  merged, applications under 0.3 s dropped), from pressure where it is
  logged and from `dV/dt < −0.3 g` where it is not (EK9, RX8): where it
  began, the 95th percentile and median of `−(dV/dt + g·grade)`, the
  lateral g at the peak. Drag is not removed here; C2 does that with the
  speed stored beside it.
- **Apex table.** Per lap and corner of the track's corner table: speed
  at the track's apex, `V²κ` of the track there, the lap's own peak
  ground-frame lateral in the corner and the accelerometer's. A corner
  is grip-limited when the p90 of its laps' peaks is within 15 % of the
  car's highest corner at the layout.

Gate C1 on the dataset through 2026-11-14: 1,431 build laps (1,296 dry,
135 wet) in 250 sessions give 1,131,368 full-throttle samples (502,945 in
a straight line), 3,006 coast samples, 10,311 braking zones and 13,622
apex rows.

| Check | Threshold | Measured | |
|---|---|---|---|
| Every model car's cell has its tables | straight samples, braking zones, apex rows for ≥ 99 % of build laps (dry) | 10 of 10 cells; 1,106 of 1,106 laps | pass |
| Cells under three days | listed, no vote in C3 | Inferno 86 Motegi (2); KK-F Fuji (2) and Suzuka (2); KK-SII Fuji (1); KK-F-V2 Tsukuba (1); every SFJ cell; RX8 | reported |
| Channel coverage | reported | throttle and brake on every session of the Inferno 86 (98), KK-F (38), KK-F-V2 (3) and SFJ (18); KK-SII brake 66 of 66, throttle 63 (none usable on its three sessions of 2026-04-17); EK9 and RX8 neither | reported |
| Lateral level, max / min across layouts with three days | reported (review: 1.06–1.12 and ≤ 1.03) | Inferno 86 1.061; KK-SII 1.007 | reported |

**Straight tables** (median `dV/dt + g·grade` in g, by 5 m/s bin from the
speed shown; bins with 50 samples):

| Car, layout | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 |
|---|---|---|---|---|---|---|---|---|---|
| Inferno 86, Fuji | 0.295 | 0.309 | 0.200 | 0.198 | 0.141 | 0.129 | 0.091 | 0.072 | 0.050 |
| Inferno 86, Sodegaura | | 0.302 | 0.217 | 0.214 | 0.152 | 0.138 | 0.130 | | |
| Inferno 86, Motegi | | | 0.219 | 0.217 | 0.158 | 0.140 | 0.107 | | |
| Inferno 86, Tsukuba | | | 0.195 | 0.217 | 0.141 | 0.151 | | | |
| KK-SII, Suzuka | | | 0.379 | 0.272 | 0.257 | 0.194 | 0.114 | 0.070 | 0.031 |
| KK-SII, Tsukuba | | | 0.379 | 0.308 | 0.270 | 0.193 | 0.152 | | |
| KK-SII, Fuji | | | | 0.293 | 0.228 | 0.165 | 0.111 | 0.045 | 0.014 |
| KK-F, Tsukuba | | | 0.392 | 0.317 | 0.284 | 0.206 | 0.145 | | |
| KK-F, Fuji | | | 0.433 | 0.293 | 0.250 | 0.186 | 0.121 | 0.042 | 0.023 |
| KK-F, Suzuka | | | 0.418 | 0.277 | 0.265 | 0.226 | 0.120 | 0.059 | 0.025 |
| KK-F-V2, Tsukuba | | | | 0.373 | 0.333 | 0.247 | 0.175 | 0.145 | |
| SFJ, Fuji | | | | 0.323 | 0.257 | 0.195 | 0.135 | 0.066 | 0.023 |
| SFJ, Motegi | | | 0.316 | 0.312 | 0.270 | 0.197 | 0.140 | 0.092 | |
| SFJ, Suzuka | | | 0.428 | 0.354 | 0.291 | 0.230 | 0.139 | 0.086 | 0.038 |

The Inferno 86's row is the gear staircase the review described (0.20 g at
25–35 m/s, 0.14 at 35–45, 0.09 at 45–50). Only Fuji and Suzuka reach
50 m/s, so only they can say anything about drag. The KK-F-V2 is 0.03–0.06
g above the KK-F at every speed on its one day, which is the new intake.

**Lateral level** (median over the grip-limited corners of the corner's
p90; ground frame): Inferno 86 Tsukuba 1.374, Fuji 1.339, Motegi 1.315,
Sodegaura 1.296; KK-SII Suzuka 1.674, Tsukuba 1.662, Fuji 1.656; KK-F Fuji
1.720, Tsukuba 1.696, Suzuka 1.687; KK-F-V2 1.662; SFJ 1.44–1.50; EK9
1.279; RX8 1.087 on 13 laps. Wet over dry on that statistic: KK-SII 0.71
(Fuji) and 0.92 (Tsukuba, one day), KK-F 0.81 and 0.83, Inferno 86 0.90
(Sodegaura) and 0.93 (Tsukuba), EK9 0.78 on one lap. The review's 0.79 was
a median over sessions chosen by their grip; a p90 over the fast laps of
wet sessions reads the driest of them.

**Braking** (p90 over zones of the zone's p95, grade removed): Inferno 86
0.89–1.04 g by layout, KK-SII 1.25–1.32, KK-F 1.20–1.39, KK-F-V2 1.25, SFJ
1.00–1.17, EK9 0.89 (zones from `dV/dt`). 4.8–5.2 zones a lap at Tsukuba,
7–10 at Fuji, Sodegaura and Motegi, 9–14 at Suzuka. **Lift-off**
deceleration: Inferno 86 0.12 g from 30 to 60 m/s; KK-SII 0.18 / 0.21 /
0.29 g at 30 / 40 / 50 m/s and KK-F 0.22 / 0.26 g at 40 / 50. Coasting in a straight
line with the brakes off is rare (3,006 fixes in all), so these rest on
52–510 samples a bin.

#### C2. The car model from the tables (`car.py`, `car_gate.py`, gate `check-car-fit`, 2026-11-15)

**A car is a chassis with one or more powertrains, and how it is driven
is not the car** (decided by the owner, 2026-11-15: "the shift light
shouldn't make it a different car, we should have some way to adjust for
level of aggression"; "we should have some way to share the car params
but allow for power changes, to cover the KK-F-V2 and swap for the 86").

- **The chassis** is what a car's raw cars share: drag, the cost of
  cornering at full throttle, the braking peak, the lateral level, the
  wet scale. It is read from all of their laps together.
- **A powertrain** is an engine with its gearing: power at the wheels per
  mass against rpm (in bins of 250 rpm), the speed per 1,000 rpm of each
  gear, and one `p_specific` for a car whose logs have no rpm or no gear.
  `car.FAMILY` and `car.POWERTRAINS` say which raw car and date is which:
  the KK-F-V2 is a powertrain of the KK-F; the Inferno 86 has its first
  engine to 2026-05-31 and its second from 2026-06-01 (replaced between
  its sessions of 2026-04-06 and 2026-07-15, with a 4.1 final drive in
  place of 4.3). Each powertrain after a car's first carries its `scale`,
  its power over the first one's at equal rpm; one with too few samples
  for a curve of its own (the KK-F-V2's single day) is the first one's
  curve at that scale.
- **The shift point is the driver's.** `shift_rpm` is read per day from
  that day's logs (`car.shift_rpm_by_day`: the 99th percentile of
  full-throttle rpm in the gears the car shifts up out of, the median over
  those gears) and handed to the solver beside the pace knob. It reads
  about 120 rpm under the rpm the change is made at, because the samples
  next to a gear change are not in the table.
- **A gear channel is made where the logger has none** (`car.with_gears`).
  In one gear, speed over rpm is one number, so a day's values fall into
  one tight cluster per gear; a sample takes the gear of the cluster
  within 3 % of it. On the cars that log a gear the clusters are the
  logged gears exactly (KK-F: 4.68 / 5.95 / 6.83 / 7.95 / 9.29 for gears
  2 to 6); on the KK-SII 99.2 % of the full-throttle samples get a gear,
  on the SFJ 98.2 %. Such a gear is numbered among the gears seen at full
  throttle, not as the gearbox numbers them; the tyre's rolling
  circumference would turn a cluster into an overall ratio
  (`16.67 · C / k`, with `C` in metres and `k` the cluster in m/s per
  1,000 rpm) and so into a gear number.
- **The gear set is the day's setup, like the shift point**
  (`car.gear_sets_by_day`, `car.complete_sets`). The KK-SII has
  per-circuit gearing (the owner, 2026-11-15) and the logs show two sets:
  4.47 / 5.49 / 6.53 / 7.72 m/s per 1,000 rpm on ten Tsukuba days from
  2026-03-18 to 05-24, and 4.80 / 6.24 / 7.32 / 8.62 on nine days at Fuji
  and Suzuka and on its first Tsukuba day, 2026-02-18 (the owner: "first
  day at Tsukuba in the KK-SII was the Suzuka gears"). The SFJ ran three.
  A day shows only the gears it used at full throttle, so each day is
  completed from the other days that ran the same set, across the car's
  powertrains: the KK-F-V2's one day, which shows three gears, has the
  KK-F's five, while the Inferno 86's second engine, 5 % taller in every
  gear, falls into a set of its own. The day's set goes to the solver
  where the car is driven through its gears; a powertrain's own `gears`
  are the set most of its days ran.
- **A gear made from speed over rpm does not drive the car**
  (`solver.INFERRED_GEARS_DRIVE`, off). Driven through its curve and the
  day's gear set, the KK-SII's held-out speed trace was further out on 19
  of 22 folds than on its one level (S1 +0.04 m/s in the median, +0.09 at
  most) and the simulation 0.2–0.4 m/s faster over the top quarter of
  speeds on 21 of them, where it was already fast on all but one; that
  put 2026-05-24 at Tsukuba outside (1.52 against 1.44, floor 1.25). By
  the rule cornering drag entered on, the curve stays out and the KK-SII
  is driven on `p_specific` as before. The gears are not the cause: a
  flat curve through the same gears gives the one level's numbers. The
  candidate is the time lost in the shifts (D1 below), which the solver
  does not have and the level, being lower than the curve over the rev
  band, happens to make up for; not tested. Its gears still give the
  gear sets, the shift point and the drag.
- **The air is the day's.** `car.air_state` gives each session the density
  of its moist air and the SAE J1349 factor from the hourly weather
  (humidity out of the pressure, dry pressure and the square root of
  temperature). A sample's power is brought to standard air (99 kPa dry,
  25 °C, 1.157 kg/m³) before anything is pooled, drag is scaled with
  density, and the solver is handed the session's air back.
- `car_gate.Drive` turns a lap's keys into the car the solver drives: the
  powertrain the raw car had that day, the day's shift point, the
  session's air. Both gates drive every logged lap through it.

How each number is read:

- **Drag, from the same rpm in different gears** (`car.fit_drag_rpm`). At
  one rpm an engine makes one power whatever gear it is in, so within a
  day, a powertrain and an rpm bin, `a·V` falls with `V³` at the slope
  `−c`. That needs nothing about the shape of the power curve and works
  at every layout. The first estimator took the slope at the top of the
  straight table of the layouts that reach 50 m/s, with the power as one
  number over those bins: on simulated laps of the Inferno 86's first
  engine, whose power falls off above 7,000 rpm, it read 42 % high. It
  remains for a car without rpm (`car.fit_drag`) and is reported beside
  the new one. Held within ± 30 % of a published value where there is
  one (the Inferno 86: 2.4e-4 /m).
- **`k_corner_drag`**: the median over full-throttle samples above 0.5 g
  of lateral of `(a_straight − a) / lat²`, against the straight bin of
  the same layout and powertrain: the engine gives `P/V − k·lat²`. In the
  default solver since gate C3 first ran (below).
- **`mu_b`**: the p90 over dry braking zones of the zone's p95 less the
  drag at the speed of the peak; `n_brake` = 1.
- **`wet.grip_scale`**: the median over layouts of wet over dry lateral
  level.
- **Day levels** (`car.day_levels`): the day's power over its
  powertrain's curve at equal rpm, which the gear the car was in and the
  rpm it was shifted at do not move.
- **Mass** is published, not fitted: 495 kg (KK-SII) and 525 kg (KK-F),
  the class minimum with driver. The Inferno 86's 1,350 kg is the mass
  behind the review's drag prior and is a placeholder until the car's
  weighed mass with driver is entered (`car.PUBLISHED`); nothing in the
  solver uses mass, only the sanity band below.

On the dataset through 2026-11-14, in standard air:

| Chassis | Inferno 86 | KK-SII | KK-F (with the KK-F-V2) |
|---|---|---|---|
| `c_drag`, 1e-4 /m, across gears | 2.66 (2.30–3.23 by layout and engine) | 6.19 | 6.59 |
| the same at the top of the table | 2.68, 2.54 (Fuji) | 6.13, 6.24 | 6.95, 7.82 |
| `k_corner_drag`, /g | 0.047 | 0.032 | 0.025 |
| `mu_b` (to 2026-11-15: over every zone) | 0.89 (0.86–0.99 by layout) | 1.16 (1.15–1.17) | 1.18 (1.07–1.24) |
| `mu_b` (from C4: the zones braked in a straight line) | 0.95 | 1.21 | 1.25 |
| `brake_ellipse` (C4) | 0.69 | 0.52 | 0.63 |
| lateral level of the apex table | 1.34 | 1.67 | 1.70 |
| `mu_y` by convention (to 2026-11-15) | 1.221 | 1.511 | 1.522 |
| `mu_y` by convention (from C4) | 1.190 | 1.433 | 1.464 |
| `wet.grip_scale` | 0.92 | 0.81 | 0.82 |

| Powertrain | Days | `p_specific`, W/kg | Scale | Peak, W/kg at rpm | Shift point, rpm | Gears, m/s per 1,000 rpm |
|---|---|---|---|---|---|---|
| Inferno 86, first engine | 26 | 74.5 | | 79.8 at 7,000 | 7,290 (6,980–7,525 by day) | 3.43 / 4.87 / 6.21 / 7.64 / 10.00 |
| Inferno 86, second engine | 6 | 81.0 | 1.08 | 85.6 at 7,250 | 7,423 (7,339–7,526) | 3.56 / 5.13 / 6.55 / 7.99 |
| KK-F | 12 | 132.7 | | 136.1 at 6,250 | 6,481 (6,437–6,555) | 4.69 / 5.96 / 6.85 / 7.95 / 9.29 |
| KK-F-V2 | 1 | 151.6 | 1.12 | the KK-F's curve × 1.12 | 6,561 | the KK-F's (its day shows 5.98 / 6.85 / 7.95) |
| KK-SII | 19 | 124.4 | | 129.9 at 6,000 | 6,888 (6,544–6,936) | from speed over rpm: 4.47 / 5.49 / 6.53 / 7.72 (10 days), 4.80 / 6.24 / 7.32 / 8.62 (9 days) |

Power against rpm (W/kg):

| rpm | 5,000 | 5,500 | 6,000 | 6,500 | 7,000 | 7,250 | 7,500 |
|---|---|---|---|---|---|---|---|
| Inferno 86, first engine | 57.9 | 63.3 | 70.7 | 76.8 | 79.8 | 77.2 | 74.4 |
| Inferno 86, second engine | | 71.0 | 76.6 | 83.7 | 85.4 | 85.6 | |
| KK-F | 121.0 | 128.5 | 136.0 | 135.9 | | | |
| KK-SII | 120.6 | 125.3 | 129.9 | 125.5 | | | |

The KK-SII has rpm and no gear channel. Its gear is made from speed over
rpm, which gives it the curve, the shift point and the gear sets above;
the solver drives it on its one `p_specific` (see the rule above).

| Check | Threshold | Measured | |
|---|---|---|---|
| Power agrees between layouts | each layout's own within 7 % of its powertrain's, at equal rpm | −4.7 to +2.3 % (Inferno 86, first engine), −2.2 to +2.0 % (second), −3.7 to +2.4 % (KK-SII), −2.2 to +0.8 % (KK-F) | pass |
| Power at the wheels over published crank power per mass | 0.55–0.80, reported | Inferno 86 0.68 and 0.74 (at the placeholder mass), KK-SII 0.70 | reported |
| Recovery on simulated laps | power within 3 %, `c_drag` within 20 % | power −0.5 to +1.8 %, drag −0.9 to −0.3 % on five powertrains | pass |
| In-sample speed RMSE at each lap's own u | median over laps ≤ the cell's day floor + 0.4 m/s | 9 of 9 cells, 7 of them under their floor; the widest is the Inferno 86 at Sodegaura, 1.43 against 1.24 | pass |
| `ref_u` per layout | recorded | u between a car's layouts: Inferno 86 6.1 %, KK-SII 6.6 %, KK-F 7.3 % | recorded |

**`mu_y` is a convention** (`car_gate.set_convention`): the value at which
the median over the car's layouts of the reference u is 1, a layout's
reference u being the median over its days of the u of the day's best
build lap, each lap driven with its own day's powertrain, shift point and
air. It came out 9–10 % under the apex table's level on all three cars
(11–14 % since C4, where the braking zones stopped costing the time
the grip had to make up)
(1.221 against 1.339, 1.511 against 1.668, 1.522 against 1.698): the
table's level is a p90 of per-lap peaks, the upper edge of what the car
did, and a lap driven at that level everywhere is a lap nobody drove.
`mu_b` is a measurement and is not moved.

**`ref_u`** (median over the layout's days; days; sd over them):

| | Fuji | Motegi | Sodegaura | Suzuka | Tsukuba |
|---|---|---|---|---|---|
| Inferno 86 | 1.016 (10; 0.045) | 1.032 (2; 0.009) | 0.982 (16; 0.032) | | 0.973 (4; 0.097) |
| KK-SII | 0.939 (1) | | | 1.002 (7; 0.019) | 0.999 (12; 0.025) |
| KK-F | 0.973 (2; 0.003) | | | 1.000 (2; 0.019) | 1.044 (9; 0.030) |

This is the number a lap time predicted for a layout the car has not run
rests on (#202). With d ln T / d ln u of about −0.3, 6–7 % of u is 2 % of
lap time. The KK-SII's two layouts with more than one day agree to 0.2 %
(as in the review); what is left of its spread is Fuji, one day. Before
the air was taken out the formula cars read 9.4–9.7 %, low at Fuji and
high at Tsukuba together: Fuji is some 550 m up, the SAE factor there is
1.03–1.10, and an engine making that much less power had been read as
less grip.

**Recovery** (`car_gate.recover`). Each powertrain is simulated on the
car's layouts at u = 0.97, 1.00 and 1.03, at its own shift point; each
lap is made a session by the definitions the logged ones use (25 Hz,
accelerations as 0.5 s differences, full throttle where the engine is
what limits, brakes on where the tyres slow the car, `gear` and `rpm`
where the car has them) and put through C1's selection and C2's
estimator. Returned over true:

| | Power | Drag | `k_corner_drag` | Braking peak | Lateral level | Gears | Shift point |
|---|---|---|---|---|---|---|---|
| Inferno 86, first engine | −0.4 % | −0.9 % | 0.046 for 0.047 | −1.4 % | +2.2 % | exact | 7,171 for 7,290 |
| Inferno 86, second engine | −0.2 % | −0.8 % | 0.045 for 0.047 | −1.4 % | +2.3 % | exact | 7,297 for 7,423 |
| KK-SII | −0.2 % | −0.3 % | 0.016 for 0.032 | −3.6 % | +1.7 % | | |
| KK-F | +1.8 % | −0.3 % | 0.016 for 0.025 | −3.8 % | +1.7 % | | |
| KK-F-V2 | −0.5 % | −0.8 % | 0.021 for 0.025 | −4.0 % | +1.6 % | exact | 6,438 for 6,561 |

The cornering-drag estimate comes back right for the Inferno 86 and
0.004–0.016 low for the formula cars, so their own `k` is probably that
much above what their tables give; it is not corrected. No speed slope
of grip is estimated, so that clause of the issue has nothing to find.

**In-sample speed** (`car_gate.lap_scores`): every dry build lap
simulated at its own lap time with its day's powertrain, shift point and
air; RMS difference of speed over the track's nodes, median over laps,
beside the day floor (`car_gate.day_floor`: the same difference between
two logged laps of the car at the layout, from different days and within
0.5 % of lap time), m/s:

| Car, layout | Laps, days | Simulated | Day floor | Over | One power, standard air (as first built) |
|---|---|---|---|---|---|
| Inferno 86, Fuji | 111, 10 | 1.26 | 1.46 | −0.20 | 1.57 |
| Inferno 86, Motegi | 15, 2 | 1.07 | 1.29 | −0.21 | 1.16 |
| Inferno 86, Sodegaura | 355, 16 | 1.43 | 1.24 | +0.19 | 1.54 |
| Inferno 86, Tsukuba | 37, 4 | 1.34 | 1.53 | −0.19 | 1.37 |
| KK-SII, Suzuka | 106, 7 | 1.48 | 1.61 | −0.13 | 1.53 |
| KK-SII, Tsukuba | 193, 12 | 1.09 | 1.17 | −0.08 | 1.06 |
| KK-SII, Fuji | 25, 1 | 1.44 | no pair | not scored | 1.59 |
| KK-F, Fuji | 57, 2 | 1.33 | 1.86 | −0.52 | 1.40 |
| KK-F, Suzuka | 41, 2 | 1.52 | 1.51 | +0.02 | 1.51 |
| KK-F, Tsukuba | 189, 9 | 1.13 | 1.22 | −0.09 | 1.17 |

The last column is the same clause with cornering drag on but one
`p_specific` per car in standard air, the model as it stood before the
powertrains (the KK-F's Tsukuba cell had 166 laps then, without the
KK-F-V2's). The floors are higher than the review's 0.94–1.10 because a
pair here is any two laps of like lap time on different days, not a
selected few. No lap is outside what `solve_u` can reach; a build lap's u
runs from 0.83 to 1.08 (5th and 95th percentile over the cells).

**How cornering drag came into the default (2026-11-15).** As first
built the solver had no cost of cornering at full throttle, and the gate
failed one cell: the Inferno 86 at Sodegaura, 1.72 against a floor of
1.24. Four things were tried there and on three other cells, each lap at
its own u (m/s, one power per car, cornering drag off unless stated):

| Variant | 86 Sodegaura | 86 Fuji | KK-F Tsukuba | KK-SII Suzuka |
|---|---|---|---|---|
| as first built | 1.725 | 1.756 | 1.287 | 1.713 |
| cornering drag on (`k_corner_drag`) | 1.513 | 1.595 | 1.154 | 1.520 |
| each day's own power level | 1.730 | 1.801 | 1.285 | 1.702 |
| the layout's own braking peak | 1.772 | 1.819 | 1.286 | 1.715 |
| vertical curvature off | 1.751 | 1.894 | 1.263 | 1.800 |

Cornering drag lowered the error in ten of ten cells; vertical curvature
is worth 0.09–0.14 m/s at Fuji and Suzuka and nothing on the flat
layouts. The issue lets a term into the default only if it lowers the
error on the top quarter of speeds and does not raise the
leave-one-layout-out error; C3 measured both (lower in ten of ten cells,
twice) and the owner asked for it to be tried. `solver.CORNER_DRAG_DEFAULT`
is the one switch. "Each day's own power level" did not help then because
it was a level read per speed bin, which mixes the engine with the gear
the car was in; the powertrains, the shift point and the air are what
that row was reaching for.

**Beside the gate.**

- **The numbers agree with the design review's where it measured the same
  thing**: drag 2.7e-4 on the Inferno 86 (1.5–2.4e-4 at Fuji) and 6.2e-4
  on the KK-SII (6.0e-4); `k` 0.047 and 0.032 (0.04–0.06 and 0.02–0.05);
  the Inferno 86's braking peak 0.89 (0.90) with 13 % between layouts.
- **What the day levels are now.** At equal rpm and in standard air the
  Inferno 86's first engine has a day sd of 3.7 % over 26 days and its
  second 2.4 % over 6; KK-SII 3.4 %, KK-F 2.4 %. Read per speed bin with
  dry density only, the Inferno 86 had 6.0 % (the review: 5.6 %), which
  was the engine, the final drive, the shift point and the air in one
  number. What the correction leaves on the Inferno 86 still goes with
  density (a slope of 0.67 in the logs, R² 0.46): hot days read low by
  more than standard air accounts for.
- **The KK-F carries more drag than the KK-SII** (6.6e-4 against 6.2e-4),
  as its extra wing element says it should, and more power (133 against
  124 W/kg).
- **The KK-F-V2 is the KK-F with 12 % more power and nothing else
  changed.** Its scale is 1.12 at equal rpm in standard air on its one
  day (2026-11-13). Against the KK-F's day in the same air (2026-10-23,
  density 1.212 against 1.215) the uncorrected ratio is 1.09 / 1.09 /
  1.10 / 1.11 in the speed bins from 30, 35, 40 and 45 m/s: the same at
  every speed, which is what power looks like. Speed per 1,000 rpm is the
  same in every gear to 0.3 %, the revs used are the same, and apex
  speeds and lateral peaks agree within 0.5 m/s and 0.06 g except at the
  two fast corners, where the V2 is 1.0–1.4 m/s quicker. It is one day;
  the KK-F's own days scatter by 2.4 %.
- **Braking is the least stable number.** `mu_b` by layout: Inferno 86
  0.86 (Sodegaura) to 0.99 (Fuji); KK-F 1.07 (Suzuka, two days) to 1.24
  (Fuji, two days); KK-SII 1.15–1.17.

#### C3. The car on held-out days and layouts (`car_holdout.py`, gate `check-car-holdout`, 2026-11-15)

**Folds**: leave-one-day-out per layout and leave-one-layout-out. The car
is read from its tables without the held-out laps, `mu_y` set by the
convention on what is left, and each held-out build lap simulated at its
own lap time with the powertrain the car had that day, the day's shift
point and the session's air (`car_gate.Drive`). The shift point of a
held-out day comes from that day's own logs: it is the driver's, like
the lap time. **Statistics** per lap, a fold reading the median: S1, the
RMS difference of speed over the nodes; S2, the simulated less the logged
top speed, mean over the layout's straights (stretches between two apexes
of 200 m or more on which the car gains 5 m/s); S3, the same for the
minimum speed at the apexes; and the braking point, where the simulated
lap stops gaining speed less where the lap's brake application (peak
≥ 0.5 g) began. **Controls** through the same folds: the fold's own car
with its power × 0.85 and `mu_y` raised by the convention to hold the
reference laps; and another car's parameters (the KK-SII's for the
Inferno 86, the Inferno 86's for both formula cars). **Verdict**: a model
is inside a fold when S1 ≤ the fold's day floor + 0.4 m/s and |S2|,
|S3| ≤ 1.0 m/s; a fold where a control is inside, or that has no floor,
has no power against that control. A car that is outside on any powered
fold fails; one that is not passes when it has a powered day fold and a
powered layout fold for each control, and is otherwise not scored. The
track is the layout's own, built from every car's laps.

On simulated laps (the unit test) the gate passes the car that drove them
and finds both controls outside on every fold.

Gate C3 on the dataset through 2026-11-14 (73 folds of three laps or
more: 63 by day with 1,102 held-out laps, 10 by layout; two minutes, 24
before the solver's passes were compiled):

| Check | Threshold | Measured | |
|---|---|---|---|
| KK-F (with the KK-F-V2's day) | inside on every powered fold; both controls powered by day and by layout | inside 13 of 13 day folds and 3 of 3 layouts; both controls outside on all of them | pass |
| KK-SII | the same | inside 19 of 19 day folds, both controls outside on all 19; inside at Suzuka and Tsukuba as layouts, both controls outside (Fuji, one day, has no floor) | pass |
| Inferno 86 | the same | inside 27 of 31 day folds and 3 of 4 layouts; outside on four powered day folds, all before the shift light, and at Tsukuba as a layout | **fail**, cause below |
| Braking point | median within ± 5 m | Inferno 86 −20 m, KK-F −11 m, KK-SII −1.5 m by day (−17, −11, −1 by layout): the simulation brakes early on two cars | **fail** |
| Top-speed band | on a held-out layout's top quarter of speeds, the model's drag within 0.3 m/s of the best of 0 / 0.5 / 1 / 1.5 × it | the model's drag is the best of the four on all seven cells | pass |
| Cornering drag | in the default only if it lowers the top-quarter error and does not raise the leave-one-layout-out error | with it, S1 lower in 10 of 10 cells (by 0.08–0.23 m/s) and the top quarter in 10 of 10 (0.02–0.34) | met; it is on |

**Since 2026-11-16 the default carries two more terms** (C4 below), and
the gate reads: Inferno 86 inside on 29 of 31 day folds and 4 of 4
layouts (**fail** on two days, Sodegaura 2025-03-20 and Tsukuba
2025-07-26, both on S2); KK-SII 19 of 19; KK-F 13 of 13 and 3 of 3;
braking point +10.8 m (KK-SII), +3.3 m (Inferno 86), +0.6 m (KK-F) by
day (by the statistic as corrected in C4), so within ± 5 m on two cars
and late on the third. The table above
and the medians below are the gate as it stood on 2026-11-15.

Medians over the folds, S1 / S2 / S3 in m/s:

| | fitted | power × 0.85 | other car |
|---|---|---|---|
| Inferno 86, by day | 1.36 / +0.57 / +0.42 | 1.37 / −0.74 / +1.03 | 3.02 / +2.55 / −0.47 |
| Inferno 86, by layout | 1.31 / +0.61 / +0.11 | 1.42 / −0.76 / +1.06 | 3.08 / +2.83 / −0.77 |
| KK-SII, by day | 1.14 / +0.22 / +0.12 | 1.39 / −1.54 / +0.83 | 2.11 / −1.14 / +1.21 |
| KK-SII, by layout | 1.46 / +0.31 / +0.05 | 1.64 / −1.46 / +0.97 | 2.11 / −0.61 / +1.23 |
| KK-F, by day | 1.20 / +0.06 / +0.35 | 1.50 / −1.48 / +0.93 | 2.57 / −1.70 / +1.45 |
| KK-F, by layout | 1.40 / +0.06 / +0.34 | 1.80 / −1.56 / +1.15 | 2.52 / −1.25 / +1.59 |

**How the gate moved as the model did** (Inferno 86, day folds inside
and its median S2; the same run each time):

| Model | Inferno 86 inside | S2 | KK-SII | KK-F |
|---|---|---|---|---|
| one power per car, no cornering drag | 17 of 31 | +0.89 | fail (16 of 19) | pass |
| + cornering drag | 19 of 31 | +0.62 | pass (19 of 19) | pass |
| + powertrains, shift point, air, drag across gears | 27 of 31 | +0.57 | pass (19 of 19) | pass (13 of 13) |

The Inferno 86 by period of its history, day folds:

| | To 2025-10-05 (before the shift light) | 2025-10-13 to 2026-04-06 | Second engine |
|---|---|---|---|
| inside | 12 of 16 | 9 of 9 | 6 of 6 |
| S1 over its floor | +0.12 | −0.24 | +0.05 |
| S2 | +0.72 | +0.17 | +0.54 |
| the low-power control inside | 11 of 16 | 1 of 9 | 3 of 6 |

The second engine's six days, which the first model could not tell from
a 10 % error in power, are all inside with the car's own chassis and a
scale of 1.08 on the engine. So are the nine days after the shift light.

**What is left: two points go back to C2.**

1. **The Inferno 86 on its early, hot days.** The four failing day folds
   are Sodegaura on 2025-03-20, 05-05 and 06-09 and Tsukuba on
   2025-07-26 (S2 +1.58, a 36 °C day), and the failing layout is Tsukuba
   (S2 +1.21 over 37 laps on four days). Before the shift light the car
   is 0.7 m/s fast at the end of the straights with its own shift point
   and standard-air correction, and a car with 85 % of the power fits 11
   of those 16 days. C2's day levels say the same from the tables: at
   equal rpm what the air correction leaves on this car still goes with
   density (a slope of 0.67, R² 0.46), and the days before the shift
   light are its spring and summer. The candidate is a term for what
   heat costs this engine beyond standard air (the intake draws from a
   hot engine bay; the SAE factor knows only the ambient); by the issue's
   rule it would have to lower the leave-one-layout-out error to enter.
   Not built.
2. **The simulation brakes early**: 22 m on the Inferno 86 at Sodegaura
   (9 at Fuji, 11 at Tsukuba, 5 at Motegi), 8–22 m on the KK-F, and
   within 3 m on the KK-SII at Tsukuba. `mu_b` is one p90 over every
   zone of the car, which for the Inferno 86 is mostly Sodegaura's (0.86
   against 0.99 at Fuji), and a build lap's u of about 0.95 scales it
   down again; a stop from 55 to 25 m/s at 0.85 g instead of 1.0 g is
   22 m longer. The KK-SII's −66 m at Suzuka is the statistic, not the
   car: on a stretch with a lift or a dab between two apexes "the last
   stop that began on the stretch" is not always the stop for the corner
   at its end. The clause needs a braking peak that follows speed, and a
   matching of stops to corners, before it can be scored there.

**A gear change takes no time in the solver**, and the cars that lift to
shift are the ones still a little fast at the end of the straights
(Inferno 86 +0.2 to +0.5 m/s on its days after the shift light, KK-SII
+0.2 to +0.3, the flat-shifting KK-F +0.06). That reading was withdrawn
once, when cornering drag turned out to be most of what it had been
credited with; at this size it is consistent with a lift of a few tenths
of a second per shift and is inside the limits. Shift dips are a
reported variant of #206.

**What changed on the Inferno 86, from the owner and the logs
(2026-11-15).** The model above was prompted by two changes the day
levels showed and the owner named.

- **From October 2025 the gears are held about 350 rpm longer.** The
  owner: "in october I got a new shift light and started shifting
  slightly later". Full-throttle rpm at its 99th percentile is
  6,990–7,250 on every Sodegaura and Tsukuba day to 2025-10-05 and
  7,330–7,440 on every day after; third gear tops out at 34–35 m/s before and 36–36.7 after; the
  share of full-throttle samples above 7,000 rpm goes from 7 % to 29 %.
  The first day at the higher figure is Motegi on 2025-10-04; Sodegaura
  the next day is still at the lower one; from Fuji on 2025-10-13 every
  day is high. The engine is the same one: at equal rpm, weather
  corrected, the two cold Sodegaura days after it read 1–2 % above the
  last Sodegaura day before it.
- **Between 2026-04-06 and 2026-07-15 the engine was replaced** (the
  owner: "between april and July, the engine was replaced"), **and the
  final drive went from 4.3 to 4.1 with it.** Engine rpm per m/s of rear
  wheel speed in third gear is 207.0–207.3 on every day to 2026-04-06 and
  197.4–197.8 from 2026-07-15, and 163.0 against 155.5 in fourth: a
  factor of 0.954 in both, and 4.10 / 4.30 is 0.9535. Wheel speed over
  GPS speed is 0.97–0.99 throughout, so it is not the tyres. The new
  engine makes 8 % more power at equal rpm and holds it to 7,250 rpm
  where the old one falls off above 7,000 (the curves are in C2).

Read per speed bin, as the first model did, all of this was "power": the
bin from 35 to 40 m/s was fourth gear (0.13–0.15 g) and is now third
(0.18–0.21 g), +24 % at both Sodegaura and Fuji, while the bins that kept
their gear rose 4–13 %; and the same-month comparisons a year apart read
+6 % at Fuji, +14 % at Sodegaura and +7 % at Tsukuba with the weather
taken out.

**Recorded beside the gate, as the issue asks**: `ref_u` per layout with
its day sd is in C2 above. **Not built**: the era ladder (the table by
period above is its first row), the class's cross-circuit ratios and
wet-over-dry lap time against the model, the per-corner residual table,
wind on the main straight, and the standing comparison columns
(interpolation, the other-day median trace, the within-session floor).

#### C4. Two more terms: the braking limit's shape and the shift time (2026-11-16)

Both are measured per car and both are in the default solver
(`solver.BRAKE_SHAPE_DEFAULT`, `solver.SHIFT_LOSS_DEFAULT`), **on by the
owner's decision of 2026-11-16**: neither meets the rule cornering drag
entered on in every cell, and one car is worse with them, both recorded
below.

**Where the simulated lap left the logged one.** At every car's stops
from 30 m/s or more the simulation was 0.3 to 1.0 m/s fast just before
the driver brakes and then 1 to 2 m/s *slow* for about 100 m, level
again by the apex: it began braking earlier (gate C3's braking point:
17 m early on the Inferno 86, 11 m on the KK-F). Over the first 60 m
its deceleration was the logged one; what it lacked was deceleration
while cornering. The time it lost there it made up with grip, which is
why its apexes were fast (S3 +0.43 m/s on the Inferno 86).

**The braking limit's shape** (`car.braking_shape`, `grip.brake_ellipse`).
#206 writes the braking limit as a diamond: cornering at a share `x` of
the lateral limit leaves `1 − x` of `mu_b`. The braking zones say
otherwise. With `x` the lateral at a zone's peak deceleration over the
apex table's level and `y` that peak over `mu_b`, the 90th percentile of
`y` in bands of `x`:

| `x` | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | diamond at 0.65 | ellipse at 0.65 |
|---|---|---|---|---|---|---|---|---|
| Inferno 86 (4,002 zones) | 0.90 | 0.84 | 0.70 | 0.61 | 0.51 | 0.43 | 0.35 | 0.76 |
| KK-SII (2,080) | 0.81 | 0.70 | 0.61 | 0.56 | 0.47 | 0.40 | | |
| KK-F (1,842) | 0.86 | 0.78 | 0.56 | 0.53 | 0.50 | 0.46 | | |

`brake_ellipse` is the median over the bands of how far the 90th
percentile lies from the diamond towards the ellipse `√(1 − x²)`: 0.69
(Inferno 86), 0.52 (KK-SII), 0.63 (KK-F). The solver takes
`(1 − e)·(1 − x) + e·√(1 − x²)` of `mu_b`. It is a blend and not an
exponent on purpose: a power is not the same number to the last bit in
numpy and in compiled code, and a square root is.

**`mu_b` is now the level of the zones braked in a straight line**
(`car.braking_level`: a peak with under 0.2 of the lateral level): 0.95,
1.21, 1.25, where the 90th percentile over every zone was 0.89, 1.16,
1.18. A stop made while cornering has less than `mu_b` to brake with,
which is the shape's to say and not a lower `mu_b`; and read against
the level over every zone the shape estimator gave a simulated car's
share back 0.09 high. Read against the straight-line level it gives it
back (+0.03 in the median over fifteen simulated cars with shares from
0.2 to 0.8, seven laps a layout; the gate's own three laps a layout
read 0.10 to 0.12 high on few zones), and `mu_b` within 2 %.

**The shift time** (`car.shift_times`, a powertrain's `shift_s`). Across
an upshift the full-throttle table has a gap; the speed gained over it
falls short of what the power either side of it gives, and the
shortfall over the engine's own acceleration there is the time it gave
nothing, less what the same test reads over one second inside a gear
(0.00 to 0.05 s):

| upshift out of the | 1st seen | 2nd | 3rd | 4th |
|---|---|---|---|---|
| Inferno 86, first engine (gears 2 to 6) | 0.28 | 0.36 | 0.58 | 0.29 |
| Inferno 86, second engine | 0.25 | 0.33 | 0.62 | |
| KK-SII | 0.01 | 0.08 | 0.05 | |
| KK-F | 0.02 | 0.02 | 0.00 | 0.01 |
| SFJ | 0.15 | 0.15 | 0.20 | |

(seconds; the Inferno 86's slow one is fourth to fifth, across the
gate). A shift is the engine's work `P · shift_s` lost. Taken at a place
on the road it would make the lap time jump as u moves a corner's speed
across a shift point, so the solver takes it as 25 % less power over a
band of speed above the shift, as wide as loses that work on the flat
(`SolverCar.shift_bands`): the lap time stays continuous in u and
nothing changes in the passes. A version that coasts for the measured
time after each shift gave the same speed errors to 0.01 m/s on the
Inferno 86's build laps. On a grade the band over- or understates the
loss by the ratio of the accelerations.

**Gate C3 with each, re-run whole** (`mu_y` set again by its convention
each time; day folds, medians; S1 / S2 / S3 in m/s). The first rows are
with `mu_b` over every zone and the shape read against it (0.79, 0.57,
0.73), as the terms were first tried; the last of each car is the
default as it now stands:

| | inside | S1 | S2 | S3 | braking point | `mu_y` |
|---|---|---|---|---|---|---|
| Inferno 86, neither | 27 of 31 | 1.36 | +0.56 | +0.43 | −16.7 m | 1.222 |
| shift time | 26 | 1.26 | +0.26 | +0.54 | −14.2 m | 1.260 |
| braking shape | 21 | 1.09 | +0.85 | −0.21 | −4.5 m | 1.152 |
| both | 29 | 1.05 | +0.52 | −0.03 | −1.2 m | 1.192 |
| **both, `mu_b` in a straight line (the default)** | **29** | **1.11** | +0.54 | −0.03 | −1.2 m | 1.190 |
| KK-F, neither | 13 of 13 | 1.20 | +0.04 | +0.35 | −10.5 m | 1.522 |
| both | 13 | 1.18 | +0.20 | −0.01 | −1.0 m | 1.464 |
| **the default** | **13** | **1.23** | +0.21 | −0.02 | +0.6 m | 1.464 |
| KK-SII, neither | 19 of 19 | 1.14 | +0.22 | +0.12 | −1.5 m | 1.511 |
| braking shape (it has no shift) | 19 | 1.26 | +0.26 | −0.14 | +7.0 m | 1.434 |
| **the default** | **19** | **1.29** | +0.29 | −0.12 | +8.5 m | 1.433 |

Each alone moves the Inferno 86's error from one statistic to another
(the shift time takes the straights' and leaves fast apexes; the braking
shape takes the braking zones' and leaves fast straights); together they
bring it inside on 29 of 31 days and on all four layouts (3 before),
with the braking point 1 m from the logged one. What is left outside is
Sodegaura 2025-03-20 (four laps, S2 +1.02) and Tsukuba 2025-07-26
(S2 +1.26). The KK-F's braking point comes to 1 m as well and its
held-out layouts improve (S1 1.40 to 1.28). Gate C2 passes; in sample
the Inferno 86 is at 0.93 (Sodegaura) to 1.40 m/s (Motegi, floor 1.28).

**Against them.** The KK-SII, whose braking point the diamond already
had, brakes 8.5 m late, and its S1 by day rises from 1.14 to 1.29 (still
inside on all 19). The straight-line `mu_b` is the estimator that gives
a known car back, and it scores a little worse than the level over
every zone (S1 0.03 to 0.06 higher on each car): the logged laps do not
hold the peak they reach, and a share 0.1 to 0.25 lower scored better
still on the KK-SII and the KK-F (1.17 and 1.05), which was not taken
because it is a fit to the gate and not a measurement. On the
leave-one-layout-out cells (one term switched, `mu_y` left) the braking
shape lowers S1 in 6 of 10 and brings the braking point nearer in 6 of
10; the shift time lowers S1 in 5 of 7 and is worse at the Inferno 86's
two fast layouts; cornering drag, beside the two, is at 8 of 10 where
it entered on ten of ten.

**The braking point, measured where it can be** (same day). The gate's
braking point is where the simulated lap stops gaining speed less where
the driver's brake application began. Twice a lap at Suzuka the car
lifts for a fast corner and brakes 120 to 170 m later (into turn 1 and
into Spoon), and there the statistic read −122 and −168 m while the two
laps stop gaining speed within 14 to 18 m of each other. A lap's value
is the median over its straights with a stop, so a Suzuka day read −5 m
or −120 m according to whether a fourth straight registered a stop: the
KK-SII's "braking point" jumped from July, when it moved from Tsukuba to
Suzuka, and nothing in the driving did. The statistic now takes a
straight only if the logged lap gained speed on it until it braked (its
speed peak within 30 m of the application; `car_holdout.BRAKE_CLEAN_M`),
and the gate reports `peak_to_peak` beside it, the same from speed peak
to speed peak over every straight. Nothing else in gate C3 moved. By
day, with the default:

| | braking point | speed peak to speed peak |
|---|---|---|
| Inferno 86 | +3.3 m | +11 m |
| KK-F | +0.6 m | +12 m |
| KK-SII | +10.8 m | +14 m |

(The braking points in the table of configurations above are by the
statistic as it then was.) The KK-SII is +6 to +20 m on each of its
twelve Tsukuba days and +7 to +15 m on each of its seven Suzuka days:
no step in the summer.

**What is still missing: the stop does not begin at once.** Speed peak
to speed peak the simulation is 11 to 14 m late on all three cars; the
application is a poor reference because the driver lifts before it.
From the stops made from full throttle at 30 m/s or more (1,559 of
them, 51 sessions):

| | Inferno 86 | KK-F | KK-SII |
|---|---|---|---|
| lift to brake application | 0.32 s | 0.24 s | 0.16 s |
| speed peak to application | 0.20 s, 7 m | 0.24 s, 12 m | 0.08 s, 3 m |
| brake pressure to 80 % of the stop's highest | 0.24 s | 0.36 s | 0.20 s |

The solver goes from full drive to full braking between two nodes. A
lift of 0.16 to 0.32 s and half a build of 0.2 to 0.36 s are 12 to 19 m
at these speeds, which is the size of what it is late by. Twenty metres
after the application it is 0.8 to 1.6 m/s fast on every car, and by
the apex 0.2 to 1.2 m/s slow. Under the diamond the two errors hid each
other: too little braking while cornering made it start early, and no
lift or build let it start late. This is also why a lower braking share
scored better than the measured one. The transition is the next term
(tried below).

**The onset, tried (same day; `solver.BRAKE_ONSET_DEFAULT`, off).** The
braking-zone table now has, per stop, the time from the last sample at
full throttle to the application (`lift_s`, between the middles of the
two samples) and from the application to 80 % of the stop's highest
pressure (`build_s`). `car.brake_onset` takes the lift plus half the
build over the dry stops from full throttle at 30 m/s or more
(`grip.onset_s`): 0.43 s (Inferno 86: 0.28 + 0.24 to 80 %), 0.25 s
(KK-SII: 0.12 + 0.20), 0.40 s (KK-F: 0.20 + 0.32), each steady from
day to day (0.36–0.53, 0.20–0.32, 0.33–0.44). It is the driver's on
this car, like the shift point, and is carried per car for now.

The solver cuts every speed peak of the lap flat where it is that long
in time (`_kernel.open_peaks`): the lap leaves the drive curve early,
coasts, and joins the same braking curve. It is the opening of the
speed trace by a window of that time (the lowest speed in the window
after each node, then the highest of those in the window before it),
which is why it needs no event: minima are untouched, a top already
that flat is untouched, the result moves smoothly with u, and it is
minima and maxima only, so the numpy and the compiled form agree to the
last bit. A simulated car's onset comes back 0.02 to 0.035 s high.

Gate C3 re-run whole with it (day folds, medians):

| | inside | S1 | S2 | S3 | speed peak to speed peak |
|---|---|---|---|---|---|
| Inferno 86, without | 29 of 31 | 1.11 | +0.54 | −0.03 | +11 m |
| with | **31** | 1.10 | +0.06 | −0.02 | −4 m |
| KK-SII, without | 19 of 19 | 1.29 | +0.29 | −0.12 | +14 m |
| with | 19 | 1.29 | +0.04 | −0.12 | +4 m |
| KK-F, without | 13 of 13 | 1.23 | +0.21 | −0.02 | +12 m |
| with | 12 | 1.21 | −0.22 | −0.01 | −4 m |

S1 is lower on 41 of the 63 day folds and higher on none. The Inferno
86 is inside on every day and every layout, so it **passes** gate C3
for the first time; its fast straights (S2) were the missing onset. The
KK-F loses one day, the KK-F-V2's (2026-11-13, S2 −1.16: with the onset
its top speed is short, on a powertrain with one day and a borrowed
curve). The braking point against the application reads −12.6, +0.2 and
−17.0 m, which is the lift itself (the logged lap stops gaining speed 7,
3 and 12 m before the application): against the logged speed peak the
simulation is within 4 m on all three (8 m on the KK-F's held-out
layouts). Left off pending the owner's decision; the gate's braking
point would have to be read speed peak to speed peak with it.

The KK-SII's three days of 2026-05-22 to 05-24 are the one real shift
in its braking: the straight-line peak falls to 1.00–1.09 g from 1.19–
1.26 g on every other day, with the pedal pressure, while cornering,
power and lap time are as before. The team's notes for the 22nd have
"spitting rain/misting" and, in the next session, "ended hitting a vita
in T1".

**The KK-SII stays on its level.** Through its curve and gears with the
shift time its S1 is higher on 12 of 19 days and on two of three
layouts (S2 +0.29 to +0.40), so `solver.INFERRED_GEARS_DRIVE` stays off.

#### C5. Review of the car model by seven reviewers (2026-11-16/17)

Asked by the owner: are these models physically right, and accurate? An
advocate, a physics critic, an estimation critic, a validation critic,
the downstream consumer (tyre heat), a researcher in the logs and one of
outside practice; two rounds, the second one deciding measurement each
with its threshold written before the run. Nothing below changed the
code; the scripts were in the session's scratch and are gone, so this is
the record. "In sample" means the car fitted on every day and each lap
driven at its own lap time.

**1. The simulated lap brakes harder and corners softer than the car,
and the pace knob hides it.** Logged straight-line deceleration is
0.81–0.87 of simulated; the simulation is +0.85 to +2.4 m/s fast where
the logged car brakes and −0.3 to −1.2 m/s slow on entry and mid-corner
in all ten car-layout cells; 0.4–2.8 s of phase error sits behind a
right lap time (in sample). In heat: lateral −0.5 to −3.0 %,
longitudinal +0.7 to +6.0 %, in ten of ten cells.

**2. `mu_b` is the stop the driver makes one time in ten; `mu_y` was
already what is held.** Over 2,502 straight-line stops and 7,230 apexes
on laps within 2 % of u = 1, deceleration is held at 0.80 / 0.80 / 0.78
of `mu_b` (Inferno 86 / KK-SII / KK-F) and lateral at 0.87 / 0.89 / 0.89
of the apex table's level, which is where the convention puts `mu_y`
(0.89 / 0.86 / 0.86). The held ratio of braking to cornering is 0.64–0.65
on all three cars; the solver runs 0.80–0.85. Braking use moves from day
to day (sd 0.05–0.08 within a layout, range 0.68–0.94) and cornering use
does not (0.01–0.02); the two do not move together.

**3. Held out, a car with less braking beats the fitted one.** Car
re-fitted per fold, laps paired: `mu_b` × 0.84 scores a lower S1 on 21
of 31, 9 of 13 and 17 of 19 days. A car with drag +10 %, `mu_b` × 0.84
and shape −0.2 is inside wherever the fitted car is (28 of 29, 12 of 13,
18 of 19 folds). Drag +10 % and power −5 % are coin flips; power +5 %
is told (22, 12, 18 days). `mu_y` against `mu_b` cancels exactly, power
against drag, shift time and cornering drag nearly, `mu_b` against shape
and onset nearly.

**4. Handing the solver the measured braking use closes the braking
bias only in the median** (in sample). Scaled by the stop's plateau over
`mu_b` (0.88 / 0.85 / 0.78) the bias is within ± 0.5 m/s in 5 of 10
cells with one number a car and 3 of 10 with the day's; entry stays at
−0.5 m/s whatever the braking level. The use has to be measured against
the limit the solver applies, `mu_b · u`: there the level that zeroes
the bias is 0.83–0.93 in nine cells (read after the run, not fitted).

**5. The gate could not fail a model on speed.** S1 against the floor
+ 0.4 m/s admits 1.3–2.0 m/s of error that repeats lap after lap; the
model's is 0.40–1.13 m/s. "Both controls outside on every powered fold"
is true by definition: power × 0.85 is inside on 16 of 31 Inferno 86 day
folds. The verdict turns on S2 near ± 1.0 m/s ("the Inferno 86 passes
with the onset" holds in 76 % of resamples).

**6. S1 is not what the tyre model needs.** It explains 4 % of the
per-tyre heat error; a signed speed error weighted by where lateral heat
is made, with S2, explains 66 %. Worst-tyre heat bias is 2.3 % without
the three C4 terms, 2.4 % with the braking shape and shift time, 2.6 %
with the onset (worse in ten of ten cells). Everything contested in C4
is worth under 1.2 % of heat; 1 m/s where lateral heat is made is 4.1 %.

**7. One clean test, thresholds first** (134 wet build laps of the three
cars and the SFJ's 51 laps, never scored before): repeatable speed error
1.36 m/s without the C4 terms, 1.19 with shape and shift (−0.17
[−0.20, −0.14]), 1.19 with the onset; worst-tyre heat 3.4 / 3.5 / 3.6 %.
The gain is on the cars the terms were developed on (Inferno 86 wet
−0.53, KK-SII wet −0.13 m/s); on the SFJ nothing moves. On speed alone
the power × 0.85 Inferno 86 scores better than the fitted one in the
wet; heat tells them apart in every group.

**8. At a track the car has not run, the pace level is the error.**
Leave-one-layout-out, per-tyre heat: 2.4 % with u from the lap's own
time, 5.9 % with u carried from the car's other layouts (8 of 10 folds
over 3 %; Inferno 86 1.0 → 7.1 %). The held-out layout wants a u of
0.92–1.07 and that predicts every fold's shift within 0.3 %.

**9. The per-corner factor is not the track's as a whole; banking is
the part that is** (in sample, 13 shared limited corners between unlike
cars). A factor from an unlike car removes −16 % of the corner-to-corner
variance of apex-speed error (formula to Inferno 86: −58, −66 %);
between the two formula cars +63 and +71 %. The part that repeats
between unlike cars (r = +0.76, +0.87) goes with the elevation model's
cross-slope (r = +0.68 to +0.78): mean +0.032 rad into a corner, 6.6 %
of lateral grip, from 0.009 at Sodegaura to 0.059 at Tsukuba. The other
part follows the car class's line. With cross-slope held the formula
cars' grip still rises with V² (ClA 0.45 and 0.70 m²).

**10. The onset** trims the speed peak and leaves the braking curve
(its last node is still 10–14 m late; `peak_m` read its first); its top
is 1.00 s wide where the logged one is 0.67 s; it changes no count in
items 3 and 4.

**11. Banking, as physics from the elevation model** (a third round,
2026-11-17; thresholds first; a prototype pass outside the repository
that matches the solver to 2e-6 m/s at zero bank). The bank at each 2 m
node is the slope of a line through the 5 m mesh's heights from −5 to
+5 m across the reference line, ± 10 m mean along it, bridges and
sections that are not a plane interpolated (202 nodes at Suzuka, 175
under the crossover; 6 at Motegi). Its noise is 0.003–0.004 rad at Fuji,
Sodegaura and Tsukuba, 0.010 at Motegi, 0.019 at Suzuka. Mean bank into
the corners is 0.031–0.037 rad on four layouts and 0.011 at Sodegaura,
the largest 0.08–0.10 rad. Against the accelerometer's reading of it,
where the bank is large and there are four sessions or more, the slope
is 0.92 and 0.97 (Inferno 86 at Fuji and Tsukuba), 0.99 (KK-F at
Tsukuba), 0.74 (KK-SII at Suzuka), r = +0.62 to +0.80; Tsukuba's
hairpins read 0.080 and 0.082 rad against 0.075–0.093 from two cars'
sensors. In the lap, with the tyre's lateral `V²κ·cos − g·sin` and the
load `g·cosθ·cos + V²κ·sin`, at full strength and `mu_y` set again:

| | without | with the bank |
|---|---|---|
| heat at a layout the car has not run, u carried (worst tyre, median of 10 folds) | 5.9 % | 3.7 % (better in 6) |
| corner-to-corner variance of apex-speed error removed | | 30 / 17 / 12 % |
| reference u between layouts (sd) | 0.080 / 0.069 / 0.060 | 0.128 / 0.073 / 0.033 |
| repeatable error, dry (layouts equal) | 0.781 m/s | 0.784 [−0.007, +0.010] |
| repeatable error, wet laps | | −0.077 m/s [−0.091, −0.062] |
| lateral heat, left against right | | +6 % (no) |

(Inferno 86 / KK-SII / KK-F.) Half and one-and-a-half strength are both
worse on the apex speeds, so nothing says the elevation model under- or
over-reads. The Inferno 86's wider spread of u is Tsukuba alone (0.956
to 0.901; its other three layouts close from 0.071 to 0.032): with the
right bank the hairpins overshoot (KK-F +0.19 to +0.62 m/s) and the
fast corners fall further behind (−1.02 to −1.62 m/s), which is the car
or the driver and not the road. `mu_y` then sits 16 / 24 / 21 % under
the apex table (9 / 16 / 8 % against the table in the tyre's frame).
The reviewer's rule, set beforehand, puts the bank into the track model
as a recorded array and leaves the solver term off; the elevation model
loses crown and kerbs inside 5 m and reads the ground under a bridge,
and #251's survey of Fuji is the truth to hold the 0.92 against.

**Defects found:** `brake_ellipse` is estimated against the apex table's
level and applied against `mu_y · u`, 11–16 % lower; there is no rolling
resistance (the Inferno 86's drag area reads 0.62 m² against about
0.53); `k_corner_drag` returns half its planted value on the KK-SII and
has no limit in the gate; the committed recovery tests the onset only
at zero.

**What the reviewers ask for.** The car's limits apart from the
driver's use of them, each measured; a gate that carries per-tyre heat
(lateral and longitudinal apart) and the repeatable error by phase, with
layouts weighted equally and paired controls a few per cent wrong;
recovery on every parameter with noise and the real spread of pace; a
term named with its one statistic before it is run and scored once on
laps not looked at; cant from the elevation model before any more
braking terms.

#### C6. Braking: how it is judged (2026-11-21)

**The measure** (owner's decision, 2026-11-21: one plain number for how
well the simulated braking matches the real braking). Wherever the
logged lap slows by 0.2 g or more, the difference between the simulated
and the logged deceleration at the same place on the track, as one RMS
figure in g per lap (`car_holdout.braking_match`; deceleration from the
change of V² over 16 m centred on each 2 m node,
`car_holdout.deceleration_g`).

**It is read at three levels of pace** (owner, 2026-11-21: "to make sure
our solutions scale with pace"): on the laps above the 75th, the 90th
and the 95th percentile of pace among the car's laps at the layout
(`car_holdout.PACE_LEVELS`, `fast_laps`). The fast laps matter most and
repeat best; three levels show whether the simulation holds as the pace
rises. Percentiles and not a margin of lap time: the same share of the
laps at every layout. Gate C3 scores the day folds (the car fitted
without the lap's day, the lap driven at its own lap time), each figure
the median over days of the day's median lap.

**The yardstick is reality against itself** (`car_holdout.pace_yardsticks`):
the same measure between the logged lap and the nearest three laps in
lap time of the same set, from other days and from its own day.

**The gate is a score, per car and circuit** (owner, 2026-11-21: "let's
move away from the gate being a hard pass/fail, and treat it as
attainment score, our goal is to maximize the number of passes"; "per
circuit/car"). A cell is one car at one layout at one level of pace on
one measure: speed round the lap (S1), the size of the end-of-straight
and of the apex speed error, and the braking. It passes when the
simulated lap is no further from the logged lap than a logged lap of
another day is (`car_holdout.attainment`); a cell whose set holds one
day's laps has nothing to be set against and is not scored. Gate C3
reports the count and nothing in it is a pass or a fail any more: the
verdict per car on S1, S2 and S3 over every quick lap and the top-speed
band are reported beside it.

**Today: 34 of 76 cells** (32 of the 108 cannot be scored). By measure:
speed round the lap 8 of 19, end of straights 14 of 19, apex 10 of 19,
braking 2 of 19. By level: P75 13 of 32, P90 9 of 24, P95 12 of 20.
Passes of the levels scored:

| car | layout | speed round the lap | end of straights | apex | braking |
|---|---|---|---|---|---|
| Inferno 86 | Fuji | 1 of 3 | 2 of 3 | 1 of 3 | 0 of 3 |
| Inferno 86 | Motegi | 0 of 1 | 1 of 1 | 0 of 1 | 0 of 1 |
| Inferno 86 | Sodegaura | 3 of 3 | 3 of 3 | 1 of 3 | 1 of 3 |
| Inferno 86 | Tsukuba | not scored: its fast laps are one day's | | | |
| KK-F | Fuji | 1 of 1 | 0 of 1 | 1 of 1 | 0 of 1 |
| KK-F | Suzuka | 0 of 2 | 1 of 2 | 0 of 2 | 0 of 2 |
| KK-F | Tsukuba | 1 of 3 | 3 of 3 | 3 of 3 | 0 of 3 |
| KK-SII | Suzuka | 2 of 3 | 3 of 3 | 1 of 3 | 0 of 3 |
| KK-SII | Tsukuba | 0 of 3 | 1 of 3 | 3 of 3 | 1 of 3 |

(The KK-SII at Fuji has one day and no day fold.)

**The braking point is gone** (owner, 2026-11-21: "get rid of the old
gate, use the new metric"). The braking
point (`brake_m`, where the simulated lap stops gaining speed less where
the logged brake came on, ± 5 m, with `peak_m` beside it) is not in
`car_holdout.py`: an 8 % error in the braking level moved it 2 m where
8 % of power moved it 8 m and the driver's onset 17 m. The sections above
that quote a braking point (C3, C4) are the record of what it read while
it stood.

Pooled over a car's layouts (the median over all its days), the braking
reads:

| car | level | laps / days | simulated against logged | two logged laps, other days | one day |
|---|---|---|---|---|---|
| Inferno 86 | P75 | 131 / 20 | 0.221 g | 0.205 g | 0.155 g |
| | P90 | 54 / 12 | 0.230 | 0.197 | 0.155 |
| | P95 | 27 / 10 | 0.233 | 0.205 | 0.149 |
| KK-F | P75 | 75 / 10 | 0.293 | 0.214 | 0.167 |
| | P90 | 30 / 8 | 0.288 | 0.170 | 0.170 |
| | P95 | 16 / 5 | 0.254 | 0.159 | 0.198 |
| KK-SII | P75 | 77 / 17 | 0.264 | 0.211 | 0.186 |
| | P90 | 31 / 10 | 0.271 | 0.196 | 0.148 |
| | P95 | 16 / 5 | 0.249 | 0.199 | 0.185 |

Pooled, every car is over its yardstick at every level: the Inferno 86
by 0.02–0.03 g, the KK-SII by 0.05–0.08 g, the KK-F by 0.08–0.12 g. Over every
held-out lap, each paired with laps within 0.5 % of its lap time, the
three read 0.22 / 0.27 / 0.28 g against 0.21 / 0.23 / 0.22 g: taken over
all the laps the simulation looked about as close to a logged lap as a
second logged lap is, and the fast laps show it is not.

**The speed measures on the same laps and yardstick**, pooled over a
car's layouts in the same way. Simulated against logged / two logged
laps of other days; S2 and S3 as sizes:

| car | level | S1 (m/s) | end of straights (m/s) | apex (m/s) |
|---|---|---|---|---|
| Inferno 86 | P75 | 0.95 / 1.17 | 0.27 / 0.39 | 0.25 / 0.29 |
| | P90 | 0.90 / 0.99 | 0.22 / 0.37 | 0.33 / 0.24 |
| | P95 | 0.95 / 1.03 | 0.24 / 0.31 | 0.45 / 0.28 |
| KK-F | P75 | 1.20 / 1.18 | 0.26 / 0.45 | 0.17 / 0.33 |
| | P90 | 1.19 / 1.08 | 0.23 / 0.43 | 0.23 / 0.40 |
| | P95 | 1.18 / 0.97 | 0.25 / 0.68 | 0.09 / 0.38 |
| KK-SII | P75 | 1.15 / 1.04 | 0.16 / 0.21 | 0.39 / 0.24 |
| | P90 | 1.14 / 0.92 | 0.26 / 0.17 | 0.29 / 0.31 |
| | P95 | 0.97 / 0.93 | 0.23 / 0.11 | 0.18 / 0.34 |

- **Round the lap the Inferno 86 is nearer a logged lap than a second
  logged lap is** at every level; the two formula cars are further, and
  the KK-F by more as the pace rises (its error stays at 1.2 m/s while
  two logged laps close from 1.18 to 0.97): it does not scale with pace.
  Two logged laps of one day are 0.76–0.93 m/s apart; no car is there.
- **Over every quick lap the picture was kinder**: 1.10 / 1.28 / 1.33
  m/s against 1.31 / 1.38 / 1.31 between logged laps. The Inferno 86's
  end-of-straight error (+0.56 m/s over every lap) is 0.27 by P75: it
  belongs to slower laps on which the real car did not reach its speed.
- **The apex and end-of-straight sizes do not settle** at any level
  (the KK-F's end-of-straight yardstick runs 0.45, 0.43, 0.68): each is
  a mean over a handful of straights or corners a lap.
- **Above P95 there is nothing to score against.** At the 99th
  percentile a layout's set is one to four laps of one day; where a
  figure can be formed (the KK-F: 0.59 m/s from two laps) it means
  nothing, and the Inferno 86's fastest lap at Tsukuba is 4.6 % clear
  of its other days. P75 keeps 75–131 laps a car, P90 30–54 and P95
  16–27; at P95 the formula cars rest on five days.

**What it says of the braking limit's shape** (the term of C4, on since
2026-11-16; gate C3's comparison on the layout folds, every held-out
lap): with the shape the braking is further from the logged lap's in
nine of ten cells (0.19–0.34 g against 0.17–0.28 g with the diamond;
nearer only for the KK-F at Fuji), where S1 is lower in six of ten. The
term was let in on S1 and the braking point; by this measure it does not
earn its place. Nothing is changed on that account yet.

**How to read the yardsticks.** Two logged laps each carry their own
lap-to-lap variation, so a simulation that matched the driver's usual
braking would read under the other-day figure; the one-day figure is
about the best any simulation of this driver could do.

**Laps are compared by position, and nothing is moved to line them up.**
Braking at another place counts as much as braking at another strength.
That makes the number feel a position error, measured as a fast logged
lap against itself moved along the track: 1 m reads 0.02–0.03 g, 2 m
0.04–0.05, 3 m 0.06–0.08, 5 m 0.10–0.13 (it adds to a real difference as
the root of the sum of squares). A GPS offset moves a whole lap one way,
so the shift it makes at each corner follows the corner's direction; one
offset fitted per day comes out at 2.5–3 m and explains the corners'
shifts beyond chance on the KK-F only (45 % against 18 %; 14–22 % against
15–19 % on the other two), while the shifts themselves are 3.5–6 m rms.
So drift is an upper 2.5–3 m, about 0.01 g on the other-day yardstick and
a fifth to a half of its gap to the one-day one. Two logged laps of
different days carry two offsets and a simulated lap against a logged
one carries one. When two versions of the simulation are scored on the
same laps the offset is the same for both: it sets how low the number
can go, not which version is nearer.

**What it does not say.** Why a lap differs (the car's limit or the
driver's use of it), or whether the difference is in where the braking
starts or how hard it is. A power error reaches it through the pace knob
(below).

**What softer braking does to it** (in sample, the lead's probe before
the team: `mu_b` × 0.83 / 0.87 / 0.73 for the Inferno 86 / KK-F /
KK-SII): 0.18 / 0.20 / 0.19 g on the fast laps against 0.23 / 0.27 /
0.31 in sample for the default, nearer on 92–100 % of days. The measure
answers to a braking change.

**The team's first stage, as a record** (six members: the tyre model as
consumer, the coaching use, the logs, validation on planted cases,
physics, outside practice; a round of positions and a round of debate,
thresholds first; the scripts were in the session's scratch). It
proposed a set of four measures; the owner set it aside as too
complicated and asked for the one above. What it measured stands:

- **The simulation sheds 11–15 % more speed than the driver** over the
  middle of a stop (between 20 % and 90 % of the speed shed: +0.151 /
  +0.111 / +0.151 as a log ratio), on 98 % of days, and as much after
  turn-in as before it. 57–63 % of a stop's energy is shed after
  turn-in.
- **The fitted `mu_b` is about right as a sustained level and the driver
  is below it.** Over the hard straight part of big stops the best tenth
  sit at 0.94–0.95 of it (the 0.5 s peak reads 1.04–1.10: a spike); over
  all braking time the driver is at a median 0.82–0.84 of the limit and
  the simulation at 1.00.
- **The corners with a low level are corners where less is asked**: the
  level follows the pedal pressure (R² 0.78–0.98 across corners), is
  stable over days and repeats between cars (r = 0.94–0.99). Not grade,
  and no downforce is shown (under 2 standard errors).
- **Heat cannot judge braking.** The braking part of a tyre's heat is
  the energy shed, whatever the level: it is 13–27 % of a front tyre's
  heat and today's error in it is at most 0.004–0.006 bar; no heat
  statistic moves beyond its day floor for a level wrong by 0.08. One
  per cent of heat is 0.003–0.005 bar, not the 0.01 of C5. Braking does
  not bear on the pace level at a layout the car has not run (Inferno 86
  at Tsukuba: −0.106 whatever the braking).
- **`brake_m` and the braking-zone speed error are not measures of
  braking.** An 8 % level error moves `brake_m` by 2 m, 8 % of power by
  8 m and the driver's onset by 17 m; the speed error in the braking
  zones answers to power one for one.
- **No measure at a lap's own u is clean of power.** The simulated car
  brakes at `mu_b·u` and 8 % of power moves u by 4 %, so every ratio of
  decelerations reads half a power error. Power has to be right first.
- **A limit cannot be told from a cautious driver** in one driver's laps:
  a driver 8 % under the limit in the right car reads as a car 8 % too
  high. Only "not too low" can be shown. On the formula cars the
  deceleration still rises with pedal pressure: their limit has not been
  reached in the logs; the Inferno 86's has.
- **The KK-F's brake channel is late** by 0.13 s (6 m) against the onset
  of its deceleration. A "use" measured from the application inherits
  it (the lead's 0.87 for that car).
- **A shape statistic that needs no `mu_y`** (the physics member's: the
  time-weighted p90 of deceleration over the stop's own peak, in bands of
  lateral over the lap's own apex level) moves ± 0.054 for ± 0.2 of
  `brake_ellipse` and under 0.007 for level, power or a cautious driver.
  On hard stops it reads the logged edge fuller than the model by 0.028–
  0.048 (± 0.020–0.024), beyond 2 standard errors on the KK-F only. Not
  built.

#### C7. The driver's braking use, and the committee that planned it (2026-11-21)

**The question** (owner, 2026-11-21): "make a committee to plan how we
can improve our simulation to increase our attainment scores and report
the plan back". Seven seats (the score's arithmetic, braking, the
corners, the drive, the pace knob, outside practice, a skeptic), a round
of positions and a round of debate, every seat measuring on one pack of
the gate's held-out traces with the gate's own scorer, thresholds
written before the runs; nothing was built in the repository until the
owner said "build it" to step 1.

**The plan it reported.**

| step | what | on the score | verdict |
|---|---|---|---|
| 1 | the driver's braking use, apart from the car's limit | 34 → 56 of 76 | built (below) |
| 2 | apex speed at a given lap time | 7 to 15 cells in reach | to be investigated; no term on the table moves it |
| 3 | the Inferno 86's power from 2025-11 | inside chance | real in the logs; not a step for the count |
| 4 | the brake's onset (`BRAKE_ONSET_DEFAULT`) | −4 on top of step 1 | stays off until the drive is right |
| — | the knob off braking; grip against speed; the car's own line; no shift time; the KK-SII through its gears | −3 to −10 | dropped |

**Step 1 as built** (issue #312's split: the car's limits apart from how
it is driven). The car's braking limit stays the car's (`grip.mu_b`,
`grip.brake_ellipse`). The model gains a `driver` entry: `brake_use`,
the level the driver brakes at as a share of `mu_b·u`, and
`brake_use_ellipse`, the shape held under cornering in place of the
limit's. A lap is braked at `brake_use · mu_b · u` (`SolverCar.brake_use`;
the pace knob multiplies it like the limit, the kernel is untouched and a
car without a `driver` is the old arithmetic to the last bit).
`Drive.brake_use = False` brakes at the limit, for the comparison.

*The reading* (`car_gate.set_use`, rule written before it was scored).
From the logged speed of every dry build lap of the car, each simulated
at its own lap time for its u: the nodes that slow by 0.2 g or more
(the braking measure's own zone), 0.5 s or more into the stop (past the
brake's build); per node the braking force (what slows the car less the
drag and the grade) over `mu_b·u` times the load, against the lateral
over `mu_y·u` times the load; the median of each per band of lateral 0.1
wide from 0 to 0.9 (200 nodes or more); then `A·(1 − x) + B·√(1 − x²)` by
least squares over the bands, the solver's own form: the level `A + B`
and the shape `B / (A + B)`. The median and not a high percentile: this
is what the driver does, where `car.braking_shape` reads what the car
can. A lap's u depends on how it is braked, so the reading is repeated
with the laps braked as read until the level moves by less than 0.015
(each repeat moves it by a ninth of the one before: two readings, the
second within 0.002 of where it settles). The convention for `mu_y` is
set after it, and moves the share with `mu_y` so that every lap stays
the same lap (`car_gate._with_mu_y`): softer braking takes lap time that
the knob gives back, so `mu_y` rises by 2.8–3.8 %.

| car | `brake_use` | level at u = 1 | held shape (the limit's) | `mu_y` (before) | read back from laps simulated with it |
|---|---|---|---|---|---|
| Inferno 86 | 0.80 | 0.76 g | 0.59 (0.69) | 1.160 (1.128) | −0.6 %, 0.59 |
| KK-SII | 0.85 | 1.03 g | 0.28 (0.52) | 1.318 (1.270) | −0.1 %, 0.29 |
| KK-F | 0.84 | 1.05 g | 0.47 (0.63) | 1.372 (1.334) | −0.1 %, 0.48 |

Gate C2 carries the read-back as a check (within 2 % of the level and
0.05 of the shape): the committee's first rule, a median at low lateral,
read a planted level 1.4–5.9 % low and was replaced by this one.

**Gate C3 with it: 56 of 76** (34 with the fold's car braked at its
limit, `mu_y` set again: 25 cells won, 3 lost). The reading is made in
every fold from the laps that are left.

| measure | with the use | at the limit | won | lost |
|---|---|---|---|---|
| speed round the lap | 14 of 19 | 8 | 6 | 0 |
| end of straights | 16 of 19 | 14 | 2 | 0 |
| apex | 8 of 19 | 10 | 1 | 3 |
| braking | 18 of 19 | 2 | 16 | 0 |

By level: P75 23 of 32 (13), P90 18 of 24 (9), P95 15 of 20 (12).

| car | layout | speed round the lap | end of straights | apex | braking |
|---|---|---|---|---|---|
| Inferno 86 | Fuji | 1 of 3 | 3 of 3 | 0 of 3 | 3 of 3 |
| Inferno 86 | Motegi | 1 of 1 | 1 of 1 | 0 of 1 | 1 of 1 |
| Inferno 86 | Sodegaura | 3 of 3 | 3 of 3 | 0 of 3 | 3 of 3 |
| KK-F | Fuji | 1 of 1 | 1 of 1 | 0 of 1 | 1 of 1 |
| KK-F | Suzuka | 2 of 2 | 1 of 2 | 0 of 2 | 2 of 2 |
| KK-F | Tsukuba | 3 of 3 | 3 of 3 | 3 of 3 | 2 of 3 |
| KK-SII | Suzuka | 3 of 3 | 3 of 3 | 2 of 3 | 3 of 3 |
| KK-SII | Tsukuba | 0 of 3 | 1 of 3 | 3 of 3 | 3 of 3 |

On the layout folds (the use read on the car's other layouts) S1 is
lower in 7 of 10 cells and the braking nearer the logged lap's in 10 of
10; the worst tyre's heat with the pace carried is 3.0 % against 3.7 %
in the median (lower in 4 of 10, the differences 0.1–0.6 points), and at
the lap's own time 2.1 % either way. The held shape against the diamond:
S1 lower in 9 of 10, the braking nearer in 9 of 10 (with the limit's
shape it was worse in 9 of 10, C6). The verdict per car, reported: the
Inferno 86 inside on 30 of 31 days (29), the formula cars on every day.
The low-power control keeps the fold's reading of the driver (a wrong
power is planted, not another driver: the level in g at the reference
pace is held, `convene(..., use=)`).

**What the committee tested it against** (its own prototype, the level
and shape applied after the convention; the counts are of the same 76
cells):

- *Chance.* The count moves by 2.0–2.7 cells under resampling of laps
  and days (5–95 %: 27–36 on the old default); the difference of two
  versions on the same laps has an sd of 1.7–3.9. A gain under about 5
  cells is not told from noise. Step 1 is +22 today and +19.5 ± 3.7
  resampled, above zero in every resample; its own count resampled is 51
  (46–56), so 56 is the top of its range.
- *A placebo.* A 40 m blur of the old trace moved 4 m earlier scores 44
  (braking 14 of 19) with no change to the car. Step 1 beats it by 12
  (above zero in 99 %), equally at the three levels, passes 15 cells it
  does not, and the same blur on step 1's own traces loses 11.
- *Planted wrong cars on the changed model.* Power × 0.93: 35, × 1.07:
  32, × 0.85: 1, another car: 7 (by layout folds 32, 30, 3, 7).
- *The level planted wrong.* × 0.9: 46; × 1.1: 51. The count tells a
  level too low, not one too high; a plain scale of `mu_b` scores 54 at
  × 0.77 and 51 at × 0.84 and cannot tell them apart, which is why the
  level is read from the logs and not chosen by the count.
- *Layout folds*: 54 against 31.
- *The knob stays on braking.* The level fixed in g (not multiplied by
  u) scores 48: 0 won, 8 lost, seven at P90 and P95. In the logs the
  straight-line level rises 0.9 / 1.3 / 2.0 % per 1 % of u within a day.
- *Heat at a layout the car has not run*: 5.3 % with the use applied
  after the convention (u then sits 1.6–3.4 % high), 2.9 % with the
  convention set under it; hence the order above.
- *The shape's part is not shown.* The level alone scores 53, the level
  and shape 56: inside chance.

**What step 1 leaves wrong.** The simulated car is 0.5–0.85 m/s slow over
the logged braking nodes (it was 0.2–0.45 m/s fast) and 0.06–0.10 g soft
where the driver slows by 0.6 g or more: one level cannot be right on
the hard stops and slow enough into the corner. It loses 0.15–0.48 s a
lap through the braking nodes and takes it back in the corners: u is up
3 % and the Inferno 86's apexes are over by 0.35–1.0 m/s at every level
(0 of 7 apex cells, from 2 of 7). A level read as the one that sheds
each stop's speed in the stop's own distance passes the same 56 cells
and leaves the apexes where they are.

**Step 2, open (issue #313): apex speed at a given lap time.** Of the 20 cells still
failing 11 are apex cells, and it is the only area with more cells in
reach than the noise (the logged lap's own apex windows put in: +11 apex
and +3 speed cells). What was measured:

- *(Qualified 2026-11-22: a slope against lap time within a day can be
  made by selection alone, see "Withdrawn as a reading of the knob"
  below; this bullet and the next record what the logs hold, not how
  pace moves the lap.)* Within a day, per 1 % of lap time the logged apex speed rises
  0.28–0.37 m/s and the simulated 0.34–0.41; the logged top speed
  0.21–0.32 and the simulated 0.20–0.23: the knob puts too much of a
  faster lap into the corners.
- On the formula cars the apex error follows corner speed: 0.5–0.9 m/s
  over under 22 m/s, 0.7 m/s under above 30 m/s (1.2–1.6 m/s between the
  two, stable across days, levels and cars, untouched by step 1). On the
  Inferno 86 after step 1 it is a level: every class over by 0.2–0.5 m/s.
- Tried and not taken: grip rising with speed (0.6–1.2 standard errors
  in the logs; 53 against 56 on step 1, its reading swinging between
  layouts); curvature from the car's own fast laps (−10); the knob split
  between braking and cornering in a measured ratio (53); the knob on the
  exit drive.
- Four Suzuka corners where the cars hold 1.40–1.61 against a level of
  1.32–1.33 are not missing bank (the accelerometer reads at most
  0.023 rad more lean than the elevation model, where 0.03–0.10 rad would
  be needed): at two of them the lap's own curvature at its minimum
  speed is 9–13 % under the track's peak. A track matter (the line
  through linked corners), not a car term.

**The pace knob is the driver's (owner's decision, 2026-11-22).** "The
pace knob shouldn't change the grip, it should change how we use the
grip ... spending less time at the max"; "pace knob should be a driver
knob". Today u is a scale on `mu_y` and `mu_b`: a slower lap is a car
with less grip driven at its limit at every apex and through every stop.
It does two jobs at once, the grip of the day and the condition (the
track, the tyres, the water) and how hard the lap was driven, and a wet
lap differs from a dry one by that scale and nothing else
(`wet.grip_scale` multiplies the same two numbers, so it only moves
where u = 1 sits). The decision: grip is the car's, the day's and the
condition's; the lap's pace is what the driver does with it. Not built;
its form is #313's to find, and how the day's grip is told from the
day's driving by a lap time is open.

**The car's limit is its best moments put together (owner's decision,
2026-11-22).** A lap time and a speed trace fix only the product of the
car's limit and the driver's use of it, so the true limit cannot be read
from one driver's laps (the driver committee's planted test: a driver
who never reaches the limit and a weaker car driven to it read back as
the same car). The owner: "while we cannot find the true limit, we can
certainly use momentary max performances to derive a limit that is
higher than has ever been attained so far". The car model therefore
holds the best the car has been shown to do in each respect, each taken
where it was best (cornering, braking, the two together, drive under
lateral load), so that the whole is above anything one lap has reached;
it is a floor under the true limit and is labelled so. The driver model
is every way a lap falls short of it, and the lap with no shortfall (the
"max driver", #263) is by definition faster than any logged lap. Not
built. The committee's reading of it (`docs/lapsim_driver_model.md`,
clause 1): per corner the expected best of m passes (m the owner's
choice; 20 by default) of the lateral over the tyres' load, on the mean
of the two readings of each pass with the sensors' noise taken out, the
median over corners: 1.278 / 1.539 / 1.492 at m = 20 (Inferno 86 /
KK-SII / KK-F), where the convention was 1.160 / 1.318 / 1.372; the lap with every use at 1 is
faster than the best logged lap in every cell (thinnest the Inferno 86
at Fuji, 0.80 %). A maximum or a 99th percentile rises as days are
added (2–4 %) and reads the sensor; a 90th percentile reads the driver.

**The owner's decisions on the committee's design (2026-11-23; #327,
`docs/lapsim_driver_model.md`).** Seven were asked. (1) The uniform
knob, one use of the limit the same everywhere, is accepted now; a
direction by phase or corner is added only if the stint-level test finds
one. (2) m = 20, with the levels at 10 and 50 reported beside it. (3) A
dry day's level (about 2 %) is the driver's and goes into the knob; the
air and the powertrain of the day stay the car's. (4) The drive out of a
corner goes through the driven axle on all three cars; the committee's
default was the Inferno 86 only, because the formula cars' reading is
confounded, and that evidence stands as recorded. (5) The check on the
car is the max driver against the best logged lap; the sum of the best
segments is a diagnostic beside it. (6) A wet day's water level is
solved from the target lap time with the driver at their typical wet
use; there is no wetness input. (7) A target outside the logged range of
pace is answered and flagged, not refused. Its first step (the car's
limit, the knob as the driver's use of it, the max driver) was built
the same day: C9 below. Open and not the owner's: which curvature the limit is read
on (the car's own path against the track's line; the gap is larger than
the max driver's margins) and the plant of the limit's statistic, which
comes before the first step is accepted.

*What a faster lap is made of* (scratch, 2026-11-22; 1,124 dry build
laps on 63 days; the within-day slope per 1 % of pace, that is a lap
time 1 % shorter, ± the standard error over resampled days; one function
reads the logged speed and the simulated speed of the same lap at the
track's nodes, in the tyres' frame):

| per 1 % of pace within a day | Inferno 86 | KK-SII | KK-F | simulated (the knob) |
|---|---|---|---|---|
| u | | | | +3.2 to +3.3 % |
| lateral held, 95th percentile of the lap | +1.6 ± 0.2 % | +1.5 ± 0.2 % | +2.0 ± 0.3 % | +3.2 % |
| lateral at the apexes | +2.0 ± 0.2 % | +1.8 ± 0.2 % | +1.8 ± 0.1 % | +2.8 to +2.9 % |
| the lap's own peak lateral per corner over the corner's p90 (apex table), median corner | +1.2 ± 0.2 % | +1.4 ± 0.2 % | +1.3 ± 0.4 % | |
| the same, the lap's best corners | +1.2 ± 0.2 % | +1.1 ± 0.2 % | +0.1 ± 0.7 % | |
| corners within 5 % of their own p90 (40 in 100 on the median lap) | +8.1 ± 1.0 in 100 | +8.7 ± 1.7 | +8.0 ± 3.4 | every corner, always |
| braking force, 95th percentile of the braking nodes | +1.2 ± 0.6 % | +3.8 ± 0.7 % | +3.1 ± 0.4 % | +3.1 to +3.2 % |
| lap time within 15 % of the usual peaks (a third of the lap) | +3.1 ± 0.1 points | +2.7 ± 0.2 | +3.4 ± 0.2 | +5.0 to +6.1 |
| apex speed | +1.07 ± 0.08 % | +1.18 ± 0.08 % | +1.12 ± 0.09 % | +1.29 to +1.41 % |
| top speed | +0.66 ± 0.12 % | +0.28 ± 0.06 % | +0.40 ± 0.16 % | +0.17 to +0.49 % |

A single scratch reading, in sample, re-checked the same day by two
members of the driver committee: on all laps it reproduces; on the laps
within 1 % of the day's best the lateral shortfall holds on the Inferno
86 (+1.8 ± 0.4 % against +3.2 %), is weaker on the KK-SII (+2.1 ± 0.6 %)
and inside noise on the KK-F (+2.4 ± 0.6 %), and the apex-speed row's
excess is gone, so that part is carried by the slower laps. "Corners at
their own maximum" follows by arithmetic from a uniform shift of a 6 %
corner-to-corner scatter and is no evidence of its own.

**Withdrawn as a reading of the knob (the driver committee's selection
test, 2026-11-22; `docs/lapsim_driver_model.md`).** Laps simulated with
no knob at all, only the measured corner-to-corner scatter, give
within-day slopes of the same size (lateral at the apexes +2.0 to +2.4 %
per 1 % of pace against 2.05 logged on the Inferno 86 at Sodegaura), and
a direction planted in the knob is not read back by such slopes: taking
the faster laps of a day takes the laps whose corners happened to go
well. The table above is what the logs hold; it does not show how the
driver's pace moves the lap, and the paragraph below is kept as it was
written, not as a finding. What scatter does not make is the braking
row: the Inferno 86's peak braking hardly rises with pace (ABS), the
formula cars' rises more than a uniform knob gives.

Both pictures are in it. The peaks the driver reaches do rise with pace,
by about half of what the knob gives them (1.2–2.0 % against 3.2 %; the
formula cars' braking peaks by all of it, the Inferno 86's by a third);
the rest of a quicker lap is more corners taken at their maximum (8 more
in 100) and more speed down the straights than the knob has. Whether the
part of the peaks that moves is the grip of that hour or the driver's
commitment a lap time cannot say. A use that is one share of the limit
everywhere is the knob's arithmetic under another name: a driver knob
differs only where the use is uneven round the lap.

*Wet laps, scored once* (scratch, 2026-11-22; the gates take dry laps
only). The car and the braking use read from the dry laps, 134 wet build
laps each at its own lap time, all laps of a cell together and the car
not held out by day, so not the gate's number. The three cells with two
wet days or more pass 11 of 12 measures against another wet day's lap (9
of 12 braked at the limit), but that yardstick is loose (S1 between two
wet days 1.4–2.1 m/s). S1 is 1.6–1.8 m/s on the Inferno 86 (1.0–1.2 on
its dry laps) and 1.17 on the KK-F at Tsukuba (1.02). The braking use
does not carry to the wet on the Inferno 86: read from its wet laps as
from its dry ones the share is 1.33 times the dry one (1.30 Sodegaura,
1.28 Tsukuba), 1.04 on the KK-SII and 0.79 on the KK-F, and the shape
held is rounder on the formula cars (0.60–0.95 against 0.30–0.54 at
Tsukuba). One wet scale on cornering and braking alike is wrong per car.
Re-checked in g by a member of the driver committee (each wet day's p90
against the car's dry p90 at the layout, drag out): the direction holds
and the size is smaller: lateral 0.77–0.88 and straight-line braking
0.82–0.94 on the Inferno 86 (braking above lateral on 5 of 6 days),
0.77–0.88 and 0.62–0.96 on the KK-F (below on 4 of 5), 0.70–0.78 and
0.73–0.79 on the KK-SII; by brake pressure against deceleration the
limit moved on one of the Inferno 86's wet days and the driver's use on
three.
Tsukuba's wet track in place of the dry one changes little (91 laps): S1
0.03–0.05 m/s lower, the ends of the straights 0.06–0.21 m/s higher, u
up 2.0–2.5 %, the apex errors and the braking shares as they were.

**Step 3: the Inferno 86 is stronger in a straight line from 2025-11.**
The day's full-throttle level over the first engine's curve at equal rpm
(`car.day_levels`, in standard air) is 0.92–1.02 on the 18 days to
2025-10-25 (median 0.993) and 1.01–1.07 on the 7 days from 2025-11-01
(median 1.049): +4 to +7 % between 5,000 and 7,000 rpm and nothing from
7,250 rpm; +5 % in 2nd and 3rd, +8 % in 4th, +10 % in 5th; speed per rpm
in each gear unchanged (not tyre size or final drive). On top of step 1
no reading of it wins a cell beyond chance (the day's own level 55, a
level against temperature from other days 55, a period 54) and none
repairs an apex cell, so it is not a step for the count; it is a
powertrain period if the cause is the car's. The owner knows of no
engine change (only the shift light, 2026-11-21). What the raw logs say
(`IntakeAirT`, `AmbientTemp`, `MAP`, `Baro`, `Lambda`, `ECT`, `OilTemp`
are on every Inferno 86 log; read at full throttle between 5,000 and
7,000 rpm, 168 logs):

- The intake air is the ambient air: it reads 0–2 °C above the car's own
  ambient sensor at full throttle on every day. There is no heat soak
  for a correction to use.
- The level falls with temperature beyond the SAE correction: −2.7 ± 0.6
  % per 10 °C alone (scatter 3.6 → 2.6 %), a step alone +6.1 ± 1.1 %
  (2.4 %), both together +4.1 ± 1.4 % and −1.5 ± 0.6 % per 10 °C (2.2 %).
  The days from 2025-11 are all cold (−5.5 to 9.5 °C), but the cold days
  before it (−2, 4 and 7 °C) read 0.99–1.02, so temperature alone does
  not give the step. The second engine alone: −4.0 ± 1.2 % per 10 °C
  (6 days).
- **The mixture changed in October 2025.** At full throttle the logged
  lambda is 0.76–0.81, different from day to day, on every day to
  2025-10-13 but one; on 2025-10-04 and from 2025-10-25 on every day it
  reads 0.750–0.757 (0.750 on 12 of 16 days), the second engine's days
  included: the channel at a floor or the fuelling on a fixed target. With the highest full-throttle rpm
  rising from 7,410 to 7,440–7,650 from 2025-10-04, that is the mark of
  an engine calibration change three weeks before the level steps.
  Manifold pressure over barometric pressure, coolant and oil
  temperature show no step.

Asked with these readings, the owner answered on 2026-11-22 that nothing
was changed on the car in that time: no calibration, no fuel. So no
powertrain period is made from 2025-11 (a car is not split without a
change to the car), and the two marks are read otherwise: the higher rpm
from 2025-10-04 is the shift light, which is the driver's, and a lambda
of 0.750 on 12 of 16 days is taken as the channel at its floor and not
as the fuelling. The step in the level stands as measured and
unexplained; part of it is the cold of those days (−1.5 ± 0.6 % per
10 °C beyond the SAE correction).

**Step 4: the onset stays off.** It is real in the logs (0.2–0.4 s, the
same on fast laps as on all) but on top of step 1 it loses 4 cells (the
tops are then 0.16–0.61 m/s short): most of what it repaired was the
late start that step 1 now gives.

**The test a further step is held to** (the skeptic's, adjusted in the
debate): the rule and the expected cells written before the run, the
value from the logs with the count at ± 10 % of it beside; day folds and
layout folds; the gain above zero in 95 % of day resamples (about +5
cells), or, for a step too small for the count to decide, its own
statistic named beforehand with the count not below zero; a measure down
by 2 or more needs a named cause; the placebo beaten, and a blur of the
candidate's own traces must not gain; the planted wrong cars at least 10
(power ± 7 %) and 20 (power × 0.85, another car) cells below; heat at an
unseen layout no higher in the median and no fold worse by more than a
point, measured with the convention set under the change; the reading
returns a planted value within 2 %; nothing indexed by layout, corner or
node and nothing lent by another car; the two forms of the solver
identical.

**Findings about the score** (not proposals: the owner has settled it).
About 66 is the ceiling and not 76: the median logged lap of a layout's
other days, with no car at all, scores 59–66. The braking measure rewards
anything that lowers u (power × 1.07 is nearer in 19 of 19 braking
cells). The apex measure is a signed mean over the apexes, so errors
cancel in it, and it does not tell a power error of 7–10 %. In 9 cells
the other-day yardstick is below the same-day one; 20 cells rest on two
days; one lap removed flips 18 cells. The fitted GPS offset of a day is
1.5–7 m (median 3 m) and costs no cell. The levels sort days (the
Inferno 86's fastest laps are its cold days) as much as the driver's
pace. Nothing published scores a simulation against the driver's own
day-to-day variation (outside seat; Fambro et al. 2000 has ordinary
drivers at about 75 % of the available friction in measured stops).

**Cost.** Gate C3 reads the use in each of its 73 folds, two solves of
every lap that is left: it takes about four minutes.

#### C8. What #205 still asked to be reported (2026-11-22)

Eight reported items of #205 were never built. The owner took them one by
one on 2026-11-22: three built, one moved, four struck.

| item | verdict | why |
|---|---|---|
| the per-corner residual table | built (gate C3) | it tells the track's error from the car's, which is #313's question |
| the era ladder | built (gate C2) | what changed on a car or in how it is driven shows as a step |
| the formula cars' lateral against another driver's lap | built (gate C2) | the one outside figure beside the car's cornering limit |
| the exceedance of the envelope | moved to #327 | there it is a bound: no use above 1 |
| simulated top speed against the speed trap | struck | the end-of-straight measure and the top-speed band of gate C3 do it, held out |
| the class's cross-circuit ratios and wet-over-dry lap time | struck | the review found it tells neither a wrong power nor a wrong car, and it needs record laps kept current |
| wind along the main straight | struck | the end-of-straight measure passes 16 of 19 cells and hourly wind is coarse |
| the masked-run comparison columns | struck | another day's logged lap, the same day's and the measured ceiling (about 66) are that yardstick now |

**The apex speed corner by corner** (`car_holdout.corner_table`,
`corner_agreement`; held out by day, the median over days of the day's
median lap; the simulated lap's lowest speed in ± 20 m of the apex less
the logged lap's). `S3` is the mean of these over a lap's corners, so
errors of opposite sign cancel there and stand here.

- **At Tsukuba the errors are the same on all three cars**: over its eight
  corners the correlation between two cars' errors is +0.87 (Inferno 86 and
  KK-F), +0.89 (Inferno 86 and KK-SII) and +0.89 (the two formula cars).
  The two hairpins and the first corner are over on the formula cars (+0.2
  to +0.9 m/s), the last corner under on every car (−1.05 to −1.34), the
  third under on every car (−0.70 to −1.18). An error that unlike cars
  share at the same corners is the track's: its bank or its line through
  the corner, not the car and not the driver.
- **At Suzuka the two formula cars agree** (+0.86 over 16 corners), which
  says less, the cars being alike: corner 7 is −2.57 and −3.13 m/s and
  corner 13 −1.49 and −2.59, in the linked corners the driver committee
  put down to the line.
- **At Fuji the Inferno 86 and the KK-F do not agree** (−0.27 over 10
  corners): what is wrong there is not the track's alone.
- The furthest single corner is the Inferno 86 at Motegi's corner 10,
  +2.72 m/s on two days.

**The era ladder** (`car_gate.era_ladder`): a car's days in order with the
powertrain, the tyre compound where the day's sessions are labelled, the
day's best build lap and its u, the day's full-throttle level and the
driver's shift point. The u of a day's best lap runs from 0.794 to 1.086
on the Inferno 86 over 32 days (0.794 and 0.881 are Tsukuba days of
2025), 0.956 to 1.075 on the KK-SII over 20 and 0.953 to 1.025 on the
KK-F over 13. On the Inferno 86 the days from 2025-11 at Fuji read
1.07–1.09 against 0.98–1.05 before, beside the level's step
(1.05–1.07 against 0.97–1.01): the ladder shows the step and, like
everything read from a lap time, cannot say whether a day's u is its
grip or its driving. The compound is labelled on 20 of the Inferno 86's
32 days and on almost none of the formula cars'.

**Another driver's lap in the class** (`car.CLASS_LAPS`,
`car_gate.class_lap_rows`): one published dry reference lap of a Super
FJ at Tsukuba from 2009, never fitted to.

| at Tsukuba | the other driver's Super FJ | the KK-SII |
|---|---|---|
| lap time | 59.9 s | 60.2 s (best dry build lap) |
| lateral | 1.51–1.54 g (95th percentile over 0.5 s) | 1.66 g (apex table: ground frame, the car's own line, the bank in it); `mu_y` by convention 1.318 |
| braking | 1.23 g | 1.25 g (90th percentile of the zones' peaks, drag in); `mu_b` 1.21 |

The lap time and the braking are the other driver's within 0.5 % and
2 %. The apex table's lateral is 8–10 % above theirs, which is the frame
(Tsukuba's hairpins lean 8 % into the turn) and the statistic (a 90th
percentile of per-lap peaks); the driver committee's reading of the
cornering limit in the tyres' frame (1.539 for the KK-SII at m = 20,
`docs/lapsim_driver_model.md`) sits inside the other driver's band.

#### C9. The car's limits and the driver's use of them (issue #312, 2026-11-23)

The build of the driver committee's design (`docs/lapsim_driver_model.md`)
as the owner ruled on it on 2026-11-23, in the stages issue #312 sets
out. Each stage's "must show" is recorded here.

**Stage 0, the baseline.** Gates C2 and C3 at `a2d9952`, and every day
fold's laps kept as solved (u and the speed at the track's nodes; 1,102
laps in nine car and layout cells): 56 of 76.

**Stage 1, the sample tables at version 5: columns added, none changed.**

- The apex table has `lat_tyre_g`, the lap's own Doppler lateral as what
  the tyres hold of it on the road's bank at the apex (`L·cos b − sin b`,
  b the lean into the corner, the bank's median over ± 10 m;
  `bank.to_tyres_frame`), and `load`, the load on the tyres there over
  the weight at the apex speed (`cos θ·cos b + V²(κ_v + |κ|·sin b) / g`,
  the same median; `bank.road_arrays`, the solver's own arrays). Lateral
  over load is the share of the grip a pass used.
- A braking zone has its *held part*: the fixes where the pressure is at
  80 % or more of the zone's highest. Over it `decel_held_g` is the mean
  deceleration (a level the car held, not a peak), `decel_held_accel_g`
  the same from the accelerometer's longitudinal as the sensor reads it,
  with `v_held_ms`, `lat_held_tyre_g` and `held_s`. They are the second
  reading and the held level that the braking limit of stage 8 is read
  from.
- Full-throttle and coast samples and braking zones have their lateral
  in the tyres' frame beside the ground-frame one (`lat_tyre_g`,
  `lat_peak_tyre_g`, `lat_mean_tyre_g`).

`k_corner_drag`, `brake_ellipse` and `mu_b` are still read from the
ground-frame columns. Reading them in the tyres' frame changes simulated
laps, so it is not part of a step whose test is that no lap changes; it
is stage 9, judged by the score. Must show, and does: gate C2's report
and the three car models are identical to the baseline's, byte for byte
but for the time of the fit, and gate C3's report is identical.

**Stage 2, the cornering limit's statistic** (`limits.py`; the driver
committee's clause 1.1, written out in issue #312 because the
committee's probes were scratch). A pass is one dry build lap through
one grip-limited corner. It has two readings of the lateral the tyres
hold over the load on them: the Doppler's (`lat_tyre_g / load`) and the
accelerometer's, put on the Doppler's scale for its session (an
accelerometer reads less than the tyres hold by the body's roll). The
pass's value is their geometric mean. Per corner, every dry day pooled,
the level is the expected largest of m passes drawn from its n without
replacement, `Σ w_i·x_(i)` with `w_i = C(i−1, m−1) / C(n, m)`
(`limits.expected_best`); the car's level is the median over the
corners with 2m passes or more, every layout together. m = 20 by the
owner's decision. The model keeps it as `grip.mu_y_limit`, with its
workings in `meta.limit`; the solver does not use it until stage 4, and
every fold of gate C3 reads its own (the fit is handed the tables
without the fold).

The committee's uncorrected figures come back exactly from the
package's tables (1.261 / 1.481 / 1.525 at m = 10, 1.287 / 1.496 /
1.549 at m = 20, 1.304 / 1.516 / 1.602 at m = 50; Inferno 86 / KK-F /
KK-SII), and so does their table of which corners support which m. The
noise correction is this build's own (the committee's was applied by
hand: +0.73 / +0.28 / +0.66 % at m = 20): the independent noise left in
the two-reading mean has a quarter of the variance of the two readings'
difference within a corner and day, and a corner's passes are drawn
towards its mean by the noise-free share of their spread. It takes
0.34 / 0.21 / 0.55 % off. As gate C2 reports it:

| | Inferno 86 | KK-SII | KK-F |
|---|---|---|---|
| level, m = 20 [5 %, 95 % of a day bootstrap] | 1.283 [1.261, 1.285] | 1.541 [1.514, 1.551] | 1.493 [1.475, 1.511] |
| m = 10 / m = 50 | 1.257 / 1.299 | 1.516 / 1.593 | 1.478 / 1.512 |
| supporting corners at m = 20 | 15 (Fuji, Sodegaura) | 14 (Suzuka, Tsukuba) | 5 (Tsukuba) |
| layouts that borrow the level | Motegi, Tsukuba | Fuji | Fuji, Suzuka |
| the same, each sensor read where the other peaks | 1.249 | 1.524 | 1.466 |
| on the Doppler alone, uncorrected | 1.312 | 1.584 | 1.614 |
| the convention's `mu_y` (what the solver uses today) | 1.160 | 1.318 | 1.372 |

A car or a fold with no corner that supports m on two readings is read
on the Doppler alone and says so (`one_sensor`); that reads 2–3 % high
where both exist, and 8 % on the KK-F, whose Doppler-only corners are at
Fuji and Suzuka on its other loggers. The KK-F's fold without Tsukuba is
such a fold.

**Stage 3, the statistic on a planted car** (`car_gate.plant_limit`,
gate C2; the committee's clause 5.4: "nobody has run that plant for
this statistic; it comes before step 1 is accepted"). The rule was
written on #312 before the run: on every top, the level read is within
1 % of the expected best of m of the planted passes and no more than
1 % above the planted limit. **On the plant as built it fails, on every
car and every top, and the statistic was not changed to pass; the
plant itself then turned out to overstate the effect (below).**

A car with the logged car's limit is driven for 12 days of 10 laps on
the layout most of its supporting corners are at, each corner pass at
its own use of the limit (the lap is driven at the limit on the track
with that corner's curvature divided by its use, which is the same
lap), through the package's own table builder, with noise added to both
readings' series *before* the corner's largest value is taken. The
passes' spread is the car's own, and the noise is set so that the
statistic finds in the plant the noise it finds in the logs (0.013 /
0.016 / 0.010 in the two-reading mean, which takes 0.032 / 0.046 /
0.031 g per reading, smoothed over 0.5 s as the readings are). Against
the expected best of 20 of the planted passes:

| top of a corner's passes | Inferno 86 | KK-SII | KK-F |
|---|---|---|---|
| bell-shaped: the committee's reading (each sensor's own peak) | +1.8 % | +2.0 % | +2.2 % |
| capped at the limit | +3.5 % | +3.6 % | +2.7 % |
| bell-shaped with slow mistakes in 2 % of passes | +2.8 % | +3.0 % | +2.2 % |
| bell-shaped: each sensor read where the other peaks | −0.9 % | −0.7 % | −0.1 % |
| capped, read so | +1.5 % | +1.0 % | +1.0 % |
| with mistakes, read so | −0.6 % | −0.7 % | −0.4 % |
| the table builder alone, no noise | −0.5 to +0.6 % | −0.5 to +0.3 % | −0.2 to +0.2 % |

The cause is one the limits seat named and did not measure: a reading
is the *largest* value of a corner's fixes, and the largest of noisy
fixes is raised by the noise, on both sensors alike. That is a shift of
every pass, not a spread, so no correction built on the spread of the
two readings' difference can see it; the correction takes off the 0.3
to 0.6 % it was built for and leaves the rest. The logs show the same
thing directly: over 5,429 passes a sensor's own peak sits above its
value at the fix where the *other* sensor peaks by 3.9 % and 3.1 %
(Doppler and accelerometer, Inferno 86), 3.0 % and 2.4 % (KK-SII) and
1.4 % and 1.1 % (KK-F), and the two sensors' difference is fix-to-fix
(sd 2.1–3.1 % on a corner's plateau, 0.91–0.94 correlated from one fix
to the next) far more than pass-to-pass (0.6–1.1 %).

Reading each sensor where the other one peaks is a choice the sensor's
own noise has no part in. It is kept in the apex table (tables
version 6: `lat_tyre_cross_g`, `lat_accel_cross_g`) and reported beside
the committee's reading by `limits.car_limit(..., cross=True)`. On the
plant it is within 1 % on a bell-shaped top and with mistakes, and
1.0–1.5 % high on a capped one (the mean of two readings still has
noise, and a top with no spread of its own has nothing to hide it in);
it reads low where the two peaks are apart on a corner that is not
flat. An error the two readings share by day and corner (2.5 % planted,
one draw of twelve days) adds 1.3–2.9 % to either reading on the
Inferno 86 and the KK-F and 0.3–1.4 % on the KK-SII, as the design says
it must: no reading removes it.

**The plant overstates the effect, by its own second number.** Its
noise was set to match one thing the logs show (the noise the statistic
finds in the two-reading mean). It then predicts a second: how far the
level by own peaks sits above the level read where the other sensor
peaks. On the planted layout's own supporting corners:

| own peak over cross, m = 20 | Inferno 86 (Fuji) | KK-SII (Suzuka) | KK-F (Tsukuba) |
|---|---|---|---|
| the plant, bell-shaped top | +2.7 % | +2.7 % | +2.4 % |
| the logs, the car's level on those corners | +1.7 % | +1.1 % | +1.8 % |
| the logs, median over those corners | +1.1 % | +0.9 % | +0.7 % |
| the logs, the car's level over every layout | +2.7 % | +1.1 % | +1.8 % |

(Run again at the last commit, the planted car now driving through its
driven axle and the noise drawn afresh, the figures are of the same
kind and not the same: own peaks +0.9 to +3.1 % over the three tops and
cars, the cross reading −1.7 to +0.8 %, and the gap 2.1 / 3.0 / 2.8 %
against the logs' 1.7 / 1.1 / 1.8 %. One draw of twelve days moves a
figure by about a point.)

The plant's gap is 1.3 to 3 times the logs'. The likely reason is the
planted pass itself: a point mass at the limit holds the same lateral
through the whole of a corner, so its largest value is taken over many
fixes at one level, where a driven pass rises to a peak and falls
away. So the plant shows the mechanism and its sign, and its sizes
(+1.8 to +3.6 %) are upper figures, not the bias of the logged cars.

What the logs themselves bound. The truth lies between the two
readings: a sensor's own peak is raised by its noise, and its value
where the other peaks is lowered by however far apart the peaks are.
The committee's reading is therefore high by something between nothing
and the gap: at most 2.7 % on the Inferno 86, 1.1 % on the KK-SII and
1.8 % on the KK-F, and by the plant's split (about two thirds of its
gap above the truth, one third below) more likely 1 to 2 % on the
Inferno 86 and the KK-F and under 1 % on the KK-SII. Which reading is
the car's limit is the owner's to rule: 1.283 / 1.541 / 1.493 by own
peaks, 1.249 / 1.524 / 1.466 read where the other sensor peaks. The
default stays the committee's until then. Nothing in the stages that
follow depends on the choice except the max driver's margins
(stage 6): with the knob solved from a lap time, a simulated lap is the
same lap at any level of the limit. The rule written before the run
fails on the plant as built; because the plant does not reproduce the
logs' own gap, gate C2 reports the item and does not fail on it: a
criterion added after the run (the item is gated only where the plant's
gap is 0.8 to 1.25 of the logs'), by this build and not by the owner.

**Stage 4, `mu_y` is the car's limit and the knob is the driver's use
of it** (`car_gate.set_limit`; the committee's clauses 3.1 and 5.1; the
owner's ruling of 2026-11-22 that the pace knob is the driver's and not
a scale on the grip). The convention for `mu_y` is gone from the cars
the gates set: `mu_y` is `grip.mu_y_limit`, which no pace moves, and
the solver's knob u is the share of it a lap uses, in cornering and in
braking alike (q = −ln u). The kernel is untouched. A layout's
reference is the median use of its day-best laps, and the median of
those over the car's layouts is the car's *typical use*
(`meta.use_ref`, `car_gate.typical_use`): it is what is carried to a
layout the car has not run, where the convention carried "u = 1".
Reports keep `u` as the pace against the typical use, so that their
columns read as before, and gain `use` and `q`.

| | Inferno 86 | KK-SII | KK-F |
|---|---|---|---|
| the limit, `mu_y` | 1.283 | 1.541 | 1.493 |
| typical use (q) | 0.904 (0.101) | 0.855 (0.156) | 0.919 (0.085) |
| limit × typical use: the convention's `mu_y` | 1.160 | 1.318 | 1.372 |
| the driver's braking as a share of `mu_b` at a use of 1 | 0.885 | 0.989 | 0.914 |
| the same at the typical use: the share as it was read | 0.800 | 0.846 | 0.840 |
| use on held-out laps, P05 / median / P95 / highest (day folds) | 0.80 / 0.86 / 0.94 / 0.99 | 0.78 / 0.83 / 0.89 / 0.92 | 0.83 / 0.88 / 0.92 / 0.95 |

The braking use was read as a share of `mu_b` times the knob, so it
rises as `mu_y` rises to the limit and every lap stays the lap it was
(`car_gate._with_mu_y`); it is not read again in this stage. Must show,
and does, against the baseline's 1,102 held-out laps solved in nine car
and layout cells:

- every lap has the baseline's speed at every node within 0.00004 m/s
  (the largest difference over all laps and nodes; identical to the
  solver's tolerance, not bitwise: the kernel is handed a different
  `mu_y` and solves a different u for the same product);
- limit × use equals the baseline's `mu_y` × u, and the braking level
  in g equals the baseline's, to six figures on every lap;
- the score is 56 of 76, the same verdict in every one of the 108
  cells, the largest change in a cell's figure 4e-7;
- worst-tyre heat with the pace carried to a layout the car has not run
  is 3.0 %, and 2.1 % at the lap's own time, as before;
- no held-out lap needs a use above 1 (0 of 2,231 laps over the day and
  layout folds; the highest is 0.998, the Inferno 86 on a layout fold).
  So nothing was built for q below 0: the kernel's knob scales
  cornering and braking together, a cap on one would need a change to
  it, and at this stage no logged lap asks for it (with the driven axle
  in, stage 7, five laps of the Inferno 86 do: the end of C9).

Two things the stage showed that the design did not say.

*A car braked "at its limit" is not the same car under the two
meanings.* Under the convention the reference lap had u = 1, so a car
with no driver's braking use braked at `mu_b` at the reference pace.
With `mu_y` at the limit the same car has a use of 0.86–0.92 on that
lap and brakes at `mu_b` times it: the uniform knob lowers the braking
with the cornering, which is a braking use by another name. The
comparison arm "braked at the car's limit" of gate C3 would have read
49 of 76 for it where it read 34. The arm is kept as it was (`mu_b` at
the typical use: `car_holdout.convene` holds a share of 1 at the
typical use, which is a share above 1 at a use of 1), so that "what the
driver's braking use is worth" still compares the same two cars. The
max driver (stage 6) is the other thing "at the limit" can mean: every
use at 1.

*The KK-SII has no room left in its braking.* Its share is 0.989 of
`mu_b` at a use of 1: a lap at the car's cornering limit would brake at
the car's braking limit, with the driver's held shape. The committee's
"knob's end lap" (clause 3.1: the braking use rising to 1 below q = 0)
has under 2 % of room on this car, 9 % on the KK-F and 12 % on the
Inferno 86, until stage 8 reads `mu_b` as the best of m stops.

**Stage 5, the driver's braking use read against the limit**
(`car_gate.at_limit`; the committee's clause 2.1). The car is put at
its limit first and the use is read there, as a share of the limit the
laps are driven on; there is no reading "before the convention" any
more, because there is no convention to move it afterwards. It comes
out 0.886 / 0.990 / 0.914 of `mu_b` at a use of 1 (0.885 / 0.989 /
0.914 by stage 4's arithmetic), the held shape 0.59 / 0.28 / 0.47 as
before, and read back from laps simulated with it within 0.4 %. Against
the baseline: the braking level in g within 0.17 % on every lap (1.5 %
allowed by clause 5.1), node speeds within 0.045 m/s (0.002–0.009 rms),
56 of 76 with the same verdict in every cell, heat 3.0 % carried and
2.1 % at the lap's own time.

**Stage 6, the max driver against the best logged lap**
(`car_gate.max_driver_rows`, gate C2; the committee's clause 1.8, "the
owner's plain number"). The lap with every use at 1 and the driver's
habits off: the car at its cornering limit and at `mu_b` on the limit's
own shape, with the powertrain, shift point, gears and air of the best
lap's day, on the track's line. Lap time against the car's best logged
dry build lap:

| car, layout | m = 10 | m = 20 | m = 50 | m = 20, 5 % end | m = 20, read where the other sensor peaks |
|---|---|---|---|---|---|
| Inferno 86, Fuji | −0.42 % | −0.90 % | −1.19 % | −0.48 % | −0.28 % |
| Inferno 86, Motegi (borrowed) | −1.44 | −1.86 | −2.12 | −1.50 | −1.31 |
| Inferno 86, Sodegaura | −2.02 | −2.60 | −2.94 | −2.10 | −1.84 |
| Inferno 86, Tsukuba (borrowed) | −2.90 | −3.38 | −3.68 | −2.96 | −2.74 |
| KK-SII, Fuji (borrowed) | −4.48 | −4.82 | −5.52 | −4.45 | −4.59 |
| KK-SII, Suzuka | −2.09 | −2.46 | −3.18 | −2.06 | −2.21 |
| KK-SII, Tsukuba | −3.80 | −4.22 | −5.08 | −3.77 | −3.93 |
| KK-F, Fuji (borrowed) | −2.94 | −3.16 | −3.43 | −2.89 | −2.76 |
| KK-F, Suzuka (borrowed) | −2.00 | −2.23 | −2.54 | −1.93 | −1.79 |
| KK-F, Tsukuba | −2.27 | −2.53 | −2.87 | −2.20 | −2.04 |

Faster in ten of ten cells at every m, at the 5 % end of the day
bootstrap, and with the limit read where the other sensor peaks: at
this stage the owner's check passes on either reading (it no longer
does on the cross reading once stages 7 and 9 are in: the table at the
end of C9). The thin
cell is the committee's, the Inferno 86 at Fuji: 0.90 % at m = 20 (the
committee had 0.80 %), 0.48 % at the 5 % end (0.44 %), and 0.28 % on the
cross reading, which is the one the plant of stage 3 reads as the
nearer to a planted car. With the car cornering at the driver's typical
use and braking at the limit the same lap is 1.4 % *slower* than the
best lap for the Inferno 86 at Fuji and the KK-SII at Suzuka, and from
1.4 % faster to 0.3 % slower in the other eight cells: the margin is
the cornering limit's. Not built: the sum of the expected
best of m segment times (the diagnostic beside it), and the level read
on the track's line instead of the car's own path (clause 1.9, open:
the track's curvature reads 3–4 % under the car's own at the formula
cars' apexes, so their margins above are larger than that reading would
give).

**The limit handed to every fold at 1.1 and at 0.9 of itself** (the
check stage 4 promised, run after stage 5 so that the use is read again
each time): no cell of the 108 changes its verdict either way; node
speeds stay within 0.06 and 0.10 m/s (under 0.01 rms) and the braking
level in g within 0.17 and 0.24 %. At 1.1 the KK-SII's braking share
reads 1.08 of `mu_b` and the KK-F's 1.00; at 0.9 some laps solve to a
use above 1. A lap is the same lap at any level of the limit: the level
decides only what the numbers are called and how far the max driver is
from them.

**Stage 7, drive out of a corner goes through the driven axle**
(`SolverCar.drive_share`, `solver.DRIVEN_AXLE_DEFAULT`; the committee's
clause 1.3; on for all three cars by the owner's decision of 2026-11-23,
where the committee's default was the Inferno 86 only). The grip left
for drive is `r·√((μ_y·G)² − lat²)`, r the rear's share of the static
weight from production's facts (`car.drive_share`: 0.47 on the Inferno
86, 0.58 on the formula cars; a stated input, not weighed, not fitted;
it leaves out the load that moves rearwards under drive). The share
rides in the last entry of the car's power table, so that neither form
of the solver gains an argument; with a share of 1 both are the
arithmetic they were, and with the term switched off every one of the
1,102 held-out laps is identical to stage 5's to the last bit (largest
difference 0.0 at every node). The two forms agree to the last bit with
it on. The driver's use and the typical use are read with the term on.

| | with the driven axle | the whole car's grip |
|---|---|---|
| score (day folds) | **59 of 76** | 56 |
| by car: Inferno 86 / KK-SII / KK-F | 19 / 19 / 21 | 19 / 18 / 19 |
| by measure: speed / end of straights / apex / braking | 17 / 16 / 8 / 18 | 14 / 16 / 8 / 18 |
| layout folds: S1 lower with it | 8 of 10 cells | |
| worst-tyre heat, pace carried to an unseen layout | 2.6 % | 3.0 % |
| the same at the lap's own lap time | 1.9 % | 2.1 % |
| typical use (Inferno 86 / KK-SII / KK-F) | 0.916 / 0.877 / 0.935 | 0.904 / 0.855 / 0.919 |
| braking share of `mu_b` at a use of 1 | 0.87 / 0.97 / 0.90 | 0.886 / 0.990 / 0.914 |

It wins five cells and loses two: the KK-SII's speed round the lap at
Tsukuba at all three levels of pace and the KK-F's apex speed at Suzuka
at P75 and P90, for the KK-SII's apex speed at Tsukuba at P75 and P90.
Three cells is inside what the count moves by chance (about 5 are
needed), so the count alone does not carry it; the heat at a layout
the car has not run does fall (2.6 % against 3.0 %), and S1 is lower in
eight of ten layout folds. The Inferno 86, where the committee thought
the evidence strongest, changes no cell; the gains are on the formula
cars, where the committee would not have built it. The exits are slower
with it, so a lap at a given lap time takes more of the cornering limit:
the typical use rises by 1–2 points, three of the Inferno 86's 518 laps
on layout folds now solve to a use above 1 (the highest 1.007; none on
a day fold, where the highest is 1.000), and the max driver's margin at
Fuji falls to 0.67 % at m = 20 (0.24 % at the 5 % end, and 0.01 % with
the limit read where the other sensor peaks: level with the best lap).
Of the committee's test for a later term (clause 5.3) this run has the
day and layout folds, the cells won and lost per car and the heat at an
unseen layout; it has not the day-resampled gain, the blur placebo or
the planted wrong cars' distance, and it is on by the owner's decision,
not by that test.

**Stage 9, the fit reads its laterals in the tyres' frame**
(`car.TYRES_FRAME_READS`). The cost of cornering at full throttle, the
choice of the stops braked in a straight line and the braking limit's
shape are read from the tables' `*_tyre_g` columns, and the lateral
level the braking is set against is the apex table's in the same frame
(`meta.lat_level_braking`). The solver applies all three to what the
tyres hold, so that is the frame they are fitted in now.

| | Inferno 86 | KK-SII | KK-F |
|---|---|---|---|
| `k_corner_drag`, /g | 0.047 → 0.050 | 0.032 → 0.034 | 0.025 → 0.027 |
| `brake_ellipse` (the limit's shape) | 0.69 → 0.71 | 0.52 → 0.45 | 0.63 → 0.62 |
| `mu_b` | 0.954 → 0.953 | 1.212 → 1.210 | 1.246 → 1.246 |
| the lateral level the braking is read against | 1.339 → 1.293 | 1.668 → 1.622 | 1.698 → 1.645 |

Gate C3: 60 of 76 against stage 7's 59 (one cell won, the KK-SII's apex
speed at Suzuka at P75; none lost), worst-tyre heat 2.6 % carried and
1.9 % at the lap's own time, unchanged; recovery on simulated laps
passes as before. A change of one cell is nothing by the count; it is
kept because it is the frame the solver uses, and it costs nothing.

**Stage 8, the best braking the car has held** (`limits.braking_best`,
gate C2; the committee's clause 1.2), **built as a reading and not put
into `mu_b`.** Per stopping place (stops that begin within 60 m of each
other), every dry day pooled: the expected best of m stops of the
deceleration *held* over the part of the stop with the pressure at 80 %
or more of its highest (0.4 s or more, in a straight line), less the
drag, read by the GPS speed and by the accelerometer (put on the GPS's
scale per session), with the independent noise taken out; the median
over the places braked at the limit (within 15 % of the layout's
highest, as a corner is grip-limited) with 2m stops or more.

| | Inferno 86 | KK-SII | KK-F |
|---|---|---|---|
| best of 20 stops [5 %, 95 %] | 0.944 [0.905, 0.969] | 1.210 [1.198, 1.225] | 1.099 [1.042, 1.119] |
| places it stands on | 5 (Fuji, Sodegaura) | 4 (Suzuka, Tsukuba) | 3 (Tsukuba) |
| `mu_b` today (90th percentile of the stops' peaks) | 0.953 | 1.210 | 1.246 |
| the reading over `mu_b` | 0.99 | 1.00 | 0.88 |

The reading the committee asked for comes out at or under the number
it was to replace: a level held is under a peak, and the expected best
of twenty held levels is about what a 90th percentile of peaks already
was. On the KK-F, whose two readings exist at Tsukuba only, it is 12 %
under (its Fuji stops, on the GPS alone, hold 1.26–1.28). Putting it
into `mu_b` would *lower* the car's braking limit, which is not what
"the best the car has been shown to do" was for, and it would take the
KK-F's driver above the limit at a use of 1 (a share of 1.02). So
`mu_b` stays as it is and the reading is reported beside it
(`meta.braking_best`). It would not move a simulated lap either way:
the laps are braked as the driver brakes, and that level in g is read
from the logs whatever `mu_b` is called. What it would move is the max
driver, the comparison arm braked at the limit, and the room the knob's
end lap has. Whether the car's braking limit is this reading, today's
peaks, or per layout (the Inferno 86 holds about 0.92–0.95 at Sodegaura
and 1.03–1.05 at Fuji) is the owner's.

**Issue #312 after the nine stages.** Built: the tables in both frames,
the cornering limit and its two readings, the knob as the driver's use,
the uses read against the limit, the max driver, the driven axle on all
three cars, the fit in the tyres' frame, the braking reading. The score
went from 56 to 60 of 76 (the driven axle +3, the tyres' frame +1; the
limit, the knob and the uses changed no lap) and the heat at a layout
the car has not run from 3.0 % to 2.6 %. For the owner: which reading
is the cornering limit (own peaks or cross), and what the braking limit
is. Not built: the sum of the expected best of m segment times beside
the max driver; the level read on the track's line (clause 1.9); a cap
for a use above 1.

As the stages stand together (gates C2 and C3 at the last commit):

| | Inferno 86 | KK-SII | KK-F |
|---|---|---|---|
| the cornering limit `mu_y` (own peaks, m = 20) | 1.283 | 1.541 | 1.493 |
| typical use (q) | 0.918 (0.085) | 0.878 (0.130) | 0.936 (0.067) |
| the driver's braking share of `mu_b` at a use of 1 | 0.87 | 0.97 | 0.90 |
| drive share (the rear's) | 0.47 | 0.58 | 0.58 |

The max driver against the best logged lap, with the driven axle and
the tyres' frame in: faster in ten of ten cells at m = 20 and at its
5 % end, thinnest the Inferno 86 at Fuji at 0.60 % (0.17 % at the 5 %
end, 0.10 % at m = 10); with the limit read where the other sensor
peaks, nine of ten: at Fuji the Inferno 86's max driver is then 0.05 %
*slower* than its best lap (0.32 % slower at m = 10, 0.24 % faster at
m = 50), so the owner's check fails on that reading in that cell at the
m the owner ruled, and seven of that cell's 111 held-out laps would
solve to a use above 1 on it. The other nine cells are faster by 0.9 to
4.2 % on it. That is the constraint that separates the two readings:
"higher than has ever been attained" holds on own peaks and not on the
cross reading, while own peaks may be up to 2.7 % high on this car.
Five of the 2,231 held-out laps solve to a use above 1, all the Inferno
86's (three on layout folds, two on day folds; the highest 1.009): they
are driven that far above the limit, not capped.

**The owner's two rulings, 2026-11-24, and what closed the issue.**

1. *The cornering limit is read on each sensor's own peak* (the
   committee's reading; `grip.mu_y_limit` as built). It is the reading
   on which the max driver is faster than every best logged lap, which
   is the owner's own test of a limit; it may read high by up to the
   gap to the cross reading (2.7 / 1.1 / 1.8 %), and is labelled so.
   The cross reading stays in the tables and in gate C2 beside it.
2. *`mu_b` stays the 90th percentile of the stops' peaks.* The best of m
   stops of a held deceleration is reported beside it and not used: it
   would lower the braking limit. The braking shape is therefore not
   read by that statistic either.

Two things stage 6 had left were then built (tables version 7,
`limits.car_limit(..., track=True)`, `car_gate.best_segments`):

*The level read on the track's line* (the committee's clause 1.9, which
was open): the same statistic on the pass as the solver would have it,
the track's curvature at the apex at the lap's speed there, in the
tyres' frame; one reading, so nothing is taken out for noise. At m = 20
it reads 1.282 / 1.534 / 1.510 against 1.283 / 1.541 / 1.493 on the
car's own path: within 1.1 %, not the 3–4 % that a pass-by-pass
comparison of the two curvatures gave the committee (the expected best
of twenty passes is set by the top of a corner's passes, where the two
differ less than they do in the middle). The max driver at that level
is faster than the best logged lap in ten of ten cells, by 0.59 % for
the Inferno 86 at Fuji (0.60 % on the car's own path) and 2.0 to 4.3 %
on the formula cars. So the clause's worry does not arise, and the
level stays on the car's own path, where it has two readings.

*The lap stitched from its best parts* (the diagnostic of the owner's
ruling 5): the lap cut half-way between apexes, and per part the
expected best of 20 laps, as the limit takes the expected best of 20
passes; no noise is taken out of the timing. Against the fastest whole
lap it is 0.4 % faster for the Inferno 86 at Fuji and Tsukuba and 0.4 %
*slower* at Sodegaura (355 laps: the expected best of twenty is not the
best of all), 0.2 to 1.2 % faster for the KK-SII and 0.3 to 1.3 % for
the KK-F. The max driver is faster than the stitched lap in all nine
cells that have twenty laps (by 0.2 % for the Inferno 86 at Fuji, the
thinnest again, and by 0.7 to 3.4 % elsewhere).

Left out of #312, on purpose: a cap for a use above 1 (five laps of
2,231 reach 1.009; it needs a change to the kernel and belongs with the
calculator's contract, #206); the committee's full test for the driven
axle (it is on by the owner's decision); the braking shape by the best
of m stops (it goes with `mu_b`, which stays).

#### D1 (in part). The lap solver (`solver.py`, 2026-11-15)

C2's remaining clauses and all of C3 need `simulate_lap` and `solve_u`,
which #206 specifies under D1 and says can be built before C. This is
that part of D1 and no more: the limits and the integrator exactly as
#206 writes them (grip `G = u·w·g·N` with `N = cos θ + V²κ_v/g`; the
cornering limit in closed form; the braking diamond; drive as the lesser
of `P/V` and what is left of the grip; `dV/dt = a_tyre − c·V² − g·grade`;
trapezoidal steps in `V²` over the track's own 2 m nodes; a forward and a
backward pass from the slowest apex). The engine's power at a speed is
one number, or a curve against rpm read at the rpm of the gear the
driver is in (the lowest gear under the driver's shift point there,
`SolverCar.power_at`); the day's air scales the power and the drag.
`SimLap` is per step, not per
node: a step's mean speed times its time is its length, so distance and
the closed-lap identity hold to rounding. One term is added to #206's
drive rule: the engine gives `P/V − k·lat²` (`k_corner_drag`), on by
default since gate C3 let it in (`solver.CORNER_DRAG_DEFAULT`).

One thing differs from the text. The far end of a forward step is found
by fixed-point iteration, and at the edge of the envelope that iteration
does not converge: what is left of the grip grows as a square root of the
distance from the limit, and the iterate hops across the root (found on a
constant radius with drag, where the car must sit 0.002 m/s under the
limit so that the remaining grip carries the drag). Where four rounds do
not settle to 1e-7 m/s the step is solved by bisection.

Unit tests (the clauses of gate D1 that #206 states in closed form):
constant radius to 1e-9; power and drag on a straight against
`V³(s) = p/c + (V₀³ − p/c)·e^(−3cs)` within 0.01 m/s; braking with drag
against its closed form within 0.01 m/s and the braking point within one
node; terminal speed on a 5 % grade to 0.01 %; a crest against the closed
form of the cornering limit; distance to 1e-12; `Σ V·long·dt = 0`; lap
time and node speeds invariant to the start node; the mirrored track
mirrors the lateral exactly; halving the step moves the lap time by under
0.05 %; no step over the combined rule by 0.005; T(u) strictly falling;
`solve_u` round trip to 1e-6 and `no_root` for a lap the car cannot do.

**Shift dips, measured and not yet in the solver.** Across an upshift
the full-throttle table has a gap; the speed gained over it against what
the power seen just before and after would have given, as seconds of
drive lost (median; the same test over one second inside a gear reads
0.01–0.06 s on every car, which is the method's own bias):

| | 1→2 | 2→3 | 3→4 | 4→5 | 5→6 |
|---|---|---|---|---|---|
| Inferno 86, first engine | | 0.34 | 0.44 | 0.76 | 0.86 (21 shifts) |
| Inferno 86, second engine | | 0.30 | 0.41 | 0.78 | |
| KK-SII (gears as seen, lowest = 1) | 0.04 | 0.16 | 0.15 | | |
| KK-F | | 0.03 | 0.06 | 0.04 | 0.05 |

The Inferno 86 is 0.7–0.9 m/s short after every upshift and the solver,
which shifts in no time, is not: that is the size and the sign of its
bias over the top quarter of speeds in gate C3 (+0.56 m/s by day), and
the KK-F, which loses nothing, has none (+0.04). The KK-SII loses about
0.1 s in its two upper shifts.

**How D2 and D3 are judged (owner's decision, 2026-11-23).** #206
writes its two gates on logged laps as fixed limits on laps within
1.02 × the session best: D3 as lap-mean `V·|lat|` simulated over logged
within 0.03 and right turns minus left within ± 0.06, D2 as the wet
elasticities within 0.3. They take gate C3's form instead (C6, C7): the
same measures, each with a logged lap of another day beside it, at P75,
P90 and P95 of pace, a cell passing when the simulated lap is no
further from the logged lap than a logged lap of another day is. The
fixed limits are superseded as what passes; the unit tests of gate D1
keep their tolerances. Neither gate is built: D3 can be measured now,
and D2 is rewritten against the driver committee's wet clause once the
car's limits and the knob of that design are built (#312, #206).

**The outside reference of gate D1 (2026-11-24; `TestReferenceOvals`).**
Two stadium ovals with arcs of radius 160/π m and 400 m straights, one
with a curvature step (1,120 m) and one with 40 m clothoid ramps
(1,200 m), lapped by a point mass (μ_y 1.2, μ_x 1.0, 100 W/kg, c =
3.36e-4 /m, g = 9.81; on a flat road g enters only with the grip, so
the reference's g is a factor on μ in the tests). The lap times are an
ODE reference's and a closed form's, from #206's review.

| lap | reference | the solver, 2 m | 0.25 m |
|---|---|---|---|
| clothoid oval, ellipse both sides | 35.7254 s | +0.032 % | +0.0009 % |
| stepped oval, ellipse | 35.1056 s | −0.004 % | −0.0004 % |
| stepped oval, braking on the diamond | 35.1036 s | −0.0003 % (the rule as written) | +0.004 % |
| clothoid oval, braking on the diamond and the drive by power alone | 36.0337 s | | +0.013 % |
| clothoid oval, the rule as written (the drive within the circle of the grip) | none | 36.0887 s | 36.0787 s |

"Ellipse both sides" is braking and drive each within the ellipse with
μ_x along the car, which the solver has had exactly since the driven
axle went in: the braking shape at the ellipse and the drive at
μ_x / μ_y of what the cornering leaves (`brake_ellipse = 1`,
`drive_share = 1 / 1.2`). The error falls by a factor of three with
each halving of the step. **The issue's figure for "the rule above",
36.0337 s, is not that rule's lap.** An independent integration with
each pair of rules puts it at braking on the diamond with the drive
held by the engine alone ("diamond on braking only", as the review's
own comment calls the stepped oval's figure); the drive rule the issue
writes, `min(P/V, √((μ_y G)² − (V²κ)²))`, holds the drive on the way out
of each clothoid and laps 0.125 % slower, 36.0787 s. So the clause
"36.0337 s within 0.03 %" could not have been met by the rule it names;
it is replaced by the two references the solver can be set to exactly,
and the written rule's own figure is pinned as the solver's. Also
checked once, outside the tests: TUM's `calc_vel_profile` (loaded by
path, its package does not import on this Python) gives 35.7305 s and
35.1130 s on the two ovals at 0.25 m for the ellipse and 36.5277 s for
the diamond both sides (references 35.7254, 35.1056, 36.5213), within
0.02 %; a test against it is in the suite and skips where the package
does not import.

Gate D1's other closed forms are now tests too: a downhill helix held
at the root of the combined rule (to 1e-4, on the diamond and on a
shape of 0.6), and the engine's work at full throttle against the
tyres' (6e-6). **The heat chain on a simulated lap** is
`heat.sim_heat`: the lap is put on an even 100 Hz clock (its steps are
even in distance, so a bin's plain mean would weight the slow part of a
second too little), the lateral is what the tyres hold and the
longitudinal what an accelerometer reads, and the bins count from the
lap's start. A right-hand lap heats the left tyres and the mirrored
track swaps them to 1e-12; halving the step moves the heat by under
0.1 %; 25 Hz against 100 Hz by under 0.5 % (the review had 0.3 % above
10 Hz). `heat.bin_1s(..., phase_s=)` moves the bins' edges, to measure
what their phase is worth.

**Gate D1 on logged laps (`heat_gate.py`, `lapsim check-heat-chain`,
2026-11-24).** The chain of `heat.py` counts a lap's seconds from the
lap's start; production bins a whole stint on its rolling clock and sums
a lap from the stint's bins. On a stint of one lap the two give the same
bins to 1e-6 (a unit test against `statespace._bin_stint`). On the
1,906 laps in 182 stints that production fits, from the samples
production bins and with its fitted shares (1,782 laps have a heat from
both sides), with the limits written into the gate before the run:

| | formula cars | Inferno 86 | limit |
|---|---|---|---|
| a lap's heat over production's, mean size of the log ratio, worst tyre | 0.0052 | 0.0040 | 0.01 |
| its mean (signed), any tyre | ±0.0001 | ±0.0002 | |
| the same per stint (the stint's mean), mean size, worst tyre | 0.0017 | 0.0008 | 0.003 |
| the bins' edges half a second back: 90th percentile of what a lap's heat moves, worst tyre | 0.0111 | 0.0058 | 0.02 |

**Pass.** A lap's heat by this chain is production's on average to
0.02 % and lap by lap to 0.4–0.5 %, and what is left is the phase of
the bins: moving the edges half a second moves a lap by about as much
(0.1–0.5 % in the mean, 1.1 % at the 90th percentile on the formula
cars' front right). The twelve laps that hold a discarded
window come back with no heat from either side.

*The 0.5 s smoothing of issue #207's E0, measured and not adopted
here.* #207 would smooth the lateral and the longitudinal over 0.5 s
before the positive parts are taken, on both sides of a comparison.
This chain does not, as production does not. A lap's heat as this
chain has it over the same lap smoothed first:

| | four-tyre range |
|---|---|
| logged laps, accelerometer (1,370) | +0.7 to +1.3 % |
| logged laps, GPS-derived (401 of the formula cars) | +2.8 to +4.8 % |
| simulated laps at each car's typical use (ten cells) | +0.6 to +1.0 % in the mean, 1.3 % at most on a tyre |

So the two chains are 0.6 to 1.3 % apart on a simulated lap and as far
on an accelerometer's, and several per cent on a GPS-derived series
(its noise passes through the positive parts, which is what the
smoothing is for). #207's gate E1 asks a simulated lap put through the
observed side's chain to return the table's heat within 0.002: that
holds only if the table is made by the chain the observed side is read
by. The rule for #206's table and #207's comparison is therefore one
chain on both sides, whichever it is; the two are not interchangeable.

**Gate D3, lateral heat by side (2026-11-24; gate C3's report,
`car_holdout.MOMENT_MEASURES`).** In gate C3's form by the owner's
decision of 2026-11-23, and on gate C3's own folds: two more measures
per held-out lap, each with a logged lap of another day beside it, at
P75, P90 and P95 of pace; they are not counted in the 76 cells.
`lat_heat` is the log of the simulated lap's lap-mean `V·|lat|` over the
logged lap's, and `side` the same for right turns less left turns. The
simulated lap's lateral is what the tyres hold on the track's curvature
and bank at its speed; the logged lap's is **its own**, the Doppler
lateral on the road's bank at the track's nodes, which the tables keep
since version 8 (`CarTables.lateral`). So a wrong sign, a mirrored
track or a curvature wrong on one side shows here, where gate C3's heat
(the track's curvature on both sides) could not show it.

| | cells no further than another day's lap | simulated, range over the scored cells | another day's lap |
|---|---|---|---|
| lateral heat | 10 of 19 | 0.006–0.032 | 0.007–0.028 |
| right less left | 12 of 19 | 0.013–0.043 | 0.006–0.065 |

22 of 38. With their signs, over every held-out lap of the day folds
(the median over days of the day's median lap), and what a mirrored
track or a swapped sign would read by side at the layout (twice the log
of the logged laps' right over left):

| car | layout | days | lateral heat | right less left | a mirror would read |
|---|---|---|---|---|---|
| Inferno 86 | Fuji | 10 | +0.003 | 0.000 | 1.27 |
| Inferno 86 | Motegi | 2 | −0.040 | −0.029 | 0.69 |
| Inferno 86 | Sodegaura | 16 | −0.007 | −0.015 | 1.20 |
| Inferno 86 | Tsukuba | 3 | −0.005 | −0.028 | 1.70 |
| KK-SII | Suzuka | 7 | −0.016 | −0.023 | 0.15 |
| KK-SII | Tsukuba | 12 | −0.012 | −0.033 | 1.62 |
| KK-F | Fuji | 2 | −0.014 | +0.014 | 1.29 |
| KK-F | Suzuka | 2 | −0.014 | −0.012 | 0.17 |
| KK-F | Tsukuba | 9 | −0.016 | −0.021 | 1.64 |

- **No sign or mirror defect**, and the clause can see one at four of
  the five layouts: a mirror would read 0.7 to 1.7 by side there against
  the 0.043 at most that is read. **At Suzuka it is nearly blind**: the
  figure of eight loads its two sides alike, so a mirror would read 0.15
  to 0.17, hardly over the 0.12 that counts as a defect. Suzuka's sign
  rests on the track's other checks (B2's corner drawings and by-side
  curvature), not on this one. `lat_heat` does not move under a mirror at
  all.
- **The simulated lap has 0.5 to 1.6 % less lateral heat than the logged
  lap** in eight of nine cells (4 % less on the Inferno 86's two days at
  Motegi), where two logged laps of different days differ by 0.7 to
  2.8 %. **Most of it is the logged lap's lateral on the straights.** On
  the nodes where the track's line asks under 0.05 g (14 to 45 % of a
  lap) the logged laps carry 1.1 to 4.8 % of their lateral heat and the
  track's line at the same speed 0.5 to 2.5 %: an excess of 0.6 to 1.0 %
  of the lap's lateral heat at seven of ten cells (1.6 % on the KK-SII
  at Fuji, 2.4 % on the Inferno 86 at Motegi). It is the noise of the
  Doppler lateral passing through the absolute value, with whatever the
  car really weaves, and a simulated lap has neither. In the turning
  nodes the track's line at the lap's own speed is within 1 % of the
  logged lateral heat (−2.1 % at Motegi). Two logged laps both carry
  the excess, so the yardstick does not, and the simulated lap is scored
  against a bar that is stricter than it looks by about this 1 %: the
  10 of 19 on `lat_heat` is read with that in mind. Not a defect by the
  rule of #206, and not corrected: production's chain is given the
  logged lateral as it is.
- **The simulated lap leans to the left turns nearly everywhere**: the
  right turns carry less of the simulated heat than of the logged in
  seven of nine cells (−0.012 to −0.033; Tsukuba −0.028, −0.033, −0.021
  on the three cars). That it has one sign at four of five layouts
  means three cars agreeing at Tsukuba does not make it Tsukuba's
  curvature: it is a lean common to the layouts. **It is in the
  simulated speed, not in the track**: the track's line driven at each
  logged lap's own speed, against that lap's own lateral, has no such
  lean (right less left 0.000 to +0.029 at eight of ten cells, −0.009
  at two). So the simulated car is a little slower through right turns
  than through left ones, relative to the logged car, at most layouts.
  It is placed further below ("The lean to the left turns, placed"):
  not the bank either; the logged cars hold more lateral in their right
  turns. Under the 0.06 at which #206 sends one to the per-corner table.

The lateral traces are the speed traces' laps row for row (built from
the same lap list, shapes checked at the build), and gates C2 and C3
are unchanged line for line by the tables' new version.

**Gate D1, the pace table (`pace.py`, `pace_gate.py`, `lapsim
check-pace-table`, 2026-11-24).** A car's laps at a layout over the
whole range of pace, for whoever needs the heat of a lap of a given lap
time without a solver (#207, #208, the calculators). As built it differs
from the table #206 first specified in the four ways its note of
2026-11-23 asked for:

- **The axis is the driver's use of the car's limit**, `q = −ln(use)`:
  twelve laps evenly in q from a use of 0.40 to a use of 1, driven as
  one batch (and, since the cap was built, two more to the knob's end
  lap: below). The lap at a use of 1 is the knob's lap at q = 0, braked
  as the driver brakes; it is **not the max driver's lap** (clause 3.1
  of the driver committee's design: every use at 1 and the driver's
  habits off), which is 0.3 to 0.9 s faster at these layouts and is kept
  beside the table (`max_driver_lap_s`).
- **It holds laps, not heat.** Each knot is kept as its speed at the
  track's nodes. Per-tyre heat is defined by the force shares of the fit
  in hand (#208: leaving a cell out moves the same laps' heat by up to
  11 % per tyre), so `PaceTable.heat(lap_time, shares)` splits the lap
  with the caller's shares, through gate D1's chain unless the caller
  gives another (the smoothed chain of #207's E0 is E's to settle).
  Between the knots heat and use are read log-log against lap time.
- **It says what it assumes of the car** (`PaceTable.assumes`). A lap
  that has not been driven has no day, so: the car's latest powertrain
  unless one is named; that powertrain's own shift point and gears;
  standard air; braked as the driver brakes; the road's bank; dry. This
  was #206's to settle without the owner. The KK-SII's per-circuit
  gearing is flagged in its tables and changes nothing in them: its gear
  is inferred, so it is driven on one specific power.
- **A target lap time gets one of three answers.** `ok`;
  `outside_support`, answered and flagged (owner, 2026-11-23): its use
  lies outside the car's logged range, or it is slower than the table's
  slow end and is read off the last two knots; `no_solution`: it is
  faster than the table's fastest lap, and the table gives no heat, with
  that lap's time and the max driver's beside it. As first built the
  table ended at a use of 1; with the cap (below) it ends at the knob's
  end lap. None of the cars' 1,129 dry build laps is faster than the lap
  at a use of 1. The
  logged range is the use of every dry build
  lap of the car at the layout, each driven with its own day's car
  (`car_gate.use_range`, saved by gate C2 as the model's
  `meta.use_range`); at a layout the car has not run it is the car's
  range over the layouts it has, and the table says so.

*The out-lap.* Each knot also has the part of a lap from the pit exit's
join to the line, begun at the speed the logged out-laps have at the
join and driven at the knot's use; the car is driven on round the lap,
so it reaches the line as a lap that goes on. Gate B2 now keeps the join
speed beside the join's place and the rolling time (`meta.pit`):

| layout | join (m) | speed at the join (m/s; 10th–90th) | rolling before it (s) | out-laps |
|---|---|---|---|---|
| Fuji | 752 | 21.0 (9.5–31.4) | 44 | 48 |
| Motegi | 240 | 19.3 (15.9–23.1) | 29 | 29 |
| Sodegaura | 328 | 23.5 (13.8–29.6) | 40 | 60 |
| Suzuka | 641 | 30.6 (23.4–36.9) | 26 | 34 |
| Tsukuba | 215 | 19.4 (15.8–24.2) | 16 | 60 |

*The gate*, with #206's limit (1 %) written in before the run: fifteen
tables (three cars at five layouts, five of them at a layout the car has
not run), each held at twenty lap times evenly between its ends to the
lap solved directly at that lap time with the table's own car, under two
sets of shares (the car's and another car's).

| | worst of the fifteen tables | limit |
|---|---|---|
| per-tyre heat, table over direct solve (size of the log ratio) | 0.0088 (KK-SII, Tsukuba) | 0.01 |
| the use read off the table | 0.0013 | 0.01 |
| the out-lap's time and per-tyre heat | 0.0089 (KK-SII, Sodegaura) | 0.01 |

**Pass**, with little room, and more knots would not buy much. Nine
knots give 1.18 %, twelve 0.89 %, sixteen 0.58 %, about as one over the
number of knots and not as a smooth curve's error falls. Solved directly
every 0.05 s over 3 s of lap time, a tyre's heat is jagged about its
trend by up to 0.36 % (0.18 % rms; the KK-SII's front right at Tsukuba)
and 0.19 % at Fuji, the jag repeating every 1.5 to 2 s of lap time: it
is the one-second bins of the heat chain sliding along the lap as the
lap time changes, the phase effect gate D1 measured on logged laps, and
it is in the knots and in the direct solve alike. It is largest on the
shortest laps (58 to 67 s, where the gate's worst tables are). So
perhaps half of the 0.9 % is a floor that belongs to the chain and not
to the table. A table of nine nodes, as the calculators interpolate
today (#208), would not hold 1 %.

*Reported.* (1) **The powertrain a table assumes is worth up to 2.8 % of
a tyre's heat** at one lap time (the median logged lap of each layout):
the first engine of the Inferno 86, or the KK-F before its new intake,
needs 2–6 % more use for the same lap time; a rear tyre gets up to
2.8 % less heat and a front up to 1.3 % more. A table is therefore per
powertrain, and F's per-fold tables must be built with the powertrain of
the laps they are read against. (2) Of the cars' 1,129 dry build laps
read through their layout's table, none is faster than the lap at a use
of 1 (nor than the max driver's), and 78 are flagged outside the support. The support is each lap's
use with its own day's car, and the table's car is the latest powertrain
in standard air, which does the same lap time at another use; 30 of the
78 are the Inferno 86 at Fuji, where both engines ran. The cause was not
separated lap by lap. A consumer that knows a lap's day should build the
table for that day's powertrain.

**Gate D2, the wet (`wet.py`, `wet_gate.py`, `lapsim check-wet`,
2026-11-24).** A wet lap is solved to its own lap time, so the wet grip
scale, the day's water level and the driver's use are one product that
no simulated lap depends on. Two things are not absorbed by the lap
time, and they are what a wet gate can judge: **braking falling less
than cornering** (the driver committee's clause 1.6) and **the wet
line** (a wet track exists at Tsukuba only). Everything below was
written on #206 before the first run: the readings, the arms, the form
and the entry rule.

*What there is to judge with.* 135 wet build laps: the Inferno 86 at
Tsukuba (41 laps, 4 days) and Sodegaura (23, 2), the KK-F at Tsukuba
(33, 4) and Fuji (6, 1), the KK-SII at Tsukuba (17, 1) and Fuji (14, 1).
In gate C3's form (a logged lap of another day beside the simulated
one, the owner's ruling of 2026-11-23) four cells per measure can be
scored, three of them one car at one layout. **The owner ruled on
2026-11-24 that the wet is scored on same-day laps** ("let's do same
day scoring for wet, wet is harder, we know that"; asked first as a
question, measured, then confirmed): that scores 13 cells per
measure (nine of them on one wet day), and it suits the wet, where two
days differ in water. It is a much stricter bar: against a same-day lap
the dry gate itself reaches 31 of 104 (speed 10 of 26, straights 11,
apex 1, braking 9) where it reaches 60 of 76 against another day's. D2
is scored on same-day laps and read beside that 31 of 104, with the
other-day cells shown.

*The readings* (per car, its wet passes pooled; the expected best of m
wet passes over the expected best of the same m dry passes at the
corner or stopping place, m half the wet passes, 3 to 20; the median
over corners):

| | wet cornering | wet braking | the wet braking term (their quotient) | `wet.grip_scale` as fitted |
|---|---|---|---|---|
| Inferno 86 | 0.874 | 0.932 | 1.07 | 0.92 |
| KK-SII | 0.739 | 0.755 | 1.02 | 0.81 |
| KK-F | 0.822 | 0.686 | 0.83 | 0.82 |

The Inferno 86's are the committee's (0.83–0.88 and 0.92): its braking
falls less than its cornering, on every fold (a term of 1.01 to 1.15
without each wet day in turn). The KK-F's braking falls *more* than its
cornering (0.83 to 0.99 by fold), which the committee put with the
driver (no ABS; braking further from lock-up). A wet day's own
cornering ratio runs from 0.66 to 0.93: the day's water, as far as the
logs give it.

*The arms*, each wet build lap solved to its own lap time with its
day's car: **A**, the null, the dry track and a lower use and nothing
else; **B**, with the wet braking term, read without the lap's own wet
day; **C**, on the wet line (the wet track's curvature laid on the dry
track's nodes, so the logged laps are read where they are); **D**,
both. What a fold holds out is the wet term only: it is read without
the lap's own wet day. The car is the dry car gate C2 saved, fitted on
every dry lap; no wet lap is in that fit but for `wet.grip_scale`,
which the lap time absorbs.

*The wet line was checked on the real pair before its result was
read*: laid on Tsukuba's dry track, the wet track's nodes are 0.5 m
from the dry line in the median (1.5 m at most), the laid-over
curvature lines up with the dry one at no shift (correlation 0.998),
and its apexes are 2.5 % tighter in the median (from 4 % wider to 10 %
tighter over the eight corners). Arm C drove the wet line where it is.

| arm | same-day cells | other-day cells | same-day cells won / lost against A | car-days nearer than A on its own measure | needed |
|---|---|---|---|---|---|
| A, the null | 14 of 52 | 13 of 16 | | | |
| B, the wet braking term | 14 of 52 | 13 of 16 | 1 / 1 | braking: 9 of 13 | 10 |
| C, the wet line (Tsukuba) | 11 of 28 | 13 of 16 | 2 / 2 | apex: 6 of 9 | 8 |
| D, both (Tsukuba) | 10 of 28 | 12 of 16 | 2 / 3 | speed round the lap: 5 of 9 | 8 |

- **The null already does the wet about as well as the dry.** With no
  wet term at all, a wet lap as a lower use of the same car on the dry
  track is no further from the logged lap than a same-day lap in 14 of
  52 cells (27 %; the dry gate 31 of 104, 30 %) and than another day's
  in 13 of 16 (81 %; the dry gate 60 of 76, 79 %).
- **Neither term meets the entry rule, and both stay off**
  (`solver.WET_BRAKE_DEFAULT = False`; the wet line is not driven). The
  rule: nearer than the null on the term's own measure on a number of
  car-days a coin reaches under one time in twenty, and no more same-day
  cells lost than won.
- *The braking term* is nearer on 9 of 13 car-days (10 needed): on five
  of the Inferno 86's six and both of the KK-SII's, and on two of the
  KK-F's five, the car whose term is under 1. One day short of the
  rule; it moves a day's braking measure by 0.002 to 0.034 g where the
  measure is 0.13 to 0.24 g. Not taken, and not re-cut by car to make
  it pass.
- *The wet line* is nearer at the apexes on 6 of 9 car-days (8 needed).
  Where it helps it helps a good deal (0.61 to 0.38 m/s on the KK-SII's
  day, 0.63 to 0.41 and 0.74 to 0.52 on two of the KK-F's) and where it
  does not it is as much worse (0.38 to 0.57 and 0.19 to 0.33 on two of
  the Inferno 86's days): one wet line pooled from every car and every
  wet day is right for some days and wrong for others. That the line
  moves with the water is the candidate reason; it was not tested.

So the wet stays what it was in the simulation: the same car at a lower
use on the dry track. `wet.py` keeps the readings and
`SolverCar.wet_brake` the term (a wet lane's braking level is multiplied
by it; nothing in the kernel changed, and gates C2 and C3 are unchanged
byte for byte), for when there are more wet days.

*The pace table's wet entry is the dry table, with the wet laps in its
support.* Read as built, the table flagged every one of the wet laps as
outside its support (their use of the dry limit is 0.48 to 0.77, under
every car's dry range), which is not the ruling: only a target wetter
than the wettest logged day is to be flagged. So gate C2 now also saves
the use of each wet build lap as the null drives it
(`meta.use_range_wet`), and a table's support runs down to the slowest
of them (at the layout, or over the layouts the car has run in the wet;
`support["use_min_dry"]` keeps the dry laps' slow end). Of the 134 wet
build laps of the three cars 121 are then answered `ok` and 13 flagged
(the table's car is the latest powertrain in standard air), and the dry
laps flagged fall from 78 to 7. No separate wet table is needed until a
wet term enters. A slow dry lap and a wet lap of one lap time get one
answer: there is no wetness input, by the owner's ruling.

**The cap: a use cannot pass 1 (2026-11-24; `solver._knob`,
`SolverCar.use_cap`).** The driver committee's clause 3.1, built at the
owner's request to finish #206. With the knob u above 1 the cornering
stays at the car's limit and the braking level goes on rising,
`min(mu_b · brake_use · u, mu_b)`, to the car's own at u = 1 /
`brake_use`: *the knob's end lap*, braked at the car's level and still
on the shape the driver holds. Lap time is continuous through 1, falls
to the end lap and is flat beyond it; a target faster than the end lap
has no solution.

- **It is built above the two solver forms, not in them.** A lane above
  its cap is handed to the kernel or to the numpy form with its limits
  already set (cornering at the cap, braking at its capped level, the
  knob at 1); a lap first solved above 1 is solved again on the capped
  lap time. Neither form changed, so they agree to the last bit as
  before, and a lap at or under a use of 1 is the old arithmetic exactly.
- **It is asked for where a lap is answered**: a pace table, and the
  fitted car's held-out laps in gate C3 (`from_model(..., use_cap=True)`,
  `Drive.use_cap`), and only for a car whose `mu_y` is its limit. *As
  first built it was on for every limit car, and by the rule written
  before the run that was a defect*: it moved gate C2's recovery on
  simulated laps and the making of gate C3's planted wrong cars, which
  use the knob as a free scale (a known car driven at 1.1; a car with
  0.85 of the power, which needs more than its limit to make the lap
  time). The fit's own workings, a recovery, a planted car and the
  gate's comparison arms (another drive rule, a drag that is not the
  car's, a flat road) are driven without it.
- **What moved, against the rule** (gates C2 and D2 and the pace gate
  under a use of 1 byte for byte; gate C3 only where a fold's car solved
  a lap above 1): C2 and D2 are byte for byte. Gate C3 stays at 60 of
  76, and what differs is the Inferno 86's folds that hold one of the
  five laps (of 2,231) above 1: under the cap the highest of them needs
  a knob of 1.102 where it needed 1.009, since past 1 only the braking
  rises; a day's median speed error there moves by under 0.03 m/s.
- **The pace table goes on to the end lap.** A car at its limit has two
  more knots, from a use of 1 to the knob's end, held to the same 1 %
  (0.96 % at most over the fifteen tables). The end lap is faster than
  the lap at a use of 1 by 0.27 to 0.60 s on the Inferno 86 and 0.19 to
  0.40 s on the KK-F, and by 0.05 to 0.11 s on the KK-SII, whose driver
  already brakes at 0.97 of the car's level. The max driver's lap is a
  further 0.15 to 0.42 s beyond it (the held shape's cost), and no knob
  reaches it: there is no habit left to give up past the braking level.
  A target between a use of 1 and the end lap is answered and flagged
  (no logged lap is there), with the cornering use (1) and the braking
  use given apart, since the knob there is not a use.

*An export for a consumer without a solver* (`PaceTable.export(share)`,
for #208's calculators): the knots as plain numbers (lap time, knob, q,
cornering and braking use, heat per tyre, the out-lap's time and heat),
with the support, what the table assumes and its two fast ends. It is of
one set of shares and is made again for another; the table itself holds
laps.

**The frame factor (2026-11-24; `heat_gate.frame_factor`, reported by
`lapsim check-heat-chain`).** #206's frame clause: a simulated lap is in
the tyres' frame on the position clock (the lateral the tyres hold on
the road's bank; `dV/dt` with the grade), and production's τ and K were
fitted on the heat of the sensor's series. Where a table's heat has to
meet them it is multiplied by the per-tyre median, over the logged laps
of a production car at a layout from one kind of source, of production's
chain on the sensor's series over the same lap's own Doppler lateral in
the tyres' frame and its `dV/dt` with the grade, through the same chain.
No simulated lap is in it.

| production car, source | factor by layout and tyre | after it, a session is off by (worst tyre; median, 90th percentile) |
|---|---|---|
| formula cars, accelerometer | 0.986–1.011 at Suzuka and Tsukuba; 0.925–1.030 at Fuji (the rears 0.93–0.94 on 44 laps of 7 sessions) | 2.0 %, 4.4 % (60 sessions) |
| formula cars, GPS-derived | 1.057–1.099 | 1.1 %, 2.9 % (36) |
| Inferno 86, accelerometer | 1.017–1.040 at four layouts | 1.5 %, 5.4 % (63) |

An accelerometer on the road's plane reads what the tyres hold, and the
formula cars' factor is 1 within about 1 % where they have most laps.
The formula cars' rears at Fuji (0.93–0.94) are not explained.
The Inferno 86's accelerometer heat is 2–4 % over the Doppler's, on
every tyre and layout alike, and a GPS-derived series 6–10 % over it:
the noise that passes through the positive parts (the smoothing figures
above). What a factor leaves session by session is what #206 first
estimated (1.3 % in the median and 5.3 % at the 90th percentile over
125 sessions). It goes away if production is refitted on the tyres'
frame (#255); whether a table read against production's heat is right
after it is #207's gate E1, which needs each fold's shares and is not
measured here.

**The lean to the left turns, placed (2026-11-24).** The simulated car is
slow through right turns relative to left ones, against the logged car:

- *At the apex*: with the car and the corner's speed accounted for, the
  simulated apex speed is 1.4 % (± 0.5 %) lower in right turns than in
  left turns relative to the logged one (95 corners of gate C3's corner
  table); seven of ten car and layout cells, Fuji the other way on both
  formula cars (+2.0 %, +2.9 %).
- *Not the track's curvature* (the track's line at each logged lap's own
  speed has no such lean) *and not the bank*: with every road taken as
  flat the lean is the same (−1.0 % with the bank, −1.0 % without, the
  mean over the ten cells).
- *The logged formula cars hold more lateral in right turns, and that is
  consistent with their lean.* The cornering limit's statistic by side
  is higher in the right-hand corners by 4.5 % on the KK-SII (8 right
  corners, 6 left) and 2.7 % on the KK-F (4 and 1); the simulation has
  one limit for both ways, and 4.5 % of lateral is about 2.2 % of apex
  speed, against the KK-SII's −2.3 % and −2.5 % at Suzuka and Tsukuba.
  **The Inferno 86's lean is not accounted for**: its two sides differ
  by 0.6 % (9 and 6 corners), which is 0.3 % of apex speed, against
  −0.6 %, −2.2 % and −4.2 % at Fuji, Tsukuba and Motegi.
- *The owner, 2026-11-24, of the formula cars:* they are symmetrical, so
  this is the corners and not the car. These circuits run mostly clockwise (58
  right corners to 37 left), and the corners a driver leans on hardest
  are mostly rights. Taking the higher side as the limit of a
  symmetrical car would raise `mu_y` by 2.7 % on the KK-SII (1.541 to
  1.582) and 0.7 % on the KK-F (1.493 to 1.503, on four corners) and
  nothing on the Inferno 86, and the max driver's lap by 0.4–0.8 s and
  0.1–0.2 s; it moves the ceiling only (a logged lap is solved to its
  lap time) and would not remove the lean. Not applied: it changes the
  limit's reading ruled on 2026-11-24 and is the owner's to decide.
- What would remove the lean is the driver using more of the limit in
  some corners than in others, which is #313's; nothing is fitted by
  side or by corner here.

**Dropped from #206's first text, with the reason.** The `centroid`
columns of the pace table: no consumer has named them since E's review.
Lateral heat by side with the logged lap read by other sensors than the
Doppler: the tyres' frame on both sides is the ruled frame, the
accelerometer's chain is held by gate D1's parity, and the frame factor
above is what separates the sensors.

**Left open when #206 closed**: more wet days before the wet terms can
be judged again (gate D2 is the instrument); the higher-side reading of
a symmetrical car's limit (the owner's); the driver's use by corner,
which the lean points at (#313); production on the tyres' frame (#255)
and the table read against production's heat (#207).

**Segment E (#207): the owner's rulings (2026-11-24).** (1) *Production
is not retrained yet.* E0, the refit of production on GPS-derived heat
(#255), is set aside; it is not a step of E. The simulation's heat is
compared with production's heat as it is today (production's chain on
the sensor's series) through the frame factor above, and one chain is
on both sides: production's, unsmoothed, which gate D1 holds the
simulated side to. (2) *E is scored in gate C3's form*: a cell is a car
at a layout at a level of pace (P75, P90, P95) on one tyre, and it
passes when the simulated lap's heat is no further from the logged
lap's than a logged lap of another day is. (3) *No fixed limits at all*
("get rid of the 5 % all together, use the vs real life paired runs
like we do elsewhere", after E was first run with #207's 0.03, 0.02 and
0.05 beside the score): a car with 0.85 of the power was inside 5 % in
every cell, so the limit told nothing. Every figure of E is a simulated
lap against the logged lap with a logged lap of another day beside it,
the transfer arms and the lap time not given included. What the
adoption of the simulation is, restated when asked the same day: at
gate F's last clause, for the entries with no logged laps (their heat
from a simulated lap, written into `tire_model.json` with a `source`
flag and shown as an estimate in the calculators); an entry with logged
laps keeps its logged heat, and the pressure model's own fit does not
stop.

**Gate E1 and arm B of gate E2: a simulated lap's heat per tyre against
production's (2026-11-24; `heat_match.py`, in gate C3's report).** On
gate C3's own folds, every held-out dry build lap is solved at its lap
time by the fold's car and that lap (not a pace-table read, so the
table's 0.9 % is not in the verdict) goes through production's chain.
*Observed* heat is production's chain on the sensor's series of the same
lap. The simulated lap is brought to it by the fold's frame factor, read
from logged laps the fold does not hold out (a day fold: the layout's
other days from the same kind of source; a layout fold: the car's other
layouts). A cell is a car at a layout at a level of pace on one tyre,
passing when the simulated heat is no further from the logged lap's
than a logged lap of another day *from the same kind of source* is.
Beside it the same with no sensor in it: against the lap's own heat in
the tyres' frame, no factor. The rules, the expected signs and the
planted cars were written on #207 before the run; of 1,129 held-out
laps three of mixed source are left out.

| arm | E1: held-out days | | E2 arm B: a layout the car was not fitted on | |
|---|---|---|---|---|
| | production's frame | tyres' frame | production's frame | tyres' frame |
| the fitted car | **40 of 76** | 46 of 76 | **41 of 76** | 47 of 76 |
| planted: 0.85 of the power, grip raised | 30 | 37 | 27 | 36 |
| planted: another car's parameters | 19 | 18 | 13 | 18 |
| the car's other layouts, unscaled (no simulated lap) | | | 4 | 3 |

- **What arm B was given, and what it was not.** The lap's own lap time
  and the track's own line; only the car was unseen (fitted on its other
  layouts). The track from other cars (arms C and D) and a lap time that
  is not given are measured further below.
- **The simulated lap is about as far from the logged lap as another
  day's logged lap is.** In the median cell it is 1.4 % (held-out days)
  and 1.5 % (a layout the car was not fitted on) from the logged lap's
  heat on a tyre, where a logged lap of another day and like lap time
  is 1.3 % (a same-day lap 1.1 %). That is why it passes about half the
  cells and not all: the bar is reality's own repeatability, and the
  simulation sits on it.
- **What it beats, by a wide margin, is the car's own logged heat at its
  other layouts, unscaled**: 4 cells of 76, 2 to 26 % off (the
  right-hand tyres 20–26 % low at Suzuka; 9–25 % high at Fuji and
  Motegi).
- **It is told from another car's parameters, and not from 15 % less
  power.** 40 and 41 cells against 19 and 13 for another car's
  parameters; against 30 and 27 for the car with 0.85 of the power,
  which with the four tyres of a lap moving together is about ten
  groups of 19 against seven, not shown to differ. At a given lap time
  on the track's own line a tyre's heat is set mostly by the track and
  the lap time; the car's fit is worth a fraction of a per cent of it
  (0.2 % or less, arms A and B below). Per-tyre heat at a matched lap
  time is blunt, as #207's finding 9 said.
- **Where it is furthest**: the Inferno 86's fronts at Motegi (−5.7,
  −6.0 % on held-out days; −4.3, −3.6 % with the layout held out; two
  days) and the KK-SII's one day at Fuji (−3.5, −3.7, +1.6, +3.7 % by
  tyre). Both are cells of one or two days, with no other day to be set
  against.
- **The total, written down beforehand as about 1 % low**: in the
  tyres' frame the four-tyre mean is 0.5 to 1.0 % low on the formula
  cars in five of six cells (the KK-SII at Tsukuba level), and not on
  the Inferno 86 (+0.3 to +0.4 %; −2.0 % at Motegi).
- **The signs written down beforehand are there on the formula cars**:
  their left tyres, loaded in right turns, read low on held-out days
  (nine of ten left-tyre figures between −1.1 and −1.6 %, the tenth
  −0.5 %) and their right tyres between −1.4 and +1.9 %. The Inferno 86
  shows no such pattern (within ±1 % at Sodegaura; its rear left +2.4 %
  at Fuji and its rear right +2.2 % at Tsukuba), as its limit by side
  said it would not.
- **What carrying the frame factor to a layout costs by itself**: under
  1 % on the fronts (2 % on one) and about 1 % on the Inferno 86's
  rears; 1–3 % on the formula cars' rears; 6 % on the KK-SII's
  accelerometer rears at Fuji (13 laps), which is the unexplained Fuji figure of the frame
  factor above. It is in production's-frame numbers and not in the
  tyres'-frame ones, which is why the two differ most there.

Every existing line of gate C3's report is unchanged by the addition.

**Arms C and D of gate E2: the track from other cars, and the lap time
not given (2026-11-24; `heat_match.transfer_laps`, a third item in gate
C3's report).** The cells were recounted first (#207): four have three
scored dry days of the car and three days and 30 laps of other cars
behind the track (the Inferno 86 at Fuji and Tsukuba, the KK-SII and the
KK-F at Tsukuba), and two more are listed (the formula cars at Fuji, on
one and two days). No other cell has a track from another class of car.
The track from other cars is gate B3's saved track built without the
car's class, its curvature laid on the own track's nodes; the bank and
the grade stay the own track's (the bank is the road's). Four arms on
every dry build lap of the cell, each solved to the lap's own lap time,
and then each day's best lap at the car's typical use with no lap time
given. Every arm is scored as the fitted car is: its cells against a
logged lap of another day, in the tyres' frame (no factor).

| arm | cells no further than another day's lap, of 40 | from the logged lap, median cell | another day's lap | cells won / lost against A |
|---|---|---|---|---|
| A: the car's own fit, its own track | 22 | 1.2 % | 1.5 % | |
| B: the car fitted without the layout | 24 | 1.1 % | 1.5 % | 2 / 0 |
| C: the track from other cars | 25 | 1.1 % | 1.5 % | 6 / 3 |
| **D: neither has seen the cell** | **25** | 1.2 % | 1.5 % | 6 / 3 |
| the car's logged heat at its other layouts, unscaled | 1 | 8.7 % | 1.5 % | 0 / 21 |
| A\*: no lap time given, own fit and track | 10 | 3.0 % | 1.2 % | 3 / 15 |
| B\*: no lap time given, the car unseen | 9 | 3.2 % | 1.2 % | 2 / 15 |
| **D\*: nothing of the cell given** | **4** | 4.6 % | 1.2 % | 2 / 20 |

- **With the lap time given, not having seen the cell costs nothing
  that can be told.** The car fitted without the layout on a track from
  other cars passes 25 cells where the car's own fit on its own track
  passes 22, and is 1.2 % from the logged lap in the median cell where
  another day's lap is 1.5 %. Between the arms the car moves a tyre's
  heat by 0.2 % or less and the track by up to 1.6 % (the formula cars
  at Tsukuba on a track from the Inferno 86's 37 laps, 0.8 to 1.6 %
  less heat than on their own line; under 0.4 % at Fuji and the other
  way round), and the two add to within 0.04 %. That is gate B3's
  by-class difference (0.017 at Tsukuba, 0.007 at Fuji) carried into
  heat, and it is of the size of the day-to-day difference itself.
- **The car's logged heat at its other layouts, unscaled, passes one
  cell of 40** and is 8.7 % from the logged lap in the median cell.
- Arm D has no lap for three of the Inferno 86's 111 laps at Fuji:
  under the cap of a use of 1, the car fitted without Fuji cannot make
  their lap times on the other cars' line. They are counted in the
  report (`D_no_lap`) and left out of arm D's figures.

*The lap time not given.* Each day's best lap of the cell is driven at
the car's typical use of its limit, which is what is carried to a
layout it has not run (the starred arms above).

| cell, nothing given (D\*) | simulated lap time against the day's best | another day's best lap against it | worst tyre's heat |
|---|---|---|---|
| Inferno 86, Fuji | 0.7 % slow | 1.4 % | 4.3 % low |
| Inferno 86, Tsukuba | 3.1 % fast | 4.7 % | 9.8 % high |
| KK-SII, Tsukuba | 0.6 % (in size) | 0.6 % | 2.3 % |
| KK-F, Tsukuba | 0.9 % slow | 1.0 % | 5.6 % low |
| *KK-F, Fuji* | 0.7 % fast | 0.5 % | 1.1 % |
| *KK-SII, Fuji* | 1.1 % fast | one day | 2.5 % |

- **The lap time is supplied about as well as days repeat.** With
  nothing of the cell given the simulated lap time is 0.8 % from the
  day's best in the median cell, where another day's best lap at the
  layout is 1.0 % from it; it is no further in three of the five cells
  that have two days (and within a tenth of a per cent of it in a
  fourth). The Inferno 86's 3.1 % at Tsukuba is inside the 4.7 % its own
  four days differ by.
- **The heat of that lap is not the logged lap's.** D\* passes 4 cells
  of 40 where D passes 25, and is 4.6 % from the logged lap in the
  median cell where another day's lap of like lap time is 1.2 %. Heat
  is steep in lap time (three to five times the lap-time miss), and the
  yardstick is a lap of *like lap time*: a lap time right to within
  what days repeat by is not near enough for heat. Even on the car's
  own fit and track (A\*) it passes 10.
- **Where the miss comes from differs by cell.** At the Inferno 86 at
  Tsukuba it is the driver's level there: the lap is 2.9 % off on the
  car's own fit and its own track already (its best laps use less of
  the limit than the car's typical use, on four days), and carrying the
  car and the track adds 0.2 % of lap time. At the Inferno 86 at Fuji
  and the KK-F at Tsukuba it is the carrying: 1.6, 2.8 and 4.3 % on the
  worst tyre as the car and then the track become unseen, and 3.4, 4.3
  and 5.6 %.
- **The lap compared against is the day's best, and that is not the lap
  production feeds.** The typical use is defined from days' best laps,
  so the simulated lap time is a best-lap pace. A session's
  25th-percentile lap, which is what #207 says F feeds, is 0.8 to 1.5 %
  slower than the day's best on the formula cars and 1.9 to 2.2 % on
  the Inferno 86 (a median lap 1.4 to 3.4 %). A simulated lap time
  prefilled as it stands would therefore put 3 to 6 % more heat on top.
  A typical use for the lap that is fed (the use of a session's
  25th-percentile lap) is the way to supply a lap time; it is not
  built, and is F's to settle (#208).

**Gate E2's wet clause (2026-11-24; `wet_gate.wet_heat`, an item in
`lapsim check-wet`).** A wet lap is driven as gate D2 left it (the same
car at a lower use on the dry track, at the wet lap's own lap time) and
its heat set against the logged wet lap's own heat in the tyres' frame;
the car is the dry car, in whose fit no wet lap is.

| cell | wet laps / days | worst tyre | four-tyre mean | right less left |
|---|---|---|---|---|
| Inferno 86, Tsukuba | 41 / 4 | 3.6 % | +2.4 % | +1.2 % |
| KK-F, Tsukuba | 33 / 4 | 1.2 % | +0.9 % | +0.4 % |
| *Inferno 86, Sodegaura* | 23 / 2 | 1.1 % | +0.1 % | +1.6 % |
| *KK-SII, Tsukuba* | 17 / 1 | 1.8 % | +1.1 % | +1.1 % |
| *KK-SII, Fuji* | 14 / 1 | 1.5 % | −0.4 % | +0.9 % |
| *KK-F, Fuji* | 6 / 1 | 3.8 % | −1.9 % | +2.1 % |

Against a logged lap of the same wet day (the owner's form for the
wet) 27 of 48 cells pass, with no wet term; the worst tyre is 1.7 %
from the logged wet lap in the median cell and 3.8 % at most, and the
two cells with three wet days read 3.6 % and 1.2 %. The right-hand
tyres read 0.4 to 2.1 % above the left-hand ones in every cell, the
same lean as in the dry. The clause as #207 wrote it (the simulation
"given the cell's neighbours' dry data and wet data from other cells")
is this with the car's own fit and track; the unseen-car form was not
run on the wet laps, where the dry arms show the car's transfer at
0.2 %.

**The pace response: heat as the lap gets slower (2026-11-24;
`heat_match.band_heat`, a fourth item in gate C3's report).** #207's E3
and the slow-lap item of E1 were first set aside as written (slopes
against an allowance from a D2 that was replaced; laps the gates' tables
do not hold), and built when the owner asked why, as one paired
measurement. The gates' laps are the build laps, which are within 2 %
of their session's best; here every clean dry lap within 12 % of it is
taken, in three bands, each solved to its own lap time by the car's own
fit on its own track and set against the logged lap's own heat in the
tyres' frame. No table is rebuilt: only the lap time and the logged
heat are needed. A cell is a car at a layout on one tyre, against a
logged lap *of the same band*, another day and like lap time.

| band (lap time over the session's best) | laps | cells | from the logged lap, median cell | another day's lap of the band | four-tyre mean, median cell |
|---|---|---|---|---|---|
| fast: the build laps (within 2 %) | 1,126 | 27 of 36 | 1.1 % | 1.6 % | −0.7 % |
| mid: up to 4 % | 490 | 27 of 36 | 1.5 % | 2.3 % | −0.9 % |
| slow: 4 to 12 % | 347 | 20 of 28 | 2.0 % | 3.1 % | −1.4 % |

- **The simulated heat stays nearer the logged lap than another logged
  lap of the same pace does, in every band.** It passes three quarters
  of the cells in each (27 of 36, 27 of 36, 20 of 28). Its distance
  from the logged lap grows as the lap gets slower, from 1.1 to 2.0 %,
  and the distance between two logged laps grows faster, from 1.6 to
  3.1 %: slow laps are slow for different reasons and differ more from
  each other than from a lap slowed evenly.
- **In the mean it drifts low, not high.** The expectation written
  beforehand was a little too much heat on slow laps (a driven slow lap
  loses its time in places and coasts there). It is the other way: the
  four-tyre mean goes from 0.7 % low to 1.4 % low in the median cell
  (3.2 % low at most, on two laps). A driven slow lap puts slightly more
  heat into the tyres than a lap of the same time slowed evenly.
- **The lean between the sides grows with the slowness, on the formula
  cars at Tsukuba.** Right less left goes from 3.4 to 4.8 to 7.3 % on
  the KK-SII there and from 1.7 to 3.4 to 5.4 % on the KK-F (1.3, 2.2,
  2.6 % on the Inferno 86); the left tyres are 3.5 to 5.1 % low in the
  slow band. Elsewhere it does not move with pace (Sodegaura 0.9, 1.4,
  1.1 %; Suzuka 1.5, 1.1, 1.1 %). So where a slow lap is slow is not
  even between the corners at Tsukuba: the right turns keep more of
  their pace than the lefts do. It is the lean of gate D3 seen from
  another side, and is #313's (the driver's use by corner).
- For a target lap time slower than the laps the car has logged fast,
  the simulated heat is therefore usable as it is: within 2 % in the
  median cell down to 12 % off the pace, with the caution on the left
  tyres at Tsukuba.

**What is left of E, and what is not built.** E0 is set aside by the
owner. Every other clause that today's laps can carry is run. The lap
production feeds (a session's 25th-percentile lap) and the lap time the
simulation supplies are F's (#208).

**Segment F (#208): the plan the owner approved (2026-11-24), before any
of it is built.** #208 was read against B–E as built and rewritten; it
holds the plan only. The rulings: (1) *Paired days.* The cold pressure
the calculators would recommend from a simulated lap (flying lap 5,
through the fold model, no TPMS reading in it) is set against the one a
logged day's own heat gives, with another logged day of the same raw
car, powertrain and kind of source at the same track beside it. A cell
is a raw car at a track on one tyre and passes when the simulation is no
further from the day than another day is. No fixed limit in bar (#208's
0.006, 0.008 and 0.010 bar are gone) and no levels of pace: a stint is
fed one lap time. (2) *With nothing given, the real run is given nothing
either*: another single day's logs with their own lap time, heat and
out-lap, and beside it the ceiling, the logged entry built without the
day. (3) *Sodegaura and Suzuka are scored with only the car unseen*,
labelled and tallied apart: a track from another class of car exists at
Tsukuba and Fuji only. (4) *The gate is a report; the switchover is the
owner's call on it* ("I will make the call when you have data"). Three
cells have neither the car nor the track seen and a cell's four tyres
move together, so the count is about three groups: the distances, the
planted controls and the simulation against what it replaces (the car's
other layouts) carry the result. (5) *Simulated entries are built from a
committed export*: each knot's flying lap and out-lap as the series the
heat chain reads, which `build_warmup_table` splits through the shares
of the fit in hand at every retrain (an entry's heat is of one fit's
shares, so it cannot be written once). (6) *The tracks are committed
with their bank, with attribution to GSI, and no flat-road arm is run*
("attribution is fine ... commit the bank track info, add attribution,
no flat stuff"). Read for it: GSI's Survey Act Q&A (Q1-16: processing
without reproduction needs no approval; Q2-4: a map made from the
elevation model does) and its content terms, which ask for the source
and a note that the content was processed
(「地理院タイル（標高タイル（基盤地図情報数値標高モデル））を加工して作成」). Neither page addresses the
numeric outputs of a simulation. **Done on 2026-11-24**: the six track
files are in `data/lapsim/tracks/` (the gates read and write them there;
`lapsim --tracks-dir` points elsewhere), the attribution is in
`data/lapsim/README.md` and in `dem.py`'s note, and the tracks of gate
B3's folds, which are caches built from some cars' laps and carry no
bank, stay in `workspace/lapsim/tracks/folds/`. The calculators' about
text gets the same attribution when a simulated entry first reaches them
(adoption): nothing derived from the tiles is in `tire_model.json` yet.

**What F has to build first.** Nothing of F exists:
`validate._evaluate_fold` has no hook for a heat source and `predict.py`
still ends at a 90 s lap. Four laps are called typical and differ: the
simulation's typical use is the use of a day's best lap; production's
holdout types in a session's 25th-percentile flying lap; the calculators
prefill the median of all the cell's flying laps, beside the median of
per-lap heat; #208's given lap is the median of a session's ordinary
flying laps from flying lap 4 on. A simulated entry has to mean what a
logged one means, so F0 builds a *fed use* beside `meta.use_ref`: per
raw car, the use of a session's median flying lap, carried to a layout
the car has not run as the typical use is. The harness's first output
needs no simulation: the yardstick itself, another day against a day in
mbar with each pair's lap-time gap, which has not been measured.

**Gate F, the harness and the yardstick (F0; 2026-11-24; `validate.py`,
`lapsim/pressure.py`, `lapsim/pressure_gate.py`, `lapsim check-pressure
--yardstick`).** Everything in this paragraph was built and measured
before a simulated lap was scored, and the pairing, the rows and the
cell list were not changed afterwards.

*The harness.* `validate._evaluate_fold` is split into the fit
(`_fit_fold`), the preparation of the held-out laps with the fold model's
κ and shares (`_prepare_fold_laps`) and the scoring (`_score_fold_laps`),
and takes `model=` and `lap_inputs_fn=`; fold 0 of the standing holdout is
the same 789 rows to the last bit. A heat source is the *entry* it would
write for the held-out cell (`pressure.Entry`: a typical lap time, the
heat per tyre at it, a pace curve, an out-lap), placed in a copy of the
fold's model (`pressure.place`) and read by `_calculator_lap_inputs`, its
interpolation included. `pressure.recommend` is the cold pressure for a
pit-exit stint and a tyre at the end of flying lap 5 (10 beside it):
through the fold's K and τ (the compound's where the session has a human
or seed label; no label from the EM), from the stint's own pit-exit
temperature and the session's weather, at the team's hot-pressure target
for the car and compound (`hot_pressure_targets.yaml`; it only sets the
unit and is the same on every side). No TPMS pressure enters. A fold's
fit takes 33 to 65 s, so it is fitted once and kept in
`workspace/lapsim/cache/pressure_folds/`, keyed by the sessions left out,
the dataset's files and the bytes of the fit's source.

*Per-tyre heat is of the fold's shares.* The held-out laps' heat is
`attach_lap_heat` under the fold model; a logged lap's bins are read once
(`heat_gate.sensor_lap_bins`, `tyres_frame_lap_bins`) and split per fold;
`holdout.shares_of(model)` reads the shares of the model in hand
(`production_shares` is it on `tire_model.json`, and is not used inside a
fold).

*The folds and the cells.* Seven folds, each a pooled car at a track with
every session of the cell left out (68 for the formula cars at Tsukuba, of
which 63 reach a fit). A *day* is a dry held-out session with three
*ordinary* flying laps (from flying lap 4 on, none over 1.12 × the stint's
best on the full lap list, a heat on every tyre) and a pit-exit stint; its
own recommendation is from the median heat of those laps at their median
lap time. Per raw car:

| raw car, track | sessions / days scored | kind of source | scored |
|---|---|---|---|
| KK-SII, Tsukuba | 26 / 12 | GPS-derived 15, accelerometer 11 | yes |
| KK-F, Tsukuba | 11 / 5 | GPS-derived 6, accelerometer 5 | yes |
| Inferno 86, Fuji | 16 / 9 | accelerometer | yes |
| Inferno 86, Sodegaura | 19 / 7 | accelerometer | yes |
| KK-SII, Suzuka | 6 / 4 | accelerometer | yes |
| Inferno 86, Motegi | 2 / 2 | accelerometer | listed |
| Inferno 86, Tsukuba | 2 / 2 | accelerometer | listed |
| KK-SII, Fuji | 2 / 1 | one of each | listed |
| KK-F-V2, Tsukuba | 2 / 1 | accelerometer | listed |

Five raw-car cells have three dry days, so twenty cells (a raw car at a
track on one tyre) are scored. The counts are lower than the review's
(which counted sessions that reach a fit: 40 on 17 days for the formula
cars at Tsukuba) because a day needs its own steady laps.

*The two forms and the yardstick.* The figure of any arm is its distance
from the day's own recommendation in mbar, per pit-exit stint and tyre; a
cell is the median over days of the day's median stint. `heat`, at a
given lap time: every side at the day's lap time, on the out-lap the fold
model itself answers with. `chain`, nothing given: each side at the lap
time and with the out-lap its own entry supplies. *Another day* is a
logged day of the same raw car, powertrain and kind of source at the
track: in `heat` up to three sessions of other days nearest in lap time,
each read from its five ordinary laps nearest the day's lap time (the
pair's lap-time gap kept); in `chain` every other day as if it alone had
written the entry (production's own builders on that day's laps). *The
ceiling* is the raw car's logged entry at the cell built without the day.
Per cell, front left / front right / rear left / rear right, mbar:

| raw car, track | `heat`: another day | pair's lap-time gap | `heat`: ceiling | `chain`: another day | lap-time gap | `chain`: ceiling |
|---|---|---|---|---|---|---|
| KK-SII, Tsukuba | 4.1 / 2.3 / 4.2 / 3.1 | 0.23 % | 7.2 / 3.9 / 5.3 / 2.2 | 7.4 / 3.7 / 8.4 / 4.6 | 1.4 % | 4.1 / 2.1 / 7.7 / 3.0 |
| KK-F, Tsukuba | 3.1 / 1.7 / 4.3 / 2.2 | 0.24 % | 10.9 / 4.8 / 6.4 / 1.7 | 4.5 / 4.0 / 6.7 / 6.1 | 0.7 % | 7.7 / 4.6 / 4.1 / 3.6 |
| Inferno 86, Fuji | 7.9 / 6.3 / 7.9 / 20.1 | 0.65 % | 3.1 / 3.7 / 12.8 / 11.4 | 24.5 / 20.1 / 23.2 / 23.0 | 2.7 % | 20.2 / 11.0 / 18.5 / 11.8 |
| Inferno 86, Sodegaura | 4.6 / 4.1 / 5.0 / 6.3 | 0.33 % | 4.6 / 2.2 / 5.3 / 3.7 | 7.7 / 9.7 / 7.6 / 7.6 | 1.3 % | 10.0 / 6.1 / 10.4 / 12.0 |
| KK-SII, Suzuka | 4.6 / 0.4 / 2.8 / 1.5 | 0.50 % | 1.4 / 3.0 / 0.05 / 2.4 | 4.7 / 1.1 / 4.1 / 0.8 | 0.7 % | 4.5 / 2.6 / 4.5 / 4.0 |

- **Two logged days are 4.1 mbar apart at a given lap time and 7.0 mbar
  with nothing given** (the median cell; 0.4 to 20.1 and 0.8 to 24.5).
  The lap time is most of the difference between the two: a pair's laps
  are 0.3 % apart in lap time when they are chosen near the day's and
  1.3 % when a day brings its own.
- **The Inferno 86 at Fuji is its own case**: two days there differ by
  2.7 % in lap time and 20 to 25 mbar with nothing given, and its rear
  right by 20 mbar even at a given lap time. Whatever stands in that cell
  is read against that.
- **The ceiling is no better than another day at a given lap time** (3.8
  mbar in the median cell, 9 of 20 cells no further than another day) and
  a little better with nothing given (5.4 mbar, 12 of 20): a cell's median
  lap is a steadier guess of a day's lap time than one other day's is.
  A cell passing about half the time is what "as near as another day"
  looks like for an entry made of the cell's own logs.
- **What stands in for a cell without logged laps today is far off**: the
  car's other layouts, unscaled (#260's last step), 20.1 mbar at a given
  lap time (3 of 20 cells) and 18.4 with nothing given (5 of the 12 cells
  where another car's lap time gives it one); the fallback as coded 52.6
  (1 of 20) and 65.0 (3 of 20).

*The harness checks* (they keep their tolerances; they are not a match to
reality): an entry read from a fold's model and placed again is the same
rows and the holdout the same frame, bit for bit, in 17 of 17 folds
(13,407 rows); no row of a held-out cell is in a fold model's entry
tables; with a stand-in entry materialised in every cell without logged
laps (the formula cars at Motegi and Sodegaura, the Inferno 86 at Suzuka)
no row of the leave-date-out holdout moves (0 of 6,725).

*The reference level on seen cells* (leave-date-out, ten folds of whole
dates, 138 sessions' worth of fits): the fold model's own entry for the
cell, pooled and with constants that have seen the cell, is 3.5 mbar from
the day at a given lap time and 7.0 with nothing given in the median
cell, which is the yardstick again. *The pooled formula-car entry* costs
nothing that can be told at a given lap time (4.5 mbar pooled against
5.0 for the raw car's own, median of eight cells at Tsukuba) and about
2.5 mbar with nothing given (6.7 against 4.1): the KK-F reads +6.5 to
+9.3 mbar on every tyre (the pooled lap time is 1.8 % slower than its
days' own) and the KK-SII 2 to 3 mbar further on three tyres.

*Pressure against the TPMS* (S2, reported, never ranking) at flying lap 5
of pit-exit stints with production's own inputs, hot-pressure MAE in
mbar, constants fitted without the cell: the ceiling 33 to 70 by cell
(median 41), the car's other layouts 31 to 78 (50), the fallback 32 to
164 (87), and the cell with only the date left out 30 to 62 (33). At
Tsukuba the fallback and the other layouts score *better* against the
TPMS than the cell's own logged entry does (32 and 31 against 41 on the
KK-SII), which is finding 2 of #208 again (a heat source can be nearer
the TPMS by being wrong) and why S2 does not rank.

**Gate F, the simulation scored (2026-11-24; `lapsim check-pressure`,
about three minutes once the seventeen fold fits are kept).** The
simulation's side of a fold is `pressure_gate.sim_cell`: the car fitted
without the layout as a layout fold of gate C3 fits it; the track from
the other classes' laps where they have three days and thirty laps
(Tsukuba and Fuji; a day is driven on a track built without any lap of
its own date, the smallest being 25 laps), else the cell's own track;
`pace.build_table` for the scored day's powertrain; its laps exported
(`PaceTable.laps_export`) and split through the fold's shares; times the
frame factor of the day's kind of source, read from the car's build laps
at its other layouts under the fold's shares; and the entry's own lap
time from the *fed use*.

| raw car, track | track from | fed use (typical use) | read at |
|---|---|---|---|
| KK-SII, Tsukuba | Inferno 86, 37 laps on 4 days | 0.814 (0.862) | Suzuka 0.837, Fuji 0.790 |
| KK-F, Tsukuba | the same | 0.803 (0.849) | Fuji 0.803, Suzuka 0.802 |
| Inferno 86, Fuji | KK-F, KK-SII, SFJ, 105 laps on 5 days | 0.845 (0.915) | Sodegaura 0.829, Tsukuba 0.845, Motegi 0.940 |
| Inferno 86, Sodegaura | its own (only the car unseen) | 0.861 (0.923) | Fuji 0.861, Tsukuba 0.846, Motegi 0.924 |
| KK-SII, Suzuka | its own (only the car unseen) | 0.837 (0.888) | Tsukuba 0.847, Fuji 0.828 |

*What of the cell the simulated side has still seen.* The pit exit: a
track's `meta.pit` (where an out-lap joins the driven line, how fast and
after how long) is the median over the layout's out-laps of every car,
the held-out car's among them, and a track from other cars keeps the own
track's. It enters the out-lap of an entry with nothing given and no
flying lap. And the layout's reference line, bank and grade are the own
track's in every cell (the bank is the road's); only the curvature is
from the other cars' laps.

*The fed use* (`car_gate.fed_use`) is the use of a session's median
flying lap, the median over a raw car's dry sessions at a layout and then
over its layouts (`pressure_gate.fed_laps`: the lap index's full laps
under the pressure model's own lap filter; on the 148 sessions both
sides have, the session medians agree with production's to 0.01 % in the
median and 0.02 % at the 90th percentile). It is 0.05 to 0.07 under the
typical use, which is a best-lap pace. Its spread between a car's layouts
is 0.00 to 0.05 on the formula cars and 0.08 to 0.11 on the Inferno 86,
nearly all of that Motegi's two sessions (0.92 to 0.94 where the others
read 0.83 to 0.86). The KK-F-V2 has no dry session away from Tsukuba and
so no fed use; it has one day and is listed. The KK-F's laps away from
Tsukuba are all GPS-derived, so it has a frame factor for that kind of
source only, and its five accelerometer sessions at Tsukuba have no
simulated entry: three or four of its five days are scored.

*At a given lap time* (`heat`; signed distance in mbar, above zero a
higher cold pressure, so less heat than the day's own):

| raw car, track | simulation, fl / fr / rl / rr | another day | cells passed |
|---|---|---|---|
| KK-SII, Tsukuba | +5.0 / +0.6 / +3.2 / −6.5 | 4.1 / 2.3 / 4.2 / 3.1 | 2 of 4 |
| KK-F, Tsukuba | +7.3 / +2.5 / +3.2 / −2.1 | 4.7 / 2.7 / 4.9 / 1.2 | 2 of 4 |
| Inferno 86, Fuji | +2.7 / +2.5 / −6.4 / −2.3 (size 3.3 / 3.1 / 10.7 / 4.9) | 7.9 / 6.3 / 7.9 / 20.1 | 3 of 4 |
| Inferno 86, Sodegaura | −0.3 / +0.2 / −0.0 / −1.3 (size 3.0 / 2.2 / 4.3 / 3.4) | 4.6 / 4.1 / 5.0 / 6.3 | 4 of 4 |
| KK-SII, Suzuka | +2.4 / −2.3 / −0.5 / −6.4 | 4.6 / 0.4 / 2.8 / 1.5 | 2 of 4 |

*With nothing given* (`chain`; the simulation at the lap time of its fed
use and with its own out-lap):

| raw car, track | simulation, size, fl / fr / rl / rr | another day | its lap time against the day's | another day's | cells passed |
|---|---|---|---|---|---|
| KK-SII, Tsukuba | 8.4 / 3.4 / 9.4 / 7.6 | 7.4 / 3.7 / 8.4 / 4.6 | 1.1 % | 1.4 % | 1 of 4 |
| KK-F, Tsukuba | 12.0 / 3.8 / 5.5 / 1.9 | 9.0 / 2.2 / 10.8 / 2.1 | 0.8 % | 0.7 % | 2 of 4 |
| Inferno 86, Fuji | 16.0 / 18.5 / 10.9 / 9.3 | 24.5 / 20.1 / 23.2 / 23.0 | 2.1 % | 2.7 % | 4 of 4 |
| Inferno 86, Sodegaura | 8.8 / 4.3 / 4.6 / 4.4 | 7.7 / 9.7 / 7.6 / 7.6 | 0.6 % | 1.3 % | 3 of 4 |
| KK-SII, Suzuka | 8.0 / 3.3 / 6.3 / 2.3 | 4.7 / 1.1 / 4.1 / 0.8 | 1.3 % | 0.7 % | 0 of 4 |

**Two arms are compared on the stints both have** (`pressure_gate.attain`).
It matters in one place: the KK-F at Tsukuba, where the simulation has an
entry for the GPS-derived days only (5 or 6 of the 9 to 13 stints), so
"another day" in its rows above is the yardstick on those stints and not
the one in the yardstick's own table. (The first run of the gate set the
simulation's stints against the yardstick over all of its own; that gave
11 of 20 with nothing given where equal rows give 10. No other count
moved.)

Every arm beside the yardstick (cells no further from the day than
another day is, of 20; the distance in the median scored cell, mbar; the
cells in which the simulation is nearer the day than the arm, and
further, on the stints the two share):

| arm | `heat` | mbar | sim nearer / further | `chain` | mbar | sim nearer / further |
|---|---|---|---|---|---|---|
| **the simulation** | **13** | **3.4** | | **10** | **6.9** | |
| another day, on the simulation's stints | | 4.4 | | | 7.6 | |
| another day, on all of its own (the yardstick as recorded) | | 4.1 | | | 7.0 | |
| the ceiling: the cell's logged entry without the day | 9 | 3.8 | 13 / 7 | 12 | 5.4 | 12 / 8 |
| the car's other layouts, unscaled | 3 | 20.1 | 19 / 1 | 5 of 12 | 18.4 | 11 / 1 |
| the fallback as coded | 1 | 52.6 | 17 / 3 | 3 | 65.0 | 19 / 1 |
| planted: the day's heat × 1.05 | 1 | 10.4 | 20 / 0 | 4 | 10.9 | 17 / 3 |
| planted: the day's heat × 0.95 | 1 | 10.5 | 20 / 0 | 4 | 11.0 | 17 / 3 |
| planted: another car's parameters | 5 | 8.9 | 16 / 4 | 9 | 9.7 | 15 / 5 |
| planted: 0.85 of the power | 11 | 4.0 | 11 / 9 | 11 | 7.4 | 12 / 8 |
| planted: the simulation's lap time + 1 % | | | | 8 | 10.0 | 16 / 4 |
| planted: the simulation's lap time − 1 % | | | | 15 | 5.7 | 9 / 11 |

- **At a given lap time the simulation is as near the day as another
  logged day is**: 13 of 20 cells, 3.4 mbar from the day's own
  recommendation where another day is 4.4. With neither the car nor the
  track having seen the cell 7 of 12 (3.7 against 4.4); with only the car
  unseen 6 of 8 (3.4 against 4.3). It is nearer the day than the cell's
  own logged entry built without the day in 13 cells and further in 7.
  This is E's finding carried through the fold's constants, shares and
  frame factor into bar: nothing was lost on the way.
- **With nothing given it is on the yardstick too**: 10 of 20 cells, 6.9
  mbar where another day, given nothing either, is 7.6. Neither seen: 7
  of 12 (8.9 against 8.7); only the car unseen: 3 of 8 (4.5 against
  6.1). The cell's own logged entry without the day passes 12 (5.4 mbar);
  the simulation is nearer the day than it in 12 cells and further in 8.
- **The count is five groups and the groups do not agree, which is what
  the plan said of it.** With nothing given the Inferno 86 at Fuji passes
  on all four tyres because two days there are 20 to 25 mbar apart, not
  because the simulation is near (9 to 19 mbar); the KK-SII at Suzuka
  fails on all four with distances of 2 to 8 mbar because its days there
  repeat to 1 to 5. The distances are the result.
- **The lap time the fed use supplies is as good as another day's**: 0.6
  to 2.1 % from the day's in the scored cells, where another day's is
  0.7 to 2.7 % and the ceiling's 0.7 to 1.7 %. One per cent of lap time
  is 4.3 to 8.2 mbar at flying lap 5 by cell, so it is most of the
  distance with nothing given: 3 to 7 mbar on the formula cars and at
  Sodegaura, 17 at Fuji. It is slow on the formula cars (+0.5 % on the
  KK-SII at Tsukuba, +1.3 % at Suzuka, signed), and moving the
  simulation's lap time 1 % faster scores better
  (15 cells, 5.7 mbar) while 1 % slower scores worse (8, 10.0).
- **The misses, by where they point.** *The lean by side* (#313, the
  driver's use by corner): on the formula cars at Tsukuba the left tyres
  read 3 to 7 mbar less heat than the day's and the rear right 2 to 7
  more (KK-SII +5.0 / +0.6 / +3.2 / −6.5; KK-F +7.3 / +2.5 / +3.2 /
  −2.1), the sign E wrote down; at Suzuka the rear right reads 6.4 more.
  *The track or the chain at one layout*: the Inferno 86's rear left at
  Fuji, 6.4 mbar more heat (E: +2.4 % there). *The fed use*: the KK-SII
  at Suzuka, 1.3 % slow, which is 7 mbar on every tyre. Nothing points at
  the car model: every cell of one car does not miss alike.
- **What the gate detects, and what it does not.** A heat 5 % off is
  seen in both forms (1 and 4 cells of 20; 10 to 11 mbar from the day,
  further than the simulation in 20 and 17 cells), and so are another
  car's parameters (5 and 9 cells; 9 to 10 mbar, further than the
  simulation in 16 and 15) and a lap time 1 % slow. **A car with 0.85 of the power is not told from the fitted car in
  either form** (11 cells in both; 4.0 and 7.4 mbar): at a given lap time
  the heat is set by the track and the lap time, as in E, and with
  nothing given the wrong car's fed use is read from the same logged laps
  and hands back about the same lap time. The expectation written before
  the run had it showing with nothing given; it does not, and the gate
  cannot detect a wrong power.
- **Against what it replaces** the simulation is nearer the day than the
  car's other layouts in 19 of 20 cells at a given lap time and 11 of 12
  with nothing given, and nearer than the fallback as coded in 17 and 19
  of 20. The expectation was every cell; one or two are lost to each (a
  single tyre where the stand-in happens to sit close).
- **The entry as the calculators read it.** With the table's fourteen
  knots as the curve's nodes, read linearly as the ports read a curve,
  the recommendation is within 0.24 mbar of the table read log-log. On
  nine nodes from 0.75 to 1.5 × the entry's lap time, the grid of a
  logged curve, it is up to 4.3 mbar off (the Inferno 86's rears at Fuji
  3.0 and 2.5, Motegi 2.6 to 4.3). The ports interpolate whatever nodes
  an entry has, so a simulated entry is written with its knots.
- **Flying lap 10**: 13 of 20 at a given lap time (4.4 mbar against 5.7)
  and 11 of 20 with nothing given (9.4 against 10.5).
- **The listed cells**, with no yardstick: the Inferno 86 at Motegi (two
  days) 1 to 9 mbar at a given lap time, its fronts +9.2 and +9.4 (E: its
  fronts read 4 to 6 % low there), and 14 to 31 with nothing given (the
  fed lap time is 2.9 % slow, read against a fed use carried from
  Sodegaura and Fuji where Motegi's own two sessions read 0.93); the
  Inferno 86 at Tsukuba 0.8 to 3.2 and 0.4 to 4.2; the KK-SII at Fuji
  2.8 to 4.5 and 3.2 to 12.7.
- **Pressure against the TPMS** (S2, flying lap 5, on the rows the
  simulation has, MAE in mbar): the simulation 33 to 69 by cell (median
  43), the day's own heat 33 to 71 (43), the ceiling 33 to 70 (41), the
  cell with only its date left out 30 to 62 (33). The day's own heat
  scaled down by 5 % scores *better* in three of the five cells (the two
  at Tsukuba and Sodegaura, by 4 to 5 mbar) and worse at Suzuka and
  Fuji; scaled up it is worse in four. So S2 moves by 4 to 6 mbar for a
  heat error the recommendation moves by 10, in a direction set by the
  cell's constants: reported, not counted. The simulation's bias is the
  day's own heat's within 4 mbar in every cell (+0.2 against −2.5 at
  Suzuka, +32.5 against +29.3 on the KK-SII at Tsukuba, −45.4 against
  −46.2 at Fuji): what is left against the TPMS is the pressure model's
  (#262), not the heat's.

*The calculator as it is today, by the same measure* (asked by the owner
on 2026-11-25; `pressure_gate.report_current`). The calculator reads one
entry per pooled car and track. It is scored three ways with the same
pairing, rows and cells: at a cell it has no logged laps for (the
fallback as coded, on the cell folds); at a cell it has, on days whose
date was left out of the fit (the ten leave-date-out folds, every other
session's laps put through the fold's model so that another day stands
beside it on the same stints); and as shipped (`tire_model.json`, which
has seen the day, so that row is in sample).

| | at a given lap time | another day | nothing given | another day | against the TPMS |
|---|---|---|---|---|---|
| **the simulation, at a cell without logged laps** | **3.4** (13 of 20) | 4.4 | **6.9** (10 of 20) | 7.6 | 43 |
| the calculator today, at a cell without logged laps | 52.6 (1 of 20) | 4.1 | 65.0 (3 of 20) | 7.0 | 87 |
| the calculator today, at a cell with logged laps, the date left out | 3.5 (7 of 20) | 3.5 | 7.0 (7 of 20) | 5.8 | 33 |
| the calculator as shipped, the day in its fit | 3.4 (9 of 20) | 3.5 | 6.6 (9 of 20) | 5.9 | 32 |

(mbar, the median scored cell; cells no further from the day than
another day; the last column is hot-pressure MAE at flying lap 5 and does
not rank a heat source.) By cell, the four tyres' median:

| raw car, track | `heat`: simulation, cell unseen | calculator, cell unseen | calculator, cell logged | `chain`: simulation, cell unseen | calculator, cell unseen | calculator, cell logged |
|---|---|---|---|---|---|---|
| KK-SII, Tsukuba | 4.6 | 26.5 | 3.6 | 8.0 | 64.2 | 5.0 |
| KK-F, Tsukuba | 2.9 | 30.8 | 4.3 | 4.6 | 76.6 | 8.0 |
| Inferno 86, Fuji | 4.1 | 40.3 | 7.4 | 13.4 | 38.4 | 11.5 |
| Inferno 86, Sodegaura | 3.2 | 112 | 3.8 | 4.5 | 25.6 | 9.9 |
| KK-SII, Suzuka | 3.6 | 150 | 2.1 | 4.8 | 92.9 | 4.7 |

- **Where the calculator has no logged laps, the simulation is ten to
  fifteen times nearer the day**: 3.4 mbar against 52.6 at a given lap
  time and 6.9 against 65.0 with nothing given, in every one of the five
  cells (26 to 150 mbar for the fallback at a given lap time). Against
  the TPMS, 43 mbar against 87.
- **A simulated entry at a cell nobody has logged recommends as near the
  day as the calculator's own logged entry does at a cell it has**: 3.4
  against 3.5 mbar at a given lap time and 6.9 against 7.0 with nothing
  given. By cell it is nearer in three of five at a given lap time (the
  KK-F at Tsukuba, Fuji, Sodegaura) and further at the KK-SII's two; with
  nothing given nearer at the KK-F's Tsukuba and at Sodegaura, level at
  Suzuka, further at the KK-SII's Tsukuba and at Fuji. The logged entry is
  the pooled car's median lap, which costs the KK-F 8 mbar with nothing
  given.
- **The yardstick is passed by neither more than half the time**, and by
  the logged entry less often than by the simulation (7 of 20 in both
  forms against 13 and 10): "as near as another day" is a bar an entry
  made of the cell's own logs sits on, not under.
- **What a logged cell has that a simulated entry cannot give is the
  constants**: against the TPMS the calculator with the cell in its fit
  is at 33 mbar and the simulation at a cell left out at 43, the same 43
  the day's own heat scores under those constants (#262). The two rows
  are through different fits (the cell left out, the date left out), so
  their mbar are set beside their own yardsticks and not against each
  other tyre by tyre.
- As shipped, in sample, the calculator is 3.4 and 6.6 mbar from the day
  (9 of 20): leaving the date out costs it 0.1 to 0.4 mbar.

*What F0 was to settle for the export.* Its clock is the chain's own:
each knot's flying lap and out-lap as one-second bins of speed and of the
four non-negative parts of the accelerations (`lapsim/export.py`), which
is everything `lap_heat` reads, so any fit's shares split an export to
the table's own heat exactly. A table of fourteen knots is 155 kB of JSON
at Tsukuba and 350 kB at Suzuka, against 0.5 to 1.5 MB for the node
speeds, and takes under 0.1 s to build and export. A refit's shares move
the carried frame factor by 0.1 % or less in three of four folds and by
up to 1.2 % on a tyre where leaving the cell out moves `g_transfer` from
2.49 to 1.04 (the formula cars without Tsukuba), so the factor is read
again from the laps' bins at each retrain and is not a number to store.

*Reported beside* (each is in the same report; nothing above rests on
one).

- **The wet and damp cells** (`pressure_gate.wet_arms`): a wet or damp
  day's own recommendation against the simulation's, which is the dry
  table read at the wet lap time (typed in, #261) with no wet term (gate
  D2), at a given lap time. The yardstick in the wet is another session
  of the *same* day, same raw car, powertrain, kind of source and
  condition, as in D2 (wet days do not repeat). Twenty-three sessions in
  five cells of a raw car at a track; three cells (twelve with the tyres) have
  a same-day pair, on one or two days each, so none reaches three days
  and none is "scored". The simulation is no further from the wet day
  than a run of the same wet day in 9 of those 12 cells, 2.4 mbar where
  the same-day run is 6.0; with the car's wet and damp sessions at every
  other track left out of the fit as well (track and condition both new)
  9 of 12, 2.8 against 6.4. On every wet stint, paired or
  not, the simulation is 0.9 to 9.0 mbar from the day, the day's own heat
  scaled by 5 % 5 to 12, and the fold model as it stands 4 to 216. The
  KK-SII's rear right at Tsukuba reads 7 mbar more heat in the wet, as it
  does in the dry.
- **The chronological first visit** (`pressure_gate.first_visits`): the
  first scored day of a car at a track where the car had run elsewhere
  before, with the pressure model fitted on the sessions before that
  date only, beside the same sessions under the cell's fold. Four visits
  with rows, eight sessions (the formula cars at Fuji, 2026-02-26, and at
  Suzuka, 2026-07-19; the Inferno 86 at Sodegaura, 2025-07-24, and at
  Motegi, 2025-10-04). **Leaving a cell out does not flatter the figure**:
  the simulation's distance from the day at a given lap time is 4.1, 2.0,
  3.0 and 3.1 mbar (mean over the day's stints and tyres) under the fit
  of what was logged before, and 3.8, 1.9, 2.8 and 2.7 under the cell's
  fold; with nothing given 6.6, 4.3, 6.0 and 14.4 against 6.8, 4.8, 5.8
  and 14.5. Against the TPMS the earlier fits are worse where little had
  been logged (the formula cars at Fuji 123 mbar against 78, Sodegaura 70
  against 54) and the same elsewhere (44 against 48, 62 against 61),
  whatever the heat source: that is the constants, not the heat. The
  simulated side is the cell fold's in both (the car fitted without the
  layout, not without its later days), so this checks the pressure
  model's side of the fold only.
- **The pooled simulated entry** (`pressure.pooled_entry`,
  `pressure_gate.pooled_sim_arms`): per tyre the geometric mean of the
  raw cars' simulated entries at one reference lap time, each of its
  latest powertrain, weighted by the raw car's share of the pooled car's
  usable laps in the fold's fit. With Tsukuba left out the only formula
  car with laps in the fit is the KK-SII, so the KK-F's days are read
  through the KK-SII's entry: at a given lap time 1.1 to 4.1 mbar from
  the day where its own simulated entry is 2.1 to 7.3 (no further on
  three tyres), and with nothing given 1.8 to 16.1 against 1.9 to 12.0,
  2 to 4 mbar further on three tyres, because the pooled lap time is 1.0
  to 2.1 % slower than the KK-F's days. With Suzuka left out the KK-SII is read through an
  entry weighed with the KK-F's laps and is no further for it (0.7 to
  6.0 against 2.3 to 8.0 with nothing given). The cost of pooling is the
  lap time, as for the logged entry.

*Not built*: adoption (the committed export, the entries with their
`source` flag in `tire_model.json`, the calculators' line and their
attribution), which waits for the owner's call on this report, as
planned; and #260's changes to `predict.py` (the 90 s constant, the
prior), which #208 lists as not in F. The fold cache is of the bytes of
the fit's source and of the loaders it reads the timeseries through
(`tire_etl/kinematics.py`, `dataset.py`); adding those two to the key
refitted all seventeen folds and gave the yardstick back to the digit.

**The goal restated by the owner, and the shootout (2026-11-25; the plan,
not yet run).** Asked how the simulation compares with the calculator as
it is, the owner restated what the programme is for, in these words:
"our goal is to build a simulation based model that will replace the
current calculator"; "the goal is to delete the current calculator model
and replace it with one that does a simulation IN THE BROWSER"; "the
simulation based model is everything built in 202. It is supposed to be a
track model, a car model, a driver model, a heat model, and a tire
pressure model, all of which stack together to predict the tire pressure
growth"; and, on how it is judged, "we evaluate the calculator's accuracy
using MAE over folds of held out data". This replaces two things the plan
of 2026-11-24 said: that an entry with logged laps keeps its logged heat
(under a replacement every cell's heat is the simulated stack's), and
that the simulation reaches the calculators only as precomputed entries.
What is built today is not that yet: the lap solver is Python (numpy and
numba) with no port, the tyre pressure model is the current one (its K,
τ and κ fitted on logged heat, the simulated heat brought to it by the
frame factor), and gate F read the simulation through a pace table.

*How the replacement is evaluated.* By the calculator's own measure: the
standing holdout (`tire-predict-holdout`: folds of held-out sessions,
calculator inputs, hot pressure against the TPMS, MAE and bias per tyre,
pooled, per car and per condition), the simulation-based calculator
beside the current one on the same folds, fits and rows
(`lapsim check-pressure-holdout`). Gate F's paired-day measure is
reported beside it. One thing is known of this MAE and is to be read with
it: on these data a heat 5 % too low scores better in three of five
cells (gate F, S2; #208 findings 1 and 2), so a small difference in MAE
between two heat sources is not by itself evidence for either.

*The shootout: a precomputed pace table against a direct simulation.*
The owner's instruction of 2026-11-25. Both contestants are the same
stack (track, car, driver, heat, pressure); they differ in where the lap
is driven.

- **A, the precomputed pace table.** Per car, powertrain and layout the
  fourteen knots are driven ahead of time and shipped as their laps'
  one-second bins (`lapsim/export.py`, 155 to 350 kB a table). The
  calculator holds no solver: it reads the heat of a lap time between the
  knots and splits it with the fit's shares.
- **B, a direct simulation.** The calculator holds the track, the car
  model and the solver and drives the lap it is asked for: at the lap
  time typed in (the knob solved for it), or at the car's fed use where
  none is typed, with the out-lap driven from the pit exit's join; the
  heat chain then runs on that lap. In the browser this is the solver
  ported to JavaScript (and to C# for the desktop and Android heads).

What B can take that A cannot, and so where the two can differ beyond
A's interpolation (0.9 % of a tyre's heat at most between knots, 0.24
mbar in gate F): the air of the day (the calculators already ask for the
ambient temperature; a table is in standard air), a powertrain, shift
point or gear set other than the table's, a lap time past the table's
ends, and the wet as its own use. What A has that B has not: no solver
port to write and to hold to the Python reference, and nothing to
compute in the client.

The shootout, on the same folds, fits and rows throughout:

1. **Accuracy, by the calculator's measure**: MAE over the held-out
   folds for the current calculator, A and B. B is driven by the Python
   solver, which is what a faithful port computes, so accuracy is settled
   before a port is written. B is given what a user types (track, car,
   compound, condition, ambient temperature, lap time); a second arm, B+,
   is also given the day's logged shift point and gear set, to show what
   more inputs could buy.
2. **Accuracy, by gate F's measure**: the recommended cold pressure
   against the day's own with another day beside it, A and B.
3. **A against B directly**: the difference between their two
   recommendations on the same rows (median and worst), and where it
   comes from (interpolation, the air, the out-lap, the ends of the
   table).
4. **Coverage**: the held-out sessions each can answer, and why not
   where not (no lap at the lap time, outside the logged support, no
   frame factor).
5. **Cost**: what each ships (bytes per car and layout), what each
   computes per prediction, and for B the port: the kernel in JavaScript
   timed in the browser's engine on one lap solved for a lap time (the
   compiled Python form takes 1 to 4 ms, the numpy form 0.5 to 4 s), and
   the parity it has to hold with the Python solver, whose two forms are
   held identical to the last bit today.

Expected before the run: A and B within about 1 % of heat of each other
at a lap time inside the table, so 1 to 3 mbar apart and not told apart
by an MAE of 30 to 70 mbar; any difference that shows will come from the
air and from lap times near the table's ends. The call between them is
the owner's on the report; nothing is adopted, ported or deleted before
it. The form of adoption written on 2026-11-24 (entries materialised
from a committed export) is A's and waits on it too.

**The standing holdout on simulated heat, and the shootout's first item
(2026-11-25; `pressure_gate.check_pressure_holdout`, `just tire-lapsim
check-pressure-holdout`; 7 min 46 s on four workers of two threads, the
folds side by side).** The calculator's own evaluation: 2 sessions per
(track, car, condition) bucket per fold, 15 folds before the buckets run
out, 128 sessions, 5,751 lap × tyre rows; calculator inputs (the session's
25th-percentile lap time typed in). Each fold is fitted once and scored
by four calculators on the same rows: the current one (the fold model as
it stands: 0.0387 bar on fold 0, the standing holdout's own figure) and
the simulation-based one as a pace table, as a lap driven directly for
the typed lap time in the session's air, and the same with the day's
logged shift point and gears. **A fold's held-out sessions, and only
those, are out of every model under the simulated entry** (owner: "make
sure the holdout is applying to the training of the underlying model as
well"; "you should just be excluding the sessions that are not used"):
the pressure model's fit, the car model with its limit, braking use and
typical use (`held_out_drive`), the track's curvature (`track_without`),
the frame factor and the fed use. A fold does not take out which laps
count as build laps, the track's reference line, bank, grade and pit
exit, or the compound label of an unlabelled session (the standing
holdout's own, an EM on the full dataset).

| hot-pressure MAE, bar | current calculator | simulation, pace table | simulation, driven directly | with the day's shift point and gears |
|---|---|---|---|---|
| **pooled** (5,751 rows) | **0.0437** | **0.0446** | **0.0447** | 0.0447 |
| FJ (3,307) | 0.0353 | 0.0366 | 0.0366 | 0.0366 |
| Inferno 86 (2,444) | 0.0551 | 0.0555 | 0.0558 | 0.0558 |
| dry (4,650) | 0.0434 | 0.0446 | 0.0447 | 0.0447 |
| damp (224) | 0.0732 | 0.0752 | 0.0751 | 0.0751 |
| wet (877) | 0.0377 | 0.0372 | 0.0371 | 0.0371 |
| pooled bias | +0.0070 | +0.0096 | +0.0096 | +0.0096 |

- **The simulation-based calculator is about 1 mbar behind the current
  one**: per session +0.7 ± 0.3 mbar for the table and +0.8 ± 0.3 for the
  direct lap (mean and its standard error over 128 sessions), nearer the
  TPMS on 63 and 57 of them. Every arm answers all 128 sessions.
- **The table and the direct lap are not told apart**: their predictions
  differ by 0.8 mbar in the median row and 4.3 at the 95th percentile,
  and the direct lap's MAE less the table's is +0.1 ± 0.1 mbar. The air
  of the day and the day's shift point and gears buy nothing that this
  measure sees.
- **At levels of pace** (the laps above a percentile of pace among the
  held-out laps of the same raw car, track and condition; MAE in bar):

  | level | laps | current | table | direct |
  |---|---|---|---|---|
  | all | 1,469 | 0.0437 | 0.0446 | 0.0447 |
  | P75 | 371 | 0.0411 | 0.0417 | 0.0419 |
  | P90 | 150 | 0.0394 | 0.0401 | 0.0405 |
  | P95 | 80 | 0.0410 | 0.0411 | 0.0417 |

  The gap does not close on fast laps. On dry laps it is a steady 1 to
  1.5 mbar on the FJ at every level and grows on the Inferno 86 from
  under 1 mbar to 2 to 3 at P90 (0.0486 against 0.0507 and 0.0517). Every
  arm is fed the session's 25th-percentile lap time on every lap, so a
  fast lap here is one driven harder than its plan, and every arm
  over-predicts more there (+17 mbar dry at P90 for the current one, 3 to
  4 more for the simulation).
- **To be read with it**: this MAE moves by 4 to 6 mbar for a heat 5 %
  off and prefers less heat in three of five cells (gate F, S2), and the
  constants under every arm were fitted on logged heat.

*The same question by gate F's measure* (the ten leave-date-out folds,
dry days, the held-out sessions out of every model; an item of
`check-pressure`): on the cells the calculator has logged, the simulated
entry of the day's own raw car is 3.2 mbar from the day's own
recommendation at a given lap time and 6.0 with nothing given, the
calculator's pooled logged entry 3.5 and 7.0, another day 3.5 and 5.8;
the simulated entry passes 13 of 20 cells in both forms where the logged
entry passes 7, and is nearer the day in 15 cells and further in 5.
Against the TPMS on those rows it is 45.4 mbar where the logged entry is
44.1. So the two measures disagree in sign by about 1 mbar each: the
simulated entry recommends nearer what a day's own heat recommends, and
scores a little further from the TPMS.

*The KK-F fitted apart from the KK-SII* (the owner, 2026-11-25: "let's
split out the KK-F and see if it gets better"; `check-pressure-holdout
--split KK-F KK-F-V2`, `pressure_gate.split_pooling`, an experiment's
switch that writes nothing). The same folds and rows; the KK-F and the
KK-F-V2 are one pooled car with its own K, τ, κ, shares and entries, and
the "FJ" that is left is the KK-SII alone. Hot-pressure MAE in bar,
pooled fit and then split:

| | rows | current calculator | simulation, pace table |
|---|---|---|---|
| every car | 5,751 | 0.0437 → 0.0436 | 0.0446 → 0.0446 |
| KK-SII, all | 2,175 | 0.0318 → **0.0302** | 0.0331 → **0.0316** |
| KK-SII, dry | 1,802 | 0.0312 → 0.0289 | 0.0327 → 0.0307 |
| KK-SII, wet | 373 | 0.0347 → 0.0363 | 0.0350 → 0.0363 |
| KK-F, all | 1,132 | 0.0421 → 0.0446 | 0.0434 → 0.0458 |
| KK-F, dry | 668 | 0.0349 → 0.0342 | 0.0365 → 0.0345 |
| KK-F, wet | 240 | 0.0328 → 0.0266 | 0.0327 → 0.0276 |
| KK-F, damp | 224 | 0.0732 → **0.0948** | 0.0752 → **0.0989** |
| Inferno 86 | 2,444 | 0.0551 → 0.0551 | 0.0555 → 0.0555 |

- **It gets better on the KK-SII and in the dry on both cars, and worse
  on the KK-F's damp sessions; in total it is level.** The KK-SII gains
  1.2 ± 0.7 mbar a session (better on 36 of 53) and its bias falls from
  +7.6 to +2.7 mbar. The KK-F in the dry gains 0.7 mbar on the current
  calculator and 2.0 on the simulation-based one, and its bias goes from
  −11.5 to +2.3; in the wet it gains 5 to 6 mbar. Its four damp sessions
  (224 rows) lose 22 mbar, which is the whole of its loss: over its 22
  sessions the change is +1.3 ± 3.6 mbar.
- **With the cars apart the simulation-based calculator is level with
  the current one on the KK-F in the dry** (0.0345 against 0.0342, where
  pooled it was 0.0365 against 0.0349), and still 1.4 to 1.8 mbar behind
  on the KK-SII.
- What it would take to keep: the damp cell of a car with four damp
  sessions needs the condition chain to fall back sensibly (it reads the
  KK-F's own thin damp fit where pooled it shared the KK-SII's), and the
  KK-F's constants rest on one track. Not adopted; the pooling in
  `warmup_table.CAR_FIT_ALIASES` is unchanged.

*A lap time under anything the models were trained on* (the owner,
2026-11-25; `check-pressure-holdout --fastest 0.25`,
`pressure_gate.fastest_sessions`). Per raw car and track with ten dry
sessions, the fastest quarter of them by the lap time a driver would
type is held out of every model, one fold a cell: 28 sessions in five
cells. **The data cannot ask for much**: the held-out sessions' typed lap
time is 0.4 to 0.8 % under the fastest typed lap time left in training
(median 0.5 %), and only 6 of the 28 are under every single lap left at
the cell (by 0.0 to 1.0 %).

| hot-pressure MAE, bar | rows | current calculator | simulation, pace table | simulation, driven directly |
|---|---|---|---|---|
| the 28 fastest sessions | 1,192 | 0.0387 | 0.0401 | 0.0403 |
| FJ, Tsukuba | 643 | 0.0308 | 0.0331 | 0.0334 |
| FJ, Suzuka | 112 | 0.0355 | 0.0363 | 0.0364 |
| Inferno 86, Fuji | 181 | 0.0490 | **0.0465** | **0.0458** |
| Inferno 86, Sodegaura | 256 | 0.0528 | 0.0548 | 0.0557 |
| the 6 sessions under every lap left | 232 | 0.0443 | 0.0449 | 0.0458 |

- **The simulation is not better at a pace it was not trained on, as far
  as these data reach**: +1.2 ± 0.7 mbar a session against the current
  calculator, the same gap as in the standing folds. It is nearer in one
  cell of four, the Inferno 86 at Fuji (2.5 to 3.2 mbar), the cell whose
  held-out sessions are furthest under their training pace (0.8 %).
- The current calculator's pace curve is itself a simulation of a faster
  lap (each logged lap scaled to the target lap time), so half a per
  cent past the logged pace is inside what it already does. A test of a
  lap time several per cent under anything logged does not exist in the
  dataset: the fastest quarter of a cell's sessions is that close to the
  rest.
- One of the 28 has no directly driven lap: the Inferno 86's fastest
  session at Fuji asks for a lap time the car fitted without its fastest
  sessions cannot make under the cap of a use of 1. The table answers it
  from its fastest knot, flagged; a direct simulation has no lap to give.

*The same with most of the sessions held out* (the owner, 2026-11-25:
"you can create the real test by splitting the data properly so that we
hold out a lot of sessions"; `--fastest 0.5`, `0.8`, `0.9`). Holding out
more of a cell's fastest dry sessions leaves only its slowest in every
model, and the held-out lap times go well under them:

| fastest held out | sessions | under the fastest typed lap time left (median; furthest) | under every lap left | left per cell at the track | current | pace table | direct |
|---|---|---|---|---|---|---|---|
| 25 % | 28 | 0.5 %; 2.4 % | 6 | 8 to 22 | 0.0387 | 0.0401 | 0.0403 |
| 50 % | 53 | 0.8 %; 3.7 % | 16 | 5 to 15 | 0.0411 | 0.0423 | 0.0427 |
| 80 % | 85 | 2.0 %; 5.8 % | 62 | 2 to 6 | 0.0459 | 0.0481 | 0.0485 |
| 90 % | 95 | 2.8 %; 5.9 % | 80 | 1 to 3 | 0.0456 | 0.0472 | 0.0480 |

(hot-pressure MAE in bar on the rows every arm has.) Per session, the
simulation less the current calculator: +0.9 ± 0.5 and +1.1 ± 0.6 mbar
(table, direct) at 50 %, +1.2 ± 0.7 and +1.4 ± 0.7 at 80 %, +1.0 ± 0.6
and +1.6 ± 0.7 at 90 %.

- **The simulation is not a lot better at a lap time nothing was trained
  on; it is about 1 to 1.5 mbar behind at every split, as in the standing
  folds.** On the sessions under every single lap left in training: 61
  sessions at 80 % (0.0441 against 0.0466 and 0.0472) and 79 at 90 %
  (0.0447 against 0.0465 and 0.0475).
- **Where the extrapolation is longest it is level.** On the sessions
  more than 3 % under their training pace: 13 at the 80 % split, 0.0559
  against 0.0551 and 0.0552 (−0.2 ± 2.2 and −1.4 ± 2.5 mbar a session);
  39 at the 90 % split, 0.0463 against 0.0461 and 0.0473 (−0.7 ± 1.1 and
  0.0 ± 1.3).
- **It differs by cell, the same way at every split.** At 80 %: the
  Inferno 86 at Fuji is nearer with the simulation (0.0648 against 0.0611
  and 0.0606; −2.5 ± 1.6 and −4.0 ± 1.7 mbar, nearer on 14 of 20
  sessions); at Sodegaura further (0.0546 against 0.0597 and 0.0604;
  +3.7 ± 1.7); the FJ at Tsukuba further (0.0370 against 0.0399 and
  0.0406; +2.3 ± 0.8); the FJ at Suzuka level, and at 90 %, with one
  session left there, the current calculator is off by +22 mbar in the
  mean where the simulation is at −4 (0.0399 against 0.0352).
- **The direct lap runs out before the table does.** One of the Inferno
  86's Fuji sessions has no directly driven lap at every split (the car
  fitted from the slow sessions cannot make the lap time under the cap);
  the table answers it flagged, and flags 9 of 85 and 17 of 95 as outside
  its support.
- What this test is: a pace not seen *at that track*. The car's sessions
  at its other tracks stay in every model, and at 80 and 90 % a cell's
  training is a handful of its slowest dry sessions, which may be slow
  for reasons that are not pace.

**Heat in the tyres' frame, so that no frame factor exists (#376;
2026-11-26; the default since the owner's promotion the same day).** The
pressure model's heat had been the sensor verdict's series, the
accelerometer or a GPS-derived fallback per 20 s window. A simulated lap
therefore had to be brought onto that series by a frame factor per (car,
layout, kind of source). The holdout read that factor by the held-out
session's own kind of sensor, which no driver knows before a session.
The pressure model now rebuilds every session's heat input in the frame
a simulated lap is in, so that the fit and the simulated lap meet with a
factor of 1 and no sensor verdict is read by any heat. The sensor's
series is kept for comparison (`--heat-frame sensor`), and only there
does a frame factor exist: every frame factor quoted earlier in §2.12
(gates D1, E1/E2 and F) is a sensor-frame run from before 2026-11-26.

- **The series** (`lapsim/tyres_frame.py`, query time, nothing in the
  extractor):
  - **Lateral:** what the tyres hold on the road's bank, `V·Δψ/Δt / g`
    from the Doppler course (`frame.ground_series`) through
    `bank.to_tyres_frame`.
  - **Longitudinal:** `ΔV/Δt / g` plus the track's grade, on the
    position clock, from the committed track JSON.
  - **Coverage:** the whole session, out-lap, in-lap and pit lane
    included.
  - **Laying a fix on the track:** each fix goes onto the track's line
    by its nearest node (a KD-tree) and `track.project_onto`. A fix more
    than 15 m from the line (pit lane, paddock), or on a layout without
    a committed track, is in the ground frame with bank and grade 0.
  - **Standing still:** the car's own acceleration is 0 where it
    stands.
  - **Gaps:** a row between two usable values at most 1.0 s apart
    (`FILL_MAX_S`, one bin of the chain) is linear between them. The
    Doppler course is undefined for a few fixes now and then, and each of
    its 0.5 s differences then lacks a value for about 0.3 s, at speed.
  - **Edges:** a row before the session's first, or after its last,
    usable value by at most 1.0 s holds it (a file that starts or ends at
    speed).
  - **Holes:** a longer stretch, and any stretch over a fix the extractor
    marked as a GPS spike, is a hole.
  - **Holes left:** across the pooled cars' 175 TPMS sessions, 40 laps in
    19 sessions, all real GPS gaps or discard windows.
    - Without the bridging and the edges (the first version, 2026-11-26,
      found by the KK-F-V2 Fuji test) it was 149 laps in 97 sessions,
      the last lap of a file that ends at speed among them.
  - **Clock:** it is handed to the fit on the kinematics clock
    (`tyres_kinematics`: `load_kinematics`' own rows with `lat_g` and
    `long_g` replaced, source `tyres`, `on_line` per row).
  - **Check against the existing chain:** per lap it reads within
    0.1–0.5 % of the gates' `heat_gate.tyres_frame_lap_bins` on eight
    sessions (four layouts, both kinds of sensor). The difference is the
    kinematics clock against the position clock in the 1 s bins.
  - **What the logged series reads over it:** the frame factors, again.
    Logged over this series is 1.025 / 1.026 / 1.051 / 1.050 on an
    Inferno 86 Sodegaura session, 1.07 / 1.07 / 1.06 / 1.06 on a KK-SII
    GPS session at Tsukuba, and 1.00 / 1.07 / 0.90 / 0.94 on a KK-SII IMU
    session at Fuji: the FJ's low IMU rears at Fuji are the
    accelerometer.
- **The switch** (`tire_model/heat_frame.py`):
  - The frame in force is `tyres` outside any `heat_frame.using(...)`;
    `using("sensor")` makes every kinematics load of the fit and of the
    holdout's preparation hand out the sensor's series instead
    (`statespace._session_timeseries`, cached per frame;
    `pace_sim.load_lap_samples`). A `heat_frame=None` argument means the
    frame in force.
  - `build_warmup_table(..., heat_frame=)` and
    `validate._fit_fold(..., heat_frame=)` fit on it.
  - The model records `heat_input.frame` (absent: `sensor`, so a sensor
    fit is byte for byte what it was: fold 0 refitted equals its cached
    model in every key but `fit_at_utc`).
  - `attach_lap_heat` reads the frame from the model it is given.
  - `pressure_gate.served_factor` is 1 on every tyre for such a model.
  - A tyres-frame fold has its own cache key (`_tyres_signature`: the
    code that builds the series and the tracks' bytes).
  - CLI: `--heat-frame tyres|sensor` (default `tyres`) on
    `tire-predict-holdout`, `tire-build-warmup-table` and `tire-lapsim
    check-pressure-holdout` (rows in `holdout_rows.parquet`, the sensor's
    in `holdout_rows__sensor.parquet`; before the promotion the default
    run's rows were the sensor's and the tyres' run wrote
    `holdout_rows__tyres.parquet`).
  - The other lap-sim gates compare with production, so they read heat
    in the frame of the committed `tire_model.json`
    (`heat_frame.model_frame`): the sensor's until it is retrained in
    this frame. `holdout_rows` refuses a fold whose frame is not the one
    in force, and gate F leaves the model as shipped unscored when its
    frame is not the run's.
  - **The kind of sensor is a label only.** `tyres_kinematics` keeps the
    sensor verdict's `lat_source` as `sensor_source`, which
    `heat_gate.sensor_lap_bins` reports as the lap's `source` (imu / gps
    / mixed): the holdout's rows by kind of logger and gate F's pairing
    "of the same kind of source" read it, no heat does, and every kind
    is served a factor of 1.
- **Where the frame cannot be built** (`heat_frame.fallbacks`, recorded
  in the model as `heat_input.frame_fallbacks`, with a warning once per
  layout or session):
  - a layout with no committed track in `data/lapsim/tracks/` (today
    `motegi_east`, `fuji_short`, `suzuka_west` and `minami`; two
    one-lap sessions among them reach the fit's laps, an Inferno 86's at
    `fuji_short` and one at `motegi_east`, pooled into their venue's
    entry) is in the ground frame with bank and grade 0;
  - a session with no layout or no GPS geometry has no heat in this
    frame and is left out of the fit (the four with no layout today have
    no track either and were never in it);
  - a dataset with no sessions table is a `HeatFrameError`, never a
    skipped file.
- **Cost:** a fold fits in about 70 s against 44 s (1.5 GB). The
  15-fold `check-pressure-holdout --heat-frame tyres` takes 12 min 14 s
  on four workers, at 2.4 GB a worker.

The gate, written in #376 before any tyres-frame row was scored:
- the 15 standing folds refitted on this series (fold ids identical to
  the sensor run), the table served with a factor of 1, the
  calculator's entries from the same series;
- both calculators within ±0.30 mbar of their current values:
  - the calculator against 43.71;
  - the table against today's table with a source-blind factor
    (simfix's `table_gm`, 44.08), with 44.64 reported beside;
- κ_FJ's sd across the folds below today's;
- the FJ-at-Fuji rear bias not grown by more than 1 mbar.

The gate's "today" is the standing holdout of 2026-11-25, the values
it was written against; it is scored as written. That run predates
#388's count of a first-lap anchor's flying laps, which the tyres'-frame
run has, so #376's own effect is read against A-master below.

| check | today (2026-11-25) | tyres' frame | |
|---|---|---|---|
| calculator MAE (mbar) | 43.71 | 43.60 | pass |
| table, factor 1 | 44.08 (`table_gm`; 44.64 as served) | 43.76 | **fail**, 0.02 past the band on the good side |
| κ_FJ across 15 folds | sd 2.51e-7, 9e-10 to 1.14e-6 | sd 2.55e-7, 2.2e-7 to 1.36e-6 | **fail** |
| FJ Fuji rears, bias (51 rows, wet) | calculator −49.96, table −41.72 | −44.56, −41.51 | pass |

The first version of the series (2026-11-26, without the bridging and
the edges) read 43.66, 43.80 (−0.28: inside the band), sd 2.54e-7, and
−44.97 / −41.93. Its table check passed, the κ check failed.

- **κ:** it no longer runs into 0 on any fold, but its spread did not
  shrink. #255's reading, that κ_FJ follows the fleet's GPS share, is
  not borne out by the spread.
- **The table:** it improved by more than the band allowed. The band was
  written for a change expected to be about 0, so it fails by the
  letter.

**#376 alone.** A-master is master 47426819c324 (with #388) in the
sensor's frame on the same 15 folds and 5,751 rows. MAE in mbar; the
paired difference is #376 less A-master per lap and tyre, ± session /
date-clustered SE; "per session" is the mean of the sessions' own
differences, ± its date-clustered SE.

| arm | A-master | #376 | paired | per session | nearer, of 128 |
|---|---|---|---|---|---|
| calculator | 43.79 | 43.60 | −0.18 ± 0.16 / 0.19 | −0.24 ± 0.20 | 85 |
| table as served | 44.73 | 43.76 | −0.97 ± 0.27 / 0.35 | −0.76 ± 0.29 | 81 |
| direct | 44.83 | 43.84 | −0.99 ± 0.27 / 0.35 | −0.77 ± 0.28 | 85 |
| direct_setup | 44.84 | 43.85 | −0.99 ± 0.27 / 0.35 | −0.77 ± 0.28 | 85 |

| calculator / table | A-master | #376 | paired |
|---|---|---|---|
| FJ, calculator | 35.37 | 35.18 | −0.19 ± 0.22 / 0.24 |
| FJ, table | 36.71 | 35.17 | −1.54 ± 0.44 / 0.55 |
| Inferno 86, calculator | 55.17 | 54.99 | −0.18 ± 0.24 / 0.30 |
| Inferno 86, table | 55.59 | 55.38 | −0.21 ± 0.22 / 0.28 |
| KK-SII, calculator | 31.80 | 31.34 | −0.46 ± 0.20 / 0.20 |
| KK-SII, table | 33.15 | 31.20 | −1.96 ± 0.52 / 0.67 |
| KK-F, calculator | 42.25 | 42.57 | +0.33 ± 0.52 / 0.55 |
| KK-F, table | 43.54 | 42.80 | −0.74 ± 0.76 / 0.90 |
| dry (100 sessions), calculator | 43.40 | 43.19 | −0.21 ± 0.15 / 0.18 |
| wet (24), calculator | 38.23 | 37.73 | −0.50 ± 0.24 / 0.25 |
| damp (4), calculator | 73.58 | 75.08 | +1.50 ± 2.73 / 2.99 |

The rms change in the calculator's predictions is 4.25 mbar (MDE 0.71),
in the table's 6.11 (MDE 1.03).

**The gap the simulation has to the calculator** (simulation less
calculator per lap and tyre, mbar, ± session / date SE; in brackets, the
sessions where the simulation is better). The logged-out-lap column is
read off #376's rows by an out-lap swap (simfix's `outlap_swap.py`); it
is not yet an arm of `check-pressure-holdout` and has no A-master
counterpart.

| | A-master, table as served | #376, table as served | #376, logged out-lap |
|---|---|---|---|
| pooled | +0.94 ± 0.31 / 0.40 | +0.16 ± 0.21 / 0.28 | −0.04 ± 0.14 / 0.19 (56 of 128) |
| Inferno 86 | +0.42 | +0.39 | +0.22 |
| KK-SII | +1.36 | −0.14 | −0.47 |
| KK-F | +1.29 | +0.22 | +0.21 |
| FJ, GPS-logged sessions | | +0.05 | −0.40 |
| FJ, IMU-logged sessions | | −0.08 | −0.10 |

Before #388 (the standing holdout of 2026-11-25): calculator 43.71, the
simulation with both D fixes 43.83 (+0.12 ± 0.15 / 0.22 from the
calculator), the table as served 44.64, direct 44.74; #376's first
version 43.66, 43.59, 43.80, 43.88.

What the numbers say:
- **The gap closes.** The table's gap to the calculator falls from
  +0.94 to +0.16 mbar, most of it on the FJ (−1.54 on the table against
  −0.19 on the calculator). With the logged out-lap the simulation is at
  parity, nominally 0.04 mbar ahead.
- **Within noise for the calculator.** Its −0.18 is about one SE, under
  its MDE of 0.71; the table's −0.97 is about three.
- **The two calculators move closer.** Their row-by-row prediction
  difference is 3.8 mbar rms, against 4.3.
- **The constants absorb the level.** K rises 2–5 % on every car and
  axle (the tyres'-frame heat is that much below the sensor heat on
  average). τ moves under 1 %, except the Inferno 86's dry fronts
  (−2.9 %).
- **This holdout can exercise only what it does.** The option's value
  is that a simulated entry needs no factor and no verdict at a layout
  or sensor setup it has not seen. The holdout tests seen cells.

**Promotion.** The owner turned the tyres' frame on by default on
2026-11-26, on these numbers with the gate at 2 of 4. The retrain on it
moves K by +2 to +5 %, every logged entry's heat by the same, the pace
curves, and κ; `schema_version` stays 5 (the calculators read none of
the new keys).

**The unseen Fuji day** (the KK-F-V2 on 2026-11-25, all dry, not in any
fit; mbar): the cold-pressure advice with the lap time typed is 1.5 from
the day's own for the simulation against 3.6 for the current calculator
(4 days); prefilled, 2.5 against 19.3; the hot pressure 51.5 against
51.9. GPS and IMU heat now agree within 1 % on it, against 7–10 %
before.

**The cutover: simulated entries in `tire_model.json` (#208's adoption;
2026-11-27).** The shape is
the one Segment F approved, form A: a precomputed pace table, no solver in
the client. Gate F and the standing holdout did not tell form A from the
direct solve.

- **The committed export** (`lapsim export-entries`, `lapsim/sim_export.py`,
  `data/lapsim/exports/`):
  - one file per raw car and committed track: the pace table of the raw
    car's latest powertrain (the Inferno 86's second engine, the KK-F, the
    KK-F-V2, the KK-SII), each knot's flying lap and out-lap as the heat
    chain's 1 s bins (`TableExport`);
  - the raw car's fed use, the layout's own where it has run the layout,
    else the median over its layouts;
  - the table's support (dry and wet ranges of use);
  - what it assumes, and the commit, date and dataset it was made at;
  - an index with GSI's attribution.
  - It is made as the holdout's `served` arm makes a cell
    (`pressure_gate.held_out_drive` with nothing held out). Its numbers are
    rounded to 7 significant digits (the lead): the files halve, and the
    heat moves by under a millionth.
  - No heat is stored. Every retrain splits the bins through its own shares.
  - **The tracks are those committed before the 2026-11 refresh.** The
    car models, fed uses and pace tables are on all data (through
    2026-11-25). Since #376 the fit's heat also reads the tracks' bank and
    grade, so rebuilding them is a follow-up with its own refit.
- **Placement** (`tire_model/sim_entries.place_simulated`, at
  `build_warmup_table(write_artifacts=True)` only):
  - **The track:** a raw car's export at a venue's main layout
    (`VENUES[..].default_layout`) serves that track.
  - **Only the shipped artifact:** a fold's fit never gets simulated rows,
    and `fold_model` refuses one that has them.
  - **Only the tyres' frame:** a sensor-frame model gets no simulated rows,
    and records why.
- **What a cell is served** (`lapsim/served.py`, the shipping rules):
  - **dry:** the table read log-log at the fed use's lap time, with the
    knots as its curve (`method: "sim"`, `support`);
  - **wet and damp:** served only where the model has a logged typical lap
    time for exactly that condition. The dry table is read there (#208's
    ruling), and the lap-time row stays the logged one. Elsewhere the cell
    keeps its logged answer (the lead, 2026-11-27: a dry pace must not
    become a wet prefill);
  - **out-lap:** the car's own logged out-lap wherever the condition chain
    reaches one at the track. Only where none does is it the simulated
    out-lap, which is driven at the flying knob (#377).
- **Beside, not in place of** (schema §9, `predict._entry_row`):
  - A simulated row is added beside the logged rows at its key and answers
    only its own exact condition.
  - At the exact step the order is raw sim, raw logged, pooled sim, pooled
    logged, decided by `source` (absent means logged), never by row order.
  - Every condition fallback and every averaged step (track and global)
    reads logged rows only.
  - So every cell the simulation does not serve reads exactly what it read
    before. `place_simulated` refuses to write if any moved
    (`logged_cells_moved`).
  - The FJ raw cars' rows go in #383's raw tables, so a saved FJ reads the
    pooled rows, which stay logged.
- **What the artifact says** (recorded as `simulation` and `entry_coverage`):
  - `simulation`: the export's making, the conditions, the rules, the cells
    and the attribution;
  - `entry_coverage`: per offered cell, which kind of row (sim, logged,
    averaged, none) each lookup answers with, read through the calculators'
    one chain walker (`predict._answering_rows`), with the reason where the
    simulation does not answer.
  - The calculators show GSI's attribution whenever a prediction reads a
    simulated row.
- **The arm scored is the arm shipped.**
  - `check-pressure-holdout` has a `served` arm that places each fold's cell
    with the same function and rounding. It differs from shipping only in
    using the day's powertrain, and in that every model under it holds the
    fold's sessions out.
  - `--served-steps` scores the committee's rules (simfix's `table_logout`)
    and each step from them to the shipping ones in the same run, as side
    arms kept out of the common rows: the simulated out-lap, the raw-car
    rows, the rounding.
  - `SIM_CONDITIONS` is dry, damp and wet, on the re-evaluation below.
- **Two commits** (the lead's ruling (B)):
  - The Python commit carries the code, the export and the tests, with
    `tire_model.json` untouched. predict.py reads today's artifact as
    before, pinned to the parity fixture by
    `test_the_refactored_lookups_answer_as_before_on_the_committed_artifact`.
  - The calculators' commit carries the ports, which read `source` in the
    §9 order, the artifact regenerated with simulated entries, and the
    parity fixture.
- **Rehearsal** (the saved pace tables of 2026-11-24, placed in fold 0's
  cached model, no dataset read):
  - 60 offered cells: 28 answer simulated heat, 5 of them sim-only (the FJ
    raw cars at Motegi and Sodegaura, the Inferno 86 at Suzuka); 16 are
    logged and 16 averaged;
  - 0 logged cells moved, against 6 before §9, when the Inferno 86's
    simulated rows reached the KK-F-V2's track average;
  - 15 export files, 2.4 MB rounded.

**The re-evaluation** (2026-11-27). The standing holdout ran on the final
data: the fix stack (#376, #371, #373, #383, #395, #393, #391), the
dataset through 2026-11-25 (429 recordings), and the export of 20 files
(4 raw cars × 5 tracks, 3.2 MB). It used 15 folds and 129 held-out
sessions. The current calculator and the shipped `served` arm were
scored, and master 47426819c324's calculator on the same fold ids.

On the 5,651 lap × tyre rows all three have (128 sessions), in mbar,
paired per lap and tyre, ± session / date-clustered SE:

| | master | calculator | served | served − calculator |
|---|---|---|---|---|
| pooled | 42.40 | 36.15 | 36.08 | −0.06 ± 0.17 / 0.22 (59 of 128 nearer) |
| FJ | 35.37 | 31.14 | 30.94 | −0.20 ± 0.15 / 0.20 |
| KK-SII | 32.06 | 29.95 | 29.50 | −0.46 ± 0.19 / 0.23 |
| KK-F | 41.72 | 33.41 | 33.71 | +0.30 ± 0.21 / 0.25 |
| Inferno 86 | 52.31 | 43.22 | 43.35 | +0.12 ± 0.34 / 0.45 |
| dry (100 sessions) | 41.73 | 34.93 | 35.17 | +0.24 ± 0.15 / 0.21 |
| damp (4) | 72.29 | 42.83 | 42.64 | −0.19 ± 0.25 / 0.27 |
| wet (24) | 38.22 | 40.76 | 39.16 | −1.60 ± 0.59 / 0.68 (15 of 24 nearer) |

- **The shipped arm is at parity with the calculator** pooled and in the
  dry, and better in the wet. On that the lead set `SIM_CONDITIONS` to
  all three. Its value beyond the holdout is coverage: 7 car × track
  pairs answer from the simulation alone (the FJ raw cars at Motegi and
  Sodegaura, the Inferno 86 at Suzuka).
- **Against master**, the calculator is −6.25 ± 1.71 mbar (date SE;
  nearer on 91 of 128) and the shipped arm −6.31.
  - The scores are on identical pressure readings. The gain is mostly the
    fix stack's reading of the calculator's inputs (the pit-exit anchor,
    the stint clock).
  - The wet is the exception: there the calculator is +2.55 ± 1.22 against
    master and the shipped arm +0.95 ± 0.82.
- **No day of the refresh is scored.** A holdout bucket is (track, raw car,
  condition) with at least 10 dry sessions. The KK-F-V2's Fuji day has 9
  trained dry sessions. The KK-SII's five Motegi sessions have TPMS but no
  lap the fit's filters keep, so that cell has no logged entry and is
  sim-only. The refresh enters as training data, and as the simulation's
  fed uses at those two cells.

**Batches, and speed.** The solver works on a batch of laps of one
track, each with its own car and its own u (`solver.solve_laps`,
`solver.drive_laps`; one lap is a batch of one), in two forms that give
the same numbers to the last bit:

- `_tensor.py` is numpy with the laps as an axis: a step along the track
  is taken by every lap at once, and so is a step of the search for u.
  A lap of a batch is driven exactly as it would be alone, which takes
  three things: each lap is turned to its own slowest apex, a forward
  step that does not settle is bisected only for the laps it does not
  settle on, and a lap that has closed is left alone while the others
  go round again. It is the reference, and what runs without numba.
- `_kernel.py` is the same statements compiled with numba, one lane per
  lap, the lanes side by side on the cores (in the `tire-model` extra).

The passes are a recurrence along the lap (a node's speed comes from the
node before), so numpy still walks the nodes in Python, about 200 µs a
node whatever the batch; the batch axis divides that among the laps.
Per lap at Fuji (2,253 nodes; eight threads):

| laps at once | 1 | 10 | 100 | 1,000 |
|---|---|---|---|---|
| driven at a given u, numpy | 490 ms | 68 ms | 8.2 ms | 1.7 ms |
| driven at a given u, numba | 0.52 ms | | 0.27 ms | 0.21 ms |
| u solved for a lap time, numpy | 4.1 s | 1.0 s | 125 ms | 19 ms |
| u solved for a lap time, numba | 3.9 ms | 2.5 ms | 1.2 ms | 0.77 ms |

(The scalar Python loop the solver began as took 8.9 ms a lap. Compiling
the numpy form as it stands gives 0.6 ms a lap from a hundred laps and
10 ms for one, so the compiled form is written per lane instead.) The
search for u is scipy's `brentq` written out, in both forms; it agrees
with scipy's to 2e-10 in u. Tests hold the two forms to identical node
speeds, lap times, u and statuses on thirteen profiles and on mixed
batches with laps that have no root.

The gates run compiled: C3 went from 23 min 36 s (the scalar Python
loop) to 2 min 13 s (compiled, lap by lap) to 1 min 31 s (batches over
the cores) to 50 s (the fit below), and C2 from 17 s to 5 s, the reports
unchanged in every line each time.

**The fit.** A fold refits the car, so gate C3 fits it 76 times. The fit
is grouped statistics over the full-throttle table (300,000 to 470,000
rows a car), and what made it slow was Python around them: the
powertrain of a sample looked up once per sample (now once per car and
day), loops over days, gears and layouts each calling pandas on its own
group (now one grouped call for all of them: `shift_rpm_by_day`,
`gear_sets_by_day`, `power_scale_by`, the corners' 90th percentiles),
whole-table copies where a statistic reads five columns, and the table
relabelled inside every fit. With the first of those gone a fit took
0.62 s, and 0.26 s with the rest, with every number of the model and of
its unrounded workings identical on 95 fits (four cars, every
leave-one-day-out and leave-one-layout-out fold). What is left is
about twenty-five grouped medians and quantiles over those rows at 5 to
10 ms each, which is pandas' own compiled speed; a batch axis over the
folds would not lower it, because the cost is per row and not per call.
Of the time gate C3 spends in its folds, about half is now the fits, a
quarter the conventions (every day-best lap solved for u, two or three
rounds, for the fitted car and the low-power control of each fold) and
a sixth the held-out laps. The gate's time moves with the machine: three
runs of the same code took 50, 50 and 56 s, and one taken straight after
a long profiling run 85 s.

**The gates before the car (B1 to C1).** They work session by session
(250 to 315 of them): read the timeseries, put it on one clock, lay the
laps on the track. Run from an empty workspace:

| gate | before | after |
|---|---|---|
| B1 `check-geometry` | 18 s | 14 s |
| B2 `check-track-truth` | 56 s | 29 s |
| B3 `check-track-holdout` | 33 s | 19 s |
| C0 `check-frame` | 38 s | 26 s |
| C1 `check-envelope-samples` | 43 s | 33 s |

with every report, every track JSON and every cached table identical.
What was slow was work repeated per lap that belongs to the session or
the track, and numpy taking columns one at a time:

- the band of the build laps at the pit exit (two percentiles over laps
  × bins) was recomputed for every out-lap (`pit.build_band`, once per
  track);
- numpy's `nanmedian` along an axis takes each row or column in Python
  (`geometry.nan_median` sorts them all at once and takes each line's
  middle, to the last bit);
- laying a lap on the reference line built a points × 81 candidates
  array to find each point's nearest vertex (compiled:
  `_kernel.nearest_vertex`, the same vertex);
- the heat replay masked the whole session and interpolated its speed
  again for every lap, and the gyro scale filtered and sorted the whole
  session's gyro for every lap (once per session now; a lap is a slice);
- a clock looked up in itself moved by a constant was a binary search
  per sample (`geometry.sorted_bounds`, one walk);
- the kinematics loader made one Python string per row for its source
  labels.

What is left is per session and real: about 25 ms of parquet reads,
30 ms in the kinematics loader, 20 ms of geometry and 25 ms for the
frame record, already array code. Each gate does them again for every
session, so the next step for these gates is not more vectorizing but
keeping a session's geometry between gates.

A lap takes 0.3–1.2 ms compiled (1,039–2,878 nodes); `solver.solve_u_near` solves a
lap's u in about eight laps (a bracket opened around a guess, the last
lap's u in the same session, and Brent's method inside it) where
`solve_u`'s bisection takes two dozen. With `mu_y` still at the apex table's level, no
cornering drag and u = 1 the solver laps Tsukuba in 58.99 s (KK-SII), Suzuka in 135.77 s (KK-SII),
Fuji in 122.53 s and Sodegaura in 74.30 s (Inferno 86): 3–5 % under the
cars' reference laps, which is what the convention for `mu_y` then
removes.

## 3. Test methodology

> How the current calculator and the simulation-based one are compared
> (the standing holdout scored by both, the holdout by pace, the cars
> fitted apart, gate F at a track the car has not run), with the command
> for each: [`docs/calculator_evaluation.md`](calculator_evaluation.md).

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
  skipped, and so is a first-lap anchor's own lap: a lap whose middle on
  the stint clock is not past the reading (the lap before a start
  reading, whose end the stint clock puts a few hundredths of a second
  after it).
- **Compound.** The driver always selects the tire, so held-out sessions
  use the human or seed label where one exists and otherwise the compound
  the EM infers for that session on the full dataset (argmax, laps- and
  responsibility-weighted across axles). The inference uses the session's
  own temperatures and set pressure, so for unlabeled sessions this is a
  mild leak in exchange for standing in what the driver knew.
- **Blacklist applied.** Confirmed broken sensors are masked in both
  training and evaluation — we don't grade the model against channels we
  already know are broken.
- **Inputs: calculator or oracle.** By default (`--inputs calculator`) each
  held-out lap is predicted from what the calculator has: the fold model's
  ⟨g²⟩ scaled along the pace curve at the session's median lap time (the
  target a driver would enter — the session's **25th-percentile** flying lap,
  since drivers are optimistic about their pace), the bucket's typical
  out-lap followed by the clock `N × lap time`, and the start temperature
  the driver types in (a separate measurement at standstill, not the dash),
  stood in by the out-lap's first valid TPMS reading at pit exit (the
  start fields are always filled in practice; leaving them blank was
  measured at +1.2 °C MAE and dropped as an option). Stints without a
  pit-exit out-lap anchor on their first lap as before, with no out-lap
  segment, and N counts the laps driven from that reading, from its lap
  boundary: anchored at the start of the stint's first lap, the end of
  `lap_within_stint` = N is N + 1 laps from it; anchored at that lap's
  end (or the next lap's start), N. A first lap that is an out-lap but
  did not leave the pits counts as one of the N (#374). Until 2026-11-26
  they were fed N = `lap_within_stint`, one lap short when anchored at
  the first lap's start. On the standing holdout (15 folds, 5,751 rows),
  545 rows from 17 sessions are on first-lap anchors. The fix moves N by
  +1 on 481 of them and by −1 on 5 (anchored at lap 2's start), and
  leaves 59 as they were. The bias on those rows goes from +2.3 to +11.4
  mbar, so the short clock was hiding an over-prediction. MAE on those
  rows goes from 39.3 to 40.1 mbar; pooled, from 43.71 to 43.79 mbar
  (per session +0.05 ± 0.13, date-clustered, against an MDE of 0.7).
  Every pit-exit row is unchanged. The simulation-based arms of
  `check-pressure-holdout` move alike (table 44.64 → 44.73, direct 44.74
  → 44.83 mbar), so their gap to the current calculator stays at +0.9
  and +1.0 mbar.
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

The holdout scores every lap in two domains. The temperature residual is in
the model's observable, the pressure-implied gas temperature (§2.2a). The
pressure residual is the predicted hot pressure, `P_start_abs · T_hot_K /
T_start_K` from the pit-exit anchor — exactly the step the calculators
apply — minus the TPMS hot pressure, in bar. Because the observable is the
gas temperature the two are the same information in different units; the
pressure one is what the driver gets. (An empirical pressure gain γ was
evaluated while the target was the TPMS temperature; with the gas
temperature as observable it is 1 by construction and was removed.)

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

### 4.0b The schema-v5 share model (every-session 20-fold holdout, 106 sessions, 4655 lap × corner, calculator inputs, 2026-10-09)

Lap-end hot-pressure residual (predicted − observed TPMS reading), pooled
and per car, for the promoted model (§2.0: ``g_transfer`` the only fitted
distribution constant, no efficiencies) against the geometry-based split
it replaced (same |g|·V input and κ; four spec-sheet geometry numbers plus
fitted drive/brake terms per car):

| | FL | FR | RL | RR |
|---|---|---|---|---|
| geometry split, MAE (bar) | 0.048 | 0.050 | 0.051 | 0.052 |
| **promoted model, MAE (bar)** | **0.049** | **0.051** | **0.050** | **0.052** |
| promoted model, bias (bar) | +0.003 | +0.007 | +0.014 | +0.005 |
| FJ, MAE / bias | 0.033 / +0.009 | 0.031 / +0.011 | 0.043 / +0.022 | 0.042 / +0.008 |
| Inferno 86, MAE / bias | 0.068 / −0.004 | 0.074 / +0.001 | 0.059 / +0.005 | 0.064 / +0.003 |

The same accuracy on known circuits with one fitted distribution constant
per car instead of two fitted plus four hand-entered, and 0.0595 against
0.0641 bar on unseen circuits in the harness's leave-one-track-out. The
FJ's +0.01–0.02 bar held-out bias is the fold effect noted under the
previous build (the same model scores the FJ unbiased in-sample through
the same path); its cause is an open item.

### 4.0a Per-second fit vs per-lap (every-session 20-fold holdout, 106 sessions, 4655 lap × corner, calculator inputs, 2026-10-04)

Hot-pressure MAE in bar, same folds and laps. "TPMS per-lap" is the v0.19
model (TPMS temperature as the target, ⟨g²⟩ at the 75th percentile);
"gas per-lap" is the v1.0 release (gas temperature, median ⟨g²⟩); "gas
per-second" is §2.6a.

| | TPMS per-lap (v0.19) | gas per-lap (v1.0) | gas per-second |
|---|---|---|---|
| pooled | 0.050 | 0.054 | **0.051** |
| pooled bias | +0.000 | +0.009 | +0.010 |
| FJ dry | 0.035 | **0.030** | 0.031 |
| FJ Suzuka dry | 0.054 | **0.038** | **0.038** |
| FJ wet | 0.043 | 0.049 | **0.042** |
| FJ damp (3 sessions) | 0.079 | 0.113 | 0.099 |
| Inferno 86 dry | **0.063** | 0.072 | 0.067 |
| Inferno 86 Fuji dry | **0.054** | 0.069 | 0.062 |
| Inferno 86 Sodegaura dry | 0.069 | 0.073 | 0.071 |
| Inferno 86 wet | 0.064 | 0.066 | 0.070 |
| within ±0.05 bar | 62 % | 61 % | 62 % |
| oracle inputs, pooled | — | 0.055 | 0.052 |

Paired bootstrap over sessions, pooled MAE: per-second − gas per-lap
−0.0023 bar (95 % CI −0.0049 to −0.0002); per-second − TPMS per-lap
+0.0013 bar (−0.0020 to +0.0048), i.e. indistinguishable from the v0.19
model overall while using the physically right observable. The Inferno 86
recovers most of what the gas target had cost it (dry 0.072 → 0.067, Fuji
0.069 → 0.062) but stays 0.005 bar behind the TPMS fit (CI −0.000 to
+0.011); what remains there is whole-stint offsets on a few long Sodegaura
and Fuji sessions, not quantisation. The lap-1 bias of +0.023 bar decaying
to zero by lap 5 is unchanged: the out-lap segment predicts heat that the
pit-exit-referenced gas does not yet show (§2.2b).

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

7. ~~**Per-second fitting on pressure.**~~ **Done** (2026-10-04, §2.6a,
   §4.0a): the default Pass 1 fits the discretized ODE at 1 Hz against the
   pressure-implied gas temperature; held-out hot-pressure MAE 0.054 →
   0.051 bar pooled (CI excludes zero), Inferno 86 dry 0.072 → 0.067. Not
   yet done inside it: the anchor pressure as a per-stint nuisance
   parameter and a tread-to-gas lag. The original reasoning follows. Fit
   the discretized ODE at 1 Hz against the pressure-implied gas
   temperature instead of the per-lap closed form. Tested offline against
   the TPMS temperature
   (2026-10) it bought nothing, because that observable lags the gas by
   2–3 min and the within-lap detail was sensor dynamics. Against pressure
   it should: the 0.03 bar quantisation becomes information (the *time* a
   rising pressure crosses each step locates it to a fraction of a step,
   where one lap-end sample cannot), the pit-exit anchor pressure becomes a
   per-stint nuisance parameter with a half-step prior instead of a
   whole-stint offset the fit absorbs into K, and measured g²(t) through
   the lap identifies τ and K from the lap's shape. It is also the only
   framework where a tread-to-gas lag, the out-lap and warm starts can be
   fitted as what they are. The scratch harness from the 2026-10 state-space
   experiment (1 Hz per-stint arrays with the moving mask and anchors, exact
   vectorised recurrence, ~5 min per two-car fit) is the starting point;
   production would be a `tire_model/statespace.py` replacing Pass 1 with the
   same artifact tables out. Decision rule: adopt if the every-session
   holdout's hot-pressure MAE beats the per-lap gas-target fit, with the
   Inferno 86 (0.064 bar on the TPMS target) as the bucket to watch.
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
# (out-lap: --outlap-time-s 90 to override the bucket's typical out-lap, --no-outlap to skip it)

# Predict per-corner cold pressures, with rain condition
just tire-predict --track tsukuba_2000 --car KK-SII --lap 5 --ambient 15 \
                  --condition damp --hot-all 1.5

# Predict with a target lap time (scales each corner's heat input along the
# simulated-lap curve and sets time-on-track t = N × target)
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

### Heat-input / cooling experiments

```
uv run python -m motorsports_data_notebook.tire_model.heat_experiment \
    --form g2 --form sliding --form "sliding:glim=1.15" --form "power:p=2.5" \
    [--p-mode anchor|instant] [--p-exp N] [--cool-mode speed] [--cool-beta B] \
    [--split-compound] [--car FJ] [--n-folds 5] [--out-dir DIR]
```

Form specs: ``g2``; ``sliding`` (``V·g²``), ``sliding:glim=<mult>`` (×
the car's p99.5 g) or ``sliding:glim=FJ:2.2,Inferno 86:1.5``,
``sliding:rr=<G²>``; ``power:p=<exp>``. See §2.6b.

### Two-node tread + carcass experiment

```
uv run python -m motorsports_data_notebook.tire_model.twonode \
    [--mode per_corner|shared|shared_speed] [--car "Inferno 86"] [--n-folds 5] [--out-dir DIR]
```

Prints the IR gate counts and reasons, the held-out single-node vs
two-node pressure/tread residuals, and the fitted constants per fold. See
§2.6c.

## 7. Files of interest

| Path | What |
|---|---|
| `src/motorsports_data_notebook/tire_model/energy_balance.py` | Pure physics functions (T_eff, warmup_curve, Gay-Lussac, T_road proxy, discretized recurrence) |
| `src/motorsports_data_notebook/tire_model/warmup_table.py` | Data prep + two-pass scipy fit + JSON serializer + sensor audit + blacklist |
| `src/motorsports_data_notebook/tire_model/pace_sim.py` | Pace model: the logged laps simulated per second at other lap times through the force shares → per-corner `g2_vs_lap_time` curves; `tire-pace-check` |
| `src/motorsports_data_notebook/tire_model/compounds.py` | Compound label loading (sidecar + notes fallback, wheel-set mapping, condition seeds) |
| `src/motorsports_data_notebook/tire_model/compound_infer.py` | Decomposed compound K: EM with partial supervision, forced selection |
| `src/motorsports_data_notebook/tire_model/predict.py` | `predict_cold_pressure(...)` and the fallback chain for K / τ / ⟨q⟩ |
| `src/motorsports_data_notebook/tire_model/validate.py` | `tire-predict-validate` (notes-recorded ground truth), `tire-predict-holdout` (held-out generalization test, temperature and pressure domains) |
| `data/tire_dataset/tire_model.json` | The committed fitted artifact (diff-friendly) |
| `data/tire_dataset/tire_compounds.yaml` | Human-curated compound history: per-session `compound:`, `wheel_sets`, `condition_seeds` |
| `data/tire_dataset/sensor_blacklist.yaml` | Human-curated list of broken (session, corner) channels |
| `scripts/regen_tire_predict_fixture.py` | Regenerates the Python-parity fixture pinned by the C# and web test suites |
| `tests/tire_model/test_energy_balance.py` | Physics functions in isolation |
| `tests/tire_model/test_warmup_table.py` | Lookup builders, anchors, blacklist, pace scaling on synthetic laps |
| `tests/tire_model/test_compound_infer.py` | EM recovery on synthetic mixtures: pinned labels, latent posteriors, forced selection |
| `tests/tire_model/test_predict.py` | Fallback chain hits every level with mocked artifacts |
