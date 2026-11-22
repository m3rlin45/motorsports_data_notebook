# Lap simulation: a driver model

The car's limits apart from the driver's use of them. Design of 2026-11-17; the body of issue #312.

Part of #202. Follows the review of the car model (#205) by seven reviewers on 2026-11-16/17; its measurements are in `docs/tire_model.md` §2.12 C5 and are quoted here with the reviewer who made them. Feeds #206 (the pace knob, D2), #208 (heat at a track the car has not run) and #263 (where the car can go faster). A design, not yet built; the numbers are the review's probes and are re-measured by the gate below.

**Built so far (2026-11-21): the braking use.** The model's `driver` entry holds the level the driver brakes at as a share of `mu_b·u` (0.80 / 0.85 / 0.84) and the shape held under cornering, read from the logged deceleration at each lap's own u (`car_gate.set_use`); the laps are braked on it and the car's limit stays the car's. Gate C3's score went from 34 to 56 of 76 cells. The reading rule, what it was tested against and what it leaves wrong are in `docs/tire_model.md` §2.12 C7. Still as designed here and not built: the cornering use with the apex table's level in place of the convention, the tables in the tyres' frame, the max driver, and the four decisions below.

## What

Keep two things apart that the car model now mixes: **what the car can do** (its limits, read from the best it has been seen to do) and **how it is driven** (how much of each limit the driver uses, and how the controls move). The owner's rule for the shift point, "how it is driven is not the car", applied to grip.

With the two apart there is a third thing for free: a **max driver**, every use set to 1, which is the lap the car has shown it can do, put together (#263).

## Why

- **The simulated lap brakes harder and corners softer than the car.** Logged straight-line deceleration is 0.81–0.87 of simulated (physics); the simulation is +0.85 to +2.4 m/s fast where the logged car brakes and −0.3 to −1.2 m/s slow on entry and mid-corner in ten of ten car-layout cells (logs); lateral heat is 0.5–3.0 % low and longitudinal heat 0.7–6.0 % high in ten of ten (consumer).
- **The two grip levels are not on one footing.** `mu_b` is the 90th percentile of the stops' peaks, and the solver makes every stop that one. `mu_y` is a convention, and the convention turns out to be what the driver holds: lateral through the apex is 0.87 / 0.89 / 0.89 of the apex table's level (Inferno 86 / KK-SII / KK-F) and the convention sits at 0.89 / 0.86 / 0.86 (estimation, 7,230 apexes).
- **Braking is held lower than cornering, by the same amount on every car.** Deceleration from 80 % of the stop's pressure to turn-in is 0.80 / 0.80 / 0.78 of `mu_b` (estimation, 2,502 stops). Held braking over held cornering is 0.64–0.65 on all three cars; the solver runs 0.80–0.85.
- **Held-out laps agree.** With the car re-fitted per fold and laps paired, `mu_b` × 0.84 scores a lower S1 than the fitted car on 21 of 31, 9 of 13 and 17 of 19 days (advocate).
- **One pace knob cannot do it.** u multiplies `mu_y` and `mu_b` together, so only their ratio is seen (estimation: `mu_y` +7 % with `mu_b` −7 % scores within 0.005 m/s), and the ratio is what is wrong.

## The split

| the car: limits | the driver: use |
|---|---|
| power against rpm per powertrain, the day's gear set, drag, cornering drag | shift point, per day (exists) |
| lateral level: the apex table's (no convention) | **cornering use**: lateral held over that level |
| `mu_b`: the straight-line stops' peak level; the limit's shape under cornering | **braking use**: deceleration held over the limit the solver applies |
| shift time (gearbox; measured per powertrain) | lift, brake build, release: recorded |
| wet grip scale | pace of the lap (u), and its spread from lap to lap |

- **Cornering use** is a constant of this driver on this car: 0.87–0.89, moving 0.01–0.02 from day to day. `mu_y` becomes level × use. The convention (the median reference lap has u = 1) stays as the normalisation, and the measured use beside it becomes a check: they agree within 0.02–0.03 today.
- **Braking use** is set by the ratio of held braking to held cornering, both measured on the same laps, because the ratio is the only thing a lap time lets the solver see. It moves from day to day (0.68–0.94; sd 0.05–0.08 within a layout) and does not move with cornering use (r −0.21 to −0.32).
- **Mistakes** are scatter about the driver's typical lap, not a bias: they stay in the spread of u and of the residuals.

## What the review already tried, and what it changes here

Handing the solver a measured braking use (logs, in sample): with the stop's plateau over `mu_b` (0.88 / 0.85 / 0.78) the straight-line braking bias is within ± 0.5 m/s in 5 of 10 cells with one number a car, in 3 of 10 with the day's own, and it overshoots on the formula cars. So:

1. **The definition has to be the one the solver's stop corresponds to.** The solver sheds speed at a constant level over distance. The use is therefore the deceleration that sheds the stop's own speed over the stop's own distance, from the application to turn-in: `(V_on² − V_turn²) / (2·Δs)`, less drag, over the applied limit. A constant stop at that level passes through the logged application and the logged turn-in speed by construction. The time-mean (0.78–0.80), the plateau and the mean to pedal release (0.87 / 0.84 / 0.80) are recorded beside it; none is chosen by the gate.
2. **One number per driver and car first; the day's is recorded.** The day's value helped only the KK-F, and on the Inferno 86 it follows the day's u (r = +0.66), which has then already taken it.
3. **It does not fix corner entry.** Entry stays at −0.5 m/s whatever the braking level (logs). That error is on the lateral side: the track (cant, the banking investigation and #299) and the scale of the braking shape.
4. **The braking shape is re-measured in the driver's units.** It is now estimated against the apex table's level and applied against `mu_y·u`, 11–16 % lower. With both axes carrying their own use, the shape of what the driver uses is read from held deceleration against held lateral; the 90th-percentile shape is kept for the max driver.
5. **The onset stays out.** It trims the speed peak and leaves the braking curve; it made no difference on unscored laps and raised the heat error in ten of ten cells (validation, consumer). The lift and the build are inside the distance-mean of point 1.

## How

1. **Tables (C1).** Per stop: speed and position at the application and at turn-in (lateral reaching 0.2 of the level), so the distance-mean deceleration is a column. Per apex: the lateral held within ± 20 m. Laterals in the tyres' frame (`bank.tyre_lateral_g` at the sample's place on the track): the tables are in the ground frame today, where the road's bank counts as the car's grip (the apex table returns 8.5–15 % over a simulated car's `mu_y`). A tables version bump.
2. **Fit (`car.py`).** `cornering_use`, `braking_use` and the held shape, each from its own table in one grouped call, per day and as the car's median; the limits (`mu_b`, the lateral level, the p90 shape) unchanged and reported as the car's.
3. **Solver.** `SolverCar` carries the limits and the uses; the passes read `limit × use` and nothing else changes, so the numpy and the compiled form stay identical. `Drive` hands a lap its day's shift point as now.
4. **Max driver.** Uses at 1 and the p90 shape; reported per layout as lap time and as time against the driven lap by phase. Reported only until #263 says how a claim about time not yet found is checked.

## Gate

The review showed the present gate cannot fail a model on speed: S1 against the day floor admits 1.3–2.0 m/s of repeatable error, and a car with drag +10 %, `mu_b` × 0.84 and shape −0.2 is inside wherever the fitted car is. So the driver model is judged by a gate that can, written down before it is run:

- **Recovery.** Laps simulated with a planted braking use (0.7, 0.8, 0.9) and cornering use, with the real spread of u and noise, through the tables and the fit: each use back within 0.03.
- **The statistic it must move**, named now: straight-line braking bias (simulated minus logged speed in the braking phase) within ± 0.5 m/s in 8 of 10 cells, in sample; 0 of 10 today.
- **Paired, held out.** Car re-fitted per fold, laps paired: the driver-model car beats today's default on at least 75 % of each car's day folds in the lateral-heat-weighted speed error, and is not worse in worst-tyre heat by more than 0.2 %. Layouts weigh equally.
- **Wrong cars.** Braking use ± 0.08, cornering use ± 0.03: each must lose to the fitted one on most days, or the report says the gate cannot tell them.
- **One clean score.** Days after 2026-11-14, the wet build laps and the SFJ, scored once: repeatable error and worst-tyre heat, against 1.19 m/s and 3.5 % for today's default.
- **Reported:** residual by phase and the time ledger; the measured cornering use against the convention's; the day's braking use against the day's u.

## Not in this

- **The pace level at a track the car has not run.** Per-tyre heat is 2.4 % off with the lap's own lap time and 5.9 % with u carried from the car's other layouts (consumer); the held-out layout wants a u of 0.92–1.07. No use of the driver's explains it. It is the track's, or it needs the target lap time the calculator already takes. Banking from the elevation model, put into a prototype lap as physics, takes the 5.9 % to 3.7 % (physics, third round; §2.12 C5 item 11); it is in the track model and in every lap since 2026-11-17 (#299; §2.12 B2b), and what it leaves at Tsukuba (the hairpins overshoot, the fast corners fall behind) and in braking against cornering is this piece's to explain.
- Throttle pick-up after the apex, the driver's line, another driver. Every lap in the dataset has one driver, so a use is "this driver on this car" and is checked across days and cars, not across drivers.

## Decisions for the owner

Deferred on 2026-11-17 until the split is planned and the banking is settled:

1. The split as tabled: the lateral level and `mu_b` are the car's, the uses are the driver's.
2. Braking use from the distance-mean of the stop, one number per driver and car, the day's recorded.
3. The gate above replaces S1 against the floor for this piece; the same rework for gate C3 as a whole is its own issue.
4. The onset term: left in the code and off, or removed.


## The driver committee's design (2026-11-22)

The owner asked for a committee "to debate how we can separate car performance from driver
performance, and how we make a realistic driver centric pace knob. The car model should just
know the limits of the car, and the driver model should set where and how we underperform those
limits, in order to make the pace we want", with research on probabilistic modelling of lap
times and Monte Carlo over "the various ways we can get the same time, using different
underperformances". Seven seats (the car's limits, the driver by phase, the knob, Monte Carlo,
the day and the conditions, the uses and the build, a skeptic) and, from the third round, five
experts who had not seen the debate (a statistician of frontiers and extremes, a driver coach,
a vehicle dynamicist, a tyre specialist, a probabilistic modeller) worked through five rounds:
positions, cross-examination, the experts' own views, deciding tests with replies, and a vote.
All twelve signed the draft put to the vote; the text below is that draft **as amended by the
lead after the vote** from the dissents (section 7 lists which clauses changed in substance),
and the amended text was not put back to the members.
Nothing was built in the repository. During the debate the owner ruled twice: the pace knob is
the driver's and not a scale on the grip; and, the true limit being unreadable from one
driver's laps, "we can certainly use momentary max performances to derive a limit that is
higher than has ever been attained so far".

**What supersedes what.** This section replaces the split tabled above where they differ (the
lateral level is now the composite limit of clause 1.1, not the apex table's 90th percentile;
no cornering use is read as a constant, the knob is that use). Its clause 0 is the first thing
to read: the knob it offers is narrower than the one asked for. **The design awaits the
owner's decisions on the clauses marked [owner]; until then nothing here is built.**

**Every number in this design is in sample**: read on the laps it describes, the car fitted on
them, nothing held out unless a clause says so. Several rest on one car at one layout or on
field reasoning; the clauses say where. Clauses marked **[owner]** are decisions the committee
cannot make; each carries the default the committee would take.

### 0. What this design does and does not give the owner **[owner]**

The owner asked for a knob that changes "how we use the grip ... spending less time at the max"
and not the grip. The committee could not read from the logs how a slower lap differs from a
faster one beyond chance (clause 3.3), so what it offers is this:

- **The grip is the best the car has been shown to do, held fixed** (clause 1): a limit above
  every logged lap, which no pace setting changes. It is a floor: whatever the driver leaves
  unused at their best moments is still missing from it (a driver never at the limit and a
  weaker car driven to it read back alike).
- **The knob is one share of that limit, the same everywhere** (clause 3.1). On the simulated
  lap it is the arithmetic of today's knob: every limit lowered by the same share at every
  apex and every stop. Step 1 changes no simulated lap (clause 5.1). What changes is what the
  number means (the driver's use of a fixed limit) and that the limit no longer moves with
  pace. A driver would know the corners of a slower lap made this way (lower peaks everywhere,
  the straights flat out) and not its stops: it brakes softer and so earlier (about 3 m per 1 %
  of pace on a long stop, reasoned), where the logs hold the braking point fixed.
- **"Less time at the max" is almost empty at the paces in the logs.** The driver's typical use
  is 0.86–0.92 of the limit against a corner-to-corner spread of 2.4–3.8 % (one reading) or
  4.1–5.2 % (another), so between 0.1 and 5 in 100 corner passes reach the cap, about 2 at a
  use of 0.90. The cap matters only close to the max driver.
- **The choice**: accept a uniform knob measured from the new limit now, with a direction added
  only if the stint-level test of clause 3.3 finds one; or hold the knob as it is until that
  test has been run. Default: accept it now.

### 1. The car: limits only

1.1 **The cornering limit** is the expected best of m passes through a corner (m stated;
default 20 **[owner]**), of the lateral the tyres hold over the load on them at the apex (the
tyres' frame: bank and vertical curvature, `solver._road`), read on the mean of the two readings
of each pass (Doppler and accelerometer) with the sensors' own noise subtracted; then the median
over the grip-limited corners and the layouts; every dry day pooled; a day-bootstrap error
beside it. With m = 20: 1.278 / 1.492 / 1.539 (Inferno 86 / KK-F / KK-SII); the convention it
replaces was 1.160 / 1.372 / 1.318. It is **a floor under the true limit, up to the error the
two sensors share (the line, a kerb) and the noise correction**, and labelled so: the
correction (0.2–0.9 %) assumes a bell-shaped top to a corner's passes, which the logs cannot
tell from a capped one; a capped top would make the correction larger and the level lower. A layout without two readings
borrows the car's level, labelled (the KK-F's level is Tsukuba's). All the levels and margins in this clause and in 1.8 are on the car's own path (see 1.9).

1.2 **The braking limit** is read the same way: the expected best of m stops of a deceleration
held over a window (not a peak), drag and grade out. The brake table gains the second reading
(the accelerometer's longitudinal beside the GPS speed's change) so that the noise can be taken
out; until it has, `mu_b` stays as today and is labelled "best shown, one sensor". The shape
under cornering is read per direction by the same statistic, never as a hull.

1.3 **Drive out of a corner goes through the driven axle**: the grip left for drive is the
rear share of the grip circle less the lateral. The share is production's stated static weight
split (front 0.53 on the Inferno 86, 0.42 on the formula cars: a stated input, not weighed and
not fitted). **Built for the Inferno 86 and then held to clause 5.3** (two readings agree
there, but it has had no heat check, no layout folds and no wrong-car test, and its convention
was re-read on four folds of another car); not built for the formula cars until a test that is
not confounded (rear wheel slip against the engine's ask, or a held-out run) **[owner]**. The
static share leaves out the load moved rearwards under drive, so it understates the rear's
grip (field reasoning).

1.4 **What is found where the driver's input is saturated is the car's** (full throttle, ABS,
the top of the pedal): power, drag, cornering drag, shift time as today; no driver term may be
fitted to a shortfall that is still there at saturated input. Recorded as a known error of the
car and not owned by any term yet: at full throttle near the cornering limit the logs are
0.015–0.030 g short of the model beyond the cornering drag, the same at every pace.

1.5 **No speed axis.** The limit does not change with corner speed; downforce on the winged
formula cars is neither shown nor excluded (a wing of the likely size would own about half of
the corner-speed apex error). It enters only as a stated per-car input.

1.6 **The wet is a condition on the limit**: per car, a scale on the composite read from wet
best moments at an equal number of passes (cornering 0.83–0.88 / 0.85 / 0.79), and a wet braking
term beside it where braking does not fall with cornering (0.92 on the Inferno 86; that it is
the ABS there, and the driver braking further from lock-up on the KK-F, is field reasoning and
not read from the logs; on the KK-F it would then belong with the driver's constants of 2.1).
**A wet day's water level is a condition and not the driver's** (wet days sit at 0.73–0.91 of
the dry composite, up to eight times the dry day's 2 %): the composite is set by the driest
wet day, so each wet day's level is read from its own best moments, and at prediction it is
solved from the target lap time with the driver held at their typical wet use, within the
logged range **[owner]**. The wet rests on one to four wet days per car and layout (one on the
KK-SII), too few passes for m = 20, with the tyre unlabelled on most of the KK-F's wet
sessions: the least trusted clause of the design.

1.7 **The day's level** (about 2 % between dry days; a day sits at 0.89–0.94 of the composite)
cannot be split by the data into the day's grip and the day's driving. Default: it is the
driver's and goes into the knob; the air and the powertrain of the day stay the car's
**[owner]**. Its cost: a uniform knob puts the day into the wrong phases (between days the
longitudinal heat error differs from the within-day one, and the Inferno 86 within one
powertrain looks like grip), about 0.4–0.8 % of heat.

1.8 **The check on the car**, the owner's plain number: the lap with every use at 1 (the max
driver) against the best logged lap, per car and layout (today: faster in every cell; thinnest
the Inferno 86 at Fuji, 0.80 % at m = 20, 0.44 % at the low end of the day bootstrap). Beside
it as a diagnostic, not a gate: the sum of the expected best of m segment times, at the same m
and with the timing noise subtracted (a plain sum of best segments is a maximum many times
over and reads fast). Every new day is scored against it: a logged lap that beats
the max driver in equal air and powertrain shows the limit too low.

1.9 **Which curvature the level is read on: open.** The level of 1.1 is read on the car's own
path; the solver spends it on the track's line. On the formula cars the track's curvature
reads 3–4 % under the car's own at the apexes and above it at single corners (at Suzuka's c6
it reads 1.073 of the car's path; Suzuka's corners together lift the KK-SII's level by 1.6 %).
That gap is larger than the margins of 1.8. The level stays on the car's own path until the
level read on the track's curvature has been put through the max driver at this statistic (it
was only run on an earlier one, where every cell stayed faster by 0.4–3.9 %); bringing it to
the track's curvature would fold each track's line error into the car's number, which differs
at a layout the car has not run. Three members trust this clause least.

### 2. The driver: every way a lap falls short of those limits

2.1 **Constants of the driver, read once per car and not moved by pace**: the braking use and
the shape held under cornering (as built, `docs/tire_model.md` §2.12 C7, re-read against the new limits); the lift
before braking and the brake's build (0.14–0.32 s and 0.34–0.44 s a stop; recorded, and applied
only once the onset term's gate is re-run; the KK-SII's lift shortens 0.07 s per 1 % of pace);
full throttle arriving at a fixed share of the grip (0.92 / 1.01 / 1.01 of the old convention's
`mu_y·u`, ground frame, the same at every pace; to be read again against the composite limit,
where it sits lower), so that with 1.3 in the car no exit use is built (on the formula cars
this rests on an exit-acceleration ratio that the simulation's faster trough confounds); the
day's shift point (exists). On a car without ABS the wet braking shortfall belongs here (1.6).

2.2 **The use**: one number, clause 3.

2.3 **The scatter**: an event per corner (its phases moving together: stop and release +0.68;
corners independent of one another), a level per lap and a level per stint; each event a
bell-shaped core (2.4–3.8 % within a corner and day without sensor noise by one reading;
4.1–5.2 % within a lap, with a further 2.1–2.7 % by day and corner, by another) and a
slow-side mistake in about 2 % of passes (sized from segment counts; nobody has yet planted a
tail and got it back). This is "where" the driver is under the limit on a given lap; it has no pattern by
corner speed or phase that the data can show.

### 3. The pace knob

3.1 **One use q per lap: every limit of clause 1 is multiplied by exp(−q)**, cornering and
braking alike, each use capped at 1; the convention for `mu_y` is gone. It is today's arithmetic (per-lane scalars,
the kernel untouched), so a lap time is solved for it exactly, and lap time is continuous and
monotone in it. **Two laps mark its ends, and they are not the same lap**: the *max driver* is
every use at 1 and the driver's habits of clause 2.1 off (clause 1.8's lap); the *knob's
lap at q = 0* has the habits on (the braking use of 0.80–0.85 and the held shape still
applied) and is slower than the max driver by 0.48–0.83 s a lap (one seat's ledger, read at a
P90 lap's pace on the old convention's car, the lift and the build left out). The knob does not stop there: below q = 0 the cornering use stays capped at 1 and the braking
use rises, reaching 1 at q = ln 0.80 to ln 0.85 (about −0.2). That is *the knob's end lap*:
braking at the car's level but still on the shape the driver holds under cornering, slower
than the max driver by the held shape's cost (0.12–0.39 s a lap by the same ledger). Lap time
stays continuous and monotone throughout. A logged lap may need q < 0: at the Inferno 86's Fuji the max driver's margin over the best
lap (0.44–0.80 %, about 0.6–1.05 s) is of the size of the habits' cost (0.48–0.83 s). Which
laps do is reported.

3.2 **"Less time at the max" comes from the cap and the scatter**: a corner event's use is
min(1, exp(−q − ε)). The knob is solved first on the one typical lap (ε = 0); solving it so
that the expected lap over the scatter meets the target (q about 1 % lower) is the next step,
taken when the sampler passes its planted test.

3.3 **No direction is built until one is tested.** The knob does not move braking, cornering
and exits in different proportions, and does not treat slow and fast corners differently.
Within a day, slopes against lap time can be made by selection alone (the selection test below: shown on one cell,
the Inferno 86 at Sodegaura; the other cell's verdict was void), so they cannot be read as a
direction, and a direction planted in the knob was not read back by them either (within 2 sd
of a uniform knob on every reading). That makes a direction unreadable within a day; it does
not show there is none. A direction may
enter only by: planted recovery through the whole pipeline; not being preferred where a
uniform knob is the truth; and a contrast between stints of one day (which can only reject).
Known and left as errors: the braking of the Inferno 86 hardly rises with pace (ABS) and the
formula cars' rises more than uniform; a uniform knob moves the braking points, which the
logs hold fixed; the slow-corner trough is a V in the logs and a U in the simulation.

3.4 **The calculator's contract**: a target lap time is a stint's typical pace. A target faster
than the knob's end lap (clause 3.1) has no solution and is reported so, with the max driver's
lap beside it; a target whose q lies outside the
car's logged range of stint-typical q (about 4 % of lap time off the car's pace where it has
run; q carries to a track where it has not, and a lap time does not), or wetter than the
wettest logged day, is answered and flagged as extrapolation **[owner]**. Such a lap is not a
careful lap but another way of driving (a cool-down, tyre saving), of which the logs hold none.

### 4. The same time in different ways

4.1 **There are no distinct ways**: one lap time is a continuum of gives and takes corner by
corner (clustering finds nothing a bell curve does not), with no trade across the lap; at a
fixed lap time the release, exit and straights each move by 0.15–0.18 s, the apex and the stop
by about 0.1 s (the Inferno 86 at Sodegaura only).

4.2 **The Monte Carlo** draws clause 2.3 and drives each draw at the knob; for the laps of one
lap time it root-solves the lap level only and weights each draw by its prior density over the
lap time's sensitivity, reporting the effective sample size. It needs a scale per lap and node
in both forms of the solver.

4.3 **What it is for**: the error bar on a stint's heat, and the max driver's ledger (how often
each corner goes well; time by phase and corner between a driven lap and the composite lap).
**What it is not for**: the predicted heat (the ways move the mean by under 0.3 %), or the gate
(one draw scores 31 of 76, a mean of draws is a blur).

4.4 **Known short**: the sampler makes a third to a half of the logged spread of heat at one lap
time; its
planted recovery (the correlations and the tail) comes before any use.

### 5. The test

5.1 **Step 1 is accepted when limit × use × exp(−q) gives each lap exactly today's lap at its
own lap time**: the score (56 of 76) and the heat error (2.1 %) unchanged, to the last bit where
the kernel is handed the same numbers as today (the braking use set so that its product with
the new limit is today's) and within the reading's tolerance (1.5 %) if the use is read again.
The heat with the pace carried to a layout the car has not run (3.0 % today) rested on the
convention that is now gone: it is measured again with q carried, and reported, before step 1
is accepted.
5.2 **Every use at or below 1** is a bound, not a check.
5.3 **Any later term**: the committee's test of `docs/tire_model.md` §2.12 C7, and planted recovery at the logs' noise
through the whole reading. The single number proposed in the debate (E\*, below) is withdrawn as a chooser (it
separates nothing, even where today's knob is the truth).
5.4 **The limit's statistic**: stable as days are added by its construction (the 0–0.4 % left
from two days to all says only that the days are alike), and within its stated bias on a plant
at the measured noise with a bell-shaped top, with a capped top and with the slow-side
mistakes, each with an offset the two sensors share. Nobody has run that plant for this
statistic; it comes before step 1 is accepted.
5.5 **Heat at a layout the car has not run no worse**, and the max driver's margin reported
with every run.

### 6. The order of building

1. The tables in the tyres' frame with two readings per pass; the composite limits; the uses
   re-read; q in place of u; the max-driver report. Acceptance: 5.1. No change to the kernel.
2. The driven-axle term in both forms of the solver (the Inferno 86), read before the uses.
3. The second braking reading in the brake table; the braking best by clause 1.2.
4. The scale per lap and node in both forms; the sampler of clause 4.2 with its planted test;
   the heat error bar and the ledger.
5. The wet composites and scales (clause 1.6).
6. The stint-level test for a direction (clause 3.3).

**Dropped**: the exit term as a driver's use; the highest corner and the 99th percentile as
the limit; E\* as a gate. **Not built until tested**: a knob with a direction by phase; a use
that depends on corner radius; a speed axis; a separate day level; the onset post-pass.

### 7. The vote (round E)

All twelve signed the draft. Dissents, each on a named clause, and the clause each member
trusts least, are below. **Four clauses were changed in substance by the lead after the vote
and were not put back to it**: 1.6 (a wet day's water level as a condition, with a new owner's
default, from two dissents), 1.9 (its default reversed: the level stays on the car's own path,
from one dissent and three least-trusted votes), 3.1 (q may go below 0, from one dissent, and
the knob's end lap added by the lead) and 3.4 (the range judged in q, from one dissent). The
other amendments correct figures or wording.

| member | vote | least trusted |
|---|---|---|
| drv-limits | sign except 1.9 (not run at this statistic; now written as open) and a figure in 1.8 (corrected) | 1.9 |
| drv-phases | sign except two of its own numbers in 2.1 and 3.1 (corrected) | 2.1, "no exit use" on the formula cars |
| drv-knob | sign except 3.1's "every logged lap has q > 0" (amended: q may go below 0) | 1.9 |
| drv-montecarlo | sign except 2.3 and the spread quoted in 0 (both readings now quoted) | 2.3's slow-mistake tail |
| drv-conditions | sign except 1.6 (a wet day's water level is a condition: amended) | 1.6 |
| drv-consumer | sign except 3.4 (judge the range in q, not in lap time: amended) | 1.7 |
| drv-skeptic | sign except 0 (the grip is "the best shown, held fixed", not "measured": amended) and 5.1 (amended) | 5.4 |
| exp-frontier | sign except 1.1's "floor" (qualified: amended) and 5.4 (amended) | 1.6 |
| exp-bayes | sign | 2.3's variance shares |
| exp-coach | sign | 3.1's braking side |
| exp-dynamics | sign; withdrew its own default on 1.3 (all three cars) | 1.9 |
| exp-tyre | sign except 1.6 (the KK-F's wet braking shortfall is the driver's: noted in 1.6 and 2.1) | 1.6 |

**The clauses the committee trusts least**: 1.6 (the wet: three members), 1.9 (which curvature:
three), 2.3 (the scatter's sizes: two), then 1.7, 2.1, 3.1's braking side and 5.4 (one each).

### 8. The deciding tests (round D), as they returned

All in sample. E\* is the single number proposed in the debate for "a slower lap looks like a
real slower lap": for pairs of same-day laps 1–4 % apart, the RMS round the lap of the
simulated less the logged change of speed per 1 % of pace, with one half of the days against
the other as reality against itself (0.178 / 0.198 / 0.225 m/s per 1 % for today's knob against
0.147 / 0.147–0.160 / 0.238).

- **The selection test** (two cells, 20 simulated worlds per arm). Laps with **no
  knob**, only per-corner scatter, give within-day slopes of the logged size: lateral at the
  apexes +2.0 to +2.4 % per 1 % of pace (logged 2.05 on the Inferno 86; today's knob 2.50). A
  lap-level direction planted in the knob cannot be read back from within-day slopes (within
  2 sd of a uniform knob on every reading), and **E\* separates no arm**, even where today's
  knob is the truth. What is left that selection does not make: the braking slope (the Inferno
  86's logged 1.22 sits below every arm; the KK-SII's 4.40 is matched only by a direction
  carried at the stint level, whose ratio was itself read from these logs), and part of the
  Inferno 86's lateral gap, which scatter not tied to grip would also make. The KK-SII's
  lateral verdict is void (the plant over-makes its spread). The probabilistic modeller's claim stands in the
  weaker form: the slopes can be made by selection, whether or not they are.
- **The expected best of m passes through the owner's check**. With the sensor noise taken
  out the levels are 1.253 / 1.478 / 1.517 at m = 10, 1.278 / 1.492 / 1.539 at m = 20 and
  1.293 / 1.510 / 1.589 at m = 50 (Inferno 86 / KK-F / KK-SII). The max driver beats the best
  logged lap in all 30 cells at the day-bootstrap 5 % level; the thin cell is the Inferno 86 at
  Fuji (0.21 / 0.44 / 0.74 %). Only 15 of 30 cells stand on their own layout's passes; the KK-F's
  level is Tsukuba's alone. Against the driver coach's theoretical best lap (sum of best segments,
  pooled over days; a comparison, not a gate) the Inferno 86 at Fuji is slower at every m and
  the KK-F at Tsukuba at m = 10 and 20. Neither check picks m.
- **A speed axis for the limit**. No rise of the limit with corner speed is
  shown; downforce is not excluded. Against the tyre specialist's size for a small wing set (+5.4 to +14.4 %
  per 1000 (m/s)², the low end likely) the KK-SII excludes the upper half; the KK-F's +7.5 is
  inside the range but rests on five Tsukuba corners and on the load division. A wing of the
  likely size would own about half of the corner-speed apex pattern.
- **The driven axle**. Rear share from
  production, nothing fitted. The slow-minus-fast exit gap falls 0.39 / 0.81 / 0.79 → 0.03 /
  0.50 / 0.45 m/s (Inferno 86 / KK-F / KK-SII); the apex gap rises 0.06–0.07 m/s (u climbs
  1–2 % and spends the time in the corners); the exit's wrong pace slope is halved; the count
  is 58 of 76 (5 won, 3 lost: inside chance), 46 with its own traces blurred. Against it:
  logged over simulated exit acceleration by corner-speed band is 0.92 / 0.90 / 1.06 on the
  Inferno 86 but 1.02 / 0.91 / 0.95 (KK-SII) and 1.00 / 1.00 / 0.93 (KK-F): the formula cars
  leave slow corners as hard as the whole car's grip allows (confounded: the simulation is
  0.3–0.9 m/s faster at those minimums). Verdict: supported on the Inferno 86, not settled on
  the formula cars. The convention was re-read with the term on for four folds only.
- **The single number, and mistakes**. The E\* rule: a form beats today's if today's is worse in
  95 % or more of 1,000 day resamples, per car; four of 18 tests clear it, all on the KK-SII,
  with no correction for the number of tests. Mistakes have a heavy tail on every car: the slow
  side beyond 3 sd holds 1.8–2.4 % of segment passes against 0.5–0.6 % on the fast side and
  0.2 % for a Gaussian; the worst two segments do not carry the loss by the rule set (75–79 %
  against 67–72 % for a Gaussian).

### 9. What the literature added

Four researchers and a report (kept outside the repository's tracked files): no published work
separates a car's limit from a driver's use of it from one driver's logs without a tyre model,
extra sensors or drivers changing cars; published driver models are a few shares of the car's
envelope per phase with timing terms; nothing published samples the set of driver shortfalls
that give one lap time, though the method is standard (draw all but one, solve the last so the
lap time matches, weight by the prior over the lap time's sensitivity); and no source measures
where one driver's slower lap loses its time, which only these logs can answer.
