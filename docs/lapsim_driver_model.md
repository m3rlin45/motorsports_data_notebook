# Lap simulation: a driver model

The car's limits apart from the driver's use of them. Design of 2026-11-17; the body of issue #312.

Part of #202. Follows the review of the car model (#205) by seven reviewers on 2026-11-16/17; its measurements are in `docs/tire_model.md` §2.12 C5 and are quoted here with the reviewer who made them. Feeds #206 (the pace knob, D2), #208 (heat at a track the car has not run) and #263 (where the car can go faster). A design, not yet built; the numbers are the review's probes and are re-measured by the gate below.

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
