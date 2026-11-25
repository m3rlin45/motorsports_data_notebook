# Evaluating the calculator: the current model against the simulation-based one

How the tire pressure calculator's accuracy is measured, how the
simulation-based calculator (issues #202–#208: a track model, a car
model, a driver model, a heat model and the tire pressure model, stacked)
is set beside the current one, and how to run every comparison again.
The numbers of the first runs (2026-11-24/25) and the reasoning behind
each choice are in `docs/tire_model.md` §2.12 ("Gate F" onward); this
page is the method and the commands.

**Status.** The simulation-based calculator is not cut over. Everything
here compares it with the current calculator without changing what ships:
`tire_model.json`, `predict.py` and the two calculator heads are as they
were, and nothing here writes under `data/`.

## One command

```bash
just tire-sim-compare
```

runs every comparison below in order and leaves a report for each in
`workspace/lapsim/reports/`. The first run fits about fifty folds of the
pressure model (roughly an hour on eight cores); the fits are kept in
`workspace/lapsim/cache/pressure_folds/`, keyed by the sessions left out,
the dataset's files and the bytes of the fit's source, so later runs take
minutes unless the fit or the data changed. Each command also runs alone
(`just tire-lapsim <command>`, under `ulimit -v 10000000`).

The lap simulation's own inputs (the lap index, the cars' sample tables,
the tracks from other cars' laps) are built on first use and kept in
`workspace/lapsim/cache/` and `workspace/lapsim/tracks/folds/`; the
tracks themselves are committed (`data/lapsim/tracks/`).

## The measure the calculator is judged by: MAE over folds of held-out sessions

`just tire-lapsim check-pressure-holdout` → `FH_pressure_holdout.md`

This is the calculator's standing holdout (`just tire-predict-holdout
--n-folds 20`): sessions are bucketed by (track, car, condition), two per
bucket are held out per fold (dry buckets of ten sessions or more, rain
buckets of three), the pressure model is refitted without them, and each
held-out lap is predicted from *calculator inputs* only: the pit-exit
temperature, the cell's out-lap, N laps of the session's 25th-percentile
lap time (what a driver would type) and the compound. The figure is the
hot-pressure residual at the end of each lap, prediction less the TPMS
reading: MAE and signed bias in bar, per tyre, pooled, per car, per
condition and per cell.

Each fold is fitted once and scored by four calculators on the same rows:

| arm | what answers for the session's cell |
|---|---|
| `calculator` | the fold's own entry: the current calculator, row for row the standing holdout |
| `table` | the simulation, as a precomputed pace table read at the typed lap time |
| `direct` | the simulation, the lap of the typed lap time driven for the asking, in the session's air |
| `direct_setup` | the same with the day's logged shift point and gears |

**The holdout applies to every model.** A fold's held-out sessions, and
only those, are out of everything a simulated answer is made of: the
pressure model's fit (K, τ, κ, the force shares, the compound
multipliers), the car model with its limit, braking use and typical use,
the track's curvature, the frame factor and the fed use. Do not hold out
more (a whole car, a whole track) in this evaluation, and do not leave a
held-out session in any of them. What a fold does not remove: which laps
count as build laps, the track's reference line, bank, grade and pit
exit, and the compound label of an unlabelled session (the standing
holdout's own practice).

The report also has the same rows **at levels of pace** (P75, P90, P95:
the laps above that percentile of pace among the held-out laps of the
same raw car, track and condition) and **the shootout** between the pace
table and the direct lap (how far apart their predictions are, and their
MAE session by session).

How to read it:

- Lead with the pooled and per-car MAE; the per-session difference with
  its standard error says whether a gap is more than noise.
- Every arm predicts every lap of a session from one typed lap time, so a
  "fast lap" in the pace levels is one driven harder than its plan.
- This MAE responds weakly to the heat source: a heat 5 % too low scores
  *better* in three of five cells (gate F, S2), and the constants under
  every arm were fitted on logged heat. A difference of about 1 mbar
  between two heat sources is not by itself evidence for either.

## A lap time under anything trained on

`just tire-lapsim pressure-pace-margins` → `FHM_pressure_pace_margins.md`
`just tire-lapsim check-pressure-holdout --fastest 0.8` → `FHP_pressure_holdout_pace.md`

The same evaluation with the folds replaced by a holdout by pace: per raw
car and track with ten dry sessions, the fastest fraction of them (by
typed lap time) is held out of every model, one fold a cell. The first
command makes no fit and shows what each fraction asks for; at a quarter
the held-out sessions are only half a per cent under their training pace,
at 80 to 90 % they are 2 to 6 % under and most are under every single lap
left in training. The report adds the breakdown by how far under the
training pace a session is (under every lap left; more than 2 % and 3 %
under), over every cell and per cell.

Know what it is: a pace not seen *at that track* (the car's sessions at
its other tracks stay in training), and at 80 to 90 % a cell's training
is a handful of its slowest dry sessions.

## A change to the pooling of the cars

`just tire-lapsim check-pressure-holdout --split KK-F KK-F-V2` → `FHS_pressure_holdout_split.md`
`just tire-lapsim compare-pressure-holdouts --a "" --b __split` → `FHC_pressure_holdout_compare.md`

`--split` fits the raw cars named as a pooled car of their own (named
after the first) on the same folds; it changes the pooling for that
process only and writes nothing. `compare-pressure-holdouts` then sets
two saved runs against each other row for row: each arm's MAE and bias in
both, per raw car, per raw car and condition, and the change session by
session. The saved rows are
`workspace/lapsim/cache/pressure_rows/holdout_rows<tag>.parquet` (tags:
none, `__split`, `__fast<percent>`); any two can be compared.

## A track the car has not run: gate F

`just tire-lapsim check-pressure` → `F_pressure.md`

The question the lap simulation was built for. A fold is a pooled car at
a track with **every** one of its sessions left out, so the pressure
constants, the car model and (at Tsukuba and Fuji) the track have not
seen the cell. The figure is not against the TPMS: it is the cold
pressure the calculator would recommend for the end of flying lap 5,
against the one the day's own logged heat gives, in mbar, with **another
logged day beside it** as the yardstick. A cell (a raw car at a track on
one tyre) passes when an arm is no further from the day than another day
is. Two forms: `heat`, every side at the day's lap time; `chain`, each
side at the lap time and with the out-lap its own entry supplies.

The report carries, by the same measure:

- the simulation, with its planted controls (a heat 5 % off, another
  car's parameters, 0.85 of the power, a lap time 1 % off);
- the calculator today: its fallback at a cell without logged laps, its
  own entry at a cell it has logged (the day's date left out), and as
  shipped;
- the simulated entry in place of the logged one on the cells the
  calculator has logged (leave-date-out);
- pressure against the TPMS at flying lap 5, reported beside;
- the wet and damp cells (against a run of the same wet day), the
  chronological first visit, and the pooled entry's cost per raw car.

Two arms are always compared on the stints both have.

## Which measure answers which question

| question | measure | command |
|---|---|---|
| Is a calculator more accurate in everyday use? | hot-pressure MAE over held-out folds | `check-pressure-holdout` |
| Does it hold up on the fastest laps? | the same at P75, P90, P95 | in the same report |
| Does it hold up at a pace it was not trained on? | the same, fastest sessions held out | `check-pressure-holdout --fastest F` |
| Would fitting a car apart help? | the same folds refitted, row for row | `--split`, then `compare-pressure-holdouts` |
| Pace table or direct simulation? | the two side by side on the same rows | in every holdout report |
| What does it do at a track the car has never run? | recommended cold pressure against paired real days | `check-pressure` |

## Changing something and measuring again

- A change to the pressure model's fit, or to the dataset, refits every
  fold by itself (the cache is of the fit's source and the data's files;
  add a module to `pressure_gate._FIT_SOURCES` when the fit starts to
  depend on it). A change to the lap simulation or to the scoring does
  not refit.
- A change to a track or to the car model: run the lap simulation's own
  gates first (`check-track-truth`, `check-car-fit`, `check-car-holdout`),
  then `just tire-sim-compare`.
- Write the numbers that matter into `docs/tire_model.md` §2.12: the
  reports live in `workspace/`, which is not committed.
- Compare like with like: the same folds, the same rows (every table
  here is on the rows all its arms have), and the per-session difference
  with its standard error before calling a gap real.

## Where the code is

| piece | file |
|---|---|
| the standing holdout, split into fit / prepare / score | `tire_model/validate.py` (`_fit_fold`, `_prepare_fold_laps`, `_score_fold_laps`) |
| an entry, placing it in a fold's model, the recommended cold pressure | `tire_model/lapsim/pressure.py` |
| the folds, the arms, every report above | `tire_model/lapsim/pressure_gate.py` |
| a pace table as the laps the heat chain reads | `tire_model/lapsim/export.py`, `pace.py` |
| the fed use (the lap an entry stands for) | `tire_model/lapsim/car_gate.py` (`fed_use`) |
| the commands | `tire_model/lapsim/cli.py`, `justfile` |
