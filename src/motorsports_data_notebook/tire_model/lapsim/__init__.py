"""Lap simulation: separable track and car models (issues #202–#208).

Exploration code. Nothing here is read by the production pressure model;
reports and fitted artifacts go to ``workspace/lapsim/`` (not committed).

Segment B (#204), the track model:

- :mod:`.geometry`: every lap's position, course and speed on one clock,
  and the laps a track is built from (B1).
- :mod:`.track`: the curvature profile of a layout from the median of its
  build laps' Doppler course, and the inertial profile it is checked
  against (B2).
- :mod:`.dem`: ground elevation from the GSI tiles, for B2's altitude
  check and the road's bank.
- :mod:`.bank`: the road's bank at a track's nodes, from the elevation
  model across the line (#299).
- :mod:`.drawings`: the operators' published corner geometry, transcribed,
  for B2's check against it.
- :mod:`.pit`: where an out-lap from the pit joins the driven line.
- :mod:`.holdout`: the ladder of holdouts, day against day up to car
  against car, through production's heat chain (B3).
- :mod:`.gates`: the gate reports; :mod:`.cli` runs them.

Segment C (#205), the car model:

- :mod:`.heat`: production's heat chain on one lap of any series.
- :mod:`.frame`: the session frame record, what a session's accelerometer
  reads of a ground-frame motion (C0).
- :mod:`.samples`: what each car did on its build laps, as tables: full
  throttle, coast, braking zones and apexes (C1).
- :mod:`.car`: the car model read from those tables (C2).
- :mod:`.car_gate`: gate C2 through the solver: the convention for
  ``mu_y`` with ``ref_u``, recovery of a known car, in-sample speed.
- :mod:`.car_holdout`: the car on held-out days and layouts against two
  planted wrong cars (C3).

Segment D (#206), the solver:

- :mod:`.solver`: the quasi-steady-state lap solver and ``solve_u``.
- :mod:`.heat`: production's chain on a simulated lap (``sim_heat``).
- :mod:`.heat_gate`: that chain against production's own per-lap heat on
  logged laps (gate D1).
- :mod:`.pace`: the pace table, a car's laps at a layout from a slow use
  of its limit to the max driver's, with the out-lap from the pit join;
  :mod:`.pace_gate` holds it to a direct solve (gate D1).
- :mod:`.car_holdout` also carries gate D3: lateral heat by side.

The car's limits and the driver (#312):

- :mod:`.limits`: the car's cornering limit as the expected best of m
  passes, and the best braking it has held.
"""
