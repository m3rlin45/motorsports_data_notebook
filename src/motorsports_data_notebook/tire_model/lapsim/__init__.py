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
"""
