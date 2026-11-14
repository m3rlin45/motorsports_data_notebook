"""Lap simulation: separable track and car models (issues #202–#208).

Exploration code. Nothing here is read by the production pressure model;
reports and fitted artifacts go to ``workspace/lapsim/`` (not committed).

Segment B (#204), the track model:

- :mod:`.geometry`: every lap's position, course and speed on one clock,
  and the laps a track is built from (B1).
- :mod:`.track`: the curvature profile of a layout from the median of its
  build laps' Doppler course, and the inertial profile it is checked
  against (B2).
- :mod:`.gates`: the gate reports; :mod:`.cli` runs them.
"""
