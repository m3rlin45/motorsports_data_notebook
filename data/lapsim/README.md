# Lap simulation data

What the lap simulation (`src/motorsports_data_notebook/tire_model/lapsim/`,
`docs/tire_model.md` §2.12) produces that other code is built from. The
gates' reports and caches are not here: they go to `workspace/lapsim/`.

## `tracks/`

One file per layout, written by gate B2 (`just tire-lapsim check-track-truth`):
the reference line, the curvature the heat chain reads
(`kappa_heat_1_per_m`), altitude and grade from the logs, the corners, the
pit exit's join (`meta.pit`) and the road's bank (`bank_rad`, `bank_flag`,
`meta.bank`). `<layout>__wet.json` is the line driven on wet days, where a
layout has the laps for one. A track is rebuilt by running the gate; the
tracks of gate B3's folds (built from some cars' laps only) are caches and
stay in `workspace/lapsim/tracks/folds/`.

| file | layout |
|---|---|
| `fuji_gp.json` | Fuji Speedway, Grand Prix course |
| `motegi_full.json` | Mobility Resort Motegi, road course |
| `sodegaura.json` | Sodegaura Forest Raceway |
| `suzuka_full.json` | Suzuka Circuit, full course |
| `tsukuba_2000.json` | Tsukuba Circuit, Course 2000 |
| `tsukuba_2000__wet.json` | the same, the line driven in the wet |

### Attribution: the road's bank

The bank (the cross-slope of the road at each node of a track) is derived
from the elevation tiles of the Geospatial Information Authority of Japan
(GSI): the 5 m elevation model, read across the driven line and processed
into an angle (`lapsim/bank.py`, `lapsim/dem.py`).

- 出典：国土地理院ウェブサイト（地理院タイル：https://maps.gsi.go.jp/development/ichiran.html）
- 「地理院タイル（標高タイル（基盤地図情報数値標高モデル））を加工して作成」
- Source: GSI Tiles (elevation tiles, Fundamental Geospatial Data digital
  elevation model), Geospatial Information Authority of Japan; processed by
  this project. The tiles themselves are not redistributed here.

Everything else in a track file comes from the cars' own logs. A lap
simulated on these tracks, and anything computed from one (a pace table, a
simulated entry of the tire pressure model), carries the bank in it and
the same attribution.
