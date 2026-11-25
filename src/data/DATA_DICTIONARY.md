# Data dictionary: `data/processed/persons.parquet`

Owner: Sharan Raj JANAKIRAMAN (M1). Generated from the file built by `python -m src.data.make_dataset`.

- **871,115 people** (one row each), **382,721 accidents**, 2019-2025. KSI rate **18.1%**.
- Codes are kept as **text** (`'7'`, not `7`): they are categories, not quantities. Numeric columns are floats.
- Missing = `NaN` (all raw codes -1, empty, '.', N/A, and 0 where 0 means "unknown" were converted; see `src/data/clean.py`).
- `0` is a **real category** in `obs`, `obsm`, `choc`, `infra`, `situ`, `vosp`, `secu1-3`, `locp`.
- Removed: `grav` (the outcome, replaced by `KSI`), free-text address and road numbers, duplicate IDs, `lartpc` and `occutc` (over 98% missing).
- Fixed: speed limits outside 10-130 km/h and ages over 100 set to missing (see EDA notebook, section 3). Reasons in `src/data/make_dataset.py`.
- Official meaning of every code: ONISR, "Description des bases de données annuelles des accidents corporels".

| Column | Group | Meaning | Type | Missing % | Distinct values |
|---|---|---|---|---|---|
| `person_id` | key | Unique person ID (year_rownumber), built by M1 | code (text) | 0.0 | 871,115 |
| `Num_Acc` | key | Accident ID. Use it as the group for GroupKFold | code (text) | 0.0 | 382,721 |
| `id_vehicule` | key | Vehicle ID (a pedestrian gets the vehicle that hit them) | code (text) | 0.0 | 649,419 |
| `place` | person | Seat position in the vehicle (10 = pedestrian) | code (text) | 0.0 | 10 |
| `catu` | person | Road-user type: 1 driver, 2 passenger, 3 pedestrian | code (text) | 0.0 | 3 |
| `sexe` | person | Sex: 1 male, 2 female | code (text) | 1.4 | 2 |
| `an_nais` | person | Birth year (numeric). Age = year - an_nais | number | 1.5 | 107 |
| `trajet` | person | Trip purpose (home-work, leisure, ...) | code (text) | 28.1 | 6 |
| `secu1` | person | 1st safety equipment (0 none, 1 belt, 2 helmet, ...) | code (text) | 1.3 | 10 |
| `secu2` | person | 2nd safety equipment | code (text) | 41.1 | 10 |
| `secu3` | person | 3rd safety equipment | code (text) | 96.3 | 10 |
| `locp` | person | Pedestrian location (0 = not a pedestrian) | code (text) | 46.1 | 10 |
| `actp` | person | Pedestrian action | code (text) | 92.1 | 11 |
| `etatp` | person | Pedestrian alone or in a group | code (text) | 92.1 | 3 |
| `year` | key | Accident year. 2019-2023 = train, 2024-2025 = test | number | 0.0 | 7 |
| `jour` | accident | Day of month | code (text) | 0.0 | 31 |
| `mois` | accident | Month | code (text) | 0.0 | 12 |
| `hrmn` | accident | Time HH:MM | code (text) | 0.0 | 1,440 |
| `lum` | accident | Light: 1 day, 2 dusk/dawn, 3 night no lights, 4 lights off, 5 lights on | code (text) | 0.0 | 5 |
| `dep` | accident | Department code (97x = overseas) | code (text) | 0.0 | 107 |
| `com` | accident | Commune code (very many values) | code (text) | 0.0 | 25,588 |
| `agg` | accident | 1 outside towns, 2 inside towns | code (text) | 0.0 | 2 |
| `int` | accident | Intersection type | code (text) | 0.0 | 9 |
| `atm` | accident | Weather | code (text) | 0.0 | 9 |
| `col` | accident | Collision type | code (text) | 0.4 | 7 |
| `lat` | accident | Latitude (numeric) | number | 0.0 | 330,232 |
| `long` | accident | Longitude (numeric) | number | 0.0 | 338,730 |
| `catr` | location | Road category: 1 motorway, 2 national, 3 departmental, 4 municipal, ... | code (text) | 0.0 | 8 |
| `circ` | location | Traffic regime (one-way, two-way, ...) | code (text) | 5.7 | 4 |
| `nbv` | location | Number of lanes (numeric) | number | 4.2 | 12 |
| `vosp` | location | Reserved lane (0 none, cycle lane, ...) | code (text) | 0.8 | 4 |
| `prof` | location | Road slope | code (text) | 0.0 | 4 |
| `plan` | location | Road shape (straight, curve, ...) | code (text) | 0.0 | 4 |
| `larrout` | location | Road width, m (mostly empty) | number | 81.9 | 219 |
| `surf` | location | Surface state (dry, wet, ice, ...) | code (text) | 0.0 | 9 |
| `infra` | location | Infrastructure (0 none, tunnel, bridge, roundabout ...) | code (text) | 0.9 | 10 |
| `situ` | location | Accident situation (on road, shoulder, ...) | code (text) | 0.0 | 7 |
| `vma` | location | Speed limit km/h (numeric; values outside 10-130 set to missing) | number | 1.6 | 27 |
| `senc` | vehicle | Direction of travel | code (text) | 6.1 | 3 |
| `catv` | vehicle | Vehicle category (7 car, 1 bicycle, 33 motorcycle, ...) | code (text) | 0.2 | 30 |
| `obs` | vehicle | Fixed obstacle hit (0 none, 2 tree, ...) | code (text) | 0.0 | 18 |
| `obsm` | vehicle | Moving obstacle hit (0 none, pedestrian, vehicle, ...) | code (text) | 0.0 | 7 |
| `choc` | vehicle | Initial impact point (0 none) | code (text) | 0.0 | 10 |
| `manv` | vehicle | Main manoeuvre before the accident | code (text) | 6.2 | 26 |
| `motor` | vehicle | Engine type | code (text) | 6.6 | 6 |
| `KSI` | TARGET | 1 = killed or hospitalised > 24 h, 0 = otherwise | number | 0.0 | 2 |
