# Ground-truth dataset schema (`datatide_ground_truth.parquet`)

This document describes the **canonical processed table** used for EDA and downstream ML: how it is **aggregated and joined** (no ocean or statistical model is fit inside the build script), what each column means, and where values originate.

**Build script:** `scripts/process/build_ground_truth_dataset.py`  
**Configuration:** `configs/process.yaml` (column selection, county→environment mapping, SCCOOS county allowlist)  
**Companion:** `configs/fetch.yaml` (raw pull targets, including CDIP bundle names)  
**Local output:** `data/processed/datatide_ground_truth.parquet` + `datatide_ground_truth_meta.json`  
**Note:** The `data/` tree is **gitignored** (except `.gitkeep`); clones must **regenerate** files locally (see `context/README.md`).

---

## Grain and design

- **One row per laboratory result record** from the Tier-1 California beach bacteria monitoring export (multiple rows per station per day are possible if several parameters or replicate samples exist).
- **Primary time key for joins:** `sample_date` — calendar date (`YYYY-MM-DD`) derived from Tier-1 `SampleDate` (parsed in the ETL).
- **Spatial logic for CDIP:** each row’s monitoring coordinates (`Station_UpperLat`, `Station_UpperLon`) are used to pick the **nearest** CDIP buoy (great-circle distance) among bundles listed in `configs/fetch.yaml` → `cdip.bundles`.
- **Spatial logic for CCE1 moorings:** same station coordinates pick the **nearest** of the configured mooring IDs (typically **13** and **15** from `data/raw/cce_moorings/OS_CCE1_*_D_*.nc`); daily means at **shallowest depth** (CTD: `TEMP`, `PSAL`; ADCP: `UCUR`, `VCUR`) merge on (`sample_date`, `cce_mooring_id`).
- **HF radar:** latest `hf_*_water_u_*.csv` / `hf_*_water_v_*.csv` pair under `hf_radar/` is reduced to **daily** mean u/v per grid cell, then the **nearest** cell to the station (within the fetch bbox) fills `hf_*` columns (rolling NRT window only—see `fetch_hf_radar.py`).
- **Regional joins** (precipitation, tide gauge, SCCOOS) use **`CountyName`** on the bacteria row mapped through `configs/process.yaml` → `county_to_env` and `sccoos_join_counties` where applicable.

---

## Build pipeline (aggregation / join order)

1. **Read** the newest `data/raw/ca_swrcb_bacteria/download_*.csv` in chunks (see `process.yaml` → `bacteria_chunk_rows`).
2. **Derive** `sample_date` from `SampleDate`; normalize `CountyName` (strip whitespace).
3. **Map** county → `precip_bucket` and `tide_station_id` via `county_to_env`.
4. **Merge** `data/raw/ca_swrcb_stations/download_*.csv` (latest file) on `Station_id` → station supplement columns.
5. **Merge** latest `data/raw/ca_swrcb_beach_detail/download_*.csv` on `BeachName_id` → columns prefixed `beach_detail_*`.
6. **Merge** daily GHCN precipitation from `data/raw/noaa_precip/cdo_PRCP_*.json` on (`sample_date`, `precip_bucket`).
7. **Merge** daily tidal range from `data/raw/noaa_tides/tides_hilo_*.json` on (`sample_date`, `tide_station_id`).
8. **Merge** SCCOOS Del Mar daily means on `sample_date`, then **null out** SCCOOS columns for counties **not** in `sccoos_join_counties`.
9. **Assign** `cdip_bundle` by nearest buoy; **merge** daily CDIP means on (`sample_date`, `cdip_bundle`).
10. **Merge** San Diego County monthly coastal JSON on (`CountyName`, `calendar_month`).
11. **Assign** HF radar u/v/speed/distance from latest griddap CSV pair (per-row nearest grid cell, bbox-limited).
12. **Merge** IBWC Tijuana daily means on `sample_date` (`ibwc_tijuana_*`).
13. **Merge** optional BWTF **statewide daily summary** on `sample_date` (only if `bwtf_water_quality_*.json` shards exist after an authenticated fetch).
14. **Assign** nearest CCE mooring id; **merge** CCE daily CTD/ADCP means on (`sample_date`, `cce_mooring_id`).
15. **Write** Parquet (Zstd) and `datatide_ground_truth_meta.json`.

---

## Overlapping vs distinct columns

Some fields describe the **same kind of quantity** (e.g. beach name, agency, coordinates) but come from **different exports** or **different geometry**. The wide table keeps both so you can audit joins and catch mismatches; for modeling you typically **choose one column per concept** unless you explicitly want redundancy checks.

### Same concept, two Tier-1 sources (often aligned, not guaranteed identical)

- **On the bacteria row:** `Beach_Name`, `BeachType`, `AB411Beach`, `USEPAID`, `WaterBodyName`, `Agency_Name`, `Beach_UpperLat`, `Beach_UpperLon`, etc.
- **From the beach-detail merge:** `beach_detail_Beach_Name`, `beach_detail_BeachType`, `beach_detail_AB411Beach`, `beach_detail_USEPAID`, `beach_detail_WaterBodyName`, `beach_detail_Agency_Name`, `beach_detail_Beach_UpperLat`, `beach_detail_Beach_UpperLon`, etc.

**Join key:** `BeachName_id`. If the two programs disagree (refresh lag, spelling, or correction in one file only), both columns remain useful for QA.

### Coordinates: same “family,” different points (not duplicates)

| Column group | Meaning |
|--------------|---------|
| `Station_UpperLat` / `Station_UpperLon` | Monitoring **station** location on the result row—used for **nearest** CDIP, HF radar, and CCE mooring assignment. |
| `Beach_UpperLat` / `Beach_UpperLon` | **Beach** point from the bacteria export. |
| `beach_detail_Beach_UpperLat` / `…Lon` / `…LowerLat` / `…LowerLon` | Beach-detail file **extent / corners** when provided. |

Use **station** coords for anything tied to how joins were computed; use beach coords for maps or beach-centric narratives.

### Dates and join keys

- **`SampleDate`** — raw value from the state CSV (often `M/D/YYYY` text).
- **`sample_date`** — normalized **`YYYY-MM-DD`** for all environmental merges.

`precip_bucket` and `tide_station_id` are **lookup keys** (county → regional rain bucket / CO-OPS gauge), not measurements.

### Environmental layers: different instruments or footprints

- **`regional_ghcn_prcp_mm`** — daily rain at a **regional GHCN** station, not at the water’s edge.
- **`sccoos_delmar_*`** — **one** nearshore mooring (Del Mar); nulled outside `sccoos_join_counties` in `process.yaml`.
- **`cdip_*`** — **nearest CDIP buoy** wave statistics (realtime files → often short history unless archive URLs are used).
- **`cce_*`** — **nearest CCE1 mooring** (IDs 13/15 in current config); offshore CTD/ADCP at shallow depth—**not** surf-zone water.
- **`hf_*`** — HF radar **surface** currents (grid nearest station); **not** the same as mooring `cce_ucur` / `cce_vcur`.
- **`ibwc_*`** — **Tijuana River** stage/discharge (runoff / transboundary context), not beach concentration.
- **`bwtf_*`** — **Statewide same-day summary** of community BWTF numeric results in a CA bounding box when JSON shards exist; **not** the Tier-1 lab result on that row.
- **`sd_coastal_*`** — **Monthly** San Diego county program aggregates; different temporal grain than a single lab result.

---

## Column reference

Columns appear in a stable order in Parquet; the list below groups them by role.

### Tier 1 — Bacteria spine (State Water Board / data.ca.gov)

These come from the **bacteria monitoring results** CSV unless noted. Official portal hub: [Beach water quality postings and closures](https://lab.data.ca.gov/dataset/beach-water-quality-postings-and-closures).

| Column | Description | Source |
|--------|-------------|--------|
| `RESULTS id` | Internal result row identifier in the state export. | Bacteria CSV |
| `SampleDate` | Sample date as published (often `M/D/YYYY` text in raw file). | Bacteria CSV |
| `StartTime` | Sample start time when present. | Bacteria CSV |
| `Parameter` | Analyte / indicator (e.g. Enterococcus, E. coli). | Bacteria CSV |
| `Qualifier` | Lab/result qualifier code when present. | Bacteria CSV |
| `Result` | Reported numeric or categorical result (as provided). | Bacteria CSV |
| `Unit` | Measurement unit for `Result`. | Bacteria CSV |
| `AnalysisMethod` | Method description or code. | Bacteria CSV |
| `SampleType` | Sample matrix / program type field from export. | Bacteria CSV |
| `LabID` | Laboratory identifier when present. | Bacteria CSV |
| `Weather` | Field observations (weather) when captured in export. | Bacteria CSV |
| `TidalHeight` | Categorical/text tidal stage at sampling when present. | Bacteria CSV |
| `SurfHeight` | Surf height field when present. | Bacteria CSV |
| `Turbidity` | Turbidity when present. | Bacteria CSV |
| `WaterColor` | Water color when present. | Bacteria CSV |
| `Station_id` | Monitoring station key (join key to stations export). | Bacteria CSV |
| `BeachName_id` | Beach identifier (links to beach detail export conceptually). | Bacteria CSV |
| `Station_Name` | Station label. | Bacteria CSV |
| `Station_UpperLat` / `Station_UpperLon` | Coordinates used for mapping and **nearest CDIP buoy**. | Bacteria CSV |
| `CountyName` | County or program grouping label on the result row (used for regional joins). | Bacteria CSV |
| `Beach_Name` | Beach name. | Bacteria CSV |
| `BeachType` | Beach type classification from export. | Bacteria CSV |
| `WaterBodyName` | Named water body when present. | Bacteria CSV |
| `AB411Beach` | AB 411 program flag field from export. | Bacteria CSV |
| `USEPAID` | USEPA identifier field when present. | Bacteria CSV |
| `Beach_UpperLat` / `Beach_UpperLon` | Beach polygon/point metadata from export. | Bacteria CSV |
| `Agency_Name` | Sampling or managing agency name. | Bacteria CSV |

### Derived keys and join keys (ETL)

| Column | Description | How computed |
|--------|-------------|--------------|
| `sample_date` | ISO date `YYYY-MM-DD`. | `pd.to_datetime(SampleDate)` then normalized to date string. |
| `precip_bucket` | Regional GHCN bucket name (e.g. `los_angeles`). | `CountyName` → `county_to_env.precip_bucket` in `process.yaml`. |
| `tide_station_id` | NOAA CO-OPS station number string. | `CountyName` → `county_to_env.tide_station_id`. |
| `calendar_month` | `YYYY-MM` period for monthly contextual joins. | From `sample_date`. |
| `cdip_bundle` | CDIP bundle id (e.g. `cdip_100`) for nearest buoy. | Nearest of buoy lat/lon in NetCDF metadata to station coordinates. |

### Tier 1 — Station supplement (separate SWRCB export)

Merged on `Station_id` from latest `data/raw/ca_swrcb_stations/download_*.csv`.

| Column | Description |
|--------|-------------|
| `station_description_ref` | `Station_Description` from stations file (renamed to avoid ambiguity). |
| `agency_station_id` | `AgencyStationIdentifier`. |
| `station_county_code` | `CountyCode` from stations file. |
| `station_datum` | Vertical `Datum` (e.g. survey reference) for the station record. |

### Tier 2 — NOAA / GHCN precipitation (NCEI CDO JSON)

Built by scanning `data/raw/noaa_precip/cdo_PRCP_*.json`. Values are **millimeters per day** (CDO stores tenths of mm; ETL divides by 10).

| Column | Description |
|--------|-------------|
| `regional_ghcn_prcp_mm` | Daily precipitation at the **regional** GHCN station for `precip_bucket` on `sample_date`. |

**Caveat:** This is **not** rain at the beach; it is rain at the chosen regional gauge (see `fetch.yaml` → `noaa_precip.stations`).

### Tier 2 — NOAA Tides & Currents (high/low predictions)

From `data/raw/noaa_tides/tides_hilo_*.json`. Per day and gauge, tidal range = max(high) − min(low) among prediction extrema whose local timestamp falls on that calendar day.

| Column | Description |
|--------|-------------|
| `tide_range_hilo_m` | Tidal range in **meters** for `tide_station_id` on `sample_date`. |

### Tier 3 — SCCOOS ERDDAP (Del Mar mooring, 1 m)

Sources: `data/raw/sccoos_erddap/sccoos_delmar_temperature_*.csv` and `sccoos_delmar_salinity_*.csv`. Second header row (units) skipped. All timestamps parsed as UTC; rows aggregated to **daily mean** per `sample_date` across shards.

| Column | Description |
|--------|-------------|
| `sccoos_delmar_temp_1m_c` | Mean **1 m temperature (°C)** at Del Mar (~32.93°N) on `sample_date`. |
| `sccoos_delmar_salinity_1m_psu` | Mean **1 m salinity (PSU)** on `sample_date`. |

**Caveat:** Applied only to counties in `sccoos_join_counties`; other counties are set to null after merge. This is a **regional** nearshore signal, not station-specific.

### Tier 2 — CDIP waves (nearest buoy, daily mean)

Sources: latest `*p1_rt*.nc` per bundle under `data/raw/cdip/`, configured in `fetch.yaml` → `cdip.bundles`. Variables: `waveHs` (m), `waveTp` (s), indexed by `waveTime`. Daily mean computed per `sample_date` and `cdip_bundle`.

| Column | Description |
|--------|-------------|
| `cdip_wave_hs_m_mean` | Mean significant wave height (**m**) for assigned buoy on `sample_date`. |
| `cdip_wave_tp_s_mean` | Mean peak period (**s**). |

**Caveat:** Realtime NetCDF files typically cover a **rolling recent** window. Historical bacteria dates often have **null** wave fields until **archive** THREDDS paths are used in fetch config.

### Tier 5 — San Diego County coastal program (monthly, county-wide)

Source: latest `data/raw/sd_county_beach/beach_advisories_*.json`. Each JSON row is a **monthly aggregate** for the county program (not per beach, not per lab sample).

| Column | Description |
|--------|-------------|
| `sd_coastal_date` | Month-end (or report) timestamp from source JSON. |
| `sd_coastal_frequency` | e.g. `monthly`. |
| `sd_coastal_*` | Advisory/closure/beach-day counts and totals as in Socrata export. |

**Join:** `CountyName == "San Diego"` and `calendar_month` match. Rows for other counties have nulls in `sd_coastal_*` columns.

---

## Additional merged columns (summary)

| Block | Key columns | Notes |
|--------|-------------|--------|
| Beach detail | `beach_detail_*` | Extra metadata vs bacteria row alone. |
| HF radar | `hf_water_u_mps`, `hf_water_v_mps`, `hf_current_speed_mps`, `hf_grid_dist_km` | **Short NRT window**; most historical bacteria dates will be null unless you point fetch at an archive product. |
| CCE moorings | `cce_mooring_id`, `cce_temp_shallow_c`, `cce_psal_shallow_psu`, `cce_ucur_shallow_mps`, `cce_vcur_shallow_mps` | **Offshore** background; nearest mooring is a pragmatic aggregate, not a surf-zone measurement. |
| IBWC | `ibwc_tijuana_stage_m_daily_mean`, `ibwc_tijuana_discharge_cms_daily_mean` | Parsed from legacy text feed when fetch succeeds. |
| BWTF | `bwtf_ca_median_result`, `bwtf_ca_n_samples`, `bwtf_ca_median_entero` | **Statewide same-day context** from community samples in a CA bbox; null if no JSON shards. |

Raw files remain under `data/raw/<source>/` for deeper joins (e.g. site-level BWTF) beyond this table.

---

## Provenance and reproducibility

- After each build, read **`data/processed/datatide_ground_truth_meta.json`**: input bacteria path, row counts, CDIP buoy list, layer list.
- **Do not commit** Parquet/CSV under `data/` to GitHub; re-run fetch + process locally (see `context/README.md` for time order and duration).

---

## Related documentation

- [`DATASETS.md`](DATASETS.md) — tiered sourcing strategy and portals.  
- [`data/processed/README.md`](../data/processed/README.md) — quick Parquet FAQ (binary vs “lines” in editors).  
- [`structure.md`](structure.md) — repo layout and artifact index.
