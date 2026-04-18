# INTERFACES.md

Contracts for **tabular modeling**: column semantics, keys, time cutoff, and **train/serve** parity. Enables another LLM or engineer to implement ETL and `app/` without guessing.

**Primary keys:** Derive from **Tier 1** California beach water quality data: **monitoring results** joined to **monitoring stations** and **beach detail** (`context/DATASETS.md`). Do not treat Surfrider BWTF or CCE as the primary key spine unless explicitly scoping a side experiment.

---

## Prediction unit (grain)

- **Modeling grain:** **daily** rows: `(station_id_or_beach_key, date)` — one row per beach-site per calendar day (or per sample day aligned to monitoring).
- **`as_of_time`:** still defined for leakage (features through end of *D−1* or start of *D*); all environmental inputs are **aggregated to that calendar day** (or previous days for lags), **not** left at hourly resolution in the training table.
- **Operational cadence:** you may **retrain, evaluate, or publish forecasts week-by-week** (e.g. rolling 7-day batches); that is a **schedule**, not an hourly tide series requirement.
- **Target** = label tied to a **documented** sampling or posting window (e.g. exceedance on day *D* given features through cutoff on *D* or *D−1*).

---

## Feature table (logical schema)

Names are **suggested**; rename consistently in `configs/*.yaml` and code.

| Column | Type | Source tier | Notes |
|--------|------|-------------|-------|
| `station_id` | string | **Tier 1** | From statewide monitoring stations dataset. |
| `beach_key` | string (optional) | **Tier 1** | From beach detail / regulatory grouping. |
| `county` | string | **Tier 1** | |
| `latitude`, `longitude` | float | **Tier 1** | For spatial joins to buoys, tides, radar. |
| `day_of_year` | int | Calendar | From `as_of_date`. |
| `fib_lag1d` | float | **Tier 1** (primary) | Prior **official** FIB reading; optional **Tier 10** only as separate columns if merged carefully. |
| `rain_sum_24h` … `rain_sum_7d` | float | **Tier 2** NOAA precip | Windows end at feature cutoff. |
| `dry_days_since_rain` | int | Derived | From precip series. |
| `tide_*` (e.g. range, mean high/low) | float | **Tier 2** NOAA Tides | Built from **high/low predictions** (`hilo`) aggregated **per calendar day** at join station — not hourly tide series in the feature matrix. |
| `wave_hs_mean`, `wave_tp_mean`, `wave_dir` | float | **Tier 2** CDIP | Nearest / best buoy from mapping table. |
| `sst_mean`, `salinity_mean` | float | **Tier 3** SCCOOS (optional) | ERDDAP/THREDDS aggregates. |
| `surface_current_u`, `surface_current_v` | float (optional) | **Tier 3** HF radar | Grid interpolation to shore. |
| `sst_cce_mean`, `salinity_cce_mean` | float (optional) | **Tier 9** CCE | **Background** only; naming should reflect offshore context. |
| `discharge_ibwc` | float (optional) | **Tier 12** | **South Bay / border** subset only. |

**Join keys:** `station_id` / coordinates map to `{noaa_tide_station, cdip_buoy_id, precip_station_id, sccoos_erddap_dataset, hfr_grid_cell}` via **`data/external/`** reference tables (versioned CSV or Parquet).

---

## Labels (supervision)

| Column | Type | Definition |
|--------|------|------------|
| `target_exceedance` | int {0,1} | 1 if indicator **exceeds** applicable **California** action level for that site/sample (per **Tier 1** fields and posted rules). |
| `fib_value` | float (optional) | Raw count / MPN for regression or evaluation. |

**Surfrider BWTF:** use as **auxiliary** labels or validation only unless `DECISIONS.md` records a deliberate scope change.

---

## Model outputs (for `app/` or batch)

| Field | Type | Meaning |
|-------|------|---------|
| `station_id` / `beach_key` | string | |
| `as_of_time` | ISO8601 | Feature cutoff. |
| `p_exceedance` | float [0,1] | |
| `binary_alert` | int {0,1} | After tuned threshold. |
| `model_tier` | string (optional) | e.g. `statewide` vs `south_bay_enhanced` |

---

## Leakage rules (mandatory)

1. **No future Tier 1 labs** in features: any `fib_*` input must be from samples **strictly before** the target window.
2. **No future environmental data** after `as_of_time`.
3. **Time-based CV** only (blocks by season/year).

---

## External I/O (fetch layer)

| System | Pattern |
|--------|---------|
| **lab.data.ca.gov (Tier 1)** | CKAN/DataStore or file exports per dataset page—**implement per current portal docs**. |
| **CDIP** | THREDDS / NetCDF per [cdip.ucsd.edu](https://cdip.ucsd.edu/). |
| **NOAA Tides** | REST `api.tidesandcurrents.noaa.gov` ([prod API](https://api.tidesandcurrents.noaa.gov/api/prod/)). |
| **NOAA precip** | NCEI hourly / LCD pathways; see catalog links in `DATASETS.md`. |
| **SCCOOS** | ERDDAP tabledap/griddap; THREDDS **[VERIFY]**. |
| **BWTF** | [bwtf.surfrider.org](https://bwtf.surfrider.org/) public portal/API as documented there. |
| **IBWC** | **[VERIFY]** official gauge access before production. |

Legacy examples also appear in `resources/deep-research-report.md`; **tier list in `DATASETS.md` wins** for implementation priority.
