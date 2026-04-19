# INTERFACES.md

Contracts for **tabular modeling**: column semantics, keys, time cutoff, and **train/serve** parity. Enables another LLM or engineer to implement ETL and the **`site/`** web UI without guessing.

**Primary keys:** Derive from **Tier 1** California beach water quality data: **monitoring results** joined to **monitoring stations** and **beach detail** (`context/DATASETS.md`). Do not treat Surfrider BWTF or CCE as the primary key spine unless explicitly scoping a side experiment.

**Demo / UI mock data:** For a plain-language pipeline summary, example JSON, and glossary for tools like Gemini, see **`MODEL_OUTPUTS_FOR_DEMO.md`**.

---

## Prediction unit (grain)

### Tabular / ML baseline (site–day)

- **Default feature table grain:** **daily** rows: `(station_id, date)` with environmental lags aligned to an **`as_of_time`** (features through end of *D−1* or start of *D* per `ASSUMPTIONS.md`).
- **Operational cadence:** **week-by-week** retrain or score is a **schedule**, not a requirement that every beach samples weekly—cadence varies by program (`DECISIONS.md`).

### Hierarchical Bayesian model (planned)

- **Latent process:** continuous **daily** \(\eta_{s,t}\) (e.g. log or \(\log_{10}\) expected concentration) per station \(s\).
- **Observations:** **irregular**—only on days with Tier 1 lab results; likelihood with **censoring** at detection limits (`DECISIONS.md`).
- **Do not assume** uniform weekly sampling; aggregating to forced weekly bins before modeling discards information and misstates uncertainty.
- **Public / dashboard display:** optional **7-day rolling** summary of posterior **median** for communication only—distinct from the latent daily process (`context/plan.md`).

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
| `fib_lag1d` | float | **Tier 1** (primary) | Prior **official** FIB reading (not BWTF as default spine); optional **Tier 10** only as separate columns if merged carefully. |
| `rain_sum_24h` … `rain_sum_7d` (or `log_rain_*`) | float | **Tier 2** NOAA precip | Windows end at feature cutoff; **HSGP**-friendly nonlinear transforms per `DECISIONS.md`. |
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

## Model outputs (for `site/` or batch)

| Field | Type | Meaning |
|-------|------|---------|
| `station_id` / `beach_key` | string | |
| `as_of_time` or `prediction_date` | ISO8601 / date | Feature cutoff / forecast day. |
| `pred_p05` … `pred_p95` | float | Posterior quantiles of concentration or \(\log_{10}\) MPN (see training target convention). |
| `pred_mean`, `pred_sd` | float | Posterior mean / SD (optional). |
| `p_exceedance` | float [0,1] | \(\Pr(Y > \text{threshold})\) from posterior; **calibration-sensitive** if SVI understates variance (`RISKS.md`). |
| `binary_alert` | int {0,1} | After tuned threshold on `p_exceedance` or concentration. |
| `alert_level` | string (optional) | e.g. green / yellow / red for dashboards. |
| `model_tier` | string (optional) | e.g. `statewide` vs `south_bay_enhanced` |
| `run_id` | string | FK to `dim_run` for Power BI and run bundles. |

### ArviZ / NetCDF

- Persist **`arviz.InferenceData`** per run: `posterior`, `posterior_predictive`, `prior`, `prior_predictive`, `observed_data`, `log_likelihood`, `sample_stats`.
- **Path:** `artifacts/models/<model_name>_<YYYY-MM-DD>.nc` (or `.nc` group per `rules_templet.md` run bundle).
- Use for **PPC** (`az.plot_ppc`), **LOO-PIT** (`az.plot_loo_pit`), **forest** plots for REs, **trace** for NUTS subset only.

### Power BI star schema (Parquet under `artifacts/data/powerbi/`)

Recommended tables for import via Parquet or DuckDB:

| Table | Role | Key columns (summary) |
|-------|------|-------------------------|
| `fact_predictions` | One row per (`station_id`, `prediction_date`, optional `parameter`) with uncertainty | `pred_p05`…`pred_p95`, `pred_mean`, `p_exceedance`, `alert_level`, `observed` (nullable), `is_censored_*`, `run_id` |
| `dim_station` | Station / beach attributes | `station_id`, names, `county`, lat/lon, `beach_type`, nearest buoy / tide / precip ids |
| `dim_date` | Calendar | `date`, year, month, week, DoY, season flags |
| `dim_run` | Model run metadata | `run_id`, `model_name`, `model_version`, `inference_method` (SVI/NUTS), train window, `git_sha`, LOO / CRPS / AUC |
| `fact_features` | Explainability drill-down | (`prediction_date`, `station_id`): rain lags, `tide_range_m`, `wave_hs_m`, SST, salinity, discharge (nullable) |
| `fact_calibration` | Calibration tab | observed vs interval membership, PIT, CRPS per row |
| `fact_feature_importance` | Partial dependence | (`feature_name`, `x_grid`): effect `p05`/`p50`/`p95` |

**Dashboard tabs (suggested):** map of latest alerts; station time series with predictive band + observations; diagnostics (PIT, coverage by county); run leaderboard vs baselines.

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
