# Data Processing

End-to-end description of how raw source files become the JAX/NumPyro-ready arrays consumed by the hierarchical Bayesian model. Covers every step from network fetch through the artifacts emitted by `notebooks/modeling/eda.ipynb`.

Three pipeline stages:

1. **Fetch** — raw files pulled from each data provider (`scripts/fetch/`) into `data/raw/<layer>/`.
2. **Ground-truth build** — one Tier-1 bacteria row per lab result, with every environmental layer joined in (`scripts/process/build_ground_truth_dataset.py`), written to `data/processed/datatide_ground_truth.parquet`.
3. **EDA + panel build** — `notebooks/modeling/eda.ipynb` audits the ground truth, aggregates replicates, expands to a daily panel, engineers lag features, standardizes, builds temporal CV folds, and emits the arrays under `artifacts/data/panel/`.

Configuration is centralised in `configs/fetch.yaml` (what to pull) and `configs/process.yaml` (what to join, bucket, and output).

---

## 1. Fetch — raw source layers

Every script in `scripts/fetch/` writes idempotently into a layer-specific folder under `data/raw/`. `scripts/fetch/run_all.py` orchestrates them.

| Script | Layer | Output dir | Role |
|---|---|---|---|
| `fetch_ca_swrcb.py` | CA SWRCB bacteria + station supplement + beach detail | `data/raw/ca_swrcb/` | Tier-1 spine. Statewide marine Enterococcus lab results with `Station_id`, `BeachName_id`, `CountyName`, `Station_UpperLat/Lon`, `Qualifier`, `Result`. |
| `fetch_sd_county.py` | San Diego county coastal monthly | `data/raw/sd_county_beach/` | Tier-5 regional context (ancillary). |
| `fetch_surfrider_bwtf.py` | Surfrider BWTF citizen lab | `data/raw/surfrider_bwtf/` | Tier-4 statewide daily summary (optional). |
| `fetch_noaa_precip.py` | NOAA GHCN precipitation | `data/raw/noaa_precip/` | Tier-2 daily rain by precip bucket → county. |
| `fetch_noaa_tides.py` | NOAA CO-OPS tides (hi/lo) | `data/raw/noaa_tides/` | Tier-2 daily tide range by station id → county. |
| `fetch_cdip.py` | CDIP wave buoys (Hs, Tp) | `data/raw/cdip/` | Tier-2 daily waves, joined by nearest-buoy to station lat/lon. |
| `fetch_sccoos_erddap.py` | SCCOOS Del Mar pier ERDDAP | `data/raw/sccoos_erddap/` | Tier-3 daily SST/salinity, county-masked to SoCal only. |
| `fetch_cce_moorings.py` | CCE1/CCE2 mooring CTD/ADCP | `data/raw/cce_moorings/` | Tier-4 offshore T/S/U/V at shallowest depth, nearest-mooring join. |
| `fetch_hf_radar.py` | HF-radar surface currents | `data/raw/hf_radar/` | Tier-3 daily u/v within a fetch bbox (nearest grid cell). |
| `fetch_ibwc.py` | IBWC Tijuana River discharge | `data/raw/ibwc/` | Tier-5 daily stage + discharge (IB/SD only). |

Each fetcher is responsible for its own rate limiting, retry, and cache-on-disk; re-running a fetcher only downloads missing date ranges.

**In the current repo state,** the raw tree has been pruned — `data/` is gitignored and the pushed artifact is `data/processed/datatide_ground_truth.parquet`. The EDA notebook runs in *reload mode* (loads `artifacts/data/panel/*` directly) when the ground truth is absent, so fetch-stage reproducibility is gated on rerunning the fetchers above before invoking the ground-truth build.

---

## 2. Ground-truth build — `data/processed/datatide_ground_truth.parquet`

`scripts/process/build_ground_truth_dataset.py` is a pure join/aggregate step — no modelling. It produces **one row per bacteria lab result** with every environmental layer broadcast-joined on the appropriate key.

**Spine keys:** `Station_id`, `sample_date`, `CountyName`, `Station_UpperLat/Lon`, `Qualifier`, `Result`, plus supplements from `stn_sup` (station metadata) and `beach_sup` (beach detail).

**Join keys per layer (from `src/preprocessing/modeling_joins.py`):**

| Layer | Join key | Notes |
|---|---|---|
| GHCN precip | `(sample_date, precip_bucket)` | `precip_bucket = precip_bucket_by_county[CountyName]` |
| CO-OPS tides | `(sample_date, tide_station_id)` | `tide_station_id = tide_id_by_county[CountyName]` |
| SCCOOS Del Mar | `sample_date` | masked to `sccoos_counties` (SoCal); NaN elsewhere |
| CDIP waves | `(sample_date, cdip_bundle)` | `cdip_bundle` = nearest buoy by great-circle distance to `(Station_UpperLat, Station_UpperLon)` |
| SD monthly | `(CountyName, calendar_month)` | `calendar_month = sample_date.to_period("M")` |
| HF radar | `sample_date` | nearest grid cell within `hf_bbox`; outside bbox → NaN |
| IBWC Tijuana | `sample_date` | valid only when `data/raw/ibwc/` exists |
| Surfrider BWTF | `sample_date` | statewide daily summary; optional |
| CCE mooring | `(sample_date, cce_mooring_id)` | `cce_mooring_id` = nearest mooring (CCE1 vs CCE2) |
| Station + beach supplements | `Station_id`, `BeachName_id` | simple dim-table merges |

**Streaming pattern:** bacteria CSV is read in 200k-row chunks (`bacteria_chunk_rows` in `configs/process.yaml`) and written chunk-by-chunk to Parquet via `pyarrow.parquet.ParquetWriter`. This keeps memory bounded over the full state-level history.

**Coverage caveat:** if a fetcher for a given layer wasn't run (or its raw folder is empty), the join function returns an empty DataFrame and `build_ground_truth_dataset.py` fills the corresponding columns with NaN explicitly (see the HF-radar and IBWC branches around `build_ground_truth_dataset.py:215`). The EDA prune step later drops any feature whose standardized column has zero variance, so dead layers don't poison the model.

**Emitted sidecar:** `data/processed/datatide_ground_truth_meta.json` records the build timestamp, the layer inventory, and per-layer row counts.

---

## 3. EDA + panel build — `notebooks/modeling/eda.ipynb`

This is where the ground truth becomes modelling-ready arrays. The notebook operates in two modes, controlled by whether the raw Parquet exists.

### 3.1 Dual-mode entry point (cells 1 – 3)

- `STUDY_START = 2010-01-01`, `STUDY_END = 2025-12-31`, `TRAIN_END = 2022-12-31`, `VAL_END = 2023-12-31` are defined in cell 1.
- Cell 3 sets `REBUILD = GT_PATH.exists()`.
  - **Rebuild mode** (`data/processed/datatide_ground_truth.parquet` exists): loads the raw Parquet via a `read_parquet_safe` helper that reads column-by-column into NumPy arrays, sidestepping a pyarrow/pandas `Int64` round-trip bug on this env. Drops rows missing `sample_date`, `Station_id`, or `Result`. Casts `Station_id` to `int64` and `BeachName_id` to `Int64`. Filters to `[STUDY_START, STUDY_END]`.
  - **Reload mode** (Parquet missing): loads `artifacts/data/panel/enterococcus_panel.pkl` (pickled DataFrame) + `enterococcus_panel_meta.json`. Re-casts the censoring flags to bool and `sample_date` to datetime. Reconstructs a per-observation `raw` DataFrame (observed rows only) so downstream EDA cells behave identically. Reads `SMOOTH_FEATURES`, `LINEAR_FEATURES`, `FEATURE_COLS`, and `scalers` from the meta file.
- **Station-name enrichment** (end of cell 3): loads `artifacts/data/panel/station_labels.json` (produced by `scripts/process/reverse_geocode_stations.py`, see §3.9) and attaches a `Station_Name` column to both `raw` and, when already loaded, `panel`. Existing non-null names are preserved; NaN/`""` entries are filled from the cache.

### 3.2 Parameter spine audit (cells 4 – 5)

Confirms Enterococcus is the only statewide-reliable marine parameter. Prints a parameter-mix table (rows, stations, counties, date span) sorted by row count. Filters the working frame `entero = raw[raw["Parameter"] == "Enterococcus"]`.

### 3.3 Censoring audit (cells 6 – 7)

Parses MPN/CFU censoring via `parse_censoring(df)`:

- `left` — `Qualifier` starts with `<` **or** `Result ≤ LEFT_LIMITS_MPN`
- `right` — `Qualifier` starts with `>` **or** `Result ≥ RIGHT_LIMITS_MPN`
- `observed` — everything else (an uncensored interior measurement)

Limits: `LEFT_LIMITS_MPN = (1.0, 2.0, 10.0)`; `RIGHT_LIMITS_MPN = (2419.2, 2419.6, 24192.0, 24196.0)` (IDEXX Quanti-Tray caps). Left-censored results are replaced by the detection limit for plotting; the modelling layer uses the `det_low_log` / `det_high_log` arrays downstream. Outputs `is_left_cens`, `is_right_cens`, `is_observed`, `det_low`, `det_high`, `log10_result`.

### 3.4 Target distribution and log-transform justification (cells 8 – 9)

Six-panel diagnostic on observed Enterococcus:

1. Raw count histogram, linear x — shows four-plus orders of magnitude crushed at the low end.
2. Raw count histogram, log x — already approximately normal.
3. `log10(MPN)` histogram with the `log10(104)` threshold line — the modelling space.
4. Q-Q plot against Normal (manual `_norm_ppf`, no scipy) — close to the 45° line.
5. Boxplot: stored `log10_result` vs recomputed `log10(raw)` — consistency check.
6. Monthly exceedance rate time series.

Skewness and excess-kurtosis are printed for raw and log10 separately (typically raw skew > 20, log10 skew ≈ 1.5 — the "log is the right space" argument, made quantitatively rather than by test). The unconditional exceedance rate at `P(MPN > 104)` is printed as the base rate any classifier must beat.

### 3.5 Seasonal structure — justifying the seasonal naive baseline (cells 10 – 11)

Motivates the month-conditional naive baseline used in the modeling plan. A three-panel diagnostic on observed Enterococcus:

1. Monthly violin plot of `log10(MPN)` across all twelve calendar months, colored wet-season (blue, Nov–Apr) vs dry-season (orange, May–Oct), with the `log10(104)` advisory line overlaid. Shows the annual cycle as a distributional shift, not just a mean shift.
2. December-vs-June count histograms on a log x-axis — the headline "inverse profile" chart showing the wet-vs-dry pole months overlap very little on the count scale.
3. Monthly advisory-exceedance rate bar chart (`P(y > 104)` by calendar month) against the pooled rate — the decision-relevant summary for a beach manager.

Printed statistics quantify the argument: peak-vs-trough monthly mean in `log10(MPN)`, implied fold change in median counts, December vs June median counts and exceedance rates with their ratio, and the variance of `log10_result` explained by a month-only predictor (typically 10–15%). The last number is the key result — conditioning on month alone shrinks residual SD by a meaningful fraction at zero modeling cost, which is why the Phase B naive baseline in `MODELING_PLAN.md §6.1` uses `mean(counts | month)` and not the grand mean.

### 3.6 Station and county coverage (cells 12 – 13)

Per-station aggregation: sample count, date range, span (days), empirical exceedance rate. Grouping keys auto-filter columns that are entirely null (handles the reload-mode edge case where `Station_Name` was historically absent). Prints:

- quantiles of samples-per-station
- top counties by row volume
- top-12 most-sampled beaches with names (post-geocoding)
- count of stations with ≤5 observations — a flag for how heavily hierarchical pooling will be asked to carry the weight.

### 3.7 Sample cadence (cells 14 – 15)

Between-sample gap in days, grouped by station. Shows the statewide cadence is not "weekly" but a bimodal mixture of daily-in-summer and monthly-off-season — which motivates treating sampling as irregular observation of a latent daily process (i.e. a state-space model), not a fixed-cadence regression.

### 3.8 Environmental feature coverage (cells 16 – 17)

Tabulates per-row `non_null_share` and `n_non_null` for every joined environmental feature on observed Enterococcus rows. Used to decide which smooths are statewide-viable vs regional-only; also surfaces upstream layers that never populated (`hf_current_speed_mps`, `ibwc_tijuana_discharge_cms_daily_mean` in the current artifact).

### 3.9 Replicate aggregation → one row per (station, date) (cells 18 – 19)

Multiple `RESULTS id` entries per station/day are treated as replicate aliquots. Aggregation rules in `agg_day(g)`:

- `log10_result` — arithmetic mean of per-replicate `log10_result` (== log10 of the geometric mean on the raw scale).
- `is_left_cens` — True iff **all** replicates are left-censored (same for right).
- `is_mixed_cens` — True if the replicates span both an observed and a censored value (these rows keep the observed value and drop the censor flag).
- `det_low`, `det_high` — min of lower limits, max of upper limits, respectively.
- Environmental columns — mean across replicates (they're identical by construction; mean is a no-op but safe).
- Station metadata — first value.

Emitted DataFrame `obs` is the "one-row-per-(Station_id, sample_date)" observed subset.

### 3.10 Daily prediction grid (cells 20 – 21)

`expand_station(group)` generates every calendar day inside each station's active window:

- `d_min = max(first observed date, STUDY_START)`
- `d_max = min(last observed date, STUDY_END)`
- emits `pd.date_range(d_min, d_max, freq="D")`

Environmental covariates are merged on `(Station_id, sample_date)` by taking the daily mean across replicates (from `daily_feats`). `is_observed` is 1 only on sampled days. Result is the **panel DataFrame**: one row per `(Station_id, calendar_day)` inside each station's active window, with environmental context filled and bacteria values present only on `is_observed=True` rows.

### 3.11 Lag and derived features (cells 22 – 23)

`_build_lag_features(panel)` computes within-station, chronologically-sorted lag/rolling features (no cross-station leakage):

- rainfall rolling sums: `rain_24h_mm`, `rain_48h_mm`, `rain_72h_mm`, `rain_7d_mm`
- dry-days-since-rain (count of consecutive days with `rain_24h_mm == 0`)
- tide range, wave Hs/Tp, SST, salinity, surface currents, river discharge — passed through as-is but with a per-station forward/backward fill cap
- seasonality: `doy_sin`, `doy_cos` from `day_of_year`
- AR(1)-style term: `yesterday_log10_result` — previous-day's (geo-mean) `log10_result` on observed rows, NaN on unsampled days (the model sees both the value and a "was it observed?" indicator)

Returns the enriched panel plus the canonical `SMOOTH_FEATURES` list (continuous covariates getting HSGP smooths) and `LINEAR_FEATURES` list (everything that stays parametric).

### 3.12 Standardisation, CV folds, and split (cells 24 – 25)

**log1p-before-standardise for skewed positives:** `rain_*_mm` and `river_discharge_cms` are `log1p`'d first. Scalers are fit on rows with `sample_date ≤ TRAIN_END` only — training-window fit, applied to all rows. Each feature produces two panel columns: `{col}__z` (standardised value, NaN→0 fill) and `{col}__missing` (1 if the raw value was NaN, else 0). Scaler params are recorded in `scalers[col] = {"mu", "sd", "log1p"}`.

**Zero-variance prune** — `_prune_dead_features(panel, feats, scalers)` drops any feature whose standardised column has `std < 1e-8` (i.e. the upstream layer was never populated, so standardisation collapsed everything to the same imputed fill value). Drops the `__z`, `__missing`, and scaler entries together, removes the feature from `SMOOTH_FEATURES`/`LINEAR_FEATURES`, and prints `[prune] dropped zero-variance features (upstream layer never populated): - {col} ({reason})`. Runs in both modes. In the current artifact this drops `river_discharge_cms`.

**Static split** — `panel["split"] ∈ {"train", "val", "test"}` cut at `TRAIN_END` and `VAL_END`. Kept for quick interactive work; not the primary evaluation scheme.

**Temporal CV folds (the primary evaluation scheme)** — `build_temporal_folds(panel, val_years)` builds expanding-window rolling-origin folds:

- fold k: train `[STUDY_START, val_years[k] - 1d]`, validate on calendar year `val_years[k]`
- `CV_VAL_YEARS = (2020, 2021, 2022, 2023)` → 4 folds
- Test (`sample_date > 2023-12-31`) is never touched during CV

Each row receives `cv_val_year ∈ {0, 2020, 2021, 2022, 2023, -1}`:

| `cv_val_year` | Meaning |
|---|---|
| `0` | Pre-2020 rows — always train, never validation |
| `2020`–`2023` | Row belongs to the validation set of that specific fold |
| `-1` | Held-out test (2024 onward) |

Fold definitions (train/val row counts, date ranges) are also written to `enterococcus_panel_meta.json → cv_folds` for downstream notebooks.

### 3.13 Pack + save (cells 26 – 27)

`_pack_and_save_panel(panel)` emits three artifacts under `artifacts/data/panel/`:

**`enterococcus_panel.npz`** — compressed NumPy archive, the JAX-ready array pack:

| Key | Shape | Dtype | Role |
|---|---|---|---|
| `station_idx`, `county_idx`, `t_idx` | `(N,)` | `int32` | integer indices into `station_ids`/`county_names`/days-since-`date_min` |
| `X_smooth`, `X_linear` | `(N, D)` | `float32` | standardised design matrices for HSGP smooths and parametric linear terms |
| `miss_smooth`, `miss_linear` | `(N, D)` | `int8` | 1 if the raw value was NaN, else 0 |
| `y_log` | `(N,)` | `float32` | `log10(MPN)`; likelihood uses it only where `obs_mask=True` |
| `obs_mask`, `left_mask`, `right_mask` | `(N,)` | `bool` | likelihood masks |
| `det_low_log`, `det_high_log` | `(N,)` | `float64` | censoring limits on log10 scale |
| `cv_val_year` | `(N,)` | `int32` | temporal fold membership (see §3.12) |
| `station_ids`, `station_names` | `(S,)` | `int64`, `object` | station id → human-readable beach name |
| `county_names` | `(C,)` | `object` | index → county name |
| `smooth_features`, `linear_features` | `(D,)` | `object` | feature-name dictionaries |
| `date_min` | `()` | `datetime64[D]` | for reconstructing real dates from `t_idx` |

**`enterococcus_panel.pkl`** — pickled full panel DataFrame (wide view with all source columns, `__z`, `__missing`, `cv_val_year`, `Station_Name`, etc.). Falls back from Parquet write on systems where `pyarrow` 15.x is binary-incompatible with NumPy 2.x.

**`enterococcus_panel_meta.json`** — machine-readable sidecar: target, exceedance threshold, train/val cutoffs, n_rows/observed/censored, n_stations/counties, full feature lists, scaler params, date range, CV scheme (`expanding_window_rolling_origin`), `cv_val_years`, full `cv_folds` details, and a pointer to the station-labels cache.

### 3.14 Round-trip verification (cell 29)

Reloads the `.npz` and the pickled panel, prints the shape/dtype of every packed array, and prints the per-split row counts with observed/left/right breakdowns. Acts as both smoke test and reference for how the modelling notebook should load the artifacts.

---

## 4. Station-name enrichment — `scripts/process/reverse_geocode_stations.py`

Outside the notebook (but contemporaneous with the panel build) — reverse-geocodes every unique `(Station_id, Station_UpperLat, Station_UpperLon)` via Nominatim (OpenStreetMap). Respects Nominatim's 1 req/sec ToS, caches to `artifacts/data/panel/station_labels.json` so a re-run is instant. The notebook loads this cache and joins `Station_Name` into both `raw` and `panel`. All 446 stations in the current artifact have human-readable names ("Huntington Beach", "Imperial Beach", "Village of La Jolla", etc.).

Run once (idempotent; cache-hit skips):

```bash
python scripts/process/reverse_geocode_stations.py
```

---

## 5. Reproducibility checklist

To rebuild everything from scratch:

```bash
# 1. pull every raw layer
python scripts/fetch/run_all.py

# 2. join + aggregate into the Tier-1 ground truth
python scripts/process/build_ground_truth_dataset.py
#    → data/processed/datatide_ground_truth.parquet
#    → data/processed/datatide_ground_truth_meta.json

# 3. reverse-geocode station names (one-time, cached)
python scripts/process/reverse_geocode_stations.py
#    → artifacts/data/panel/station_labels.json

# 4. run EDA + panel build
jupyter nbconvert --to notebook --execute notebooks/modeling/eda.ipynb \
  --output eda.ipynb --ExecutePreprocessor.timeout=900
#    → artifacts/data/panel/enterococcus_panel.{npz,pkl}
#    → artifacts/data/panel/enterococcus_panel_meta.json
```

If `data/processed/datatide_ground_truth.parquet` is absent, step 4 still succeeds by reloading the pre-built panel under `artifacts/data/panel/` and re-running the EDA + standardisation + CV-fold + repack stages against it.

---

## 6. Known issues and mitigations

| Issue | Where | Mitigation |
|---|---|---|
| `pandas ↔ pyarrow Int64` round-trip bug crashes `to_pandas()` | Cell 3, ground-truth reload | `read_parquet_safe()` reads column-by-column into NumPy arrays |
| `pyarrow 15.x` incompatible with `numpy 2.x` Int64 column writes | Cell 25, panel save | `try/except` around `to_parquet`, falls back to `.pkl` |
| `nbformat` validation failures (missing `name` in stream outputs) | Re-executing a previously-aborted notebook | Clear cell outputs before `nbconvert --execute` |
| HF-radar and IBWC layers zero-coverage in current ground truth | Data quality | `_prune_dead_features` drops the dead standardized column and the `scalers` entry; model carries only live features |
| 77% of stations have ≤5 observations | Sample distribution | Hierarchical pooling (county random effect + optional spatial GP over `(lat, lon)`) is mandatory; iid station effects would over-fit or under-fit |
| scipy binary-incompat with NumPy 2.x on some envs | Cell 9 Q-Q plot | Manual Beasley-Springer-Moro `_norm_ppf`, no scipy dependency |
