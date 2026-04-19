# Modeling process, expected outputs, and demo data shapes

**Audience:** Designers, mockup tools (e.g. Gemini), and frontend work — so generated **fake data** matches what the science path will actually produce.

**Related:** `plan.md` (full math + sprint hours), `DECISIONS.md`, `ASSUMPTIONS.md`, `INTERFACES.md` (contracts), `GROUND_TRUTH_SCHEMA.md` (ETL Parquet), `APP_PRODUCT.md` (UI promise).

---

## 1. What is being modeled?

| Item | Choice (repo) |
|------|----------------|
| **Primary indicator** | **Enterococcus** (marine FIB) from **Tier 1** California SWRCB / `lab.data.ca.gov` — not Surfrider BWTF as the spine. |
| **Target scale** | **Log-scale concentration** (e.g. \(\log Y\) or \(\log_{10}\) **MPN** / reported units from the state CSV). Exact convention is fixed at training time and mirrored in exports. |
| **Likelihood** | **Censored Gaussian** on that log scale — lab reports often **`<` or `>`** detection limits; raw integers are **not** treated as simple Poisson counts. |
| **Scientific grain** | **Latent daily** process \(\eta_{s,t}\) per **station** \(s\) and **calendar day** \(t\). Sampling is **irregular** (only days with lab results). |
| **What users “see”** | **Forecasts** for calendar days (e.g. **7-day strip**): posterior **median** + **interval** + **probability water exceeds the regulatory threshold** — see §5. |

**Non-goals for v1:** Official county advisory replacement; joint modeling of all four bacteria before Enterococcus coverage is understood (`DECISIONS.md`).

---

## 2. End-to-end pipeline (where data comes from)

```
Tier 1 bacteria CSV + stations + beach detail
        ↓
  fetch + ETL (`build_ground_truth_dataset.py`)
        ↓
  `data/processed/datatide_ground_truth.parquet`  ← wide table: lab rows + env joins
        ↓
  Feature engineering (lags, as-of cutoff, Enterococcus filter, censoring flags)
        ↓
  Training table: mainly (station_id, as_of_date) or (station_id, target_date) daily rows
        ↓
  Baselines → then NumPyro hierarchical model (HSGP + SVI; NUTS on subset for checks)
        ↓
  Posterior summaries exported for app / BI (see §4–§6)
```

**Important:** The **ground-truth Parquet** is **not** the final model output — it is the **input layer** after environmental joins. Model outputs add **`pred_*`**, **`p_exceedance`**, **`alert_level`**, etc.

---

## 3. Inputs the model uses (conceptual)

From `INTERFACES.md` / `DECISIONS.md`, features are **tabular** with **leakage-safe** lags:

- **Identity / space:** `station_id`, `county`, lat/lon, optional `beach_key`.
- **Time:** `day_of_year`, calendar features; **no future labs** in features.
- **Tier 1 history:** prior official **FIB** reading(s), e.g. lag-1d (not BWTF by default).
- **Rain:** sums over **24h / 48h / 72h / 7d** (often log-transformed in model), `dry_days_since_rain`.
- **Tide:** **daily** range from **high/low** predictions at the mapped gauge (not full hourly series in the matrix).
- **Waves:** CDIP **Hs, Tp** (nearest buoy); often **missing** on old dates unless archive NetCDF is wired.
- **SCCOOS / HF / CCE / IBWC:** optional; many **NaNs** outside regions or time coverage (`GROUND_TRUTH_SCHEMA.md` § geographic coverage).

The **app** does not need to show all raw columns — only what you need for **map color**, **7-day chart**, and **“why”** copy (top drivers from `fact_features` / SHAP-like summaries).

---

## 4. Model outputs the app should consume

Canonical field names from `INTERFACES.md` (and `plan.md` Power BI sketch). Use these names in **mock JSON** so backend can swap real inference later.

### 4.1 Per prediction (one row per `station_id` × `prediction_date` × `parameter`)

| Field | Type | Meaning |
|-------|------|---------|
| `station_id` | string | Tier 1 monitoring station id (join to `dim_station`). |
| `prediction_date` | date (ISO `YYYY-MM-DD`) | Day the forecast is **for** (not necessarily a sample day). |
| `parameter` | string | e.g. `"Enterococcus"`. |
| `as_of_time` | ISO8601 (optional) | Instant features were frozen (for “last updated”). |
| `pred_p05` … `pred_p95` | float | Posterior quantiles of the **modeled quantity** (same scale as training — usually **log or log10 MPN**). |
| `pred_mean`, `pred_sd` | float | Optional posterior mean / SD. |
| `p_exceedance` | float in **[0, 1]** | \(\Pr(Y > \text{regulatory threshold})\) from the posterior — **primary driver for traffic-light UI**. |
| `binary_alert` | 0/1 | Optional; thresholded action. |
| `alert_level` | string | e.g. `"green"` / `"yellow"` / `"red"` / `"unknown"` — **map dot color**. |
| `confidence_tier` | string (optional) | e.g. `"high"` / `"low"` if last sample stale or env features missing (product choice). |
| `run_id` | string | Which model run produced this row. |

**Calibration note:** If **SVI** understates variance, **`p_exceedance`** can be **miscalibrated** (`ASSUMPTIONS.md`); disclaimers in UI are required (`APP_PRODUCT.md`).

### 4.2 Dimensions (for map + detail page)

**`dim_station` (one row per station)**

| Field | Example |
|-------|---------|
| `station_id` | `"12345"` (format from state data) |
| `station_name`, `beach_name` | `"Santa Monica Pier"` |
| `county` | `"Los Angeles"` |
| `latitude`, `longitude` | `34.008`, `-118.496` |
| `beach_type`, `agency_name` | strings from Tier 1 |
| `hero_image_url` | optional — product, not from model |

**`dim_date`** — standard calendar dimension for charts.

**`dim_run`** — `run_id`, `model_version`, `inference_method` (`SVI` / `NUTS`), `train_start`/`train_end`, metrics (CRPS, AUC), `git_sha`.

### 4.3 “Why?” / explainability (optional for demo)

| Table | Role |
|-------|------|
| `fact_features` | For a given `(prediction_date, station_id)`: rain lags, `tide_range_m`, `wave_hs_m`, etc. — **plain-language panel** can map top z-scores or SHAP to bullets. |
| `fact_feature_importance` | Partial dependence grids: `feature_name`, `x_value`, `effect_p50`, … |
| `fact_calibration` | For stats mode: PIT, coverage — not needed for minimal fake demo. |

---

## 5. What the UI timeline looks like (7-day forecast)

For **each beach / station**, the detail page needs **7 points** (or 7 bars):

- **`prediction_date`** = D, D+1, …, D+6 (from **today’s** forecast run, or from a chosen `as_of_time`).
- For each day: **`pred_p50`** (line), **`pred_p05`–`pred_p95`** (band), **`p_exceedance`** (secondary axis or color), **`alert_level`**.

**Display scale:** The model may live on **log10 MPN**; the **app** should either:

- show **MPN** on a log axis, or  
- show **probability / green–yellow–red** as the hero (recommended for consumers), with raw numbers in **stats mode**.

Do **not** mix **MPN** and **log10 MPN** on the same numeric axis without labeling.

---

## 6. Plausible fake numbers (for Gemini / Figma)

Use **round** numbers; keep **`p_exceedance` ∈ [0,1]** and **`alert_level`** consistent.

**Single station, one forecast run (`prediction_date` = tomorrow):**

```json
{
  "station_id": "CA-SM-EXAMPLE",
  "beach_name": "Santa Monica Pier at the Pier",
  "county": "Los Angeles",
  "latitude": 34.008,
  "longitude": -118.496,
  "parameter": "Enterococcus",
  "as_of_time": "2026-04-19T06:00:00Z",
  "prediction_date": "2026-04-20",
  "pred_p05": 1.85,
  "pred_p25": 2.10,
  "pred_p50": 2.35,
  "pred_p75": 2.62,
  "pred_p95": 3.05,
  "pred_mean": 2.38,
  "pred_sd": 0.35,
  "p_exceedance": 0.18,
  "alert_level": "green",
  "run_id": "run_2026-04-19_svi_v2"
}
```

Interpret **`pred_*`** here as **log10(MPN)** **if** you want realistic spread; then **MPN ≈ 10^2.35 ≈ 224** at the median (illustrative only — thresholds are jurisdiction-specific).

**Seven-day strip (array for chart):**

```json
{
  "station_id": "CA-SM-EXAMPLE",
  "forecast_days": [
    {"date": "2026-04-20", "pred_p50": 2.35, "pred_p05": 1.9, "pred_p95": 3.0, "p_exceedance": 0.18, "alert_level": "green"},
    {"date": "2026-04-21", "pred_p50": 2.42, "pred_p05": 2.0, "pred_p95": 3.1, "p_exceedance": 0.22, "alert_level": "green"},
    {"date": "2026-04-22", "pred_p50": 2.55, "pred_p05": 2.1, "pred_p95": 3.2, "p_exceedance": 0.35, "alert_level": "yellow"},
    {"date": "2026-04-23", "pred_p50": 2.48, "pred_p05": 2.0, "pred_p95": 3.15, "p_exceedance": 0.28, "alert_level": "yellow"},
    {"date": "2026-04-24", "pred_p50": 2.30, "pred_p05": 1.85, "pred_p95": 2.95, "p_exceedance": 0.15, "alert_level": "green"},
    {"date": "2026-04-25", "pred_p50": 2.20, "pred_p05": 1.75, "pred_p95": 2.85, "p_exceedance": 0.12, "alert_level": "green"},
    {"date": "2026-04-26", "pred_p50": 2.15, "pred_p05": 1.70, "pred_p95": 2.80, "p_exceedance": 0.10, "alert_level": "green"}
  ]
}
```

**Map list row (sortable):**

```json
{
  "station_id": "CA-SM-EXAMPLE",
  "beach_name": "Santa Monica Pier",
  "distance_km": 4.2,
  "alert_level": "green",
  "p_exceedance": 0.18,
  "one_liner": "Likely OK today; rain earlier in the week is fading."
}
```

---

## 7. What exists today vs what is still future

| Artifact | Status | Notes |
|----------|--------|--------|
| `datatide_ground_truth.parquet` | **Exists** after fetch + ETL | **Observations** + env columns — **no** `pred_*` / `p_exceedance` yet. |
| Trained NumPyro model + `InferenceData` | **Planned** (`STATUS.md` sprint) | Produces posteriors → summarized into `fact_predictions`. |
| Static JSON API for app | **Future** | Likely built from Parquet export + `dim_*` per deploy. |

**For a fake demo:** It is correct to **simulate** `pred_*`, `p_exceedance`, and `alert_level` as above; label the demo **“illustrative — not from production run”** if no `run_id` is real.

---

## 8. Copy and legal alignment

- **`p_exceedance` high** → UI says **“elevated risk”** / **“likely unsafe”**, not “you will get sick.”
- Always pair **`alert_level`** with **uncertainty** (stale sample → “confidence low”).
- Footer: **not an official advisory**; link county authority (`APP_PRODUCT.md`).

---

## 9. Quick glossary for mockup tools

| Term | Short definition |
|------|------------------|
| **MPN** | Most probable number — common FIB count scale in monitoring. |
| **Exceedance** | True concentration above the **regulatory single-sample limit** for that beach/program. |
| **`p_exceedance`** | Model’s probability of exceedance — **not** the same as observed exceedance on a past sample day. |
| **Censoring** | Lab says `<10` or `>2419` instead of an exact number — model uses it; naive regression often doesn’t. |
| **SVI / NUTS** | Inference: variational vs MCMC — affects how much you trust tail probabilities (`RISKS.md`). |

---

*Keep this file updated when `INTERFACES.md` or export schema changes.*
