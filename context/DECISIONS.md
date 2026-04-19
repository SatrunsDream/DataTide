# DECISIONS.md

Record meaningful choices: decision, alternatives, rationale, consequences, date, optional `run_id`.

---

## Template

### YYYY-MM-DD — Short title

- **Decision:** …
- **Alternatives considered:** …
- **Rationale:** …
- **Consequences / follow-ups:** …
- **Run ID (if any):** …

---

### 2026-04-18 — Primary data spine and target organism

- **Decision:** Use **Tier 1** California SWRCB / `lab.data.ca.gov` **bacteria monitoring results** as the **only** regulatory statewide spine for labels and station keys. **Surfrider BWTF** is **auxiliary** (optional offset or validation), not the source for “yesterday’s count” or primary beach ID in a statewide model.
- **Alternatives considered:** BWTF-as-spine (community sites, mixed creek/beach, non-regulatory).
- **Rationale:** Aligns with `PROJECT_BRIEF.md` and `DATASETS.md`; avoids conflating programs with different sampling design and thresholds.
- **Consequences / follow-ups:** Feature `fib_lag*` must come from **official** Tier 1 history per station; BWTF joins require explicit scope in `DECISIONS.md` if used.

### 2026-04-18 — Primary modeled indicator

- **Decision:** **Enterococcus** is the **first** modeling target (marine beaches, regulatory relevance). **E. coli** / other parameters are **secondary** or stretch; a full multivariate FIB model waits on coverage analysis (many sites measure only one or two parameters).
- **Alternatives considered:** Joint model for four bacteria with shared latent state; single pooled “any FIB” target.
- **Rationale:** Literature and California practice emphasize Enterococcus at marine beaches; joint models are rank-deficient where parameters are missing and thresholds differ by indicator.
- **Consequences / follow-ups:** Filter `Parameter == "Enterococcus"` for v1; document county/jurisdiction coverage in EDA.

### 2026-04-18 — Observation cadence vs prediction grain

- **Decision:** Treat monitoring as **irregular (station, sample_date) events** with **variable cadence** (not uniformly weekly). The **scientific** framing is a **latent daily** process (e.g. \(\eta_{s,t}\) log-concentration) with an **observation model only on days with samples**; **predict** a full daily series for evaluation and dashboards.
- **Alternatives considered:** Force one row per week; assume weekly run-rate for all beaches.
- **Rationale:** Tier 1 schema allows multiple rows per station per day; AB 411 summer cadence differs from off-season; weekly aggregation throws away information and mis-states uncertainty.
- **Consequences / follow-ups:** ETL may aggregate lab **replicates** to one observation per (station, date) before likelihood (e.g. geometric mean) to avoid double-counting—**verify** in EDA.

### 2026-04-18 — Likelihood for concentration data

- **Decision:** Primary likelihood: **censored Gaussian on \(\log Y\)** (or \(\log_{10} Y\)) with left/right censoring from **detection limits** (parse `<`, `>` from `Qualifier` / `Result` where present). **Poisson** / **negative binomial** on raw integer counts are **secondary** comparisons only (and only on uncensored subsets), because FIB reports are **not** pure Poisson counts—MPN/CFU with limits dominates.
- **Alternatives considered:** NB2 as default; ordinary least squares on \(\log Y\) ignoring censoring.
- **Rationale:** Matches operational and methodological literature on FIB (detection limits, log-scale normality in the body of the distribution).
- **Consequences / follow-ups:** Hour 0 audit of censoring fields; NumPyro `factor` or built-in censored normal for limits.

### 2026-04-18 — Nonlinear covariates and scale

- **Decision:** Encode smooth nonlinear effects with **HSGP** (Hilbert–GP) **Matérn-3/2** approximations: default **\(M \approx 30\)** basis functions, boundary factor **\(c = 1.5\)**, **log-normal** length-scale priors scaled to feature range. Add **hierarchical** station and county random effects (**non-centered** parameterization). Optional **AR(1)** residual on weekly scale for slow decay of FIB persistence.
- **Alternatives considered:** Full GP (too costly at \(\sim 10^6\) rows); pure linear GLM; spline-only GAM outside NumPyro.
- **Rationale:** HSGP is \(O(NM)\) vs \(O(N^3)\) for full GP; matches “GAM within hierarchical Bayes” goal without killing the 10-hour budget on CPU.
- **Consequences / follow-ups:** If SVI fails to converge by gate time, drop smooths on lowest-SHAP features first (keep rain + salinity per plan).

### 2026-04-18 — Feature construction (lags vs sequences)

- **Decision:** Prefer **scalar lag summaries** for rain, waves, tides (e.g. **24h / 48h / 72h / 7d** rain sums or logs; max \(H_s\) over 72h; daily tidal range) rather than a raw **7-dimensional** daily sequence per week as the default feature block—unless a sequence model is explicitly chosen later.
- **Alternatives considered:** 7-day vectors per feature for multimodal embedding; hourly stacks in the training table.
- **Rationale:** Summaries match literature at moderate \(N\), reduce collinearity, and fit the tabular + HSGP path within time budget.
- **Consequences / follow-ups:** `INTERFACES.md` / `GROUND_TRUTH_SCHEMA.md` columns must be extended with engineered lags from daily env series.

### 2026-04-18 — Inference stack (hardware and budget)

- **Decision:** On **Apple Silicon (M4 Pro)**, use **`jax` CPU** build for **NumPyro** (not **jax-metal** for production NUTS/SVI). **SVI** with **AutoLowRankMultivariateNormal** (e.g. rank 20) on **full** (or minibatched) data first; **Pareto-\(\hat k\)** diagnostic on importance weights; **NUTS** on a **stratified subset** (e.g. one county or one year) for calibration checks. **4 parallel chains** on P-cores where stable.
- **Alternatives considered:** Full-data NUTS; Metal JAX; mean-field `AutoNormal` only.
- **Rationale:** Full NUTS at \(\sim 10^6\) rows exceeds a **10-hour** window on CPU; Metal is unstable with NumPyro; low-rank MVN guide partially recovers correlations vs mean-field; Pareto-\(\hat k\) flags when SVI tails are untrustworthy for **exceedance** probabilities.
- **Consequences / follow-ups:** Document in `ASSUMPTIONS.md` and `RISKS.md`; if \(\hat k \ge 0.7\), ship NUTS-subset as primary or NeuTra / subsampled NUTS per plan notes.

### 2026-04-18 — Baselines and evaluation

- **Decision:** Require **non-Bayesian baselines** before hierarchical model: **persistence** (last observation), **seasonal climatology** (median \(\pm\)14 DoY), **logistic regression** on exceedance, **LightGBM** on \(\log\) result and on exceedance. **Time-blocked** splits (e.g. train \(\le 2022\), val 2023, test 2024–2025); **no lags crossing** split boundaries. Metrics: **CRPS**, **MAE** on \(\log_{10}\) MPN, **exceedance AUC**, **recall at fixed specificity** (operational constraint), **interval coverage**, **PSIS-LOO** where applicable.
- **Alternatives considered:** Bayesian-only deliverable; random station split only.
- **Rationale:** Time crunch + need to prove value over simple operational predictors; Searcy & Boehm–style comparators.
- **Consequences / follow-ups:** Store `artifacts/tables/baseline_metrics.csv` and model comparison tables; Bayesian win should emphasize **calibration** if point metrics tie.

### 2026-04-18 — Posterior storage and dashboards

- **Decision:** Serialize runs as **ArviZ `InferenceData`** (posterior, posterior_predictive, prior, observed_data, `log_likelihood`, sample stats) to **NetCDF** under `artifacts/models/` per run-bundle rules; use for **PPC**, **LOO-PIT**, traces (subset). **Power BI** consumes a **star schema** of Parquet facts/dims (`fact_predictions`, `dim_station`, `dim_date`, `dim_run`, `fact_features`, `fact_calibration`)—see `INTERFACES.md`.
- **Alternatives considered:** CSV-only exports; dashboard without uncertainty bands.
- **Rationale:** Reproducible calibration review; Power BI joins expect long/wide fact + dimension tables.

---

## Pending (not yet decided in repo)

- Exact **regulatory threshold** table per county/program for `target_exceedance` (must be codified from Tier 1 + official postings).
- Whether **replicate samples** are collapsed with geometric mean vs kept in hierarchical observation model.
- **IBWC / USGS** discharge feature inclusion for South Bay v1 vs phase 2.
