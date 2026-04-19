# ASSUMPTIONS.md

Document what you assume, why it is reasonable, how it could fail, and how you will test sensitivity.

---

## Data

### Tier 1 spine and fields

- **Assumption:** The **bacteria results** CSV exposes enough structure to define **Enterococcus** rows, **station** join keys, and **sample dates**; **censoring** can be recovered from `Qualifier` / `Result` patterns (e.g. `<10`, `>2419.6`) for a meaningful fraction of rows.
- **Why reasonable:** State export is used operationally; literature treats FIB on log scale with detection limits.
- **Failure mode:** Missing or inconsistent censoring codes → biased likelihood if treated as exact counts.
- **Test:** Hour-0 audit: histogram of `Result` strings, cross-tab `Parameter` × `Qualifier`; compare to lab method docs.

### Cadence and replicates

- **Assumption:** Sampling **cadence varies** by beach, season, and program (not uniform weekly); multiple rows per (station, date) may exist (parameters or replicates).
- **Why reasonable:** Matches AB 411 and county practice.
- **Failure mode:** Double-counting if replicates enter likelihood as independent.
- **Test:** Count rows per (station_id, sample_date); rule for collapse documented in `DECISIONS.md` / notebook.

### Environmental joins

- **Assumption:** **Regional** GHCN precipitation and **mapped** tide gauges approximate coastal forcing for the **first** statewide model; **nearest CDIP buoy** wave stats are adequate at daily resolution when non-missing.
- **Why reasonable:** `GROUND_TRUTH_SCHEMA.md` documents join logic; literature uses similar proxies.
- **Failure mode:** Realtime CDIP NetCDF **short history** → wave features **null** for most historical dates until archive URLs are added.
- **Test:** Missingness rates by year for `cdip_wave_*`; indicator “wave observed” in models.

### BWTF and auxiliary data

- **Assumption:** **BWTF** is **not** used as the primary label or spatial key unless a separate experiment is scoped.
- **Why reasonable:** Program design and `PROJECT_BRIEF.md`.
- **Failure mode:** Accidental merge treats community and regulatory thresholds as interchangeable.
- **Test:** Column naming and `DECISIONS.md` scope checks.

---

## Modeling / evaluation

### Latent process and likelihood

- **Assumption:** A **latent daily** log-concentration (or \(\log_{10}\) MPN) per station, observed only on sample days with **censored normal** noise, is a workable v1 for **Enterococcus**.
- **Why reasonable:** Matches state-space / nowcasting framing in the literature; handles limits better than Poisson/NB on reported values.
- **Failure mode:** Heavy contamination events need fat-tailed observation noise—Gaussian may underfit extremes.
- **Test:** Posterior predictive / rootograms; tail-weighted scoring; compare to hurdle or Student-\(t\) observation (stretch goal).

### HSGP and priors

- **Assumption:** **HSGP** with **\(M \approx 30\)**, **Matérn-3/2**, and **log-normal** length scales on standardized features is flexible enough without severe overfit when combined with hierarchical pooling.
- **Why reasonable:** Standard Riutort-Mayol / NumPyro practice; \(M\) trades bias vs variance.
- **Failure mode:** Length-scale non-identifiability → slow mixing in NUTS or biased SVI marginals.
- **Test:** NUTS subset: \(\hat R\), ESS; prior predictive on \(\eta_{s,t}\) in plausible MPN range (\(10^2\)–\(10^5\)).

### SVI vs NUTS (10-hour constraint)

- **Assumption:** **SVI + low-rank multivariate normal guide** gives a **usable** statewide fit in time; **Pareto-\(\hat k\)** and **NUTS-on-subset** agreement within \(\sim\)10% on key hyperparameter90% CI widths justify reporting SVI posteriors for dashboards.
- **Why reasonable:** Full-data NUTS is infeasible on CPU in one sprint; Yao et al. diagnostics exist.
- **Failure mode:** SVI **undercovers** tails → **exceedance probabilities** (`p_exceed`) biased (typically anti-conservative).
- **Test:** Mandatory \(\hat k\); compare SVI vs NUTS for \(\tau_{\text{stn}}\), \(\sigma_k\), \(\ell_k\), and station REs on subset; report caveat in deliverable if \(\hat k \ge 0.7\).

### Hardware

- **Assumption:** **M4 Pro, 48 GB RAM**: data matrices (\(\sim 10^6 \times\) tens of features) fit in memory as **float32**; **CPU JAX** is the stable backend for NumPyro.
- **Why reasonable:** ~200 MB dense features + overhead; Metal JAX still risky for MCMC.
- **Failure mode:** OOM if storing full dense HSGP design for all smooths at once without minibatching.
- **Test:** Monitor RSS; use minibatch SVI if needed.

### Splits and leakage

- **Assumption:** **Time-blocked** CV with **no feature lag** crossing train/test cut date reflects operational information flow.
- **Why reasonable:** Standard for forecasting FIB.
- **Failure mode:** Random station split without time blocks inflates scores.
- **Test:** Assert max(feature date) \(\le\) cutoff for each `as_of_date` row in test.

---

## Operational / product

- **Assumption:** Model output is **decision support**, not a legally binding beach advisory unless adopted by a health authority (`PROJECT_BRIEF.md`).
- **Assumption:** **Power BI** consumers need **quantiles** (`p05`–`p95`) and **exceedance probability**, not only point forecasts.
