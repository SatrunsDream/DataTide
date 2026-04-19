# Modeling Plan

End-to-end plan for building, selecting, validating, and productionizing a hierarchical Bayesian forecast of marine *Enterococcus* counts at California beaches, with matched non-Bayesian baselines for comparison. This document is the contract between `notebooks/modeling/eda.ipynb` (which produced the panel + priors) and the downstream modeling work: what will be fit, how it will be compared, what metrics decide the winner, and how the final model reaches a dashboard.

Target audience is me and anyone reviewing the modeling choices. Every design decision below is justified against the data summaries produced in `context/DATA_PROCESSING.md` and the EDA artefacts in `artifacts/data/panel/`.

---

## 1. Purpose

Predict *Enterococcus* concentration (MPN per 100 mL) at a California marine beach on a given calendar day, with calibrated uncertainty. The primary operational question the model supports is: *"what is the probability that today's count exceeds the California AB411 advisory threshold of 104 MPN/100 mL?"* A secondary question is the point forecast itself, in MPN units, so that stakeholders can read it without converting from log space.

There are two deliverables:

1. A selected, validated model and a full audit trail (prior predictive, posterior, posterior predictive, convergence, cross-validation, test-set evaluation) — committed to `notebooks/modeling/model.ipynb`.
2. A productionization pipeline that applies the selected model to a historical backtest grid and to today's prediction grid, emitting dashboard-ready tables for a teammate's Power BI or internal web dashboard — committed to `notebooks/modeling/predict.ipynb`.

---

## 2. Problem framing

This is a **continuous regression problem on log10(MPN/100 mL)** with a censored-normal observation likelihood, not a count model and not a classification model. Three reasons:

- **Not a count model.** Enterococcus MPN values are not Poisson realisations; they come from a Most-Probable-Number lookup applied to a Quanti-Tray dilution series, which is itself a noisy estimator of a latent continuous concentration. The underlying bacterial concentration follows a log-normal distribution because bacterial growth/decay/dilution is multiplicative. Treating the MPN as a Poisson count mis-specifies the data-generating process twice.
- **Not a classification model.** The operational decision (post an advisory?) is a classification, but it is *derived* from a continuous prediction, not modelled directly. Fitting a continuous model on `log10(MPN)` and evaluating `P(y > log10(104))` from its predictive distribution yields a single model that serves both the continuous forecast and the classification decision, with calibrated probabilities.
- **Censored-normal likelihood.** About 48% of observed lab rows carry a `<` or `>` qualifier (detection-limit or quantitation-limit clipping). Ignoring censoring biases estimates of both the mean and the variance and destroys calibration of `P(exceed)`. The `censoring_toy.ipynb` notebook demonstrates this failure mode quantitatively on simulated data.

The generative structure is:

```
log10(y_it) ~ Normal(mu_it, sigma)            interior observations
log10(y_it) <= log10(L_it) when left-censored  (Normal CDF contribution)
log10(y_it) >  log10(U_it) when right-censored (Normal survival contribution)

mu_it = alpha_0
        + f_season(month_it, doy_it)                # seasonal scaffold (month-conditional baseline)
        + alpha_station[s(i)] + alpha_county[c(i)]  # hierarchical intercepts on top
        + beta . X_linear_it                        # linear covariate adjustments
        + f_rain(rain_lags_it) + f_time(t_it)       # non-seasonal smooths
```

The month-conditional baseline `f_season` is the scaffold: at v1 it enters as twelve hierarchical month-level intercepts (`alpha_month[m(i)]`, exactly matching the Phase B naive baseline in §6.1), and it is *replaced* at higher rungs with progressively smoother parameterisations (polynomial → spline → HSGP on day-of-year). Station/county intercepts, linear covariates, and non-seasonal smooths are all adjustments on top of whatever seasonal parameterisation is active at that rung. **There is only ever one seasonal term in the model at a time** — a month categorical or a doy smooth, never both, to avoid identifiability conflicts.

The structure on `mu_it` is what varies across the complexity ladder in Section 5.

---

## 3. Inputs and reference artefacts

This plan consumes artefacts produced upstream and does not regenerate them. See `context/DATA_PROCESSING.md` for how they are built.

| Artefact | Path | Contents |
|---|---|---|
| Panel DataFrame | `artifacts/data/panel/enterococcus_panel.pkl` | One row per `(station_id, calendar_date)` between 2010-01-01 and 2025-12-31 with observed log10 result (or NaN), censoring flags, detection limits, lag features, missingness indicators, split label, and CV fold index. |
| JAX arrays | `artifacts/data/panel/enterococcus_panel.npz` | Packed arrays ready for NumPyro: `station_idx`, `county_idx`, `t_idx`, `X_smooth`, `X_linear`, `miss_smooth`, `miss_linear`, `y_log`, `obs_mask`, `left_mask`, `right_mask`, `det_low_log`, `det_high_log`, `cv_val_year`. |
| Metadata | `artifacts/data/panel/enterococcus_panel_meta.json` | Feature lists, fitted scalers (train-window only), CV fold definitions, station-name cache path, and empirically-elicited prior scales. |
| Station names | `artifacts/data/panel/station_labels.json` | `Station_id → beach name` from reverse-geocoded coordinates. |

Empirical prior scales (from `eda.ipynb` section 13, written to `_meta.json` under `priors`) are used as the scales for the weakly-informative priors in the Bayesian models. The intercept prior is centred on the interior-row mean of log10. All hierarchical-SD priors are Half-Normal with scale taken from the between-group SD of group-means. These are weakly informative in the Betancourt sense — they define the plausible region without forcing a specific answer. The priors consumed by the ladder are:

| Prior | Distribution | Empirical scale (current artifact) | Used in rung |
|---|---|---|---|
| `alpha_0` (grand intercept) | `Normal(loc=1.84, scale=1.0)` | interior-row mean of `log10_result` | v0 onwards |
| `alpha_month[m]` (month deviation) | non-centred with `sigma_month` | — | v1, v2, v3 |
| `sigma_month` | `HalfNormal(scale=0.23)` | between-month SD of monthly means from EDA §5 | v1, v2, v3 |
| `alpha_station[s]`, `alpha_county[c]` | non-centred with `sigma_station`, `sigma_county` | — | v2 onwards |
| `sigma_station` | `HalfNormal(scale=0.75)` | between-station SD of station means | v2 onwards |
| `sigma_county` | `HalfNormal(scale=0.34)` | between-county SD of county means | v2 onwards |
| `sigma_obs` | `HalfNormal(scale=0.70)` | residual SD after removing station × month means | all rungs |
| `beta_linear[k]` | `Normal(loc=0, scale=0.5)` | 1.5 × max marginal log10-slope per SD | v3 onwards |
| `hsgp_amplitude_seasonal` | `HalfNormal(scale=0.23)` | same as `sigma_month` (same signal, different parameterisation) | v6 |
| `hsgp_amplitude_rain` | `HalfNormal(scale=0.24)` | magnitude of empirical log10-per-log1p(mm) rain slope | v6 |

The rungs v4 (polynomial on doy) and v5 (spline on doy) use broad Normal priors on their basis coefficients with scale set so the prior predictive seasonal amplitude matches the EDA range (peak-to-trough ~0.7 log10). Details are in `src/modeling/bayesian.py` at implementation time.

---

## 4. Modeling approach — two phases

Two distinct comparisons, run sequentially. Each produces one table.

**Phase A — Bayesian complexity ladder (internal):** fit progressively richer Bayesian models, score on held-out CV, show where added complexity earns its keep. The output is a single winning Bayesian model.

**Phase B — Winner vs non-Bayesian baselines (external):** run the same cross-validated evaluation on three non-Bayesian regressors (naive seasonal-mean, OLS, XGBoost) and the Phase A winner. The output is a defensible answer to *"did the Bayesian overhead pay off versus classical alternatives?"*

Both phases use the same CV folds, the same held-out rows, and the same metric suite, so every row in every table is comparable.

---

## 5. Phase A — Bayesian complexity ladder

The ladder exists to make each modelling choice *earn* its place. At every rung we ask: does the added structure improve validation counts-MAE by enough to justify the complexity cost? If not, we stop climbing. This produces an auditable answer to "why is the final model shaped this way?" rather than an unjustified maximalist model.

**Design principle.** The month-conditional seasonal effect is the scaffold on which every other term sits. v0 establishes the no-skill floor, v1 installs the seasonal scaffold (exactly matching the Phase B naive baseline), and every rung v2 onwards adds structure as **adjustments on top of that scaffold**. Rungs v4–v6 do not *add* a seasonal term — they *replace* the month categorical from v1 with progressively smoother parameterisations of the same underlying seasonal function. At most one seasonal term is active at any rung.

The ladder progresses through three orthogonal axes — **seasonal parameterisation**, **hierarchical structure**, and **non-seasonal covariate form** — changing one axis at a time so we can attribute lift to a single change.

| Rung | Structure added / changed | What lives in `mu_it` after this rung | Justification |
|---|---|---|---|
| v0 | `mu_it = alpha_0` | pooled intercept only | Establishes the no-skill floor. If the floor is already close to the baselines, the problem is not well-posed for any regressor. |
| v1 | `+ alpha_month[m(i)]`, non-centred, hierarchical with `sigma_month ~ HalfNormal(0.23)` (EDA-elicited scale) | `alpha_0 + alpha_month` | **Seasonal scaffold — matches the Phase B naive baseline exactly.** EDA §5 shows ~0.7 log10 peak-to-trough range and ~3× swing in exceedance rate between wet and dry months. Every subsequent rung must beat a model that already knows the season. |
| v2 | `+ alpha_station + alpha_county`, non-centred partial-pooling with EDA-elicited `sigma_station=0.75`, `sigma_county=0.34` | `alpha_0 + alpha_month + alpha_station + alpha_county` | Most variance lives between stations (EDA §13). This rung adds hierarchical intercepts *on top of* the seasonal scaffold — stations are now offsets from the season-appropriate mean, not from the grand mean. |
| v3 | `+ beta . X_linear` (lag rainfall z-scores, dry-days, `yesterday_log10_result`, SST, waves, tides, salinity) — **no `doy_sin / doy_cos`; seasonality is already in v1** | v2 + linear weather slopes | Marginal correlations show `yesterday_log10_result` r≈0.62 and rainfall lags r≈0.26. These covariates explain *within-month* variation that the seasonal scaffold cannot. |
| v4 | **replace** `alpha_month` with degree-3 polynomial of day-of-year; keep everything else | v3 with polynomial seasonality | Month categorical is a 12-step piecewise-constant function. A cubic polynomial is smoother and can express late-December-through-early-February peaks that don't respect month boundaries. If validation counts-MAE doesn't improve, smoothness isn't buying anything and we stop here. |
| v5 | **replace** polynomial with natural cubic spline (6 interior knots on doy) | v3 with spline seasonality | Splines are strictly more flexible than degree-3 polynomials and control tail behaviour via natural boundary conditions. If the spline beats the polynomial, shape-flexibility (not polynomial order) is what matters. |
| v6 | **replace** spline with HSGP on doy; **add** HSGP on the rainfall-lag cluster (shared amplitude across 24h/48h/72h/7d lags) | v3 with HSGP seasonality + HSGP rainfall | HSGP length-scale is data-driven and the rainfall-lag cluster gets a single coherent response instead of four independent linear slopes. Expected final architecture based on EDA, subject to validation. |
| v7 | `sigma_obs_j ~ HalfNormal(...)` per station | v6 + heteroskedastic observation variance | Only run if v6 is the validation winner. Clean beaches should have tighter residual variance than chronic-problem sites. Only retained if held-out counts-MAE improves. |

Stopping rule: the winner is the **simplest rung whose validation counts-MAE is within 1% of the best rung's validation counts-MAE**. This forces the ladder to justify every added degree of complexity.

The same generative likelihood (censored-normal on log10) and the same empirical priors apply at every rung. What changes between rungs is only the structure of `mu_it`, and — importantly — the seasonal parameterisation evolves as a **drop-in replacement**, never an addition. This keeps the ladder identifiable (no two terms fighting over the same seasonal signal) and makes the lift attributable.

**Sanity anchor between v1 and Phase B naive.** Because v1 = `alpha_0 + alpha_month` with a censored-normal likelihood on log10, its validation counts-MAE should be *close to but slightly better than* the Phase B naive seasonal baseline (which uses raw monthly means in count space with no censoring handling and no partial pooling across months). If v1 is *worse* than the naive baseline, there is a bug in the Bayesian pipeline (likelihood, priors, or indexing) — that's the smoke test for the ladder before v2 even runs.

---

## 6. Phase B — non-Bayesian baselines

Three baselines, fit with the same covariates as the Bayesian winner, evaluated on the same CV folds. The goal is to answer: does the Bayesian model beat simpler, faster alternatives?

### 6.1 Naive — month-conditional mean of counts (floor)

For each CV fold, compute twelve constants

```
mean_mpn_by_month[m] = mean(y_counts_train  |  sample_month == m)     for m in 1..12
```

on the training window, and predict `mean_mpn_by_month[m]` for every held-out row whose sample falls in month `m`. The predictive distribution at each held-out row is the empirical log10 distribution of *training observations from the same calendar month* (used to produce PPIs and `P(exceed)`). If a validation month has no training history (edge case), the predictor falls back to the pooled mean for that fold.

**Why conditioned on month, not pooled.** Section 5 of `eda.ipynb` establishes that the distribution of `log10(MPN)` shifts materially with month: the peak-to-trough range of monthly means is ~0.73 log10 (a ~5.4× fold change in the geometric mean between peak and trough months, ~3.2× in median counts between December and June), and monthly exceedance rates swing from ~9% in late summer to ~33% in winter against a pooled rate of 19.4%. In log space, month-only conditioning explains ~6% of the variance of `log10_result` and shrinks residual SD by ~3% — a modest accuracy gain in log space but a large shift in the decision-relevant quantity (exceedance probability). A pooled-mean baseline would systematically over-predict quiet summer days and under-predict wet winter days against a clearly calendar-structured signal, so the naive baseline conditions on month. This is the weakest baseline that already respects the most obvious feature of the data.

This is still the simplest possible model in the comparison — no station, no weather, no pooling — but it is the simplest *defensible* one. Beating it is a necessary but not sufficient bar for a usable model.

### 6.2 OLS linear regression on log10(y)

Ordinary least squares on log10 with the same covariate set as Bayesian v3 (i.e. the ladder rung that introduces the full linear-covariate block):

- Station and county one-hot (no pooling — contrast with the Bayesian hierarchy).
- Linear terms for all lag features, standardized.
- `doy_sin + doy_cos` for season.
- Missingness indicators matched to the Bayesian design.

Point prediction is `10^(X . beta_hat)`. Predictive distribution is Normal with variance = OLS residual variance + linear-prediction-variance (standard frequentist predictive interval), then back-transformed to MPN.

This is the natural classical counterpart to the Bayesian model and tests whether the Bayesian overhead (censoring, partial pooling, smooths, priors) is earning its keep versus a well-specified linear model.

### 6.3 XGBoost regressor on log10(y)

Gradient-boosted trees on the same covariates. Single model, point prediction only. Hyperparameters tuned via early stopping on each fold's training-vs-validation split (not the outer CV — an inner train/val split within the training window).

XGBoost natively produces only a point estimate. To produce PPIs and `P(exceed)` for a fair apples-to-apples comparison against the other models, we apply the same assumption OLS implicitly makes: the residuals are Normal with a constant variance, estimated as the training-residual SD on log10. The resulting predictive distribution is `Normal(xgb_pred, sigma_train_residual)`. This is the simplest retrofit for probabilistic metrics and is explicitly noted as a limitation — it is likely conservative versus a properly-calibrated XGBoost (e.g., conformal prediction), but introducing conformal would add a method the user did not request.

The Bayesian model's advantage on coverage, Brier, and ECE is expected to come largely from its native handling of uncertainty rather than from this retrofit assumption. If XGBoost still beats the Bayesian model on coverage-style metrics under this conservative retrofit, that is a meaningful result.

---

## 7. Cross-validation scheme

Both phases use the same temporal CV scheme defined in `eda.ipynb` and persisted in `_meta.json["cv_folds"]`.

**Structure:** expanding-window rolling-origin, four folds.

| Fold | Train window | Validation window |
|---|---|---|
| 0 | 2010-01-01 → 2019-12-31 | 2020 |
| 1 | 2010-01-01 → 2020-12-31 | 2021 |
| 2 | 2010-01-01 → 2021-12-31 | 2022 |
| 3 | 2010-01-01 → 2022-12-31 | 2023 |

**Test window:** 2024-01-01 → 2025-12-31. Never touched during ladder development or winner selection. Fits on train + val combined (i.e. all data up through 2023-12-31) and scores only on test.

Justification for rolling-origin over random k-fold: environmental time series have strong temporal structure (seasons, persistent storms, baseline drift), so random splits leak future information into training and over-estimate accuracy. Rolling-origin respects the temporal direction of forecasting and matches the production setting.

**Development budget.** To keep iteration tractable during ladder construction, Phase A uses only two folds (val_year = 2022 and 2023) while rungs are being compared. The chosen winner is then re-fit on all four folds for its reported final numbers. Phase B baselines always use all four folds (they are fast). The test-set fit is a single fit on train+val combined.

---

## 8. Evaluation framework

All metrics are computed per CV fold, then averaged across the folds used for that comparison (two folds during ladder development, four folds for the winner and the Phase B baselines). **Every number in both comparison tables is a fold-average.** The standard deviation across folds is reported alongside each mean so that fold-to-fold stability is visible.

### 8.1 Primary selection metric

**counts-MAE** — mean absolute error in MPN units.

```
counts-MAE_fold = mean_i | y_i_mpn_true - y_i_mpn_pred |
counts-MAE = mean_fold counts-MAE_fold
```

Point predictions for the metric:

- **Naive:** `mean_mpn_by_month[m]` for the held-out row's calendar month `m` (month-conditional mean of training counts).
- **OLS:** `10^(X . beta_hat)`.
- **XGBoost:** `10^xgb_point`.
- **Bayesian:** `10^posterior_median(mu_it)`. The posterior median is used (not the posterior mean) because medians commute with monotone transforms, so `10^median(mu) = median(10^mu)`. The posterior mean of `10^mu` is Jensen-biased upward and is not a fair MAE-minimising point estimate.

Interior (non-censored) held-out rows are used for the MAE computation. Censored rows have no scalar truth to difference against, so they are excluded from counts-MAE (handled separately for probabilistic metrics below).

### 8.2 Secondary metrics, also fold-averaged

All computed per fold, averaged across folds:

| Metric | Definition |
|---|---|
| **counts-MedAE** | median of `\| y_true - y_pred \|` in MPN, per fold, then averaged. Robustness check against single-day storm outliers that dominate mean MAE. |
| **coverage at 50 / 80 / 95%** | Fraction of held-out rows whose true `log10(y)` falls inside the predictive 50% / 80% / 95% interval. Target values are 0.50, 0.80, 0.95 respectively. |
| **Brier score** | `mean( (p_exceed - 1[y > 104])^2 )` over held-out rows. Strictly proper score for probability forecasts. Lower is better. |
| **ECE** (expected calibration error) | Binned weighted gap between predicted `p_exceed` and empirical exceedance rate, averaged across 10 equal-width bins. Lower is better. |

All of these are computed first per fold, then averaged across folds. The table cell reported is `mean ± std` across folds.

### 8.3 Diagnostic plots (not ranked, but always produced)

Plots are generated per fold so that fold-to-fold behaviour is visible. Two types:

- **Reliability diagram.** For each fold, bin held-out `p_exceed` into 10 deciles; plot mean predicted probability per bin vs empirical exceedance rate in that bin. A well-calibrated model lies on the 45-degree line. A summary panel shows all four folds overlaid, with the fold-average calibration curve in bold.
- **PIT histogram.** For each held-out interior row, compute `u_i = F_pred(y_i_true)` where `F_pred` is the model's predictive CDF on log10. A calibrated model produces `u ~ Uniform(0, 1)`. A U-shape flags over-confidence (intervals too narrow); a humped histogram flags under-confidence.

These plots are the visual evidence for the scalar calibration metrics (Brier and ECE). A model can have a low Brier score while still showing a systematically biased reliability diagram — the plots are what catch the sign of the miscalibration.

---

## 9. Bayesian workflow — the extra-care clause

Bayesian models are scored by the same metrics as the baselines, but they are *built* through a stricter workflow because their value — calibrated uncertainty — is also the thing that can fail silently. A Bayesian model that did not converge, or whose priors pushed the posterior into an implausible region, will produce numerically reasonable predictions with wrong uncertainty. Each of the four steps below is a gate: the workflow does not proceed to the next step unless the current one passes.

### 9.1 Prior predictive check (before any fit)

Before sampling from the posterior, sample from the prior predictive: `y_sim ~ p(y | priors_only)`. Transform the simulated log10 draws back to MPN. The per-station distribution of simulated medians and the distribution of simulated `P(exceed)` across stations should be plausible — roughly covering the 1 to 10^4 MPN range, with exceedance rates in the 0–50% range across stations. If the prior predictive puts mass on 10^10 MPN or on universal exceedance, the priors are wrong and the fit will not be rescued by data.

Implementation: `numpyro.infer.Predictive(model, num_samples=2000)` with `posterior_samples=None`. Wrap in `arviz.from_numpyro(prior=...)`. Plot prior predictive `y` distribution, exponentiated to MPN, alongside observed histogram for visual agreement.

This check runs once per ladder rung before NUTS is invoked for that rung.

### 9.2 NUTS sampling

Standard configuration used at every rung and every fold:

- `num_warmup = 1500`, `num_samples = 2000`, `num_chains = 4`, `target_accept_prob = 0.90`.
- JAX/JIT compiled model, run on CPU (single-machine).
- Seed fixed per fold for reproducibility.
- Non-centred parameterisation for all station/county random effects from v1 onward (centred parameterisation is known to cause divergences in hierarchical normal models with moderate group-level variance).

The returned `MCMC` object is converted immediately to ArviZ `InferenceData` via `az.from_numpyro(mcmc, prior=..., posterior_predictive=...)` and saved to disk. Every later diagnostic and metric reads from the saved `InferenceData`, not from the live MCMC object — this means re-opening a diagnostic notebook never requires re-sampling.

### 9.3 Convergence diagnostics (gate before interpreting anything)

Required checks, all via ArviZ:

- **R-hat.** Every parameter should have `R-hat < 1.01`. Parameters above threshold are flagged and the sampler rerun with more warmup or a non-centred reparameterisation.
- **Effective sample size.** Bulk ESS and tail ESS both above 400 for every parameter of interest. Low ESS on a variance component is especially concerning — it typically means the posterior geometry is pathological and the parameter is not identified.
- **Divergences.** `mcmc.get_extra_fields()["diverging"].sum()` must be zero. Non-zero divergences invalidate the posterior in the neighbourhood of the divergent trajectories. The first response is reparameterisation (non-centred); cranking `target_accept_prob` is a symptomatic fix, not a cure.
- **Energy plot** (`az.plot_energy`). Marginal and transitional energy distributions should overlap. A much wider marginal indicates the sampler is struggling with the posterior geometry.
- **Trace plots** (`az.plot_trace`). Chains should be visually indistinguishable across their range.

A rung that fails any of these gates is not scored for model selection until the convergence issue is fixed. A rung whose convergence cannot be fixed is dropped from the ladder and its failure noted.

### 9.4 Posterior parameter inspection

After convergence passes, inspect the posterior for sanity before trusting predictive output:

- `az.plot_forest` for `alpha_month` — the twelve non-centred month deviations should recover the observed monthly climatology from EDA §5 (Dec/Jan/Feb elevated, Sep lowest). If the posterior forest disagrees with the raw data, something is wrong upstream.
- `az.plot_forest` for `beta_linear` — which features did the model learn a non-zero slope on? Expected: `yesterday_log10_result` and the rainfall cluster are the largest. `sst_c` is expected to collapse toward zero (or flip sign from its marginal correlation) after conditioning on season (which the month scaffold now explicitly provides).
- `az.plot_forest` for `alpha_station` — the three-regime structure (clean / borderline / chronic) visible in EDA should be visible here.
- `az.plot_pair` for `(sigma_station, sigma_county, sigma_month, sigma_obs)` — check for pathological posterior correlations between variance components (a sign of weak identifiability, especially between `sigma_month` and a seasonal smooth's amplitude hyperparameter in rungs v4–v6 if they ever coexist — they should not, per the "one seasonal term at a time" rule).
- `az.plot_posterior` for each variance component with 89% HDI reported.

The goal at this step is not to accept or reject the model — the held-out metrics do that — but to confirm the posterior is coherent with EDA expectations. A model that disagrees sharply with strong EDA signals needs explanation before its held-out metrics can be trusted.

### 9.5 Posterior predictive check

Sample `y_rep` from the posterior predictive on the training set and compare to the observed data:

- `az.plot_ppc` overlaying ~100 posterior-predictive draws against the observed histogram. They should bracket the observed distribution cleanly.
- `az.plot_bpv` with `t_stat="mean"` and with `t_stat="std"` for Bayesian p-values on the first and second moments.
- A custom per-station PPC: posterior-predictive exceedance rate per station vs empirical exceedance rate per station. This is the operationally important check — a model whose posterior predictive reproduces the observed exceedance pattern station-by-station is where the decision probability will be trustworthy.

### 9.6 Per-rung execution order

Every rung of the ladder runs in this order:

1. Prior predictive check → gate on plausibility.
2. NUTS fit per development fold.
3. Convergence diagnostics per fold → gate before scoring.
4. Posterior inspection → sanity check.
5. Posterior predictive check → sanity check.
6. Held-out evaluation on validation → produces the scalar row for the Phase A comparison table.

The baselines skip steps 1, 3, 4, 5 (they have no prior, no sampler, no posterior), run only steps 2 and 6 in adapted form (fit → evaluate), and populate the Phase B table.

---

## 10. The two comparison tables

**Phase A table — Bayesian ladder.** One row per rung, each cell a fold-average (`mean ± std`).

| Model | counts-MAE | counts-MedAE | cov-50 | cov-80 | cov-95 | Brier | ECE |
|---|---|---|---|---|---|---|---|
| v0 pooled intercept (floor) | | | | | | | |
| v1 + month scaffold (seasonal baseline) | | | | | | | |
| v2 + station/county hierarchy | | | | | | | |
| v3 + linear weather covariates | | | | | | | |
| v4 replace month → polynomial season | | | | | | | |
| v5 replace polynomial → spline season | | | | | | | |
| v6 replace spline → HSGP season + HSGP rain | | | | | | | |
| v7 + station-specific sigma | (only if needed) | | | | | | |

Decision rule: simplest rung within 1% of best `counts-MAE` wins.

**Phase B table — winner vs baselines.** One row per model, each cell a fold-average.

| Model | counts-MAE | counts-MedAE | cov-50 | cov-80 | cov-95 | Brier | ECE |
|---|---|---|---|---|---|---|---|
| Naive seasonal mean of counts (by month) | | | | | | | |
| OLS on log10 | | | | | | | |
| XGBoost regressor | | | | | | | |
| **Bayesian (winner)** | | | | | | | |

Narrative: present the ladder plot (counts-MAE vs rung), read off the winner, then present Phase B table showing the winner beats (or loses to) the baselines.

---

## 11. Final selection and test-set evaluation

After the winner is chosen from Phase A:

1. Refit the winning model architecture on **train + validation combined** — all data through 2023-12-31.
2. Predict on the held-out test window (2024-01-01 → 2025-12-31).
3. Compute the same metric suite on the test window. This is reported alongside the CV numbers as an out-of-sample confirmation; it is not used for model selection (the selection is already done).
4. Run the full diagnostic suite (posterior parameter inspection, posterior predictive check, reliability diagram, PIT histogram) on the final-fit `InferenceData`.
5. Produce per-station fan charts on the test window — observed scatter + predictive 50% / 80% bands + advisory threshold line — for the six most-sampled beaches. These are the plots that go into any stakeholder-facing write-up.

The final `InferenceData` is saved to `artifacts/modeling/{winner}/final/idata.nc` and is what the productionization notebook consumes.

---

## 12. Productionization — `notebooks/modeling/predict.ipynb`

The prediction notebook is deliberately thin: it loads the final model artifacts, scores an input dataframe, and writes dashboard-ready tables. It does not re-fit and does not re-compute diagnostics.

### 12.1 Inputs

- `artifacts/modeling/{winner}/final/idata.nc` — posterior samples.
- `artifacts/data/panel/enterococcus_panel_meta.json` — feature list, scalers, station index.
- Either the full historical panel (for backtest rescore) or a daily prediction grid for "today" (per-station feature vectors for the current date).

### 12.2 Outputs

Two tables always emitted, one optional:

**`predictions_backtest.parquet`** — full historical rescore, one row per `(station_id, sample_date)` in the panel. Columns:

| Column | Type | Meaning |
|---|---|---|
| `station_id`, `station_name`, `county` | ids | merged from station label cache |
| `sample_date` | date | prediction date |
| `median_mpn` | float | posterior median of `10^mu` in MPN units |
| `q10_mpn`, `q25_mpn`, `q75_mpn`, `q90_mpn` | float | posterior predictive quantiles in MPN (50% and 80% PPI endpoints) |
| `p_exceed` | float in [0,1] | posterior probability `y > 104` |
| `p_exceed_hdi_low`, `p_exceed_hdi_high` | float | 89% HDI on `p_exceed` across posterior draws |

**`predictions_today.parquet`** — single-day forecast for every station given the current day's features. Same column schema as backtest; one row per station.

**`predictions_samples.parquet`** (optional, controlled by flag) — long-format posterior draws, for the internal website that wants to display full distributions rather than summaries. Columns: `station_id, sample_date, draw_idx, log10_y, mpn`. Default draw count: 500 (sufficient for histogram display without making the file unreasonably large).

### 12.3 Consumption by downstream dashboards

- **Power BI dashboard.** Reads `predictions_backtest.parquet` and `predictions_today.parquet` as ordinary tables. The median and quantile columns power fan charts; `p_exceed` powers a beach-level heatmap or advisory panel. No Python dependency in the BI runtime.
- **Internal website (Python-backed).** Reads `predictions_samples.parquet` for per-station distribution displays, or loads `idata.nc` via `az.from_netcdf` for any new diagnostic the team wants to compute. This is the flexible consumer; the website code lives downstream of this project.

### 12.4 Entry-point signature

```python
def predict(
    panel_df: pd.DataFrame,        # rows to score: must have the feature columns and station_idx
    idata_path: Path,              # artifacts/modeling/{winner}/final/idata.nc
    output_dir: Path,              # where the parquet files go
    include_samples: bool = False, # emit predictions_samples.parquet?
    n_sample_draws: int = 500,     # how many posterior draws to retain if include_samples
) -> None
```

Lives in `src/modeling/predict.py`. The notebook is a thin wrapper that calls this function twice (once for the historical panel, once for today's grid) and emits a small summary print of row counts and file sizes.

---

## 13. Repository layout

Files this plan will produce, in addition to what already exists.

```
src/modeling/
    baselines.py          # NaiveSeasonalMeanCounts, OLSLog10, XGBoostRegressor; common interface
    bayesian.py           # build_model(rung, data_spec) — NumPyro model for each v0..v7
    inference.py          # run_nuts() -> InferenceData; saves to artifacts/
    cv.py                 # run_cv_ladder() — fits and scores all rungs per fold
    predict.py            # predict() — productionization entrypoint

src/evaluation/
    metrics.py            # counts_mae, counts_medae, coverage, brier, ece
    calibration.py        # reliability_diagram, pit_histogram
    compare.py            # score_all(), build_comparison_table()

src/viz/
    ladder.py             # counts-MAE vs rung plot
    forecast.py           # per-station fan charts, calendar risk heatmap
    diagnostics.py        # thin wrappers around az.plot_* with project defaults

notebooks/modeling/
    eda.ipynb             # EXISTING — panel + priors + CV folds + preview
    censoring_toy.ipynb   # EXISTING — teaching demo for censored likelihood
    model.ipynb           # NEW — the monolith: baselines + ladder + comparison + winner + test + diagnostics
    predict.ipynb         # NEW — thin productionization notebook

artifacts/modeling/
    v{N}/
        fold_{year}/
            idata.nc              # ArviZ InferenceData per fold
            metrics.json          # scalar row for that rung/fold
    {winner}/
        final/
            idata.nc              # train+val fit, used by predict.ipynb
            metrics.json          # test-set scalars

context/
    MODELING_PLAN.md      # THIS FILE
```

---

## 14. Execution phases

The work proceeds in phases so that each phase produces a testable artefact before the next is started. Earlier phases do not depend on later phases, and each phase is independently runnable.

| Phase | Scope | Output |
|---|---|---|
| 1 | `src/evaluation/{metrics,calibration,compare}.py` + `src/modeling/baselines.py` + `model.ipynb` sections 0–1 | Working Phase B baseline comparison table (naive seasonal-mean, OLS, XGBoost) across four CV folds |
| 2 | `src/modeling/{bayesian,inference,cv}.py` + `model.ipynb` section 2 (Bayesian ladder v0–v6, v7 optional) | Phase A comparison table, ladder plot, winner chosen |
| 3 | `model.ipynb` sections 3–5 | Phase B table populated with Bayesian winner, test-set fit + evaluation, final diagnostic plots |
| 4 | `src/modeling/predict.py` + `predict.ipynb` | `predictions_backtest.parquet`, `predictions_today.parquet`, optional samples file |

Each phase ends with its artefact committed and the notebook executable end-to-end before the next phase starts.
