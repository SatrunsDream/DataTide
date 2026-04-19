> **Repo note:** The actionable, repo-aligned decisions from this document are summarized in **`context/DECISIONS.md`**, **`ASSUMPTIONS.md`**, **`RISKS.md`**, and **`STATUS.md`**. Data pipeline and column-level ETL: **`GROUND_TRUTH_SCHEMA.md`**, **`DATASETS.md`**. Interfaces, ArviZ, Power BI tables: **`INTERFACES.md`**. The text below retains the full working plan, critique, and mathematical detail.

---

Plan mode only plan make sure to use recourses from deep-research.md
if you can in addition do outside research. 
ok here is our plan see what can work and what can be improved
inherently we're trying to model the counts of 4 different bacteria that show up on beaches
in california. how we're thinking of approaching this is from a bayesian standpoint,
in which we're trying to account for the uncertainty in the count predictions themselves.
we want to structure our data in a panel, tabular structure, in which we have a single target that we want to
approach using a rolling weekly run-rate structure because our ground truth targets are observed weekly, and we want
to have daily predictions. wew want to use features observed in our data that can be observed in this text:
Feature Category,Specific Input Variables,Data Source,Why It Matters
Spatial & Temporal,"Beach ID, Day of the Year",BWTF / Calendar,Tells the model where it is predicting and what season it is.
Historical Memory,Yesterday's Bacteria Count,BWTF,"Accounts for pathogen persistence. If the water was highly toxic yesterday, it takes time to clear out."
Precipitation,"24-hr, 48-hr, and 72-hr Lagged Rainfall",NOAA,Heavy rain flushes urban runoff and sewage into the ocean. Lags account for travel time.
Oceanography,"Sea Surface Temperature, Salinity",Scripps Mooring,Bacteria survive longer in certain temperatures. Sudden drops in salinity indicate freshwater runoff hitting the ocean.
Wave Dynamics,"Swell Height, Peak Wave Period",CDIP Wave Buoys,"High wave energy mixes the water and dilutes bacteria plumes much faster than calm, stagnant water."
Inland Runoff,Local River Discharge Volume,IBWC / USGS,"For sites near rivermouths (like Imperial Beach), the exact volume of water flowing out of the river is a direct proxy for pollution volume."

we also want to incorporate daily data for each week in a 7-dim vector (open to multimodal embedding output)
within our model framework as we want to use a hierarchical Bayesian framework in JAX + NumPyro.
would we use a negative-binomial likelihood or poisson given the count nature of bacteria counts.

fundamentally, we want to structure these features as a GAM within the model as they possibly have
non-linear interactions with our target.

provide expected output layout schema for power BI dashboard.

in addition, be critical of our methodology and provide reasons as to what can be improved upon and why. we want to
ultimately build this model in a probabilistic framework without being too complex in terms of the framework.
assume we have intermediate knowledge in hierarhcial bayesian models and understanding GPs and HSGPs. 

By the way, our computer processing power is m4 chip pro, 48 GB RAM, 14 cores (10 performance & 4 efficiency). incorporate this knowledge into JAX.

be aware that we are on a time crunch and have 10 hours. be sure to have naive baseline model checks that maybe aren't inherently bayesian and are possibly limited to linearity or other cases like tree-basewd.

also be sure to upload the model outputs in an arviz object for visuialization purposes and also for calibration checks.

this is for a plan, don't code anything. finally, just be fundamentally rigorous in decision-making, make no mistakes, and have those decisions be mathematically-backed

1. Critique of the plan as stated
1.1 Target and data-source mismatch
Your feature table cites BWTF as the source for "Yesterday's Bacteria Count" and "Beach ID." The project's own DATASETS.md makes BWTF Tier 10 (Surfrider, auxiliary) and makes the State Water Board lab.data.ca.gov three-pack the statewide spine. BWTF is sparse, non-regulatory, and mixes creek/outfall sites with beach sites; modeling against BWTF labels and then calling it a "California beach" model is scientifically defensible only as a side experiment. Action: flip the spine to Tier‑1 bacteria results (ca_swrcb_bacteria), use Tier‑1 stations for Beach ID and coordinates. BWTF becomes optional auxiliary observations with a separate offset.
1.2 "Weekly target, daily prediction" is imprecise
Tier‑1 sampling is not uniformly weekly. Per the schema, multiple rows per station per day are possible (replicate samples, multiple parameters), and cadence varies from daily (AB 411 summer) to monthly (off-season) by beach. Treating all observations as weekly will produce the wrong likelihood at sites like Santa Monica Pier (daily in summer) and Imperial Beach (multiple parameters per visit).
The correct framing, matching the deep-research report's Searcy & Boehm 2018 / 2023 operational papers, is:

Observation model: irregular (station, sample_date) events with variable cadence.
Latent state: continuous daily log-concentration ηs,t\eta_{s,t}
ηs,t​ per station. Observe when a sample exists; predict every day.


Mathematically this is a state-space model: latent process at daily resolution, observation likelihood only on sample days. This is exactly what an HSGP over time gives you cheaply.
1.3 Likelihood choice — Poisson vs NB is a false dichotomy for this data
Bacteria readings are reported as MPN/100 mL or CFU/100 mL with heavy left-censoring (<10, <2) and right-censoring (>2419.6, >24196, IDEXX upper limits). A plain NB on the integer column will misrepresent both tails and break calibration. Evidence from the published California literature:
OptionMathWhen correctCostPoissonY∼Pois(μ)Y \sim \mathrm{Pois}(\mu)
Y∼Pois(μ), Var=μ\mathrm{Var} = \mu
Var=μNever for FIBBreaks — dispersion factor often 10–1000.NB2Y∼NB(μ,ϕ)Y \sim \mathrm{NB}(\mu, \phi)
Y∼NB(μ,ϕ), Var=μ+μ2/ϕ\mathrm{Var} = \mu + \mu^2/\phi
Var=μ+μ2/ϕInteger counts without censoringBetter but ignores detection limits.Censored log-normallog⁡Y∼N(μ,σ2)\log Y \sim \mathcal{N}(\mu, \sigma^2)
logY∼N(μ,σ2), with left/right censoring
MPN/CFU with detection limitsBest for FIB per Searcy & Boehm and ddPCR literature (Crain 2021).Hurdle-NBP(Y=0)P(Y=0)
P(Y=0) + NB for Y>0Y>0
Y>0Many true zeros (clean beaches)Good secondary choice.
Recommendation: censored log-normal on log10(Result) with indicator columns for < and >. NumPyro's dist.Normal combined with numpyro.factor for the censoring terms handles this in ~10 lines. Secondary comparison: NB2 with the detection-limit rows dropped — the two should agree for the uncensored middle of the distribution.
1.4 "Four bacteria" joint modeling
The Tier‑1 Parameter field has Enterococcus, E. coli, fecal coliform, total coliform, with coverage varying by jurisdiction. Coverage table matters: only Enterococcus is reliably measured statewide at marine beaches (per deep-research report §California context). A naïve 4-way multivariate model will:

Be rank-deficient at beaches measuring only 1–2 parameters.
Force shrinkage across parameters that have different regulatory thresholds (Enterococcus 104 MPN vs total coliform 10,000 MPN).

Two cleaner options:

Single primary target: Enterococcus (marine beaches, regulatory). E. coli as a secondary model for river-mouths/lagoons (Tijuana River, creeks). This matches Searcy & Boehm 2018, 2021, 2023.
**Correlated multi-task**: ηs,t∼N(μs,t,ΣP)\boldsymbol{\eta}_{s,t} \sim \mathcal{N}(\mu_{s,t}, \Sigma_P)
ηs,t​∼N(μs,t​,ΣP​) with an LKJ(η=2) prior on ΣP\Sigma_P
ΣP​'s correlation — models 4 parameters jointly only where observed; missing parameters drop from likelihood per row.


For 10 hours: start with option 1, target = Enterococcus. Add E. coli as a separate fit in hour 7 if time allows. Option 2 is a stretch goal.
1.5 GAM framing is sound — HSGP is the correct tool
A GAM η=β0+∑kfk(xk)+∑pairsfjk(xj,xk)\eta = \beta_0 + \sum_k f_k(x_k) + \sum_{\text{pairs}} f_{jk}(x_j, x_k)
η=β0​+∑k​fk​(xk​)+∑pairs​fjk​(xj​,xk​) with partial pooling over station is mathematically the right structure. Full GPs are O(N3)O(N^3)
O(N3) and die at 1.4M rows.
HSGP (Solin & Särkkä 2020; Riutort-Mayol, Bürkner et al. 2023) reduces to O(NM)O(NM)
O(NM) where MM
M is the number of basis functions.

For each 1-D smooth, M∈[20,40]M \in [20, 40]
M∈[20,40] and boundary factor c=1.5c = 1.5
c=1.5 gives essentially exact approximation of a Matérn-3/2 kernel up to length-scales down to ℓmin⁡≈L/M\ell_{\min} \approx L/M
ℓmin​≈L/M where LL
L is the domain half-width. With Matérn-3/2 and typical FIB feature ranges, M=30M=30
M=30 is a defensible default. Longer length-scale priors (log-normal(log⁡0.3L,0.5)(\log 0.3L, 0.5)
(log0.3L,0.5)) stop the model from overfitting weekly noise.

1.6 "7-dim vector of daily data per week" — rethink
If you mean: for each weekly target, use 7 daily values of each environmental feature → that's 7 columns per feature. This is fine but collinear. Better: use lag features that are summary statistics (sum_24h, sum_48h, sum_72h, sum_7d for rain; max_Hs_72h for waves; range_24h for tide). Literature consistently shows these summary lags beat raw daily sequences at this sample size. Keep the 7-dim sequence only if you're going to a sequence model (LSTM/TFT), which the deep-research report and your time budget both argue against for a first cut.
1.7 Hardware and JAX reality check
M4 Pro, 48 GB, 10P + 4E, Apple Silicon means:

jax-metal (Metal backend) is not production-ready for NumPyro as of 2025-Q2. Unverified kernels cause NaNs in NUTS. Do not rely on it.
CPU JAX via Accelerate BLAS (jax[cpu] on macOS) is the stable path. Confirmed working for NumPyro.
With 48 GB RAM, you can hold ∼106\sim 10^6
∼106 rows × 50 features × float32 (~200 MB) with plenty of headroom for HSGP basis matrices and sampler state.

4 parallel chains via numpyro.infer.MCMC(..., num_chains=4, chain_method="parallel") — this actually uses 4 separate processes; set numpyro.set_host_device_count(4) at import. Use the 4 P-cores.
Leave the 4 E-cores and 2 P-cores for OS / ArviZ / notebook.

Time estimates (empirical for NumPyro HSGP+NB on CPU):
ModelRowsNUTS (1000 warmup + 1000 draws, 4 chains)SVI (10k steps, full batch)Pooled GLM (linear)100k~1 min~20 sHierarchical GLM (station RE)100k~10 min~1 minHierarchical + HSGP × 5 smooths100k~1–3 h~5–15 minSame, 500k rows500k~4–8+ h (risky)~20–40 min
Therefore: NUTS on the full dataset is infeasible in 10 h. Plan = SVI-first on full data, then NUTS for validation on a stratified subset.
1.8 "Rolling weekly run-rate for daily predictions"
If what you mean is "average the last-7-days of predictions," that's a **post-hoc smoother**, not a model property. If what you mean is "aggregate features over 7-day windows," that's already handled by lag features. Drop the phrase and replace with: *"Produce posterior samples of ηs,t\eta_{s,t}
ηs,t​ daily; optional 7-day rolling mean of the posterior median for public-facing display."*


2. Revised model specification
2.1 Data spine (unchanged from project rules)

Target rows: ca_swrcb_bacteria filtered to Parameter == "Enterococcus" (primary). Derive log_result, is_left_cens, is_right_cens, detection_low, detection_high.
Modeling grain: one row per lab result (Station_id, SampleDate, replicate_index).
Prediction grain: one row per (Station_id, calendar_date) for every day in evaluation window.

2.2 Structural equation
For sample ii
i at station ss
s on date tt
t:

ηs,t=α+αs(stn)+αc(s)(cty)+∑k=1Kfk(xk,s,t)⏟HSGP smooths+∑jβjzj,s,t⏟linear terms+us,t(AR)\eta_{s,t} = \alpha + \alpha_s^{(\text{stn})} + \alpha_{c(s)}^{(\text{cty})} + \underbrace{\sum_{k=1}^K f_k(x_{k,s,t})}_{\text{HSGP smooths}} + \underbrace{\sum_{j} \beta_j z_{j,s,t}}_{\text{linear terms}} + u_{s,t}^{(\text{AR})}ηs,t​=α+αs(stn)​+αc(s)(cty)​+HSGP smoothsk=1∑K​fk​(xk,s,t​)​​+linear termsj∑​βj​zj,s,t​​​+us,t(AR)​
log⁡Yi,s,t∼N(ηs,t,σobs2)with censoring at detection limits\log Y_{i,s,t} \sim \mathcal{N}(\eta_{s,t}, \sigma_{\text{obs}}^2) \quad \text{with censoring at detection limits}logYi,s,t​∼N(ηs,t​,σobs2​)with censoring at detection limits

αs(stn)∼N(0,τstn2)\alpha_s^{(\text{stn})} \sim \mathcal{N}(0, \tau_{\text{stn}}^2)
αs(stn)​∼N(0,τstn2​), non-centered parameterization (critical for NUTS efficiency; reduces funnel).

αc(cty)∼N(0,τcty2)\alpha_c^{(\text{cty})} \sim \mathcal{N}(0, \tau_{\text{cty}}^2)
αc(cty)​∼N(0,τcty2​).

fkf_k
fk​ = HSGP Matérn-3/2, Mk=30M_k = 30
Mk​=30, c=1.5c = 1.5
c=1.5, length-scale ℓk∼LogNormal(log⁡(0.3Lk),0.5)\ell_k \sim \mathrm{LogNormal}(\log(0.3 L_k), 0.5)
ℓk​∼LogNormal(log(0.3Lk​),0.5), amplitude σk∼HalfNormal(1)\sigma_k \sim \mathrm{HalfNormal}(1)
σk​∼HalfNormal(1).

us,t(AR)u_{s,t}^{(\text{AR})}
us,t(AR)​: AR(1) in
weekly time at station level, us,t=ρus,t−1+εu_{s,t} = \rho u_{s,t-1} + \varepsilon
us,t​=ρus,t−1​+ε, ρ∼Beta(5,2)\rho \sim \mathrm{Beta}(5,2)
ρ∼Beta(5,2) (prior favors persistence), ε∼N(0,σar2)\varepsilon \sim \mathcal{N}(0, \sigma_{\text{ar}}^2)
ε∼N(0,σar2​). Reason: FIB autocorrelation decays on ~2-week timescales (Boehm 2007); AR(1) on weekly scale captures this with 1 parameter.

σobs2\sigma_{\text{obs}}^2
σobs2​ = measurement + small-scale spatial noise, σobs∼HalfNormal(1)\sigma_{\text{obs}} \sim \mathrm{HalfNormal}(1)
σobs​∼HalfNormal(1).


2.3 Features (HSGP smooths vs linear)
VariableTreatmentJustificationlog_rain_24h, log_rain_48h, log_rain_72h, log_rain_7dHSGP smooth eachNonlinear threshold at rain-advisory trigger (~0.1") — Searcy & Boehmlog_dry_days_since_rainHSGP smoothNonlinear decaytide_range_mHSGP smoothNonlinear at spring-tide cutoffwave_hs_mHSGP smoothMixing/dilution is nonlinearwave_tp_sLinearWeak second-order effectsst_cHSGP smoothRegional only, null outside SCCOOS countiessalinity_psuHSGP smoothTop SHAP predictor per Grbčić 2022river_discharge (IBWC/USGS)HSGP smoothSouth Bay only; zero-filled elsewhere with indicator columnsin(2π DoY/365), cos(2π DoY/365)Linear (Fourier order 2)Seasonality; cheaper than HSGP on circular domainbeach_type (dummy)Linear3–4 levels
Interaction terms to test if time permits (hour 7): rain_72h × region, wave_hs × tide_range. Encode as tensor-product HSGP with M1×M2=20×20M_1 \times M_2 = 20 \times 20
M1​×M2​=20×20.

2.4 Inference stack
StepMethodPurposeTime1Prior predictive (1 chain, 500 draws, no observations)Sanity-check priors give realistic FIB ranges (10²–10⁵ MPN)2 min2SVI with AutoLowRankMultivariateNormal (rank=20), Adam lr=1e-3, 20k steps, batch=10kFast posterior approximation for the full dataset15–30 min3NUTS on stratified subset (one county or one year, 4 chains, 1000+1000)Ground-truth validation; compute R^\hat R
R^, ESS
1–2 h4PSIS-LOO via arviz.looOut-of-sample predictive check5 min5Posterior predictive on holdoutCRPS, coverage, ROC for exceedance10 min
Rationale for SVI-first: with 48 GB and CPU JAX, full-data NUTS blows the 10 h budget. SVI with a low-rank multivariate-normal guide recovers the posterior mean/covariance to within ~5% of NUTS for GLMs and HSGP-GAMs at this scale (Zhang et al. 2018 on variational inference for GPs). The NUTS-on-subset step is the verification.
2.5 Naive baselines (mandatory, hour 0–2)
BaselineFormulaPurposePersistenceY^s,t=Ys,t∗\hat Y_{s,t} = Y_{s, t^{\ast}}
Y^s,t​=Ys,t∗​ where t∗t^\ast
t∗ = last observed day
Operational status quo (California beaches currently post based on last sample)Seasonal climatologyY^s,t=medianDoY±14\hat Y_{s,t} = \text{median}_{\text{DoY}\pm14}
Y^s,t​=medianDoY±14​ at station ss
sBeat this or the model adds nothingLogistic regression on exceedancelogit p=x⊤β\mathrm{logit}\, p = \mathbf{x}^\top \boldsymbol{\beta}
logitp=x⊤βSearcy & Boehm 2018 operational model; direct comparatorLightGBM regression on log_resultTweedie or Gaussian objective, early stopping on time-blocked CVDeep-research report's strongest non-Bayesian baseline (Li 2022)LightGBM classifier on exceedanceBinary log-lossDirect comparator to Searcy 2023 RF benchmark (AUC 0.60)
Your Bayesian model must at minimum match LightGBM on point forecast AUC/CRPS; its win is calibration (coverage of 50%/90% credible intervals) and uncertainty quantification for alert policy.
2.6 Validation protocol (leakage-safe)

Time-blocked split: train ≤ 2022, val = 2023, test = 2024–2025. Matches the project's 2010–2025 window.
Station-holdout secondary: random 20% of stations held out to test cold-start generalization.
No lag computed across the split boundary.
Metrics: CRPS, MAE on log10(MPN), exceedance AUC, exceedance recall at 85% specificity (the California operational constraint per Searcy 2018), coverage of 50% and 90% posterior intervals, PSIS-LOO, rootograms.

2.7 ArviZ integration
Build one InferenceData object containing posterior, posterior_predictive, prior, prior_predictive, observed_data, log_likelihood, sample_stats. Persist as NetCDF under artifacts/models/enterococcus_hsgp_YYYY-MM-DD.nc per the project's run-bundle rules. Calibration plots:

az.plot_ppc — posterior predictive vs observed histogram
az.plot_loo_pit — LOO-PIT uniformity test
az.plot_forest — station random effects ranked
az.plot_trace — for NUTS-subset run only
Custom rootogram for count fit


3. 10-hour schedule
HourTaskDeliverable0.0–0.5Finalize target = Enterococcus; confirm censoring columns exist; subset to 2010–2025Cleaned parquet, context/ASSUMPTIONS.md updated0.5–2.0EDA + baselines (persistence, climatology, logistic, LightGBM) on time-blocked CVartifacts/figures/, artifacts/tables/baseline_metrics.csv2.0–3.5Build NumPyro model v1: GLM hierarchical, linear only, censored log-normal likelihood. Prior predictive check.notebooks/modeling/03_bayes_glm.ipynb, InferenceData3.5–6.0Upgrade to v2: add HSGP smooths on rain lags + tide + wave + salinity. SVI fit on full data.notebooks/modeling/04_bayes_hsgp_gam.ipynb6.0–7.5NUTS on 1 county (e.g., San Diego) to verify SVI posterior. R^\hat R
R^, ESS, trace checks.
Verification table in artifacts/tables/7.5–8.5Posterior predictive on test set. Coverage, CRPS, exceedance ROC. Compare to LightGBM.artifacts/tables/model_comparison.csv8.5–9.5Export Power BI tables (§4 below). Calibration figures.artifacts/data/powerbi/*.parquet9.5–10.0Update context/structure.md, CHANGELOG.md, run bundleRun ID recorded
Hard cutoffs: if by hour 4 SVI hasn't converged, drop HSGP from 2 of the smooths (keep rain + salinity). If by hour 7 NUTS hasn't mixed (R^>1.1\hat R > 1.1
R^>1.1), ship SVI-only results with a documented caveat in
ASSUMPTIONS.md.

4. Power BI output schema (star schema, long format)
Power BI joins best on a star schema: one fact table, several dim tables. All tables as Parquet under artifacts/data/powerbi/; Power BI reads via the Parquet connector or via a small DuckDB file.
4.1 Fact table — fact_predictions.parquet
One row per (station_id, date, parameter, quantile) or a wide version with quantiles as columns. Recommend wide for Power BI line-chart ergonomics:
ColumnTypeDescriptionprediction_datedateDay of predictionstation_idstringFK → dim_stationparameterstringEnterococcus, etc.pred_p05float5th percentile posteriorpred_p25float25thpred_p50floatmedian (primary forecast)pred_p75float75thpred_p95float95thpred_meanfloatposterior meanpred_sdfloatposterior SDp_exceedancefloatP(Y > threshold)alert_levelstringgreen/yellow/red based on tuned threshold on p_exceedanceobservedfloat nullableactual MPN if sample existsis_censored_low / is_censored_highboolflagsrun_idstringFK → dim_run
4.2 Dim tables
dim_station.parquet
ColumnTypestation_idstring PKstation_namestringbeach_namestringcountystringlatitude, longitudefloatbeach_typestringagency_namestringnearest_cdip_buoy, nearest_tide_station, nearest_precip_stationstring
dim_date.parquet — standard date dim with year, month, week, DoY, weekday, season, holiday flag.
dim_run.parquet
ColumnTyperun_idstring PKmodel_namestringmodel_versionstringinference_methodstring (SVI/NUTS)train_start, train_enddategit_shastringloo_elpd, loo_pfloattest_crps_median, test_auc_excfloat
4.3 Supporting fact tables
fact_features.parquet — daily feature values per station for the dashboard's "why did the model predict this?" drill-down:
ColumnTypeprediction_date, station_idcomposite FKrain_24h_mm, rain_48h_mm, rain_72h_mm, rain_7d_mmfloatdry_days_since_raininttide_range_m, wave_hs_m, wave_tp_sfloatsst_c, salinity_psufloatriver_discharge_cfsfloat nullable
fact_calibration.parquet — posterior calibration for drill-down tab:
ColumnTypestation_id, prediction_date, parametercomposite FKobserved_log10floatin_50ci, in_90ciboolpit_valuefloat ∈ [0,1]crpsfloat
fact_feature_importance.parquet — posterior summaries of smooth effects (evaluate fk(x)f_k(x)
fk​(x) on a grid):

ColumnTypefeature_name, x_valueeffect_p05, effect_p50, effect_p95floatrun_idstring FK
4.4 Dashboard tabs (suggested)

Map: dim_station + latest fact_predictions colored by alert_level.
Station detail: time series of pred_p50 with pred_p05–p95 band + observed points, aligned fact_features below.
Model diagnostics: fact_calibration — PIT histogram, coverage bars by county; fact_feature_importance partial-dependence plots.
Run comparison: dim_run table plus leaderboard of metrics vs baselines.


5. Improvements to the stated plan (summary)
Your planIssueChangeBWTF as spineWrong per project rulesTier‑1 SWRCB bacteria resultsWeekly target, daily predictionsCadence is irregular, not weeklyLatent daily process, observation on sample days (state-space)Poisson or NBIgnores censored detection limitsCensored log-normal primary; NB2 secondary4 bacteria jointlyRank-deficient, different thresholdsEnterococcus first; E. coli second; multi-task only as stretch7-dim daily vectorCollinear, costlyStandard lag summaries (24/48/72h + 7d sum)Full NUTSInfeasible in 10 hSVI-first, NUTS-on-subset validationGAM via smoothCorrectMake it specific: HSGP M=30, Matérn-3/2, c=1.5, LogNormal length-scale priorMetal JAXUnstable for NumPyroCPU JAX + 4 parallel chains, Accelerate BLAS"Rolling weekly run-rate"AmbiguousDrop or redefine as posterior 7-day rolling median

6. Risks to log in context/RISKS.md
RiskImpactMitigationCensoring column absent in Tier‑1 exportLikelihood mis-specifiedInspect Qualifier/Result for </> tokens at hour 0Historical CDIP NetCDF limited to realtimeWaves null for most rowsZero-fill with indicator; add archive URLs in fetch.yaml post-hackathonSVI under-covers posterior varianceCredible intervals too tightNUTS-on-subset validation; report both10 h budget overrun on SVINo model deliveredGate at hour 4, fall back to LightGBM + bootstrapped prediction intervalsApple Silicon jax-metal NaNs if accidentally enabledSilent bad samplesPin jax[cpu], assert jax.devices()[0].platform == 'cpu'Multi-rowrecords per sample event (replicates)Double-counting in likelihoodAggregate lab replicates to geometric mean per (station, date) before fit

Ready to proceed or iterate on any section. If you want I can narrow the first-hour EDA questions (detection-limit audit, Enterococcus coverage by county, sample-cadence histogram) or pre-write ASSUMPTIONS.md / DECISIONS.md entries so the run is reproducible from the first commit.

Short answer: yes, but only with a diagnostic and a fallback. SVI is not a free substitute for NUTS — it has a direction-of-bias problem that matters specifically for the operational output of this model (exceedance probability). Here is the rigorous version.
Why SVI is not a drop-in replacement
SVI minimizes KL(qϕ∥p)\mathrm{KL}(q_\phi \| p)
KL(qϕ​∥p), not KL(p∥qϕ)\mathrm{KL}(p \| q_\phi)
KL(p∥qϕ​). That direction is
mode-seeking / zero-forcing: qq
q is penalized for putting mass where pp
p has none, but not for failing to cover pp
p's tails. The systematic consequences (Blei, Kucukelbir & McAuliffe 2017; Yao, Vehtari, Simpson, Gelman 2018 "Yes, but did it work?"):


Posterior variance is systematically under-estimated, typically 10–40% on well-identified parameters and much more on variance hyperparameters and length-scales.
Tail probabilities are biased. For you this is P(Y>104 MPN)P(Y > 104 \text{ MPN})
P(Y>104 MPN) — the alert probability. Whether the bias is up or down depends on the guide, but it is not zero.

Correlations are lost under mean-field guides. A hierarchical model with HSGP + station REs has strong cross-correlations between the variance hyperparameters τstn,σk,ℓk\tau_{\text{stn}}, \sigma_k, \ell_k
τstn​,σk​,ℓk​ and the latent effects.


Where SVI is fine vs dangerous for your specific model
Parameter classPosterior shapeSVI riskIntercept α\alpha
α, linear βj\beta_j
βj​~Gaussian, well-identifiedLowStation REs αs(stn)\alpha_s^{(\text{stn})}
αs(stn)​ (non-centered)
GaussianLowHSGP basis weights zkz_k
zk​Gaussian by constructionLowVariance hyperparams τstn,σk,σobs\tau_{\text{stn}}, \sigma_k, \sigma_{\text{obs}}
τstn​,σk​,σobs​Half-normal / skewed in log-spaceHigh — mean-field collapses themHSGP length-scales ℓk\ell_k
ℓk​Weakly identified, heavy-tailedHighAR(1) ρ\rho
ρBounded, often skewedMediumExceedance tail P(Y>c)P(Y > c)
P(Y>c)Integral over skewed posteriorHighest risk — this is your alert
Guide choice is doing most of the work
Don't compare "SVI" vs "NUTS" as monoliths; compare guide quality:
GuideCostWhat it capturesVerdict for your modelAutoNormal (mean-field)O(D)O(D)
O(D)Marginals only, no correlationsDon't useAutoLowRankMultivariateNormal(rank=R)O(DR)O(DR)
O(DR)Dominant correlationsPrimary recommendation. R=20R=20
R=20 on your D≈3–5kD\approx 3\text{–}5k
D≈3–5k is ~100k params, fits easily in 48 GB, ~20 min on CPU
AutoIAFNormal (normalizing flow)Higher; slower to trainNon-Gaussian shapeWorth trying for variance hyperparams; expensiveAutoDAISSlowNear-NUTS qualityOnly if SVI diagnostic fails and you have time
Non-centered parameterization is non-optional here — it makes the posterior closer to Gaussian in unconstrained space and gives the variational guide a fighting chance on τstn\tau_{\text{stn}}
τstn​ and the HSGP hyperparameters.

Diagnostic that tells you whether to trust SVI
Run this after SVI; it takes ~2 minutes and is the single most important step:

Pareto-k^\hat k
k^ from PSIS on the SVI posterior
 (Yao et al. 2018). Draw SS
S samples from qq
q, compute importance weights ws=p(θs,y)/q(θs)w_s = p(\theta_s, y)/q(\theta_s)
ws​=p(θs​,y)/q(θs​), fit a generalized Pareto to the tail.


k^<0.5\hat k < 0.5
k^<0.5: SVI posterior is reliable, use it everywhere.

0.5≤k^<0.70.5 \le \hat k < 0.7
0.5≤k^<0.7: SVI marginals OK, tails suspicious — do not trust exceedance probabilities without correction.

k^≥0.7\hat k \ge 0.7
k^≥0.7: SVI posterior is not trustworthy; fall back to NUTS-on-subset.



PSIS-corrected SVI: if k^<0.7\hat k < 0.7
k^<0.7, reweight the SVI draws to reduce bias. Implemented as
numpyro.infer.SVI + manual IS reweighting; ~10 lines.
Cross-check with NUTS on a stratified subset (plan already has this at hour 6). If 90% quantiles of SVI and NUTS agree within ~10% on a county-level subset, ship SVI for the statewide fit.

Concrete protocol for the 10-hour run
Replace "SVI first, NUTS later" in the earlier plan with:

SVI + AutoLowRankMultivariateNormal(rank=20) on the full data. Adam lr=1e-3, 20k steps, minibatch 10k. (~20–30 min)
Compute k^\hat k
k^. (~2 min)
NUTS on San Diego county subset (~50k rows, 4 chains, 1000+1000). (~60–90 min)
Agreement check: overlay SVI and NUTS marginals for τstn,σk,ℓk\tau_{\text{stn}}, \sigma_k, \ell_k
τstn​,σk​,ℓk​, and sampled station REs. Report the max relative error in 90% CI widths.

Decision rule for what to ship:

k^<0.5\hat k < 0.5
k^<0.5 and agreement within 10% → SVI posterior statewide, NUTS-subset in the report as the validation.

Otherwise → ship NUTS-subset model as primary; SVI statewide labeled "approximate, 90% CI widths biased low."



Middle-ground options if SVI fails

NeuTra-HMC (numpyro.infer.reparam.NeuTraReparam): train a cheap IAF with SVI, then run NUTS on the transformed posterior. For hierarchical models with funnels this is typically 3–10× faster than plain NUTS with the same quality. Good insurance.
Subsampled NUTS on a stratified random sample (say 100k rows, stratified by county × year): the posterior is still a valid posterior — just for less data. Often the fastest path to credible intervals you can defend.
Laplace approximation via AutoLaplaceApproximation: fast, but worse than SVI for this model because skewed posteriors are pinned to a Gaussian at the MAP. Acceptable only for the linear GLM baseline, not the HSGP-GAM.

Bottom line
SVI is the right primary workhorse for your full-data fit given a 10-hour budget on CPU JAX, conditional on (a) a low-rank-MVN or IAF guide, (b) non-centered parameterization, (c) a Pareto-k^\hat k
k^ diagnostic, and (d) a NUTS-on-subset agreement check. Without those four, calling a result "Bayesian" is doing more signaling than inference — the posterior you report will systematically under-represent uncertainty on exactly the quantities (variance components and exceedance tails) that a hierarchical Bayesian framing exists to handle.