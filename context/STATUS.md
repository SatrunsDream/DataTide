# STATUS.md

Current state: what is done, what is next, blockers.

## Done

- Repository scaffold aligned with `context/rules_templet.md`; notebooks under `notebooks/eda/`, `modeling/`, `reporting/`.
- Strategic research in `resources/deep-research-report.md`; **tiered source stack** in `context/DATASETS.md`.
- **Fetch pipeline:** `configs/fetch.yaml` + `scripts/fetch/*.py` + `run_all.py` → per-source folders under `data/raw/` (default window **2010-01-01 → 2025-12-31**).
- **Processed ground truth:** `scripts/process/build_ground_truth_dataset.py` → `data/processed/datatide_ground_truth.parquet` + meta JSON; full schema in `context/GROUND_TRUTH_SCHEMA.md`.
- **Documentation:** `context/README.md` (replication, timing), `context/plan.md` (Bayesian modeling plan + critique), `context/DECISIONS.md` / `ASSUMPTIONS.md` / `RISKS.md` populated from plan adoption.
- **`data/` gitignored** (except `.gitkeep`); data not pushed to GitHub.

## Next (modeling sprint — see `context/plan.md`)

1. **Hour 0–0.5:** Filter **Enterococcus**; build `log_result`, censoring flags; confirm columns; refresh `ASSUMPTIONS.md` if export differs.
2. **Hour 0.5–2:** EDA + **baselines** (persistence, DoY climatology, logistic exceedance, LightGBM); time-blocked CV; `artifacts/tables/baseline_metrics.csv`.
3. **Hour 2–3.5:** NumPyro **v1** hierarchical GLM, **censored log-normal**, non-centered REs; prior predictive; ArviZ `InferenceData`.
4. **Hour 3.5–6:** **v2** add **HSGP** smooths (rain lags, tide range, waves, salinity); **SVI** full/minibatch; Pareto-\(\hat k\).
5. **Hour 6–7.5:** **NUTS** on subset (e.g. San Diego); \(\hat R\), ESS; compare to SVI marginals.
6. **Hour 7.5–8.5:** Posterior predictive on holdout; CRPS, coverage, exceedance ROC vs LightGBM.
7. **Hour 8.5–9.5:** **Power BI** Parquet star schema under `artifacts/data/powerbi/` (`INTERFACES.md`); calibration figures.
8. **Hour 9.5–10:** Run bundle in `artifacts/logs/runs/`; update `CHANGELOG.md`, `structure.md`.

## Blockers

- *(none listed — add NOAA token, portal URL, or auth issues here when they appear)*
