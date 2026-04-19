"""
Build the monolithic modeling notebook programmatically.

Run `python scripts/build_model_notebook.py` to (re-)materialise
`notebooks/modeling/model.ipynb`. The notebook is structured per
`context/MODELING_PLAN.md`:

    \u00a70  Setup and configuration
    \u00a71  Panel load + CV folds + priors
    \u00a72  Phase B baselines (naive seasonal, OLS log10, XGBoost log10)
    \u00a73  Phase A Bayesian ladder v0..v6 (with prior-predictive checks)
    \u00a74  Joint comparison table (fold-averaged, per-model)
    \u00a75  Winner diagnostics (trace, forest, PPC, reliability, PIT)
    \u00a76  Winner retrain on train+val and test-set evaluation (2024)
    \u00a77  Artifact export for productionisation
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


OUT = Path("notebooks/modeling/model.ipynb")


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src.strip("\n"))


cells: list[dict] = []

# ---------------------------------------------------------------------------
# \u00a70  Setup
# ---------------------------------------------------------------------------

cells.append(md(r"""
# Modeling \u2014 baselines + Bayesian ladder + winner + test-set evaluation

This notebook executes the plan in `context/MODELING_PLAN.md`. Every model
\u2014 three non-Bayesian baselines and eight Bayesian ladder rungs \u2014 implements
the same `Forecaster` protocol defined in `src/evaluation/compare.py`, so the
comparison machinery is uniform end-to-end.

Primary selection metric: **counts-MAE** (fold-averaged). Secondary: counts-MedAE,
coverage at 50/80/95%, Brier, ECE. Plan explicitly excludes log-MAE, CRPS-log,
and fold-error as ranking criteria.
"""))

cells.append(md(r"""
## \u00a70. Setup

- JAX on CPU, deterministic.
- All data loaded from `artifacts/data/panel/` (built by the EDA notebook).
- NUTS defaults are intentionally moderate (`num_warmup=300`, `num_samples=300`,
  `num_chains=1`) for Phase-A development speed. The winner is re-fit with
  production-grade settings (`800 / 800 / 2`) in \u00a76.
"""))

cells.append(code(r"""
import os, warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# project imports
import sys
PROJ = Path.cwd().resolve()
while not (PROJ / "src").exists() and PROJ != PROJ.parent:
    PROJ = PROJ.parent
sys.path.insert(0, str(PROJ))

from src.modeling.cv        import load_panel, iter_folds
from src.modeling.baselines import NaiveSeasonalMeanCounts, OLSLog10, XGBoostLog10
from src.modeling.bayesian  import Rung, RUNG_LABELS
from src.modeling.inference import BayesianRung, NutsConfig
from src.evaluation.compare import score_model_on_fold, fold_scores_to_table, average_over_folds
from src.evaluation.metrics import EXCEEDANCE_LOG10, EXCEEDANCE_MPN
from src.evaluation.calibration import (
    reliability_curve, pit_values, plot_reliability, plot_pit,
)

plt.rcParams.update({"figure.dpi": 100, "font.size": 9})
print(f"project root: {PROJ}")
"""))

cells.append(code(r"""
# -------- user-editable configuration --------
DEV_FOLDS     = [2022, 2023]                # quick Phase-A ladder folds
ALL_FOLDS     = [2020, 2021, 2022, 2023]    # full fold set for Phase-B baselines
TEST_YEARS    = [2024]                      # held-out test window
NUTS_DEV      = NutsConfig(num_warmup=300, num_samples=300, num_chains=1, progress_bar=False)
NUTS_PROD     = NutsConfig(num_warmup=800, num_samples=800, num_chains=2, progress_bar=False)
ARTIFACT_DIR  = Path("artifacts/modeling")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
print(f"dev folds {DEV_FOLDS}  |  all folds {ALL_FOLDS}  |  test {TEST_YEARS}")
"""))

# ---------------------------------------------------------------------------
# \u00a71  Load data
# ---------------------------------------------------------------------------

cells.append(md(r"""
## \u00a71. Panel load + CV folds + priors

Loads the observed subset of the processed panel (11.6K rows), the CV schema
(`cv_val_year` \u2208 {0, 2020, 2021, 2022, 2023, \u22121} meaning *always-train*,
*fold-k*, *test*), and the empirical priors elicited in the EDA.
"""))

cells.append(code(r"""
t0 = time.time()
bundle = load_panel()
print(f"loaded panel in {time.time()-t0:.1f}s:")
print(f"  n_observations = {len(bundle.y_log):,}")
print(f"  n_stations     = {bundle.n_stations}  (with name: {bundle.meta.get('n_stations_with_name')})")
print(f"  n_counties     = {bundle.n_counties}")
print(f"  date range     = {bundle.meta['date_min']} \u2192 {bundle.meta['date_max']}")
print(f"  train_end / val_end = {bundle.meta['train_end']} / {bundle.meta['val_end']}")
print(f"  censoring      = left {bundle.left_mask.sum()}  right {bundle.right_mask.sum()}  interior {(~(bundle.left_mask|bundle.right_mask)).sum()}")
print(f"  empirical priors: {list(bundle.priors.keys())}")
"""))

cells.append(code(r"""
# Build CV folds once; every model re-uses these slices.
dev_folds = {vy: f for vy, f in zip(DEV_FOLDS, iter_folds(bundle, val_years=DEV_FOLDS))}
all_folds = {vy: f for vy, f in zip(ALL_FOLDS, iter_folds(bundle, val_years=ALL_FOLDS))}

print(f"{'fold':>6}  {'n_train':>8}  {'n_val':>6}")
for vy, f in all_folds.items():
    print(f"{vy:>6}  {f.n_train:>8}  {f.n_val:>6}")
"""))

# ---------------------------------------------------------------------------
# \u00a72  Phase B baselines
# ---------------------------------------------------------------------------

cells.append(md(r"""
## \u00a72. Phase B \u2014 baseline comparison

Three honest non-Bayesian baselines, all reporting the full metric suite:

| id | name | structure | predictive distribution |
|----|------|-----------|--------------------------|
| B1 | `NaiveSeasonalMeanCounts` | per-month empirical log10 bootstrap | non-parametric |
| B2 | `OLSLog10` | OLS on smooth+linear+month one-hots | Normal(\u03bc, \u03c3_resid) |
| B3 | `XGBoostLog10` | gradient-boosted trees on the same features | Normal(\u03bc, \u03c3_resid) |

The OLS predictive distribution is not the Bayesian posterior predictive \u2014
it's the training-residual Normal approximation, which is the honest
apples-to-apples retrofit for probabilistic metrics (see `MODELING_PLAN \u00a76`).
"""))

cells.append(code(r"""
baseline_rows = []
for vy, fold in all_folds.items():
    for M in (NaiveSeasonalMeanCounts, OLSLog10, XGBoostLog10):
        m = M()
        t1 = time.time()
        s = score_model_on_fold(m, fold)
        baseline_rows.append(s)
        print(f"  fold {vy}  {m.name:34s}  "
              f"dt={time.time()-t1:5.1f}s  MAE={s.counts_mae:>8.1f}  "
              f"cov50={s.coverage['cov-50']:.2f}  Brier={s.brier:.3f}  ECE={s.ece:.3f}")

baseline_per_fold = fold_scores_to_table(baseline_rows)
baseline_summary  = average_over_folds(baseline_per_fold)
baseline_summary[['model', 'counts-MAE', 'counts-MedAE', 'Brier', 'ECE', 'cov-50', 'cov-80', 'cov-95']]
"""))

# ---------------------------------------------------------------------------
# \u00a73  Bayesian ladder
# ---------------------------------------------------------------------------

cells.append(md(r"""
## \u00a73. Phase A \u2014 Bayesian ladder v0 \u2192 v6

Each rung adds exactly one structural element, building on the seasonal
scaffold the EDA justified (see `MODELING_PLAN \u00a75`):

| rung | what it adds | seasonal parametrisation |
|------|--------------|---------------------------|
| v0   | pooled intercept | none |
| v1   | + `alpha_month[m]` scaffold | categorical |
| v2   | + station/county hierarchy (non-centred) | categorical |
| v3   | + linear weather covariates | categorical |
| v4   | polynomial(doy, deg=3) **replaces** month | polynomial |
| v5   | natural cubic spline(doy, 6 knots) **replaces** polynomial | spline |
| v6   | periodic Fourier HSGP(doy) + shared-amplitude rain slopes | HSGP |

Censored log-normal likelihood on log10 is identical across rungs \u2014 any metric
change is attributable to `mu_it` structure alone.

For Phase-A development we run on two folds (2022, 2023) with short chains
(300 warmup, 300 samples, 1 chain). The winner is re-fit with production
settings (2 chains, 800/800) in \u00a76.
"""))

cells.append(code(r"""
# Prior predictive check on v0 and v6  \u2014  are implied y_log draws plausible?
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
rungs_to_check = [Rung.v0, Rung.v6]
for ax, rung in zip(axes, rungs_to_check):
    m = BayesianRung(rung=rung, priors=bundle.priors, nuts=NUTS_DEV)
    prior_y = m.prior_predictive(dev_folds[DEV_FOLDS[-1]], num_samples=400)
    ax.hist(prior_y.ravel(), bins=80, density=True, color='steelblue', alpha=0.6, label='prior predictive')
    ax.hist(bundle.y_log, bins=80, density=True, color='crimson', alpha=0.4, label='observed y_log')
    ax.set_title(f"{rung.value}: prior-predictive vs observed (log10)")
    ax.legend(fontsize=8)
    ax.set_xlabel("log10 MPN")
plt.tight_layout()
plt.show()
"""))

cells.append(code(r"""
ladder_rows = []
fit_cache = {}           # fold_val_year \u2192 { rung : BayesianRung(fitted) }
for vy in DEV_FOLDS:
    fold = dev_folds[vy]
    fit_cache[vy] = {}
    for rung in list(Rung):
        m = BayesianRung(rung=rung, priors=bundle.priors, nuts=NUTS_DEV)
        t1 = time.time()
        try:
            s = score_model_on_fold(m, fold)
            fit_cache[vy][rung] = m
            ladder_rows.append(s)
            print(f"  fold {vy}  {m.name:34s}  dt={time.time()-t1:5.1f}s  "
                  f"MAE={s.counts_mae:>8.1f}  MedAE={s.counts_medae:>6.1f}  "
                  f"Brier={s.brier:.3f}  ECE={s.ece:.3f}")
        except Exception as e:
            print(f"  fold {vy}  {m.name:34s}  FAILED: {e}")

ladder_per_fold = fold_scores_to_table(ladder_rows)
ladder_summary  = average_over_folds(ladder_per_fold)
ladder_summary[['model', 'counts-MAE', 'counts-MedAE', 'Brier', 'ECE', 'cov-50', 'cov-80', 'cov-95']]
"""))

cells.append(code(r"""
# Ladder gain plot  \u2014  counts-MAE and Brier as a function of rung
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
per_model = (
    ladder_per_fold.groupby('model', sort=False)
    .agg(mae_mean=('counts-MAE', 'mean'), mae_std=('counts-MAE', 'std'),
         brier_mean=('Brier', 'mean'), brier_std=('Brier', 'std'),
         ece_mean=('ECE', 'mean'))
    .reset_index()
)
x = np.arange(len(per_model))
axes[0].errorbar(x, per_model['mae_mean'], yerr=per_model['mae_std'], fmt='o-', color='C0')
axes[0].set_xticks(x); axes[0].set_xticklabels([m.split(' ', 1)[0] for m in per_model['model']], rotation=0)
axes[0].set_ylabel("counts-MAE (fold-averaged)"); axes[0].set_title("Ladder \u2014 point accuracy in MPN")
axes[0].grid(True, alpha=0.3)

axes[1].errorbar(x, per_model['brier_mean'], yerr=per_model['brier_std'], fmt='o-', color='C2', label='Brier')
axes[1].plot(x, per_model['ece_mean'], 's--', color='C3', label='ECE')
axes[1].set_xticks(x); axes[1].set_xticklabels([m.split(' ', 1)[0] for m in per_model['model']], rotation=0)
axes[1].set_ylabel("exceedance prob. metric"); axes[1].set_title("Ladder \u2014 calibration against 104 MPN")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
"""))

# ---------------------------------------------------------------------------
# \u00a74  Joint comparison table
# ---------------------------------------------------------------------------

cells.append(md(r"""
## \u00a74. Joint comparison \u2014 baselines \u222a ladder

Both phases on their common fold set (2022, 2023 for ladder; all four for
baselines). The selection metric is fold-averaged `counts-MAE` among rungs
with acceptable calibration (the EDA recommended cov-50 within 5 points of
0.50, cov-95 within 5 points of 0.95).
"""))

cells.append(code(r"""
# Restrict baselines to the dev-fold set for apples-to-apples ladder comparison
bl_dev = baseline_per_fold[baseline_per_fold['fold_val_year'].isin(DEV_FOLDS)].copy()
joint_per_fold = pd.concat([bl_dev, ladder_per_fold], ignore_index=True)
joint_summary  = average_over_folds(joint_per_fold)

display_cols = ['model', 'counts-MAE', 'counts-MedAE', 'Brier', 'ECE', 'cov-50', 'cov-80', 'cov-95']
joint_summary[display_cols]
"""))

cells.append(code(r"""
# Winner pick: min fold-averaged counts-MAE among models meeting calibration floors
CAL_BOUNDS = {'cov-50': (0.40, 0.60), 'cov-95': (0.85, 1.00)}

def meets_calibration(row):
    for k, (lo, hi) in CAL_BOUNDS.items():
        if not (lo <= row[k + '__mean'] <= hi):
            return False
    return True

eligible = joint_summary[joint_summary.apply(meets_calibration, axis=1)].copy()
eligible = eligible.sort_values('counts-MAE__mean')
print(f"eligible models ({len(eligible)}):")
for _, r in eligible.iterrows():
    print(f"  {r['model']:40s}  MAE={r['counts-MAE__mean']:>8.1f} \u00b1 {r['counts-MAE__std']:>7.1f}  "
          f"Brier={r['Brier__mean']:.3f}  ECE={r['ECE__mean']:.3f}  "
          f"cov50={r['cov-50__mean']:.2f}  cov95={r['cov-95__mean']:.2f}")

winner_name = eligible.iloc[0]['model']
print(f"\nwinner: {winner_name}")
"""))

# ---------------------------------------------------------------------------
# \u00a75  Winner diagnostics
# ---------------------------------------------------------------------------

cells.append(md(r"""
## \u00a75. Winner diagnostics

For the selected winner \u2014 be it a baseline or a Bayesian rung \u2014 we run the
full diagnostic suite from ArviZ plus the plan's calibration plots:

- Trace + rank plots for top-level parameters
- Energy + divergences diagnostic
- Posterior summary table (mean \u00b1 HDI, ESS, R-hat) for priors
- Posterior predictive check against observed y_log
- Reliability diagram + PIT histogram

If the winner is a baseline (no posterior), skip the ArviZ diagnostics and
only show reliability + PIT.
"""))

cells.append(code(r"""
# Re-build the winner on the last dev fold so we have an in-memory reference
winner_rung = None
for r in Rung:
    if RUNG_LABELS[r] == winner_name:
        winner_rung = r; break

ref_fold_year = DEV_FOLDS[-1]
ref_fold = dev_folds[ref_fold_year]

is_bayesian = winner_rung is not None
if is_bayesian:
    winner = fit_cache[ref_fold_year][winner_rung]
    idata = winner.to_inferencedata()
    az_summary = az.summary(idata, var_names=[
        v for v in ['alpha_0', 'sigma_month', 'sigma_station', 'sigma_county',
                    'sigma_obs', 'hsgp_amp_doy', 'hsgp_amp_rain']
        if v in idata.posterior.data_vars
    ], round_to=3)
    print(az_summary)
else:
    print(f"winner {winner_name} is a baseline \u2014 skipping MCMC diagnostics")
"""))

cells.append(code(r"""
if is_bayesian:
    # posterior predictive samples on the validation fold for plotting
    pred = winner.predict(ref_fold)
    samples = pred.samples_log10
    y_true  = ref_fold.y_log_val

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    # PPC
    axes[0].hist(y_true, bins=60, density=True, color='crimson', alpha=0.45, label='observed')
    axes[0].hist(samples[:30].ravel(), bins=60, density=True, color='steelblue',
                 alpha=0.40, label='posterior predictive (30 draws)')
    axes[0].legend(fontsize=8); axes[0].set_xlabel("log10 MPN")
    axes[0].set_title(f"{winner_name}: PPC on val {ref_fold_year}")

    # Reliability
    from src.evaluation.metrics import exceedance_prob_from_samples
    exc = exceedance_prob_from_samples(samples)
    rc = reliability_curve(y_true, exc)
    plot_reliability(axes[1], rc, label='posterior', color='steelblue',
                     show_base_rate=float((y_true > EXCEEDANCE_LOG10).mean()))
    axes[1].set_title(f"{winner_name}: reliability (exceedance)")
    axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.show()
else:
    print("(baseline winner \u2014 run \u00a76 first to get retrained predictions for diagnostics)")
"""))

cells.append(code(r"""
if is_bayesian:
    pit = pit_values(ref_fold.y_log_val, winner.predict(ref_fold).samples_log10)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    plot_pit(ax, pit, color='steelblue', label=f'{winner_name} PIT')
    ax.legend(fontsize=8); ax.set_title(f"PIT histogram \u2014 {winner_name}")
    plt.tight_layout(); plt.show()
"""))

# ---------------------------------------------------------------------------
# \u00a76  Retrain on train+val, evaluate on test (2024)
# ---------------------------------------------------------------------------

cells.append(md(r"""
## \u00a76. Retrain winner on train+val and evaluate on 2024 test

This is the plan's Phase-C step: take the selected model, re-fit it on the
*union* of training + all validation folds (pre-2024), and evaluate once on
the 2024 test year. The test fold has been untouched by CV.

Because 2024 rows are marked `cv_val_year == -1` in the panel, we build a
synthetic `FoldData` where every row `cv_val_year \u2265 0` is training and every
row `cv_val_year == -1` is test.
"""))

cells.append(code(r"""
from src.evaluation.compare import FoldData

def build_test_fold(b, test_year: int = 2024) -> FoldData:
    is_test  = (b.cv_val_year == -1)   # reserved test rows
    is_train = (b.cv_val_year >= 0)    # pre-test rows = train+val union
    return FoldData(
        fold_val_year=test_year,
        y_log_train=b.y_log[is_train],
        month_train=b.month[is_train],
        station_idx_train=b.station_idx[is_train],
        county_idx_train=b.county_idx[is_train],
        X_smooth_train=b.X_smooth[is_train],
        X_linear_train=b.X_linear[is_train],
        miss_smooth_train=b.miss_smooth[is_train],
        miss_linear_train=b.miss_linear[is_train],
        left_mask_train=b.left_mask[is_train],
        right_mask_train=b.right_mask[is_train],
        det_low_log_train=b.det_low_log[is_train],
        det_high_log_train=b.det_high_log[is_train],
        y_log_val=b.y_log[is_test],
        month_val=b.month[is_test],
        station_idx_val=b.station_idx[is_test],
        county_idx_val=b.county_idx[is_test],
        X_smooth_val=b.X_smooth[is_test],
        X_linear_val=b.X_linear[is_test],
        miss_smooth_val=b.miss_smooth[is_test],
        miss_linear_val=b.miss_linear[is_test],
        smooth_features=list(b.smooth_features),
        linear_features=list(b.linear_features),
        n_stations=b.n_stations,
        n_counties=b.n_counties,
    )

test_fold = build_test_fold(bundle)
print(f"test fold: n_train={test_fold.n_train:,}  n_test={test_fold.n_val:,}")
"""))

cells.append(code(r"""
# Retrain winner with production NUTS settings (or refit baseline).
if is_bayesian:
    final_model = BayesianRung(rung=winner_rung, priors=bundle.priors, nuts=NUTS_PROD)
else:
    final_model = {
        'B1 naive seasonal-mean (by month)': NaiveSeasonalMeanCounts,
        'B2 OLS log10':                       OLSLog10,
        'B3 XGBoost log10':                   XGBoostLog10,
    }[winner_name]()

t0 = time.time()
test_score = score_model_on_fold(final_model, test_fold)
print(f"test-set evaluation took {time.time()-t0:.1f}s")
print(test_score.as_flat_dict())
"""))

# ---------------------------------------------------------------------------
# \u00a77  Artifact export
# ---------------------------------------------------------------------------

cells.append(md(r"""
## \u00a77. Artifact export for productionisation

Everything a downstream teammate needs to build a dashboard on top of this
model. Written under `artifacts/modeling/`:

- `winner_predictions_test.parquet` \u2014 point forecast (MPN median), 50/80/95 %
  predictive intervals (MPN), exceedance probability, station / county / date
  per test row. This is the file a Power BI analyst consumes.
- `winner_posterior_samples_test.npz` \u2014 raw `(S, N_test)` log10 posterior
  samples for any custom aggregation the internal website needs.
- `winner_inferencedata.nc` \u2014 ArviZ NetCDF of the production fit (Bayesian
  winners only). Full trace, suitable for re-analysis.
- `winner_meta.json` \u2014 selected rung, NUTS config, priors used, metric suite.
"""))

cells.append(code(r"""
pred = final_model.predict(test_fold)
samples_log10 = pred.samples_log10

q_lo50, q_hi50 = np.quantile(samples_log10, [0.25, 0.75], axis=0)
q_lo80, q_hi80 = np.quantile(samples_log10, [0.10, 0.90], axis=0)
q_lo95, q_hi95 = np.quantile(samples_log10, [0.025, 0.975], axis=0)
exc_prob = (samples_log10 > EXCEEDANCE_LOG10).mean(axis=0)

out_rows = pd.DataFrame({
    "station_idx":  test_fold.station_idx_val,
    "station_id":   bundle.station_ids[test_fold.station_idx_val],
    "station_name": bundle.station_names[test_fold.station_idx_val],
    "county_idx":   test_fold.county_idx_val,
    "county_name":  bundle.county_names[test_fold.county_idx_val],
    "month":        test_fold.month_val,
    "point_mpn_median":   pred.point_mpn,
    "pi50_low_mpn":       10.0 ** q_lo50,
    "pi50_high_mpn":      10.0 ** q_hi50,
    "pi80_low_mpn":       10.0 ** q_lo80,
    "pi80_high_mpn":      10.0 ** q_hi80,
    "pi95_low_mpn":       10.0 ** q_lo95,
    "pi95_high_mpn":      10.0 ** q_hi95,
    "p_exceed_104mpn":    exc_prob,
})
pred_path = ARTIFACT_DIR / "winner_predictions_test.parquet"
try:
    out_rows.to_parquet(pred_path, index=False)
except Exception:
    pred_path = pred_path.with_suffix(".pkl")
    out_rows.to_pickle(pred_path)
print(f"wrote predictions  \u2192 {pred_path}  ({len(out_rows)} rows)")
out_rows.head()
"""))

cells.append(code(r"""
samples_path = ARTIFACT_DIR / "winner_posterior_samples_test.npz"
np.savez_compressed(samples_path, samples_log10=samples_log10)
print(f"wrote posterior samples (log10)  \u2192 {samples_path}  shape={samples_log10.shape}")

if is_bayesian:
    idata_path = ARTIFACT_DIR / "winner_inferencedata.nc"
    idata_prod = final_model.to_inferencedata()
    idata_prod.to_netcdf(idata_path)
    print(f"wrote InferenceData  \u2192 {idata_path}")

meta = {
    "winner": winner_name,
    "rung": (winner_rung.value if is_bayesian else None),
    "nuts": (NUTS_PROD.__dict__ if is_bayesian else None),
    "metric_suite": ["counts-MAE", "counts-MedAE", "Brier", "ECE",
                      "cov-50", "cov-80", "cov-95", "log-RMSE"],
    "test_score": test_score.as_flat_dict(),
    "dev_folds": DEV_FOLDS,
    "all_folds": ALL_FOLDS,
    "test_years": TEST_YEARS,
}
meta_path = ARTIFACT_DIR / "winner_meta.json"
with open(meta_path, "w") as fh:
    json.dump(meta, fh, indent=2, default=str)
print(f"wrote meta  \u2192 {meta_path}")
"""))

cells.append(md(r"""
### Next steps

- `notebooks/modeling/predict.ipynb` (not built yet) will consume these
  artifacts to produce a rolling forward-prediction grid over all 446 stations
  for the current operational window.
- The Power-BI / internal-web teammate reads `winner_predictions_test.parquet`
  directly; the raw posterior is available in `winner_posterior_samples_test.npz`
  for any custom aggregation they need.
"""))

# ---------------------------------------------------------------------------
# build the notebook
# ---------------------------------------------------------------------------

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
