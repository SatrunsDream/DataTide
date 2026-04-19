"""
Regenerate `notebooks/modeling/predict.ipynb` from this single file.

`predict.ipynb` is the scheduled-forecast companion to `model.ipynb`:

    model.ipynb    \u2014 trains + selects + evaluates + saves a winner bundle
    predict.ipynb  \u2014 **only** loads the saved winner and writes a new week of
                     forecasts. Runs in ~30-60 s on CPU because there is no
                     MCMC \u2014 just `Predictive` on the val-side design.

Run:
    python scripts/build_predict_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/modeling/predict.ipynb")


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.lstrip("\n").splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.lstrip("\n").splitlines(keepends=True),
    }


cells: list[dict] = []

# -----------------------------------------------------------------------------
# \u00a70  intro
# -----------------------------------------------------------------------------

cells.append(md(r"""
# Predict \u2014 reload the winner model and forecast the next week

Scheduled-forecast companion to `model.ipynb`. There is **no MCMC here** \u2014
we load the posterior parameter samples saved by `model.ipynb \u00a77` and push
them through the validation-side of the model on a synthetic future fold.

Inputs (all written by `model.ipynb`):

- `artifacts/modeling/winner_model.npz`      \u2014 posterior parameter samples
- `artifacts/modeling/winner_run_meta.json`  \u2014 rung, NUTS config, provenance

Outputs (written by this notebook):

- `artifacts/modeling/winner_next_week_forecast.parquet` \u2014 **tidy, Power-BI ready**
- `artifacts/modeling/winner_next_week_samples.npz`      \u2014 raw `(S, N_future)` log10 draws
- `artifacts/modeling/winner_next_week_index.parquet`    \u2014 `row_idx \u2192 station/date`
- `artifacts/modeling/winner_next_week_meta.json`        \u2014 manifest

Switch scenarios by editing `HORIZON_DAYS` / `START_DATE` / `WEATHER_OVERRIDE`
below. Typical weather = monthly climatology from pre-test rows (the default).
"""))

# -----------------------------------------------------------------------------
# \u00a70  setup
# -----------------------------------------------------------------------------

cells.append(md(r"""
## \u00a70. Setup
"""))

cells.append(code(r"""
import os, warnings, json, time
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
# Prediction is cheap \u2014 stay on CPU so this runs anywhere.
os.environ["JAX_PLATFORMS"] = os.environ.get("FORCE_JAX_PLATFORM", "cpu")
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
PROJ = Path.cwd().resolve()
while not (PROJ / "src").exists() and PROJ != PROJ.parent:
    PROJ = PROJ.parent
sys.path.insert(0, str(PROJ))

from src.modeling.cv         import load_panel
from src.modeling.inference  import BayesianRung
from src.modeling.production import (
    build_future_fold, compute_climatology, export_forecast_bundle,
    read_forecast_frame,
)

plt.rcParams.update({"figure.dpi": 100, "font.size": 9})
ARTIFACT_DIR = PROJ / "artifacts" / "modeling"
print(f"project root : {PROJ}")
print(f"artifact dir : {ARTIFACT_DIR}")
"""))

# -----------------------------------------------------------------------------
# \u00a71  reload
# -----------------------------------------------------------------------------

cells.append(md(r"""
## \u00a71. Reload the winner \u2014 no refit

`BayesianRung.from_saved(...)` rebuilds the numpyro model closure and plugs
the saved posterior parameter samples back in, so `.predict(fold)` is
immediately available. If you get a `FileNotFoundError` here, re-run
`model.ipynb \u00a76\u2013\u00a77` first.
"""))

cells.append(code(r"""
with open(ARTIFACT_DIR / "winner_run_meta.json") as fh:
    run_meta = json.load(fh)
print("winner        :", run_meta["winner"])
print("rung          :", run_meta["rung"])
print("is_bayesian   :", run_meta["is_bayesian"])
print("test counts-MAE  :", run_meta["test_score"].get("counts-MAE"))

if not run_meta["is_bayesian"]:
    raise RuntimeError(
        "The selected winner is a baseline \u2014 refit it in model.ipynb \u00a76 and "
        "call .predict() directly on the future fold (this predict.ipynb only "
        "knows how to reload a Bayesian rung)."
    )

winner_path = ARTIFACT_DIR / run_meta["artifacts"]["model_bundle"]
t0 = time.time()
final_model = BayesianRung.from_saved(winner_path)
print(f"reloaded winner in {time.time()-t0:.2f}s \u2014 {final_model.name}")
print(f"posterior sample keys: {list(final_model._posterior_samples.keys())}")
n_draws = next(iter(final_model._posterior_samples.values())).shape[0]
print(f"total posterior draws (chains merged): {n_draws}")
"""))

# -----------------------------------------------------------------------------
# \u00a72  future fold
# -----------------------------------------------------------------------------

cells.append(md(r"""
## \u00a72. Build the next-week fold

The future fold has one row per (station, day). Date-derived features
(month, doy) come from the dates themselves; weather lag features come from
**monthly climatology** computed on pre-test rows. Override any feature by
editing `WEATHER_OVERRIDE` below.
"""))

cells.append(code(r"""
HORIZON_DAYS    = 7
START_DATE      = None            # None = day after last panel date; set to "YYYY-MM-DD" to force
WEATHER_OVERRIDE: dict[str, np.ndarray] | None = None   # e.g. {"rain_24h_mm": np.full(N, 2.0)}

bundle = load_panel()
print(f"panel: n={len(bundle.y_log):,}  n_stations={bundle.n_stations}  "
      f"last date={str(np.datetime64(bundle.date_min) + np.timedelta64(int(bundle.t_idx.max()), 'D'))[:10]}")

clim = compute_climatology(bundle, use_rows=(bundle.cv_val_year >= 0))
future_fold, future_index = build_future_fold(
    bundle,
    start_date=START_DATE,
    horizon_days=HORIZON_DAYS,
    climatology=clim,
    weather_override=WEATHER_OVERRIDE,
)
print(f"future fold: {len(future_index):,} rows  "
      f"({future_index['date'].min().date()} \u2192 {future_index['date'].max().date()})")
future_index.head()
"""))

# -----------------------------------------------------------------------------
# \u00a73  predict
# -----------------------------------------------------------------------------

cells.append(md(r"""
## \u00a73. Posterior-predictive draws for the future fold

`final_model.predict(future_fold)` vectorises over every posterior parameter
draw and every future row, yielding a `(S, N_future)` matrix of log10 MPN
samples. The point forecast is `median(10**samples)` per row; intervals are
quantiles of `10**samples`.
"""))

cells.append(code(r"""
t0 = time.time()
future_pred = final_model.predict(future_fold)
print(f"predicted in {time.time()-t0:.1f}s   samples shape = {future_pred.samples_log10.shape}")
"""))

# -----------------------------------------------------------------------------
# \u00a74  export
# -----------------------------------------------------------------------------

cells.append(md(r"""
## \u00a74. Export the teammate bundle

One call writes the four files the dashboards consume:

| file | purpose |
|---|---|
| `winner_next_week_forecast.parquet` | tidy median / PI / exceedance per (station, date) \u2014 Power BI |
| `winner_next_week_samples.npz`      | raw `(S, N_future)` log10 draws \u2014 internal web |
| `winner_next_week_index.parquet`    | `row_idx \u2192 (station, date, county)` lookup |
| `winner_next_week_meta.json`        | shapes + dates + provenance |
"""))

cells.append(code(r"""
written = export_forecast_bundle(
    ARTIFACT_DIR,
    tag="winner_next_week",
    index_df=future_index,
    samples_log10=future_pred.samples_log10,
    meta_extra={
        "winner":            run_meta["winner"],
        "rung":              run_meta["rung"],
        "horizon_days":      HORIZON_DAYS,
        "weather_strategy":  ("monthly climatology (pre-2024)" if WEATHER_OVERRIDE is None
                              else f"climatology + override keys={list(WEATHER_OVERRIDE)}"),
        "start_date":        str(future_index['date'].min().date()),
    },
)
for k, p in written.items():
    print(f"wrote {k:<8s}  \u2192 {p}")
"""))

# -----------------------------------------------------------------------------
# \u00a75  preview
# -----------------------------------------------------------------------------

cells.append(md(r"""
## \u00a75. Preview \u2014 tidy table + advisory heatmap

Same view the dashboard will render. Counties are sorted by earliest-day
exceedance risk (highest at top) so the most actionable advisory stands out.
"""))

cells.append(code(r"""
tidy = read_forecast_frame(written["forecast"])
display(tidy.head(15))

heat = (
    tidy.assign(date=tidy["date"].dt.date)
        .groupby(["county_name", "date"])["p_exceed_104mpn"]
        .mean()
        .unstack("date")
)
heat = heat.sort_values(by=heat.columns[0], ascending=False)

fig, ax = plt.subplots(figsize=(1 + 0.8 * HORIZON_DAYS, 0.35 * len(heat) + 1.2))
im = ax.imshow(heat.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(heat.shape[1]))
ax.set_xticklabels([str(d) for d in heat.columns], rotation=35, ha='right')
ax.set_yticks(range(heat.shape[0]))
ax.set_yticklabels(heat.index)
ax.set_title(f"next {HORIZON_DAYS}-day advisory risk   P(single-sample > 104 MPN)  by county")
plt.colorbar(im, ax=ax, label='exceedance prob', fraction=0.03, pad=0.02)
plt.tight_layout(); plt.show()
"""))

cells.append(code(r"""
# Station-level view: top 20 stations by mean next-week exceedance risk.
top20 = (
    tidy.groupby(["station_id", "station_name", "county_name"])["p_exceed_104mpn"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
        .rename(columns={"p_exceed_104mpn": "mean_p_exceed_next_week"})
)
display(top20)
"""))

cells.append(code(r"""
# Sanity: quick residual-free trace-like plot of one station's predictive
# distribution over the week (median + 50/80/95 PI, counts scale, log y-axis).
stn_idx = int(top20.iloc[0]["station_id"])      # highest-risk station id
rows = tidy[tidy["station_id"] == stn_idx].sort_values("date")
if len(rows) > 0:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = pd.to_datetime(rows["date"])
    ax.fill_between(x, rows["pi95_low_mpn"], rows["pi95_high_mpn"], alpha=0.12, color='C0', label='95% PI')
    ax.fill_between(x, rows["pi80_low_mpn"], rows["pi80_high_mpn"], alpha=0.2,  color='C0', label='80% PI')
    ax.fill_between(x, rows["pi50_low_mpn"], rows["pi50_high_mpn"], alpha=0.35, color='C0', label='50% PI')
    ax.plot(x, rows["point_mpn_median"], 'o-', color='C0', label='median')
    ax.axhline(104, ls='--', color='crimson', lw=0.8, label='104 MPN advisory')
    ax.set_yscale('log')
    ax.set_ylabel("MPN / 100 mL")
    ax.set_title(f"next-week forecast  \u2014  station_id {stn_idx}  ({rows.iloc[0]['station_name']})")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
"""))

# -----------------------------------------------------------------------------
# footer
# -----------------------------------------------------------------------------

cells.append(md(r"""
## Next steps

- Schedule this notebook on cron (or a CI job) so a fresh bundle is written
  once a week. Nothing about it requires an interactive kernel.
- If forecasts drift from reality for >N weeks, re-run `model.ipynb` end-to-end
  so the panel, priors, and posterior get refreshed.
- For a **storm** scenario, set e.g.
  `WEATHER_OVERRIDE = {"rain_24h_mm": np.full(n_future, 25.0)}` at the top of
  \u00a72 and re-run \u00a72\u2013\u00a75. The exported artifacts overwrite the "typical" bundle
  \u2014 rename the output tag if you want to keep both.
"""))


# -----------------------------------------------------------------------------
# assemble notebook
# -----------------------------------------------------------------------------

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "nbconvert_exporter": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NOTEBOOK_PATH, "w") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote {NOTEBOOK_PATH}  ({len(cells)} cells)")
