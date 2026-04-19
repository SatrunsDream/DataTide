"""
Unified scorer and fold-averager.

All models in the comparison \u2014 baselines and Bayesian ladder rungs \u2014 implement
a common minimal protocol:

    model.fit(fold_data: FoldData) -> None
    model.predict(fold_data: FoldData) -> Prediction

where `Prediction` carries posterior-predictive samples on the log10 scale.
The scorer in this module wraps every metric into one call and returns a
per-fold dataframe plus a per-model fold-averaged summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .metrics import FoldScore, score_fold


@dataclass
class Prediction:
    """What every model returns from `.predict(fold_data)`."""

    samples_log10: np.ndarray   # (S, N_val)
    point_log10: np.ndarray     # (N_val,)  \u2014 used only for log-RMSE
    point_mpn: np.ndarray       # (N_val,)  \u2014 counts-MAE/MedAE point, must come from posterior median


@dataclass
class FoldData:
    """Container passed to `.fit` and `.predict`.

    Every model consumes the same fields, but not every model uses every field.
    The baselines use only `(y_log_train, month_train, month_val)`; the
    Bayesian models use the full covariate arrays.
    """

    fold_val_year: int

    # training arrays (all observed rows in the training window)
    y_log_train: np.ndarray
    month_train: np.ndarray
    station_idx_train: np.ndarray
    county_idx_train: np.ndarray
    X_smooth_train: np.ndarray
    X_linear_train: np.ndarray
    miss_smooth_train: np.ndarray
    miss_linear_train: np.ndarray
    left_mask_train: np.ndarray
    right_mask_train: np.ndarray
    det_low_log_train: np.ndarray
    det_high_log_train: np.ndarray

    # validation arrays (observed rows in the held-out year)
    y_log_val: np.ndarray
    month_val: np.ndarray
    station_idx_val: np.ndarray
    county_idx_val: np.ndarray
    X_smooth_val: np.ndarray
    X_linear_val: np.ndarray
    miss_smooth_val: np.ndarray
    miss_linear_val: np.ndarray
    doy_train: np.ndarray | None = None
    doy_val: np.ndarray | None = None

    # shared feature metadata (names + index lookups)
    smooth_features: list[str] = field(default_factory=list)
    linear_features: list[str] = field(default_factory=list)

    # shared global dims
    n_stations: int = 0
    n_counties: int = 0

    @property
    def n_train(self) -> int:
        return int(self.y_log_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.y_log_val.shape[0])


class Forecaster(Protocol):
    """Every baseline and Bayesian rung implements this two-method protocol."""

    name: str

    def fit(self, fold: FoldData) -> None: ...
    def predict(self, fold: FoldData) -> Prediction: ...


# ---------------------------------------------------------------------------
# fold scoring
# ---------------------------------------------------------------------------

def score_model_on_fold(model: Forecaster, fold: FoldData) -> FoldScore:
    """Fit + predict + score \u2014 every caller does this, so centralise it."""
    model.fit(fold)
    pred = model.predict(fold)
    return score_fold(
        model_name=model.name,
        fold_val_year=fold.fold_val_year,
        y_true_log10=fold.y_log_val,
        y_point_log10=pred.point_log10,
        y_point_mpn=pred.point_mpn,
        samples_log10=pred.samples_log10,
    )


def fold_scores_to_table(rows: list[FoldScore]) -> pd.DataFrame:
    """Per-fold table, one row per (model, fold)."""
    return pd.DataFrame([r.as_flat_dict() for r in rows])


def average_over_folds(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Fold-average every metric per model. Returns mean \u00b1 std cells."""
    metric_cols = [
        c for c in per_fold.columns
        if c not in ("model", "fold_val_year", "n_val")
    ]
    agg = (
        per_fold.groupby("model", sort=False)[metric_cols]
        .agg(["mean", "std", "count"])
    )

    out_cols: dict[str, list[float | str]] = {"model": list(agg.index)}
    for m in metric_cols:
        mean_s = agg[(m, "mean")].to_list()
        std_s = agg[(m, "std")].to_list()
        out_cols[m] = [f"{mu:.3f} \u00b1 {sd:.3f}" for mu, sd in zip(mean_s, std_s)]
        out_cols[f"{m}__mean"] = mean_s
        out_cols[f"{m}__std"] = std_s
    return pd.DataFrame(out_cols)
