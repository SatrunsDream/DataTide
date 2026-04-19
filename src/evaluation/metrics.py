"""
Metric suite for the DataTide modeling plan.

Every model in the comparison (baselines + Bayesian ladder) is evaluated by the
same functions in this module. Primary metric is counts-MAE; secondary metrics
are counts-MedAE, predictive-interval coverage at 50/80/95%, Brier score against
the 104-MPN advisory threshold, and Expected Calibration Error (ECE) on
exceedance probability.

Convention for model outputs:

- `y_true_log10` : np.ndarray shape (N,)  — observed log10(MPN/100 mL)
- `y_pred_log10_samples` : np.ndarray shape (S, N)  — posterior (or
   predictive) samples on the log10 scale. Baselines emit a large Normal draw
   approximation; Bayesian models emit actual posterior-predictive draws.
- `y_point_mpn` : np.ndarray shape (N,) — point prediction in MPN units.
   Choice of summary (median, mean) is the model's to make; for the selection
   metric we use the posterior median in count space (median of `10**samples`)
   since `10^mean(log10)` is the *geometric* mean (Jensen).

All fold-level scores are returned as floats; the fold-averager in
`src/evaluation/compare.py` aggregates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

EXCEEDANCE_MPN = 104.0
EXCEEDANCE_LOG10 = float(np.log10(EXCEEDANCE_MPN))


# ---------------------------------------------------------------------------
# point accuracy in count space
# ---------------------------------------------------------------------------

def counts_mae(y_true_mpn: np.ndarray, y_pred_mpn: np.ndarray) -> float:
    """Mean absolute error in MPN/100 mL units. PRIMARY selection metric."""
    y_true_mpn = np.asarray(y_true_mpn, dtype=float)
    y_pred_mpn = np.asarray(y_pred_mpn, dtype=float)
    return float(np.mean(np.abs(y_true_mpn - y_pred_mpn)))


def counts_medae(y_true_mpn: np.ndarray, y_pred_mpn: np.ndarray) -> float:
    """Median absolute error in MPN units. Robustness check alongside MAE."""
    y_true_mpn = np.asarray(y_true_mpn, dtype=float)
    y_pred_mpn = np.asarray(y_pred_mpn, dtype=float)
    return float(np.median(np.abs(y_true_mpn - y_pred_mpn)))


# ---------------------------------------------------------------------------
# interval accuracy  (coverage at 50/80/95% posterior predictive intervals)
# ---------------------------------------------------------------------------

def ppi_coverage(
    y_true_log10: np.ndarray,
    samples_log10: np.ndarray,
    alphas: Sequence[float] = (0.50, 0.80, 0.95),
) -> dict[str, float]:
    """Empirical coverage at each level.

    For each level alpha, compute the central (alpha)-interval from the
    posterior-predictive samples on the log10 scale and return the fraction of
    truths that fall inside it. Well-calibrated intervals have coverage ~= alpha.
    """
    y_true_log10 = np.asarray(y_true_log10, dtype=float)
    samples_log10 = np.asarray(samples_log10, dtype=float)
    assert samples_log10.ndim == 2, "samples must be (S, N)"
    assert samples_log10.shape[1] == y_true_log10.shape[0]

    out: dict[str, float] = {}
    for alpha in alphas:
        lo_q = (1.0 - alpha) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(samples_log10, lo_q, axis=0)
        hi = np.quantile(samples_log10, hi_q, axis=0)
        cov = float(np.mean((y_true_log10 >= lo) & (y_true_log10 <= hi)))
        out[f"cov-{int(alpha * 100)}"] = cov
    return out


# ---------------------------------------------------------------------------
# exceedance probability  (Brier score + ECE against the 104 MPN threshold)
# ---------------------------------------------------------------------------

def exceedance_prob_from_samples(
    samples_log10: np.ndarray,
    threshold_log10: float = EXCEEDANCE_LOG10,
) -> np.ndarray:
    """P(y > threshold) per row, computed as the sample fraction > threshold."""
    samples_log10 = np.asarray(samples_log10, dtype=float)
    return np.mean(samples_log10 > threshold_log10, axis=0).astype(float)


def brier_score(
    y_true_log10: np.ndarray,
    exc_prob: np.ndarray,
    threshold_log10: float = EXCEEDANCE_LOG10,
) -> float:
    """Brier score for the P(y > threshold) forecast vs the 0/1 outcome."""
    y_true_log10 = np.asarray(y_true_log10, dtype=float)
    exc_prob = np.asarray(exc_prob, dtype=float)
    y_bin = (y_true_log10 > threshold_log10).astype(float)
    return float(np.mean((exc_prob - y_bin) ** 2))


def expected_calibration_error(
    y_true_log10: np.ndarray,
    exc_prob: np.ndarray,
    n_bins: int = 10,
    threshold_log10: float = EXCEEDANCE_LOG10,
) -> float:
    """ECE with equal-width probability bins.

    `ECE = sum_b (n_b / N) * |mean(pred_b) - mean(truth_b)|`

    Captures systematic over/under-confidence on the exceedance probability.
    """
    y_true_log10 = np.asarray(y_true_log10, dtype=float)
    exc_prob = np.asarray(exc_prob, dtype=float)
    y_bin = (y_true_log10 > threshold_log10).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(y_bin)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (exc_prob >= lo) & (exc_prob <= hi)
        else:
            mask = (exc_prob >= lo) & (exc_prob < hi)
        if not mask.any():
            continue
        weight = mask.sum() / n
        ece += weight * abs(exc_prob[mask].mean() - y_bin[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# residual diagnostics  (log space) \u2014 not used for selection, but useful
# ---------------------------------------------------------------------------

def log_rmse(y_true_log10: np.ndarray, y_point_log10: np.ndarray) -> float:
    """Root-mean-squared error on the log10 scale. Diagnostic only."""
    y_true_log10 = np.asarray(y_true_log10, dtype=float)
    y_point_log10 = np.asarray(y_point_log10, dtype=float)
    return float(np.sqrt(np.mean((y_true_log10 - y_point_log10) ** 2)))


# ---------------------------------------------------------------------------
# single-call metric bundle
# ---------------------------------------------------------------------------

@dataclass
class FoldScore:
    """All metrics for a single (model, fold) evaluation."""

    model_name: str
    fold_val_year: int
    n_val: int
    counts_mae: float
    counts_medae: float
    log_rmse: float
    coverage: dict[str, float]
    brier: float
    ece: float

    def as_flat_dict(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "model": self.model_name,
            "fold_val_year": self.fold_val_year,
            "n_val": self.n_val,
            "counts-MAE": self.counts_mae,
            "counts-MedAE": self.counts_medae,
            "log-RMSE": self.log_rmse,
            "Brier": self.brier,
            "ECE": self.ece,
        }
        row.update(self.coverage)
        return row


def score_fold(
    *,
    model_name: str,
    fold_val_year: int,
    y_true_log10: np.ndarray,
    y_point_log10: np.ndarray,
    y_point_mpn: np.ndarray,
    samples_log10: np.ndarray,
) -> FoldScore:
    """Compute the full metric bundle for one model on one CV fold."""
    y_true_mpn = 10.0 ** y_true_log10
    cov = ppi_coverage(y_true_log10, samples_log10)
    exc = exceedance_prob_from_samples(samples_log10)
    return FoldScore(
        model_name=model_name,
        fold_val_year=int(fold_val_year),
        n_val=int(len(y_true_log10)),
        counts_mae=counts_mae(y_true_mpn, y_point_mpn),
        counts_medae=counts_medae(y_true_mpn, y_point_mpn),
        log_rmse=log_rmse(y_true_log10, y_point_log10),
        coverage=cov,
        brier=brier_score(y_true_log10, exc),
        ece=expected_calibration_error(y_true_log10, exc),
    )
