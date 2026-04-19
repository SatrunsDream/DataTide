"""
Calibration diagnostics and their matplotlib renderings.

Two diagnostics:

- Reliability diagram for the P(exceed 104) forecast. X-axis is the predicted
  exceedance probability, binned; Y-axis is the empirical exceedance rate in
  each bin. The 45-degree line is perfect calibration.

- PIT (Probability Integral Transform) histogram for the continuous log10
  predictive distribution. For each observed row, PIT(i) = CDF_i(y_i) under the
  predictive distribution. A well-calibrated predictive distribution produces
  a uniform-on-[0,1] PIT histogram. Deviations diagnose over-dispersion
  (U-shape), under-dispersion (inverted-U), and bias (slope).

Neither diagnostic reduces to a single scalar \u2014 they produce plots and
the supporting per-bin / per-observation arrays. The scalar ECE is in
`metrics.expected_calibration_error`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import EXCEEDANCE_LOG10


@dataclass
class ReliabilityCurve:
    """Per-bin summary for the reliability diagram."""

    bin_centers: np.ndarray     # predicted probability, bin midpoint
    empirical_rate: np.ndarray  # observed exceedance fraction in bin
    bin_counts: np.ndarray      # # observations in bin (for point sizing)


def reliability_curve(
    y_true_log10: np.ndarray,
    exc_prob: np.ndarray,
    n_bins: int = 10,
    threshold_log10: float = EXCEEDANCE_LOG10,
) -> ReliabilityCurve:
    """Equal-width bins of predicted probability; empirical rate per bin."""
    y_true_log10 = np.asarray(y_true_log10, dtype=float)
    exc_prob = np.asarray(exc_prob, dtype=float)
    y_bin = (y_true_log10 > threshold_log10).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rates = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (exc_prob >= lo) & (exc_prob <= hi)
        else:
            mask = (exc_prob >= lo) & (exc_prob < hi)
        counts[i] = int(mask.sum())
        if mask.any():
            rates[i] = float(y_bin[mask].mean())
    return ReliabilityCurve(
        bin_centers=centers, empirical_rate=rates, bin_counts=counts
    )


def pit_values(y_true_log10: np.ndarray, samples_log10: np.ndarray) -> np.ndarray:
    """Empirical PIT values under the predictive distribution.

    PIT(i) is the empirical CDF at the observed y, evaluated from the
    posterior-predictive draws for that row. Uniform on [0, 1] under perfect
    calibration.

    samples_log10 : shape (S, N)
    """
    y_true_log10 = np.asarray(y_true_log10, dtype=float)
    samples_log10 = np.asarray(samples_log10, dtype=float)
    assert samples_log10.ndim == 2
    S, N = samples_log10.shape
    assert N == y_true_log10.shape[0]
    # fraction of samples <= observed, per row
    return (samples_log10 <= y_true_log10[None, :]).mean(axis=0)


# ---------------------------------------------------------------------------
# matplotlib renderers \u2014 kept here (not in src/viz) so calibration is one import
# ---------------------------------------------------------------------------

def plot_reliability(
    ax,
    curve: ReliabilityCurve,
    *,
    label: str | None = None,
    color: str = "C0",
    show_base_rate: float | None = None,
) -> None:
    """Render a reliability diagram onto an existing Axes."""
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", lw=1, label="perfect calibration")
    sizes = 10.0 + 0.6 * np.sqrt(curve.bin_counts.clip(min=1))
    m = ~np.isnan(curve.empirical_rate)
    ax.scatter(
        curve.bin_centers[m],
        curve.empirical_rate[m],
        s=sizes[m],
        color=color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
        label=label,
    )
    ax.plot(curve.bin_centers[m], curve.empirical_rate[m], color=color, alpha=0.6, lw=1)
    if show_base_rate is not None:
        ax.axhline(show_base_rate, color="crimson", linestyle=":", lw=1,
                   label=f"base rate = {show_base_rate:.2%}")
    ax.set_xlabel("predicted P(y > 104)")
    ax.set_ylabel("empirical exceedance rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")


def plot_pit(ax, pit: np.ndarray, *, n_bins: int = 20, color: str = "C0", label: str | None = None) -> None:
    """Render a PIT histogram. Uniform bars = well-calibrated predictive dist."""
    ax.hist(pit, bins=n_bins, range=(0, 1), density=True, color=color, alpha=0.70,
            edgecolor="white", linewidth=0.6, label=label)
    ax.axhline(1.0, color="crimson", linestyle="--", lw=1, label="uniform (ideal)")
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2)
