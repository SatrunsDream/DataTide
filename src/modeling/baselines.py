"""
Non-Bayesian baselines for the Phase B comparison.

All three baselines implement the `Forecaster` protocol defined in
`src/evaluation/compare.py` so they can be scored by the same machinery as
the Bayesian ladder.

Each `.predict(fold)` returns a `Prediction(samples_log10, point_log10,
point_mpn)`. For the naive baseline we bootstrap from the empirical per-month
log10 distribution; for OLS and XGBoost we approximate the predictive
distribution as `Normal(mu_hat, sigma_resid)` on the log10 scale and sample
from it. This gives every model a comparable set of posterior-predictive-like
samples to feed into coverage / Brier / ECE.

Point predictions in MPN space are the **posterior median** of those samples
in counts space, `median(10 ** samples)`, following the MODELING_PLAN (we do
not use `10 ** mean(log10)`, which is the geometric mean not the median \u2014
Jensen's inequality).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.compare import FoldData, Prediction


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _point_mpn_from_samples(samples_log10: np.ndarray) -> np.ndarray:
    """Posterior median in count space: median(10 ** samples), per row."""
    return np.median(10.0 ** samples_log10, axis=0)


def _point_log10_from_samples(samples_log10: np.ndarray) -> np.ndarray:
    """Posterior median on the log10 scale (for log-RMSE diagnostic only)."""
    return np.median(samples_log10, axis=0)


def _censored_normal_logl_dummy(*_args, **_kwargs):  # pragma: no cover
    """Place-holder: baselines ignore censoring by construction."""
    return 0.0


# ---------------------------------------------------------------------------
# B1  \u2014  Naive seasonal-mean-of-counts baseline  (Phase B floor)
# ---------------------------------------------------------------------------

@dataclass
class NaiveSeasonalMeanCounts:
    """Per-month empirical distribution from training counts.

    Plan reference: `context/MODELING_PLAN.md \u00a76.1`.

    Fit: per month m in 1..12, cache the *set* of training log10 values whose
    sample month was m. Predict: for each held-out row, draw S samples from
    the cached log10-set for that row's month (with replacement). This is an
    empirical bootstrap of the per-month log10 distribution, respecting the
    shape of the observed training distribution rather than assuming
    Normality.

    If a validation month has no training history, fall back to the pooled
    training distribution for that fold (rare edge case).
    """

    name: str = "B1 naive seasonal-mean (by month)"
    n_samples: int = 2000

    def fit(self, fold: FoldData) -> None:
        by_month: dict[int, np.ndarray] = {}
        for m in range(1, 13):
            mask = fold.month_train == m
            if mask.any():
                by_month[m] = fold.y_log_train[mask].astype(np.float64)
        self._by_month = by_month
        self._pool = fold.y_log_train.astype(np.float64)
        rng = np.random.default_rng(0)
        self._rng = rng

    def predict(self, fold: FoldData) -> Prediction:
        rng = self._rng
        S = self.n_samples
        N = fold.n_val
        samples = np.empty((S, N), dtype=np.float32)
        for m in range(1, 13):
            idx = np.flatnonzero(fold.month_val == m)
            if idx.size == 0:
                continue
            pool = self._by_month.get(m, self._pool)
            draws = rng.choice(pool, size=(S, idx.size), replace=True)
            samples[:, idx] = draws.astype(np.float32)
        return Prediction(
            samples_log10=samples,
            point_log10=_point_log10_from_samples(samples),
            point_mpn=_point_mpn_from_samples(samples),
        )


# ---------------------------------------------------------------------------
# B2  \u2014  OLS linear regression on log10  (classical counterpart)
# ---------------------------------------------------------------------------

@dataclass
class OLSLog10:
    """OLS on `log10(y)` with the same covariate block as Bayesian v3.

    Features: all z-scored smooth + linear features from the panel artifact
    plus their missingness indicators, plus station one-hots and a 12-level
    month one-hot (matching the v3 Bayesian rung's design). No partial pooling
    \u2014 that's the whole point: this is the no-hierarchy counterpart.

    Predictive distribution: Normal(mu_hat_i, sigma_resid) on log10 scale,
    where sigma_resid is the training-residual SD.
    """

    name: str = "B2 OLS log10"
    n_samples: int = 2000
    # Station one-hots without pooling would create ~(N_stations-1) columns of
    # which most are all-zero on any single fold (stations with 0 training rows
    # that fold), making the normal equations rank-deficient. The honest
    # "no-hierarchy" counterpart to the Bayesian ladder is OLS on the panel
    # covariates + month one-hots. We keep the flag so the user can opt in.
    include_station_one_hot: bool = False

    def _design(
        self, fold: FoldData, which: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if which == "train":
            X_smooth = fold.X_smooth_train
            X_linear = fold.X_linear_train
            miss_smooth = fold.miss_smooth_train
            miss_linear = fold.miss_linear_train
            month = fold.month_train
            station = fold.station_idx_train
        else:
            X_smooth = fold.X_smooth_val
            X_linear = fold.X_linear_val
            miss_smooth = fold.miss_smooth_val
            miss_linear = fold.miss_linear_val
            month = fold.month_val
            station = fold.station_idx_val

        parts = [
            np.ones((month.shape[0], 1), dtype=np.float32),
            X_smooth.astype(np.float32),
            X_linear.astype(np.float32),
            miss_smooth.astype(np.float32),
            miss_linear.astype(np.float32),
        ]
        # 12-level month one-hot (drop m=6 as reference to avoid collinearity with intercept)
        month_oh = np.zeros((month.shape[0], 11), dtype=np.float32)
        cols = [m for m in range(1, 13) if m != 6]
        for j, m in enumerate(cols):
            month_oh[:, j] = (month == m).astype(np.float32)
        parts.append(month_oh)

        if self.include_station_one_hot:
            # sparse-ish station one-hot; keep only the first (n_stations-1) for identifiability
            S = fold.n_stations
            one_hot = np.zeros((station.shape[0], S - 1), dtype=np.float32)
            keep = station < (S - 1)  # rows whose station index is not the reference
            rows = np.flatnonzero(keep)
            one_hot[rows, station[keep]] = 1.0
            parts.append(one_hot)

        X = np.concatenate(parts, axis=1)
        return X, month

    def fit(self, fold: FoldData) -> None:
        X_tr, _ = self._design(fold, "train")
        y_tr = np.ascontiguousarray(fold.y_log_train.astype(np.float64))

        # Drop zero-variance columns (constant intercept already captured in
        # column 0; stations / months with no training rows would otherwise
        # produce all-zero columns). Then solve normal equations with a small
        # ridge: solve is O(D^3), stable, and ~100000x faster than lstsq on
        # OpenBLAS / ARM for this shape.
        keep = X_tr.std(axis=0) > 1e-10
        keep[0] = True  # always keep intercept column
        X_kept = np.ascontiguousarray(X_tr[:, keep].astype(np.float64))
        D = X_kept.shape[1]
        lam = 1e-6
        # OpenBLAS 0.3.30 on ARM64 macOS deadlocks on the 2-D @ 1-D GEMV
        # pattern (X.T @ y) for large N; einsum ("ij,i->j" and "ij,ik->jk")
        # lowers into a non-BLAS path that is both fast and reliable here.
        XtX = np.einsum("ij,ik->jk", X_kept, X_kept)
        XtX += lam * np.eye(D)
        Xty = np.einsum("ij,i->j", X_kept, y_tr)
        beta_kept = np.linalg.solve(XtX, Xty)
        beta_full = np.zeros(X_tr.shape[1], dtype=np.float64)
        beta_full[keep] = beta_kept

        self._beta = beta_full
        self._kept_cols = keep
        # Residuals: same deadlock class, so avoid `@` on the full design.
        mu_tr = np.einsum("ij,j->i", X_tr.astype(np.float64), self._beta)
        self._sigma = float(np.std(y_tr - mu_tr))
        self._rng = np.random.default_rng(1)

    def predict(self, fold: FoldData) -> Prediction:
        X_val, _ = self._design(fold, "val")
        # See `fit` for the einsum-over-`@` rationale (OpenBLAS ARM deadlock).
        mu = np.einsum("ij,j->i", X_val.astype(np.float64), self._beta).astype(np.float32)
        rng = self._rng
        S = self.n_samples
        samples = rng.normal(loc=mu[None, :], scale=self._sigma, size=(S, mu.shape[0])).astype(np.float32)
        return Prediction(
            samples_log10=samples,
            point_log10=_point_log10_from_samples(samples),
            point_mpn=_point_mpn_from_samples(samples),
        )


# ---------------------------------------------------------------------------
# B3  \u2014  XGBoost regressor on log10  (non-linear counterpart)
# ---------------------------------------------------------------------------

@dataclass
class XGBoostLog10:
    """Gradient-boosted trees on log10(y).

    Predictive distribution is `Normal(xgb_pred, sigma_train_resid)` \u2014 the
    same retrofit OLS uses, explicitly noted in the plan as the honest
    apples-to-apples retrofit for probabilistic metrics. See MODELING_PLAN
    \u00a76.3 for the limitation note.
    """

    name: str = "B3 XGBoost log10"
    n_samples: int = 2000
    n_estimators: int = 600
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    early_stopping_rounds: int = 40

    def _design(self, fold: FoldData, which: str) -> tuple[np.ndarray, np.ndarray]:
        if which == "train":
            X_s, X_l, m_s, m_l = fold.X_smooth_train, fold.X_linear_train, fold.miss_smooth_train, fold.miss_linear_train
            month, station, county = fold.month_train, fold.station_idx_train, fold.county_idx_train
        else:
            X_s, X_l, m_s, m_l = fold.X_smooth_val, fold.X_linear_val, fold.miss_smooth_val, fold.miss_linear_val
            month, station, county = fold.month_val, fold.station_idx_val, fold.county_idx_val

        X = np.concatenate([
            X_s.astype(np.float32),
            X_l.astype(np.float32),
            m_s.astype(np.float32),
            m_l.astype(np.float32),
            month.astype(np.float32).reshape(-1, 1),
            station.astype(np.float32).reshape(-1, 1),
            county.astype(np.float32).reshape(-1, 1),
        ], axis=1)
        return X, month

    def fit(self, fold: FoldData) -> None:
        import xgboost as xgb  # local import: optional dep
        X_tr, _ = self._design(fold, "train")
        y_tr = fold.y_log_train.astype(np.float32)

        # inner 80/20 split on the training window for early stopping
        rng = np.random.default_rng(2)
        n = X_tr.shape[0]
        perm = rng.permutation(n)
        n_inner_val = max(200, int(0.2 * n))
        inner_val_idx = perm[:n_inner_val]
        inner_tr_idx = perm[n_inner_val:]

        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method="hist",
            early_stopping_rounds=self.early_stopping_rounds,
            verbosity=0,
            random_state=3,
        )
        self._model.fit(
            X_tr[inner_tr_idx],
            y_tr[inner_tr_idx],
            eval_set=[(X_tr[inner_val_idx], y_tr[inner_val_idx])],
            verbose=False,
        )
        # residual-SD on full training set (with best n_estimators)
        mu_train = self._model.predict(X_tr)
        self._sigma = float(np.std(y_tr - mu_train))
        self._rng = np.random.default_rng(4)

    def predict(self, fold: FoldData) -> Prediction:
        X_val, _ = self._design(fold, "val")
        mu = self._model.predict(X_val).astype(np.float32)
        rng = self._rng
        S = self.n_samples
        samples = rng.normal(loc=mu[None, :], scale=self._sigma, size=(S, mu.shape[0])).astype(np.float32)
        return Prediction(
            samples_log10=samples,
            point_log10=_point_log10_from_samples(samples),
            point_mpn=_point_mpn_from_samples(samples),
        )
