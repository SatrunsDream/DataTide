"""
Bayesian ladder rungs v0..v4 implemented in NumPyro.

Design is a single parameterised model function plus per-rung design-matrix
builders, so the ladder lives in one place and every rung produces identical
prior / posterior / posterior-predictive output shapes. Rungs differ only in
which blocks of `mu_it` are active:

    v0: alpha_0                                        (pooled intercept)
    v1: + alpha_month[m(i)]                             (month scaffold)
    v2: + alpha_station[s(i)] + alpha_county[c(i)]     (hierarchy on top)
    v3: + beta . X_linear  (weather, no doy_sin/cos)    (linear weather)
    v4: replace alpha_month with natural spline(doy)   (spline season)

Censored log-normal likelihood on log10 at every rung \u2014 censoring handling is
constant across the ladder so that any accuracy lift is attributable to
`mu_it` structure alone.

Memory note: earlier revisions of this module registered `mu` and
`doy_effect` as `numpyro.deterministic` sites. Those tensors are (N_train,)-
shaped, so NumPyro would cache `2 \u00d7 num_chains \u00d7 num_samples \u00d7 N_train`
floats of state per rung, which blew up to >1 GB per fit and OOM-killed the
more complex rungs. The deterministics are now removed entirely \u2014 `mu` is
reconstructed from the sampled parameters at predict time via `Predictive`,
which is exactly what the posterior-predictive pipeline already does.

Priors are taken from `artifacts/data/panel/enterococcus_panel_meta.json["priors"]`
and elicited empirically in `notebooks/modeling/eda.ipynb` section 13.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
import numpy as np

from src.modeling import _jax_compat  # noqa: F401  (must precede numpyro import)

import numpyro
import numpyro.distributions as dist

from src.evaluation.compare import FoldData


DAYS_PER_YEAR = 365.25


class Rung(str, Enum):
    v0 = "v0"
    v1 = "v1"
    v2 = "v2"
    v3 = "v3"
    v4 = "v4"


RUNG_LABELS: dict[Rung, str] = {
    Rung.v0: "v0 pooled intercept",
    Rung.v1: "v1 + month scaffold",
    Rung.v2: "v2 + station/county hierarchy",
    Rung.v3: "v3 + linear weather",
    Rung.v4: "v4 spline season",
}


# ---------------------------------------------------------------------------
# feature-index book-keeping  (which columns are rainfall, which are doy, etc)
# ---------------------------------------------------------------------------

DOY_FOURIER_FEATURES = ["doy_sin", "doy_cos"]


# ---------------------------------------------------------------------------
# basis constructor  (natural spline on day-of-year for v4)
# ---------------------------------------------------------------------------

def _natural_spline_basis(doy: np.ndarray, n_knots: int = 6) -> np.ndarray:
    """Truncated-power natural cubic spline basis on doy in [1, 365].

    Simple and self-contained: `[t, t^2, t^3] + (t-k_i)^3_+` for k_i at
    equally-spaced interior knots. Produces `3 + n_knots` basis columns.
    This is a natural (non-periodic) spline; minor boundary artefact between
    Dec 31 and Jan 1 is acceptable given the already-noisy data there.
    """
    t = (doy - 1.0) / 364.0       # scale to [0, 1]
    knots = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
    cols = [t, t ** 2, t ** 3]
    for k in knots:
        cols.append(np.maximum(t - k, 0.0) ** 3)
    return np.stack(cols, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# model specification  (what is active at each rung)
# ---------------------------------------------------------------------------

@dataclass
class RungSpec:
    rung: Rung
    use_month: bool
    use_station_county: bool
    use_linear: bool
    use_spline: bool
    spline_knots: int = 6

    @classmethod
    def from_rung(cls, rung: Rung) -> "RungSpec":
        r = Rung(rung)
        # v4 replaces the month scaffold with a smooth spline, so use_month is
        # False there. Every other rung >= v1 keeps the discrete month effect.
        return cls(
            rung=r,
            use_month=r in (Rung.v1, Rung.v2, Rung.v3),
            use_station_county=r in (Rung.v2, Rung.v3, Rung.v4),
            use_linear=r in (Rung.v3, Rung.v4),
            use_spline=r is Rung.v4,
        )


# ---------------------------------------------------------------------------
# design-matrix builder  (CPU numpy; used for both fit and predict sides)
# ---------------------------------------------------------------------------

@dataclass
class Design:
    """Per-rung design dict. JAX-array-ready; one per (train|val) side."""

    month_idx: jnp.ndarray                  # (N,)  int32 in 0..11 (0=Jan)
    station_idx: jnp.ndarray                # (N,)  int32
    county_idx: jnp.ndarray                 # (N,)  int32
    X_linear: jnp.ndarray | None            # (N, D_linear)  or None
    X_spline: jnp.ndarray | None            # (N, 3 + n_knots) or None
    n_stations: int
    n_counties: int
    n_linear: int
    linear_names: list[str] = field(default_factory=list)


def _doy_from_fold(fold: FoldData, which: str) -> np.ndarray:
    """Recover integer doy from the encoded doy_sin/doy_cos columns."""
    if which == "train":
        X_lin = fold.X_linear_train
    else:
        X_lin = fold.X_linear_val

    names = fold.linear_features
    try:
        i_sin = names.index("doy_sin")
        i_cos = names.index("doy_cos")
    except ValueError:
        # fallback: doy_sin/doy_cos are z-scored so we can't invert; assume raw
        raise RuntimeError("doy_sin / doy_cos not in linear features; cannot recover doy")

    # doy_sin/doy_cos columns are z-scored in the panel, so we need to un-z
    # them using the scalers from the panel meta. For simplicity compute the
    # full-range trig from the *de-standardised* values.
    raw_sin = X_lin[:, i_sin]
    raw_cos = X_lin[:, i_cos]
    # approximate raw: for standardised sin/cos (mean 0, sd ~1/sqrt(2)), angle
    # is still recoverable by atan2 since the ratio is scale-invariant.
    theta = np.arctan2(raw_sin, raw_cos)           # in [-pi, pi]
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    doy = (theta / (2 * np.pi)) * DAYS_PER_YEAR
    return np.clip(doy, 1.0, DAYS_PER_YEAR).astype(np.float32)


def build_design(spec: RungSpec, fold: FoldData, which: str) -> Design:
    if which == "train":
        month = fold.month_train.astype(np.int32) - 1   # 0..11
        station = fold.station_idx_train.astype(np.int32)
        county = fold.county_idx_train.astype(np.int32)
        X_smooth = fold.X_smooth_train
        X_linear_full = fold.X_linear_train
    else:
        month = fold.month_val.astype(np.int32) - 1
        station = fold.station_idx_val.astype(np.int32)
        county = fold.county_idx_val.astype(np.int32)
        X_smooth = fold.X_smooth_val
        X_linear_full = fold.X_linear_val

    # Linear covariate block: all smooth + linear features EXCEPT doy_sin/doy_cos
    # (the month scaffold / spline handles seasonality).
    D_smooth = X_smooth.shape[1]
    lin_names_raw = fold.linear_features
    doy_exclude_idx = [i for i, n in enumerate(lin_names_raw) if n in DOY_FOURIER_FEATURES]
    keep_linear_idx = [i for i in range(len(lin_names_raw)) if i not in doy_exclude_idx]
    X_linear_kept = X_linear_full[:, keep_linear_idx]

    covariate_block = np.concatenate([X_smooth, X_linear_kept], axis=1).astype(np.float32)
    covariate_names = (
        [f"smooth::{n}" for n in fold.smooth_features]
        + [f"linear::{lin_names_raw[i]}" for i in keep_linear_idx]
    )

    X_linear = jnp.asarray(covariate_block) if spec.use_linear else None

    X_spline = None
    if spec.use_spline:
        doy = _doy_from_fold(fold, which)
        X_spline = jnp.asarray(_natural_spline_basis(doy, n_knots=spec.spline_knots))

    return Design(
        month_idx=jnp.asarray(month),
        station_idx=jnp.asarray(station),
        county_idx=jnp.asarray(county),
        X_linear=X_linear,
        X_spline=X_spline,
        n_stations=fold.n_stations,
        n_counties=fold.n_counties,
        n_linear=int(covariate_block.shape[1]),
        linear_names=covariate_names,
    )


# ---------------------------------------------------------------------------
# the model function  (censored log-normal on log10, rung-parameterised mu_it)
# ---------------------------------------------------------------------------

def _prior_scale(priors: dict, key: str, default: float) -> float:
    node = priors.get(key, {})
    return float(node.get("scale", default))


def _prior_loc(priors: dict, key: str, default: float) -> float:
    node = priors.get(key, {})
    return float(node.get("loc", default))


def make_model(spec: RungSpec, priors: dict):
    """Return a NumPyro model closure for the requested rung.

    Priors:
    - `alpha_0`      ~ Normal(loc=priors.intercept.loc, scale=1.0)
    - `sigma_month`  ~ HalfNormal(priors.sigma_month.scale)           [v1..v3]
    - `z_month`      ~ Normal(0, 1)^12, alpha_month = z * sigma_month
    - `sigma_station`/`sigma_county` ~ HalfNormal(...)                [v2+]
    - `z_station`/`z_county` non-centred, alpha = z * sigma
    - `beta_linear[k]` ~ Normal(0, priors.beta_linear.scale)          [v3+]
    - `beta_spline[k]` ~ Normal(0, 0.5)                               [v4]
    - `sigma_obs`    ~ HalfNormal(priors.sigma_obs.scale)

    Important: we intentionally DO NOT register `mu` or any (N_train,)-shape
    tensor as a `numpyro.deterministic`. Doing so would force NumPyro to cache
    an (num_chains, num_samples, N_train) array that can easily exceed 1 GB
    on this panel and has no downstream use (Predictive reconstructs mu on
    the validation design at predict time). Small entity-level deterministics
    (alpha_month[12], alpha_station[~800], alpha_county[~16]) are fine and are
    kept for post-hoc interpretation.
    """

    alpha_0_loc = _prior_loc(priors, "intercept", 1.5)
    alpha_0_scale = _prior_scale(priors, "intercept", 1.0)
    sigma_month_scale = _prior_scale(priors, "sigma_month", 0.23)
    sigma_station_scale = _prior_scale(priors, "sigma_station", 0.75)
    sigma_county_scale = _prior_scale(priors, "sigma_county", 0.34)
    sigma_obs_scale = _prior_scale(priors, "sigma_obs", 0.70)
    beta_linear_scale = _prior_scale(priors, "beta_linear", 0.5)

    def model(design: Design,
              y_log: jnp.ndarray | None = None,
              left_mask: jnp.ndarray | None = None,
              right_mask: jnp.ndarray | None = None,
              det_low_log: jnp.ndarray | None = None,
              det_high_log: jnp.ndarray | None = None):

        N = design.month_idx.shape[0]

        alpha_0 = numpyro.sample("alpha_0", dist.Normal(alpha_0_loc, alpha_0_scale))
        mu = jnp.full((N,), alpha_0)

        if spec.use_month:
            sigma_month = numpyro.sample("sigma_month", dist.HalfNormal(sigma_month_scale))
            z_month = numpyro.sample("z_month", dist.Normal(0.0, 1.0).expand([12]))
            alpha_month = numpyro.deterministic("alpha_month", z_month * sigma_month)
            mu = mu + alpha_month[design.month_idx]

        if spec.use_station_county:
            sigma_station = numpyro.sample("sigma_station", dist.HalfNormal(sigma_station_scale))
            z_station = numpyro.sample("z_station", dist.Normal(0.0, 1.0).expand([design.n_stations]))
            alpha_station = numpyro.deterministic("alpha_station", z_station * sigma_station)
            mu = mu + alpha_station[design.station_idx]

            sigma_county = numpyro.sample("sigma_county", dist.HalfNormal(sigma_county_scale))
            z_county = numpyro.sample("z_county", dist.Normal(0.0, 1.0).expand([design.n_counties]))
            alpha_county = numpyro.deterministic("alpha_county", z_county * sigma_county)
            mu = mu + alpha_county[design.county_idx]

        if spec.use_linear and design.X_linear is not None:
            D = design.X_linear.shape[1]
            beta_linear = numpyro.sample(
                "beta_linear",
                dist.Normal(0.0, beta_linear_scale).expand([D]),
            )
            mu = mu + design.X_linear @ beta_linear

        if spec.use_spline and design.X_spline is not None:
            D = design.X_spline.shape[1]
            beta_spline = numpyro.sample("beta_spline", dist.Normal(0.0, 0.5).expand([D]))
            mu = mu + design.X_spline @ beta_spline

        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(sigma_obs_scale))

        # prior-predictive / posterior-predictive: no y \u2192 sample y from the likelihood
        if y_log is None:
            numpyro.sample("y_log", dist.Normal(mu, sigma_obs))
            return

        # censored log-normal likelihood on log10:
        # - interior rows:      log p(y | mu, sigma)
        # - left-censored:      log Phi((det_low  - mu) / sigma)
        # - right-censored:     log (1 - Phi((det_high - mu) / sigma))
        interior_mask = ~(left_mask | right_mask)

        # interior contribution
        logp_interior = jss.norm.logpdf(y_log, mu, sigma_obs)
        numpyro.factor("ll_interior", jnp.sum(jnp.where(interior_mask, logp_interior, 0.0)))

        # left-censored contribution
        if left_mask is not None:
            logcdf_left = jss.norm.logcdf(det_low_log, mu, sigma_obs)
            numpyro.factor("ll_left", jnp.sum(jnp.where(left_mask, logcdf_left, 0.0)))

        # right-censored contribution
        if right_mask is not None:
            logsf_right = jss.norm.logsf(det_high_log, mu, sigma_obs)
            numpyro.factor("ll_right", jnp.sum(jnp.where(right_mask, logsf_right, 0.0)))

    return model
