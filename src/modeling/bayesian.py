"""
Bayesian ladder rungs v0..v6 implemented in NumPyro.

Design is a single parameterised model function plus per-rung design-matrix
builders, so the ladder lives in one place and every rung produces identical
prior / posterior / posterior-predictive output shapes. Rungs differ only in
which blocks of `mu_it` are active \u2014 exactly as specified in
`context/MODELING_PLAN.md \u00a75`:

    v0: alpha_0                                       (pooled intercept)
    v1: + alpha_month[m(i)]                            (month scaffold)
    v2: + alpha_station[s(i)] + alpha_county[c(i)]    (hierarchy on top)
    v3: + beta . X_linear (no doy_sin, no doy_cos)    (linear weather)
    v4: replace alpha_month with polynomial_3(doy)    (polynomial season)
    v5: replace polynomial with natural spline(doy)   (spline season)
    v6: replace spline with periodic-HSGP(doy)        (HSGP season)
        + shared-amplitude rain slope block           (HSGP rain)

Censored log-normal likelihood on log10 at every rung \u2014 censoring handling is
constant across the ladder so that any accuracy lift is attributable to
`mu_it` structure alone.

Priors are taken from `artifacts/data/panel/enterococcus_panel_meta.json["priors"]`
and elicited empirically in `notebooks/modeling/eda.ipynb` section 13.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.modeling import _jax_compat  # noqa: F401  (must precede jax / numpyro import)

import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
import numpy as np

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
    v5 = "v5"
    v6 = "v6"


RUNG_LABELS: dict[Rung, str] = {
    Rung.v0: "v0 pooled intercept",
    Rung.v1: "v1 + month scaffold",
    Rung.v2: "v2 + station/county hierarchy",
    Rung.v3: "v3 + linear weather",
    Rung.v4: "v4 polynomial season",
    Rung.v5: "v5 spline season",
    Rung.v6: "v6 HSGP season + HSGP rain",
}


# ---------------------------------------------------------------------------
# feature-index book-keeping  (which columns are rainfall, which are doy, etc)
# ---------------------------------------------------------------------------

RAIN_FEATURES = ["rain_24h_mm", "rain_48h_mm", "rain_72h_mm", "rain_7d_mm"]
DOY_FOURIER_FEATURES = ["doy_sin", "doy_cos"]


def _col_indices(names: list[str], wanted: list[str]) -> np.ndarray:
    """Return 1-d int32 array of indices of `wanted` in `names` (missing \u2192 skipped)."""
    name_to_idx = {n: i for i, n in enumerate(names)}
    return np.array([name_to_idx[w] for w in wanted if w in name_to_idx], dtype=np.int32)


# ---------------------------------------------------------------------------
# basis constructors  (polynomial, spline, periodic-Fourier for HSGP-on-doy)
# ---------------------------------------------------------------------------

def _polynomial_basis(doy: np.ndarray, degree: int = 3) -> np.ndarray:
    """Centred / scaled polynomial basis of day-of-year, degree `degree`.

    Centring at day 183 (summer solstice-ish) and scaling by 365/2 keeps the
    columns O(1) so that Normal(0, 1) priors on coefficients imply O(1) log10
    seasonal effect \u2014 the prior predictive check will sanity-tune this.
    """
    t = (doy - 183.0) / 182.5
    cols = [t ** k for k in range(1, degree + 1)]
    return np.stack(cols, axis=1).astype(np.float32)


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


def _periodic_fourier_basis(doy: np.ndarray, K: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Periodic Fourier basis for periodic HSGP on day-of-year.

    Returns `(Phi_cos, Phi_sin)`, each shape `(N, K)`. With a spectral-decay
    prior `b_k ~ Normal(0, amplitude / k^decay)` the resulting function is a
    smooth periodic GP on the annual cycle. K=8 captures sub-monthly structure;
    K=4 is already enough to capture winter/summer asymmetry.
    """
    theta = 2.0 * np.pi * (doy / DAYS_PER_YEAR)
    k = np.arange(1, K + 1)[None, :]                           # (1, K)
    arg = theta[:, None] * k                                   # (N, K)
    return np.cos(arg).astype(np.float32), np.sin(arg).astype(np.float32)


# ---------------------------------------------------------------------------
# model specification  (what is active at each rung)
# ---------------------------------------------------------------------------

@dataclass
class RungSpec:
    rung: Rung
    use_month: bool
    use_station_county: bool
    use_linear: bool
    use_poly: bool
    use_spline: bool
    use_hsgp: bool
    hsgp_K: int = 8
    spline_knots: int = 6
    poly_degree: int = 3

    @classmethod
    def from_rung(cls, rung: Rung) -> "RungSpec":
        r = Rung(rung)
        return cls(
            rung=r,
            use_month=r in (Rung.v1, Rung.v2, Rung.v3),
            use_station_county=r in (Rung.v2, Rung.v3, Rung.v4, Rung.v5, Rung.v6),
            use_linear=r in (Rung.v3, Rung.v4, Rung.v5, Rung.v6),
            use_poly=r is Rung.v4,
            use_spline=r is Rung.v5,
            use_hsgp=r is Rung.v6,
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
    X_poly: jnp.ndarray | None              # (N, deg)
    X_spline: jnp.ndarray | None            # (N, 3 + n_knots)
    Phi_cos: jnp.ndarray | None             # (N, K)
    Phi_sin: jnp.ndarray | None             # (N, K)
    X_rain: jnp.ndarray | None              # (N, 4)
    n_stations: int
    n_counties: int
    n_linear: int
    linear_names: list[str] = field(default_factory=list)


def _doy_from_fold(fold: FoldData, which: str) -> np.ndarray:
    """Approximate day-of-year from encoded doy_sin/doy_cos columns.

    Prefer the exact `FoldData.doy_*` values when available; this helper is a
    backward-compatible fallback for callers that still construct folds
    manually without the explicit day-of-year arrays.
    """
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


def _resolve_doy(fold: FoldData, which: str) -> np.ndarray:
    """Use exact doy values when the fold provides them, else fall back."""
    doy = fold.doy_train if which == "train" else fold.doy_val
    if doy is not None:
        return np.asarray(doy, dtype=np.float32)
    return _doy_from_fold(fold, which)


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
    # (the month scaffold / polynomial / spline / HSGP handles seasonality).
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

    X_poly = X_spline = Phi_cos = Phi_sin = X_rain = None
    if spec.use_poly or spec.use_spline or spec.use_hsgp:
        doy = _resolve_doy(fold, which)
        if spec.use_poly:
            X_poly = jnp.asarray(_polynomial_basis(doy, degree=spec.poly_degree))
        if spec.use_spline:
            X_spline = jnp.asarray(_natural_spline_basis(doy, n_knots=spec.spline_knots))
        if spec.use_hsgp:
            Pc, Ps = _periodic_fourier_basis(doy, K=spec.hsgp_K)
            Phi_cos, Phi_sin = jnp.asarray(Pc), jnp.asarray(Ps)

    if spec.use_hsgp:
        # shared-amplitude rain slope block (the "HSGP on rain" from the plan)
        rain_idx = _col_indices(fold.smooth_features, RAIN_FEATURES)
        if len(rain_idx) > 0:
            X_rain = jnp.asarray(X_smooth[:, rain_idx].astype(np.float32))

    return Design(
        month_idx=jnp.asarray(month),
        station_idx=jnp.asarray(station),
        county_idx=jnp.asarray(county),
        X_linear=X_linear,
        X_poly=X_poly,
        X_spline=X_spline,
        Phi_cos=Phi_cos,
        Phi_sin=Phi_sin,
        X_rain=X_rain,
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
    - `sigma_month`  ~ HalfNormal(scale=priors.sigma_month.scale)     [v1..v3]
    - `alpha_month`  ~ non-centred Normal(0, sigma_month) of shape (12,)
    - `sigma_station`/`sigma_county` ~ HalfNormal(scale=priors...)    [v2+]
    - `alpha_station` / `alpha_county` non-centred
    - `beta_linear[k]` ~ Normal(0, priors.beta_linear.scale)          [v3+]
    - `beta_poly[k]` ~ Normal(0, 0.5)                                 [v4]
    - `beta_spline[k]` ~ Normal(0, 0.5)                               [v5]
    - HSGP amplitudes and Fourier coefficients                        [v6]
    - `sigma_obs`    ~ HalfNormal(scale=priors.sigma_obs.scale)
    """

    alpha_0_loc = _prior_loc(priors, "intercept", 1.5)
    alpha_0_scale = _prior_scale(priors, "intercept", 1.0)
    sigma_month_scale = _prior_scale(priors, "sigma_month", 0.23)
    sigma_station_scale = _prior_scale(priors, "sigma_station", 0.75)
    sigma_county_scale = _prior_scale(priors, "sigma_county", 0.34)
    sigma_obs_scale = _prior_scale(priors, "sigma_obs", 0.70)
    beta_linear_scale = _prior_scale(priors, "beta_linear", 0.5)
    hsgp_doy_amp_scale = _prior_scale(priors, "hsgp_amplitude_seasonal", 0.23)
    hsgp_rain_amp_scale = _prior_scale(priors, "hsgp_amplitude_rain", 0.24)

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

        if spec.use_poly and design.X_poly is not None:
            D = design.X_poly.shape[1]
            beta_poly = numpyro.sample("beta_poly", dist.Normal(0.0, 0.5).expand([D]))
            mu = mu + design.X_poly @ beta_poly

        if spec.use_spline and design.X_spline is not None:
            D = design.X_spline.shape[1]
            beta_spline = numpyro.sample("beta_spline", dist.Normal(0.0, 0.5).expand([D]))
            mu = mu + design.X_spline @ beta_spline

        if spec.use_hsgp and design.Phi_cos is not None and design.Phi_sin is not None:
            K = design.Phi_cos.shape[1]
            amp_doy = numpyro.sample("hsgp_amp_doy", dist.HalfNormal(hsgp_doy_amp_scale))
            # spectral-decay prior: harmonic k has SD = amp / k  (Matern-like)
            k_vec = jnp.arange(1, K + 1, dtype=jnp.float32)
            b_cos = numpyro.sample("b_cos", dist.Normal(0.0, 1.0).expand([K]))
            b_sin = numpyro.sample("b_sin", dist.Normal(0.0, 1.0).expand([K]))
            doy_effect = amp_doy * (
                (design.Phi_cos @ (b_cos / k_vec)) + (design.Phi_sin @ (b_sin / k_vec))
            )
            mu = mu + numpyro.deterministic("doy_effect", doy_effect)

            if design.X_rain is not None:
                R = design.X_rain.shape[1]
                amp_rain = numpyro.sample("hsgp_amp_rain", dist.HalfNormal(hsgp_rain_amp_scale))
                z_rain = numpyro.sample("z_rain", dist.Normal(0.0, 1.0).expand([R]))
                beta_rain = numpyro.deterministic("beta_rain", amp_rain * z_rain)
                mu = mu + design.X_rain @ beta_rain

        mu = numpyro.deterministic("mu", mu)
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
