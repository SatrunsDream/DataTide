"""
NUTS runner, prior / posterior predictive machinery, and a `Forecaster`
wrapper that lets a Bayesian rung drop into the same comparison pipeline as
the baselines.

A Bayesian rung is constructed as `BayesianRung(rung=Rung.vK, priors=...)`
and then fed to `src.evaluation.compare.score_model_on_fold` just like any
baseline. Under the hood:

- `.fit(fold)` builds the train-side design, runs NUTS on the censored
   log-normal model, and stashes the MCMC samples.
- `.predict(fold)` builds the val-side design and draws posterior-predictive
   samples `y_new_log10 ~ Normal(mu_val, sigma_obs)` with `Predictive`.

The posterior draws are also converted to an ArviZ `InferenceData` so the
downstream notebook can do `az.summary`, `az.plot_trace`, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np

from src.modeling import _jax_compat  # noqa: F401  (must precede numpyro import)

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive

from src.evaluation.compare import FoldData, Prediction
from src.modeling.bayesian import (
    Design,
    Rung,
    RUNG_LABELS,
    RungSpec,
    build_design,
    make_model,
)


# ---------------------------------------------------------------------------
# NUTS defaults  (tunable from the notebook)
# ---------------------------------------------------------------------------

@dataclass
class NutsConfig:
    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 2
    target_accept_prob: float = 0.85
    max_tree_depth: int = 10
    seed: int = 0
    progress_bar: bool = False
    # How to dispatch multiple chains:
    #   "sequential" \u2014 one chain at a time (safe everywhere, slow on GPU)
    #   "parallel"   \u2014 vmap across chains on a single device (best on GPU/TPU)
    #   "vectorized" \u2014 single big draw, num_chains stacked (also GPU-friendly)
    #   None         \u2014 auto: "parallel" if on GPU/TPU/METAL, else "sequential"
    chain_method: str | None = None


# ---------------------------------------------------------------------------
# posterior-predictive helpers
# ---------------------------------------------------------------------------

def _posterior_predictive_samples(
    model_fn,
    posterior_samples: dict[str, np.ndarray],
    design_val: Design,
    seed: int = 1,
) -> np.ndarray:
    """Draw `y_log_val` from the posterior predictive.

    NumPyro's `Predictive` handles the plumbing: for each posterior draw it
    re-executes the model with `y_log=None` on the validation design, which
    causes the likelihood-sample statement to draw new y's instead of
    conditioning on observed ones.
    """
    predictive = Predictive(
        model_fn,
        posterior_samples=posterior_samples,
        return_sites=("y_log",),
    )
    rng = jax.random.PRNGKey(seed)
    draws = predictive(rng, design=design_val)
    return np.asarray(draws["y_log"], dtype=np.float32)


def _prior_predictive_samples(
    model_fn,
    design: Design,
    num_samples: int = 500,
    seed: int = 2,
) -> np.ndarray:
    """Draw from the prior predictive for a design. Used for prior checks."""
    predictive = Predictive(
        model_fn,
        num_samples=num_samples,
        return_sites=("y_log",),
    )
    rng = jax.random.PRNGKey(seed)
    draws = predictive(rng, design=design)
    return np.asarray(draws["y_log"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Forecaster wrapper for Bayesian rungs  (plugs into compare.score_model_on_fold)
# ---------------------------------------------------------------------------

@dataclass
class BayesianRung:
    """Forecaster for a Bayesian ladder rung."""

    rung: Rung
    priors: dict
    nuts: NutsConfig = field(default_factory=NutsConfig)
    name: str = ""

    def __post_init__(self):
        self.rung = Rung(self.rung)
        self.spec = RungSpec.from_rung(self.rung)
        if not self.name:
            self.name = RUNG_LABELS[self.rung]
        self._mcmc: MCMC | None = None
        self._model_fn = None
        self._posterior_samples: dict[str, np.ndarray] | None = None
        self._design_train: Design | None = None

    # --- fit / predict (Forecaster protocol) ---

    def fit(self, fold: FoldData) -> None:
        self._model_fn = make_model(self.spec, self.priors)
        self._design_train = build_design(self.spec, fold, "train")

        y_log = jnp.asarray(fold.y_log_train)
        left_mask = jnp.asarray(fold.left_mask_train)
        right_mask = jnp.asarray(fold.right_mask_train)
        # det_low / det_high are NaN at interior rows (they only apply to
        # censored rows). Replace NaNs with 0 so `jnp.where` doesn't propagate
        # NaN through its non-selected branch; the interior rows' contribution
        # is zeroed out by the mask anyway.
        det_low = jnp.asarray(np.nan_to_num(fold.det_low_log_train, nan=0.0))
        det_high = jnp.asarray(np.nan_to_num(fold.det_high_log_train, nan=0.0))

        kernel = NUTS(
            self._model_fn,
            target_accept_prob=self.nuts.target_accept_prob,
            max_tree_depth=self.nuts.max_tree_depth,
        )
        chain_method = self.nuts.chain_method
        if chain_method is None:
            # Auto-select: parallelise across chains on an accelerator, stay
            # sequential on CPU (parallel on CPU is usually *slower* because
            # vmap forces one big XLA program with no threading headroom).
            backend = jax.default_backend()
            chain_method = "parallel" if backend in ("gpu", "tpu", "metal") else "sequential"
        mcmc = MCMC(
            kernel,
            num_warmup=self.nuts.num_warmup,
            num_samples=self.nuts.num_samples,
            num_chains=self.nuts.num_chains,
            progress_bar=self.nuts.progress_bar,
            chain_method=chain_method,
        )
        rng = jax.random.PRNGKey(self.nuts.seed)
        mcmc.run(
            rng,
            design=self._design_train,
            y_log=y_log,
            left_mask=left_mask,
            right_mask=right_mask,
            det_low_log=det_low,
            det_high_log=det_high,
        )
        self._mcmc = mcmc
        self._posterior_samples = mcmc.get_samples()

    def predict(self, fold: FoldData) -> Prediction:
        assert self._model_fn is not None and self._posterior_samples is not None, \
            "BayesianRung.fit must be called before predict"

        design_val = build_design(self.spec, fold, "val")
        samples = _posterior_predictive_samples(
            self._model_fn,
            self._posterior_samples,
            design_val,
            seed=self.nuts.seed + 11,
        )  # (S, N_val)

        samples_log10 = samples.astype(np.float32)
        point_log10 = np.median(samples_log10, axis=0)
        point_mpn = np.median(10.0 ** samples_log10, axis=0)
        return Prediction(
            samples_log10=samples_log10,
            point_log10=point_log10,
            point_mpn=point_mpn,
        )

    # --- introspection (used by the notebook for diagnostics) ---

    def to_inferencedata(self) -> az.InferenceData:
        """Convert the last MCMC run into an ArviZ InferenceData object."""
        assert self._mcmc is not None, "fit() first"
        return az.from_numpyro(self._mcmc)

    def prior_predictive(self, fold: FoldData, *, num_samples: int = 500) -> np.ndarray:
        """Draw from the prior predictive on the training design. Plan \u00a79.1 check."""
        model_fn = make_model(self.spec, self.priors)
        design = build_design(self.spec, fold, "train")
        return _prior_predictive_samples(model_fn, design, num_samples=num_samples)
