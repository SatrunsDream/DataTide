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

from src.modeling import _jax_compat  # noqa: F401  (must precede jax / numpyro import)

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np

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


def _subsample_fold(fold: FoldData, frac: float, seed: int) -> FoldData:
    """Return a `FoldData` with the train side uniformly subsampled to `frac`.

    Validation side is left untouched \u2014 we always score on the full val set
    to keep the MAE/MedAE/calibration metrics comparable across rungs. Only
    training is subsampled, and only for dev fits where `subsample_frac<1`.
    """
    n = len(fold.y_log_train)
    k = max(int(round(n * frac)), 64)        # never go below 64 train rows
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    idx = np.sort(idx)

    return FoldData(
        fold_val_year=fold.fold_val_year,
        y_log_train=fold.y_log_train[idx],
        month_train=fold.month_train[idx],
        station_idx_train=fold.station_idx_train[idx],
        county_idx_train=fold.county_idx_train[idx],
        X_smooth_train=fold.X_smooth_train[idx],
        X_linear_train=fold.X_linear_train[idx],
        miss_smooth_train=fold.miss_smooth_train[idx],
        miss_linear_train=fold.miss_linear_train[idx],
        left_mask_train=fold.left_mask_train[idx],
        right_mask_train=fold.right_mask_train[idx],
        det_low_log_train=fold.det_low_log_train[idx],
        det_high_log_train=fold.det_high_log_train[idx],
        y_log_val=fold.y_log_val,
        month_val=fold.month_val,
        station_idx_val=fold.station_idx_val,
        county_idx_val=fold.county_idx_val,
        X_smooth_val=fold.X_smooth_val,
        X_linear_val=fold.X_linear_val,
        miss_smooth_val=fold.miss_smooth_val,
        miss_linear_val=fold.miss_linear_val,
        smooth_features=fold.smooth_features,
        linear_features=fold.linear_features,
        n_stations=fold.n_stations,
        n_counties=fold.n_counties,
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
    # max_tree_depth=8 caps leapfrog steps per iteration at 2^8=256. The default
    # NumPyro value is 10 (->1024 steps), which is overkill for this panel and
    # makes each iteration ~4x slower when the sampler saturates the cap (which
    # it does on the hierarchical rungs with 778 station intercepts). Raise to
    # 9 or 10 only for the production winner if divergences warrant it.
    max_tree_depth: int = 8
    seed: int = 0
    # Progress bar is on by default so every `fit(...)` prints one tqdm bar
    # per chain \u2014 you can watch warmup / sampling advance live in the notebook.
    # Set progress_bar=False when running a scripted benchmark where tqdm
    # noise would clutter stdout.
    progress_bar: bool = True
    # How to dispatch multiple chains:
    #   "sequential" \u2014 one chain at a time (safe everywhere, shows one bar
    #                   per chain sequentially \u2014 easiest to read in Jupyter)
    #   "parallel"   \u2014 vmap across chains on a single device (best on GPU/TPU,
    #                   bars are combined into one multi-line view)
    #   "vectorized" \u2014 single big draw, num_chains stacked (also GPU-friendly)
    #   None         \u2014 auto: "parallel" if on GPU/TPU/METAL, else "sequential"
    chain_method: str | None = None
    # Train on a uniform random subsample of the training rows. 1.0 = full data,
    # 0.2 = 20% of rows. Ladder ranking is stable under subsampling (verified
    # empirically on this panel); we use subsample_frac < 1.0 for dev fits and
    # 1.0 for the production winner refit. Note: censored + interior rows are
    # subsampled together, so the censoring mask semantics are preserved.
    subsample_frac: float = 1.0


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

        # Optional dev-speed subsampling: pick a random fraction of training
        # rows and rebuild the design on that slice. FoldData has no random-
        # access accessor, so we materialise a tiny sub-fold dataclass.
        design_fold = fold
        if 0.0 < self.nuts.subsample_frac < 1.0:
            design_fold = _subsample_fold(fold, self.nuts.subsample_frac, self.nuts.seed)

        self._design_train = build_design(self.spec, design_fold, "train")

        y_log = jnp.asarray(design_fold.y_log_train)
        left_mask = jnp.asarray(design_fold.left_mask_train)
        right_mask = jnp.asarray(design_fold.right_mask_train)
        # det_low / det_high are NaN at interior rows (they only apply to
        # censored rows). Replace NaNs with 0 so `jnp.where` doesn't propagate
        # NaN through its non-selected branch; the interior rows' contribution
        # is zeroed out by the mask anyway.
        det_low = jnp.asarray(np.nan_to_num(design_fold.det_low_log_train, nan=0.0))
        det_high = jnp.asarray(np.nan_to_num(design_fold.det_high_log_train, nan=0.0))

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

        if self.nuts.progress_bar:
            # One-line header so multiple model fits in a loop are visually
            # separable. The per-chain tqdm bars NumPyro draws land right
            # below this line.
            sub_note = (
                f"  subsample={self.nuts.subsample_frac*100:.0f}%  (N={len(y_log):,})"
                if self.nuts.subsample_frac < 1.0 else f"  N={len(y_log):,}"
            )
            print(
                f"  [NUTS] {self.name:<34s}  "
                f"warmup={self.nuts.num_warmup}  samples={self.nuts.num_samples}  "
                f"chains={self.nuts.num_chains} ({chain_method}){sub_note}"
            )

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

    # --- persistence (so productionisation can skip the refit) ---

    def save(self, path) -> None:
        """Save just enough state to reconstruct a predict-ready rung.

        We persist `(rung, priors, nuts, posterior_samples)` as a compressed
        numpy archive. The MCMC trajectory, design matrices, and model closure
        are *not* stored \u2014 they can be rebuilt on load from the sampled
        parameters alone, which is what `.predict()` actually consumes.

        Writing format: a single `.npz` with:
          - one array per posterior site (named as in `_posterior_samples`)
          - a `__meta__` 0-d array holding a JSON blob with rung/priors/nuts.
        """
        import json as _json
        from pathlib import Path as _Path

        assert self._posterior_samples is not None, \
            "BayesianRung.save: nothing fit yet \u2014 call .fit() first."

        samples_np = {k: np.asarray(v) for k, v in self._posterior_samples.items()}

        # Forbid collision with the metadata key, which we prepend.
        if "__meta__" in samples_np:
            raise ValueError("posterior sample dict contains reserved key '__meta__'")

        meta_blob = _json.dumps({
            "rung": self.rung.value,
            "name": self.name,
            "priors": self.priors,
            "nuts": self.nuts.__dict__,
            "site_names": list(samples_np.keys()),
            "n_samples_total": int(next(iter(samples_np.values())).shape[0]),
        })
        out = {"__meta__": np.array(meta_blob)}
        out.update(samples_np)

        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)

    @classmethod
    def from_saved(cls, path) -> "BayesianRung":
        """Reconstruct a predict-ready `BayesianRung` from `save(...)` output.

        Does NOT run MCMC. After load, you can call `.predict(fold)` directly.
        `.to_inferencedata()` is unavailable because we don't round-trip the
        full `mcmc.states` \u2014 load the companion `winner_inferencedata.nc` if
        you need diagnostics.
        """
        import json as _json
        from pathlib import Path as _Path

        path = _Path(path)
        npz = np.load(path, allow_pickle=True)

        if "__meta__" not in npz.files:
            raise ValueError(
                f"{path} is not a BayesianRung save \u2014 missing '__meta__' key"
            )
        meta = _json.loads(str(npz["__meta__"]))

        nuts = NutsConfig(**meta["nuts"])
        obj = cls(rung=Rung(meta["rung"]), priors=meta["priors"], nuts=nuts,
                  name=meta.get("name", ""))

        samples = {k: np.asarray(npz[k]) for k in npz.files if k != "__meta__"}
        obj._model_fn = make_model(obj.spec, obj.priors)
        obj._posterior_samples = samples
        return obj
