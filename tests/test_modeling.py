import os
import subprocess
import sys

import numpy as np

from src.evaluation.compare import FoldData
from src.modeling import _jax_compat
from src.modeling.bayesian import Rung, RungSpec, _polynomial_basis, build_design
from src.modeling.cv import iter_folds, load_panel


def _toy_fold() -> FoldData:
    n_train, n_val = 3, 2
    return FoldData(
        fold_val_year=2020,
        y_log_train=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        month_train=np.array([1, 4, 7], dtype=np.int32),
        station_idx_train=np.zeros(n_train, dtype=np.int32),
        county_idx_train=np.zeros(n_train, dtype=np.int32),
        X_smooth_train=np.zeros((n_train, 1), dtype=np.float32),
        X_linear_train=np.zeros((n_train, 2), dtype=np.float32),
        miss_smooth_train=np.zeros((n_train, 1), dtype=np.int8),
        miss_linear_train=np.zeros((n_train, 2), dtype=np.int8),
        left_mask_train=np.zeros(n_train, dtype=bool),
        right_mask_train=np.zeros(n_train, dtype=bool),
        det_low_log_train=np.full(n_train, np.nan, dtype=np.float64),
        det_high_log_train=np.full(n_train, np.nan, dtype=np.float64),
        y_log_val=np.array([0.4, 0.5], dtype=np.float32),
        month_val=np.array([9, 12], dtype=np.int32),
        station_idx_val=np.zeros(n_val, dtype=np.int32),
        county_idx_val=np.zeros(n_val, dtype=np.int32),
        X_smooth_val=np.zeros((n_val, 1), dtype=np.float32),
        X_linear_val=np.zeros((n_val, 2), dtype=np.float32),
        miss_smooth_val=np.zeros((n_val, 1), dtype=np.int8),
        miss_linear_val=np.zeros((n_val, 2), dtype=np.int8),
        doy_train=np.array([1, 91, 181], dtype=np.int16),
        doy_val=np.array([273, 365], dtype=np.int16),
        smooth_features=["rain_24h_mm"],
        linear_features=["doy_sin", "doy_cos"],
        n_stations=1,
        n_counties=1,
    )


def test_build_design_prefers_exact_doy_arrays():
    fold = _toy_fold()
    spec = RungSpec.from_rung(Rung.v4)

    design_train = build_design(spec, fold, "train")
    design_val = build_design(spec, fold, "val")

    np.testing.assert_allclose(
        np.asarray(design_train.X_poly),
        _polynomial_basis(fold.doy_train.astype(np.float32)),
    )
    np.testing.assert_allclose(
        np.asarray(design_val.X_poly),
        _polynomial_basis(fold.doy_val.astype(np.float32)),
    )


def test_load_panel_folds_include_doy():
    bundle = load_panel()
    fold = next(iter(iter_folds(bundle, val_years=[2020])))

    assert bundle.doy.shape == bundle.y_log.shape
    assert fold.doy_train is not None
    assert fold.doy_val is not None
    assert int(fold.doy_train.min()) >= 1
    assert int(fold.doy_val.max()) <= 366


def test_jax_compat_subprocess_defaults_to_safe_backend():
    env = os.environ.copy()
    env.pop("JAX_PLATFORMS", None)
    env.pop("JAX_PLATFORM_NAME", None)

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from src.modeling.inference import BayesianRung; "
                "import jax; "
                "print(jax.default_backend())"
            ),
        ],
        cwd="/Users/kylechoi/DataTide",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() in {"cpu", "gpu", "tpu"}


def test_configure_platform_env_preserves_explicit_choice(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)

    _jax_compat._configure_platform_env()

    assert os.environ["JAX_PLATFORMS"] == "gpu"


def test_configure_platform_env_normalizes_metal_case(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "metal")
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)

    _jax_compat._configure_platform_env()

    assert os.environ["JAX_PLATFORMS"] == "METAL"
