"""Orchestrate the modeling pipeline from processed ground truth to model artifacts.

This runner bridges the current notebook-driven workflow:

1. Confirm `data/processed/datatide_ground_truth.parquet` exists.
2. Execute `notebooks/modeling/eda.ipynb` to rebuild panel artifacts.
3. Regenerate `notebooks/modeling/model.ipynb` from the scripted template.
4. Execute `notebooks/modeling/model.ipynb` to fit/evaluate models and export results.

Usage:
    python -m src.modeling.run_pipeline
    python -m src.modeling.run_pipeline --skip-eda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir() and (parent / "notebooks").is_dir():
            return parent
    return here.parents[2]


ROOT = _repo_root()
NOTEBOOK_DIR = ROOT / "notebooks" / "modeling"
GROUND_TRUTH = ROOT / "data" / "processed" / "datatide_ground_truth.parquet"
EDA_NOTEBOOK = NOTEBOOK_DIR / "eda.ipynb"
MODEL_NOTEBOOK = NOTEBOOK_DIR / "model.ipynb"
EDA_OUTPUT = "eda.executed.ipynb"
MODEL_OUTPUT = "model.executed.ipynb"


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    env.setdefault("JAX_PLATFORMS", "cpu")
    return env


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    printable = " ".join(cmd)
    print(f"[run] {printable}")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def _execute_notebook(notebook: Path, output_name: str, *, timeout: int, env: dict[str, str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook.relative_to(ROOT)),
        "--output",
        output_name,
        f"--ExecutePreprocessor.timeout={timeout}",
    ]
    _run(cmd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DataTide modeling pipeline.")
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip executing notebooks/modeling/eda.ipynb and reuse existing panel artifacts.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip executing notebooks/modeling/model.ipynb.",
    )
    parser.add_argument(
        "--eda-timeout",
        type=int,
        default=900,
        help="Notebook execution timeout in seconds for the EDA step.",
    )
    parser.add_argument(
        "--model-timeout",
        type=int,
        default=0,
        help="Notebook execution timeout in seconds for the modeling step (0 = unlimited).",
    )
    args = parser.parse_args()

    if not GROUND_TRUTH.is_file():
        raise SystemExit(
            f"Missing processed ground truth: {GROUND_TRUTH}\n"
            "Build it first with scripts/process/build_ground_truth_dataset.py."
        )

    env = _base_env()
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    if not args.skip_eda:
        _execute_notebook(EDA_NOTEBOOK, EDA_OUTPUT, timeout=args.eda_timeout, env=env)

    _run([sys.executable, str((ROOT / "scripts" / "build_model_notebook.py").relative_to(ROOT))], env=env)

    if not args.skip_model:
        _execute_notebook(MODEL_NOTEBOOK, MODEL_OUTPUT, timeout=args.model_timeout, env=env)

    print("[ok] pipeline complete")
    print(f"[ok] ground truth   -> {GROUND_TRUTH.relative_to(ROOT)}")
    print(f"[ok] panel dir      -> {(ROOT / 'artifacts' / 'data' / 'panel').relative_to(ROOT)}")
    print(f"[ok] model outputs  -> {(ROOT / 'artifacts' / 'modeling').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
