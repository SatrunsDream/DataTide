"""Backward-compatible entry point — builds the **full** ground-truth Parquet.

Prefer: `python scripts/process/build_ground_truth_dataset.py`
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _main():
    print("[note] Prefer scripts/process/build_ground_truth_dataset.py (same build, clearer name).")
    path = Path(__file__).resolve().parent / "build_ground_truth_dataset.py"
    spec = importlib.util.spec_from_file_location("build_ground_truth_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    _main()
