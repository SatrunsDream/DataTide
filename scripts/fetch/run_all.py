"""
Run all fetch scripts sequentially from repo root.

Usage (from repo root):
  python scripts/fetch/run_all.py
  python scripts/fetch/run_all.py --only fetch_noaa_tides.py fetch_sd_county.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ORDER = [
    "fetch_ca_swrcb.py",
    "fetch_noaa_tides.py",
    "fetch_noaa_precip.py",
    "fetch_cdip.py",
    "fetch_sccoos_erddap.py",
    "fetch_hf_radar.py",
    "fetch_cce_moorings.py",
    "fetch_surfrider_bwtf.py",
    "fetch_sd_county.py",
    "fetch_ibwc.py",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        nargs="*",
        metavar="SCRIPT",
        help=f"Subset of {DEFAULT_ORDER}",
    )
    args = p.parse_args()
    scripts = args.only if args.only else DEFAULT_ORDER
    here = Path(__file__).resolve().parent
    for name in scripts:
        path = here / name
        if not path.is_file():
            print(f"[skip] missing {name}")
            continue
        print(f"\n=== {name} ===")
        r = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
        if r.returncode != 0:
            print(f"[warn] {name} exited {r.returncode}")


if __name__ == "__main__":
    main()
