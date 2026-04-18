"""Download CCE / OceanSITES NetCDF files from NDBC THREDDS fileServer (Tier 4 background)."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.fetch_common import load_fetch_config, raw_out_dir, write_json  # noqa: E402

_BASE = "https://dods.ndbc.noaa.gov/thredds/fileServer/oceansites/DATA/CCE1"


def _bundles(sec: dict) -> list[tuple[str, str]]:
    rows = sec.get("bundles")
    if rows:
        out: list[tuple[str, str]] = []
        for b in rows:
            url = b["nc_url"]
            name = b.get("name") or Path(url.split("?", 1)[0]).stem
            out.append((name, url))
        return out
    legacy = sec.get("nc_url")
    if legacy:
        stem = Path(legacy.split("?", 1)[0]).stem
        return [(stem, legacy)]
    return []


def main() -> None:
    cfg = load_fetch_config()
    sec = cfg["cce_moorings"]
    pairs = _bundles(sec)
    if not pairs:
        print("[skip] cce_moorings: set bundles[] or nc_url in configs/fetch.yaml")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = raw_out_dir(cfg, sec["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    timeout = float(cfg["fetch"]["http_timeout_sec"])
    meta_rows: list[dict] = []

    for name, url in pairs:
        fname = Path(url.split("?", 1)[0]).name
        dest = out_root / f"{fname.rsplit('.', 1)[0]}_{stamp}.nc"
        req = Request(url, headers={"User-Agent": cfg["fetch"]["user_agent"]})
        with urlopen(req, timeout=timeout) as resp, dest.open("wb") as f:
            shutil.copyfileobj(resp, f)
        sz = dest.stat().st_size
        meta_rows.append({"name": name, "url": url, "file": dest.name, "bytes": sz})
        print(f"[ok] CCE {name}: {dest.name} ({sz} bytes)")

    write_json(
        out_root / f"cce_{stamp}_meta.json",
        {
            "bundles": meta_rows,
            "note": "Tier 4 offshore mooring context — not a nearshore beach spine.",
        },
    )


if __name__ == "__main__":
    main()
