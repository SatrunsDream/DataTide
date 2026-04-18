"""HF radar surface currents via NOAA/IOOS ERDDAP griddap (Tier 3).

The ucsdHfr* products on coastwatch.pfeg.noaa.gov expose a **rolling near-real-time**
time window (not full2010+ history). Pulls are suitable for recent coverage and ops;
historic HF radar for modeling may require separate CORDC archives.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.fetch_common import load_fetch_config, raw_out_dir, stream_download, write_json  # noqa: E402


def _griddap_time_epoch_range(erddap_base: str, dataset_id: str, cfg: dict, timeout: float) -> tuple[int, int]:
    url = f"{erddap_base.rstrip('/')}/info/{dataset_id}/index.csv"
    req = Request(url, headers={"User-Agent": cfg["fetch"]["user_agent"]})
    with urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("attribute,time,actual_range"):
            rest = line.split(",double,", 1)[1].strip().strip('"')
            parts = rest.replace('"', "").split(",")
            lo = int(float(parts[0].strip()))
            hi = int(float(parts[1].strip()))
            return lo, hi
    raise ValueError(f"Could not parse time actual_range for {dataset_id}")


def main() -> None:
    cfg = load_fetch_config()
    sec = cfg["hf_radar"]
    dataset_id = sec.get("griddap_id") or sec.get("tabledap_id")
    if not dataset_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_root = raw_out_dir(cfg, sec["output_dir"])
        out_root.mkdir(parents=True, exist_ok=True)
        write_json(
            out_root / f"README_{stamp}.json",
            {
                "status": "not_configured",
                "hint": "Set hf_radar.griddap_id (e.g. ucsdHfrW6) in configs/fetch.yaml",
            },
        )
        print("[skip] hf_radar: set griddap_id in configs/fetch.yaml")
        return

    erddap_base = sec.get("erddap_base", "https://coastwatch.pfeg.noaa.gov/erddap")
    lat_lo = float(sec["lat_min"])
    lat_hi = float(sec["lat_max"])
    lon_lo = float(sec["lon_min"])
    lon_hi = float(sec["lon_max"])
    tstride = int(sec.get("time_stride_seconds", 21600))
    variables = sec.get("variables") or ["water_u", "water_v"]

    timeout = float(cfg["fetch"]["http_timeout_sec"])
    t0, t1 = _griddap_time_epoch_range(erddap_base, dataset_id, cfg, timeout)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = raw_out_dir(cfg, sec["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    pulled: list[dict] = []
    for var in variables:
        url = (
            f"{erddap_base.rstrip('/')}/griddap/{dataset_id}.csv"
            f"?{var}[({t0}):{tstride}:({t1})]"
            f"[({lat_lo}):1:({lat_hi})][({lon_lo}):1:({lon_hi})]"
        )
        dest = out_root / f"hf_{dataset_id}_{var}_{stamp}.csv"
        try:
            n = stream_download(url, dest, cfg, timeout=timeout)
        except HTTPError as e:
            print(f"[fail] HF radar {var}: HTTP {e.code}")
            raise
        pulled.append({"variable": var, "file": dest.name, "bytes": n})
        print(f"[ok] HF radar {dataset_id} {var}: {dest.name} ({n} bytes)")

    write_json(
        out_root / f"hf_radar_{stamp}_meta.json",
        {
            "griddap_id": dataset_id,
            "time_epoch_range": [t0, t1],
            "bbox": {"lat": [lat_lo, lat_hi], "lon": [lon_lo, lon_hi]},
            "time_stride_seconds": tstride,
            "downloads": pulled,
            "note": "Rolling NRT window on IOOS ERDDAP; not a full historical archive.",
        },
    )
    print(f"[ok] HF radar: {len(pulled)} file(s) under {out_root.name}/")


if __name__ == "__main__":
    main()
