"""Stream-download one CDIP NetCDF (can be large) into data/raw/cdip/."""

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


def _bundles(sec: dict) -> list[tuple[str, str]]:
    raw = sec.get("bundles")
    if raw:
        return [(b["name"], b["opendap_nc_url"]) for b in raw]
    legacy = sec.get("opendap_nc_url")
    if legacy:
        fname = Path(legacy.split("?", 1)[0]).stem or "cdip"
        return [(fname, legacy)]
    return []


def main() -> None:
    cfg = load_fetch_config()
    sec = cfg["cdip"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = raw_out_dir(cfg, sec["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    pairs = _bundles(sec)
    if not pairs:
        print("[skip] cdip: set bundles[] or opendap_nc_url in configs/fetch.yaml")
        return

    timeout = float(cfg["fetch"]["http_timeout_sec"])
    meta_files: list[dict] = []
    for name, url in pairs:
        fname = Path(url.split("?", 1)[0]).name or f"{name}.nc"
        base = fname.rsplit(".", 1)[0]
        dest = out_root / f"{base}_{stamp}.nc"
        req = Request(url, headers={"User-Agent": cfg["fetch"]["user_agent"]})
        with urlopen(req, timeout=timeout) as resp, dest.open("wb") as f:
            shutil.copyfileobj(resp, f)
        sz = dest.stat().st_size
        meta_files.append({"name": name, "url": url, "file": dest.name, "bytes": sz})
        print(f"[ok] CDIP {name}: {dest.name} ({sz} bytes)")

    write_json(
        out_root / f"cdip_{stamp}_meta.json",
        {
            "bundles": meta_files,
            "note": "Realtime files; use CDIP archive THREDDS paths for long historical windows.",
        },
    )


if __name__ == "__main__":
    main()
