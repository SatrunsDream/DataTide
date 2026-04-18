"""Pull San Diego County beach advisory aggregates (Socrata SODA JSON) into data/raw/sd_county_beach/."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.fetch_common import http_get_text, load_fetch_config, raw_out_dir, write_json  # noqa: E402


def main() -> None:
    cfg = load_fetch_config()
    sec = cfg["sd_county"]
    limit = int(sec.get("soda_limit", 50000))
    base = sec["soda_resource"]
    sep = "&" if "?" in base else "?"
    url = f"{base}{sep}{urlencode({'$limit': limit})}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = raw_out_dir(cfg, sec["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    text = http_get_text(url, cfg=cfg)
    dest = out_root / f"beach_advisories_{stamp}.json"
    dest.write_text(text, encoding="utf-8")
    try:
        rows = json.loads(text)
        n = len(rows) if isinstance(rows, list) else "?"
    except json.JSONDecodeError:
        n = "invalid json"
    write_json(out_root / f"sd_county_{stamp}_meta.json", {"url": url, "rows": n})
    print(f"[ok] SD County Socrata: {dest.name} (rows={n})")


if __name__ == "__main__":
    main()
