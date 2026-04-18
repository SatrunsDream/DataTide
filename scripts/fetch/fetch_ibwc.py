"""Attempt IBWC legacy gauge text; on failure write meta for manual download."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.fetch_common import http_get_text, load_fetch_config, raw_out_dir, write_json  # noqa: E402


def main() -> None:
    cfg = load_fetch_config()
    sec = cfg["ibwc"]
    url = sec.get("legacy_gauge_text_url")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = raw_out_dir(cfg, sec["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        text = http_get_text(url, cfg=cfg)
        (out_root / f"tijuana_gauge_{stamp}.txt").write_text(text, encoding="utf-8", errors="replace")
        write_json(out_root / f"ibwc_{stamp}_meta.json", {"url": url, "success": True})
        print(f"[ok] IBWC: saved text ({len(text)} chars)")
    except HTTPError as e:
        write_json(
            out_root / f"ibwc_{stamp}_meta.json",
            {
                "url": url,
                "success": False,
                "http_status": e.code,
                "note": "Use waterdata.ibwc.gov or updated gauge endpoint; verify in browser.",
            },
        )
        print(f"[fail] IBWC HTTP {e.code} — see meta JSON; manual pull may be required")


if __name__ == "__main__":
    main()
