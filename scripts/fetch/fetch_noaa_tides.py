"""
Fetch NOAA CO-OPS tide **predictions** per station as **high/low only** (`interval=hilo`).

Aligned with **daily** site-day features (not hourly): ~4 tide extrema per day. Yearly chunks avoid
CO-OPS range limits and keep runs fast. When you operate **week-by-week**, refresh or extend these
series in batches; aggregation still rolls up to `(station, date)` for the model.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.fetch_common import fetch_range_dates, http_get_text, load_fetch_config, raw_out_dir, write_json  # noqa: E402


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def year_segments(range_start: date, range_end: date) -> list[tuple[date, date]]:
    """Non-overlapping [lo, hi] segments per calendar year within [range_start, range_end]."""
    out: list[tuple[date, date]] = []
    y = range_start.year
    y_end = range_end.year
    while y <= y_end:
        lo = date(y, 1, 1)
        hi = date(y, 12, 31)
        seg_lo = max(lo, range_start)
        seg_hi = min(hi, range_end)
        if seg_lo <= seg_hi:
            out.append((seg_lo, seg_hi))
        y += 1
    return out


def main() -> None:
    cfg = load_fetch_config()
    sec = cfg["noaa_tides"]
    start_s, end_s = fetch_range_dates(cfg)
    range_start = date.fromisoformat(start_s)
    range_end = date.fromisoformat(end_s)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = raw_out_dir(cfg, sec["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    segments = year_segments(range_start, range_end)

    for st in sec["stations"]:
        sid = st["id"]
        name = st["name"]
        meta_chunks: list[dict] = []
        for seg_lo, seg_hi in segments:
            params = {
                "product": sec["product"],
                "application": "DataTide",
                "begin_date": _ymd(seg_lo),
                "end_date": _ymd(seg_hi),
                "datum": sec["datum"],
                "station": sid,
                "time_zone": "gmt",
                "units": sec["units"],
                "interval": sec["interval"],
                "format": "json",
            }
            url = f"{sec['base_url']}?{urlencode(params)}"
            text = http_get_text(url, cfg=cfg)
            ytag = seg_lo.year
            dest = out_root / f"tides_hilo_{name}_{sid}_{ytag}_{stamp}.json"
            dest.write_text(text, encoding="utf-8")
            meta_chunks.append({"year": ytag, "file": dest.name, "url": url})
            print(f"[ok] NOAA tides {name} ({sid}) {ytag}: {dest.name}")
        write_json(
            out_root / f"tides_{name}_{sid}_{stamp}_meta.json",
            {"station_id": sid, "station_name": name, "range": [start_s, end_s], "chunks": meta_chunks},
        )


if __name__ == "__main__":
    main()
