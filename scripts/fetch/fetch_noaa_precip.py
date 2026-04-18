"""Fetch NOAA NCEI CDO v2 data (daily PRCP example). Requires NOAA_CDO_TOKEN env var."""

from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.fetch_common import fetch_range_dates, load_fetch_config, raw_out_dir, write_json  # noqa: E402


def _yearly_ranges(start_s: str, end_s: str) -> list[tuple[str, str]]:
    """CDO /data rejects ranges ≥1 year; return inclusive calendar slices within each year."""
    start = date.fromisoformat(start_s)
    end = date.fromisoformat(end_s)
    out: list[tuple[str, str]] = []
    for y in range(start.year, end.year + 1):
        a = date(y, 1, 1) if y > start.year else start
        b = date(y, 12, 31) if y < end.year else end
        if a <= b:
            out.append((a.isoformat(), b.isoformat()))
    return out


def _precip_stations(sec: dict) -> list[tuple[str, str]]:
    rows = sec.get("stations")
    if rows:
        return [(r["name"], r["stationid"]) for r in rows]
    legacy = sec.get("stationid")
    if legacy:
        slug = legacy.replace(":", "_").replace("GHCND_", "").lower()
        return [(slug, legacy)]
    return []


def _cdo_get(req: Request, timeout: float, max_retries: int) -> str:
    """GET with retries for transient CDO failures (503, 429, timeouts)."""
    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except HTTPError as e:
            last_err = e
            if e.code in (429, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise
        except URLError as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise
    raise last_err  # pragma: no cover


def main() -> None:
    token = os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        print("[skip] NOAA_CDO_TOKEN not set — register at https://www.ncdc.noaa.gov/cdo-web/token")
        return

    cfg = load_fetch_config()
    sec = cfg["noaa_precip"]
    st_list = _precip_stations(sec)
    if not st_list:
        print("[skip] noaa_precip: set stations[] or stationid in configs/fetch.yaml")
        return

    start, end = fetch_range_dates(cfg)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = raw_out_dir(cfg, sec["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    timeout = float(cfg["fetch"]["http_timeout_sec"])
    delay = float(sec.get("request_delay_sec", 0.0))
    max_retries = int(sec.get("max_retries", 5))
    written: list[str] = []
    for st_name, stationid in st_list:
        for y_start, y_end in _yearly_ranges(start, end):
            params = {
                "datasetid": sec["datasetid"],
                "stationid": stationid,
                "startdate": y_start,
                "enddate": y_end,
                "datatypeid": sec["datatypeid"],
                "limit": sec.get("limit", 1000),
                "offset": 1,
            }
            url = f"{sec['cdo_data_url']}?{urlencode(params)}"
            req = Request(url, headers={"token": token, "User-Agent": cfg["fetch"]["user_agent"]})
            body = _cdo_get(req, timeout, max_retries)

            dest = out_root / (
                f"cdo_{sec['datatypeid']}_{stationid.replace(':', '_')}_{st_name}_{y_start}_{y_end}_{stamp}.json"
            )
            dest.write_text(body, encoding="utf-8")
            written.append(dest.name)
            print(f"[ok] CDO PRCP {st_name} {y_start}..{y_end} -> {dest.name}")
            if delay > 0:
                time.sleep(delay)

    write_json(
        out_root / f"cdo_{stamp}_meta.json",
        {
            "range": {"start": start, "end": end},
            "stations": [name for name, _ in st_list],
            "files": written,
            "note": "One CDO request per station per calendar year (< 1 year limit). Raise limit if needed.",
        },
    )
    print(f"[ok] NOAA CDO: {len(written)} file(s) ({len(st_list)} station(s)) under {out_root.name}/")


if __name__ == "__main__":
    main()
