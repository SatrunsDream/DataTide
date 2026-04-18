"""Pull SCCOOS ERDDAP tabledap datasets (Tier 3) as monthly CSV shards."""

from __future__ import annotations

import sys
from calendar import monthrange
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.fetch_common import (  # noqa: E402
    fetch_range_dates,
    load_fetch_config,
    raw_out_dir,
    stream_download,
    write_json,
)


def _month_slices(range_start: date, range_end: date):
    y, m = range_start.year, range_start.month
    while date(y, m, 1) <= range_end:
        lo = date(y, m, 1)
        hi = date(y, m, monthrange(y, m)[1])
        a = max(lo, range_start)
        b = min(hi, range_end)
        if a <= b:
            yield a, b
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


def _iso_day(d: date, *, end_of_day: bool = False) -> str:
    if end_of_day:
        return f"{d.isoformat()}T23:59:59Z"
    return f"{d.isoformat()}T00:00:00Z"


def main() -> None:
    cfg = load_fetch_config()
    sec = cfg["sccoos_erddap"]
    datasets = sec.get("datasets") or []
    if not datasets:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_root = raw_out_dir(cfg, sec["output_dir"])
        out_root.mkdir(parents=True, exist_ok=True)
        write_json(
            out_root / f"README_{stamp}.json",
            {
                "status": "not_configured",
                "hint": "Add sccoos_erddap.datasets[] in configs/fetch.yaml (see Tier 3 example).",
            },
        )
        print("[skip] sccoos_erddap: add datasets[] in configs/fetch.yaml")
        return

    start_s, end_s = fetch_range_dates(cfg)
    range_start = date.fromisoformat(start_s)
    range_end = date.fromisoformat(end_s)
    base = sec["base_url"].rstrip("/")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = raw_out_dir(cfg, sec["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    pulled: list[dict] = []
    for ds in datasets:
        tid = ds["tabledap_id"]
        name = ds.get("name") or tid.replace("-", "_").lower()
        vars_ = ds.get("variables") or ["time"]
        var_q = ",".join(vars_)

        for a, b in _month_slices(range_start, range_end):
            ymtag = f"{a.year:04d}{a.month:02d}"
            dest = out_root / f"sccoos_{name}_{ymtag}_{stamp}.csv"
            url = (
                f"{base}/tabledap/{tid}.csv"
                f"?{var_q}"
                f"&time>={_iso_day(a)}"
                f"&time<={_iso_day(b, end_of_day=True)}"
            )
            try:
                n = stream_download(url, dest, cfg, timeout=float(cfg["fetch"]["http_timeout_sec"]))
            except HTTPError as e:
                if e.code == 404:
                    print(f"[warn] {name} {ymtag}: no data (404) — check sensor dates vs fetch range")
                    if dest.is_file():
                        dest.unlink(missing_ok=True)
                    continue
                raise
            pulled.append({"dataset": name, "month": ymtag, "file": dest.name, "bytes": n})
            print(f"[ok] SCCOOS {name} {ymtag}: {dest.name} ({n} bytes)")

    write_json(
        out_root / f"sccoos_erddap_{stamp}_meta.json",
        {"range": [start_s, end_s], "downloads": pulled},
    )
    print(f"[ok] SCCOOS ERDDAP: {len(pulled)} shard(s) under {out_root.name}/")


if __name__ == "__main__":
    main()
