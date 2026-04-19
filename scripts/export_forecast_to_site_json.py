#!/usr/bin/env python3
"""Export `winner_next_week_forecast.parquet` (or any tidy forecast from export_forecast_bundle) to JSON for the static site.

Usage:
  python scripts/export_forecast_to_site_json.py \\
    --forecast artifacts/modeling/winner_next_week_forecast.parquet \\
    --out site/public/data/datatide_forecast.json

Optional:
  --station-labels artifacts/data/panel/station_labels.json
      Merge lat/lon by station_id (string key in labels file).

The site loads `/data/datatide_forecast.json` at runtime (no API key). Re-run after each predict notebook export.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _risk_from_p(p: float) -> tuple[str, str, str]:
    """Return (status, statusText) style for UI; matches site thresholds ~ mock."""
    if p < 0.2:
        return "safe", "safe", "Safe to swim"
    if p < 0.5:
        return "caution", "caution", "Exercise caution"
    return "avoid", "avoid", "Likely unsafe"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--forecast",
        type=Path,
        default=REPO / "artifacts" / "modeling" / "winner_next_week_forecast.parquet",
        help="Tidy forecast parquet from export_forecast_bundle",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "site" / "public" / "data" / "datatide_forecast.json",
    )
    ap.add_argument(
        "--station-labels",
        type=Path,
        default=REPO / "artifacts" / "data" / "panel" / "station_labels.json",
        help="Optional JSON {station_id_str: {lat, lon, name?, county?}}",
    )
    args = ap.parse_args()

    if not args.forecast.exists():
        raise SystemExit(f"forecast file not found: {args.forecast}")

    df = pd.read_parquet(args.forecast)
    required = {"station_id", "station_name", "date", "p_exceed_104mpn", "point_mpn_median", "point_log10_median"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"forecast parquet missing columns: {sorted(missing)}")

    labels: dict = {}
    if args.station_labels.exists():
        with open(args.station_labels, encoding="utf-8") as fh:
            raw = json.load(fh)
            # normalize keys to str
            labels = {str(k): v for k, v in raw.items()}

    # one row per station × date → group by station
    df = df.copy()
    df["station_id_str"] = df["station_id"].astype(str)
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    beaches: list[dict] = []
    for sid, g in df.groupby("station_id_str", sort=False):
        g = g.sort_values("date")
        first = g.iloc[0]
        p_today = float(first["p_exceed_104mpn"])
        status, risk_key, status_text = _risk_from_p(p_today)

        geo = labels.get(sid, {})
        lat = geo.get("lat")
        lng = geo.get("lon")

        forecast_rows = []
        for _, row in g.iterrows():
            p = float(row["p_exceed_104mpn"])
            st, _, _ = _risk_from_p(p)
            lo_mpn = row.get("pi95_low_mpn")
            hi_mpn = row.get("pi95_high_mpn")
            pred_p05 = pred_p95 = None
            if pd.notna(lo_mpn) and float(lo_mpn) > 0:
                pred_p05 = round(math.log10(float(lo_mpn)), 3)
            if pd.notna(hi_mpn) and float(hi_mpn) > 0:
                pred_p95 = round(math.log10(float(hi_mpn)), 3)
            forecast_rows.append(
                {
                    "day": pd.Timestamp(row["date"]).strftime("%a"),
                    "date": row["date_str"],
                    "status": st,
                    "pExceedance": p,
                    "predP50": float(row["point_log10_median"]),
                    "linearP50": float(row["point_mpn_median"]),
                    "predP05": pred_p05,
                    "predP95": pred_p95,
                }
            )

        beach = {
            "id": sid,
            "name": str(first["station_name"]),
            "location": str(first.get("county_name", "")),
            "distance": "—",
            "status": risk_key,
            "pExceedance": p_today,
            "predP50": float(first["point_log10_median"]),
            "linearP50": float(first["point_mpn_median"]),
            "confidenceTier": "high",
            "asOfTime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "statusText": status_text,
            "reason": f"Model estimate: {p_today:.0%} chance a single sample exceeds 104 MPN (advisory threshold).",
            "lastSample": "—",
            "forecast": forecast_rows,
            "imageUrl": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1200&q=80",
            "coordinates": {"lat": float(lat) if lat is not None else 34.05, "lng": float(lng) if lng is not None else -118.25},
            "shorelineBearing": 270,
            "officialAdvisoryUrl": "https://lab.data.ca.gov/dataset/beach-water-quality",
            "dataOriginUrl": "https://lab.data.ca.gov/dataset/beach-water-quality",
        }
        beaches.append(beach)

    try:
        src_rel = str(args.forecast.resolve().relative_to(REPO))
    except ValueError:
        src_rel = str(args.forecast)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_parquet": src_rel,
        "threshold_mpn": 104.0,
        "n_stations": len(beaches),
        "beaches": beaches,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}  ({len(beaches)} stations)")


if __name__ == "__main__":
    main()
