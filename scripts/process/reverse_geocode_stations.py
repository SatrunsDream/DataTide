"""Reverse-geocode every distinct (Station_id, lat, lon) in the panel
to a human-readable beach/park/locality name. Result is cached to
artifacts/data/panel/station_labels.json so the EDA notebook can reload
without hitting Nominatim again.

Usage:  python scripts/process/reverse_geocode_stations.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
PANEL_PKL = ROOT / "artifacts/data/panel/enterococcus_panel.pkl"
CACHE = ROOT / "artifacts/data/panel/station_labels.json"

NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
HEADERS = {"User-Agent": "DataTide-EDA/1.0 (reverse-geocode beach stations)"}


def _label_from_nominatim(lat: float, lon: float) -> dict:
    params = {
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "format": "jsonv2",
        "zoom": 17,
        "addressdetails": 1,
    }
    r = requests.get(NOMINATIM, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    j = r.json()
    addr = j.get("address", {})
    primary = (
        addr.get("beach")
        or addr.get("park")
        or addr.get("leisure")
        or addr.get("natural")
        or addr.get("neighbourhood")
        or addr.get("suburb")
        or addr.get("hamlet")
        or addr.get("village")
        or addr.get("town")
        or addr.get("city_district")
        or addr.get("city")
    )
    county = addr.get("county")
    display = j.get("display_name", "")
    return {
        "name": primary or display.split(",")[0],
        "county_nominatim": county,
        "display": display,
    }


def main():
    panel = pd.read_pickle(PANEL_PKL)
    stations = (
        panel[["Station_id", "Station_UpperLat", "Station_UpperLon", "CountyName"]]
        .drop_duplicates("Station_id")
        .sort_values("Station_id")
        .reset_index(drop=True)
    )
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    total = len(stations)
    print(f"stations to label: {total}  (cached: {len(cache)})")

    for i, row in stations.iterrows():
        sid = str(int(row["Station_id"]))
        if sid in cache:
            continue
        lat = float(row["Station_UpperLat"])
        lon = float(row["Station_UpperLon"])
        try:
            info = _label_from_nominatim(lat, lon)
        except Exception as exc:
            info = {"name": None, "error": f"{type(exc).__name__}: {exc}"}
        info["lat"] = lat
        info["lon"] = lon
        info["county"] = row["CountyName"]
        cache[sid] = info
        if i % 25 == 0:
            CACHE.write_text(json.dumps(cache, indent=2))
            print(f"  [{i+1:>3}/{total}] {sid}  {info.get('name','?')}")
        time.sleep(1.05)  # Nominatim ToS: 1 req/sec max

    CACHE.write_text(json.dumps(cache, indent=2))
    ok = sum(1 for v in cache.values() if v.get("name"))
    print(f"done. {ok}/{len(cache)} have names; cache at {CACHE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
