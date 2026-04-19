"""Emit `site/public/data/surf_spots.json` from `artifacts/data/panel/station_labels.json`.

Run from repo root:
  python scripts/build_surf_spots_json.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "artifacts" / "data" / "panel" / "station_labels.json"
OUT = REPO / "site" / "public" / "data" / "surf_spots.json"


def main() -> None:
    if not LABELS.exists():
        raise SystemExit(f"missing {LABELS}")
    raw = json.loads(LABELS.read_text(encoding="utf-8"))
    spots = []
    for sid, row in raw.items():
        lat, lon = row.get("lat"), row.get("lon")
        if lat is None or lon is None:
            continue
        spots.append(
            {
                "id": str(sid),
                "name": str(row.get("name", "Spot")),
                "county": str(row.get("county", row.get("county_nominatim", ""))),
                "lat": float(lat),
                "lng": float(lon),
                "shorelineBearing": 270,
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": "artifacts/data/panel/station_labels.json", "n": len(spots), "spots": spots}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(spots)} spots)")


if __name__ == "__main__":
    main()
