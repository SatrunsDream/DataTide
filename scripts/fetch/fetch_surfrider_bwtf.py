"""Fetch Surfrider Blue Water Task Force samples via AWS AppSync GraphQL (Tier 4).

The public web app authenticates with Amazon Cognito. The bundled API key does **not**
authorize `listWaterQualityDatas`. Set **BWTF_COGNITO_ID_TOKEN** in `.env` to a valid
**Cognito IdToken** (from browser devtools → Application → Local Storage after logging in
at https://bwtf.surfrider.org/ ). Tokens expire (~1h); refresh and re-run for long pulls.

Data are supplementary community Enterococcus / E.coli context — not the statewide Tier 1 spine.
"""

from __future__ import annotations

import json
import os
import sys
import time
from calendar import monthrange
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.fetch_common import fetch_range_dates, load_fetch_config, raw_out_dir, write_json  # noqa: E402

LIST_QUERY = """
query ListBWTF($limit: Int!, $nextToken: String, $filter: ModelWaterQualityDataFilterInput) {
  listWaterQualityDatas(filter: $filter, limit: $limit, nextToken: $nextToken) {
    items {
      id
      collectionTime
      labID
      siteID
      author
      comments
      organizationName
      publicationTime
      testedBy
      samples {
        collectionTime
        method
        modifier
        result
        substance
        units
      }
      location {
        id
        name
        labID
        coordinate { latitude longitude }
      }
      weather {
        airTemperature
        waterTemperature
        waveHeight
        tide
      }
    }
    nextToken
  }
}
"""


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


def _post_graphql(url: str, token: str, payload: dict, timeout: float, ua: str) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": token,
            "User-Agent": ua,
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    token = (os.environ.get("BWTF_COGNITO_ID_TOKEN") or "").strip()
    cfg = load_fetch_config()
    sec = cfg["surfrider_bwtf"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = raw_out_dir(cfg, sec["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    if not token:
        write_json(
            out_root / f"README_{stamp}.json",
            {
                "status": "auth_required",
                "portal": sec.get("portal_url"),
                "env": "BWTF_COGNITO_ID_TOKEN",
                "how_to": (
                    "Log in at bwtf.surfrider.org, open DevTools → Application → Local Storage, "
                    "copy the Cognito IdToken (JWT) into .env, then re-run this script."
                ),
            },
        )
        print("[skip] surfrider_bwtf: set BWTF_COGNITO_ID_TOKEN in .env (see README json)")
        return

    gql_url = sec.get(
        "graphql_url",
        "https://esvbxbhkmzgh5ojw3g2hvsx7du.appsync-api.us-west-2.amazonaws.com/graphql",
    )
    page_size = int(sec.get("page_size", 200))
    delay = float(sec.get("request_delay_sec", 0.35))
    start_s, end_s = fetch_range_dates(cfg)
    range_start = date.fromisoformat(start_s)
    range_end = date.fromisoformat(end_s)
    timeout = float(cfg["fetch"]["http_timeout_sec"])
    ua = cfg["fetch"]["user_agent"]

    total_items = 0
    shard_meta: list[dict] = []

    for a, b in _month_slices(range_start, range_end):
        ymtag = f"{a.year:04d}{a.month:02d}"
        t0 = f"{a.isoformat()}T00:00:00.000Z"
        t1 = f"{b.isoformat()}T23:59:59.999Z"
        filt = {"collectionTime": {"between": [t0, t1]}}
        next_token: str | None = None
        month_items: list[dict] = []

        while True:
            variables: dict = {"limit": page_size, "filter": filt}
            if next_token:
                variables["nextToken"] = next_token
            try:
                data = _post_graphql(
                    gql_url,
                    token,
                    {"query": LIST_QUERY, "variables": variables},
                    timeout,
                    ua,
                )
            except HTTPError as e:
                msg = e.read().decode("utf-8", errors="replace")[:500]
                print(f"[fail] BWTF {ymtag}: HTTP {e.code} {msg}")
                raise

            if data.get("errors"):
                print(f"[fail] BWTF {ymtag}: {data['errors'][:2]}")
                raise SystemExit(1)

            conn = data.get("data", {}).get("listWaterQualityDatas") or {}
            batch = conn.get("items") or []
            month_items.extend(batch)
            total_items += len(batch)
            next_token = conn.get("nextToken")
            if not next_token:
                break
            if delay > 0:
                time.sleep(delay)

        if month_items:
            dest = out_root / f"bwtf_water_quality_{ymtag}_{stamp}.json"
            dest.write_text(json.dumps(month_items, indent=2), encoding="utf-8")
            shard_meta.append({"month": ymtag, "file": dest.name, "rows": len(month_items)})
            print(f"[ok] surfrider_bwtf {ymtag}: {len(month_items)} row(s) -> {dest.name}")
        else:
            print(f"[--] surfrider_bwtf {ymtag}: no rows")

    write_json(
        out_root / f"bwtf_{stamp}_meta.json",
        {
            "range": [start_s, end_s],
            "graphql_url": gql_url,
            "shards": shard_meta,
            "total_rows": total_items,
        },
    )
    print(f"[ok] surfrider_bwtf: {total_items} total row(s) in {len(shard_meta)} month file(s)")


if __name__ == "__main__":
    main()
