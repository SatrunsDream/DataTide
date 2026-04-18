"""
Download Tier 1 California beach water quality CSVs (State Water Board / data.ca.gov).

Uses direct download URLs (streaming) — bacteria file is ~1.6GB; see http_timeout_sec_large in fetch.yaml.
Fallback: CKAN package_show on ckan_action_url if direct_url is null.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.fetch_common import (  # noqa: E402
    ckan_package_show,
    ckan_resource_urls,
    fetch_range_dates,
    load_fetch_config,
    pick_resource_url,
    raw_out_dir,
    stream_download,
    write_json,
)


def main() -> None:
    cfg = load_fetch_config()
    section = cfg.get("ca_swrcb") or {}
    action_url = section["ckan_action_url"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    large_timeout = float(cfg["fetch"].get("http_timeout_sec_large", cfg["fetch"]["http_timeout_sec"]))

    for res in section.get("resources", []):
        key = res["key"]
        out_rel = res["output_dir"]
        out_dir = raw_out_dir(cfg, out_rel)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / f"fetch_meta_{stamp}.json"

        direct = res.get("direct_url")
        if direct:
            ext = ".csv" if "csv" in direct.lower() else ".bin"
            dest = out_dir / f"download_{stamp}{ext}"
            timeout = large_timeout if key == "bacteria" else float(cfg["fetch"]["http_timeout_sec"])
            try:
                n = stream_download(direct, dest, cfg, timeout=timeout)
            except (OSError, HTTPError) as e:
                write_json(meta_path, {"success": False, "error": str(e), "url": direct})
                print(f"[fail] {key}: {e}")
                continue
            write_json(meta_path, {"source": "direct_url", "url": direct, "key": key, "bytes": n})
            print(f"[ok] {key}: {dest.name} ({n} bytes)")
            continue

        pkg_id = res["package_id"]
        try:
            result = ckan_package_show(action_url, pkg_id, cfg)
        except HTTPError as e:
            write_json(
                meta_path,
                {
                    "success": False,
                    "error": str(e),
                    "package_id": pkg_id,
                    "hint": "Set direct_url in configs/fetch.yaml (data.ca.gov resource download link)",
                },
            )
            print(f"[fail] {key}: CKAN package_show HTTP error {e.code} — see {meta_path}")
            continue

        if not result.get("success"):
            write_json(meta_path, {"success": False, "ckan_response": result, "package_id": pkg_id})
            print(f"[fail] {key}: CKAN success=false — see {meta_path}")
            continue

        urls = ckan_resource_urls(result)
        url = pick_resource_url(urls, "CSV")
        if not url:
            write_json(
                meta_path,
                {"success": False, "resources_found": urls, "hint": "Set direct_url in configs/fetch.yaml"},
            )
            print(f"[fail] {key}: no downloadable URL — see {meta_path}")
            continue

        ext = Path(url.split("?", 1)[0]).suffix or ".bin"
        if ext not in (".csv", ".json", ".zip", ".geojson"):
            ext = ".bin"
        dest = out_dir / f"package_{pkg_id[:8]}_{stamp}{ext}"
        timeout = large_timeout if "monitoring-results" in url.lower() or key == "bacteria" else float(
            cfg["fetch"]["http_timeout_sec"]
        )
        n = stream_download(url, dest, cfg, timeout=timeout)
        write_json(
            meta_path,
            {
                "success": True,
                "package_id": pkg_id,
                "resource_url": url,
                "bytes": n,
                "fetch_window": fetch_range_dates(cfg),
            },
        )
        print(f"[ok] {key}: {dest.name} ({n} bytes)")


if __name__ == "__main__":
    main()
