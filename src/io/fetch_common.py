"""Shared helpers for scripts/fetch data downloads."""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml

from src.utils.paths import repo_root


def _load_repo_dotenv() -> None:
    env_path = repo_root() / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


_load_repo_dotenv()


def load_fetch_config(path: str | Path | None = None) -> dict[str, Any]:
    root = repo_root()
    cfg_path = Path(path) if path else root / "configs" / "fetch.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def raw_out_dir(cfg: dict[str, Any], *sub: str) -> Path:
    root = repo_root()
    rel = cfg["fetch"]["output_root"]
    return root.joinpath(rel, *sub)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fetch_range_dates(cfg: dict[str, Any]) -> tuple[str, str]:
    dr = cfg["fetch"]["date_range"]
    start = dr["start"]
    if dr.get("end"):
        end = dr["end"]
    else:
        end = date.today().isoformat()
    return start, end


def http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> bytes:
    if timeout is None:
        timeout = float((cfg or {}).get("fetch", {}).get("http_timeout_sec", 60.0))
    ua = (cfg or {}).get("fetch", {}).get("user_agent", "DataTide/fetch")
    h = {"User-Agent": ua}
    if headers:
        h.update(headers)
    req = Request(url, headers=h)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_get_text(url: str, **kwargs: Any) -> str:
    return http_get(url, **kwargs).decode("utf-8", errors="replace")


def write_bytes(path: Path, data: bytes) -> None:
    ensure_parent(path)
    path.write_bytes(data)


def write_json(path: Path, obj: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def stream_download(url: str, dest: Path, cfg: dict[str, Any], *, timeout: float | None = None) -> int:
    """Stream a URL to disk; returns byte size. Use for large CSV/NetCDF."""
    if timeout is None:
        timeout = float(cfg["fetch"]["http_timeout_sec"])
    ua = cfg["fetch"]["user_agent"]
    req = Request(url, headers={"User-Agent": ua})
    ensure_parent(dest)
    with urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)
    return dest.stat().st_size


def ckan_package_show(action_url: str, package_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """GET CKAN action API package_show."""
    url = f"{action_url.rstrip('/')}/package_show?id={package_id}"
    raw = http_get_text(url, cfg=cfg)
    return json.loads(raw)


def ckan_resource_urls(result: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return list of (resource_id, name, url) from successful package_show."""
    if not result.get("success"):
        return []
    pkg = result["result"]
    out: list[tuple[str, str, str]] = []
    for r in pkg.get("resources", []):
        rid = r.get("id", "")
        name = r.get("name") or r.get("description") or rid
        url = r.get("url") or ""
        if url:
            out.append((rid, name, url))
    return out


def pick_resource_url(
    resources: list[tuple[str, str, str]],
    prefer_format: str | None = "CSV",
) -> str | None:
    if not resources:
        return None
    if prefer_format:
        for _, _, url in resources:
            if url.lower().endswith((".csv", ".zip")):
                return url
    return resources[0][2]
