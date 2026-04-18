"""Build the consolidated **ground-truth** modeling table (Tier-1 spine + joined environmental layers).

Output: `data/processed/datatide_ground_truth.parquet` (see `data/processed/README.md`).

Tier-1: CA SWRCB bacteria monitoring results (+ station supplement from stations export).
Tier-2: regional GHCN daily precipitation, CO-OPS tidal range (hilo), CDIP wave stats (nearest buoy).
Tier-3: SCCOOS Del Mar 1 m T/S daily means (only for configured Southern California counties).
Tier-5: San Diego County coastal program **monthly** advisory aggregates (county-wide; join only matches San Diego).

Not row-merged here (different spatial/temporal grain or auth): HF radar grids, CCE mooring NetCDF,
Surfrider BWTF (Cognito), IBWC — use as separate feature tables or future spatial joins.

Usage (repo root):
  python scripts/process/build_ground_truth_dataset.py
  python scripts/process/build_ground_truth_dataset.py --max-chunks 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.modeling_joins import (  # noqa: E402
    assign_nearest_cdip_bundle,
    load_cdip_buoy_meta_and_daily,
    load_precip_daily,
    load_sccoos_delmar_daily,
    load_sd_county_coastal_monthly,
    load_station_supplement,
    load_tide_daily,
)
from src.utils.paths import repo_root  # noqa: E402


def _latest_match(raw: Path, pattern: str) -> Path | None:
    matches = sorted(raw.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build datatide_ground_truth.parquet from raw downloads.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Stop after N bacteria chunks (testing).")
    args = parser.parse_args()

    cfg_path = repo_root() / "configs" / "process.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fetch_path = repo_root() / "configs" / "fetch.yaml"
    with fetch_path.open(encoding="utf-8") as f:
        fetch_cfg = yaml.safe_load(f)

    raw_root = repo_root() / cfg["paths"]["data_raw"]
    out_root = repo_root() / cfg["paths"]["data_processed"]
    out_root.mkdir(parents=True, exist_ok=True)

    bacteria_path = _latest_match(raw_root, cfg["inputs"]["bacteria_glob"])
    if bacteria_path is None:
        raise SystemExit(f"No bacteria CSV under {raw_root} (glob {cfg['inputs']['bacteria_glob']!r}).")

    cols = list(cfg["bacteria_columns"])
    header = pd.read_csv(bacteria_path, nrows=0).columns.tolist()
    missing = set(cols) - set(header)
    if missing:
        raise SystemExit(f"Bacteria CSV missing columns: {sorted(missing)}")

    county_env = {k: v for k, v in (cfg.get("county_to_env") or {}).items()}
    precip_bucket_by_county = {k: v.get("precip_bucket") for k, v in county_env.items()}
    tide_id_by_county = {
        k: str(v["tide_station_id"]) if v.get("tide_station_id") is not None else None
        for k, v in county_env.items()
    }

    precip = load_precip_daily(raw_root / "noaa_precip")
    tides = load_tide_daily(raw_root / "noaa_tides")
    precip["sample_date"] = precip["sample_date"].astype(str)
    tides["sample_date"] = tides["sample_date"].astype(str)
    tides["tide_station_id"] = tides["tide_station_id"].astype(str)

    sccoos_counties = set(cfg.get("sccoos_join_counties") or [])
    sccoos = load_sccoos_delmar_daily(raw_root / "sccoos_erddap")
    if len(sccoos):
        sccoos["sample_date"] = sccoos["sample_date"].astype(str)

    cdip_names = [b["name"] for b in (fetch_cfg.get("cdip") or {}).get("bundles") or []]
    cdip_meta, cdip_daily = load_cdip_buoy_meta_and_daily(raw_root / "cdip", cdip_names)
    if len(cdip_daily):
        cdip_daily["sample_date"] = cdip_daily["sample_date"].astype(str)
        cdip_daily["cdip_bundle"] = cdip_daily["cdip_bundle"].astype(str)

    stations_path = _latest_match(raw_root, cfg["inputs"]["stations_glob"])
    stn_sup = load_station_supplement(stations_path) if stations_path else pd.DataFrame()

    sd_path_dir = raw_root / "sd_county_beach"
    sd_monthly = load_sd_county_coastal_monthly(sd_path_dir) if sd_path_dir.is_dir() else pd.DataFrame()

    out_parquet = out_root / cfg["output"]["parquet_name"]
    out_meta = out_root / cfg["output"]["meta_name"]
    chunk_rows = int(cfg.get("bacteria_chunk_rows") or 200_000)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("Install pyarrow: pip install pyarrow") from e

    writer = None
    total_rows = 0
    chunks_done = 0

    meta_cdip_buoys = cdip_meta.to_dict(orient="records") if len(cdip_meta) else []

    for chunk in pd.read_csv(
        bacteria_path,
        usecols=lambda c: c in set(cols),
        chunksize=chunk_rows,
        low_memory=False,
    ):
        chunk["sample_date"] = pd.to_datetime(chunk["SampleDate"], errors="coerce").dt.strftime("%Y-%m-%d")
        chunk["CountyName"] = chunk["CountyName"].astype(str).str.strip()
        chunk["precip_bucket"] = chunk["CountyName"].map(precip_bucket_by_county)
        chunk["tide_station_id"] = chunk["CountyName"].map(tide_id_by_county)

        chunk["Station_id"] = pd.to_numeric(chunk["Station_id"], errors="coerce").astype("Int64")
        if len(stn_sup):
            chunk = chunk.merge(stn_sup, on="Station_id", how="left")

        chunk = chunk.merge(precip, on=["sample_date", "precip_bucket"], how="left")
        chunk = chunk.merge(tides, on=["sample_date", "tide_station_id"], how="left")

        if len(sccoos):
            chunk = chunk.merge(sccoos, on="sample_date", how="left")
            mask_scc = ~chunk["CountyName"].isin(sccoos_counties)
            for c in ("sccoos_delmar_temp_1m_c", "sccoos_delmar_salinity_1m_psu"):
                if c in chunk.columns:
                    chunk.loc[mask_scc, c] = np.nan

        if len(cdip_meta) and len(cdip_daily):
            lat = pd.to_numeric(chunk["Station_UpperLat"], errors="coerce").to_numpy(dtype=float)
            lon = pd.to_numeric(chunk["Station_UpperLon"], errors="coerce").to_numpy(dtype=float)
            chunk["cdip_bundle"] = assign_nearest_cdip_bundle(lat, lon, cdip_meta)
            chunk = chunk.merge(
                cdip_daily,
                on=["sample_date", "cdip_bundle"],
                how="left",
            )
        else:
            chunk["cdip_bundle"] = ""
            chunk["cdip_wave_hs_m_mean"] = np.nan
            chunk["cdip_wave_tp_s_mean"] = np.nan

        if len(sd_monthly):
            chunk["calendar_month"] = pd.to_datetime(chunk["sample_date"], errors="coerce").dt.to_period(
                "M"
            ).astype(str)
            sd_cols = [c for c in sd_monthly.columns if c not in ("CountyName", "calendar_month")]
            chunk = chunk.merge(
                sd_monthly[["CountyName", "calendar_month"] + sd_cols],
                on=["CountyName", "calendar_month"],
                how="left",
            )
        else:
            chunk["calendar_month"] = pd.to_datetime(chunk["sample_date"], errors="coerce").dt.to_period(
                "M"
            ).astype(str)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table.schema, compression="zstd")
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        total_rows += len(chunk)
        chunks_done += 1
        print(f"[chunk {chunks_done}] rows={len(chunk):_} cumulative={total_rows:_}")
        if args.max_chunks is not None and chunks_done >= args.max_chunks:
            break

    if writer is None:
        raise SystemExit("No rows written.")
    writer.close()

    meta = {
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bacteria_csv": str(bacteria_path.relative_to(repo_root())),
        "output_parquet": str(out_parquet.relative_to(repo_root())),
        "total_rows": total_rows,
        "chunks": chunks_done,
        "precip_rows": int(len(precip)),
        "tide_daily_rows": int(len(tides)),
        "sccoos_daily_rows": int(len(sccoos)),
        "cdip_daily_rows": int(len(cdip_daily)),
        "cdip_buoys": meta_cdip_buoys,
        "sd_coastal_monthly_rows": int(len(sd_monthly)),
        "layers": [
            "tier1_bacteria_spine",
            "tier1_station_supplement",
            "tier2_ghcn_regional_precip",
            "tier2_coops_tidal_range",
            "tier2_cdip_nearest_buoy_waves",
            "tier3_sccoos_delmar_ts_regional",
            "tier5_sd_county_monthly_coastal_context",
        ],
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] {out_parquet.name}: {total_rows:_} rows -> {out_parquet}")
    print(f"[ok] meta -> {out_meta}")


if __name__ == "__main__":
    main()
