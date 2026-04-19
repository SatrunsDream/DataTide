# Processed datasets (`data/processed/`)

This folder holds **derived, analysis-ready** tables built from `data/raw/`. Raw files stay immutable; regenerate processed outputs when you refresh downloads.

**Full column dictionary, sources, and join logic:** [`context/GROUND_TRUTH_SCHEMA.md`](../../context/GROUND_TRUTH_SCHEMA.md).  
**End-to-end replication (fetch + build, timing):** [`context/README.md`](../../context/README.md).  
This directory is **not committed** to Git (see repo `.gitignore`); only `.gitkeep` is tracked.

## Canonical ground-truth table

| File | Description |
|------|-------------|
| `datatide_ground_truth.parquet` | **Primary modeling / EDA table**: one row per **Tier-1 bacteria lab result** (CA SWRCB / data.ca.gov), with environmental context joined where spatial/temporal alignment is defensible. |
| `datatide_ground_truth_meta.json` | Row counts, input paths, and which layers were merged. |

### Build command

From the repo root (after `pip install -r requirements.txt`):

```bash
python scripts/process/build_ground_truth_dataset.py
```

Smoke test (first chunks only):

```bash
python scripts/process/build_ground_truth_dataset.py --max-chunks 2
```

### Why does my editor show “few lines” for the Parquet file?

**Parquet is a compressed binary columnar format**, not text. IDEs often try to open it as text and report a meaningless “line” count. Check the real size and row count with Python or your file explorer:

```bash
python -c "import pandas as pd; df=pd.read_parquet('data/processed/datatide_ground_truth.parquet'); print(len(df), 'rows;', df.shape[1], 'cols')"
```

A **~1.4M-row** table with ~30–40 columns often compresses to **tens of MB** with Zstandard (`zstd`) because county names, station names, and repeated dates compress well. **Small on disk does not mean small scientifically.**

### What is merged into `datatide_ground_truth.parquet`?

1. **Tier 1 — Bacteria spine**  
   Latest `data/raw/ca_swrcb_bacteria/download_*.csv` (config: `configs/process.yaml` → `bacteria_columns`).

2. **Tier 1 — Station supplement**  
   Latest `ca_swrcb_stations/download_*.csv` → adds `station_datum`, `agency_station_id`, etc.

3. **Tier 2 — Regional precipitation**  
   GHCN daily PRCP from `data/raw/noaa_precip/cdo_PRCP_*.json`, keyed by **county → regional bucket** (`county_to_env` in `process.yaml`).

4. **Tier 2 — Tidal range**  
   CO-OPS high/low predictions: `data/raw/noaa_tides/tides_hilo_*.json` → `tide_range_hilo_m` for the gauge mapped to that county.

5. **Tier 2 — Waves (CDIP)**  
   Latest realtime NetCDF per buoy under `data/raw/cdip/`. Each sample row is assigned the **nearest** CDIP buoy by great-circle distance from `Station_UpperLat` / `Station_UpperLon`, then merged **by date + buoy id**.  
   *Limitation:* CDIP realtime files may only cover a **recent** window; older sample dates can have missing wave fields unless you point fetches at archive THREDDS paths.

6. **Tier 3 — SCCOOS Del Mar mooring**  
   Daily mean **1 m temperature and salinity** from `sccoos_delmar_*` ERDDAP CSV shards. Joined **by date only** and **only for counties listed in** `sccoos_join_counties` in `process.yaml` (regional Southern California signal — not valid for Northern California rows).

7. **Tier 5 — San Diego County coastal program**  
   **Monthly**, **county-wide** advisory statistics from `sd_county_beach/beach_advisories_*.json`. Merged on `CountyName == "San Diego"` and **calendar month**. These columns are contextual; they are **not** per-beach or per-sample.

### What is *not* in this table (yet)?

- **HF radar** (gridded currents): different geometry — needs interpolation to beach points or a grid lookup.
- **CCE / OceanSITES moorings** (Tier 4): offshore context; not the statewide beach spine.
- **Surfrider BWTF** (Tier 4): separate program and auth (Cognito); keep as a side table or future join.
- **IBWC / other Tier 5** feeds: verify endpoints and schema before merging.

Use those sources in notebooks or extend `build_ground_truth_dataset.py` once join logic is defined.

## Legacy filename

`datatide_modeling_daily_env.parquet` may still exist from an earlier run (bacteria + precip + tide only). The **canonical** artifact is `datatide_ground_truth.parquet`. Remove stale files if you do not need them for comparison.

### CDIP wave columns often sparse

Realtime CDIP NetCDF files cover a **rolling recent** time window. Most historical bacteria sample dates will have **null** `cdip_wave_*` values until you ingest **archive** THREDDS URLs with full history (see `configs/fetch.yaml` CDIP bundles). Nearest-buoy assignment is still valid for rows inside the NetCDF time range.

### `build_modeling_dataset.py`

`scripts/process/build_modeling_dataset.py` is a **shim** that runs the same pipeline as `build_ground_truth_dataset.py`. Either script **overwrites** `datatide_ground_truth.parquet` on each full run.
