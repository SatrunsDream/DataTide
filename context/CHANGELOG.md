# CHANGELOG.md

## Unreleased

- Initial repository layout for DataTide (science + deployment paths).
- Notebooks grouped under `notebooks/eda/`, `notebooks/modeling/`, `notebooks/reporting/`; all Markdown docs under `context/`.
- Added `resources/paper/` for curated research papers.
- Filled `context/PROJECT_BRIEF.md`, `DATASETS.md` (feature matrix), `INTERFACES.md`, `GLOSSARY.md`, `STATUS.md` for beach FIB prediction context.
- Reoriented documentation to **State Water Board** `lab.data.ca.gov` Tier 1 backbone (bacteria + stations + beach detail); tiered CDIP, NOAA, SCCOOS, HF radar, CCE, Surfrider BWTF, SD County, IBWC; CCE demoted to background-only.
- Added `configs/fetch.yaml`, `src/io/fetch_common.py`, and `scripts/fetch/*` to download into per-source folders under `data/raw/` (default date window **2010-01-01 → 2025-12-31**, subset in aggregate step as needed).
- Tier 1 pulls use **data.ca.gov datastore dump** CSV URLs; NOAA tides **yearly chunks**; `stream_download` for large files.
- NOAA tides fetch uses **`interval=hilo`** (high/low only) for **daily** modeling grain; docs updated for week-by-week ops vs hourly tides.
