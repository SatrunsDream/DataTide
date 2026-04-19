# CHANGELOG.md

## Unreleased

- **`GROUND_TRUTH_SCHEMA.md`:** section **Geographic coverage: NorCal vs SoCal, NaNs, and what is fixable** (HF bbox / `griddap_id` defaults vs extending to NorCal; SCCOOS Del Mar county mask; CDIP/CCE caveats); **`DATASETS.md`** HF bullet cross-link; **`fetch.yaml`** HF comment block.
- **`GROUND_TRUTH_SCHEMA.md`:** section **Overlapping vs distinct columns** (when bacteria vs `beach_detail_*` overlap; station vs beach vs beach-detail geometry; dates/keys; how env layers differ—CDIP, SCCOOS, CCE, HF, IBWC, BWTF, SD coastal).
- **Modeling plan adoption:** `context/plan.md` (Bayesian hierarchical + HSGP + censored likelihood + SVI/NUTS protocol) distilled into `DECISIONS.md`, `ASSUMPTIONS.md`, `RISKS.md`; `STATUS.md` sprint checklist; `INTERFACES.md` extended (latent daily grain, ArviZ, Power BI star schema); `PROJECT_BRIEF` and `GLOSSARY` updated.
- **Ground-truth ETL:** `datatide_ground_truth.parquet` builder, `GROUND_TRUTH_SCHEMA.md`, `data/processed/README.md`; `data/**` gitignored except `.gitkeep`.
- Initial repository layout for DataTide (science + deployment paths).
- Notebooks grouped under `notebooks/eda/`, `notebooks/modeling/`, `notebooks/reporting/`; all Markdown docs under `context/`.
- Added `resources/paper/` for curated research papers.
- Filled `context/PROJECT_BRIEF.md`, `DATASETS.md` (feature matrix), `INTERFACES.md`, `GLOSSARY.md`, `STATUS.md` for beach FIB prediction context.
- Reoriented documentation to **State Water Board** `lab.data.ca.gov` Tier 1 backbone (bacteria + stations + beach detail); tiered CDIP, NOAA, SCCOOS, HF radar, CCE, Surfrider BWTF, SD County, IBWC; CCE demoted to background-only.
- Added `configs/fetch.yaml`, `src/io/fetch_common.py`, and `scripts/fetch/*` to download into per-source folders under `data/raw/` (default date window **2010-01-01 → 2025-12-31**, subset in aggregate step as needed).
- Tier 1 pulls use **data.ca.gov datastore dump** CSV URLs; NOAA tides **yearly chunks**; `stream_download` for large files.
- NOAA tides fetch uses **`interval=hilo`** (high/low only) for **daily** modeling grain; docs updated for week-by-week ops vs hourly tides.
