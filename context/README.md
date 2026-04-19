# DataTide

End-to-end data science layout: exploration and reporting in `notebooks/` (by phase), reusable code in `src/`, governed data under `data/`, generated outputs in `artifacts/`, Python serving scaffold in `app/`, and the **consumer web UI** in `site/` (React/Vite).

**Documentation in this folder:** operational rules in [`rules_templet.md`](rules_templet.md) (see [`development_rules.md`](development_rules.md)), repo map and results log in [`structure.md`](structure.md), decisions in [`DECISIONS.md`](DECISIONS.md), assumptions in [`ASSUMPTIONS.md`](ASSUMPTIONS.md), changelog in [`CHANGELOG.md`](CHANGELOG.md). Supporting notes: [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md), [`DATASETS.md`](DATASETS.md) (**tiered source stack—start with State Water Board `lab.data.ca.gov` Tier 1**), [`GROUND_TRUTH_SCHEMA.md`](GROUND_TRUTH_SCHEMA.md) (**processed Parquet: columns, sources, joins, overlapping vs distinct fields**), [`plan.md`](plan.md) (**Bayesian modeling plan + critique; see DECISIONS/ASSUMPTIONS for adopted summary**), [`APP_PRODUCT.md`](APP_PRODUCT.md) (**consumer app / web: personas, screens, design, Open-Meteo roadmap**), [`MODEL_OUTPUTS_FOR_DEMO.md`](MODEL_OUTPUTS_FOR_DEMO.md) (**modeling pipeline summary + expected API fields + fake JSON for demos**), [`GLOSSARY.md`](GLOSSARY.md), [`INTERFACES.md`](INTERFACES.md), [`STATUS.md`](STATUS.md), [`RISKS.md`](RISKS.md).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

---

## Replicate data locally (nothing under `data/` is on GitHub)

The **`data/`** directory is **gitignored** except `.gitkeep` files. After cloning, you must **download raw inputs** and **build** the processed table yourself.

### 1) Environment and secrets

```bash
# Windows PowerShell: Copy-Item .env.example .env
cp .env.example .env
# Edit .env:
#   NOAA_CDO_TOKEN=...   (required for NOAA precipitation — free token from NCEI CDO)
#   BWTF_COGNITO_ID_TOKEN=...   (optional — only for Surfrider BWTF bulk GraphQL; omit to skip)
```

### 2) Pull raw data

From the **repository root**:

```bash
python scripts/fetch/run_all.py
```

This runs every fetcher in `scripts/fetch/run_all.py` (Tier 1 state data, CDIP, tides, precip, SCCOOS, HF radar, Tier 4–5, etc.) according to **`configs/fetch.yaml`**.

**Individual sources** (same root), e.g.:

```bash
python scripts/fetch/fetch_ca_swrcb.py
python scripts/fetch/fetch_noaa_precip.py
python scripts/fetch/fetch_noaa_tides.py
python scripts/fetch/fetch_cdip.py
python scripts/fetch/fetch_sccoos_erddap.py
python scripts/fetch/fetch_surfrider_bwtf.py
```

**Configuration:** `configs/fetch.yaml` — default window **2010-01-01 → 2025-12-31** (set `end: null` to pull through today). Tier 1 may need **`direct_url`** values if CKAN responses differ. **BWTF** skips without `BWTF_COGNITO_ID_TOKEN` (see `.env.example`).

### 3) Build the ground-truth Parquet

```bash
python scripts/process/build_ground_truth_dataset.py
```

**Output:** `data/processed/datatide_ground_truth.parquet` and `datatide_ground_truth_meta.json`.  
**Shim:** `python scripts/process/build_modeling_dataset.py` runs the same pipeline.

**Smoke test** (first chunks only):

```bash
python scripts/process/build_ground_truth_dataset.py --max-chunks 2
```

### 4) How long it takes (order-of-magnitude)

| Step | Duration | Notes |
|------|-----------|--------|
| `pip install -r requirements.txt` | ~1–5 min | Depends on machine and cache. |
| `fetch_ca_swrcb.py` (bacteria CSV) | **~15–45+ min** | Large stream (~1.5GB+); set generous timeout in `fetch.yaml`. |
| `fetch_noaa_precip.py` | **~10–60+ min** | Many station × year API calls + delay between requests. |
| `fetch_noaa_tides.py` | **~10–40 min** | Many station × year JSON files. |
| Other fetchers (CDIP, SCCOOS, …) | **~5–30 min** each | SCCOOS has many monthly shards; CDIP is a few NetCDF files. |
| **Full `run_all.py`** | **~1–3+ hours** typical | Serial script order; network and rate limits dominate. |
| `build_ground_truth_dataset.py` | **~1–2 min** | For ~1.4M bacteria rows on a typical laptop; scales with CSV size. |

Treat these as **rough** bounds; first-time runs and slow disks increase wall time.

### 5) Accessing outputs in Python

```python
import pandas as pd
df = pd.read_parquet("data/processed/datatide_ground_truth.parquet")
```

**Column dictionary, join logic, and redundant-field guidance:** [`GROUND_TRUTH_SCHEMA.md`](GROUND_TRUTH_SCHEMA.md).  
**Parquet vs IDE “line count”:** [`data/processed/README.md`](../data/processed/README.md).

## Where things live

| Area | Path (repo root) |
|------|-------------------|
| Rules | [`context/rules_templet.md`](rules_templet.md), [`context/development_rules.md`](development_rules.md) |
| Repo map + results index | [`context/structure.md`](structure.md) |
| EDA notebooks | `notebooks/eda/` |
| Modeling notebooks | `notebooks/modeling/` |
| Reporting notebooks | `notebooks/reporting/` (summaries, stakeholder-facing model narrative) |
| Library code | `src/` |
| Fetch scripts | `scripts/fetch/` → `data/raw/<source>/` |
| CLI scripts | `scripts/` |
| Config | `configs/` |
| Data tiers | `data/raw`, `data/interim`, `data/processed`, `data/external` |
| Outputs | `artifacts/` |
| Web UI (React) | `site/` |
| App / serving (Python) | `app/` |
| Containers | `docker/` |
| Research papers | `resources/paper/` |

Use `src.utils.paths.repo_root()` for paths relative to the repo (see [`rules_templet.md`](rules_templet.md)).

## Git and data

**The whole `data/` tree is gitignored** (except `**/.gitkeep` placeholders) so large raw/processed files are **not pushed to GitHub**. Recreate everything with the **Replicate data locally** steps above. Optional small manifests under `artifacts/` may be committed per project policy (`rules_templet.md`).
