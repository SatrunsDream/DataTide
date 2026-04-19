# DATASETS.md

Where data comes from, how it maps to **model features**, and **tiered priority** for ingestion. Written so **another LLM or teammate** can implement ETL without drifting to the wrong backbone.

**Architecture rule:** The **core** of this project is a **statewide California beach bacteria target table** from the **State Water Resources Control Board** (via **California’s beach water quality open data**), plus **station** and **beach metadata** from the same program family. **Ocean, wave, rain, tide, and runoff** features are **joined around** that spine. **Do not** build the product around **CCE moorings alone**—treat **CCE** as a **Tier 4 background** signal only.

**Related:** `context/PROJECT_BRIEF.md`, `context/INTERFACES.md`, `context/GLOSSARY.md`, `context/DECISIONS.md` (target organism, likelihood, inference), `context/plan.md` (full Bayesian methodology narrative), `resources/deep-research-report.md` (literature; may mention older CEDEN-centric paths).

---

## Clean stack (summary)

| Layer | Sources |
|--------|---------|
| **CORE TARGETS** | (1) CA Beach Water Quality Monitoring Results – Bacteria, (2) Monitoring Stations, (3) Beach Detail Information |
| **CORE FEATURES** | (4) CDIP waves, (5) NOAA Tides & Currents, (6) NOAA precipitation |
| **HIGH-VALUE EXPANSIONS** | (7) SCCOOS ERDDAP / THREDDS, (8) HF radar surface currents, (9) CCE moorings, (10) Surfrider BWTF |
| **REGIONAL SPECIAL CASE** | (11) San Diego County Beach and Bay Program, (12) IBWC / Tijuana River flow **[VERIFY endpoint]** |

---

## Tier 1 — Must-use core (State Water Board backbone)

### 1) California Beach Water Quality Monitoring Results – Bacteria

**Use for:** Primary **target table**: observed **FIB** values (Enterococcus, E. coli, fecal coliform, total coliform per dataset fields); **raw count** or **exceedance** modeling; **site–date history** features (e.g. lags of official monitoring results).

**Dataset page:** [https://lab.data.ca.gov/dataset/beach-water-quality-postings-and-closures/7bd961cf-abe4-433b-8033-378161237ff3](https://lab.data.ca.gov/dataset/beach-water-quality-postings-and-closures/7bd961cf-abe4-433b-8033-378161237ff3)

**Access:** `scripts/fetch/fetch_ca_swrcb.py` streams **`https://data.ca.gov/datastore/dump/<resource_id>?format=csv`** (avoids `/download/` **403** from some networks). Fallback: portal CSV or CKAN `package_show` on `data.ca.gov`.

### 2) California Beach Water Quality Monitoring Stations

**Use for:** **Station metadata**—coordinates, names, **county**, **agency**, keys for **spatial joins** to CDIP buoys, tide stations, rain gauges, HF radar grids, SCCOOS products.

**Dataset page:** [https://lab.data.ca.gov/dataset/beach-water-quality-postings-and-closures/98e628ff-d012-4982-ad32-b9f9ad8ab524](https://lab.data.ca.gov/dataset/beach-water-quality-postings-and-closures/98e628ff-d012-4982-ad32-b9f9ad8ab524)

**Access:** Same script as (1): `fetch_ca_swrcb.py` (resource `stations` in `configs/fetch.yaml`).

### 3) California Beach Detail Information

**Use for:** **Beach-level** identifiers and **regulatory** grouping; **standardizing joins** across station/beach naming variants.

**Dataset page:** [https://lab.data.ca.gov/dataset/beach-water-quality-postings-and-closures/fcbc9250-06e3-437d-b0c6-3cc5ddde93fc](https://lab.data.ca.gov/dataset/beach-water-quality-postings-and-closures/fcbc9250-06e3-437d-b0c6-3cc5ddde93fc)

**Access:** Same script as (1): `fetch_ca_swrcb.py` (resource `beach_detail`).

---

## Tier 2 — Strongest environmental features

### 4) CDIP (waves)

**Use for:** Significant wave height, peak period, swell direction, recent wave-energy summaries (mixing, resuspension, transport).

**Primary:** [https://cdip.ucsd.edu/](https://cdip.ucsd.edu/) (data access / THREDDS patterns as documented there)

**Broader California ocean context:** [https://sccoos.org/data-access/](https://sccoos.org/data-access/)

### 5) NOAA Tides & Currents

**Use for:** **Daily** tide context (height extrema, tidal range, phase proxies) where a beach maps to a nearby tide station. Fetches use **`interval=hilo`** (high/low predictions only), not hourly tides—matching **(site, date)** modeling and avoiding huge hourly pulls/timeouts.

**API documentation:** [https://api.tidesandcurrents.noaa.gov/api/prod/](https://api.tidesandcurrents.noaa.gov/api/prod/)

### 6) NOAA precipitation

**Use for:** `rain_24h`, `rain_48h`, `rain_72h`, `rain_7d`, `dry_days_since_rain`, etc.

**Reference / catalog pages (ISO metadata views):**

- [Hourly precipitation dataset overview (C00684)](https://data.noaa.gov/metaview/page?header=none&view=getDataView&xml=NOAA/NESDIS/ncei/ncei-nc//iso/xml/C00684.xml)
- [Hourly precipitation archive overview (C00313)](https://data.noaa.gov/metaview/page?header=none&view=getDataView&xml=NOAA/NESDIS/ncei/ncei-nc/iso/xml/C00313.xml)

**Note:** Actual pulls typically go through **NCEI/CDO or documented hourly products** + station selection (airport / major stations per region). Wire exact endpoints during implementation; document latency (training vs operational).

---

## Tier 3 — High-value coastal circulation (Southern California expansion)

### 7) SCCOOS — ERDDAP / THREDDS

**Use for:** Nearshore **SST** and **salinity**, **surface currents**, **ROMS** nowcast/forecast context, shore stations, gliders, Del Mar mooring pathways per SCCOOS documentation.

- **Data access:** [https://sccoos.org/data-access/](https://sccoos.org/data-access/)
- **ERDDAP:** [https://erddap.sccoos.org/erddap/](https://erddap.sccoos.org/erddap/)
- **THREDDS:** [https://thredds.sccoos.org/thredds/catalog.html](https://thredds.sccoos.org/thredds/catalog.html) — **[VERIFY]** catalog stability at implementation time.

### 8) HF radar surface currents

**Use for:** Current speed/direction, alongshore transport, plume indicators.

- **SCCOOS overview:** [https://sccoos.org/high-frequency-radar/](https://sccoos.org/high-frequency-radar/)
- **IOOS HF radar:** [https://ioos.noaa.gov/project/hf-radar/](https://ioos.noaa.gov/project/hf-radar/)

---

## Tier 4 — Useful secondary

### 9) CCE moorings (background only)

**Use for:** **Offshore** temperature/salinity regime, event-scale **background** ocean state—not the primary nearshore column.

- **CCE project:** [https://mooring.ucsd.edu/cce/](https://mooring.ucsd.edu/cce/)
- **OceanSITES / NDBC THREDDS (CCE1 catalog):** [https://dods.ndbc.noaa.gov/thredds/catalog/oceansites/DATA/CCE1/catalog.html](https://dods.ndbc.noaa.gov/thredds/catalog/oceansites/DATA/CCE1/catalog.html)
- **CCE2 / CCE3** analogous paths: **[VERIFY]** before relying in production ETL.

### 10) Surfrider Blue Water Task Force (BWTF)

**Use for:** **Supplemental** Enterococcus observations; **creeks / stormwater / outfalls**; gap-filling and validation vs non-government sampling—not a replacement for the **statewide official** monitoring spine.

- **Program:** [https://www.surfrider.org/programs/blue-water-task-force](https://www.surfrider.org/programs/blue-water-task-force)
- **Public data portal:** [https://bwtf.surfrider.org/](https://bwtf.surfrider.org/)

---

## Tier 5 — Regional special case

### 11) San Diego County Beach and Bay Program

**Use for:** **San Diego–specific** advisories, closures, local validation and messaging—not the **statewide** backbone.

**Program page:** [https://www.sandiegocounty.gov/content/sdc/deh/lwqd/beachandbay.html](https://www.sandiegocounty.gov/content/sdc/deh/lwqd/beachandbay.html)

*(Optional open-data API history, e.g. Socrata advisory aggregates, can supplement—document resource IDs in `configs/` when used.)*

### 12) IBWC / Tijuana River flow

**Use for:** **South Bay / Imperial Beach / Tijuana River–influenced** models only; **not** required for the statewide base model.

**Status:** **[VERIFY]** official gauge/portal/API endpoint at implementation time (portal vs legacy feeds; bot/WAF issues possible for naive HTTP).

---

## Model input features (canonical table)

| Feature category | Specific input variables | Data source (priority) | Why it matters |
|------------------|-------------------------|------------------------|----------------|
| **Spatial & temporal** | Station/beach IDs; day of year; seasonality | **Tier 1** monitoring results + stations + beach detail; calendar | Where and when you predict; seasonal FIB basins. |
| **Historical memory** | Prior official (or last-available) **FIB** reading | **Tier 1** bacteria results; optional **Tier 10** BWTF for extra sites | Persistence / decay of contamination; **strict lags** only. |
| **Precipitation** |24/48/72 h (and 7 d) rainfall; dry-day counts | **Tier 2** NOAA precip | Runoff and rain advisory mechanics. |
| **Tides** | Daily tidal range / high–low summaries (from **`hilo`** predictions, not hourly) | **Tier 2** NOAA Tides & Currents | Flushing, exchange at mouths/outfalls; **daily** features only. |
| **Wave dynamics** | Hs, Tp, direction, energy summaries | **Tier 2** CDIP | Mixing, dilution, resuspension. |
| **Oceanography / circulation** | SST, salinity, currents, model context | **Tier 3** SCCOOS (ERDDAP/THREDDS), HF radar; **Tier 9** CCE as background | Plumes, stratification, transport; **CCE is supplementary**. |
| **Community augmentation** | Extra Enterococcus samples | **Tier 10** Surfrider BWTF | Gaps, creeks, validation—not primary labels unless scoped. |
| **Inland runoff** | River discharge | **Tier 12** IBWC (+ USGS where relevant) | **Regional** South Bay; not statewide default. |

### Draft skeleton (your column headings) — mapping

| Your category | Variables | Repo source | Note |
|---------------|-----------|-------------|------|
| Spatial & temporal | Beach ID, day of year | **Tier 1** stations + beach detail + calendar | “Beach ID” = monitoring `station_id` / beach key from state data (not BWTF-only). |
| Historical memory | Yesterday’s bacteria count | **Tier 1** bacteria results | Same persistence logic; **Surfrider BWTF** optional extra rows only. |
| Precipitation | 24 / 48 / 72 h lagged rain | **Tier 2** NOAA precip | Daily or subdaily raw → **daily** lag sums for `(site, date)`. |
| Oceanography | SST, salinity | SCCOOS / moorings (**Tier 3**), CCE background (**Tier 9**) | Aggregate to **daily** at join point. |
| Wave dynamics | Hs, peak period | **Tier 2** CDIP | **Daily** mean/max over lookback from buoy. |
| Inland runoff | River discharge | **Tier 12** IBWC / **USGS** | **Daily** (or last-available) for rivermouth sites. |

**Week-by-week operations:** retrain or score in **batches of days**; the **feature table stays daily**, not hourly.

### How to use this in code

- **Grain:** one row per **monitoring event** or per **site–date** prediction unit—**align** with Tier 1 schema once ETL is fixed.
- **Joins:** build `data/external/beach_to_environment.parquet` (or similar): station → `{noaa_tide_station, cdip_buoy, precip_station, erddap_grid, hfr_tile}`.
- **Labels:** derive **exceedance** from **Tier 1** fields and **California** posting rules documented in state materials; do not silently mix Surfrider thresholds with state regulatory thresholds.

---

## Schemas (high level)

- **Tier 1 bacteria results:** station/beach keys, sample datetime, indicator columns, units, QA flags—**freeze** real column names after first successful download (`INTERFACES.md`).
- **Tier 1 stations / beach detail:** lat/lon, county, agency, beach groupings.
- **Environmental time series:** align to **sample timestamp** or **decision cutoff** with documented aggregation (mean, max, sum for rain).

---

## Processed ground-truth table (EDA + modeling)

The repo builds a **single Parquet** that **does not replace** the tiered philosophy above but **materializes** the Tier-1 spine with defensible environmental joins for analysis:

| Output | Location | Builder |
|--------|----------|---------|
| Main table | `data/processed/datatide_ground_truth.parquet` | `scripts/process/build_ground_truth_dataset.py` |
| Run metadata | `data/processed/datatide_ground_truth_meta.json` | (same script) |

**Full column-by-column documentation** (sources, join keys, aggregation rules, caveats): [`GROUND_TRUTH_SCHEMA.md`](GROUND_TRUTH_SCHEMA.md).

**What is inside (summary):**

- **Spine:** Latest Tier-1 bacteria CSV (`ca_swrcb_bacteria/download_*.csv`), one row per result record.
- **Station context:** Left join of latest Tier-1 **stations** export on `Station_id`.
- **Precipitation:** Daily GHCN PRCP from `noaa_precip` JSON, keyed by **county → regional bucket** (`configs/process.yaml` → `county_to_env`).
- **Tides:** Daily tidal range from CO-OPS **hilo** JSON files, keyed by **county → tide station id**.
- **Waves (CDIP):** Daily mean `waveHs` / `waveTp` from latest realtime NetCDF per configured buoy; each bacteria row gets the **nearest** buoy by great-circle distance from `Station_UpperLat` / `Station_UpperLon` (sparse for historical dates unless archive NetCDF URLs are configured).
- **SCCOOS:** Daily mean Del Mar 1 m temperature and salinity from ERDDAP monthly CSV shards; joined by **date** only and **nulled** outside counties listed in `sccoos_join_counties` (regional Southern California signal).
- **San Diego County program:** Monthly **county-wide** advisory aggregates from Socrata JSON; merged on `CountyName == "San Diego"` and calendar month—not per-beach.

**Not row-merged in that file:** HF radar grids, CCE moorings, BWTF (auth/program), IBWC—see `GROUND_TRUTH_SCHEMA.md` for rationale; raw files remain under `data/raw/` for separate features or future ETL.

**Git:** The entire `data/` directory is **ignored** by Git (except `.gitkeep` placeholders). Clones must **re-run** fetch and process scripts locally; see `context/README.md`.

---

## Caveats

- **State portal vs lab subdomain:** `lab.data.ca.gov` hosts the dataset pages listed above; follow current **download/API** instructions on each page (patterns may evolve).
- **Spatial mismatch:** buoys, tide gauges, HF radar grids are **not co-located** with swim sites—document distance and bearing in the join table.
- **Leakage:** lagged FIB, rain, and ocean features must use only information **available before** the prediction cutoff (`INTERFACES.md`).
- **CEDEN / older paths:** may still appear in literature (`resources/deep-research-report.md`); **Tier 1 lab.data.ca.gov beach bacteria datasets** are the **authoritative statewide spine** for this repo unless superseded by a project decision in `DECISIONS.md`.
