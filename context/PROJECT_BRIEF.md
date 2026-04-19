# PROJECT_BRIEF.md

## Goal

Build **machine learning** and a future **app** that support **prediction of unsafe California beach water** (fecal indicator bacteria—**FIB**), with **Enterococcus** and other indicators as published in **official statewide monitoring**. **Product vision for the app / web** (personas, screens, design system, surf/weather parity via Open-Meteo): **`APP_PRODUCT.md`**.

**Probabilistic layer (project plan):** Quantify **uncertainty** in concentrations and **exceedance** probabilities suitable for **decision support** and dashboards—not only point predictions. The preferred path is a **hierarchical** model in **JAX + NumPyro** with **HSGP** smooths for nonlinear environmental effects and a **censored log-normal** observation model for detection limits; **SVI** on full data plus **NUTS** on a subset for calibration checks within time budget (`context/plan.md`, `DECISIONS.md`). **Naive baselines** (persistence, climatology, logistic, **LightGBM**) are mandatory comparators.

**Data backbone (non-negotiable for design):** The **California State Water Resources Control Board** beach water quality open-data products on **`lab.data.ca.gov`**—**(1) statewide bacteria monitoring results**, **(2) monitoring stations**, **(3) beach detail information**—form the **core target and metadata spine**. All **environmental** layers (waves, tides, rain, SCCOOS/HF radar, optional CCE, Surfrider BWTF, county/regional feeds) are **joined to that spine**, not the reverse.

The modeling approach remains **tabular, lag-aware, leakage-safe** (see `INTERFACES.md`), with **South Bay / Tijuana River** enhancements (**IBWC** and related) as a **regional special case**, not the statewide default.

## Success criteria

- **Tier 1 data ingested first:** Bacteria results + stations + beach detail documented and reproducible from `configs/` + `src/io/`.
- **Documented feature lineage:** Every model input has a **named column**, **tier/source**, and **rationale** (`DATASETS.md`).
- **Time-correct features:** Lags computed only from information **available at prediction time** (`INTERFACES.md`).
- **Defensible baseline:** Tabular classifier with **time-blocked validation** and **threshold tuning** for **exceedance recall** where appropriate.
- **Explainability:** SHAP or equivalent, aligned with domain expectations (rain, tide, waves, circulation).
- **Traceability:** Raw caches under `data/raw/`, transforms in `src/`, artifacts and `context/structure.md` updated per project rules.

## Constraints

- **Data latency:** NOAA precip and some NCEI products may lag; operational inference may need alternate real-time sources—document per deployment.
- **Licensing:** Respect State of California open data, NOAA, CDIP, SCCOOS, Surfrider, and county terms.
- **Not an official advisory:** Outputs are **decision support** unless adopted by a health authority.
- **Site specificity:** California practice is often **beach- or region-specific**; optional **enhanced** submodels (e.g. South Bay) sit **on top of** the statewide framework.

## Further reading

- **Authoritative source stack (tiers + URLs):** `context/DATASETS.md`
- **Processed table schema (columns + joins):** `context/GROUND_TRUTH_SCHEMA.md`
- **Modeling methodology decisions:** `context/DECISIONS.md`, `context/ASSUMPTIONS.md`, `context/RISKS.md`, `context/plan.md`
- Literature / legacy endpoint notes: `resources/deep-research-report.md`
- Feature–source matrix: `DATASETS.md` → *Model input features*
