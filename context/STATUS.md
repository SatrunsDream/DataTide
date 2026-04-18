# STATUS.md

Current state: what is done, what is next, blockers.

## Done

- Repository scaffold aligned with `context/rules_templet.md`; notebooks under `notebooks/eda/`, `modeling/`, `reporting/`.
- Strategic research captured in `resources/deep-research-report.md`.
- **Tiered source stack** (State Water Board `lab.data.ca.gov` Tier 1 backbone + CDIP / NOAA / SCCOOS / HF radar / CCE / BWTF / regional) recorded in `context/DATASETS.md`; aligned **PROJECT_BRIEF**, **INTERFACES**, **GLOSSARY**.

## Next

- Run `scripts/fetch/` (see `context/README.md`); confirm Tier 1 downloads or set `direct_url` in `configs/fetch.yaml`.
- Ingest **Tier 1** three-pack; freeze column names in `INTERFACES.md` and `configs/`.
- Build **`data/external/`** join table: monitoring station → tide / CDIP / precip / SCCOOS / HF radar.
- **[VERIFY]** SCCOOS ERDDAP IDs, IBWC endpoint, Surfrider BWTF export; chunk NOAA tide/precip requests if APIs limit range.
- **Aggregate script** (site-day; raw pull defaults **2010–2025**, then filter years as needed): separate step after raw pulls stable.
- First **EDA notebook** under `notebooks/eda/`; log artifacts in `context/structure.md`.

## Blockers

- *(none)*
