# GLOSSARY.md

Domain terms, acronyms, and field meanings for beach water quality modeling.

| Term | Definition |
|------|------------|
| **SWRCB / State Water Board** | **California State Water Resources Control Board**; publishes statewide **beach water quality** monitoring open data used as this project’s **Tier 1** backbone (`lab.data.ca.gov`). |
| **Tier 1 (this project)** | The three **must-use** datasets: **(1)** Beach Water Quality Monitoring Results – Bacteria, **(2)** Monitoring Stations, **(3)** Beach Detail Information—see `DATASETS.md` for URLs. |
| **BWTF (Surfrider)** | **Surfrider Foundation Blue Water Task Force**—**community** monitoring with a **public portal** ([bwtf.surfrider.org](https://bwtf.surfrider.org/)). **Augmentation / validation** source (**Tier 10**), not the official statewide regulatory spine. |
| **FIB** | **Fecal indicator bacteria** (e.g. **Enterococcus**, **E. coli**, fecal coliform, total coliform—**which columns exist** depends on **Tier 1** data). |
| **Enterococcus** | Primary **marine beach** indicator in much of California practice; often compared to a posted **action threshold**. |
| **Exceedance** | Reading **above** the applicable health **action level** for the indicator and jurisdiction. |
| **CCE** | **California Current Ecosystem** long-term mooring program; **Tier 9** **background** ocean state via **OceanSITES/NDBC**—**not** the main nearshore data layer. |
| **CDIP** | **Coastal Data Information Program** (UCSD)—California wave buoys (**Tier 2**). |
| **SCCOOS** | **Southern California Coastal Ocean Observing System**—**Tier 3** ERDDAP/THREDDS/HF radar pathways. |
| **ERDDAP** | SCCOOS (and other) **data server** for tabular and gridded environmental data; common access pattern for SST, currents, model fields. |
| **HF radar** | **High-frequency radar** **surface currents**; **Tier 3** for alongshore transport / plume context ([SCCOOS](https://sccoos.org/high-frequency-radar/), [IOOS](https://ioos.noaa.gov/project/hf-radar/)). |
| **IBWC** | **International Boundary and Water Commission**—**Tier 12** **Tijuana River / border** hydrology for **South Bay** special-case models only; **endpoint VERIFY** at implementation. |
| **NOAA Tides & Currents** | **CO-OPS** API for tides and water levels (**Tier 2**). |
| **NCEI / hourly precip** | NOAA **hourly precipitation** and related archives—**Tier 2** rain features (see metaview links in `DATASETS.md`). |
| **CEDEN** | **California Environmental Data Exchange Network**—legacy/auxiliary water-quality warehouse; **not** the primary beach bacteria spine for this repo (see `DATASETS.md`). |
| **Lag feature** | Predictor using only data **strictly before** the prediction **cutoff** time. |
| **Nowcast / forecast** | **Nowcast** = same-day / near-real-time; **forecast** = multi-day lead—define in config/`ASSUMPTIONS.md`. |
