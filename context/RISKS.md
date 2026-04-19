# RISKS.md

Known risks, failure modes, mitigations. Aligned with `context/plan.md` (modeling sprint) and implementation reality.

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Censoring fields missing or ambiguous** in Tier 1 export | Wrong likelihood; biased \(\hat\eta\) and exceedance | Audit `Qualifier`/`Result` early; document parse rules; fallback rows flagged `is_censored_unknown` |
| **CDIP / wave history mostly null** (realtime NetCDF only) | Wave smooths weak or inactive for long history | Zero-fill + `has_wave_data` indicator; add **archive** THREDDS URLs in `configs/fetch.yaml`; prioritize rain/tide/salinity in v1 |
| **SVI posterior too tight** (mean-field or poor guide) | **Understated uncertainty**; `p_exceed` not calibrated | **AutoLowRankMultivariateNormal** (not `AutoNormal` only); **Pareto-\(\hat k\)** after SVI; **NUTS on subset**; report both; if \(\hat k \ge 0.7\), subset NUTS as primary for uncertainty |
| **jax-metal / GPU path on Mac** | **NaNs**, silent bad samples in MCMC | Pin **`jax[cpu]`**; assert `jax.devices()[0].platform == 'cpu'` in modeling entrypoint |
| **10-hour budget overrun** | No delivered model | **Gates:** hour 4 drop HSGPs; hour 7 ship **LightGBM + bootstrap CIs** if SVI fails; document in `ASSUMPTIONS.md` |
| **Double-counting lab replicates** | Inflated precision, wrong coverage | Aggregate to one obs per (station, date) or explicit hierarchical replicate model—decide in EDA |
| **Multi-parameter / multi-jurisdiction thresholds** | Wrong `target_exceedance` | v1 **Enterococcus** only; codify threshold table from official rules per program |
| **Spatial join mismatch** (buoy/tide/gauge miles from beach) | Residual bias by region | Document distance in `dim_station`; sensitivity: drop coastal stations with poor join quality |
| **BWTF accidentally used as spine** | Non-comparable to regulatory narrative | Enforce Tier 1 in ETL and `DECISIONS.md`; code review on label columns |
| **Data not in Git** | Collaborators clone empty `data/` | `context/README.md` replication steps; `datatide_ground_truth_meta.json` records input paths |

**Related:** `DECISIONS.md` (inference and spine choices), `ASSUMPTIONS.md` (SVI validation assumptions).
