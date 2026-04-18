# DataTide

End-to-end data science layout: exploration and reporting in `notebooks/` (by phase), reusable code in `src/`, governed data under `data/`, generated outputs in `artifacts/`, and an `app/` layer for later deployment.

**Documentation in this folder:** operational rules in [`rules_templet.md`](rules_templet.md) (see [`development_rules.md`](development_rules.md)), repo map and results log in [`structure.md`](structure.md), decisions in [`DECISIONS.md`](DECISIONS.md), assumptions in [`ASSUMPTIONS.md`](ASSUMPTIONS.md), changelog in [`CHANGELOG.md`](CHANGELOG.md). Supporting notes: [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md), [`DATASETS.md`](DATASETS.md), [`GLOSSARY.md`](GLOSSARY.md), [`INTERFACES.md`](INTERFACES.md), [`STATUS.md`](STATUS.md), [`RISKS.md`](RISKS.md).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Where things live

| Area | Path (repo root) |
|------|-------------------|
| Rules | [`context/rules_templet.md`](rules_templet.md), [`context/development_rules.md`](development_rules.md) |
| Repo map + results index | [`context/structure.md`](structure.md) |
| EDA notebooks | `notebooks/eda/` |
| Modeling notebooks | `notebooks/modeling/` |
| Reporting notebooks | `notebooks/reporting/` (summaries, stakeholder-facing model narrative) |
| Library code | `src/` |
| CLI scripts | `scripts/` |
| Config | `configs/` |
| Data tiers | `data/raw`, `data/interim`, `data/processed`, `data/external` |
| Outputs | `artifacts/` |
| App / serving | `app/` |
| Containers | `docker/` |

Use `src.utils.paths.repo_root()` for paths relative to the repo (see [`rules_templet.md`](rules_templet.md)).

## Git and data

`data/raw/` and `data/interim/` are ignored except `.gitkeep`. Commit small processed sets or manifests only when policy allows.
