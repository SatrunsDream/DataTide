"""Export Power BI-ready tables from processed project artifacts.

This script builds a small star schema under ``POWER BI/data`` so a beginner
can import a few clean tables instead of wrangling the raw project datasets.

Primary sources:
- data/processed/datatide_ground_truth.parquet
- artifacts/data/panel/enterococcus_panel.parquet

Default scope:
- Dates: 2015-01-01 through 2025-12-31
- Parameter focus: Enterococcus daily panel for dashboarding
- Detail table: all available lab-result parameters for drill-through

Usage:
  .venv/bin/python scripts/process/export_power_bi_dataset.py
  .venv/bin/python scripts/process/export_power_bi_dataset.py \
      --start-date 2015-01-01 --end-date 2025-12-31 \
      --output-dir "POWER BI/data"
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def _first_non_blank(series: pd.Series) -> Any:
    for value in series:
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return pd.NA


def _rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    cols = {old: new for old, new in mapping.items() if old in df.columns}
    return df.rename(columns=cols)


def _safe_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def build_dim_date(start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    dim = pd.DataFrame({"date": dates})
    dim["date_key"] = dim["date"].dt.strftime("%Y%m%d").astype("int64")
    dim["year"] = dim["date"].dt.year.astype("int64")
    dim["quarter"] = "Q" + dim["date"].dt.quarter.astype(str)
    dim["month_number"] = dim["date"].dt.month.astype("int64")
    dim["month_name"] = dim["date"].dt.strftime("%B")
    dim["year_month"] = dim["date"].dt.strftime("%Y-%m")
    dim["week_of_year"] = dim["date"].dt.isocalendar().week.astype("int64")
    dim["day_of_month"] = dim["date"].dt.day.astype("int64")
    dim["day_of_week_number"] = (dim["date"].dt.dayofweek + 1).astype("int64")
    dim["day_of_week_name"] = dim["date"].dt.strftime("%A")
    dim["day_of_year"] = dim["date"].dt.dayofyear.astype("int64")
    dim["is_weekend"] = dim["day_of_week_number"].isin([6, 7])
    dim["season"] = dim["month_number"].map(_season_from_month)
    return dim


def build_dim_station(
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    station_cols = [
        "sample_date",
        "Station_id",
        "Station_Name",
        "Beach_Name",
        "CountyName",
        "BeachType",
        "WaterBodyName",
        "Agency_Name",
        "USEPAID",
        "AB411Beach",
        "Station_UpperLat",
        "Station_UpperLon",
        "Beach_UpperLat",
        "Beach_UpperLon",
        "beach_detail_NearestCityName",
        "beach_detail_BeachAccess",
        "beach_detail_AttendanceSummer",
        "beach_detail_AttendanceWinter",
        "beach_detail_SwimSeasonLength",
        "beach_detail_Beach_Length",
        "beach_detail_WaterShedName",
        "beach_detail_WaterBodyType",
        "beach_detail_WaterBodyClass",
        "station_description_ref",
        "agency_station_id",
    ]
    station_frame = ground_truth[[c for c in station_cols if c in ground_truth.columns]].copy()
    station_frame["Station_id"] = pd.to_numeric(station_frame["Station_id"], errors="coerce")
    station_frame = station_frame.dropna(subset=["Station_id"]).copy()
    station_frame["Station_id"] = station_frame["Station_id"].astype("int64")
    station_frame = station_frame.sort_values("sample_date", ascending=False)

    grouped = station_frame.groupby("Station_id", dropna=False)
    dim = grouped.agg(_first_non_blank).reset_index()
    if "sample_date" in dim.columns:
        dim = dim.drop(columns=["sample_date"])
    dim = _rename_columns(
        dim,
        {
            "Station_id": "station_id",
            "Station_Name": "station_name",
            "Beach_Name": "beach_name",
            "CountyName": "county",
            "BeachType": "beach_type",
            "WaterBodyName": "water_body_name",
            "Agency_Name": "agency_name",
            "USEPAID": "usepa_id",
            "AB411Beach": "ab411_beach",
            "Station_UpperLat": "station_latitude",
            "Station_UpperLon": "station_longitude",
            "Beach_UpperLat": "beach_latitude",
            "Beach_UpperLon": "beach_longitude",
            "beach_detail_NearestCityName": "nearest_city_name",
            "beach_detail_BeachAccess": "beach_access",
            "beach_detail_AttendanceSummer": "attendance_summer",
            "beach_detail_AttendanceWinter": "attendance_winter",
            "beach_detail_SwimSeasonLength": "swim_season_length",
            "beach_detail_Beach_Length": "beach_length",
            "beach_detail_WaterShedName": "watershed_name",
            "beach_detail_WaterBodyType": "water_body_type",
            "beach_detail_WaterBodyClass": "water_body_class",
            "station_description_ref": "station_description",
            "agency_station_id": "agency_station_id",
        },
    )
    dim["ab411_beach"] = dim["ab411_beach"].astype(str)
    return dim.sort_values(["county", "beach_name", "station_id"], na_position="last").reset_index(drop=True)


def build_dim_parameter(ground_truth: pd.DataFrame) -> pd.DataFrame:
    values = (
        ground_truth["Parameter"]
        .dropna()
        .astype(str)
        .str.strip()
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return pd.DataFrame(
        {
            "parameter_id": np.arange(1, len(values) + 1, dtype=np.int64),
            "parameter_name": values,
        }
    )


def build_fact_beach_day(panel: pd.DataFrame) -> pd.DataFrame:
    fact = panel.copy()
    fact["date"] = pd.to_datetime(fact["sample_date"], errors="coerce")
    fact["date_key"] = fact["date"].dt.strftime("%Y%m%d").astype("int64")
    fact["station_id"] = pd.to_numeric(fact["Station_id"], errors="coerce").astype("int64")
    fact["observed_mpn"] = np.power(10.0, pd.to_numeric(fact["log10_result"], errors="coerce"))
    fact["ab411_threshold_mpn"] = 104.0
    fact["exceeds_ab411"] = _safe_bool((fact["observed_mpn"] > fact["ab411_threshold_mpn"]) & fact["is_observed"])
    fact["has_wave_data"] = _safe_bool(fact["wave_hs_m"].notna())
    fact["has_sst_data"] = _safe_bool(fact["sst_c"].notna())
    fact["has_salinity_data"] = _safe_bool(fact["salinity_psu"].notna())

    keep_cols = [
        "date",
        "date_key",
        "station_id",
        "CountyName",
        "log10_result",
        "observed_mpn",
        "is_observed",
        "is_left_cens",
        "is_right_cens",
        "det_low",
        "det_high",
        "n_replicates",
        "ab411_threshold_mpn",
        "exceeds_ab411",
        "rain_24h_mm",
        "rain_48h_mm",
        "rain_72h_mm",
        "rain_7d_mm",
        "dry_days_since_rain",
        "tide_range_m",
        "wave_hs_m",
        "wave_tp_s",
        "sst_c",
        "salinity_psu",
        "hf_current_speed_mps",
        "river_discharge_cms",
        "yesterday_log10_result",
        "has_wave_data",
        "has_sst_data",
        "has_salinity_data",
    ]
    fact = fact[keep_cols]
    fact = _rename_columns(
        fact,
        {
            "CountyName": "county",
            "is_left_cens": "is_left_censored",
            "is_right_cens": "is_right_censored",
            "det_low": "detection_limit_low_mpn",
            "det_high": "detection_limit_high_mpn",
            "n_replicates": "sample_replicates",
            "hf_current_speed_mps": "current_speed_mps",
        },
    )
    bool_cols = [
        "is_observed",
        "is_left_censored",
        "is_right_censored",
        "exceeds_ab411",
        "has_wave_data",
        "has_sst_data",
        "has_salinity_data",
    ]
    for col in bool_cols:
        fact[col] = _safe_bool(fact[col])
    return fact.sort_values(["date", "station_id"]).reset_index(drop=True)


def build_fact_lab_results(ground_truth: pd.DataFrame, dim_parameter: pd.DataFrame) -> pd.DataFrame:
    fact = ground_truth.copy()
    fact["date"] = pd.to_datetime(fact["sample_date"], errors="coerce")
    fact["date_key"] = fact["date"].dt.strftime("%Y%m%d").astype("int64")
    fact["station_id"] = pd.to_numeric(fact["Station_id"], errors="coerce").astype("Int64")
    fact["beach_id"] = pd.to_numeric(fact["BeachName_id"], errors="coerce").astype("Int64")
    fact["result_numeric"] = pd.to_numeric(fact["Result"], errors="coerce")
    param_map = dict(
        zip(dim_parameter["parameter_name"].astype(str), dim_parameter["parameter_id"].astype("int64"))
    )
    fact["parameter_id"] = fact["Parameter"].astype(str).map(param_map).astype("Int64")

    keep_cols = [
        "RESULTS id",
        "date",
        "date_key",
        "station_id",
        "beach_id",
        "parameter_id",
        "Parameter",
        "Result",
        "result_numeric",
        "Unit",
        "Qualifier",
        "SampleType",
        "AnalysisMethod",
        "CountyName",
        "Station_Name",
        "Beach_Name",
        "Agency_Name",
        "Weather",
        "SurfHeight",
        "TidalHeight",
        "Turbidity",
        "WaterColor",
        "regional_ghcn_prcp_mm",
        "regional_ghcn_prcp_mm_24h",
        "regional_ghcn_prcp_mm_48h",
        "regional_ghcn_prcp_mm_72h",
        "regional_ghcn_prcp_mm_7d",
        "dry_days_since_rain",
        "tide_range_hilo_m",
        "cdip_wave_hs_m_mean",
        "cdip_wave_tp_s_mean",
        "sccoos_delmar_temp_1m_c",
        "sccoos_delmar_salinity_1m_psu",
        "hf_current_speed_mps",
        "ibwc_tijuana_discharge_cms_daily_mean",
        "cce_temp_shallow_c",
        "cce_psal_shallow_psu",
    ]
    fact = fact[[c for c in keep_cols if c in fact.columns]]
    fact = _rename_columns(
        fact,
        {
            "RESULTS id": "result_id",
            "Parameter": "parameter_name",
            "Result": "result_raw",
            "Unit": "unit",
            "Qualifier": "qualifier",
            "SampleType": "sample_type",
            "AnalysisMethod": "analysis_method",
            "CountyName": "county",
            "Station_Name": "station_name",
            "Beach_Name": "beach_name",
            "Agency_Name": "agency_name",
            "Weather": "weather",
            "SurfHeight": "surf_height",
            "TidalHeight": "tidal_height",
            "Turbidity": "turbidity",
            "WaterColor": "water_color",
            "regional_ghcn_prcp_mm": "regional_precip_mm",
            "regional_ghcn_prcp_mm_24h": "regional_precip_24h_mm",
            "regional_ghcn_prcp_mm_48h": "regional_precip_48h_mm",
            "regional_ghcn_prcp_mm_72h": "regional_precip_72h_mm",
            "regional_ghcn_prcp_mm_7d": "regional_precip_7d_mm",
            "tide_range_hilo_m": "tide_range_m",
            "cdip_wave_hs_m_mean": "wave_hs_m",
            "cdip_wave_tp_s_mean": "wave_tp_s",
            "sccoos_delmar_temp_1m_c": "sst_c",
            "sccoos_delmar_salinity_1m_psu": "salinity_psu",
            "hf_current_speed_mps": "current_speed_mps",
            "ibwc_tijuana_discharge_cms_daily_mean": "river_discharge_cms",
        },
    )
    return fact.sort_values(["date", "station_id", "parameter_name"]).reset_index(drop=True)


def build_predictions_template() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "run_id",
            "date",
            "date_key",
            "station_id",
            "model_name",
            "pred_mpn_p05",
            "pred_mpn_p25",
            "pred_mpn_p50",
            "pred_mpn_p75",
            "pred_mpn_p95",
            "p_exceedance",
            "alert_level",
        ]
    )


def export_tables(
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_path = ROOT / "data" / "processed" / "datatide_ground_truth.parquet"
    panel_path = ROOT / "artifacts" / "data" / "panel" / "enterococcus_panel.parquet"

    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Missing ground truth parquet: {ground_truth_path}")
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing enterococcus panel parquet: {panel_path}")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    ground_truth_cols = [
        "RESULTS id",
        "sample_date",
        "Station_id",
        "BeachName_id",
        "Parameter",
        "Result",
        "Unit",
        "Qualifier",
        "SampleType",
        "AnalysisMethod",
        "CountyName",
        "Station_Name",
        "Beach_Name",
        "Agency_Name",
        "Weather",
        "SurfHeight",
        "TidalHeight",
        "Turbidity",
        "WaterColor",
        "BeachType",
        "WaterBodyName",
        "USEPAID",
        "AB411Beach",
        "Station_UpperLat",
        "Station_UpperLon",
        "Beach_UpperLat",
        "Beach_UpperLon",
        "station_description_ref",
        "agency_station_id",
        "beach_detail_NearestCityName",
        "beach_detail_BeachAccess",
        "beach_detail_AttendanceSummer",
        "beach_detail_AttendanceWinter",
        "beach_detail_SwimSeasonLength",
        "beach_detail_Beach_Length",
        "beach_detail_WaterShedName",
        "beach_detail_WaterBodyType",
        "beach_detail_WaterBodyClass",
        "regional_ghcn_prcp_mm",
        "regional_ghcn_prcp_mm_24h",
        "regional_ghcn_prcp_mm_48h",
        "regional_ghcn_prcp_mm_72h",
        "regional_ghcn_prcp_mm_7d",
        "dry_days_since_rain",
        "tide_range_hilo_m",
        "cdip_wave_hs_m_mean",
        "cdip_wave_tp_s_mean",
        "sccoos_delmar_temp_1m_c",
        "sccoos_delmar_salinity_1m_psu",
        "hf_current_speed_mps",
        "ibwc_tijuana_discharge_cms_daily_mean",
        "cce_temp_shallow_c",
        "cce_psal_shallow_psu",
    ]
    ground_truth = pd.read_parquet(ground_truth_path, columns=ground_truth_cols)
    ground_truth["sample_date"] = pd.to_datetime(ground_truth["sample_date"], errors="coerce")
    ground_truth = ground_truth.loc[
        ground_truth["sample_date"].between(start_ts, end_ts, inclusive="both")
    ].copy()

    panel_cols = [
        "Station_id",
        "sample_date",
        "CountyName",
        "log10_result",
        "is_left_cens",
        "is_right_cens",
        "det_low",
        "det_high",
        "n_replicates",
        "is_observed",
        "rain_24h_mm",
        "rain_48h_mm",
        "rain_72h_mm",
        "rain_7d_mm",
        "dry_days_since_rain",
        "tide_range_m",
        "wave_hs_m",
        "wave_tp_s",
        "sst_c",
        "salinity_psu",
        "hf_current_speed_mps",
        "river_discharge_cms",
        "yesterday_log10_result",
    ]
    panel = pd.read_parquet(panel_path, columns=panel_cols)
    panel["sample_date"] = pd.to_datetime(panel["sample_date"], errors="coerce")
    panel = panel.loc[panel["sample_date"].between(start_ts, end_ts, inclusive="both")].copy()

    dim_date = build_dim_date(start_date=start_date, end_date=end_date)
    dim_station = build_dim_station(ground_truth)
    dim_parameter = build_dim_parameter(ground_truth)
    fact_beach_day = build_fact_beach_day(panel)
    fact_lab_results = build_fact_lab_results(ground_truth, dim_parameter)
    predictions_template = build_predictions_template()

    tables: dict[str, pd.DataFrame] = {
        "dim_date": dim_date,
        "dim_station": dim_station,
        "dim_parameter": dim_parameter,
        "fact_beach_day": fact_beach_day,
        "fact_lab_results": fact_lab_results,
        "fact_model_predictions_template": predictions_template,
    }

    manifest_tables: dict[str, Any] = {}
    for name, frame in tables.items():
        out_path = output_dir / f"{name}.parquet"
        frame.to_parquet(out_path, index=False)
        manifest_tables[name] = {
            "file": str(out_path.relative_to(ROOT)),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
        }

    template_csv_path = output_dir / "fact_model_predictions_template.csv"
    predictions_template.to_csv(template_csv_path, index=False)
    manifest_tables["fact_model_predictions_template_csv"] = {
        "file": str(template_csv_path.relative_to(ROOT)),
        "rows": int(len(predictions_template)),
        "columns": int(len(predictions_template.columns)),
    }

    manifest = {
        "built_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_range": {"start": start_date, "end": end_date},
        "sources": {
            "ground_truth": str(ground_truth_path.relative_to(ROOT)),
            "enterococcus_panel": str(panel_path.relative_to(ROOT)),
        },
        "tables": manifest_tables,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Power BI-ready parquet tables.")
    parser.add_argument("--start-date", default="2015-01-01", help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default="2025-12-31", help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--output-dir",
        default="POWER BI/data",
        help="Directory where parquet files should be written.",
    )
    args = parser.parse_args()

    manifest = export_tables(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=ROOT / args.output_dir,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
