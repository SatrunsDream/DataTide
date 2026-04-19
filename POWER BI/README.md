# Power BI Starter Pack

This folder is the clean handoff point for Power BI.

The data exports under [data](</Users/kylechoi/DataTide/POWER BI/data>) are already filtered to the requested **2015-01-01 through 2025-12-31** range and shaped for a beginner-friendly Power BI model.

## What is in here

- `dim_date.parquet`: calendar table for slicers and time-series visuals.
- `dim_station.parquet`: station and beach metadata.
- `dim_parameter.parquet`: the four monitored bacteria parameters.
- `fact_beach_day.parquet`: one row per `station_id + date` for the **Enterococcus daily panel**.
- `fact_lab_results.parquet`: detailed lab-result fact table for drill-through and QA.
- `fact_model_predictions_template.parquet` and `.csv`: empty schema you can populate after your predictive model writes probabilities.
- `manifest.json`: row counts and source paths so you can sanity-check imports.

## Recommended beginner model

Start with only these three tables:

1. `dim_date.parquet`
2. `dim_station.parquet`
3. `fact_beach_day.parquet`

Create these relationships in Power BI:

- `dim_date[date]` -> `fact_beach_day[date]` as one-to-many
- `dim_station[station_id]` -> `fact_beach_day[station_id]` as one-to-many

Use single-direction filtering from dimensions to facts.

After that works, add:

- `fact_lab_results.parquet`
- `dim_parameter.parquet`

Then create:

- `dim_parameter[parameter_id]` -> `fact_lab_results[parameter_id]`
- `dim_date[date]` -> `fact_lab_results[date]`
- `dim_station[station_id]` -> `fact_lab_results[station_id]`

## Beginner import steps

1. Open Power BI Desktop.
2. Click `Get data`.
3. Search for `Parquet`.
4. Import these files from [data](</Users/kylechoi/DataTide/POWER BI/data>).
5. In Model view, create the relationships listed above.
6. Mark `dim_date` as your date table using the `date` column.
7. Hide key columns you do not want on the report canvas, like `date_key`.

## First visuals to build

### 1. Beach risk time series

- Visual: Line chart
- Axis: `dim_date[date]`
- Values: `% Exceed AB411`
- Legend: `dim_station[beach_name]`

### 2. County comparison

- Visual: Clustered column chart
- Axis: `dim_station[county]`
- Values: `Average Enterococcus (MPN)`

### 3. Map of beaches

- Visual: Map
- Latitude: `dim_station[station_latitude]`
- Longitude: `dim_station[station_longitude]`
- Size or color: `% Exceed AB411`
- Tooltip: `beach_name`, `county`, `attendance_summer`, `attendance_winter`

### 4. Conditions people usually use

- Visual: Scatter chart
- X: `wave_hs_m`
- Y: `sst_c`
- Size: `observed_mpn`
- Tooltip: `rain_24h_mm`, `tide_range_m`, `salinity_psu`, `exceeds_ab411`

This is a nice way to show the decision gap: people often respond to wave and weather comfort, while your water-quality view adds the hidden public-health dimension.

## Starter DAX measures

Create these measures first:

```DAX
Average Enterococcus (MPN) =
AVERAGE ( fact_beach_day[observed_mpn] )

Observed Beach Days =
CALCULATE (
    COUNTROWS ( fact_beach_day ),
    fact_beach_day[is_observed] = TRUE ()
)

Exceed Days =
CALCULATE (
    COUNTROWS ( fact_beach_day ),
    fact_beach_day[exceeds_ab411] = TRUE ()
)

% Exceed AB411 =
DIVIDE ( [Exceed Days], [Observed Beach Days] )

Average Rain 72h =
AVERAGE ( fact_beach_day[rain_72h_mm] )

Average Wave Height =
AVERAGE ( fact_beach_day[wave_hs_m] )
```

## How to add model outputs later

When your model is ready, write predictions into the same shape as:

- [fact_model_predictions_template.csv](</Users/kylechoi/DataTide/POWER BI/data/fact_model_predictions_template.csv>)

At minimum, populate:

- `run_id`
- `date`
- `station_id`
- `model_name`
- `pred_mpn_p50`
- `p_exceedance`
- `alert_level`

Then relate that table to `dim_date` and `dim_station` the same way as `fact_beach_day`.

## Refreshing the folder

From the repo root:

```bash
.venv/bin/python scripts/process/export_power_bi_dataset.py
```

If you want a different date range:

```bash
.venv/bin/python scripts/process/export_power_bi_dataset.py --start-date 2018-01-01 --end-date 2025-12-31
```
