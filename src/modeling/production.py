"""
Production helpers: turn a trained `BayesianRung` into next-week forecasts.

The ground-truth panel is historical. To predict Enterococcus counts for the
next 7 days at every station, we need a *future* `FoldData`:

    - identity per station (station_idx, county_idx)      \u2014 from the panel
    - date-derived features per day (month, doy_sin/cos)   \u2014 deterministic
    - weather lag features (rain_*, dry_days, ...)         \u2014 UNKNOWN in advance

For the weather features, two practical strategies exist:

  climatology   \u2014 use the training-data mean of each feature *per month*
                  (the default). This produces the "typical mid-April weather"
                  prediction; no external weather feed required.

  passthrough   \u2014 the caller supplies a `(N_future, D)` weather array, e.g.
                  from an NWS short-range forecast. We just plug it in.

Both strategies go through the same `build_future_fold(...)` function, which
produces a `FoldData` whose `*_val` side contains the future rows. Feed that
to `BayesianRung.predict(...)` to get a `Prediction` with posterior-predictive
draws in log10(MPN). The tidy-export helper here then folds those into a
teammate-friendly parquet + raw-samples archive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.compare import FoldData
from src.modeling.cv import PanelBundle


DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------------------
# climatology:  mean(training feature | month)  \u2192 plausible "typical" weather
# ---------------------------------------------------------------------------

@dataclass
class Climatology:
    """Per-month mean of every smooth + linear feature, computed on training rows."""

    smooth: np.ndarray          # (12, D_smooth)   month 1..12 -> mean vector
    linear: np.ndarray          # (12, D_linear)
    miss_smooth: np.ndarray     # (12, D_smooth)   fraction "missing" per month (all 0 in the usual case)
    miss_linear: np.ndarray     # (12, D_linear)
    smooth_features: list[str]
    linear_features: list[str]


def compute_climatology(bundle: PanelBundle, *, use_rows: np.ndarray | None = None) -> Climatology:
    """Compute the per-month mean weather design for use as a default future scenario.

    Parameters
    ----------
    bundle       : PanelBundle
    use_rows     : optional boolean mask over `bundle.*` arrays. Defaults to
                   all rows with `cv_val_year >= 0` (i.e. everything *except*
                   the 2024 test hold-out) so no test-window weather leaks
                   into the climatology.
    """
    if use_rows is None:
        use_rows = bundle.cv_val_year >= 0

    months = bundle.month[use_rows]
    Xs = bundle.X_smooth[use_rows]
    Xl = bundle.X_linear[use_rows]
    Ms = bundle.miss_smooth[use_rows]
    Ml = bundle.miss_linear[use_rows]

    D_smooth = Xs.shape[1]
    D_linear = Xl.shape[1]
    smooth = np.zeros((12, D_smooth), dtype=np.float32)
    linear = np.zeros((12, D_linear), dtype=np.float32)
    miss_s = np.zeros((12, D_smooth), dtype=np.float32)
    miss_l = np.zeros((12, D_linear), dtype=np.float32)

    for m in range(1, 13):
        sel = months == m
        if sel.any():
            smooth[m - 1] = Xs[sel].mean(axis=0)
            linear[m - 1] = Xl[sel].mean(axis=0)
            miss_s[m - 1] = Ms[sel].mean(axis=0)
            miss_l[m - 1] = Ml[sel].mean(axis=0)

    return Climatology(
        smooth=smooth,
        linear=linear,
        miss_smooth=miss_s,
        miss_linear=miss_l,
        smooth_features=list(bundle.smooth_features),
        linear_features=list(bundle.linear_features),
    )


# ---------------------------------------------------------------------------
# future-fold construction
# ---------------------------------------------------------------------------

def _doy_fourier(doy: np.ndarray, z_mean: float = 0.0, z_sd: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Raw (unstandardised) doy sin/cos. z_mean / z_sd override if the panel's
    standardiser was recorded in meta. Not needed for v0..v4 because `_doy_from_fold`
    recovers the angle from the ratio alone, which is z-scale invariant."""
    theta = 2.0 * np.pi * (doy / DAYS_PER_YEAR)
    return np.sin(theta).astype(np.float32), np.cos(theta).astype(np.float32)


def build_future_fold(
    bundle: PanelBundle,
    *,
    start_date: date | str | None = None,
    horizon_days: int = 7,
    climatology: Climatology | None = None,
    weather_override: dict[str, np.ndarray] | None = None,
    training_mask: np.ndarray | None = None,
) -> tuple[FoldData, pd.DataFrame]:
    """Build a `FoldData` where the `_val` side is a future (station \u00d7 day) grid.

    Parameters
    ----------
    bundle           : loaded PanelBundle
    start_date       : first future day (inclusive). Defaults to "day after
                       the last observed date in the panel + 1".
    horizon_days     : how many days to forecast (default 7).
    climatology      : precomputed climatology; if None, computed on the fly
                       using every row with `cv_val_year >= 0`.
    weather_override : optional `{feature_name: (N_future,) array}` of
                       standardised weather values to plug in instead of the
                       climatology means. Missing features fall back to
                       climatology.
    training_mask    : boolean mask selecting rows for the `_train` side of
                       the returned fold. Defaults to "every observed panel
                       row" \u2014 this is needed because predict() re-uses the
                       same training-side arrays for the numpyro model spec
                       even though it only reads the val side.

    Returns
    -------
    fold            : FoldData with `y_log_val` filled with NaN (unknown), all
                      other `_val` arrays aligned and standardised.
    index_df        : `DataFrame` with columns
                      [row_idx, date, station_idx, station_id, station_name,
                      county_idx, county_name, month] so downstream consumers
                      can join posterior samples to station/date.
    """
    # -- date grid --------------------------------------------------------
    date_min_ns = bundle.date_min
    last_t = int(bundle.t_idx.max())
    last_date = (date_min_ns + np.timedelta64(last_t, "D")).astype("datetime64[D]").astype(date)
    if start_date is None:
        start = last_date + timedelta(days=1)
    elif isinstance(start_date, str):
        start = date.fromisoformat(start_date)
    else:
        start = start_date

    future_dates = [start + timedelta(days=i) for i in range(horizon_days)]
    future_dates_np = np.array([np.datetime64(d) for d in future_dates])

    n_stations = bundle.n_stations
    n_fut = horizon_days * n_stations

    # station \u00d7 day outer product. Iterate days outer, stations inner, so that
    # rows 0..n_stations-1 are day 1, etc.
    station_idx = np.tile(np.arange(n_stations, dtype=np.int32), horizon_days)
    day_of_row = np.repeat(np.arange(horizon_days, dtype=np.int32), n_stations)

    # Resolve a county per station from the panel (first observed county for
    # each station). This is a lookup table; stations don't move counties.
    county_by_station = _station_to_county_lookup(bundle)

    county_idx = county_by_station[station_idx].astype(np.int32)
    dates_for_rows = future_dates_np[day_of_row]

    # -- date-derived features -------------------------------------------
    month_val = np.array(
        [pd.Timestamp(d).month for d in dates_for_rows], dtype=np.int8
    )
    doy_val = np.array(
        [pd.Timestamp(d).dayofyear for d in dates_for_rows], dtype=np.float32
    )

    # -- weather features via climatology --------------------------------
    if climatology is None:
        climatology = compute_climatology(bundle)

    # climatology.smooth is (12, D_smooth); index by month-1
    X_smooth_val = climatology.smooth[month_val - 1].astype(np.float32)
    X_linear_val = climatology.linear[month_val - 1].astype(np.float32)
    miss_smooth_val = climatology.miss_smooth[month_val - 1].astype(np.int8)
    miss_linear_val = climatology.miss_linear[month_val - 1].astype(np.int8)

    # Overwrite doy_sin / doy_cos on the linear side with the *actual* future
    # doy (climatology averages would be month-mean not day-of-year exact).
    raw_sin, raw_cos = _doy_fourier(doy_val)
    if "doy_sin" in climatology.linear_features:
        i_sin = climatology.linear_features.index("doy_sin")
        X_linear_val[:, i_sin] = raw_sin
    if "doy_cos" in climatology.linear_features:
        i_cos = climatology.linear_features.index("doy_cos")
        X_linear_val[:, i_cos] = raw_cos

    # Any caller-supplied weather override
    if weather_override:
        name_to_s = {n: i for i, n in enumerate(climatology.smooth_features)}
        name_to_l = {n: i for i, n in enumerate(climatology.linear_features)}
        for name, arr in weather_override.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape != (n_fut,):
                raise ValueError(
                    f"weather_override['{name}'] must have shape ({n_fut},); got {arr.shape}"
                )
            if name in name_to_s:
                X_smooth_val[:, name_to_s[name]] = arr
            elif name in name_to_l:
                X_linear_val[:, name_to_l[name]] = arr
            else:
                raise KeyError(f"weather_override feature '{name}' not in panel features")

    # -- train-side carry-over -------------------------------------------
    # The Bayesian model function needs a train design only to size the latent
    # hierarchical vectors (n_stations, n_counties). We pass any observed rows
    # so those shapes line up; `.predict()` never reads `_train` arrays.
    if training_mask is None:
        training_mask = np.ones_like(bundle.y_log, dtype=bool)

    fold = FoldData(
        fold_val_year=-1,      # sentinel for "future"
        # train side (dummy, just needs to be shape-consistent)
        y_log_train=bundle.y_log[training_mask],
        month_train=bundle.month[training_mask],
        station_idx_train=bundle.station_idx[training_mask],
        county_idx_train=bundle.county_idx[training_mask],
        X_smooth_train=bundle.X_smooth[training_mask],
        X_linear_train=bundle.X_linear[training_mask],
        miss_smooth_train=bundle.miss_smooth[training_mask],
        miss_linear_train=bundle.miss_linear[training_mask],
        left_mask_train=bundle.left_mask[training_mask],
        right_mask_train=bundle.right_mask[training_mask],
        det_low_log_train=bundle.det_low_log[training_mask],
        det_high_log_train=bundle.det_high_log[training_mask],
        # val side = future
        y_log_val=np.full(n_fut, np.nan, dtype=np.float32),    # unknown
        month_val=month_val,
        station_idx_val=station_idx,
        county_idx_val=county_idx,
        X_smooth_val=X_smooth_val,
        X_linear_val=X_linear_val,
        miss_smooth_val=miss_smooth_val,
        miss_linear_val=miss_linear_val,
        smooth_features=list(bundle.smooth_features),
        linear_features=list(bundle.linear_features),
        n_stations=bundle.n_stations,
        n_counties=bundle.n_counties,
    )

    index_df = pd.DataFrame({
        "row_idx":      np.arange(n_fut, dtype=np.int64),
        "date":         pd.to_datetime(dates_for_rows),
        "station_idx":  station_idx,
        "station_id":   bundle.station_ids[station_idx],
        "station_name": bundle.station_names[station_idx],
        "county_idx":   county_idx,
        "county_name":  bundle.county_names[county_idx],
        "month":        month_val,
        "doy":          doy_val.astype(int),
    })

    return fold, index_df


def _station_to_county_lookup(bundle: PanelBundle) -> np.ndarray:
    """For each station_idx, return its county_idx (first observed mapping).

    Stations don't move counties in this panel (station_id is unique per
    beach), but cv.load_panel doesn't surface the mapping explicitly, so we
    recover it from observed rows.
    """
    n_stations = bundle.n_stations
    out = np.full(n_stations, -1, dtype=np.int32)
    seen = np.zeros(n_stations, dtype=bool)
    for s, c in zip(bundle.station_idx, bundle.county_idx):
        if not seen[s]:
            out[s] = c
            seen[s] = True
    if (out < 0).any():
        missing = np.where(out < 0)[0]
        # Fallback: broadcast county 0. Shouldn't happen if the panel has an
        # observation for every station, but be defensive.
        out[missing] = 0
    return out


# ---------------------------------------------------------------------------
# tidy-export helpers
# ---------------------------------------------------------------------------

EXCEEDANCE_THRESHOLD_MPN = 104.0            # single-sample regulatory advisory
EXCEEDANCE_THRESHOLD_LOG10 = float(np.log10(EXCEEDANCE_THRESHOLD_MPN))


def summarise_posterior_mpn(samples_log10: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the standard summary statistics every dashboard wants.

    Back-transformation to MPN is done before computing medians / quantiles,
    NOT by exponentiating summaries of `log10`. That preserves the
    interpretability the stakeholders asked for.
    """
    samples_mpn = 10.0 ** samples_log10            # (S, N)
    point_mpn = np.median(samples_mpn, axis=0)
    # median in log-space is useful for plotting the "central line" on a
    # log-scale chart without having to re-log the median-of-MPN downstream.
    point_log10 = np.median(samples_log10, axis=0)
    q_lo50, q_hi50 = np.quantile(samples_mpn, [0.25, 0.75], axis=0)
    q_lo80, q_hi80 = np.quantile(samples_mpn, [0.10, 0.90], axis=0)
    q_lo95, q_hi95 = np.quantile(samples_mpn, [0.025, 0.975], axis=0)
    p_exceed = (samples_log10 > EXCEEDANCE_THRESHOLD_LOG10).mean(axis=0)
    return {
        "point_mpn_median":   point_mpn,
        "point_log10_median": point_log10,
        "pi50_low_mpn":       q_lo50,   "pi50_high_mpn": q_hi50,
        "pi80_low_mpn":       q_lo80,   "pi80_high_mpn": q_hi80,
        "pi95_low_mpn":       q_lo95,   "pi95_high_mpn": q_hi95,
        "p_exceed_104mpn":    p_exceed,
    }


def _write_dataframe(df: pd.DataFrame, path_no_ext: Path, *, label: str = "") -> Path:
    """Write `df` as parquet if the installed pyarrow works, else fall back to pickle.

    Some environments ship incompatible (`pyarrow < 16`) + `numpy >= 2` wheels
    which fail even for trivial DataFrames. Instead of crashing the forecast
    pipeline, we detect the failure once, write a pickle, and print a single
    advisory so the user knows Power BI will need the pickle or a pyarrow
    upgrade.
    """
    parquet_path = Path(str(path_no_ext) + ".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        pkl_path = Path(str(path_no_ext) + ".pkl")
        df.to_pickle(pkl_path)
        print(
            f"  \u26a0  could not write {label or 'dataframe'} as parquet "
            f"({type(exc).__name__}: {exc}); wrote pickle instead \u2014 "
            f"`pip install -U pyarrow` to fix if Power BI needs parquet."
        )
        return pkl_path


def read_forecast_frame(path: Path | str) -> pd.DataFrame:
    """Robust read for the tidy frames written by `export_forecast_bundle`.

    Accepts either the `.parquet` or `.pkl` path produced by the exporter, so
    downstream code never has to care which fallback happened.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"unknown forecast frame extension: {path}")


def export_forecast_bundle(
    out_dir: Path | str,
    *,
    tag: str,
    index_df: pd.DataFrame,
    samples_log10: np.ndarray,
    meta_extra: dict | None = None,
) -> dict[str, Path]:
    """Write the three teammate-facing artifacts for a forecast horizon.

    Files written (all under `out_dir`):

      - `<tag>_forecast.parquet`   tidy one-row-per-(station, date) with
                                    median, 50/80/95 % PI, and p_exceed. This
                                    is the **Power BI** entry point.

      - `<tag>_samples.npz`        raw `(S, N)` posterior-predictive draws in
                                    log10(MPN). This is the **internal web**
                                    entry point \u2014 any custom aggregation a
                                    teammate wants (e.g. week-level exceedance,
                                    cross-station correlation plots) is a
                                    one-liner on this array.

      - `<tag>_index.parquet`      row_idx \u2192 (station, date, county, month)
                                    mapping so the samples npz can be joined
                                    back to a tidy frame.

      - `<tag>_meta.json`          provenance: samples shape, tag, date range,
                                    threshold, any caller-supplied extras.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarise_posterior_mpn(samples_log10)
    tidy = index_df.copy()
    for k, v in summary.items():
        tidy[k] = v

    forecast_path = _write_dataframe(tidy, out_dir / f"{tag}_forecast", label="forecast")

    samples_path = out_dir / f"{tag}_samples.npz"
    np.savez_compressed(
        samples_path,
        samples_log10=samples_log10.astype(np.float32),
        row_idx=index_df["row_idx"].to_numpy(),
    )

    index_path = _write_dataframe(index_df, out_dir / f"{tag}_index", label="index")

    meta = {
        "tag": tag,
        "n_rows": int(len(index_df)),
        "n_samples": int(samples_log10.shape[0]),
        "date_min": str(index_df["date"].min().date()) if len(index_df) else None,
        "date_max": str(index_df["date"].max().date()) if len(index_df) else None,
        "exceedance_threshold_mpn": EXCEEDANCE_THRESHOLD_MPN,
        "forecast_path": str(forecast_path.name),
        "samples_path": str(samples_path.name),
        "index_path": str(index_path.name),
    }
    if meta_extra:
        meta.update(meta_extra)
    meta_path = out_dir / f"{tag}_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    return {
        "forecast": forecast_path,
        "samples": samples_path,
        "index": index_path,
        "meta": meta_path,
    }
