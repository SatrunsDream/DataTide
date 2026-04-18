"""Load auxiliary environmental tables from `data/raw` for joining to the Tier-1 bacteria spine."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# GHCN station id suffix → bucket name matching configs/fetch.yaml noaa_precip.stations[].name
GHCND_TO_PRECIP_BUCKET: dict[str, str] = {
    "USW00024283": "crescent_city",
    "USW00024243": "eureka",
    "USW00023234": "san_francisco",
    "USW00023259": "monterey",
    "USW00023129": "santa_barbara",
    "USW00023174": "los_angeles",
    "USW00023188": "san_diego",
}

_TIDE_FILE_RE = re.compile(r"_(\d{6,8})_(\d{4})_")


def load_precip_daily(noaa_precip_dir: Path) -> pd.DataFrame:
    """One row per (calendar date, precip_bucket). Values are mm/day (CDO PRCP tenths → mm)."""
    rows: list[dict] = []
    for path in sorted(noaa_precip_dir.glob("cdo_PRCP_*.json")):
        if "meta" in path.name.lower():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for rec in payload.get("results") or []:
            station = str(rec.get("station") or "")
            suffix = station.split(":")[-1] if ":" in station else station
            bucket = GHCND_TO_PRECIP_BUCKET.get(suffix)
            if not bucket:
                continue
            d = rec.get("date")
            if not d:
                continue
            day = str(d)[:10]
            val = rec.get("value")
            try:
                tenths = float(val)
            except (TypeError, ValueError):
                continue
            rows.append({"sample_date": day, "precip_bucket": bucket, "regional_ghcn_prcp_mm": tenths / 10.0})
    if not rows:
        return pd.DataFrame(columns=["sample_date", "precip_bucket", "regional_ghcn_prcp_mm"])
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["sample_date", "precip_bucket"], keep="last")


def load_tide_daily(noaa_tides_dir: Path) -> pd.DataFrame:
    """Per calendar day and tide gauge: tidal range (m) from high/low predictions."""
    out_rows: list[dict] = []
    for path in sorted(noaa_tides_dir.glob("tides_hilo_*.json")):
        m = _TIDE_FILE_RE.search(path.name)
        if not m:
            continue
        station_id = m.group(1)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        preds = payload.get("predictions") or []
        by_day: dict[str, dict[str, list[float]]] = {}
        for p in preds:
            t = p.get("t")
            typ = p.get("type")
            if not t or typ not in ("H", "L"):
                continue
            day = str(t).split()[0]
            try:
                v = float(p.get("v"))
            except (TypeError, ValueError):
                continue
            bucket = by_day.setdefault(day, {"H": [], "L": []})
            bucket[typ].append(v)
        for day, hl in by_day.items():
            highs, lows = hl["H"], hl["L"]
            if not highs or not lows:
                continue
            tr = max(highs) - min(lows)
            out_rows.append(
                {"sample_date": day, "tide_station_id": station_id, "tide_range_hilo_m": round(tr, 4)}
            )
    if not out_rows:
        return pd.DataFrame(columns=["sample_date", "tide_station_id", "tide_range_hilo_m"])
    df = pd.DataFrame(out_rows)
    return df.drop_duplicates(subset=["sample_date", "tide_station_id"], keep="last")


def load_station_supplement(stations_csv: Path) -> pd.DataFrame:
    """Extra station metadata not always present on bacteria rows (e.g. vertical datum)."""
    df = pd.read_csv(
        stations_csv,
        usecols=lambda c: c
        in {"Station_id", "Datum", "CountyCode", "AgencyStationIdentifier", "Station_Description"},
        low_memory=False,
    )
    df["Station_id"] = pd.to_numeric(df["Station_id"], errors="coerce").astype("Int64")
    df = df.drop_duplicates(subset=["Station_id"], keep="last")
    return df.rename(
        columns={
            "Datum": "station_datum",
            "CountyCode": "station_county_code",
            "AgencyStationIdentifier": "agency_station_id",
            "Station_Description": "station_description_ref",
        }
    )


def load_sccoos_delmar_daily(sccoos_dir: Path) -> pd.DataFrame:
    """Daily mean Del Mar (SCCOOS) 1 m temperature and salinity from ERDDAP monthly CSV shards."""
    temp_rows: list[pd.DataFrame] = []
    sal_rows: list[pd.DataFrame] = []
    for path in sorted(sccoos_dir.glob("sccoos_delmar_temperature_*.csv")):
        try:
            df = pd.read_csv(path, skiprows=[1], low_memory=False)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if "time" not in df.columns or "T_1m" not in df.columns:
            continue
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time", "T_1m"])
        df["sample_date"] = df["time"].dt.strftime("%Y-%m-%d")
        temp_rows.append(df[["sample_date", "T_1m"]])
    for path in sorted(sccoos_dir.glob("sccoos_delmar_salinity_*.csv")):
        try:
            df = pd.read_csv(path, skiprows=[1], low_memory=False)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if "time" not in df.columns or "S_1m" not in df.columns:
            continue
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time", "S_1m"])
        df["sample_date"] = df["time"].dt.strftime("%Y-%m-%d")
        sal_rows.append(df[["sample_date", "S_1m"]])

    out: dict[str, pd.Series | str] = {}
    if temp_rows:
        tdf = pd.concat(temp_rows, ignore_index=True)
        out["sccoos_delmar_temp_1m_c"] = tdf.groupby("sample_date", sort=False)["T_1m"].mean()
    if sal_rows:
        sdf = pd.concat(sal_rows, ignore_index=True)
        out["sccoos_delmar_salinity_1m_psu"] = sdf.groupby("sample_date", sort=False)["S_1m"].mean()
    if not out:
        return pd.DataFrame(columns=["sample_date", "sccoos_delmar_temp_1m_c", "sccoos_delmar_salinity_1m_psu"])
    df_out = pd.DataFrame(out).reset_index()
    return df_out


def _cdip_bundle_digits(bundle_name: str) -> str:
    m = re.search(r"(\d{3})", bundle_name)
    return m.group(1) if m else ""


def _latest_cdip_nc(cdip_dir: Path, digits: str) -> Path | None:
    cands = list(cdip_dir.glob(f"*{digits}*p1_rt*.nc"))
    if not cands:
        cands = list(cdip_dir.glob(f"*{digits}*.nc"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def load_cdip_buoy_meta_and_daily(
    cdip_dir: Path, bundle_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-bundle deploy location + daily mean wave height / peak period from latest realtime NetCDF each."""
    try:
        import xarray as xr
    except ImportError:
        return (
            pd.DataFrame(columns=["cdip_bundle", "cdip_buoy_lat", "cdip_buoy_lon"]),
            pd.DataFrame(
                columns=["sample_date", "cdip_bundle", "cdip_wave_hs_m_mean", "cdip_wave_tp_s_mean"]
            ),
        )

    meta_rows: list[dict] = []
    daily_parts: list[pd.DataFrame] = []
    for name in bundle_names:
        digits = _cdip_bundle_digits(name)
        if not digits:
            continue
        path = _latest_cdip_nc(cdip_dir, digits)
        if path is None:
            continue
        try:
            ds = xr.open_dataset(path)
            lat = float(ds["metaDeployLatitude"].values)
            lon = float(ds["metaDeployLongitude"].values)
            meta_rows.append({"cdip_bundle": name, "cdip_buoy_lat": lat, "cdip_buoy_lon": lon})
            wh = ds["waveHs"].to_series()
            wh.index = pd.to_datetime(wh.index)
            tp = ds["waveTp"].to_series()
            tp.index = pd.to_datetime(tp.index)
            df = pd.DataFrame({"wave_hs": wh, "wave_tp": tp})
            df = df.dropna(how="all")
            df["sample_date"] = df.index.strftime("%Y-%m-%d")
            g = df.groupby("sample_date", sort=False).mean(numeric_only=True).reset_index()
            g = g.rename(
                columns={"wave_hs": "cdip_wave_hs_m_mean", "wave_tp": "cdip_wave_tp_s_mean"}
            )
            g["cdip_bundle"] = name
            daily_parts.append(g)
            ds.close()
        except (OSError, KeyError, ValueError, TypeError):
            continue

    meta = pd.DataFrame(meta_rows)
    if not daily_parts:
        daily = pd.DataFrame(
            columns=["sample_date", "cdip_bundle", "cdip_wave_hs_m_mean", "cdip_wave_tp_s_mean"]
        )
    else:
        daily = pd.concat(daily_parts, ignore_index=True)
        daily = daily.drop_duplicates(subset=["sample_date", "cdip_bundle"], keep="last")
    return meta, daily


def assign_nearest_cdip_bundle(
    lat: np.ndarray, lon: np.ndarray, meta: pd.DataFrame
) -> np.ndarray:
    """Vectorized nearest-buoy label; NaN coords → empty string."""
    n = len(lat)
    out = np.array([""] * n, dtype=object)
    if meta is None or len(meta) == 0:
        return out
    bla = meta["cdip_buoy_lat"].to_numpy(dtype=float)
    blo = meta["cdip_buoy_lon"].to_numpy(dtype=float)
    bundles = meta["cdip_bundle"].to_numpy()
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        return out
    lat_e = np.radians(lat[ok, None])
    lon_e = np.radians(lon[ok, None])
    bla_e = np.radians(bla[None, :])
    blo_e = np.radians(blo[None, :])
    dlat = bla_e - lat_e
    dlon = blo_e - lon_e
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_e) * np.cos(bla_e) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    r_km = 6371.0 * c
    idx = np.argmin(r_km, axis=1)
    assigned = bundles[idx]
    out[ok] = assigned
    return out


def load_sd_county_coastal_monthly(sd_dir: Path) -> pd.DataFrame:
    """San Diego County beach program monthly aggregates (county-wide, not per sampling station)."""
    files = sorted(sd_dir.glob("beach_advisories_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return pd.DataFrame()
    try:
        rows = json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "calendar_month": pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str),
            "CountyName": "San Diego",
        }
    )
    for c in df.columns:
        out[f"sd_coastal_{c}"] = df[c]
    return out.drop_duplicates(subset=["calendar_month", "CountyName"], keep="last")
