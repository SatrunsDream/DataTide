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
    df = df.drop_duplicates(subset=["sample_date", "precip_bucket"], keep="last")

    df["sample_date"] = pd.to_datetime(df["sample_date"])
    df = df.sort_values(["precip_bucket", "sample_date"])

    def compute_lags(group):
        bucket = group.name
        group = group.set_index("sample_date").asfreq("D", fill_value=0.0)
        group["precip_bucket"] = bucket
        # We assume 24h lag means the rainfall of the previous day, 48h means previous 2 days, etc.
        # This prevents target leakage.
        group["regional_ghcn_prcp_mm_24h"] = group["regional_ghcn_prcp_mm"].shift(1).fillna(0)
        group["regional_ghcn_prcp_mm_48h"] = group["regional_ghcn_prcp_mm"].rolling(window=2, min_periods=1).sum().shift(1).fillna(0)
        group["regional_ghcn_prcp_mm_72h"] = group["regional_ghcn_prcp_mm"].rolling(window=3, min_periods=1).sum().shift(1).fillna(0)
        group["regional_ghcn_prcp_mm_7d"] = group["regional_ghcn_prcp_mm"].rolling(window=7, min_periods=1).sum().shift(1).fillna(0)

        # dry_days_since_rain: consecutive days where rain <= 0.1mm BEFORE current day
        is_rain = group["regional_ghcn_prcp_mm"] > 0.1
        rain_events = is_rain.cumsum()
        dry_days = group.groupby(rain_events).cumcount()
        group["dry_days_since_rain"] = dry_days.shift(1).fillna(0).astype(int)

        return group.reset_index()

    if not df.empty:
        df = df.groupby("precip_bucket").apply(compute_lags).reset_index(drop=True)
    
    df["sample_date"] = df["sample_date"].dt.strftime("%Y-%m-%d")
    return df


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


def load_beach_detail_supplement(beach_csv: Path) -> pd.DataFrame:
    """Tier-1 beach metadata export; join on BeachName_id. Columns prefixed to avoid clashing with bacteria fields."""
    df = pd.read_csv(beach_csv, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "Beach_ UpperLon" in df.columns:
        df = df.rename(columns={"Beach_ UpperLon": "Beach_UpperLon"})
    if "BeachName_id" not in df.columns:
        return pd.DataFrame()
    df["BeachName_id"] = pd.to_numeric(df["BeachName_id"], errors="coerce").astype("Int64")
    df = df.drop_duplicates(subset=["BeachName_id"], keep="last")
    out = df[["BeachName_id"]].copy()
    for c in df.columns:
        if c == "BeachName_id":
            continue
        safe = str(c).strip().replace(" ", "_")
        out[f"beach_detail_{safe}"] = df[c].values
    return out


def load_ibwc_tijuana_daily(ibwc_dir: Path) -> pd.DataFrame:
    """Parse IBWC legacy text table → daily mean stage (m) and discharge (m³/s)."""
    files = sorted(ibwc_dir.glob("tijuana_gauge_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return pd.DataFrame(
            columns=["sample_date", "ibwc_tijuana_stage_m_daily_mean", "ibwc_tijuana_discharge_cms_daily_mean"]
        )
    text = files[0].read_text(encoding="utf-8", errors="replace")
    row_re = re.compile(
        r"^(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})\s+([\d.]+)\s+([\d.]+)\s*$",
        re.MULTILINE,
    )
    rows: list[tuple[str, float, float]] = []
    for m in row_re.finditer(text):
        day, _t, stage_s, q_s = m.groups()
        try:
            stage = float(stage_s)
            q = float(q_s)
        except ValueError:
            continue
        iso = pd.to_datetime(day, format="%m/%d/%Y", errors="coerce")
        if pd.isna(iso):
            continue
        rows.append((iso.strftime("%Y-%m-%d"), stage, q))
    if not rows:
        return pd.DataFrame(
            columns=["sample_date", "ibwc_tijuana_stage_m_daily_mean", "ibwc_tijuana_discharge_cms_daily_mean"]
        )
    raw = pd.DataFrame(rows, columns=["sample_date", "_st", "_q"])
    g = raw.groupby("sample_date", sort=False).agg(
        ibwc_tijuana_stage_m_daily_mean=("_st", "mean"),
        ibwc_tijuana_discharge_cms_daily_mean=("_q", "mean"),
    ).reset_index()
    return g


def load_bwtf_state_daily(bwtf_dir: Path) -> pd.DataFrame:
    """Surfrider BWTF JSON shards → daily median numeric result (Enterococcus / E.coli) in CA bbox.

    Site-level alignment is future work; this is a weak statewide same-day context signal.
    """
    paths = sorted(bwtf_dir.glob("bwtf_water_quality_*.json"))
    if not paths:
        return pd.DataFrame(
            columns=["sample_date", "bwtf_ca_median_result", "bwtf_ca_n_samples", "bwtf_ca_median_entero"]
        )
    ca_lat = (32.4, 42.1)
    ca_lon = (-124.5, -114.1)
    day_vals: dict[str, list[float]] = {}
    day_entero: dict[str, list[float]] = {}
    for path in paths:
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            loc = item.get("location") or {}
            coord = loc.get("coordinate") or {}
            try:
                lat = float(coord.get("latitude"))
                lon = float(coord.get("longitude"))
            except (TypeError, ValueError):
                continue
            if not (ca_lat[0] <= lat <= ca_lat[1] and ca_lon[0] <= lon <= ca_lon[1]):
                continue
            t = item.get("collectionTime")
            if not t:
                continue
            day = str(t)[:10]
            if len(day) < 10:
                continue
            for s in item.get("samples") or []:
                sub = str(s.get("substance") or "").lower()
                raw_r = s.get("result")
                try:
                    val = float(raw_r)
                except (TypeError, ValueError):
                    continue
                day_vals.setdefault(day, []).append(val)
                if "entero" in sub:
                    day_entero.setdefault(day, []).append(val)
    if not day_vals:
        return pd.DataFrame(
            columns=["sample_date", "bwtf_ca_median_result", "bwtf_ca_n_samples", "bwtf_ca_median_entero"]
        )
    rows = []
    for day, vals in sorted(day_vals.items()):
        ent = day_entero.get(day) or []
        rows.append(
            {
                "sample_date": day,
                "bwtf_ca_median_result": float(np.median(vals)),
                "bwtf_ca_n_samples": len(vals),
                "bwtf_ca_median_entero": float(np.median(ent)) if ent else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """lat1/lon1 shape (n,1), lat2/lon2 shape (1,m) or broadcastable → (n,m) km."""
    r = 6371.0
    ph1 = np.radians(lat1)
    ph2 = np.radians(lat2)
    dph = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dph / 2) ** 2 + np.cos(ph1) * np.cos(ph2) * np.sin(dl / 2) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return r * c


def load_hf_radar_daily_lookup(hf_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], tuple[float, float, float, float] | None]:
    """Latest HF radar u/v CSV pair → per-UTC-date grid (lat, lon, u, v) and bounding box."""
    u_files = sorted(hf_dir.glob("hf_*_water_u_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    v_files = sorted(hf_dir.glob("hf_*_water_v_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not u_files or not v_files:
        return {}, None
    u_path, v_path = u_files[0], v_files[0]
    u = pd.read_csv(u_path, skiprows=[1], low_memory=False)
    v = pd.read_csv(v_path, skiprows=[1], low_memory=False)
    for col in ("time", "latitude", "longitude"):
        if col not in u.columns or col not in v.columns:
            return {}, None
    u["time"] = pd.to_datetime(u["time"], utc=True, errors="coerce")
    v["time"] = pd.to_datetime(v["time"], utc=True, errors="coerce")
    u = u.dropna(subset=["time", "latitude", "longitude"])
    v = v.dropna(subset=["time", "latitude", "longitude"])
    var_u = [c for c in u.columns if c not in ("time", "latitude", "longitude")][0]
    var_v = [c for c in v.columns if c not in ("time", "latitude", "longitude")][0]
    u = u.rename(columns={var_u: "_u"})
    v = v.rename(columns={var_v: "_v"})
    merged = u.merge(v, on=["time", "latitude", "longitude"], how="inner")
    merged["sample_date"] = merged["time"].dt.strftime("%Y-%m-%d")
    agg = (
        merged.groupby(["sample_date", "latitude", "longitude"], sort=False)[["_u", "_v"]]
        .mean()
        .reset_index()
    )
    lat_min = float(agg["latitude"].min())
    lat_max = float(agg["latitude"].max())
    lon_min = float(agg["longitude"].min())
    lon_max = float(agg["longitude"].max())
    bbox = (lat_min, lat_max, lon_min, lon_max)
    by_day: dict[str, dict[str, np.ndarray]] = {}
    for day, g in agg.groupby("sample_date", sort=False):
        by_day[str(day)] = {
            "lat": g["latitude"].to_numpy(dtype=float),
            "lon": g["longitude"].to_numpy(dtype=float),
            "u": g["_u"].to_numpy(dtype=float),
            "v": g["_v"].to_numpy(dtype=float),
        }
    return by_day, bbox


def assign_hf_radar_to_chunk(
    chunk: pd.DataFrame,
    by_day: dict[str, dict[str, np.ndarray]],
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    """Add hf_water_u_mps, hf_water_v_mps, hf_current_speed_mps, hf_grid_dist_km (nearest grid cell)."""
    chunk["hf_water_u_mps"] = np.nan
    chunk["hf_water_v_mps"] = np.nan
    chunk["hf_current_speed_mps"] = np.nan
    chunk["hf_grid_dist_km"] = np.nan
    if not by_day or bbox is None:
        return chunk
    lat_min, lat_max, lon_min, lon_max = bbox
    lat = pd.to_numeric(chunk["Station_UpperLat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(chunk["Station_UpperLon"], errors="coerce").to_numpy()
    dates = chunk["sample_date"].astype(str).to_numpy()
    in_box = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= lat_min)
        & (lat <= lat_max)
        & (lon >= lon_min)
        & (lon <= lon_max)
    )
    idxs = np.flatnonzero(in_box)
    if idxs.size == 0:
        return chunk
    for day in np.unique(dates[idxs]):
        g = by_day.get(str(day))
        if g is None or len(g["lat"]) == 0:
            continue
        sel = idxs[dates[idxs] == day]
        sla = lat[sel]
        slo = lon[sel]
        glat = g["lat"]
        glon = g["lon"]
        d = _haversine_km(sla[:, None], slo[:, None], glat[None, :], glon[None, :])
        nn = np.argmin(d, axis=1)
        dkm = d[np.arange(sla.shape[0], dtype=int), nn]
        u = g["u"][nn]
        v = g["v"][nn]
        sp = np.hypot(u, v)
        ii = chunk.index[sel]
        chunk.loc[ii, "hf_water_u_mps"] = u
        chunk.loc[ii, "hf_water_v_mps"] = v
        chunk.loc[ii, "hf_current_speed_mps"] = sp
        chunk.loc[ii, "hf_grid_dist_km"] = dkm
    return chunk


def assign_nearest_cce_mooring(lat: np.ndarray, lon: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Vectorized nearest CCE mooring id (int); invalid coords → -1."""
    n = len(lat)
    out = np.full(n, -1, dtype=np.int64)
    if meta is None or len(meta) == 0:
        return out
    mids = meta["cce_mooring_id"].to_numpy(dtype=np.int64)
    mlat = meta["cce_mooring_lat"].to_numpy(dtype=float)
    mlon = meta["cce_mooring_lon"].to_numpy(dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        return out
    lat_e = np.radians(lat[ok, None])
    lon_e = np.radians(lon[ok, None])
    bla = np.radians(mlat[None, :])
    blo = np.radians(mlon[None, :])
    dlat = bla - lat_e
    dlon = blo - lon_e
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_e) * np.cos(bla) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    r_km = 6371.0 * c
    idx = np.argmin(r_km, axis=1)
    out[ok] = mids[idx]
    return out


def load_cce_mooring_daily(cce_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OceanSITES CCE1 CTD + ADCP: daily means at shallowest depth; meta for two mooring IDs (13, 15)."""
    try:
        import xarray as xr
    except ImportError:
        return (
            pd.DataFrame(columns=["cce_mooring_id", "cce_mooring_lat", "cce_mooring_lon"]),
            pd.DataFrame(
                columns=[
                    "sample_date",
                    "cce_mooring_id",
                    "cce_temp_shallow_c",
                    "cce_psal_shallow_psu",
                    "cce_ucur_shallow_mps",
                    "cce_vcur_shallow_mps",
                ]
            ),
        )

    moorings: dict[int, dict[str, float | None]] = {}

    def _mooring_lat_lon(path: Path) -> tuple[int, float, float]:
        m = re.search(r"OS_CCE1_(\d+)_D_", path.name)
        mid = int(m.group(1)) if m else -1
        ds = xr.open_dataset(path)
        la = float(ds["LATITUDE"].values.flat[0])
        lo = float(ds["LONGITUDE"].values.flat[0])
        ds.close()
        return mid, la, lo

    def _daily_ctd(path: Path) -> pd.DataFrame:
        with xr.open_dataset(path) as ds:
            t = ds["TEMP"].isel(DEPTH=0).load()
            ser_t = t.to_series()
            ser_t.index = pd.to_datetime(ser_t.index, utc=True, errors="coerce")
            psal = ds["PSAL"].isel(DEPTH=0).load().to_series()
            psal.index = pd.to_datetime(psal.index, utc=True, errors="coerce")
        df = pd.DataFrame({"temp": ser_t, "psal": psal}).dropna(how="all")
        df = df[~df.index.isna()]
        df["sample_date"] = df.index.strftime("%Y-%m-%d")
        return df.groupby("sample_date", sort=False).mean(numeric_only=True).reset_index()

    def _daily_adcp(path: Path) -> pd.DataFrame:
        with xr.open_dataset(path) as ds:
            u = ds["UCUR"].isel(DEPTH=0).load().to_series()
            u.index = pd.to_datetime(u.index, utc=True, errors="coerce")
            v = ds["VCUR"].isel(DEPTH=0).load().to_series()
            v.index = pd.to_datetime(v.index, utc=True, errors="coerce")
        df = pd.DataFrame({"ucur": u, "vcur": v}).dropna(how="all")
        df = df[~df.index.isna()]
        df["sample_date"] = df.index.strftime("%Y-%m-%d")
        return df.groupby("sample_date", sort=False).mean(numeric_only=True).reset_index()

    daily_parts: list[pd.DataFrame] = []
    for mid in (13, 15):
        ctd_cands = list(cce_dir.glob(f"OS_CCE1_{mid}_D_CTD*.nc"))
        adcp_cands = list(cce_dir.glob(f"OS_CCE1_{mid}_D_ADCP*.nc"))
        ctd_p = max(ctd_cands, key=lambda p: p.stat().st_mtime) if ctd_cands else None
        adcp_p = max(adcp_cands, key=lambda p: p.stat().st_mtime) if adcp_cands else None
        if ctd_p:
            m_id, la, lo = _mooring_lat_lon(ctd_p)
            moorings[m_id] = {"lat": la, "lon": lo}
        elif adcp_p:
            m_id, la, lo = _mooring_lat_lon(adcp_p)
            moorings[m_id] = {"lat": la, "lon": lo}
        else:
            continue
        dfc = _daily_ctd(ctd_p) if ctd_p else pd.DataFrame(columns=["sample_date"])
        dfa = _daily_adcp(adcp_p) if adcp_p else pd.DataFrame(columns=["sample_date"])
        if len(dfc) and len(dfa):
            m = dfc.merge(dfa, on="sample_date", how="outer")
        elif len(dfc):
            m = dfc.copy()
            m["ucur"] = np.nan
            m["vcur"] = np.nan
        elif len(dfa):
            m = dfa.copy()
            m["temp"] = np.nan
            m["psal"] = np.nan
        else:
            continue
        m["cce_mooring_id"] = mid
        m = m.rename(
            columns={
                "temp": "cce_temp_shallow_c",
                "psal": "cce_psal_shallow_psu",
                "ucur": "cce_ucur_shallow_mps",
                "vcur": "cce_vcur_shallow_mps",
            }
        )
        daily_parts.append(m)

    if not moorings:
        return (
            pd.DataFrame(columns=["cce_mooring_id", "cce_mooring_lat", "cce_mooring_lon"]),
            pd.DataFrame(
                columns=[
                    "sample_date",
                    "cce_mooring_id",
                    "cce_temp_shallow_c",
                    "cce_psal_shallow_psu",
                    "cce_ucur_shallow_mps",
                    "cce_vcur_shallow_mps",
                ]
            ),
        )

    meta = pd.DataFrame(
        [
            {"cce_mooring_id": k, "cce_mooring_lat": v["lat"], "cce_mooring_lon": v["lon"]}
            for k, v in sorted(moorings.items())
        ]
    )
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    if len(daily):
        daily = daily.drop_duplicates(subset=["sample_date", "cce_mooring_id"], keep="last")
    return meta, daily
