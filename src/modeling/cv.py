"""
Fold iterator and data slicer for the DataTide modeling plan.

The panel artifact packs every row (~295K) including un-observed daily grid
points that only exist so we can forward-fill environmental lags. CV fitting
and scoring happens on the *observed* subset only (`obs_mask=True`). Inside
the observed subset we split by `cv_val_year`:

    cv_val_year == 0            -> pre-2020 rows  (always train)
    cv_val_year in val_years[k] -> fold k validation
    cv_val_year != val_years[k] -> fold k training
    cv_val_year == -1           -> test-set hold-out (never touched by CV)

`iter_folds` yields a `FoldData` object for each requested fold. The caller
passes the in-memory `PanelBundle` (loaded once) rather than re-reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import json
import numpy as np

from src.evaluation.compare import FoldData


def _project_root() -> Path:
    """Walk up from this file to the repo root (directory containing `src/`).

    Using an absolute, module-anchored path means `load_panel()` works from
    any cwd (notebooks, scripts, tests) without the caller having to chdir.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir() and (parent / "artifacts").is_dir():
            return parent
    # fallback: 3 levels up from src/modeling/cv.py  ->  repo root
    return here.parents[2]


PROJECT_ROOT = _project_root()
PANEL_DIR = PROJECT_ROOT / "artifacts" / "data" / "panel"
NPZ_PATH = PANEL_DIR / "enterococcus_panel.npz"
META_PATH = PANEL_DIR / "enterococcus_panel_meta.json"


@dataclass
class PanelBundle:
    """In-memory panel + metadata. Load once, reuse across folds."""

    # observed-only slices (pre-filtered for convenience)
    y_log: np.ndarray
    month: np.ndarray
    station_idx: np.ndarray
    county_idx: np.ndarray
    t_idx: np.ndarray
    X_smooth: np.ndarray
    X_linear: np.ndarray
    miss_smooth: np.ndarray
    miss_linear: np.ndarray
    left_mask: np.ndarray
    right_mask: np.ndarray
    det_low_log: np.ndarray
    det_high_log: np.ndarray
    cv_val_year: np.ndarray

    # global dims + metadata
    smooth_features: list[str]
    linear_features: list[str]
    n_stations: int
    n_counties: int
    station_ids: np.ndarray
    station_names: np.ndarray
    county_names: np.ndarray
    date_min: np.datetime64
    priors: dict
    meta: dict


def _month_from_t_idx(t_idx: np.ndarray, date_min: np.datetime64) -> np.ndarray:
    """Vectorised month-of-year from day offsets. Returns int8 in 1..12."""
    dates = np.asarray(date_min) + t_idx.astype("timedelta64[D]")
    pd_dates = dates.astype("datetime64[D]").astype(object)
    return np.array([d.month for d in pd_dates], dtype=np.int8)


def load_panel(
    npz_path: Path = NPZ_PATH,
    meta_path: Path = META_PATH,
) -> PanelBundle:
    """Load the panel artifact and slice to observed rows.

    We filter to `obs_mask=True` here because every downstream consumer (fit,
    predict, score) only touches observed rows. Un-observed rows exist only to
    supply forward-filled environmental context and are irrelevant once the
    lag features have already been computed upstream.
    """
    npz = np.load(npz_path, allow_pickle=True)
    meta = json.load(open(meta_path))

    obs = npz["obs_mask"].astype(bool)
    t_idx = npz["t_idx"][obs].astype(np.int64)
    date_min = np.datetime64(str(npz["date_min"]))
    month = _month_from_t_idx(t_idx, date_min)

    return PanelBundle(
        y_log=npz["y_log"][obs].astype(np.float32),
        month=month,
        station_idx=npz["station_idx"][obs].astype(np.int32),
        county_idx=npz["county_idx"][obs].astype(np.int32),
        t_idx=t_idx.astype(np.int32),
        X_smooth=npz["X_smooth"][obs].astype(np.float32),
        X_linear=npz["X_linear"][obs].astype(np.float32),
        miss_smooth=npz["miss_smooth"][obs].astype(np.int8),
        miss_linear=npz["miss_linear"][obs].astype(np.int8),
        left_mask=npz["left_mask"][obs].astype(bool),
        right_mask=npz["right_mask"][obs].astype(bool),
        det_low_log=npz["det_low_log"][obs].astype(np.float64),
        det_high_log=npz["det_high_log"][obs].astype(np.float64),
        cv_val_year=npz["cv_val_year"][obs].astype(np.int32),
        smooth_features=list(npz["smooth_features"]),
        linear_features=list(npz["linear_features"]),
        n_stations=int(npz["station_ids"].shape[0]),
        n_counties=int(npz["county_names"].shape[0]),
        station_ids=npz["station_ids"],
        station_names=npz["station_names"],
        county_names=npz["county_names"],
        date_min=date_min,
        priors=meta.get("priors", {}),
        meta=meta,
    )


def iter_folds(
    bundle: PanelBundle,
    val_years: Iterable[int] | None = None,
) -> Iterator[FoldData]:
    """Yield `FoldData` for each requested validation year.

    If `val_years` is None, use all folds in `meta["cv_val_years"]`. A row
    belongs to fold k's validation if `cv_val_year == val_years[k]`; everything
    else with `cv_val_year >= 0` and `< val_years[k]` is training. Test rows
    (cv_val_year == -1) are excluded from both sides.
    """
    if val_years is None:
        val_years = list(bundle.meta.get("cv_val_years", []))
    for vy in val_years:
        vy = int(vy)
        is_val = bundle.cv_val_year == vy
        # expanding-window: train = everything observed strictly before vy
        #   cv_val_year == 0        -> pre-2020 rows (always train)
        #   0 < cv_val_year < vy    -> earlier validation years, now train
        is_train = (bundle.cv_val_year >= 0) & (bundle.cv_val_year < vy) & (~is_val)

        yield _slice_fold(bundle, vy, is_train, is_val)


def _slice_fold(
    bundle: PanelBundle,
    fold_val_year: int,
    is_train: np.ndarray,
    is_val: np.ndarray,
) -> FoldData:
    return FoldData(
        fold_val_year=fold_val_year,
        y_log_train=bundle.y_log[is_train],
        month_train=bundle.month[is_train],
        station_idx_train=bundle.station_idx[is_train],
        county_idx_train=bundle.county_idx[is_train],
        X_smooth_train=bundle.X_smooth[is_train],
        X_linear_train=bundle.X_linear[is_train],
        miss_smooth_train=bundle.miss_smooth[is_train],
        miss_linear_train=bundle.miss_linear[is_train],
        left_mask_train=bundle.left_mask[is_train],
        right_mask_train=bundle.right_mask[is_train],
        det_low_log_train=bundle.det_low_log[is_train],
        det_high_log_train=bundle.det_high_log[is_train],
        y_log_val=bundle.y_log[is_val],
        month_val=bundle.month[is_val],
        station_idx_val=bundle.station_idx[is_val],
        county_idx_val=bundle.county_idx[is_val],
        X_smooth_val=bundle.X_smooth[is_val],
        X_linear_val=bundle.X_linear[is_val],
        miss_smooth_val=bundle.miss_smooth[is_val],
        miss_linear_val=bundle.miss_linear[is_val],
        smooth_features=list(bundle.smooth_features),
        linear_features=list(bundle.linear_features),
        n_stations=bundle.n_stations,
        n_counties=bundle.n_counties,
    )
