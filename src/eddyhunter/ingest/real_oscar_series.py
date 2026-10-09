"""Build a long, OSCAR-only (plus whatever SWOT happens to overlap) real
time series -- the "best model possible" data source once more than a
handful of SWOT-paired days exist.

Unlike `real_local.build_real_dataset` (keyed on SWOT granules, since that
was all there was), this iterates every OSCAR day found on disk -- OSCAR is
a full daily gridded product, so it alone gives long, dense, real coverage
for vorticity-primary detection. SWOT SSH is attached only on days where a
real SWOT granule happens to fall within `max_date_gap_days`; every other
day just has ssha=NaN everywhere, which the existing pipeline already
handles correctly (it's exactly the "sparse SWOT cross-check" case the
validation step was built for).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import LinearNDInterpolator

from .real_local import BBox, _load_swot_granule, _date_from_oscar_filename


def build_oscar_series(
    oscar_dir: str,
    bbox: BBox,
    swot_paths: Optional[List[str]] = None,
    grid_step_deg: float = 0.05,
    max_date_gap_days: int = 1,
) -> xr.Dataset:
    oscar_files = sorted(Path(oscar_dir).glob("*.nc"), key=lambda p: _date_from_oscar_filename(str(p)) or pd.Timestamp(0))
    if not oscar_files:
        raise RuntimeError(f"No OSCAR .nc files found in {oscar_dir}")

    target_lons = np.arange(bbox.lon_min, bbox.lon_max + 1e-9, grid_step_deg)
    target_lats = np.arange(bbox.lat_min, bbox.lat_max + 1e-9, grid_step_deg)
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)

    # Real SWOT granules, keyed by day, attached only where they exist.
    swot_by_day = {}
    if swot_paths:
        for p in swot_paths:
            swot = _load_swot_granule(p, bbox)
            if swot["n_points"] >= 50 and swot["timestamp"] is not None:
                swot_by_day.setdefault(swot["timestamp"].normalize(), []).append(swot)

    u_frames, v_frames, ssha_frames, times, diagnostics = [], [], [], [], []
    src_lon_grid = src_lat_grid = None  # same OSCAR source grid every file -- cache the bbox subset coords

    for f in oscar_files:
        day = _date_from_oscar_filename(str(f))
        if day is None:
            continue
        ds = xr.open_dataset(f)
        lon = ds["lon"].values
        lat = ds["lat"].values
        if src_lon_grid is None:
            lon_mask = (lon >= bbox.lon_min) & (lon <= bbox.lon_max)
            lat_mask = (lat >= bbox.lat_min) & (lat <= bbox.lat_max)
            lon_sub, lat_sub = lon[lon_mask], lat[lat_mask]
            src_lon_grid, src_lat_grid = np.meshgrid(lon_sub, lat_sub)
        u = ds["u"].isel(time=0).values.T[np.ix_(lat_mask, lon_mask)]
        v = ds["v"].isel(time=0).values.T[np.ix_(lat_mask, lon_mask)]
        ds.close()

        pts = list(zip(src_lon_grid.ravel(), src_lat_grid.ravel()))
        u_on_grid = LinearNDInterpolator(pts, u.ravel())(lon_grid, lat_grid)
        v_on_grid = LinearNDInterpolator(pts, v.ravel())(lon_grid, lat_grid)

        # Attach SWOT SSH only if a real granule falls within the gap tolerance of this day.
        ssha_on_grid = np.full_like(u_on_grid, np.nan)
        best_gap, best_swot = None, None
        for d2, swots in swot_by_day.items():
            gap = abs((d2 - day).days)
            if gap <= max_date_gap_days and (best_gap is None or gap < best_gap):
                best_gap, best_swot = gap, swots
        if best_swot is not None:
            from scipy.interpolate import griddata
            all_lon = np.concatenate([s["lon"] for s in best_swot])
            all_lat = np.concatenate([s["lat"] for s in best_swot])
            all_ssha = np.concatenate([s["ssha"] for s in best_swot])
            ssha_on_grid = griddata((all_lon, all_lat), all_ssha, (lon_grid, lat_grid), method="linear")
            diagnostics.append(f"{day.date()}: OSCAR + real SWOT ({sum(s['n_points'] for s in best_swot)} pts, gap={best_gap}d)")
        else:
            diagnostics.append(f"{day.date()}: OSCAR only, no SWOT within {max_date_gap_days}d")

        u_frames.append(u_on_grid)
        v_frames.append(v_on_grid)
        ssha_frames.append(ssha_on_grid)
        times.append(pd.Timestamp(day))

    ds_out = xr.Dataset(
        {
            "u": (("time", "lat", "lon"), np.stack(u_frames)),
            "v": (("time", "lat", "lon"), np.stack(v_frames)),
            "ssha": (("time", "lat", "lon"), np.stack(ssha_frames)),
        },
        coords={"time": times, "lat": target_lats, "lon": target_lons},
        attrs={
            "source": "real OSCAR L4 daily currents (full series) + real SWOT L2 LR SSH Expert "
                      "wherever a granule happens to overlap -- not SWOT-pass-keyed like the "
                      "original 5-day ingest",
            "diagnostics": "\n".join(diagnostics),
        },
    )
    return ds_out
