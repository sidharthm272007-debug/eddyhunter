"""Load real, already-downloaded SWOT L2 Expert + OSCAR L4 granules from
local disk and combine them into the same (time, lat, lon) / (ssha, u, v)
shape the synthetic generator produces.

This is the "I already have the files" counterpart to
eddyhunter.ingest.earthdata (which downloads them). Real SWOT L2 data is a
along-track SWATH product (2D lat/lon per pixel, two ~50km-wide strips
either side of the ground track, ~531 km gap in the middle) -- not a
regular lat/lon grid -- so unlike the synthetic generator, this module has
to interpolate the scattered, quality-filtered swath points onto a regular
grid before anything downstream (which assumes a regular grid, for
np.gradient-based vorticity etc.) can use it.

SWOT quality: ssh_karin_qual is a bitmask; 0 means no flags raised at all.
We keep only qual==0 pixels -- strict, but real SWOT passes are wide enough
that plenty of good pixels remain even over a regional subset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata


@dataclass
class BBox:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


def _load_swot_granule(path: str, bbox: BBox) -> dict:
    """Return quality-filtered, bbox-clipped (lon, lat, ssha) point arrays
    plus the granule's representative timestamp.
    """
    ds = xr.open_dataset(path)
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    ssha = ds["ssha_karin"].values
    qual = ds["ssh_karin_qual"].values

    good = np.isfinite(ssha) & (qual == 0)
    in_box = (lon >= bbox.lon_min) & (lon <= bbox.lon_max) & (lat >= bbox.lat_min) & (lat <= bbox.lat_max)
    mask = good & in_box

    # granule timestamp: median of the per-line time coordinate
    t = pd.to_datetime(ds["time"].values)
    t = t[np.isfinite(t.values.astype("float64"))] if hasattr(t, "values") else t
    timestamp = pd.Timestamp(np.array(t, dtype="datetime64[ns]")[len(t) // 2]) if len(t) else None

    out = {
        "lon": lon[mask], "lat": lat[mask], "ssha": ssha[mask],
        "n_points": int(mask.sum()), "timestamp": timestamp,
        "granule": Path(path).name,
    }
    ds.close()
    return out


def _date_from_oscar_filename(path: str) -> Optional[pd.Timestamp]:
    m = re.search(r"(\d{8})", Path(path).name)
    return pd.Timestamp(m.group(1)) if m else None


def _load_oscar_grid(path: str, bbox: BBox):
    ds = xr.open_dataset(path)
    lon = ds["lon"].values
    lat = ds["lat"].values
    # dims are (time, longitude, latitude) -- squeeze time, then transpose to (lat, lon)
    u = ds["u"].isel(time=0).values.T
    v = ds["v"].isel(time=0).values.T
    ds.close()

    lon_mask = (lon >= bbox.lon_min) & (lon <= bbox.lon_max)
    lat_mask = (lat >= bbox.lat_min) & (lat <= bbox.lat_max)
    return lon[lon_mask], lat[lat_mask], u[np.ix_(lat_mask, lon_mask)], v[np.ix_(lat_mask, lon_mask)]


def build_real_dataset(
    swot_paths: List[str],
    oscar_dir: str,
    bbox: BBox,
    grid_step_deg: float = 0.25,
    max_date_gap_days: int = 1,
) -> xr.Dataset:
    """Match each SWOT granule to the same-day OSCAR file, regrid the SWOT
    swath onto a regular grid at `grid_step_deg` resolution over `bbox`,
    and stack into one (time, lat, lon) Dataset with ssha, u, v.
    """
    oscar_files = sorted(Path(oscar_dir).glob("*.nc"))
    oscar_by_date = {}
    for f in oscar_files:
        d = _date_from_oscar_filename(str(f))
        if d is not None:
            oscar_by_date[d.normalize()] = str(f)

    target_lons = np.arange(bbox.lon_min, bbox.lon_max + 1e-9, grid_step_deg)
    target_lats = np.arange(bbox.lat_min, bbox.lat_max + 1e-9, grid_step_deg)
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)

    ssha_frames, u_frames, v_frames, times, diagnostics = [], [], [], [], []

    for path in sorted(swot_paths):
        swot = _load_swot_granule(path, bbox)
        if swot["n_points"] < 50 or swot["timestamp"] is None:
            diagnostics.append(f"{swot['granule']}: skipped ({swot['n_points']} good points in bbox)")
            continue

        day = swot["timestamp"].normalize()
        oscar_path = oscar_by_date.get(day)
        if oscar_path is None:
            # fall back to nearest available OSCAR date within the gap tolerance
            candidates = [(abs((d - day).days), d) for d in oscar_by_date]
            candidates.sort()
            if candidates and candidates[0][0] <= max_date_gap_days:
                oscar_path = oscar_by_date[candidates[0][1]]
            else:
                diagnostics.append(f"{swot['granule']}: no OSCAR file within {max_date_gap_days}d of {day.date()}")
                continue

        ssha_on_grid = griddata(
            (swot["lon"], swot["lat"]), swot["ssha"], (lon_grid, lat_grid), method="linear",
        )

        oscar_lon, oscar_lat, u_oscar, v_oscar = _load_oscar_grid(oscar_path, bbox)
        u_on_grid = griddata(
            (np.meshgrid(oscar_lon, oscar_lat)[0].ravel(), np.meshgrid(oscar_lon, oscar_lat)[1].ravel()),
            u_oscar.ravel(), (lon_grid, lat_grid), method="linear",
        )
        v_on_grid = griddata(
            (np.meshgrid(oscar_lon, oscar_lat)[0].ravel(), np.meshgrid(oscar_lon, oscar_lat)[1].ravel()),
            v_oscar.ravel(), (lon_grid, lat_grid), method="linear",
        )

        ssha_frames.append(ssha_on_grid)
        u_frames.append(u_on_grid)
        v_frames.append(v_on_grid)
        times.append(swot["timestamp"])
        diagnostics.append(
            f"{swot['granule']}: {swot['n_points']} good SWOT points -> "
            f"{np.isfinite(ssha_on_grid).sum()} grid cells; paired with {Path(oscar_path).name}"
        )

    if not times:
        raise RuntimeError("No SWOT granule could be matched to an OSCAR file / had enough good data:\n"
                            + "\n".join(diagnostics))

    order = np.argsort(times)
    ds = xr.Dataset(
        {
            "ssha": (("time", "lat", "lon"), np.stack([ssha_frames[i] for i in order])),
            "u": (("time", "lat", "lon"), np.stack([u_frames[i] for i in order])),
            "v": (("time", "lat", "lon"), np.stack([v_frames[i] for i in order])),
        },
        coords={
            "time": [times[i] for i in order],
            "lat": target_lats, "lon": target_lons,
        },
        attrs={"source": "real SWOT L2 LR SSH Expert + OSCAR L4 (local files)", "diagnostics": "\n".join(diagnostics)},
    )
    return ds
