"""Cleaning / QC / regridding utilities shared by synthetic and real data.

Real SWOT and OSCAR products carry their own quality-flag conventions; this
module keeps a thin, format-agnostic layer: drop/interpolate missing data,
optionally apply a caller-supplied quality mask, and regrid one dataset onto
another's lat/lon/time grid so SWOT SSH and OSCAR velocity can be combined.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def apply_quality_mask(ds: xr.Dataset, quality_var: str, good_values) -> xr.Dataset:
    """Mask out cells whose quality flag is not in `good_values`."""
    if quality_var not in ds:
        return ds
    good = ds[quality_var].isin(list(good_values))
    return ds.where(good)


def interpolate_gaps(ds: xr.Dataset, max_gap_cells: int = 3) -> xr.Dataset:
    """Fill small NaN gaps by linear interpolation along lat and lon in turn;
    leaves larger gaps (e.g. whole missing swaths) as NaN rather than
    inventing data over them.
    """
    out = ds
    for dim in ("lon", "lat"):
        if dim in out.dims:
            out = out.interpolate_na(dim=dim, method="linear", limit=max_gap_cells)
    return out


def regrid_like(ds: xr.Dataset, target: xr.Dataset, method: str = "linear") -> xr.Dataset:
    """Interpolate `ds` onto `target`'s lat/lon (and time, if present) grid."""
    coords = {}
    if "lat" in target.coords:
        coords["lat"] = target["lat"]
    if "lon" in target.coords:
        coords["lon"] = target["lon"]
    if "time" in target.coords and "time" in ds.coords:
        coords["time"] = target["time"]
    return ds.interp(**coords, method=method)


def clean_pipeline(ds: xr.Dataset, quality_var: str = None, good_values=(0,),
                    max_gap_cells: int = 3) -> xr.Dataset:
    """Standard clean-up: QC mask -> small-gap interpolation."""
    if quality_var is not None:
        ds = apply_quality_mask(ds, quality_var, good_values)
    ds = interpolate_gaps(ds, max_gap_cells=max_gap_cells)
    return ds
