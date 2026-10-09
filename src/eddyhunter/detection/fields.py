"""Derived-field computations on a (lat, lon) velocity/SSH grid.

Everything here operates on a single time slice: 2D arrays of shape
(nlat, nlon) plus the 1D lat/lon coordinate vectors. Distances are computed
on a local equirectangular (tangent-plane) projection, which is accurate
enough at mesoscale-eddy scales (tens to ~200 km).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def latlon_to_local_xy_m(lat, lon, lat0, lon0):
    """Equirectangular projection to meters, centered at (lat0, lon0)."""
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    y = dlat * EARTH_RADIUS_M
    x = dlon * EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))
    return x, y


def grid_spacing_m(lats: np.ndarray, lons: np.ndarray):
    """Per-row (varies with latitude) grid spacing in meters."""
    dlat_deg = np.mean(np.diff(lats))
    dlon_deg = np.mean(np.diff(lons))
    dy = np.deg2rad(dlat_deg) * EARTH_RADIUS_M
    dx = np.deg2rad(dlon_deg) * EARTH_RADIUS_M * np.cos(np.deg2rad(lats))  # per-row
    return dx, dy  # dx: array (nlat,), dy: scalar


def compute_vorticity(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Relative vorticity zeta = dv/dx - du/dy, s^-1. NaN-safe-ish (NaNs propagate)."""
    dx_per_row, dy = grid_spacing_m(lats, lons)
    # np.gradient along axis=1 (lon/x) and axis=0 (lat/y)
    dv_dx = np.gradient(v, axis=1) / dx_per_row[:, None]
    du_dy = np.gradient(u, axis=0) / dy
    return dv_dx - du_dy


def compute_strain(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray):
    """Normal and shear strain components, for the Okubo-Weiss parameter."""
    dx_per_row, dy = grid_spacing_m(lats, lons)
    du_dx = np.gradient(u, axis=1) / dx_per_row[:, None]
    dv_dy = np.gradient(v, axis=0) / dy
    du_dy = np.gradient(u, axis=0) / dy
    dv_dx = np.gradient(v, axis=1) / dx_per_row[:, None]

    s_n = du_dx - dv_dy       # normal strain
    s_s = dv_dx + du_dy       # shear strain
    zeta = dv_dx - du_dy      # relative vorticity
    return s_n, s_s, zeta


def okubo_weiss(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """W = s_n^2 + s_s^2 - zeta^2. W << 0 => rotation (vortex) dominated."""
    s_n, s_s, zeta = compute_strain(u, v, lats, lons)
    return s_n ** 2 + s_s ** 2 - zeta ** 2


@dataclass
class DerivedFields:
    u: np.ndarray
    v: np.ndarray
    ssha: np.ndarray
    speed: np.ndarray
    vorticity: np.ndarray
    okubo_weiss: np.ndarray
    lats: np.ndarray
    lons: np.ndarray


def derive_fields(u, v, ssha, lats, lons) -> DerivedFields:
    speed = np.sqrt(u ** 2 + v ** 2)
    zeta = compute_vorticity(u, v, lats, lons)
    ow = okubo_weiss(u, v, lats, lons)
    return DerivedFields(u=u, v=v, ssha=ssha, speed=speed, vorticity=zeta,
                          okubo_weiss=ow, lats=lats, lons=lons)
