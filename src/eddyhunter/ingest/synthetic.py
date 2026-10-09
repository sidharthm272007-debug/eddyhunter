"""Synthetic rotating-vortex data generator.

Produces SSH-anomaly and surface-velocity fields with known, ground-truth
mesoscale eddies so the full detection -> tracking -> forecasting -> PPO
pipeline can be validated end-to-end before any real SWOT/OSCAR data is
involved. Mirrors the structure the real ingestion module will hand off:
an xarray.Dataset with dims (time, lat, lon) and variables (ssha, u, v),
plus a ground-truth eddy-track table for scoring detector/tracker/forecaster
accuracy later.

Vortex model
------------
Each synthetic eddy is a Gaussian ("Lilly") vortex: tangential velocity

    v_theta(r) = v_max * (r / R) * exp(0.5 * (1 - (r / R)**2))

which is zero at the center, peaks at r = R, and decays smoothly outward.
sign = +1 is cyclonic (CCW in the Northern-Hemisphere convention used here),
sign = -1 is anticyclonic. Sea-surface-height anomaly is the geostrophic
partner of that velocity field: a Gaussian bump/dip in SSH whose sign is
opposite the rotation sign for cyclones in the NH (cyclones = low SSH,
anticyclones = high SSH), amplitude scaled by vortex strength.

Eddies drift at a constant (configurable, per-eddy) velocity across time
steps so the resulting dataset exercises tracking and forecasting, not
just single-frame detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr


EARTH_RADIUS_KM = 6371.0


def km_per_degree_lat() -> float:
    return np.pi * EARTH_RADIUS_KM / 180.0


def km_per_degree_lon(lat_deg: float) -> float:
    return np.pi * EARTH_RADIUS_KM / 180.0 * np.cos(np.deg2rad(lat_deg))


@dataclass
class SyntheticEddySpec:
    """Ground-truth generator spec for one eddy, at t=0."""

    name: str
    lat0: float
    lon0: float
    radius_km: float          # R in the Gaussian vortex profile
    v_max: float              # peak tangential speed, m/s
    sign: int                 # +1 cyclonic, -1 anticyclonic
    drift_u: float = 0.0       # eastward drift, m/s
    drift_v: float = 0.0       # northward drift, m/s
    ssh_coeff: float = 0.12    # m of SSH anomaly per (m/s * radius-scale)
    lifetime_steps: Optional[int] = None  # None = lives whole run
    birth_step: int = 0


@dataclass
class GroundTruthRecord:
    time_index: int
    time: pd.Timestamp
    eddy_name: str
    lat: float
    lon: float
    radius_km: float
    v_max: float
    sign: int


def _vortex_uv_ssha(lat_grid, lon_grid, lat0, lon0, radius_km, v_max, sign, ssh_coeff):
    """Return (u, v, ssha) contributions from one Gaussian vortex."""
    dlat = lat_grid - lat0
    dlon = lon_grid - lon0
    y_km = dlat * km_per_degree_lat()
    x_km = dlon * km_per_degree_lon(lat0)
    r = np.sqrt(x_km ** 2 + y_km ** 2) + 1e-9
    theta = np.arctan2(y_km, x_km)

    r_norm = r / radius_km
    v_theta = sign * v_max * r_norm * np.exp(0.5 * (1.0 - r_norm ** 2))

    u = -v_theta * np.sin(theta)
    v = v_theta * np.cos(theta)

    # Geostrophic SSH partner: Gaussian bump, sign opposite the rotation
    # sign (NH convention: anticyclone = high SSH = sign -1 -> +ssha).
    ssha = -sign * ssh_coeff * (v_max / 0.5) * np.exp(-0.5 * r_norm ** 2)

    return u, v, ssha


def generate_synthetic_dataset(
    eddies: List[SyntheticEddySpec],
    lat_range=(20.0, 30.0),
    lon_range=(-60.0, -50.0),
    grid_step_deg: float = 0.1,
    n_steps: int = 20,
    dt_hours: float = 24.0,
    start_time: str = "2024-01-01",
    noise_std_ssha: float = 0.01,
    noise_std_vel: float = 0.02,
    missing_frac: float = 0.03,
    seed: int = 0,
) -> "tuple[xr.Dataset, pd.DataFrame]":
    """Generate a synthetic (time, lat, lon) dataset plus ground-truth tracks.

    Returns
    -------
    ds : xarray.Dataset
        Variables: ssha (m), u (m/s), v (m/s). Dims: time, lat, lon.
        Some fraction of cells are set to NaN to emulate QC-flagged /
        missing swath data, the same shape of problem the real SWOT/OSCAR
        loader has to clean up.
    truth : pandas.DataFrame
        One row per (time step, live eddy) with its true centroid, radius,
        strength and rotation sign -- the answer key for evaluating the
        detector, tracker and forecaster later.
    """
    rng = np.random.default_rng(seed)

    lats = np.arange(lat_range[0], lat_range[1] + 1e-9, grid_step_deg)
    lons = np.arange(lon_range[0], lon_range[1] + 1e-9, grid_step_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)  # shape (nlat, nlon)

    times = pd.date_range(start=start_time, periods=n_steps, freq=pd.Timedelta(hours=dt_hours))

    ssha = np.zeros((n_steps, lats.size, lons.size), dtype=np.float64)
    u = np.zeros_like(ssha)
    v = np.zeros_like(ssha)

    truth_rows: List[GroundTruthRecord] = []

    # Track live centroid per eddy across steps (starts at spec lat0/lon0).
    live_lat = {e.name: e.lat0 for e in eddies}
    live_lon = {e.name: e.lon0 for e in eddies}

    for t_idx in range(n_steps):
        for e in eddies:
            end_step = e.birth_step + (e.lifetime_steps if e.lifetime_steps is not None else n_steps)
            if t_idx < e.birth_step or t_idx >= end_step:
                continue

            if t_idx > e.birth_step:
                dt_s = dt_hours * 3600.0
                dlat = (e.drift_v * dt_s) / 1000.0 / km_per_degree_lat()
                dlon = (e.drift_u * dt_s) / 1000.0 / km_per_degree_lon(live_lat[e.name])
                live_lat[e.name] += dlat
                live_lon[e.name] += dlon

            u_c, v_c, ssha_c = _vortex_uv_ssha(
                lat_grid, lon_grid,
                live_lat[e.name], live_lon[e.name],
                e.radius_km, e.v_max, e.sign, e.ssh_coeff,
            )
            u[t_idx] += u_c
            v[t_idx] += v_c
            ssha[t_idx] += ssha_c

            truth_rows.append(GroundTruthRecord(
                time_index=t_idx,
                time=times[t_idx],
                eddy_name=e.name,
                lat=live_lat[e.name],
                lon=live_lon[e.name],
                radius_km=e.radius_km,
                v_max=e.v_max,
                sign=e.sign,
            ))

    # Sensor-like noise.
    ssha += rng.normal(0, noise_std_ssha, size=ssha.shape)
    u += rng.normal(0, noise_std_vel, size=u.shape)
    v += rng.normal(0, noise_std_vel, size=v.shape)

    # Simulate missing/QC-flagged data (swath gaps, bad quality cells).
    if missing_frac > 0:
        mask = rng.random(ssha.shape) < missing_frac
        ssha[mask] = np.nan
        u[mask] = np.nan
        v[mask] = np.nan

    ds = xr.Dataset(
        {
            "ssha": (("time", "lat", "lon"), ssha),
            "u": (("time", "lat", "lon"), u),
            "v": (("time", "lat", "lon"), v),
        },
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={
            "title": "Synthetic SWOT/OSCAR-like rotating vortex dataset",
            "source": "eddyhunter.ingest.synthetic",
        },
    )

    truth = pd.DataFrame([r.__dict__ for r in truth_rows])
    return ds, truth


def default_demo_eddies() -> List[SyntheticEddySpec]:
    """A hand-picked scene: one cyclone, one anticyclone, drifting westward
    (mimicking real mesoscale eddy propagation), plus a short-lived weak
    eddy that should get filtered out as noise by the persistence threshold.
    """
    return [
        SyntheticEddySpec(
            name="cyclone_A", lat0=25.0, lon0=-54.0, radius_km=60.0,
            v_max=0.55, sign=+1, drift_u=-0.03, drift_v=0.01,
        ),
        SyntheticEddySpec(
            name="anticyclone_B", lat0=23.5, lon0=-57.0, radius_km=75.0,
            v_max=0.65, sign=-1, drift_u=-0.025, drift_v=-0.005,
        ),
        SyntheticEddySpec(
            name="weak_transient_C", lat0=27.0, lon0=-52.0, radius_km=25.0,
            v_max=0.12, sign=+1, drift_u=-0.01, drift_v=0.0,
            lifetime_steps=4, birth_step=3,
        ),
    ]


if __name__ == "__main__":
    ds, truth = generate_synthetic_dataset(default_demo_eddies())
    print(ds)
    print(truth.head())
