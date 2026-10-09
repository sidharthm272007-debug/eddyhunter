"""Real SWOT SSH-anomaly + OSCAR velocity ingestion via NASA Earthdata.

Both products are archived at PO.DAAC, so one Earthdata Login covers both.
Authentication reads EARTHDATA_USERNAME + EARTHDATA_PASSWORD, or
EARTHDATA_TOKEN, from the environment -- never hardcode credentials here.

Usage:
    export EARTHDATA_USERNAME=...   # or EARTHDATA_TOKEN=...
    export EARTHDATA_PASSWORD=...

    from eddyhunter.ingest.earthdata import fetch_swot_oscar
    ds = fetch_swot_oscar(
        bbox=(-62, 18, -48, 32),               # (west, south, east, north)
        temporal=("2024-01-01", "2024-01-31"),
    )

`ds` comes back in the same (time, lat, lon) / (ssha, u, v) shape the
synthetic generator produces, so it drops straight into
`eddyhunter.detection.pipeline.detect_eddies_all_steps` unchanged.

Dataset short names: PO.DAAC's catalog names change occasionally (e.g. a
SWOT product version bump). If a search comes back empty, check
https://podaac.jpl.nasa.gov and pass the current short_name explicitly --
the defaults below were current as of this project's writing but are not
guaranteed to stay so.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .clean import clean_pipeline, regrid_like

DEFAULT_SWOT_SHORT_NAME = "SWOT_L2_LR_SSH_Expert_2.0"
DEFAULT_OSCAR_SHORT_NAME = "OSCAR_L4_OC_third-deg"


def _login():
    import earthaccess

    token = os.environ.get("EARTHDATA_TOKEN")
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")

    if token:
        os.environ.setdefault("EARTHDATA_TOKEN", token)
        auth = earthaccess.login(strategy="environment")
    elif username and password:
        auth = earthaccess.login(strategy="environment")
    else:
        raise RuntimeError(
            "No Earthdata credentials found. Set EARTHDATA_USERNAME + "
            "EARTHDATA_PASSWORD, or EARTHDATA_TOKEN, in the environment. "
            "Get a token at https://urs.earthdata.nasa.gov/profile/tokens"
        )
    if not auth.authenticated:
        raise RuntimeError("Earthdata login failed -- check credentials / token.")
    return auth


@dataclass
class FetchConfig:
    swot_short_name: str = DEFAULT_SWOT_SHORT_NAME
    oscar_short_name: str = DEFAULT_OSCAR_SHORT_NAME
    download_dir: str = "/tmp/eddyhunter_downloads"
    max_granules_per_product: int = 60


def _search_and_open(short_name: str, bbox, temporal, cfg: FetchConfig) -> xr.Dataset:
    import earthaccess

    results = earthaccess.search_data(
        short_name=short_name, bounding_box=bbox, temporal=temporal,
        count=cfg.max_granules_per_product,
    )
    if not results:
        raise RuntimeError(
            f"No granules found for short_name={short_name!r}, bbox={bbox}, "
            f"temporal={temporal}. Check the product name at podaac.jpl.nasa.gov "
            f"and your date range/region."
        )

    os.makedirs(cfg.download_dir, exist_ok=True)
    files = earthaccess.download(results, local_path=cfg.download_dir)
    ds = xr.open_mfdataset(files, combine="by_coords", decode_times=True)
    return ds


def fetch_swot_oscar(
    bbox: Tuple[float, float, float, float],
    temporal: Tuple[str, str],
    config: FetchConfig = FetchConfig(),
) -> xr.Dataset:
    """Download, clean, and co-register SWOT SSH anomaly + OSCAR u/v for a
    region and time window. Returns a (time, lat, lon) Dataset with ssha,
    u, v -- the same shape `generate_synthetic_dataset` produces.
    """
    _login()

    swot_raw = _search_and_open(config.swot_short_name, bbox, temporal, config)
    oscar_raw = _search_and_open(config.oscar_short_name, bbox, temporal, config)

    # Variable names vary by product version; try the common SWOT/OSCAR
    # conventions and fail loudly (rather than silently picking the wrong
    # field) if none match.
    ssha_var = next((v for v in ("ssha", "ssha_karin", "sea_surface_height_anomaly") if v in swot_raw), None)
    u_var = next((v for v in ("u", "ug", "u_geos") if v in oscar_raw), None)
    v_var = next((v for v in ("v", "vg", "v_geos") if v in oscar_raw), None)
    if ssha_var is None or u_var is None or v_var is None:
        raise RuntimeError(
            f"Could not find expected variable names. SWOT vars: {list(swot_raw.data_vars)}; "
            f"OSCAR vars: {list(oscar_raw.data_vars)}. Update the candidate lists in "
            f"eddyhunter.ingest.earthdata.fetch_swot_oscar to match."
        )

    quality_var = next((v for v in ("quality_flag", "ssha_karin_qual") if v in swot_raw), None)
    swot_clean = clean_pipeline(swot_raw, quality_var=quality_var, good_values=(0,))
    oscar_clean = clean_pipeline(oscar_raw)

    oscar_on_swot = regrid_like(oscar_clean, swot_clean)

    ds = xr.Dataset(
        {
            "ssha": swot_clean[ssha_var],
            "u": oscar_on_swot[u_var],
            "v": oscar_on_swot[v_var],
        }
    )
    # Normalize dim names to lat/lon/time if the source used something else.
    rename = {}
    for cand, target in (("latitude", "lat"), ("longitude", "lon")):
        if cand in ds.dims and target not in ds.dims:
            rename[cand] = target
    if rename:
        ds = ds.rename(rename)

    return ds
