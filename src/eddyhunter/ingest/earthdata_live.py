"""Live NASA Earthdata fetch: download the most recent OSCAR + SWOT granules
directly from PO.DAAC (no manual download step required), then hand the
downloaded files to the same proven ingestion path
(`real_oscar_series.build_oscar_series`) the local-folder live-refresh
already uses -- real_local.py's SWOT/OSCAR loaders don't care whether a file
arrived via a manual download or an automated one.

This is what makes a web-hosted deployment "live" without needing the
user's own laptop folders: the server pulls new granules itself. There is
still no such thing as true streaming for this data -- OSCAR updates once a
day, SWOT revisits a given spot every 1-21 days -- so "live" here means
"checks NASA for anything new since the last sync," not a continuous feed;
see live.py's module docstring for the fuller version of this point.

Authentication reads EARTHDATA_USERNAME + EARTHDATA_PASSWORD (or
EARTHDATA_TOKEN) from the environment -- NEVER hardcode credentials here,
never log their values, and never type them into this file. On Streamlit
Community Cloud, set them in the app's own "Secrets" settings in the
Streamlit Cloud web UI (which only you can access) and app.py copies them
into the environment at startup; anywhere else, export them as real
environment variables before running the app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from .earthdata import DEFAULT_OSCAR_SHORT_NAME, DEFAULT_SWOT_SHORT_NAME
from .real_local import BBox


@dataclass
class LiveFetchConfig:
    swot_short_name: str = DEFAULT_SWOT_SHORT_NAME
    oscar_short_name: str = DEFAULT_OSCAR_SHORT_NAME
    days_back: int = 5
    max_granules_per_product: int = 40


def _login():
    import earthaccess

    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    token = os.environ.get("EARTHDATA_TOKEN")
    if not (token or (username and password)):
        raise RuntimeError(
            "No Earthdata credentials found in the environment. Set EARTHDATA_USERNAME + "
            "EARTHDATA_PASSWORD (or EARTHDATA_TOKEN) -- on Streamlit Community Cloud this goes in "
            "the app's own Secrets settings, never in code or chat."
        )
    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        raise RuntimeError("Earthdata login failed -- check the stored credentials are still valid.")
    return auth


def fetch_recent_granules(
    bbox: BBox,
    download_dir: str,
    config: LiveFetchConfig = LiveFetchConfig(),
) -> Tuple[int, int]:
    """Download whatever OSCAR + SWOT granules PO.DAAC has published for the
    last `config.days_back` days over `bbox`, into
    `download_dir/oscar` and `download_dir/swot`. Returns
    (n_oscar_files, n_swot_files); callers then point
    `real_oscar_series.build_oscar_series` at those two folders exactly as
    the manual local-folder flow already does -- this function only handles
    getting the files onto disk.
    """
    import earthaccess

    _login()

    end = pd.Timestamp.now("UTC").normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=config.days_back)
    temporal = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    # earthaccess bounding_box order is (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)
    ea_bbox = (bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max)

    oscar_dir = Path(download_dir) / "oscar"
    swot_dir = Path(download_dir) / "swot"
    oscar_dir.mkdir(parents=True, exist_ok=True)
    swot_dir.mkdir(parents=True, exist_ok=True)

    oscar_results = earthaccess.search_data(
        short_name=config.oscar_short_name, bounding_box=ea_bbox, temporal=temporal,
        count=config.max_granules_per_product,
    )
    oscar_files = earthaccess.download(oscar_results, local_path=str(oscar_dir)) if oscar_results else []

    swot_results = earthaccess.search_data(
        short_name=config.swot_short_name, bounding_box=ea_bbox, temporal=temporal,
        count=config.max_granules_per_product,
    )
    swot_files = earthaccess.download(swot_results, local_path=str(swot_dir)) if swot_results else []

    return len(oscar_files), len(swot_files)
