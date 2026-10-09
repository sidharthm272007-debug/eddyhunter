"""End-to-end single-time-step and full-dataset eddy detection pipeline:

    raw grid -> derived fields -> candidate cores -> ring point clouds
    -> alpha complex / persistent H1 -> noise filtering -> eddy polygons
    -> physical validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import xarray as xr

from .fields import derive_fields
from .point_cloud import find_candidate_cores, build_ring_point_clouds
from .topology import analyze_rings, filter_by_persistence
from .polygons import build_eddies, Eddy
from .validation import validate_eddies, ValidationConfig


@dataclass
class DetectionConfig:
    min_abs_ssha: float = 0.02
    core_neighborhood_cells: int = 5
    ring_box_radius_cells: int = 15
    ring_band_low_frac: float = 0.15
    ring_band_high_frac: float = 0.75
    ring_min_points: int = 10
    min_persistence_m: float = 5_000.0   # meters; noise-floor on H1 persistence
    min_area_km2: float = 20.0
    validation: ValidationConfig = None
    # Which field candidate cores/rings are built from: "ssha" (default --
    # matches the synthetic generator and a fully-gridded SSH product) or
    # "vorticity" (required for a real single-pass SWOT swath, which is too
    # geometrically narrow to ever close an SSH ring -- see point_cloud.py's
    # module docstring and find_candidate_cores' docstring for why).
    source_field: str = "ssha"
    # noise-floor amplitude threshold on the *source field itself* --
    # min_abs_ssha is reused for SSH; vorticity lives on a totally different
    # scale (~1e-5 s^-1), so it gets its own floor.
    min_abs_vorticity: float = 2.0e-6

    def __post_init__(self):
        if self.validation is None:
            self.validation = ValidationConfig()


def detect_eddies_single_step(
    u: np.ndarray, v: np.ndarray, ssha: np.ndarray,
    lats: np.ndarray, lons: np.ndarray,
    time_index: int,
    config: DetectionConfig = DetectionConfig(),
) -> List[Eddy]:
    fields = derive_fields(u, v, ssha, lats, lons)

    if config.source_field == "vorticity":
        source_array = fields.vorticity
        min_abs = config.min_abs_vorticity
    else:
        source_array = fields.ssha
        min_abs = config.min_abs_ssha

    cores = find_candidate_cores(
        fields, min_abs_ssha=min_abs,
        neighborhood_cells=config.core_neighborhood_cells,
        source_field=source_array,
    )
    if not cores:
        return []

    rings = build_ring_point_clouds(
        fields, cores,
        box_radius_cells=config.ring_box_radius_cells,
        band_low_frac=config.ring_band_low_frac,
        band_high_frac=config.ring_band_high_frac,
        min_points=config.ring_min_points,
        source_field=source_array,
    )
    if not rings:
        return []

    topo_results = analyze_rings(rings)
    surviving = filter_by_persistence(topo_results, config.min_persistence_m)
    if not surviving:
        return []

    eddies = build_eddies(
        surviving, fields, time_index,
        min_area_km2=config.min_area_km2,
        source_field=config.source_field,
    )

    cores_by_id = {c.core_id: c for c in cores}
    eddies = validate_eddies(eddies, cores_by_id, config.validation, source_field=config.source_field)
    return eddies


def detect_eddies_all_steps(ds: xr.Dataset, config: DetectionConfig = DetectionConfig()) -> List[Eddy]:
    all_eddies: List[Eddy] = []
    lats = ds["lat"].values
    lons = ds["lon"].values
    for t_idx in range(ds.sizes["time"]):
        u = ds["u"].isel(time=t_idx).values
        v = ds["v"].isel(time=t_idx).values
        ssha = ds["ssha"].isel(time=t_idx).values
        eddies = detect_eddies_single_step(u, v, ssha, lats, lons, t_idx, config)
        all_eddies.extend(eddies)
    return all_eddies


def eddies_to_dataframe(eddies: List[Eddy]):
    import pandas as pd
    return pd.DataFrame([e.to_dict() for e in eddies])
