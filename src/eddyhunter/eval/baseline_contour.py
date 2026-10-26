"""Independent second eddy detector: closed-contour SSH eddy identification --
the same family of method AVISO's own published eddy atlas (and the
widely-used py-eddy-tracker library it is built from) uses (Chelton et al.
2011 / Mason et al. 2014: grow iso-contours outward from a local SSH
extremum, keep the largest closed one that stays roughly circular). Used
only to cross-validate this project's own topological (persistent-H1)
detector against a structurally different method -- never imported by the
live dashboard, so it carries no deploy-time cost or risk.

We tried installing the actual py-eddy-tracker package first (it's what
AVISO's published atlas is built with) and running it directly on our real
data as the cross-check. It turned out to be pinned to numpy/numba/netCDF4
versions from ~2022 with no installable wheels for any Python version
available in development here, so getting it running was turning into a
multi-day dependency fight -- not viable against a deadline. This module
implements the same underlying algorithm it's built on instead, in this
project's own already-working stack (numpy/scipy/shapely + contourpy for
contour extraction), which is a legitimate independent second opinion
precisely because it is a structurally different construction (growing
iso-contours) from our own (persistent H1 on a ring point cloud), even
though both ultimately key off the same SSH field.

Needs `contourpy` (pip install contourpy). Deliberately NOT added to
requirements.txt: this module is a one-off/offline analysis tool, never
imported by app.py, so it should never affect what Streamlit Cloud installs
or how much memory the live app uses -- the same mistake that caused the
earlier OOM crash (see live.py's sync_from_nasa_and_refresh docstring) is
not worth risking again for an evaluation exhibit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from shapely.geometry import Polygon, Point

from ..detection.fields import DerivedFields, latlon_to_local_xy_m
from ..detection.point_cloud import find_candidate_cores


@dataclass
class ContourEddy:
    core_id: int
    time_index: int
    centroid_lat: float
    centroid_lon: float
    polygon_lonlat: Polygon
    area_km2: float
    rotation: str          # "cyclonic" | "anticyclonic"
    shape_error: float      # 0 = perfect circle, 100 = degenerate
    amplitude: float         # |SSH extremum - contour level|, m
    mean_speed: float        # mean current speed (m/s) on the boundary


def _shape_error(polygon: Polygon) -> float:
    area = polygon.area
    perim = polygon.length
    if perim <= 0:
        return 100.0
    return float(100.0 * (1.0 - (4 * np.pi * area) / (perim ** 2)))


def _area_km2(polygon: Polygon, ref_lat: float) -> float:
    lons, lats = polygon.exterior.coords.xy
    x_m, y_m = latlon_to_local_xy_m(np.array(lats), np.array(lons), ref_lat, float(np.mean(lons)))
    return float(Polygon(zip(x_m, y_m)).area / 1e6)


def _mean_speed_on_boundary(speed: np.ndarray, lats: np.ndarray, lons: np.ndarray, polygon: Polygon) -> float:
    bx, by = polygon.exterior.coords.xy
    vals = []
    for lon_pt, lat_pt in zip(bx, by):
        j = int(np.argmin(np.abs(lons - lon_pt)))
        i = int(np.argmin(np.abs(lats - lat_pt)))
        v = speed[i, j]
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def detect_eddies_contour(
    fields: DerivedFields,
    time_index: int,
    source_field_name: str = "ssha",
    n_levels: int = 20,
    max_shape_error: float = 70.0,
    min_amplitude: float = None,
    min_area_km2: float = 20.0,
) -> List[ContourEddy]:
    """One independent detection pass over a single time slice. Reuses this
    project's own local-extrema finder (so the comparison isolates "ring/
    contour construction method", not "where to look"), then for each
    candidate grows contours of `source_field_name` outward from the
    extremum and keeps the largest single-extremum, non-edge-touching,
    roughly-circular one -- breaking ties by maximum mean boundary speed,
    the same selection rule py-eddy-tracker itself uses.

    `source_field_name` MUST match whatever field the detector being
    cross-checked actually ran on -- "ssha" for the synthetic scene (full
    areal SSH coverage), "vorticity" for real data. This project's own real
    detector switched from ssha to vorticity years before this module
    existed, for exactly this reason: a single SWOT swath pass only covers
    a ~50km-wide strip (real_full_report.json's own
    n_with_swot_ssh_crosscheck is typically single digits out of hundreds
    of eddies), so an ssha-based baseline on real data would be searching
    an almost entirely-NaN field and finding a near-disjoint handful of
    candidates purely by swath-overlap luck -- a 0% agreement rate that
    would reflect the field choice, not whether either detector is finding
    real structure. Comparing two methods only means something when they
    run on the same input.
    """
    import contourpy

    source = fields.ssha if source_field_name == "ssha" else fields.vorticity
    # rotation sign convention flips with the source field, same as
    # detection/polygons.py's build_eddies(): a local SSH maximum is
    # anticyclonic (NH) but a local *vorticity* maximum is cyclonic.
    rotation_for_sign = (
        (lambda s: "anticyclonic" if s > 0 else "cyclonic") if source_field_name == "ssha"
        else (lambda s: "cyclonic" if s > 0 else "anticyclonic")
    )
    # Amplitude/noise-floor thresholds are field-scale-dependent: SSH
    # anomaly lives around 0.01-0.5m, relative vorticity around 1e-6-1e-4
    # s^-1 -- four to five orders of magnitude apart. Mirrors
    # detection/pipeline.py's DetectionConfig (min_abs_ssha=0.02,
    # min_abs_vorticity=2e-6), the project's own real-data detector.
    min_abs_core = 0.02 if source_field_name == "ssha" else 2.0e-6
    if min_amplitude is None:
        min_amplitude = 0.01 if source_field_name == "ssha" else 2.0e-6

    cores = find_candidate_cores(fields, min_abs_ssha=min_abs_core, source_field=source)
    lats, lons = fields.lats, fields.lons
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    field_filled = np.where(np.isfinite(source), source, 0.0)

    cg = contourpy.contour_generator(x=lon_grid, y=lat_grid, z=field_filled,
                                      line_type=contourpy.LineType.Separate)

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())
    edge_tol_lon = (lon_max - lon_min) * 0.01
    edge_tol_lat = (lat_max - lat_min) * 0.01

    out: List[ContourEddy] = []
    for core in cores:
        other_cores = [c for c in cores if c.core_id != core.core_id and c.sign == core.sign]
        levels = np.linspace(0.0, core.ssha_value, n_levels + 1)[1:]

        best = None
        for lev in levels:
            try:
                lines = cg.lines(float(lev))
            except Exception:
                continue
            for line in lines:
                if line.shape[0] < 4 or not np.allclose(line[0], line[-1]):
                    continue
                poly = Polygon(line)
                if not poly.is_valid or poly.area == 0:
                    continue
                if not poly.contains(Point(core.lon, core.lat)):
                    continue
                minx, miny, maxx, maxy = poly.bounds
                if (minx - lon_min < edge_tol_lon or lon_max - maxx < edge_tol_lon or
                        miny - lat_min < edge_tol_lat or lat_max - maxy < edge_tol_lat):
                    continue
                if any(poly.contains(Point(oc.lon, oc.lat)) for oc in other_cores):
                    continue
                se = _shape_error(poly)
                if se > max_shape_error:
                    continue
                amplitude = abs(core.ssha_value - lev)
                if amplitude < min_amplitude:
                    continue
                area_km2 = _area_km2(poly, core.lat)
                if area_km2 < min_area_km2:
                    continue
                mean_speed = _mean_speed_on_boundary(fields.speed, lats, lons, poly)
                cand = ContourEddy(
                    core_id=core.core_id, time_index=time_index,
                    centroid_lat=core.lat, centroid_lon=core.lon,
                    polygon_lonlat=poly, area_km2=area_km2,
                    rotation=rotation_for_sign(core.sign),
                    shape_error=se, amplitude=amplitude, mean_speed=mean_speed,
                )
                if best is None or (np.isfinite(mean_speed) and (not np.isfinite(best.mean_speed) or mean_speed > best.mean_speed)):
                    best = cand
        if best is not None:
            out.append(best)
    return out
