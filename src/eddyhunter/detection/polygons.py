"""Convert a surviving (ring, H1 bar) pair into an eddy polygon with physical
properties: centroid, area, mean velocity/vorticity, persistence score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from shapely.geometry import Polygon, MultiPoint

from .fields import DerivedFields, latlon_to_local_xy_m
from .topology import TopologyResult

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class Eddy:
    eddy_id: str
    time_index: int
    centroid_lat: float
    centroid_lon: float
    polygon_lonlat: Polygon          # outer boundary, (lon, lat) coords
    area_km2: float
    persistence_m: float
    mean_u: float
    mean_v: float
    mean_speed: float
    mean_vorticity: float
    rotation: str                     # "cyclonic" | "anticyclonic"
    n_ring_points: int
    mean_ssha: float = float("nan")   # mean SWOT SSH anomaly under the polygon, when available -- cross-check only
    validated: bool = False
    validation_notes: List[str] = field(default_factory=list)

    def to_dict(self):
        d = {
            "eddy_id": self.eddy_id, "time_index": self.time_index,
            "centroid_lat": self.centroid_lat, "centroid_lon": self.centroid_lon,
            "area_km2": self.area_km2, "persistence_m": self.persistence_m,
            "mean_u": self.mean_u, "mean_v": self.mean_v, "mean_speed": self.mean_speed,
            "mean_vorticity": self.mean_vorticity, "rotation": self.rotation,
            "mean_ssha": self.mean_ssha,
            "n_ring_points": self.n_ring_points, "validated": self.validated,
            "validation_notes": ";".join(self.validation_notes),
            "polygon_wkt": self.polygon_lonlat.wkt,
        }
        return d


def _polygon_from_ring_points(lat_pts: np.ndarray, lon_pts: np.ndarray) -> Optional[Polygon]:
    """Outer boundary of the ring point set, as a smooth ellipse.

    Real-data ring points are noisy, so their raw convex hull is jagged and
    pointy -- it reads as a scribble on a map, not as "an eddy." Published
    eddy atlases (Chelton et al. 2011; Mason et al. 2014) report eddies as
    smooth closed contours, and most downstream displays further simplify
    those to an ellipse fit. We do the same here: fit an ellipse to the ring
    points via PCA (principal axes = ring's major/minor spread, in local
    meters so it's not distorted by the lat/lon aspect ratio), then rescale
    it so its area exactly matches the raw convex hull's area. This keeps
    every physical quantity derived from polygon area (area_km2, and the
    mean-field sampling that uses "inside the polygon") numerically
    consistent with before -- only the *boundary shape* becomes smooth.
    """
    pts = MultiPoint(list(zip(lon_pts, lat_pts)))
    hull = pts.convex_hull
    if hull.geom_type != "Polygon" or hull.area == 0:
        return None

    lat0, lon0 = float(np.mean(lat_pts)), float(np.mean(lon_pts))
    x_m, y_m = latlon_to_local_xy_m(lat_pts, lon_pts, lat0, lon0)
    xy = np.column_stack([x_m, y_m])
    if xy.shape[0] < 3:
        return hull

    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    a0, b0 = float(np.sqrt(eigvals[0])), float(np.sqrt(eigvals[1]))
    angle = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Convex hull's own centroid in local meters, so the ellipse is centered
    # on the actual point cloud rather than pulled by its own hull vertices.
    hull_x_m, hull_y_m = latlon_to_local_xy_m(
        np.array(hull.exterior.coords.xy[1]), np.array(hull.exterior.coords.xy[0]), lat0, lon0)
    cx, cy = float(np.mean(hull_x_m)), float(np.mean(hull_y_m))

    hull_area_m2 = hull.area * (EARTH_RADIUS_M ** 2) * (np.pi / 180.0) ** 2 * np.cos(np.deg2rad(lat0))
    # Recompute hull area directly in meters instead (robust to the above
    # approximation): reproject hull vertices to local meters and measure.
    hull_local = Polygon(zip(hull_x_m, hull_y_m))
    hull_area_m2 = hull_local.area

    k = np.sqrt(hull_area_m2 / (np.pi * a0 * b0)) if (a0 > 0 and b0 > 0) else 1.0
    a, b = a0 * k, b0 * k

    theta = np.linspace(0.0, 2 * np.pi, 48)
    ex = a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
    ey = a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)
    ex, ey = ex + cx, ey + cy

    ell_lat = lat0 + np.rad2deg(ey / EARTH_RADIUS_M)
    ell_lon = lon0 + np.rad2deg(ex / (EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))))

    ellipse = Polygon(zip(ell_lon, ell_lat))
    if not ellipse.is_valid or ellipse.area == 0:
        return hull
    return ellipse


def _polygon_area_km2(polygon_lonlat: Polygon, ref_lat: float) -> float:
    lons, lats = polygon_lonlat.exterior.coords.xy
    x_m, y_m = latlon_to_local_xy_m(np.array(lats), np.array(lons), ref_lat, float(np.mean(lons)))
    local_poly = Polygon(zip(x_m, y_m))
    return local_poly.area / 1e6  # m^2 -> km^2


def _mean_field_in_polygon(fields: DerivedFields, polygon_lonlat: Polygon, var: np.ndarray):
    lon_grid, lat_grid = np.meshgrid(fields.lons, fields.lats)
    minx, miny, maxx, maxy = polygon_lonlat.bounds
    box = (lon_grid >= minx) & (lon_grid <= maxx) & (lat_grid >= miny) & (lat_grid <= maxy)
    rows, cols = np.nonzero(box)
    if rows.size == 0:
        return float("nan")

    try:
        from shapely import contains_xy
        inside = contains_xy(polygon_lonlat, lon_grid[rows, cols], lat_grid[rows, cols])
    except ImportError:  # pragma: no cover - older shapely fallback
        from matplotlib.path import Path
        path = Path(np.array(polygon_lonlat.exterior.coords))
        inside = path.contains_points(np.column_stack([lon_grid[rows, cols], lat_grid[rows, cols]]))

    sel_rows, sel_cols = rows[inside], cols[inside]
    if sel_rows.size == 0:
        return float("nan")
    vals = var[sel_rows, sel_cols]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def build_eddies(
    topo_results: List[TopologyResult],
    fields: DerivedFields,
    time_index: int,
    min_area_km2: float = 20.0,
    source_field: str = "ssha",
) -> List[Eddy]:
    """`source_field` says which field the candidate cores/rings were built
    from ("ssha" or "vorticity") -- it only changes the sign convention used
    to label rotation, since a local SSH maximum and a local vorticity
    maximum imply *opposite* rotations in the Northern Hemisphere (high SSH
    anomaly = anticyclonic; positive relative vorticity = cyclonic). All the
    region covered by this project's real-data scene is Northern Hemisphere
    (down to the equator, where the convention itself breaks down -- not
    handled specially here).
    """
    eddies: List[Eddy] = []
    for i, res in enumerate(topo_results):
        best = res.best_bar
        if best is None:
            continue
        ring = res.ring

        polygon = _polygon_from_ring_points(ring.points_lat, ring.points_lon)
        if polygon is None or not polygon.is_valid or polygon.area == 0:
            continue

        area_km2 = _polygon_area_km2(polygon, ring.core.lat)
        if area_km2 < min_area_km2:
            continue

        mean_u = _mean_field_in_polygon(fields, polygon, fields.u)
        mean_v = _mean_field_in_polygon(fields, polygon, fields.v)
        mean_vort = _mean_field_in_polygon(fields, polygon, fields.vorticity)
        # NOTE: averaging u, v component-wise cancels around a closed
        # rotation (tangential vectors point every which way), so mean_u/
        # mean_v describe the eddy's net *translation/drift*, not its
        # rotational strength. The rotational strength is the mean of the
        # scalar speed field itself.
        mean_speed = _mean_field_in_polygon(fields, polygon, fields.speed)
        mean_ssha = _mean_field_in_polygon(fields, polygon, fields.ssha)

        if source_field == "vorticity":
            rotation = "cyclonic" if ring.core.sign > 0 else "anticyclonic"
        else:
            rotation = "anticyclonic" if ring.core.sign > 0 else "cyclonic"

        eddies.append(Eddy(
            eddy_id=f"t{time_index}_c{ring.core.core_id}",
            time_index=time_index,
            centroid_lat=float(np.mean(ring.points_lat)),
            centroid_lon=float(np.mean(ring.points_lon)),
            polygon_lonlat=polygon,
            area_km2=area_km2,
            persistence_m=best.persistence_m,
            mean_u=mean_u, mean_v=mean_v, mean_speed=mean_speed,
            mean_vorticity=mean_vort,
            rotation=rotation,
            n_ring_points=ring.points_xy_m.shape[0],
            mean_ssha=mean_ssha,
        ))
    return eddies
