"""Motion estimation and short-term position forecasting for eddy tracks.

This is "Track B": for each live track we estimate displacement/velocity
from its recent history, extrapolate a next-step position (constant-
velocity, or a short linear regression when more history is available),
and attach a confidence score built from track length, fit quality, and
the eddy's own persistence score (a weak/noisy eddy's future position is
less trustworthy than a strong, long-lived one's).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..tracking.associate import Track, TrackPoint, haversine_km
from ..detection.fields import latlon_to_local_xy_m, EARTH_RADIUS_M


@dataclass
class TrackBRecord:
    track_id: str
    time_index: int
    eddy_id: str
    centroid_lat: float
    centroid_lon: float
    polygon_lonlat: object
    persistence_m: float
    mean_speed: float
    rotation: str
    track_length: int
    velocity_u_ms: float          # eastward drift velocity, m/s
    velocity_v_ms: float          # northward drift velocity, m/s
    displacement_km_last_step: float
    predicted_lat: float
    predicted_lon: float
    confidence: float
    fit_r2: float


def _local_xy_km(lats: np.ndarray, lons: np.ndarray, ref_lat: float, ref_lon: float):
    x_m, y_m = latlon_to_local_xy_m(lats, lons, ref_lat, ref_lon)
    return x_m / 1000.0, y_m / 1000.0


def _xy_km_to_latlon(x_km: float, y_km: float, ref_lat: float, ref_lon: float):
    dlat = (y_km * 1000.0) / EARTH_RADIUS_M * 180.0 / np.pi
    dlon = (x_km * 1000.0) / (EARTH_RADIUS_M * np.cos(np.deg2rad(ref_lat))) * 180.0 / np.pi
    return ref_lat + dlat, ref_lon + dlon


def estimate_motion_and_forecast(
    track: Track,
    dt_hours: float = 24.0,
    max_history: int = 5,
    strong_persistence_m: float = 30_000.0,
) -> TrackBRecord:
    """Compute velocity + next-position forecast + confidence for a track's
    *current* (most recent) point, using up to `max_history` prior points.
    """
    pts: List[TrackPoint] = track.points[-max_history:]
    latest = pts[-1]
    ref_lat, ref_lon = latest.eddy.centroid_lat, latest.eddy.centroid_lon

    lats = np.array([p.eddy.centroid_lat for p in pts])
    lons = np.array([p.eddy.centroid_lon for p in pts])
    x_km, y_km = _local_xy_km(lats, lons, ref_lat, ref_lon)
    steps = np.arange(len(pts), dtype=float)
    dt_s = dt_hours * 3600.0

    if len(pts) == 1:
        vx_kmstep, vy_kmstep, r2 = 0.0, 0.0, 0.0
        fit_quality = 0.3
    elif len(pts) == 2:
        vx_kmstep = x_km[-1] - x_km[-2]
        vy_kmstep = y_km[-1] - y_km[-2]
        r2 = 1.0  # a 2-point line fits itself perfectly; treat as moderate trust below
        fit_quality = 0.6
    else:
        vx_kmstep, bx = np.polyfit(steps, x_km, 1)
        vy_kmstep, by = np.polyfit(steps, y_km, 1)
        pred_x = vx_kmstep * steps + bx
        pred_y = vy_kmstep * steps + by
        ss_res = np.sum((x_km - pred_x) ** 2 + (y_km - pred_y) ** 2)
        ss_tot = np.sum((x_km - x_km.mean()) ** 2 + (y_km - y_km.mean()) ** 2) + 1e-9
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        fit_quality = 0.5 + 0.5 * r2

    velocity_u_ms = (vx_kmstep * 1000.0) / dt_s
    velocity_v_ms = (vy_kmstep * 1000.0) / dt_s

    pred_lat, pred_lon = _xy_km_to_latlon(vx_kmstep, vy_kmstep, ref_lat, ref_lon)

    if len(pts) >= 2:
        disp_km = haversine_km(lats[-2], lons[-2], lats[-1], lons[-1])
    else:
        disp_km = 0.0

    len_score = min(1.0, (track.length - 1) / 4.0)
    persistence_score = min(1.0, latest.eddy.persistence_m / strong_persistence_m)
    confidence = float(np.clip(0.4 * len_score + 0.3 * fit_quality + 0.3 * persistence_score, 0.0, 1.0))

    return TrackBRecord(
        track_id=track.track_id, time_index=latest.time_index, eddy_id=latest.eddy.eddy_id,
        centroid_lat=latest.eddy.centroid_lat, centroid_lon=latest.eddy.centroid_lon,
        polygon_lonlat=latest.eddy.polygon_lonlat, persistence_m=latest.eddy.persistence_m,
        mean_speed=latest.eddy.mean_speed, rotation=latest.eddy.rotation,
        track_length=track.length, velocity_u_ms=velocity_u_ms, velocity_v_ms=velocity_v_ms,
        displacement_km_last_step=disp_km, predicted_lat=pred_lat, predicted_lon=pred_lon,
        confidence=confidence, fit_r2=r2,
    )


def build_track_b_output(tracks: List[Track], dt_hours: float = 24.0) -> List[TrackBRecord]:
    """Track-B record for every (track, time) point -- not just the latest --
    so the whole history can be scored against ground truth later.
    """
    records: List[TrackBRecord] = []
    for tr in tracks:
        for i in range(len(tr.points)):
            sub_track = Track(track_id=tr.track_id, points=tr.points[: i + 1])
            records.append(estimate_motion_and_forecast(sub_track, dt_hours=dt_hours))
    return records


def track_b_to_dataframe(records: List[TrackBRecord]):
    import pandas as pd
    rows = []
    for r in records:
        d = r.__dict__.copy()
        poly = d.pop("polygon_lonlat", None)
        d["polygon_wkt"] = poly.wkt if poly is not None else None
        rows.append(d)
    return pd.DataFrame(rows)
