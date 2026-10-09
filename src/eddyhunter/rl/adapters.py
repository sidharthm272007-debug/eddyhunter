"""Turn real Track-B output (from the TDA detector + tracker + forecaster)
into ReplayStep sequences the PPO navigation env can step through, so a
trained policy can be evaluated on actual detected-and-forecasted eddies
instead of just the procedural training distribution.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..forecast.predict import TrackBRecord
from ..detection.fields import latlon_to_local_xy_m
from .env import ReplayStep


def track_b_records_to_replay_steps(
    records: List[TrackBRecord], dt_hours: float = 24.0, strong_persistence_m: float = 30_000.0,
) -> List[ReplayStep]:
    """`records` must be one track's TrackBRecords, already time-ordered."""
    if not records:
        return []

    ref_lat, ref_lon = records[0].centroid_lat, records[0].centroid_lon
    dt_s = dt_hours * 3600.0

    steps = []
    for r in records:
        x_m, y_m = latlon_to_local_xy_m(np.array([r.centroid_lat]), np.array([r.centroid_lon]), ref_lat, ref_lon)
        pred_x_m, pred_y_m = latlon_to_local_xy_m(np.array([r.predicted_lat]), np.array([r.predicted_lon]), ref_lat, ref_lon)
        vel_u_kmstep = r.velocity_u_ms * dt_s / 1000.0
        vel_v_kmstep = r.velocity_v_ms * dt_s / 1000.0
        steps.append(ReplayStep(
            eddy_x_km=float(x_m[0]) / 1000.0, eddy_y_km=float(y_m[0]) / 1000.0,
            pred_x_km=float(pred_x_m[0]) / 1000.0, pred_y_km=float(pred_y_m[0]) / 1000.0,
            vel_u_kmstep=vel_u_kmstep, vel_v_kmstep=vel_v_kmstep,
            persistence_norm=min(1.0, r.persistence_m / strong_persistence_m),
            confidence=r.confidence,
        ))
    return steps
