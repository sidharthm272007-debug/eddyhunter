"""Associate per-time-step eddy detections into continuous tracks.

Matching cost between an eddy at time t and a candidate at time t+1 combines:
 - centroid distance (km)
 - polygon overlap (1 - IoU)
 - relative area difference
 - relative persistence-score difference
 - motion-consistency: distance between the candidate and where the track's
   current constant-velocity estimate predicts it should be (0 for a brand
   new track with no velocity estimate yet). The velocity estimate itself
   is produced by eddyhunter.forecast.predict and is not recomputed here;
   TrackPoint.predicted_lat/lon stay None through plain `track_eddies`
   calls, which simply disables this term (distance/overlap/area/
   persistence already gate matches tightly enough for well-separated
   mesoscale eddies). Call eddyhunter.forecast.predict.build_track_b_output
   after tracking to get forecasts for evaluation/RL use.

Only same-rotation-sign eddies (cyclonic<->cyclonic, anticyclonic<->
anticyclonic) are allowed to match. Candidate pairs costlier than a gating
distance are forbidden outright (treated as "no possible match") so a track
never jumps to an unrelated eddy on the other side of the domain. Solved
per consecutive time-step pair with the Hungarian algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..detection.polygons import Eddy
from ..detection.fields import latlon_to_local_xy_m

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def polygon_iou(poly_a, poly_b) -> float:
    if not poly_a.intersects(poly_b):
        return 0.0
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return float(inter / union) if union > 0 else 0.0


@dataclass
class TrackPoint:
    time_index: int
    eddy: Eddy
    predicted_lat: Optional[float] = None
    predicted_lon: Optional[float] = None


@dataclass
class Track:
    track_id: str
    points: List[TrackPoint] = field(default_factory=list)
    missed_streak: int = 0

    @property
    def rotation(self) -> str:
        return self.points[0].eddy.rotation

    @property
    def length(self) -> int:
        return len(self.points)

    def latest(self) -> TrackPoint:
        return self.points[-1]


@dataclass
class AssociationConfig:
    max_gate_km: float = 120.0           # forbid matches farther apart than this, per missed step
    w_distance: float = 1.0
    w_iou: float = 1.0
    w_area: float = 0.5
    w_persistence: float = 0.3
    w_motion: float = 0.8
    distance_scale_km: float = 60.0      # normalizes centroid distance to ~[0,2]
    max_missed_steps: int = 2            # tolerate up to this many consecutive non-detections
                                          # before permanently closing a track (handles brief
                                          # swath gaps / one-frame detector misses)


def _pairwise_cost(prev_point: TrackPoint, cand: Eddy, cfg: AssociationConfig,
                    dt_steps: int = 1) -> float:
    if prev_point.eddy.rotation != cand.rotation:
        return np.inf

    d_km = haversine_km(prev_point.eddy.centroid_lat, prev_point.eddy.centroid_lon,
                         cand.centroid_lat, cand.centroid_lon)
    if d_km > cfg.max_gate_km * dt_steps:
        return np.inf

    iou = polygon_iou(prev_point.eddy.polygon_lonlat, cand.polygon_lonlat)
    area_rel_diff = abs(prev_point.eddy.area_km2 - cand.area_km2) / max(prev_point.eddy.area_km2, 1.0)
    pers_rel_diff = abs(prev_point.eddy.persistence_m - cand.persistence_m) / max(prev_point.eddy.persistence_m, 1.0)

    motion_cost = 0.0
    if prev_point.predicted_lat is not None:
        motion_cost = haversine_km(prev_point.predicted_lat, prev_point.predicted_lon,
                                    cand.centroid_lat, cand.centroid_lon) / cfg.distance_scale_km

    cost = (
        cfg.w_distance * (d_km / cfg.distance_scale_km)
        + cfg.w_iou * (1.0 - iou)
        + cfg.w_area * area_rel_diff
        + cfg.w_persistence * pers_rel_diff
        + cfg.w_motion * motion_cost
    )
    return cost


def track_eddies(eddies_by_time: Dict[int, List[Eddy]], cfg: AssociationConfig = AssociationConfig()) -> List[Track]:
    time_indices = sorted(eddies_by_time.keys())
    if not time_indices:
        return []

    active_tracks: List[Track] = []
    finished_tracks: List[Track] = []
    next_track_num = 0

    # seed tracks from first time step
    for eddy in eddies_by_time[time_indices[0]]:
        t = Track(track_id=f"trk{next_track_num:04d}", points=[TrackPoint(time_indices[0], eddy)])
        next_track_num += 1
        active_tracks.append(t)

    for i in range(1, len(time_indices)):
        t_idx = time_indices[i]
        candidates = eddies_by_time[t_idx]

        if not active_tracks or not candidates:
            cost_matrix = np.empty((len(active_tracks), 0))
        else:
            cost_matrix = np.full((len(active_tracks), len(candidates)), np.inf)
            for ti, tr in enumerate(active_tracks):
                prev_point = tr.latest()
                dt_steps = t_idx - prev_point.time_index  # accounts for any prior missed steps
                for ci, cand in enumerate(candidates):
                    cost_matrix[ti, ci] = _pairwise_cost(prev_point, cand, cfg, dt_steps)

        matched_tracks = set()
        matched_cands = set()

        if cost_matrix.size > 0 and np.isfinite(cost_matrix).any():
            row_ind, col_ind = linear_sum_assignment(np.where(np.isfinite(cost_matrix), cost_matrix, 1e6))
            for r, c in zip(row_ind, col_ind):
                if np.isfinite(cost_matrix[r, c]):
                    active_tracks[r].points.append(TrackPoint(t_idx, candidates[c]))
                    active_tracks[r].missed_streak = 0
                    matched_tracks.add(r)
                    matched_cands.add(c)

        # unmatched active tracks: tolerate a short gap (sensor dropout /
        # one-frame detector miss) before permanently closing the track.
        still_active = []
        for ti, tr in enumerate(active_tracks):
            if ti in matched_tracks:
                still_active.append(tr)
            else:
                tr.missed_streak += 1
                if tr.missed_streak <= cfg.max_missed_steps:
                    still_active.append(tr)
                else:
                    finished_tracks.append(tr)

        # unmatched candidates become new tracks
        for ci, cand in enumerate(candidates):
            if ci not in matched_cands:
                nt = Track(track_id=f"trk{next_track_num:04d}", points=[TrackPoint(t_idx, cand)])
                next_track_num += 1
                still_active.append(nt)

        active_tracks = still_active

    finished_tracks.extend(active_tracks)
    return finished_tracks


def tracks_to_dataframe(tracks: List[Track]):
    import pandas as pd
    rows = []
    for tr in tracks:
        for p in tr.points:
            row = p.eddy.to_dict()
            row["track_id"] = tr.track_id
            row["track_length"] = tr.length
            rows.append(row)
    return pd.DataFrame(rows)
