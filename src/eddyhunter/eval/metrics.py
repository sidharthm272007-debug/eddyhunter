"""Evaluation metrics for detection, tracking and forecasting, scored
against the synthetic dataset's known ground truth (real SWOT/OSCAR data
has no such ground truth, so this suite is the one that can give hard
numbers -- it's what "test on synthetic first" is for).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..tracking.associate import haversine_km, Track
from ..forecast.predict import TrackBRecord


def evaluate_detection_against_truth(eddies_df: pd.DataFrame, truth_df: pd.DataFrame,
                                      max_match_km: float = 60.0) -> dict:
    """Greedy nearest-neighbor match (same time step, compatible rotation
    sign where truth's sign is known) between detected eddies and the
    synthetic ground-truth table. Returns centroid error stats and recall.
    """
    matches = []
    n_truth = 0

    for t_idx, truth_t in truth_df.groupby("time_index"):
        n_truth += len(truth_t)
        det_t = eddies_df[(eddies_df.time_index == t_idx) & (eddies_df.validated)]
        used_det = set()
        for _, trow in truth_t.iterrows():
            best_d, best_i = None, None
            expected_rotation = "cyclonic" if trow["sign"] > 0 else "anticyclonic"
            for i, drow in det_t.iterrows():
                if i in used_det or drow["rotation"] != expected_rotation:
                    continue
                d = haversine_km(trow["lat"], trow["lon"], drow["centroid_lat"], drow["centroid_lon"])
                if d <= max_match_km and (best_d is None or d < best_d):
                    best_d, best_i = d, i
            if best_i is not None:
                used_det.add(best_i)
                matches.append(best_d)

    matches = np.array(matches)
    return {
        "n_truth_eddy_steps": n_truth,
        "n_matched": int(matches.size),
        "recall": float(matches.size / n_truth) if n_truth else 0.0,
        "mean_centroid_error_km": float(matches.mean()) if matches.size else float("nan"),
        "median_centroid_error_km": float(np.median(matches)) if matches.size else float("nan"),
    }


def evaluate_track_continuity(tracks: List[Track], n_time_steps: int) -> dict:
    """How contiguous are tracks? Reports mean track length relative to the
    full run length, and the number of distinct tracks per rotation sign
    (fewer, longer tracks = better continuity; many short tracks = lots of
    identity switches).
    """
    lengths = [t.length for t in tracks]
    return {
        "n_tracks": len(tracks),
        "mean_track_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_track_length": int(max(lengths)) if lengths else 0,
        "mean_track_length_frac_of_run": float(np.mean(lengths) / n_time_steps) if lengths else 0.0,
    }


def evaluate_forecast_error(records: List[TrackBRecord]) -> dict:
    """For every track, compare its predicted (t -> t+1) position against
    the eddy's actual matched position at t+1 (when the track continues
    there), in km.
    """
    by_track = {}
    for r in records:
        by_track.setdefault(r.track_id, []).append(r)

    errors = []
    for track_id, recs in by_track.items():
        recs = sorted(recs, key=lambda r: r.time_index)
        for i in range(len(recs) - 1):
            cur, nxt = recs[i], recs[i + 1]
            if nxt.time_index - cur.time_index != 1:
                continue  # skip across a tracking gap; forecast was for immediate next step
            err_km = haversine_km(cur.predicted_lat, cur.predicted_lon, nxt.centroid_lat, nxt.centroid_lon)
            errors.append(err_km)

    errors = np.array(errors)
    return {
        "n_forecast_steps": int(errors.size),
        "mean_forecast_error_km": float(errors.mean()) if errors.size else float("nan"),
        "median_forecast_error_km": float(np.median(errors)) if errors.size else float("nan"),
        "p90_forecast_error_km": float(np.percentile(errors, 90)) if errors.size else float("nan"),
    }
