"""Cross-validate this project's own topological eddy detections against the
independent closed-contour baseline in `baseline_contour.py`. This is the
real-world credibility exhibit: unlike `metrics.py` (scored against
synthetic ground truth, which real satellite data doesn't have), this
compares two *different detection methods* run on the same real SWOT/OSCAR
field, so agreement between them is real evidence neither one is just
finding noise.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import xarray as xr

from ..detection.fields import derive_fields
from .baseline_contour import detect_eddies_contour, ContourEddy

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def run_contour_baseline_over_time(ds: xr.Dataset, **kwargs) -> pd.DataFrame:
    """Run the independent contour detector on every time step of `ds`
    (must have u, v, ssha, lat, lon, time) and return one row per detected
    eddy.
    """
    lats = ds["lat"].values
    lons = ds["lon"].values
    rows: List[dict] = []
    for t_idx in range(ds.sizes["time"]):
        u = ds["u"].isel(time=t_idx).values
        v = ds["v"].isel(time=t_idx).values
        ssha = ds["ssha"].isel(time=t_idx).values
        fields = derive_fields(u, v, ssha, lats, lons)
        for e in detect_eddies_contour(fields, t_idx, **kwargs):
            rows.append(dict(
                time_index=e.time_index, centroid_lat=e.centroid_lat, centroid_lon=e.centroid_lon,
                rotation=e.rotation, area_km2=e.area_km2, shape_error=e.shape_error,
                amplitude=e.amplitude, mean_speed=e.mean_speed,
            ))
    return pd.DataFrame(rows)


def cross_validate(ours_df: pd.DataFrame, baseline_df: pd.DataFrame, max_match_km: float = 60.0) -> dict:
    """Greedy nearest-neighbor match (same time step, same rotation sign)
    between our validated detections and the independent contour-baseline
    detections. Reports it both directions: what fraction of OUR eddies the
    baseline also finds (precision-style -- are we finding real structures,
    not noise), and what fraction of the BASELINE's eddies we also find
    (recall-style against an independent method, not just our own
    synthetic ground truth).
    """
    ours = ours_df[ours_df.get("validated", True) == True] if "validated" in ours_df.columns else ours_df
    matched_ours_idx = set()
    matched_base_idx = set()
    offsets_km = []

    for t_idx, ours_t in ours.groupby("time_index"):
        base_t = baseline_df[baseline_df.time_index == t_idx]
        used_base = set()
        for oi, orow in ours_t.iterrows():
            best_d, best_bi = None, None
            for bi, brow in base_t.iterrows():
                if bi in used_base or brow["rotation"] != orow["rotation"]:
                    continue
                d = haversine_km(orow["centroid_lat"], orow["centroid_lon"], brow["centroid_lat"], brow["centroid_lon"])
                if d <= max_match_km and (best_d is None or d < best_d):
                    best_d, best_bi = d, bi
            if best_bi is not None:
                used_base.add(best_bi)
                matched_ours_idx.add(oi)
                matched_base_idx.add((t_idx, best_bi))
                offsets_km.append(best_d)

    n_ours = int(len(ours))
    n_base = int(len(baseline_df))
    n_matched = int(len(matched_ours_idx))
    offsets = np.array(offsets_km)

    return {
        "method": "closed-contour SSH identification (Chelton/Mason-style, same family as AVISO's "
                  "published eddy atlas and py-eddy-tracker), run independently on this project's own "
                  "real SWOT/OSCAR field",
        "n_our_eddies": n_ours,
        "n_baseline_eddies": n_base,
        "n_agreeing": n_matched,
        "our_agreement_rate": float(n_matched / n_ours) if n_ours else 0.0,
        "baseline_coverage_rate": float(n_matched / n_base) if n_base else 0.0,
        "mean_centroid_offset_km": float(offsets.mean()) if offsets.size else float("nan"),
        "median_centroid_offset_km": float(np.median(offsets)) if offsets.size else float("nan"),
        "max_match_km": max_match_km,
        "note": "\"Agreeing\" means an independent, structurally different detector (iso-contour growing "
                "on SSH, not persistent homology on a ring point cloud) places an eddy of the same "
                "rotation sign within max_match_km of ours in the same snapshot. This is not a claim "
                "that either method is ground truth -- it is evidence that our detections correspond "
                "to real SSH structure a standard, independently-implemented method also finds, rather "
                "than artifacts of the topological method alone.",
    }
