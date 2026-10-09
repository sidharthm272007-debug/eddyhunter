"""Alpha complex + persistent H1 homology over a ring point cloud, via gudhi.

Each H1 bar (birth, death) in the alpha-complex filtration describes a loop
that appears at filtration value `birth` and is filled in (killed) at
`death`. For an alpha complex, filtration values are squared circumradii
(meters^2 here, since input points are in a local meters projection), so we
work in sqrt-space to get physically meaningful radii in meters.

Persistence score follows the project's spec: P = death - birth (computed
in sqrt/meters space here for interpretability), and a bar with large P is
a loop that survives across a wide range of scales -- a real, well-formed
ring -- rather than a one-off triangulation artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import gudhi

from .point_cloud import RingPointCloud


@dataclass
class H1Bar:
    birth_m: float
    death_m: float

    @property
    def persistence_m(self) -> float:
        return self.death_m - self.birth_m


@dataclass
class TopologyResult:
    ring: RingPointCloud
    bars: List[H1Bar]

    @property
    def best_bar(self) -> Optional[H1Bar]:
        if not self.bars:
            return None
        return max(self.bars, key=lambda b: b.persistence_m)


def compute_h1_persistence(points_xy_m: np.ndarray, max_alpha_square: Optional[float] = None) -> List[H1Bar]:
    """Build an alpha complex from 2D points (meters) and return its H1 bars."""
    if points_xy_m.shape[0] < 3:
        return []

    alpha_complex = gudhi.AlphaComplex(points=points_xy_m)
    simplex_tree = alpha_complex.create_simplex_tree(
        max_alpha_square=max_alpha_square if max_alpha_square is not None else float("inf")
    )
    simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    intervals = simplex_tree.persistence_intervals_in_dimension(1)

    bars = []
    for birth_sq, death_sq in intervals:
        if not np.isfinite(death_sq):
            continue  # loop that never closes within the complex; ignore
        birth_m = float(np.sqrt(max(birth_sq, 0.0)))
        death_m = float(np.sqrt(max(death_sq, 0.0)))
        bars.append(H1Bar(birth_m=birth_m, death_m=death_m))
    return bars


def analyze_rings(rings: List[RingPointCloud]) -> List[TopologyResult]:
    results = []
    for ring in rings:
        bars = compute_h1_persistence(ring.points_xy_m)
        results.append(TopologyResult(ring=ring, bars=bars))
    return results


def filter_by_persistence(results: List[TopologyResult], min_persistence_m: float) -> List[TopologyResult]:
    """Drop candidates whose best H1 bar doesn't survive the noise threshold."""
    kept = []
    for res in results:
        best = res.best_bar
        if best is not None and best.persistence_m >= min_persistence_m:
            kept.append(res)
    return kept
