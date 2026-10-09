"""Physical plausibility checks for candidate eddies.

Each candidate already comes from an SSH-anomaly-consistent ring (that
agreement is built into how the point cloud was constructed), so what's
left to check here is: does the velocity/vorticity field agree with the
rotation label implied by the SSH sign, is the circulation strong enough to
not be noise, and is the core actually enclosed by its own ring polygon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .polygons import Eddy
from .point_cloud import CandidateCore


@dataclass
class ValidationConfig:
    min_mean_speed: float = 0.03       # m/s
    min_abs_circulation: float = 1.0e4  # m^2/s  (= mean_vorticity * area_m2)
    max_area_km2: float = 200_000.0    # sanity upper bound
    require_core_enclosed: bool = True


def validate_eddy(
    eddy: Eddy,
    core: CandidateCore,
    config: ValidationConfig = ValidationConfig(),
    source_field: str = "ssha",
) -> Eddy:
    """`source_field` must match what was passed to detection -- it decides
    which check is a real independent cross-check vs. which would be
    circular.

    When `source_field == "ssha"` (candidates/rings built from SSH, as on
    synthetic data / a gridded SSH product), checking that velocity-derived
    vorticity sign agrees with the SSH-implied rotation is a genuine
    independent check. When `source_field == "vorticity"` (real single-pass
    SWOT data, where rotation is read directly off the same vorticity field
    used to build the ring), that same check would just be comparing the
    vorticity sign to itself -- so instead we use real SWOT SSH, where the
    narrow swath happens to cover this eddy's polygon, as the independent
    cross-check. Coverage is sparse by construction (the swath is a ~50km
    strip), so this is a soft/informational note, not a pass/fail gate: not
    finding SSH coverage under a given eddy is expected, not a red flag.
    """
    notes = []
    ok = True

    if not (np.isfinite(eddy.mean_vorticity) and np.isfinite(eddy.mean_speed)):
        eddy.validated = False
        eddy.validation_notes = ["missing velocity/vorticity data in polygon"]
        return eddy

    if source_field == "vorticity":
        notes.append("rotation read directly from vorticity (primary detection field for real data)")
        if np.isfinite(eddy.mean_ssha):
            expected_ssha_sign = +1 if eddy.rotation == "anticyclonic" else -1
            if np.sign(eddy.mean_ssha) == expected_ssha_sign:
                notes.append(f"cross-check: SWOT SSH sign agrees with vorticity-implied rotation ({eddy.mean_ssha:+.3f} m)")
            else:
                notes.append(f"cross-check: SWOT SSH sign disagrees with vorticity-implied rotation ({eddy.mean_ssha:+.3f} m) -- informational only, not failing the eddy")
        else:
            notes.append("cross-check: no SWOT SSH coverage under this eddy (narrow swath) -- cannot cross-check")
    else:
        expected_sign = +1 if eddy.rotation == "cyclonic" else -1
        vorticity_sign_ok = np.sign(eddy.mean_vorticity) == expected_sign
        if not vorticity_sign_ok:
            ok = False
            notes.append(f"vorticity sign disagrees with SSH-implied rotation ({eddy.rotation})")
        else:
            notes.append("vorticity sign agrees with SSH-implied rotation")

    area_m2 = eddy.area_km2 * 1e6
    circulation = eddy.mean_vorticity * area_m2
    if abs(circulation) < config.min_abs_circulation:
        ok = False
        notes.append(f"circulation too weak ({circulation:.2e} m^2/s)")
    else:
        notes.append(f"circulation magnitude {circulation:.2e} m^2/s OK")

    if eddy.mean_speed < config.min_mean_speed:
        ok = False
        notes.append(f"mean speed too weak ({eddy.mean_speed:.3f} m/s)")

    if eddy.area_km2 > config.max_area_km2:
        ok = False
        notes.append("polygon implausibly large")

    if config.require_core_enclosed:
        core_in_poly = eddy.polygon_lonlat.contains(
            __import__("shapely.geometry", fromlist=["Point"]).Point(core.lon, core.lat)
        )
        if not core_in_poly:
            ok = False
            notes.append("SSH core not enclosed by ring polygon")
        else:
            notes.append("SSH core enclosed by ring polygon")

    eddy.validated = ok
    eddy.validation_notes = notes
    return eddy


def validate_eddies(
    eddies: List[Eddy],
    cores_by_id,
    config: ValidationConfig = ValidationConfig(),
    source_field: str = "ssha",
) -> List[Eddy]:
    out = []
    for e in eddies:
        core_id = int(e.eddy_id.split("_c")[-1])
        core = cores_by_id[core_id]
        out.append(validate_eddy(e, core, config, source_field=source_field))
    return out
