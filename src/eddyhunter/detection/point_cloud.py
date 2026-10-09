"""Turn a velocity/SSH field into the ring-shaped spatial point clouds that
persistent-H1 homology can read as loops.

Why rings, not raw valid-grid points
-------------------------------------
Persistent H1 on a 2D point cloud detects *holes* — regions with no points
that are encircled by points. A disk of densely-packed points has trivial
H1 (its interior is already "filled in" by the Delaunay/alpha triangulation).
A set of points sampled in an *annulus* around an empty center, by contrast,
produces exactly one robust H1 generator, whose persistence tracks how
clean and well-separated the ring is from the noise.

Because SSH anomaly is the geostrophic streamfunction, its iso-contours
*are* the flow streamlines. A closed, roughly circular band of constant
|SSH anomaly| around a local extremum is therefore a direct proxy for a
closed geostrophic streamline, i.e. candidate eddy. We detect local extrema
of |ssha| as candidate eddy cores, then select the grid points whose ssha
value (same sign as the core) falls in a mid-amplitude band around that
core, each tagged back to its candidate core id. Running persistent
homology per-candidate (rather than one huge global complex) keeps this
fast and gives each candidate its own clean H1 bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter, label

from .fields import DerivedFields, latlon_to_local_xy_m


@dataclass
class CandidateCore:
    core_id: int
    row: int
    col: int
    lat: float
    lon: float
    ssha_value: float
    sign: int  # +1 for SSH maximum (anticyclone, NH), -1 for SSH minimum (cyclone, NH)


@dataclass
class RingPointCloud:
    core: CandidateCore
    points_xy_m: np.ndarray       # (n, 2) local-tangent-plane meters, for alpha complex
    points_lat: np.ndarray
    points_lon: np.ndarray
    rows: np.ndarray
    cols: np.ndarray


def _robust_noise_std(field: np.ndarray) -> float:
    """Median-absolute-deviation based noise estimate, robust to the handful
    of large-amplitude real eddy cells that would otherwise inflate a plain
    std() estimate.
    """
    valid = field[np.isfinite(field)]
    if valid.size == 0:
        return 0.0
    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    return 1.4826 * mad


def find_candidate_cores(
    fields: DerivedFields,
    min_abs_ssha: float = 0.02,
    neighborhood_cells: int = 9,
    smoothing_sigma_cells: float = 2.0,
    noise_sigma_multiple: float = 5.0,
    source_field: Optional[np.ndarray] = None,
) -> List[CandidateCore]:
    """Local maxima/minima of a smoothed field, above a data-driven noise
    floor, as eddy-core candidates.

    `source_field` lets the caller run this same extrema-finding machinery
    on something other than SSH anomaly -- notably relative vorticity. SSH
    rings only work when the input has full 2D areal coverage around each
    core (true for a gridded/mapped product, false for a single raw SWOT
    swath pass, which is only ~1-2 pixels wide in cross-track direction at
    coarse resolution and never closes a ring). Vorticity computed from a
    gridded velocity product like OSCAR *does* have that areal coverage, so
    detecting on vorticity and treating SWOT SSH purely as a cross-check
    (not the primary field) is what real single-pass data needs; this
    mirrors the project's own stated role for SSH ("retained for
    comparison and validation") more literally than building rings from
    SSH ever did. Defaults to `fields.ssha` for backward compatibility.

    The raw field is smoothed first (Gaussian, sigma in grid cells) purely
    for *locating* extrema -- this suppresses single-cell sensor noise from
    masquerading as its own candidate core without altering the physical
    values used everywhere else downstream. The amplitude threshold is the
    larger of the caller's floor and `noise_sigma_multiple` times a robust
    (MAD-based) noise estimate of the field, so the detector adapts to how
    noisy the actual input is rather than relying on one hand-picked number.
    """
    ssha = source_field if source_field is not None else fields.ssha
    valid = np.isfinite(ssha)

    # NaN-aware Gaussian smoothing: a plain gaussian_filter on a NaN-filled
    # array lets the kernel blend in filler values from outside the valid
    # region, which is catastrophic when valid coverage is sparse (e.g. a
    # real SWOT swath covering a few percent of the grid) -- the "filled"
    # background bleeds into the real signal and washes it out. Instead we
    # smooth (data*mask) and (mask) separately with the same kernel and
    # divide, so each output cell is a weighted average using only nearby
    # *valid* cells (standard normalized-convolution trick).
    zero_filled = np.where(valid, ssha, 0.0)
    mask_f = valid.astype(float)
    num = gaussian_filter(zero_filled, sigma=smoothing_sigma_cells)
    den = gaussian_filter(mask_f, sigma=smoothing_sigma_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(den > 1e-6, num / den, np.nan)

    # Noise is estimated from the HIGH-PASS residual (raw - smoothed) at
    # valid cells only, not the raw field's own spread: raw std conflates
    # real mesoscale signal with sensor noise, and the residual isolates
    # small-scale noise regardless of how much of the grid is covered.
    residual = np.where(valid & np.isfinite(smoothed), ssha - smoothed, np.nan)
    noise_std = _robust_noise_std(residual)
    threshold = max(min_abs_ssha, noise_sigma_multiple * noise_std)

    valid = valid & np.isfinite(smoothed)

    # fill with +/-inf (not NaN) before max/min_filter so invalid cells never
    # win a local extremum comparison and never poison a neighboring window
    footprint = np.ones((neighborhood_cells, neighborhood_cells))
    smoothed_for_max = np.where(valid, smoothed, -np.inf)
    smoothed_for_min = np.where(valid, smoothed, np.inf)
    local_max = (smoothed_for_max == maximum_filter(smoothed_for_max, footprint=footprint)) & valid & (smoothed > threshold)
    local_min = (smoothed_for_min == minimum_filter(smoothed_for_min, footprint=footprint)) & valid & (smoothed < -threshold)

    cores: List[CandidateCore] = []
    cid = 0
    for mask, sign in ((local_max, +1), (local_min, -1)):
        rows, cols = np.nonzero(mask)
        for r, c in zip(rows, cols):
            cores.append(CandidateCore(
                core_id=cid, row=int(r), col=int(c),
                lat=float(fields.lats[r]), lon=float(fields.lons[c]),
                ssha_value=float(ssha[r, c]), sign=sign,
            ))
            cid += 1
    return cores


def build_ring_point_clouds(
    fields: DerivedFields,
    cores: List[CandidateCore],
    box_radius_cells: int = 15,
    band_low_frac: float = 0.15,
    band_high_frac: float = 0.75,
    min_points: int = 10,
    source_field: Optional[np.ndarray] = None,
) -> List[RingPointCloud]:
    """For each candidate core, select nearby grid points whose source-field
    value (same sign as the core) is between band_low_frac and
    band_high_frac of the core's peak magnitude -- the 'ring'. Must be
    called with the same `source_field` passed to `find_candidate_cores`
    (defaults to `fields.ssha` for backward compatibility).

    Band-matching alone is not enough to define a ring: on a field that
    stays close to its peak magnitude across a wide area (a broad, flat
    vorticity patch rather than a sharply peaked Gaussian-like SSH bump --
    exactly what a basin-scale current, as opposed to a compact mesoscale
    eddy, looks like), "same sign and in this amplitude band" can match
    roughly half the search box, including cells that have nothing to do
    with this particular core -- they just happen to sit in the band by
    coincidence of the box they were swept into. We additionally require the
    selected cells to form a single connected component (4-connectivity)
    containing a cell adjacent to the core itself, which is what actually
    enforces "this is one contiguous ring around this core" rather than "any
    same-sign, same-amplitude cell within `box_radius_cells`".
    """
    ssha = source_field if source_field is not None else fields.ssha
    nlat, nlon = ssha.shape
    clouds: List[RingPointCloud] = []

    for core in cores:
        r0, c0 = core.row, core.col
        r_lo, r_hi = max(0, r0 - box_radius_cells), min(nlat, r0 + box_radius_cells + 1)
        c_lo, c_hi = max(0, c0 - box_radius_cells), min(nlon, c0 + box_radius_cells + 1)

        sub_ssha = ssha[r_lo:r_hi, c_lo:c_hi]
        valid = np.isfinite(sub_ssha)

        peak_mag = abs(core.ssha_value)
        same_sign_val = sub_ssha * core.sign  # positive where same rotational sign
        in_band = valid & (same_sign_val > band_low_frac * peak_mag) & (same_sign_val < band_high_frac * peak_mag)

        # Restrict to the connected component touching the core's own cell
        # (dilated by 1 cell, since the core cell itself is the peak and is
        # usually just *outside* the band by construction).
        labeled, _ = label(in_band)
        rr, cc = core.row - r_lo, core.col - c_lo
        rr_lo, rr_hi = max(0, rr - 1), min(labeled.shape[0], rr + 2)
        cc_lo, cc_hi = max(0, cc - 1), min(labeled.shape[1], cc + 2)
        neighborhood_labels = labeled[rr_lo:rr_hi, cc_lo:cc_hi]
        candidate_labels = neighborhood_labels[neighborhood_labels > 0]
        if candidate_labels.size == 0:
            continue
        core_label = np.bincount(candidate_labels.ravel()).argmax()
        in_band = in_band & (labeled == core_label)

        rows, cols = np.nonzero(in_band)
        if rows.size < min_points:
            continue

        abs_rows = rows + r_lo
        abs_cols = cols + c_lo
        lat_pts = fields.lats[abs_rows]
        lon_pts = fields.lons[abs_cols]

        x_m, y_m = latlon_to_local_xy_m(lat_pts, lon_pts, core.lat, core.lon)
        points_xy = np.column_stack([x_m, y_m])

        clouds.append(RingPointCloud(
            core=core, points_xy_m=points_xy,
            points_lat=lat_pts, points_lon=lon_pts,
            rows=abs_rows, cols=abs_cols,
        ))

    return clouds
