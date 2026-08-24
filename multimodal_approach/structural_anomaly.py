"""Self-referential structural anomaly detection from the post-implant MRI.

No CT, no diffusion MRI, no FLAIR, no pre-implant baseline, no normative
population, and no manually placed electrode fiducials — only the two
isotropic-ish series already in the repository
(``dataset/MRI-with-electrodes/DCM``): ``t1_mprage_tra_p2_iso`` and
``t2_space_TR_p2_iso``. Given that, a trained normative autoencoder (the
"reconstruction-error" approach used in published unsupervised MRI anomaly
detection — see the package README for citations) is not reproducible here:
there is no population to train it on. What *is* reproducible with only this
single scan is the classic single-subject alternative used in the same
literature for epilepsy lesion screening when no population reference is
available: **hemispheric asymmetry mapping**. Most of a healthy brain is
close to left/right mirror-symmetric; a voxel whose intensity differs
sharply from its mirror counterpart, in a way this same brain's own
asymmetry distribution says is unusual, is a structural anomaly candidate.

Method, in order:

1. Segment a head mask (Otsu threshold + largest connected component) so
   background air never enters the statistics.
2. Flag signal-void voxels (dark relative to their local neighbourhood —
   the electrode contacts and shafts) and exclude them from anomaly scoring.
   This is instrumentation, not pathology, exactly the same reasoning this
   repo already applies to excluding the ``MKR...`` hardware clock channels
   from the timeseries detector (see ``extreme_event_agent.edf_workflow.read_edf``).
3. Find the true sagittal midline in patient space (not a voxel-index guess
   — this dataset's acquisition is tilted a few degrees, confirmed from its
   own ``ImageOrientationPatient``) by searching for the mirror plane that
   maximizes self-correlation of the (masked) brain against its own reflection.
4. Compute a per-voxel asymmetry index ``(v - mirror(v)) / (v + mirror(v))``
   and self-referentially z-score it against this same brain's own
   asymmetry distribution (median/MAD, so ~0 asymmetry is the null
   expectation and outliers are relative to this subject, not a population).
5. Repeat independently on T2 (resampled onto the T1 grid using the real
   DICOM patient-space affine — see ``dicom_geometry``), and keep only
   voxels where **both contrasts agree in sign** — susceptibility artifact,
   noise, and single-sequence bias field rarely agree between two
   differently-weighted acquisitions, so this cross-contrast agreement step
   is a real, if partial, defence against single-sequence false positives.

This is a proxy for a lesion detector, not one: sensitivity/specificity here
are unmeasured (there is no lesion mask to check against), and the technique
literature it is modelled on the caveat this section leans on — see the
package README's discussion of what this can and cannot claim.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .dicom_geometry import VolumeGeometry, list_series, load_series_geometry

__all__ = ["StructuralAnomalyResult", "run_structural_anomaly", "find_top_anomaly_clusters"]


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Standard Otsu automatic threshold; no external segmentation tool needed."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    sum_all = float((hist * centers).sum())
    sum_bg, weight_bg = 0.0, 0.0
    best_thresh, best_var = float(centers[0]), -1.0
    for index in range(bins):
        weight_bg += hist[index]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += hist[index] * centers[index]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var, best_thresh = var_between, float(centers[index])
    return best_thresh


def _head_mask(volume: np.ndarray) -> np.ndarray:
    """Head tissue vs. background air: Otsu threshold, largest connected component."""
    nonzero = volume[volume > 0]
    threshold = _otsu_threshold(nonzero) if nonzero.size else 0.0
    mask = volume > threshold
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, index=range(1, n_components + 1))
    largest = int(np.argmax(sizes)) + 1
    return ndimage.binary_fill_holes(labeled == largest)


def _artifact_mask(volume: np.ndarray, head_mask: np.ndarray,
                    median_size: int = 5, z_threshold: float = 6.0) -> np.ndarray:
    """Signal-void (electrode-artifact) candidate voxels: dark relative to local neighbourhood.

    Same median/MAD-threshold pattern ``extreme_event_agent`` uses for
    timeseries outliers, applied spatially instead of temporally.
    """
    local_median = ndimage.median_filter(volume, size=median_size)
    contrast = volume - local_median
    values = contrast[head_mask]
    if values.size == 0:
        return np.zeros_like(head_mask, dtype=bool)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826 + 1e-6
    z_score = (contrast - median) / mad
    candidate = (z_score < -z_threshold) & head_mask
    return ndimage.binary_dilation(candidate, iterations=1)


def _erode_mm(mask: np.ndarray, geometry: VolumeGeometry, margin_mm: float) -> np.ndarray:
    """Shrink a mask inward by ~``margin_mm``, using the geometry's own voxel size.

    The Otsu head mask in ``_head_mask`` is a threshold on raw intensity, not
    a real skull-strip — it keeps scalp and skull along with brain, and its
    boundary is exactly where T1 (0.9 mm) vs. resampled-T2 partial-volume
    disagreement, and any residual unmasked artifact halo, is worst. A small
    inward margin trades a thin rim of true cortex (a real cost — a
    surface-adjacent lesion near this margin would be missed) for removing a
    boundary layer that was otherwise dominating the anomaly map with
    non-anatomical (scalp/skull/partial-volume) asymmetry — see this
    package's README for the visual before/after.
    """
    if margin_mm <= 0:
        return mask
    voxel_size_mm = float(np.linalg.norm(geometry.d_row))  # approx isotropic voxel edge
    iterations = max(1, round(margin_mm / voxel_size_mm))
    return ndimage.binary_erosion(mask, iterations=iterations)


def _strided_geometry(geometry: VolumeGeometry, stride: int) -> VolumeGeometry:
    return VolumeGeometry(
        volume=geometry.volume[::stride, ::stride, ::stride],
        origin=geometry.origin,
        d_slice=geometry.d_slice * stride,
        d_row=geometry.d_row * stride,
        d_col=geometry.d_col * stride,
        series=geometry.series,
    )


def _mirror_volume_on_grid(geometry: VolumeGeometry, volume_on_grid: np.ndarray,
                            midline_x: float, coords: np.ndarray) -> np.ndarray:
    """Reflect ``volume_on_grid`` (already defined on ``geometry``'s grid) across x = midline_x."""
    mirrored_xyz = coords.copy()
    mirrored_xyz[..., 0] = 2.0 * midline_x - coords[..., 0]
    voxel_coords = geometry.patient_to_voxel(mirrored_xyz)
    coords_for_map = np.moveaxis(voxel_coords, -1, 0)
    return ndimage.map_coordinates(volume_on_grid, coords_for_map, order=1, mode="constant", cval=0.0)


def _resample_onto(source_geometry: VolumeGeometry, source_volume: np.ndarray,
                    target_geometry: VolumeGeometry, target_coords: np.ndarray,
                    order: int = 1) -> np.ndarray:
    """Resample ``source_volume`` (on ``source_geometry``'s grid) onto ``target_geometry``'s grid.

    Uses each series' own DICOM patient-space affine — both series were
    acquired in the same scanner session, so this needs no external
    registration software, only the geometry already in the files.
    """
    voxel_coords = source_geometry.patient_to_voxel(target_coords)
    coords_for_map = np.moveaxis(voxel_coords, -1, 0)
    return ndimage.map_coordinates(source_volume, coords_for_map, order=order, mode="constant", cval=0.0)


def _search_midline(geometry: VolumeGeometry, head_mask: np.ndarray, artifact_mask: np.ndarray,
                     stride: int = 4, radius_mm: float = 20.0, step_mm: float = 1.0) -> tuple[float, float]:
    """Find the sagittal plane x = midline that best mirrors the (masked) brain onto itself."""
    small = _strided_geometry(geometry, stride)
    head_small = head_mask[::stride, ::stride, ::stride]
    artifact_small = artifact_mask[::stride, ::stride, ::stride]
    valid_small = head_small & ~artifact_small
    coords_small = small.full_coordinate_grid()
    x_small = coords_small[..., 0]
    center = float(np.median(x_small[valid_small])) if valid_small.any() else 0.0

    base_vals = small.volume[valid_small]
    best_mid, best_corr = center, -2.0
    for mid in np.arange(center - radius_mm, center + radius_mm + step_mm, step_mm):
        mirrored = _mirror_volume_on_grid(small, small.volume, float(mid), coords_small)
        mirrored_vals = mirrored[valid_small]
        if base_vals.std() < 1e-6 or mirrored_vals.std() < 1e-6:
            continue
        corr = float(np.corrcoef(base_vals, mirrored_vals)[0, 1])
        if corr > best_corr:
            best_corr, best_mid = corr, float(mid)
    return best_mid, best_corr


def _asymmetry_and_z(volume_on_grid: np.ndarray, mirrored: np.ndarray,
                      valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denom = volume_on_grid + mirrored
    usable = valid_mask & (np.abs(denom) > 1e-3)
    asymmetry = np.zeros_like(volume_on_grid, dtype=np.float32)
    asymmetry[usable] = (volume_on_grid[usable] - mirrored[usable]) / denom[usable]
    values = asymmetry[usable]
    if values.size == 0:
        return np.zeros_like(asymmetry), usable
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826 + 1e-6
    z_score = np.zeros_like(asymmetry)
    z_score[usable] = (asymmetry[usable] - median) / mad
    return z_score, usable


@dataclass
class StructuralAnomalyResult:
    """Everything ``run_structural_anomaly`` produces, all on the T1 grid.

    ``combined_anomaly`` is the T1/T2-agreement-gated signed z-score map —
    the primary output. ``t1_anomaly_z``/``t2_anomaly_z`` are each contrast's
    own z-score before the agreement gate, kept for inspection/debugging.
    ``hemisphere_summary`` reduces ``combined_anomaly`` to one number per
    hemisphere (mean/max |z|), the input ``extreme_event_prior`` consumes.
    """

    combined_anomaly: np.ndarray
    t1_anomaly_z: np.ndarray
    t2_anomaly_z: np.ndarray
    head_mask: np.ndarray
    artifact_mask: np.ndarray
    midline_x_mm: float
    midline_mirror_correlation: float
    hemisphere_summary: dict[str, object]
    t1_geometry: VolumeGeometry
    timings_seconds: dict[str, float] = field(default_factory=dict)


def find_top_anomaly_clusters(result: StructuralAnomalyResult, threshold: float = 4.0,
                               min_voxels: int = 5, top_n: int = 5,
                               min_z_percentile: float | None = 55.0) -> list[dict[str, object]]:
    """Reduce ``combined_anomaly`` to a short, ranked list of contiguous anomaly clusters.

    A single "hottest voxel" is easy to pick but is exactly as likely to be a
    one-voxel noise spike as a real finding; a whole-volume mean (the
    hemisphere summary) is stable but cannot point at *where*. This sits
    between the two: connected-component clusters of voxels at or above
    ``threshold``, ranked by total mass (``sum(|combined_anomaly|)`` over the
    cluster — rewards clusters that are both strong and spatially coherent,
    not just one extreme voxel), each reporting its peak voxel (the actual
    "best view" location a figure should slice through) alongside its
    centroid, size, and patient-space position.

    ``min_z_percentile`` exists because of a real finding on this dataset:
    with no filter at all, every one of the top five clusters sat at patient
    Z around the 10th-40th percentile of the head mask — i.e. skull
    base/mastoid/upper-neck level, not cerebrum. Mastoid air-cell aeration is
    routinely left/right asymmetric in healthy people for no pathological
    reason, and that bony, air-adjacent region is exactly where a uniform
    few-millimetre mask erosion (``brain_margin_mm``) does the least to
    separate true tissue from a sharp, non-anatomical intensity edge. Rather
    than silently keep ranking those highest, this discards any cluster
    whose peak voxel falls below the given percentile of the (already
    brain_margin-eroded) head mask's own Z range before ranking what
    remains — a crude cerebrum floor, not a real segmentation boundary, so a
    lesion very low in the temporal or occipital pole could still be
    excluded by it. Pass ``None`` to disable and see the unfiltered ranking
    (including skull-base clusters) instead.
    """
    z_map = None
    z_cutoff = None
    if min_z_percentile is not None:
        z_map = result.t1_geometry.z_coordinate_map()
        head_z = z_map[result.head_mask]
        if head_z.size:
            z_cutoff = float(np.percentile(head_z, min_z_percentile))

    mask = np.abs(result.combined_anomaly) >= threshold
    if z_cutoff is not None:
        mask = mask & (z_map >= z_cutoff)
    labeled, n_components = ndimage.label(mask)
    clusters: list[dict[str, object]] = []
    for label_id in range(1, n_components + 1):
        component = labeled == label_id
        size = int(component.sum())
        if size < min_voxels:
            continue
        values = result.combined_anomaly[component]
        peak_index = int(np.argmax(np.abs(values)))
        peak_voxel = tuple(int(c) for c in np.argwhere(component)[peak_index])
        centroid_voxel = tuple(float(c) for c in ndimage.center_of_mass(component))
        peak_xyz = result.t1_geometry.voxel_to_patient(*peak_voxel)
        clusters.append({
            "voxel_count": size,
            "total_mass": float(np.abs(values).sum()),
            "mean_abs_anomaly": float(np.abs(values).mean()),
            "peak_value": float(values[peak_index]),
            "peak_voxel_kij": peak_voxel,
            "centroid_voxel_kij": centroid_voxel,
            "peak_patient_xyz_mm": [float(c) for c in peak_xyz],
            "hemisphere": "right" if float(peak_xyz[0]) < result.midline_x_mm else "left",
        })
    clusters.sort(key=lambda c: c["total_mass"], reverse=True)
    return clusters[:top_n]


def run_structural_anomaly(dicom_dir: str | Path, t1_series_number: str | None = None,
                            t2_series_number: str | None = None, brain_margin_mm: float = 4.0,
                            verbose: bool = True) -> StructuralAnomalyResult:
    def log(message: str) -> None:
        if verbose:
            print(f"[structural_anomaly] {message}", flush=True)

    timings: dict[str, float] = {}

    def timed(label, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        timings[label] = time.perf_counter() - start
        log(f"{label}: {timings[label]:.1f}s")
        return result

    series_map = list_series(dicom_dir)
    if t1_series_number is None:
        t1_series_number = next(num for num, s in series_map.items() if "t1" in s.description.lower())
    if t2_series_number is None:
        t2_series_number = next(num for num, s in series_map.items() if "t2" in s.description.lower())

    geom_t1 = timed("load_t1", load_series_geometry, series_map[t1_series_number])
    geom_t2 = timed("load_t2", load_series_geometry, series_map[t2_series_number])
    log(f"T1 {geom_t1.volume.shape}, T2 {geom_t2.volume.shape}")

    head_mask_t1 = timed("head_mask_t1", _head_mask, geom_t1.volume)
    artifact_mask_t1 = timed("artifact_mask_t1", _artifact_mask, geom_t1.volume, head_mask_t1)
    head_mask_t1 = _erode_mm(head_mask_t1, geom_t1, brain_margin_mm)

    midline_x, midline_corr = timed("midline_search", _search_midline, geom_t1, head_mask_t1, artifact_mask_t1)
    log(f"midline x = {midline_x:.2f} mm (mirror self-correlation r = {midline_corr:.3f})")

    coords_t1 = timed("coordinate_grid", geom_t1.full_coordinate_grid)

    smoothed_t1 = ndimage.gaussian_filter(geom_t1.volume, sigma=1.0)
    mirrored_t1 = timed("mirror_t1", _mirror_volume_on_grid, geom_t1, smoothed_t1, midline_x, coords_t1)
    valid_t1 = head_mask_t1 & ~artifact_mask_t1
    z_t1, usable_t1 = _asymmetry_and_z(smoothed_t1, mirrored_t1, valid_t1)

    t2_on_t1 = timed("resample_t2_onto_t1", _resample_onto, geom_t2, geom_t2.volume, geom_t1, coords_t1, 1)
    head_mask_t2 = _head_mask(t2_on_t1)
    artifact_mask_t2 = _artifact_mask(t2_on_t1, head_mask_t2)
    head_mask_t2 = _erode_mm(head_mask_t2, geom_t1, brain_margin_mm)
    smoothed_t2 = ndimage.gaussian_filter(t2_on_t1, sigma=1.0)
    mirrored_t2 = timed("mirror_t2", _mirror_volume_on_grid, geom_t1, smoothed_t2, midline_x, coords_t1)
    valid_t2 = head_mask_t2 & ~artifact_mask_t2
    z_t2, usable_t2 = _asymmetry_and_z(smoothed_t2, mirrored_t2, valid_t2)

    agree = usable_t1 & usable_t2 & (np.sign(z_t1) == np.sign(z_t2)) & (z_t1 != 0)
    combined = np.zeros_like(z_t1)
    combined[agree] = np.sign(z_t1[agree]) * np.minimum(np.abs(z_t1[agree]), np.abs(z_t2[agree]))
    log(f"{int(agree.sum())} / {int(valid_t1.sum())} in-mask voxels have T1/T2-agreeing asymmetry sign")

    x_map = geom_t1.x_coordinate_map()
    # DICOM LPS: +x is toward patient Left, so x < midline is patient Right.
    head_right = head_mask_t1 & (x_map < midline_x)
    head_left = head_mask_t1 & (x_map >= midline_x)
    right_mask = valid_t1 & (x_map < midline_x)
    left_mask = valid_t1 & (x_map >= midline_x)

    def _summary(mask: np.ndarray, head_side_mask: np.ndarray) -> dict[str, object]:
        # artifact_fraction is reported alongside every score for exactly the
        # confound this module's README flags: this implant is placed
        # asymmetrically (some shafts have no contralateral counterpart), so
        # a side with more excluded artifact voxels also has a smaller,
        # differently-composed pool of voxels feeding its anomaly summary —
        # a reviewer should be able to see that before trusting a left/right
        # difference as anatomical rather than an artifact-coverage effect.
        artifact_here = int((artifact_mask_t1 & head_side_mask).sum())
        head_here = int(head_side_mask.sum())
        if not mask.any():
            return {"voxel_count": 0, "mean_abs_anomaly": None, "max_abs_anomaly": None,
                     "artifact_voxel_count": artifact_here, "artifact_fraction_of_head": (
                         artifact_here / head_here if head_here else None)}
        values = np.abs(combined[mask])
        return {
            "voxel_count": int(mask.sum()),
            "mean_abs_anomaly": float(values.mean()),
            "max_abs_anomaly": float(values.max()),
            "p99_abs_anomaly": float(np.percentile(values, 99)),
            "artifact_voxel_count": artifact_here,
            "artifact_fraction_of_head": artifact_here / head_here if head_here else None,
        }

    hemisphere_summary = {
        "midline_x_mm": midline_x,
        "midline_mirror_correlation": midline_corr,
        "right_hemisphere": _summary(right_mask, head_right),
        "left_hemisphere": _summary(left_mask, head_left),
    }

    return StructuralAnomalyResult(
        combined_anomaly=combined,
        t1_anomaly_z=z_t1,
        t2_anomaly_z=z_t2,
        head_mask=head_mask_t1,
        artifact_mask=artifact_mask_t1,
        midline_x_mm=midline_x,
        midline_mirror_correlation=midline_corr,
        hemisphere_summary=hemisphere_summary,
        t1_geometry=geom_t1,
        timings_seconds=timings,
    )
