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

A second, independent channel: local texture heterogeneity
-------------------------------------------------------------
The asymmetry channel above is blind **by construction** to any abnormality
that is not lateralized — a bilateral or midline structural change produces
no left/right difference at all, so it cannot show up in
``combined_anomaly`` no matter how real it is (see the package README,
"Honest limits"). Focal solid masses (the pediatric-oncology literature on
heterogeneity texture analysis in neuroblastoma is one well-known example,
but the same idea shows up generally in single-sequence unsupervised lesion
screening) tend to disrupt the *local* intensity texture of the tissue they
replace even when they sit on the midline or occur bilaterally, so a second,
symmetry-independent channel — local intensity heterogeneity, itself
self-referentially z-scored against this same head's own texture
distribution (same median/MAD null-hypothesis pattern as step 4, just
one-sided: elevated local heterogeneity relative to this subject's own
typical tissue, not relative to a population) — is computed in parallel and
cross-contrast-gated (both T1 and T2 elevated) the same way as the asymmetry
channel. This is reported as ``combined_heterogeneity``, kept fully separate
from ``combined_anomaly`` rather than merged into one score, so a reviewer
can always tell *which* mechanism (asymmetry vs. local heterogeneity, or
both) is behind any flagged region. Like the asymmetry channel, this is a
proxy, not a lesion detector: no lesion mask exists here to measure it
against, and normal structures (vasculature, sulcal CSF, partial-volume
edges) also have naturally elevated local heterogeneity, which the T1/T2
agreement gate only partially screens out.
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

    Applied to the already skull-stripped ``brain_mask`` (see
    ``_brain_extract``), not the raw Otsu head mask — its remaining job is a
    residual safety margin at the brain surface (T1 0.9 mm vs.
    resampled-T2 partial-volume disagreement, and any thin dura/bone-marrow
    rim ``_brain_extract``'s bone/air threshold did not fully catch), not
    stripping scalp/skull bulk, which ``_brain_extract`` now does directly by
    tissue signal instead of by a fixed distance. A small inward margin
    trades a thin rim of true cortex (a real cost — a surface-adjacent
    lesion near this margin would be missed) for removing a boundary layer
    that would otherwise dominate the anomaly map with non-anatomical
    (partial-volume/residual-dura) asymmetry.
    """
    if margin_mm <= 0:
        return mask
    voxel_size_mm = float(np.linalg.norm(geometry.d_row))  # approx isotropic voxel edge
    iterations = max(1, round(margin_mm / voxel_size_mm))
    return ndimage.binary_erosion(mask, iterations=iterations)


def _low_percentile_mask(values_by_voxel: np.ndarray, sample_mask: np.ndarray, percentile: float,
                          above: bool) -> np.ndarray:
    """``values_by_voxel >= pth percentile`` (``above=True``) or ``<=`` (``above=False``), within ``sample_mask``.

    The percentile is computed only from ``sample_mask``'s own voxels — this
    subject's own intensity distribution, not a population reference — same
    self-referential pattern the rest of this module uses throughout.
    """
    sample = values_by_voxel[sample_mask]
    if sample.size == 0:
        return np.zeros_like(sample_mask)
    cutoff = float(np.percentile(sample, percentile))
    return (values_by_voxel >= cutoff) if above else (values_by_voxel <= cutoff)


def _brain_extract(head_mask: np.ndarray, t1: np.ndarray, t2_on_t1: np.ndarray,
                    geometry: VolumeGeometry, bone_percentile: float = 45.0,
                    closing_iterations: int = 4, seed_margin_mm: float = 6.0) -> np.ndarray:
    """Strip scalp and skull from ``head_mask`` using T1+T2 signal, no external skull-strip tool.

    Cortical bone is close to a signal void on essentially every clinical MR
    sequence (very few mobile protons), unlike CSF (dark T1, bright T2) or
    brain parenchyma (moderate signal on both) — so "low signal on *both*
    T1 and T2" is a reasonable per-voxel proxy for bone or air, distinct from
    both other tissue classes, using only the two contrasts this dataset
    already has (no population atlas, no separate segmentation model). The
    calvarium forms a nearly complete thin shell of that low-signal tissue
    around the brain, so removing those voxels from ``head_mask`` breaks the
    connection between scalp (outside the skull) and brain (inside it) in
    most places; ``binary_closing`` patches the small gaps noise leaves in
    that break before the steps below.

    A first version of this function closed ``candidate`` (the brain-side
    mask) before labelling and ``fill_holes``-ing its largest component —
    checked against this dataset, not assumed correct, and it failed twice
    over, both caught only by inspecting the actual voxel counts (see this
    package's README, "Honest limits", for why that check exists at all):

    1. ``binary_closing`` fills small gaps — applied to ``candidate``, that
       is exactly backwards: the whole point of removing ``bone_or_air`` is
       to *open* a gap between scalp and brain, and closing the result can
       re-bridge that same gap through any few-voxel pinhole the percentile
       threshold missed (this bone/air classifier is a per-voxel intensity
       test with no morphological cleanup, so the shell it produces is
       porous with noise-sized holes almost everywhere). The fix is to close
       the *bone/air shell itself* (solidify it, patch its own pinholes)
       before subtracting it, not close what is left after subtracting it.
    2. Even with a real gap, labelling ``candidate`` and blindly
       ``fill_holes``-ing its largest component cannot tell "this enclosure
       is a ventricle" from "this enclosure is an eye socket that lost its
       thin bony roof" — both read as an interior hole of the same
       component and both got filled in as brain. The fix is a
       seed-and-constrained-grow instead of label-and-fill: erode
       ``candidate`` inward by ``seed_margin_mm`` (comfortably wider than
       residual orbital-roof/cribriform-plate leaks) so any lingering bridge
       to an extracranial pocket is severed; take the largest connected
       component of *that* — now unambiguously seeded inside the cranial
       cavity — as the seed; ``binary_propagation`` (geodesic dilation
       constrained to stay within ``candidate``) then grows the seed back
       out to the true brain surface *without ever being able to cross into
       a disconnected pocket*. ``binary_fill_holes`` runs only at the very
       end, on that already brain-constrained result, where an enclosure
       really is intracranial.

    Still a heuristic, not a validated skull-strip (see this package's
    README, "Honest limits") — it can leave a thin residual rim wherever
    cortical bone is thick enough or fatty enough (marrow) to not read as a
    clean signal void, and ``seed_margin_mm`` erosion is itself a fixed
    distance, not a segmentation, so it could in principle sever a genuine
    thin isthmus of brain tissue (unlikely at 6 mm on a normal cerebrum, not
    checked on pathologically distorted anatomy).

    ``bone_percentile``/``closing_iterations`` defaults were tuned by
    inspection on this dataset, the same way ``min_z_percentile=55`` in
    ``find_top_anomaly_clusters`` was — not derived from an atlas, and worth
    re-checking on other scans. At the first values tried (15th percentile,
    2 closing iterations) the "bone" shell this classifier found was real
    but too porous: even after seeding and constrained propagation, the
    result stayed 95% the size of ``head_mask`` (i.e. essentially un-stripped
    scalp+skull+brain) because ``candidate`` never actually disconnected —
    checked by comparing connected-component sizes directly, not assumed
    from how the classifier looked in isolation. Raising the percentile (to
    45) and closing iterations (to 4) — both make the shell thicker/more
    solid before subtracting it — produced a single dominant component whose
    volume (~1.1-1.2 L after the margin erosion below) sits in the normal
    adult whole-brain range, and whose surface visibly tracks the cortex
    across axial and coronal slices rather than following the skull. Both
    checks (component-size ratio, physically plausible volume) are what
    "tuned by inspection" means here — not a claim of validated accuracy.
    ``head_mask`` is also lightly closed and hole-filled before any of this
    (a few percent of raw Otsu ``head_mask`` voxels are themselves porous
    from noise/vessel voids, found by the same component-size check), so the
    bone/air classifier isn't fighting speckle noise in its own input.
    """
    head_mask = ndimage.binary_fill_holes(ndimage.binary_closing(head_mask, iterations=2))
    bone_or_air = (
        head_mask
        & _low_percentile_mask(t1, head_mask, bone_percentile, above=False)
        & _low_percentile_mask(t2_on_t1, head_mask, bone_percentile, above=False)
    )
    bone_or_air = ndimage.binary_closing(bone_or_air, iterations=closing_iterations)
    candidate = head_mask & ~bone_or_air

    seed = _erode_mm(candidate, geometry, seed_margin_mm)
    labeled, n_components = ndimage.label(seed)
    if n_components == 0:
        return candidate
    sizes = ndimage.sum(seed, labeled, index=range(1, n_components + 1))
    largest = int(np.argmax(sizes)) + 1
    brain_seed = labeled == largest

    brain = ndimage.binary_propagation(brain_seed, mask=candidate)
    return ndimage.binary_fill_holes(brain)


def _csf_mask(t1: np.ndarray, t2_on_t1: np.ndarray, brain_mask: np.ndarray,
              low_percentile: float = 25.0, high_percentile: float = 75.0) -> np.ndarray:
    """Low-T1/high-T2 voxels within ``brain_mask``: the classic CSF (fluid) signature.

    Ventricles, sulcal CSF, and subarachnoid space are dark on T1 and bright
    on T2 — the opposite intensity relationship from bone/air (dark on
    both, see ``_brain_extract``) and distinct from parenchyma (moderate on
    both), so the same two-contrast-agreement idea this module already uses
    for anomaly scoring also separates this third tissue class cleanly, with
    no separate fluid-segmentation tool. Percentiles are computed from
    ``brain_mask`` itself (post-skull-strip), not the raw head mask, so they
    reflect the intracranial tissue distribution rather than being diluted
    by scalp/skull intensities.

    This channel exists because CSF is a leading source of false anomaly
    flags for both the asymmetry channel (ventricle asymmetry — e.g. one
    occipital horn larger than the other — is a common normal variant, not
    pathology) and, especially, the local-heterogeneity channel (a sharp
    tissue/CSF partial-volume edge has genuinely high local variance for a
    purely geometric reason — see this package's README, "Honest limits",
    for a real example found on this dataset before this mask existed).
    """
    return (
        brain_mask
        & _low_percentile_mask(t1, brain_mask, low_percentile, above=False)
        & _low_percentile_mask(t2_on_t1, brain_mask, high_percentile, above=True)
    )


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


def _local_heterogeneity_index(volume: np.ndarray, window: int = 5) -> np.ndarray:
    """Windowed coefficient of variation: local intensity spread relative to local level.

    Plain local standard deviation would run hottest wherever the tissue is
    simply *bright* (T1 fat, T2 fluid), confounding "structurally
    heterogeneous" with "high signal". Dividing by the local mean (the
    coefficient-of-variation normalization) asks the question this channel
    actually wants: does this neighbourhood's intensity vary a lot *relative
    to its own local level*, independent of what that level is. Computed
    with ``ndimage.uniform_filter`` (mean and mean-of-squares, so variance
    falls out algebraically) rather than a generic windowed function — same
    box size as ``window``, exact and O(n) instead of a slow per-voxel loop.
    """
    mean = ndimage.uniform_filter(volume, size=window)
    mean_sq = ndimage.uniform_filter(volume * volume, size=window)
    variance = np.clip(mean_sq - mean * mean, 0.0, None)
    return np.sqrt(variance) / (np.abs(mean) + 1e-3)


def _heterogeneity_and_z(volume: np.ndarray, valid_mask: np.ndarray,
                          window: int = 5) -> np.ndarray:
    """Self-referential z-score of local heterogeneity: median/MAD, one-sided.

    Same null-hypothesis pattern as ``_asymmetry_and_z`` (this subject's own
    distribution defines "normal", not a population) but there is no sign to
    speak of here — only "more locally heterogeneous than this brain's own
    typical tissue" is the anomalous direction, so unlike the asymmetry
    z-score this is used through its positive tail only (see
    ``run_structural_anomaly``, where two contrasts must each be positive to
    agree, not just sign-matched).
    """
    heterogeneity = _local_heterogeneity_index(volume, window)
    z_score = np.zeros_like(heterogeneity)
    values = heterogeneity[valid_mask]
    if values.size == 0:
        return z_score
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826 + 1e-6
    z_score[valid_mask] = (heterogeneity[valid_mask] - median) / mad
    return z_score


@dataclass
class StructuralAnomalyResult:
    """Everything ``run_structural_anomaly`` produces, all on the T1 grid.

    ``combined_anomaly`` is the T1/T2-agreement-gated signed z-score map —
    the primary output. ``t1_anomaly_z``/``t2_anomaly_z`` are each contrast's
    own z-score before the agreement gate, kept for inspection/debugging.
    ``hemisphere_summary`` reduces ``combined_anomaly`` to one number per
    hemisphere (mean/max |z|), the input ``extreme_event_prior`` consumes.
    ``combined_heterogeneity`` is the independent, symmetry-blind texture
    channel (module docstring, "A second, independent channel") — always
    non-negative, gated the same T1/T2-agreement way but on "both elevated"
    rather than "same sign". ``heterogeneity_summary`` reduces it to one
    whole-head number (not per-hemisphere — the point of this channel is
    exactly that it does not assume a lateralized abnormality).
    ``brain_mask`` is the skull-stripped intracranial mask (``_brain_extract``)
    after the ``brain_margin_mm`` erosion — scalp and skull excluded, both
    anomaly channels are scored only inside it. ``csf_mask`` (``_csf_mask``)
    marks the CSF voxels *within* ``brain_mask`` that are additionally
    excluded from scoring — kept as its own field, not merged into
    ``brain_mask``, so the two exclusions (bone/scalp vs. fluid) stay
    separately auditable.
    """

    combined_anomaly: np.ndarray
    t1_anomaly_z: np.ndarray
    t2_anomaly_z: np.ndarray
    combined_heterogeneity: np.ndarray
    t1_heterogeneity_z: np.ndarray
    t2_heterogeneity_z: np.ndarray
    brain_mask: np.ndarray
    csf_mask: np.ndarray
    artifact_mask: np.ndarray
    midline_x_mm: float
    midline_mirror_correlation: float
    hemisphere_summary: dict[str, object]
    heterogeneity_summary: dict[str, object]
    t1_geometry: VolumeGeometry
    t2_on_t1: np.ndarray
    timings_seconds: dict[str, float] = field(default_factory=dict)


def find_top_anomaly_clusters(result: StructuralAnomalyResult, anomaly_map: np.ndarray | None = None,
                               threshold: float = 4.0, min_voxels: int = 5, top_n: int = 5,
                               min_z_percentile: float | None = None) -> list[dict[str, object]]:
    """Reduce an anomaly volume to a short, ranked list of contiguous clusters.

    ``anomaly_map`` defaults to ``result.combined_anomaly`` (the asymmetry
    channel); pass ``result.combined_heterogeneity`` to rank clusters from
    the independent texture channel instead — everything below (thresholding,
    the cerebrum-Z floor, ranking by mass, peak/centroid reporting) works
    identically on either map since both are gated, self-referential z-score
    volumes on the same T1 grid.

    A single "hottest voxel" is easy to pick but is exactly as likely to be a
    one-voxel noise spike as a real finding; a whole-volume mean (the
    hemisphere summary) is stable but cannot point at *where*. This sits
    between the two: connected-component clusters of voxels at or above
    ``threshold``, ranked by total mass (``sum(|combined_anomaly|)`` over the
    cluster — rewards clusters that are both strong and spatially coherent,
    not just one extreme voxel), each reporting its peak voxel (the actual
    "best view" location a figure should slice through) alongside its
    centroid, size, and patient-space position.

    ``min_z_percentile`` defaults to off (``None``) — it exists only as an
    optional extra height floor, not a required step. It used to default to
    55 for a real, checked reason: before this package's ``result.brain_mask``
    was an actual tissue-based skull-strip (see ``_brain_extract``), every
    one of the top five clusters on this dataset sat at patient Z around the
    10th-40th percentile of the (then scalp+skull-including) head mask —
    skull base/mastoid/upper-neck level, not cerebrum — because a uniform
    millimetre-scale mask erosion could not tell true tissue from a sharp,
    non-anatomical bone/air edge. Now that ``brain_mask`` excludes skull and
    skull-base tissue directly, by signal, not by height, that failure mode
    is gone at the source (checked: re-running with the old height floor
    still enabled after brain extraction now discards *every* cluster on
    this dataset — it is filtering out genuine low-cerebrum/temporal/
    cerebellar findings instead of skull base, since the brain mask's own Z
    range no longer includes the neck/skull-base bulk the percentile used to
    be calibrated against). Pass an explicit percentile to re-enable this as
    a belt-and-suspenders filter if a future dataset's brain extraction
    turns out to be less clean than this one's.
    """
    if anomaly_map is None:
        anomaly_map = result.combined_anomaly

    z_map = None
    z_cutoff = None
    if min_z_percentile is not None:
        z_map = result.t1_geometry.z_coordinate_map()
        brain_z = z_map[result.brain_mask]
        if brain_z.size:
            z_cutoff = float(np.percentile(brain_z, min_z_percentile))

    mask = np.abs(anomaly_map) >= threshold
    if z_cutoff is not None:
        mask = mask & (z_map >= z_cutoff)
    labeled, n_components = ndimage.label(mask)
    clusters: list[dict[str, object]] = []
    for label_id in range(1, n_components + 1):
        component = labeled == label_id
        size = int(component.sum())
        if size < min_voxels:
            continue
        values = anomaly_map[component]
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
    coords_t1 = timed("coordinate_grid", geom_t1.full_coordinate_grid)
    t2_on_t1 = timed("resample_t2_onto_t1", _resample_onto, geom_t2, geom_t2.volume, geom_t1, coords_t1, 1)

    # Skull-strip using both contrasts (see _brain_extract) instead of the
    # raw Otsu head_mask_t1 (scalp+skull+brain) that fed anomaly scoring
    # before — both channels below are now scored on brain tissue only.
    brain_mask_t1 = timed("brain_extract", _brain_extract, head_mask_t1, geom_t1.volume, t2_on_t1, geom_t1)
    csf_mask_t1 = timed("csf_mask", _csf_mask, geom_t1.volume, t2_on_t1, brain_mask_t1)
    log(f"brain-extracted {int(brain_mask_t1.sum())} / {int(head_mask_t1.sum())} head-mask voxels "
        f"({int(csf_mask_t1.sum())} of those flagged CSF)")
    brain_mask_t1 = _erode_mm(brain_mask_t1, geom_t1, brain_margin_mm)

    # csf_mask_t1 itself (kept undilated on the result, for audit) only
    # catches the CSF *core* — its own percentile test is strict by design.
    # The highest local heterogeneity actually sits one voxel further out, at
    # the partial-volume rim mixing brain and CSF signal (a real, checked
    # failure mode — see this package's README, "Honest limits"): neither
    # pure CSF nor pure parenchyma, so it can dodge the CSF test while still
    # reading as sharply heterogeneous for the same purely geometric reason.
    # A one-voxel dilation, used only for scoring exclusion (not stored as
    # the reported csf_mask), absorbs that rim without eating meaningfully
    # into periventricular white matter beyond it.
    csf_exclusion = ndimage.binary_dilation(csf_mask_t1, iterations=1)

    # Both artifact masks are normalized against actual brain-tissue signal
    # now (brain_mask_t1), not the raw scalp+skull+brain head mask — a
    # cleaner local-median baseline for the same signal-void detector. One
    # shared mask for both contrasts also means T1 and T2 agree by
    # construction on *where* tissue is, not just incidentally.
    artifact_mask_t1 = timed("artifact_mask_t1", _artifact_mask, geom_t1.volume, brain_mask_t1)
    artifact_mask_t2 = timed("artifact_mask_t2", _artifact_mask, t2_on_t1, brain_mask_t1)

    midline_x, midline_corr = timed("midline_search", _search_midline, geom_t1, brain_mask_t1, artifact_mask_t1)
    log(f"midline x = {midline_x:.2f} mm (mirror self-correlation r = {midline_corr:.3f})")

    smoothed_t1 = ndimage.gaussian_filter(geom_t1.volume, sigma=1.0)
    mirrored_t1 = timed("mirror_t1", _mirror_volume_on_grid, geom_t1, smoothed_t1, midline_x, coords_t1)
    valid_t1 = brain_mask_t1 & ~artifact_mask_t1 & ~csf_exclusion
    z_t1, usable_t1 = _asymmetry_and_z(smoothed_t1, mirrored_t1, valid_t1)

    smoothed_t2 = ndimage.gaussian_filter(t2_on_t1, sigma=1.0)
    mirrored_t2 = timed("mirror_t2", _mirror_volume_on_grid, geom_t1, smoothed_t2, midline_x, coords_t1)
    valid_t2 = brain_mask_t1 & ~artifact_mask_t2 & ~csf_exclusion
    z_t2, usable_t2 = _asymmetry_and_z(smoothed_t2, mirrored_t2, valid_t2)

    agree = usable_t1 & usable_t2 & (np.sign(z_t1) == np.sign(z_t2)) & (z_t1 != 0)
    combined = np.zeros_like(z_t1)
    combined[agree] = np.sign(z_t1[agree]) * np.minimum(np.abs(z_t1[agree]), np.abs(z_t2[agree]))
    log(f"{int(agree.sum())} / {int(valid_t1.sum())} in-mask voxels have T1/T2-agreeing asymmetry sign")

    # Second, independent channel: local texture heterogeneity (module
    # docstring, "A second, independent channel") — computed on the same
    # smoothed volumes and gated on the same common valid mask, but with no
    # mirroring/midline involved, so it can flag bilateral or midline
    # findings the asymmetry channel above cannot see by construction.
    common_valid = valid_t1 & valid_t2
    het_t1 = timed("heterogeneity_t1", _heterogeneity_and_z, smoothed_t1, common_valid)
    het_t2 = timed("heterogeneity_t2", _heterogeneity_and_z, smoothed_t2, common_valid)
    het_agree = common_valid & (het_t1 > 0) & (het_t2 > 0)
    combined_het = np.zeros_like(het_t1)
    combined_het[het_agree] = np.minimum(het_t1[het_agree], het_t2[het_agree])
    log(f"{int(het_agree.sum())} / {int(common_valid.sum())} in-mask voxels have T1/T2-agreeing "
        f"elevated local heterogeneity")

    x_map = geom_t1.x_coordinate_map()
    # DICOM LPS: +x is toward patient Left, so x < midline is patient Right.
    brain_right = brain_mask_t1 & (x_map < midline_x)
    brain_left = brain_mask_t1 & (x_map >= midline_x)
    right_mask = valid_t1 & (x_map < midline_x)
    left_mask = valid_t1 & (x_map >= midline_x)

    def _summary(mask: np.ndarray, brain_side_mask: np.ndarray) -> dict[str, object]:
        # artifact_fraction is reported alongside every score for exactly the
        # confound this module's README flags: this implant is placed
        # asymmetrically (some shafts have no contralateral counterpart), so
        # a side with more excluded artifact voxels also has a smaller,
        # differently-composed pool of voxels feeding its anomaly summary —
        # a reviewer should be able to see that before trusting a left/right
        # difference as anatomical rather than an artifact-coverage effect.
        artifact_here = int((artifact_mask_t1 & brain_side_mask).sum())
        brain_here = int(brain_side_mask.sum())
        if not mask.any():
            return {"voxel_count": 0, "mean_abs_anomaly": None, "max_abs_anomaly": None,
                     "artifact_voxel_count": artifact_here, "artifact_fraction_of_brain": (
                         artifact_here / brain_here if brain_here else None)}
        values = np.abs(combined[mask])
        return {
            "voxel_count": int(mask.sum()),
            "mean_abs_anomaly": float(values.mean()),
            "max_abs_anomaly": float(values.max()),
            "p99_abs_anomaly": float(np.percentile(values, 99)),
            "artifact_voxel_count": artifact_here,
            "artifact_fraction_of_brain": artifact_here / brain_here if brain_here else None,
        }

    hemisphere_summary = {
        "midline_x_mm": midline_x,
        "midline_mirror_correlation": midline_corr,
        "right_hemisphere": _summary(right_mask, brain_right),
        "left_hemisphere": _summary(left_mask, brain_left),
    }

    # Whole-head, not per-hemisphere: this channel's entire premise is that
    # it does not assume the abnormality is lateralized (see module
    # docstring), so splitting it by side would misrepresent what it can see.
    het_values = combined_het[het_agree]
    heterogeneity_summary = {
        "voxel_count": int(het_agree.sum()),
        "mean_heterogeneity_z": float(het_values.mean()) if het_values.size else None,
        "max_heterogeneity_z": float(het_values.max()) if het_values.size else None,
        "p99_heterogeneity_z": float(np.percentile(het_values, 99)) if het_values.size else None,
    }

    return StructuralAnomalyResult(
        combined_anomaly=combined,
        t1_anomaly_z=z_t1,
        t2_anomaly_z=z_t2,
        combined_heterogeneity=combined_het,
        t1_heterogeneity_z=het_t1,
        t2_heterogeneity_z=het_t2,
        brain_mask=brain_mask_t1,
        csf_mask=csf_mask_t1,
        artifact_mask=artifact_mask_t1,
        midline_x_mm=midline_x,
        midline_mirror_correlation=midline_corr,
        hemisphere_summary=hemisphere_summary,
        heterogeneity_summary=heterogeneity_summary,
        t1_geometry=geom_t1,
        t2_on_t1=t2_on_t1,
        timings_seconds=timings,
    )
