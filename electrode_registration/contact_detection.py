"""Candidate electrode-shaft trajectories from the MRI's own signal-void (artifact) mask — no CT needed.

``multimodal_approach.structural_anomaly._artifact_mask`` already flags
signal-void voxels (dark relative to local neighbourhood — electrode
contacts and shafts are metal, a near-total signal void on MR) so they can
be *excluded* from anomaly scoring. This module reuses that exact,
already-tested mask for a second purpose: as raw material for candidate
contact/shaft geometry, the same "reuse what's already tested" discipline
the rest of this repository follows (e.g. ``object_model/figure.py`` reusing
``_beta_gamma_z_scores``/``_seizure_graph_layout`` rather than
recomputing them).

**What this can and cannot claim.** ``detect_shaft_candidates`` finds
*candidate* elongated dark-voxel clusters — it does not know which one is
which named shaft, or even reliably how many of them are real shaft
segments versus vessels, sulcal shadow, or noise. Checked directly on
``sEEG-HFOs-8.edf``'s MRI: an unconstrained connected-component sweep at
several link distances (4-8 mm) consistently finds 70-98 candidate clusters
of at least 15 voxels — far more than the 12 shafts this patient's montage
documents. That is reported honestly by ``registration.py``'s
precision/recall metrics rather than hidden by quietly tuning parameters
until the count happens to come out to 12 (which would be exactly the kind
of "numbers with no way to check them"
``multimodal_approach/README.md``'s "Why this, and not full electrode
localization" already refuses to produce). ``max_link_mm=6.0`` here is
chosen from typical SEEG contact pitch (~3.5-5 mm centre-to-centre for this
kind of depth electrode) plus a margin for imaging blur, not tuned against
the expected shaft count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

__all__ = ["ShaftCandidate", "detect_shaft_candidates"]

# Typical adjacent-contact spacing for this style of depth electrode is
# roughly 3.5-5 mm centre-to-centre; doubling that gives enough margin to
# bridge the imaging-blur gap between two physically adjacent contacts'
# own signal-void blobs without being so generous that unrelated, merely
# nearby shafts chain together (see the module docstring's own sweep).
DEFAULT_MAX_LINK_MM = 6.0
DEFAULT_MIN_BLOB_VOXELS = 4


@dataclass
class ShaftCandidate:
    """One connected group of artifact-mask blobs, fit to a 3-D line.

    ``candidate_id`` is an arbitrary, rank-ordered label (``"cand_0"`` is
    the largest by voxel count) — it names nothing about which real shaft
    this might be; that association, when made at all, is
    ``registration.py``'s job, and is always a *guess*, never a
    verification. ``direction_unit`` is the first principal axis (PCA) of
    the pooled voxel point cloud; ``endpoint_a``/``endpoint_b`` are that
    axis's extremes, so ``length_mm`` is the cluster's own longest span, not
    a measurement of any implanted electrode's true physical length.
    ``hemisphere`` is derived from where the candidate's own voxels sit
    relative to the DICOM midline (``StructuralAnomalyResult.midline_x_mm``),
    the same convention ``structural_anomaly.py`` already uses — never from
    a name, since this candidate has none yet.
    """

    candidate_id: str
    hemisphere: str
    blob_count: int
    voxel_count: int
    centroid_patient_xyz_mm: tuple[float, float, float]
    endpoint_a_patient_xyz_mm: tuple[float, float, float]
    endpoint_b_patient_xyz_mm: tuple[float, float, float]
    length_mm: float
    direction_unit: tuple[float, float, float]


def _primitive_blobs(artifact_mask: np.ndarray, geometry: Any, midline_x_mm: float,
                     min_voxels: int) -> list[dict[str, Any]]:
    """Connected components of ``artifact_mask``, each reduced to size/centroid/hemisphere/points.

    Full 26-connectivity (``np.ones((3,3,3))``) so a thin, diagonally
    running shaft segment is not artificially split by 6-connectivity.
    Components spanning both hemispheres significantly (neither side holds
    at least 85% of the component's voxels) are dropped outright here, not
    just down-weighted: a real implanted SEEG shaft never crosses the
    midline, so a mixed-hemisphere blob is very unlikely to be one (more
    likely a vessel, falx, or other midline structure the artifact
    detector also flags).
    """
    labeled, n = ndimage.label(artifact_mask, structure=np.ones((3, 3, 3)))
    blobs = []
    for label_id in range(1, n + 1):
        component = labeled == label_id
        size = int(component.sum())
        if size < min_voxels:
            continue
        voxel_coords = np.argwhere(component)
        patient_coords = geometry.voxel_to_patient(voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2])
        frac_right = float((patient_coords[:, 0] < midline_x_mm).mean())
        if frac_right >= 0.85:
            hemisphere = "right"
        elif frac_right <= 0.15:
            hemisphere = "left"
        else:
            continue
        blobs.append({"size": size, "hemisphere": hemisphere,
                      "centroid": patient_coords.mean(axis=0), "points": patient_coords})
    return blobs


def _link_blobs_into_candidates(blobs: list[dict[str, Any]], max_link_mm: float) -> list[list[int]]:
    """Group same-hemisphere blobs into chains via a proximity graph (build_seizure_graph's own convention).

    Two blobs are linked if their centroids are within ``max_link_mm`` of
    each other *and* they are on the same hemisphere; connected components
    of that graph are the shaft candidates. The same threshold-pruned
    proximity-graph idea ``multimodal_approach.structural_graph.build_structural_anomaly_graph``
    already uses for anomaly clusters, applied here to raw artifact blobs
    instead — one graph-construction convention, reused rather than
    reinvented, for a different purpose.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(len(blobs)))
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            if blobs[i]["hemisphere"] != blobs[j]["hemisphere"]:
                continue
            distance = float(np.linalg.norm(blobs[i]["centroid"] - blobs[j]["centroid"]))
            if distance <= max_link_mm:
                graph.add_edge(i, j)
    return [sorted(component) for component in nx.connected_components(graph)]


def detect_shaft_candidates(result: Any, max_link_mm: float = DEFAULT_MAX_LINK_MM,
                            min_blob_voxels: int = DEFAULT_MIN_BLOB_VOXELS,
                            min_candidate_voxels: int = 15) -> list[ShaftCandidate]:
    """Find candidate shaft trajectories in ``result.artifact_mask``, ranked by voxel mass.

    ``result`` is a ``multimodal_approach.structural_anomaly.StructuralAnomalyResult``
    (needs ``artifact_mask``, ``midline_x_mm``, ``t1_geometry``). Two-stage
    clustering (see the module docstring for why one stage alone
    over- or under-merges): primitive same-hemisphere connected components
    of the artifact mask (``_primitive_blobs``), then a distance-pruned
    proximity graph over their centroids (``_link_blobs_into_candidates``,
    ``max_link_mm``) whose connected components become the returned
    candidates, each refit to one pooled 3-D line. Candidates below
    ``min_candidate_voxels`` total are dropped. Returned list is sorted by
    ``voxel_count`` descending — ``registration.py`` takes the top-N per
    hemisphere from this ranking, the same "ranked by mass, not tuned to a
    target count" principle ``structural_anomaly.find_top_anomaly_clusters``
    already uses.
    """
    geometry = result.t1_geometry
    blobs = _primitive_blobs(result.artifact_mask, geometry, result.midline_x_mm, min_blob_voxels)
    chains = _link_blobs_into_candidates(blobs, max_link_mm)

    candidates = []
    for chain in chains:
        total_voxels = sum(blobs[i]["size"] for i in chain)
        if total_voxels < min_candidate_voxels:
            continue
        points = np.concatenate([blobs[i]["points"] for i in chain], axis=0)
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        projections = centered @ direction
        length = float(projections.max() - projections.min())
        endpoint_a = centroid + direction * projections.min()
        endpoint_b = centroid + direction * projections.max()
        candidates.append(ShaftCandidate(
            candidate_id="", hemisphere=blobs[chain[0]]["hemisphere"], blob_count=len(chain),
            voxel_count=total_voxels, centroid_patient_xyz_mm=tuple(float(c) for c in centroid),
            endpoint_a_patient_xyz_mm=tuple(float(c) for c in endpoint_a),
            endpoint_b_patient_xyz_mm=tuple(float(c) for c in endpoint_b),
            length_mm=length, direction_unit=tuple(float(c) for c in direction)))

    candidates.sort(key=lambda c: c.voxel_count, reverse=True)
    for index, candidate in enumerate(candidates):
        candidate.candidate_id = f"cand_{index}"
    return candidates
