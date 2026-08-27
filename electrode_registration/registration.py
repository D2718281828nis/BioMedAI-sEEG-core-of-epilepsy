"""Best-guess assignment of shaft candidates to documented shafts, and per-contact position placement.

Nothing in this module verifies an electrode identity. ``register_shaft_candidates``
picks, for each documented shaft, the single candidate whose fitted length
is closest to that shaft's PDF-documented length — a plausibility-ranked
guess among 70-98 detected candidates (see ``contact_detection.py``'s
module docstring), never a confirmation that the picked candidate actually
is that shaft. Every consumer of this module's output must keep that
distinction visible — see ``metrics.py`` for how the resulting numbers are
framed as internal consistency, not accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contact_detection import ShaftCandidate
from .reference_geometry import SEEG_HFOS_8_SHAFT_REFERENCE, ShaftReference

__all__ = ["ShaftRegistration", "ContactPosition", "register_shaft_candidates",
          "assign_contact_positions", "extract_contact_roi", "contacts_outside_volume"]


@dataclass
class ShaftRegistration:
    """One documented shaft matched (or not) to its best-guess candidate.

    ``matched`` is ``False`` when no candidate remains to pair with this
    shaft (fewer detected candidates on this hemisphere than documented
    shafts) — an honest gap, not silently skipped. ``length_error_mm`` is
    the one "registration error" this module can honestly report: **not** a
    3-D position error against verified ground truth (none exists — see
    this package's README), but the plausibility gap between the matched
    candidate's own fitted length and this shaft's PDF-documented length. A
    small value means the candidate's *span* is consistent with what the
    clinician's note says this shaft should measure; it says nothing about
    whether the candidate is actually positioned where that shaft actually
    is.
    """

    shaft_name: str
    hemisphere: str
    reference: ShaftReference
    matched: bool
    candidate: ShaftCandidate | None
    length_error_mm: float | None
    length_relative_error: float | None


def register_shaft_candidates(candidates: list[ShaftCandidate],
                              reference: dict[str, ShaftReference] | None = None) -> list[ShaftRegistration]:
    """Match the top-N candidates per hemisphere to the N documented shafts on that side, by length similarity.

    N per hemisphere comes from ``reference`` itself (default
    ``SEEG_HFOS_8_SHAFT_REFERENCE`` — 7 right, 5 left on this dataset),
    never tuned to make the candidate count come out even; ``candidates``
    (already ranked by voxel mass — see ``contact_detection.detect_shaft_candidates``)
    is truncated to the top N per side before matching, so a hemisphere
    with more spurious candidates than real shafts cannot let a low-mass
    noise blob steal a match a higher-mass candidate already covers.

    Matching is a linear assignment (``scipy.optimize.linear_sum_assignment``)
    minimizing total absolute length error between candidates and shafts —
    Hungarian-optimal over the whole hemisphere at once, not a greedy
    nearest-first pairing that an early bad match could lock in.
    """
    reference = reference or SEEG_HFOS_8_SHAFT_REFERENCE
    registrations: list[ShaftRegistration] = []
    for hemisphere in ("right", "left"):
        shafts = [ref for ref in reference.values() if ref.hemisphere == hemisphere]
        hemisphere_candidates = [c for c in candidates if c.hemisphere == hemisphere][:len(shafts)]

        match_map: dict[int, int] = {}
        if hemisphere_candidates and shafts:
            cost = np.array([[abs(candidate.length_mm - shaft.length_mm) for candidate in hemisphere_candidates]
                             for shaft in shafts])
            from scipy.optimize import linear_sum_assignment
            shaft_rows, candidate_cols = linear_sum_assignment(cost)
            match_map = dict(zip(shaft_rows.tolist(), candidate_cols.tolist()))

        for shaft_index, shaft in enumerate(shafts):
            if shaft_index in match_map:
                candidate = hemisphere_candidates[match_map[shaft_index]]
                length_error = abs(candidate.length_mm - shaft.length_mm)
                registrations.append(ShaftRegistration(
                    shaft_name=shaft.name, hemisphere=hemisphere, reference=shaft, matched=True,
                    candidate=candidate, length_error_mm=length_error,
                    length_relative_error=(length_error / shaft.length_mm) if shaft.length_mm else None))
            else:
                registrations.append(ShaftRegistration(
                    shaft_name=shaft.name, hemisphere=hemisphere, reference=shaft, matched=False,
                    candidate=None, length_error_mm=None, length_relative_error=None))
    return registrations


@dataclass(frozen=True)
class ContactPosition:
    """One approximate contact position, evenly spaced along a matched candidate's fitted line.

    ``contact_label`` is ``"<shaft>_contact_<k>"`` (1-indexed along the
    fitted line's own direction) — **not** the true numbered contact
    ordering from the EDF montage (e.g. ``"EEG PM3"``): which end of the
    fitted line corresponds to contact 1 (entry) versus the deepest contact
    is unknown from the artifact mask alone (no proximal/distal marker
    survives voxel-intensity clustering), so this numbering is an
    arbitrary but fixed convention (the fitted line's own SVD sign), stated
    here rather than silently presented as if it matched the real
    recording's own contact 1..N.
    """

    contact_label: str
    shaft_name: str
    hemisphere: str
    patient_xyz_mm: tuple[float, float, float]


def assign_contact_positions(registrations: list[ShaftRegistration]) -> list[ContactPosition]:
    """Place each matched shaft's documented contact count evenly along its candidate's fitted line.

    Only matched registrations contribute — an unmatched shaft gets no
    positions at all (this gap is exactly what ``metrics.py``'s
    ``coverage`` measures). Evenly spacing ``reference.contact_count``
    points between the candidate's own fitted endpoints is a modelling
    choice, not a measurement: real contacts are not necessarily perfectly
    evenly spaced, and any noise in the fitted line's endpoints propagates
    directly into every placed position — which is why every position this
    function produces still has to pass ``contacts_outside_volume`` before
    being used for anything downstream.
    """
    positions: list[ContactPosition] = []
    for registration in registrations:
        if not registration.matched:
            continue
        candidate = registration.candidate
        count = registration.reference.contact_count
        endpoint_a = np.array(candidate.endpoint_a_patient_xyz_mm)
        endpoint_b = np.array(candidate.endpoint_b_patient_xyz_mm)
        for k in range(count):
            fraction = k / max(count - 1, 1)
            xyz = endpoint_a + fraction * (endpoint_b - endpoint_a)
            positions.append(ContactPosition(
                contact_label=f"{registration.shaft_name}_contact_{k + 1}", shaft_name=registration.shaft_name,
                hemisphere=registration.hemisphere, patient_xyz_mm=tuple(float(c) for c in xyz)))
    return positions


def extract_contact_roi(result: Any, position_xyz_mm: tuple[float, float, float],
                        radius_mm: float = 15.0) -> dict[str, Any]:
    """Crop the T1 volume (and its artifact mask) around one patient-space position.

    Returns a plain dict (not a dataclass wrapping large arrays):
    ``t1_roi``/``artifact_mask_roi`` (cropped 3-D arrays), ``voxel_bounds_kij``
    (the actual (k, i, j) slice bounds used, already clipped to the volume),
    ``center_voxel_kij``, and ``in_bounds`` (whether the *centre* position
    itself — not just the crop — falls inside the volume; see
    ``contacts_outside_volume`` for the same check applied to a whole list
    of positions at once). Radius is converted to voxels per axis using
    each axis's own physical spacing (``t1_geometry.d_slice``/``d_row``/
    ``d_col``), not assumed isotropic.
    """
    geometry = result.t1_geometry
    kij = geometry.patient_to_voxel(np.array(position_xyz_mm))
    voxel_size = np.array([np.linalg.norm(geometry.d_slice), np.linalg.norm(geometry.d_row),
                           np.linalg.norm(geometry.d_col)])
    half_span = np.maximum(1, np.round(radius_mm / voxel_size)).astype(int)
    shape = np.array(geometry.volume.shape)
    center = np.round(kij).astype(int)
    in_bounds = bool(np.all((kij >= 0) & (kij < shape)))
    lo = np.clip(center - half_span, 0, shape)
    hi = np.clip(center + half_span + 1, 0, shape)
    slices = tuple(slice(int(lo[axis]), int(hi[axis])) for axis in range(3))
    return {
        "center_voxel_kij": tuple(int(c) for c in center),
        "voxel_bounds_kij": tuple((int(lo[axis]), int(hi[axis])) for axis in range(3)),
        "t1_roi": geometry.volume[slices],
        "artifact_mask_roi": result.artifact_mask[slices],
        "in_bounds": in_bounds,
    }


def contacts_outside_volume(result: Any, positions: list[ContactPosition]) -> list[str]:
    """Contact labels whose assigned position's voxel index falls outside the T1 volume's own array bounds.

    Straightforward and always well-defined regardless of how the position
    was derived (unlike the length/registration metrics, this needs no
    reference table at all): a position that maps outside ``[0, shape)`` is
    definitely not a usable coordinate, independent of whether it is
    otherwise "correct".
    """
    geometry = result.t1_geometry
    shape = np.array(geometry.volume.shape)
    outside = []
    for position in positions:
        kij = geometry.patient_to_voxel(np.array(position.patient_xyz_mm))
        if not np.all((kij >= 0) & (kij < shape)):
            outside.append(position.contact_label)
    return outside
