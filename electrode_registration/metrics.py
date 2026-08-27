"""Contact-level spatial metrics — all defined as internal-consistency checks, never accuracy against verified ground truth.

`dataset/истинное положение.pdf` has no numeric coordinates (see
``reference_geometry.py``'s module docstring), and this dataset has no CT
or manually placed fiducial file either — so "distance error",
"precision/recall", and "coverage" cannot honestly mean what they mean in a
validated electrode-localization pipeline (e.g. iELVis/GARDEL/SEEGA
checked against a manually-digitized or CT-derived truth). Every metric
here instead cross-checks three genuinely independent sources this
repository/dataset actually has: the EDF channel-naming montage (exact
contact counts, from the recording itself), the PDF's documented
shaft length/contact count (approximate, hand-drawn), and the MRI's own
artifact-mask-derived candidate geometry (data-derived, unconstrained). See
this package's ``README.md``, "Honest limits", before reporting any number
from here as if it were localization accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .reference_geometry import SEEG_HFOS_8_SHAFT_REFERENCE, montage_shaft_contact_counts
from .registration import ContactPosition, ShaftRegistration, contacts_outside_volume

__all__ = ["RegistrationMetrics", "compute_registration_metrics", "compare_montage_to_reference"]

DEFAULT_PLAUSIBLE_RELATIVE_ERROR = 0.5


def compare_montage_to_reference(montage_names: list[str],
                                 reference: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Per-shaft contact-count agreement between the EDF montage (exact) and the PDF table (approximate).

    This is the one check in this package that needs no MRI, no artifact
    mask, and no candidate detection at all — pure metadata cross-check.
    On ``sEEG-HFOs-8.edf``, every one of the 12 documented shafts agrees
    exactly (0 discrepancies): the montage's own bipolar-pair counts
    (``+1`` contact per pair) match the PDF's contact-count column for
    every shaft, right and left — real, checked evidence that the PDF and
    this specific recording describe the same implant, even though neither
    supplies a 3-D coordinate.
    """
    reference = reference or SEEG_HFOS_8_SHAFT_REFERENCE
    montage_counts = montage_shaft_contact_counts(montage_names)
    rows = []
    for name, ref in reference.items():
        edf_count = montage_counts.get(name)
        rows.append({
            "shaft_name": name, "hemisphere": ref.hemisphere,
            "pdf_contact_count": ref.contact_count, "edf_montage_contact_count": edf_count,
            "agree": edf_count == ref.contact_count,
        })
    return rows


@dataclass
class RegistrationMetrics:
    """No field here is validated against a numeric 3-D ground truth — see this package's README.

    ``recall`` — fraction of documented shafts (12 on this dataset) whose
    matched candidate's length falls within ``plausible_relative_error`` of
    that shaft's documented length.
    ``precision`` — fraction of *matched* registrations that are
    length-plausible (a matched-but-implausible pairing "used up" a shaft
    slot without actually resembling it).
    ``coverage`` — fraction of the EDF montage's own total contact count
    (exact) that ended up with *some* assigned candidate position, however
    approximate, versus none at all.
    ``contacts_outside_volume`` — the one metric here needing no reference
    table and no plausibility threshold: contact labels whose assigned
    position maps outside the T1 volume's own bounds.
    """

    plausible_relative_error: float
    total_shafts: int
    matched_shafts: int
    plausible_shafts: int
    recall: float
    precision: float
    coverage: float
    mean_length_error_mm: float | None
    median_length_relative_error: float | None
    contacts_outside_volume: list[str]
    per_shaft: list[dict[str, Any]]


def compute_registration_metrics(registrations: list[ShaftRegistration], positions: list[ContactPosition],
                                 montage_names: list[str], result: Any,
                                 plausible_relative_error: float = DEFAULT_PLAUSIBLE_RELATIVE_ERROR
                                 ) -> RegistrationMetrics:
    """Reduce ``register_shaft_candidates``'s output to recall/precision/coverage, all self-consistency based.

    ``plausible_relative_error`` (default 50%) is the tolerance a matched
    candidate's length must fall within, relative to its shaft's documented
    length, to count as "plausible". Deliberately loose, not tuned finer:
    ``ShaftCandidate.length_mm`` is itself only as good as the artifact
    mask's own fragmentation (see ``contact_detection.py``'s module
    docstring — 70-98 candidates detected against 12 documented shafts), so
    a genuinely correct pairing is not expected to match a hand-measured
    shaft length closely even when the assignment is right.
    """
    matched = [r for r in registrations if r.matched]
    plausible = [r for r in matched if r.length_relative_error is not None
                and r.length_relative_error <= plausible_relative_error]

    total_shafts = len(registrations)
    recall = len(plausible) / total_shafts if total_shafts else 0.0
    precision = len(plausible) / len(matched) if matched else 0.0

    montage_counts = montage_shaft_contact_counts(montage_names)
    total_montage_contacts = sum(montage_counts.values())
    covered_contacts = sum(r.reference.contact_count for r in plausible)
    coverage = covered_contacts / total_montage_contacts if total_montage_contacts else 0.0

    length_errors = [r.length_error_mm for r in plausible if r.length_error_mm is not None]
    relative_errors = [r.length_relative_error for r in plausible if r.length_relative_error is not None]

    plausible_names = {r.shaft_name for r in plausible}
    per_shaft = [{
        "shaft_name": r.shaft_name, "hemisphere": r.hemisphere, "matched": r.matched,
        "plausible": r.shaft_name in plausible_names, "length_error_mm": r.length_error_mm,
        "length_relative_error": r.length_relative_error,
        "candidate_id": r.candidate.candidate_id if r.candidate else None,
        "candidate_voxel_count": r.candidate.voxel_count if r.candidate else None,
    } for r in registrations]

    return RegistrationMetrics(
        plausible_relative_error=plausible_relative_error, total_shafts=total_shafts,
        matched_shafts=len(matched), plausible_shafts=len(plausible), recall=recall, precision=precision,
        coverage=coverage, mean_length_error_mm=float(np.mean(length_errors)) if length_errors else None,
        median_length_relative_error=float(np.median(relative_errors)) if relative_errors else None,
        contacts_outside_volume=contacts_outside_volume(result, positions), per_shaft=per_shaft)
