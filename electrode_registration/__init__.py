"""Electrode-to-DICOM candidate registration — contact geometry, patient-space transform, and honest metrics.

See ``README.md`` in this package for what is and is not validated here:
in short, no CT and no per-contact fiducial file exist for this dataset, so
every spatial number this package produces is either (a) directly measured
from the DICOM/MRI geometry itself (patient-space transforms,
``FrameOfReferenceUID`` agreement, whether a position falls inside the
volume) or (b) an internal-consistency cross-check between independent but
approximate sources (the EDF montage's own channel naming, the PDF's
documented shaft length/contact count, and the MRI's own artifact-mask
geometry) — never a validated 3-D localization accuracy claim.
"""
from .contact_detection import ShaftCandidate, detect_shaft_candidates
from .frame_reference import FrameOfReferenceCheck, check_frame_of_reference
from .metrics import RegistrationMetrics, compare_montage_to_reference, compute_registration_metrics
from .reference_geometry import SEEG_HFOS_8_SHAFT_REFERENCE, ShaftReference, montage_shaft_contact_counts
from .registration import (
    ContactPosition, ShaftRegistration, assign_contact_positions, contacts_outside_volume,
    extract_contact_roi, register_shaft_candidates,
)

__all__ = [
    "ShaftCandidate", "detect_shaft_candidates",
    "FrameOfReferenceCheck", "check_frame_of_reference",
    "RegistrationMetrics", "compare_montage_to_reference", "compute_registration_metrics",
    "SEEG_HFOS_8_SHAFT_REFERENCE", "ShaftReference", "montage_shaft_contact_counts",
    "ContactPosition", "ShaftRegistration", "assign_contact_positions", "contacts_outside_volume",
    "extract_contact_roi", "register_shaft_candidates",
]
