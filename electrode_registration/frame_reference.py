"""Confirm every DICOM series actually shares one physical coordinate frame before trusting a shared affine.

``multimodal_approach.dicom_geometry`` already resamples T2 onto the T1
grid using each series' own patient-space affine (``ImageOrientationPatient``/
``ImagePositionPatient``), which is only valid if both series were acquired
in the same scanner session against the same physical reference — exactly
what DICOM's ``FrameOfReferenceUID`` tag exists to assert. This module reads
that tag directly (not inferred from the affine) and checks it, rather than
assuming agreement the way the rest of the pipeline implicitly has been.

On ``dataset/MRI-with-electrodes/DCM``: T1 (``t1_mprage_tra_p2_iso``) and T2
(``t2_space_TR_p2_iso``) share the identical ``FrameOfReferenceUID``
(``1.3.12.2.1107.5.2.19.45833.1.20231011110841464.0.0.0``), confirmed by
reading both series' own tags directly — a real, checked fact, not an
assumption. ``SeriesInstanceUID``/``StudyInstanceUID`` differ between the
two series on this dataset (an anonymization artifact -- see
``dicom_viewer.viewer.group_series``'s own docstring for the same
observation about ``SeriesInstanceUID``), which is expected and irrelevant
to spatial validity; only ``FrameOfReferenceUID`` matters for whether a
patient-space coordinate computed on one series' grid is valid on another's
without an explicit registration step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pydicom

from multimodal_approach.dicom_geometry import list_series

__all__ = ["FrameOfReferenceCheck", "check_frame_of_reference"]


@dataclass(frozen=True)
class FrameOfReferenceCheck:
    """Result of comparing every series' own ``FrameOfReferenceUID`` tag."""

    frame_of_reference_uid_by_series: dict[str, str | None]
    all_match: bool
    shared_frame_of_reference_uid: str | None


def check_frame_of_reference(dicom_dir: str | Path) -> FrameOfReferenceCheck:
    """Read ``FrameOfReferenceUID`` from the first file of every series under ``dicom_dir``.

    The tag is a per-acquisition-session attribute, not per-slice, so
    reading it from each series' first file (``stop_before_pixels=True``,
    no image data loaded) is enough — unlike ``SeriesInstanceUID``, it is
    expected to be identical across every series from the same scanner
    session, and this function's whole job is to check that expectation
    rather than assume it.
    """
    series_map = list_series(dicom_dir)
    uids: dict[str, str | None] = {}
    for series_number, series in series_map.items():
        dataset = pydicom.dcmread(series.files[0], stop_before_pixels=True)
        uid = getattr(dataset, "FrameOfReferenceUID", None)
        uids[f"{series_number}:{series.description}"] = str(uid) if uid is not None else None

    present = [uid for uid in uids.values() if uid is not None]
    all_match = bool(present) and len(set(present)) == 1 and len(present) == len(uids)
    shared = present[0] if all_match else None
    return FrameOfReferenceCheck(frame_of_reference_uid_by_series=uids, all_match=all_match,
                                 shared_frame_of_reference_uid=shared)
