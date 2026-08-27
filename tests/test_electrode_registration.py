import numpy as np
import pytest

from electrode_registration.contact_detection import ShaftCandidate, detect_shaft_candidates
from electrode_registration.metrics import compare_montage_to_reference, compute_registration_metrics
from electrode_registration.reference_geometry import ShaftReference, montage_shaft_contact_counts
from electrode_registration.registration import (
    assign_contact_positions, contacts_outside_volume, extract_contact_roi, register_shaft_candidates,
)
from multimodal_approach.dicom_geometry import VolumeGeometry


class _StubResult:
    def __init__(self, t1_geometry, artifact_mask, midline_x_mm):
        self.t1_geometry = t1_geometry
        self.artifact_mask = artifact_mask
        self.midline_x_mm = midline_x_mm


def _synthetic_geometry(shape=(40, 60, 60)):
    volume = np.zeros(shape, dtype=np.float32)
    return VolumeGeometry(
        volume=volume, origin=np.zeros(3), d_slice=np.array([0.0, 0.0, 1.0]),
        d_row=np.array([0.0, 1.0, 0.0]), d_col=np.array([1.0, 0.0, 0.0]), series=None,
    )


def test_montage_shaft_contact_counts_matches_bipolar_pairs_plus_one():
    names = ["EEG PM1", "EEG PM2", "EEG PM3", "EEG CC1", "EEG CC2", "MKR1+"]
    counts = montage_shaft_contact_counts(names)
    assert counts == {"PM": 3, "CC": 2}


def test_compare_montage_to_reference_flags_disagreement():
    reference = {"PM": ShaftReference("PM", "right", 38.0, 8, "red")}
    names = [f"EEG PM{i}" for i in range(1, 5)]  # only 4 contacts, not the documented 8
    rows = compare_montage_to_reference(names, reference)
    assert len(rows) == 1
    assert rows[0]["shaft_name"] == "PM"
    assert rows[0]["edf_montage_contact_count"] == 4
    assert rows[0]["pdf_contact_count"] == 8
    assert rows[0]["agree"] is False


def test_detect_shaft_candidates_finds_separated_lines_on_correct_hemispheres():
    geometry = _synthetic_geometry()
    mask = np.zeros(geometry.volume.shape, dtype=bool)
    # Right-hemisphere line (x = j < midline): j=10.
    for i in range(5, 25):
        mask[10, i, 10] = True
    # Left-hemisphere line (x = j > midline): j=50, far away from the right line.
    for i in range(5, 25):
        mask[10, i, 50] = True
    result = _StubResult(geometry, mask, midline_x_mm=30.0)

    candidates = detect_shaft_candidates(result, max_link_mm=3.0, min_blob_voxels=1, min_candidate_voxels=5)
    assert len(candidates) == 2
    hemispheres = {c.hemisphere for c in candidates}
    assert hemispheres == {"right", "left"}
    for candidate in candidates:
        assert candidate.length_mm == pytest.approx(19.0, abs=1.0)
        assert candidate.voxel_count == 20


def test_detect_shaft_candidates_drops_midline_crossing_blobs():
    geometry = _synthetic_geometry()
    mask = np.zeros(geometry.volume.shape, dtype=bool)
    # A blob straddling the midline (x=j from 25 to 35, midline at 30) should be dropped.
    for j in range(25, 36):
        mask[10, 10, j] = True
    result = _StubResult(geometry, mask, midline_x_mm=30.0)

    candidates = detect_shaft_candidates(result, min_blob_voxels=1, min_candidate_voxels=1)
    assert candidates == []


def _candidate(candidate_id, hemisphere, length_mm, voxel_count=100):
    return ShaftCandidate(
        candidate_id=candidate_id, hemisphere=hemisphere, blob_count=1, voxel_count=voxel_count,
        centroid_patient_xyz_mm=(0.0, 0.0, 0.0),
        endpoint_a_patient_xyz_mm=(0.0, 0.0, 0.0), endpoint_b_patient_xyz_mm=(0.0, length_mm, 0.0),
        length_mm=length_mm, direction_unit=(0.0, 1.0, 0.0))


def test_register_shaft_candidates_matches_by_length_hungarian_optimal():
    reference = {
        "A": ShaftReference("A", "right", 10.0, 4, "x"),
        "B": ShaftReference("B", "right", 50.0, 4, "x"),
    }
    # Deliberately reversed order/closeness so a greedy nearest-first match
    # would get this wrong (cand_0 is closer to A's length than cand_1 is,
    # but the *optimal* global assignment still pairs cand_0 with B and
    # cand_1 with A once both costs are considered together)... construct
    # a case where greedy first-come-first-served differs from optimal:
    # cand order: cand_0 (length=48) then cand_1 (length=12).
    candidates = [_candidate("cand_0", "right", 48.0), _candidate("cand_1", "right", 12.0)]

    registrations = register_shaft_candidates(candidates, reference)
    by_shaft = {r.shaft_name: r for r in registrations}
    assert by_shaft["A"].matched and by_shaft["A"].candidate.candidate_id == "cand_1"
    assert by_shaft["B"].matched and by_shaft["B"].candidate.candidate_id == "cand_0"
    assert by_shaft["A"].length_error_mm == pytest.approx(2.0)
    assert by_shaft["B"].length_error_mm == pytest.approx(2.0)


def test_register_shaft_candidates_leaves_shaft_unmatched_when_no_candidate():
    reference = {"A": ShaftReference("A", "right", 10.0, 4, "x"),
                "B": ShaftReference("B", "right", 50.0, 4, "x")}
    candidates = [_candidate("cand_0", "right", 48.0)]  # only one candidate for two shafts

    registrations = register_shaft_candidates(candidates, reference)
    by_shaft = {r.shaft_name: r for r in registrations}
    assert by_shaft["B"].matched
    assert not by_shaft["A"].matched
    assert by_shaft["A"].candidate is None
    assert by_shaft["A"].length_error_mm is None


def test_assign_contact_positions_only_for_matched_shafts_and_even_spacing():
    reference = {"A": ShaftReference("A", "right", 10.0, 3, "x")}
    matched = register_shaft_candidates([_candidate("cand_0", "right", 10.0)], reference)
    positions = assign_contact_positions(matched)

    assert [p.contact_label for p in positions] == ["A_contact_1", "A_contact_2", "A_contact_3"]
    ys = [p.patient_xyz_mm[1] for p in positions]
    assert ys[0] == pytest.approx(0.0)
    assert ys[1] == pytest.approx(5.0)
    assert ys[2] == pytest.approx(10.0)


def test_extract_contact_roi_and_contacts_outside_volume():
    geometry = _synthetic_geometry(shape=(40, 60, 60))
    result = _StubResult(geometry, np.zeros(geometry.volume.shape, dtype=bool), midline_x_mm=30.0)

    inside = type("P", (), {"contact_label": "in", "patient_xyz_mm": (10.0, 10.0, 10.0)})()
    outside = type("P", (), {"contact_label": "out", "patient_xyz_mm": (1000.0, 1000.0, 1000.0)})()

    roi = extract_contact_roi(result, inside.patient_xyz_mm, radius_mm=5.0)
    assert roi["in_bounds"] is True
    assert roi["t1_roi"].size > 0

    outside_roi = extract_contact_roi(result, outside.patient_xyz_mm, radius_mm=5.0)
    assert outside_roi["in_bounds"] is False

    labels = contacts_outside_volume(result, [inside, outside])
    assert labels == ["out"]


def test_compute_registration_metrics_recall_precision_coverage():
    reference = {
        "A": ShaftReference("A", "right", 10.0, 4, "x"),   # will get a plausible match
        "B": ShaftReference("B", "right", 50.0, 4, "x"),   # will get an implausible match
        "C": ShaftReference("C", "left", 20.0, 2, "x"),    # will get no candidate at all
    }
    candidates = [
        _candidate("cand_0", "right", 10.5),   # close to A -> plausible
        _candidate("cand_1", "right", 90.0),   # far from B (or A) -> implausible whichever it lands on
    ]
    registrations = register_shaft_candidates(candidates, reference)
    positions = assign_contact_positions(registrations)
    geometry = _synthetic_geometry()
    result = _StubResult(geometry, np.zeros(geometry.volume.shape, dtype=bool), midline_x_mm=30.0)

    metrics = compute_registration_metrics(registrations, positions, ["EEG A1", "EEG A2", "EEG A3", "EEG A4",
                                                                       "EEG B1", "EEG B2", "EEG B3", "EEG B4",
                                                                       "EEG C1", "EEG C2"],
                                           result, plausible_relative_error=0.5)
    assert metrics.total_shafts == 3
    assert metrics.matched_shafts == 2   # A and B got candidates; C did not
    assert metrics.plausible_shafts == 1  # only A is within 50% relative error
    assert metrics.recall == pytest.approx(1 / 3)
    assert metrics.precision == pytest.approx(1 / 2)
    assert metrics.coverage == pytest.approx(4 / 10)  # only A's 4 contacts covered, out of 10 total
