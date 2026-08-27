import numpy as np
import pytest

from extreme_event_agent.models import AnnotatedEvent, BrainProcess
from extreme_event_agent.verification import (
    classify_temporal_accuracy, contact_overlap, lateralization_index, verify_against_annotation,
    COARSE_SECONDS, PRECISE_SECONDS, WINDOW_SECONDS,
)


def _process(**overrides):
    defaults = dict(
        event_time_seconds=100.0,
        channel_band_scores={"EEG PM3": 8.0, "EEG PA9": 7.0, "EEG PA'3": 6.0},
        onset_latency_seconds={"EEG PM3": 0.03, "EEG PA9": 0.03, "EEG PA'3": 2.0},
        likely_initiators=("EEG PM3",),
        later_recruited=("EEG PA'3",),
        earliest_contacts=("EEG PM3", "EEG PA9"),
        earliest_latency_seconds=0.03,
        prior_matched=("EEG PM3",),
        prior_source="clinical context",
        initiators_constrained_by_prior=False,
        prior_fraction_among_earliest=0.5,
        hemisphere_of_earliest="mixed",
    )
    defaults.update(overrides)
    return BrainProcess(**defaults)


def _annotation(time_seconds: float) -> AnnotatedEvent:
    return AnnotatedEvent(time_seconds=time_seconds, label="приступ + БТКП", duration_seconds=2.,
                          annotations=((time_seconds, "приступ + БТКП"),))


def test_classify_temporal_accuracy_bands():
    assert classify_temporal_accuracy(0.5) == "precise"
    assert classify_temporal_accuracy(-0.5) == "precise"
    assert classify_temporal_accuracy(PRECISE_SECONDS + 1e-9) == "coarse"
    assert classify_temporal_accuracy(COARSE_SECONDS) == "coarse"
    assert classify_temporal_accuracy(COARSE_SECONDS + 1e-9) == "window"
    assert classify_temporal_accuracy(WINDOW_SECONDS) == "window"
    assert classify_temporal_accuracy(WINDOW_SECONDS + 1e-9) == "miss"


def test_temporal_accuracy_preserves_sign():
    # Method fires at event_time_seconds + earliest_latency_seconds = 100.5s.
    process = _process(event_time_seconds=100.0, earliest_latency_seconds=0.5)

    # Annotation later than the method -> method is early -> negative delta.
    report_early = verify_against_annotation(_annotation(105.5), process)
    latency_early = report_early.annotation_conditioned_recruitment_latency[0]
    assert latency_early.delta_seconds == pytest.approx(100.5 - 105.5)
    assert latency_early.delta_seconds < 0

    # Annotation earlier than the method -> method lags -> positive delta.
    report_late = verify_against_annotation(_annotation(95.5), process)
    latency_late = report_late.annotation_conditioned_recruitment_latency[0]
    assert latency_late.delta_seconds == pytest.approx(100.5 - 95.5)
    assert latency_late.delta_seconds > 0

    # Same magnitude (5s either side of the method's own time), opposite
    # sign -> distinguishable, not collapsed by abs().
    assert latency_early.delta_seconds == pytest.approx(-latency_late.delta_seconds)


def test_blind_event_detection_is_separate_from_recruitment_latency():
    # blind_event_detection is populated only when a blind time is actually
    # given -- it must never be silently backfilled from the
    # annotation-conditioned recruitment latency, since the whole point of
    # the split is that the two are not interchangeable.
    process = _process(event_time_seconds=100.0, earliest_latency_seconds=0.5)

    report_no_blind = verify_against_annotation(_annotation(100.03), process)
    assert report_no_blind.blind_event_detection == []
    assert len(report_no_blind.annotation_conditioned_recruitment_latency) == 1
    assert report_no_blind.annotation_conditioned_recruitment_latency[0].method == "recruitment_latency"

    report_with_blind = verify_against_annotation(_annotation(100.03), process,
                                                   blind_event_time_seconds=141.6)
    assert len(report_with_blind.blind_event_detection) == 1
    blind_entry = report_with_blind.blind_event_detection[0]
    assert blind_entry.method == "blind_event_detection"
    assert blind_entry.method_time_seconds == pytest.approx(141.6)
    assert blind_entry.delta_seconds == pytest.approx(141.6 - 100.03)
    # The recruitment-latency entry is unaffected by whether a blind time
    # was also supplied.
    assert (report_with_blind.annotation_conditioned_recruitment_latency
           == report_no_blind.annotation_conditioned_recruitment_latency)


def test_lateralization_index_bounds_and_symmetry():
    assert lateralization_index(5.0, 5.0) == pytest.approx(0.0)
    assert lateralization_index(0.0, 0.0) == pytest.approx(0.0)
    assert lateralization_index(10.0, 0.0) == pytest.approx(1.0)
    assert lateralization_index(0.0, 10.0) == pytest.approx(-1.0)

    rng = np.random.default_rng(0)
    for _ in range(200):
        v_right, v_left = rng.uniform(0, 100, 2)
        index = lateralization_index(float(v_right), float(v_left))
        assert -1.0 <= index <= 1.0
        # Swapping right/left negates the index.
        assert lateralization_index(float(v_left), float(v_right)) == pytest.approx(-index)


def test_verification_report_records_context_fields():
    process = _process()

    report = verify_against_annotation(
        _annotation(100.03), process, crop_applied=True, crop_end_seconds=10550.0,
        channel_selection="balanced", masking_method="tissue_brain_extract(bone_percentile=45.0)")
    assert report.crop_applied is True
    assert report.crop_end_seconds == 10550.0
    assert report.channel_selection == "balanced"
    assert report.masking_method == "tissue_brain_extract(bone_percentile=45.0)"
    assert report.prior_used == "clinical context"

    # Fields are always present, defaulting honestly rather than being omitted.
    bare_process = _process(prior_source="")
    bare_report = verify_against_annotation(_annotation(100.03), bare_process)
    assert bare_report.crop_applied is False
    assert bare_report.crop_end_seconds is None
    assert bare_report.channel_selection is None
    assert bare_report.masking_method is None
    assert bare_report.prior_used is None
    assert bare_report.reservoir_arbitration_valid is None


def test_contact_overlap_none_without_prior():
    assert contact_overlap(_process(prior_matched=())) is None
    overlap = contact_overlap(_process())
    assert overlap.precision == pytest.approx(0.5)   # 1 of 2 earliest_contacts is prior_matched
    assert overlap.recall == pytest.approx(1.0)       # the 1 prior_matched contact is covered
    assert overlap.jaccard == pytest.approx(1 / 2)
