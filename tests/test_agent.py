import numpy as np
import pytest
from datetime import datetime, timezone
from pathlib import Path

from extreme_event_agent import AgentConfig, ClinicalEvent, ExtremeEventAgent
from extreme_event_agent import edf_workflow
from extreme_event_agent.edf_workflow import (
    _cluster_seizure_annotation,
    analyse_brain_process,
    apply_bipolar_montage,
    build_bipolar_montage,
    build_seizure_graph,
    clock_time_to_offset,
    compare_montages,
    describe_seizure_source,
    evaluate_message_passing,
    format_bipolar_montage,
    is_right_frontal,
    parse_contact_name,
    plot_all_timeseries,
    plot_message_passing,
    plot_message_passing_layouts,
    plot_message_passing_validation,
    plot_seizure_evolution,
    plot_seizure_graph,
    plot_seizure_graph_layouts,
    read_edf,
    read_edf_markers,
    run_edf,
    select_seizure_event,
    simulate_message_passing,
    summarize_montage_comparison,
)
from extreme_event_agent.models import DetectionReport, EdfRunResult, Event


def test_agent_finds_multichannel_extreme_event():
    rng = np.random.default_rng(7)
    sfreq = 100.0
    data = rng.normal(0, 0.15, (8, 3000))
    time = np.arange(400) / sfreq
    burst = 7 * np.sin(2 * np.pi * 18 * time)
    data[2:5, 1400:1800] += burst
    config = AgentConfig(window_seconds=1, step_seconds=.25, threshold_mad=5,
                         min_involved_channels=2)
    report = ExtremeEventAgent(config).run(data, sfreq, [f"C{i}" for i in range(8)])
    assert report.events
    assert report.events[0].start_seconds <= 14.5
    assert report.events[0].end_seconds >= 17.5
    assert set(report.events[0].involved_channels) >= {"C2", "C3", "C4"}
    assert [item["phase"] for item in report.audit_log][0] == "observe"


def test_agent_rejects_low_quality_data():
    data = np.zeros((2, 100))
    data[:, :30] = np.nan
    with pytest.raises(ValueError, match="Usable sample fraction"):
        ExtremeEventAgent().run(data, 10)


def test_brain_process_and_all_channel_plot(tmp_path):
    rng = np.random.default_rng(4)
    sfreq = 200.
    data = rng.normal(0, .05, (4, 2400))
    time = np.arange(400) / sfreq
    burst = 3 * np.sin(2 * np.pi * 35 * time)
    data[0, 1200:1600] += burst
    data[1, 1240:1640] += burst
    names = ["EEG PM3", "EEG CC8", "EEG L1", "EEG L2"]
    event = ClinicalEvent(6., duration_seconds=2.)
    process = analyse_brain_process(data, sfreq, names, event, baseline_seconds=5.)
    assert set(process.likely_initiators) == {"EEG PM3", "EEG CC8"}
    assert process.onset_latency_seconds["EEG PM3"] < process.onset_latency_seconds["EEG CC8"]
    pytest.importorskip("matplotlib")
    output = plot_all_timeseries(data, sfreq, names, tmp_path / "all.png", event)
    assert output.stat().st_size > 1000
    evolution = plot_seizure_evolution(data, sfreq, names, event, process,
                                       tmp_path / "evolution.png", baseline_seconds=5.)
    assert evolution.stat().st_size > 1000

    graph = build_seizure_graph(data, sfreq, names, event, process, baseline_seconds=5.)
    assert set(graph.nodes) == {"PEAK", "EEG PM3", "EEG CC8"}
    assert graph.nodes["EEG PM3"]["is_initiator"] and graph.nodes["EEG CC8"]["is_initiator"]
    assert graph.has_edge("PEAK", "EEG PM3") and graph.has_edge("PEAK", "EEG CC8")
    graph_figure = plot_seizure_graph(graph, tmp_path / "graph.png")
    assert graph_figure.stat().st_size > 1000

    layout_figures = plot_seizure_graph_layouts(graph, tmp_path, "test")
    assert set(layout_figures) == {"radial", "spring", "circular", "shell"}
    for figure in layout_figures.values():
        assert figure.stat().st_size > 1000

    channel_order, states = simulate_message_passing(graph, steps=4)
    assert states.shape == (5, len(channel_order))
    # A degree-normalized convex combination never leaves the seed's range.
    assert states.min() >= min(graph.nodes[n]["peak_z"] for n in channel_order) - 1e-6
    assert states.max() <= max(graph.nodes[n]["peak_z"] for n in channel_order) + 1e-6

    evaluation = evaluate_message_passing(data, sfreq, names, event, channel_order, states,
                                          baseline_seconds=5.)
    assert len(evaluation["elapsed_seconds"]) == len(evaluation["correlation"]) == 5
    assert evaluation["elapsed_seconds"][0] == 0.0

    mp_figure = plot_message_passing(graph, channel_order, states, tmp_path / "mp.png")
    assert mp_figure.stat().st_size > 1000
    mp_validation_figure = plot_message_passing_validation(evaluation, tmp_path / "mp_validation.png")
    assert mp_validation_figure.stat().st_size > 1000

    mp_layout_figures = plot_message_passing_layouts(graph, channel_order, states, tmp_path, "test")
    assert set(mp_layout_figures) == {"radial", "spring", "circular", "shell"}
    for figure in mp_layout_figures.values():
        assert figure.stat().st_size > 1000

    source_text = describe_seizure_source(process)
    assert "EEG PM3" in source_text and "EEG CC8" in source_text
    assert f"{event.time_seconds:.3f}" in source_text


def test_describe_seizure_source_reports_no_source_when_process_found_nothing():
    from extreme_event_agent.models import BrainProcess
    empty_process = BrainProcess(event_time_seconds=42.0, channel_band_scores={},
                                 onset_latency_seconds={}, likely_initiators=(), later_recruited=())
    text = describe_seizure_source(empty_process)
    assert "No channel crossed" in text
    assert "42.000" in text


def test_is_right_frontal_matches_bipolar_pairs_by_either_endpoint():
    # Regression test: the old RIGHT_FRONTAL regex only ever checked a pair
    # label's *first* contact number, so "PM2-3" and "CC7-8" (whose second
    # endpoint sits inside the right-frontal zone) were silently misread as
    # not right-frontal. Either endpoint must count.
    assert is_right_frontal("EEG PM3") and not is_right_frontal("EEG PM1")
    assert is_right_frontal("EEG CC9") and not is_right_frontal("EEG CC5")
    assert is_right_frontal("PM3-4") and is_right_frontal("PM2-3") and is_right_frontal("PM7-8")
    assert is_right_frontal("CC7-8") and is_right_frontal("CC9-10")
    assert not is_right_frontal("PM1-2") and not is_right_frontal("CC1-2")
    # Primed (contralateral) shafts never count, single contact or pair alike.
    assert not is_right_frontal("EEG CC'4") and not is_right_frontal("EEG PM'3")


def _write_synthetic_edf(path, sfreq=100.0, n_samples=1000, marker_channel=True):
    import mne
    n_seconds = n_samples / sfreq
    signal = np.stack([np.linspace(0, n_seconds, n_samples, endpoint=False),
                       np.linspace(0, 2 * n_seconds, n_samples, endpoint=False)]) * 1e-6
    names, types = ["EEG A1", "EEG A2"], ["eeg", "eeg"]
    if marker_channel:
        signal = np.concatenate([signal, np.zeros((1, n_samples))], axis=0)
        names.append("MKR1+")
        types.append("eeg")
    info = mne.create_info(names, sfreq, ch_types=types)
    raw = mne.io.RawArray(signal, info, verbose="ERROR")
    raw.export(path, fmt="edf", verbose="ERROR")
    return n_seconds


def test_read_edf_crop_end_seconds_truncates_data(tmp_path):
    edf_path = tmp_path / "synthetic.edf"
    sfreq, n_samples = 100.0, 1000
    _write_synthetic_edf(edf_path, sfreq=sfreq, n_samples=n_samples)

    full_data, full_sfreq, full_names = read_edf(edf_path)
    cropped_data, cropped_sfreq, cropped_names = read_edf(edf_path, crop_end_seconds=5.0)

    assert full_names == cropped_names == ["EEG A1", "EEG A2"]
    assert full_sfreq == cropped_sfreq == sfreq
    assert full_data.shape[1] == n_samples
    assert cropped_data.shape[1] == pytest.approx(5.0 * sfreq, abs=1)
    assert cropped_data.shape[1] < full_data.shape[1]
    np.testing.assert_allclose(cropped_data, full_data[:, :cropped_data.shape[1]], atol=2e-8)


def test_read_edf_crop_end_seconds_past_recording_end_is_a_no_op(tmp_path):
    edf_path = tmp_path / "synthetic.edf"
    n_seconds = _write_synthetic_edf(edf_path)
    full_data, _, _ = read_edf(edf_path)
    generously_cropped_data, _, _ = read_edf(edf_path, crop_end_seconds=n_seconds * 100)
    assert generously_cropped_data.shape == full_data.shape


def test_read_edf_markers_crop_end_seconds_truncates_data(tmp_path):
    edf_path = tmp_path / "synthetic.edf"
    sfreq = 100.0
    _write_synthetic_edf(edf_path, sfreq=sfreq, n_samples=1000)
    full_markers, _, marker_names = read_edf_markers(edf_path)
    cropped_markers, _, cropped_marker_names = read_edf_markers(edf_path, crop_end_seconds=5.0)
    assert marker_names == cropped_marker_names == ["MKR1+"]
    assert cropped_markers.shape[1] == pytest.approx(5.0 * sfreq, abs=1)
    assert cropped_markers.shape[1] < full_markers.shape[1]


def test_run_edf_rejects_unknown_montage_reference():
    with pytest.raises(ValueError, match="montage_reference"):
        run_edf("does-not-matter.edf", "does-not-matter-output", montage_reference="average")


def test_compare_montages_passes_one_stem_level_to_run_edf(monkeypatch, tmp_path):
    # Regression test: compare_montages appends "<stem>/<reference>" itself,
    # so a caller must pass the bare output directory, not one already
    # suffixed with the stem — passing the suffixed one doubled the stem in
    # the path (output/stem/stem/reference) and made run_edf fail to find
    # its own just-created directory.
    seen_output_dirs = []

    def fake_run_edf(path, output_dir, event=None, montage_reference="none", crop_end_seconds=None):
        seen_output_dirs.append(Path(output_dir))
        return montage_reference

    monkeypatch.setattr(edf_workflow, "run_edf", fake_run_edf)
    compare_montages(tmp_path / "recording.edf", tmp_path / "out", montage_references=("none", "bipolar"))
    assert seen_output_dirs == [tmp_path / "out" / "recording" / "none",
                                tmp_path / "out" / "recording" / "bipolar"]


def test_summarize_montage_comparison_reduces_two_results_to_comparable_rows():
    # Built directly from EdfRunResult/DetectionReport rather than a real EDF,
    # matching this file's existing style of testing pipeline stages with
    # synthetic data instead of file I/O; run_edf/compare_montages are thin
    # orchestration around functions already covered elsewhere in this file.
    def make_result(reference, n_events, n_involved, correlations):
        report = DetectionReport(events=[None] * n_events, sampling_frequency_hz=100.,
                                 channel_names=tuple(f"C{i}" for i in range(4)), threshold=6.)
        process = None
        if n_involved:
            from extreme_event_agent.models import BrainProcess
            process = BrainProcess(0., {}, {f"C{i}": 0. for i in range(n_involved)},
                                   likely_initiators=("C0",), later_recruited=())
        return EdfRunResult(report, process, montage={}, montage_reference=reference,
                            montage_file="montage.txt", overview_figure="overview.png",
                            evolution_figure=None, graph_figures={}, graph_graphml=None,
                            message_passing_figure=None, message_passing_validation_figure=None,
                            message_passing_evaluation={"elapsed_seconds": list(range(len(correlations))),
                                                        "correlation": correlations},
                            annotated_event=None, detected_event=None)

    results = {
        "none": make_result("none", n_events=2, n_involved=90, correlations=[0.6, 0.2, float("nan")]),
        "bipolar": make_result("bipolar", n_events=1, n_involved=12, correlations=[0.9, 0.8, 0.7]),
    }
    rows = summarize_montage_comparison(results)
    by_reference = {row["montage_reference"]: row for row in rows}
    assert by_reference["none"]["n_detected_candidates"] == 2
    assert by_reference["none"]["n_involved_channels"] == 90
    assert by_reference["none"]["message_passing_best_correlation"] == pytest.approx(0.6)
    assert by_reference["bipolar"]["n_involved_channels"] == 12
    assert by_reference["bipolar"]["message_passing_best_correlation"] == pytest.approx(0.9)
    assert by_reference["bipolar"]["message_passing_mean_correlation"] == pytest.approx(0.8)


def test_bipolar_montage_pairs_adjacent_contacts_per_shaft():
    # Mirrors this dataset's naming: two shafts ("PM" and its distinct
    # contralateral counterpart "PM'"), plus a non-matching marker channel
    # and a gap in PM's numbering (no contact 3) that must still pair
    # across the gap rather than silently dropping a neighbor relationship.
    names = ["EEG PM1", "EEG PM2", "EEG PM4", "EEG PM'1", "EEG PM'2", "MKR1+"]
    assert parse_contact_name("EEG PM3") == ("PM", 3)
    assert parse_contact_name("EEG CC'4") == ("CC'", 4)
    assert parse_contact_name("MKR1+") is None

    montage = build_bipolar_montage(names)
    assert montage["PM"] == [("EEG PM1", "EEG PM2"), ("EEG PM2", "EEG PM4")]
    assert montage["PM'"] == [("EEG PM'1", "EEG PM'2")]
    assert "MKR1+" not in format_bipolar_montage(montage)

    rendered = format_bipolar_montage(montage)
    assert "PM:\n  1-2\n  2-4" in rendered
    assert "PM':\n  1-2" in rendered

    data = np.arange(5 * 4).reshape(5, 4).astype(float)
    all_names = ["EEG PM1", "EEG PM2", "EEG PM4", "EEG PM'1", "EEG PM'2"]
    derived_data, derived_names = apply_bipolar_montage(data, all_names, montage)
    assert derived_names == ["PM1-2", "PM2-4", "PM'1-2"]
    np.testing.assert_array_equal(derived_data[0], data[0] - data[1])


def test_select_seizure_event_prefers_spread_over_score():
    # A brief, few-channel spike that scores higher than a widely spread,
    # longer-lasting candidate must still lose: spread and duration outrank
    # raw score, exactly to keep sharp interictal spikes from being mistaken
    # for the seizure.
    spike = Event(start_seconds=10., end_seconds=10.3, peak_seconds=10.1, score=40.,
                 confidence=1., involved_channels=("A",), evidence={})
    seizure = Event(start_seconds=6000., end_seconds=6030., peak_seconds=6012.,
                    score=15., confidence=.9,
                    involved_channels=tuple(f"C{i}" for i in range(20)), evidence={})
    detected = select_seizure_event([spike, seizure])
    assert detected.time_seconds == 6012.
    assert detected.involved_channel_count == 20
    assert select_seizure_event([]) is None


def test_cluster_seizure_annotation_matches_real_edf_markers():
    # Mirrors the three annotations actually embedded in sEEG-HFOs-8.edf
    # (decoded from Windows-1251), plus an unrelated distant annotation that
    # must not be folded in.
    onsets = [10392.734, 10396.445, 10399.469, 500.0]
    descriptions = ["где тут начало?", "приступ + БТКП", "клиника", "recording start"]
    event = _cluster_seizure_annotation(onsets, descriptions)
    assert event.time_seconds == 10396.445
    assert event.label == "приступ + БТКП"
    assert event.duration_seconds == pytest.approx(10399.469 - 10392.734)
    assert len(event.annotations) == 3


def test_cluster_seizure_annotation_returns_none_without_keyword():
    assert _cluster_seizure_annotation([1.0, 2.0], ["lights off", "impedance check"]) is None


def test_clock_time_to_offset_and_midnight_rollover():
    start = datetime(2026, 1, 1, 17, 27, 11, tzinfo=timezone.utc)
    assert clock_time_to_offset("17:27:14", start, 20.) == 3.
    near_midnight = datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc)
    assert clock_time_to_offset("00:00:01.500", near_midnight, 5.) == 2.5
    with pytest.raises(ValueError, match="outside"):
        clock_time_to_offset("17:28:00", start, 20.)
