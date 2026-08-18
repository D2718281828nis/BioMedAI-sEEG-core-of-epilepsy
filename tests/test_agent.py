import numpy as np
import pytest
from datetime import datetime, timezone

from extreme_event_agent import AgentConfig, ClinicalEvent, ExtremeEventAgent
from extreme_event_agent.edf_workflow import (
    _cluster_seizure_annotation,
    analyse_brain_process,
    build_seizure_graph,
    clock_time_to_offset,
    evaluate_message_passing,
    plot_all_timeseries,
    plot_message_passing,
    plot_message_passing_validation,
    plot_seizure_evolution,
    plot_seizure_graph,
    plot_seizure_graph_layouts,
    select_seizure_event,
    simulate_message_passing,
)
from extreme_event_agent.models import Event


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
