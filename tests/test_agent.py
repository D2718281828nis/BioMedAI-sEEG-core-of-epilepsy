import numpy as np
import pytest

from extreme_event_agent import AgentConfig, ClinicalEvent, ExtremeEventAgent
from extreme_event_agent.edf_workflow import analyse_brain_process, plot_all_timeseries


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
