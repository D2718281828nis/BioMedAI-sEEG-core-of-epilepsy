import numpy as np
import pytest

from extreme_event_agent import AgentConfig, ExtremeEventAgent


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
