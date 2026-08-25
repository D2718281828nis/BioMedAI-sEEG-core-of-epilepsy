import numpy as np
import pytest

from extreme_event_agent.edf_workflow import analyse_brain_process, build_seizure_graph
from extreme_event_agent.models import ClinicalEvent
from object_model.graph import build_object_model_graph


class _StubWindow:
    def __init__(self, output_names):
        self.output_names = output_names


class _StubReservoirEvaluation:
    """Duck-types just enough of model.plant.ReservoirEvaluation for build_object_model_graph."""

    def __init__(self, output_names, onset, peak):
        self.window = _StubWindow(output_names)
        self.per_channel_onset_seconds = onset
        self.per_channel_peak_score = peak


def test_object_model_graph_carries_three_layers(tmp_path):
    rng = np.random.default_rng(4)
    sfreq = 200.
    data = rng.normal(0, .05, (2, 2400))
    time = np.arange(400) / sfreq
    burst = 3 * np.sin(2 * np.pi * 35 * time)
    data[0, 1200:1600] += burst  # EEG PM3 (right)
    data[1, 1200:1600] += burst  # EEG PM'3 (left)
    names = ["EEG PM3", "EEG PM'3"]
    event = ClinicalEvent(6., duration_seconds=2.)
    process = analyse_brain_process(data, sfreq, names, event, baseline_seconds=5.)
    graph = build_seizure_graph(data, sfreq, names, event, process, baseline_seconds=5.)

    hemisphere_summary = {
        "right_hemisphere": {"mean_abs_anomaly": 0.1, "max_abs_anomaly": 0.5},
        "left_hemisphere": {"mean_abs_anomaly": 0.2, "max_abs_anomaly": 0.6},
    }
    reservoir_evaluation = _StubReservoirEvaluation(
        output_names=["EEG PM3"], onset={"EEG PM3": 0.03}, peak={"EEG PM3": 12.5})

    build_object_model_graph(graph, hemisphere_summary=hemisphere_summary,
                             reservoir_evaluation=reservoir_evaluation)

    # Dynamic (EDF) layer: beta_gamma_peak is an alias for the already-present peak_z.
    assert graph.nodes["EEG PM3"]["beta_gamma_peak"] == graph.nodes["EEG PM3"]["peak_z"]
    # Structural (DICOM) layer: every channel node gets it, keyed by its own hemisphere.
    assert graph.nodes["EEG PM3"]["hemisphere_anomaly_mean"] == pytest.approx(0.1)
    assert graph.nodes["EEG PM3"]["hemisphere_anomaly_max"] == pytest.approx(0.5)
    assert graph.nodes["EEG PM'3"]["hemisphere_anomaly_mean"] == pytest.approx(0.2)
    # Model (reservoir) layer: only the node that is actually a reservoir output channel.
    assert graph.nodes["EEG PM3"]["residual_onset_seconds"] == pytest.approx(0.03)
    assert graph.nodes["EEG PM3"]["residual_peak_score"] == pytest.approx(12.5)
    assert "residual_onset_seconds" not in graph.nodes["EEG PM'3"]
    assert "residual_peak_score" not in graph.nodes["EEG PM'3"]

    import networkx as nx
    graphml_path = tmp_path / "object_model.graphml"
    nx.write_graphml(graph, graphml_path)
    reloaded = nx.read_graphml(graphml_path)
    assert float(reloaded.nodes["EEG PM3"]["beta_gamma_peak"]) == pytest.approx(
        graph.nodes["EEG PM3"]["beta_gamma_peak"])
    assert float(reloaded.nodes["EEG PM3"]["hemisphere_anomaly_mean"]) == pytest.approx(0.1)
    assert float(reloaded.nodes["EEG PM3"]["residual_onset_seconds"]) == pytest.approx(0.03)
    assert "residual_onset_seconds" not in reloaded.nodes["EEG PM'3"]


def test_build_object_model_graph_without_dicom_or_reservoir_omits_those_layers():
    # build_seizure_graph's np.corrcoef step needs at least 2 involved
    # channels to return a 2-D correlation matrix, so this uses 2 (both
    # bursting) rather than 1 -- unrelated to what this test itself checks.
    rng = np.random.default_rng(4)
    sfreq = 200.
    data = rng.normal(0, .05, (2, 2400))
    time = np.arange(400) / sfreq
    burst = 3 * np.sin(2 * np.pi * 35 * time)
    data[0, 1200:1600] += burst
    data[1, 1200:1600] += burst
    names = ["EEG PM3", "EEG CC8"]
    event = ClinicalEvent(6., duration_seconds=2.)
    process = analyse_brain_process(data, sfreq, names, event, baseline_seconds=5.)
    graph = build_seizure_graph(data, sfreq, names, event, process, baseline_seconds=5.)

    build_object_model_graph(graph)  # no hemisphere_summary, no reservoir_evaluation

    assert "hemisphere_anomaly_mean" not in graph.nodes["EEG PM3"]
    assert "residual_onset_seconds" not in graph.nodes["EEG PM3"]
    assert graph.nodes["EEG PM3"]["beta_gamma_peak"] == graph.nodes["EEG PM3"]["peak_z"]
