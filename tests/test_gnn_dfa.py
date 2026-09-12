import networkx as nx
import numpy as np

from gnn_model.augment_dfa import augment_graph_with_dfa
from gnn_model.dfa import dfa_alpha


def test_dfa_alpha_white_noise_near_one_half():
    rng = np.random.default_rng(0)
    white_noise = rng.normal(size=20000)
    assert 0.4 < dfa_alpha(white_noise) < 0.6


def test_dfa_alpha_random_walk_near_one_point_five():
    rng = np.random.default_rng(0)
    random_walk = np.cumsum(rng.normal(size=20000))
    assert 1.3 < dfa_alpha(random_walk) < 1.7


def test_dfa_alpha_nan_for_too_short_signal():
    assert np.isnan(dfa_alpha(np.ones(10)))
    assert np.isnan(dfa_alpha(np.array([])))


def test_augment_graph_with_dfa_sets_attribute_for_known_channels_only():
    graph = nx.Graph()
    graph.add_node("PEAK", kind="peak")
    graph.add_node("EEG A1", kind="channel", role="earliest")
    graph.add_node("EEG A2", kind="channel", role="later_recruited")
    graph.add_node("EEG A3", kind="channel", role="later_recruited")  # not in `names`/`data`

    rng = np.random.default_rng(0)
    names = ["EEG A1", "EEG A2"]
    sfreq = 200.0
    data = rng.normal(size=(2, int(60 * sfreq)))

    augment_graph_with_dfa(graph, data, sfreq, names, event_time_seconds=40.0, baseline_seconds=30.0)

    assert "dfa_alpha" in graph.nodes["EEG A1"]
    assert "dfa_alpha" in graph.nodes["EEG A2"]
    assert "dfa_alpha" not in graph.nodes["EEG A3"]  # never had a signal to compute it from
    assert "dfa_alpha" not in graph.nodes["PEAK"]  # not a channel node


def test_augment_graph_with_dfa_skips_when_baseline_too_short():
    graph = nx.Graph()
    graph.add_node("EEG A1", kind="channel", role="earliest")

    rng = np.random.default_rng(0)
    names = ["EEG A1"]
    sfreq = 200.0
    data = rng.normal(size=(1, int(60 * sfreq)))

    # event_time_seconds=0.02 with a 30 s baseline clamps the window start to sample 0 and the
    # window end to sample 4 -- far too short for any DFA window size to produce an estimate.
    augment_graph_with_dfa(graph, data, sfreq, names, event_time_seconds=0.02, baseline_seconds=30.0)

    assert "dfa_alpha" not in graph.nodes["EEG A1"]
