import networkx as nx
import pytest

from gnn_model.data import load_graph_dataset
from gnn_model.model import GNNConfig
from gnn_model.train import train_gnn


def _synthetic_graph(n_channels: int = 12) -> nx.Graph:
    """A small stand-in for a build_seizure_graph()/build_object_model_graph() result:
    same node kinds/attributes, none of the EDF/DICOM machinery."""
    graph = nx.Graph()
    graph.add_node("PEAK", kind="peak", label="seizure")
    for i in range(n_channels):
        role = "earliest" if i < 3 else "later_recruited"
        hemisphere = "left" if i % 2 == 0 else "right"
        graph.add_node(f"CH{i}", kind="channel", onset_latency_seconds=float(i), peak_z=10.0 - i,
                       is_initiator=(role == "earliest"), role=role, in_prior=(i == 0),
                       hemisphere=hemisphere)
        graph.add_edge("PEAK", f"CH{i}", kind="recruitment", weight=1.0 / (1.0 + i))
    for i in range(n_channels - 1):
        graph.add_edge(f"CH{i}", f"CH{i + 1}", kind="co-activation", weight=0.6)
    return graph


def test_load_graph_dataset_masks_cover_only_channel_nodes():
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)

    assert dataset.class_names == ["earliest", "later_recruited"]
    assert dataset.data.x.shape == (13, len(dataset.feature_names))  # 12 channels + PEAK

    peak_index = dataset.node_names.index("PEAK")
    assert not dataset.data.train_mask[peak_index]
    assert not dataset.data.val_mask[peak_index]

    channel_count = sum(1 for name in dataset.node_names if graph.nodes[name].get("kind") == "channel")
    assert int(dataset.data.train_mask.sum() + dataset.data.val_mask.sum()) == channel_count
    # No node is in both splits.
    assert not bool((dataset.data.train_mask & dataset.data.val_mask).any())


def test_load_graph_dataset_requires_target_attribute():
    graph = _synthetic_graph()
    del graph.nodes["CH0"]["role"]
    with pytest.raises(ValueError, match="role"):
        load_graph_dataset(graph)


def test_train_gnn_reports_confusion_matrices_and_history():
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(hidden_channels=4, num_layers=2, dropout=0.1, seed=1)

    result = train_gnn(dataset, config=config, epochs=5, lr=0.05)

    assert len(result.history["loss"]) == 5
    assert len(result.history["val_loss"]) == 5
    num_classes = len(dataset.class_names)
    assert result.train_confusion_matrix.shape == (num_classes, num_classes)
    assert result.val_confusion_matrix.shape == (num_classes, num_classes)
    assert result.train_confusion_matrix.sum() == int(dataset.data.train_mask.sum())
    assert result.val_confusion_matrix.sum() == int(dataset.data.val_mask.sum())
    assert "GCNConv" in result.model.describe()
    # Defaults (drop_edge_p=0, early_stopping_patience=None) must run the full unregularized
    # baseline -- this is the reproducibility guarantee gnn_model_result/baseline_overfit/ relies on.
    assert result.epochs == 5
    assert result.stopped_early is False
    assert result.best_epoch is None


def test_train_gnn_early_stopping_restores_best_val_loss_checkpoint():
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(hidden_channels=16, num_layers=2, dropout=0.0, seed=1)

    result = train_gnn(dataset, config=config, epochs=100, lr=0.05, weight_decay=0.0,
                       early_stopping_patience=3)

    assert result.epochs <= 100
    assert result.best_epoch is not None
    assert result.best_epoch <= result.epochs
    if result.stopped_early:
        assert result.epochs < 100
    # The reported val_loss is the best checkpoint's, not necessarily the last epoch trained.
    assert min(result.history["val_loss"]) == pytest.approx(result.history["val_loss"][result.best_epoch - 1])


def test_train_gnn_drop_edge_p_runs_without_error():
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(hidden_channels=4, num_layers=2, dropout=0.1, seed=1)

    result = train_gnn(dataset, config=config, epochs=5, lr=0.05, drop_edge_p=0.5)

    assert len(result.history["loss"]) == 5
    assert result.stopped_early is False


def test_train_gnn_gat_architecture_runs_and_reports_attention_params():
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(architecture="gat", hidden_channels=4, heads=2, num_layers=2, dropout=0.1, seed=1)

    result = train_gnn(dataset, config=config, epochs=5, lr=0.05)

    assert len(result.history["loss"]) == 5
    num_classes = len(dataset.class_names)
    assert result.train_confusion_matrix.shape == (num_classes, num_classes)
    assert "GATv2Conv" in result.model.describe()
    assert "heads=2" in result.model.describe()


def test_train_gnn_unknown_architecture_raises():
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(architecture="transformer", seed=1)

    with pytest.raises(ValueError, match="architecture"):
        train_gnn(dataset, config=config, epochs=1)


def test_train_gnn_early_stopping_metric_val_accuracy_picks_highest_accuracy_epoch():
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(hidden_channels=8, num_layers=2, dropout=0.0, seed=1)

    result = train_gnn(dataset, config=config, epochs=40, lr=0.05, weight_decay=0.0,
                       early_stopping_patience=10, early_stopping_metric="val_accuracy")

    assert result.early_stopping_metric == "val_accuracy"
    assert result.best_epoch is not None
    best_acc = result.history["val_accuracy"][result.best_epoch - 1]
    assert best_acc == max(result.history["val_accuracy"][:result.epochs])
