import networkx as nx
import pytest

from gnn_model.data import load_graph_dataset, load_graph_kfold_datasets
from gnn_model.model import GNNConfig
from gnn_model.train import cross_validate_gnn, train_gnn


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


def test_train_gnn_early_stopping_metric_val_macro_f1_picks_highest_f1_epoch():
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(hidden_channels=8, num_layers=2, dropout=0.0, seed=1)

    result = train_gnn(dataset, config=config, epochs=40, lr=0.05, weight_decay=0.0,
                       early_stopping_patience=10, early_stopping_metric="val_macro_f1")

    assert result.early_stopping_metric == "val_macro_f1"
    assert "val_macro_f1" in result.history
    assert len(result.history["val_macro_f1"]) == result.epochs
    best_f1 = result.history["val_macro_f1"][result.best_epoch - 1]
    assert best_f1 == max(result.history["val_macro_f1"][:result.epochs])


def test_train_gnn_unknown_early_stopping_metric_raises():
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(seed=1)

    with pytest.raises(ValueError, match="early_stopping_metric"):
        train_gnn(dataset, config=config, epochs=1, early_stopping_metric="bogus")


def test_seizure_gat_single_layer_uses_heads_averaged_not_forced_to_one():
    # A single-layer SeizureGAT must still honour GNNConfig.heads (averaged, concat=False) --
    # right-sizing this architecture down to one layer should not silently also force heads=1.
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(architecture="gat", num_layers=1, heads=3, dropout=0.1, seed=1)

    result = train_gnn(dataset, config=config, epochs=2, lr=0.05)

    conv = result.model.convs[0]
    assert conv.heads == 3
    assert conv.concat is False


def test_seizure_gat_concat_heads_false_keeps_hidden_width_constant():
    # A deep stack (num_layers > 2) with concat_heads=False must not multiply its hidden width
    # by heads every layer -- that multiplicative blow-up was the original overfitting cause.
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(architecture="gat", num_layers=4, heads=3, hidden_channels=5,
                       concat_heads=False, dropout=0.1, seed=1)

    result = train_gnn(dataset, config=config, epochs=2, lr=0.05)

    convs = result.model.convs
    assert len(convs) == 4
    for conv in convs[:-1]:
        assert conv.concat is False
        assert conv.heads == 3
        assert conv.out_channels == 5  # width stays hidden_channels, not hidden_channels * heads
    assert convs[1].in_channels == 5  # hidden-to-hidden width also stays constant


def test_seizure_gat_residual_flag_reaches_every_layer():
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(architecture="gat", num_layers=3, heads=2, hidden_channels=4,
                       residual=True, dropout=0.1, seed=1)

    result = train_gnn(dataset, config=config, epochs=2, lr=0.05)

    assert all(conv.residual for conv in result.model.convs)


def test_train_gnn_label_smoothing_runs_without_error():
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(hidden_channels=4, num_layers=2, dropout=0.1, seed=1)

    result = train_gnn(dataset, config=config, epochs=5, lr=0.05, label_smoothing=0.1)

    assert len(result.history["loss"]) == 5


def test_load_graph_kfold_datasets_covers_every_channel_node_exactly_once():
    graph = _synthetic_graph()
    datasets = load_graph_kfold_datasets(graph, n_splits=3, seed=1)

    assert len(datasets) == 3
    channel_count = sum(1 for name in datasets[0].node_names
                        if graph.nodes[name].get("kind") == "channel")
    val_counts = {}
    for dataset in datasets:
        # Every fold shares identical features/edges -- only the masks differ.
        assert dataset.feature_names == datasets[0].feature_names
        assert dataset.data.x.shape == datasets[0].data.x.shape
        for name in dataset.node_names:
            if dataset.data.val_mask[dataset.node_names.index(name)]:
                val_counts[name] = val_counts.get(name, 0) + 1
    # Every channel node appears in exactly one fold's validation set (PEAK never does).
    channel_names = [n for n in datasets[0].node_names if graph.nodes[n].get("kind") == "channel"]
    for name in channel_names:
        assert val_counts.get(name) == 1
    assert "PEAK" not in val_counts
    assert sum(val_counts.values()) == channel_count


def test_cross_validate_gnn_aggregates_out_of_fold_confusion_matrix():
    graph = _synthetic_graph()
    datasets = load_graph_kfold_datasets(graph, n_splits=3, seed=1)
    config = GNNConfig(hidden_channels=4, num_layers=2, dropout=0.1, seed=1)

    result = cross_validate_gnn(datasets, config=config, epochs=5, lr=0.05)

    assert len(result.fold_results) == 3
    channel_count = sum(1 for name in datasets[0].node_names
                        if graph.nodes[name].get("kind") == "channel")
    # Every classifiable node counted exactly once across all folds' val_confusion_matrix.
    assert result.out_of_fold_confusion_matrix.sum() == channel_count
    assert result.out_of_fold_confusion_matrix.shape == (2, 2)
    assert "earliest" in result.out_of_fold_report
    assert "macro avg" in result.out_of_fold_report


def test_cross_validate_gnn_requires_at_least_two_folds():
    graph = _synthetic_graph()
    datasets = load_graph_kfold_datasets(graph, n_splits=3, seed=1)
    config = GNNConfig(seed=1)

    with pytest.raises(ValueError, match="fold"):
        cross_validate_gnn(datasets[:1], config=config, epochs=1)


def test_train_gnn_gat_is_deterministic_within_one_process():
    # Regression guard for the reproducibility bug documented in gnn_model.train's docstring:
    # two identical SeizureGAT training runs, same process, must agree bit-for-bit.
    graph = _synthetic_graph()
    dataset = load_graph_dataset(graph, val_fraction=0.4, seed=1)
    config = GNNConfig(architecture="gat", num_layers=3, heads=2, hidden_channels=4,
                       residual=True, dropout=0.3, seed=1)

    result_a = train_gnn(dataset, config=config, epochs=15, lr=0.05, drop_edge_p=0.2)
    result_b = train_gnn(dataset, config=config, epochs=15, lr=0.05, drop_edge_p=0.2)

    assert result_a.history["loss"] == result_b.history["loss"]
    assert result_a.history["val_loss"] == result_b.history["val_loss"]
    assert (result_a.train_confusion_matrix == result_b.train_confusion_matrix).all()


def test_load_graph_dataset_dfa_feature_only_included_when_present():
    graph_without = _synthetic_graph()
    dataset_without = load_graph_dataset(graph_without, val_fraction=0.4, seed=1)
    assert "dfa_alpha" not in dataset_without.feature_names
    assert "has_dfa_layer" not in dataset_without.feature_names

    graph_with = _synthetic_graph()
    graph_with.nodes["CH0"]["dfa_alpha"] = 1.1  # just one node -- still enough to add the column
    dataset_with = load_graph_dataset(graph_with, val_fraction=0.4, seed=1)
    assert "dfa_alpha" in dataset_with.feature_names
    assert "has_dfa_layer" in dataset_with.feature_names
    # Schema for a graph without dfa_alpha must be untouched -- this is what keeps
    # gnn_model_result/baseline_overfit/'s bit-for-bit reproducibility guarantee intact for
    # every graph that has never been through gnn_model.augment_dfa.
    assert dataset_without.data.x.shape[1] == len(dataset_without.feature_names)
    assert dataset_with.data.x.shape[1] == dataset_without.data.x.shape[1] + 2


def test_load_graph_kfold_datasets_group_by_shaft_keeps_shafts_together():
    graph = nx.Graph()
    graph.add_node("PEAK", kind="peak")
    # Two shafts (PM, CC), 4 contacts each -- 3 "earliest" spread across both shafts so
    # stratification still has something to balance.
    names = ["EEG PM1", "EEG PM2", "EEG PM3", "EEG PM4", "EEG CC1", "EEG CC2", "EEG CC3", "EEG CC4"]
    roles = ["earliest", "earliest", "later_recruited", "later_recruited",
            "earliest", "later_recruited", "later_recruited", "later_recruited"]
    for name, role in zip(names, roles):
        graph.add_node(name, kind="channel", onset_latency_seconds=0.0, peak_z=1.0, role=role,
                       in_prior=False, hemisphere="left")
        graph.add_edge("PEAK", name, kind="recruitment", weight=1.0)

    datasets = load_graph_kfold_datasets(graph, n_splits=2, seed=1, group_by_shaft=True)

    for dataset in datasets:
        val_names = {name for name in dataset.node_names
                    if dataset.data.val_mask[dataset.node_names.index(name)]}
        pm_in_val = {n for n in val_names if n.startswith("EEG PM")}
        cc_in_val = {n for n in val_names if n.startswith("EEG CC")}
        # Each shaft is either entirely in this fold's val set or entirely out of it.
        assert len(pm_in_val) in (0, 4)
        assert len(cc_in_val) in (0, 4)
