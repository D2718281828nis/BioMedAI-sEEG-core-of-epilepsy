"""Turn a GraphML seizure graph into a PyTorch Geometric ``Data`` for node classification.

Deliberately separate from how the graph itself gets built
(``extreme_event_agent.edf_workflow.build_seizure_graph`` /
``object_model.graph.build_object_model_graph``) -- this module only reads
node/edge attributes already on the graph, the same duck-typing discipline
``object_model.graph`` itself uses (kind="channel" nodes, a "hemisphere"
attribute, optional structural/reservoir attributes).

Node target: ``role`` (``"earliest"`` / ``"prior_early"`` / ``"later_recruited"``,
whichever subset actually appears in ``graph`` -- see
``edf_workflow.build_seizure_graph``'s docstring). ``is_initiator`` is
deliberately *excluded* from the feature set even though it is numeric and
present on every channel node: ``build_seizure_graph`` sets it from the same
``likely_initiators`` set ``role`` is derived from, so handing it to the
classifier as a feature would leak a near-answer rather than teach it
anything. ``hemisphere_anomaly_*``/``residual_*`` are set only for some nodes
(``object_model.graph``'s own "omit, don't null" rule) -- missing values are
filled with 0 plus an explicit ``has_structural_layer``/``has_reservoir_layer``
presence flag, rather than silently conflating "zero" with "not measured".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

CONTINUOUS_FEATURES = (
    "onset_latency_seconds", "peak_z", "hemisphere_anomaly_mean", "hemisphere_anomaly_max",
    "residual_onset_seconds", "residual_peak_score",
)
BINARY_FEATURES = ("in_prior", "has_structural_layer", "has_reservoir_layer")

__all__ = ["GraphDataset", "load_graph_dataset", "CONTINUOUS_FEATURES", "BINARY_FEATURES"]


@dataclass
class GraphDataset:
    """A single-graph, transductive node-classification dataset.

    ``data.train_mask``/``data.val_mask`` cover only the ``kind="channel"``
    nodes, split between the two -- there is no held-out test set, matching
    what a training-time loss/confusion-matrix report needs. Non-channel
    nodes (the synthetic ``PEAK`` node) stay in the graph for message passing
    with a neutral (all-zero) feature row and ``False`` in both masks.
    """
    data: Data
    feature_names: list[str]
    class_names: list[str]
    node_names: list[str]


def _hemisphere_categories(graph) -> list[str]:
    values = {data.get("hemisphere", "unknown") for _, data in graph.nodes(data=True)
             if data.get("kind") == "channel"}
    return sorted(values) or ["unknown"]


def _node_row(data: dict, hemisphere_categories: list[str]) -> tuple[list[float], list[float]]:
    """Returns (continuous_values, binary_and_onehot_values) for one channel node's attributes."""
    continuous = [float(data.get(name, 0.0)) for name in CONTINUOUS_FEATURES]
    binary = [
        float(bool(data.get("in_prior", False))),
        float("hemisphere_anomaly_mean" in data),
        float("residual_onset_seconds" in data),
    ]
    hemisphere = data.get("hemisphere", "unknown")
    one_hot = [1.0 if hemisphere == category else 0.0 for category in hemisphere_categories]
    return continuous, binary + one_hot


def load_graph_dataset(graph, target: str = "role", val_fraction: float = 0.3,
                       seed: int = 7) -> GraphDataset:
    """Build a ``GraphDataset`` from an ``nx.Graph`` (as produced by ``networkx.read_graphml``).

    ``target`` must be a node attribute present on every ``kind="channel"``
    node (``"role"`` by default); its distinct values, sorted, become the
    class list. Continuous features are z-scored using the channel nodes'
    own mean/std (transductive: every node's features are visible at train
    time, only ``target`` is masked) -- the standard convention for small
    single-graph node classification (Kipf & Welling's GCN on Cora, etc.).
    """
    node_names = list(graph.nodes())
    channel_names = [name for name in node_names if graph.nodes[name].get("kind") == "channel"]
    if len(channel_names) < 2:
        raise ValueError(f"Graph has only {len(channel_names)} channel node(s) -- nothing to classify.")
    missing_target = [name for name in channel_names if target not in graph.nodes[name]]
    if missing_target:
        raise ValueError(f"{len(missing_target)} channel node(s) are missing the {target!r} attribute, "
                         f"e.g. {missing_target[0]!r}.")

    hemisphere_categories = _hemisphere_categories(graph)
    continuous_rows, extra_rows = [], []
    for name in channel_names:
        continuous, extra = _node_row(graph.nodes[name], hemisphere_categories)
        continuous_rows.append(continuous)
        extra_rows.append(extra)
    continuous_matrix = np.array(continuous_rows, dtype=np.float64)
    mean = continuous_matrix.mean(axis=0)
    std = continuous_matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    continuous_matrix = (continuous_matrix - mean) / std

    feature_names = list(CONTINUOUS_FEATURES) + list(BINARY_FEATURES) + \
        [f"hemisphere_{c}" for c in hemisphere_categories]
    channel_features = np.concatenate([continuous_matrix, np.array(extra_rows, dtype=np.float64)], axis=1)

    class_names = sorted({graph.nodes[name][target] for name in channel_names})
    class_index = {name: i for i, name in enumerate(class_names)}

    num_features = len(feature_names)
    x = np.zeros((len(node_names), num_features), dtype=np.float32)
    y = np.zeros(len(node_names), dtype=np.int64)
    train_mask = np.zeros(len(node_names), dtype=bool)
    val_mask = np.zeros(len(node_names), dtype=bool)

    name_to_index = {name: i for i, name in enumerate(node_names)}
    for name, features in zip(channel_names, channel_features):
        position = name_to_index[name]
        x[position] = features
        y[position] = class_index[graph.nodes[name][target]]

    labels = [class_index[graph.nodes[name][target]] for name in channel_names]
    try:
        train_names, val_names = train_test_split(
            channel_names, test_size=val_fraction, random_state=seed, stratify=labels)
    except ValueError:
        # A class with too few members to stratify (e.g. a single example) -- fall back to a plain split
        # rather than refusing to train; the resulting confusion matrix will just say so honestly.
        train_names, val_names = train_test_split(
            channel_names, test_size=val_fraction, random_state=seed, stratify=None)
    for name in train_names:
        train_mask[name_to_index[name]] = True
    for name in val_names:
        val_mask[name_to_index[name]] = True

    edge_index: list[list[int]] = [[], []]
    edge_weight: list[float] = []
    for u, v, edge_data in graph.edges(data=True):
        i, j = name_to_index[u], name_to_index[v]
        # Undirected graph -> both directions; magnitude only (co-activation weight is a signed
        # correlation, and GCNConv's degree normalization assumes non-negative edge weights).
        weight = abs(float(edge_data.get("weight", 1.0)))
        edge_index[0].extend([i, j])
        edge_index[1].extend([j, i])
        edge_weight.extend([weight, weight])

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
    )
    return GraphDataset(data=data, feature_names=feature_names, class_names=class_names, node_names=node_names)
