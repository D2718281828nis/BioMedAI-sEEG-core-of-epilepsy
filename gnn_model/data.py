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
anything. ``hemisphere_anomaly_*``/``residual_*``/``dfa_alpha`` are set only
for some nodes (``object_model.graph``'s own "omit, don't null" rule, and
``gnn_model.augment_dfa``'s the same) -- missing values are filled with 0
plus an explicit ``has_structural_layer``/``has_reservoir_layer``/
``has_dfa_layer`` presence flag, rather than silently conflating "zero" with
"not measured".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch_geometric.data import Data

CONTINUOUS_FEATURES = (
    "onset_latency_seconds", "peak_z", "hemisphere_anomaly_mean", "hemisphere_anomaly_max",
    "residual_onset_seconds", "residual_peak_score",
)
BINARY_FEATURES = ("in_prior", "has_structural_layer", "has_reservoir_layer")
# Included in the schema only when at least one channel node actually carries it (see
# _build_graph_arrays) -- unlike CONTINUOUS_FEATURES/BINARY_FEATURES above, which are always
# present (as zero-filled columns when absent). Keeping dfa_alpha conditional, rather than just
# adding it to CONTINUOUS_FEATURES, is not cosmetic: an unconditional addition changes
# in_channels for *every* graph, DFA-augmented or not, which reshuffles the first layer's random
# weight initialization and would silently break train_gnn's bit-for-bit baseline reproducibility
# guarantee (gnn_model.train's docstring) for every existing graph that has never seen
# gnn_model.augment_dfa.
OPTIONAL_CONTINUOUS_FEATURE = "dfa_alpha"
OPTIONAL_BINARY_FEATURE = "has_dfa_layer"

# Same convention edf_workflow.parse_contact_name documents for the bipolar montage (README,
# "Bipolar montage"): "<shaft><contact number>", an optional leading "EEG " prefix, and a
# trailing "'" that is part of the shaft label (e.g. "EEG CC'8" -> shaft "CC'"). Duplicated here
# as a plain regex rather than importing extreme_event_agent, matching this module's own
# "duck-typing off the graph alone" discipline -- the shaft is a naming-convention fact, not data
# that requires reading the EDF.
_SHAFT_PATTERN = re.compile(r"^(?:EEG\s+)?([A-Za-zА-Яа-я]+'?)\d+$")


def _shaft_group(channel_name: str) -> str:
    """The electrode shaft a channel name belongs to (e.g. "EEG PM3" -> "PM"), or the whole name
    unchanged if it doesn't match the ``<shaft><contact number>`` convention."""
    match = _SHAFT_PATTERN.match(channel_name)
    return match.group(1) if match else channel_name

__all__ = ["GraphDataset", "load_graph_dataset", "load_graph_kfold_datasets",
          "CONTINUOUS_FEATURES", "BINARY_FEATURES"]


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


def _node_row(data: dict, continuous_feature_names: list[str], include_dfa: bool,
             hemisphere_categories: list[str]) -> tuple[list[float], list[float]]:
    """Returns (continuous_values, binary_and_onehot_values) for one channel node's attributes."""
    continuous = [float(data.get(name, 0.0)) for name in continuous_feature_names]
    binary = [
        float(bool(data.get("in_prior", False))),
        float("hemisphere_anomaly_mean" in data),
        float("residual_onset_seconds" in data),
    ]
    if include_dfa:
        binary.append(float(OPTIONAL_CONTINUOUS_FEATURE in data))
    hemisphere = data.get("hemisphere", "unknown")
    one_hot = [1.0 if hemisphere == category else 0.0 for category in hemisphere_categories]
    return continuous, binary + one_hot


@dataclass
class _GraphArrays:
    """Everything a ``GraphDataset`` needs except the train/val split itself -- built once and
    shared across every fold in ``load_graph_kfold_datasets`` so features/edges are never
    recomputed (or allowed to differ) between folds."""
    node_names: list[str]
    channel_names: list[str]
    feature_names: list[str]
    class_names: list[str]
    name_to_index: dict[str, int]
    labels: list[int]  # per channel_names, aligned
    x: np.ndarray
    y: np.ndarray
    edge_index: list[list[int]]
    edge_weight: list[float]


def _build_graph_arrays(graph, target: str) -> _GraphArrays:
    node_names = list(graph.nodes())
    channel_names = [name for name in node_names if graph.nodes[name].get("kind") == "channel"]
    if len(channel_names) < 2:
        raise ValueError(f"Graph has only {len(channel_names)} channel node(s) -- nothing to classify.")
    missing_target = [name for name in channel_names if target not in graph.nodes[name]]
    if missing_target:
        raise ValueError(f"{len(missing_target)} channel node(s) are missing the {target!r} attribute, "
                         f"e.g. {missing_target[0]!r}.")

    include_dfa = any(OPTIONAL_CONTINUOUS_FEATURE in graph.nodes[name] for name in channel_names)
    continuous_feature_names = list(CONTINUOUS_FEATURES) + \
        ([OPTIONAL_CONTINUOUS_FEATURE] if include_dfa else [])
    binary_feature_names = list(BINARY_FEATURES) + ([OPTIONAL_BINARY_FEATURE] if include_dfa else [])

    hemisphere_categories = _hemisphere_categories(graph)
    continuous_rows, extra_rows = [], []
    for name in channel_names:
        continuous, extra = _node_row(graph.nodes[name], continuous_feature_names, include_dfa,
                                      hemisphere_categories)
        continuous_rows.append(continuous)
        extra_rows.append(extra)
    continuous_matrix = np.array(continuous_rows, dtype=np.float64)
    mean = continuous_matrix.mean(axis=0)
    std = continuous_matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    continuous_matrix = (continuous_matrix - mean) / std

    feature_names = continuous_feature_names + binary_feature_names + \
        [f"hemisphere_{c}" for c in hemisphere_categories]
    channel_features = np.concatenate([continuous_matrix, np.array(extra_rows, dtype=np.float64)], axis=1)

    class_names = sorted({graph.nodes[name][target] for name in channel_names})
    class_index = {name: i for i, name in enumerate(class_names)}

    num_features = len(feature_names)
    x = np.zeros((len(node_names), num_features), dtype=np.float32)
    y = np.zeros(len(node_names), dtype=np.int64)
    name_to_index = {name: i for i, name in enumerate(node_names)}
    for name, features in zip(channel_names, channel_features):
        position = name_to_index[name]
        x[position] = features
        y[position] = class_index[graph.nodes[name][target]]
    labels = [class_index[graph.nodes[name][target]] for name in channel_names]

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

    return _GraphArrays(node_names=node_names, channel_names=channel_names, feature_names=feature_names,
                        class_names=class_names, name_to_index=name_to_index, labels=labels,
                        x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)


def _make_dataset(arrays: _GraphArrays, train_names: list[str], val_names: list[str]) -> GraphDataset:
    train_mask = np.zeros(len(arrays.node_names), dtype=bool)
    val_mask = np.zeros(len(arrays.node_names), dtype=bool)
    for name in train_names:
        train_mask[arrays.name_to_index[name]] = True
    for name in val_names:
        val_mask[arrays.name_to_index[name]] = True

    data = Data(
        x=torch.tensor(arrays.x, dtype=torch.float32),
        y=torch.tensor(arrays.y, dtype=torch.long),
        edge_index=torch.tensor(arrays.edge_index, dtype=torch.long),
        edge_weight=torch.tensor(arrays.edge_weight, dtype=torch.float32),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
    )
    return GraphDataset(data=data, feature_names=arrays.feature_names, class_names=arrays.class_names,
                        node_names=arrays.node_names)


def load_graph_dataset(graph, target: str = "role", val_fraction: float = 0.3,
                       seed: int = 7) -> GraphDataset:
    """Build a ``GraphDataset`` from an ``nx.Graph`` (as produced by ``networkx.read_graphml``).

    ``target`` must be a node attribute present on every ``kind="channel"``
    node (``"role"`` by default); its distinct values, sorted, become the
    class list. Continuous features are z-scored using the channel nodes'
    own mean/std (transductive: every node's features are visible at train
    time, only ``target`` is masked) -- the standard convention for small
    single-graph node classification (Kipf & Welling's GCN on Cora, etc.).

    A *single* random train/val split -- see ``load_graph_kfold_datasets``
    for the cross-validated alternative, which pools every node into a
    validation set exactly once across its folds instead of leaving most of
    them permanently on the train side of one arbitrary split.
    """
    arrays = _build_graph_arrays(graph, target)
    try:
        train_names, val_names = train_test_split(
            arrays.channel_names, test_size=val_fraction, random_state=seed, stratify=arrays.labels)
    except ValueError:
        # A class with too few members to stratify (e.g. a single example) -- fall back to a plain split
        # rather than refusing to train; the resulting confusion matrix will just say so honestly.
        train_names, val_names = train_test_split(
            arrays.channel_names, test_size=val_fraction, random_state=seed, stratify=None)
    return _make_dataset(arrays, train_names, val_names)


def load_graph_kfold_datasets(graph, target: str = "role", n_splits: int = 5,
                              seed: int = 7, group_by_shaft: bool = False) -> list[GraphDataset]:
    """Like ``load_graph_dataset``, but returns ``n_splits`` ``GraphDataset``s from a
    ``StratifiedKFold`` split instead of one ``GraphDataset`` from a single random split.

    All ``n_splits`` datasets share identical features/edges (``_build_graph_arrays`` runs once);
    only ``train_mask``/``val_mask`` differ. Every channel node ends up in exactly one fold's
    validation set, so summing each fold's ``val_confusion_matrix`` (see
    ``gnn_model.train.cross_validate_gnn``) gives a full leave-some-out confusion matrix over
    *every* node this graph has, rather than the ~30% one random split happens to hold out --
    the more trustworthy read when a class has as few as 5 total members: with
    ``n_splits=5`` (the default), each fold holds out almost exactly one of them.

    ``group_by_shaft=True`` switches to ``StratifiedGroupKFold``, grouped by each channel's
    electrode shaft (``_shaft_group`` -- adjacent contacts on the same physical depth electrode,
    e.g. every ``PM1``..``PM8``). Plain ``StratifiedKFold`` can split a shaft across train and
    val; since adjacent contacts on one shaft are exactly the channels most strongly linked by
    this graph's own co-activation edges and by real physiological proximity, that lets a
    val node's prediction lean on a near-duplicate neighbour sitting in train -- a subtler
    leak than the label itself, but a leak. Grouping by shaft closes it at the cost of a harder,
    less flattering (but more honest) validation split.
    """
    arrays = _build_graph_arrays(graph, target)
    if group_by_shaft:
        groups = [_shaft_group(name) for name in arrays.channel_names]
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(arrays.channel_names, arrays.labels, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(arrays.channel_names, arrays.labels)

    datasets = []
    for train_idx, val_idx in split_iter:
        train_names = [arrays.channel_names[i] for i in train_idx]
        val_names = [arrays.channel_names[i] for i in val_idx]
        datasets.append(_make_dataset(arrays, train_names, val_names))
    return datasets
