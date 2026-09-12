"""Add a per-channel DFA (Detrended Fluctuation Analysis) scaling exponent to an existing
seizure graph's channel nodes, computed from the EDF recording's own pre-event baseline.

Run as a module, from the repo root:

    python -m gnn_model.augment_dfa --edf dataset/sEEG-HFOs-8.edf \\
        --graph object_model_result/sEEG-HFOs-8/object_model_graph.graphml \\
        --output object_model_result/sEEG-HFOs-8/object_model_graph_dfa.graphml

Deliberately a separate, explicit step from the rest of ``gnn_model`` (see ``gnn_model.data``'s
docstring on the EDF/GraphML boundary that module otherwise keeps): this is the one place in
this package that reads raw EDF samples, and it only ever writes one new node attribute
(``dfa_alpha``) onto a *copy* of an existing graph -- it never builds a graph from scratch, and
``gnn_model.train``/``gnn_model.model`` never import ``extreme_event_agent``.

Each channel's exponent is computed over the ``--baseline-seconds`` immediately *before* the
resolved event -- interictal dynamics, not the seizure itself, so this asks whether a channel's
own pre-ictal fluctuation structure (self-similar/uncorrelated/anti-correlated -- see
``gnn_model.dfa``'s docstring) differs between "earliest" and "later_recruited" channels, not
whether the seizure changes it (which every channel's would, uninformatively, once it starts).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np

from extreme_event_agent.edf_workflow import find_annotated_event, read_edf

from .dfa import dfa_alpha

__all__ = ["augment_graph_with_dfa", "run"]


def augment_graph_with_dfa(graph, data: np.ndarray, sfreq: float, names: list[str],
                           event_time_seconds: float, baseline_seconds: float = 30.0):
    """Set ``dfa_alpha`` on every ``kind="channel"`` node of ``graph`` whose name is one of
    ``names`` and whose baseline window is long enough for a DFA estimate. Mutates and returns
    ``graph``; nodes that fail either check are left without the attribute -- the same
    "omit, don't null" rule ``object_model.graph`` already follows, since GraphML has no null
    type, and a fabricated 0 would be indistinguishable from a genuinely measured one.
    """
    name_to_index = {name: i for i, name in enumerate(names)}
    start_sample = max(0, int(round((event_time_seconds - baseline_seconds) * sfreq)))
    end_sample = int(round(event_time_seconds * sfreq))

    for node, node_data in graph.nodes(data=True):
        if node_data.get("kind") != "channel":
            continue
        index = name_to_index.get(node)
        if index is None:
            continue
        segment = data[index, start_sample:end_sample]
        alpha = dfa_alpha(segment)
        if alpha == alpha:  # not NaN
            node_data["dfa_alpha"] = float(alpha)
    return graph


def run(edf_path: str | Path, graph_path: str | Path, output_path: str | Path,
       baseline_seconds: float = 30.0) -> Path:
    """Read ``edf_path``/``graph_path``, add ``dfa_alpha`` to every channel node it can, and
    write the result to ``output_path`` (parent directories created as needed). Returns
    ``output_path``."""
    edf_path = Path(edf_path)
    graph_path = Path(graph_path)
    output_path = Path(output_path)

    annotated = find_annotated_event(edf_path)
    if annotated is None:
        raise ValueError(f"{edf_path} carries no EDF+ annotated seizure marker -- DFA needs a "
                         "resolved event time to know where the pre-event baseline ends.")
    data, sfreq, names = read_edf(edf_path)

    graph = nx.read_graphml(graph_path)
    augment_graph_with_dfa(graph, data, sfreq, names, annotated.time_seconds,
                           baseline_seconds=baseline_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edf", required=True, help="EDF recording the graph's channel names come from")
    parser.add_argument("--graph", required=True, help="Existing GraphML to augment "
                                                        "(e.g. object_model_graph.graphml)")
    parser.add_argument("--output", required=True, help="Path to write the DFA-augmented GraphML to")
    parser.add_argument("--baseline-seconds", type=float, default=30.0,
                        help="Pre-event window (seconds) each channel's DFA exponent is computed over")
    args = parser.parse_args()

    output = run(args.edf, args.graph, args.output, baseline_seconds=args.baseline_seconds)
    print(f"Wrote DFA-augmented graph to {output}")


if __name__ == "__main__":
    main()
