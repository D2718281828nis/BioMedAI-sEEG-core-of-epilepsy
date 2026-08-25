"""Layer DICOM and reservoir evidence onto an EDF recruitment graph, without merging them.

``edf_workflow.build_seizure_graph`` already produces a NetworkX graph whose
channel nodes carry the *dynamic* (EDF) layer: ``role``, ``latency_seconds``,
``in_prior``, ``hemisphere`` (by channel-name convention), and ``peak_z``
(the 13-80 Hz recruitment score). ``build_object_model_graph`` takes that
graph as-is -- same nodes, same PEAK node, same recruitment spokes, same
co-activation mesh -- and adds two more, independently-sourced attribute
groups per node:

- **structural** (DICOM, hemisphere granularity only -- see
  ``multimodal_approach/README.md``, "Honest limits": there is no verified
  per-contact 3-D electrode localization here): ``hemisphere_anomaly_mean``/
  ``hemisphere_anomaly_max``, looked up via each node's own ``hemisphere``
  attribute against a structural-anomaly ``hemisphere_summary``.
- **model** (the reservoir plant, ``model/``): ``residual_onset_seconds``/
  ``residual_peak_score``, only for nodes that are also reservoir output
  channels.

Nothing here computes a combined score across these three groups --
``multimodal_approach/structural_anomaly.py`` already keeps
``combined_anomaly``/``combined_heterogeneity`` separate on the same
principle, and doing it again here (folding EDF+DICOM+reservoir into one
number) would hide exactly the disagreement between modalities this graph
exists to make visible. A node whose EDF role says "earliest" and whose
structural anomaly is high on the *other* hemisphere is the interesting
case; averaging the two away would erase it.
"""
from __future__ import annotations

from typing import Any

__all__ = ["build_object_model_graph"]


def build_object_model_graph(graph: Any, hemisphere_summary: dict[str, Any] | None = None,
                             reservoir_evaluation: Any | None = None) -> Any:
    """Add structural and reservoir node attributes to an existing seizure graph, in place.

    ``graph`` is exactly an ``edf_workflow.build_seizure_graph`` result (or
    anything shaped like one: channel nodes with ``kind="channel"`` and a
    ``hemisphere`` attribute already set). Mutates and returns the same
    graph object.

    A structural attribute is set on a channel node only when both
    ``hemisphere_summary`` is given *and* that node's own ``hemisphere`` is
    ``"right"``/``"left"`` (never ``"unknown"``/``"mixed"``) *and* the
    corresponding value is actually present in ``hemisphere_summary`` --
    otherwise the key is omitted entirely, never set to ``None``: GraphML
    has no null type (``nx.write_graphml`` raises on a ``None``-valued
    attribute), and omitting the key is also the more honest representation
    of "not available" versus "available and zero". The same omit-don't-null
    rule applies to the reservoir attributes, which are only ever set for
    nodes in ``reservoir_evaluation.window.output_names``.

    ``reservoir_evaluation`` is duck-typed (``window.output_names``,
    ``per_channel_onset_seconds``, ``per_channel_peak_score``) rather than
    importing ``model.plant.ReservoirEvaluation`` directly, matching
    ``extreme_event_agent.verification``'s own leaf-module discipline.
    """
    channel_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "channel"]

    for name in channel_nodes:
        data = graph.nodes[name]
        # Alias, not a recomputation: peak_z already *is* the 13-80 Hz
        # recruitment score build_seizure_graph attached; beta_gamma_peak is
        # this task's own name for the same dynamic-layer quantity.
        if "peak_z" in data:
            data["beta_gamma_peak"] = data["peak_z"]

        hemisphere = data.get("hemisphere")
        if hemisphere_summary is not None and hemisphere in ("right", "left"):
            side_summary = hemisphere_summary.get(f"{hemisphere}_hemisphere") or {}
            mean_anomaly = side_summary.get("mean_abs_anomaly")
            max_anomaly = side_summary.get("max_abs_anomaly")
            if mean_anomaly is not None:
                data["hemisphere_anomaly_mean"] = float(mean_anomaly)
            if max_anomaly is not None:
                data["hemisphere_anomaly_max"] = float(max_anomaly)

    if reservoir_evaluation is not None:
        for name in reservoir_evaluation.window.output_names:
            if name not in graph.nodes:
                continue
            onset = reservoir_evaluation.per_channel_onset_seconds.get(name)
            if onset is not None:
                graph.nodes[name]["residual_onset_seconds"] = float(onset)
            peak = reservoir_evaluation.per_channel_peak_score.get(name)
            if peak is not None:
                graph.nodes[name]["residual_peak_score"] = float(peak)

    return graph
