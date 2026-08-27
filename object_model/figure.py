"""The one figure that summarizes all three modalities and the verification against ground truth.

Six panels (see ``plot_object_model_summary``): the EDF recruitment cascade
(A), the object-model graph (B), a DICOM slice through the strongest
structural cluster (C), the reservoir's per-channel residual (D), the
lateralization index per source (E), and the structural anomaly graph on
three real DICOM slices (F -- axial/coronal/sagittal, one row). F replaces
what used to be a "temporal accuracy" bar chart (delta-t per method vs. the
annotation) that sat next to E: that number is still written in full to
``verification_report.json`` and appears in this repo's dissertation-figure
exports, so dropping it from *this* composite trades one already-duplicated
number for graph structure this figure otherwise has no DICOM-side
counterpart to.

Every panel that needs a raw EDF/DICOM/reservoir computation reuses the
existing, already-tested function that produces it (``_beta_gamma_z_scores``,
``_seizure_graph_layout``, ``find_top_anomaly_clusters``,
``evaluation.per_channel_score``, ``multimodal_approach.structural_graph``'s
own graph builder and slice-drawer) rather than recomputing it a second way
-- this figure is a *view* onto results computed elsewhere, never a second
implementation of them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from extreme_event_agent.edf_workflow import (PEAK_NODE, RECRUITMENT_THRESHOLD_MAD,
                                               SIMULTANEITY_WINDOW_SECONDS, PRIOR_WINDOW_SECONDS,
                                               _beta_gamma_z_scores, _seizure_graph_layout)
from extreme_event_agent.models import AnnotatedEvent, BrainProcess
from extreme_event_agent.verification import INDETERMINATE_LI_THRESHOLD, VerificationReport

__all__ = ["plot_object_model_summary"]

_ROLE_COLORS = {"earliest": "crimson", "prior_early": "darkorange", "later_recruited": "steelblue"}
_HEMISPHERE_MARKERS = {"right": "o", "left": "s", "mixed": "^", "unknown": "d"}


def _panel_a_timeseries(ax, data: np.ndarray, sfreq: float, names: list[str],
                        event: Any, process: BrainProcess, baseline_seconds: float, analysis_seconds: float) -> None:
    times, z = _beta_gamma_z_scores(data, sfreq, event, baseline_seconds, analysis_seconds)
    if not process.onset_latency_seconds:
        ax.text(0.5, 0.5, "No channel crossed threshold", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("A. EDF recruitment cascade")
        return
    ordered_names = sorted(process.onset_latency_seconds, key=process.onset_latency_seconds.get)
    indices = [names.index(name) for name in ordered_names]
    matrix = z[:, indices].T
    relative_times = times - event.time_seconds
    n_rows = len(ordered_names)
    im = ax.imshow(matrix, aspect="auto", cmap="inferno", vmin=0,
                   vmax=max(RECRUITMENT_THRESHOLD_MAD, float(np.percentile(matrix, 99))),
                   extent=[relative_times[0], relative_times[-1], n_rows, 0])
    ax.figure.colorbar(im, ax=ax, shrink=0.7, label="z (13-80 Hz)")

    # Every real annotation this recording carries (not a hardcoded time) --
    # AnnotatedEvent.annotations is the full matched cluster (see
    # edf_workflow.find_annotated_event); a non-annotation event context
    # (ClinicalEvent/DetectedEvent) has only its own single time.
    marks = (event.annotations if isinstance(event, AnnotatedEvent) and event.annotations
            else ((event.time_seconds, event.label),))
    for onset, label in marks:
        ax.axvline(onset - event.time_seconds, color="cyan", lw=1, ls="--")

    tau_min = process.earliest_latency_seconds
    ax.axvspan(tau_min, tau_min + PRIOR_WINDOW_SECONDS, color="gold", alpha=0.10, zorder=0)
    ax.axvspan(tau_min, tau_min + SIMULTANEITY_WINDOW_SECONDS, color="lime", alpha=0.18, zorder=0)
    ax.axvline(tau_min, color="white", lw=1.0)

    prior_set, earliest_set = set(process.prior_matched), set(process.earliest_contacts)
    show_labels = n_rows <= 25
    if show_labels:
        labels = [("* " if name in prior_set else "  ") + name for name in ordered_names]
        ax.set_yticks(np.arange(n_rows) + .5, labels, fontsize=6)
        for tick, name in zip(ax.get_yticklabels(), ordered_names):
            if name in earliest_set:
                tick.set_fontweight("bold")
    else:
        ax.set_yticks([])
    ax.set_xlabel("s from event")
    ax.set_title(f"A. EDF recruitment ({n_rows} ch, row order = data; * = prior)", fontsize=9)


def _panel_b_graph(ax, graph: Any, layout: str = "radial", seed: int = 7) -> None:
    channel_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "channel"]
    if not channel_nodes:
        ax.text(0.5, 0.5, "No involved channels to graph", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("B. Object-model graph")
        ax.axis("off")
        return
    pos = _seizure_graph_layout(graph, channel_nodes, layout, seed)
    mesh_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "co-activation"]
    spoke_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "recruitment"]
    import networkx as nx
    nx.draw_networkx_edges(graph, pos, edgelist=spoke_edges, edge_color="0.88", width=0.4, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=mesh_edges, edge_color="0.6", width=0.6, alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=[PEAK_NODE], node_shape="*", node_size=500,
                           node_color="black", ax=ax)
    peak_z = {node: graph.nodes[node].get("peak_z", 0.0) for node in channel_nodes}
    vmax = max(RECRUITMENT_THRESHOLD_MAD, float(np.percentile(list(peak_z.values()), 99))) if peak_z else 6.0
    for hemisphere, marker in _HEMISPHERE_MARKERS.items():
        nodes_here = [n for n in channel_nodes if graph.nodes[n].get("hemisphere", "unknown") == hemisphere]
        if not nodes_here:
            continue
        colors = [_ROLE_COLORS.get(graph.nodes[n].get("role", "later_recruited"), "0.6") for n in nodes_here]
        sizes = [40 + 100 * min(peak_z[n] / vmax, 1.0) for n in nodes_here]
        edge_colors = ["gold" if graph.nodes[n].get("in_prior") else "0.3" for n in nodes_here]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes_here, node_shape=marker, node_size=sizes,
                               node_color=colors, edgecolors=edge_colors, linewidths=1.0, ax=ax)
    earliest_nodes = [n for n in channel_nodes if graph.nodes[n].get("role") == "earliest"]
    if earliest_nodes:
        nx.draw_networkx_labels(graph, pos, labels={n: n for n in earliest_nodes}, font_size=5, ax=ax)
    ax.set_title("B. Object-model graph (fill=role, ring=prior, shape=hemisphere, size=z)", fontsize=8)
    ax.axis("off")


def _panel_c_dicom(ax, structural_result: Any) -> None:
    from multimodal_approach.structural_anomaly import find_top_anomaly_clusters
    if structural_result is None:
        ax.text(0.5, 0.5, "No DICOM available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("C. Structural (DICOM)")
        ax.axis("off")
        return
    clusters = find_top_anomaly_clusters(structural_result)
    volume = structural_result.t1_geometry.volume
    anomaly = structural_result.combined_anomaly
    if clusters:
        k0, _, _ = clusters[0]["peak_voxel_kij"]
        title_extra = (f"peak z={clusters[0]['peak_value']:+.2f}, {clusters[0]['hemisphere']} hemisphere")
    else:
        k0 = volume.shape[0] // 2
        title_extra = "no cluster reached threshold"
    vmin, vmax = np.percentile(volume, [1, 99])
    ax.imshow(volume[k0, :, :], cmap="gray", vmin=vmin, vmax=vmax)
    masked = np.ma.masked_where(np.abs(anomaly[k0, :, :]) < 2.0, anomaly[k0, :, :])
    im = ax.imshow(masked, cmap="coolwarm", vmin=-8, vmax=8, alpha=0.7)
    ax.figure.colorbar(im, ax=ax, shrink=0.6, label="asymmetry z")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"C. DICOM axial ({title_extra})\nmasking={structural_result.masking_method}\n"
                f"midline r={structural_result.midline_mirror_correlation:.2f}", fontsize=7)


def _panel_d_residual(ax, evaluation: Any) -> None:
    if evaluation is None:
        ax.text(0.5, 0.5, "No reservoir evaluation available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("D. Reservoir residual")
        return
    output_names = evaluation.window.output_names
    order = sorted(range(len(output_names)),
                   key=lambda i: (evaluation.per_channel_onset_seconds.get(output_names[i]) is None,
                                  evaluation.per_channel_onset_seconds.get(output_names[i], float("inf"))))
    matrix = evaluation.per_channel_score[:, order].T
    times = evaluation.window.times_seconds
    im = ax.imshow(matrix, aspect="auto", cmap="inferno", vmin=0,
                   vmax=max(evaluation.threshold, float(np.percentile(matrix, 99))),
                   extent=[times[0], times[-1], len(output_names), 0])
    ax.figure.colorbar(im, ax=ax, shrink=0.7, label="per-channel z")
    ax.axvspan(times[0], evaluation.washout_end_seconds, color="grey", alpha=0.3)
    ax.axvline(0, color="cyan", lw=1, ls="--")
    for row, index in enumerate(order):
        onset = evaluation.per_channel_onset_seconds.get(output_names[index])
        if onset is not None:
            ax.plot(onset, row + .5, marker="x", color="white", markersize=4)
    ax.set_yticks(np.arange(len(output_names)) + .5, [output_names[i] for i in order], fontsize=5)
    ax.set_xlabel("s from event")
    ax.set_title(f"D. Reservoir per-channel residual (washout shaded, "
                f"selection={evaluation.window.channel_selection_method})", fontsize=7)


def _panel_e_lateralization(ax_li, report: VerificationReport) -> None:
    """Lateralization index per source -- the other half of ``_panel_e_verification``'s old E1/E2 pair.

    E1 (delta-t per method vs. the annotation) is dropped from this
    composite in favour of the new panel F (structural anomaly graph on
    real DICOM slices) -- see this module's docstring for why. `delta_t`
    itself is not lost: it is still written in full to
    ``verification_report.json`` (``report.temporal``) and appears in this
    repo's dissertation-figure exports.
    """
    ax_li.axvspan(-INDETERMINATE_LI_THRESHOLD, INDETERMINATE_LI_THRESHOLD, color="0.9", zorder=0)
    sources = [entry.source for entry in report.lateralization]
    indices = [entry.index for entry in report.lateralization]
    colors = ["0.4" if not entry.arbitration_valid else
             ("crimson" if entry.side == "right" else "steelblue" if entry.side == "left" else "grey")
             for entry in report.lateralization]
    ax_li.scatter(indices, range(len(sources)), c=colors, s=60, zorder=3)
    for y, entry in enumerate(report.lateralization):
        note = f" (n={entry.right_count}/{entry.left_count})"
        if not entry.arbitration_valid:
            note += " [not arbitration-valid]"
        ax_li.text(1.05, y, note, va="center", fontsize=6)
    ax_li.set_yticks(range(len(sources)), sources, fontsize=6)
    ax_li.set_xlim(-1.15, 1.9)
    ax_li.axvline(0, color="black", lw=0.5)
    ax_li.set_xlabel("LI (right +1 / left -1)")
    ax_li.set_title("E. Lateralization index", fontsize=8)


def _panel_f_structural_graph(axes, structural_result: Any) -> None:
    """Structural anomaly graph (asymmetry + heterogeneity clusters, spatial-proximity edges) on real slices.

    Reuses ``multimodal_approach.structural_anomaly.find_top_anomaly_clusters``,
    ``.structural_graph.build_structural_anomaly_graph``, and
    ``.structural_graph._draw_structural_anomaly_graph_anatomical`` exactly
    as ``multimodal_approach.run_multimodal`` and the standalone
    ``plot_structural_anomaly_graph_anatomical`` do -- no second
    implementation, this panel only supplies smaller axes, a compact
    legend, and shorter per-panel titles (``F1``/``F2``/``F3`` instead of
    the standalone figure's own suptitle).
    """
    if structural_result is None:
        for ax in axes:
            ax.axis("off")
        axes[1].text(0.5, 0.5, "No DICOM available", ha="center", va="center", transform=axes[1].transAxes)
        return

    from multimodal_approach.structural_anomaly import find_top_anomaly_clusters
    from multimodal_approach.structural_graph import (
        _anatomical_legend_handles, _draw_structural_anomaly_graph_anatomical, build_structural_anomaly_graph,
    )

    anomaly_clusters = find_top_anomaly_clusters(structural_result)
    heterogeneity_clusters = find_top_anomaly_clusters(structural_result,
                                                        anomaly_map=structural_result.combined_heterogeneity)
    if not anomaly_clusters and not heterogeneity_clusters:
        for ax in axes:
            ax.axis("off")
        axes[1].text(0.5, 0.5, "No anomaly/heterogeneity cluster reached threshold", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=7)
        return

    graph = build_structural_anomaly_graph(anomaly_clusters, heterogeneity_clusters)
    on_slice_tolerance_mm = 3.0
    _draw_structural_anomaly_graph_anatomical(
        axes, structural_result, graph, on_slice_tolerance_mm=on_slice_tolerance_mm,
        panel_titles=["F1. Axial", "F2. Coronal", "F3. Sagittal"], title_fontsize=7.5, label_fontsize=5.0)
    axes[0].legend(handles=_anatomical_legend_handles(on_slice_tolerance_mm), loc="lower left", fontsize=5,
                   framealpha=0.85)


def plot_object_model_summary(data: np.ndarray, sfreq: float, names: list[str], event: Any,
                              process: BrainProcess, graph: Any, structural_result: Any, evaluation: Any,
                              report: VerificationReport, output: str | Path,
                              baseline_seconds: float = 30.0, analysis_seconds: float = 8.0) -> Path:
    """Render the six-panel object-model summary figure.

    ``graph``/``structural_result``/``evaluation`` may each be ``None`` (no
    involved channels / no DICOM / no reservoir run) — the corresponding
    panel then says so explicitly rather than raising. Mandatory caption
    text at the bottom of the figure states ``report.channel_selection``
    (with a warning glyph when it is ``"recruitment"``, since that mode's
    reservoir lateralization is not an independent cross-check — see
    ``LateralizationEstimate.arbitration_valid``), whether the recording was
    cropped, the DICOM masking method, and the fixed research-candidate
    status line every figure in this repository carries.

    Row 3 (F1/F2/F3) is the structural anomaly graph on real DICOM slices
    (``_panel_f_structural_graph``) — it needs three roughly-square axes to
    read at all, so it gets its own full-width row rather than sharing a
    cell the way the panel it replaced (the old E1 temporal-accuracy chart)
    did.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(24, 21))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.0, 1.05], hspace=0.45, wspace=0.35)
    _panel_a_timeseries(fig.add_subplot(grid[0, 0]), data, sfreq, names, event, process,
                        baseline_seconds, analysis_seconds)
    _panel_b_graph(fig.add_subplot(grid[0, 1]), graph)
    _panel_c_dicom(fig.add_subplot(grid[0, 2]), structural_result)
    _panel_d_residual(fig.add_subplot(grid[1, 0]), evaluation)
    _panel_e_lateralization(fig.add_subplot(grid[1, 1:]), report)
    f_grid = grid[2, :].subgridspec(1, 3, wspace=0.25)
    _panel_f_structural_graph([fig.add_subplot(f_grid[0, i]) for i in range(3)], structural_result)

    selection_note = report.channel_selection or "n/a"
    if report.channel_selection == "recruitment":
        selection_note += " [WARNING: reservoir channels chosen by the same recruitment analysis " \
                          "being cross-checked -- lateralization agreement is not independent]"
    caption = (
        f"channel_selection={selection_note} | crop_applied={report.crop_applied} "
        f"(crop_end_seconds={report.crop_end_seconds}) | masking_method={report.masking_method} | "
        f"prior_used={report.prior_used}\n"
        "Candidates for expert review, not a diagnosis or medical device.")
    fig.suptitle(f"Object model — {event.label!r} at {event.time_seconds:.3f}s", fontsize=12, y=0.995)
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9))

    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output
