"""Graph representation of structural-anomaly clusters, the DICOM-side analogue of
``extreme_event_agent.edf_workflow.build_seizure_graph``.

The EEG graph connects channel nodes by how their *time series* co-vary
(Pearson correlation of 13-80 Hz z-score traces, threshold- and
top-k-pruned). A static post-implant MRI has no time axis to correlate over
(see the package README, "Why this, and not full electrode localization" /
the top-level README's EDF-vs-DICOM temporal-resolution table) — the one
relationship distance alone can support here is spatial proximity between
the anomaly/heterogeneity clusters ``find_top_anomaly_clusters`` already
extracts. ``build_structural_anomaly_graph`` prunes that the same way
(distance threshold + top-k-per-node) so the two graphs share one
graph-construction convention even though they are built from different
physical quantities: kept alongside each other, never merged into a single
cross-modal graph, matching this repo's "never merge into one score"
discipline (see ``structural_anomaly.py``'s ``combined_anomaly`` vs.
``combined_heterogeneity``, and ``object_model/graph.py``'s three separate
attribute layers).
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["build_structural_anomaly_graph", "plot_structural_anomaly_graph",
          "plot_structural_anomaly_graph_anatomical"]

_CHANNEL_COLORS = {"asymmetry": "crimson", "heterogeneity": "darkorange"}
_HEMISPHERE_RING_COLORS = {"right": "steelblue", "left": "seagreen"}


def _cluster_strength(cluster: dict[str, object]) -> float:
    """``mean_abs_anomaly`` for asymmetry clusters, ``mean_heterogeneity_z`` for the other channel.

    ``find_top_anomaly_clusters`` reports the same key names regardless of
    which map it ranked (see its docstring), except this one: the asymmetry
    channel's summary field is ``mean_abs_anomaly`` (signed z, reported
    unsigned) while the heterogeneity channel's own summary uses
    ``mean_heterogeneity_z`` (already one-sided, see ``structural_anomaly.py``).
    Cluster dicts themselves only ever carry ``mean_abs_anomaly`` (see
    ``find_top_anomaly_clusters``), so this falls back to it either way —
    named here only to make the "which channel this came from" ambiguity
    explicit rather than silent.
    """
    value = cluster.get("mean_abs_anomaly")
    return float(value) if value is not None else 0.0


def build_structural_anomaly_graph(anomaly_clusters: list[dict], heterogeneity_clusters: list[dict],
                                   distance_threshold_mm: float = 40.0, top_k_per_node: int = 3) -> Any:
    """Build a NetworkX graph of the top asymmetry/heterogeneity clusters, connected by spatial proximity.

    ``anomaly_clusters``/``heterogeneity_clusters`` are
    ``find_top_anomaly_clusters`` results (asymmetry channel and
    heterogeneity channel respectively — see ``structural_anomaly.py``,
    "A second, independent channel", for why these are two separate lists
    rather than one). Each cluster becomes one node, labelled ``asym_<i>``/
    ``het_<i>`` in rank order; nodes carry ``channel`` (``"asymmetry"`` or
    ``"heterogeneity"``, the one attribute distinguishing which map found
    it), ``hemisphere``, ``voxel_count``, ``total_mass``, ``strength``
    (``mean_abs_anomaly``), ``peak_value``, and the peak voxel's patient
    coordinates as three plain floats (``x_mm``/``y_mm``/``z_mm`` — not a
    tuple, since GraphML attributes must be scalar).

    An edge is drawn between two clusters only when their peak-voxel
    Euclidean distance is at most ``distance_threshold_mm`` *and* each is
    among the other's ``top_k_per_node`` closest neighbours — the same
    threshold-then-top-k pruning ``build_seizure_graph`` applies to its
    co-activation mesh, substituting distance for correlation magnitude
    (closer takes the place of "more correlated"). Edge ``kind`` is always
    ``"proximity"`` (there is no second edge kind here — no recruitment
    order exists without a time axis), weighted ``1 / (1 + distance_mm)`` so
    closer pairs draw heavier, with the raw ``distance_mm`` also stored for
    direct inspection.

    Raises ``ValueError`` if both cluster lists are empty — nothing to
    graph, the same guard ``build_seizure_graph`` applies when
    ``process.onset_latency_seconds`` is empty.
    """
    import networkx as nx

    entries: list[tuple[str, str, dict]] = (
        [(f"asym_{i}", "asymmetry", cluster) for i, cluster in enumerate(anomaly_clusters)]
        + [(f"het_{i}", "heterogeneity", cluster) for i, cluster in enumerate(heterogeneity_clusters)]
    )
    if not entries:
        raise ValueError("No anomaly or heterogeneity clusters to graph.")

    graph = nx.Graph(distance_threshold_mm=distance_threshold_mm, top_k_per_node=top_k_per_node)
    positions = np.zeros((len(entries), 3))
    node_ids = [node_id for node_id, _, _ in entries]
    for row, (node_id, channel, cluster) in enumerate(entries):
        x_mm, y_mm, z_mm = (float(c) for c in cluster["peak_patient_xyz_mm"])
        positions[row] = (x_mm, y_mm, z_mm)
        graph.add_node(
            node_id, channel=channel, hemisphere=cluster["hemisphere"],
            voxel_count=int(cluster["voxel_count"]), total_mass=float(cluster["total_mass"]),
            strength=_cluster_strength(cluster), peak_value=float(cluster["peak_value"]),
            x_mm=x_mm, y_mm=y_mm, z_mm=z_mm,
        )

    n = len(entries)
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)

    selected: set[tuple[int, int]] = set()
    for row in range(n):
        candidates = np.flatnonzero(distance[row] <= distance_threshold_mm)
        if candidates.size:
            closest = candidates[np.argsort(distance[row, candidates])[:top_k_per_node]]
            selected.update(tuple(sorted((row, int(col)))) for col in closest if col != row)
    for row, col in selected:
        d = float(distance[row, col])
        graph.add_edge(node_ids[row], node_ids[col], kind="proximity", weight=1.0 / (1.0 + d),
                       distance_mm=d)
    return graph


def plot_structural_anomaly_graph(graph: Any, output: str, seed: int = 7) -> str:
    """Render a ``build_structural_anomaly_graph`` result as a node-link figure.

    Three independent visual encodings, kept separate rather than merged —
    the same principle ``plot_seizure_graph`` follows for the EEG graph:
    **fill colour = channel** (crimson for ``asymmetry``, orange for
    ``heterogeneity`` — which map found this cluster), **ring colour =
    hemisphere** (steel blue right, sea green left — spatial, from the
    cluster's own peak-voxel position, not a prior), **size = strength**
    (``mean_abs_anomaly``/``mean_heterogeneity_z``, bigger = stronger). A
    node whose fill and ring are both large is a strong finding on a
    specific side; two same-ring nodes of different fill colour close
    together are where the asymmetry and heterogeneity channels agree on
    roughly the same location, the interesting case this graph exists to
    surface (see the module docstring: the two channels look for
    structurally different things and are not expected to usually agree).
    Proximity edges are drawn grey, heavier for closer pairs. A legend
    identifies every marker, and a boxed caption states the pruning
    parameters this graph was built with.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx

    pos = nx.spring_layout(graph, seed=seed, weight="weight")

    strengths = [data["strength"] for _, data in graph.nodes(data=True)]
    vmax = max(strengths) if strengths and max(strengths) > 0 else 1.0
    sizes = [90 + 360 * min(data["strength"] / vmax, 1.0) for _, data in graph.nodes(data=True)]

    fig, ax = plt.subplots(figsize=(10, 9))
    edges = list(graph.edges(data=True))
    if edges:
        widths = [0.5 + 2.5 * data["weight"] for _, _, data in edges]
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for u, v, _ in edges],
                               width=widths, edge_color="0.6", alpha=0.7, ax=ax)

    for node, data in graph.nodes(data=True):
        ring = _HEMISPHERE_RING_COLORS.get(data["hemisphere"], "0.5")
        nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_shape="o",
                               node_size=[90 + 360 * min(data["strength"] / vmax, 1.0)],
                               node_color=_CHANNEL_COLORS.get(data["channel"], "grey"),
                               edgecolors=ring, linewidths=2.2, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor="black",
               markersize=10, label=f"{channel} cluster")
        for channel, color in _CHANNEL_COLORS.items()
    ] + [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=color,
               markeredgewidth=2.2, markersize=10, label=f"{hemi} hemisphere")
        for hemi, color in _HEMISPHERE_RING_COLORS.items()
    ] + [Line2D([0], [0], color="0.6", label="spatial proximity")]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Structural anomaly graph — clusters connected by spatial proximity, not time")
    ax.set_axis_off()
    caption = (
        f"distance_threshold_mm={graph.graph['distance_threshold_mm']:.0f}, "
        f"top_k_per_node={graph.graph['top_k_per_node']} — no temporal edge kind exists here "
        "(static scan, no time axis); compare against build_seizure_graph's co-activation mesh."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
              bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9))
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_structural_anomaly_graph_anatomical(result: Any, graph: Any, output: str,
                                             fade_range_mm: float = 40.0,
                                             on_slice_tolerance_mm: float = 3.0) -> str:
    """Overlay ``build_structural_anomaly_graph`` on three real DICOM slices, like a DICOM viewer.

    Unlike a maximum-intensity projection (which shows the brightest voxel
    along a whole ray, not real anatomy at any one depth), this draws one
    actual axial, one coronal, and one sagittal slice — exactly what a
    DICOM viewer's three-pane layout shows — all three cut through the
    *same* physical point, the graph's strongest node (largest
    ``total_mass``), the same "all three projections through one point"
    convention ``run_multimodal._plot_overview`` already uses for a single
    cluster.

    Every graph node is projected onto every one of the three slices at its
    own peak-voxel position (unavoidable for any node not physically on that
    slice — there is no way to show a 3-D graph on 2-D anatomy without
    projecting something), but nodes are no longer treated as uniformly
    visible the way the discarded max-intensity-projection version did:
    each node's ``depth_mm`` — its true distance, along that view's
    out-of-plane axis, from the slice actually shown — is computed and used
    to (a) fade the node's opacity toward (but never to) a floor as
    ``depth_mm`` grows past ``fade_range_mm``, (b) switch its ring from a
    solid line (within ``on_slice_tolerance_mm`` of the slice — physically
    on it) to a dashed one (off-slice, i.e. projected), and (c) label
    off-slice nodes with their own ``Δ<depth_mm>mm`` rather than silently
    implying they are all equally present in the image. Edges fade by the
    deeper of their two endpoints. Same colour encoding throughout as
    ``plot_structural_anomaly_graph`` (fill = channel, ring colour =
    hemisphere, size = strength).

    This is the honest middle ground between the two easy failure modes: a
    single slice with no fade would silently teleport far-away nodes onto
    it (implying a false precision this projection cannot have); a max
    projection (this function's previous implementation) shows every voxel's
    brightest value across the whole depth, which is not what any single
    DICOM slice actually looks like. Depth is disclosed, not hidden.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    nodes = list(graph.nodes(data=True))
    voxel_kij = {
        node: result.t1_geometry.patient_to_voxel(np.array([data["x_mm"], data["y_mm"], data["z_mm"]]))
        for node, data in nodes
    }

    # Reference point: the strongest node's own voxel position, rounded to
    # the nearest slice index -- one physical point, shared by all three views.
    ref_node = max(nodes, key=lambda nd: nd[1]["total_mass"])[0]
    k0, i0, j0 = (int(round(c)) for c in voxel_kij[ref_node])

    volume = result.t1_geometry.volume
    vmin, vmax = np.percentile(volume, [1, 99])
    voxel_size_k = float(np.linalg.norm(result.t1_geometry.d_slice))
    voxel_size_i = float(np.linalg.norm(result.t1_geometry.d_row))
    voxel_size_j = float(np.linalg.norm(result.t1_geometry.d_col))

    strengths = [data["strength"] for _, data in nodes]
    node_vmax = max(strengths) if strengths and max(strengths) > 0 else 1.0

    def _size(data: dict) -> float:
        return 60 + 220 * min(data["strength"] / node_vmax, 1.0)

    def _alpha(depth_mm: float) -> float:
        return float(np.clip(1.0 - depth_mm / fade_range_mm, 0.2, 1.0))

    # (title, background slice, horizontal axis, vertical axis, depth-from-slice in mm).
    panels = [
        ("Axial (top-down)\nthrough " + ref_node, volume[k0, :, :], 2, 1,
         lambda kij: abs(kij[0] - k0) * voxel_size_k),
        ("Coronal (front-back)\nthrough " + ref_node, volume[:, i0, :], 2, 0,
         lambda kij: abs(kij[1] - i0) * voxel_size_i),
        ("Sagittal (side)\nthrough " + ref_node, volume[:, :, j0], 1, 0,
         lambda kij: abs(kij[2] - j0) * voxel_size_j),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    for ax, (title, background, h_axis, v_axis, depth_fn) in zip(axes, panels):
        ax.imshow(background, cmap="gray", vmin=vmin, vmax=vmax)
        for u, v, edge_data in graph.edges(data=True):
            depth = max(depth_fn(voxel_kij[u]), depth_fn(voxel_kij[v]))
            ux, uy = voxel_kij[u][h_axis], voxel_kij[u][v_axis]
            vx, vy = voxel_kij[v][h_axis], voxel_kij[v][v_axis]
            ax.plot([ux, vx], [uy, vy], color="0.8", linewidth=0.5 + 2.5 * edge_data["weight"],
                    alpha=_alpha(depth), zorder=1)
        # Cycle label offsets so nodes that land close together in this
        # projection (unavoidable — see the docstring's depth-collapse
        # caveat) don't render as illegibly stacked text.
        label_offsets = [(0, 9), (0, -13), (14, 9), (-14, -13), (14, -13), (-14, 9)]
        for node_index, (node, data) in enumerate(nodes):
            kij = voxel_kij[node]
            depth = depth_fn(kij)
            on_slice = depth <= on_slice_tolerance_mm
            alpha = 1.0 if on_slice else _alpha(depth)
            x, y = kij[h_axis], kij[v_axis]
            ring = _HEMISPHERE_RING_COLORS.get(data["hemisphere"], "0.5")
            ax.scatter(x, y, s=_size(data), c=_CHANNEL_COLORS.get(data["channel"], "grey"),
                       edgecolors=ring, linewidths=2.4 if on_slice else 1.3,
                       linestyle="solid" if on_slice else (0, (2, 1)), alpha=alpha, zorder=2)
            label = node if on_slice else f"{node} (Δ{depth:.0f}mm)"
            offset = label_offsets[node_index % len(label_offsets)]
            ax.annotate(label, (x, y), fontsize=6, color="white", alpha=max(alpha, 0.6),
                        ha="center", va="center", zorder=3, xytext=offset, textcoords="offset points")
        ax.set_title(title, fontsize=9.5)
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor="black",
               markersize=9, label=f"{channel} cluster")
        for channel, color in _CHANNEL_COLORS.items()
    ] + [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=color,
               markeredgewidth=2.0, markersize=9, label=f"{hemi} hemisphere")
        for hemi, color in _HEMISPHERE_RING_COLORS.items()
    ] + [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=2.4, markersize=9, label="on this slice (≤%.0fmm)" % on_slice_tolerance_mm),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.3, markersize=9, label="off-slice, projected + faded"),
        Line2D([0], [0], color="0.8", label="proximity edge (real 3-D distance)"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=7.5, framealpha=0.9)
    fig.suptitle("Structural anomaly graph on real DICOM slices — three views through one point")
    fig.text(0.5, 0.02,
             f"Solid ring + full label = physically on this slice (≤{on_slice_tolerance_mm:.0f} mm); "
             "dashed ring + faded + Δ-labelled = off-slice, projected here only for layout — "
             "its real position is elsewhere along this view's depth axis, not on this image.",
             ha="center", fontsize=8.5,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9))
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output
