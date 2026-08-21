"""Figures for the reservoir plant: the model itself, and its dynamics.

Two kinds, matching what was asked of it: a *model* figure (the state-space
block diagram and the reservoir's own connectivity graph) and *dynamic*
figures (hidden-state trajectory, predicted-vs-real output, and the
extreme-event residual score). Every figure carries a legend for its
markers/edges/colours and a boxed caption explaining what it shows, the same
convention ``extreme_event_agent.edf_workflow`` uses for its own figures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from extreme_event_agent.edf_workflow import _caption

from .plant import ReservoirEvaluation


def plot_model_architecture(evaluation: ReservoirEvaluation, output: str | Path) -> Path:
    """State-space block diagram: u(t) -> [B, A] -> x(t) -> [C, D] -> y(t).

    A schematic, not a data plot: boxes for the input, the hidden reservoir
    state (with its own self-loop for the recurrent state equation), and the
    output, annotated with the actual channel names and dimensions of this
    run so it documents *this* model instance, not a generic template.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    window = evaluation.window
    esn = evaluation.esn
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", linewidth=1.5,
                               edgecolor="0.2", facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)
        return (x, y, w, h)

    input_box = box(0.3, 2.3, 2.6, 1.6,
                    f"Input u(t)\nexogenous: {', '.join(window.input_names)}\n"
                    f"+ {esn.config.output_feedback_lag}-step delay embedding of y(t)\n"
                    f"dim = {esn.n_inputs}", "#cfe8ff")
    state_box = box(4.6, 2.0, 3.2, 2.2,
                    f"Hidden state x(t)\n(reservoir)\ndim = {esn.config.n_reservoir}\n"
                    f"spectral radius = {esn.achieved_spectral_radius:.2f}\nleak rate = {esn.config.leak_rate}",
                    "#ffe6b3")
    output_box = box(9.1, 2.3, 2.6, 1.6,
                     f"Output y(t)\n(predicted channels)\ndim = {esn.n_outputs}\n"
                     f"({window.channel_selection_method})", "#d9f2d9")

    def arrow(p0, p1, label, style="-|>", connectionstyle="arc3,rad=0.0", color="0.2"):
        patch = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16, linewidth=1.6,
                                color=color, connectionstyle=connectionstyle)
        ax.add_patch(patch)
        mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        ax.text(mid[0], mid[1] + 0.25, label, ha="center", fontsize=8, color=color)

    arrow((input_box[0] + input_box[2], input_box[1] + input_box[3] / 2),
         (state_box[0], state_box[1] + state_box[3] / 2), "B = W_in")
    arrow((state_box[0] + state_box[2], state_box[1] + state_box[3] / 2),
         (output_box[0], output_box[1] + output_box[3] / 2), "C (from x)")
    arrow((input_box[0] + input_box[2] / 2, input_box[1] + input_box[3]),
         (output_box[0] + output_box[2] / 2, output_box[1] + output_box[3]),
         "D (direct feedthrough)", connectionstyle="arc3,rad=-0.35", color="0.45")
    arrow((state_box[0] + state_box[2] * 0.75, state_box[1]),
         (state_box[0] + state_box[2] * 0.25, state_box[1]), "A = W (recurrent)",
         connectionstyle="arc3,rad=1.4", color="firebrick")

    ax.set_title("Reservoir plant — state-space block diagram\n"
                f"x(t) = (1-leak)x(t-1) + leak·tanh(B·u(t) + A·x(t-1) + bias)   |   y(t) = C·x(t) + D·u(t)")
    _caption(ax, "Only C and D (the readout) are trained, by ridge regression on the\n"
                "pre-event baseline; A (recurrent) and B (input) are fixed random weights\n"
                "set once at construction — reservoir computing's defining trick. u(t)'s\n"
                "delay-embedded taps (NARX-style) give the readout the real signal's own\n"
                "recent past to work from; the clock alone carries almost no information\n"
                "about fast EEG structure.",
            loc="lower left")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight"); plt.close(fig)
    return output


def plot_reservoir_connectivity(evaluation: ReservoirEvaluation, output: str | Path,
                                max_hidden_nodes: int = 60, edge_percentile: float = 97.0,
                                seed: int = 7) -> Path:
    """The reservoir's own random recurrent graph — the literal hidden-state network.

    All ``n_reservoir`` units would be unreadable in one figure, so a random
    (seeded) sample of ``max_hidden_nodes`` is drawn, together with every
    input/output node and only their strongest edges (top
    ``100 - edge_percentile`` % by magnitude), so the figure shows the
    heaviest structure rather than a uniform hairball. Only the *exogenous*
    input (``window.input_names``, the ``MKR...`` clock) gets its own input
    nodes here — the reservoir's much larger delay-embedded output-feedback
    input block (see :func:`model.plant._build_augmented_input`) is
    deliberately not drawn node-by-node (it would add dozens of near-identical
    nodes and drown out everything else); ``plot_model_architecture`` is
    where that feedback path is documented instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx

    esn, window = evaluation.esn, evaluation.window
    rng = np.random.default_rng(seed)
    n = esn.config.n_reservoir
    hidden_sample = np.sort(rng.choice(n, size=min(max_hidden_nodes, n), replace=False))
    hidden_nodes = [f"h{i}" for i in hidden_sample]

    graph = nx.DiGraph()
    graph.add_nodes_from(window.input_names, kind="input")
    graph.add_nodes_from(hidden_nodes, kind="hidden")
    graph.add_nodes_from(window.output_names, kind="output")

    recurrent = esn.W[np.ix_(hidden_sample, hidden_sample)]
    hidden_threshold = np.percentile(np.abs(recurrent), edge_percentile)
    for row, i in enumerate(hidden_sample):
        for col, j in enumerate(hidden_sample):
            weight = recurrent[row, col]
            if abs(weight) >= hidden_threshold and i != j:
                graph.add_edge(f"h{i}", f"h{j}", weight=float(weight), kind="recurrent")

    exogenous_count = len(window.input_names)
    input_weights = esn.W_in[hidden_sample, :exogenous_count]
    input_threshold = np.percentile(np.abs(input_weights), edge_percentile - 10)
    for row, i in enumerate(hidden_sample):
        for col, name in enumerate(window.input_names):
            weight = input_weights[row, col]
            if abs(weight) >= input_threshold:
                graph.add_edge(name, f"h{i}", weight=float(weight), kind="input")

    if esn.W_out is not None:
        readout_hidden = esn.W_out[:, hidden_sample]
        readout_threshold = np.percentile(np.abs(readout_hidden), edge_percentile - 10)
        for out_row, out_name in enumerate(window.output_names):
            for col, i in enumerate(hidden_sample):
                weight = readout_hidden[out_row, col]
                if abs(weight) >= readout_threshold:
                    graph.add_edge(f"h{i}", out_name, weight=float(weight), kind="readout")

    pos = nx.spring_layout(graph, seed=seed, weight="weight", k=1.4 / np.sqrt(max(len(graph), 1)))
    fig, ax = plt.subplots(figsize=(13, 11))

    for kind, color, width in (("recurrent", "0.75", 0.6), ("input", "#1f77b4", 1.4),
                               ("readout", "#2ca02c", 1.4)):
        edges = [(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == kind]
        extra = dict(arrows=True, arrowsize=8) if kind != "recurrent" else dict(arrows=False)
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=color, width=width,
                               alpha=0.6, ax=ax, **extra)

    nx.draw_networkx_nodes(graph, pos, nodelist=window.input_names, node_shape="s", node_size=380,
                           node_color="#1f77b4", ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=hidden_nodes, node_shape="o", node_size=90,
                           node_color="#ffb84d", edgecolors="0.4", linewidths=0.5, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=window.output_names, node_shape="D", node_size=260,
                           node_color="#2ca02c", ax=ax)
    nx.draw_networkx_labels(graph, pos, labels={name: name for name in window.input_names + window.output_names},
                            font_size=7, ax=ax)

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4", markersize=11,
              label=f"Input u(t) — {len(window.input_names)} MKR channel(s)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffb84d", markersize=9,
              label=f"Hidden reservoir unit — {len(hidden_sample)}/{n} shown"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c", markersize=10,
              label=f"Output y(t) — {len(window.output_names)} channel(s)"),
        Line2D([0], [0], color="0.75", lw=1.5, label="Recurrent edge (A = W), strongest "
              f"{100 - edge_percentile:.0f}%"),
        Line2D([0], [0], color="#1f77b4", lw=1.5, label="Input edge (B = W_in)"),
        Line2D([0], [0], color="#2ca02c", lw=1.5, label="Readout edge (C, from x to y)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8,
             title="Legend", frameon=True)
    ax.set_title(f"Reservoir connectivity — {n}-unit hidden state, "
                f"{max(len(hidden_sample), 0)} sampled for display\n"
                f"spectral radius(A) = {esn.achieved_spectral_radius:.3f}, sparsity = {esn.config.sparsity} "
                f"(+ {esn.n_inputs - exogenous_count} delay-embedding input dims not drawn — see architecture)")
    ax.axis("off")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight"); plt.close(fig)
    return output


def plot_reservoir_spectrum(evaluation: ReservoirEvaluation, output: str | Path) -> Path:
    """Eigenvalues of ``A = W`` in the complex plane, against the unit circle.

    The standard echo-state-property check as a picture: the reservoir is
    guaranteed contracting (forgets initial conditions, only the driven
    trajectory survives) when every eigenvalue sits inside the unit circle,
    i.e. spectral radius < 1; near or beyond it (as configured here by
    default, 0.95) the system runs close to the edge of stability, where
    memory is longest but explicit leak-rate integration is what keeps it
    from diverging.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eigenvalues = np.linalg.eigvals(evaluation.esn.W)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    circle = plt.Circle((0, 0), 1.0, fill=False, color="0.5", lw=1.2, ls="--")
    ax.add_artist(circle)
    ax.scatter(eigenvalues.real, eigenvalues.imag, s=18, color="firebrick", alpha=0.8,
              label="eigenvalues of A = W")
    ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8)
    limit = max(1.15, float(np.max(np.abs(eigenvalues))) * 1.1)
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set(xlabel="Re(eigenvalue)", ylabel="Im(eigenvalue)",
          title=f"Reservoir stability spectrum — spectral radius = "
                f"{evaluation.esn.achieved_spectral_radius:.3f}")
    ax.legend(loc="upper right", fontsize=8)
    _caption(ax, "Dashed circle = unit circle (radius 1).\n"
                "Points inside = the echo-state property holds without leaky\n"
                "integration; this reservoir relies on leak_rate for stability\n"
                "if its spectral radius is at or beyond 1.", loc="lower left")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170); plt.close(fig)
    return output


def plot_hidden_state_dynamics(evaluation: ReservoirEvaluation, output: str | Path) -> Path:
    """Heatmap of the hidden state ``x(t)`` across every reservoir unit and timestep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = evaluation.window
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(evaluation.hidden_states.T, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1,
                   extent=[window.times_seconds[0], window.times_seconds[-1],
                          evaluation.esn.config.n_reservoir, 0])
    ax.axvspan(window.times_seconds[0], evaluation.washout_end_seconds, color="0.5", alpha=0.25,
              label="washout (x(0)=0 transient)")
    ax.axvline(0.0, color="black", lw=1.5, ls="--", label="resolved event")
    ax.set(xlabel=f"Time relative to event (s) — {window.event.label!r} at {window.event.time_seconds:.3f} s",
          ylabel="Reservoir unit index",
          title="Hidden state x(t) — every reservoir unit's activation over time")
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("tanh-bounded activation")
    ax.legend(loc="upper right", fontsize=8)
    _caption(ax, "Each row = one reservoir unit's trajectory (bounded in [-1, 1] by tanh).\n"
                "A visible break in the texture at the dashed line means the driven\n"
                "state itself changed character at the event, not just the readout.",
            loc="lower left")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170); plt.close(fig)
    return output


def plot_output_prediction(evaluation: ReservoirEvaluation, output: str | Path,
                           max_channels_shown: int = 12) -> Path:
    """Small multiples: real vs. baseline-trained-model-predicted output, per channel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = evaluation.window
    names = window.output_names[:max_channels_shown]
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.6 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, names):
        index = window.output_names.index(name)
        axis.plot(window.times_seconds, window.output_data[:, index], color="black", lw=0.8, label="real y(t)")
        axis.plot(window.times_seconds, evaluation.predicted_output[:, index], color="firebrick", lw=0.8,
                 alpha=0.8, label="baseline-model ŷ(t)")
        axis.axvline(0.0, color="0.5", lw=1.0, ls="--")
        axis.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
        axis.set_yticks([])
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel(f"Time relative to event (s) — {window.event.label!r} at "
                        f"{window.event.time_seconds:.3f} s")
    fig.suptitle("Output y(t): real recording vs. baseline-trained plant prediction")
    _caption(axes[0], "Model trained only on t < 0 (baseline); divergence after t = 0\n"
                     "is the residual the extreme-event score below is built from.", loc="upper left")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170); plt.close(fig)
    return output


def plot_residual_heatmap(evaluation: ReservoirEvaluation, output: str | Path) -> Path:
    """Per-channel, per-timestep difference: real y(t) minus baseline-trained ŷ(t).

    ``plot_output_prediction`` overlays real and predicted traces per
    channel — clear for a handful of channels, but hard to compare
    quantitatively once there are more than two or three. This is the same
    ``evaluation.residual`` collapsed into one heatmap instead: every output
    channel's residual, independently median/MAD-normalized against its own
    *baseline* segment (the exact per-channel analogue of how
    ``run_reservoir_plant`` builds the aggregated ``score``), so channels
    with very different native amplitudes are still visually comparable on
    one shared colour scale. Where ``plot_extreme_event_score`` answers
    "when does the model stop explaining the data" with one number, this
    answers "which channels, specifically".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = evaluation.window
    baseline_mask = (window.times_seconds >= evaluation.washout_end_seconds) & (window.times_seconds < 0.0)
    residual = evaluation.residual
    center = np.median(residual[baseline_mask], axis=0)
    deviation = np.abs(residual[baseline_mask] - center)
    mad = 1.4826 * np.median(deviation, axis=0)
    scale = np.where(mad > 1e-15, mad, np.std(residual[baseline_mask], axis=0) + 1e-15)
    channel_z = (residual - center) / scale

    vmax = max(6.0, float(np.percentile(np.abs(channel_z[baseline_mask]), 99.5)) * 3)
    names = window.output_names
    fig, ax = plt.subplots(figsize=(13, max(4, 0.35 * len(names) + 1)))
    im = ax.imshow(channel_z.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[window.times_seconds[0], window.times_seconds[-1], len(names), 0])
    ax.axvspan(window.times_seconds[0], evaluation.washout_end_seconds, color="0.5", alpha=0.15,
              label="washout (excluded from baseline stats)")
    ax.axvline(0.0, color="black", lw=1.5, ls="--", label="resolved event")
    ax.set_yticks(np.arange(len(names)) + 0.5, names, fontsize=8)
    ax.set(xlabel=f"Time relative to event (s) — {window.event.label!r} at {window.event.time_seconds:.3f} s",
          ylabel="Output channel",
          title="Real minus predicted output — per-channel residual heatmap")
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("residual z-score (per-channel median/MAD, baseline-normalized)")
    ax.legend(loc="upper right", fontsize=8)
    _caption(ax, "Blue/red = model under-/over-predicts that channel at that instant\n"
                "(z-scored per channel against its own baseline); white ~ 0 means the\n"
                "baseline-trained model still explains that channel there.",
            loc="lower left")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170); plt.close(fig)
    return output


def plot_extreme_event_score(evaluation: ReservoirEvaluation, output: str | Path) -> Path:
    """The model's own extreme-event verdict: residual score vs. threshold, over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = evaluation.window
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axvspan(window.times_seconds[0], evaluation.washout_end_seconds, color="0.5", alpha=0.2,
              label="washout (excluded from detection)")
    ax.plot(window.times_seconds, evaluation.score, color="0.75", lw=0.6, label="raw residual score")
    ax.plot(window.times_seconds, evaluation.smoothed_score, color="navy", lw=1.3,
           label="smoothed score (detection basis)")
    ax.axhline(evaluation.threshold, color="firebrick", lw=1.2, ls="--",
              label=f"detection threshold ({evaluation.threshold:.1f} MAD)")
    ax.axvline(0.0, color="black", lw=1.2, ls=":", label="resolved event")
    if evaluation.onset_time_seconds is not None:
        ax.axvline(evaluation.onset_time_seconds, color="darkorange", lw=1.4,
                  label=f"model's own onset ({evaluation.onset_time_seconds:+.3f} s)")
    ax.scatter([evaluation.peak_time_seconds], [evaluation.peak_score], color="black", zorder=5,
              label=f"peak ({evaluation.peak_score:.1f} MAD @ {evaluation.peak_time_seconds:+.3f} s)")
    ax.set(xlabel="Time relative to event (s)", ylabel="median/MAD z-score of output residual",
          title="Reservoir-plant extreme-event score — "
                f"{'DETECTED' if evaluation.detected else 'not detected'} at this threshold")
    # The washout transient can dwarf every other value; let it run off the top of the
    # plot (its axvspan already marks it as excluded) rather than compress everything
    # else — real baseline/event structure, not that transient, is what this figure is for.
    post_washout = evaluation.score[window.times_seconds > evaluation.washout_end_seconds]
    y_ceiling = max(evaluation.threshold * 1.2, float(np.max(post_washout)) * 1.15, 1.0)
    ax.set_ylim(top=y_ceiling)
    ax.legend(loc="upper left", fontsize=8)
    _caption(ax, "Score = robust z-score of ||real y(t) - predicted ŷ(t)||, normalized\n"
                "against its own baseline (t < 0) segment; ~0 while the baseline-trained\n"
                "model still explains the data, large where it stops explaining it.\n"
                "Detection requires the smoothed line, not a single raw spike, above threshold.",
            loc="lower right")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170); plt.close(fig)
    return output


def plot_all(evaluation: ReservoirEvaluation, output_dir: str | Path, stem: str) -> dict[str, Path]:
    """Render every figure above, named ``<stem>_<figure>.png``. Returns ``{name: path}``."""
    output_dir = Path(output_dir)
    return {
        "architecture": plot_model_architecture(evaluation, output_dir / f"{stem}_architecture.png"),
        "connectivity": plot_reservoir_connectivity(evaluation, output_dir / f"{stem}_connectivity.png"),
        "spectrum": plot_reservoir_spectrum(evaluation, output_dir / f"{stem}_spectrum.png"),
        "hidden_state": plot_hidden_state_dynamics(evaluation, output_dir / f"{stem}_hidden_state.png"),
        "output_prediction": plot_output_prediction(evaluation, output_dir / f"{stem}_output_prediction.png"),
        "residual_heatmap": plot_residual_heatmap(evaluation, output_dir / f"{stem}_residual_heatmap.png"),
        "extreme_event_score": plot_extreme_event_score(
            evaluation, output_dir / f"{stem}_extreme_event_score.png"),
    }
