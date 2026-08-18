"""EDF orchestration, whole-recording plots, and event-centred analysis."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from scipy import signal

from .agent import ExtremeEventAgent
from .models import (AnnotatedEvent, BrainProcess, ClinicalEvent, DetectedEvent, DetectionReport,
                     EdfRunResult, Event)

MARKER = re.compile(r"^MKR\s*\d+\+?$", re.IGNORECASE)
RIGHT_FRONTAL = re.compile(r"(?:PM\s*[3-8]|CC\s*(?:8|9|10))(?:\D|$)", re.IGNORECASE)
# This dataset's annotation channel is Windows-1251 Cyrillic; MNE's default UTF-8
# decode raises on it ("Encountered invalid byte..."). cp1251 is a superset of
# ASCII, so it also decodes plain-ASCII channel names and annotations correctly.
EDF_ENCODING = "cp1251"
SEIZURE_KEYWORDS = ("приступ", "судорог", "seizure", "ictal", "бткп", "tcs")
ANNOTATION_CLUSTER_GAP_SECONDS = 10.0


def clock_time_to_offset(clock: str, recording_start: datetime, duration_seconds: float) -> float:
    """Convert ``HH:MM:SS[.ffffff]`` to seconds from the EDF start.

    Midnight rollover is supported. An out-of-recording value is rejected so a
    typo cannot silently place a clinical marker outside the plotted data.
    """
    try:
        parsed = datetime.strptime(clock, "%H:%M:%S.%f").time()
    except ValueError:
        try:
            parsed = datetime.strptime(clock, "%H:%M:%S").time()
        except ValueError as error:
            raise ValueError("Event clock must use HH:MM:SS or HH:MM:SS.ffffff.") from error
    event_datetime = datetime.combine(recording_start.date(), parsed, recording_start.tzinfo)
    if event_datetime < recording_start:
        event_datetime += timedelta(days=1)
    offset = (event_datetime - recording_start).total_seconds()
    if not 0 <= offset <= duration_seconds:
        raise ValueError(
            f"Event clock {clock} is {offset:.3f} s from EDF start, outside the "
            f"{duration_seconds:.3f} s recording.")
    return offset


def read_edf_start(path: str | Path) -> tuple[datetime, float]:
    """Read EDF start datetime and duration without preloading signal samples."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR", encoding=EDF_ENCODING)
    start = raw.info.get("meas_date")
    if start is None:
        raise ValueError(f"{path} does not contain an EDF recording start time.")
    return start, float(raw.n_times / raw.info["sfreq"])


def read_edf(path: str | Path) -> tuple[np.ndarray, float, list[str]]:
    """Load every non-marker EDF signal in volts using MNE.

    ``MKR...`` channels are excluded here, not because they are ignored, but
    because on this dataset ``MKR1+``/``MKR2+`` are a perfectly periodic 1 Hz
    square-wave hardware sync clock (every transition is exactly 0.5 s from
    the last, for the entire recording, with no anomaly around any detected
    or annotated event) — not brain signal and not an event marker. Folding a
    constant hardware clock into the statistical detector or the beta-gamma
    channel analysis would misrepresent it as neural or clinical evidence.
    Use ``read_edf_markers`` to load them for display instead.
    """
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR", encoding=EDF_ENCODING)
    names = [name for name in raw.ch_names if not MARKER.fullmatch(name.strip())]
    if not names:
        raise ValueError(f"{path} contains no non-marker signal channels.")
    return raw.get_data(picks=names), float(raw.info["sfreq"]), names


def read_edf_markers(path: str | Path) -> tuple[np.ndarray, float, list[str]]:
    """Load only the ``MKR...`` marker/trigger channels, for display purposes.

    See ``read_edf`` for why they are kept out of detection and process
    analysis. Channels are picked before loading so only their data (not the
    full multichannel recording) is read from disk.
    """
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR", encoding=EDF_ENCODING)
    names = [name for name in raw.ch_names if MARKER.fullmatch(name.strip())]
    if not names:
        return np.empty((0, raw.n_times)), float(raw.info["sfreq"]), []
    raw.pick(names)
    raw.load_data(verbose="ERROR")
    return raw.get_data(), float(raw.info["sfreq"]), names


def _cluster_seizure_annotation(onsets: list[float], descriptions: list[str]) -> AnnotatedEvent | None:
    """Pure grouping logic behind ``find_annotated_event``; see its docstring."""
    matches = {index for index, description in enumerate(descriptions)
              if any(keyword in description.lower() for keyword in SEIZURE_KEYWORDS)}
    if not matches:
        return None
    order = sorted(range(len(onsets)), key=lambda index: onsets[index])
    groups = [[order[0]]]
    for index in order[1:]:
        if onsets[index] - onsets[groups[-1][-1]] <= ANNOTATION_CLUSTER_GAP_SECONDS:
            groups[-1].append(index)
        else:
            groups.append([index])
    group = next(group for group in groups if matches & set(group))
    anchor = min((index for index in group if index in matches), key=lambda index: onsets[index])
    start, end = onsets[group[0]], onsets[group[-1]]
    return AnnotatedEvent(
        time_seconds=onsets[anchor],
        label=descriptions[anchor],
        duration_seconds=max(end - start, 0.1),
        annotations=tuple((onsets[index], descriptions[index]) for index in group),
    )


def find_annotated_event(path: str | Path) -> AnnotatedEvent | None:
    """Locate a seizure marker in the EDF's own EDF+ annotation channel.

    This reads metadata the clinician who scored the recording embedded
    directly in the file — the same way ``read_edf_start`` reads ``meas_date``
    — rather than any apriori number. An annotation counts as a seizure
    marker only when its decoded text contains one of ``SEIZURE_KEYWORDS``;
    every other annotation within ``ANNOTATION_CLUSTER_GAP_SECONDS`` of that
    match is folded into the same event, so a clinician's separate notes
    around one seizure (e.g. an "onset?" query next to a "seizure" tag) do
    not appear as unrelated events. Returns ``None`` when the file carries no
    annotations, or none of them mention a seizure.
    """
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR", encoding=EDF_ENCODING)
    onsets = [float(onset) for onset in raw.annotations.onset]
    descriptions = [str(description) for description in raw.annotations.description]
    return _cluster_seizure_annotation(onsets, descriptions)


def select_seizure_event(events: list[Event]) -> DetectedEvent | None:
    """Pick the most seizure-like candidate out of the agent's own detections.

    No expert time is consulted. A seizure is told apart from a brief
    interictal spike by its numeric footprint, not by amplitude alone: a
    sharp interictal discharge can score as high as, or higher than, the
    seizure while staying focal and short. Candidates are therefore ranked
    by channel spread first, event duration second, and raw score only as a
    final tiebreaker, so a widely and durably recruiting candidate always
    outranks a brief high-amplitude one. ``events`` is already the agent's
    finished, threshold-verified candidate list; nothing here re-detects or
    re-scores the recording.
    """
    if not events:
        return None
    best = max(events, key=lambda event: (len(event.involved_channels),
                                          event.end_seconds - event.start_seconds, event.score))
    return DetectedEvent(
        time_seconds=best.peak_seconds,
        label="data-detected extreme event (ranked by channel spread, then duration, then score)",
        duration_seconds=max(best.end_seconds - best.start_seconds, 0.1),
        involved_channel_count=len(best.involved_channels),
        score=best.score,
        confidence=best.confidence,
    )


def _beta_gamma_z_scores(data: np.ndarray, sfreq: float, event: ClinicalEvent | AnnotatedEvent | DetectedEvent,
                         baseline_seconds: float, analysis_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Sliding 250 ms 13-80 Hz band-energy, median/MAD-normalized against the
    pre-event baseline. Shared by ``analyse_brain_process`` (which reduces
    this to per-channel summary scores and onset latencies) and
    ``plot_seizure_evolution`` (which renders the full [windows, channels]
    course this returns). Returns ``(times, z)``.
    """
    high = min(80.0, sfreq / 2 * .95)
    if high <= 13:
        raise ValueError("Sampling frequency is too low for beta-gamma analysis.")
    # Restrict the costly high-frequency transform to the clinical neighbourhood.
    # The recording-wide detector still receives the complete recording in run_edf.
    context_start = max(0., event.time_seconds - baseline_seconds)
    context_end = min(data.shape[1] / sfreq, event.time_seconds + analysis_seconds + .5)
    first_sample = int(np.floor(context_start * sfreq))
    last_sample = int(np.ceil(context_end * sfreq))
    context = data[:, first_sample:last_sample]
    filtered = signal.sosfiltfilt(
        signal.butter(4, [13., high], btype="bandpass", fs=sfreq, output="sos"),
        np.nan_to_num(context), axis=1)
    win, step = max(4, round(.25 * sfreq)), max(1, round(.05 * sfreq))
    starts = np.arange(0, context.shape[1] - win + 1, step)
    if not starts.size:
        raise ValueError("Event context is shorter than one beta-gamma window.")
    times = context_start + (starts + win / 2) / sfreq
    energy = np.stack([np.mean(filtered[:, start:start + win] ** 2, axis=1) for start in starts])
    baseline = (times >= max(0, event.time_seconds - baseline_seconds)) & (times < event.time_seconds)
    if baseline.sum() < 4:
        raise ValueError("Insufficient pre-event baseline for process analysis.")
    center = np.median(energy[baseline], axis=0)
    mad = 1.4826 * np.median(np.abs(energy[baseline] - center), axis=0)
    scale = np.where(mad > 1e-20, mad, np.maximum(np.std(energy[baseline], axis=0), 1e-20))
    return times, np.clip((energy - center) / scale, 0, 50)


def analyse_brain_process(data: np.ndarray, sfreq: float, names: list[str],
                          event: ClinicalEvent | AnnotatedEvent | DetectedEvent,
                          baseline_seconds: float = 30.0, analysis_seconds: float = 8.0) -> BrainProcess:
    """Rank beta-gamma activation and estimate robust recruitment latency.

    Recruitment is the first post-event window above six MADs. ``event`` only
    supplies a time to center the window on — an expert ``ClinicalEvent``, a
    ``find_annotated_event``-read ``AnnotatedEvent``, or an automatically
    ``select_seizure_event``-picked ``DetectedEvent`` — and is kept separate
    from the data-derived measurements.
    """
    times, z = _beta_gamma_z_scores(data, sfreq, event, baseline_seconds, analysis_seconds)
    after = (times >= event.time_seconds) & (times <= event.time_seconds + analysis_seconds)
    if not after.any():
        raise ValueError("Clinical event is outside the recording.")
    scores = np.max(z[after], axis=0)
    latency: dict[str, float] = {}
    for index, name in enumerate(names):
        crossings = np.flatnonzero(after & (z[:, index] >= 6.))
        if crossings.size:
            latency[name] = float(times[crossings[0]] - event.time_seconds)
    first = min(latency.values(), default=0.)
    initiators = tuple(name for name in names
                       if RIGHT_FRONTAL.search(name)
                       and latency.get(name, np.inf) <= first + .25)
    if not initiators and latency:
        # Fall back to the earliest measured contacts, never merely the largest
        # amplitude: "initiation" is a temporal claim.
        initiators = tuple(name for name, delay in sorted(latency.items(), key=lambda item: item[1])
                           if delay <= first + .05)
    later = tuple(name for name, delay in sorted(latency.items(), key=lambda item: item[1])
                  if name not in initiators and delay > first + .05)
    return BrainProcess(event.time_seconds, dict(zip(names, map(float, scores))), latency,
                        initiators, later)


def plot_seizure_evolution(data: np.ndarray, sfreq: float, names: list[str],
                           event: ClinicalEvent | AnnotatedEvent | DetectedEvent, process: BrainProcess,
                           output: str | Path, baseline_seconds: float = 30.0,
                           analysis_seconds: float = 8.0) -> Path:
    """Visualize how the seizure recruits channels from onset to peak.

    Renders a channel-by-time heatmap of the same 13-80 Hz median/MAD z-score
    ``analyse_brain_process`` computes, restricted to exactly the channels it
    already found involved (``process.onset_latency_seconds`` — never a
    separately re-picked "top N") and ordered by their recruitment latency,
    so the image reads top-to-bottom as the cascade ``process`` already
    measured: initiators first, later-recruited contacts below. The dashed
    line at 0 s marks ``event`` — per the clinical annotation this analysis
    was centred on, the point the asymmetric tonic activity was scored as
    becoming a generalized bilateral tonic-clonic seizure, i.e. the peak this
    cascade builds toward, not necessarily the very first twitch.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not process.onset_latency_seconds:
        raise ValueError("BrainProcess found no involved channels to visualize.")
    times, z = _beta_gamma_z_scores(data, sfreq, event, baseline_seconds, analysis_seconds)
    ordered_names = sorted(process.onset_latency_seconds, key=process.onset_latency_seconds.get)
    indices = [names.index(name) for name in ordered_names]
    matrix = z[:, indices].T
    relative_times = times - event.time_seconds
    fig, ax = plt.subplots(figsize=(14, max(4, .35 * len(ordered_names) + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="inferno", vmin=0,
                   vmax=max(6., float(np.percentile(matrix, 99))),
                   extent=[relative_times[0], relative_times[-1], len(ordered_names), 0])
    ax.axvline(0, color="cyan", lw=1.5, ls="--")
    ax.axvline(event.duration_seconds, color="cyan", lw=1, ls=":")
    ax.set_yticks(np.arange(len(ordered_names)) + .5, ordered_names, fontsize=8)
    ax.set_xlabel(f"Time relative to event peak (s) — {event.label!r} at {event.time_seconds:.3f} s")
    ax.set_title("Beta/gamma recruitment cascade — involved channels ordered by onset latency "
                "(initiators first)")
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("median/MAD z-score (13-80 Hz energy)")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)
    return output


PEAK_NODE = "PEAK"


def build_seizure_graph(data: np.ndarray, sfreq: float, names: list[str],
                        event: ClinicalEvent | AnnotatedEvent | DetectedEvent, process: BrainProcess,
                        baseline_seconds: float = 30.0, analysis_seconds: float = 8.0,
                        correlation_threshold: float = 0.5, top_k_per_node: int = 4):
    """Build a NetworkX graph of how the seizure recruits channels toward the peak.

    Nodes are exactly the channels ``analyse_brain_process`` already found
    involved (``process.onset_latency_seconds`` — never a separately
    re-picked subset), plus one synthetic ``PEAK`` node standing for
    ``event`` itself. Two kinds of edges, both measured, none assumed:

    - ``PEAK``-to-channel "recruitment" spokes, weighted by how soon after
      the peak each channel was recruited (heavier for earlier recruitment) —
      the graph counterpart of ``plot_seizure_evolution``'s onset-latency
      ordering, i.e. "...goes to peak".
    - channel-to-channel "co-activation" edges from the Pearson correlation
      of the same 13-80 Hz z-score time courses that heatmap renders,
      threshold- and top-k-pruned exactly as the sister notebook
      (`sEEG_temporal_wavelet_graph_colab.ipynb`) prunes its db4-correlation
      graphs — i.e. "...how it starts [and] evolves", read from which
      channels' activity actually co-varies, not an assumed propagation path.

    Node attributes (``onset_latency_seconds``, ``peak_z``, ``is_initiator``)
    and edge attributes (``kind``, ``weight``) are enough to recreate
    ``plot_seizure_graph``'s figure, or to export/analyse the graph directly
    (e.g. ``networkx.write_graphml``).
    """
    import networkx as nx
    if not process.onset_latency_seconds:
        raise ValueError("BrainProcess found no involved channels to graph.")
    times, z = _beta_gamma_z_scores(data, sfreq, event, baseline_seconds, analysis_seconds)
    involved = sorted(process.onset_latency_seconds, key=process.onset_latency_seconds.get)
    indices = [names.index(name) for name in involved]
    correlation = np.corrcoef(z[:, indices], rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)

    graph = nx.Graph(event_time_seconds=event.time_seconds, event_label=event.label)
    graph.add_node(PEAK_NODE, kind="peak", label=event.label)
    for name in involved:
        latency = process.onset_latency_seconds[name]
        graph.add_node(name, kind="channel", onset_latency_seconds=latency,
                       peak_z=process.channel_band_scores.get(name, 0.0),
                       is_initiator=name in process.likely_initiators)
        graph.add_edge(PEAK_NODE, name, kind="recruitment", weight=1.0 / (1.0 + latency),
                       latency_seconds=latency)

    magnitude = np.abs(correlation)
    selected = set()
    for row in range(len(involved)):
        candidates = np.flatnonzero(magnitude[row] >= correlation_threshold)
        if candidates.size:
            strongest = candidates[np.argsort(magnitude[row, candidates])[-top_k_per_node:]]
            selected.update(tuple(sorted((row, int(col)))) for col in strongest if col != row)
    for row, col in selected:
        graph.add_edge(involved[row], involved[col], kind="co-activation",
                       weight=float(correlation[row, col]))
    return graph


def _seizure_graph_layout(graph, channel_nodes: list[str], layout: str, seed: int):
    """Compute a ``{node: (x, y)}`` layout for ``plot_seizure_graph``.

    Four layouts, all deterministic (``seed``) and all placing ``PEAK`` at
    the origin, so the same figure-reading convention ("centre = peak")
    holds across every one of them — only how the channels are arranged
    around it differs:

    - ``"radial"`` (default): angle from a spring layout of only the
      co-activation mesh (so correlated channels cluster angularly), radius
      from recruitment latency (closer = recruited sooner). Reads outside-in
      as the seizure converges on the peak; this is the layout used before
      multiple layouts existed.
    - ``"spring"``: a single standard force-directed layout over the whole
      graph (mesh + recruitment spokes together, weighted), letting both
      edge kinds jointly shape the picture instead of only the mesh.
    - ``"circular"``: channels placed evenly around a circle ordered by
      recruitment latency, so position reads left-to-right/around as a
      clock face of "when", with no correlation structure involved at all —
      a plain, uncluttered reference to compare the other layouts against.
    - ``"shell"``: two concentric rings, initiators on the inner ring and
      all other involved channels on the outer ring, emphasizing the
      initiator/later-recruited split ``analyse_brain_process`` already
      makes rather than latency as a continuum or correlation structure.
    """
    import networkx as nx
    mesh = nx.Graph()
    mesh.add_nodes_from(channel_nodes)
    mesh.add_edges_from((u, v, d) for u, v, d in graph.edges(data=True) if d.get("kind") == "co-activation")

    if layout == "radial":
        angular_pos = nx.spring_layout(mesh, seed=seed, weight="weight")
        max_latency = max(graph.nodes[node]["onset_latency_seconds"] for node in channel_nodes) or 1.0
        pos = {PEAK_NODE: (0.0, 0.0)}
        for node in channel_nodes:
            x, y = angular_pos[node]
            angle = np.arctan2(y, x)
            radius = 0.2 + 0.8 * (graph.nodes[node]["onset_latency_seconds"] / max_latency)
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))
        return pos

    if layout == "spring":
        return nx.spring_layout(graph, seed=seed, weight="weight")

    if layout == "circular":
        ordered = sorted(channel_nodes, key=lambda node: graph.nodes[node]["onset_latency_seconds"])
        pos = nx.circular_layout(ordered)
        pos[PEAK_NODE] = (0.0, 0.0)
        return pos

    if layout == "shell":
        initiators = [node for node in channel_nodes if graph.nodes[node]["is_initiator"]]
        others = [node for node in channel_nodes if node not in initiators]
        shells = [shell for shell in (initiators, others) if shell]
        pos = nx.shell_layout(channel_nodes, nlist=shells) if shells else {}
        pos[PEAK_NODE] = (0.0, 0.0)
        return pos

    raise ValueError(f"Unknown layout {layout!r}; choose one of "
                     "'radial', 'spring', 'circular', 'shell'.")


def plot_seizure_graph(graph, output: str | Path, layout: str = "radial", seed: int = 7) -> Path:
    """Render a ``build_seizure_graph`` result as a node-link figure.

    ``layout`` selects how channels are arranged around ``PEAK``; see
    ``_seizure_graph_layout`` for the four choices. ``PEAK`` sits at the
    centre as a black star in every one. Initiators (right-frontal contacts
    crossing threshold first) are drawn as gold diamonds; other involved
    channels as circles colour-mapped by their peak z-score. Recruitment
    spokes are faint; the co-activation mesh (red for negative, grey for
    positive correlation) carries the visual weight.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    channel_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "channel"]
    if not channel_nodes:
        raise ValueError("Graph has no channel nodes to plot.")
    pos = _seizure_graph_layout(graph, channel_nodes, layout, seed)

    initiators = [node for node in channel_nodes if graph.nodes[node]["is_initiator"]]
    others = [node for node in channel_nodes if node not in initiators]
    peak_z = {node: graph.nodes[node]["peak_z"] for node in channel_nodes}
    vmax = max(6.0, float(np.percentile(list(peak_z.values()), 99))) if peak_z else 6.0

    mesh_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "co-activation"]
    mesh_colors = ["firebrick" if graph.edges[u, v]["weight"] < 0 else "0.65" for u, v in mesh_edges]
    spoke_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "recruitment"]

    fig, ax = plt.subplots(figsize=(13, 13))
    nx.draw_networkx_edges(graph, pos, edgelist=spoke_edges, edge_color="0.85", width=0.5, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=mesh_edges, edge_color=mesh_colors, width=0.9,
                           alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=[PEAK_NODE], node_shape="*", node_size=1400,
                           node_color="black", ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=others, node_shape="o", node_size=220,
                           node_color=[peak_z[node] for node in others], cmap="inferno",
                           vmin=0, vmax=vmax, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=initiators, node_shape="D", node_size=320,
                           node_color=[peak_z[node] for node in initiators], cmap="inferno",
                           vmin=0, vmax=vmax, edgecolors="gold", linewidths=2.0, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels={node: node for node in channel_nodes},
                            font_size=6, ax=ax)
    ax.annotate(graph.graph["event_label"], pos[PEAK_NODE], xytext=(0, -18),
               textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(f"Seizure recruitment graph ({layout} layout) — PEAK: "
                f"{graph.graph['event_label']!r} at {graph.graph['event_time_seconds']:.3f} s\n"
                "gold diamonds = initiators; mesh = measured co-activation "
                "(red = negative correlation)")
    ax.axis("off")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)
    return output


DEFAULT_SEIZURE_GRAPH_LAYOUTS = ("radial", "spring", "circular", "shell")


def plot_seizure_graph_layouts(graph, output_dir: str | Path, stem: str,
                               layouts: tuple[str, ...] = DEFAULT_SEIZURE_GRAPH_LAYOUTS,
                               seed: int = 7) -> dict[str, Path]:
    """Render the same graph in every layout in ``layouts``.

    No single layout is "the" seizure graph: ``radial`` reads temporally
    (outside-in to the peak), ``spring`` lets edges alone decide structure,
    ``circular`` is a plain latency clock face with no correlation structure
    at all, and ``shell`` isolates the initiator/later-recruited split. One
    file per layout, named ``<stem>_seizure_graph_<layout>.png``. Returns a
    ``{layout: path}`` dict.
    """
    output_dir = Path(output_dir)
    return {layout: plot_seizure_graph(graph, output_dir / f"{stem}_seizure_graph_{layout}.png",
                                       layout=layout, seed=seed)
           for layout in layouts}


def simulate_message_passing(graph, steps: int = 6, alpha: float = 0.5) -> tuple[list[str], np.ndarray]:
    """Diffuse each channel's real post-peak activation through the graph.

    A single linear message-passing update, run for ``steps`` iterations:
    ``h(t+1) = alpha * h(t) + (1 - alpha) * D^-1 W h(t)``, where ``W`` is the
    absolute co-activation weight between channels (edge sign captures phase
    relationship, not connection strength, so only magnitude drives
    diffusion) and ``D`` is each channel's row-sum degree. The seed ``h(0)``
    is each channel's already-measured ``peak_z`` — the maximum z-score
    ``analyse_brain_process`` found for it across the post-peak analysis
    window, not a synthetic or uniform value, though also not exactly the
    instantaneous value at ``event.time_seconds`` (see
    ``evaluate_message_passing``'s step-0 correlation, which is therefore
    informative rather than trivially 1.0) — so this models how the graph's
    *static*, measured co-activation structure alone would spread that real
    starting condition outward. ``evaluate_message_passing`` then checks the
    result against what the recording actually did next. Returns
    ``(channel_order, states)`` where ``states`` is
    ``[steps + 1, n_channels]``, row 0 being the seed.
    """
    channel_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "channel"]
    if not channel_nodes:
        raise ValueError("Graph has no channel nodes to propagate.")
    count = len(channel_nodes)
    index_of = {node: index for index, node in enumerate(channel_nodes)}
    weights = np.zeros((count, count), dtype=float)
    for u, v, edge_data in graph.edges(data=True):
        if edge_data.get("kind") != "co-activation":
            continue
        i, j = index_of[u], index_of[v]
        weights[i, j] = weights[j, i] = abs(edge_data["weight"])
    degree = weights.sum(axis=1)
    transition = np.divide(weights, degree[:, None], out=np.zeros_like(weights), where=degree[:, None] > 0)

    state = np.array([graph.nodes[node]["peak_z"] for node in channel_nodes], dtype=float)
    states = [state.copy()]
    for _ in range(steps):
        state = alpha * state + (1 - alpha) * (transition @ state)
        states.append(state.copy())
    return channel_nodes, np.stack(states)


def evaluate_message_passing(data: np.ndarray, sfreq: float, names: list[str],
                             event: ClinicalEvent | AnnotatedEvent | DetectedEvent,
                             channel_order: list[str], states: np.ndarray,
                             baseline_seconds: float = 30.0, analysis_seconds: float = 8.0,
                             elapsed_seconds_per_step: float | None = None) -> dict[str, list[float]]:
    """Check simulated diffusion against what the recording actually did next.

    For each propagation step, the simulated state is spatially (i.e.
    cross-channel, at one instant) Pearson-correlated against the real
    measured 13-80 Hz z-score at ``event.time_seconds + step *
    elapsed_seconds_per_step`` (default ``analysis_seconds / steps``, so the
    last step lands at the end of the same analysis window
    ``analyse_brain_process`` uses). This is a real check, not a
    demonstration: a falling correlation means the graph's static structure,
    built from one moment, does not explain the seizure's actual temporal
    evolution beyond that moment.
    """
    steps = states.shape[0] - 1
    if elapsed_seconds_per_step is None:
        elapsed_seconds_per_step = analysis_seconds / max(steps, 1)
    times, z = _beta_gamma_z_scores(data, sfreq, event, baseline_seconds, analysis_seconds)
    indices = [names.index(name) for name in channel_order]
    elapsed_seconds, correlation = [], []
    for step in range(steps + 1):
        elapsed = step * elapsed_seconds_per_step
        nearest = int(np.argmin(np.abs(times - (event.time_seconds + elapsed))))
        real_row, simulated_row = z[nearest, indices], states[step]
        if np.std(real_row) < 1e-12 or np.std(simulated_row) < 1e-12:
            value = float("nan")
        else:
            value = float(np.corrcoef(simulated_row, real_row)[0, 1])
        elapsed_seconds.append(elapsed)
        correlation.append(value)
    return {"elapsed_seconds": elapsed_seconds, "correlation": correlation}


def plot_message_passing(graph, channel_order: list[str], states: np.ndarray, output: str | Path,
                         layout: str = "spring", seed: int = 7, max_panels: int = 6) -> Path:
    """Small-multiples of the network state at each propagation step.

    Positions come from ``_seizure_graph_layout`` (same convention as
    ``plot_seizure_graph``: ``PEAK`` at centre); node colour is the simulated
    state at that step, on one shared colour scale so panels are directly
    comparable. At most ``max_panels`` evenly-spaced steps are shown (all of
    them, for the default ``simulate_message_passing`` step count). Defaults
    to the ``"spring"`` layout rather than ``"radial"``: here the step
    progression itself already carries the temporal "starts/evolves/peaks"
    story, so a layout using both edge kinds tends to keep loosely-connected
    outlier channels from dominating each panel's axis scale the way
    ``"radial"``'s latency-only radius can.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    pos = _seizure_graph_layout(graph, channel_order, layout, seed)
    total_steps = states.shape[0]
    panel_steps = np.unique(np.linspace(0, total_steps - 1, min(max_panels, total_steps)).astype(int))
    vmin, vmax = float(states.min()), max(float(states.max()), 1e-6)
    mesh_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "co-activation"]

    fig, axes = plt.subplots(1, len(panel_steps), figsize=(4.2 * len(panel_steps), 4.6))
    axes = np.atleast_1d(axes)
    for axis, step in zip(axes, panel_steps):
        nx.draw_networkx_edges(graph, pos, edgelist=mesh_edges, edge_color="0.8", width=0.4,
                               alpha=0.5, ax=axis)
        nx.draw_networkx_nodes(graph, pos, nodelist=channel_order, node_size=60,
                               node_color=states[step], cmap="inferno", vmin=vmin, vmax=vmax, ax=axis)
        nx.draw_networkx_nodes(graph, pos, nodelist=[PEAK_NODE], node_shape="*", node_size=260,
                               node_color="black", ax=axis)
        axis.set_title(f"step {step}")
        axis.axis("off")
    fig.suptitle(f"Message-passing diffusion of the peak-moment state — {graph.graph['event_label']!r}")
    colorbar_source = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin, vmax))
    fig.colorbar(colorbar_source, ax=list(axes), shrink=0.7, label="simulated activation (z-score units)")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170); plt.close(fig)
    return output


def plot_message_passing_validation(evaluation: dict[str, list[float]], output: str | Path) -> Path:
    """Line plot of simulated-vs-real spatial correlation across propagation steps.

    Reads as an evaluation, not a demo: correlation near 1 at a given
    elapsed time means the graph's static co-activation structure, alone,
    reproduces which channels were actually active then; a falling curve
    means the real dynamics outrun what a graph built from a single moment
    can explain.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(evaluation["elapsed_seconds"], evaluation["correlation"], marker="o", color="navy")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(xlabel="Elapsed time after peak (s)",
          ylabel="Spatial correlation: simulated vs. measured",
          title="Message-passing validation against observed post-peak dynamics",
          ylim=(-1.05, 1.05))
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170); plt.close(fig)
    return output


def plot_all_timeseries(data: np.ndarray, sfreq: float, names: list[str], output: str | Path,
                        event: ClinicalEvent | AnnotatedEvent | DetectedEvent | None = None,
                        max_points: int = 12000) -> Path:
    """Render every channel over the complete EDF, downsampling only the display.

    The marker style reports its own provenance: a ``ClinicalEvent`` (solid
    crimson) is an expert CLI-supplied time; an ``AnnotatedEvent`` (solid
    teal) is the clinician's own marker read from the EDF's annotation
    channel; a ``DetectedEvent`` (dashed orange) is an algorithmic guess.
    Only the dashed style should ever be read as unconfirmed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stride = max(1, int(np.ceil(data.shape[1] / max_points)))
    shown = data[:, ::stride]
    times = np.arange(0, data.shape[1], stride)[:shown.shape[1]] / sfreq
    shown -= np.nanmedian(shown, axis=1, keepdims=True)
    scale = 1.4826 * np.nanmedian(np.abs(shown), axis=1)
    scale = np.where(scale > 0, scale, np.nanstd(shown, axis=1))
    normalized = shown / np.where(scale > 0, scale, 1)[:, None]
    offsets = np.arange(len(names))[::-1] * 8.
    fig, ax = plt.subplots(figsize=(20, max(7, .28 * len(names))))
    for trace, offset in zip(normalized, offsets):
        ax.plot(times, np.clip(trace, -3.5, 3.5) + offset, color="black", lw=.35)
    if event:
        if isinstance(event, DetectedEvent):
            color, style, prefix = "darkorange", "--", "detected: "
        elif isinstance(event, AnnotatedEvent):
            color, style, prefix = "teal", "-", "EDF annotation: "
        else:
            color, style, prefix = "crimson", "-", ""
        ax.axvline(event.time_seconds, color=color, lw=1.5, ls=style)
        ax.axvspan(event.time_seconds, event.time_seconds + event.duration_seconds,
                   color=color, alpha=.15)
        ax.annotate(f"{prefix}{event.label}\n{event.time_seconds:.3f} s", (event.time_seconds, offsets[0]),
                    xytext=(8, 10), textcoords="offset points", color=color, rotation=90,
                    va="bottom", fontsize=9)
    ax.set_yticks(offsets, names)
    ax.set(xlabel="Time from EDF start (s)", ylabel="EEG channels (from `names`)",
           title="Whole-recording sEEG overview (robust-normalized display)")
    ax.margins(x=0); fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)
    return output


def run_edf(path: str | Path, output_dir: str | Path, event: ClinicalEvent | None = None) -> EdfRunResult:
    """Run detection, context analysis, and whole-recording visualization.

    ``event`` (an explicit expert ``--event-time``/``--event-clock``) always
    wins when given. Otherwise the EDF's own annotation channel is checked
    via ``find_annotated_event`` — the clinician's own real-time markup, read
    from the file rather than typed as an apriori number — and used if it
    names a seizure. Only when neither is available does ``select_seizure_event``
    fall back to the agent's blind statistical ranking. Whichever one is used
    drives the beta/gamma process analysis, the whole-recording overview
    figure (which includes the ``MKR...`` marker channels for visual/QC
    context even though ``read_edf`` keeps them out of detection and process
    analysis), and — when the process found involved channels — the
    ``plot_seizure_evolution`` heatmap, every layout from
    ``plot_seizure_graph_layouts``, and a ``simulate_message_passing`` /
    ``evaluate_message_passing`` run rendered by ``plot_message_passing`` and
    ``plot_message_passing_validation``. See :class:`EdfRunResult` for the
    full, named result shape.
    """
    data, sfreq, names = read_edf(path)
    report = ExtremeEventAgent().run(data, sfreq, names)
    annotated = None if event is not None else find_annotated_event(path)
    detected = None if (event is not None or annotated is not None) else select_seizure_event(report.events)
    context = event or annotated or detected
    process = analyse_brain_process(data, sfreq, names, context) if context else None

    output_dir = Path(output_dir)
    stem = Path(path).stem
    marker_data, _, marker_names = read_edf_markers(path)
    overview_data = np.concatenate([data, marker_data], axis=0) if marker_names else data
    overview_names = names + marker_names
    overview_figure = plot_all_timeseries(overview_data, sfreq, overview_names,
                                          output_dir / f"{stem}_all_timeseries.png", context)

    evolution_figure = None
    graph_figures: dict[str, Path] = {}
    graph_graphml = None
    message_passing_figure = message_passing_validation_figure = None
    message_passing_evaluation = None
    if process is not None and process.onset_latency_seconds:
        evolution_figure = plot_seizure_evolution(
            data, sfreq, names, context, process, output_dir / f"{stem}_seizure_evolution.png")
        graph = build_seizure_graph(data, sfreq, names, context, process)
        graph_figures = plot_seizure_graph_layouts(graph, output_dir, stem)
        import networkx as nx
        graph_graphml = output_dir / f"{stem}_seizure_graph.graphml"
        nx.write_graphml(graph, graph_graphml)
        channel_order, states = simulate_message_passing(graph)
        message_passing_evaluation = evaluate_message_passing(
            data, sfreq, names, context, channel_order, states)
        message_passing_figure = plot_message_passing(
            graph, channel_order, states, output_dir / f"{stem}_message_passing.png")
        message_passing_validation_figure = plot_message_passing_validation(
            message_passing_evaluation, output_dir / f"{stem}_message_passing_validation.png")

    return EdfRunResult(report, process, overview_figure, evolution_figure, graph_figures, graph_graphml,
                        message_passing_figure, message_passing_validation_figure,
                        message_passing_evaluation, annotated, detected)
