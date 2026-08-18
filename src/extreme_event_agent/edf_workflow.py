"""EDF orchestration, whole-recording plots, and event-centred analysis."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from scipy import signal

from .agent import ExtremeEventAgent
from .models import AnnotatedEvent, BrainProcess, ClinicalEvent, DetectedEvent, DetectionReport, Event

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


def run_edf(path: str | Path, output_dir: str | Path,
            event: ClinicalEvent | None = None
            ) -> tuple[DetectionReport, BrainProcess | None, Path, Path | None,
                      AnnotatedEvent | None, DetectedEvent | None]:
    """Run detection, context analysis, and whole-recording visualization.

    ``event`` (an explicit expert ``--event-time``/``--event-clock``) always
    wins when given. Otherwise the EDF's own annotation channel is checked
    via ``find_annotated_event`` — the clinician's own real-time markup, read
    from the file rather than typed as an apriori number — and used if it
    names a seizure. Only when neither is available does ``select_seizure_event``
    fall back to the agent's blind statistical ranking. Whichever one is used
    still drives the beta/gamma process analysis, the whole-recording overview
    figure, and (when the process found involved channels) a second recruitment
    figure from ``plot_seizure_evolution``; the two fallbacks are also returned
    individually so callers can record provenance. The overview figure includes
    the ``MKR...`` marker channels for visual/QC context even though ``read_edf``
    keeps them out of detection and process analysis (see its docstring).
    """
    data, sfreq, names = read_edf(path)
    report = ExtremeEventAgent().run(data, sfreq, names)
    annotated = None if event is not None else find_annotated_event(path)
    detected = None if (event is not None or annotated is not None) else select_seizure_event(report.events)
    context = event or annotated or detected
    process = analyse_brain_process(data, sfreq, names, context) if context else None
    marker_data, _, marker_names = read_edf_markers(path)
    overview_data = np.concatenate([data, marker_data], axis=0) if marker_names else data
    overview_names = names + marker_names
    plot = plot_all_timeseries(overview_data, sfreq, overview_names,
                               Path(output_dir) / f"{Path(path).stem}_all_timeseries.png", context)
    evolution_plot = None
    if process is not None and process.onset_latency_seconds:
        evolution_plot = plot_seizure_evolution(
            data, sfreq, names, context, process,
            Path(output_dir) / f"{Path(path).stem}_seizure_evolution.png")
    return report, process, plot, evolution_plot, annotated, detected
