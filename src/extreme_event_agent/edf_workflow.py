"""EDF orchestration, whole-recording plots, and event-centred analysis."""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
from scipy import signal

from .agent import ExtremeEventAgent
from .models import BrainProcess, ClinicalEvent, DetectionReport

MARKER = re.compile(r"^MKR\s*\d+\+?$", re.IGNORECASE)
RIGHT_FRONTAL = re.compile(r"(?:PM\s*[3-8]|CC\s*(?:8|9|10))(?:\D|$)", re.IGNORECASE)


def read_edf(path: str | Path) -> tuple[np.ndarray, float, list[str]]:
    """Load every non-marker EDF signal in volts using MNE."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    names = [name for name in raw.ch_names if not MARKER.fullmatch(name.strip())]
    if not names:
        raise ValueError(f"{path} contains no non-marker signal channels.")
    return raw.get_data(picks=names), float(raw.info["sfreq"]), names


def analyse_brain_process(data: np.ndarray, sfreq: float, names: list[str], event: ClinicalEvent,
                          baseline_seconds: float = 30.0, analysis_seconds: float = 8.0) -> BrainProcess:
    """Rank beta-gamma activation and estimate robust recruitment latency.

    Sliding 250 ms band-energy is standardized against the pre-event baseline
    using median/MAD. Recruitment is the first post-event window above six MADs.
    The expert annotation is kept separate from the data-derived measurements.
    """
    high = min(80.0, sfreq / 2 * .95)
    if high <= 13:
        raise ValueError("Sampling frequency is too low for beta-gamma analysis.")
    filtered = signal.sosfiltfilt(
        signal.butter(4, [13., high], btype="bandpass", fs=sfreq, output="sos"),
        np.nan_to_num(data), axis=1)
    win, step = max(4, round(.25 * sfreq)), max(1, round(.05 * sfreq))
    starts = np.arange(0, data.shape[1] - win + 1, step)
    times = (starts + win / 2) / sfreq
    energy = np.stack([np.mean(filtered[:, start:start + win] ** 2, axis=1) for start in starts])
    baseline = (times >= max(0, event.time_seconds - baseline_seconds)) & (times < event.time_seconds)
    if baseline.sum() < 4:
        raise ValueError("Insufficient pre-event baseline for process analysis.")
    center = np.median(energy[baseline], axis=0)
    mad = 1.4826 * np.median(np.abs(energy[baseline] - center), axis=0)
    scale = np.where(mad > 1e-20, mad, np.maximum(np.std(energy[baseline], axis=0), 1e-20))
    z = np.clip((energy - center) / scale, 0, 50)
    after = (times >= event.time_seconds) & (times <= event.time_seconds + analysis_seconds)
    if not after.any():
        raise ValueError("Clinical event is outside the recording.")
    scores = np.max(z[after], axis=0)
    latency: dict[str, float] = {}
    for index, name in enumerate(names):
        crossings = np.flatnonzero(after & (z[:, index] >= 6.))
        if crossings.size:
            latency[name] = float(times[crossings[0]] - event.time_seconds)
    initiators = tuple(name for name in names if RIGHT_FRONTAL.search(name) and name in latency)
    if not initiators:
        initiators = tuple(names[i] for i in np.argsort(scores)[::-1][:min(3, len(names))])
    first = min(latency.values(), default=0.)
    later = tuple(name for name, delay in sorted(latency.items(), key=lambda item: item[1])
                  if name not in initiators and delay > first + .05)
    return BrainProcess(event.time_seconds, dict(zip(names, map(float, scores))), latency,
                        initiators, later)


def plot_all_timeseries(data: np.ndarray, sfreq: float, names: list[str], output: str | Path,
                        event: ClinicalEvent | None = None, max_points: int = 12000) -> Path:
    """Render every channel over the complete EDF, downsampling only the display."""
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
        ax.axvline(event.time_seconds, color="crimson", lw=1.5)
        ax.axvspan(event.time_seconds, event.time_seconds + event.duration_seconds,
                   color="crimson", alpha=.15)
        ax.annotate(f"{event.label}\n{event.time_seconds:.3f} s", (event.time_seconds, offsets[0]),
                    xytext=(8, 10), textcoords="offset points", color="crimson", rotation=90,
                    va="bottom", fontsize=9)
    ax.set_yticks(offsets, names)
    ax.set(xlabel="Time from EDF start (s)", ylabel="All non-marker EEG channels",
           title="Whole-recording sEEG overview (robust-normalized display)")
    ax.margins(x=0); fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)
    return output


def run_edf(path: str | Path, output_dir: str | Path,
            event: ClinicalEvent | None = None) -> tuple[DetectionReport, BrainProcess | None, Path]:
    """Run detection, context analysis, and whole-recording visualization."""
    data, sfreq, names = read_edf(path)
    report = ExtremeEventAgent().run(data, sfreq, names)
    process = analyse_brain_process(data, sfreq, names, event) if event else None
    plot = plot_all_timeseries(data, sfreq, names,
                               Path(output_dir) / f"{Path(path).stem}_all_timeseries.png", event)
    return report, process, plot
