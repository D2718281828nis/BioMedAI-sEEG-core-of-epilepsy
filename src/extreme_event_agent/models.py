from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    """Detection policy. Times are expressed in seconds."""

    window_seconds: float = 2.0
    step_seconds: float = 0.25
    channel_fraction: float = 0.1
    threshold_mad: float = 6.0
    merge_gap_seconds: float = 0.5
    min_involved_channels: int = 2
    min_quality: float = 0.8
    max_iterations: int = 3

    def __post_init__(self) -> None:
        if self.window_seconds <= 0 or not 0 < self.step_seconds <= self.window_seconds:
            raise ValueError("Require 0 < step_seconds <= window_seconds.")
        if not 0 < self.channel_fraction <= 1:
            raise ValueError("channel_fraction must be in (0, 1].")
        if self.threshold_mad <= 0 or self.max_iterations < 1:
            raise ValueError("threshold_mad and max_iterations must be positive.")


@dataclass(frozen=True)
class Event:
    start_seconds: float
    end_seconds: float
    peak_seconds: float
    score: float
    confidence: float
    involved_channels: tuple[str, ...]
    evidence: dict[str, float]


@dataclass
class DetectionReport:
    events: list[Event]
    sampling_frequency_hz: float
    channel_names: tuple[str, ...]
    threshold: float
    audit_log: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class ClinicalEvent:
    """Expert-provided annotation; it is never used to tune detection."""

    time_seconds: float
    label: str = "асимметричный тонический приступ"
    duration_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.time_seconds < 0 or self.duration_seconds <= 0:
            raise ValueError("Clinical event time must be non-negative and duration positive.")


@dataclass(frozen=True)
class AnnotatedEvent:
    """A seizure marker read from the EDF's own EDF+ annotation channel.

    This is the clinician's own real-time markup, embedded in the recording
    file itself — not a number typed on a command line, and not a
    statistical guess. ``find_annotated_event`` locates it by matching
    seizure-indicative keywords in the decoded annotation text and folding
    every annotation within a short gap of a match into one event, so the
    clinician's separate notes about the same seizure (e.g. an "onset?"
    query alongside a "seizure" tag) don't appear as unrelated events.
    ``annotations`` keeps the full matched cluster, verbatim, for audit.
    """

    time_seconds: float
    label: str
    duration_seconds: float
    annotations: tuple[tuple[float, str], ...]


@dataclass(frozen=True)
class DetectedEvent:
    """A seizure candidate chosen from the agent's own detections.

    Unlike :class:`ClinicalEvent`, nothing here comes from an expert or a
    known timestamp: the time, duration, and channel count are read off the
    highest-ranked entry in ``DetectionReport.events`` after ``select_seizure_event``
    reorders them by spatial spread. It exists so a run without any expert
    annotation can still drive the beta/gamma process analysis and the
    figure marker, while keeping this data-derived provenance visibly
    distinct from an expert-confirmed time.
    """

    time_seconds: float
    label: str
    duration_seconds: float
    involved_channel_count: int
    score: float
    confidence: float


@dataclass(frozen=True)
class BrainProcess:
    """Data-derived beta/gamma recruitment around an expert annotation."""

    event_time_seconds: float
    channel_band_scores: dict[str, float]
    onset_latency_seconds: dict[str, float]
    likely_initiators: tuple[str, ...]
    later_recruited: tuple[str, ...]


@dataclass
class EdfRunResult:
    """Everything one ``run_edf`` call produces, named instead of positional.

    ``graph_figures`` maps layout name to its rendered PNG (empty when
    ``process`` found no involved channels). ``message_passing_evaluation``
    is the ``{"elapsed_seconds": [...], "correlation": [...]}`` result of
    checking simulated diffusion against what the recording actually did
    next; ``None`` under the same condition as the message-passing figures.
    """

    report: DetectionReport
    process: BrainProcess | None
    overview_figure: Path
    evolution_figure: Path | None
    graph_figures: dict[str, Path]
    graph_graphml: Path | None
    message_passing_figure: Path | None
    message_passing_validation_figure: Path | None
    message_passing_evaluation: dict[str, list[float]] | None
    annotated_event: AnnotatedEvent | None
    detected_event: DetectedEvent | None
