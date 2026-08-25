from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


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
class ContactPrior:
    """Contacts nominated by external clinical context, not by the data.

    This is deliberately a plain input, not a derived result: ``shafts`` is
    read from an expert (the clinical picture supplied alongside a
    recording), and ``analyse_brain_process`` never edits it based on what
    the signal shows. Keeping it as an explicit, named argument — rather
    than a bare module constant — is what makes it possible to see, in the
    function signature itself, which part of ``likely_initiators`` is prior
    and which part is measured (see ``edf_workflow.SEEG_HFOS_8_CLINICAL_PRIOR``
    for this dataset's actual value, and ``edf_workflow.prior_matches`` for
    how a channel is tested against it). Passing ``prior=None`` disables it
    entirely, for a region-agnostic reading of the same recording.
    """

    shafts: Mapping[str, Sequence[int]]
    source: str
    description: str


@dataclass(frozen=True)
class BrainProcess:
    """Data-derived beta/gamma recruitment around an expert annotation.

    ``likely_initiators`` and ``later_recruited`` are kept for backward
    compatibility; the fields below them make explicit which part of
    ``likely_initiators`` came from the data and which from the external
    prior (see ``edf_workflow.analyse_brain_process``'s docstring for the
    exact classification rule):

    ``earliest_contacts``/``earliest_latency_seconds`` are the globally
    earliest threshold crossing(s) and their latency, computed with **no**
    reference to the prior at all — this is what closes the blind spot
    described in the package README: a contact recruited first that happens
    to sit outside the prior's contact list is reported here even though it
    is excluded from ``likely_initiators``. ``prior_matched`` is every
    involved contact the prior names, regardless of latency.
    ``prior_source`` documents where the prior came from (empty string when
    ``prior=None``). ``initiators_constrained_by_prior`` is ``True`` exactly
    when the wider, prior-only recruitment window contributed a contact to
    ``likely_initiators`` that the prior-free ``earliest_contacts`` set did
    not already include. ``prior_fraction_among_earliest`` is
    ``|earliest_contacts ∩ prior_matched| / |earliest_contacts|`` (``0.0``
    when there are no earliest contacts) — 1.0 means every globally-earliest
    contact is one the prior also names; 0.0 means none of them are.
    ``hemisphere_of_earliest`` is ``"right"``/``"left"``/``"mixed"``/
    ``"unknown"``, read off ``earliest_contacts`` alone via this dataset's
    montage naming convention (see ``edf_workflow.hemisphere_of_channel``) —
    the one field in this class that can directly agree or disagree with the
    prior's own right-frontal hypothesis.
    """

    event_time_seconds: float
    channel_band_scores: dict[str, float]
    onset_latency_seconds: dict[str, float]
    likely_initiators: tuple[str, ...]
    later_recruited: tuple[str, ...]
    earliest_contacts: tuple[str, ...] = ()
    earliest_latency_seconds: float = 0.0
    prior_matched: tuple[str, ...] = ()
    prior_source: str = ""
    initiators_constrained_by_prior: bool = False
    prior_fraction_among_earliest: float = 0.0
    hemisphere_of_earliest: str = "unknown"


@dataclass
class EdfRunResult:
    """Everything one ``run_edf`` call produces, named instead of positional.

    ``montage`` is the bipolar shaft/contact-pair grouping read straight off
    the channel names (``build_bipolar_montage``) — always describes the
    recording's referential contact structure, independent of
    ``montage_reference``. ``montage_reference`` ("none" or "bipolar")
    records which signal reference every other field in this result was
    actually computed against — "none" analyses the recording's native
    channels as loaded; "bipolar" re-references to adjacent-contact
    differences first (``apply_bipolar_montage``), changing detection,
    process analysis, the recruitment graph, and message passing all at
    once. ``graph_figures`` maps layout name to its rendered PNG (empty when
    ``process`` found no involved channels). ``message_passing_evaluation``
    is the ``{"elapsed_seconds": [...], "correlation": [...]}`` result of
    checking simulated diffusion against what the recording actually did
    next; ``None`` under the same condition as the message-passing figures.
    ``message_passing_figures`` mirrors ``graph_figures`` but for the
    diffusion figure — the same signal-propagation-from-source-to-PEAK story,
    one file per layout — while ``message_passing_figure`` keeps the single
    default (``"spring"``) rendering for backward compatibility.
    ``source_summary`` is ``describe_seizure_source``'s plain-language
    statement of the located source (channel(s) and absolute time) and the
    involved-channel spread; ``source_summary_file`` is where it was written.
    Both are ``None`` under the same condition as the graph figures.
    """

    report: DetectionReport
    process: BrainProcess | None
    montage: dict[str, list[tuple[str, str]]]
    montage_reference: str
    montage_file: Path
    overview_figure: Path
    evolution_figure: Path | None
    graph_figures: dict[str, Path]
    graph_graphml: Path | None
    message_passing_figure: Path | None
    message_passing_validation_figure: Path | None
    message_passing_evaluation: dict[str, list[float]] | None
    annotated_event: AnnotatedEvent | None
    detected_event: DetectedEvent | None
    message_passing_figures: dict[str, Path] = field(default_factory=dict)
    source_summary: str | None = None
    source_summary_file: Path | None = None
