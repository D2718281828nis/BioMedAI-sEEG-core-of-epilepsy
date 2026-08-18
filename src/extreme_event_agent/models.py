from __future__ import annotations

from dataclasses import dataclass, field


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
class BrainProcess:
    """Data-derived beta/gamma recruitment around an expert annotation."""

    event_time_seconds: float
    channel_band_scores: dict[str, float]
    onset_latency_seconds: dict[str, float]
    likely_initiators: tuple[str, ...]
    later_recruited: tuple[str, ...]
