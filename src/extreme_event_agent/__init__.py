"""Agentic extreme-event discovery for multichannel time series."""

from .agent import ExtremeEventAgent
from .models import (AgentConfig, AnnotatedEvent, BrainProcess, ClinicalEvent, DetectedEvent,
                     DetectionReport, EdfRunResult, Event)

__all__ = ["AgentConfig", "AnnotatedEvent", "BrainProcess", "ClinicalEvent", "DetectedEvent",
           "DetectionReport", "EdfRunResult", "Event", "ExtremeEventAgent"]
