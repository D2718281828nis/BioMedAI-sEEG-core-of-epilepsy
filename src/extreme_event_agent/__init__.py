"""Agentic extreme-event discovery for multichannel time series."""

from .agent import ExtremeEventAgent
from .models import AgentConfig, BrainProcess, ClinicalEvent, DetectionReport, Event

__all__ = ["AgentConfig", "BrainProcess", "ClinicalEvent", "DetectionReport", "Event", "ExtremeEventAgent"]
