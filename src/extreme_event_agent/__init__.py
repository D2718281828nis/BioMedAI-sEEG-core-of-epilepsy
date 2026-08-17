"""Agentic extreme-event discovery for multichannel time series."""

from .agent import ExtremeEventAgent
from .models import AgentConfig, DetectionReport, Event

__all__ = ["AgentConfig", "DetectionReport", "Event", "ExtremeEventAgent"]
