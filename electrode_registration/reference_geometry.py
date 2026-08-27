"""Dataset-specific shaft reference table, transcribed by hand from ``dataset/истинное положение.pdf``.

That PDF is a clinician's orientation sketch: RadiAnt-viewer screenshots (a
trial-licensed third-party tool — every page carries its own "You have N
days left in your trial period" watermark) with one circle hand-drawn per
shaft over coronal/sagittal/axial views, plus a short table of shaft length
and contact count. Critically, **it contains no numeric 3-D coordinates**,
and several of its own screenshots carry MRI-software warnings reading
"Warning: Interpolated Image, all values are approximate". This module
transcribes only the one part of it that is a plain, unambiguous fact —
shaft name, hemisphere, documented length, and documented contact count —
not the circled regions themselves, which cannot be turned into a
trustworthy coordinate without a human re-clicking them.

``SEEG_HFOS_8_SHAFT_REFERENCE`` is therefore used throughout
``electrode_registration`` strictly as a **count/length cross-check**
against what the artifact-mask-based detector in ``contact_detection.py``
finds on its own from the MRI signal — never as spatial ground truth. See
this package's own ``README.md``, "Honest limits", for why: there is no CT,
no manually placed per-shaft fiducial file, and no numeric coordinate table
anywhere in this dataset, the same gap ``multimodal_approach/README.md``
already documents under "Why this, and not full electrode localization".
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ShaftReference", "SEEG_HFOS_8_SHAFT_REFERENCE", "montage_shaft_contact_counts"]


@dataclass(frozen=True)
class ShaftReference:
    """One shaft's documented (not measured-here) length/contact-count, from the PDF's own table."""

    name: str
    hemisphere: str  # "right" (unprimed) / "left" (primed) -- same convention as edf_workflow.hemisphere_of_channel
    length_mm: float
    contact_count: int
    color: str


# Transcribed verbatim from dataset/истинное положение.pdf, pages 1-6
# ("СПРАВА:" = right, unprimed shafts; "СЛЕВА:" = left, primed shafts).
# Format per entry there: "<shaft> <length_mm>, <contact_count> к <color>".
SEEG_HFOS_8_SHAFT_REFERENCE: dict[str, ShaftReference] = {
    "R":   ShaftReference("R",   "right", 50.0, 10, "black"),
    "FP":  ShaftReference("FP",  "right", 38.0, 8,  "brown"),
    "FD":  ShaftReference("FD",  "right", 25.0, 6,  "yellow"),
    "PM":  ShaftReference("PM",  "right", 38.0, 8,  "red"),
    "CC":  ShaftReference("CC",  "right", 49.0, 10, "yellow"),
    "SA":  ShaftReference("SA",  "right", 30.0, 6,  "black"),
    "PA":  ShaftReference("PA",  "right", 44.0, 10, "orange"),
    "CR'": ShaftReference("CR'", "left",  46.0, 10, "brown"),
    "CC'": ShaftReference("CC'", "left",  49.0, 10, "lilac"),
    "PM'": ShaftReference("PM'", "left",  27.0, 6,  "blue"),
    "SA'": ShaftReference("SA'", "left",  38.0, 8,  "orange"),
    "PA'": ShaftReference("PA'", "left",  38.0, 8,  "green"),
}


def montage_shaft_contact_counts(names: list[str]) -> dict[str, int]:
    """Per-shaft contact counts read straight from EDF channel names — exact, not from the PDF.

    Reuses ``extreme_event_agent.edf_workflow.parse_contact_name`` (the same
    parser ``build_bipolar_montage`` already uses), so a shaft's contact
    count here is never computed a second, different way. Unlike
    ``SEEG_HFOS_8_SHAFT_REFERENCE`` (approximate, from a hand-drawn sketch),
    this is exact: every contact the recording actually has a channel for is
    counted once, no more.
    """
    from extreme_event_agent.edf_workflow import parse_contact_name

    counts: dict[str, int] = {}
    for name in names:
        parsed = parse_contact_name(name)
        if parsed is None:
            continue
        shaft, _ = parsed
        counts[shaft] = counts.get(shaft, 0) + 1
    return counts
