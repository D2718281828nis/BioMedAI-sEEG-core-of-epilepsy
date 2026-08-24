"""Hemisphere-level structural prior for tier-3 blind extreme-event candidates.

``extreme_event_agent.edf_workflow.select_seizure_event`` picks the primary
blind-detector candidate by channel spread, then duration, then score — a
purely temporal/statistical signal, deliberately never tuned against a known
answer (see its own docstring). The structural-anomaly map in
``structural_anomaly.py`` is a spatial signal available *before* any event
time is known at all, so combining them is not "tuning against the label" —
it is genuinely independent evidence, the same way this repo already
cross-checks the blind detector against the EDF's own annotation, or the
bipolar montage against the referential one.

**Scope.** There is no verified per-contact 3-D electrode localization in
this repository yet (that is the harder problem discussed in the package
README — MRI-only signal-void contact detection with per-contact QC). What
*is* available now, with zero additional apriori input, is the hemisphere
each channel already names: ``is_right_frontal`` (and this dataset's own
montage convention) already treats unprimed shafts (``PM``, ``CC``, ...) as
right-hemisphere and primed shafts (``PM'``, ``CC'``, ...) as their
contralateral, left-hemisphere counterpart. So this module only compares at
hemisphere granularity, not per-contact — a coarser but honestly-supported
claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = ["StructuralPriorReport", "apply_structural_prior"]

# extreme_event_agent.edf_workflow.parse_contact_name only matches a single
# referential contact ("EEG PM3") — by design, per its own docstring, since
# it feeds build_bipolar_montage. A bipolar-referenced candidate's channel
# names are pair labels instead ("PM3-4", "SA'4-5" — see
# apply_bipolar_montage), which parse_contact_name does not match at all, so
# every montage_reference="bipolar" candidate would silently classify as
# hemisphere-unknown if reused here. This pattern accepts both forms and, like
# extreme_event_agent.edf_workflow.is_right_frontal, keeps the trailing "'"
# that marks the primed contralateral shaft.
_SHAFT_PATTERN = re.compile(r"^(?:EEG\s+)?([A-Za-zА-Яа-я]+'?)\s*\d+(?:-\d+)?$")


def _hemisphere_of(channel_name: str) -> str | None:
    """Right for an unprimed shaft, left for its primed contralateral counterpart.

    Mirrors the same convention ``is_right_frontal`` relies on: the montage's
    own naming, not a coordinate. Returns ``None`` for a name that doesn't
    fit a single-contact or bipolar-pair SEEG label (e.g. a non-SEEG channel).
    """
    match = _SHAFT_PATTERN.match(channel_name.strip())
    if match is None:
        return None
    shaft = match.group(1)
    return "left" if shaft.endswith("'") else "right"


def _event_hemisphere_fractions(involved_channels: list[str]) -> dict[str, object]:
    sides = [side for side in (_hemisphere_of(name) for name in involved_channels) if side is not None]
    if not sides:
        return {"right_fraction": 0.0, "left_fraction": 0.0, "classified_channel_count": 0}
    right = sum(1 for side in sides if side == "right")
    left = len(sides) - right
    return {
        "right_fraction": right / len(sides),
        "left_fraction": left / len(sides),
        "classified_channel_count": len(sides),
    }


@dataclass
class StructuralPriorReport:
    """Every candidate annotated with its hemisphere balance and structural-alignment score.

    ``temporal_pick_index`` is the same pick ``select_seizure_event`` would
    make (ranked by channel spread, then duration, then score — read
    straight off ``candidates``, never recomputed differently here).
    ``structural_pick_index`` is which candidate a *purely spatial* signal —
    MRI hemisphere-asymmetry magnitude times the candidate's own channel
    hemisphere balance — would prefer, evaluated completely independently.
    ``agree`` says whether an independent temporal method and an independent
    spatial method landed on the same candidate: a genuine cross-check, not
    a re-ranking of one by the other.
    """

    candidates: list[dict[str, Any]]
    temporal_pick_index: int | None
    structural_pick_index: int | None
    agree: bool
    hemisphere_summary: dict[str, Any]


def apply_structural_prior(events: list[dict[str, Any]],
                            hemisphere_summary: dict[str, Any]) -> StructuralPriorReport:
    """Annotate blind-detector candidates with a hemisphere structural-alignment score.

    ``events`` is a list of dicts shaped like ``extreme_event_agent.models.Event``
    (``start_seconds``, ``end_seconds``, ``score``, ``involved_channels``, ...) —
    e.g. ``analysis.json["detection"]["events"]`` as already written by
    ``extreme_event_agent``'s CLI. ``hemisphere_summary`` is
    ``StructuralAnomalyResult.hemisphere_summary`` from ``structural_anomaly.py``.
    Never mutates a threshold or a candidate's own score/confidence — every
    input field is carried through unchanged; only new keys are added.
    """
    right_info = hemisphere_summary.get("right_hemisphere", {}) or {}
    left_info = hemisphere_summary.get("left_hemisphere", {}) or {}
    right_score = right_info.get("mean_abs_anomaly") or 0.0
    left_score = left_info.get("mean_abs_anomaly") or 0.0
    hemisphere_diff = right_score - left_score  # positive => MRI flags right as more structurally anomalous

    annotated: list[dict[str, Any]] = []
    for event in events:
        fractions = _event_hemisphere_fractions(list(event.get("involved_channels", [])))
        laterality = fractions["right_fraction"] - fractions["left_fraction"]  # positive => mostly right channels
        alignment_score = hemisphere_diff * laterality  # positive => MRI and channel hemisphere agree
        annotated.append({**event, **fractions, "structural_alignment_score": alignment_score})

    if not annotated:
        return StructuralPriorReport([], None, None, True, hemisphere_summary)

    def temporal_key(item: dict[str, Any]) -> tuple[float, float, float]:
        return (len(item.get("involved_channels", [])),
                item["end_seconds"] - item["start_seconds"],
                item["score"])

    temporal_index = max(range(len(annotated)), key=lambda i: temporal_key(annotated[i]))
    structural_index = max(range(len(annotated)), key=lambda i: annotated[i]["structural_alignment_score"])

    return StructuralPriorReport(
        candidates=annotated,
        temporal_pick_index=temporal_index,
        structural_pick_index=structural_index,
        agree=(temporal_index == structural_index),
        hemisphere_summary=hemisphere_summary,
    )
