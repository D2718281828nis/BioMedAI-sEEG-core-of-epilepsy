"""Score any method's output against this recording's own annotated event.

The only ground truth `sEEG-HFOs-8.edf` has is its own EDF+ annotation
cluster (see ``edf_workflow.find_annotated_event``): `начало приступа?` at
10392.734s, `приступ + БТКП` at 10396.445s, `клиника` at 10399.469s — a
~6.7s spread that is itself the natural scale of "how precisely can an
expert even pin this down". This module compares every method this repo
produces against that one reference, with named, justified tolerance bands
for time and a normalized index for space, and refuses to silently drop the
context (crop, channel-selection method, masking method, prior) that any of
those numbers depends on.

**Why only two live temporal methods, not three.** A literal reading of this
task names three EDF-side temporal methods: a "blind broadband ensemble", a
"targeted narrowband" method, and a "tier-3 blind fallback". Only two of
those exist as code in this installable package: ``t_targeted``
(``analyse_brain_process``'s 13-80 Hz recruitment analysis — see
``edf_workflow.py``) and the blind statistical detector
(``ExtremeEventAgent``, reached as tier 3 by
``edf_workflow.select_seizure_event``). The "≈+39.6s broadband ensemble"
figure this task's own reference numbers cite is
``sEEG_extreme_event_detector_colab.ipynb``'s five-method ensemble, which
``MANIFEST.md`` already documents as explicitly *not* part of the
installable package. Rather than duplicate that notebook's ensemble into
this package or silently relabel the tier-3 blind detector's own number as
if it were a third, independent method, this module reports exactly the two
methods that exist here as ``t_targeted``/``t_blind`` and leaves the
notebook's own number to the notebook and the top-level README, where it is
already discussed.

**Why lateralization, not "which contact".** DICOM's own asymmetry map is
hemisphere-granular only (no verified per-contact 3-D electrode
localization exists in this repository — see
``multimodal_approach/README.md``, "Honest limits"), so the only axis on
which EDF, DICOM, and reservoir evidence are directly comparable is which
*side* each implicates, not which contact. ``lateralization_index`` is
therefore the one quantity every source below reduces to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .edf_workflow import hemisphere_of_channel
from .models import AnnotatedEvent, BrainProcess

__all__ = ["TemporalAccuracy", "LateralizationEstimate", "ContactOverlap", "VerificationReport",
          "classify_temporal_accuracy", "lateralization_index", "contact_overlap",
          "verify_against_annotation", "PRECISE_SECONDS", "COARSE_SECONDS", "WINDOW_SECONDS",
          "INDETERMINATE_LI_THRESHOLD"]

TemporalBand = Literal["precise", "coarse", "window", "miss"]

# Tolerance bands for |delta_seconds|, each with a concrete reason, not a
# round-number guess:
# - PRECISE: tighter than the ~6.7s spread between this recording's own
#   earliest ("начало приступа?", 10392.734s) and latest ("клиника",
#   10399.469s) annotations around the same seizure -- i.e. more precise
#   than the expert's own annotation granularity for this event.
# - COARSE: within the ictal phase as commonly reported in seizure-timing
#   literature -- a "roughly right" lead, not a precise onset.
# - WINDOW: within the seizure event as a whole (onset through generalized
#   tonic-clonic phase), the loosest band that still means "found the right
#   event", not an unrelated part of the recording.
PRECISE_SECONDS = 1.0
COARSE_SECONDS = 10.0
WINDOW_SECONDS = 60.0
# Lateralization index below this magnitude is treated as "no side
# resolved" rather than forced to a sign -- an LI of, say, 0.02 is far more
# likely to be measurement noise around a genuinely symmetric or
# non-lateralized process than a real, if weak, right/left preference.
INDETERMINATE_LI_THRESHOLD = 0.05


def classify_temporal_accuracy(delta_seconds: float) -> TemporalBand:
    """Band ``|delta_seconds|`` into ``"precise"``/``"coarse"``/``"window"``/``"miss"``.

    Only the magnitude decides the band; the sign of ``delta_seconds``
    itself is a separate, deliberately preserved piece of information (see
    ``TemporalAccuracy``) -- a method that fires 0.5s *before* the
    annotation and one that fires 0.5s *after* it are both ``"precise"``
    here, but are not the same finding.
    """
    magnitude = abs(delta_seconds)
    if magnitude <= PRECISE_SECONDS:
        return "precise"
    if magnitude <= COARSE_SECONDS:
        return "coarse"
    if magnitude <= WINDOW_SECONDS:
        return "window"
    return "miss"


@dataclass(frozen=True)
class TemporalAccuracy:
    """One method's time, compared against the recording's own annotated event.

    ``delta_seconds`` is signed (``method_time_seconds - reference_time_seconds``)
    and is never ``abs()``'d before being stored anywhere in this module --
    a negative value (the method fired *before* the annotation) is a
    materially different finding from a positive one of the same magnitude
    (predictive vs. lagging vs. plausibly a false alarm), and collapsing
    that distinction at storage time would make it unrecoverable later.
    """

    method: str
    method_time_seconds: float
    reference_time_seconds: float
    delta_seconds: float
    band: TemporalBand


def _temporal_accuracy(method: str, method_time_seconds: float, reference_time_seconds: float) -> TemporalAccuracy:
    delta = method_time_seconds - reference_time_seconds
    return TemporalAccuracy(method=method, method_time_seconds=method_time_seconds,
                            reference_time_seconds=reference_time_seconds, delta_seconds=delta,
                            band=classify_temporal_accuracy(delta))


def lateralization_index(v_right: float, v_left: float) -> float:
    """``(v_right - v_left) / (v_right + v_left)``, clamped to ``[-1, 1]``.

    ``v_right``/``v_left`` should already be normalized (e.g. a rate or a
    per-voxel mean, not a raw count) so that hemispheres with different
    channel/voxel counts are comparable -- see each ``_*_lateralization``
    helper below for how each source does that. Both-zero (or
    exactly-opposite-sign, canceling) input returns ``0.0`` rather than
    dividing by zero.
    """
    total = v_right + v_left
    if abs(total) < 1e-12:
        return 0.0
    return float(np.clip((v_right - v_left) / total, -1.0, 1.0))


def _lateralization_side(index: float) -> str:
    if abs(index) < INDETERMINATE_LI_THRESHOLD:
        return "indeterminate"
    return "right" if index > 0 else "left"


@dataclass(frozen=True)
class LateralizationEstimate:
    """One source's normalized right/left estimate and the side it implies.

    ``right_value``/``left_value`` are already normalized per hemisphere
    (a rate for EDF, a per-voxel mean for DICOM, a per-output-channel mean
    for the reservoir) so they are comparable across sources despite
    different underlying channel/voxel counts, which are still reported
    (``right_count``/``left_count``) for context. ``arbitration_valid`` is
    ``True`` for every source except a reservoir estimate built from a
    ``"recruitment"``-selected plant (see ``model.plant.ReservoirWindow.arbitration_valid``):
    a reservoir plant whose output channels came from
    ``analyse_brain_process`` cannot independently confirm or contest that
    same analysis's own lateralization, so this flag exists precisely so a
    reader does not mistake circular agreement for a cross-check.
    """

    source: str
    right_value: float
    left_value: float
    right_count: int
    left_count: int
    index: float
    side: str
    arbitration_valid: bool = True


def _edf_lateralization(process: BrainProcess) -> LateralizationEstimate:
    """EDF-agnostic lateralization: fraction of ``earliest_contacts`` per hemisphere.

    Deliberately reads ``process.earliest_contacts`` (prior-free -- see
    ``edf_workflow.analyse_brain_process``'s docstring), never
    ``likely_initiators``, which by construction can only ever contain
    contacts the clinical prior already names and so cannot supply an
    independent lateralization estimate. The denominator for each
    hemisphere's rate is every channel ``analyse_brain_process`` actually
    scored (``process.channel_band_scores`` -- computed for every input
    channel, not only the involved ones), so a hemisphere with more
    montage channels is not automatically favored.
    """
    sides = {name: hemisphere_of_channel(name) for name in process.channel_band_scores}
    right_total = sum(1 for side in sides.values() if side == "right")
    left_total = sum(1 for side in sides.values() if side == "left")
    earliest_sides = [sides.get(name) for name in process.earliest_contacts]
    right_rate = (earliest_sides.count("right") / right_total) if right_total else 0.0
    left_rate = (earliest_sides.count("left") / left_total) if left_total else 0.0
    index = lateralization_index(right_rate, left_rate)
    return LateralizationEstimate(source="edf_earliest_contacts", right_value=right_rate, left_value=left_rate,
                                  right_count=right_total, left_count=left_total, index=index,
                                  side=_lateralization_side(index))


def _dicom_lateralization(hemisphere_summary: dict[str, Any]) -> LateralizationEstimate | None:
    """DICOM lateralization straight off ``structural_anomaly.py``'s own per-hemisphere means.

    ``mean_abs_anomaly`` is already a per-voxel mean, so no extra
    normalization by voxel count is needed here (unlike the EDF/reservoir
    sources, which sum over a variable number of channels). Returns
    ``None`` if either hemisphere has no scored voxels at all (e.g. no
    DICOM was available) rather than fabricating a 0.0.
    """
    right = hemisphere_summary.get("right_hemisphere") or {}
    left = hemisphere_summary.get("left_hemisphere") or {}
    right_value, left_value = right.get("mean_abs_anomaly"), left.get("mean_abs_anomaly")
    if right_value is None or left_value is None:
        return None
    index = lateralization_index(float(right_value), float(left_value))
    return LateralizationEstimate(source="dicom_mean_abs_anomaly", right_value=float(right_value),
                                  left_value=float(left_value), right_count=int(right.get("voxel_count", 0) or 0),
                                  left_count=int(left.get("voxel_count", 0) or 0), index=index,
                                  side=_lateralization_side(index))


def _reservoir_lateralization(reservoir_evaluation: Any | None) -> list[LateralizationEstimate]:
    """Two independent reservoir-derived lateralization estimates, or ``[]`` without one.

    Duck-typed against ``model.plant.ReservoirEvaluation`` (``window``,
    ``per_channel_peak_score``, ``per_channel_onset_seconds``) rather than
    importing ``model`` directly, so this module stays a leaf the rest of
    the repo's packages can depend on without a reverse dependency.

    - **strength**: mean of ``max(0, per_channel_peak_score)`` (clipped at
      0 -- a negative z-score is "more predictable than baseline", not
      evidence of anything) over each hemisphere's output channels.
    - **earliness**: mean of ``1 / (1 + onset_seconds)`` over output
      channels whose smoothed score actually crossed threshold
      *after* the reference event (``onset_seconds >= 0``); a channel whose
      only crossing happened during the pre-event baseline is excluded
      here, not counted as "early" -- that crossing reflects reservoir
      warm-up/transient behavior, not a response to the event this function
      is trying to lateralize.

    Both are divided by that hemisphere's own output-channel count, so an
    imbalanced ``"recruitment"``-selected channel list does not by itself
    bias the result -- though see ``LateralizationEstimate.arbitration_valid``
    for why that channel list still matters.
    """
    if reservoir_evaluation is None:
        return []
    output_names = list(reservoir_evaluation.window.output_names)
    sides = {name: hemisphere_of_channel(name) for name in output_names}
    right_names = [name for name in output_names if sides.get(name) == "right"]
    left_names = [name for name in output_names if sides.get(name) == "left"]
    arbitration_valid = bool(getattr(reservoir_evaluation.window, "arbitration_valid", False))

    def _strength(names: list[str]) -> float:
        if not names:
            return 0.0
        return sum(max(0.0, reservoir_evaluation.per_channel_peak_score.get(name, 0.0))
                   for name in names) / len(names)

    def _earliness(names: list[str]) -> float:
        if not names:
            return 0.0
        total = 0.0
        for name in names:
            onset = reservoir_evaluation.per_channel_onset_seconds.get(name)
            if onset is not None and onset >= 0:
                total += 1.0 / (1.0 + onset)
        return total / len(names)

    estimates = []
    for source, aggregate in (("reservoir_residual_strength", _strength),
                              ("reservoir_residual_earliness", _earliness)):
        right_value, left_value = aggregate(right_names), aggregate(left_names)
        index = lateralization_index(right_value, left_value)
        estimates.append(LateralizationEstimate(
            source=source, right_value=right_value, left_value=left_value, right_count=len(right_names),
            left_count=len(left_names), index=index, side=_lateralization_side(index),
            arbitration_valid=arbitration_valid))
    return estimates


def _pairwise_agreement(estimates: list[LateralizationEstimate]) -> list[dict[str, Any]]:
    """Sign agreement between every pair of ``estimates``.

    ``agree`` is ``None`` (not ``True``/``False``) whenever either side is
    ``"indeterminate"`` -- there is no side to agree or disagree *about*.
    """
    pairs = []
    for i in range(len(estimates)):
        for j in range(i + 1, len(estimates)):
            a, b = estimates[i], estimates[j]
            agree = None if "indeterminate" in (a.side, b.side) else (a.side == b.side)
            pairs.append({"source_a": a.source, "source_b": b.source, "agree": agree,
                         "both_arbitration_valid": a.arbitration_valid and b.arbitration_valid})
    return pairs


@dataclass(frozen=True)
class ContactOverlap:
    """How much ``earliest_contacts`` (data) overlaps ``prior_matched`` (external prior).

    This measures whether the data *supports* the clinical hypothesis --
    not a localization claim in its own right (see
    ``edf_workflow.analyse_brain_process``'s docstring on
    ``prior_fraction_among_earliest``, which this is a close relative of,
    computed against the full ``prior_matched`` set rather than only the
    ``earliest`` bucket).
    """

    precision: float
    recall: float
    jaccard: float
    earliest_count: int
    prior_count: int
    overlap_count: int


def contact_overlap(process: BrainProcess) -> ContactOverlap | None:
    """``None`` when ``process`` was built with no prior (``prior_matched`` empty)."""
    if not process.prior_matched:
        return None
    earliest, prior = set(process.earliest_contacts), set(process.prior_matched)
    overlap, union = earliest & prior, earliest | prior
    return ContactOverlap(
        precision=len(overlap) / len(earliest) if earliest else 0.0,
        recall=len(overlap) / len(prior) if prior else 0.0,
        jaccard=len(overlap) / len(union) if union else 0.0,
        earliest_count=len(earliest), prior_count=len(prior), overlap_count=len(overlap))


@dataclass
class VerificationReport:
    """Everything ``verify_against_annotation`` produces, in one JSON-able record.

    ``crop_applied``/``crop_end_seconds``/``channel_selection``/
    ``masking_method``/``prior_used`` are required context, not optional
    metadata: none of the numbers above are interpretable without knowing
    whether the recording was cropped (see ``edf_workflow.read_edf``'s own
    non-physiological-tail discussion), which reservoir channel-selection
    strategy was used (``model.plant.CHANNEL_SELECTION_METHODS``), which
    brain-extraction parameters produced any DICOM number
    (``multimodal_approach.structural_anomaly.StructuralAnomalyResult.masking_method``),
    and whether a clinical prior constrained the EDF side at all. Each is
    ``None`` only when genuinely not applicable (e.g. ``masking_method`` when
    no DICOM was analysed), never silently omitted.
    """

    temporal: list[TemporalAccuracy]
    lateralization: list[LateralizationEstimate]
    lateralization_agreement: list[dict[str, Any]]
    contact_overlap: ContactOverlap | None
    crop_applied: bool
    crop_end_seconds: float | None
    channel_selection: str | None
    masking_method: str | None
    prior_used: str | None
    reservoir_arbitration_valid: bool | None = field(default=None)


def verify_against_annotation(annotation: AnnotatedEvent, process: BrainProcess,
                              blind_event_time_seconds: float | None = None,
                              hemisphere_summary: dict[str, Any] | None = None,
                              reservoir_evaluation: Any | None = None,
                              crop_applied: bool = False, crop_end_seconds: float | None = None,
                              channel_selection: str | None = None,
                              masking_method: str | None = None) -> VerificationReport:
    """Score every available method's time and lateralization against ``annotation``.

    ``annotation`` is the recording's own EDF+ annotated event (tier 2 --
    see ``edf_workflow.find_annotated_event``), the one piece of ground
    truth this recording has. ``process`` supplies ``t_targeted`` (its
    ``earliest_latency_seconds``, relative to whatever event it was itself
    centred on) and the EDF-agnostic lateralization estimate.
    ``blind_event_time_seconds``, when given, is the tier-3 blind
    detector's own pick (``t_blind`` -- see this module's docstring for why
    that, not a third notebook-only ensemble, is what "blind" means here).
    ``hemisphere_summary``/``reservoir_evaluation`` are each optional and
    independently omittable -- a report with only EDF evidence is still a
    valid (if incomplete) report, never an error.
    """
    reference_time = annotation.time_seconds
    targeted_time = process.event_time_seconds + process.earliest_latency_seconds
    temporal = [_temporal_accuracy("t_targeted", targeted_time, reference_time)]
    if blind_event_time_seconds is not None:
        temporal.append(_temporal_accuracy("t_blind", blind_event_time_seconds, reference_time))

    lateralization = [_edf_lateralization(process)]
    if hemisphere_summary is not None:
        dicom_estimate = _dicom_lateralization(hemisphere_summary)
        if dicom_estimate is not None:
            lateralization.append(dicom_estimate)
    lateralization += _reservoir_lateralization(reservoir_evaluation)

    reservoir_arbitration_valid = (
        bool(getattr(reservoir_evaluation.window, "arbitration_valid", False))
        if reservoir_evaluation is not None else None)

    return VerificationReport(
        temporal=temporal,
        lateralization=lateralization,
        lateralization_agreement=_pairwise_agreement(lateralization),
        contact_overlap=contact_overlap(process),
        crop_applied=crop_applied,
        crop_end_seconds=crop_end_seconds,
        channel_selection=channel_selection,
        masking_method=masking_method,
        prior_used=(process.prior_source or None),
        reservoir_arbitration_valid=reservoir_arbitration_valid)
