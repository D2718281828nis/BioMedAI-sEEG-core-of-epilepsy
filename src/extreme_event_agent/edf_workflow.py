"""EDF orchestration, whole-recording plots, and event-centred analysis."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from scipy import signal

from .agent import ExtremeEventAgent
from .models import (AnnotatedEvent, BrainProcess, ClinicalEvent, ContactPrior, DetectedEvent,
                     DetectionReport, EdfRunResult, Event)

MARKER = re.compile(r"^MKR\s*\d+\+?$", re.IGNORECASE)
# Matches this dataset's SEEG naming convention: an optional "EEG " prefix, a
# shaft label (letters, optionally ending in "'" for the contralateral/left
# shaft — e.g. "PM" vs "PM'"), then the contact number along that shaft.
CONTACT_PATTERN = re.compile(r"^(?:EEG\s+)?([A-Za-zА-Яа-я]+'?)\s*(\d+)$")
# A looser pattern used only by prior_matches: matches a shaft label plus
# one contact number, or two ("3-4") for a bipolar pair label — searched
# rather than anchored, so it still finds "PM3" inside "EEG PM3".
_CONTACT_NUMBERS_PATTERN = re.compile(r"([A-Za-z]+)\s*(\d+)(?:-(\d+))?")
# Same convention as CONTACT_PATTERN, but also accepting a bipolar pair label
# ("PM3-4") — used only to classify hemisphere by shaft naming (unprimed =
# right, primed = its distinct contralateral counterpart), not to parse a
# contact number, so both single-contact and pair forms are equally valid
# here. Shared with multimodal_approach.extreme_event_prior, which imports
# hemisphere_of_channel from this module rather than keeping its own copy of
# this pattern, so hemisphere classification has exactly one implementation.
_HEMISPHERE_PATTERN = re.compile(r"^(?:EEG\s+)?([A-Za-zА-Яа-я]+'?)\s*\d+(?:-\d+)?$")

# This dataset's clinical context named PM3-8 and CC8-10 as the contacts of
# interest before any signal was analysed — it is supplied information, not
# something the pipeline discovered. Wrapping it in ContactPrior (rather than
# a bare dict) keeps that provenance attached to the value itself instead of
# living only in this file's comments or in the README, so any code holding a
# ContactPrior instance can report where it came from.
SEEG_HFOS_8_CLINICAL_PRIOR = ContactPrior(
    shafts={"PM": range(3, 9), "CC": range(8, 11)},
    source="clinical context supplied with the recording (not derived from the signal)",
    description="right frontal",
)

# The recruitment threshold analyse_brain_process applies to the 13-80 Hz
# median/MAD z-score: a contact "crosses" once its energy (searched across
# the pre-event baseline window and the post-event analysis window alike)
# exceeds this many MADs above its own pre-event baseline. Set externally
# (matches ExtremeEventAgent's own default AgentConfig.threshold_mad), not
# fit to this recording.
RECRUITMENT_THRESHOLD_MAD = 6.0
# Two windows measured relative to tau_min (the globally earliest crossing,
# irrespective of any prior — see analyse_brain_process): SIMULTANEITY_WINDOW_SECONDS
# resolves near-simultaneous crossings into one "earliest" group ("occurred
# at essentially the same instant as the very first crossing"; a purely
# data-derived rule). PRIOR_WINDOW_SECONDS is wider and only ever applies to
# contacts a ContactPrior already names — it is the window within which a
# named contact still counts as an early responder even if it did not tie
# for absolute first. Both are properties of this classification rule, not
# of the underlying detector.
SIMULTANEITY_WINDOW_SECONDS = 0.05
PRIOR_WINDOW_SECONDS = 0.25


def prior_matches(name: str, prior: ContactPrior | None) -> bool:
    """True if ``name`` is a contact (or bipolar pair) named by ``prior``.

    Handles a single referential contact (``"EEG PM3"``) and a bipolar pair
    label (``"PM3-4"``) alike: a pair counts as matching if *either*
    endpoint falls in range, since the pair's local gradient still spans
    that zone — a bipolar ``"PM2-3"``/``"CC7-8"`` pair does touch a named
    right-frontal contact even though its first-listed number does not.
    Only the unprimed shafts ``prior.shafts`` actually lists ever match;
    their primed contralateral counterparts (e.g. ``PM'``, ``CC'``) never
    do, because the shaft-label capture excludes ``'`` and so simply fails
    to match a primed name at all. ``prior=None`` (region-agnostic mode)
    always returns ``False``.
    """
    if prior is None:
        return False
    match = _CONTACT_NUMBERS_PATTERN.search(name.strip())
    if match is None:
        return False
    shaft, first, second = match.group(1), match.group(2), match.group(3)
    valid = prior.shafts.get(shaft)
    if valid is None:
        return False
    numbers = {int(first)} | ({int(second)} if second else set())
    return bool(numbers & set(valid))


def is_right_frontal(name: str) -> bool:
    """True if ``name`` is a contact (or bipolar pair) in PM3-8 or CC8-10.

    Thin, backward-compatible wrapper over ``prior_matches`` against
    ``SEEG_HFOS_8_CLINICAL_PRIOR`` — kept under its original name because
    existing code and tests reference it directly. See ``prior_matches`` for
    the matching rule itself.
    """
    return prior_matches(name, SEEG_HFOS_8_CLINICAL_PRIOR)


def hemisphere_of_channel(name: str) -> str | None:
    """``"right"`` for an unprimed SEEG shaft, ``"left"`` for its primed counterpart.

    Reads this dataset's own montage naming convention (unprimed = right
    hemisphere, ``'``-suffixed = the distinct contralateral shaft — see
    ``parse_contact_name``), not a 3-D coordinate: there is no per-contact
    stereotactic localization in this repository. Matches a single
    referential contact or a bipolar pair label alike, the same as
    ``prior_matches``. Returns ``None`` for a name that isn't an SEEG contact
    label at all (e.g. ``"MKR1+"``). This is the one shared implementation —
    ``multimodal_approach.extreme_event_prior`` imports it rather than
    keeping its own copy, so a contact's hemisphere is never computed two
    different ways in this repository.
    """
    match = _HEMISPHERE_PATTERN.match(name.strip())
    if match is None:
        return None
    shaft = match.group(1)
    return "left" if shaft.endswith("'") else "right"


def _hemisphere_of_group(names: tuple[str, ...]) -> str:
    """Reduce ``hemisphere_of_channel`` over several contacts to one label.

    ``"right"``/``"left"`` only when every classifiable contact in ``names``
    agrees; ``"mixed"`` when both hemispheres are present; ``"unknown"`` when
    ``names`` is empty or none of its entries are classifiable SEEG labels.
    """
    sides = {hemisphere_of_channel(name) for name in names} - {None}
    if not sides:
        return "unknown"
    if sides == {"right"}:
        return "right"
    if sides == {"left"}:
        return "left"
    return "mixed"


# This dataset's annotation channel is Windows-1251 Cyrillic; MNE's default UTF-8
# decode raises on it ("Encountered invalid byte..."). cp1251 is a superset of
# ASCII, so it also decodes plain-ASCII channel names and annotations correctly.
EDF_ENCODING = "cp1251"
SEIZURE_KEYWORDS = ("приступ", "судорог", "seizure", "ictal", "бткп", "tcs")
ANNOTATION_CLUSTER_GAP_SECONDS = 10.0


def clock_time_to_offset(clock: str, recording_start: datetime, duration_seconds: float) -> float:
    """Convert ``HH:MM:SS[.ffffff]`` to seconds from the EDF start.

    Midnight rollover is supported. An out-of-recording value is rejected so a
    typo cannot silently place a clinical marker outside the plotted data.
    """
    try:
        parsed = datetime.strptime(clock, "%H:%M:%S.%f").time()
    except ValueError:
        try:
            parsed = datetime.strptime(clock, "%H:%M:%S").time()
        except ValueError as error:
            raise ValueError("Event clock must use HH:MM:SS or HH:MM:SS.ffffff.") from error
    event_datetime = datetime.combine(recording_start.date(), parsed, recording_start.tzinfo)
    if event_datetime < recording_start:
        event_datetime += timedelta(days=1)
    offset = (event_datetime - recording_start).total_seconds()
    if not 0 <= offset <= duration_seconds:
        raise ValueError(
            f"Event clock {clock} is {offset:.3f} s from EDF start, outside the "
            f"{duration_seconds:.3f} s recording.")
    return offset


def read_edf_start(path: str | Path) -> tuple[datetime, float]:
    """Read EDF start datetime and duration without preloading signal samples."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR", encoding=EDF_ENCODING)
    start = raw.info.get("meas_date")
    if start is None:
        raise ValueError(f"{path} does not contain an EDF recording start time.")
    return start, float(raw.n_times / raw.info["sfreq"])


def read_edf(path: str | Path, crop_end_seconds: float | None = None) -> tuple[np.ndarray, float, list[str]]:
    """Load every non-marker EDF signal in volts using MNE.

    ``MKR...`` channels are excluded here, not because they are ignored, but
    because on this dataset ``MKR1+``/``MKR2+`` are a perfectly periodic 1 Hz
    square-wave hardware sync clock (every transition is exactly 0.5 s from
    the last, for the entire recording, with no anomaly around any detected
    or annotated event) — not brain signal and not an event marker. Folding a
    constant hardware clock into the statistical detector or the beta-gamma
    channel analysis would misrepresent it as neural or clinical evidence.
    Use ``read_edf_markers`` to load them for display instead.

    ``crop_end_seconds``, when given, discards everything after that time
    before any data is read — e.g. a non-physiological segment (equipment
    test, surgical procedure) appended after the recording of interest, which
    would otherwise silently enter the blind detector's own recording-wide
    normalization statistics and candidate ranking. ``None`` (the default)
    reads the complete file, unchanged from prior behavior.
    """
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR", encoding=EDF_ENCODING)
    if crop_end_seconds is not None:
        max_tmax = (raw.n_times - 1) / raw.info["sfreq"]
        raw.crop(tmin=0.0, tmax=min(crop_end_seconds, max_tmax))
    raw.load_data(verbose="ERROR")
    names = [name for name in raw.ch_names if not MARKER.fullmatch(name.strip())]
    if not names:
        raise ValueError(f"{path} contains no non-marker signal channels.")
    return raw.get_data(picks=names), float(raw.info["sfreq"]), names


def read_edf_markers(path: str | Path, crop_end_seconds: float | None = None) -> tuple[np.ndarray, float, list[str]]:
    """Load only the ``MKR...`` marker/trigger channels, for display purposes.

    See ``read_edf`` for why they are kept out of detection and process
    analysis, and for what ``crop_end_seconds`` does — pass the same value
    given to ``read_edf`` so the marker trace overlaid on a whole-recording
    plot spans the same range as the signal channels it is drawn alongside.
    Channels are picked before loading so only their data (not the full
    multichannel recording) is read from disk.
    """
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR", encoding=EDF_ENCODING)
    if crop_end_seconds is not None:
        max_tmax = (raw.n_times - 1) / raw.info["sfreq"]
        raw.crop(tmin=0.0, tmax=min(crop_end_seconds, max_tmax))
    names = [name for name in raw.ch_names if MARKER.fullmatch(name.strip())]
    if not names:
        return np.empty((0, raw.n_times)), float(raw.info["sfreq"]), []
    raw.pick(names)
    raw.load_data(verbose="ERROR")
    return raw.get_data(), float(raw.info["sfreq"]), names


def parse_contact_name(name: str) -> tuple[str, int] | None:
    """Split an SEEG channel name into ``(shaft, contact_number)``.

    ``"EEG PM3"`` -> ``("PM", 3)``; ``"EEG CC'4"`` -> ``("CC'", 4)`` — the
    trailing ``'`` is part of the shaft label, since it marks a distinct
    (contralateral) electrode in this dataset's naming convention, not a
    variant of the unprimed shaft. Returns ``None`` for names that don't fit
    the pattern (e.g. ``MKR1+``).
    """
    match = CONTACT_PATTERN.match(name.strip())
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def build_bipolar_montage(names: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Group EDF channel names into per-shaft bipolar (adjacent-contact) pairs.

    This reads the montage directly out of the channel list already in the
    file — nothing here is a separate, apriori electrode map. Names are
    parsed with ``parse_contact_name``; channels that don't match (e.g.
    ``MKR1+``) are skipped. Within each shaft, contacts are sorted
    numerically and paired consecutively (contact 1 with 2, 2 with 3, ...) —
    the standard bipolar/"referential neighbor" derivation for depth
    electrodes. If a contact number is absent from the recording, the pair
    spanning the gap is still formed from the two nearest present numbers
    (e.g. present contacts 1, 2, 4 pair as 1-2 and 2-4), matching what most
    SEEG review software does by default rather than silently dropping a
    contact's neighbor relationship. Returns ``{shaft: [(name_a, name_b), ...]}``,
    shafts in first-seen order and each shaft's pairs ordered along the shaft.
    """
    shafts: dict[str, list[tuple[int, str]]] = {}
    for name in names:
        parsed = parse_contact_name(name)
        if parsed is None:
            continue
        shaft, contact = parsed
        shafts.setdefault(shaft, []).append((contact, name))
    montage: dict[str, list[tuple[str, str]]] = {}
    for shaft, contacts in shafts.items():
        contacts.sort(key=lambda item: item[0])
        montage[shaft] = [(contacts[index][1], contacts[index + 1][1])
                          for index in range(len(contacts) - 1)]
    return montage


def format_bipolar_montage(montage: dict[str, list[tuple[str, str]]]) -> str:
    """Render a ``build_bipolar_montage`` result as a compact ``"1-2"``-style listing.

    Channel names are reduced to their bare contact numbers within each
    shaft (e.g. ``"EEG PM3"``/``"EEG PM4"`` -> ``"3-4"``) since that number
    is what a clinician reading a montage sheet expects, with the shaft
    label given once as a section header.
    """
    lines = []
    for shaft, pairs in montage.items():
        lines.append(f"{shaft}:")
        for name_a, name_b in pairs:
            contact_a = parse_contact_name(name_a)[1]
            contact_b = parse_contact_name(name_b)[1]
            lines.append(f"  {contact_a}-{contact_b}")
    return "\n".join(lines)


def apply_bipolar_montage(data: np.ndarray, names: list[str],
                          montage: dict[str, list[tuple[str, str]]]
                          ) -> tuple[np.ndarray, list[str]]:
    """Compute bipolar-referenced signals from a ``build_bipolar_montage`` result.

    Each derivation is the earlier contact's signal minus the next one along
    the shaft (``data[a] - data[b]``), labelled ``"<shaft><a>-<b>"`` (e.g.
    ``"PM3-4"``). This is the standard SEEG bipolar reference, used to
    suppress common-reference and volume-conducted artifacts shared by
    neighboring contacts; it is a re-referencing, not a filter, so it does
    not change what a subsequent detector or process analysis measures in
    kind, only which reference the amplitudes are relative to.
    """
    index_of = {name: index for index, name in enumerate(names)}
    derived_data, derived_names = [], []
    for shaft, pairs in montage.items():
        for name_a, name_b in pairs:
            if name_a not in index_of or name_b not in index_of:
                continue
            derived_data.append(data[index_of[name_a]] - data[index_of[name_b]])
            contact_a = parse_contact_name(name_a)[1]
            contact_b = parse_contact_name(name_b)[1]
            derived_names.append(f"{shaft}{contact_a}-{contact_b}")
    return np.stack(derived_data), derived_names


def _cluster_seizure_annotation(onsets: list[float], descriptions: list[str]) -> AnnotatedEvent | None:
    """Pure grouping logic behind ``find_annotated_event``; see its docstring."""
    matches = {index for index, description in enumerate(descriptions)
              if any(keyword in description.lower() for keyword in SEIZURE_KEYWORDS)}
    if not matches:
        return None
    order = sorted(range(len(onsets)), key=lambda index: onsets[index])
    groups = [[order[0]]]
    for index in order[1:]:
        if onsets[index] - onsets[groups[-1][-1]] <= ANNOTATION_CLUSTER_GAP_SECONDS:
            groups[-1].append(index)
        else:
            groups.append([index])
    group = next(group for group in groups if matches & set(group))
    anchor = min((index for index in group if index in matches), key=lambda index: onsets[index])
    start, end = onsets[group[0]], onsets[group[-1]]
    return AnnotatedEvent(
        time_seconds=onsets[anchor],
        label=descriptions[anchor],
        duration_seconds=max(end - start, 0.1),
        annotations=tuple((onsets[index], descriptions[index]) for index in group),
    )


def find_annotated_event(path: str | Path) -> AnnotatedEvent | None:
    """Locate a seizure marker in the EDF's own EDF+ annotation channel.

    This reads metadata the clinician who scored the recording embedded
    directly in the file — the same way ``read_edf_start`` reads ``meas_date``
    — rather than any apriori number. An annotation counts as a seizure
    marker only when its decoded text contains one of ``SEIZURE_KEYWORDS``;
    every other annotation within ``ANNOTATION_CLUSTER_GAP_SECONDS`` of that
    match is folded into the same event, so a clinician's separate notes
    around one seizure (e.g. an "onset?" query next to a "seizure" tag) do
    not appear as unrelated events. Returns ``None`` when the file carries no
    annotations, or none of them mention a seizure.
    """
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR", encoding=EDF_ENCODING)
    onsets = [float(onset) for onset in raw.annotations.onset]
    descriptions = [str(description) for description in raw.annotations.description]
    return _cluster_seizure_annotation(onsets, descriptions)


def select_seizure_event(events: list[Event]) -> DetectedEvent | None:
    """Pick the most seizure-like candidate out of the agent's own detections.

    No expert time is consulted. A seizure is told apart from a brief
    interictal spike by its numeric footprint, not by amplitude alone: a
    sharp interictal discharge can score as high as, or higher than, the
    seizure while staying focal and short. Candidates are therefore ranked
    by channel spread first, event duration second, and raw score only as a
    final tiebreaker, so a widely and durably recruiting candidate always
    outranks a brief high-amplitude one. ``events`` is already the agent's
    finished, threshold-verified candidate list; nothing here re-detects or
    re-scores the recording.
    """
    if not events:
        return None
    best = max(events, key=lambda event: (len(event.involved_channels),
                                          event.end_seconds - event.start_seconds, event.score))
    return DetectedEvent(
        time_seconds=best.peak_seconds,
        label="data-detected extreme event (ranked by channel spread, then duration, then score)",
        duration_seconds=max(best.end_seconds - best.start_seconds, 0.1),
        involved_channel_count=len(best.involved_channels),
        score=best.score,
        confidence=best.confidence,
    )


_FILTER_EDGE_GUARD_SECONDS = 5.0


def _beta_gamma_z_scores(data: np.ndarray, sfreq: float, event: ClinicalEvent | AnnotatedEvent | DetectedEvent,
                         baseline_seconds: float, analysis_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Sliding 250 ms 13-80 Hz band-energy, median/MAD-normalized against the
    pre-event baseline. Shared by ``analyse_brain_process`` (which reduces
    this to per-channel summary scores and onset latencies) and
    ``plot_seizure_evolution`` (which renders the full [windows, channels]
    course this returns). Returns ``(times, z)``.

    ``sosfiltfilt`` is run over ``_FILTER_EDGE_GUARD_SECONDS`` more context
    than is actually needed on the low end, and that guard is discarded
    before ``times``/the returned ``z`` are ever built — not cosmetic:
    without it, the filter's own edge padding produces a spurious energy
    transient right at the start of the context array, shared across many
    channels simultaneously (a numerical artifact of the padding, not
    biology), which ``analyse_brain_process`` could otherwise pick up as a
    fake, suspiciously-synchronized "earliest crossing" whenever its search
    window reaches back to this function's raw start (confirmed by varying
    ``baseline_seconds`` and watching the spurious crossing track the
    boundary instead of staying at a fixed absolute time).
    """
    high = min(80.0, sfreq / 2 * .95)
    if high <= 13:
        raise ValueError("Sampling frequency is too low for beta-gamma analysis.")
    # Restrict the costly high-frequency transform to the clinical neighbourhood.
    # The recording-wide detector still receives the complete recording in run_edf.
    usable_start = max(0., event.time_seconds - baseline_seconds)
    context_start = max(0., usable_start - _FILTER_EDGE_GUARD_SECONDS)
    context_end = min(data.shape[1] / sfreq, event.time_seconds + analysis_seconds + .5)
    first_sample = int(np.floor(context_start * sfreq))
    last_sample = int(np.ceil(context_end * sfreq))
    context = data[:, first_sample:last_sample]
    filtered = signal.sosfiltfilt(
        signal.butter(4, [13., high], btype="bandpass", fs=sfreq, output="sos"),
        np.nan_to_num(context), axis=1)
    win, step = max(4, round(.25 * sfreq)), max(1, round(.05 * sfreq))
    starts = np.arange(0, context.shape[1] - win + 1, step)
    if not starts.size:
        raise ValueError("Event context is shorter than one beta-gamma window.")
    times = context_start + (starts + win / 2) / sfreq
    energy = np.stack([np.mean(filtered[:, start:start + win] ** 2, axis=1) for start in starts])
    keep = times >= usable_start
    times, energy = times[keep], energy[keep]
    baseline = (times >= usable_start) & (times < event.time_seconds)
    if baseline.sum() < 4:
        raise ValueError("Insufficient pre-event baseline for process analysis.")
    center = np.median(energy[baseline], axis=0)
    mad = 1.4826 * np.median(np.abs(energy[baseline] - center), axis=0)
    scale = np.where(mad > 1e-20, mad, np.maximum(np.std(energy[baseline], axis=0), 1e-20))
    return times, np.clip((energy - center) / scale, 0, 50)


def _caption(ax, text: str, loc: str = "lower left") -> None:
    """Small boxed explanatory note anchored inside ``ax``, out of the data's way.

    Every figure below plots a number that means something specific (a
    z-score threshold, a correlation, an offset); this puts that meaning in
    the figure itself instead of only in a docstring the viewer may never
    read.
    """
    from matplotlib.offsetbox import AnchoredText
    anchored = AnchoredText(text, loc=loc, prop=dict(size=8), frameon=True, borderpad=0.6)
    anchored.patch.set(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6")
    ax.add_artist(anchored)


def describe_seizure_source(process: BrainProcess) -> str:
    """Plain-language statement of the located source and its reach.

    Turns the numbers ``build_seizure_graph``/``plot_seizure_graph`` already
    plot into a sentence a reviewer can read without opening a figure: which
    channel(s) crossed the recruitment threshold first (the likely source),
    at what absolute recording time, how many channels were ultimately
    involved, and the latency span between the first and last of them.
    Nothing is computed afresh here — every number quoted already lives in
    ``process`` (produced by ``analyse_brain_process``); this only narrates
    it. Returns an empty-cascade sentence, never raises, when ``process``
    found no involved channels.
    """
    if not process.onset_latency_seconds:
        return ("No channel crossed the recruitment threshold around "
                f"{process.event_time_seconds:.3f} s: no source could be located.")
    involved = sorted(process.onset_latency_seconds, key=process.onset_latency_seconds.get)
    first_latency = process.onset_latency_seconds[involved[0]]
    last_latency = process.onset_latency_seconds[involved[-1]]
    source_channels = process.likely_initiators or (involved[0],)
    lines = [
        f"Reference (peak) time: {process.event_time_seconds:.3f} s from EDF start; "
        "all latencies below are measured relative to it.",
        f"Likely source: {', '.join(source_channels)} — first channel(s) to cross the "
        f"6-MAD recruitment threshold, at {process.event_time_seconds + first_latency:.3f} s "
        f"({first_latency:+.3f} s relative to the reference time).",
        f"Involved channels: {len(involved)} total, recruited between "
        f"{first_latency:+.3f} s and {last_latency:+.3f} s relative to the reference time "
        f"({involved[-1]} last).",
    ]
    if process.later_recruited:
        shown = ", ".join(process.later_recruited[:15])
        more = ", ..." if len(process.later_recruited) > 15 else ""
        lines.append(f"Later-recruited (spread beyond the source): {shown}{more}")
    return "\n".join(lines)


def analyse_brain_process(data: np.ndarray, sfreq: float, names: list[str],
                          event: ClinicalEvent | AnnotatedEvent | DetectedEvent,
                          baseline_seconds: float = 30.0, analysis_seconds: float = 8.0,
                          prior: ContactPrior | None = SEEG_HFOS_8_CLINICAL_PRIOR) -> BrainProcess:
    """Rank beta-gamma activation and classify each channel's recruitment.

    Recruitment is the first window above ``RECRUITMENT_THRESHOLD_MAD``,
    searched across ``[event.time_seconds - baseline_seconds, event.time_seconds
    + analysis_seconds]`` — i.e. including the ``baseline_seconds`` *before*
    ``event`` itself, not only after it. This is deliberate: ``event`` is a
    clinician's real-time annotation, typically pressed once a seizure is
    already clinically visible, which commonly lags the true electrographic
    onset by some seconds; restricting the search to times at or after
    ``event.time_seconds`` would make it structurally impossible for
    ``likely_initiators``/``tau_min`` to ever reflect a channel that
    initiated *before* the clinician noticed, which is exactly the case an
    "initiator" analysis exists to catch. A channel found here can therefore
    carry a *negative* latency. The baseline used for the underlying z-score
    normalization (see ``_beta_gamma_z_scores``) is unaffected — it still
    only ever draws its median/MAD from ``times < event.time_seconds``, so a
    channel that is already ramping up in the last second or two of that
    window is a minority contamination of a mostly-quiet ``baseline_seconds``
    baseline, not a circular self-comparison; median/MAD's robustness to a
    minority of outliers is exactly why this stays meaningful. ``event``
    only supplies a time to center the window on — an expert
    ``ClinicalEvent``, a ``find_annotated_event``-read ``AnnotatedEvent``, or
    an automatically ``select_seizure_event``-picked ``DetectedEvent`` — and
    is kept separate from the data-derived measurements.

    Every channel with a measured recruitment latency is classified into
    exactly one of three mutually exclusive, jointly exhaustive categories,
    relative to ``tau_min`` — the single earliest crossing across *all*
    channels, computed with no reference to ``prior`` whatsoever:

    - **earliest** (``BrainProcess.earliest_contacts``): ``latency <= tau_min
      + SIMULTANEITY_WINDOW_SECONDS``. Purely data-derived; a channel lands
      here whether or not ``prior`` names it.
    - **prior_early**: not already ``earliest``, named by ``prior``, and
      ``latency <= tau_min + PRIOR_WINDOW_SECONDS`` — a wider window that
      only ever applies to prior-named contacts.
    - **later_recruited** (``BrainProcess.later_recruited``): everything
      else with a measured latency.

    This closes a real defect the two-branch version of this function had: a
    channel outside ``prior`` whose latency fell in
    ``(tau_min, tau_min + SIMULTANEITY_WINDOW_SECONDS]`` used to satisfy
    neither the initiator condition (not in ``prior``) nor the later-recruited
    condition (``delay > tau_min + SIMULTANEITY_WINDOW_SECONDS`` was false),
    and so vanished from both reported tuples — worst of all for the single
    globally-earliest channel itself, exactly where the data could have
    contradicted ``prior``. The three categories above are checked below to
    partition ``onset_latency_seconds`` exactly, with no such gap.

    ``BrainProcess.likely_initiators`` is kept for backward compatibility and
    computed as ``prior_early ∪ (earliest ∩ prior)`` — i.e. every contact
    that is both named by ``prior`` and within the applicable window of
    ``tau_min``. ``initiators_constrained_by_prior`` is ``True`` exactly when
    ``prior_early`` is non-empty, i.e. when the prior's wider window
    contributed a contact that the prior-free ``earliest`` set alone would
    not have. Passing ``prior=None`` disables all of this: ``prior_early`` is
    always empty, ``likely_initiators`` reduces to ``earliest``, and
    ``initiators_constrained_by_prior`` is always ``False``.
    """
    times, z = _beta_gamma_z_scores(data, sfreq, event, baseline_seconds, analysis_seconds)
    search_window = ((times >= event.time_seconds - baseline_seconds)
                     & (times <= event.time_seconds + analysis_seconds))
    if not search_window.any():
        raise ValueError("Clinical event is outside the recording.")
    scores = np.max(z[search_window], axis=0)
    latency: dict[str, float] = {}
    for index, name in enumerate(names):
        crossings = np.flatnonzero(search_window & (z[:, index] >= RECRUITMENT_THRESHOLD_MAD))
        if crossings.size:
            latency[name] = float(times[crossings[0]] - event.time_seconds)

    ordered = tuple(sorted(latency, key=latency.get))
    tau_min = latency[ordered[0]] if ordered else 0.0

    earliest = tuple(name for name in ordered
                     if latency[name] <= tau_min + SIMULTANEITY_WINDOW_SECONDS)
    earliest_set = set(earliest)
    prior_early = tuple(name for name in ordered
                        if name not in earliest_set
                        and prior_matches(name, prior)
                        and latency[name] <= tau_min + PRIOR_WINDOW_SECONDS)
    prior_early_set = set(prior_early)
    later = tuple(name for name in ordered
                  if name not in earliest_set and name not in prior_early_set)

    categorised = earliest_set | prior_early_set | set(later)
    if categorised != set(latency) or len(earliest) + len(prior_early) + len(later) != len(latency):
        raise ValueError(
            "Recruitment categorisation must partition every channel with a measured "
            f"latency exactly once: {len(latency)} channel(s) measured, "
            f"{len(earliest)} earliest + {len(prior_early)} prior_early + {len(later)} "
            f"later_recruited = {len(earliest) + len(prior_early) + len(later)} categorised, "
            f"union covers {len(categorised)}.")

    prior_matched = tuple(name for name in ordered if prior_matches(name, prior))
    # prior=None is the region-agnostic mode: there is no prior to intersect
    # earliest with, so the latency-only earliest set *is* likely_initiators
    # (prior_early is always empty in this mode, so this is not a special
    # case of the formula below so much as its degenerate, correct limit).
    likely_initiators = earliest if prior is None else tuple(
        name for name in ordered
        if name in prior_early_set or (name in earliest_set and prior_matches(name, prior)))
    prior_fraction_among_earliest = (
        len(earliest_set & set(prior_matched)) / len(earliest) if earliest else 0.0)

    return BrainProcess(
        event_time_seconds=event.time_seconds,
        channel_band_scores=dict(zip(names, map(float, scores))),
        onset_latency_seconds=latency,
        likely_initiators=likely_initiators,
        later_recruited=later,
        earliest_contacts=earliest,
        earliest_latency_seconds=tau_min,
        prior_matched=prior_matched,
        prior_source=prior.source if prior is not None else "",
        initiators_constrained_by_prior=bool(prior_early),
        prior_fraction_among_earliest=prior_fraction_among_earliest,
        hemisphere_of_earliest=_hemisphere_of_group(earliest),
    )


def plot_seizure_evolution(data: np.ndarray, sfreq: float, names: list[str],
                           event: ClinicalEvent | AnnotatedEvent | DetectedEvent, process: BrainProcess,
                           output: str | Path, baseline_seconds: float = 30.0,
                           analysis_seconds: float = 8.0) -> Path:
    """Visualize how the seizure recruits channels from onset to peak.

    Renders a channel-by-time heatmap of the same 13-80 Hz median/MAD z-score
    ``analyse_brain_process`` computes, restricted to exactly the channels it
    already found involved (``process.onset_latency_seconds`` — never a
    separately re-picked "top N") and ordered by their recruitment latency,
    so the image reads top-to-bottom as the cascade ``process`` already
    measured, earliest first — this row order comes from the data alone and
    is unaffected by ``process.prior_matched``.

    Two further, independent things are drawn on top of that data-only
    ordering, deliberately kept visually separate so a viewer can't mistake
    one for the other: a diamond marker in front of a row's label if that
    contact is named by the external clinical prior
    (``process.prior_matched`` — supplied information, drawn but never
    allowed to move the row); and, on the time axis, ``tau_min``
    (``process.earliest_latency_seconds``, white line) with the
    simultaneity window ``[tau_min, tau_min + SIMULTANEITY_WINDOW_SECONDS]``
    and the wider prior window ``[tau_min, tau_min + PRIOR_WINDOW_SECONDS]``
    shaded and labelled — the exact rule ``analyse_brain_process`` used to
    classify every row, made visible instead of only living in code. A white
    "x" marks each row's own measured crossing instant, and every row in
    ``process.earliest_contacts`` (the globally earliest crossing(s),
    regardless of whether the prior names them) is bolded, so the contact
    that would contradict the prior — if the data ever produced one — cannot
    be edited out of this figure the way it used to be able to disappear
    from both reported tuples (see ``analyse_brain_process``'s docstring for
    that fixed defect). The dashed line at 0 s marks ``event`` itself — per
    the clinical annotation this analysis was centred on, the point the
    asymmetric tonic activity was scored as becoming a generalized bilateral
    tonic-clonic seizure, i.e. the peak this cascade builds toward, not
    necessarily the very first twitch.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not process.onset_latency_seconds:
        raise ValueError("BrainProcess found no involved channels to visualize.")
    times, z = _beta_gamma_z_scores(data, sfreq, event, baseline_seconds, analysis_seconds)
    ordered_names = sorted(process.onset_latency_seconds, key=process.onset_latency_seconds.get)
    indices = [names.index(name) for name in ordered_names]
    matrix = z[:, indices].T
    relative_times = times - event.time_seconds
    n_rows = len(ordered_names)
    fig, ax = plt.subplots(figsize=(14, max(4, .35 * n_rows + 1.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="inferno", vmin=0,
                   vmax=max(RECRUITMENT_THRESHOLD_MAD, float(np.percentile(matrix, 99))),
                   extent=[relative_times[0], relative_times[-1], n_rows, 0])
    ax.axvline(0, color="cyan", lw=1.5, ls="--")
    ax.axvline(event.duration_seconds, color="cyan", lw=1, ls=":")

    tau_min = process.earliest_latency_seconds
    ax.axvspan(tau_min, tau_min + PRIOR_WINDOW_SECONDS, color="gold", alpha=0.10, zorder=0)
    ax.axvspan(tau_min, tau_min + SIMULTANEITY_WINDOW_SECONDS, color="lime", alpha=0.18, zorder=0)
    ax.axvline(tau_min, color="white", lw=1.1, ls="-", zorder=1)
    ax.text(tau_min + SIMULTANEITY_WINDOW_SECONDS / 2, n_rows, "simultaneity\nwindow",
           ha="center", va="top", fontsize=6.5, color="lime")
    ax.text(tau_min + PRIOR_WINDOW_SECONDS, n_rows, " prior window", ha="left", va="top",
           fontsize=6.5, color="gold")

    prior_set = set(process.prior_matched)
    earliest_set = set(process.earliest_contacts)
    for row, name in enumerate(ordered_names):
        ax.plot(process.onset_latency_seconds[name], row + .5, marker="x", color="white",
               markersize=5, markeredgewidth=1.2, zorder=2)
    labels = [("◆ " if name in prior_set else "   ") + name for name in ordered_names]
    ax.set_yticks(np.arange(n_rows) + .5, labels, fontsize=8)
    for tick, name in zip(ax.get_yticklabels(), ordered_names):
        if name in earliest_set:
            tick.set_fontweight("bold")
            tick.set_color("gold" if name in prior_set else "lime")
    ax.set_xlabel(f"Time relative to event peak (s) — {event.label!r} at {event.time_seconds:.3f} s")
    ax.set_title("Beta/gamma recruitment cascade — rows ordered by measured onset latency "
                "(data only; earliest first)")
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("median/MAD z-score (13-80 Hz energy)")
    colorbar.ax.axhline(RECRUITMENT_THRESHOLD_MAD, color="cyan", lw=1)
    colorbar.ax.text(1.6, RECRUITMENT_THRESHOLD_MAD, f"{RECRUITMENT_THRESHOLD_MAD:g} MAD\nthreshold",
                     transform=colorbar.ax.get_yaxis_transform(), fontsize=6, va="center")
    _caption(ax, "Row order and 'x' crossing markers are data-derived (measured onset latency);\n"
                "◆ before a label = named by the external clinical prior (process.prior_matched);\n"
                "bold gold/green label = globally earliest crossing(s) (process.earliest_contacts),\n"
                "gold if also named by the prior, green if not — the latter is the contact that\n"
                "would contradict the prior, made visible rather than silently dropped.",
            loc="upper right")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)
    return output


PEAK_NODE = "PEAK"


def build_seizure_graph(data: np.ndarray, sfreq: float, names: list[str],
                        event: ClinicalEvent | AnnotatedEvent | DetectedEvent, process: BrainProcess,
                        baseline_seconds: float = 30.0, analysis_seconds: float = 8.0,
                        correlation_threshold: float = 0.5, top_k_per_node: int = 4):
    """Build a NetworkX graph of how the seizure recruits channels toward the peak.

    Nodes are exactly the channels ``analyse_brain_process`` already found
    involved (``process.onset_latency_seconds`` — never a separately
    re-picked subset), plus one synthetic ``PEAK`` node standing for
    ``event`` itself. Two kinds of edges, both measured, none assumed:

    - ``PEAK``-to-channel "recruitment" spokes, weighted by how soon after
      the peak each channel was recruited (heavier for earlier recruitment) —
      the graph counterpart of ``plot_seizure_evolution``'s onset-latency
      ordering, i.e. "...goes to peak".
    - channel-to-channel "co-activation" edges from the Pearson correlation
      of the same 13-80 Hz z-score time courses that heatmap renders,
      threshold- and top-k-pruned exactly as the sister notebook
      (`sEEG_temporal_wavelet_graph_colab.ipynb`) prunes its db4-correlation
      graphs — i.e. "...how it starts [and] evolves", read from which
      channels' activity actually co-varies, not an assumed propagation path.

    Node attributes (``onset_latency_seconds``/``latency_seconds`` — same
    value, the latter added for a self-explanatory GraphML export —
    ``peak_z``, ``is_initiator``, ``role``, ``in_prior``, ``hemisphere``) and
    edge attributes (``kind``, ``weight``) are enough to recreate
    ``plot_seizure_graph``'s figure, or to export/analyse the graph directly
    (e.g. ``networkx.write_graphml``). ``role`` is one of ``"earliest"``,
    ``"prior_early"``, or ``"later_recruited"`` — the same three-way,
    prior-independent-then-prior-widened partition ``analyse_brain_process``
    computes over ``process.onset_latency_seconds`` (see its docstring);
    ``in_prior`` is whether ``process.prior_matched`` names the channel at
    all, independent of ``role``. Drawing both lets a reader see directly
    whether a node's data-derived role and its prior membership agree.
    """
    import networkx as nx
    if not process.onset_latency_seconds:
        raise ValueError("BrainProcess found no involved channels to graph.")
    times, z = _beta_gamma_z_scores(data, sfreq, event, baseline_seconds, analysis_seconds)
    involved = sorted(process.onset_latency_seconds, key=process.onset_latency_seconds.get)
    indices = [names.index(name) for name in involved]
    correlation = np.corrcoef(z[:, indices], rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)

    earliest_set = set(process.earliest_contacts)
    prior_early_set = set(process.likely_initiators) - earliest_set
    prior_matched_set = set(process.prior_matched)

    def _role(name: str) -> str:
        if name in earliest_set:
            return "earliest"
        if name in prior_early_set:
            return "prior_early"
        return "later_recruited"

    graph = nx.Graph(event_time_seconds=event.time_seconds, event_label=event.label,
                     hemisphere_of_earliest=process.hemisphere_of_earliest,
                     earliest_latency_seconds=process.earliest_latency_seconds,
                     prior_source=process.prior_source,
                     initiators_constrained_by_prior=process.initiators_constrained_by_prior)
    graph.add_node(PEAK_NODE, kind="peak", label=event.label)
    for name in involved:
        latency = process.onset_latency_seconds[name]
        graph.add_node(name, kind="channel", onset_latency_seconds=latency, latency_seconds=latency,
                       peak_z=process.channel_band_scores.get(name, 0.0),
                       is_initiator=name in process.likely_initiators,
                       role=_role(name), in_prior=name in prior_matched_set,
                       hemisphere=hemisphere_of_channel(name) or "unknown")
        graph.add_edge(PEAK_NODE, name, kind="recruitment", weight=1.0 / (1.0 + latency),
                       latency_seconds=latency)

    magnitude = np.abs(correlation)
    selected = set()
    for row in range(len(involved)):
        candidates = np.flatnonzero(magnitude[row] >= correlation_threshold)
        if candidates.size:
            strongest = candidates[np.argsort(magnitude[row, candidates])[-top_k_per_node:]]
            selected.update(tuple(sorted((row, int(col)))) for col in strongest if col != row)
    for row, col in selected:
        graph.add_edge(involved[row], involved[col], kind="co-activation",
                       weight=float(correlation[row, col]))
    return graph


def _seizure_graph_layout(graph, channel_nodes: list[str], layout: str, seed: int):
    """Compute a ``{node: (x, y)}`` layout for ``plot_seizure_graph``.

    Four layouts, all deterministic (``seed``) and all placing ``PEAK`` at
    the origin, so the same figure-reading convention ("centre = peak")
    holds across every one of them — only how the channels are arranged
    around it differs:

    - ``"radial"`` (default): angle from a spring layout of only the
      co-activation mesh (so correlated channels cluster angularly), radius
      from recruitment latency (closer = recruited sooner). Reads outside-in
      as the seizure converges on the peak; this is the layout used before
      multiple layouts existed.
    - ``"spring"``: a single standard force-directed layout over the whole
      graph (mesh + recruitment spokes together, weighted), letting both
      edge kinds jointly shape the picture instead of only the mesh.
    - ``"circular"``: channels placed evenly around a circle ordered by
      recruitment latency, so position reads left-to-right/around as a
      clock face of "when", with no correlation structure involved at all —
      a plain, uncluttered reference to compare the other layouts against.
    - ``"shell"``: two concentric rings, initiators on the inner ring and
      all other involved channels on the outer ring, emphasizing the
      initiator/later-recruited split ``analyse_brain_process`` already
      makes rather than latency as a continuum or correlation structure.
    """
    import networkx as nx
    mesh = nx.Graph()
    mesh.add_nodes_from(channel_nodes)
    mesh.add_edges_from((u, v, d) for u, v, d in graph.edges(data=True) if d.get("kind") == "co-activation")

    if layout == "radial":
        angular_pos = nx.spring_layout(mesh, seed=seed, weight="weight")
        max_latency = max(graph.nodes[node]["onset_latency_seconds"] for node in channel_nodes) or 1.0
        pos = {PEAK_NODE: (0.0, 0.0)}
        for node in channel_nodes:
            x, y = angular_pos[node]
            angle = np.arctan2(y, x)
            radius = 0.2 + 0.8 * (graph.nodes[node]["onset_latency_seconds"] / max_latency)
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))
        return pos

    if layout == "spring":
        return nx.spring_layout(graph, seed=seed, weight="weight")

    if layout == "circular":
        ordered = sorted(channel_nodes, key=lambda node: graph.nodes[node]["onset_latency_seconds"])
        pos = nx.circular_layout(ordered)
        pos[PEAK_NODE] = (0.0, 0.0)
        return pos

    if layout == "shell":
        initiators = [node for node in channel_nodes if graph.nodes[node]["is_initiator"]]
        others = [node for node in channel_nodes if node not in initiators]
        shells = [shell for shell in (initiators, others) if shell]
        pos = nx.shell_layout(channel_nodes, nlist=shells) if shells else {}
        pos[PEAK_NODE] = (0.0, 0.0)
        return pos

    raise ValueError(f"Unknown layout {layout!r}; choose one of "
                     "'radial', 'spring', 'circular', 'shell'.")


_GRAPH_ROLE_COLORS = {"earliest": "crimson", "prior_early": "darkorange", "later_recruited": "steelblue"}
_GRAPH_ROLE_LABELS = {
    "earliest": f"earliest (data: latency ≤ τmin+{SIMULTANEITY_WINDOW_SECONDS:g}s)",
    "prior_early": f"prior_early (in prior & latency ≤ τmin+{PRIOR_WINDOW_SECONDS:g}s)",
    "later_recruited": "later_recruited (everything else)",
}
# Above this many channel nodes, only the semantically load-bearing ones
# (earliest, plus every prior-named contact) keep their text label — dense
# label clutter otherwise makes a ~100-channel graph unreadable. The earliest
# contact(s) always keep theirs regardless of graph size.
_GRAPH_LABEL_ALL_BELOW = 40


def plot_seizure_graph(graph, output: str | Path, layout: str = "radial", seed: int = 7) -> Path:
    """Render a ``build_seizure_graph`` result as a node-link figure.

    ``layout`` selects how channels are arranged around ``PEAK``; see
    ``_seizure_graph_layout`` for the four choices. ``PEAK`` sits at the
    centre as a black star in every one. Every channel node encodes three
    independent things, on purpose kept as three separate visual channels
    rather than folded into one: **fill colour = role** (crimson/orange/blue
    for ``earliest``/``prior_early``/``later_recruited`` — data-derived, see
    ``build_seizure_graph``); **ring colour = prior membership** (gold ring
    when ``in_prior`` is true, thin grey otherwise — externally supplied);
    **size = peak z-score** (bigger = stronger activation). A node whose
    fill and ring agree (crimson-or-orange fill with a gold ring, or blue
    fill with a grey ring) is a case where the data-derived role and the
    external prior line up; a mismatch (blue fill with a gold ring, or
    crimson fill with a grey ring) is exactly where they don't — the most
    informative thing this figure can show, and the reason the two
    encodings are never merged into a single colour. Every earliest-role
    node keeps its text label even when other labels are thinned for
    legibility above ``_GRAPH_LABEL_ALL_BELOW`` channels. Recruitment
    spokes are faint; the co-activation mesh (red for negative, grey for
    positive correlation) carries the visual weight. A legend identifies
    every marker and edge kind, and a boxed caption states the data-only
    located source (earliest contact(s), hemisphere) alongside
    ``likely_initiators`` in words, mirroring ``describe_seizure_source``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx
    channel_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "channel"]
    if not channel_nodes:
        raise ValueError("Graph has no channel nodes to plot.")
    pos = _seizure_graph_layout(graph, channel_nodes, layout, seed)

    role_of = {node: graph.nodes[node].get("role", "later_recruited") for node in channel_nodes}
    in_prior_of = {node: bool(graph.nodes[node].get("in_prior", False)) for node in channel_nodes}
    peak_z = {node: graph.nodes[node]["peak_z"] for node in channel_nodes}
    vmax = (max(RECRUITMENT_THRESHOLD_MAD, float(np.percentile(list(peak_z.values()), 99))) if peak_z
           else RECRUITMENT_THRESHOLD_MAD)
    sizes = {node: 90 + 260 * min(peak_z[node] / vmax, 1.0) for node in channel_nodes}

    mesh_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "co-activation"]
    mesh_colors = ["firebrick" if graph.edges[u, v]["weight"] < 0 else "0.65" for u, v in mesh_edges]
    spoke_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "recruitment"]

    fig, ax = plt.subplots(figsize=(14, 13))
    nx.draw_networkx_edges(graph, pos, edgelist=spoke_edges, edge_color="0.85", width=0.5, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=mesh_edges, edge_color=mesh_colors, width=0.9,
                           alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=[PEAK_NODE], node_shape="*", node_size=1400,
                           node_color="black", ax=ax)
    for role, color in _GRAPH_ROLE_COLORS.items():
        role_nodes = [node for node in channel_nodes if role_of[node] == role]
        prior_nodes = [node for node in role_nodes if in_prior_of[node]]
        other_nodes = [node for node in role_nodes if not in_prior_of[node]]
        if prior_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=prior_nodes, node_shape="o",
                                   node_size=[sizes[node] for node in prior_nodes],
                                   node_color=color, edgecolors="gold", linewidths=2.2, ax=ax)
        if other_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=other_nodes, node_shape="o",
                                   node_size=[sizes[node] for node in other_nodes],
                                   node_color=color, edgecolors="0.35", linewidths=0.6, ax=ax)

    earliest_nodes = {node for node in channel_nodes if role_of[node] == "earliest"}
    if len(channel_nodes) > _GRAPH_LABEL_ALL_BELOW:
        prior_nodes_all = {node for node in channel_nodes if in_prior_of[node]}
        labelled_nodes = earliest_nodes | prior_nodes_all
    else:
        labelled_nodes = set(channel_nodes)
    nx.draw_networkx_labels(graph, pos, labels={node: node for node in labelled_nodes},
                            font_size=6, ax=ax)
    ax.annotate(graph.graph["event_label"], pos[PEAK_NODE], xytext=(0, -18),
               textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(f"Seizure recruitment graph ({layout} layout) — PEAK: "
                f"{graph.graph['event_label']!r} at {graph.graph['event_time_seconds']:.3f} s")
    ax.axis("off")

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=18,
              label="PEAK — resolved event"),
        *[Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=_GRAPH_ROLE_LABELS[role])
         for role, color in _GRAPH_ROLE_COLORS.items()],
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", markeredgecolor="gold",
              markeredgewidth=2.2, markersize=10, label="ring = in_prior (named by the external clinical prior)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", markeredgecolor="0.35",
              markersize=10, label="thin ring = not named by the prior"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", markersize=6,
              label="small = low peak z-score"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", markersize=15,
              label="large = high peak z-score"),
        Line2D([0], [0], color="0.85", lw=2, label="Recruitment path to PEAK (weight = 1/(1+latency))"),
        Line2D([0], [0], color="0.65", lw=2, label="Co-activation edge (positive correlation)"),
        Line2D([0], [0], color="firebrick", lw=2, label="Co-activation edge (negative correlation)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
             fontsize=8, title="Legend: fill = role (data), ring = prior, size = peak z", frameon=True)

    if channel_nodes:
        initiators = [node for node in channel_nodes if graph.nodes[node]["is_initiator"]]
        earliest_names = sorted(earliest_nodes, key=lambda node: graph.nodes[node]["onset_latency_seconds"])
        min_latency = min(graph.nodes[node]["onset_latency_seconds"] for node in channel_nodes)
        source_text = (
            "Located source (data only, no prior)\n"
            f"Earliest contact(s): {', '.join(earliest_names) if earliest_names else '(none)'}\n"
            f"Hemisphere of earliest: {graph.graph.get('hemisphere_of_earliest', 'unknown')}\n"
            f"Time: {graph.graph['event_time_seconds'] + min_latency:.3f} s from EDF start "
            f"({min_latency:+.3f} s vs. PEAK)\n"
            f"likely_initiators (prior ∩ data): "
            f"{', '.join(initiators) if initiators else '(none)'}\n"
            f"Involved channels: {len(channel_nodes)}")
        _caption(ax, source_text, loc="lower left")

    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)
    return output


DEFAULT_SEIZURE_GRAPH_LAYOUTS = ("radial", "spring", "circular", "shell")


def plot_seizure_graph_layouts(graph, output_dir: str | Path, stem: str,
                               layouts: tuple[str, ...] = DEFAULT_SEIZURE_GRAPH_LAYOUTS,
                               seed: int = 7) -> dict[str, Path]:
    """Render the same graph in every layout in ``layouts``.

    No single layout is "the" seizure graph: ``radial`` reads temporally
    (outside-in to the peak), ``spring`` lets edges alone decide structure,
    ``circular`` is a plain latency clock face with no correlation structure
    at all, and ``shell`` isolates the initiator/later-recruited split. One
    file per layout, named ``<stem>_seizure_graph_<layout>.png``. Returns a
    ``{layout: path}`` dict.
    """
    output_dir = Path(output_dir)
    return {layout: plot_seizure_graph(graph, output_dir / f"{stem}_seizure_graph_{layout}.png",
                                       layout=layout, seed=seed)
           for layout in layouts}


def simulate_message_passing(graph, steps: int = 6, alpha: float = 0.5) -> tuple[list[str], np.ndarray]:
    """Diffuse each channel's real post-peak activation through the graph.

    A single linear message-passing update, run for ``steps`` iterations:
    ``h(t+1) = alpha * h(t) + (1 - alpha) * D^-1 W h(t)``, where ``W`` is the
    absolute co-activation weight between channels (edge sign captures phase
    relationship, not connection strength, so only magnitude drives
    diffusion) and ``D`` is each channel's row-sum degree. The seed ``h(0)``
    is each channel's already-measured ``peak_z`` — the maximum z-score
    ``analyse_brain_process`` found for it across the post-peak analysis
    window, not a synthetic or uniform value, though also not exactly the
    instantaneous value at ``event.time_seconds`` (see
    ``evaluate_message_passing``'s step-0 correlation, which is therefore
    informative rather than trivially 1.0) — so this models how the graph's
    *static*, measured co-activation structure alone would spread that real
    starting condition outward. ``evaluate_message_passing`` then checks the
    result against what the recording actually did next. Returns
    ``(channel_order, states)`` where ``states`` is
    ``[steps + 1, n_channels]``, row 0 being the seed.
    """
    channel_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "channel"]
    if not channel_nodes:
        raise ValueError("Graph has no channel nodes to propagate.")
    count = len(channel_nodes)
    index_of = {node: index for index, node in enumerate(channel_nodes)}
    weights = np.zeros((count, count), dtype=float)
    for u, v, edge_data in graph.edges(data=True):
        if edge_data.get("kind") != "co-activation":
            continue
        i, j = index_of[u], index_of[v]
        weights[i, j] = weights[j, i] = abs(edge_data["weight"])
    degree = weights.sum(axis=1)
    transition = np.divide(weights, degree[:, None], out=np.zeros_like(weights), where=degree[:, None] > 0)

    state = np.array([graph.nodes[node]["peak_z"] for node in channel_nodes], dtype=float)
    states = [state.copy()]
    for _ in range(steps):
        state = alpha * state + (1 - alpha) * (transition @ state)
        states.append(state.copy())
    return channel_nodes, np.stack(states)


def evaluate_message_passing(data: np.ndarray, sfreq: float, names: list[str],
                             event: ClinicalEvent | AnnotatedEvent | DetectedEvent,
                             channel_order: list[str], states: np.ndarray,
                             baseline_seconds: float = 30.0, analysis_seconds: float = 8.0,
                             elapsed_seconds_per_step: float | None = None) -> dict[str, list[float]]:
    """Check simulated diffusion against what the recording actually did next.

    For each propagation step, the simulated state is spatially (i.e.
    cross-channel, at one instant) Pearson-correlated against the real
    measured 13-80 Hz z-score at ``event.time_seconds + step *
    elapsed_seconds_per_step`` (default ``analysis_seconds / steps``, so the
    last step lands at the end of the same analysis window
    ``analyse_brain_process`` uses). This is a real check, not a
    demonstration: a falling correlation means the graph's static structure,
    built from one moment, does not explain the seizure's actual temporal
    evolution beyond that moment.
    """
    steps = states.shape[0] - 1
    if elapsed_seconds_per_step is None:
        elapsed_seconds_per_step = analysis_seconds / max(steps, 1)
    times, z = _beta_gamma_z_scores(data, sfreq, event, baseline_seconds, analysis_seconds)
    indices = [names.index(name) for name in channel_order]
    elapsed_seconds, correlation = [], []
    for step in range(steps + 1):
        elapsed = step * elapsed_seconds_per_step
        nearest = int(np.argmin(np.abs(times - (event.time_seconds + elapsed))))
        real_row, simulated_row = z[nearest, indices], states[step]
        if np.std(real_row) < 1e-12 or np.std(simulated_row) < 1e-12:
            value = float("nan")
        else:
            value = float(np.corrcoef(simulated_row, real_row)[0, 1])
        elapsed_seconds.append(elapsed)
        correlation.append(value)
    return {"elapsed_seconds": elapsed_seconds, "correlation": correlation}


def plot_message_passing(graph, channel_order: list[str], states: np.ndarray, output: str | Path,
                         layout: str = "spring", seed: int = 7, max_panels: int = 6) -> Path:
    """Small-multiples of the network state at each propagation step.

    Positions come from ``_seizure_graph_layout`` (same convention as
    ``plot_seizure_graph``: ``PEAK`` at centre); node colour is the simulated
    state at that step, on one shared colour scale so panels are directly
    comparable. At most ``max_panels`` evenly-spaced steps are shown (all of
    them, for the default ``simulate_message_passing`` step count). Defaults
    to the ``"spring"`` layout rather than ``"radial"``: here the step
    progression itself already carries the temporal "starts/evolves/peaks"
    story, so a layout using both edge kinds tends to keep loosely-connected
    outlier channels from dominating each panel's axis scale the way
    ``"radial"``'s latency-only radius can.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx
    pos = _seizure_graph_layout(graph, channel_order, layout, seed)
    total_steps = states.shape[0]
    panel_steps = np.unique(np.linspace(0, total_steps - 1, min(max_panels, total_steps)).astype(int))
    vmin, vmax = float(states.min()), max(float(states.max()), 1e-6)
    mesh_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "co-activation"]

    fig, axes = plt.subplots(1, len(panel_steps), figsize=(4.2 * len(panel_steps), 4.9))
    axes = np.atleast_1d(axes)
    for axis, step in zip(axes, panel_steps):
        nx.draw_networkx_edges(graph, pos, edgelist=mesh_edges, edge_color="0.8", width=0.4,
                               alpha=0.5, ax=axis)
        nx.draw_networkx_nodes(graph, pos, nodelist=channel_order, node_size=60,
                               node_color=states[step], cmap="inferno", vmin=vmin, vmax=vmax, ax=axis)
        nx.draw_networkx_nodes(graph, pos, nodelist=[PEAK_NODE], node_shape="*", node_size=260,
                               node_color="black", ax=axis)
        axis.set_title(f"step {step}")
        axis.axis("off")
    fig.text(0.5, 1.10,
            "Each panel = one diffusion step seeded from every channel's real measured peak "
            "activation, spread across the co-activation edges above; compare against "
            "plot_message_passing_validation for whether this matches what the recording did next.",
            ha="center", va="bottom", fontsize=7.5)
    fig.suptitle(f"Message-passing diffusion of the peak-moment state ({layout} layout) — "
                f"{graph.graph['event_label']!r}", y=1.02)
    colorbar_source = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin, vmax))
    fig.colorbar(colorbar_source, ax=list(axes), shrink=0.7, label="simulated activation (z-score units)")

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=16,
              label="PEAK — resolved event"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", markersize=9,
              label="Channel (colour = simulated activation this step)"),
        Line2D([0], [0], color="0.7", lw=1.5, label="Co-activation edge (diffusion path)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, -0.04))

    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight"); plt.close(fig)
    return output


DEFAULT_MESSAGE_PASSING_LAYOUTS = ("spring", "radial", "circular", "shell")


def plot_message_passing_layouts(graph, channel_order: list[str], states: np.ndarray,
                                 output_dir: str | Path, stem: str,
                                 layouts: tuple[str, ...] = DEFAULT_MESSAGE_PASSING_LAYOUTS,
                                 seed: int = 7, max_panels: int = 6) -> dict[str, Path]:
    """Render the same ``simulate_message_passing`` run in every layout in ``layouts``.

    Mirrors ``plot_seizure_graph_layouts`` for the *dynamic* figure: the
    identical diffusion states are drawn on top of every one of
    ``_seizure_graph_layout``'s four arrangements, so how the signal spreads
    from the source toward ``PEAK`` can be read outside-in by latency
    (``radial``), from combined graph structure (``spring``), against a
    latency-only clock face with no correlation structure (``circular``), or
    split by the initiator/later-recruited grouping (``shell``) — the same
    choice already offered for the static recruitment graph, now for its
    propagation. One file per layout, named
    ``<stem>_message_passing_<layout>.png``.
    """
    output_dir = Path(output_dir)
    return {layout: plot_message_passing(
                graph, channel_order, states,
                output_dir / f"{stem}_message_passing_{layout}.png",
                layout=layout, seed=seed, max_panels=max_panels)
           for layout in layouts}


def plot_message_passing_validation(evaluation: dict[str, list[float]], output: str | Path) -> Path:
    """Line plot of simulated-vs-real spatial correlation across propagation steps.

    Reads as an evaluation, not a demo: correlation near 1 at a given
    elapsed time means the graph's static co-activation structure, alone,
    reproduces which channels were actually active then; a falling curve
    means the real dynamics outrun what a graph built from a single moment
    can explain.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(evaluation["elapsed_seconds"], evaluation["correlation"], marker="o", color="navy")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(xlabel="Elapsed time after peak (s)",
          ylabel="Spatial correlation: simulated vs. measured",
          title="Message-passing validation against observed post-peak dynamics",
          ylim=(-1.05, 1.05))
    _caption(ax, "+1 = diffusion matches which channels were really active;\n"
                "0 = no relationship; -1 = opposite pattern.", loc="lower right")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170); plt.close(fig)
    return output


def _render_timeseries_panel(ax, data: np.ndarray, sfreq: float, names: list[str], max_points: int,
                             start_seconds: float = 0.0) -> np.ndarray:
    """Downsample-for-display, robust-normalize, and plot one panel of channel traces.

    Shared by both panels ``plot_all_timeseries`` draws: the stride is
    computed from *this* array's own sample count against ``max_points``, so
    a short zoomed window (few samples) ends up barely downsampled at all
    even though the whole-recording panel (many samples) is downsampled
    heavily — each panel keeps whatever resolution its own time span allows.
    Returns the per-channel y-offsets used, so a caller can anchor event
    annotations at the same height as the top trace.
    """
    stride = max(1, int(np.ceil(data.shape[1] / max_points)))
    shown = data[:, ::stride]
    times = start_seconds + np.arange(0, data.shape[1], stride)[:shown.shape[1]] / sfreq
    shown = shown - np.nanmedian(shown, axis=1, keepdims=True)
    scale = 1.4826 * np.nanmedian(np.abs(shown), axis=1)
    scale = np.where(scale > 0, scale, np.nanstd(shown, axis=1))
    normalized = shown / np.where(scale > 0, scale, 1)[:, None]
    offsets = np.arange(len(names))[::-1] * 8.
    for trace, offset in zip(normalized, offsets):
        ax.plot(times, np.clip(trace, -3.5, 3.5) + offset, color="black", lw=.35)
    ax.set_yticks(offsets, names)
    ax.margins(x=0)
    return offsets


def _mark_event(ax, event: ClinicalEvent | AnnotatedEvent | DetectedEvent, label_y: float,
                include_annotation_cluster: bool) -> None:
    """Draw one event marker (line + shaded duration + label) on ``ax``.

    The marker style reports its own provenance: a ``ClinicalEvent`` (solid
    crimson) is an expert CLI-supplied time; an ``AnnotatedEvent`` (solid
    teal) is the clinician's own marker read from the EDF's annotation
    channel; a ``DetectedEvent`` (dashed orange) is an algorithmic guess.
    Only the dashed style should ever be read as unconfirmed.

    When ``include_annotation_cluster`` is set and ``event`` is an
    ``AnnotatedEvent`` with more than one matched annotation (see
    ``find_annotated_event``/``_cluster_seizure_annotation``), every other
    annotation in the cluster is also marked (thin grey dotted line) and
    labelled with its own text — the clinician's real-time notes bracketing
    the seizure (e.g. an "onset?" query before it, a clinical-sign note
    during it), so the panel shows the seizure's actual course as scored,
    not just its single anchor instant.
    """
    if isinstance(event, DetectedEvent):
        color, style, prefix = "darkorange", "--", "detected: "
    elif isinstance(event, AnnotatedEvent):
        color, style, prefix = "teal", "-", "EDF annotation: "
    else:
        color, style, prefix = "crimson", "-", ""
    ax.axvline(event.time_seconds, color=color, lw=1.5, ls=style)
    ax.axvspan(event.time_seconds, event.time_seconds + event.duration_seconds,
               color=color, alpha=.15)
    ax.annotate(f"{prefix}{event.label}\n{event.time_seconds:.3f} s", (event.time_seconds, label_y),
                xytext=(8, 10), textcoords="offset points", color=color, rotation=90,
                va="bottom", fontsize=9)
    if include_annotation_cluster and isinstance(event, AnnotatedEvent) and len(event.annotations) > 1:
        for onset, description in event.annotations:
            if onset == event.time_seconds:
                continue
            ax.axvline(onset, color="0.4", lw=1.0, ls=":")
            ax.annotate(description, (onset, label_y), xytext=(8, 10), textcoords="offset points",
                        color="0.3", rotation=90, va="bottom", fontsize=8)


def _mark_recruitment_path(ax, process: BrainProcess, names: list[str], offsets: np.ndarray,
                           event: ClinicalEvent | AnnotatedEvent | DetectedEvent) -> None:
    """Overlay the measured recruitment cascade on a zoomed raw-timeseries panel.

    One point per channel ``process.onset_latency_seconds`` measured a
    crossing for, placed at that channel's own actual crossing instant
    (``event.time_seconds + latency``) on its own trace — ``likely_initiators``
    in crimson, every other ``later_recruited`` channel in goldenrod. Points
    are connected in latency order (earliest, an initiator, to latest) so
    the line traces one visible path from source to where the seizure
    reached next, in the order it actually got there — the same latencies
    ``plot_seizure_evolution`` already renders as a heatmap, here overlaid
    directly on the real waveform that produced them instead of an abstract
    channel-by-time image.
    """
    offset_of = dict(zip(names, offsets))
    initiators = set(process.likely_initiators)
    ordered = sorted((name for name in process.onset_latency_seconds if name in offset_of),
                     key=process.onset_latency_seconds.get)
    if not ordered:
        return
    points = [(event.time_seconds + process.onset_latency_seconds[name], offset_of[name]) for name in ordered]
    xs, ys = zip(*points)
    ax.plot(xs, ys, color="crimson", lw=1.0, alpha=0.5, zorder=3)
    for name in ordered:
        is_initiator = name in initiators
        ax.plot(event.time_seconds + process.onset_latency_seconds[name], offset_of[name], marker="o",
               markersize=5.5 if is_initiator else 3.5, color="crimson" if is_initiator else "goldenrod",
               markeredgecolor="black", markeredgewidth=0.4, zorder=4)
    for tick, name in zip(ax.get_yticklabels(), names):
        if name in initiators:
            tick.set_color("crimson"); tick.set_fontweight("bold")


def plot_all_timeseries(data: np.ndarray, sfreq: float, names: list[str], output: str | Path,
                        event: ClinicalEvent | AnnotatedEvent | DetectedEvent | None = None,
                        process: BrainProcess | None = None, max_points: int = 12000,
                        zoom_before_seconds: float = 15.0, zoom_after_seconds: float = 60.0) -> Path:
    """Render every channel over the complete EDF, downsampling only the display.

    With no ``event``, this is one panel: the whole recording, downsampled
    to ``max_points`` for display. With an ``event``, a second panel is
    added beneath it: a ``zoom_before_seconds``-before to
    ``zoom_after_seconds``-after window around ``event.time_seconds``,
    rendered at that window's own (much finer) resolution rather than the
    whole recording's. The top panel alone compresses a seizure lasting a
    few tens of seconds into a handful of pixels on a recording that may
    span hours — real enough to mark *that* something happened, useless for
    seeing *how*: this panel is where the seizure's actual shape (onset
    ramp, sustained ictal activity, decline) becomes visible, and — for an
    ``AnnotatedEvent`` — where every annotation the clinician made around it
    (not just the one seizure-keyword match) is shown as its own marker.

    When ``process`` (``analyse_brain_process``'s result for the same
    ``event``) is also given, ``_mark_recruitment_path`` overlays the
    data-derived recruitment cascade on the zoomed panel: one point per
    involved channel at its own measured crossing instant, connected in
    latency order from ``likely_initiators`` (crimson) onward through every
    ``later_recruited`` channel (goldenrod) — the actual "way from initiators
    to the event" the seizure took, drawn on the real trace that produced it
    rather than a separate abstract figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if event is None:
        fig, ax = plt.subplots(figsize=(20, max(7, .28 * len(names))))
        _render_timeseries_panel(ax, data, sfreq, names, max_points)
        ax.set(xlabel="Time from EDF start (s)", ylabel="EEG channels (from `names`)",
              title="Whole-recording sEEG overview (robust-normalized display)")
        _caption(ax, "Each trace is median/MAD-normalized and offset per channel\n"
                    "(display only — not physical amplitude).", loc="upper left")
        fig.tight_layout()
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180); plt.close(fig)
        return output

    panel_height = max(7, .28 * len(names))
    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(20, panel_height * 1.7),
                                           gridspec_kw={"height_ratios": [1, 1.3]})
    offsets = _render_timeseries_panel(ax_full, data, sfreq, names, max_points)
    _mark_event(ax_full, event, offsets[0], include_annotation_cluster=False)
    ax_full.set(xlabel="Time from EDF start (s)", ylabel="EEG channels (from `names`)",
               title="Whole-recording sEEG overview (robust-normalized display)")
    _caption(ax_full, "Each trace is median/MAD-normalized and offset per channel\n"
                "(display only — not physical amplitude).", loc="upper left")

    total_seconds = data.shape[1] / sfreq
    effective_before_seconds = zoom_before_seconds
    if process is not None and process.onset_latency_seconds:
        earliest_latency = min(process.onset_latency_seconds.values())
        if earliest_latency < -zoom_before_seconds:
            # A recruitment crossing lands earlier than the default window: widen the
            # window itself (+2s margin) rather than plot a point with no underlying
            # trace under it -- an overlay point past the edge of the rendered signal
            # is exactly the "empty part of the figure" this widening avoids.
            effective_before_seconds = -earliest_latency + 2.0
    zoom_start = max(0.0, event.time_seconds - effective_before_seconds)
    zoom_end = min(total_seconds, event.time_seconds + zoom_after_seconds)
    start_sample, end_sample = int(round(zoom_start * sfreq)), int(round(zoom_end * sfreq))
    zoom_offsets = _render_timeseries_panel(ax_zoom, data[:, start_sample:end_sample], sfreq, names,
                                            max_points, start_seconds=zoom_start)
    _mark_event(ax_zoom, event, zoom_offsets[0], include_annotation_cluster=True)
    caption = ("Same robust normalization as above, but over just the event\n"
              "window at (near-)full time resolution, so the seizure's own\n"
              "onset/course/decline is visible instead of compressed away.")
    if process is not None and process.onset_latency_seconds:
        _mark_recruitment_path(ax_zoom, process, names, zoom_offsets, event)
        caption += ("\n● crimson = likely_initiators, ● goldenrod = later_recruited,\n"
                   "placed at each channel's own measured recruitment crossing;\n"
                   "line = recruitment order, initiator(s) to last channel reached.")
    ax_zoom.set(xlabel="Time from EDF start (s)", ylabel="EEG channels (from `names`)",
               title=f"Seizure course, −{effective_before_seconds:.0f}s/+{zoom_after_seconds:.0f}s "
                     f"around {event.label!r} at {event.time_seconds:.3f} s")
    _caption(ax_zoom, caption, loc="upper left")

    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)
    return output


def run_edf(path: str | Path, output_dir: str | Path, event: ClinicalEvent | None = None,
           montage_reference: str = "none", crop_end_seconds: float | None = None) -> EdfRunResult:
    """Run detection, context analysis, and whole-recording visualization.

    ``montage_reference`` selects which signal reference every skill below
    analyses: ``"none"`` (default) uses the recording's native/referential
    channels exactly as loaded; ``"bipolar"`` re-references to
    ``apply_bipolar_montage``'s adjacent-contact differences first, so
    detection, process analysis, the recruitment graph, and message passing
    all run against local spatial gradients instead — this suppresses
    whatever the two contacts shared (reference noise, distant volume
    conduction) and keeps only what differs between them. The bipolar
    montage *structure* (``build_bipolar_montage``, read from the channel
    names) is always written to ``<stem>_montage.txt``, independent of
    ``montage_reference``, since that structure exists whether or not it's
    actually applied. Use ``compare_montages`` to run both and get a
    result you can directly compare.

    ``crop_end_seconds`` is passed straight to ``read_edf``/``read_edf_markers``
    to discard a known non-physiological tail (see ``read_edf``) before the
    blind detector, the whole-recording overview figure, or any other skill
    below ever sees it. ``None`` (the default) analyses the complete file.

    ``event`` (an explicit expert ``--event-time``/``--event-clock``) always
    wins when given. Otherwise the EDF's own annotation channel is checked
    via ``find_annotated_event`` — the clinician's own real-time markup, read
    from the file rather than typed as an apriori number, and unaffected by
    ``montage_reference`` since it reads text annotations, not the signal —
    and used if it names a seizure. Only when neither is available does
    ``select_seizure_event`` fall back to the agent's blind statistical
    ranking, which *is* affected by ``montage_reference``. Whichever one is
    used drives the beta/gamma process analysis, the whole-recording
    overview figure (which includes the ``MKR...`` marker channels for
    visual/QC context even though ``read_edf`` keeps them out of detection
    and process analysis), and — when the process found involved channels —
    a plain-text ``describe_seizure_source`` summary (source channel(s) and
    absolute time, written to ``<stem>_source_summary.txt``), the
    ``plot_seizure_evolution`` heatmap, every layout from
    ``plot_seizure_graph_layouts``, a ``simulate_message_passing`` /
    ``evaluate_message_passing`` run rendered by ``plot_message_passing`` and
    ``plot_message_passing_validation``, and that same diffusion re-rendered
    in every layout by ``plot_message_passing_layouts``. See
    :class:`EdfRunResult` for the full, named result shape.
    """
    if montage_reference not in ("none", "bipolar"):
        raise ValueError(f"montage_reference must be 'none' or 'bipolar', got {montage_reference!r}.")
    data, sfreq, names = read_edf(path, crop_end_seconds=crop_end_seconds)
    bipolar_montage = build_bipolar_montage(names)
    if montage_reference == "bipolar":
        data, names = apply_bipolar_montage(data, names, bipolar_montage)

    report = ExtremeEventAgent().run(data, sfreq, names)
    annotated = None if event is not None else find_annotated_event(path)
    detected = None if (event is not None or annotated is not None) else select_seizure_event(report.events)
    context = event or annotated or detected
    process = analyse_brain_process(data, sfreq, names, context) if context else None

    output_dir = Path(output_dir)
    stem = Path(path).stem
    montage_file = output_dir / f"{stem}_montage.txt"
    montage_file.parent.mkdir(parents=True, exist_ok=True)
    montage_file.write_text(format_bipolar_montage(bipolar_montage), encoding="utf-8")
    marker_data, _, marker_names = read_edf_markers(path, crop_end_seconds=crop_end_seconds)
    overview_data = np.concatenate([data, marker_data], axis=0) if marker_names else data
    overview_names = names + marker_names
    overview_figure = plot_all_timeseries(overview_data, sfreq, overview_names,
                                          output_dir / f"{stem}_all_timeseries.png", context, process)

    evolution_figure = None
    graph_figures: dict[str, Path] = {}
    graph_graphml = None
    message_passing_figure = message_passing_validation_figure = None
    message_passing_figures: dict[str, Path] = {}
    message_passing_evaluation = None
    source_summary = source_summary_file = None
    if process is not None and process.onset_latency_seconds:
        source_summary = describe_seizure_source(process)
        source_summary_file = output_dir / f"{stem}_source_summary.txt"
        source_summary_file.parent.mkdir(parents=True, exist_ok=True)
        source_summary_file.write_text(source_summary, encoding="utf-8")
        evolution_figure = plot_seizure_evolution(
            data, sfreq, names, context, process, output_dir / f"{stem}_seizure_evolution.png")
        graph = build_seizure_graph(data, sfreq, names, context, process)
        graph_figures = plot_seizure_graph_layouts(graph, output_dir, stem)
        import networkx as nx
        graph_graphml = output_dir / f"{stem}_seizure_graph.graphml"
        nx.write_graphml(graph, graph_graphml)
        channel_order, states = simulate_message_passing(graph)
        message_passing_evaluation = evaluate_message_passing(
            data, sfreq, names, context, channel_order, states)
        message_passing_figure = plot_message_passing(
            graph, channel_order, states, output_dir / f"{stem}_message_passing.png")
        message_passing_figures = plot_message_passing_layouts(graph, channel_order, states,
                                                                output_dir, stem)
        message_passing_validation_figure = plot_message_passing_validation(
            message_passing_evaluation, output_dir / f"{stem}_message_passing_validation.png")

    return EdfRunResult(report, process, bipolar_montage, montage_reference, montage_file, overview_figure,
                        evolution_figure, graph_figures, graph_graphml, message_passing_figure,
                        message_passing_validation_figure, message_passing_evaluation, annotated, detected,
                        message_passing_figures=message_passing_figures, source_summary=source_summary,
                        source_summary_file=source_summary_file)


def compare_montages(path: str | Path, output_dir: str | Path, event: ClinicalEvent | None = None,
                     montage_references: tuple[str, ...] = ("none", "bipolar"),
                     crop_end_seconds: float | None = None) -> dict[str, EdfRunResult]:
    """Run ``run_edf`` once per entry in ``montage_references``, one subdirectory each.

    Directly answers "how does montage choice change the result": every
    downstream skill — detection, initiators, the recruitment graph,
    message-passing validation — runs against the *same* recording and
    (whenever tier 1/2 resolves an event) the *same* event time, varying
    only the signal reference the event-independent tier-3 fallback and all
    signal-derived analysis see. ``crop_end_seconds`` is forwarded to every
    ``run_edf`` call unchanged, so every montage reference analyses the same
    (possibly truncated) range — see ``read_edf``. Results land in
    ``<output_dir>/<stem>/<montage_reference>/``. Pass the result to
    ``summarize_montage_comparison`` for one directly comparable table.
    """
    output_dir = Path(output_dir)
    stem = Path(path).stem
    return {reference: run_edf(path, output_dir / stem / reference, event, montage_reference=reference,
                               crop_end_seconds=crop_end_seconds)
           for reference in montage_references}


def summarize_montage_comparison(results: dict[str, EdfRunResult]) -> list[dict[str, object]]:
    """Reduce a ``compare_montages`` result to one comparable row per montage.

    Reports what actually differs, not just structure: the number of
    candidates the blind detector found, how many channels the process
    analysis found involved, the likely-initiator set, the co-activation
    mesh's edge count (a rough proxy for how much of the referential
    correlation structure was shared-reference artifact rather than surviving
    re-referencing), and the message-passing validation's best and mean
    spatial correlation against real subsequent dynamics.

    ``earliest_contacts``/``hemisphere_of_earliest`` are included alongside
    ``likely_initiators`` precisely because they are prior-free: two montages
    agreeing on ``likely_initiators`` is partly guaranteed by both having run
    against the same ``ContactPrior`` contact list, so it is not, by itself,
    an independent cross-check of the right-frontal hypothesis. Whether
    ``earliest_contacts`` (and ``hemisphere_of_earliest``) also agree across
    montages is the check that owes nothing to the prior.
    """
    rows = []
    for reference, result in results.items():
        mesh_edge_count = None
        if result.graph_graphml is not None:
            import networkx as nx
            graph = nx.read_graphml(result.graph_graphml)
            mesh_edge_count = sum(1 for _, _, edge_data in graph.edges(data=True)
                                  if edge_data.get("kind") == "co-activation")
        correlations = (result.message_passing_evaluation or {}).get("correlation", [])
        finite = [value for value in correlations if value == value]  # drop NaN without importing math/np here
        rows.append({
            "montage_reference": reference,
            "n_channels_analysed": len(result.report.channel_names),
            "n_detected_candidates": len(result.report.events),
            "n_involved_channels": len(result.process.onset_latency_seconds) if result.process else 0,
            "likely_initiators": list(result.process.likely_initiators) if result.process else [],
            "earliest_contacts": list(result.process.earliest_contacts) if result.process else [],
            "hemisphere_of_earliest": result.process.hemisphere_of_earliest if result.process else "unknown",
            "co_activation_edges": mesh_edge_count,
            "message_passing_best_correlation": max(finite) if finite else None,
            "message_passing_mean_correlation": (sum(finite) / len(finite)) if finite else None,
        })
    return rows
