"""The reservoir as an input-output plant over one EDF recording.

Wires :class:`model.reservoir.EchoStateNetwork` to real sEEG data with a
specific, deliberate input/output split: the ``MKR...`` hardware-clock
channels are the plant's observed *exogenous* input ``u(t)`` (``extreme_event_agent``
keeps them out of detection because they carry no brain signal — but that
is exactly what makes them a clean, artifact-free reference input to drive
a state-space system with, independent of the very channels being modelled
as output); a subset of the real EEG channels is the observed output
``y(t)``.

A clock alone, though, is nearly constant between its 1 Hz pulses and so
carries almost no information correlated with fast EEG structure — a
reservoir driven by it alone settles near a fixed point and its linear
readout collapses to predicting close to each channel's mean, missing the
real waveform entirely. ``run_reservoir_plant`` therefore drives the
reservoir with an *augmented* input: ``u(t)`` concatenated with a short
delay embedding of the target's own recent, already-observed past
(``y(t-1), ..., y(t-lag)``) — a standard NARX ("nonlinear autoregressive
with exogenous input") extension to reservoir computing, and the same
delay-embedding idea behind the "next-generation RC" approach in this
project's own reference implementation. This never leaks ``y(t)`` itself
(only strictly earlier samples), so a one-step-ahead prediction built from
it is still a legitimate, checkable forecast — including through the
extreme event, where the baseline-fit autoregressive relationship is
exactly what may no longer hold.

The readout is trained only on the pre-event baseline — i.e. what the
mapping from (reservoir state, augmented input) to real channel activity
looks like during "normal" dynamics. :func:`run_reservoir_plant` then runs
the *same* trained model forward through the extreme event, and measures
the residual between what the nominal model predicts and what the
recording actually did — a classic observer-residual fault/anomaly signal
from control theory, used here as the model's own extreme-event evaluation,
independent of (but checked against) ``extreme_event_agent``'s own
detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from extreme_event_agent.edf_workflow import (EDF_ENCODING, MARKER, analyse_brain_process,
                                              find_annotated_event, hemisphere_of_channel)
from extreme_event_agent.models import AnnotatedEvent, BrainProcess, ClinicalEvent, DetectedEvent

from .reservoir import EchoStateNetwork, ReservoirConfig

EventContext = ClinicalEvent | AnnotatedEvent | DetectedEvent
DEFAULT_MAX_OUTPUT_CHANNELS = 12
# "recruitment" (default) is what channel_selection has always used: the same
# analyse_brain_process recruitment analysis this plant is meant to
# independently check against, so agreement between the two is not an
# independent cross-check by construction. "balanced" fixes that: it never
# looks at recruitment/latency, only at hemisphere (from channel naming) and
# pre-event variance — so an arbitration between EDF and reservoir evidence
# is only valid (see ReservoirWindow.arbitration_valid) when this mode was
# used.
CHANNEL_SELECTION_METHODS = ("recruitment", "balanced")


@dataclass
class ReservoirWindow:
    """The plant's input ``u``, target output ``y``, and bookkeeping for one run.

    ``times_seconds`` is relative to the resolved event (``0.0`` = event
    time), so every downstream figure and threshold reads directly as
    "before/after the event" without re-deriving an offset. ``input_data``
    is ``[T, n_inputs]`` (the ``MKR...`` channels); ``output_data`` is
    ``[T, n_outputs]`` (the selected EEG channels) — both already
    time-major, ready for :class:`~model.reservoir.EchoStateNetwork`.

    ``arbitration_valid`` is ``True`` only when ``channel_selection_method``
    is ``"balanced_hemisphere_variance"`` (i.e. ``build_window`` was called
    with ``channel_selection="balanced"``) -- the only mode whose channel
    choice owes nothing to ``analyse_brain_process``'s own recruitment
    analysis. A lateralization estimate read off a plant built with
    ``"recruitment"``/``"brain_process_initiators_plus_spread"`` channels is
    circular with respect to that same recruitment analysis and should not
    be reported as an independent confirmation of it -- this flag lets a
    caller (e.g. ``extreme_event_agent.verification``) enforce that without
    re-deriving which selection method was used.
    """

    times_seconds: np.ndarray
    sfreq: float
    input_names: list[str]
    input_data: np.ndarray
    output_names: list[str]
    output_data: np.ndarray
    event: EventContext
    baseline_seconds: float
    analysis_seconds: float
    channel_selection_method: str
    process: BrainProcess | None
    arbitration_valid: bool = False


@dataclass
class ReservoirEvaluation:
    """One trained-and-run reservoir plant, plus its extreme-event verdict.

    ``score`` is a robust (median/MAD) z-score of the per-timestep output
    residual magnitude, normalized against the *baseline* portion of
    ``score`` only — by construction it centers near 0 during the baseline
    the readout was trained on and is only "surprising" where real dynamics
    depart from what that trained model predicts. ``threshold`` (6 MAD,
    matching ``analyse_brain_process``'s own recruitment threshold) and
    ``detected``/``onset_time_seconds`` turn that into the same kind of
    checkable claim ``evaluate_message_passing`` makes: does this
    independent, differently-built model also flag something at the known
    event, or not.

    ``reservoir_input_names``/``reservoir_input_data`` are what the
    reservoir was *actually* driven with — ``window.input_names``/
    ``input_data`` (the exogenous ``MKR...`` clock) plus the NARX delay
    embedding of ``window.output_data`` (see module docstring); ``esn.n_inputs
    == len(reservoir_input_names)``. ``window.input_names``/``input_data``
    themselves are untouched, so they still describe the plant's literal
    exogenous input.

    ``score``/``detected``/``onset_time_seconds``/``training_rmse`` collapse
    ``residual`` across every output channel into one scalar per timestep —
    useful as a single extreme-event verdict, but it cannot say *where*: a
    channel with a naturally larger baseline residual would dominate
    ``magnitude`` and be mistaken for "the" locus. ``per_channel_score``
    (``[T, n_outputs]``) instead normalizes each output channel's residual
    independently against its *own* baseline (same median/MAD-then-std
    fallback ``score`` itself uses), so a spatial read of *which* channel's
    predictability breaks down, and *when*, is possible without one
    high-amplitude channel drowning out the rest.
    ``per_channel_onset_seconds``/``per_channel_peak_score``/
    ``per_channel_peak_time_seconds`` reduce that per-channel course the same
    way the scalar ``onset_time_seconds``/``peak_score``/``peak_time_seconds``
    reduce ``score`` — smoothed-crossing onset (``None`` if a channel never
    crosses ``threshold``), raw peak value, and raw peak time, one entry per
    ``window.output_names``. This is a *residual* location, not a lesion
    location: "where this model's prediction of normal dynamics fails
    first," not a claim about anatomical source.
    """

    esn: EchoStateNetwork
    window: ReservoirWindow
    reservoir_input_names: list[str]
    reservoir_input_data: np.ndarray
    hidden_states: np.ndarray
    predicted_output: np.ndarray
    residual: np.ndarray
    score: np.ndarray
    smoothed_score: np.ndarray
    threshold: float
    peak_time_seconds: float
    peak_score: float
    detected: bool
    onset_time_seconds: float | None
    washout_end_seconds: float
    training_rmse: dict[str, float] = field(default_factory=dict)
    per_channel_score: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    per_channel_onset_seconds: dict[str, float | None] = field(default_factory=dict)
    per_channel_peak_score: dict[str, float] = field(default_factory=dict)
    per_channel_peak_time_seconds: dict[str, float] = field(default_factory=dict)


def resolve_event_context(path: str | Path, event_time: float | None = None,
                          event_label: str = "асимметричный тонический приступ",
                          event_duration: float = 2.0) -> EventContext:
    """The same three-tier event resolution ``run_edf`` uses, kept lean.

    Tier 1 (explicit ``event_time``) and tier 2 (``find_annotated_event``,
    reading only the EDF's text annotations) are cheap — neither loads
    signal data. Tier 3 (blind statistical detection) is only reached, and
    only then loads the full recording, when neither of the above resolves
    anything; unlike ``run_edf`` this function does *not* always pay that
    cost, since a plant model needs exactly one event time, not a full
    candidate report.
    """
    if event_time is not None:
        return ClinicalEvent(event_time, event_label, event_duration)
    annotated = find_annotated_event(path)
    if annotated is not None:
        return annotated
    from extreme_event_agent.agent import ExtremeEventAgent
    from extreme_event_agent.edf_workflow import read_edf, select_seizure_event
    data, sfreq, names = read_edf(path)
    report = ExtremeEventAgent().run(data, sfreq, names)
    detected = select_seizure_event(report.events)
    if detected is None:
        raise ValueError(f"{path}: no explicit --event-time, no EDF annotation, and blind detection "
                         "found no candidate — no event time available to build a window around.")
    return detected


def _read_edf_window(path: str | Path, tmin: float, tmax: float
                     ) -> tuple[np.ndarray, dict[str, int], float, list[str], list[str]]:
    """Load exactly ``[tmin, tmax]`` of every channel — signal *and* ``MKR...``.

    Unlike ``extreme_event_agent.edf_workflow.read_edf`` (whole-recording,
    ``MKR...`` excluded, tail-only crop), this needs an arbitrary window
    around an event that can be hours into the recording, and needs the
    ``MKR...`` channels *included* — they are this plant's input.
    """
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR", encoding=EDF_ENCODING)
    max_tmax = (raw.n_times - 1) / raw.info["sfreq"]
    if tmin >= max_tmax:
        raise ValueError(f"Window start {tmin:.3f}s is at or past the recording end ({max_tmax:.3f}s).")
    raw.crop(tmin=max(0.0, tmin), tmax=min(tmax, max_tmax))
    raw.load_data(verbose="ERROR")
    all_names = list(raw.ch_names)
    input_names = [name for name in all_names if MARKER.fullmatch(name.strip())]
    output_names_all = [name for name in all_names if name not in input_names]
    return raw.get_data(), {name: index for index, name in enumerate(all_names)}, \
        float(raw.info["sfreq"]), input_names, output_names_all


def _select_balanced_channels(eeg_data: np.ndarray, eeg_names: list[str], sfreq: float,
                              baseline_seconds: float, max_output_channels: int) -> list[str]:
    """Split output channels evenly across hemispheres, ranked within each by baseline-only variance.

    Deliberately never looks at recruitment, latency, or any post-event
    sample -- unlike ``"recruitment"`` selection, this makes no claim at all
    about where the event started, only about which channels are typically
    the noisiest during quiet baseline activity, split evenly so neither
    hemisphere's channel count can itself bias a downstream lateralization
    estimate. Variance is computed over the baseline segment only
    (``eeg_data[:, :baseline_samples]``), never the post-event portion, so
    the choice is genuinely blind to the event this plant will later be run
    through. Channels ``hemisphere_of_channel`` cannot classify (not an SEEG
    contact/pair label) are excluded rather than silently assigned a side.
    """
    baseline_samples = max(1, min(eeg_data.shape[1], int(round(baseline_seconds * sfreq))))
    baseline = eeg_data[:, :baseline_samples]
    variance = np.var(baseline, axis=1)
    sides = [hemisphere_of_channel(name) for name in eeg_names]
    per_side = max(1, max_output_channels // 2)
    selected: list[str] = []
    for side in ("right", "left"):
        indices = sorted((i for i, s in enumerate(sides) if s == side), key=lambda i: variance[i], reverse=True)
        selected += [eeg_names[i] for i in indices[:per_side]]
    return selected[:max_output_channels]


def _select_output_channels(eeg_data: np.ndarray, eeg_names: list[str], sfreq: float,
                            baseline_seconds: float, analysis_seconds: float, max_output_channels: int,
                            channel_selection: str = "recruitment"
                            ) -> tuple[list[str], str, BrainProcess | None]:
    """Pick which EEG channels the plant tries to reproduce as output ``y``.

    ``channel_selection="recruitment"`` (default) reuses ``analyse_brain_process``
    — the same audited recruitment analysis ``build_seizure_graph`` already
    relies on — against a synthetic event placed exactly at
    ``baseline_seconds`` into this already-windowed array, so the channel
    selection is grounded in the same evidence as the rest of the pipeline
    rather than a fresh, separate heuristic. Prefers the likely-initiator
    (source) channels plus an evenly-spaced sample of later-recruited ones,
    capped at ``max_output_channels`` so the readout and every figure stay
    legible. Falls back to the highest-variance channels in the window only
    if that analysis finds nothing involved (e.g. a window with no real
    event in it) — never silently empty output.

    ``channel_selection="balanced"`` uses ``_select_balanced_channels``
    instead — see its docstring for why: a plant whose output channels come
    from ``analyse_brain_process`` cannot then be used to independently
    confirm or contest that same analysis's lateralization (see
    ``ReservoirWindow.arbitration_valid`` and
    ``CHANNEL_SELECTION_METHODS``'s module comment). Falls through to the
    ``"recruitment"`` path (never raises) only if no channel in this window
    has a classifiable hemisphere at all — an edge case, not the normal
    behavior of ``"balanced"``.
    """
    if channel_selection not in CHANNEL_SELECTION_METHODS:
        raise ValueError(f"channel_selection must be one of {CHANNEL_SELECTION_METHODS}, "
                         f"got {channel_selection!r}.")
    if channel_selection == "balanced":
        selected = _select_balanced_channels(eeg_data, eeg_names, sfreq, baseline_seconds, max_output_channels)
        if selected:
            return selected, "balanced_hemisphere_variance", None

    synthetic_event = ClinicalEvent(time_seconds=baseline_seconds,
                                    duration_seconds=max(0.1, analysis_seconds * 0.25))
    process = None
    try:
        process = analyse_brain_process(eeg_data, sfreq, eeg_names, synthetic_event,
                                        baseline_seconds=baseline_seconds, analysis_seconds=analysis_seconds)
    except ValueError:
        process = None
    if process is not None and process.onset_latency_seconds:
        selected = list(process.likely_initiators)
        later = [name for name in process.later_recruited if name not in selected]
        budget = max_output_channels - len(selected)
        if later and budget > 0:
            step = max(1, len(later) // budget)
            selected += later[::step][:budget]
        if not selected:
            selected = sorted(process.onset_latency_seconds, key=process.onset_latency_seconds.get)
        return selected[:max_output_channels], "brain_process_initiators_plus_spread", process
    variance = np.var(eeg_data, axis=1)
    order = np.argsort(variance)[::-1][:max_output_channels]
    return [eeg_names[i] for i in sorted(order)], "highest_variance_fallback", process


def build_window(path: str | Path, context: EventContext, baseline_seconds: float = 60.0,
                 analysis_seconds: float = 20.0, output_channels: list[str] | None = None,
                 max_output_channels: int = DEFAULT_MAX_OUTPUT_CHANNELS,
                 channel_selection: str = "recruitment") -> ReservoirWindow:
    """Load one ``[event - baseline_seconds, event + analysis_seconds]`` window.

    Widened relative to ``analyse_brain_process``'s own default (30 s / 8 s):
    the reservoir's readout needs enough *baseline* samples to fit a
    meaningful ridge regression (default 60 s at 256 Hz is ~15 000 samples),
    and the plant needs to be run for long enough past the event
    (default 20 s) for a genuine divergence to appear rather than reading
    one or two noisy points.

    ``channel_selection`` (``"recruitment"`` or ``"balanced"``; ignored when
    ``output_channels`` is given explicitly) is forwarded to
    ``_select_output_channels`` — see its docstring and
    ``CHANNEL_SELECTION_METHODS``'s module comment for what each means and
    why it matters for whether this window's later lateralization estimate
    is independent of ``analyse_brain_process``'s own recruitment analysis.
    """
    tmin = max(0.0, context.time_seconds - baseline_seconds)
    tmax = context.time_seconds + analysis_seconds
    data, index, sfreq, input_names, eeg_names_all = _read_edf_window(path, tmin, tmax)
    if not input_names:
        raise ValueError(f"{path} has no MKR... marker channel to use as the plant's input.")
    actual_baseline = context.time_seconds - tmin
    input_data = data[[index[name] for name in input_names]]
    eeg_data_all = data[[index[name] for name in eeg_names_all]]

    process = None
    if output_channels is not None:
        missing = [name for name in output_channels if name not in eeg_names_all]
        if missing:
            raise ValueError(f"output_channels not found in this window's recording: {missing}")
        method = "user_specified"
    else:
        output_channels, method, process = _select_output_channels(
            eeg_data_all, eeg_names_all, sfreq, actual_baseline, analysis_seconds, max_output_channels,
            channel_selection=channel_selection)
    eeg_index = {name: i for i, name in enumerate(eeg_names_all)}
    output_data = eeg_data_all[[eeg_index[name] for name in output_channels]]
    times = (np.arange(data.shape[1]) / sfreq) - actual_baseline
    return ReservoirWindow(times_seconds=times, sfreq=sfreq, input_names=input_names,
                           input_data=input_data.T, output_names=list(output_channels),
                           output_data=output_data.T, event=context, baseline_seconds=actual_baseline,
                           analysis_seconds=analysis_seconds, channel_selection_method=method, process=process,
                           arbitration_valid=(method == "balanced_hemisphere_variance"))


def _baseline_zscore(values: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """Per-channel median/MAD z-score of ``values`` against ``values[baseline_mask]``.

    Same robust-scale convention ``run_reservoir_plant``'s own residual
    ``score`` uses (median/MAD, falling back to std when MAD is degenerate)
    — reused here so the reservoir is actually driven by its input: raw EEG
    is ~1e-4 V, three-plus orders of magnitude below ``bias_scaling`` and the
    recurrent term's typical size, so without this ``W_in @ u(t)`` is
    negligible next to ``bias`` and the state barely reacts to ``u(t)`` at
    all — it just relaxes to its own bias/recurrence fixed point and stays
    there, regardless of what the real signal does.
    """
    baseline = values[baseline_mask]
    center = np.median(baseline, axis=0)
    mad = 1.4826 * np.median(np.abs(baseline - center), axis=0)
    scale = np.where(mad > 1e-12, mad, np.maximum(np.std(baseline, axis=0), 1e-12))
    return (values - center) / scale


def _build_augmented_input(input_names: list[str], U: np.ndarray, output_names: list[str], Y: np.ndarray,
                           lag: int, baseline_mask: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Concatenate exogenous input ``U`` with a ``lag``-step delay embedding of ``Y``.

    At row ``t`` this places ``Y[t-1], ..., Y[t-lag]`` alongside ``U[t]`` —
    always strictly earlier samples, never ``Y[t]`` itself, so nothing here
    lets a one-step-ahead predictor "see" the value it is trying to predict.
    Before the embedding has ``lag`` real samples to draw from (the very
    start of the window), missing taps are filled with ``Y[0]`` (the
    earliest available real sample, not zero) so a channel that never
    crosses 0 doesn't get an artificial jump into the embedding right at the
    start. See the module docstring for why this is needed at all: ``U``
    alone (the near-constant ``MKR...`` clock) carries almost no information
    about fast EEG structure.

    Both ``U`` and ``Y`` are first passed through ``_baseline_zscore`` (per
    channel, against ``baseline_mask``) before any shifting or
    concatenation, so every value the reservoir is actually driven with is
    O(1) rather than raw-volt EEG. This only rescales the *input features*
    built here — the caller's own ``Y`` (the ``fit_readout``/``predict``
    target) is untouched, since normalizing a local reassignment of a numpy
    array never mutates the caller's array.
    """
    U = _baseline_zscore(U, baseline_mask)
    Y = _baseline_zscore(Y, baseline_mask)
    T = U.shape[0]
    blocks = [U]
    names = list(input_names)
    for step in range(1, lag + 1):
        shifted = np.empty_like(Y)
        if step < T:
            shifted[step:] = Y[:-step]
            shifted[:step] = Y[0]
        else:
            shifted[:] = Y[0]
        blocks.append(shifted)
        names.extend(f"{name}[t-{step}]" for name in output_names)
    return names, np.concatenate(blocks, axis=1)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered box-car smoothing, same length as ``values``.

    A single noisy sample crossing 6 MAD is common even in a well-fit
    baseline (residual magnitude is not perfectly Gaussian); smoothing over
    a short window before checking against the threshold is the same
    temporal-support principle ``ExtremeEventAgent`` already applies
    (candidate merging, minimum window count) — one instant does not make an
    event, a *sustained* elevation does. Edge samples (within half a window
    of either end) are averaged over their truncated, still-available
    neighborhood rather than padded with zeros, so they are not artificially
    pulled toward 0.
    """
    if window <= 1:
        return values.copy()
    ones_kernel = np.ones(window)
    summed = np.convolve(values, ones_kernel, mode="same")
    counts = np.convolve(np.ones_like(values), ones_kernel, mode="same")
    return summed / counts


def _per_channel_evaluation(residual: np.ndarray, times_seconds: np.ndarray, baseline_scored: np.ndarray,
                            washout: int, threshold: float, smoothing_samples: int, output_names: list[str]
                            ) -> tuple[np.ndarray, dict[str, float | None], dict[str, float], dict[str, float]]:
    """Per-output-channel counterpart of ``run_reservoir_plant``'s scalar score/onset/peak.

    Each channel's own absolute residual is median/MAD-normalized against
    its *own* baseline segment (std fallback, same as the scalar path) —
    computed independently per column of ``residual``, so a channel with a
    larger native residual amplitude cannot inflate another channel's score.
    Onset/peak follow the exact same convention as the scalar path: onset
    from the smoothed (sustained-elevation) score, peak from the raw score,
    both searched only from ``washout`` onward.
    """
    per_channel_score = np.empty_like(residual)
    onset_seconds: dict[str, float | None] = {}
    peak_score: dict[str, float] = {}
    peak_time_seconds: dict[str, float] = {}
    for index, name in enumerate(output_names):
        channel_magnitude = np.abs(residual[:, index])
        center = float(np.median(channel_magnitude[baseline_scored]))
        mad = 1.4826 * float(np.median(np.abs(channel_magnitude[baseline_scored] - center)))
        scale = mad if mad > 1e-12 else max(float(np.std(channel_magnitude[baseline_scored])), 1e-12)
        channel_score = (channel_magnitude - center) / scale
        per_channel_score[:, index] = channel_score
        smoothed = _moving_average(channel_score, smoothing_samples)

        raw_searchable = channel_score[washout:]
        smoothed_searchable = smoothed[washout:]
        searchable_times = times_seconds[washout:]
        peak_local = int(np.argmax(raw_searchable))
        crossings_local = np.flatnonzero(smoothed_searchable >= threshold)
        onset_seconds[name] = float(searchable_times[crossings_local[0]]) if crossings_local.size else None
        peak_score[name] = float(raw_searchable[peak_local])
        peak_time_seconds[name] = float(searchable_times[peak_local])
    return per_channel_score, onset_seconds, peak_score, peak_time_seconds


def run_reservoir_plant(window: ReservoirWindow, config: ReservoirConfig | None = None,
                        smoothing_samples: int = 5) -> ReservoirEvaluation:
    """Train the plant on baseline dynamics, run it through the event, score the residual.

    Steps, each a direct state-space-plant operation:

    0. ``_build_augmented_input`` baseline-zscores the exogenous clock input
       and the real target (per channel, against the pre-event baseline —
       see ``_baseline_zscore``), then concatenates the clock with a
       ``config.output_feedback_lag``-step delay embedding of the
       now-normalized target (see module docstring) — the actual ``u(t)``
       the reservoir is driven with. Only these input *features* are
       rescaled; ``Y`` itself (below) stays in raw units, since it is the
       ``fit_readout``/``predict`` target.
    1. ``EchoStateNetwork.run_states`` drives the hidden state ``x(t)``
       across the *entire* window (baseline and event alike) using only
       that augmented input — legitimate because every tap is a strictly
       earlier, already-observed real sample, and the state equation never
       looks at the target *at* the timestep being predicted.
    2. ``fit_readout`` trains the output equation (``y = Cx + Du``) using
       only the baseline portion (after ``config.washout``) — the plant
       learns what *normal* reservoir-to-channel dynamics look like.
    3. ``predict`` applies that fixed, baseline-only readout across the
       whole window, including the event.
    4. The residual (real minus predicted output), root-mean-squared across
       output channels per timestep, is median/MAD-normalized against its
       own baseline segment into ``score`` — near 0 where the model still
       explains the data, large where it does not.

    Peak/onset search is restricted to samples *after* ``config.washout``
    (from the very start of the window, i.e. deep in the baseline): before
    that point the state is still relaxing away from its arbitrary
    ``x(0) = 0`` initial condition (the same transient ``fit_readout``
    itself discards), which produces a large but meaningless residual spike
    that would otherwise masquerade as the extreme event. ``score`` itself
    still covers the whole window (so the transient is visible, not hidden,
    in ``plot_extreme_event_score``); only detection ignores it.

    ``detected``/``onset_time_seconds`` are read off ``smoothed_score`` — a
    ``smoothing_samples``-wide moving average of ``score`` — rather than the
    raw, single-sample series: a lone spike above 6 MAD is common even in a
    well-fit baseline (the residual magnitude is not perfectly Gaussian) and
    should not by itself read as "detected"; a *sustained* elevation should.
    ``peak_score``/``peak_time_seconds`` stay on the raw ``score``, so the
    single highest real point is always reported even where it fails the
    sustained check — see ``describe_evaluation``.
    """
    config = config or ReservoirConfig()
    Y = window.output_data
    baseline_mask = window.times_seconds < 0.0
    reservoir_input_names, U = _build_augmented_input(
        window.input_names, window.input_data, window.output_names, Y, config.output_feedback_lag, baseline_mask)
    esn = EchoStateNetwork(n_inputs=U.shape[1], n_outputs=len(window.output_names), config=config)
    X = esn.run_states(U)

    baseline_indices = np.flatnonzero(baseline_mask)
    if baseline_indices.size - config.washout < 2:
        raise ValueError(f"Only {baseline_indices.size} baseline samples but washout={config.washout}; "
                         "widen baseline_seconds or shrink washout.")
    rmse = esn.fit_readout(X[baseline_mask], U[baseline_mask], Y[baseline_mask], washout=config.washout)
    training_rmse = dict(zip(window.output_names, (float(value) for value in rmse)))

    Y_hat = esn.predict(X, U)
    residual = Y - Y_hat
    magnitude = np.sqrt(np.mean(residual ** 2, axis=1))

    baseline_scored = baseline_indices[config.washout:]
    center = float(np.median(magnitude[baseline_scored]))
    mad = 1.4826 * float(np.median(np.abs(magnitude[baseline_scored] - center)))
    scale = mad if mad > 1e-12 else max(float(np.std(magnitude[baseline_scored])), 1e-12)
    score = (magnitude - center) / scale
    smoothed_score = _moving_average(score, smoothing_samples)

    threshold = 6.0
    raw_searchable = score[config.washout:]
    smoothed_searchable = smoothed_score[config.washout:]
    searchable_times = window.times_seconds[config.washout:]
    peak_local = int(np.argmax(raw_searchable))
    crossings_local = np.flatnonzero(smoothed_searchable >= threshold)
    onset_time = float(searchable_times[crossings_local[0]]) if crossings_local.size else None

    per_channel_score, per_channel_onset, per_channel_peak, per_channel_peak_time = _per_channel_evaluation(
        residual, window.times_seconds, baseline_scored, config.washout, threshold, smoothing_samples,
        window.output_names)

    return ReservoirEvaluation(esn=esn, window=window, reservoir_input_names=reservoir_input_names,
                               reservoir_input_data=U, hidden_states=X, predicted_output=Y_hat,
                               residual=residual, score=score, smoothed_score=smoothed_score,
                               threshold=threshold, peak_time_seconds=float(searchable_times[peak_local]),
                               peak_score=float(raw_searchable[peak_local]), detected=bool(crossings_local.size),
                               onset_time_seconds=onset_time,
                               washout_end_seconds=float(window.times_seconds[config.washout]),
                               training_rmse=training_rmse, per_channel_score=per_channel_score,
                               per_channel_onset_seconds=per_channel_onset,
                               per_channel_peak_score=per_channel_peak,
                               per_channel_peak_time_seconds=per_channel_peak_time)


def describe_evaluation(evaluation: ReservoirEvaluation) -> str:
    """Plain-language statement of the reservoir plant's extreme-event verdict.

    Mirrors ``extreme_event_agent.edf_workflow.describe_seizure_source`` in
    spirit: narrates numbers already computed in ``evaluation``, doesn't
    compute anything new, and states the honest result even when the model
    does not flag the event (never overclaims a detection that didn't cross
    threshold).
    """
    window = evaluation.window
    lines = [
        f"Reference event: {window.event.label!r} at {window.event.time_seconds:.3f} s "
        f"(window: {window.baseline_seconds:.1f} s baseline, {window.analysis_seconds:.1f} s after).",
        f"Plant exogenous input u(t): {', '.join(window.input_names)}. "
        f"Plant output y(t) ({window.channel_selection_method}): {', '.join(window.output_names)}.",
        f"Reservoir actually driven by u(t) plus a {evaluation.esn.config.output_feedback_lag}-step "
        f"delay embedding of y(t) ({len(evaluation.reservoir_input_names)} input dims total) — see "
        "module docstring for why the clock alone is not enough to explain fast EEG structure.",
        f"Reservoir: {evaluation.esn.config.n_reservoir} hidden units, achieved spectral radius "
        f"{evaluation.esn.achieved_spectral_radius:.3f} (target {evaluation.esn.config.spectral_radius}).",
        f"Baseline readout RMSE per channel: "
        + ", ".join(f"{name}={value:.4g}" for name, value in evaluation.training_rmse.items()),
    ]
    if evaluation.detected:
        near_event = abs(evaluation.onset_time_seconds) <= 5.0
        proximity = ("consistent with the resolved event" if near_event else
                    "well outside a +-5 s neighbourhood of the resolved event — plausibly a separate "
                    "interictal or artifactual transient the reservoir also fails to explain, not "
                    "necessarily the seizure onset itself; verify against the raw trace before treating "
                    "it as seizure-related")
        lines.append(
            f"Extreme event DETECTED by the plant's residual: sustained score crosses "
            f"{evaluation.threshold:.1f} MAD at {evaluation.onset_time_seconds:+.3f} s relative to the "
            f"reference event ({proximity}). Peak score is {evaluation.peak_score:.2f} MAD at "
            f"{evaluation.peak_time_seconds:+.3f} s — the baseline-trained model stops explaining the "
            "real channel dynamics there too.")
    elif evaluation.peak_score >= evaluation.threshold:
        lines.append(
            f"Extreme event NOT flagged by the plant's residual: the single highest point "
            f"({evaluation.peak_score:.2f} MAD at {evaluation.peak_time_seconds:+.3f} s) does cross the "
            f"{evaluation.threshold:.1f} MAD threshold, but the smoothed (sustained-elevation) score never "
            "does — a brief spike, not a confirmed onset; report it, don't discard it.")
    else:
        lines.append(
            f"Extreme event NOT flagged by the plant's residual: peak score is only "
            f"{evaluation.peak_score:.2f} MAD (threshold {evaluation.threshold:.1f}) at "
            f"{evaluation.peak_time_seconds:+.3f} s — this reservoir/output-channel configuration does "
            "not reproduce the recruitment analysis's own detection; report both, don't discard this one.")
    return "\n".join(lines)
