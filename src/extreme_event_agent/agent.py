from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy import signal

from .models import AgentConfig, DetectionReport, Event


class ExtremeEventAgent:
    """A bounded plan-act-observe-reflect agent with deterministic tools.

    The agent does not ask an LLM to interpret patient data. It selects and runs
    numerical tools, evaluates data quality and candidate evidence, then records
    every decision in an audit log. This makes runs reproducible and reviewable.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    def run(
        self,
        data: np.ndarray,
        sampling_frequency_hz: float,
        channel_names: list[str] | tuple[str, ...] | None = None,
    ) -> DetectionReport:
        audit: list[dict[str, object]] = []
        values, names, quality = self._validate(data, sampling_frequency_hz, channel_names)
        audit.append({"phase": "observe", "tool": "validate", "quality": quality})
        if quality < self.config.min_quality:
            raise ValueError(f"Usable sample fraction {quality:.3f} is below min_quality.")

        features, starts = self._extract_features(values, sampling_frequency_hz)
        channel_scores = self._robust_channel_scores(features)
        keep = max(1, int(np.ceil(values.shape[0] * self.config.channel_fraction)))
        global_score = np.sort(channel_scores, axis=1)[:, -keep:].mean(axis=1)
        center = float(np.median(global_score))
        scale = self._mad_scale(global_score)
        threshold = center + self.config.threshold_mad * scale
        audit.append({"phase": "act", "tool": "feature_ensemble", "windows": len(starts),
                      "features": ["rms", "peak_to_peak", "line_length", "high_frequency"]})

        # Reflection is deliberately bounded. If no candidate exists, relax only
        # enough to return review candidates; never silently claim they are events.
        candidate_mask = global_score >= threshold
        used_threshold = threshold
        for iteration in range(self.config.max_iterations):
            audit.append({"phase": "reflect", "iteration": iteration + 1,
                          "threshold": used_threshold, "candidate_windows": int(candidate_mask.sum())})
            if candidate_mask.any() or iteration + 1 == self.config.max_iterations:
                break
            used_threshold = max(center + 3.0 * scale, used_threshold - scale)
            candidate_mask = global_score >= used_threshold

        events = self._verify_and_merge(candidate_mask, global_score, channel_scores, starts,
                                        sampling_frequency_hz, names, used_threshold, center, scale)
        audit.append({"phase": "finalize", "accepted_events": len(events),
                      "config": asdict(self.config)})
        return DetectionReport(events, float(sampling_frequency_hz), names, used_threshold, audit)

    def _validate(self, data, sfreq, channel_names):
        values = np.asarray(data, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 4:
            raise ValueError("data must have shape [channels, samples] with at least 4 samples.")
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("sampling_frequency_hz must be positive.")
        names = tuple(channel_names or (f"channel_{i}" for i in range(values.shape[0])))
        if len(names) != values.shape[0] or len(set(names)) != len(names):
            raise ValueError("channel_names must be unique and match the channel count.")
        finite = np.isfinite(values)
        quality = float(finite.mean())
        for index in range(values.shape[0]):
            good = finite[index]
            fill = np.median(values[index, good]) if good.any() else 0.0
            values[index, ~good] = fill
        return values, names, quality

    def _extract_features(self, values, sfreq):
        window = int(round(self.config.window_seconds * sfreq))
        step = int(round(self.config.step_seconds * sfreq))
        if window < 4 or values.shape[1] < window:
            raise ValueError("Recording is shorter than one valid analysis window.")
        starts = np.arange(0, values.shape[1] - window + 1, max(1, step))
        output = np.empty((len(starts), values.shape[0], 4), dtype=float)
        for row, start in enumerate(starts):
            chunk = signal.detrend(values[:, start:start + window], axis=1, type="linear")
            difference = np.diff(chunk, axis=1)
            output[row, :, 0] = np.sqrt(np.mean(chunk * chunk, axis=1))
            output[row, :, 1] = np.ptp(chunk, axis=1)
            output[row, :, 2] = np.mean(np.abs(difference), axis=1)
            output[row, :, 3] = np.sqrt(np.mean(difference * difference, axis=1))
        return output, starts

    def _robust_channel_scores(self, features):
        center = np.median(features, axis=0, keepdims=True)
        mad = np.median(np.abs(features - center), axis=0, keepdims=True)
        fallback = np.std(features, axis=0, keepdims=True)
        scale = np.where(1.4826 * mad > 1e-12, 1.4826 * mad,
                         np.where(fallback > 1e-12, fallback, 1.0))
        z = np.clip((features - center) / scale, 0.0, 25.0)
        return np.sort(z, axis=2)[:, :, -2:].mean(axis=2)

    @staticmethod
    def _mad_scale(values):
        mad = 1.4826 * float(np.median(np.abs(values - np.median(values))))
        return mad if mad > 1e-12 else max(float(np.std(values)), 1e-12)

    def _verify_and_merge(self, mask, scores, channel_scores, starts, sfreq, names,
                          threshold, center, scale):
        selected = np.flatnonzero(mask)
        if not len(selected):
            return []
        max_gap = self.config.window_seconds + self.config.merge_gap_seconds
        groups = [[int(selected[0])]]
        for index in selected[1:]:
            if (starts[index] - starts[groups[-1][-1]]) / sfreq <= max_gap:
                groups[-1].append(int(index))
            else:
                groups.append([int(index)])
        events = []
        for group in groups:
            peak = group[int(np.argmax(scores[group]))]
            involved = np.flatnonzero(channel_scores[peak] >= max(3.0, threshold))
            if len(involved) < min(self.config.min_involved_channels, len(names)):
                continue
            strength = max(0.0, (float(scores[peak]) - center) / scale)
            confidence = float(np.clip((strength - 3.0) / 7.0, 0.0, 1.0))
            events.append(Event(
                start_seconds=float(starts[group[0]] / sfreq),
                end_seconds=float(starts[group[-1]] / sfreq + self.config.window_seconds),
                peak_seconds=float(starts[peak] / sfreq), score=float(scores[peak]),
                confidence=confidence, involved_channels=tuple(names[i] for i in involved),
                evidence={"robust_sigma": strength, "candidate_windows": float(len(group)),
                          "involved_channel_count": float(len(involved))},
            ))
        return sorted(events, key=lambda event: event.score, reverse=True)
