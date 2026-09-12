"""Detrended Fluctuation Analysis (Peng et al. 1994) -- the same method
``sEEG_extreme_event_detector_colab.ipynb``'s five-method ensemble uses (see the top-level
README, "Data-driven extreme-event detection notebook"), reimplemented here in plain
NumPy/SciPy so ``gnn_model`` doesn't need that notebook or a new dependency to use it as a
node feature.

DFA estimates a scaling exponent alpha describing a time series' long-range (self-similar)
fluctuation structure: alpha ~ 0.5 for uncorrelated noise, alpha ~ 1.0 for 1/f ("pink") noise,
alpha > 1 for a non-stationary, random-walk-like signal, alpha < 0.5 for anti-correlated
("mean-reverting") fluctuations. The question this module exists to ask is whether that
exponent, measured per sEEG channel over its own pre-event baseline, differs systematically
between "earliest" (seizure-initiating) and "later_recruited" channels -- a candidate feature
``gnn_model.data`` cannot compute on its own since it has no access to the raw signal (see that
module's docstring on the EDF/GraphML boundary this package otherwise keeps).
"""
from __future__ import annotations

import numpy as np

__all__ = ["dfa_alpha"]


def dfa_alpha(signal: np.ndarray, min_window: int = 4, max_window: int | None = None,
             num_windows: int = 20, order: int = 1) -> float:
    """Return the DFA scaling exponent of a 1-D ``signal``, or ``nan`` if it is too short to
    estimate one (fewer than two usable window sizes).

    ``min_window``/``max_window`` (samples; ``max_window`` defaults to ``len(signal) // 4``,
    the standard rule of thumb so every window has at least 4 non-overlapping repeats) bound the
    log-spaced window sizes ``num_windows`` are drawn from; ``order`` is the polynomial order used
    to detrend each window (1 = linear, the standard/original DFA).
    """
    signal = np.asarray(signal, dtype=np.float64)
    n_samples = signal.shape[0]
    if max_window is None:
        max_window = n_samples // 4
    if max_window < min_window or n_samples < 2 * min_window:
        return float("nan")

    profile = np.cumsum(signal - signal.mean())
    window_sizes = np.unique(np.logspace(np.log10(min_window), np.log10(max_window),
                                         num=num_windows).astype(int))

    log_sizes, log_fluctuations = [], []
    for window in window_sizes:
        n_segments = n_samples // window
        if n_segments < 2:
            continue
        segments = profile[:n_segments * window].reshape(n_segments, window)
        t = np.arange(window, dtype=np.float64)
        design = np.vander(t, order + 1)
        # Least-squares polynomial fit per segment via the pseudo-inverse of the shared design
        # matrix -- one matmul for all segments instead of a Python-level polyfit loop per segment.
        coeffs = np.linalg.pinv(design) @ segments.T
        trends = (design @ coeffs).T
        rms = np.sqrt(np.mean((segments - trends) ** 2, axis=1))
        log_sizes.append(np.log(window))
        log_fluctuations.append(np.log(rms.mean()))

    if len(log_sizes) < 2:
        return float("nan")
    slope, _ = np.polyfit(log_sizes, log_fluctuations, 1)
    return float(slope)
