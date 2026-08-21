"""Echo State Network: the hidden-state core of the reservoir plant.

A classic Echo State Network (Jaeger, 2001), written out explicitly as a
discrete-time nonlinear state-space system so the Automatic Control Theory
reading is literal, not a metaphor:

    x(t) = (1 - leak) * x(t-1) + leak * tanh(W_in @ u(t) + W @ x(t-1) + bias)   # state equation
    y(t) = W_out @ [x(t); u(t)]                                                # output equation

``W`` (the *reservoir*) and ``W_in`` are fixed random matrices, generated
once at construction and never trained — only ``W_out`` (the linear
readout, playing the combined role of a state-space system's ``C`` and
``D`` matrices) is fit, by ridge regression, to real target data. This is
reservoir computing's central trick: a large, fixed, nonlinear dynamical
system supplies rich temporal features "for free"; all the learning happens
in one linear layer on top of it.

This class is agnostic to what ``u(t)`` actually contains — :mod:`model.plant`
is the module that decides it should be the ``MKR...`` clock *plus* a short
delay embedding of the target's own recent past (a NARX-style extension;
see its module docstring), since a reservoir driven only by a near-constant
clock has essentially no information to reconstruct fast EEG structure
from — that concern belongs entirely to the caller, not to this generic
state-space core.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReservoirConfig:
    """Hyperparameters of the hidden-state dynamics and its training.

    Defaults follow standard ESN practice (see e.g. Jaeger 2001; Lukoševičius
    & Jaeger 2009): a spectral radius near/at 1 for a system operating at the
    edge of stability (rich, slowly-decaying memory) without violating the
    echo state property, a fractional leak rate so the state integrates
    several input steps rather than reacting to only the latest one, and a
    sparse reservoir (most units not directly coupled) so the dynamics are
    heterogeneous rather than one global oscillation.
    """

    n_reservoir: int = 400
    spectral_radius: float = 0.95
    input_scaling: float = 1.0
    leak_rate: float = 0.3
    sparsity: float = 0.1
    bias_scaling: float = 0.2
    ridge_alpha: float = 1e-2
    washout: int = 100
    output_feedback_lag: int = 6
    seed: int = 7

    def __post_init__(self) -> None:
        if self.n_reservoir < 4:
            raise ValueError("n_reservoir must be at least 4.")
        if not 0 < self.spectral_radius <= 3:
            raise ValueError("spectral_radius must be in (0, 3] (values >1 remain usable with leaky "
                             "integration but are unusual outside chaotic-system modelling).")
        if not 0 < self.leak_rate <= 1:
            raise ValueError("leak_rate must be in (0, 1].")
        if not 0 < self.sparsity <= 1:
            raise ValueError("sparsity must be in (0, 1].")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative.")
        if self.washout < 0:
            raise ValueError("washout must be non-negative.")
        if self.output_feedback_lag < 0:
            raise ValueError("output_feedback_lag must be non-negative.")


class EchoStateNetwork:
    """A fixed random reservoir plus a trainable linear readout.

    ``W`` (state matrix, analogous to ``A``) and ``W_in`` (input matrix,
    analogous to ``B``) are drawn once from ``config.seed`` and rescaled so
    ``W``'s spectral radius matches ``config.spectral_radius`` — the
    standard echo-state-property control: too high and the state's memory of
    its own history dominates and never forgets transients (unstable
    long-term dependence on initial conditions); too low and the reservoir
    forgets its input almost immediately (little temporal memory to read
    out). ``W_out`` (the readout, analogous to ``[C  D]`` acting on
    ``[x; u]``) starts unset and is only assigned by :meth:`fit_readout`.
    """

    def __init__(self, n_inputs: int, n_outputs: int, config: ReservoirConfig | None = None) -> None:
        if n_inputs < 1 or n_outputs < 1:
            raise ValueError("n_inputs and n_outputs must each be at least 1.")
        self.config = config or ReservoirConfig()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        rng = np.random.default_rng(self.config.seed)
        n = self.config.n_reservoir

        self.W_in = rng.uniform(-1.0, 1.0, size=(n, n_inputs)) * self.config.input_scaling
        mask = rng.random((n, n)) < self.config.sparsity
        raw = rng.uniform(-1.0, 1.0, size=(n, n)) * mask
        spectral_radius_raw = float(np.max(np.abs(np.linalg.eigvals(raw)))) if mask.any() else 0.0
        self.W = raw * (self.config.spectral_radius / spectral_radius_raw) if spectral_radius_raw > 1e-12 else raw
        self.bias = rng.uniform(-1.0, 1.0, size=n) * self.config.bias_scaling
        self.achieved_spectral_radius = float(np.max(np.abs(np.linalg.eigvals(self.W)))) if n <= 2000 else None

        self.W_out: np.ndarray | None = None

    def step(self, u_t: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        """One state-equation update: the reservoir's instantaneous dynamics."""
        pre_activation = self.W_in @ u_t + self.W @ x_prev + self.bias
        leak = self.config.leak_rate
        return (1.0 - leak) * x_prev + leak * np.tanh(pre_activation)

    def run_states(self, U: np.ndarray) -> np.ndarray:
        """Drive the reservoir with input sequence ``U`` ``[T, n_inputs]``.

        Purely input-driven (no output feedback loop), starting from
        ``x(0) = 0``, so it can be run forward across data the readout was
        never fit on — including the extreme event itself — using only the
        input signal, exactly as a physical plant's state evolves from a
        known control input regardless of whether the observer "believes"
        the resulting output. Returns hidden states ``X`` ``[T, n_reservoir]``.
        """
        if U.ndim != 2 or U.shape[1] != self.n_inputs:
            raise ValueError(f"U must have shape [T, {self.n_inputs}], got {U.shape}.")
        states = np.empty((U.shape[0], self.config.n_reservoir))
        state = np.zeros(self.config.n_reservoir)
        for t in range(U.shape[0]):
            state = self.step(U[t], state)
            states[t] = state
        return states

    def fit_readout(self, X: np.ndarray, U: np.ndarray, Y: np.ndarray, washout: int | None = None
                    ) -> np.ndarray:
        """Ridge-regress the linear readout ``W_out`` onto target output ``Y``.

        ``washout`` (default ``config.washout``) drops the first samples,
        which still carry the arbitrary ``x(0) = 0`` initial condition
        rather than the reservoir's true (input-dependent) trajectory — the
        standard echo-state-network transient discard. Fits on
        ``[X, U]`` jointly (state *and* direct input feedthrough, the ``[C
        D]`` block of a state-space output equation) rather than on ``X``
        alone, so a component of the target that tracks the input directly
        does not have to be reconstructed indirectly through the reservoir.
        Returns the per-output RMSE on the same (washout-trimmed) data it was
        fit on — a training diagnostic, not a held-out validation score.
        """
        washout = self.config.washout if washout is None else washout
        if X.shape[0] != U.shape[0] or X.shape[0] != Y.shape[0]:
            raise ValueError("X, U, and Y must share the same number of timesteps.")
        if X.shape[0] - washout < 2:
            raise ValueError(f"Only {X.shape[0]} samples but washout={washout}; nothing left to fit.")
        extended = np.concatenate([X, U], axis=1)[washout:]
        target = Y[washout:]
        regularizer = self.config.ridge_alpha * np.eye(extended.shape[1])
        self.W_out = np.linalg.solve(extended.T @ extended + regularizer, extended.T @ target).T
        predicted = extended @ self.W_out.T
        return np.sqrt(np.mean((predicted - target) ** 2, axis=0))

    def predict(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Apply the trained readout: ``y(t) = W_out @ [x(t); u(t)]`` for every ``t``."""
        if self.W_out is None:
            raise ValueError("Readout is not fitted; call fit_readout first.")
        extended = np.concatenate([X, U], axis=1)
        return extended @ self.W_out.T
