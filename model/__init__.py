"""Reservoir-computing state-space model of sEEG channel dynamics.

Frames the recording exactly the way Automatic Control Theory frames a
plant: a hidden internal state ``x(t)`` (the reservoir), driven by an
observed input ``u(t)`` (the ``MKR...`` hardware-clock channels), producing
an observed output ``y(t)`` (a subset of the real EEG channels) through a
trained linear readout —

    x(t) = (1 - leak) * x(t-1) + leak * tanh(W_in u(t) + W x(t-1) + bias)   # state equation
    y(t) = C x(t) + D u(t)                                                 # output equation

— the reservoir's ``W`` plays the role of the state matrix ``A``, ``W_in``
the input matrix ``B``, and the trained readout splits into ``C`` (from the
hidden state) and ``D`` (direct input feedthrough), by analogy with a
linear state-space plant ``x' = Ax + Bu, y = Cx + Du``. Unlike a linear
plant, ``A``/``B`` here are fixed and nonlinear (the "reservoir"); only the
linear readout (``C``, ``D``) is trained, which is reservoir computing's
whole point — see :mod:`model.reservoir`.

The readout is trained only on the pre-event baseline, so it encodes what
"normal" reservoir-to-channel dynamics look like. :mod:`model.plant` then
runs the same trained model forward across the extreme event and measures
how far the real recording diverges from what the nominal model predicts —
a residual/observer-based extreme-event evaluation, the same logic behind
fault detection in control theory. See ``run_model.py`` for the end-to-end
script and :mod:`model.visualize` for the figures it produces.
"""

from .reservoir import EchoStateNetwork, ReservoirConfig
from .plant import (ReservoirEvaluation, ReservoirWindow, build_window, resolve_event_context,
                    run_reservoir_plant)

__all__ = ["EchoStateNetwork", "ReservoirConfig", "ReservoirEvaluation", "ReservoirWindow",
          "build_window", "resolve_event_context", "run_reservoir_plant"]
