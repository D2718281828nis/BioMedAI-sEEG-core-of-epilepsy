"""Assemble EDF (`extreme_event_agent`), DICOM (`multimodal_approach`), and the
reservoir plant (`model`) into one object-level model of the recording, and
verify each against the recording's own annotated event
(`extreme_event_agent.verification`).

This is the only package that imports all three of the others together --
each of them stays free of a dependency on either of its siblings, so this
package, not any one of them, is where "the object model" actually lives.
See `run_object_model.py` for the end-to-end CLI, `graph.py` for how the
three evidence layers are attached (never merged) to one NetworkX graph, and
`figure.py` for the five-panel summary figure.
"""

from .graph import build_object_model_graph

__all__ = ["build_object_model_graph"]
