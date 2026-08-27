"""Multimodal (DICOM x timeseries) cross-checks for the sEEG pipeline.

This package is deliberately scoped to what the actual dataset supports: one
post-implant T1 + T2 MRI session (``dataset/MRI-with-electrodes/DCM``), no
CT, no diffusion MRI, no pre-implant baseline, and no manually placed
electrode fiducials. See ``multimodal_approach/README.md`` for what that
does and does not let this module claim.
"""

from .dicom_geometry import VolumeGeometry, load_series_geometry
from .structural_anomaly import StructuralAnomalyResult, run_structural_anomaly
from .extreme_event_prior import StructuralPriorReport, apply_structural_prior
from .structural_graph import (
    build_structural_anomaly_graph, plot_structural_anomaly_graph, plot_structural_anomaly_graph_anatomical,
)

__all__ = [
    "VolumeGeometry",
    "load_series_geometry",
    "StructuralAnomalyResult",
    "run_structural_anomaly",
    "StructuralPriorReport",
    "apply_structural_prior",
    "build_structural_anomaly_graph",
    "plot_structural_anomaly_graph",
    "plot_structural_anomaly_graph_anatomical",
]
