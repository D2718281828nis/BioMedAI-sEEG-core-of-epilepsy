"""Graph neural network pipeline: train a node classifier on an already-built seizure graph.

Separate from every other top-level package (``model/`` the reservoir plant,
``object_model/`` the multi-layer graph builder, ``multimodal_approach/`` the
structural block): this one never touches an EDF or DICOM file directly, it
only consumes the GraphML those pipelines already wrote. See
``gnn_model.run_gnn`` for the entry point.
"""
