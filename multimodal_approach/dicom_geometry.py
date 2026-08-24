"""DICOM series loading with full patient-space (LPS) geometry.

``dicom_viewer.viewer.load_series`` intentionally returns only axis-aligned
voxel spacing — its own docstring calls this "not a true patient-space
reformat via the full direction-cosine matrix" and notes it only holds up
because this dataset's tilt is small. That approximation is fine for
clicking through slices, but two things this package needs cannot tolerate
it:

* determining true anatomical left/right (the montage's ``PM``/``PM'``
  naming already encodes a left/right convention — see
  ``extreme_event_agent.edf_workflow.is_right_frontal`` — and confirming it
  against the image itself needs the real patient X axis, not a voxel index
  that is only approximately aligned to it);
* resampling the T2 series onto the T1 grid (a few-degree orientation error
  compounds over a 25 cm field of view into several millimetres of
  misregistration at the edges).

So this module rebuilds the actual affine from each series'
``ImageOrientationPatient``/``ImagePositionPatient``/``PixelSpacing`` instead
of assuming axis alignment. It reuses ``dicom_viewer.viewer`` for file
discovery/series grouping rather than duplicating that logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

from dicom_viewer.viewer import Series, find_dicom_files, group_series

__all__ = ["VolumeGeometry", "load_series_geometry", "list_series"]


@dataclass
class VolumeGeometry:
    """One loaded series plus the affine mapping its voxel indices to patient LPS mm.

    Voxel indices are ``(k, i, j)`` = ``(slice, row, col)``, matching
    ``volume.shape`` and matching how ``pydicom`` exposes ``pixel_array``
    (rows, columns) stacked slice-by-slice. Patient coordinates follow DICOM's
    LPS convention: +x = toward patient Left, +y = toward patient Posterior,
    +z = toward patient Head (Superior).
    """

    volume: np.ndarray  # (n_slices, n_rows, n_cols) float32, rescaled
    origin: np.ndarray  # (3,) mm, patient position of voxel (0, 0, 0)
    d_slice: np.ndarray  # (3,) mm displacement per +1 slice index
    d_row: np.ndarray  # (3,) mm displacement per +1 row index
    d_col: np.ndarray  # (3,) mm displacement per +1 col index
    series: Series

    def voxel_to_patient(self, k, i, j) -> np.ndarray:
        """Map voxel index (broadcastable arrays or scalars) to patient (x, y, z) mm."""
        k = np.asarray(k, dtype=np.float64)
        i = np.asarray(i, dtype=np.float64)
        j = np.asarray(j, dtype=np.float64)
        return (self.origin
                + k[..., None] * self.d_slice
                + i[..., None] * self.d_row
                + j[..., None] * self.d_col)

    def patient_to_voxel(self, xyz: np.ndarray) -> np.ndarray:
        """Inverse of ``voxel_to_patient``: patient (..., 3) mm -> voxel (..., 3) = (k, i, j)."""
        xyz = np.asarray(xyz, dtype=np.float64)
        basis = np.stack([self.d_slice, self.d_row, self.d_col], axis=1)  # columns: d_slice, d_row, d_col
        basis_inv = np.linalg.inv(basis)
        return (xyz - self.origin) @ basis_inv.T

    def _coordinate_component_map(self, axis: int) -> np.ndarray:
        """One patient-space coordinate component (mm) of every voxel, shape == volume.shape.

        Cheaper than ``full_coordinate_grid`` when only one axis is needed
        (hemisphere classification needs only X; a crude cerebrum-vs-skull-base
        floor needs only Z) — avoids materializing the full (k, i, j, 3) grid.
        """
        nk, ni, nj = self.volume.shape
        k_term = np.arange(nk, dtype=np.float32)[:, None, None] * np.float32(self.d_slice[axis])
        i_term = np.arange(ni, dtype=np.float32)[None, :, None] * np.float32(self.d_row[axis])
        j_term = np.arange(nj, dtype=np.float32)[None, None, :] * np.float32(self.d_col[axis])
        return (k_term + i_term + j_term + np.float32(self.origin[axis])).astype(np.float32)

    def x_coordinate_map(self) -> np.ndarray:
        """Patient-space X coordinate (mm) of every voxel, shape == volume.shape.

        +x is toward patient Left (DICOM LPS), so this alone is enough to
        classify voxels as left-/right-hemisphere once a midline X is known.
        """
        return self._coordinate_component_map(0)

    def z_coordinate_map(self) -> np.ndarray:
        """Patient-space Z coordinate (mm) of every voxel, shape == volume.shape.

        +z is toward patient Head/Superior (DICOM LPS) — used to tell cerebrum
        from neck/skull-base by height, not just to know left from right.
        """
        return self._coordinate_component_map(2)

    def full_coordinate_grid(self) -> np.ndarray:
        """Patient (x, y, z) mm for every voxel, shape (n_slices, n_rows, n_cols, 3), float32.

        Only materialized on demand (~150 MB for this dataset's T1 grid) —
        needed for mirroring across the sagittal midline, not for the cheaper
        per-hemisphere classification (``x_coordinate_map``).
        """
        nk, ni, nj = self.volume.shape
        k = np.arange(nk, dtype=np.float32)
        i = np.arange(ni, dtype=np.float32)
        j = np.arange(nj, dtype=np.float32)
        grid = (
            np.einsum("k,c->kc", k, self.d_slice.astype(np.float32))[:, None, None, :]
            + np.einsum("i,c->ic", i, self.d_row.astype(np.float32))[None, :, None, :]
            + np.einsum("j,c->jc", j, self.d_col.astype(np.float32))[None, None, :, :]
            + self.origin.astype(np.float32)
        )
        return grid


def list_series(dicom_dir) -> dict[str, Series]:
    """Discover and group every DICOM series under ``dicom_dir`` (see ``dicom_viewer``)."""
    files = find_dicom_files(Path(dicom_dir))
    if not files:
        raise FileNotFoundError(f"No DICOM files found under {dicom_dir}")
    return group_series(files)


def load_series_geometry(series: Series) -> VolumeGeometry:
    """Load one series' pixel data plus its real patient-space affine.

    Slice order follows ``series.files`` (already sorted by ``InstanceNumber``
    in ``group_series``). ``d_slice`` uses the actual position delta between
    the first two slices rather than a nominal spacing tag, matching how
    ``dicom_viewer.viewer.load_series`` derives slice spacing, so both loaders
    agree on physical slice order.
    """
    datasets = [pydicom.dcmread(f) for f in series.files]

    slope = float(getattr(datasets[0], "RescaleSlope", 1) or 1)
    intercept = float(getattr(datasets[0], "RescaleIntercept", 0) or 0)
    volume = np.stack([ds.pixel_array.astype(np.float32) for ds in datasets], axis=0)
    volume = volume * slope + intercept

    iop = np.array(datasets[0].ImageOrientationPatient, dtype=np.float64)
    d_col, d_row = iop[:3], iop[3:]  # DICOM: first triplet = row direction cosine (col index increases)
    row_spacing, col_spacing = (float(x) for x in datasets[0].PixelSpacing)  # (row spacing, col spacing)

    origin = np.array(datasets[0].ImagePositionPatient, dtype=np.float64)
    if len(datasets) >= 2 and getattr(datasets[1], "ImagePositionPatient", None) is not None:
        d_slice = np.array(datasets[1].ImagePositionPatient, dtype=np.float64) - origin
    else:
        normal = np.cross(d_col, d_row)
        thickness = float(getattr(datasets[0], "SpacingBetweenSlices", None)
                           or getattr(datasets[0], "SliceThickness", 1.0))
        d_slice = normal * thickness

    geometry = VolumeGeometry(
        volume=volume,
        origin=origin,
        d_slice=d_slice,
        d_row=d_row * row_spacing,
        d_col=d_col * col_spacing,
        series=series,
    )

    # Sanity check: the affine must reproduce the second slice's own recorded
    # position (when present) to within sub-millimetre error. Catches a wrong
    # index order or a sign flip immediately instead of silently mirroring
    # the wrong axis later.
    if len(datasets) >= 2 and getattr(datasets[1], "ImagePositionPatient", None) is not None:
        predicted = geometry.voxel_to_patient(1, 0, 0)
        actual = np.array(datasets[1].ImagePositionPatient, dtype=np.float64)
        error = float(np.linalg.norm(predicted - actual))
        if error > 0.5:
            raise ValueError(
                f"VolumeGeometry affine disagrees with slice 1's ImagePositionPatient by "
                f"{error:.3f} mm (expected < 0.5 mm) — geometry construction is wrong.")

    return geometry
