"""End-to-end CLI: detect shaft candidates, register them against the documented montage, write metrics + a figure.

Run as a module (relative imports inside the package), from the repo root:

    python -m electrode_registration.run_electrode_registration \\
        --dicom-dir dataset/MRI-with-electrodes/DCM --edf dataset/sEEG-HFOs-8.edf \\
        --output electrode_registration_result

This package exists to answer, as honestly as this specific dataset allows
(no CT, no per-contact fiducial file — see ``README.md``, "Honest limits"):
does the MRI's own signal-void geometry plausibly line up with what the EDF
montage and the clinician's own shaft-length note say should be there? Every
number this writes is an internal-consistency check between independent,
approximate sources, never a validated 3-D localization accuracy claim.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from extreme_event_agent.edf_workflow import EDF_ENCODING
from multimodal_approach.structural_anomaly import run_structural_anomaly

from .contact_detection import detect_shaft_candidates
from .frame_reference import check_frame_of_reference
from .metrics import compare_montage_to_reference, compute_registration_metrics
from .registration import assign_contact_positions, extract_contact_roi, register_shaft_candidates

_HEMISPHERE_COLORS = {"right": "steelblue", "left": "seagreen"}


def _read_edf_channel_names(edf_path: str | Path) -> list[str]:
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR", encoding=EDF_ENCODING)
    return list(raw.ch_names)


def _plot_registration_overview(result, candidates, registrations, positions, output_path: Path) -> None:
    """Axial/coronal/sagittal slices through the assigned positions' own centroid (or the volume centre).

    All detected candidates are drawn as faint grey line segments —
    intentionally cluttered, since honestly showing how many of them there
    are (70-98, against 12 documented shafts) is the point, not something
    to crop out of the picture. Assigned contact positions are drawn on top,
    coloured by hemisphere, solid for a length-plausible match and hollow
    (open marker) for a matched-but-implausible one; unmatched shafts have
    no positions to draw at all, which is itself the coverage gap
    ``metrics.py`` quantifies.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = result.t1_geometry
    volume = geometry.volume
    vmin, vmax = np.percentile(volume, [1, 99])

    plausible_shaft_names = {
        r.shaft_name for r in registrations
        if r.matched and r.length_relative_error is not None and r.length_relative_error <= 0.5
    }

    if positions:
        center_xyz = np.mean([p.patient_xyz_mm for p in positions], axis=0)
    else:
        center_xyz = np.array(geometry.voxel_to_patient(volume.shape[0] / 2, volume.shape[1] / 2,
                                                         volume.shape[2] / 2))
    k0, i0, j0 = (int(round(c)) for c in geometry.patient_to_voxel(center_xyz))
    k0 = int(np.clip(k0, 0, volume.shape[0] - 1))
    i0 = int(np.clip(i0, 0, volume.shape[1] - 1))
    j0 = int(np.clip(j0, 0, volume.shape[2] - 1))

    panels = [
        ("Axial", volume[k0, :, :], 2, 1),
        ("Coronal", volume[:, i0, :], 2, 0),
        ("Sagittal", volume[:, :, j0], 1, 0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    for ax, (title, background, h_axis, v_axis) in zip(axes, panels):
        ax.imshow(background, cmap="gray", vmin=vmin, vmax=vmax)
        for candidate in candidates:
            a_kij = geometry.patient_to_voxel(np.array(candidate.endpoint_a_patient_xyz_mm))
            b_kij = geometry.patient_to_voxel(np.array(candidate.endpoint_b_patient_xyz_mm))
            ax.plot([a_kij[h_axis], b_kij[h_axis]], [a_kij[v_axis], b_kij[v_axis]],
                    color="0.6", linewidth=0.4, alpha=0.35, zorder=1)
        for position in positions:
            kij = geometry.patient_to_voxel(np.array(position.patient_xyz_mm))
            color = _HEMISPHERE_COLORS.get(position.hemisphere, "grey")
            plausible = position.shaft_name in plausible_shaft_names
            ax.scatter(kij[h_axis], kij[v_axis], s=22, c=color if plausible else "none",
                      edgecolors=color, linewidths=1.4, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor=color,
              markersize=8, label=f"{hemi} — plausible match")
        for hemi, color in _HEMISPHERE_COLORS.items()
    ] + [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="0.4",
              markersize=8, label="matched, not length-plausible"),
        Line2D([0], [0], color="0.6", alpha=0.6, label="all detected candidates (unfiltered)"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("Electrode-to-DICOM candidate registration — best-guess assignment, not verified localization")
    fig.text(0.5, 0.02,
             "Grey lines = every detected artifact-mask candidate (70-98 typical, vs. 12 documented shafts). "
             "Dots = assigned contact positions for matched shafts only. See README.md, \"Honest limits\".",
             ha="center", fontsize=8.5,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9))
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run(dicom_dir: str | Path, edf_path: str | Path, output_dir: str | Path, max_link_mm: float = 6.0,
       plausible_relative_error: float = 0.5, roi_radius_mm: float = 15.0) -> dict[str, object]:
    """Run the full candidate-detection -> registration -> metrics pipeline once; returns/writes the JSON summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_check = check_frame_of_reference(dicom_dir)
    result = run_structural_anomaly(dicom_dir)
    candidates = detect_shaft_candidates(result, max_link_mm=max_link_mm)
    registrations = register_shaft_candidates(candidates)
    positions = assign_contact_positions(registrations)

    channel_names = _read_edf_channel_names(edf_path)
    montage_comparison = compare_montage_to_reference(channel_names)
    metrics = compute_registration_metrics(registrations, positions, channel_names, result,
                                           plausible_relative_error=plausible_relative_error)

    roi_examples = [
        {"contact_label": position.contact_label, **{
            key: value for key, value in extract_contact_roi(result, position.patient_xyz_mm,
                                                              radius_mm=roi_radius_mm).items()
            if key != "t1_roi" and key != "artifact_mask_roi"}}
        for position in positions[:5]
    ]

    figure_path = output_dir / "electrode_registration_overview.png"
    _plot_registration_overview(result, candidates, registrations, positions, figure_path)

    payload = {
        "dicom_dir": str(dicom_dir),
        "edf": str(edf_path),
        "frame_of_reference": asdict(frame_check),
        "montage_vs_pdf_reference": montage_comparison,
        "candidate_count": len(candidates),
        "candidates": [asdict(c) for c in candidates],
        "registrations": [
            {**{k: v for k, v in asdict(r).items() if k not in ("candidate", "reference")},
             "candidate_id": r.candidate.candidate_id if r.candidate else None,
             "reference": asdict(r.reference)}
            for r in registrations
        ],
        "contact_position_count": len(positions),
        "roi_examples_mm_radius": roi_radius_mm,
        "roi_examples": roi_examples,
        "metrics": asdict(metrics),
        "figure": str(figure_path),
    }
    (output_dir / "electrode_registration_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dicom-dir", default="dataset/MRI-with-electrodes/DCM")
    parser.add_argument("--edf", default="dataset/sEEG-HFOs-8.edf")
    parser.add_argument("--output", default="electrode_registration_result")
    parser.add_argument("--max-link-mm", type=float, default=6.0)
    parser.add_argument("--plausible-relative-error", type=float, default=0.5)
    args = parser.parse_args()

    payload = run(args.dicom_dir, args.edf, args.output, max_link_mm=args.max_link_mm,
                 plausible_relative_error=args.plausible_relative_error)
    metrics = payload["metrics"]
    print(f"[electrode_registration] FrameOfReferenceUID all match: "
          f"{payload['frame_of_reference']['all_match']}")
    print(f"[electrode_registration] candidates detected: {payload['candidate_count']} "
          f"(vs. {metrics['total_shafts']} documented shafts)")
    print(f"[electrode_registration] recall={metrics['recall']:.2f} precision={metrics['precision']:.2f} "
          f"coverage={metrics['coverage']:.2f}")
    print(f"[electrode_registration] contacts outside volume: {len(metrics['contacts_outside_volume'])}")
    print(f"\nWrote electrode registration outputs to {args.output}")


if __name__ == "__main__":
    main()
