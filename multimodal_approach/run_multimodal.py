"""Run the structural-anomaly x extreme-event cross-check end to end.

Run as a module from the repo root (matches ``model/run_model.py``'s own
convention, and for the same reason — relative imports inside the package):

    python -m multimodal_approach.run_multimodal \\
        --dicom-dir dataset/MRI-with-electrodes/DCM \\
        --agent-output seeg_agent_output/sEEG-HFOs-8 \\
        --output multimodal_result

Writes, to ``--output``:

* ``structural_anomaly.npz`` — the combined T1/T2-agreement-gated anomaly
  z-score volume plus the head/artifact masks, on the T1 grid;
* ``top_anomaly_clusters.json`` — up to five connected anomaly clusters
  (``find_top_anomaly_clusters``), ranked by total mass, each with its peak
  voxel, size, and patient-space location;
* ``structural_anomaly_overview.png`` — axial/coronal/sagittal all sliced
  through the single strongest cluster's peak voxel at once (crosshair-marked
  in every view), not a generic geometric mid-slice;
* ``hemisphere_summary.json`` — per-hemisphere mean/max |anomaly z|;
* ``structural_prior_report.json`` — one entry per montage reference found
  under ``--agent-output`` (``none``/``bipolar``), each with every tier-3
  blind candidate annotated with its hemisphere balance and structural
  alignment score, the temporal-only pick, the structural-only pick, and
  whether they agree; plus, when present, where the file's own EDF+
  annotation (tier 2 — the actual known event) falls, reported the same way
  the rest of this repo reports its blind-vs-annotated comparisons: after
  the fact, for review, never fed back into any threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .extreme_event_prior import apply_structural_prior
from .structural_anomaly import find_top_anomaly_clusters, run_structural_anomaly


def _plot_overview(result, clusters: list[dict], output_path: Path) -> None:
    """Axial/coronal/sagittal, all three sliced through the *same* point at once.

    Geometric mid-slices (the previous version of this figure) have no
    reason to pass through wherever the anomaly map is actually strongest —
    on this dataset they mostly missed it. Instead this slices through the
    peak voxel of ``clusters[0]`` (the highest-total-mass connected anomaly
    cluster from ``find_top_anomaly_clusters`` — see its docstring for why
    total mass, not a bare peak voxel, ranks clusters), with a crosshair
    marking that exact point in every view so the same location can be read
    across all three projections together, plus a caption naming its
    hemisphere, size, and patient-space position.
    """
    volume = result.t1_geometry.volume
    anomaly = result.combined_anomaly
    nk, ni, nj = volume.shape

    if clusters:
        top = clusters[0]
        k0, i0, j0 = top["peak_voxel_kij"]
        x_mm, y_mm, z_mm = top["peak_patient_xyz_mm"]
        location_note = (
            f"Best view: strongest anomaly cluster ({top['voxel_count']} voxels, "
            f"peak z={top['peak_value']:+.2f}, mean |z|={top['mean_abs_anomaly']:.2f}) — "
            f"{top['hemisphere']} hemisphere, patient xyz=({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm"
        )
        if len(clusters) > 1:
            location_note += f"  [{len(clusters) - 1} more cluster(s) not shown — see top_anomaly_clusters.json]"
    else:
        k0, i0, j0 = nk // 2, ni // 2, nj // 2
        location_note = "No anomaly cluster reached the reporting threshold — showing geometric mid-slices instead."

    # (title, in-plane image, in-plane overlay, crosshair x, crosshair y) — x/y are
    # in each panel's own displayed (column, row) axes, all three passing through (k0, i0, j0).
    slices = [
        ("Axial", volume[k0, :, :], anomaly[k0, :, :], j0, i0),
        ("Coronal", volume[:, i0, :], anomaly[:, i0, :], j0, k0),
        ("Sagittal", volume[:, :, j0], anomaly[:, :, j0], i0, k0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    vmin, vmax = np.percentile(volume, [1, 99])
    im = None
    for ax, (title, gray, overlay, cross_x, cross_y) in zip(axes, slices):
        ax.imshow(gray, cmap="gray", vmin=vmin, vmax=vmax)
        masked = np.ma.masked_where(np.abs(overlay) < 2.0, overlay)
        im = ax.imshow(masked, cmap="coolwarm", vmin=-8, vmax=8, alpha=0.7)
        ax.axvline(cross_x, color="yellow", linewidth=0.7, alpha=0.8)
        ax.axhline(cross_y, color="yellow", linewidth=0.7, alpha=0.8)
        if clusters:
            ax.plot(cross_x, cross_y, marker="o", markersize=14, markerfacecolor="none",
                     markeredgecolor="lime", markeredgewidth=2)
        ax.set_title(title)
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, label="T1/T2-agreeing asymmetry z-score (|z| < 2 hidden)")
    fig.suptitle(
        "Structural anomaly — best-view slices, all three projections through one point "
        f"(midline x={result.midline_x_mm:.1f} mm, mirror self-correlation r={result.midline_mirror_correlation:.2f})"
    )
    fig.text(0.5, 0.03, location_note, ha="center", fontsize=9.5,
              bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9))
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _load_events(analysis_json: Path) -> list[dict]:
    data = json.loads(analysis_json.read_text())
    return data.get("detection", {}).get("events", [])


def _load_annotated_event(analysis_json: Path) -> dict | None:
    data = json.loads(analysis_json.read_text())
    return data.get("annotated_event")


def _report_for_montage(montage_dir: Path, hemisphere_summary: dict) -> dict | None:
    analysis_json = montage_dir / "analysis.json"
    if not analysis_json.exists():
        return None
    events = _load_events(analysis_json)
    report = apply_structural_prior(events, hemisphere_summary)

    annotated = _load_annotated_event(analysis_json)
    annotated_summary = None
    if annotated is not None:
        # Reported for review only — the annotated (tier-2) event is never
        # used to pick a threshold or a candidate anywhere in this module.
        matching = [
            candidate for candidate in report.candidates
            if candidate["start_seconds"] <= annotated["time_seconds"] <= candidate["end_seconds"]
        ]
        annotated_summary = {
            "annotated_time_seconds": annotated["time_seconds"],
            "annotated_label": annotated["label"],
            "overlapping_blind_candidates": len(matching),
            "temporal_pick_overlaps_annotation": (
                report.temporal_pick_index is not None
                and report.candidates[report.temporal_pick_index] in matching
            ),
            "structural_pick_overlaps_annotation": (
                report.structural_pick_index is not None
                and report.candidates[report.structural_pick_index] in matching
            ),
        }

    return {
        "montage_reference": montage_dir.name,
        "candidate_count": len(report.candidates),
        "temporal_pick_index": report.temporal_pick_index,
        "structural_pick_index": report.structural_pick_index,
        "temporal_and_structural_agree": report.agree,
        "candidates": report.candidates,
        "annotated_event_check": annotated_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dicom-dir", type=Path, default=Path("dataset/MRI-with-electrodes/DCM"))
    parser.add_argument("--agent-output", type=Path, default=Path("seeg_agent_output/sEEG-HFOs-8"),
                         help="Directory holding montage subdirectories (none/, bipolar/) from seeg-event-agent.")
    parser.add_argument("--output", type=Path, default=Path("multimodal_result"))
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    result = run_structural_anomaly(args.dicom_dir)

    np.savez_compressed(
        args.output / "structural_anomaly.npz",
        combined_anomaly=result.combined_anomaly,
        t1_anomaly_z=result.t1_anomaly_z,
        t2_anomaly_z=result.t2_anomaly_z,
        head_mask=result.head_mask,
        artifact_mask=result.artifact_mask,
    )
    clusters = find_top_anomaly_clusters(result)
    (args.output / "top_anomaly_clusters.json").write_text(json.dumps(clusters, indent=2))
    _plot_overview(result, clusters, args.output / "structural_anomaly_overview.png")
    (args.output / "hemisphere_summary.json").write_text(json.dumps(result.hemisphere_summary, indent=2))
    print(f"[run_multimodal] hemisphere summary: {json.dumps(result.hemisphere_summary, indent=2)}")
    if clusters:
        print(f"[run_multimodal] top anomaly cluster: {json.dumps(clusters[0], indent=2)}")
    else:
        print("[run_multimodal] no anomaly cluster reached the reporting threshold")

    montage_reports = []
    if args.agent_output.exists():
        for montage_dir in sorted(p for p in args.agent_output.iterdir() if p.is_dir()):
            report = _report_for_montage(montage_dir, result.hemisphere_summary)
            if report is not None:
                montage_reports.append(report)
    else:
        print(f"[run_multimodal] {args.agent_output} not found — skipping extreme-event prior "
              f"(run seeg-event-agent first, or point --agent-output at its output directory).")

    (args.output / "structural_prior_report.json").write_text(json.dumps(montage_reports, indent=2))
    for report in montage_reports:
        print(f"[run_multimodal] montage={report['montage_reference']} "
              f"candidates={report['candidate_count']} "
              f"temporal_pick={report['temporal_pick_index']} "
              f"structural_pick={report['structural_pick_index']} "
              f"agree={report['temporal_and_structural_agree']}")
        if report["annotated_event_check"] is not None:
            print(f"  annotated_event_check={report['annotated_event_check']}")

    print(f"[run_multimodal] wrote outputs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
