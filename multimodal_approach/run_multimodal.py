"""Run the structural-anomaly x extreme-event cross-check end to end.

Run as a module from the repo root (matches ``model/run_model.py``'s own
convention, and for the same reason — relative imports inside the package):

    python -m multimodal_approach.run_multimodal \\
        --dicom-dir dataset/MRI-with-electrodes/DCM \\
        --agent-output seeg_agent_output/sEEG-HFOs-8 \\
        --output multimodal_result

Writes, to ``--output``:

* ``structural_anomaly.npz`` — the combined T1/T2-agreement-gated asymmetry
  and heterogeneity z-score volumes plus the head/artifact masks, on the T1
  grid;
* ``top_anomaly_clusters.json`` — up to five connected clusters from the
  asymmetry channel (``find_top_anomaly_clusters``), ranked by total mass,
  each with its peak voxel, size, and patient-space location;
* ``top_heterogeneity_clusters.json`` — the same, for the independent local
  texture-heterogeneity channel (catches bilateral/midline findings the
  asymmetry channel is blind to by construction — see
  ``structural_anomaly.py``'s module docstring);
* ``structural_anomaly_overview.png`` — axial/coronal/sagittal all sliced
  through the single strongest asymmetry cluster's peak voxel at once
  (crosshair-marked in every view), not a generic geometric mid-slice;
* ``structural_anomaly_t1t2_fusion.png`` — a second figure, cropped tight
  around each channel's own top cluster, with a **T1/T2 color-fused**
  anatomical background (T1 on the red channel, T2 on green+blue, so
  pure-T1-bright tissue reads warm/orange, pure-T2-bright tissue reads cyan,
  and tissue bright on both reads near-white) instead of T1 grayscale alone
  — lets a reviewer see the actual T1 vs. T2 appearance of a flagged region
  at a glance, not just its abstract z-score;
* ``hemisphere_summary.json`` — per-hemisphere mean/max |anomaly z|;
* ``heterogeneity_summary.json`` — whole-head (not per-hemisphere, by
  design) mean/max |heterogeneity z|;
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


def _fused_rgb_slice(t1_2d: np.ndarray, t2_2d: np.ndarray,
                      t1_vmin: float, t1_vmax: float, t2_vmin: float, t2_vmax: float) -> np.ndarray:
    """T1 on red, T2 on green+blue: a two-contrast color fusion, not two separate grayscales.

    Percentile windows (``t1_vmin``/``t1_vmax``/...) are computed once over
    the *whole* volume by the caller, not per-slice, so brightness stays
    comparable across the panels in one figure. T1-bright/T2-dark tissue
    (e.g. fat, subacute blood) reads warm/orange; T2-bright/T1-dark tissue
    (e.g. CSF, edema, most simple fluid) reads cyan; bright on both reads
    near-white; dark on both stays black — the same convention radiologists
    use informally when flipping between T1 and T2 to characterize a lesion,
    made simultaneous instead of sequential.
    """
    t1_norm = np.clip((t1_2d - t1_vmin) / (t1_vmax - t1_vmin + 1e-6), 0.0, 1.0)
    t2_norm = np.clip((t2_2d - t2_vmin) / (t2_vmax - t2_vmin + 1e-6), 0.0, 1.0)
    rgb = np.stack([t1_norm, t2_norm, t2_norm], axis=-1)
    return rgb


def _plot_t1t2_fusion(result, asymmetry_clusters: list[dict], heterogeneity_clusters: list[dict],
                       output_path: Path, crop_mm: float = 45.0) -> None:
    """T1/T2 color-fused anatomy (see ``_fused_rgb_slice``), cropped around each channel's top cluster.

    Two rows, one per evidence channel — asymmetry (``combined_anomaly``) on
    top, local heterogeneity (``combined_heterogeneity``) on the bottom —
    each sliced through *that channel's own* strongest cluster (they need
    not be the same location; that is expected, since the two channels are
    looking for different things, see ``structural_anomaly.py``'s module
    docstring). Cropped to a ``crop_mm``-radius box around the peak voxel
    (``structural_anomaly_overview.png`` shows the whole head; this trades
    that context for enough zoom to actually judge the fused T1/T2 tissue
    appearance by eye) rather than full-slice, matching how a radiologist
    would zoom in on a candidate region rather than read it off a whole-head
    thumbnail.
    """
    t1 = result.t1_geometry.volume
    t2 = result.t2_on_t1
    t1_vmin, t1_vmax = np.percentile(t1, [1, 99])
    t2_vmin, t2_vmax = np.percentile(t2, [1, 99])
    voxel_mm = float(np.linalg.norm(result.t1_geometry.d_row))
    half_span = max(4, int(round(crop_mm / voxel_mm)))

    def _crop(gray2d, overlay2d, cx, cy):
        x0, x1 = max(0, cx - half_span), cx + half_span
        y0, y1 = max(0, cy - half_span), cy + half_span
        return gray2d[y0:y1, x0:x1], overlay2d[y0:y1, x0:x1], cx - x0, cy - y0

    rows = [
        ("Asymmetry channel (combined_anomaly)", result.combined_anomaly, asymmetry_clusters,
         (-8, 8), "coolwarm"),
        ("Heterogeneity channel (combined_heterogeneity)", result.combined_heterogeneity, heterogeneity_clusters,
         (0, 8), "inferno"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    for row_axes, (row_title, overlay_vol, clusters, (ovmin, ovmax), cmap) in zip(axes, rows):
        if not clusters:
            for ax in row_axes:
                ax.axis("off")
            row_axes[1].text(0.5, 0.5, f"{row_title}: no cluster reached threshold",
                              ha="center", va="center", transform=row_axes[1].transAxes)
            continue
        top = clusters[0]
        k0, i0, j0 = top["peak_voxel_kij"]
        panels = [
            ("Axial", t1[k0, :, :], t2[k0, :, :], overlay_vol[k0, :, :], j0, i0),
            ("Coronal", t1[:, i0, :], t2[:, i0, :], overlay_vol[:, i0, :], j0, k0),
            ("Sagittal", t1[:, :, j0], t2[:, :, j0], overlay_vol[:, :, j0], i0, k0),
        ]
        for ax, (title, t1_2d, t2_2d, overlay2d, cx, cy) in zip(row_axes, panels):
            t1_crop, overlay_crop, cross_x, cross_y = _crop(t1_2d, overlay2d, cx, cy)
            t2_crop, _, _, _ = _crop(t2_2d, overlay2d, cx, cy)
            fused = _fused_rgb_slice(t1_crop, t2_crop, t1_vmin, t1_vmax, t2_vmin, t2_vmax)
            ax.imshow(fused)
            masked = np.ma.masked_where(np.abs(overlay_crop) < 2.0, overlay_crop)
            im = ax.imshow(masked, cmap=cmap, vmin=ovmin, vmax=ovmax, alpha=0.75)
            ax.plot(cross_x, cross_y, marker="+", markersize=16, markeredgecolor="lime", markeredgewidth=2)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        row_axes[0].set_ylabel(row_title, fontsize=9)
        fig.colorbar(im, ax=row_axes, shrink=0.75, label="z-score (|z| < 2 hidden)")
    fig.suptitle(
        "T1/T2 fused structural view — T1=red, T2=green+blue (white = bright on both, "
        f"cyan = T2-only, orange = T1-only); each row through its own channel's top cluster, "
        f"±{crop_mm:.0f} mm crop"
    )
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
        combined_heterogeneity=result.combined_heterogeneity,
        t1_heterogeneity_z=result.t1_heterogeneity_z,
        t2_heterogeneity_z=result.t2_heterogeneity_z,
        brain_mask=result.brain_mask,
        csf_mask=result.csf_mask,
        artifact_mask=result.artifact_mask,
    )
    clusters = find_top_anomaly_clusters(result)
    (args.output / "top_anomaly_clusters.json").write_text(json.dumps(clusters, indent=2))
    # Heterogeneity z-scores run on a different scale than asymmetry z-scores
    # (different underlying quantity — see structural_anomaly.py), so this
    # reuses the same threshold/min_voxels defaults only as a starting point,
    # not because the two are known to be comparable; re-check on other data.
    heterogeneity_clusters = find_top_anomaly_clusters(result, anomaly_map=result.combined_heterogeneity)
    (args.output / "top_heterogeneity_clusters.json").write_text(json.dumps(heterogeneity_clusters, indent=2))
    _plot_overview(result, clusters, args.output / "structural_anomaly_overview.png")
    _plot_t1t2_fusion(result, clusters, heterogeneity_clusters,
                       args.output / "structural_anomaly_t1t2_fusion.png")
    (args.output / "hemisphere_summary.json").write_text(json.dumps(result.hemisphere_summary, indent=2))
    (args.output / "heterogeneity_summary.json").write_text(json.dumps(result.heterogeneity_summary, indent=2))
    print(f"[run_multimodal] hemisphere summary: {json.dumps(result.hemisphere_summary, indent=2)}")
    print(f"[run_multimodal] heterogeneity summary: {json.dumps(result.heterogeneity_summary, indent=2)}")
    if clusters:
        print(f"[run_multimodal] top anomaly cluster: {json.dumps(clusters[0], indent=2)}")
    else:
        print("[run_multimodal] no anomaly cluster reached the reporting threshold")
    if heterogeneity_clusters:
        print(f"[run_multimodal] top heterogeneity cluster: {json.dumps(heterogeneity_clusters[0], indent=2)}")
    else:
        print("[run_multimodal] no heterogeneity cluster reached the reporting threshold")

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
