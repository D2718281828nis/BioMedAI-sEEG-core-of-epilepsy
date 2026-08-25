"""Build and verify the object model for one EDF recording (+ optional DICOM).

Run as a module (relative imports inside the package), from the repo root:

    python -m object_model.run_object_model --edf dataset/sEEG-HFOs-8.edf \\
        --dicom-dir dataset/MRI-with-electrodes/DCM --crop-end-seconds 10550 \\
        --channel-selection balanced --output object_model_result

Ties together, in order: the EDF recruitment analysis and graph
(``extreme_event_agent``), the structural asymmetry map (``multimodal_approach``,
skipped with a printed notice if ``--dicom-dir`` is omitted or does not
exist), the reservoir plant (``model``), and the cross-modal verification
against the recording's own annotated event (``extreme_event_agent.verification``).
Writes, to ``<output>/<edf-stem>/``: ``verification_report.json``,
``object_model_graph.graphml`` (only when the EDF process found involved
channels), and ``object_model_summary.png`` (the five-panel figure).

This script *requires* the EDF to carry its own EDF+ annotation (tier 2 --
see ``edf_workflow.find_annotated_event``): verification needs a ground
truth to score against, and the blind tier-3 fallback has no such truth to
offer. Use ``extreme_event_agent``'s own CLI/API directly for a recording
with no annotation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from extreme_event_agent.agent import ExtremeEventAgent
from extreme_event_agent.edf_workflow import (SEEG_HFOS_8_CLINICAL_PRIOR, analyse_brain_process,
                                              build_seizure_graph, find_annotated_event, read_edf,
                                              select_seizure_event)
from extreme_event_agent.models import ContactPrior
from extreme_event_agent.verification import verify_against_annotation

from .graph import build_object_model_graph
from .figure import plot_object_model_summary


def run(edf_path: str | Path, output_dir: str | Path, dicom_dir: str | Path | None = None,
       crop_end_seconds: float | None = None, channel_selection: str = "recruitment",
       prior: ContactPrior | None = SEEG_HFOS_8_CLINICAL_PRIOR,
       reservoir_baseline_seconds: float = 60.0, reservoir_analysis_seconds: float = 20.0,
       max_output_channels: int = 12) -> dict[str, object]:
    """Run the full object-model pipeline once; returns the JSON-serializable summary it also writes."""
    edf_path = Path(edf_path)
    output_dir = Path(output_dir) / edf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = find_annotated_event(edf_path)
    if annotated is None:
        raise ValueError(f"{edf_path} carries no EDF+ annotated seizure marker -- verify_against_annotation "
                         "needs this recording's own ground truth (see this module's docstring).")

    data, sfreq, names = read_edf(edf_path, crop_end_seconds=crop_end_seconds)
    process = analyse_brain_process(data, sfreq, names, annotated, prior=prior)

    blind_report = ExtremeEventAgent().run(data, sfreq, names)
    blind_event = select_seizure_event(blind_report.events)

    graph = None
    if process.onset_latency_seconds:
        graph = build_seizure_graph(data, sfreq, names, annotated, process)

    structural_result = None
    if dicom_dir is not None and Path(dicom_dir).exists():
        from multimodal_approach.structural_anomaly import run_structural_anomaly
        structural_result = run_structural_anomaly(dicom_dir)
    elif dicom_dir is not None:
        print(f"[object_model] --dicom-dir {dicom_dir} does not exist -- skipping the structural block.")

    from model.plant import build_window, run_reservoir_plant
    window = build_window(edf_path, annotated, baseline_seconds=reservoir_baseline_seconds,
                          analysis_seconds=reservoir_analysis_seconds, max_output_channels=max_output_channels,
                          channel_selection=channel_selection)
    evaluation = run_reservoir_plant(window)

    if graph is not None:
        build_object_model_graph(
            graph, hemisphere_summary=structural_result.hemisphere_summary if structural_result else None,
            reservoir_evaluation=evaluation)

    report = verify_against_annotation(
        annotation=annotated, process=process,
        blind_event_time_seconds=blind_event.time_seconds if blind_event else None,
        hemisphere_summary=structural_result.hemisphere_summary if structural_result else None,
        reservoir_evaluation=evaluation,
        crop_applied=crop_end_seconds is not None, crop_end_seconds=crop_end_seconds,
        channel_selection=channel_selection,
        masking_method=structural_result.masking_method if structural_result else None)

    verification_file = output_dir / "verification_report.json"
    verification_file.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")

    graph_file = None
    if graph is not None:
        import networkx as nx
        graph_file = output_dir / "object_model_graph.graphml"
        nx.write_graphml(graph, graph_file)

    figure_file = plot_object_model_summary(
        data, sfreq, names, annotated, process, graph, structural_result, evaluation, report,
        output_dir / "object_model_summary.png")

    return {
        "edf": str(edf_path),
        "annotated_event": asdict(annotated),
        "verification_report": asdict(report),
        "verification_report_file": str(verification_file),
        "object_model_graph_file": str(graph_file) if graph_file else None,
        "object_model_summary_figure": str(figure_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edf", required=True, help="EDF recording to model (must carry an EDF+ annotation)")
    parser.add_argument("--dicom-dir", default=None, help="DICOM series directory; omit to skip the structural block")
    parser.add_argument("--crop-end-seconds", type=float, default=None,
                        help="Discard everything after this many seconds before any analysis sees it "
                             "(e.g. a known non-physiological tail)")
    parser.add_argument("--channel-selection", choices=("recruitment", "balanced"), default="recruitment",
                        help="Reservoir output-channel strategy -- see model.plant.CHANNEL_SELECTION_METHODS")
    parser.add_argument("--output", default="object_model_result")
    parser.add_argument("--max-output-channels", type=int, default=12)
    args = parser.parse_args()

    payload = run(args.edf, args.output, dicom_dir=args.dicom_dir, crop_end_seconds=args.crop_end_seconds,
                 channel_selection=args.channel_selection, max_output_channels=args.max_output_channels)
    print(json.dumps({k: v for k, v in payload.items() if k != "verification_report"}, indent=2, ensure_ascii=False))
    print(f"\nWrote object model outputs to {Path(args.output) / Path(args.edf).stem}")


if __name__ == "__main__":
    main()
