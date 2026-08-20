import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .agent import ExtremeEventAgent
from .edf_workflow import clock_time_to_offset, compare_montages, read_edf_start, summarize_montage_comparison
from .models import AgentConfig, ClinicalEvent, EdfRunResult


def _write_analysis_json(recording_output: Path, event: ClinicalEvent | None,
                         result_obj: EdfRunResult) -> Path:
    report, process = result_obj.report, result_obj.process
    annotated, detected = result_obj.annotated_event, result_obj.detected_event
    payload = {"montage_reference": result_obj.montage_reference,
               "detection": asdict(report),
               "montage": result_obj.montage, "montage_file": str(result_obj.montage_file),
               "clinical_annotation": asdict(event) if event else None,
               "annotated_event": asdict(annotated) if annotated else None,
               "detected_event": asdict(detected) if detected else None,
               "brain_process": asdict(process) if process else None,
               "figure": str(result_obj.overview_figure),
               "evolution_figure": str(result_obj.evolution_figure) if result_obj.evolution_figure else None,
               "graph_figures": {layout: str(figure) for layout, figure in result_obj.graph_figures.items()},
               "graph_graphml": str(result_obj.graph_graphml) if result_obj.graph_graphml else None,
               "message_passing_figure": (str(result_obj.message_passing_figure)
                                          if result_obj.message_passing_figure else None),
               "message_passing_validation_figure": (
                   str(result_obj.message_passing_validation_figure)
                   if result_obj.message_passing_validation_figure else None),
               "message_passing_evaluation": result_obj.message_passing_evaluation}
    result_file = recording_output / "analysis.json"
    result_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auditably analyse a NumPy recording, one EDF, or every EDF in a directory.")
    parser.add_argument("input", help="[channels, samples] .npy, .edf, or directory")
    parser.add_argument("--sfreq", type=float, help="Required only for NumPy input")
    parser.add_argument("--channels", help="Optional NumPy channel-name text file")
    parser.add_argument("--output", default="seeg_agent_output", help="JSON file for NumPy or EDF output directory")
    parser.add_argument("--event-time", type=float, help="Expert event time in seconds from EDF start")
    parser.add_argument("--event-clock", help="Expert event wall-clock time, HH:MM:SS[.ffffff]")
    parser.add_argument("--event-label", default="асимметричный тонический приступ")
    parser.add_argument("--event-duration", type=float, default=2.)
    parser.add_argument("--montages", default="none,bipolar",
                        help="Comma-separated signal references to analyse and compare: "
                             "'none' (recording's native reference), 'bipolar' "
                             "(adjacent-contact re-referencing), or both (default).")
    parser.add_argument("--crop-end-seconds", type=float, default=None,
                        help="Discard everything in each EDF after this many seconds from the "
                             "start before any analysis (blind detection, whole-recording "
                             "figure, event context) sees it. Use this to exclude a known "
                             "non-physiological tail, e.g. an equipment test or surgical "
                             "procedure appended after the recording of interest. Ignored for "
                             ".npy input. Default: analyse the complete file.")
    args = parser.parse_args()
    montage_references = tuple(reference.strip() for reference in args.montages.split(",") if reference.strip())
    for reference in montage_references:
        if reference not in ("none", "bipolar"):
            parser.error(f"--montages entries must be 'none' or 'bipolar', got {reference!r}")
    if args.event_time is not None and args.event_clock is not None:
        parser.error("use either --event-time or --event-clock, not both")
    source = Path(args.input)
    if source.suffix.lower() == ".npy":
        if args.sfreq is None:
            parser.error("--sfreq is required for NumPy input")
        names = None
        if args.channels:
            names = [line.strip() for line in Path(args.channels).read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        report = ExtremeEventAgent(AgentConfig()).run(np.load(source), args.sfreq, names)
        output = Path(args.output)
        if output.suffix.lower() != ".json":
            output = output / "extreme_events.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(report.events)} event(s) to {output}")
        return

    paths = (sorted(path for path in source.rglob("*") if path.suffix.lower() == ".edf")
             if source.is_dir() else [source])
    if not paths or any(path.suffix.lower() != ".edf" for path in paths):
        parser.error("input must be a .npy, .edf, or a directory containing EDF files")
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        event_time = args.event_time
        if args.event_clock:
            start, duration = read_edf_start(path)
            event_time = clock_time_to_offset(args.event_clock, start, duration)
        event = (ClinicalEvent(event_time, args.event_label, args.event_duration)
                 if event_time is not None else None)
        recording_output = output_dir / path.stem
        results = compare_montages(path, output_dir, event, montage_references=montage_references,
                                   crop_end_seconds=args.crop_end_seconds)

        for reference, result_obj in results.items():
            recording_montage_output = recording_output / reference
            result_file = _write_analysis_json(recording_montage_output, event, result_obj)
            annotated, detected = result_obj.annotated_event, result_obj.detected_event
            if event:
                context_note = f"clinical event at {event.time_seconds:.3f}s"
            elif annotated:
                context_note = (f"EDF-annotated event at {annotated.time_seconds:.3f}s "
                                f"({annotated.label!r})")
            elif detected:
                context_note = (f"auto-detected event at {detected.time_seconds:.3f}s "
                                f"({detected.involved_channel_count} channels, "
                                f"score={detected.score:.2f})")
            else:
                context_note = "no event context"
            figures_note = f"wrote {result_file} and {result_obj.overview_figure}"
            if result_obj.evolution_figure:
                figures_note += f" and {result_obj.evolution_figure}"
            if result_obj.graph_figures:
                figures_note += f" and {len(result_obj.graph_figures)} graph layout(s)"
            if result_obj.message_passing_figure:
                figures_note += " and message-passing figures"
            print(f"{path} [{reference}]: {len(result_obj.report.events)} candidate(s); "
                 f"{context_note}; {figures_note}")

        if len(results) > 1:
            comparison_rows = summarize_montage_comparison(results)
            comparison_file = recording_output / "montage_comparison.json"
            comparison_file.write_text(json.dumps(comparison_rows, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
            print(f"{path}: montage comparison —")
            for row in comparison_rows:
                print(f"  {row['montage_reference']}: {row['n_detected_candidates']} candidate(s), "
                     f"{row['n_involved_channels']} involved channel(s), "
                     f"{row['co_activation_edges']} co-activation edge(s), "
                     f"message-passing best correlation="
                     f"{row['message_passing_best_correlation']}")
            print(f"{path}: wrote {comparison_file}")


if __name__ == "__main__":
    main()
