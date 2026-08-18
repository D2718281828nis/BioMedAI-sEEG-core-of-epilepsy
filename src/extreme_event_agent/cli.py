import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .agent import ExtremeEventAgent
from .edf_workflow import clock_time_to_offset, read_edf_start, run_edf
from .models import AgentConfig, ClinicalEvent


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
    args = parser.parse_args()
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
        recording_output.mkdir(parents=True, exist_ok=True)
        report, process, plot, evolution_plot, annotated, detected = run_edf(path, recording_output, event)
        payload = {"source": str(path), "detection": asdict(report),
                   "clinical_annotation": asdict(event) if event else None,
                   "annotated_event": asdict(annotated) if annotated else None,
                   "detected_event": asdict(detected) if detected else None,
                   "brain_process": asdict(process) if process else None,
                   "figure": str(plot),
                   "evolution_figure": str(evolution_plot) if evolution_plot else None}
        result = recording_output / "analysis.json"
        result.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
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
        figures_note = f"wrote {result} and {plot}"
        if evolution_plot:
            figures_note += f" and {evolution_plot}"
        print(f"{path}: {len(report.events)} candidate(s); {context_note}; {figures_note}")


if __name__ == "__main__":
    main()
