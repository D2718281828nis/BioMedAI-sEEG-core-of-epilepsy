import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .agent import ExtremeEventAgent
from .edf_workflow import run_edf
from .models import AgentConfig, ClinicalEvent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auditably analyse a NumPy recording, one EDF, or every EDF in a directory.")
    parser.add_argument("input", help="[channels, samples] .npy, .edf, or directory")
    parser.add_argument("--sfreq", type=float, help="Required only for NumPy input")
    parser.add_argument("--channels", help="Optional NumPy channel-name text file")
    parser.add_argument("--output", default="seeg_agent_output", help="JSON file for NumPy or EDF output directory")
    parser.add_argument("--event-time", type=float, help="Expert event time in seconds from EDF start")
    parser.add_argument("--event-label", default="асимметричный тонический приступ")
    parser.add_argument("--event-duration", type=float, default=2.)
    args = parser.parse_args()
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

    paths = sorted(source.rglob("*.edf")) if source.is_dir() else [source]
    if not paths or any(path.suffix.lower() != ".edf" for path in paths):
        parser.error("input must be a .npy, .edf, or a directory containing EDF files")
    event = (ClinicalEvent(args.event_time, args.event_label, args.event_duration)
             if args.event_time is not None else None)
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        report, process, plot = run_edf(path, output_dir, event)
        payload = {"source": str(path), "detection": asdict(report),
                   "clinical_annotation": asdict(event) if event else None,
                   "brain_process": asdict(process) if process else None,
                   "figure": str(plot)}
        result = output_dir / f"{path.stem}_analysis.json"
        result.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{path}: {len(report.events)} candidate(s); wrote {result} and {plot}")


if __name__ == "__main__":
    main()
