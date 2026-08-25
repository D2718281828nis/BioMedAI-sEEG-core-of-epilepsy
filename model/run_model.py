"""Build, train, run, and evaluate the reservoir plant on one EDF recording.

Run as a module (needed for its relative imports), from the repo root:

    python -m model.run_model dataset/sEEG-HFOs-8.edf --output model_result

Resolves the event the same way ``extreme_event_agent`` does (explicit time
> EDF annotation > blind detection), windows the recording around it,
selects output channels from the same audited recruitment analysis the rest
of the repo uses, trains the reservoir plant's readout on the pre-event
baseline, runs it through the event, and writes every figure from
``model.visualize`` plus a JSON/text summary to the output directory.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from extreme_event_agent.models import AnnotatedEvent, ClinicalEvent, DetectedEvent

from .plant import build_window, describe_evaluation, resolve_event_context, run_reservoir_plant
from .reservoir import ReservoirConfig
from .visualize import plot_all


def _event_to_dict(event) -> dict[str, object]:
    if isinstance(event, AnnotatedEvent):
        tier = "edf_annotation"
    elif isinstance(event, DetectedEvent):
        tier = "blind_detection"
    elif isinstance(event, ClinicalEvent):
        tier = "explicit"
    else:
        tier = "unknown"
    payload = asdict(event)
    payload["tier"] = tier
    return payload


def run(input_path: str | Path, output_dir: str | Path, event_time: float | None = None,
       baseline_seconds: float = 60.0, analysis_seconds: float = 20.0,
       max_output_channels: int = 12, config: ReservoirConfig | None = None,
       channel_selection: str = "recruitment") -> dict[str, object]:
    """Run the full pipeline once; returns the JSON-serializable summary it also writes.

    ``channel_selection`` ("recruitment", the default, or "balanced") is
    forwarded to ``build_window`` — see its docstring and
    ``model.plant.CHANNEL_SELECTION_METHODS``'s module comment for what each
    means and why it matters for whether ``ReservoirWindow.arbitration_valid``
    ends up true.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    context = resolve_event_context(input_path, event_time=event_time)
    window = build_window(input_path, context, baseline_seconds=baseline_seconds,
                          analysis_seconds=analysis_seconds, max_output_channels=max_output_channels,
                          channel_selection=channel_selection)
    evaluation = run_reservoir_plant(window, config=config)
    figures = plot_all(evaluation, output_dir, stem)

    summary_text = describe_evaluation(evaluation)
    summary_file = output_dir / f"{stem}_model_summary.txt"
    summary_file.write_text(summary_text, encoding="utf-8")

    payload = {
        "input_edf": str(input_path),
        "event": _event_to_dict(context),
        "window": {
            "baseline_seconds": window.baseline_seconds,
            "analysis_seconds": window.analysis_seconds,
            "sfreq": window.sfreq,
            "input_names": window.input_names,
            "output_names": window.output_names,
            "channel_selection_method": window.channel_selection_method,
            "arbitration_valid": window.arbitration_valid,
        },
        "reservoir_config": asdict(evaluation.esn.config),
        "achieved_spectral_radius": evaluation.esn.achieved_spectral_radius,
        "training_rmse": evaluation.training_rmse,
        "extreme_event_evaluation": {
            "threshold_mad": evaluation.threshold,
            "peak_score_mad": evaluation.peak_score,
            "peak_time_seconds": evaluation.peak_time_seconds,
            "detected": evaluation.detected,
            "onset_time_seconds": evaluation.onset_time_seconds,
        },
        "per_channel_evaluation": {
            "onset_seconds": evaluation.per_channel_onset_seconds,
            "peak_score": evaluation.per_channel_peak_score,
            "peak_time_seconds": evaluation.per_channel_peak_time_seconds,
        },
        "figures": {name: str(path) for name, path in figures.items()},
        "summary": summary_text,
        "summary_file": str(summary_file),
    }
    (output_dir / f"{stem}_model_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", nargs="?", default="dataset/sEEG-HFOs-8.edf", help="EDF recording to model")
    parser.add_argument("--output", default="model_result", help="Output directory for figures/summary")
    parser.add_argument("--event-time", type=float, default=None,
                        help="Explicit event time in seconds from EDF start; default: resolve from "
                             "the EDF's own annotation, falling back to blind detection")
    parser.add_argument("--baseline-seconds", type=float, default=60.0)
    parser.add_argument("--analysis-seconds", type=float, default=20.0)
    parser.add_argument("--max-output-channels", type=int, default=12)
    parser.add_argument("--channel-selection", choices=("recruitment", "balanced"), default="recruitment",
                        help="'recruitment' (default) picks output channels from "
                             "analyse_brain_process's own likely_initiators/later_recruited -- "
                             "NOT independent of that analysis. 'balanced' splits channels evenly "
                             "by hemisphere and ranks by baseline-only variance, so the reservoir's "
                             "lateralization estimate (ReservoirWindow.arbitration_valid) is an "
                             "actual cross-check rather than circular with the EDF recruitment analysis.")
    parser.add_argument("--n-reservoir", type=int, default=400)
    parser.add_argument("--spectral-radius", type=float, default=0.95)
    parser.add_argument("--leak-rate", type=float, default=0.3)
    parser.add_argument("--ridge-alpha", type=float, default=1e-2,
                        help="Readout ridge-regression regularization strength")
    parser.add_argument("--output-feedback-lag", type=int, default=6,
                        help="How many past real output samples (NARX delay embedding) additionally "
                             "drive the reservoir alongside the exogenous MKR input; see model/plant.py")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = ReservoirConfig(n_reservoir=args.n_reservoir, spectral_radius=args.spectral_radius,
                             leak_rate=args.leak_rate, ridge_alpha=args.ridge_alpha,
                             output_feedback_lag=args.output_feedback_lag, seed=args.seed)
    payload = run(args.input, args.output, event_time=args.event_time,
                 baseline_seconds=args.baseline_seconds, analysis_seconds=args.analysis_seconds,
                 max_output_channels=args.max_output_channels, config=config,
                 channel_selection=args.channel_selection)
    print(payload["summary"])
    print(f"\nWrote {len(payload['figures'])} figure(s) and summary to {args.output}")


if __name__ == "__main__":
    main()
