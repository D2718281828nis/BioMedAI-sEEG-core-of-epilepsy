import argparse
import json
from dataclasses import asdict

import numpy as np

from .agent import ExtremeEventAgent
from .models import AgentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Find extreme events in a [channels, samples] .npy array.")
    parser.add_argument("input", help="Path to a NumPy .npy file")
    parser.add_argument("--sfreq", type=float, required=True, help="Sampling frequency in Hz")
    parser.add_argument("--channels", help="Optional text file containing one channel name per line")
    parser.add_argument("--output", default="extreme_events.json")
    args = parser.parse_args()
    names = None
    if args.channels:
        with open(args.channels, encoding="utf-8") as stream:
            names = [line.strip() for line in stream if line.strip()]
    report = ExtremeEventAgent(AgentConfig()).run(np.load(args.input), args.sfreq, names)
    payload = asdict(report)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(f"Wrote {len(report.events)} event(s) to {args.output}")


if __name__ == "__main__":
    main()
