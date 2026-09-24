"""Reproduce the small synthetic sEEG recording shipped in this directory."""
from pathlib import Path
import json
import numpy as np

SEED = 20260924
SFREQ = 100.0
DURATION_SECONDS = 30.0
EVENT_TIME_SECONDS = 20.0
CHANNELS = ("PM3", "PM4", "PM5", "PM6", "PM7", "PM8", "CC8", "CC9", "CC10", "PA1", "SA1", "CR1")


def make_recording() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n_samples = round(SFREQ * DURATION_SECONDS)
    time = np.arange(n_samples) / SFREQ
    # Correlated low-amplitude background plus independent sensor noise.
    common = 5e-6 * np.sin(2 * np.pi * 9 * time) + 3e-6 * np.sin(2 * np.pi * 3 * time)
    data = np.vstack([common + rng.normal(0, 7e-6, n_samples) for _ in CHANNELS])

    # The current real-recording result identifies PM3-PM8 and CC8-CC10 as
    # early/right-frontal contacts, followed by wider recruitment.  The toy
    # recording mirrors that ordering without copying patient measurements.
    delays = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.05, 0.08, 0.11, 0.45, 0.65, 0.85)
    amplitudes = (110, 105, 100, 95, 90, 85, 100, 95, 90, 72, 68, 64)
    for row, (delay, amplitude_uv) in enumerate(zip(delays, amplitudes)):
        local = time - (EVENT_TIME_SECONDS + delay)
        envelope = np.where((local >= 0) & (local <= 2.0), np.sin(np.pi * local / 2.0) ** 2, 0.0)
        burst = amplitude_uv * 1e-6 * envelope * (
            np.sin(2 * np.pi * 28 * local + row * 0.17)
            + 0.45 * np.sin(2 * np.pi * 55 * local + row * 0.11)
        )
        data[row] += burst
    return data.astype(np.float64)


def main() -> None:
    directory = Path(__file__).resolve().parent
    data = make_recording()
    time = np.arange(data.shape[1]) / SFREQ
    # CSV is sample-major for convenient inspection in spreadsheets and
    # dataframe tools; the CLI transposes the channel columns on load.
    np.savetxt(
        directory / "brain_toy.csv",
        np.column_stack((time, data.T)),
        delimiter=",",
        header=",".join(("time_seconds", *CHANNELS)),
        comments="",
        fmt="%.9e",
    )
    (directory / "brain_toy_channels.txt").write_text("\n".join(CHANNELS) + "\n", encoding="utf-8")
    metadata = {
        "description": "Synthetic multichannel sEEG-like extreme-event example; not patient data.",
        "seed": SEED,
        "shape": list(data.shape),
        "sampling_frequency_hz": SFREQ,
        "duration_seconds": DURATION_SECONDS,
        "event_time_seconds": EVENT_TIME_SECONDS,
        "event_duration_seconds": 2.0,
        "units": "volts",
        "channel_names_file": "brain_toy_channels.txt",
        "data_file": "brain_toy.csv",
        "csv_layout": "rows are samples; columns are time_seconds followed by channels",
        "early_channels": list(CHANNELS[:9]),
        "later_channels": list(CHANNELS[9:]),
    }
    (directory / "brain_toy_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
