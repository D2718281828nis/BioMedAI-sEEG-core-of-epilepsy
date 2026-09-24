# Brain toy dataset

This directory contains a deterministic, synthetic brain example for running
the repository without downloading the clinical EDF. It complements any
finance and aviation examples; it does not modify or derive data from them.

## Files

- `brain_toy.csv`: human-readable table with 3,000 sample rows, a
  `time_seconds` column, and 12 channel columns. The CLI converts it to its
  internal `[channels, samples]` representation.
- `brain_toy_channels.txt`: channel labels in array-row order.
- `brain_toy_metadata.json`: sampling rate, event timing, units, and provenance.
- `generate_brain_toy.py`: deterministic generator for auditing or rebuilding
  all three files.

The recording is 30 seconds at 100 Hz, expressed in volts. It contains a
low-amplitude correlated sEEG-like background and one synthetic beta/gamma
burst beginning at 20 seconds. The early contacts (`PM3`–`PM8`, `CC8`–`CC10`)
and subsequent `PA1`/`SA1`/`CR1` spread reproduce the *pattern* of the current
repository result. No patient samples or identifying data are included, and
this toy signal is not physiologically validated or suitable for clinical
use.

## Run the independent brain demonstration

From the repository root:

```bash
python -m extreme_event_agent.cli data/brain_toy.csv \
  --sfreq 100 \
  --output brain_toy_events.json
```

If the package has not been installed, use `PYTHONPATH=src` before the command.
CSV channel names are read from the header; `brain_toy_channels.txt` is also
provided for consumers that need a separate ordered label file. The detector
is blind to `event_time_seconds` in the metadata; that timestamp
is provided only so the detected candidates can be checked after the run.
Rebuild the exact checked-in data with:

```bash
python data/generate_brain_toy.py
```
