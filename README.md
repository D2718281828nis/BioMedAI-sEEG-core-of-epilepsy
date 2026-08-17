# BioMedAI-sEEG-core-of-epilepsy
Biomakreks of sEEG timeseries analysis to find core of epilepsy as a dynamic process.

sourced dataset: https://zenodo.org/records/21967993
'sEEG-HFOs-8.edf': seizure 17:27:11, clinicaly confirmed 17:27:14


## Google Colab notebooks

Open [`sEEG_EDF_viewer_colab.ipynb`](sEEG_EDF_viewer_colab.ipynb) in Google
Colab to inspect the channel metadata, browse time windows, and plot the power
spectrum of `dataset/sEEG-HFOs-8.edf`. The notebook also explains how to upload
the file when it is not already present in the Colab runtime.

## Interactive discrete wavelet viewer

Open [`sEEG_DWT_viewer_colab.ipynb`](sEEG_DWT_viewer_colab.ipynb) in Google
Colab and run its cells from top to bottom. The notebook loads or uploads
`sEEG-HFOs-8.edf`, exposes every signal channel in an interactive selector,
excludes channels whose names begin with `MKR`, and lets you choose the mother
wavelet used to plot aligned approximation and detail components.

## Agentic extreme-event discovery

The installable `extreme_event_agent` package adds a reproducible agentic
workflow for finding previously unknown extreme events in multichannel time
series. Its bounded **plan → act → observe → reflect** loop validates data,
calls deterministic signal-analysis tools, adapts its candidate threshold when
needed, verifies spatial support, and keeps a complete audit trail. Numerical
tools—not an LLM—make the medical-data decisions, so identical inputs and
configuration produce identical results. Results are candidates for expert
review, not diagnoses.

Install and scan a NumPy array shaped `[channels, samples]`:

```bash
python -m pip install -e .
seeg-event-agent recording.npy --sfreq 1000 --channels channel_names.txt \
  --output extreme_events.json
```

Python API:

```python
from extreme_event_agent import AgentConfig, ExtremeEventAgent

agent = ExtremeEventAgent(AgentConfig(window_seconds=2, step_seconds=0.25))
report = agent.run(data, sampling_frequency_hz=1000, channel_names=names)
for event in report.events:
    print(event.start_seconds, event.involved_channels, event.confidence)
```

The detector combines robust window-level RMS, peak-to-peak amplitude, line
length, and high-frequency difference energy. It forms a focal multichannel
consensus, uses median/MAD normalization, merges adjacent candidates, and
rejects candidates without enough independently involved channels. Configure
window size, channel fraction, evidence threshold, and spatial support through
`AgentConfig`; do not tune them against a clinical label without held-out
validation.
