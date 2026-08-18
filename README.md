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

### Analyse and visualize EDF recordings

The CLI can now process one EDF or recursively process **all EDF files** in a
directory. It creates a full-duration figure containing every non-marker EEG
time series, an auditable candidate report, and (when an expert time is given)
a separate beta/gamma recruitment analysis:

```bash
seeg-event-agent dataset/ --output seeg_agent_output \
  --event-time 808 --event-label "асимметричный тонический приступ"
```

`--event-time` is seconds from the start of each EDF; do not pass wall-clock
time without first converting it using the EDF start time. The red figure mark
is an expert annotation, not a detector result. The process model uses sliding
13–80 Hz energy, robust pre-event median/MAD normalization, and a six-MAD
recruitment threshold. Contacts matching PM3–8 and CC8–10 are reported as the
right-frontal hypothesis only when they cross that data-derived threshold;
later crossings describe rapid spread. This operationalizes the supplied
clinical context without treating it as ground truth. The outputs remain
research candidates requiring review of the raw EDF, montage, video, and
clinical record; they are not a diagnosis or medical device.

#### Run locally in VS Code

1. Open this repository as the VS Code workspace and put EDF files in
   `dataset/` (the folder is intentionally not committed).
2. Open **Terminal → New Terminal**, create an isolated environment, and install
   the project:

   **Windows PowerShell**

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -e .
   ```

   **macOS/Linux**

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .
   ```

3. Run **Python: Select Interpreter** from the Command Palette and choose the
   `.venv` interpreter.
4. Open **Run and Debug** (`Ctrl+Shift+D`), select **Analyse all EDF files**, and
   press `F5`. Edit `--event-time` in [`.vscode/launch.json`](.vscode/launch.json)
   if the event is not 808 seconds from the EDF start.
5. Alternatively, select **Analyse EDF by clinical clock** to enter an EDF path
   and use the documented `17:27:14` clinical time. `--event-clock` reads the
   start time from each EDF, handles midnight rollover, and rejects times outside
   the recording. Use this only when the clinical and EDF clocks are synchronized.

The equivalent terminal commands are:

```bash
# Known offset from EDF start
seeg-event-agent dataset --output seeg_agent_output --event-time 808

# Known clinical wall-clock time
seeg-event-agent dataset/sEEG-HFOs-8.edf --output seeg_agent_output \
  --event-clock 17:27:14
```

Each recording is written to its own `seeg_agent_output/<edf-name>/` directory:

* `analysis.json` contains detected candidates, the independent clinical marker,
  beta/gamma channel scores, recruitment latencies, and the right-frontal process
  hypothesis;
* `<edf-name>_all_timeseries.png` contains every non-marker EEG channel over the
  complete recording and the red clinical-event marker.

If VS Code reports `No module named extreme_event_agent`, verify that `.venv` is
the selected interpreter and repeat `python -m pip install -e .` in the VS Code
terminal. Run without `--event-time` and `--event-clock` when no independently
confirmed event time is available; detection and full-recording plotting still
run, but no clinical label or event-centred process analysis is added.
