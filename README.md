# BioMedAI-sEEG-core-of-epilepsy
Biomakreks of sEEG timeseries analysis to find core of epilepsy as a dynamic process.

sourced dataset: https://zenodo.org/records/21967993

`sEEG-HFOs-8.edf` carries its own EDF+ annotation channel — the clinician's
real-time markup, embedded in the file rather than typed as a separate
number. Decoded with the correct `cp1251` (Windows-1251 Cyrillic) codec
instead of MNE's default UTF-8, it reads:

| onset (s from EDF start) | annotation | meaning |
|---|---|---|
| 10392.734 | `начало приступа?` | "is seizure starts here?" |
| 10396.445 | `приступ + БТКП` | "seizure + bilateral tonic-clonic seizure" |
| 10399.469 | `клиника` | "clinical [onset]" |

This is the file's own ground truth for the one asymmetric-tonic-to-bilateral
seizure described below, and it is what `extreme_event_agent` now reads
automatically (see [Analyse and visualize EDF recordings](#analyse-and-visualize-edf-recordings)).
It supersedes an earlier note here citing an offset of 808 s / clock
`17:27:14`: that figure cannot be reconciled with this EDF's own (anonymized)
`meas_date` header — `--event-clock 17:27:14` raises `ValueError: ... outside
the 14095.000 s recording` — so it was likely copied from an external
clinical or video-system clock that was never synchronized to this file.


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
directory. It creates a full-duration overview figure (every EEG channel plus
the `MKR...` marker channels), an auditable candidate report, a beta/gamma
recruitment analysis centred on a seizure time, and — when that analysis
finds involved channels — a second figure visualizing how the seizure
recruits them from onset to peak:

```bash
seeg-event-agent dataset/ --output seeg_agent_output
```

`--event-time`/`--event-clock` are optional. No apriori event time is
required, because the seizure time is resolved through a three-tier
priority, each level falling back to the next only when the one above is
unavailable:

1. **`--event-time`/`--event-clock`** — an expert time typed on the command
   line. Written to `analysis.json` as `clinical_annotation`; drawn **solid
   crimson** in the figure.
2. **The EDF's own EDF+ annotation channel** — `find_annotated_event` reads
   every annotation already embedded in the file (`cp1251`-decoded; MNE's
   default UTF-8 decode raises on Cyrillic annotation text) and looks for one
   whose text names a seizure (`приступ`, `судорог`, `seizure`, `ictal`,
   `бткп`, `tcs`). Every other annotation within 10 s of that match is folded
   in, so a clinician's separate notes about the same seizure (e.g. an
   "onset?" query beside a "seizure" tag) read as one event. This is
   metadata the clinician already wrote into the recording, not a number
   supplied externally. Written as `annotated_event`; drawn **solid teal**.
3. **`select_seizure_event`** — only when a file has no such annotation, the
   primary candidate is picked from the agent's own verified detections,
   ranked by channel spread, then duration, then score, since a real seizure
   recruits far more contacts for far longer than the brief high-amplitude
   interictal spikes that otherwise score just as high. Written as
   `detected_event`; drawn **dashed orange** — the only marker that should
   ever be read as an unconfirmed algorithmic guess.

On `sEEG-HFOs-8.edf` this resolves to tier 2: the `приступ + БТКП` annotation
at 10396.445 s (see above). Running the beta/gamma process analysis on that
time reproduces the clinical picture from data alone — `likely_initiators`
comes out as `PM3–PM8` and `CC8–CC10` (right frontal), and `later_recruited`
spans both primed and unprimed `PA`/`SA`/`CR` contacts (bilateral frontal and
parietal) — without any apriori time ever entering the pipeline. By contrast,
tier 3's blind fallback (tested by temporarily ignoring the annotation) finds
only two multichannel groups in the whole ~3.9 h recording and neither is
this one; a supplementary whole-recording 13–80 Hz band-energy scan turns up
several hundred similarly-sized bursts throughout, matching the clinical note
that dense interictal epileptiform activity confounds a blind read. Prefer an
EDF with a real annotation, or `--event-time`/`--event-clock`, over the tier-3
fallback; treat `detected_event` as a lead for expert review, not a result.

`--event-time` is seconds from the start of each EDF; do not pass wall-clock
time without first converting it using the EDF start time — and confirm that
time is actually synchronized to this EDF's `meas_date` before trusting
`--event-clock` (see the note above). The process model uses sliding 13–80 Hz
energy, robust pre-event median/MAD normalization, and a six-MAD recruitment
threshold. Contacts matching PM3–8 and CC8–10 are reported as the
right-frontal hypothesis only when they cross that data-derived threshold;
later crossings describe rapid spread. This operationalizes the supplied
clinical context without treating it as ground truth. The outputs remain
research candidates requiring review of the raw EDF, montage, video, and
clinical record; they are not a diagnosis or medical device.

#### `MKR...` marker channels

`read_edf` keeps `MKR1+`/`MKR2+` out of both the statistical detector and the
beta/gamma process analysis: checked directly, every transition on both
channels is exactly 0.5 s from the last for the entire ~3.9 h recording, with
no anomaly around any detected or annotated event — a hardware sync clock,
not brain signal or an event marker. Including a channel like that in
`likely_initiators`/`later_recruited` would misrepresent a hardware artifact
as neural or clinical evidence. They are, however, loaded (via
`read_edf_markers`, which reads only those two channels) and included in the
whole-recording overview figure for visual/QC context — a reviewer can
confirm the clock is behaving as expected next to the real signal. If a
different EDF's marker channel turns out to carry real event information
(irregular transitions, i.e. actual button presses), inspect it with
`read_edf_markers` before trusting it as a seizure time.

#### Seizure evolution figure

When the resolved event yields a `BrainProcess` with involved channels,
`plot_seizure_evolution` renders `<edf-name>_seizure_evolution.png`: a
channel-by-time heatmap of the same 13–80 Hz median/MAD z-score
`analyse_brain_process` computes, restricted to exactly the channels it
already found involved and ordered by their recruitment latency — initiators
at the top, later-recruited contacts at the bottom, never a separately
re-picked "top N". A dashed line marks the event time; since tier 2's
annotation labels 10396.445 s as the point the seizure was scored as already
*"seizure + bilateral tonic-clonic"* (with the clinician's own preceding
"where does it start?" note at 10392.734 s), this is the cascade's **peak**,
not necessarily its first twitch — the figure's baseline window (30 s before,
8 s after, by default) is there specifically to show the build-up leading
into it, not just the instant of crossing.

On `sEEG-HFOs-8.edf` this shows what looks like the tonic-clonic phase
itself: 98 of 100 channels cross the recruitment threshold, and roughly 90 of
them do so in the very first analysis window after the peak — consistent
with the generalized, whole-montage EEG/EMG signature of an established
bilateral tonic-clonic seizure rather than a clean focal cascade (that focal
cascade, if visible at all, would be in the seconds *before* the peak, where
the figure shows much sparser activity). A handful of contacts are
distinctly recruited later — `EEG PA9` (0.24 s), `EEG PA'3` (0.39 s), `EEG
SA'2`/`EEG SA'3` (2.0 s), `EEG PA'4` (2.1 s), `EEG CR'5` (5.7 s) — and these
are exactly the primed (left-hemisphere) contacts the tier-3 blind detector
independently flagged as a separate late candidate group, a small but
genuine cross-check between two unrelated parts of this pipeline.

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
   press `F5`. It runs with no expert event time, so the seizure time comes
   from tier 2 or 3 above (the EDF's own annotation channel, or the blind
   detector as a last resort); add `--event-time <seconds>` in
   [`.vscode/launch.json`](.vscode/launch.json) if you have an independently
   confirmed offset and want that to drive the process analysis instead.
5. Alternatively, select **Analyse EDF by clinical clock** to enter an EDF path
   and a clinical wall-clock time. `--event-clock` reads the start time from
   each EDF, handles midnight rollover, and rejects times outside the
   recording. Use this only when you have independently confirmed the
   clinical and EDF clocks are synchronized — for the bundled
   `sEEG-HFOs-8.edf` they are not (see the note at the top of this file), so
   use **Analyse all EDF files** for it instead.

The equivalent terminal commands are:

```bash
# No apriori event time: the EDF's own annotation channel resolves it when
# present (tier 2), the blind detector otherwise (tier 3)
seeg-event-agent dataset --output seeg_agent_output

# Known offset from EDF start
seeg-event-agent dataset --output seeg_agent_output --event-time <seconds>

# Known clinical wall-clock time, only if synchronized to this EDF's meas_date
seeg-event-agent dataset/sEEG-HFOs-8.edf --output seeg_agent_output \
  --event-clock <HH:MM:SS>
```

Each recording is written to its own `seeg_agent_output/<edf-name>/` directory:

* `analysis.json` contains detected candidates and whichever event tier
  resolved: `clinical_annotation` (tier 1, `--event-time`/`--event-clock`),
  `annotated_event` (tier 2, the EDF's own annotation channel — includes the
  full matched annotation cluster for audit), or `detected_event` (tier 3,
  the blind fallback) — plus beta/gamma channel scores, recruitment
  latencies, the right-frontal process hypothesis computed from whichever one
  was used, and `evolution_figure` (`null` when no channels were involved);
* `<edf-name>_all_timeseries.png` contains every EEG and `MKR...` marker
  channel over the complete recording and the event marker — solid crimson
  for `clinical_annotation`, solid teal for `annotated_event`, dashed orange
  for `detected_event`;
* `<edf-name>_seizure_evolution.png` (only when `brain_process` found
  involved channels) is the recruitment-cascade heatmap described above.

If VS Code reports `No module named extreme_event_agent`, verify that `.venv` is
the selected interpreter and repeat `python -m pip install -e .` in the VS Code
terminal.
