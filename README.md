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

## Data-driven extreme-event detection notebook

Open [`sEEG_extreme_event_detector_colab.ipynb`](sEEG_extreme_event_detector_colab.ipynb)
in Google Colab. It scans every non-`MKR` channel with five independent math
methods — conventional time-domain features, Dynamic Time Warping, Detrended
Fluctuation Analysis, Discrete Wavelet Transform energy, and Kuramoto phase
synchronization across delta/theta/alpha/beta/gamma — each robustly
standardized against the recording before the three strongest per-window
method scores are combined into one ensemble, with zero apriori event time
anywhere in that pipeline.

Section 3b then locates the EDF's own annotated seizure the same way the
Python package does (`cp1251`-decoded EDF+ annotations, keyword-matched and
clustered) and, everywhere useful, compares it against the blind ensemble:

* Section 10 computes, **for each of the five methods individually and for
  the combined ensemble**, the score and recording-wide percentile at the
  EDF-annotated peak — a direct, quantified answer to "how would this method
  alone have predicted the real event", not just a plot;
* Sections 11 and 11b render the whole-recording ensemble/method-contribution
  plots and both the strongest-window and the known-peak trace-plus-heatmap
  views with the EDF-annotated peak marked (teal) on every one of them, so
  the blind top-ranked window and the real event are always directly
  comparable on the same figures.

On `sEEG-HFOs-8.edf` they disagree: cached runs of this notebook (before this
comparison existed) returned the recording maximum at 10584 s with
`detected=False`, nowhere near the 10396.445 s peak — a second, independent
confirmation (after the package's own `select_seizure_event` fallback) that
dense interictal activity in this recording can outscore the true seizure in
a purely statistical, blind scan.

## Temporal wavelet correlation graph notebook

Open [`sEEG_temporal_wavelet_graph_colab.ipynb`](sEEG_temporal_wavelet_graph_colab.ipynb)
in Google Colab. It splits every non-`MKR` channel into consecutive 2-second
windows, builds one sparse NetworkX graph per window from thresholded,
top-*k*-pruned `db4` wavelet-coefficient correlations, and saves the whole
temporal sequence as PyTorch tensors. `KNOWN_EVENT_INTERVAL` is the 10396–
10398 s window containing the EDF-annotated peak (`приступ + БТКП` at
10396.445 s, read from the file — see the annotation table above), not an
apriori guess; that window's graph is singled out for visualization, exactly
mirroring `build_seizure_graph`'s co-activation-mesh construction in the
Python package but from `db4` wavelet correlation instead of 13–80 Hz
z-score correlation.

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
directory. By default it analyses **both signal references** — the
recording's native/referential channels and a bipolar (adjacent-contact)
re-referencing — and writes a comparison, so every result below always ships
with the "does this hold up under a different reference" check already run
(see [Bipolar montage](#bipolar-montage-and-comparing-it-against-the-native-reference)).
For each montage reference, it creates a full-duration overview figure
(every EEG channel plus the `MKR...` marker channels), an auditable
candidate report, a beta/gamma recruitment analysis centred on a seizure
time, and — when that analysis finds involved channels — several more
figures visualizing how the seizure recruits them from onset to peak: a
channel-by-time heatmap, a NetworkX node-link graph in four different
layouts, and a message-passing simulation checked against what the
recording actually did next:

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

#### Bipolar montage, and comparing it against the native reference

`build_bipolar_montage` reads the montage straight out of the channel names
already in the EDF — no separate electrode map, and it needs no resolved
event, so `run_edf` always writes it to `<edf-name>_montage.txt`. Each name
is parsed as `<shaft><contact number>` (`parse_contact_name`; e.g. `"EEG
PM3"` → shaft `"PM"`, contact `3` — a trailing `'`, as in `"EEG CC'4"`, is
part of the shaft label, since it marks this dataset's distinct
contralateral electrode, not a variant of the unprimed one). Within each
shaft, contacts are sorted numerically and paired with their numeric
neighbor — the standard bipolar/"referential neighbor" derivation for depth
electrodes — pairing across any gap in the numbering rather than dropping
the relationship. On `sEEG-HFOs-8.edf` this yields 12 shafts and 88 bipolar
pairs (`shaft: pair count`): `R: 9`, `FP: 7`, `FD: 5`, `PM: 7`, `CC: 9`,
`SA: 5`, `PA: 9`, `CC': 9`, `CR': 9`, `PM': 5`, `SA': 7`, `PA': 7` — each
rendered as e.g. `PM:\n  1-2\n  2-3\n  ...\n  7-8` by `format_bipolar_montage`.
This structure — `montage`/`montage_file` in `analysis.json` — is written
unconditionally, independent of which reference was actually analysed.

`apply_bipolar_montage(data, names, montage)` computes the actual
bipolar-referenced signals (`data[a] - data[b]`, labelled `"PM3-4"`): a
re-referencing, not a filter, so it does not change what a detector measures
in kind, only which reference the amplitudes are relative to — it is a
spatial high-pass filter, suppressing whatever two adjacent contacts share
(the recording's common reference, distant volume-conducted activity) and
keeping only their local gradient.

**This is wired into `run_edf` as `montage_reference` (`"none"` or
`"bipolar"`), and the CLI runs both by default** — `--montages none,bipolar`,
overridable to a single value if you only want one. Each lands in its own
`<edf-name>/<montage_reference>/` subdirectory, and `compare_montages`/
`summarize_montage_comparison` produce a top-level `montage_comparison.json`
per recording, reporting what actually differs: candidate count, involved
channel count, likely initiators, co-activation mesh edge count, and the
message-passing validation's best/mean correlation. Event resolution itself
is unaffected — `find_annotated_event` reads the EDF's own text annotations,
not the signal — so both montages are compared at the *same* event time.

On `sEEG-HFOs-8.edf`, `none` finds 98 involved channels and 273 co-activation
edges against `bipolar`'s 82 and 227 — consistent with bipolar suppressing
some of the shared-reference correlation structure referential montage
carries — while message-passing's best correlation against real subsequent
dynamics improves slightly (0.62 → 0.68). Both montages agree on the same
right-frontal `likely_initiators` (`PM3–8`/`CC8–10`), a genuine cross-check
that this hypothesis isn't an artifact of reference choice; bipolar's
initiators are written as pair labels (`PM2-3`, ..., `CC9-10`) rather than
single contacts. (Getting bipolar pair labels correctly classified as
right-frontal required fixing a real bug: the previous `RIGHT_FRONTAL` regex
only checked a pair's *first* contact number, so `PM2-3`/`CC7-8` — whose
*second* endpoint is the one in range — were silently misclassified. Fixed
as `is_right_frontal`, which checks either endpoint.)

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

#### Seizure recruitment graph

`build_seizure_graph`/`plot_seizure_graph` render the same recruitment as a
NetworkX node-link diagram instead of a heatmap:
`<edf-name>_seizure_graph.png`, plus the graph itself as
`<edf-name>_seizure_graph.graphml` for reuse outside this pipeline. Nodes are
exactly `process.onset_latency_seconds`'s channels (same "no re-picked top N"
rule as the heatmap) plus one synthetic `PEAK` node standing for the
resolved event. Two edge kinds, both measured, neither assumed:

* **recruitment spokes** — `PEAK` to every channel, weighted by how soon
  after the peak it was recruited (this is the "...goes to peak" half);
* **co-activation mesh** — Pearson correlation of the channels' own 13–80 Hz
  z-score time courses (the same ones the heatmap plots), threshold- and
  top-*k*-pruned exactly as `sEEG_temporal_wavelet_graph_colab.ipynb` prunes
  its db4-correlation graphs, so this reuses the repo's existing
  graph-building convention rather than inventing a new one (the
  "...how it starts [and] evolves" half, read from what actually co-varies,
  not an assumed propagation path).

`plot_seizure_graph` takes a `layout` argument, and `run_edf` renders all
four via `plot_seizure_graph_layouts`, one file each
(`<edf-name>_seizure_graph_<layout>.png`) — no single arrangement is "the"
seizure graph:

* **`radial`** (the original, still the default) — angle from a spring
  layout of only the co-activation mesh, radius from recruitment latency, so
  the picture reads outside-in as the seizure converges on `PEAK` at the
  centre;
* **`spring`** — one standard force-directed layout over the *whole* graph
  (mesh and recruitment spokes together), letting both edge kinds jointly
  shape the picture instead of only the mesh;
* **`circular`** — channels placed evenly around a circle ordered by
  latency, a plain "clock face of when" with no correlation structure
  involved at all, useful as an uncluttered reference for the other three;
* **`shell`** — two concentric rings, initiators inner and every other
  involved channel outer, isolating the initiator/later-recruited split
  `analyse_brain_process` already makes rather than latency as a continuum.

Every layout still draws initiators as gold diamonds and other involved
channels as circles colour-mapped by peak z-score. On `sEEG-HFOs-8.edf`, with
~90 of 98 channels tied at the same first-window latency, `radial` packs most
nodes into one dense inner arc — an honest consequence of not re-picking a
smaller "top N", not a rendering bug — while the handful of later-recruited
outliers (`EEG CR'5`, `EEG SA'1/2/3`, `EEG PA'3/4`) visibly sit apart from it
at larger radius, the same channels flagged in the heatmap above; the other
three layouts show the identical node set from correlation-only, latency-only,
and initiator-split perspectives instead.

#### Message-passing temporal dynamics

`simulate_message_passing`/`evaluate_message_passing` turn the static graph
into a real, checkable claim about *time*, not just structure. Each involved
channel's already-measured `peak_z` (the maximum post-peak z-score
`analyse_brain_process` found for it) seeds one linear diffusion update, run
for several steps, over the same co-activation mesh the graph already has —
`h(t+1) = alpha·h(t) + (1 − alpha)·D⁻¹Wh(t)`, degree-normalized so no channel
accumulates unbounded activation. `evaluate_message_passing` then spatially
(cross-channel) Pearson-correlates each propagation step against what the
recording's own 13–80 Hz z-scores *actually* looked like at the matching
elapsed real time — this is the "temporal dynamic evaluation": does the
static graph's structure, diffused forward, predict the real subsequent
spread, or not?

`run_edf` always renders both outputs when the process found involved
channels: `<edf-name>_message_passing.png` (`plot_message_passing`, one
network-state panel per step, shared colour scale, `spring` layout by
default so a loosely-connected outlier channel does not dominate any single
panel's axis scale) and `<edf-name>_message_passing_validation.png`
(`plot_message_passing_validation`, correlation vs. elapsed time). The raw
`{"elapsed_seconds": [...], "correlation": [...]}` is also written to
`analysis.json` as `message_passing_evaluation`.

**On `sEEG-HFOs-8.edf` the answer is: not very well, and that is the honest
result.** Correlation opens at 0.62 (step 0 is not trivially 1.0, because the
seed is each channel's *maximum* post-peak z-score, not literally its
instantaneous value at `event.time_seconds`), falls to roughly 0.05–0.35 and
stays there through 8 s post-peak. A single linear diffusion over a graph
built from one moment does not reproduce this recording's actual
spatiotemporal evolution — plausible, given a generalizing bilateral
tonic-clonic seizure is a far richer process than one static correlation
snapshot can encode. Treat this figure pair the same way as everything else
here: a measured check on a simplified model, not a demonstration that the
model works.

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
   confirmed offset and want that to drive the process analysis instead. It
   analyses both montage references by default (roughly twice the runtime of
   one, since it is the full pipeline run twice); add `--montages bipolar`
   there if you only want one.
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
# present (tier 2), the blind detector otherwise (tier 3). Runs both montage
# references (the default --montages none,bipolar) and writes a comparison.
seeg-event-agent dataset --output seeg_agent_output

# Only one montage reference, if you don't want to pay for both
seeg-event-agent dataset --output seeg_agent_output --montages bipolar

# Known offset from EDF start
seeg-event-agent dataset --output seeg_agent_output --event-time <seconds>

# Known clinical wall-clock time, only if synchronized to this EDF's meas_date
seeg-event-agent dataset/sEEG-HFOs-8.edf --output seeg_agent_output \
  --event-clock <HH:MM:SS>
```

Each recording is written to its own `seeg_agent_output/<edf-name>/`
directory, with one subdirectory per montage reference actually run
(`none/`, `bipolar/`, or both):

* `<edf-name>/montage_comparison.json` (only when more than one montage
  reference was run) is `summarize_montage_comparison`'s per-montage table —
  candidate count, involved channel count, likely initiators, co-activation
  edge count, message-passing best/mean correlation;
* `<edf-name>/<montage_reference>/analysis.json` contains detected
  candidates, `montage_reference` itself, `montage` (the bipolar
  shaft/contact-pair grouping, see below — always describes the referential
  structure, regardless of which reference was analysed) and `montage_file`,
  and whichever event tier resolved: `clinical_annotation` (tier 1,
  `--event-time`/`--event-clock`), `annotated_event` (tier 2, the EDF's own
  annotation channel — includes the full matched annotation cluster for
  audit, and is identical across montages since it comes from the file's
  text annotations, not the signal), or `detected_event` (tier 3, the blind
  fallback, which *does* vary by montage) — plus beta/gamma channel scores,
  recruitment latencies, the right-frontal process hypothesis computed from
  whichever one was used, `evolution_figure`, `graph_figures` (one path per
  layout) and `graph_graphml`, and `message_passing_figure`/
  `message_passing_validation_figure`/`message_passing_evaluation` (all
  `null`/empty when no channels were involved);
* `<edf-name>/<montage_reference>/<edf-name>_montage.txt` is the bipolar
  montage (see below), written unconditionally — it needs no resolved event;
* `<edf-name>/<montage_reference>/<edf-name>_all_timeseries.png` contains
  every analysed channel plus `MKR...` markers over the complete recording
  and the event marker — solid crimson for `clinical_annotation`, solid teal
  for `annotated_event`, dashed orange for `detected_event`;
* `<edf-name>/<montage_reference>/<edf-name>_seizure_evolution.png` (only
  when `brain_process` found involved channels) is the recruitment-cascade
  heatmap described above;
* `<edf-name>/<montage_reference>/<edf-name>_seizure_graph_<layout>.png`
  (`radial`/`spring`/`circular`/`shell`, same condition) are the node-link
  renderings, and `..._seizure_graph.graphml` is the underlying NetworkX graph;
* `<edf-name>/<montage_reference>/<edf-name>_message_passing.png` and
  `..._message_passing_validation.png` (same condition) are the
  diffusion-simulation panels and its validation-against-reality plot.

If VS Code reports `No module named extreme_event_agent`, verify that `.venv` is
the selected interpreter and repeat `python -m pip install -e .` in the VS Code
terminal.
