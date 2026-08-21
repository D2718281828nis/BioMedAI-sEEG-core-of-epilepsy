# BioMedAI-sEEG-core-of-epilepsy

**[English](#english)** | **[Русский](#русский)**

---

## English

Biomarkers of sEEG timeseries analysis to find core of epilepsy as a dynamic process.

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


### Google Colab notebooks

Open [`sEEG_EDF_viewer_colab.ipynb`](sEEG_EDF_viewer_colab.ipynb) in Google
Colab to inspect the channel metadata, browse time windows, and plot the power
spectrum of `dataset/sEEG-HFOs-8.edf`. The notebook also explains how to upload
the file when it is not already present in the Colab runtime.

### Interactive discrete wavelet viewer

Open [`sEEG_DWT_viewer_colab.ipynb`](sEEG_DWT_viewer_colab.ipynb) in Google
Colab and run its cells from top to bottom. The notebook loads or uploads
`sEEG-HFOs-8.edf`, exposes every signal channel in an interactive selector,
excludes channels whose names begin with `MKR`, and lets you choose the mother
wavelet used to plot aligned approximation and detail components.

### Data-driven extreme-event detection notebook

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

**Data hygiene matters here.** After ≈10550 s, `sEEG-HFOs-8.edf` contains a
surgical cauterization procedure, not brain activity. Earlier cached runs of
this notebook scanned the file end to end, including that segment, and
returned the recording maximum at 10584 s (*inside* the cauterization
segment) with `detected=False` — a non-physiological artifact both became the
reported "detection" and inflated the recording's own normalization
statistics enough to hide the real seizure. The notebook now crops to 0–10550 s
before any analysis (`ANALYSIS_END_SECONDS`); on the cleaned recording the
ensemble crosses its own automatic threshold (`detected=True`) with its top
candidate 39.6 s after the annotated onset — a late, coarse detection, not a
miss. See [`sEEG_blind_vs_targeted_detection_colab.ipynb`](sEEG_blind_vs_targeted_detection_colab.ipynb)
for the full before/after comparison and what it does and doesn't demonstrate
about interictal-vs-ictal separability.

### Temporal wavelet correlation graph notebook

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

### Agentic extreme-event discovery

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

#### Analyse and visualize EDF recordings

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

`--crop-end-seconds` discards everything in each EDF after the given time,
before any analysis sees it — use it to exclude a known non-physiological
tail. `sEEG-HFOs-8.edf` needs this: after ≈10550 s it contains a surgical
cauterization procedure, not brain activity, which otherwise both pollutes
the blind detector's own recording-wide normalization statistics and can
itself become the reported "detection" (see the tier-3 discussion below).

```bash
seeg-event-agent dataset/ --output seeg_agent_output --crop-end-seconds 10550
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
tier 3's blind fallback (tested by temporarily ignoring the annotation, on
the file cropped to 0–10550 s per the data hygiene note above) finds 4
candidate groups; the strongest two sit 41.6 s and 126.1 s after the
annotated onset (scores 24.5 and 25.0, involving 46 and 13 channels
respectively) — a late, coarse lead pointing at roughly the right place, not
a precise or reliably-timed one. Scanning the *uncropped* file instead finds
only 2 candidates, and the one nearest the annotation scores markedly lower
(16.6 vs. 24.5 cropped) for the same channels at essentially the same time —
the cauterization segment's non-physiological amplitude both dilutes that
candidate's rank and, on other detector configurations, can outscore it
entirely (see the notebook comparison linked above). Either way, tier 3 is
imprecise by construction: it can suggest roughly when something happened,
not confirm it is the seizure specifically. Prefer an EDF with a real
annotation, or `--event-time`/`--event-clock`, over the tier-3 fallback;
treat `detected_event` as a lead for expert review, not a result.

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

##### Bipolar montage, and comparing it against the native reference

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

##### `MKR...` marker channels

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

##### Seizure evolution figure

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

##### Seizure recruitment graph

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

##### Message-passing temporal dynamics

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

`run_edf` always renders these outputs when the process found involved
channels: `<edf-name>_message_passing.png` (`plot_message_passing`, one
network-state panel per step, shared colour scale, `spring` layout by
default so a loosely-connected outlier channel does not dominate any single
panel's axis scale), the same diffusion re-rendered in every one of
`plot_seizure_graph_layouts`'s four layouts via `plot_message_passing_layouts`
(`<edf-name>_message_passing_<layout>.png` — `radial`/`spring`/`circular`/`shell`,
so the propagation from source to `PEAK` can be read against latency,
combined structure, a plain clock face, or the initiator split, the same
choice already offered for the static graph), and
`<edf-name>_message_passing_validation.png` (`plot_message_passing_validation`,
correlation vs. elapsed time). Every message-passing figure carries a legend
identifying its markers/edges and a boxed caption explaining what it shows.
The raw `{"elapsed_seconds": [...], "correlation": [...]}` is also written to
`analysis.json` as `message_passing_evaluation`.

Alongside these, `describe_seizure_source` turns `process` into a
plain-language statement of the located source — which channel(s) crossed
the recruitment threshold first (the likely source), at what absolute
recording time, and how many channels were ultimately involved — written to
`<edf-name>_source_summary.txt` and to `analysis.json` as `source_summary`.
The same source (initiator channel(s) and absolute time) is also boxed
directly on `plot_seizure_graph`'s figure, next to a legend identifying every
marker and edge kind and a colourbar for the peak-z-score colour scale.

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

##### Run locally in VS Code

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
  layout) and `graph_graphml`, `message_passing_figure`/`message_passing_figures`
  (one path per layout)/`message_passing_validation_figure`/
  `message_passing_evaluation`, and `source_summary`/`source_summary_file`
  (`describe_seizure_source`'s plain-language source statement — all
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
* `<edf-name>/<montage_reference>/<edf-name>_message_passing.png`,
  `..._message_passing_<layout>.png` (`radial`/`spring`/`circular`/`shell`),
  and `..._message_passing_validation.png` (same condition) are the
  diffusion-simulation panels in every layout and its validation-against-reality
  plot;
* `<edf-name>/<montage_reference>/<edf-name>_source_summary.txt` (same
  condition) is `describe_seizure_source`'s plain-language statement of the
  located source channel(s) and time.

If VS Code reports `No module named extreme_event_agent`, verify that `.venv` is
the selected interpreter and repeat `python -m pip install -e .` in the VS Code
terminal.

### Reservoir-computing state-space model (`model/`)

`model/` adds a second, independent way to look at the same seizure: not a
detector, but a **plant** — in the Automatic Control Theory sense of the
word. An Echo State Network (a classic reservoir-computing architecture) is
wired up as an explicit discrete-time nonlinear state-space system,

```
x(t) = (1 - leak) x(t-1) + leak * tanh(B u(t) + A x(t-1) + bias)   # state equation
y(t) = C x(t) + D u(t)                                             # output equation
```

with a literal input/output split read straight off the EDF: the `MKR...`
hardware-clock channels (kept out of detection everywhere else in this repo
because they carry no brain signal) are the plant's exogenous input `u(t)`;
a subset of real EEG channels — the same right-frontal `likely_initiators`
plus a spread of `later_recruited` channels `analyse_brain_process` already
found — is the observed output `y(t)`. `A` (the reservoir) and `B` are fixed
random matrices generated once; only the linear readout (`C`, `D`) is
trained, by ridge regression, on the **pre-event baseline only** — reservoir
computing's whole trick is that one cheap linear fit on top of a large fixed
nonlinear dynamical system is enough to capture rich temporal structure.

Since the clock alone is nearly constant between its 1 Hz pulses and carries
almost no information correlated with fast EEG structure, `u(t)` is
augmented with a short delay embedding of the target's own recent,
already-observed past (`y(t-1), ..., y(t-lag)`) — a standard NARX
("nonlinear autoregressive with exogenous input") extension to reservoir
computing, and the same delay-embedding idea this project's reference
implementation ([D2718281828nis/ML-Reservoir_Computing](https://github.com/D2718281828nis/ML-Reservoir_Computing))
uses for its own next-generation-RC notebook. This never leaks `y(t)`
itself, only strictly earlier samples, so the resulting one-step-ahead
forecast is a legitimate, checkable prediction straight through the extreme
event, not just on the baseline it was fit to.

`run_reservoir_plant` then runs that *same*, baseline-only-trained model
forward across the whole window (baseline and event alike) and measures the
residual between what it predicts and what the recording actually did — a
classic observer-residual fault/anomaly signal from control theory, used
here as this model's own, independent extreme-event evaluation: a
median/MAD z-score of the residual magnitude, thresholded at 6 MAD (matching
`analyse_brain_process`'s own recruitment threshold) after a short
moving-average smoothing so a single noisy sample can't count as "detected"
on its own.

Run it (module invocation is required, for its relative imports):

```bash
python -m model.run_model dataset/sEEG-HFOs-8.edf --output model_result
```

On `sEEG-HFOs-8.edf` this resolves the same tier-2 annotation
(`приступ + БТКП` at 10396.445 s) `run_edf` does, trains on a 60 s
pre-event baseline, and evaluates 20 s past it: baseline readout RMSE lands
around 4–9×10⁻⁵ per channel (the NARX-augmented prediction visibly tracks
the real waveform, not just its mean — see `..._output_prediction.png`),
and the residual score peaks at ~70+ MAD a few seconds after the annotated
peak — a sustained, independently-built confirmation of the same event, via
a completely different mechanism (prediction residual, not spatial
recruitment) than the rest of this repository.

`model_result/<edf-name>_*.png` (seven figures per run):

* `_architecture.png` — the state-space block diagram (`u → B,A → x → C,D → y`)
  for *this* run's actual dimensions;
* `_connectivity.png` — the reservoir's own random recurrent graph (a
  legible sample of hidden units plus the exogenous input/output nodes and
  their strongest edges, with a legend);
* `_spectrum.png` — eigenvalues of `A` against the unit circle (the
  echo-state-property picture);
* `_hidden_state.png` — every reservoir unit's activation over time;
* `_output_prediction.png` — real vs. baseline-trained-model-predicted
  output, per channel;
* `_residual_heatmap.png` — the same real-minus-predicted difference as one
  per-channel, per-timestep heatmap (each channel independently
  median/MAD-normalized against its own baseline, so channels of very
  different native amplitude are still visually comparable);
* `_extreme_event_score.png` — the aggregated residual score vs. threshold,
  with the model's own onset/peak marked.

`<edf-name>_model_summary.txt`/`_model_result.json` carry the same numbers
as plain text/JSON — reservoir configuration, per-channel training RMSE, and
the extreme-event verdict (`describe_evaluation`, this model's counterpart
to `describe_seizure_source`). As with everything else in this repository, a
"detected" result here is a candidate for expert review — checked against,
not treated as more authoritative than, `extreme_event_agent`'s own
spatial-recruitment-based detection.

---

## Русский

Биомаркеры динамики sEEG-сигналов для поиска очага (ядра) эпилепсии как динамического процесса.

источник датасета: https://zenodo.org/records/21967993

`sEEG-HFOs-8.edf` содержит собственный канал аннотаций EDF+ — разметку
клинициста в реальном времени, встроенную прямо в файл, а не отдельное
число. При корректном декодировании кодировкой `cp1251` (Windows-1251,
кириллица) вместо стандартной для MNE UTF-8, аннотации выглядят так:

| начало (с от старта EDF) | аннотация | значение |
|---|---|---|
| 10392.734 | `начало приступа?` | «здесь ли начинается приступ?» |
| 10396.445 | `приступ + БТКП` | «приступ + билатеральный тонико-клонический приступ» |
| 10399.469 | `клиника` | «клиническое [начало]» |

Это собственная эталонная разметка файла для единственного асимметричного
тонического → билатерального приступа, описанного ниже, и именно её теперь
автоматически читает `extreme_event_agent` (см. [Анализ и визуализация
EDF-записей](#анализ-и-визуализация-edf-записей)). Она отменяет более
раннюю заметку об офсете 808 с / времени `17:27:14`: эта цифра не
согласуется с собственным (анонимизированным) заголовком `meas_date` этого
EDF — `--event-clock 17:27:14` выбрасывает `ValueError: ... outside the
14095.000 s recording` — так что она, вероятно, была скопирована с внешних
клинических часов или часов видеосистемы, никогда не синхронизированных с
этим файлом.

### Ноутбуки Google Colab

Откройте [`sEEG_EDF_viewer_colab.ipynb`](sEEG_EDF_viewer_colab.ipynb) в
Google Colab, чтобы посмотреть метаданные каналов, просмотреть временные
окна и построить спектр мощности `dataset/sEEG-HFOs-8.edf`. Ноутбук также
объясняет, как загрузить файл, если его ещё нет в среде выполнения Colab.

### Интерактивный просмотрщик дискретного вейвлет-преобразования

Откройте [`sEEG_DWT_viewer_colab.ipynb`](sEEG_DWT_viewer_colab.ipynb) в
Google Colab и выполните его ячейки сверху вниз. Ноутбук загружает (или
предлагает загрузить) `sEEG-HFOs-8.edf`, показывает все сигнальные каналы в
интерактивном селекторе, исключает каналы с именами, начинающимися на
`MKR`, и позволяет выбрать материнский вейвлет для построения выровненных
компонент аппроксимации и детализации.

### Ноутбук data-driven обнаружения экстремальных событий

Откройте [`sEEG_extreme_event_detector_colab.ipynb`](sEEG_extreme_event_detector_colab.ipynb)
в Google Colab. Он сканирует каждый не-`MKR` канал пятью независимыми
математическими методами — классические временные признаки, Dynamic Time
Warping, Detrended Fluctuation Analysis, энергия дискретного
вейвлет-преобразования и синхронизация фаз Курамото по диапазонам
delta/theta/alpha/beta/gamma — каждый робастно стандартизуется относительно
записи, после чего три сильнейших по каждому окну метода объединяются в
один ансамбль, причём никакое априорное время события нигде в этом
пайплайне не используется.

Раздел 3b затем находит собственную аннотированную запись приступа в EDF
так же, как это делает python-пакет (`cp1251`-декодированные
EDF+-аннотации, поиск по ключевым словам и кластеризация), и там, где это
полезно, сравнивает её со слепым ансамблем:

* Раздел 10 вычисляет **для каждого из пяти методов по отдельности и для
  объединённого ансамбля** оценку и перцентиль по всей записи в точке
  аннотированного EDF-пика — прямой количественный ответ на вопрос «как
  этот метод сам по себе предсказал бы реальное событие», а не просто
  график;
* Разделы 11 и 11b строят графики ансамбля/вклада методов по всей записи, а
  также совмещённые виды «сильнейшее окно» и «известный пик» (трасса +
  тепловая карта) с отмеченным (бирюзовым) аннотированным EDF-пиком на
  каждом из них, так что слепое top-ранжированное окно и реальное событие
  всегда напрямую сопоставимы на одних и тех же рисунках.

**Здесь важна гигиена данных.** После ≈10550 с `sEEG-HFOs-8.edf` содержит
хирургическую процедуру каутеризации, а не мозговую активность. Более
ранние закэшированные прогоны этого ноутбука сканировали файл целиком,
включая этот сегмент, и возвращали максимум записи на 10584 с (*внутри*
сегмента каутеризации) с `detected=False` — нефизиологичный артефакт
одновременно стал заявленным «обнаружением» и настолько исказил
собственную нормировочную статистику записи, что скрыл настоящий приступ.
Теперь ноутбук обрезает запись до 0–10550 с перед любым анализом
(`ANALYSIS_END_SECONDS`); на очищенной записи ансамбль пересекает
собственный автоматический порог (`detected=True`), при этом лучший
кандидат находится через 39.6 с после аннотированного начала — позднее,
грубое обнаружение, но не промах. См.
[`sEEG_blind_vs_targeted_detection_colab.ipynb`](sEEG_blind_vs_targeted_detection_colab.ipynb)
для полного сравнения «до/после» и того, что оно демонстрирует (а что нет)
в вопросе разделимости интериктальной и иктальной активности.

### Ноутбук temporal wavelet correlation graph

Откройте [`sEEG_temporal_wavelet_graph_colab.ipynb`](sEEG_temporal_wavelet_graph_colab.ipynb)
в Google Colab. Он разбивает каждый не-`MKR` канал на последовательные
2-секундные окна, строит для каждого окна разреженный граф NetworkX по
порогово- и top-*k*-обрезанным корреляциям коэффициентов вейвлета `db4`, и
сохраняет всю временную последовательность как тензоры PyTorch.
`KNOWN_EVENT_INTERVAL` — это окно 10396–10398 с, содержащее аннотированный
EDF-пик (`приступ + БТКП` на 10396.445 с, прочитано из файла — см. таблицу
аннотаций выше), а не априорная догадка; граф именно этого окна выделяется
для визуализации, точно повторяя построение co-activation mesh в
`build_seizure_graph` из python-пакета, но на основе вейвлет-корреляции
`db4` вместо корреляции z-оценок 13–80 Гц.

### Агентное обнаружение экстремальных событий

Устанавливаемый пакет `extreme_event_agent` добавляет воспроизводимый
агентный workflow для поиска ранее неизвестных экстремальных событий в
многоканальных временных рядах. Его ограниченный цикл **план → действие →
наблюдение → рефлексия** проверяет качество данных, вызывает
детерминированные инструменты сигнального анализа, адаптирует порог
кандидатов при необходимости, проверяет пространственную поддержку и ведёт
полный аудиторский журнал. Решения по медицинским данным принимают
численные инструменты — не LLM, — поэтому одинаковые входные данные и
конфигурация всегда дают одинаковый результат. Результаты — это кандидаты
для экспертной проверки, а не диагнозы.

Установка и сканирование массива NumPy формы `[channels, samples]`:

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

Детектор объединяет робастные RMS по окну, амплитуду peak-to-peak, длину
линии (line length) и высокочастотную энергию разности. Он формирует
фокальный многоканальный консенсус, использует медианно-MAD-нормировку,
объединяет соседние кандидаты и отбраковывает кандидатов без достаточного
числа независимо вовлечённых каналов. Размер окна, долю каналов, порог
доказательности и пространственную поддержку можно настроить через
`AgentConfig`; не подбирайте их под клиническую метку без отложенной
(held-out) валидации.

#### Анализ и визуализация EDF-записей

CLI теперь умеет обрабатывать один EDF-файл или рекурсивно **все
EDF-файлы** в каталоге. По умолчанию анализируются **обе сигнальные
референции** — родные/референциальные каналы записи и биполярное (по
соседним контактам) перереференцирование — и записывается сравнение, так
что каждый результат ниже всегда сопровождается уже выполненной проверкой
«сохраняется ли это при другой референции» (см. [Биполярный
монтаж](#биполярный-монтаж-и-его-сравнение-с-родной-референцией)). Для
каждой референции монтажа создаётся рисунок обзора всей записи (все
EEG-каналы плюс маркерные каналы `MKR...`), аудируемый отчёт кандидатов,
анализ рекрутирования бета/гамма-диапазона, центрированный на времени
приступа, и — когда этот анализ находит вовлечённые каналы — ещё несколько
рисунков, визуализирующих, как приступ рекрутирует их от начала до пика:
тепловая карта «канал × время», граф-узлы NetworkX в четырёх разных
layout'ах и симуляция message passing, проверяемая против того, что реально
происходило в записи дальше:

```bash
seeg-event-agent dataset/ --output seeg_agent_output
```

`--crop-end-seconds` отбрасывает всё в каждом EDF после указанного времени,
ещё до того, как это увидит любой анализ — используйте это, чтобы исключить
известный нефизиологический «хвост». `sEEG-HFOs-8.edf` требует этого: после
≈10550 с в нём находится хирургическая процедура каутеризации, а не
мозговая активность, которая иначе и загрязняет собственную общерекордную
нормировочную статистику слепого детектора, и сама может стать заявленным
«обнаружением» (см. обсуждение уровня 3 ниже).

```bash
seeg-event-agent dataset/ --output seeg_agent_output --crop-end-seconds 10550
```

`--event-time`/`--event-clock` необязательны. Никакое априорное время
события не требуется, потому что время приступа разрешается через
трёхуровневый приоритет, где каждый уровень используется в откате только
тогда, когда предыдущий недоступен:

1. **`--event-time`/`--event-clock`** — время, введённое экспертом в
   командной строке. Записывается в `analysis.json` как `clinical_annotation`;
   на рисунке рисуется **сплошной малиновой** линией.
2. **Собственный канал аннотаций EDF+ файла** — `find_annotated_event`
   читает все аннотации, уже встроенные в файл (`cp1251`-декодирование;
   декодирование по умолчанию в UTF-8 в MNE вызывает ошибку на кириллическом
   тексте аннотаций), и ищет ту, чей текст называет приступ (`приступ`,
   `судорог`, `seizure`, `ictal`, `бткп`, `tcs`). Каждая другая аннотация в
   пределах 10 с от этого совпадения объединяется с ней, так что отдельные
   заметки клинициста об одном и том же приступе (например, запрос «начало?»
   рядом с меткой «приступ») читаются как одно событие. Это метаданные, уже
   записанные клиницистом в саму запись, а не число, заданное извне.
   Записывается как `annotated_event`; рисуется **сплошной бирюзовой**
   линией.
3. **`select_seizure_event`** — только когда у файла нет такой аннотации,
   основной кандидат выбирается из собственных верифицированных обнаружений
   агента, ранжированных по охвату каналов, затем по длительности, затем по
   оценке, поскольку настоящий приступ рекрутирует гораздо больше контактов
   на гораздо более долгое время, чем короткие высокоамплитудные
   интериктальные спайки, которые иначе набирают такую же оценку.
   Записывается как `detected_event`; рисуется **пунктирной оранжевой**
   линией — единственный маркер, который следует читать как неподтверждённую
   алгоритмическую догадку.

На `sEEG-HFOs-8.edf` это разрешается на уровне 2: аннотация `приступ + БТКП`
на 10396.445 с (см. выше). Запуск анализа процесса бета/гамма на этом
времени воспроизводит клиническую картину исключительно по данным —
`likely_initiators` оказываются `PM3–PM8` и `CC8–CC10` (правая лобная
область), а `later_recruited` охватывает как «штрихованные» (primed), так и
«нештрихованные» контакты `PA`/`SA`/`CR` (билатеральные лобные и теменные)
— без какого-либо априорного времени, входящего в пайплайн. Для сравнения,
слепой откат уровня 3 (протестированный путём временного игнорирования
аннотации на файле, обрезанном до 0–10550 с согласно заметке о гигиене
данных выше) находит 4 группы-кандидата; две сильнейшие расположены на
41.6 с и 126.1 с после аннотированного начала (оценки 24.5 и 25.0, вовлекая
46 и 13 каналов соответственно) — позднее, грубое указание примерно на
нужное место, но не точное и не надёжное по времени. Сканирование же
*необрезанного* файла находит только 2 кандидата, причём ближайший к
аннотации оценивается заметно ниже (16.6 против 24.5 на обрезанном файле)
для тех же каналов примерно в то же время — нефизиологическая амплитуда
сегмента каутеризации одновременно размывает ранг этого кандидата и, при
других конфигурациях детектора, может полностью его перебить (см. сравнение
в ноутбуке по ссылке выше). В любом случае уровень 3 неточен по своей
конструкции: он может лишь примерно подсказать, когда что-то произошло, но
не подтвердить, что это именно приступ. Предпочитайте EDF с реальной
аннотацией или `--event-time`/`--event-clock` откату уровня 3; относитесь к
`detected_event` как к наводке для экспертной проверки, а не как к
результату.

`--event-time` — это секунды от начала каждого EDF; не передавайте время
настенных часов без предварительного пересчёта через время начала EDF — и
убедитесь, что это время действительно синхронизировано с `meas_date` этого
EDF, прежде чем доверять `--event-clock` (см. заметку выше). Модель процесса
использует скользящую энергию 13–80 Гц, робастную преднормировку
median/MAD относительно времени до события и порог рекрутирования в шесть
MAD. Контакты, соответствующие PM3–8 и CC8–10, сообщаются как гипотеза
«правая лобная область» только тогда, когда они пересекают этот
определённый по данным порог первыми; более поздние пересечения описывают
быстрое распространение. Это операционализирует предоставленный клинический
контекст, не относясь к нему как к эталону истины. Результаты остаются
исследовательскими кандидатами, требующими проверки сырого EDF, монтажа,
видео и клинической записи; это не диагноз и не медицинское изделие.

##### Биполярный монтаж и его сравнение с родной референцией

`build_bipolar_montage` читает монтаж прямо из имён каналов, уже
присутствующих в EDF — без отдельной карты электродов, и ему не требуется
разрешённое событие, поэтому `run_edf` всегда записывает его в
`<edf-name>_montage.txt`. Каждое имя разбирается как `<шафт><номер
контакта>` (`parse_contact_name`; например, `"EEG PM3"` → шафт `"PM"`,
контакт `3` — завершающий штрих `'`, как в `"EEG CC'4"`, является частью
метки шафта, поскольку он обозначает отдельный контралатеральный электрод
этого датасета, а не вариант нештрихованного). Внутри каждого шафта контакты
сортируются по номеру и попарно соединяются с числовым соседом —
стандартная биполярная/«соседняя референциальная» деривация для глубинных
электродов — соединяя через любой пропуск в нумерации, а не отбрасывая
связь. На `sEEG-HFOs-8.edf` это даёт 12 шафтов и 88 биполярных пар (`шафт:
число пар`): `R: 9`, `FP: 7`, `FD: 5`, `PM: 7`, `CC: 9`, `SA: 5`, `PA: 9`,
`CC': 9`, `CR': 9`, `PM': 5`, `SA': 7`, `PA': 7` — каждая отображается как,
например, `PM:\n  1-2\n  2-3\n  ...\n  7-8` функцией `format_bipolar_montage`.
Эта структура — `montage`/`montage_file` в `analysis.json` — записывается
безусловно, независимо от того, какая референция реально анализировалась.

`apply_bipolar_montage(data, names, montage)` вычисляет реальные
биполярно-референцированные сигналы (`data[a] - data[b]`, с меткой
`"PM3-4"`): это перереференцирование, а не фильтр, поэтому оно не меняет
то, что измеряет детектор, по своей природе — только то, относительно какой
референции взяты амплитуды — это пространственный фильтр верхних частот,
подавляющий то, что разделяют два соседних контакта (общую референцию
записи, удалённую объёмную проводимость), и сохраняющий только их локальный
градиент.

**Это подключено к `run_edf` как `montage_reference` (`"none"` или
`"bipolar"`), и CLI по умолчанию запускает оба** — `--montages
none,bipolar`, можно переопределить на одно значение, если нужен только
один вариант. Каждый попадает в свой собственный подкаталог
`<edf-name>/<montage_reference>/`, а `compare_montages`/
`summarize_montage_comparison` формируют `montage_comparison.json`
верхнего уровня на запись, сообщая, что реально отличается: число
кандидатов, число вовлечённых каналов, вероятные инициаторы, число рёбер
co-activation mesh и лучшую/среднюю корреляцию валидации message passing.
Само разрешение события не зависит от этого — `find_annotated_event` читает
собственные текстовые аннотации EDF, а не сигнал, — так что оба монтажа
сравниваются при одном и том же времени события.

На `sEEG-HFOs-8.edf` `none` находит 98 вовлечённых каналов и 273 ребра
co-activation против 82 и 227 у `bipolar` — что согласуется с тем, что
биполярный монтаж подавляет часть структуры корреляции общей референции,
которую несёт референциальный монтаж, — при этом лучшая корреляция message
passing с реальной последующей динамикой немного улучшается (0.62 → 0.68).
Оба монтажа сходятся на одних и тех же правых лобных `likely_initiators`
(`PM3–8`/`CC8–10`) — подлинная перекрёстная проверка того, что эта гипотеза
не является артефактом выбора референции; инициаторы биполярного монтажа
записаны как метки пар (`PM2-3`, ..., `CC9-10`), а не отдельных контактов.
(Чтобы биполярные метки пар корректно классифицировались как «правая
лобная область», потребовалось исправить реальный баг: прежнее регулярное
выражение `RIGHT_FRONTAL` проверяло только *первый* номер контакта пары,
так что `PM2-3`/`CC7-8` — у которых в диапазоне именно *второй* конец —
тихо классифицировались неверно. Исправлено как `is_right_frontal`, которая
проверяет любой из концов.)

##### Маркерные каналы `MKR...`

`read_edf` держит `MKR1+`/`MKR2+` вне и статистического детектора, и
анализа процесса бета/гамма: при прямой проверке каждый переход на обоих
каналах отстоит ровно на 0.5 с от предыдущего на протяжении всей записи
длительностью ~3.9 ч, без каких-либо аномалий вокруг любого обнаруженного
или аннотированного события — это аппаратные часы синхронизации, а не
мозговой сигнал или маркер события. Включение такого канала в
`likely_initiators`/`later_recruited` исказило бы аппаратный артефакт,
представив его как нейронное или клиническое доказательство. Тем не менее
они загружаются (через `read_edf_markers`, который читает только эти два
канала) и включаются в рисунок обзора всей записи для визуального/QC-
контекста — проверяющий может убедиться, что часы ведут себя как ожидается,
рядом с реальным сигналом. Если в другом EDF маркерный канал окажется
несущим реальную информацию о событии (нерегулярные переходы, т.е. реальные
нажатия кнопки), проверьте его через `read_edf_markers`, прежде чем
доверять ему как времени приступа.

##### Рисунок эволюции приступа

Когда разрешённое событие даёт `BrainProcess` с вовлечёнными каналами,
`plot_seizure_evolution` строит `<edf-name>_seizure_evolution.png`:
тепловую карту «канал × время» той же медианно-MAD z-оценки энергии
13–80 Гц, которую вычисляет `analyse_brain_process`, ограниченную ровно
теми каналами, которые он уже нашёл вовлечёнными, и упорядоченную по
латентности рекрутирования — инициаторы сверху, позже рекрутированные
контакты снизу, никогда отдельно переотобранный «топ N». Пунктирная линия
отмечает время события; поскольку аннотация уровня 2 помечает 10396.445 с
как момент, когда приступ уже был оценён как *«приступ + билатеральный
тонико-клонический»* (с собственной предшествующей заметкой клинициста «где
начало?» на 10392.734 с), это **пик** каскада, а не обязательно его первое
подёргивание — окно baseline рисунка (по умолчанию 30 с до, 8 с после)
специально показывает нарастание, ведущее к нему, а не только момент
пересечения порога.

На `sEEG-HFOs-8.edf` это показывает то, что похоже на саму
тонико-клоническую фазу: 98 из 100 каналов пересекают порог рекрутирования,
и примерно 90 из них делают это в самом первом окне анализа после пика —
что согласуется с генерализованной, охватывающей весь монтаж ЭЭГ/ЭМГ-
сигнатурой уже установившегося билатерального тонико-клонического приступа,
а не с чистым фокальным каскадом (этот фокальный каскад, если он вообще
виден, был бы в секундах *до* пика, где рисунок показывает гораздо более
разреженную активность). Несколько контактов явно рекрутируются позже —
`EEG PA9` (0.24 с), `EEG PA'3` (0.39 с), `EEG SA'2`/`EEG SA'3` (2.0 с),
`EEG PA'4` (2.1 с), `EEG CR'5` (5.7 с) — и это как раз штрихованные
(левополушарные) контакты, которые слепой детектор уровня 3 независимо
отметил как отдельную позднюю группу кандидатов — небольшая, но подлинная
перекрёстная проверка между двумя независимыми частями этого пайплайна.

##### Граф рекрутирования приступа

`build_seizure_graph`/`plot_seizure_graph` отображают то же рекрутирование
как диаграмму узлов-связей NetworkX вместо тепловой карты:
`<edf-name>_seizure_graph.png`, а также сам граф как
`<edf-name>_seizure_graph.graphml` для повторного использования вне этого
пайплайна. Узлы — это ровно каналы `process.onset_latency_seconds` (то же
правило «без переотобранного топ N», что и у тепловой карты) плюс один
синтетический узел `PEAK`, обозначающий разрешённое событие. Два вида
рёбер, оба измеренные, ни одно не предполагаемое:

* **спицы рекрутирования** — от `PEAK` к каждому каналу, взвешенные тем,
  как скоро после пика он был рекрутирован (это половина «...идёт к
  пику»);
* **сетка co-activation** — корреляция Пирсона собственных временных
  курсов z-оценки 13–80 Гц каналов (тех же, что строит тепловая карта),
  пороговая и top-*k*-обрезанная точно так же, как
  `sEEG_temporal_wavelet_graph_colab.ipynb` обрезает свои графы
  db4-корреляции, так что это повторно использует уже существующую в
  репозитории конвенцию построения графов, а не изобретает новую (половина
  «...как начинается [и] развивается», прочитанная из того, что реально
  совместно варьируется, а не предполагаемый путь распространения).

`plot_seizure_graph` принимает аргумент `layout`, и `run_edf` строит все
четыре через `plot_seizure_graph_layouts`, по одному файлу на каждый
(`<edf-name>_seizure_graph_<layout>.png`) — ни одно расположение не
является «тем самым» графом приступа:

* **`radial`** (исходный, всё ещё по умолчанию) — угол из spring-layout
  только сетки co-activation, радиус — из латентности рекрутирования, так
  что картина читается «снаружи внутрь» как приступ сходится к `PEAK` в
  центре;
* **`spring`** — один стандартный force-directed layout по *всему* графу
  (сетка и спицы рекрутирования вместе), позволяя обоим видам рёбер
  совместно формировать картину, а не только сетке;
* **`circular`** — каналы размещены равномерно по окружности, упорядоченные
  по латентности, простой «циферблат» без какой-либо корреляционной
  структуры вообще — полезный неперегруженный ориентир для остальных трёх;
* **`shell`** — два концентрических кольца, инициаторы внутри и все
  остальные вовлечённые каналы снаружи, выделяя разделение
  инициатор/позже-рекрутированный, которое уже делает
  `analyse_brain_process`, а не латентность как континуум.

Каждый layout по-прежнему рисует инициаторов золотыми ромбами, а остальные
вовлечённые каналы — кружками, окрашенными по пиковой z-оценке. На
`sEEG-HFOs-8.edf`, где ~90 из 98 каналов связаны одной и той же латентностью
первого окна, `radial` упаковывает большинство узлов в одну плотную
внутреннюю дугу — честное следствие отказа от переотбора меньшего «топ N»,
а не ошибка рендеринга — тогда как горстка позже рекрутированных выбросов
(`EEG CR'5`, `EEG SA'1/2/3`, `EEG PA'3/4`) заметно располагается отдельно
от неё на большем радиусе — те же каналы, отмеченные на тепловой карте
выше; три остальных layout'а показывают тот же самый набор узлов с точки
зрения только корреляции, только латентности и разделения по инициаторам
соответственно.

##### Временная динамика message passing

`simulate_message_passing`/`evaluate_message_passing` превращают статический
граф в реальное, проверяемое утверждение о *времени*, а не только о
структуре. Уже измеренное значение `peak_z` каждого вовлечённого канала
(максимальная пост-пиковая z-оценка, найденная для него
`analyse_brain_process`) служит затравкой для одного обновления линейной
диффузии, выполняемого несколько шагов, по той же сетке co-activation,
которая уже есть у графа — `h(t+1) = alpha·h(t) + (1 − alpha)·D⁻¹Wh(t)`,
нормированной по степени, так что ни один канал не накапливает
неограниченную активацию. `evaluate_message_passing` затем пространственно
(между каналами) коррелирует по Пирсону каждый шаг распространения с тем,
как реально выглядели собственные z-оценки записи 13–80 Гц в
соответствующее прошедшее реальное время — это и есть «оценка временной
динамики»: предсказывает ли структура статического графа, распространённая
вперёд, реальное последующее распространение, или нет?

`run_edf` всегда строит эти рисунки, когда процесс нашёл вовлечённые
каналы: `<edf-name>_message_passing.png` (`plot_message_passing`, одна
панель состояния сети на каждый шаг, общая цветовая шкала, layout `spring`
по умолчанию, чтобы слабо связанный канал-выброс не доминировал в масштабе
оси ни одной отдельной панели), та же диффузия, перерисованная в каждом из
четырёх layout'ов `plot_seizure_graph_layouts` через
`plot_message_passing_layouts` (`<edf-name>_message_passing_<layout>.png` —
`radial`/`spring`/`circular`/`shell`, так что распространение от источника
к `PEAK` можно читать относительно латентности, комбинированной структуры,
простого циферблата или разделения по инициаторам — тот же выбор, что уже
предложен для статического графа), и
`<edf-name>_message_passing_validation.png` (`plot_message_passing_validation`,
корреляция в зависимости от прошедшего времени). Каждый рисунок message
passing несёт легенду, идентифицирующую его маркеры/рёбра, и текстовую
подпись в рамке, поясняющую, что на нём показано. Сырые данные
`{"elapsed_seconds": [...], "correlation": [...]}` также записываются в
`analysis.json` как `message_passing_evaluation`.

Наряду с этим, `describe_seizure_source` превращает `process` в текстовое
утверждение о найденном источнике на понятном языке — какой канал(ы) первым
пересёк порог рекрутирования (вероятный источник), в какое абсолютное время
записи, и сколько каналов в итоге оказалось вовлечено — записывается в
`<edf-name>_source_summary.txt` и в `analysis.json` как `source_summary`.
Тот же источник (канал(ы)-инициатор(ы) и абсолютное время) также вынесен
текстовым блоком прямо на рисунок `plot_seizure_graph`, рядом с легендой,
идентифицирующей каждый вид маркера и ребра, и цветовой шкалой для пиковой
z-оценки.

**На `sEEG-HFOs-8.edf` ответ таков: не очень хорошо, и это честный
результат.** Корреляция начинается с 0.62 (шаг 0 не тривиально равен 1.0,
потому что затравка — это *максимальная* пост-пиковая z-оценка каждого
канала, а не буквально её мгновенное значение в `event.time_seconds`),
падает примерно до 0.05–0.35 и остаётся там на протяжении 8 с после пика.
Одна линейная диффузия по графу, построенному в один момент, не
воспроизводит реальную пространственно-временную эволюцию этой записи — что
правдоподобно, учитывая, что генерализующийся билатеральный тонико-
клонический приступ представляет собой гораздо более богатый процесс, чем
может закодировать один статический корреляционный снимок. Относитесь к
этой паре рисунков так же, как ко всему остальному здесь: как к измеренной
проверке упрощённой модели, а не как к демонстрации, что модель работает.

##### Запуск локально в VS Code

1. Откройте этот репозиторий как рабочую область VS Code и поместите
   EDF-файлы в `dataset/` (эта папка намеренно не коммитится).
2. Откройте **Terminal → New Terminal**, создайте изолированное окружение и
   установите проект:

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

3. Выполните **Python: Select Interpreter** из палитры команд и выберите
   интерпретатор `.venv`.
4. Откройте **Run and Debug** (`Ctrl+Shift+D`), выберите **Analyse all EDF
   files** и нажмите `F5`. Запуск идёт без экспертного времени события,
   поэтому время приступа берётся из уровня 2 или 3 выше (собственный канал
   аннотаций EDF или, в крайнем случае, слепой детектор); добавьте
   `--event-time <seconds>` в [`.vscode/launch.json`](.vscode/launch.json),
   если у вас есть независимо подтверждённый офсет и вы хотите, чтобы
   именно он управлял анализом процесса. По умолчанию анализируются обе
   референции монтажа (примерно вдвое дольше, чем одна, поскольку это
   полный прогон пайплайна дважды); добавьте туда `--montages bipolar`,
   если нужна только одна.
5. Либо выберите **Analyse EDF by clinical clock**, чтобы ввести путь к EDF
   и клиническое время настенных часов. `--event-clock` читает время начала
   из каждого EDF, обрабатывает переход через полночь и отклоняет времена
   вне записи. Используйте это только тогда, когда вы независимо
   подтвердили, что клинические часы и часы EDF синхронизированы — для
   входящего в комплект `sEEG-HFOs-8.edf` это не так (см. заметку в начале
   этого файла), поэтому для него используйте вместо этого **Analyse all
   EDF files**.

Эквивалентные команды терминала:

```bash
# Без априорного времени события: собственный канал аннотаций EDF разрешает его,
# если он есть (уровень 2), иначе слепой детектор (уровень 3). Запускает обе
# референции монтажа (по умолчанию --montages none,bipolar) и пишет сравнение.
seeg-event-agent dataset --output seeg_agent_output

# Только одна референция монтажа, если не хотите платить за обе
seeg-event-agent dataset --output seeg_agent_output --montages bipolar

# Известный офсет от начала EDF
seeg-event-agent dataset --output seeg_agent_output --event-time <seconds>

# Известное клиническое время настенных часов, только если синхронизировано с meas_date этого EDF
seeg-event-agent dataset/sEEG-HFOs-8.edf --output seeg_agent_output \
  --event-clock <HH:MM:SS>
```

Каждая запись записывается в свой каталог `seeg_agent_output/<edf-name>/`, с
одним подкаталогом на каждую реально запущенную референцию монтажа
(`none/`, `bipolar/` или обе):

* `<edf-name>/montage_comparison.json` (только если было запущено более
  одного монтажа) — это таблица `summarize_montage_comparison` по монтажам:
  число кандидатов, число вовлечённых каналов, вероятные инициаторы, число
  рёбер co-activation, лучшая/средняя корреляция message passing;
* `<edf-name>/<montage_reference>/analysis.json` содержит обнаруженных
  кандидатов, саму `montage_reference`, `montage` (группировку биполярных
  пар шафт/контакт, см. ниже — всегда описывает референциальную структуру,
  независимо от того, какая референция анализировалась) и `montage_file`, а
  также то, какой уровень события разрешился: `clinical_annotation`
  (уровень 1, `--event-time`/`--event-clock`), `annotated_event` (уровень 2,
  собственный канал аннотаций EDF — включает весь совпавший кластер
  аннотаций для аудита, и идентичен между монтажами, поскольку берётся из
  текстовых аннотаций файла, а не из сигнала) или `detected_event` (уровень
  3, слепой откат, который *действительно* меняется в зависимости от
  монтажа) — плюс оценки бета/гамма по каналам, латентности рекрутирования,
  гипотезу процесса «правая лобная область», вычисленную на основе того,
  что использовалось, `evolution_figure`, `graph_figures` (по одному пути на
  layout) и `graph_graphml`, `message_passing_figure`/`message_passing_figures`
  (по одному пути на layout)/`message_passing_validation_figure`/
  `message_passing_evaluation`, и `source_summary`/`source_summary_file`
  (текстовое утверждение источника от `describe_seizure_source` — все
  `null`/пусты, если вовлечённых каналов не было);
* `<edf-name>/<montage_reference>/<edf-name>_montage.txt` — биполярный
  монтаж (см. ниже), записывается безусловно — ему не требуется разрешённое
  событие;
* `<edf-name>/<montage_reference>/<edf-name>_all_timeseries.png` содержит
  все анализируемые каналы плюс маркеры `MKR...` по всей записи и маркер
  события — сплошной малиновый для `clinical_annotation`, сплошной
  бирюзовый для `annotated_event`, пунктирный оранжевый для `detected_event`;
* `<edf-name>/<montage_reference>/<edf-name>_seizure_evolution.png` (только
  если `brain_process` нашёл вовлечённые каналы) — тепловая карта каскада
  рекрутирования, описанная выше;
* `<edf-name>/<montage_reference>/<edf-name>_seizure_graph_<layout>.png`
  (`radial`/`spring`/`circular`/`shell`, то же условие) — отрисовки
  узлов-связей, а `..._seizure_graph.graphml` — сам граф NetworkX;
* `<edf-name>/<montage_reference>/<edf-name>_message_passing.png`,
  `..._message_passing_<layout>.png` (`radial`/`spring`/`circular`/`shell`)
  и `..._message_passing_validation.png` (то же условие) — панели симуляции
  диффузии в каждом layout и график её валидации против реальности;
* `<edf-name>/<montage_reference>/<edf-name>_source_summary.txt` (то же
  условие) — текстовое утверждение `describe_seizure_source` о найденном
  канале(ах)-источнике и времени.

Если VS Code сообщает `No module named extreme_event_agent`, убедитесь, что
`.venv` выбран как интерпретатор, и повторите `python -m pip install -e .`
в терминале VS Code.

### Модель на основе reservoir computing (`model/`)

`model/` добавляет второй, независимый взгляд на тот же приступ: не
детектор, а **плант** — в смысле теории автоматического управления. Echo
State Network (классическая архитектура reservoir computing) собрана как
явная нелинейная дискретная state-space система:

```
x(t) = (1 - leak) x(t-1) + leak * tanh(B u(t) + A x(t-1) + bias)   # уравнение состояния
y(t) = C x(t) + D u(t)                                             # уравнение выхода
```

с буквальным разделением вход/выход, взятым прямо из EDF: аппаратные каналы
часов `MKR...` (везде в остальном репозитории исключаемые из детекции,
поскольку не несут мозгового сигнала) — экзогенный вход планта `u(t)`;
подмножество реальных EEG-каналов — те же правые лобные `likely_initiators`
плюс разброс каналов `later_recruited`, уже найденных
`analyse_brain_process` — наблюдаемый выход `y(t)`. `A` (резервуар) и `B` —
фиксированные случайные матрицы, сгенерированные один раз; обучается только
линейный readout (`C`, `D`), гребневой регрессией, **только на baseline до
события** — весь фокус reservoir computing в том, что одной дешёвой
линейной подгонки поверх большой фиксированной нелинейной динамической
системы достаточно, чтобы захватить богатую временную структуру.

Поскольку сами часы почти постоянны между импульсами 1 Гц и несут почти
никакой информации, коррелирующей с быстрой структурой ЭЭГ, `u(t)`
дополняется коротким delay-embedding'ом собственного недавнего, уже
наблюдённого прошлого целевого сигнала (`y(t-1), ..., y(t-lag)`) —
стандартное NARX-расширение («нелинейная авторегрессия с экзогенным
входом») для reservoir computing, та же идея delay embedding, которую
использует референсная реализация этого проекта
([D2718281828nis/ML-Reservoir_Computing](https://github.com/D2718281828nis/ML-Reservoir_Computing))
в своём ноутбуке next-generation RC. Это никогда не «подглядывает» сам
`y(t)`, только строго более ранние отсчёты, так что получающийся прогноз на
один шаг вперёд — легитимное, проверяемое предсказание прямо через
экстремальное событие, а не только на baseline, на котором он был обучен.

Затем `run_reservoir_plant` запускает *ту же* модель, обученную только на
baseline, вперёд по всему окну (и baseline, и событие) и измеряет невязку
(residual) между тем, что она предсказывает, и тем, что реально произошло в
записи — классический сигнал ошибки наблюдателя (observer-residual) для
обнаружения неисправностей/аномалий из теории управления, используемый
здесь как собственная, независимая оценка экстремального события этой
модели: медианно-MAD z-оценка величины невязки, с порогом в 6 MAD
(совпадающим с собственным порогом рекрутирования `analyse_brain_process`)
после короткого сглаживания скользящим средним, чтобы один шумный отсчёт
сам по себе не мог засчитаться как «обнаружение».

Запуск (требуется вызов как модуля — из-за относительных импортов):

```bash
python -m model.run_model dataset/sEEG-HFOs-8.edf --output model_result
```

На `sEEG-HFOs-8.edf` это разрешает ту же аннотацию уровня 2 (`приступ +
БТКП` на 10396.445 с), что и `run_edf`, обучается на 60-секундном baseline
до события и оценивает 20 с после него: RMSE readout на baseline получается
порядка 4–9×10⁻⁵ на канал (NARX-дополненное предсказание заметно
отслеживает реальную форму волны, а не только её среднее — см.
`..._output_prediction.png`), а невязочная оценка достигает пика ~70+ MAD
через несколько секунд после аннотированного пика — устойчивое, независимо
построенное подтверждение того же события совершенно другим механизмом
(невязка предсказания, а не пространственное рекрутирование), чем остальная
часть этого репозитория.

`model_result/<edf-name>_*.png` (семь рисунков за запуск):

* `_architecture.png` — блок-схема state-space (`u → B,A → x → C,D → y`) с
  реальными размерностями этого конкретного запуска;
* `_connectivity.png` — собственный случайный рекуррентный граф резервуара
  (читаемая выборка скрытых юнитов плюс узлы экзогенного входа/выхода и их
  сильнейшие рёбра, с легендой);
* `_spectrum.png` — собственные значения `A` относительно единичной
  окружности (картина echo-state property);
* `_hidden_state.png` — активация каждого юнита резервуара во времени;
* `_output_prediction.png` — реальный выход против предсказанного обученной
  на baseline моделью, по каждому каналу;
* `_residual_heatmap.png` — та же разница «реальное минус предсказанное» в
  виде одной тепловой карты «канал × момент времени» (каждый канал
  независимо нормирован median/MAD относительно своего собственного
  baseline, так что каналы с очень разной собственной амплитудой всё равно
  визуально сопоставимы);
* `_extreme_event_score.png` — агрегированная невязочная оценка
  относительно порога, с отмеченными собственными началом/пиком модели.

`<edf-name>_model_summary.txt`/`_model_result.json` несут те же числа в виде
текста/JSON — конфигурацию резервуара, RMSE обучения по каждому каналу и
вердикт по экстремальному событию (`describe_evaluation`, аналог
`describe_seizure_source` для этой модели). Как и всё остальное в этом
репозитории, результат «обнаружено» здесь — кандидат для экспертной
проверки: он сверяется с собственной детекцией `extreme_event_agent`,
основанной на пространственном рекрутировании, а не считается более
авторитетным, чем она.
