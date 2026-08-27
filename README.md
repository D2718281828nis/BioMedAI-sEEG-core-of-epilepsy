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
time does not *discover* a localization from the signal — it checks whether
the contacts an external clinical context already named (PM3–8, CC8–10)
crossed a data-derived recruitment threshold within a data-derived window of
the globally earliest crossing across the whole montage. On this file they
do: `likely_initiators` comes out as `PM3–PM8` and `CC8–CC10`, and
`later_recruited` spans both primed and unprimed `PA`/`SA`/`CR` contacts
(bilateral frontal and parietal). What *is* derived from data alone, with no
apriori contact list involved at any point, is every recruitment latency,
the single globally earliest crossing (`τmin` — 0.035 s after the annotated
peak on this recording), and which contacts tie for it. See [How
`likely_initiators` is computed](#how-likely_initiators-is-computed) below
for the exact rule, a defect the earlier two-branch version of this had, and
this file's own `earliest_contacts`/`hemisphere_of_earliest` — the prior-free
reading of the same recording. By contrast, tier 3's blind fallback (tested by temporarily ignoring the annotation, on
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

##### How `likely_initiators` is computed

`likely_initiators` is the one output this whole pipeline is built around,
and it mixes two things that must stay visibly distinct: a **prior**
(`ContactPrior`) supplied from outside the recording, and a computation the
signal alone determines. Both `analyse_brain_process` (`edf_workflow.py`)
and `BrainProcess` (`models.py`) keep them as separate fields rather than
merging them into one number silently.

1. Sliding 250 ms energy in the 13–80 Hz band, per channel
   (`_beta_gamma_z_scores`).
2. Robust median/MAD normalization against each channel's own pre-event
   baseline.
3. Recruitment latency `τc` — the first post-event window where a channel's
   z-score exceeds `RECRUITMENT_THRESHOLD_MAD` (6 MAD).
4. `τmin = minc τc` — the single earliest crossing **across every channel**,
   computed with no reference to any contact list at all.
5. Classification, relative to `τmin`, into three mutually exclusive,
   jointly exhaustive categories:
   - `earliest = {c : τc ≤ τmin + 0.05s}` (`SIMULTANEITY_WINDOW_SECONDS`) —
     purely data-derived, whether or not the prior names a contact;
   - `prior_early = {c ∈ prior : τc ≤ τmin + 0.25s} \ earliest`
     (`PRIOR_WINDOW_SECONDS`) — a wider window that only ever applies to
     contacts the prior already names;
   - everything else with a measured latency is `later_recruited`.
6. Result:

   ```
   likely_initiators = prior_early ∪ (earliest ∩ prior)
   ```

   where, on this dataset, `prior` is `{PM3, ..., PM8, CC8, CC9, CC10}`
   (`SEEG_HFOS_8_CLINICAL_PRIOR`, `edf_workflow.py`). Passing `prior=None` to
   `analyse_brain_process` disables all of this — `likely_initiators`
   reduces to `earliest` and `initiators_constrained_by_prior` is always
   `False` — for a region-agnostic reading of the same recording.

**Derived from data:** every `τc`, `τmin`, whether a contact crosses
threshold at all, and the composition of `earliest`. **Set externally:**
the prior's contact list itself, the two windows (0.05 s, 0.25 s), and the
6 MAD threshold — none of these three numbers changed in this revision;
what changed is that they are now named constants
(`SIMULTANEITY_WINDOW_SECONDS`, `PRIOR_WINDOW_SECONDS`,
`RECRUITMENT_THRESHOLD_MAD`) instead of bare literals.

**The blind spot, and why `earliest_contacts` is now published.** In the
prior-constrained branch, only contacts the prior names can ever become
`likely_initiators` — so if the globally earliest crossing belonged to a
contact *outside* the prior, it used to be reported as neither an initiator
(not in the prior) nor later-recruited (its latency was too close to
`τmin` for the old `> τmin + 0.05s` later-recruited test to be true either)
— it silently disappeared from both tuples, worst of all for the one
contact whose timing could have contradicted the prior. `BrainProcess` now
separately publishes `earliest_contacts` (the prior-free earliest set) and
`hemisphere_of_earliest`, precisely so this can be checked rather than taken
on trust, and `analyse_brain_process` raises `ValueError` if its own
three-way partition of measured latencies is ever not exact.

On `sEEG-HFOs-8.edf` (tier-2 event, native reference), `τmin = 0.035s` after
the annotated peak, and **92 of the 98 involved channels tie for it** — this
seizure's peak recruits nearly the whole montage within one 50 ms window,
not a handful of focal contacts. Of those 92, only 9
(`prior_fraction_among_earliest ≈ 0.098`) are contacts the clinical prior
names; the rest span both hemispheres (57 right/35 left), so
`hemisphere_of_earliest` reads **`"mixed"`**, not `"right"`.
`initiators_constrained_by_prior` is `False` here — the prior's wider 0.25 s
window never had to reach past the tie to find PM3–8/CC8–10, because they
were already inside it. The honest reading is *not* "the data independently
confirms a right-frontal source"; it is "PM3–8/CC8–10 are among the ~94% of
this montage that crossed threshold together at the seizure's peak, and it
is the prior's contact list, not the recruitment timing, that singles them
out as `likely_initiators`." A different recording, or this one examined
earlier in the cascade rather than at the annotated peak, could easily
produce `hemisphere_of_earliest = "right"` or `"left"` instead — nothing in
this rule is tuned toward either answer.

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
`likely_initiators` (`PM3–8`/`CC8–10`, written as bipolar pair labels
`PM2-3`, ..., `CC9-10` under `bipolar`) — but that agreement is partly
guaranteed by construction, not purely a cross-check: both runs test the
*same* prior contact list against their own data-derived threshold, so
matching `likely_initiators` mainly shows that those prior contacts crossed
threshold within the prior window under both references, which is weaker
than an independent method converging on them. The check that owes nothing
to the prior is `earliest_contacts` (see [How `likely_initiators` is
computed](#how-likely_initiators-is-computed) above): on this recording both
montages' earliest sets are large — 92 of 98 channels for `none`, 76 of 82
for `bipolar` — and mixed-hemisphere in the same rough proportion (57
right/35 left vs. 49 right/27 left), so `hemisphere_of_earliest` reads
`"mixed"` under both references. *That* agreement — not the
`likely_initiators` match — is the genuine cross-check that reference choice
isn't driving the result. (Getting bipolar pair labels correctly matched
against the prior required fixing a real bug: the previous `RIGHT_FRONTAL`
regex only checked a pair's *first* contact number, so `PM2-3`/`CC7-8` —
whose *second* endpoint is the one in range — were silently misclassified.
Fixed as `is_right_frontal`, now a thin wrapper over the general
`prior_matches`, which checks either endpoint.)

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
already found involved and ordered by their recruitment latency, earliest at
the top — this row order comes from the data alone, never a separately
re-picked "top N" and never moved by the prior. A dashed line marks the
event time; since tier 2's annotation labels 10396.445 s as the point the
seizure was scored as already *"seizure + bilateral tonic-clonic"* (with the
clinician's own preceding "where does it start?" note at 10392.734 s), this
is the cascade's **peak**, not necessarily its first twitch — the figure's
baseline window (30 s before, 8 s after, by default) is there specifically
to show the build-up leading into it, not just the instant of crossing. On
top of that data-only ordering, the figure draws the prior and the
classification rule as two visually separate layers (see [How
`likely_initiators` is computed](#how-likely_initiators-is-computed)): a ◆
before a row's label marks a contact the clinical prior names
(`process.prior_matched`) without moving that row; `τmin`, the simultaneity
window, and the wider prior window are drawn on the time axis and labelled;
a small "×" marks each row's own measured crossing; and every row in
`process.earliest_contacts` is bolded — gold if the prior also names it,
green if not, so the contact that would contradict the prior (if the data
ever produced one) is visually impossible to miss.

On `sEEG-HFOs-8.edf` this shows what looks like the tonic-clonic phase
itself: 98 of 100 channels cross the recruitment threshold, and 92 of them
tie for the globally earliest crossing (`τmin`, 0.035 s after the peak) —
consistent with the generalized, whole-montage EEG/EMG signature of an
established bilateral tonic-clonic seizure rather than a clean focal cascade
(that focal cascade, if visible at all, would be in the seconds *before* the
peak, where the figure shows much sparser activity). Most of the figure's
rows are therefore bold; only 9 of those 92 are gold (prior-named), the rest
green — the same 92-channel, mixed-hemisphere tie the blind-spot discussion
above quantifies. A handful of contacts are distinctly recruited later —
`EEG PA9` (0.24 s), `EEG PA'3` (0.39 s), `EEG SA'2`/`EEG SA'3` (2.0 s), `EEG
PA'4` (2.1 s), `EEG CR'5` (5.7 s) — and these are exactly the primed
(left-hemisphere) contacts the tier-3 blind detector independently flagged
as a separate late candidate group, a small but genuine cross-check between
two unrelated parts of this pipeline.

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

Every layout draws each channel node with three independent encodings, kept
visually separate rather than merged: **fill colour = role**
(crimson/orange/blue for `earliest`/`prior_early`/`later_recruited` — the
same data-derived, prior-independent-then-prior-widened partition [How
`likely_initiators` is computed](#how-likely_initiators-is-computed)
describes), **ring colour = prior membership** (gold ring when the node's
`in_prior` attribute is true, thin grey otherwise — externally supplied),
and **size = peak z-score**. A node whose fill and ring agree (a
crimson/orange fill with a gold ring, or a blue fill with a grey ring) is
where the data-derived role and the external prior line up; a mismatch is
exactly where they don't — the most informative thing this figure can show,
which is why the two encodings are never folded into one colour. All four
node attributes (`role`, `in_prior`, `latency_seconds`, `hemisphere`) are
written into `<edf-name>_seizure_graph.graphml`, so the same check can be
run outside this pipeline (e.g. `networkx.read_graphml`). On
`sEEG-HFOs-8.edf`, with 92 of 98 channels tied at the same first-window
latency (`τmin`, see above), `radial` packs most nodes — crimson, only 9 with
a gold ring — into one dense inner arc; an honest consequence of not
re-picking a smaller "top N", not a rendering bug — while the handful of
later-recruited (blue) outliers (`EEG CR'5`, `EEG SA'1/2/3`, `EEG PA'3/4`)
visibly sit apart from it at larger radius, the same channels flagged in the
heatmap above; the other three layouts show the identical node set from
correlation-only, latency-only, and role-split perspectives instead.

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
  recruitment latencies, and `likely_initiators` under `brain_process`
  (which also carries `earliest_contacts`, `earliest_latency_seconds`,
  `prior_matched`, `prior_source`, `initiators_constrained_by_prior`, and
  `prior_fraction_among_earliest`/`hemisphere_of_earliest` — see [How
  `likely_initiators` is computed](#how-likely_initiators-is-computed)),
  `evolution_figure`, `graph_figures` (one path per
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

`model_result/<edf-name>_*.png` (nine figures per run):

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
* `_residual_timeseries.png` — `evaluation.residual` itself (`model.visualize.plot_residual_timeseries`),
  per channel, as small-multiples time series in the recording's own
  physical units (volts) rather than the heatmap's per-channel z-score —
  how large the prediction error actually is, not how surprising it is
  relative to that channel's own baseline noise;
* `_baseline_vs_event_accuracy.png` — `model.visualize.plot_baseline_vs_event_accuracy`,
  a bar chart of baseline-fit RMSE vs. post-event RMSE per channel plus
  overall, each pair annotated with its ratio — the figure form of
  "evaluate the model's accuracy by comparing baseline against the event"
  (see `model.visualize.compute_baseline_vs_event_rmse`, also written to
  `_model_result.json` as `baseline_vs_event_rmse`);
* `_extreme_event_score.png` — the aggregated residual score vs. threshold,
  with the model's own onset/peak marked.

On `sEEG-HFOs-8.edf` (default config), `compute_baseline_vs_event_rmse` finds
every one of the 12 output channels degrading after the event with no
exceptions — baseline RMSE 3.7–9.1×10⁻⁵ V rising to 7.2–17.8×10⁻⁵ V, ratios
1.4×–3.4× (`PA9` worst at 3.4×, `CC10` least at 1.4×), overall RMS ratio
**2.3×**. `EEG PA9`'s residual trace shows a visibly blocky/staircase
pattern rather than continuous EEG texture in `_residual_timeseries.png` —
worth checking that channel's raw trace for amplifier saturation/clipping
before reading its residual (the largest ratio of the twelve) as purely
physiological.

`<edf-name>_model_summary.txt`/`_model_result.json` carry the same numbers
as plain text/JSON — reservoir configuration, per-channel training RMSE, and
the extreme-event verdict (`describe_evaluation`, this model's counterpart
to `describe_seizure_source`). As with everything else in this repository, a
"detected" result here is a candidate for expert review — checked against,
not treated as more authoritative than, `extreme_event_agent`'s own
spatial-recruitment-based detection.

##### Reservoir stability spectrum (`_spectrum.png`)

`model.visualize.plot_reservoir_spectrum` plots the eigenvalues of the
reservoir's own recurrent weight matrix `A` (`W` in code) — one dot per
eigenvalue of the `n_reservoir × n_reservoir` fixed random matrix that
drives the state equation `x(t) = (1-leak)·x(t-1) + leak·tanh(B·u(t) +
A·x(t-1) + bias)`. `A` is generated once, randomly, at construction and
never trained (only the readout `C`/`D` is fit by ridge regression — that
is the whole reservoir-computing trick). This figure is a **diagnostic of
that fixed system's stability**, not of how well the model fits this
recording — it says nothing about `y(t)` or the event at all.

Each dot is plotted in the complex plane (real part on x, imaginary part on
y) against a dashed unit circle (radius 1); the achieved spectral radius
(the magnitude of `A`'s largest eigenvalue, rescaled at construction to
land at `--spectral-radius`, default 0.95) is stated in the title. This is
the standard **echo-state property** check: a recursively-driven system
like this one is only guaranteed to forget its arbitrary initial condition
`x(0) = 0` and settle into a trajectory driven purely by the real input —
the property the whole readout-fitting approach depends on — when every
eigenvalue sits strictly inside the unit circle (spectral radius < 1). On
`sEEG-HFOs-8.edf` (default config, `n_reservoir=400`) all 400 eigenvalues
sit inside the dashed circle, confirming the reservoir was actually built
as configured and is contracting, not just claimed to be. Pushing
`--spectral-radius` to or past 1 (tested in the hyperparameter sweep above)
moves points outside or onto the circle; in that regime stability no longer
comes from `A`'s own contraction but depends entirely on `leak_rate`'s
explicit integration term — the caption on the figure states this directly.
Read it alongside `_architecture.png` (what the system *is*) and
`_connectivity.png` (what it looks like as a network); `_output_prediction.png`,
`_residual_heatmap.png`, `_residual_timeseries.png`, and
`_baseline_vs_event_accuracy.png` are where fit quality is actually judged.

##### Tuning the reservoir for a tighter baseline fit

Every reservoir hyperparameter is already a CLI flag
(`--n-reservoir`/`--spectral-radius`/`--leak-rate`/`--ridge-alpha`/
`--output-feedback-lag`), so improving the plant's fit needs no code
change. A one-parameter-at-a-time sweep against `sEEG-HFOs-8.edf`'s default
window (holding the others at their CLI defaults) found:

| parameter | direction that helps baseline RMSE | effect size |
|---|---|---|
| `n_reservoir` | bigger (200→1000) | 6.68×10⁻⁵ → 4.51×10⁻⁵ (mostly diminishing returns past ~800) |
| `spectral_radius` | almost no effect (0.5→1.05) | 5.57×10⁻⁵ → 5.67×10⁻⁵ — the NARX delay embedding, not the reservoir's own recurrent dynamics, carries most of the signal here |
| `leak_rate` | higher/less-leaky (0.05→1.0) | 1.04×10⁻⁴ → 3.48×10⁻⁵ — the single strongest lever |
| `ridge_alpha` | lower regularization (1→1e-5) | 1.37×10⁻⁴ → 1.31×10⁻⁵ — also strong, but the least-regularized end risks fitting baseline noise rather than signal |
| `output_feedback_lag` | weak, mildly better higher (1→20) | 6.16×10⁻⁵ → 5.56×10⁻⁵ |

This is a legitimate target to optimize — it is baseline-only fit quality,
never touching the event window or its label, the same "never tune against
a known answer" boundary `MANIFEST.md` already states for detection
thresholds elsewhere in this repo. Picking a moderate point on this
frontier rather than the sweep's extreme (to avoid leak_rate=1.0, which
zeroes the state equation's own recurrent memory term and reduces the ESN
to a memoryless nonlinear map of the NARX input, and to avoid pushing
ridge_alpha down to the noise-fitting end) —

```bash
python -m model.run_model dataset/sEEG-HFOs-8.edf --output model_result_tuned \
  --n-reservoir 800 --spectral-radius 0.95 --leak-rate 0.7 \
  --ridge-alpha 1e-3 --output-feedback-lag 10
```

— cuts overall baseline RMSE from 1.99×10⁻⁴ V (default) to **1.87×10⁻⁵ V**
(≈3× tighter fit) and, checked rather than assumed, the *event* behaviour
improves in the same direction rather than being traded away: overall
baseline-vs-event RMSE ratio rises from 2.3× to **3.0×**, peak score rises
from 73.5 MAD to **153.5 MAD**, and the fraction of post-event samples over
the 6-MAD threshold rises from 20.7% to **33.0%** — a better-fit plant is
also a more sensitive one here, not merely a better-looking baseline trace.
The pre-event −47…−48 s transient (see above) stays at essentially the same
time under every configuration tested, further evidence it is a real
feature of the recording rather than an artifact of one particular
hyperparameter choice.

**`--channel-selection balanced`.** By default (`recruitment`) the reservoir's
output channels `y(t)` come straight from `analyse_brain_process`'s own
`likely_initiators`/`later_recruited` — useful for checking whether the
reservoir's *residual* independently flags the same event, but *not* a valid
input for cross-checking that analysis's own lateralization, since the
channels themselves already came from it. `--channel-selection balanced`
picks channels a different way instead: evenly split across
`hemisphere_of_channel`, ranked within each half by pre-event-only variance —
never latency, never recruitment, never anything from after the event.
`ReservoirWindow.arbitration_valid` is `True` only in this mode; every
lateralization estimate built from it downstream (see below) carries this
flag so it can never be silently mistaken for an independent confirmation
when it isn't. `run_reservoir_plant` also now scores each output channel's
residual *independently* (`per_channel_score`/`per_channel_onset_seconds`/
`per_channel_peak_score`, median/MAD-normalized per channel against its own
baseline, the same way the scalar `score` already is) instead of only the
whole-window collapsed scalar — a spatial read of *where* the model's
prediction breaks down first, not only *whether* it does.

### Object model: combining EDF, DICOM, and the reservoir (`object_model/`)

The three pieces above each measure this recording differently, and each
can only give part of the picture:

| Source | Temporal resolution | Spatial resolution |
|---|---|---|
| EDF (`extreme_event_agent`) | fractions of a second | down to a single contact |
| DICOM (`multimodal_approach`) | **none — a static post-implant scan has no time axis** | hemisphere only (no verified per-contact 3-D electrode localization exists here — see `multimodal_approach/README.md`, "Honest limits") |
| Reservoir (`model/`, `channel_selection="balanced"`) | fractions of a second | down to a single output channel, if `per_channel_score` isn't collapsed |

`object_model/` is the one package that imports all three of the others
together (each of them stays free of a dependency on either sibling) and
does two things with that: **verify** every method against the one ground
truth this recording has (its own EDF+ annotation), and **assemble** the
three evidence sources onto one graph without ever merging them into a
single score.

**`extreme_event_agent.verification.verify_against_annotation`** scores:

- *Temporal accuracy* — signed `delta_seconds = method_time − t_БТКП`
  (10396.445 s), never `abs()`'d before storage, so a method that fires
  early and one that fires late by the same amount are kept distinguishable.
  Banded into `precise` (≤1 s — tighter than the ~6.7 s spread between this
  recording's own earliest and latest annotation of the same seizure),
  `coarse` (≤10 s — the ictal phase), `window` (≤60 s — the event as a
  whole), or `miss`. Exactly two live methods exist in this installable
  package: `t_targeted` (`analyse_brain_process`'s earliest crossing) and
  `t_blind` (`ExtremeEventAgent`'s own tier-3 pick) — the ≈+39.6 s
  "broadband ensemble" figure elsewhere in this README is
  `sEEG_extreme_event_detector_colab.ipynb`'s five-method ensemble, which
  `MANIFEST.md` already documents as outside the installable package, so it
  is not fabricated here as a third live method.
- *Lateralization* — `LI = (v_right − v_left) / (v_right + v_left) ∈ [-1, 1]`,
  each `v` normalized by its own hemisphere's channel/voxel count first, so
  sources with very different counts are still comparable: `edf_earliest_contacts`
  (rate of `process.earliest_contacts` per hemisphere — prior-free, unlike
  `likely_initiators`, which by definition can never disagree with the
  prior's own side), `dicom_mean_abs_anomaly` (straight off `hemisphere_summary`),
  and, when a reservoir evaluation is given, `reservoir_residual_strength`/
  `reservoir_residual_earliness`. `|LI| < 0.05` reads as `"indeterminate"`,
  never forced to a side.
- *Contact overlap* — precision/recall/Jaccard of `earliest_contacts` (data)
  against `prior_matched` (external prior) — whether the data supports the
  clinical hypothesis, not a localization claim.

Every `VerificationReport` carries `crop_applied`/`channel_selection`/
`masking_method`/`prior_used` — the context every number above depends on,
never silently omitted, written to `verification_report.json`.

**`object_model.graph.build_object_model_graph`** takes an existing
`build_seizure_graph` result and adds two more attribute groups per channel
node: structural (`hemisphere_anomaly_mean`/`hemisphere_anomaly_max`, from
DICOM, keyed by the node's own hemisphere) and model
(`residual_onset_seconds`/`residual_peak_score`, only for nodes that are
also reservoir output channels — absent, never `None`, for the rest, since
GraphML has no null type). Three separately-named layers, never averaged
into one score — `structural_anomaly.py` already keeps `combined_anomaly`/
`combined_heterogeneity` separate on the same principle, and a node whose
EDF role and structural/reservoir evidence *disagree* is exactly the case
merging would hide.

Run it:

```bash
python -m object_model.run_object_model --edf dataset/sEEG-HFOs-8.edf \
  --dicom-dir dataset/MRI-with-electrodes/DCM --crop-end-seconds 10550 \
  --channel-selection balanced --output object_model_result
```

writes, to `object_model_result/<edf-name>/`: `verification_report.json`,
`object_model_graph.graphml` (three-layer node attributes, only when the EDF
process found involved channels), and `object_model_summary.png` — one
figure, five panels: the EDF recruitment cascade (row order from data,
prior-named contacts marked, never moved), the object-model graph
(fill = role, ring = prior, shape = hemisphere, size = peak z), a DICOM
slice through the strongest structural cluster, the reservoir's per-channel
residual (sorted by onset, washout shaded), and the verification summary
(Δt per method with tolerance bands; LI per source with the indeterminate
band shaded) — with the channel-selection mode, crop status, masking
method, and research-candidate status line captioned on every render.

On `sEEG-HFOs-8.edf` (`--crop-end-seconds 10550 --channel-selection
balanced`), a genuinely mixed result: `t_targeted` lands `precise`
(+0.035 s), `t_blind` lands `window` (+41.6 s, consistent with the tier-3
discussion above). The four lateralization sources do **not** agree:
`edf_earliest_contacts` reads barely `right` (LI ≈ +0.08 — 92 of 98
channels tie for earliest, see "How `likely_initiators` is computed"
above, so this is a weak signal, not a confident one), `dicom_mean_abs_anomaly`
reads `left` (LI ≈ −0.11), `reservoir_residual_strength` reads `right`
(LI ≈ +0.53), and `reservoir_residual_earliness` reads `left` (LI ≈ −1.0,
driven by output channels that only ever crossed threshold during the
pre-event baseline on the right side). Reported exactly as disagreement,
not resolved into a single answer — see [What to look at in the
result](#what-to-look-at-in-the-result) below for how to read this.

##### What to look at in the result

No outcome below is preferred in advance — the point of separating these
numbers is that the answer is visible and checkable, not that it comes out
any particular way.

1. **`hemisphere_of_earliest`** (from `analyse_brain_process`, prior-free —
   see "How `likely_initiators` is computed" above): which contact fired
   first with no contact list involved at all. `"right"` would confirm the
   clinical prior with genuinely independent data; `"left"` would flip the
   picture and line up with the structural finding instead; on this
   recording it is `"mixed"` — 92 of 98 channels tie for first, so the
   temporal channel does not resolve laterality by itself at all, and every
   `LI` above should be read with that in mind.
2. **The reservoir's Δt** against the ≈+39.6 s broadband ensemble figure
   discussed earlier in this README: a materially smaller value would mean
   a model-based criterion finds the transition earlier than a purely
   statistical one — a direct probe of where blind statistical detection's
   own limits sit, not assumed one way or the other.
3. **`reservoir_residual_strength`/`reservoir_residual_earliness` LI at
   `channel_selection=balanced`** — the third, genuinely independent voice
   in whatever EDF and DICOM disagree about, precisely because
   `arbitration_valid` is only ever `True` in this mode.
4. **`check_implant_hypothesis`'s `implant_proximity_correlation`**
   (`multimodal_approach/structural_anomaly.py`) — on this recording it
   comes out near zero (≈0.02) despite the coarse per-hemisphere
   artifact-fraction and mean-anomaly ratios looking suggestively close
   (≈0.82 vs. ≈0.80) — i.e. the voxel-wise check does *not* support "the
   structural channel is substantially just measuring the implant", even
   though the coarser per-hemisphere numbers alone might have suggested it.
   Both numbers are reported; neither is dismissed in favor of the other.

##### Structural anomaly graph (DICOM side, `multimodal_approach/structural_graph.py`)

`build_seizure_graph` (above) is a graph built from *time-series
correlation* — the EEG side of "graph approaches for both time series and
images". A static post-implant MRI has no time axis to correlate over (see
the table at the top of this section), so `multimodal_approach` now has its
own graph, built on the one relationship distance alone can support:
**spatial proximity** between the asymmetry/heterogeneity clusters
`find_top_anomaly_clusters` already ranks. `build_structural_anomaly_graph`
prunes it the same way `build_seizure_graph` prunes its co-activation mesh
— a distance threshold plus top-*k*-per-node — substituting distance for
correlation magnitude; edge `kind` is always `"proximity"` (there is no
second, temporal edge kind possible here), weighted `1/(1+distance_mm)`.
Kept fully separate from the EEG graph and from any combined score — the
same "never merge into one number" discipline `combined_anomaly`/
`combined_heterogeneity` and `object_model/graph.py`'s three separate
attribute layers already follow.

On `sEEG-HFOs-8.edf` this produces three disconnected pieces: the strongest
finding overall (`asym_0`, 2037 voxels, left temporal, peak z=+9.53) sits
within 22.6 mm of a second, independently-ranked left-temporal cluster
(`asym_3`) — weak evidence of one coherent region, not a one-voxel fluke;
a second coherent pair (`asym_1`/`asym_4`) sits on the *opposite*
(right) side — the asymmetry channel's own top clusters are not all on one
side, exactly the tension the lateralization index above has to average
over; and the four heterogeneity clusters (`het_0`–`het_3`) turn out to be
one anatomical neighbourhood — the periventricular CSF-boundary artifact
`multimodal_approach/README.md`'s "Honest limits" already documents by
name — split into several pieces by connected-component labelling, visible
as one dense sub-graph here rather than four independent findings the
ranked list alone would suggest.

`plot_structural_anomaly_graph_anatomical` renders this graph on **three
real DICOM slices** — axial, coronal, sagittal, all cut through the same
physical point (the graph's own strongest node), the DICOM-viewer three-pane
convention, not an abstract layout and not a maximum-intensity projection
(an earlier version used one; it looked tidier but is not what any actual
slice looks like). Since most nodes are not physically on whichever single
slice a given panel shows, each node's real distance from that slice
(`depth_mm`) is computed and disclosed rather than hidden: solid ring, full
opacity, and a bare label only within a few millimetres of the slice shown;
otherwise a dashed ring, fading opacity, and a `Δ<depth_mm>mm` label —
disclosure over false precision. See `multimodal_approach/README.md`,
"Structural anomaly graph", for the full node table and method.

##### Citable dissertation figures

`top_idea_figures/` holds renamed, citation-ready copies of a subset of
the figures above — dissertation naming convention
(`ch3_3-11_{slug}_BioMedAI-sEEG-core-of-epilepsy_{YYYYMMDD}.png`), each PNG
paired with a `.json` of exactly the numbers it depicts, the same
PNG+JSON-sibling convention `multimodal_result/` already uses:

* `..._edf-recruitment-cascade_...` — `plot_seizure_evolution`'s band-energy
  z-score heatmap and recruitment latencies (JSON: the full `BrainProcess`);
* `..._coactivation-graph_...` — `plot_seizure_graph`'s recruitment +
  co-activation graph (JSON: every node/edge attribute);
* `..._reservoir-residual-lateralization_...` — `model/visualize.py`'s
  per-channel residual heatmap (JSON: per-channel onset/peak plus the
  `reservoir_residual_strength`/`reservoir_residual_earliness` lateralization
  entries from `verification_report.json`);
* `..._reservoir-architecture_...` — `model/visualize.py`'s
  `plot_model_architecture`, the state-space block diagram
  (`u(t) -> [B, A] -> x(t) -> [C, D] -> y(t)`) showing *how the reservoir
  works* -- which weights are fixed-random vs. trained, and this run's
  actual dimensions (JSON: the state/output equations, `reservoir_config`,
  and achieved spectral radius);
* `..._reservoir-connectivity_...` — `model/visualize.py`'s
  `plot_reservoir_connectivity`, the reservoir's own literal random graph
  showing *what it looks like* as a network (input/hidden/output nodes,
  strongest recurrent/input/readout edges) (JSON: sampling/threshold
  parameters and the same `reservoir_config`);
* `..._object-model-three-layer-summary_...` — the five-panel
  `object_model_summary.png` (JSON: the complete `verification_report.json`).

These are static exports of one run on `sEEG-HFOs-8.edf`
(`--crop-end-seconds 10550 --channel-selection balanced`), not regenerated
automatically — rerun `model.run_model`/`object_model.run_object_model`
(above) after any change to the code they depend on.

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
EDF-записей](#анализ-и-визуализация-edf-записей)).

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
времени не *открывает* локализацию по сигналу — он проверяет, пересекли ли
контакты, уже названные внешним клиническим контекстом (PM3–8, CC8–10),
определённый по данным порог рекрутирования в пределах определённого по
данным окна относительно глобально самого раннего пересечения по всему
монтажу. На этом файле — да: `likely_initiators` оказываются `PM3–PM8` и
`CC8–CC10`, а `later_recruited` охватывает как «штрихованные» (primed), так
и «нештрихованные» контакты `PA`/`SA`/`CR` (билатеральные лобные и
теменные). Из данных, без какого-либо априорного списка контактов, выводятся
все латентности рекрутирования, единственное глобально самое раннее
пересечение (`τmin` — 0.035 с после аннотированного пика на этой записи), и
то, какие контакты делят это первенство. См. [Как вычисляется
`likely_initiators`](#как-вычисляется-likely_initiators) ниже — точное
правило, дефект, который был у прежней двухветочной версии, и собственные
`earliest_contacts`/`hemisphere_of_earliest` этого файла — свободное от
приора прочтение той же записи. Для сравнения,
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

##### Как вычисляется `likely_initiators`

`likely_initiators` — это единственный результат, вокруг которого построен
весь этот пайплайн, и он смешивает две вещи, которые обязаны оставаться
явно различимыми: **приор** (`ContactPrior`), заданный извне записи, и
вычисление, полностью определяемое сигналом. И `analyse_brain_process`
(`edf_workflow.py`), и `BrainProcess` (`models.py`) хранят их как отдельные
поля, а не молча сливают в одно число.

1. Скользящая энергия 250 мс в полосе 13–80 Гц по каждому каналу
   (`_beta_gamma_z_scores`).
2. Робастная нормировка median/MAD относительно предсобытийного baseline
   каждого канала.
3. Латентность рекрутирования `τc` — первое послесобытийное окно, где
   z-оценка канала превышает `RECRUITMENT_THRESHOLD_MAD` (6 MAD).
4. `τmin = minc τc` — единственное самое раннее пересечение **по всем
   каналам**, вычисленное без какой-либо ссылки на список контактов.
5. Классификация относительно `τmin`, на три взаимоисключающие и
   исчерпывающие категории:
   - `earliest = {c : τc ≤ τmin + 0.05с}` (`SIMULTANEITY_WINDOW_SECONDS`) —
     чисто по данным, вне зависимости от того, назван ли контакт приором;
   - `prior_early = {c ∈ prior : τc ≤ τmin + 0.25с} \ earliest`
     (`PRIOR_WINDOW_SECONDS`) — более широкое окно, применимое только к
     контактам, уже названным приором;
   - всё остальное с измеренной латентностью — `later_recruited`.
6. Итог:

   ```
   likely_initiators = prior_early ∪ (earliest ∩ prior)
   ```

   где на этом датасете `prior` — это `{PM3, ..., PM8, CC8, CC9, CC10}`
   (`SEEG_HFOS_8_CLINICAL_PRIOR`, `edf_workflow.py`). Передача `prior=None`
   в `analyse_brain_process` отключает всё это — `likely_initiators`
   сводится к `earliest`, а `initiators_constrained_by_prior` всегда равно
   `False` — для региональн-агностического прочтения той же записи.

**Выводится из данных:** каждое `τc`, `τmin`, факт пересечения порога любым
контактом, состав `earliest`. **Задано извне:** сам список контактов приора,
два окна (0.05 с, 0.25 с) и порог 6 MAD — ни одно из этих трёх чисел не
изменилось в этой правке; изменилось то, что теперь это именованные
константы (`SIMULTANEITY_WINDOW_SECONDS`, `PRIOR_WINDOW_SECONDS`,
`RECRUITMENT_THRESHOLD_MAD`), а не голые литералы.

**Слепое пятно, и зачем теперь публикуется `earliest_contacts`.** В ветке,
ограниченной приором, инициаторами могут стать только контакты, названные
приором — так что если глобально самое раннее пересечение принадлежало бы
контакту *вне* приора, оно раньше не попадало ни в инициаторы (не в приоре),
ни в поздно рекрутированные (его латентность была слишком близка к `τmin`,
чтобы старое условие `> τmin + 0.05с` для later_recruited оказалось
истинным) — оно тихо исчезало из обоих кортежей, причём хуже всего для того
самого контакта, чьё время могло бы противоречить приору. `BrainProcess`
теперь отдельно публикует `earliest_contacts` (свободное от приора самое
раннее множество) и `hemisphere_of_earliest` именно для того, чтобы это
можно было проверить, а не принимать на веру, а `analyse_brain_process`
выбрасывает `ValueError`, если её собственное трёхстороннее разбиение
измеренных латентностей вдруг оказывается не точным.

На `sEEG-HFOs-8.edf` (событие уровня 2, родная референция) `τmin = 0.035с`
после аннотированного пика, и **92 из 98 вовлечённых каналов делят это
первенство** — пик этого приступа рекрутирует почти весь монтаж в пределах
одного 50-мс окна, а не горстку фокальных контактов. Из этих 92 только 9
(`prior_fraction_among_earliest ≈ 0.098`) — контакты, названные клиническим
приором; остальные охватывают оба полушария (57 правых/35 левых), так что
`hemisphere_of_earliest` читается как **`"mixed"`**, а не `"right"`.
`initiators_constrained_by_prior` здесь равно `False` — более широкому
0.25-секундному окну приора не пришлось выходить за пределы этого
первенства, чтобы найти PM3–8/CC8–10, поскольку они уже были внутри него.
Честное прочтение — *не* «данные независимо подтверждают правый лобный
источник»; это «PM3–8/CC8–10 — среди тех ~94% этого монтажа, что пересекли
порог одновременно на пике приступа, и именно список контактов приора, а не
время рекрутирования, выделяет их как `likely_initiators`». Другая запись —
или эта же, рассмотренная раньше в каскаде, а не на аннотированном пике, —
вполне могла бы дать `hemisphere_of_earliest = "right"` или `"left"`;
ничто в этом правиле не настроено в пользу того или иного ответа.

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
Оба монтажа сходятся на одних и тех же `likely_initiators` (`PM3–8`/`CC8–10`,
записанных как метки пар `PM2-3`, ..., `CC9-10` под `bipolar`) — но это
совпадение отчасти гарантировано самой конструкцией, а не является чистой
перекрёстной проверкой: оба прогона проверяют *один и тот же* список
контактов приора против собственного определённого по данным порога, так
что совпадение `likely_initiators` в основном показывает, что эти контакты
приора пересекли порог в пределах окна приора при обеих референциях — а
это слабее, чем независимая сходимость двух разных методов к одному ответу.
Проверка, ничем не обязанная приору, — это `earliest_contacts` (см. [Как
вычисляется `likely_initiators`](#как-вычисляется-likely_initiators) выше):
на этой записи самые ранние множества обоих монтажей велики — 92 из 98
каналов для `none`, 76 из 82 для `bipolar` — и смешаны по полушариям в
примерно той же пропорции (57 правых/35 левых против 49 правых/27 левых),
так что `hemisphere_of_earliest` читается как `"mixed"` при обеих
референциях. *Именно это* совпадение — а не совпадение `likely_initiators`
— и есть подлинная перекрёстная проверка того, что выбор референции не
определяет результат. (Чтобы биполярные метки пар корректно сопоставлялись
с приором, потребовалось исправить реальный баг: прежнее регулярное
выражение `RIGHT_FRONTAL` проверяло только *первый* номер контакта пары,
так что `PM2-3`/`CC7-8` — у которых в диапазоне именно *второй* конец —
тихо классифицировались неверно. Исправлено как `is_right_frontal`, теперь
тонкая обёртка над общей `prior_matches`, которая проверяет любой из концов.)

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
латентности рекрутирования, самые ранние сверху — этот порядок строк
выведен исключительно из данных, никогда отдельно переотобранный «топ N» и
никогда не сдвигаемый приором. Пунктирная линия отмечает время события;
поскольку аннотация уровня 2 помечает 10396.445 с как момент, когда приступ
уже был оценён как *«приступ + билатеральный тонико-клонический»* (с
собственной предшествующей заметкой клинициста «где начало?» на 10392.734
с), это **пик** каскада, а не обязательно его первое подёргивание — окно
baseline рисунка (по умолчанию 30 с до, 8 с после) специально показывает
нарастание, ведущее к нему, а не только момент пересечения порога. Поверх
этого выведенного из данных порядка рисунок показывает приор и правило
классификации как два визуально разделённых слоя (см. [Как вычисляется
`likely_initiators`](#как-вычисляется-likely_initiators)): значок ◆ перед
подписью строки отмечает контакт, названный клиническим приором
(`process.prior_matched`), не сдвигая эту строку; `τmin`, окно
одновременности и более широкое окно приора нанесены на ось времени и
подписаны; маленький «×» отмечает собственный измеренный момент пересечения
каждой строки; и каждая строка из `process.earliest_contacts` выделена
жирным — золотым, если её также называет приор, зелёным, если нет, так что
контакт, который мог бы противоречить приору (если бы данные его дали),
визуально невозможно не заметить.

На `sEEG-HFOs-8.edf` это показывает то, что похоже на саму
тонико-клоническую фазу: 98 из 100 каналов пересекают порог рекрутирования,
и 92 из них делят глобально самое раннее пересечение (`τmin`, 0.035 с после
пика) — что согласуется с генерализованной, охватывающей весь монтаж
ЭЭГ/ЭМГ-сигнатурой уже установившегося билатерального тонико-клонического
приступа, а не с чистым фокальным каскадом (этот фокальный каскад, если он
вообще виден, был бы в секундах *до* пика, где рисунок показывает гораздо
более разреженную активность). Поэтому большинство строк рисунка выделены
жирным; из этих 92 только 9 золотые (названные приором), остальные зелёные
— то самое разбиение на 92 канала со смешанными полушариями, которое
количественно описано в разделе о слепом пятне выше. Несколько контактов
явно рекрутируются позже — `EEG PA9` (0.24 с), `EEG PA'3` (0.39 с), `EEG
SA'2`/`EEG SA'3` (2.0 с), `EEG PA'4` (2.1 с), `EEG CR'5` (5.7 с) — и это как
раз штрихованные (левополушарные) контакты, которые слепой детектор уровня
3 независимо отметил как отдельную позднюю группу кандидатов — небольшая,
но подлинная перекрёстная проверка между двумя независимыми частями этого
пайплайна.

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

Каждый layout рисует каждый узел-канал с тремя независимыми кодировками,
намеренно разделёнными визуально, а не слитыми: **цвет заливки — роль**
(малиновый/оранжевый/синий для `earliest`/`prior_early`/`later_recruited` —
то же выведенное из данных разбиение, сначала не зависящее от приора, затем
расширяемое им, что описано в [Как вычисляется
`likely_initiators`](#как-вычисляется-likely_initiators)), **цвет обводки —
принадлежность приору** (золотая обводка, когда атрибут узла `in_prior`
истинен, тонкая серая иначе — внешний вход), и **размер — пиковая
z-оценка**. Узел, у которого заливка и обводка согласуются (малиновая или
оранжевая заливка с золотой обводкой, либо синяя заливка с серой обводкой)
— это случай, где выведенная из данных роль и внешний приор совпадают;
несовпадение — это как раз то место, где они расходятся, самая
информативная часть этого рисунка, поэтому эти две кодировки никогда не
сливаются в один цвет. Все четыре атрибута узла (`role`, `in_prior`,
`latency_seconds`, `hemisphere`) записываются в
`<edf-name>_seizure_graph.graphml`, так что ту же проверку можно выполнить
вне этого пайплайна (например, через `networkx.read_graphml`). На
`sEEG-HFOs-8.edf`, где 92 из 98 каналов связаны одной и той же латентностью
первого окна (`τmin`, см. выше), `radial` упаковывает большинство узлов —
малиновых, лишь 9 с золотой обводкой — в одну плотную внутреннюю дугу —
честное следствие отказа от переотбора меньшего «топ N», а не ошибка
рендеринга — тогда как горстка позже рекрутированных (синих) выбросов
(`EEG CR'5`, `EEG SA'1/2/3`, `EEG PA'3/4`) заметно располагается отдельно
от неё на большем радиусе — те же каналы, отмеченные на тепловой карте
выше; три остальных layout'а показывают тот же самый набор узлов с точки
зрения только корреляции, только латентности и разделения по ролям
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
  монтажа) — плюс оценки бета/гамма по каналам, латентности рекрутирования и
  `likely_initiators` внутри `brain_process` (который также несёт
  `earliest_contacts`, `earliest_latency_seconds`, `prior_matched`,
  `prior_source`, `initiators_constrained_by_prior` и
  `prior_fraction_among_earliest`/`hemisphere_of_earliest` — см. [Как
  вычисляется `likely_initiators`](#как-вычисляется-likely_initiators)),
  `evolution_figure`, `graph_figures` (по одному пути на
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

`model_result/<edf-name>_*.png` (девять рисунков за запуск):

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
* `_residual_timeseries.png` — сам `evaluation.residual`
  (`model.visualize.plot_residual_timeseries`), по каждому каналу, как
  временной ряд «реальное минус предсказанное» в собственных физических
  единицах записи (вольты), а не в z-оценке тепловой карты — насколько
  велика ошибка предсказания на самом деле, а не насколько она удивительна
  относительно собственного baseline-шума этого канала;
* `_baseline_vs_event_accuracy.png` — `model.visualize.plot_baseline_vs_event_accuracy`,
  столбчатая диаграмма RMSE на baseline против RMSE после события по
  каждому каналу и в целом, каждая пара подписана отношением — рисунок,
  реализующий «оценить точность модели через сравнение baseline и события»
  (см. `model.visualize.compute_baseline_vs_event_rmse`, записывается также
  в `_model_result.json` как `baseline_vs_event_rmse`);
* `_extreme_event_score.png` — агрегированная невязочная оценка
  относительно порога, с отмеченными собственными началом/пиком модели.

На `sEEG-HFOs-8.edf` (конфигурация по умолчанию) `compute_baseline_vs_event_rmse`
находит, что все 12 выходных каналов без единого исключения деградируют
после события — RMSE на baseline 3.7–9.1×10⁻⁵ В растёт до 7.2–17.8×10⁻⁵ В,
отношения 1.4×–3.4× (`PA9` худший — 3.4×, `CC10` наименьший — 1.4×), общее
отношение по RMS **2.3×**. Невязка `EEG PA9` показывает заметно ступенчатый
паттерн вместо непрерывной текстуры ЭЭГ на `_residual_timeseries.png` —
стоит проверить сырую трассу этого канала на насыщение/клиппинг усилителя,
прежде чем читать его невязку (наибольшее отношение из двенадцати) как
чисто физиологическую.

`<edf-name>_model_summary.txt`/`_model_result.json` несут те же числа в виде
текста/JSON — конфигурацию резервуара, RMSE обучения по каждому каналу и
вердикт по экстремальному событию (`describe_evaluation`, аналог
`describe_seizure_source` для этой модели). Как и всё остальное в этом
репозитории, результат «обнаружено» здесь — кандидат для экспертной
проверки: он сверяется с собственной детекцией `extreme_event_agent`,
основанной на пространственном рекрутировании, а не считается более
авторитетным, чем она.

##### Спектр устойчивости резервуара (`_spectrum.png`)

`model.visualize.plot_reservoir_spectrum` рисует собственные значения
собственной рекуррентной матрицы весов резервуара `A` (в коде — `W`) —
по одной точке на каждое собственное значение фиксированной случайной
матрицы `n_reservoir × n_reservoir`, управляющей уравнением состояния
`x(t) = (1-leak)·x(t-1) + leak·tanh(B·u(t) + A·x(t-1) + bias)`. `A`
генерируется один раз, случайно, при создании модели и никогда не
обучается (обучается ридж-регрессией только readout `C`/`D` — в этом весь
фокус reservoir computing). Этот рисунок — **диагностика устойчивости
именно этой фиксированной системы**, а не того, насколько хорошо модель
подгоняется под эту запись — он вообще ничего не говорит о `y(t)` или о
событии.

Каждая точка нанесена на комплексной плоскости (действительная часть по
оси x, мнимая — по оси y) относительно пунктирной единичной окружности
(радиус 1); достигнутый спектральный радиус (модуль наибольшего
собственного значения `A`, отмасштабированного при построении так, чтобы
попасть в `--spectral-radius`, по умолчанию 0.95) указан в заголовке. Это
стандартная проверка **echo-state property**: система с такой рекуррентной
подачей входа гарантированно «забывает» своё произвольное начальное
условие `x(0) = 0` и выходит на траекторию, определяемую только реальным
входом — свойство, от которого зависит весь подход с обучением readout —
только когда каждое собственное значение строго лежит внутри единичной
окружности (спектральный радиус < 1). На `sEEG-HFOs-8.edf` (конфигурация
по умолчанию, `n_reservoir=400`) все 400 собственных значений лежат внутри
пунктирной окружности, подтверждая, что резервуар действительно построен
так, как задано, и является сжимающим, а не просто заявлен таковым.
Повышение `--spectral-radius` до 1 или выше (протестировано в переборе
гиперпараметров ниже) выводит точки на окружность или за неё; в этом
режиме устойчивость перестаёт обеспечиваться собственным сжатием `A` и
целиком зависит от явного интеграционного члена `leak_rate` — подпись на
самом рисунке говорит об этом прямо. Читать его стоит вместе с
`_architecture.png` (чем система *является*) и `_connectivity.png` (как
она выглядит как сеть); о качестве подгонки говорят `_output_prediction.png`,
`_residual_heatmap.png`, `_residual_timeseries.png` и
`_baseline_vs_event_accuracy.png`.

**`--channel-selection balanced`.** По умолчанию (`recruitment`) выходные
каналы резервуара `y(t)` берутся прямо из собственных `likely_initiators`/
`later_recruited` анализа `analyse_brain_process` — это полезно, чтобы
проверить, отмечает ли невязка резервуара то же событие независимо, но это
**не** валидный вход для перекрёстной проверки латерализации того же
анализа, поскольку сами каналы уже взяты из него. `--channel-selection
balanced` выбирает каналы иначе: поровну по полушариям (через
`hemisphere_of_channel`), внутри каждой половины — по ранжиру дисперсии
только на предсобытийном участке, никогда не по латентности, не по
рекрутированию, не по чему-либо после события. `ReservoirWindow.arbitration_valid`
равно `True` только в этом режиме; каждая вычисленная из него ниже по
конвейеру оценка латерализации несёт этот флаг, так что её невозможно
случайно принять за независимое подтверждение, когда это не так.
`run_reservoir_plant` теперь также оценивает невязку каждого выходного
канала **независимо** (`per_channel_score`/`per_channel_onset_seconds`/
`per_channel_peak_score`, median/MAD-нормировка по каждому каналу
относительно его собственного baseline — так же, как уже нормируется
скалярная `score`) вместо только схлопнутого по всему окну скаляра —
пространственное прочтение того, *где* предсказание модели ломается первым,
а не только *ломается ли оно вообще*.

##### Настройка резервуара для более точного baseline-фита

Каждый гиперпараметр резервуара уже доступен как флаг CLI
(`--n-reservoir`/`--spectral-radius`/`--leak-rate`/`--ridge-alpha`/
`--output-feedback-lag`), так что улучшение фита планта не требует
изменения кода. Перебор по одному параметру за раз на дефолтном окне
`sEEG-HFOs-8.edf` (остальные параметры зафиксированы на значениях CLI по
умолчанию) дал:

| параметр | направление, улучшающее RMSE на baseline | величина эффекта |
|---|---|---|
| `n_reservoir` | больше (200→1000) | 6.68×10⁻⁵ → 4.51×10⁻⁵ (в основном убывающая отдача после ~800) |
| `spectral_radius` | почти нет эффекта (0.5→1.05) | 5.57×10⁻⁵ → 5.67×10⁻⁵ — большую часть сигнала здесь несёт NARX-эмбеддинг задержки, а не собственная рекуррентная динамика резервуара |
| `leak_rate` | выше/менее «протекающий» (0.05→1.0) | 1.04×10⁻⁴ → 3.48×10⁻⁵ — самый сильный рычаг |
| `ridge_alpha` | ниже регуляризация (1→1e-5) | 1.37×10⁻⁴ → 1.31×10⁻⁵ — тоже сильный эффект, но на крайнем малорегуляризованном конце есть риск подгонки под baseline-шум, а не под сигнал |
| `output_feedback_lag` | слабо, немного лучше при увеличении (1→20) | 6.16×10⁻⁵ → 5.56×10⁻⁵ |

Это легитимная цель для оптимизации — это качество фита исключительно на
baseline, никогда не затрагивающее окно события или его метку, та же
граница «никогда не настраивать под известный ответ», которую `MANIFEST.md`
уже формулирует для порогов детекции в других местах этого репозитория.
Выбор умеренной точки на этом фронте, а не крайней точки перебора (чтобы
избежать `leak_rate=1.0`, который обнуляет собственный рекуррентный член
памяти в уравнении состояния и сводит ESN к безпамятному нелинейному
отображению NARX-входа, и чтобы избежать снижения `ridge_alpha` до
шумо-подгоняющего конца) —

```bash
python -m model.run_model dataset/sEEG-HFOs-8.edf --output model_result_tuned \
  --n-reservoir 800 --spectral-radius 0.95 --leak-rate 0.7 \
  --ridge-alpha 1e-3 --output-feedback-lag 10
```

— снижает общий RMSE на baseline с 1.99×10⁻⁴ В (по умолчанию) до
**1.87×10⁻⁵ В** (фит примерно в 3 раза точнее), и, что проверено, а не
просто предположено, поведение на *событии* улучшается в том же
направлении, а не приносится в жертву: общее отношение RMSE
baseline/событие растёт с 2.3× до **3.0×**, пиковая оценка растёт с
73.5 MAD до **153.5 MAD**, а доля послесобытийных отсчётов выше порога
6 MAD растёт с 20.7% до **33.0%** — более точно подогнанный плант здесь
одновременно и более чувствителен, а не просто выглядит лучше на baseline.
Предсобытийный переходный процесс на −47…−48 с (см. выше) остаётся
практически на том же самом времени при любой из протестированных
конфигураций — ещё одно свидетельство того, что это реальная особенность
записи, а не артефакт какого-то одного выбора гиперпараметров.

### Модель объекта: объединение EDF, DICOM и резервуара (`object_model/`)

Три модальности выше измеряют эту запись по-разному, и каждая может дать
лишь часть картины:

| Источник | Временное разрешение | Пространственное разрешение |
|---|---|---|
| EDF (`extreme_event_agent`) | доли секунды | до отдельного контакта |
| DICOM (`multimodal_approach`) | **отсутствует — у статического постимплантационного снимка нет оси времени** | только полушарие (в репозитории нет верифицированной 3D-локализации отдельных контактов — см. `multimodal_approach/README.md`, «Честные ограничения») |
| Резервуар (`model/`, `channel_selection="balanced"`) | доли секунды | до отдельного выходного канала, если `per_channel_score` не схлопнута |

`object_model/` — единственный пакет, импортирующий все три остальных
вместе (каждый из них остаётся свободен от зависимости от другого-соседа),
и делает с этим две вещи: **проверяет** каждый метод против единственной
имеющейся у этой записи истины (её собственной аннотации EDF+) и
**собирает** три источника свидетельств на одном графе, никогда не сливая
их в единый балл.

**`extreme_event_agent.verification.verify_against_annotation`** оценивает:

- *Временную точность* — знаковую `delta_seconds = t_метода − t_БТКП`
  (10396.445 с), никогда не приводимую к `abs()` перед сохранением, так что
  метод, сработавший раньше, и метод, сработавший позже на ту же величину,
  остаются различимы. Разбита на полосы `precise` (≤1 с — точнее, чем ~6.7 с
  разброс между самой ранней и самой поздней аннотацией этой же записи для
  одного приступа), `coarse` (≤10 с — иктальная фаза), `window` (≤60 с —
  событие как целое) или `miss`. В этом устанавливаемом пакете реально
  существует ровно два метода: `t_targeted` (самое раннее пересечение по
  `analyse_brain_process`) и `t_blind` (собственный выбор уровня 3
  `ExtremeEventAgent`) — фигурирующая в другом месте этого README цифра
  ≈+39.6 с «широкополосного ансамбля» относится к пятиметодному ансамблю
  ноутбука `sEEG_extreme_event_detector_colab.ipynb`, который `MANIFEST.md`
  уже документирует как находящийся вне устанавливаемого пакета, поэтому
  здесь он не выдуман как третий реально вычисляемый метод.
- *Латерализацию* — `LI = (v_right − v_left) / (v_right + v_left) ∈ [-1, 1]`,
  каждое `v` сначала нормировано на число каналов/вокселей своего
  полушария, так что источники с очень разным числом сопоставимы:
  `edf_earliest_contacts` (доля `process.earliest_contacts` по полушариям —
  свободна от приора, в отличие от `likely_initiators`, которая по
  определению никогда не может разойтись со стороной приора),
  `dicom_mean_abs_anomaly` (прямо из `hemisphere_summary`), и, когда дана
  оценка резервуара, `reservoir_residual_strength`/`reservoir_residual_earliness`.
  `|LI| < 0.05` читается как `"indeterminate"`, сторона никогда не
  навязывается насильно.
- *Перекрытие контактов* — точность/полноту/коэффициент Жаккара между
  `earliest_contacts` (данные) и `prior_matched` (внешний приор) —
  насколько данные подтверждают клиническую гипотезу, а не утверждение о
  локализации.

Каждый `VerificationReport` несёт `crop_applied`/`channel_selection`/
`masking_method`/`prior_used` — контекст, от которого зависит каждое число
выше, никогда не опускаемый молча, записывается в `verification_report.json`.

**`object_model.graph.build_object_model_graph`** берёт уже построенный
результат `build_seizure_graph` и добавляет к каждому узлу-каналу ещё две
группы атрибутов: структурную (`hemisphere_anomaly_mean`/
`hemisphere_anomaly_max` из DICOM, по полушарию самого узла) и модельную
(`residual_onset_seconds`/`residual_peak_score`, только для узлов, которые
также являются выходными каналами резервуара — отсутствуют, а не `None`,
для остальных, поскольку в GraphML нет типа null). Три раздельно
именованные группы, никогда не усредняемые в один балл — `structural_anomaly.py`
уже держит `combined_anomaly`/`combined_heterogeneity` раздельно по тому же
принципу, а узел, у которого роль по EDF и структурное/резервуарное
свидетельство *расходятся*, — это как раз тот случай, который слияние бы
скрыло.

Запуск:

```bash
python -m object_model.run_object_model --edf dataset/sEEG-HFOs-8.edf \
  --dicom-dir dataset/MRI-with-electrodes/DCM --crop-end-seconds 10550 \
  --channel-selection balanced --output object_model_result
```

записывает в `object_model_result/<edf-name>/`: `verification_report.json`,
`object_model_graph.graphml` (трёхслойные атрибуты узлов, только когда
процесс EDF нашёл вовлечённые каналы) и `object_model_summary.png` — один
рисунок, пять панелей: каскад рекрутирования EDF (порядок строк из данных,
контакты приора помечены, но никогда не сдвинуты), граф модели объекта
(заливка = роль, обводка = приор, форма = полушарие, размер = пиковая
z-оценка), срез DICOM через самый сильный структурный кластер, поканальная
невязка резервуара (отсортирована по моменту срабатывания, зона washout
затенена) и сводка верификации (Δt по методам с полосами допуска; LI по
источникам с затенённой зоной неопределённости) — с подписанными на каждом
рендере режимом отбора каналов, статусом обрезки, способом маскирования и
статусной строкой про кандидатов для экспертной проверки.

На `sEEG-HFOs-8.edf` (`--crop-end-seconds 10550 --channel-selection
balanced`) — подлинно смешанный результат: `t_targeted` попадает в
`precise` (+0.035 с), `t_blind` — в `window` (+41.6 с, согласуется с
обсуждением уровня 3 выше). Четыре источника латерализации **не**
согласуются: `edf_earliest_contacts` читается едва как `right` (LI ≈ +0.08
— 92 из 98 каналов делят первенство по времени, см. «Как вычисляется
`likely_initiators`» выше, так что это слабый сигнал, не уверенный),
`dicom_mean_abs_anomaly` читается как `left` (LI ≈ −0.11),
`reservoir_residual_strength` — как `right` (LI ≈ +0.53), а
`reservoir_residual_earliness` — как `left` (LI ≈ −1.0, определяется
выходными каналами, которые справа пересекли порог только во время
предсобытийного baseline). Это сообщается именно как расхождение, а не
сводится к одному ответу — см. [Что смотреть в
результате](#что-смотреть-в-результате) ниже о том, как это читать.

##### Что смотреть в результате

Ни один исход ниже не предпочтителен заранее — смысл разделения этих чисел
в том, что ответ виден и проверяем, а не в том, что он получается
каким-то конкретным образом.

1. **`hemisphere_of_earliest`** (из `analyse_brain_process`, свободно от
   приора — см. «Как вычисляется `likely_initiators`» выше): какой контакт
   сработал первым вообще без всякого списка контактов. `"right"`
   подтвердил бы клинический приор действительно независимыми данными;
   `"left"` перевернул бы картину и совпал бы со структурной находкой
   вместо этого; на этой записи это `"mixed"` — 92 из 98 каналов делят
   первенство, так что временной канал сам по себе латеральность не
   разрешает вовсе, и каждый `LI` выше следует читать с этим в виду.
2. **Δt резервуара** против фигурирующей ранее в этом README цифры
   ≈+39.6 с широкополосного ансамбля: существенно меньшее значение
   означало бы, что модельно-ориентированный критерий находит переход
   раньше, чем чисто статистический — прямая проверка того, где именно
   лежат границы применимости слепой статистической детекции, без
   заранее принятого ответа.
3. **LI `reservoir_residual_strength`/`reservoir_residual_earliness` при
   `channel_selection=balanced`** — третий, действительно независимый
   голос в том, в чём расходятся EDF и DICOM, именно потому что
   `arbitration_valid` истинно только в этом режиме.
4. **`implant_proximity_correlation` из `check_implant_hypothesis`**
   (`multimodal_approach/structural_anomaly.py`) — на этой записи он
   получается близким к нулю (≈0.02), несмотря на то что грубые
   отношения доли артефактных вокселей и средней аномалии по полушариям
   выглядят подозрительно близкими (≈0.82 против ≈0.80) — то есть
   поvoxel-вая проверка *не* подтверждает, что структурный канал по сути
   просто измеряет имплант, хотя более грубые числа по полушариям сами по
   себе могли бы на это намекать. Приводятся оба числа; ни одно не
   отбрасывается в пользу другого.

##### Граф структурных аномалий (сторона DICOM, `multimodal_approach/structural_graph.py`)

`build_seizure_graph` (выше) — это граф, построенный на *корреляции
временных рядов*: EEG-сторона тезиса «графовые подходы и для временных
рядов, и для изображений». У статичного постимплантационного MRI нет оси
времени, по которой можно было бы что-то коррелировать (см. таблицу в
начале этого раздела), поэтому у `multimodal_approach` теперь есть
собственный граф, построенный на единственном отношении, которое
поддерживает одно только расстояние: **пространственная близость** между
кластерами асимметрии/гетерогенности, уже ранжированными
`find_top_anomaly_clusters`. `build_structural_anomaly_graph` обрезает его
той же схемой, что `build_seizure_graph` обрезает свою сетку
co-activation — порог расстояния плюс top-*k* соседей на узел — заменяя
корреляцию на расстояние; тип ребра всегда `"proximity"` (второго,
временного типа ребра здесь в принципе быть не может), вес
`1/(1+distance_mm)`. Полностью отделён от EEG-графа и от любого
объединённого счёта — тот же принцип «никогда не сливать в одно число»,
которому уже следуют `combined_anomaly`/`combined_heterogeneity` и три
раздельных слоя атрибутов в `object_model/graph.py`.

На `sEEG-HFOs-8.edf` это даёт три несвязных компонента: сильнейшая находка
во всём объёме (`asym_0`, 2037 вокселей, левая височная область, пик
z=+9.53) оказывается в пределах 22.6 мм от второго, независимо
ранжированного левовисочного кластера (`asym_3`) — слабое свидетельство
одной цельной области, а не случайного одиночного воксела; вторая цельная
пара (`asym_1`/`asym_4`) лежит на *противоположной* (правой) стороне —
собственные сильнейшие кластеры канала асимметрии не все на одной стороне,
и это как раз то напряжение, которое индексу латерализации выше приходится
усреднять. Четыре кластера гетерогенности (`het_0`–`het_3`) оказываются
одной анатомической областью — тем самым перивентрикулярным артефактом
CSF-границы, который уже назван по имени в разделе «Честные ограничения»
`multimodal_approach/README.md` — разбитой на несколько частей разметкой
связных компонент; граф показывает это как один плотный подграф, а не как
четыре независимые находки, на которые намекал бы один только
ранжированный список.

`plot_structural_anomaly_graph_anatomical` рисует этот граф на **трёх
настоящих срезах DICOM** — аксиальном, корональном, сагиттальном, все три
проходят через одну и ту же физическую точку (собственный сильнейший узел
графа), по конвенции трёхпанельного просмотра DICOM-станции, а не как
абстрактный layout и не как проекция максимальной интенсивности (более
ранняя версия использовала именно её — выглядело аккуратнее, но не
соответствовало виду настоящего среза). Поскольку большинство узлов
физически не лежат на том единственном срезе, что показан в конкретной
панели, реальное расстояние каждого узла до этого среза (`depth_mm`)
вычисляется и раскрывается, а не скрывается: сплошная обводка, полная
непрозрачность и голая подпись — только в пределах нескольких миллиметров
от показанного среза; иначе — пунктирная обводка, затухающая
непрозрачность и подпись `Δ<depth_mm>mm` — раскрытие вместо ложной
точности. Полную таблицу узлов и метод см. в `multimodal_approach/README.md`,
раздел «Structural anomaly graph».

##### Рисунки для цитирования в диссертации

`top_idea_figures/` содержит переименованные, готовые к цитированию
копии части рисунков выше — по конвенции диссертации
(`ch3_3-11_{кратко}_BioMedAI-sEEG-core-of-epilepsy_{ГГГГММДД}.png`), каждый
PNG в паре с `.json`, содержащим ровно те числа, что на нём изображены —
тот же принцип PNG+JSON-соседства, что уже используется в
`multimodal_result/`:

* `..._edf-recruitment-cascade_...` — тепловая карта z-оценки энергии по
  полосам и латентности вовлечения из `plot_seizure_evolution` (JSON: полный
  `BrainProcess`);
* `..._coactivation-graph_...` — граф рекрутирования и коактивации из
  `plot_seizure_graph` (JSON: атрибуты всех узлов и рёбер);
* `..._reservoir-residual-lateralization_...` — поканальная тепловая карта
  невязки из `model/visualize.py` (JSON: латентность/пик по каждому каналу
  плюс записи латерализации `reservoir_residual_strength`/
  `reservoir_residual_earliness` из `verification_report.json`);
* `..._reservoir-architecture_...` — `plot_model_architecture` из
  `model/visualize.py`, блок-схема пространства состояний
  (`u(t) -> [B, A] -> x(t) -> [C, D] -> y(t)`), показывающая, *как резервуар
  устроен и работает* — какие веса фиксированы случайно, а какие обучены, и
  фактические размерности этого прогона (JSON: уравнения состояния/выхода,
  `reservoir_config`, достигнутый спектральный радиус);
* `..._reservoir-connectivity_...` — `plot_reservoir_connectivity` из
  `model/visualize.py`, собственный случайный граф резервуара, показывающий,
  *как он выглядит* как сеть (узлы входа/скрытого состояния/выхода,
  сильнейшие рекуррентные/входные/считывающие рёбра) (JSON: параметры
  выборки/порога и тот же `reservoir_config`);
* `..._object-model-three-layer-summary_...` — пятипанельный
  `object_model_summary.png` (JSON: полный `verification_report.json`).

Это статичный снимок одного прогона на `sEEG-HFOs-8.edf`
(`--crop-end-seconds 10550 --channel-selection balanced`), не
перегенерируется автоматически — после изменения кода, от которого эти
рисунки зависят, нужно заново прогнать `model.run_model`/
`object_model.run_object_model` (см. выше).
