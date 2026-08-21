# Agent skill manifest

Every capability `extreme_event_agent` and this repo's notebooks currently
have, in one place. "Skill" here means a deterministic, auditable tool the
agent calls — never an LLM guess. Each entry names what it does, where it
lives, and what it deliberately does *not* claim. Update this file whenever
a skill is added, renamed, or removed; treat a stale entry as a bug.

Legend: **apriori-free** = takes no externally supplied event time/label;
**apriori-optional** = works with or without one, explicit input always wins.

## 1. Detection core

| Skill | Location | What it does |
|---|---|---|
| `ExtremeEventAgent.run` | [`agent.py`](src/extreme_event_agent/agent.py) | Bounded plan → act → observe → reflect loop. Validates data quality, extracts robust RMS/peak-to-peak/line-length/high-frequency features per sliding window, forms a multichannel consensus via median/MAD z-scoring, adapts its threshold if nothing crosses it, verifies spatial support (`min_involved_channels`), merges adjacent candidates, and keeps a full audit log. **Apriori-free.** Configured via `AgentConfig`. |
| `AgentConfig` | [`models.py`](src/extreme_event_agent/models.py) | Detection policy (window/step size, channel fraction, MAD threshold, merge gap, min involved channels, quality floor). Validates its own invariants in `__post_init__`. |
| `Event` / `DetectionReport` | [`models.py`](src/extreme_event_agent/models.py) | One verified candidate (start/end/peak/score/confidence/involved channels/evidence) and the full run's result (events + audit log). |

## 2. Event-time resolution (three-tier, apriori-optional)

No apriori event time is ever *required*; when one isn't given, the file's
own evidence is used before falling back to blind statistics. Priority:
explicit → embedded annotation → blind detection.

| Tier | Skill | Location | What it does |
|---|---|---|---|
| 1 | `ClinicalEvent` (via `--event-time`/`--event-clock`) | [`models.py`](src/extreme_event_agent/models.py), [`cli.py`](src/extreme_event_agent/cli.py) | An expert-typed time. Never used to tune detection (see its docstring). Drawn solid crimson. |
| 1 | `clock_time_to_offset` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Converts an `HH:MM:SS[.ffffff]` clinical clock to seconds from EDF start against `meas_date`, with midnight rollover and out-of-recording rejection. |
| 2 | `find_annotated_event` / `_cluster_seizure_annotation` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Reads the EDF's own EDF+ annotation channel (`cp1251`-decoded — MNE's default UTF-8 decode raises on Cyrillic annotation text), keyword-matches for a seizure marker (`SEIZURE_KEYWORDS`), and folds every annotation within `ANNOTATION_CLUSTER_GAP_SECONDS` of a match into one event. This is the clinician's own real-time markup, read from the file — not a number typed on a command line. Drawn solid teal. Produces `AnnotatedEvent`. |
| 3 | `select_seizure_event` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Only when tiers 1–2 are unavailable: picks the primary candidate from the agent's own verified detections, ranked by **channel spread, then duration, then score** — never score alone, since a brief high-amplitude interictal spike can outscore a widely/durably recruiting seizure. Drawn dashed orange (the only marker meant to read as an unconfirmed guess). Produces `DetectedEvent`. |

## 3. Data access

| Skill | Location | What it does |
|---|---|---|
| `read_edf` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Loads every non-`MKR` EDF signal channel in volts, `cp1251`-decoded. |
| `read_edf_markers` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Loads only the `MKR...` channels, for display — kept out of detection/process analysis because on `sEEG-HFOs-8.edf` they are a verified 1 Hz hardware sync clock, not brain signal or an event marker. |
| `read_edf_start` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Reads `meas_date` and duration without preloading signal samples (used by `--event-clock`). |
| `parse_contact_name` / `build_bipolar_montage` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Reads the bipolar montage straight off the channel names already in the file — no separate apriori electrode map. Groups contacts by shaft (e.g. `PM` vs. its distinct contralateral `PM'`), sorts numerically, and pairs each contact with its numeric neighbor (`1-2, 2-3, ...`), pairing across any gap in the numbering rather than dropping it. |
| `format_bipolar_montage` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Renders a montage as the compact `"shaft:\n  1-2\n  2-3\n..."` text a clinician expects; `run_edf` always writes this to `<stem>_montage.txt`, independent of whether any event was resolved or which reference was analysed. |
| `apply_bipolar_montage` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Computes the actual bipolar-referenced signals (`data[a] - data[b]`, labelled `"PM3-4"`) from a montage — a re-referencing (a spatial high-pass filter suppressing what two adjacent contacts share), not a filter that changes what's measured in kind. Wired into `run_edf` via `montage_reference="bipolar"`; the CLI's `--montages` runs it alongside the native reference by default. |

## 4. Process analysis

| Skill | Location | What it does |
|---|---|---|
| `analyse_brain_process` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Sliding 250 ms 13–80 Hz band-energy, median/MAD-normalized against the pre-event baseline. Recruitment latency = first post-event window ≥ 6 MADs. Right-frontal contacts (`PM3–8`, `CC8–10`) become `likely_initiators` only when they cross that data-derived threshold first; everything else recruited later is `later_recruited`. Works on referential channel names or bipolar pair labels alike (see `is_right_frontal`). Produces `BrainProcess`. |
| `describe_seizure_source` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Turns a `BrainProcess` into a plain-language statement of the located source — which channel(s) crossed the recruitment threshold first, at what absolute recording time, how many channels were ultimately involved, and the latency span across them — without computing anything not already in `process`. `run_edf` writes it to `<edf-name>_source_summary.txt` and `EdfRunResult.source_summary`. |
| `is_right_frontal` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Right-frontal (`PM3–8`/`CC8–10`) membership check used by `analyse_brain_process`. Handles both a single contact (`"EEG PM3"`) and a bipolar pair label (`"PM3-4"`), matching on *either* endpoint of a pair — fixes a real bug in the predecessor regex, which only checked a pair's first number and so misclassified `PM2-3`/`CC7-8` as not right-frontal even though their second contact is. |
| `_beta_gamma_z_scores` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | The shared time-resolved `[windows, channels]` z-score core behind `analyse_brain_process`, `plot_seizure_evolution`, `build_seizure_graph`, and `evaluate_message_passing` — one computation, four consumers, no duplicated math. |

## 5. Visualization

| Skill | Location | What it does |
|---|---|---|
| `plot_all_timeseries` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Whole-recording overview: every EEG + `MKR...` channel, robust-normalized, with the resolved event marked in its tier's colour/style. |
| `plot_seizure_evolution` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Channel-by-time heatmap of the recruitment cascade, restricted to exactly the channels `analyse_brain_process` found involved (never a re-picked "top N"), ordered by onset latency. |

## 6. Graph construction and layouts

| Skill | Location | What it does |
|---|---|---|
| `build_seizure_graph` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Builds a NetworkX graph: one node per involved channel plus a synthetic `PEAK` node for the resolved event. Two measured edge kinds — recruitment spokes (`PEAK`→channel, weighted by latency) and a co-activation mesh (Pearson correlation of channels' own beta/gamma z-score traces, threshold + top-*k* pruned the same way `sEEG_temporal_wavelet_graph_colab.ipynb` prunes its `db4`-correlation graphs). No propagation path is assumed — only what co-varies. |
| `plot_seizure_graph` / `_seizure_graph_layout` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Renders the graph in one of four layouts — `radial` (angle from mesh spring layout, radius from latency; the "outside-in to peak" reading), `spring` (whole-graph force-direction), `circular` (latency-only clock face, no correlation structure), `shell` (initiator vs. later-recruited rings). Carries a real `ax.legend()` identifying every marker/edge kind, a colourbar for the peak-z-score colour scale, and a boxed caption stating the located source (initiator channel(s) and absolute time) in words. |
| `plot_seizure_graph_layouts` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Convenience wrapper rendering all four layouts to `<stem>_seizure_graph_<layout>.png`. |

## 7. Message-passing temporal dynamics evaluation

| Skill | Location | What it does |
|---|---|---|
| `simulate_message_passing` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Seeds each channel with its real measured `peak_z`, then runs a degree-normalized linear diffusion (`h(t+1) = α·h(t) + (1−α)·D⁻¹Wh(t)`) over the graph's co-activation mesh for several steps. Models how the graph's *static* structure alone would spread a real starting condition — not a synthetic one. |
| `evaluate_message_passing` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Spatially (cross-channel) Pearson-correlates each simulated step against the recording's *real* measured z-score at the matching elapsed time. This is the actual "temporal dynamic evaluation": a real, checkable claim, not a demonstration — on `sEEG-HFOs-8.edf` it honestly shows the static graph does **not** reproduce the recording's real subsequent dynamics well (correlation ~0.05–0.35 after step 0). |
| `plot_message_passing` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Small-multiples of the simulated network state at each propagation step, shared colour scale, one `layout` at a time (see `_seizure_graph_layout`). Carries a `fig.legend()` for the node/edge markers and a boxed caption explaining what each panel shows. |
| `plot_message_passing_layouts` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Renders the same `simulate_message_passing` run — signal propagation from source to `PEAK` — in every layout `plot_seizure_graph_layouts` offers (`radial`/`spring`/`circular`/`shell`), one file each: `<edf-name>_message_passing_<layout>.png`. |
| `plot_message_passing_validation` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Correlation-vs-elapsed-time line plot: the validation result itself, as a figure, with a boxed caption interpreting the correlation scale. |

## 8. Orchestration

| Skill | Location | What it does |
|---|---|---|
| `run_edf` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Runs the full pipeline for one EDF against one `montage_reference` (`"none"` or `"bipolar"`): writes the montage structure unconditionally, resolves the event (tiers 1–3, unaffected by montage — it reads text annotations, not signal), runs `analyse_brain_process` against the selected reference, and — when channels were found involved — every visualization and evaluation skill above. Returns one `EdfRunResult` (named fields, not a positional tuple). |
| `compare_montages` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Runs `run_edf` once per requested montage reference (default both), each to its own `<stem>/<montage_reference>/` subdirectory, so every downstream skill is compared against the *same* recording and event. |
| `summarize_montage_comparison` | [`edf_workflow.py`](src/extreme_event_agent/edf_workflow.py) | Reduces a `compare_montages` result to one row per montage: candidate count, involved channel count, likely initiators, co-activation mesh edge count, message-passing best/mean correlation — what actually differs, not just structure. On `sEEG-HFOs-8.edf`: `none` → 98 involved channels/273 edges/0.62 best correlation; `bipolar` → 82/227/0.68 — bipolar suppresses some shared-reference correlation and fits the observed dynamics slightly better, while both montages agree on the same right-frontal initiators. |
| `EdfRunResult` | [`models.py`](src/extreme_event_agent/models.py) | The typed result of `run_edf`: report, process, the montage dict and which reference was actually analysed (`montage_reference`), the montage file path, every figure path (including the graph and message-passing layout dicts), the GraphML export path, the message-passing evaluation dict, `source_summary`/`source_summary_file` (from `describe_seizure_source`), and both event-resolution fallbacks. |
| `main` (`seeg-event-agent` CLI) | [`cli.py`](src/extreme_event_agent/cli.py) | Processes a `.npy` array, one EDF, or every EDF in a directory. For EDF input, runs `compare_montages` (`--montages`, default `none,bipolar`) and writes `analysis.json` + all figures per montage subdirectory, plus `montage_comparison.json` when more than one montage ran; prints a one-line summary per montage of which event tier resolved. |

## 9. Notebook skills (not in the installable package)

| Skill | Location | What it does |
|---|---|---|
| Five-method extreme-event ensemble | [`sEEG_extreme_event_detector_colab.ipynb`](sEEG_extreme_event_detector_colab.ipynb) | Time-domain features, Dynamic Time Warping, Detrended Fluctuation Analysis, Discrete Wavelet Transform energy, and Kuramoto phase synchronization (delta/theta/alpha/beta/gamma), each robustly standardized and combined into one ensemble. **Apriori-free** by design. |
| EDF-annotation peak location (notebook copy) | [`sEEG_extreme_event_detector_colab.ipynb`](sEEG_extreme_event_detector_colab.ipynb), Section 3b | Self-contained mirror of `find_annotated_event`/`_cluster_seizure_annotation`, so the notebook needs no dependency on the installed package. |
| Per-method known-peak scoring | [`sEEG_extreme_event_detector_colab.ipynb`](sEEG_extreme_event_detector_colab.ipynb), Section 10 | For each of the five methods *and* the combined ensemble: score and recording-wide percentile at the EDF-annotated peak specifically — quantifies "how would this method alone have predicted the real event". |
| Known-peak comparison figures | [`sEEG_extreme_event_detector_colab.ipynb`](sEEG_extreme_event_detector_colab.ipynb), Sections 10/11/11b | Marks the EDF-annotated peak (teal) on the whole-recording ensemble/method plots and renders the strongest-window and known-peak trace+heatmap views side by side via one shared `visualize_event_window` helper. |
| Per-window `db4` wavelet correlation graphs | [`sEEG_temporal_wavelet_graph_colab.ipynb`](sEEG_temporal_wavelet_graph_colab.ipynb) | Builds one sparse NetworkX graph per 2-second window from thresholded, top-*k*-pruned wavelet-coefficient correlations; saves the full temporal sequence as PyTorch tensors. The event window it highlights (`KNOWN_EVENT_INTERVAL`) is the one containing the EDF-annotated peak, not an apriori guess. |

## 10. What is deliberately *not* a skill here

- Nothing in this repo diagnoses. Every detector, graph, and evaluation
  produces **candidates for expert review**, explicitly labelled by
  provenance (expert / file-annotation / blind-statistical), never a
  clinical claim.
- No skill tunes its thresholds against a known answer. `select_seizure_event`,
  the blind ensemble, and `simulate_message_passing` are all validated
  *against* `sEEG-HFOs-8.edf`'s known peak after the fact, never fitted to it.
