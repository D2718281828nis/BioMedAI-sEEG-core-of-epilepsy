# `multimodal_approach/`

A third, independent cross-check on top of `extreme_event_agent`'s
spatial-recruitment detection and `model/`'s reservoir-residual detection:
does the post-implant MRI's **own structural asymmetry** — computed with no
apriori event time, no CT, no diffusion MRI, no normative population, and
no manually placed electrode fiducials — agree with which hemisphere the
timeseries independently implicates?

## Why this, and not full electrode localization

The obvious "combine DICOM and timeseries" move is to localize every
electrode contact in 3-D and read off exactly which anatomical structure
each channel sits in. That is a real, well-studied problem (iELVis, GARDEL,
SEEGA, BrainQuake, ...) but every one of those tools either expects a CT
(much less ambiguous than MRI for metal) or a manually placed per-shaft
fiducial file, neither of which exists for this dataset — `dataset/MRI-with-electrodes/DCM`
holds only two post-implant MR series (`t1_mprage_tra_p2_iso`,
`t2_space_TR_p2_iso`), no CT, no pre-implant baseline. Building a fully
automated, *unverified* per-contact localizer here would produce numbers
with no way to check them, which fails the standard the rest of this repo
holds itself to (see `MANIFEST.md` section 10: nothing here is asserted
without an audit trail or a stated confidence).

So this package stays at the granularity the data actually supports:
**hemisphere-level**. The montage's own naming convention already encodes
laterality — `extreme_event_agent.edf_workflow.is_right_frontal` treats
unprimed shafts (`PM`, `CC`, ...) as right-hemisphere and primed shafts
(`PM'`, `CC'`, ...) as their contralateral, left-hemisphere counterpart —
so every channel's hemisphere is already known with zero additional
apriori input. What was missing was an independent *structural* signal to
compare it against.

## What it computes

`structural_anomaly.py` — self-referential hemispheric asymmetry mapping
(see its module docstring for the full method): head/artifact segmentation,
true-patient-space midline search, per-voxel asymmetry index, self
median/MAD z-scoring, repeated independently on T1 and T2 (registered onto
the T1 grid using the DICOM files' own `ImageOrientationPatient`/
`ImagePositionPatient` — no external registration tool needed since both
series share one scanner session's patient coordinate frame), keeping only
voxels where both contrasts agree in sign. Reduced to one number per
hemisphere: mean/max |anomaly z-score|.

This is the *single-subject* analogue of the published unsupervised
MRI-anomaly-detection literature (train on a normative population,
reconstruction error at inference — not reproducible here with one scan and
no population reference) and of `model/`'s own reservoir plant, which does
the same trick in time instead of space: fit on what's "normal" (there, the
pre-event baseline; here, the brain's own mirror symmetry), then flag where
reality diverges from that expectation.

`find_top_anomaly_clusters` reduces the full volume to a short, ranked list
of connected anomaly clusters (ranked by total mass — size × strength
together, not one extreme voxel) — this is what `structural_anomaly_overview.png`
now points at: axial/coronal/sagittal all sliced through the single
strongest cluster's peak voxel at once, crosshair-marked in every view,
instead of three unrelated geometric mid-slices. It applies a "cerebrum
floor" (`min_z_percentile`, default the 55th percentile of the head mask's
own patient-Z range) for a concrete, checked reason: with no floor at all,
**every one of the top five clusters on this dataset sat at skull-base/mastoid
level**, not in the cerebrum — see "Honest limits" below.

`extreme_event_prior.py` — annotates every tier-3 blind candidate
(`analysis.json["detection"]["events"]`, written by `seeg-event-agent`) with
its own channel hemisphere balance, combines it with the MRI's
hemisphere-asymmetry summary into a `structural_alignment_score`, and
compares two **independently computed** picks: `select_seizure_event`'s own
temporal/statistical pick (channel spread, then duration, then score — see
its docstring; this file never touches that ranking or any detection
threshold) against a purely spatial pick (which candidate the MRI-derived
hemisphere score alone would prefer). Whether they agree is the actual
cross-check — the same "two independent methods, do they land in the same
place" pattern this repo already uses for bipolar-vs-referential montage
agreement and blind-detector-vs-annotation comparison.

`run_multimodal.py` ties both together and, when the EDF's own annotated
(tier-2) event is available, reports — for review only, never fed back into
any threshold — whether either pick's time window actually contains it.

## Run it

```bash
python -m multimodal_approach.run_multimodal \
  --dicom-dir dataset/MRI-with-electrodes/DCM \
  --agent-output seeg_agent_output/sEEG-HFOs-8 \
  --output multimodal_result
```

Requires `seeg-event-agent` to have already been run on the same recording
(`seeg_agent_output/<edf-name>/<montage_reference>/analysis.json` must
exist) — see the top-level `README.md`.

Writes to `--output`: `structural_anomaly.npz` (combined anomaly volume plus
masks, on the T1 grid), `structural_anomaly_overview.png` (three orthogonal
mid-slices, T1 grayscale with the anomaly map overlaid), `hemisphere_summary.json`,
and `structural_prior_report.json` (one entry per montage reference found).

## Honest limits

- **Hemisphere granularity only.** This does not claim to know which
  *contact* is anomalous, only which *side*. Extending to lobe- or
  contact-level would need real 3-D electrode coordinates — the harder,
  currently-unbuilt problem discussed above.
- **Asymmetry is a lesion proxy, not a lesion detector.** A structurally
  abnormal region does not have to be asymmetric (a bilateral or midline
  abnormality would be invisible to this method by construction), and an
  asymmetric region is not necessarily pathological (normal anatomical
  asymmetry exists). There is no lesion mask for this subject to measure
  sensitivity/specificity against — treat `hemisphere_summary` as a lead,
  exactly like this repo already labels `detected_event`.
- **The implant itself is asymmetric** (different shafts placed on each
  side), so the artifact mask matters: an unmasked signal void would read
  as a huge, spurious "anomaly" on whichever side has more hardware, not
  anatomy. `structural_anomaly.py` excludes flagged signal-void voxels
  before scoring for exactly this reason — but the mask is a heuristic
  (dark relative to local neighbourhood), not a verified electrode
  segmentation, and could miss or over-mask some artifact.
- **No brain-shift or bias-field correction.** A post-implant scan's
  cortical surface near the shafts is physically displaced from where a
  pre-implant scan would show it (brain shift) — normally corrected by
  registering to a pre-implant baseline, which does not exist here.
- **T1/T2 agreement reduces, but does not eliminate, false positives.**
  Both are affected by the same real implant and the same real anatomy, so
  correlated non-anomalous effects (e.g. a strong bias-field gradient
  present in both acquisitions) could still agree in sign without being a
  genuine structural anomaly.
- **The hemisphere comparison is not yet stable, and this was checked, not
  assumed.** Running this on `sEEG-HFOs-8.edf`'s MRI with `brain_margin_mm=0`
  (raw Otsu head mask — includes scalp/skull) gave right-hemisphere mean
  |anomaly| = 0.41 vs. left = 0.33 (right higher, matching the rest of this
  repo's independently-derived right-frontal hypothesis); the *same* data
  with `brain_margin_mm=4` (a few millimetres stripped off the mask
  boundary, the current default — see `_erode_mm` in `structural_anomaly.py`)
  gave right = 0.14 vs. left = 0.17 — **the sign flipped** from a change in
  masking alone, not in the underlying data. That is a real finding about
  this method's current reliability, not a bug being papered over: it means
  the point estimate should not be trusted on its own, only the *mechanism*
  (per-hemisphere summaries computed from an MRI-native signal, cross-checked
  against channel hemisphere composition) should be treated as demonstrated.
  Next step before trusting a specific hemisphere call: report a margin- or
  bootstrap-based sensitivity range alongside the point estimate rather than
  one number, and/or replace the Otsu+erosion mask with a real skull-strip
  (e.g. SynthStrip/HD-BET) that separates scalp/skull from brain by tissue
  type instead of by a fixed distance from an intensity-threshold boundary.
- **The single strongest cluster was skull base, not brain — until filtered,
  and that filter is a heuristic, not a segmentation.** Before
  `find_top_anomaly_clusters` had any Z floor, all five top-ranked clusters
  sat at patient Z between roughly -76 mm and -40 mm — the 10th-to-40th
  percentile of this head mask's own Z range, i.e. skull base/mastoid/upper
  neck. Mastoid air-cell aeration is routinely asymmetric between sides in
  healthy people for no pathological reason, and it is exactly the kind of
  bony, air-adjacent tissue a uniform millimetre-scale mask erosion does
  least to separate from true anatomy. The current `min_z_percentile=55`
  default was chosen by inspecting this one dataset's own head-mask Z
  distribution (`dataset/MRI-with-electrodes/DCM`'s percentiles), not
  derived from an anatomical atlas — it will need re-checking on any other
  scan's geometry, and it can just as easily discard a genuine low temporal-
  or occipital-pole finding as it discards skull base. Read
  `structural_anomaly_overview.png`'s "best view" as *the clearest cluster
  after a crude height filter*, not as *the most anomalous point in the
  brain* — `top_anomaly_clusters.json` keeps the full ranked list (up to
  five) for review, and re-running with `min_z_percentile=None` recovers the
  unfiltered ranking, skull-base clusters included, for comparison.
