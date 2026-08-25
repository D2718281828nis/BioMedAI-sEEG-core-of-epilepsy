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
(see its module docstring for the full method): tissue-based brain
extraction, artifact/CSF exclusion, true-patient-space midline search,
per-voxel asymmetry index, self median/MAD z-scoring, repeated independently
on T1 and T2 (registered onto the T1 grid using the DICOM files' own
`ImageOrientationPatient`/`ImagePositionPatient` — no external registration
tool needed since both series share one scanner session's patient coordinate
frame), keeping only voxels where both contrasts agree in sign. Reduced to
one number per hemisphere: mean/max |anomaly z-score|.

**Brain extraction — excluding skull and CSF, using only the two contrasts
already here.** Both anomaly channels used to score every voxel inside a
fixed-millimetre erosion of a raw Otsu head mask — scalp and skull included,
which the "Honest limits" section below used to flag as a real, checked
reliability problem (a masking-only change flipped which hemisphere scored
higher). `_brain_extract` (see its own docstring for the full method and the
two failure modes an earlier version of it had, each caught only by
inspecting actual voxel counts, not by how the code looked) now builds an
actual skull-strip from T1+T2 signal instead: cortical bone reads as a
near-signal-void on both contrasts, unlike CSF (dark T1, bright T2) or
parenchyma (moderate on both), so "low on both" is a workable per-voxel bone
proxy; solidifying that shell, seeding a component deep inside it, and
growing the seed back out *only within* the shell-subtracted candidate
(never by a global fill-holes, which cannot tell a ventricle from an orbit
that lost its bony roof) strips scalp/skull without an external tool. Its
`bone_percentile`/`closing_iterations` defaults (45th percentile, 4 closing
iterations) were tuned by inspection on this dataset — checked against a
connected-component-size ratio (does the candidate actually disconnect into
a brain-sized piece vs. a piece dwarfed by the rest) and a physically
plausible result (~1.1-1.2 L, the normal adult whole-brain range), not
assumed from the classifier alone. `_csf_mask` then flags CSF (low-T1,
high-T2) *within* the extracted brain and excludes it from anomaly scoring
too — ventricles/sulcal CSF are a leading source of both spurious asymmetry
(ventricle-size asymmetry is a common normal variant) and spurious
heterogeneity (a sharp tissue/CSF edge has genuinely high local variance for
a purely geometric reason — see "Honest limits" for a real example this
still only partially catches).

This is the *single-subject* analogue of the published unsupervised
MRI-anomaly-detection literature (train on a normative population,
reconstruction error at inference — not reproducible here with one scan and
no population reference) and of `model/`'s own reservoir plant, which does
the same trick in time instead of space: fit on what's "normal" (there, the
pre-event baseline; here, the brain's own mirror symmetry), then flag where
reality diverges from that expectation.

**Second, independent channel — local texture heterogeneity.** The
asymmetry channel above is blind *by construction* to anything that is not
lateralized: a bilateral or midline structural change produces zero
left/right difference no matter how real it is (see "Honest limits" below).
`structural_anomaly.py` now also computes, in parallel, a windowed
coefficient-of-variation ("how much does local intensity vary relative to
its own local level") self-referentially z-scored the same median/MAD way,
one-sided (only "more heterogeneous than this brain's own typical tissue" is
the anomalous direction) and T1/T2-agreement-gated the same way (both
contrasts must show elevated heterogeneity at a voxel, not just one). This
is the same category of signal used generally in single-sequence,
no-population tumor/lesion screening (heterogeneity texture analysis, of the
kind used for e.g. pediatric solid-tumor imaging such as neuroblastoma, is
one well-known instance) — a focal structural replacement tends to disrupt
local tissue texture even when it sits on the midline or is symmetric, which
plain asymmetry cannot see. Reported as `combined_heterogeneity`, kept fully
separate from `combined_anomaly` (never merged into one score) so a reader
can always tell which mechanism, or both, is behind a flagged region.

`find_top_anomaly_clusters` reduces the full volume to a short, ranked list
of connected anomaly clusters (ranked by total mass — size × strength
together, not one extreme voxel) — this is what `structural_anomaly_overview.png`
now points at: axial/coronal/sagittal all sliced through the single
strongest cluster's peak voxel at once, crosshair-marked in every view,
instead of three unrelated geometric mid-slices. It used to also apply a
"cerebrum floor" (`min_z_percentile`, discarding any cluster below the 55th
percentile of the head mask's own patient-Z range) for a concrete, checked
reason: with no floor and the old scalp+skull-including mask, every one of
the top five clusters sat at skull-base/mastoid level, not cerebrum. Now
that `_brain_extract` excludes skull/skull-base tissue directly, that
height-based floor is off by default (`min_z_percentile=None`) — re-checked,
not just assumed obsolete: re-enabling it after brain extraction discards
*every* remaining cluster on this dataset, because it is filtering out
genuine low-cerebrum/temporal findings the old floor was never calibrated
for once the neck/skull-base bulk it was measured against is gone. See
`find_top_anomaly_clusters`'s own docstring for the full before/after.

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

It also now writes `structural_anomaly_t1t2_fusion.png`: a second figure,
cropped tight (±45 mm by default) around each channel's own top cluster,
with a **T1/T2 color-fused** anatomical background instead of T1 grayscale
alone — T1 on the red channel, T2 on green+blue, so tissue bright on both
reads near-white, T1-only-bright reads warm/orange, T2-only-bright (most
simple fluid, CSF, edema) reads cyan. `structural_anomaly_overview.png`
answers "how strong, and roughly where"; this figure answers "what does that
spot actually look like on both contrasts at once", the way a reviewer would
flip between T1 and T2 by eye, made simultaneous instead of sequential. One
row per channel (asymmetry, heterogeneity), each sliced through *its own*
top cluster — expect the two rows to point at different locations, since the
two channels are looking for structurally different things.

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

Writes to `--output`: `structural_anomaly.npz` (both channels' volumes plus
masks, on the T1 grid), `structural_anomaly_overview.png` and
`structural_anomaly_t1t2_fusion.png` (see above), `top_anomaly_clusters.json`
and `top_heterogeneity_clusters.json` (one ranked cluster list per channel),
`hemisphere_summary.json` and `heterogeneity_summary.json` (the latter
whole-head, not per-hemisphere — see above), and `structural_prior_report.json`
(one entry per montage reference found; still consumes only the asymmetry
channel's `hemisphere_summary` — the heterogeneity channel has no lateral
signal for `extreme_event_prior.py` to compare against).

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
- **[Superseded by tissue-based brain extraction, kept for the audit
  trail.] The hemisphere comparison was not stable under the old
  scalp+skull-including mask, and that was checked, not assumed.** Running
  this on `sEEG-HFOs-8.edf`'s MRI with `brain_margin_mm=0` (raw Otsu head
  mask — includes scalp/skull) gave right-hemisphere mean |anomaly| = 0.41
  vs. left = 0.33; the *same* data with `brain_margin_mm=4` (a few
  millimetres stripped off the mask boundary, the then-default) gave
  right = 0.14 vs. left = 0.17 — the sign flipped from a change in masking
  alone, not in the underlying data. This is exactly the failure mode
  `_brain_extract` (see "What it computes" above) was built to fix by
  separating scalp/skull from brain by tissue signal instead of by a fixed
  distance from an intensity-threshold boundary — the fix this section
  used to call out as the next step. Re-run with the current tissue-based
  extraction: right = 0.097 vs. left = 0.121 (left higher) — still a real,
  unresolved disagreement with the `brain_margin_mm=0` direction above, and
  still not cross-validated against a margin- or bootstrap-based sensitivity
  range, so the point estimate should still not be trusted on its own; only
  report *which* masking method produced a given number alongside it.
- **[Superseded by tissue-based brain extraction, kept for the audit
  trail.] The single strongest cluster used to be skull base, not brain.**
  Before `find_top_anomaly_clusters` had any Z floor, and before
  `_brain_extract` existed, all five top-ranked clusters sat at patient Z
  between roughly -76 mm and -40 mm — skull base/mastoid/upper neck, not
  cerebrum — masked only by a height-percentile heuristic (`min_z_percentile`,
  see "What it computes" above for why it is now off by default instead).
  With tissue-based brain extraction now excluding that tissue directly, the
  top-ranked cluster on this same dataset is 2037 voxels in the left
  temporal lobe (patient xyz ≈ (51.5, 7.9, -39.0) mm, peak z = +9.53,
  `structural_anomaly_overview.png`/`structural_anomaly_t1t2_fusion.png`) —
  a plausible cerebral location instead of bone. This is *not* a validated
  finding (see the next point below, and "Asymmetry is a lesion proxy, not a
  lesion detector" above) — only evidence that the skull-base failure mode
  specifically is resolved, not that this new top cluster is real pathology.
- **The heterogeneity channel's top cluster used to be a ventricle
  boundary, not a lesion — reduced, not eliminated, by CSF exclusion.**
  `_csf_mask` (see "What it computes" above) was added specifically because,
  before it existed, this channel's top cluster was a
  lateral-ventricle/choroid-plexus CSF boundary (2533 voxels, patient
  xyz ≈ (-4, -18, 12) mm). The reason turned out to be partial-volume, not
  missing coverage: `_csf_mask` classifies core CSF correctly (low-T1,
  high-T2 by percentile), but the highest local heterogeneity sits at the
  *rim* immediately next to that core — a transitional voxel mixing brain
  and CSF signal, neither purely one tissue nor the other, so it dodged the
  CSF test while still reading as sharply heterogeneous. `run_structural_anomaly`
  now dilates the CSF mask by one voxel before using it for score exclusion
  (kept separate from the `csf_mask` field itself, which stays undilated for
  audit) specifically to absorb that rim — checked, not assumed to have
  worked: the top cluster dropped from 1512 voxels to 15, and moved away
  from the ventricle to patient xyz ≈ (13.5, -23.1, 9.1) mm. It did not fully
  disappear — `structural_anomaly_t1t2_fusion.png`'s bottom row still shows
  a thin residual sliver hugging a CSF boundary at a different ventricle
  location, so a real periventricular lesion and a normal ventricle wall are
  still not perfectly distinguished by this channel, only much less
  dominated by the artifact than before. Widening the dilation further would
  trade more of that residual rim against eating into genuine
  periventricular white matter — not attempted here.
