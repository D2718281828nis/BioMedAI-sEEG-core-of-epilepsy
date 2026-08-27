# `electrode_registration/`

A candidate electrode-to-DICOM registration pipeline for `sEEG-HFOs-8.edf`'s
MRI, built entirely from **the recording's own montage naming, the MRI's
own signal-void geometry, and a clinician's hand-drawn shaft sketch** —
no CT, no manually placed per-shaft fiducial file, no numeric coordinate
table anywhere in this dataset.

## Why this is a *candidate* pipeline, not a localization tool

`multimodal_approach/README.md`'s "Why this, and not full electrode
localization" already explains why that package stays at hemisphere
granularity: the standard SEEG contact-localization tools (iELVis, GARDEL,
SEEGA, BrainQuake) all expect a CT (unambiguous for metal) or a manually
placed per-shaft fiducial file, and this dataset has neither. This package
exists anyway, at a request to build one, with that limitation made
explicit and load-bearing throughout rather than papered over — every
number below is either (a) directly measured from the DICOM geometry
itself, or (b) an **internal-consistency cross-check** between independent
but approximate sources, never a validated 3-D localization accuracy claim.

The three sources this package actually has:

1. **The EDF montage's own channel naming** (`extreme_event_agent.edf_workflow.parse_contact_name`)
   — exact contact counts per shaft, straight from the recording.
2. **`dataset/истинное положение.pdf`** — a clinician's orientation sketch:
   RadiAnt-viewer screenshots (a trial-licensed tool; every page carries its
   own "You have N days left in your trial period" watermark) with one
   circle hand-drawn per shaft, plus a table of shaft length and contact
   count. **It has no numeric 3-D coordinates**, and several of its own
   screenshots carry the software's own "Warning: Interpolated Image, all
   values are approximate" — so only the length/contact-count table is
   transcribed here (`reference_geometry.py`), never the circled regions
   themselves.
3. **The MRI's own signal-void (artifact) mask** — `multimodal_approach.structural_anomaly._artifact_mask`,
   already built and tested for a different purpose (excluding electrode
   artifact from structural-anomaly scoring), reused here as raw material
   for candidate shaft geometry.

## What it computes

`contact_detection.detect_shaft_candidates` — connected components of the
artifact mask, grouped into candidate shaft chains via a distance-pruned
proximity graph (same threshold + top-k-per-node discipline
`multimodal_approach.structural_graph.build_structural_anomaly_graph`
already uses for anomaly clusters, applied here to raw artifact blobs), each
refit to one 3-D line by PCA. **Checked, not assumed**: an unconstrained
sweep at several link distances (4-8 mm) consistently finds **70-98
candidate clusters of at least 15 voxels — far more than the 12 shafts this
patient's montage documents.** That gap is reported by the metrics below,
not hidden by tuning parameters until the count happens to come out to 12.

`registration.register_shaft_candidates` — a Hungarian-optimal (`scipy.optimize.linear_sum_assignment`)
best-guess match of the top-N-by-mass candidates per hemisphere (N = the
number of documented shafts on that side — 7 right, 5 left) to those
documented shafts, by length similarity alone. **This is a plausibility
ranking, never a verified identification** — nothing here confirms
candidate `cand_3` really is shaft `PM` rather than some other shaft of
similar length on the same side.

`registration.assign_contact_positions` — places each matched shaft's
documented contact count evenly along its candidate's fitted line.
Contact numbering (`PM_contact_1`, `PM_contact_2`, ...) is an **arbitrary
but fixed convention** (the fitted line's own PCA sign) — which end is the
real contact 1 (entry) versus the deepest contact is not recoverable from
the artifact mask alone, and this numbering must never be read as matching
the EDF's own contact numbers (`EEG PM1`, `EEG PM2`, ...).

`frame_reference.check_frame_of_reference` — reads `FrameOfReferenceUID`
directly from every DICOM series (not inferred from the affine) and checks
they agree. **On this dataset they do**: T1 (`t1_mprage_tra_p2_iso`) and T2
(`t2_space_TR_p2_iso`) share the identical UID
(`1.3.12.2.1107.5.2.19.45833.1.20231011110841464.0.0.0`) — a real, checked
fact confirming both series were acquired in the same physical frame, so a
coordinate computed on T1's grid is valid on T2's without a separate
registration step. (`SeriesInstanceUID`/`StudyInstanceUID` differ between
the two series — an anonymization artifact, the same one
`dicom_viewer.viewer.group_series`'s own docstring already notes for
`SeriesInstanceUID` — expected, and irrelevant to spatial validity.)

`registration.extract_contact_roi` — crops the T1 volume (and its artifact
mask) around one candidate position, in physical units per axis (not
assumed isotropic).

`registration.contacts_outside_volume` — flags any assigned position whose
voxel index falls outside the T1 array's own bounds. The one check here
needing no reference table and no plausibility threshold at all.

`metrics.compare_montage_to_reference` — per-shaft contact-count agreement
between the EDF montage (exact) and the PDF table (approximate). Needs no
MRI at all. **On `sEEG-HFOs-8.edf`, all 12 shafts agree exactly** — real,
checked evidence the PDF and this specific recording describe the same
implant, even though neither supplies a coordinate.

`metrics.compute_registration_metrics` — the requested `recall`/`precision`/`coverage`,
defined precisely because none of them can honestly mean "against verified
3-D truth" here:

- **`recall`** — fraction of the 12 documented shafts whose matched
  candidate's length falls within 50% relative error of that shaft's
  documented length ("plausible").
- **`precision`** — fraction of *matched* registrations that are
  plausible (a matched-but-implausible pairing "used up" a shaft slot
  without actually resembling it).
- **`coverage`** — fraction of the EDF montage's own total contact count
  that ended up with *some* assigned candidate position, however
  approximate, versus none at all.
- **`contacts_outside_volume`** — always well-defined, needs no threshold.

## Run it

```bash
python -m electrode_registration.run_electrode_registration \
  --dicom-dir dataset/MRI-with-electrodes/DCM --edf dataset/sEEG-HFOs-8.edf \
  --output electrode_registration_result
```

Writes `electrode_registration_result.json` (frame-of-reference check,
montage-vs-PDF comparison, every candidate, every registration, every
metric) and `electrode_registration_overview.png` — axial/coronal/sagittal
DICOM slices with every detected candidate drawn as a faint grey line
(intentionally cluttered — 70-98 of them, shown honestly rather than
cropped out) and the assigned contact positions on top, coloured by
hemisphere, solid for a length-plausible match and hollow for a
matched-but-implausible one.

## Results on `sEEG-HFOs-8.edf`

- `FrameOfReferenceUID`: **all match** (both series share one physical frame).
- Montage-vs-PDF contact-count agreement: **12/12 shafts exact.**
- Candidates detected: **91** (against 12 documented shafts).
- `recall = 0.75` (9/12), `precision = 0.75` (9/12 of matched registrations
  are length-plausible), `coverage = 0.76`.
- `contacts_outside_volume`: **0** — every assigned position, plausible or
  not, lands inside the T1 volume.
- Visually (`electrode_registration_overview.png`), the assigned positions
  for the 9 plausible shafts trace clean, roughly parallel depth-electrode
  trajectories converging from skull toward the brain's interior on both
  sides — the qualitative shape a reviewer would expect real SEEG shafts to
  have, even though nothing here *proves* these specific candidates are the
  real shafts.
- The three implausible matches (`FP`, `SA`, `PA`, all right-hemisphere)
  had relative length errors of 51-61% — genuinely borderline, not wildly
  off, consistent with a real shaft candidate whose own fragmentation
  (see `contact_detection.py`'s docstring) makes its fitted length an
  unreliable proxy for the true implanted length, rather than evidence the
  match itself is nonsense.

## Honest limits

- **No verified 3-D ground truth exists for this dataset at all.** Every
  "error"/"precision"/"recall"/"coverage" number above is an
  internal-consistency check between independent, approximate sources —
  never a measurement against a known-correct position. Do not report any
  of these numbers as localization *accuracy*.
- **Shaft identity is a best guess, not a finding.** `register_shaft_candidates`
  picks the length-closest candidate per shaft; a similarly-sized but wrong
  candidate would be picked with equal confidence. Nothing downstream
  (`assign_contact_positions`, ROI extraction, `object_model/` — not wired
  in) should treat a `ShaftRegistration.matched=True` entry as confirmed.
- **Contact numbering direction is arbitrary.** `<shaft>_contact_1` is
  whichever end the fitted line's own PCA happened to sign positive — not
  necessarily the recording's own contact 1.
- **The artifact mask was built for a different job.** `_artifact_mask`
  (in `multimodal_approach/structural_anomaly.py`) was tuned to flag
  signal-void voxels for *exclusion* from anomaly scoring, not to cleanly
  segment individual shafts — hence 70-98 candidates against 12 real
  shafts. Reusing it here is legitimate (same underlying physics: metal is
  a signal void on MR) but its parameters were never tuned for this task.
- **No brain-shift correction, same as `multimodal_approach`.** A
  post-implant scan's tissue near the shafts is physically displaced from
  where a pre-implant scan would show it; this package makes no attempt to
  correct for that, same limitation `multimodal_approach/README.md`
  already documents for the structural-anomaly channel.
- **This package is not wired into `object_model/`.** It is deliberately
  kept a leaf, standalone experiment — its outputs are not consumed by
  `verify_against_annotation`, `build_object_model_graph`, or any figure
  elsewhere in this repository. Promoting any of its numbers into those
  pipelines would need the "matched" assumption above to actually be
  verified first.
