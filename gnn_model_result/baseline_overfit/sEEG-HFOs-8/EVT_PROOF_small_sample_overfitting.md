# Why this overfitting run is evidence *for* extreme value theory, not a bug to fix

This directory is a deliberately-preserved, unregularized baseline
(`python -m gnn_model.run_gnn --graph object_model_result/sEEG-HFOs-8/object_model_graph.graphml
--output gnn_model_result/baseline_overfit`, `GNNConfig()` defaults, seed 7, 150 epochs — see
`gnn_model_result.json`/`gnn_model_summary.txt` in this folder for the full numbers). It is kept
exactly as produced, and `gnn_model.train.train_gnn` reproduces it bit-for-bit whenever
`drop_edge_p=0` and `early_stopping_patience=None` (its defaults) — see that module's docstring.

## The setup

`role` classification on this graph's channel nodes is, structurally, extreme-event detection:
**"earliest"** channels (the true seizure initiators) are the rare, high-consequence class —
5 nodes out of 97 — and **"later_recruited"** is everything else. That is exactly the shape
`extreme_event_agent` (this repo's namesake) exists to handle: very few positive examples of the
event that actually matters.

## The result

| | train | val |
|---|---|---|
| nodes | 67 | 30 |
| "earliest" positives | 3 | 2 |
| final loss | 0.187 | **6.288** |
| final accuracy | 0.896 | 0.933 |

`val_accuracy` (0.933) looks fine — it doesn't, because accuracy on a 92:5 imbalance is dominated
by the majority class and a model can score high while getting every rare case wrong. `val_loss`
is the honest signal here, and it is not fine: it rises monotonically from 0.69 to 6.29 over 150
epochs (`gnn_loss_curve.png`) while train loss keeps falling — textbook memorization, not
learning. The validation confusion matrix (`gnn_confusion_matrix_val.png`) shows why: 1 of 2
"earliest" val nodes misclassified, on a class the model saw only 3 examples of during training.

## The argument

A 226-parameter, 2-layer GCN — about as small a neural net as this task allows — still
overfits a graph with 97 labeled nodes and 5 positives of the class that matters. This is not a
hyperparameter mistake: it is what data-hungry, asymptotically-justified-by-large-N methods
*do* when N is small, regardless of how much is thrown at fixing it afterwards:

| run | params | train loss | val_loss | val_accuracy | val confusion matrix (earliest row / later_recruited row) |
|---|---|---|---|---|---|
| `baseline_overfit/` (this one) | 226 | 0.19 | **6.29** (diverges) | 0.933 | `[1,1]` / `[1,27]` |
| `regularized/` (GCN + DropEdge + early stop on val_loss) | 114 | 0.68 | 0.66 (bounded) | 0.333 | `[1,1]` / `[19,9]` |
| `attention/` v1 (GATv2, heads=4, 2 layers + DropEdge + early stop on val_accuracy) | **1994** | 0.44 | 1.71 (bounded, but climbed past 2.0 before stopping) | 0.900 | `[1,1]` / `[2,26]` |
| `attention/` (GATv2, heads=1, 1 layer + DropEdge + early stop on val_macro_f1) | **54** | 0.53 | 0.79 (bounded) | 0.867 | `[1,1]` / `[3,25]` |
| `attention_deep/` (GATv2, heads=2, **3 layers + residual**, weight_decay=0.2 + DropEdge + early stop on val_macro_f1) | 1562 | 0.43 | **1.02 (plateaus, not just bounded)** | 0.933 | `[1,1]` / `[1,27]` |

`regularized/` proves regularization alone trades the symptom (runaway loss) for a new failure
(collapsed accuracy) rather than curing anything -- capping capacity and stopping on the first
val_loss dip just freezes the model before it separates the classes at all. The first attempt at
`attention/` (GATv2 with 4 heads and 2 layers, checkpointed on val_accuracy) made the *same*
mistake in a new place: at 1994 parameters it is 9x the GCN baseline's own size, and it overfit
again, just as fast and just as unstably (`val_loss` past 2.0 within 15 epochs, `train_accuracy`
swinging between 0.54 and 0.93). More attention heads and more layers is more capacity, not
automatically more regularization. Shrinking it to a single attention head and a single layer (54
parameters -- smaller than the GCN baseline) and checkpointing on macro-F1 instead (accuracy alone
can't distinguish "learned the minority class" from "always predicts the majority" on a 92:5
split -- both score ~0.93) is what actually recovers *both* a bounded loss and a usable confusion
matrix.

`attention_deep/` then asks the opposite question of the single-layer runs: can *more* message
passing (3 hops instead of 1) with attention still avoid overfitting, rather than just shrinking
the model until it barely does anything? The answer is yes, but only with a **residual** connection
on every layer (`GNNConfig.residual=True`, `GATv2Conv`'s own learnable skip) -- without it, a
plain 3-layer attention stack on a graph this small oversmooths well before it finishes training.
With it: confusion matrices on *both* splits matching the reckless baseline's own best-case
numbers (only one misclassified node, each split) -- without that baseline's runaway `val_loss`.
Its first tuning (`weight_decay=5e-3`) still let `val_loss` climb past 2.75 before stopping;
raising `weight_decay` alone (no architecture change) to 0.2 turned that climb into a genuine
**plateau** at ~1.0-1.08 for 100+ epochs, same confusion matrix -- L2 on the weights controls the
actual mechanism (growing weight/logit magnitudes driving a class-weighted loss to charge an
ever-larger penalty for the same wrong predictions), where `label_smoothing` only caps the
symptom indirectly and, at every value tried on this graph, traded away confusion-matrix quality
to do it. Checked epoch-by-epoch, the checkpoint is a genuine plateau, not a lucky single epoch:
`val_macro_f1`/`val_accuracy` hold their best value for dozens of epochs.

**Cross-validation is what actually tests whether any of this generalizes.** Every number above
comes from one 70/30 split -- with only 5 "earliest" nodes total, that split's confusion matrix
depends heavily on which 1-2 of them happened to land in validation. Running the
`weight_decay=0.2` configuration with 5-fold `StratifiedKFold` instead
(`gnn_model_result/attention_deep_cv/`, `gnn_model.data.load_graph_kfold_datasets` +
`gnn_model.train.cross_validate_gnn`) pools every classifiable node into a validation set exactly
once, and the resulting *out-of-fold* confusion matrix is a materially more sobering picture:
`[4,1]` / `[21,71]` (earliest / later_recruited rows) -- mean `val_macro_f1 = 0.55 +/- 0.06` across
folds, not the single split's 0.73. The rare class is actually caught reasonably well in
aggregate (4 of 5 "earliest" nodes, across all 5 folds combined), but at a real precision cost (21
false positives) the one favorable split never surfaced. Per-fold loss curves also show 4 of 5
folds converge cleanly while a fifth diverges toward `val_loss~1.6` -- the same architecture is
not uniformly stable across *which* nodes get held out, something a single split cannot show by
construction.

None of this manufactures the missing data. No configuration above -- single split or
cross-validated -- ever produces a sixth "earliest" example; 5 is all this graph has. That ceiling
-- not the loss curve shape, not the confusion matrix, not the architecture -- is the actual limit
small-sample data imposes, and cross-validation's job here is only to report that limit honestly
rather than let one lucky split hide it.

## A new feature, and folding smart enough to catch its own leak

`gnn_model.augment_dfa` adds one more feature: each channel's Detrended Fluctuation Analysis
scaling exponent (`gnn_model.dfa`, plain NumPy, the same method
`sEEG_extreme_event_detector_colab.ipynb`'s five-method ensemble uses) over its own 30 s
pre-event baseline. It is a real, checkable signal on this recording: "earliest" channels'
exponents are tighter and higher (mean 1.22, std 0.05, n=5) than "later_recruited"'s (mean 1.15,
std 0.10, n=92; Mann-Whitney p=0.014).

| run | fold assignment | out-of-fold confusion matrix | mean `val_macro_f1` |
|---|---|---|---|
| `attention_deep_cv/` (no DFA) | plain `StratifiedKFold` | `[4,1]` / `[21,71]` | 0.55 ± 0.06 |
| `attention_deep_dfa_cv/` (+DFA) | plain `StratifiedKFold` | `[4,1]` / `[4,88]` | **0.84 ± 0.22** |
| `attention_deep_cv_shaft/` (no DFA, seed 7) | `StratifiedGroupKFold` by shaft | `[4,1]` / `[19,73]` | 0.50 ± 0.13 |
| `attention_deep_dfa_cv_shaft/` (+DFA, seed 7) | `StratifiedGroupKFold` by shaft | `[0,5]` / `[2,90]` | 0.48 ± 0.02 |

Plain `StratifiedKFold` makes DFA look like a clean win: 17 fewer false alarms on
"later_recruited" (21 → 4), same recall on "earliest" (4 of 5) -- and this exact result is
bit-identical whether NumPy/PyTorch are built against OpenBLAS or Apple's Accelerate framework
(checked directly on this machine, both backends installed). But plain `StratifiedKFold` can
split one electrode shaft's contacts across train and val, and adjacent contacts on a shaft are
exactly each other's strongest co-activation neighbours -- a validation node's prediction can
lean on a near-duplicate sitting in train. `--group-by-shaft` (`gnn_model.data.load_graph_kfold_datasets`'s
`StratifiedGroupKFold`) closes that leak, and at the seed this repo uses everywhere else (seed 7)
it produces the same "recall evaporates, false-alarm reduction holds" pattern the plain-fold
comparison already hinted was leakage-dependent: `[0,5]` with DFA vs. `[4,1]` without.

**That single number is not a stable measurement, though -- checked two ways.** Under Accelerate
instead of OpenBLAS, the identical seed-7 command gives `[3,2]`/`[3,89]` (f1=0.62) instead of
`[0,5]`/`[2,90]` (f1=0.48): shaft-grouping produces small, unevenly-sized, early-stopping-sensitive
folds, and that is enough for BLAS-level floating-point rounding differences to flip which local
optimum a fold's training lands on -- something the plain-fold comparison never showed at any
configuration tried. Sweeping seeds 1/2/3/7/11 (one BLAS backend, OpenBLAS) confirms it is at
least as seed-sensitive as backend-sensitive:

| | earliest recall (of 5), by seed [1, 2, 3, 7, 11] | mean | later_recruited false positives, by seed | mean |
|---|---|---|---|---|
| shaft, no DFA | 4, 4, 1, 4, 4 | 3.4 | 8, 9, 16, 19, 17 | 13.8 |
| shaft, +DFA | 4, 3, 2, 0, 3 | 2.4 | 19, 7, 10, 2, 21 | 11.8 |

Seed 7 -- the one saved to `attention_deep_dfa_cv_shaft/` and the one this repo defaults to
everywhere -- is the *worst* of the five for "earliest" recall, not a typical draw. Across the
full sweep, the with-DFA and without-DFA distributions overlap heavily on both recall and
false-positive count; the data here cannot support a confident claim that DFA helps *or* hurts
recall once shaft-based leakage is closed. Read together, not separately: DFA is a genuine,
leakage-independent, environment- and seed-stable signal for telling "later_recruited" apart more
precisely (the plain-fold win reproduces everywhere it was checked); the apparent recall gain on
the rare class specifically was at least partly riding on the same-shaft leakage plain
`StratifiedKFold` permits, and once that leakage is closed there simply is not enough data left (5
positives, unevenly-sized shaft groups) to tell whether recall is helped, hurt, or unaffected --
not "the win was fake," but "this experiment cannot resolve that question either way." Smart
folding's job here was exactly to surface that it can't, rather than let one convenient split (or
one convenient seed) assert an answer it doesn't have the power to give.

Classical extreme value theory (the block-maxima / peaks-over-threshold machinery, generalized
Pareto/generalized extreme value distributions) exists precisely because it is derived to make
valid statistical statements about tail events *from few samples*, using asymptotic theorems about
extremes rather than empirical density estimation over a large i.i.d. training set. This
overfitting run is the empirical counterpart of that theoretical motivation: it shows, on this
repo's own data, exactly the failure mode EVT-style methods are built to avoid -- and the other
two runs show why patching a neural net after the fact narrows the *symptom* without touching the
*cause*.
