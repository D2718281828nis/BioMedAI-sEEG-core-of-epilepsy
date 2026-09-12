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

| run | val_loss | val_accuracy | val confusion matrix (earliest row / later_recruited row) |
|---|---|---|---|
| `baseline_overfit/` (this one) | **6.29** (diverges) | 0.933 | `[1,1]` / `[1,27]` |
| `regularized/` (GCN + DropEdge + early stop on val_loss) | 0.81 (bounded) | 0.667 | `[1,1]` / `[19,9]` |
| `attention/` (GATv2 + DropEdge + early stop on val_accuracy) | 1.71 (bounded) | 0.900 | `[1,1]` / `[2,26]` |

`regularized/` proves regularization alone trades the symptom (runaway loss) for a new failure
(collapsed accuracy) rather than curing anything -- capping capacity and stopping on the first
val_loss dip just freezes the model before it separates the classes at all. `attention/` (see
`gnn_model.model.SeizureGAT`, learned per-neighbour attention instead of GCN's fixed
normalization, checkpointed on val_accuracy rather than the same fragile val_loss) recovers
*both* a bounded loss and a usable confusion matrix -- but note it still catches exactly the same
1 of 2 "earliest" nodes the reckless baseline does; no amount of architecture or training-loop
engineering manufactured a second real example of the rare class that was never there. That
ceiling -- not the loss curve shape -- is the actual limit small-sample data imposes.

Classical extreme value theory (the block-maxima / peaks-over-threshold machinery, generalized
Pareto/generalized extreme value distributions) exists precisely because it is derived to make
valid statistical statements about tail events *from few samples*, using asymptotic theorems about
extremes rather than empirical density estimation over a large i.i.d. training set. This
overfitting run is the empirical counterpart of that theoretical motivation: it shows, on this
repo's own data, exactly the failure mode EVT-style methods are built to avoid -- and the other
two runs show why patching a neural net after the fact narrows the *symptom* without touching the
*cause*.
