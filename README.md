# BioMedAI-sEEG-core-of-epilepsy
Biomakreks of sEEG timeseries analysis to find core of epilepsy as a dynamic process.

## Google Colab notebooks

- Open [`sEEG_EDF_viewer_colab.ipynb`](sEEG_EDF_viewer_colab.ipynb) to inspect
  EDF recording metadata, browse channel signals, and plot power spectra.
- Open
  [`sEEG_temporal_wavelet_graph_colab.ipynb`](sEEG_temporal_wavelet_graph_colab.ipynb)
  to load `dataset/sEEG-HFOs-8.edf`, remove `MKR<i>` marker channels, and convert
  consecutive two-second windows into a temporal graph. Nodes contain `db4`
  discrete-wavelet features, while smart-pruned edges contain wavelet
  correlations. The notebook saves PyTorch tensors and explicitly analyzes the
  known event on channels `CC'4`, `CC'5`, and `CR'5` at 808–810 seconds.
