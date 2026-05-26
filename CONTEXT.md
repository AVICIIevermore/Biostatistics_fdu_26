# Context

## Terms

### Baseline

The baseline is the author-style MNIST additive-noise experiment reproduced from the original repository, including both kernel-based tests and the FR graph-based test.

### Smoke Test

A smoke test is a deliberately tiny run whose purpose is only to verify that an experiment pipeline executes end-to-end and writes outputs successfully. It is not treated as a final quantitative result.

### Type-I Error Experiment

A type-I error experiment is a null-distribution experiment where both samples are drawn independently from the same digit pool and perturbed with the same noise level using independent noise realizations.

### Experiment Directory

An experiment directory is a self-contained folder under a reproduction workspace, currently `dechao_reproduction/` for active CNN experiments, that holds the copied code, result files, and notes for one experiment only.

### Penultimate FC-128 Embedding

The penultimate FC-128 embedding is the 128-dimensional ReLU output of the CNN's `Linear(64,128)` layer before the 10-dimensional classifier logits. It should not be called the classifier output.

### Layer1 GAP Embedding

The layer1 GAP embedding is the 32-dimensional vector produced by applying global average pooling to the first convolutional block output after `Conv1 -> ReLU -> MaxPool`.
