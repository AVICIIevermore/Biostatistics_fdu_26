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

### PathMNIST Overlapping Mixture Alternative

The PathMNIST overlapping mixture alternative compares two class-balanced mixtures that share lymphocytes and smooth muscle while replacing normal colon mucosa with colorectal adenocarcinoma epithelium.

### Shared-Class Disjoint Sampling

Shared-class disjoint sampling means that, within a single two-sample test, images from any class appearing in both groups are sampled without overlap across the two groups.

### PathMNIST Image Intensity Scale

The PathMNIST image intensity scale is RGB pixel intensity represented as floating-point values in `[0, 1]` for both raw-pixel testing and CNN input.

### Frozen Classification Representation

A frozen classification representation is a CNN representation learned only from supervised class labels on the training split and selected only by validation classification accuracy before any two-sample testing is run.

### Center-Shift Source Holdout Pool

The center-shift source holdout pool is the source-domain PathMNIST subset reserved only for MMMD testing after being excluded from CNN training and checkpoint selection.

### Center-Shift External Pool

The center-shift external pool is the official PathMNIST test split used only for source-to-external MMMD testing and external-domain null checks.

### Outer Repetition

An outer repetition is one independent Monte Carlo draw of two samples from the planned two-sample distributions.

### Inner Repetition

An inner repetition is an independent two-sample test nested inside an outer repetition batch, so the actual number of tests per cell is `outer_repetitions * inner_repetitions`.

### Group Size

Group size is the number of observations in each sample of a two-sample test, written as `|X| = |Y| = n`.

### Bootstrap Replication

A bootstrap replication is one multiplier-bootstrap draw used to calibrate the MMMD cutoff for a fixed two-sample test.

### Pilot Rejection Rate

A pilot rejection rate is an empirical rejection rate based on a small number of outer repetitions and should be interpreted only for coarse trends rather than small numerical differences.
