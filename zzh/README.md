# ZZH Experiment Workspace

This folder collects the files added to `main` after commit
`c481f1ffafc552c1512f7544284c6fb895248e72` along the active ancestry path.
It is intended as a self-contained index for the numerical extension and
graph-MMMD work that was previously spread across the repository root.

## Source Commits

The relevant commits after `c481f1f` are:

- `354cc9c` Merge remote-tracking branch `origin/main` into `pure-data-test-zzh`
- `56bc806` Delete some unnecessary files
- `957be4b` Modify route
- `064c468` Change readme

## Contents

- `R/`: Core R utilities for data loading, MMMD testing, graph kernels,
  bootstrap utilities, and ROC helpers.
- `experiments/`: Simulation scripts for epsilon sensitivity, variance
  sensitivity, Type-I/ROC checks, graph-MMMD demos, mixture alternatives, and
  diagnostic runs.
- `data/`: Small example data and data-source notes, including the diabetes
  CSV used for real-data testing.
- `results_data/`: Short method note summarizing the numerical extension.
- `run_all_tasks.R`, `run_graph_demo.R`, `run_mnist_cnn_graph_mmmd.R`: Entry
  points for running grouped experiments.
- `plot_graph_mmmd_vs_existing.R`: Plotting script for comparing graph-MMMD
  with existing methods.
- `README_extensions.md`: Extended explanation of the added methods and
  experiment design.

The uv environment files should stay at the repository root. This folder is not
treated as an independent Python project.

## Quick Start

From the repository root:

```bash
Rscript zzh/run_all_tasks.R
```

For a smaller graph-MMMD check:

```bash
Rscript zzh/run_graph_demo.R
```

The scripts assume repository-root execution. If a path-related error appears,
run the corresponding root-level script or update the source paths from `R/` to
`zzh/R/`.
