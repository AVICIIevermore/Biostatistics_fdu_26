# MedMNIST PathMNIST-28 MMMD Workspace

This workspace contains all PathMNIST-28 assets and results for the biomedical image extension.

## Directory Rules

- `data/`: dataset-level files such as `pathmnist.npz`.
- `shared_model/`: shared CNN training code, logs, and checkpoints. This directory is reused by experiments and is not owned by any single experiment.
- `experiments/`: one folder per experiment. Each experiment owns its own code, logs, and `Results/` outputs.

## Current Experiments

- `experiments/00_dataset_sanity_check/`: Phase 1 dataset shape, class-count, feasibility, and example-image checks only. No CNN training and no MMMD tests.
