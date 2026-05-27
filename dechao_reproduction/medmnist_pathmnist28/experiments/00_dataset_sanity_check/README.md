# Phase 1: PathMNIST Dataset Sanity Check

Purpose: inspect the official PathMNIST-28 data file, save split shapes, class counts, sampling feasibility, and a small example montage.

This experiment does not train the CNN and does not run MMMD.

Run from the repository root with:

```bash
conda run -n cv-hw2 python dechao_reproduction/medmnist_pathmnist28/experiments/00_dataset_sanity_check/python/dataset_sanity_check.py
```

Outputs are written under `Results/` and `logs/` inside this experiment directory.
