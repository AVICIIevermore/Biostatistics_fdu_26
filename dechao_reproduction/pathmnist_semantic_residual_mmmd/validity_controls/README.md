# Semantic/Residual Validity Controls

This folder contains the follow-up validity controls from `plan_next.md`.

The controls reuse the existing PathMNIST CNN checkpoints. No model is retrained.

Planned core outputs:

- `Results/semantic_residual_centered_diagnostics.csv`
- `Results/dim_matched_residual_mmmd_summary.csv`
- `Results/dim_matched_residual_power.png`
- `Results/dim_matched_residual_type1.png`
- `Results/dim_matched_residual_excess_rejection.png`
- `Results/permutation_calibration_center_shift_summary.csv`
- `Results/permutation_vs_bootstrap_type1.png`
- `Results/permutation_vs_bootstrap_power.png`

Raw per-test CSVs are also written for resumability and auditing.
