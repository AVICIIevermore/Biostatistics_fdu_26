# Phase 5: PathMNIST Type-I Check

Scenario: `mix635_vs_mix635_null`, the matched null for the PathMNIST overlapping mixture experiment.

Configuration:

- Dataset split: official test split only
- Sampling: class-balanced, X/Y disjoint within each class
- Group sizes: `n=60,120`
- Outer repetitions: `10`
- Bootstrap replications: `B_boot=500`
- Alpha: `0.05`
- Methods: same four Phase 4 methods
- Shared checkpoint: `../../shared_model/models/pathmnist_cnn_checkpoint.pt`

Outputs live under this experiment's `Results/` directory.
