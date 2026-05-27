# Phase 3: PathMNIST MMMD Smoke Test

Purpose: verify the PathMNIST MMMD pipeline end-to-end before full sample-size power runs.

Scope:

- Scenario: `mix635_vs_mix835`
- Group size: `n=30` (`10` per class per group)
- Sampling: class-balanced and disjoint for shared classes
- Methods: `raw_pixel_gaussian5`, `cnn_final_fc128_gaussian5`
- Uses shared checkpoint: `../../shared_model/models/pathmnist_cnn_checkpoint.pt`

This smoke test is for pipeline validation only; it is not a final quantitative result.
