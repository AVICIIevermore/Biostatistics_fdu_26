# Phase 4: PathMNIST Sample-Size Power Experiment

Scenario: `mix635_vs_mix835`, the PathMNIST overlapping mixture alternative.

Configuration:

- Dataset split: official test split only
- Sampling: class-balanced, disjoint for shared classes 3 and 5
- Group sizes: `n=30,60,90,120,150`
- Outer repetitions: `10`
- Bootstrap replications: `B_boot=500`
- Alpha: `0.05`
- Methods: `raw_pixel_gaussian5`, `cnn_final_fc128_gaussian5`, `cnn_multilayer_single_gaussian`, `cnn_multilayer_gaussian15`
- Shared checkpoint: `../../shared_model/models/pathmnist_cnn_checkpoint.pt`

Outputs live under this experiment's `Results/` directory.

## Run command

From the repository root, after the shared CNN checkpoint is final:

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n cv-hw2 python dechao_reproduction/medmnist_pathmnist28/experiments/03_sample_size_power_mix635_vs_mix835/python/pathmnist_power.py --sample-sizes 30 60 90 120 150 --outer-repetitions 10 --b-bootstrap 500 --alpha 0.05 --force
```

The script appends completed `sample_size × outer_iter` blocks to CSV files and can resume if interrupted when run without `--force`.
