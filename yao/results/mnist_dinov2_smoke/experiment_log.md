# MNIST DINOv2 MMMD Smoke Experiment

Run time: 2026-05-26 16:32:08 CST

## Purpose

Run the full pipeline once on a small MNIST sample:

1. Load local MNIST `idx` gzip files from `dechao_reproduction`.
2. Convert images to frozen DINOv2 embeddings.
3. Run the paper-style MMD/MMMD tests in R.
4. Estimate Type-I error and power on a tiny smoke-test scale.

This is only an end-to-end smoke test, not a final quantitative experiment.

## Environment

Working directory:

```text
/Users/yao/Documents/fudan/MATH/生物统计学/pj/code/Biostatistics_fdu_26
```

Python environment:

```bash
uv sync
```

Relevant packages were installed through the project-level `pyproject.toml`, including `torch`, `transformers`, `medmnist`, `numpy`, `pandas`, `scikit-learn`, and `matplotlib`.

R packages used:

```text
kernlab, MASS, psych
```

## Data

Source files:

```text
dechao_reproduction/mnist_additive_noise_reproduction/data/train-images-idx3-ubyte.gz
dechao_reproduction/mnist_additive_noise_reproduction/data/train-labels-idx1-ubyte.gz
```

Digits kept:

```text
1, 2, 3, 8
```

Images embedded:

```text
60 per digit, 240 total
```

Embedding output:

```text
yao/data/embeddings/mnist_dinov2_smoke.csv
```

Embedding file shape:

```text
240 rows, 768 embedding dimensions plus id/split/label columns
```

## Embedding Command

```bash
uv run python yao/scripts/extract_mnist_embeddings.py \
  --images dechao_reproduction/mnist_additive_noise_reproduction/data/train-images-idx3-ubyte.gz \
  --labels dechao_reproduction/mnist_additive_noise_reproduction/data/train-labels-idx1-ubyte.gz \
  --keep-labels 1,2,3,8 \
  --max-per-label 60 \
  --encoder dinov2 \
  --batch-size 16 \
  --output yao/data/embeddings/mnist_dinov2_smoke.csv
```

Model:

```text
facebook/dinov2-base
```

Embedding dimension:

```text
768
```

## Testing Config

Config file:

```text
yao/configs/mnist_dinov2_smoke.R
```

Key parameters:

```text
sample_size = 30
n_reps = 5
B_boot = 50
alpha = 0.05
methods = GAUSS1, GAUSS5
```

Null / Type-I setting:

```text
Split digit 1 embeddings into two independent samples.
null_labels = c(1)
```

Power setting:

```text
alternative_x_labels = c(1, 2, 3)
alternative_y_labels = c(1, 2, 8)
```

## Testing Command

```bash
Rscript yao/scripts/run_embedding_testing.R yao/configs/mnist_dinov2_smoke.R
```

## Outputs

```text
yao/results/mnist_dinov2_smoke/embedding_mmmd_replicates.csv
yao/results/mnist_dinov2_smoke/embedding_mmmd_summary.csv
```

Summary:

| setting | method | reject_rate | se |
|---|---:|---:|---:|
| power | GAUSS1 | 0.4 | 0.2449489743 |
| power | GAUSS5 | 0.6 | 0.2449489743 |
| type1 | GAUSS1 | 0.0 | 0.0 |
| type1 | GAUSS5 | 0.0 | 0.0 |

Because `n_reps = 5` and `B_boot = 50`, these numbers should be read only as a pipeline check.

## Notes

- The MedMNIST automatic download failed due to an SSL EOF error from Python `urllib`.
- A direct Zenodo download for `pathmnist_224.npz` was avoided because the file is very large for a smoke test.
- A direct Zenodo download for `pathmnist.npz` was also too slow in this environment.
- The smoke test therefore used local standard MNIST files already present in `dechao_reproduction`.
- During this run, `extract_medmnist_embeddings.py` was patched so it creates the MedMNIST root directory automatically.
- `run_embedding_testing.R` was patched to load config variables from an explicit environment; the previous `exists(..., inherits = FALSE)` check failed inside `vapply`.
