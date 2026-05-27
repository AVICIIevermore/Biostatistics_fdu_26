# MNIST DINOv2 MMMD Power Experiment

Date: 2026-05-26

## Goal

Run the MNIST additive-noise power experiment with the same six noise levels used in the paper/dechao reproduction, but replace CNN/raw-pixel features with frozen DINOv2 embeddings. Compare the resulting MMMD rejection rate with dechao's CNN embedding summary.

## Important Caveat

The local repository did not contain the MNIST test gzip files, and network access was restricted during this run. This experiment therefore used the available MNIST train split files:

- `dechao_reproduction/mnist_additive_noise_reproduction/data/train-images-idx3-ubyte.gz`
- `dechao_reproduction/mnist_additive_noise_reproduction/data/train-labels-idx1-ubyte.gz`

The experimental structure, labels, noise levels, and testing pipeline follow the intended MNIST power setup, but the split is train rather than test.

## Configuration

- Encoder: `facebook/dinov2-base`
- Embedding dimension: 768
- Feature storage: `embeddings.npy` plus `metadata.csv`
- Noise levels: `0, 0.2, 0.4, 0.6, 0.8, 1`
- Power task: `{1, 2, 3}` vs `{1, 2, 8}`
- Outer repetitions: `2`
- Images per digit label before per-test sampling: `500`
- Per-test sample size: `100` from X and `100` from Y
- Inner tests per outer repetition/noise level: `100`
- Bootstrap repetitions per test: `100`
- Alpha: `0.05`
- MMMD method: `GAUSS5`
- R parallel cores: `8`
- Seed: `20260525`

This is a medium-scale run for checking the full pipeline. The original/dechao-scale setting is larger, especially in `outer repetitions`, `inner tests`, and `bootstrap repetitions`.

## Commands

Generate noisy DINOv2 embeddings:

```bash
.venv/bin/python yao/scripts/extract_mnist_noise_embeddings.py \
  --images dechao_reproduction/mnist_additive_noise_reproduction/data/train-images-idx3-ubyte.gz \
  --labels dechao_reproduction/mnist_additive_noise_reproduction/data/train-labels-idx1-ubyte.gz \
  --x-labels 1,2,3 \
  --y-labels 1,2,8 \
  --max-per-label 500 \
  --noise-levels 0,0.2,0.4,0.6,0.8,1 \
  --n-rep 2 \
  --seed 20260525 \
  --encoder dinov2 \
  --local-files-only \
  --batch-size 64 \
  --output yao/data/embeddings/mnist_dinov2_dechao_power_medium
```

Run MMMD testing:

```bash
Rscript yao/scripts/run_noisy_embedding_testing.R yao/configs/mnist_dinov2_dechao_power_medium.R
```

Plot against dechao's CNN summary:

```bash
.venv/bin/python yao/scripts/plot_mnist_dinov2_vs_cnn.py \
  --cnn-summary dechao_reproduction/faster_mnist_cnn_main_power_summary/Results/mnist_cnn_main_power_summary.csv \
  --dinov2-summary yao/results/mnist_dinov2_dechao_power_medium/noisy_embedding_summary.csv \
  --output-dir yao/results/mnist_dinov2_dechao_power_medium
```

## Results

| noise_sigma | method | power_mean | power_se | n_rep |
| ---: | --- | ---: | ---: | ---: |
| 0.0 | GAUSS5 | 0.995 | 0.005 | 2 |
| 0.2 | GAUSS5 | 0.830 | 0.000 | 2 |
| 0.4 | GAUSS5 | 0.225 | 0.045 | 2 |
| 0.6 | GAUSS5 | 0.055 | 0.025 | 2 |
| 0.8 | GAUSS5 | 0.060 | 0.020 | 2 |
| 1.0 | GAUSS5 | 0.065 | 0.015 | 2 |

## Outputs

- Embeddings: `yao/data/embeddings/mnist_dinov2_dechao_power_medium/embeddings.npy`
- Embedding metadata: `yao/data/embeddings/mnist_dinov2_dechao_power_medium/metadata.csv`
- Raw MMMD results: `yao/results/mnist_dinov2_dechao_power_medium/noisy_embedding_results.csv`
- DINOv2 summary: `yao/results/mnist_dinov2_dechao_power_medium/noisy_embedding_summary.csv`
- Combined DINOv2/CNN summary: `yao/results/mnist_dinov2_dechao_power_medium/mnist_dinov2_vs_cnn_power_summary.csv`
- Comparison plot PNG: `yao/results/mnist_dinov2_dechao_power_medium/mnist_dinov2_vs_cnn_power_comparison.png`
- Comparison plot PDF: `yao/results/mnist_dinov2_dechao_power_medium/mnist_dinov2_vs_cnn_power_comparison.pdf`

## Quick Interpretation

DINOv2-base features give very high power at low noise (`sigma=0` and `0.2`), but the rejection rate drops quickly after `sigma=0.4`. In this medium run, the DINOv2 curve is competitive at the lowest noise levels but less robust than dechao's raw-pixel Gaussian-5 baseline and the stronger multilayer CNN embedding baselines at moderate/high noise.
