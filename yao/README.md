# Embedding MMMD Experiments

This folder is self-contained for the planned image-embedding two-sample tests.

## Goal

Convert biomedical images to frozen visual embeddings with CLIP or DINOv2, then run the Mahalanobis aggregated MMD test from Chatterjee and Bhattacharya's kernel two-sample paper.

The statistical testing code is copied/adapted into `src/mmmd_functions.R`, so the experiment does not need to source files from the original reproduction folders.

## Files

- `scripts/extract_medmnist_embeddings.py`: downloads a MedMNIST 2D dataset and writes CLIP/DINOv2 embeddings to CSV.
- `scripts/extract_mnist_embeddings.py`: extracts CLIP/DINOv2 embeddings from local MNIST idx gzip files for smoke tests.
- `scripts/extract_manifest_embeddings.py`: extracts embeddings from a generic image manifest, useful for BBBC021 after metadata preparation.
- `src/mmmd_functions.R`: single-kernel MMD and multi-kernel Mahalanobis MMD with Gaussian multiplier bootstrap.
- `scripts/run_embedding_testing.R`: estimates Type-I error and power from an embedding CSV.
- `scripts/run_noisy_embedding_testing.R`: estimates power or Type-I error from noisy MNIST embeddings stored as CSV or as an `embeddings.npy` plus `metadata.csv` directory.
- `configs/example_medmnist_embedding.R`: example R config.
- `data/embeddings/`: suggested output location for embedding CSV files.
- `results/`: suggested output location for testing summaries.

## Python Setup

The shared `uv` project is configured at `code/Biostatistics_fdu_26/pyproject.toml`.

```bash
cd code/Biostatistics_fdu_26
uv python install 3.12
uv sync
```

## Extract Embeddings

Example with DINOv2 on PathMNIST+ 224:

```bash
cd code/Biostatistics_fdu_26
uv run python yao/scripts/extract_medmnist_embeddings.py \
  --dataset pathmnist \
  --size 224 \
  --split test \
  --encoder dinov2 \
  --batch-size 64 \
  --output yao/data/embeddings/pathmnist_dinov2.csv
```

Example with CLIP:

```bash
uv run python yao/scripts/extract_medmnist_embeddings.py \
  --dataset pathmnist \
  --size 224 \
  --split test \
  --encoder clip \
  --output yao/data/embeddings/pathmnist_clip.csv
```

Use `--max-images 500` for a small smoke test.

For a manually prepared image manifest, for example BBBC021 channel composites:

```bash
uv run python yao/scripts/extract_manifest_embeddings.py \
  --manifest yao/data/bbbc021_manifest.csv \
  --image-root yao/data/bbbc021_images \
  --split test \
  --encoder dinov2 \
  --output yao/data/embeddings/bbbc021_dinov2.csv
```

The manifest must contain `image_path,label`; optional columns are `id,split`.

For repeated noisy MNIST runs, prefer directory output to avoid very wide CSV files:

```bash
uv run python yao/scripts/extract_mnist_noise_embeddings.py \
  --split test \
  --x-labels 1,2,3 \
  --y-labels 1,2,8 \
  --noise-levels 0,0.2,0.4,0.6,0.8,1 \
  --n-rep 10 \
  --encoder dinov2 \
  --output yao/data/embeddings/mnist_dinov2_dechao_power_full
```

Directory outputs contain `embeddings.npy` and `metadata.csv`; use the directory path, without `.csv`, as `embedding_file` in `run_noisy_embedding_testing.R` configs.

For MedMNIST noise experiments where one embedding pool should be reused across
power and Type-I runs, generate a shared pool without X/Y group tags:

```bash
uv run python yao/scripts/extract_medmnist_noise_embeddings.py \
  --npz yao/data/mnist/bloodmnist_224.npz \
  --split test \
  --x-labels 0 \
  --y-labels 4 \
  --pool-only \
  --noise-levels 0,0.2,0.4,0.6,0.8,1 \
  --n-rep 10 \
  --encoder dinov2 \
  --output yao/data/embeddings/bloodmnist224_dinov2_baso_lymph_sharedpool
```

Then reuse that single embedding directory with separate configs for:

- power: `0` vs `4`
- Type-I: `0` vs `0`
- Type-I: `4` vs `4`

## Run Testing

Edit `configs/example_medmnist_embedding.R` so the label groups match the dataset, then run:

```bash
cd code/Biostatistics_fdu_26
Rscript yao/scripts/run_embedding_testing.R yao/configs/example_medmnist_embedding.R
```

The output files are:

- `embedding_mmmd_replicates.csv`: one row per repetition and method.
- `embedding_mmmd_summary.csv`: Type-I error and power estimates.

## Methods

Supported `methods` in the R config:

- `GAUSS1`: single Gaussian MMD with median bandwidth.
- `LAP1`: single Laplace MMD with median bandwidth.
- `GAUSS5`: Mahalanobis MMD with five Gaussian bandwidths around the median.
- `LAP5`: Mahalanobis MMD with five Laplace bandwidths around the median.
- `MIXED`: Mahalanobis MMD with three Gaussian and three Laplace kernels.

## Experimental Interpretation

- `setting == "type1"` splits one `null_labels` pool into two samples. Its reject rate estimates Type-I error.
- `setting == "power"` samples from `alternative_x_labels` and `alternative_y_labels`. Its reject rate estimates power, and Type-II error is `1 - power`.
