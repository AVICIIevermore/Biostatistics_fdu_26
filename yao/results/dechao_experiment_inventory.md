# Dechao MNIST Experiment Inventory and CLIP/DINOv2 Alignment Plan

## Main Dechao Experiment Families

### 1. Raw Pixel Additive-Noise Power

Directories:

```text
dechao_reproduction/mnist_additive_noise_reproduction
dechao_reproduction/faster_mnist_additive_noise_reproduction
```

Task:

```text
Power: {1,2,3} vs {1,2,8}
```

Parameters:

```text
noise_levels = 0, 0.2, 0.4, 0.6, 0.8, 1
n.rep = 10
resamp = 100
n.iter = 500
alpha = 0.05
seed = 20260521
methods = LAP MMD, Gauss MMD, LAP MMMD, Gauss MMMD, Mixed MMMD
```

Main outputs:

```text
dechao_reproduction/mnist_additive_noise_reproduction/Results/*.csv
dechao_reproduction/faster_mnist_additive_noise_reproduction/Results/*.csv
```

### 2. Raw Pixel Type-I Error

Directories:

```text
dechao_reproduction/mnist_additive_noise_type1_same_distribution
dechao_reproduction/faster_mnist_additive_noise_type1_same_distribution
```

Tasks:

```text
Type-I: {1,2,3} vs {1,2,3}
Type-I: {1,2,8} vs {1,2,8}
```

Parameters:

```text
noise_levels = 0, 0.2, 0.4, 0.6, 0.8, 1
n.rep = 10
resamp = 100
n.iter = 500
alpha = 0.05
methods = LAP MMD, Gauss MMD, LAP MMMD, Gauss MMMD, Mixed MMMD, FR where available
```

### 3. CNN Embedding Power

Directories:

```text
dechao_reproduction/faster_mnist_cnn_layer1_gaussian5_power
dechao_reproduction/faster_mnist_cnn_layer1_pool2x2_gaussian5_power
dechao_reproduction/faster_mnist_cnn_layer2_gaussian5_power
dechao_reproduction/faster_mnist_cnn_final_embedding_gaussian5_power
dechao_reproduction/faster_mnist_cnn_multilayer_single_gaussian_power
dechao_reproduction/faster_mnist_cnn_multilayer_gaussian15_power
dechao_reproduction/faster_mnist_testset_paired_raw_vs_final_gaussian5_power
dechao_reproduction/faster_mnist_cnn_main_power_summary
```

Task:

```text
Power: {1,2,3} vs {1,2,8}
```

Shared parameters:

```text
noise_levels = 0, 0.2, 0.4, 0.6, 0.8, 1
n.rep = 10
resamp = 100
n.iter = 500
B_boot = n.iter = 500
alpha = 0.05
seed = 20260525
```

CNN methods:

```text
raw_pixel_gaussian5
layer1_gaussian5              # 32-d GAP embedding
layer1_pool2x2_gaussian5      # 128-d pooled embedding
layer2_gaussian5
final_embedding_gaussian5     # 128-d penultimate FC embedding
multilayer_single_gaussian    # 3 kernels, one per layer
multilayer_gaussian15         # 15 kernels, 5 per layer
```

Main combined output:

```text
dechao_reproduction/faster_mnist_cnn_main_power_summary/Results/mnist_cnn_main_power_summary.csv
```

Selected power means from the combined summary:

```text
raw_pixel_gaussian5:
  sigma 0/0.2/0.4/0.6/0.8/1 = 0.9236, 0.8886, 0.8224, 0.6602, 0.3862, 0.1866

final_embedding_gaussian5:
  sigma 0/0.2/0.4/0.6/0.8/1 = 1.0000, 0.9992, 0.7630, 0.1954, 0.1030, ...
```

### 4. CNN Embedding Type-I Error

Directories:

```text
dechao_reproduction/faster_mnist_cnn_type1_123_vs_123_gaussian
dechao_reproduction/faster_mnist_cnn_type1_two_nulls_multilayer_single_gaussian
```

Tasks:

```text
Type-I: {1,2,3} vs {1,2,3}
Type-I: {1,2,8} vs {1,2,8}
```

Parameters:

```text
noise_levels = 0, 0.2, 0.4, 0.6, 0.8, 1  # two-null multilayer experiment
noise_levels = 0, 0.4, 0.8               # broad method comparison
n.rep = 10
resamp = 100
n.iter = 500
B_boot = 500
alpha = 0.05
```

Main outputs:

```text
dechao_reproduction/faster_mnist_cnn_type1_123_vs_123_gaussian/Results/mnist_cnn_type1_summary.csv
dechao_reproduction/faster_mnist_cnn_type1_two_nulls_multilayer_single_gaussian/Results/mnist_cnn_type1_summary.csv
```

## Alignment Plan for CLIP/DINOv2

Use the same MNIST split, noise schedule, sample size, repetitions, bootstrap count, and label groups as Dechao. Replace CNN feature extraction with frozen CLIP or DINOv2 embeddings.

### Power Experiment

Embedding generation:

```bash
cd code/Biostatistics_fdu_26
uv run python yao/scripts/extract_mnist_noise_embeddings.py \
  --split test \
  --x-labels 1,2,3 \
  --y-labels 1,2,8 \
  --noise-levels 0,0.2,0.4,0.6,0.8,1 \
  --n-rep 10 \
  --seed 20260525 \
  --encoder dinov2 \
  --batch-size 64 \
  --output yao/data/embeddings/mnist_dinov2_dechao_power_full
```

Testing:

```bash
Rscript yao/scripts/run_noisy_embedding_testing.R yao/configs/mnist_dinov2_dechao_power_full.R
```

Output:

```text
yao/results/mnist_dinov2_dechao_power_full/noisy_embedding_results.csv
yao/results/mnist_dinov2_dechao_power_full/noisy_embedding_summary.csv
```

### Type-I Experiment

For `{1,2,3}` vs `{1,2,3}`:

```bash
uv run python yao/scripts/extract_mnist_noise_embeddings.py \
  --split test \
  --x-labels 1,2,3 \
  --y-labels 1,2,3 \
  --noise-levels 0,0.2,0.4,0.6,0.8,1 \
  --n-rep 10 \
  --seed 20260525 \
  --encoder dinov2 \
  --batch-size 64 \
  --output yao/data/embeddings/mnist_dinov2_dechao_type1_123_full

Rscript yao/scripts/run_noisy_embedding_testing.R yao/configs/mnist_dinov2_dechao_type1_123_full.R
```

Repeat with `--encoder clip` and corresponding output/config names for CLIP.

## Smoke Test Completed

Completed a small DINOv2 power smoke test with:

```text
n.rep = 1
noise_levels = 0, 0.2
max_per_label = 120
n_inner = 20
B_boot = 20
sample_size = 100
```

Output:

```text
yao/results/mnist_dinov2_dechao_power_smoke/noisy_embedding_summary.csv
```

The smoke embedding has also been converted to directory format at:

```text
yao/data/embeddings/mnist_dinov2_dechao_power_smoke/
```

Smoke result:

```text
sigma=0.0, GAUSS5 power=0.95
sigma=0.2, GAUSS5 power=0.90
```

These smoke numbers are not final; they only verify that the aligned pipeline works.
