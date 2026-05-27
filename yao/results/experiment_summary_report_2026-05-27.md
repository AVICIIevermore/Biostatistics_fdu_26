# DINOv2 + MMMD on MNIST and BloodMNIST

Date: 2026-05-27

## Goal

Use frozen `DINOv2-base` embeddings together with the paper's MMMD test (`GAUSS5`) to study power and Type-I error under additive Gaussian image noise.

## Implementation Summary

The final workflow has two stages:

1. Generate noisy image embedding pools once with DINOv2 and save them as `embeddings.npy + metadata.csv`.
2. Reuse those embedding pools in R to run MMMD testing across noise levels.

For MedMNIST/BloodMNIST, the important correction was:

- power can reuse pre-generated noisy pools safely
- Type-I error should not be estimated from a single fixed noisy realization repeatedly resampled many times
- instead, multiple independent noisy pools (`n_rep = 10`) should be generated first
- for null experiments, the two samples should be split from the same label pool (`shared_null=TRUE`)

## Core Scripts

- `yao/scripts/extract_mnist_noise_embeddings.py`
- `yao/scripts/extract_medmnist_noise_embeddings.py`
- `yao/scripts/run_noisy_embedding_testing.R`
- `yao/src/mmmd_functions.R`

## Experiments Completed

### 1. MNIST medium smoke-aligned power experiment

Task:

- `A = {1,2,3}`
- `B = {1,2,8}`

Setting:

- DINOv2-base, 768-d
- six noise levels: `0, 0.2, 0.4, 0.6, 0.8, 1.0`
- medium run: `n_rep = 2`, `sample_size = 100`, `n_inner = 100`, `B_boot = 100`

Result:

| noise | power |
| ---: | ---: |
| 0.0 | 0.995 |
| 0.2 | 0.830 |
| 0.4 | 0.225 |
| 0.6 | 0.055 |
| 0.8 | 0.060 |
| 1.0 | 0.065 |

Interpretation:

- DINOv2 works very well at low noise on MNIST
- power collapses quickly once additive Gaussian noise becomes moderate
- this suggests DINOv2 embeddings are not especially robust for small handwritten digits under this perturbation

Result file:

- `yao/results/mnist_dinov2_dechao_power_medium/noisy_embedding_summary.csv`

### 2. BloodMNIST single-class experiment: `basophil(0)` vs `lymphocyte(4)`

Task:

- power: `0 vs 4`
- Type-I nulls: `0 vs 0`, `4 vs 4`

Setting:

- BloodMNIST 224 test split
- DINOv2-base, 768-d
- six noise levels
- shared noisy embedding pool with `n_rep = 10`
- `sample_size = 100`, `n_inner = 100`, `B_boot = 500`
- null experiments use shared label pools and `replace = FALSE`

Power result:

| noise | power |
| ---: | ---: |
| 0.0 | 1.000 |
| 0.2 | 1.000 |
| 0.4 | 1.000 |
| 0.6 | 1.000 |
| 0.8 | 1.000 |
| 1.0 | 1.000 |

Type-I result for `0 vs 0`:

| noise | type1 |
| ---: | ---: |
| 0.0 | 0.048 |
| 0.2 | 0.057 |
| 0.4 | 0.046 |
| 0.6 | 0.048 |
| 0.8 | 0.038 |
| 1.0 | 0.059 |

Type-I result for `4 vs 4`:

| noise | type1 |
| ---: | ---: |
| 0.0 | 0.041 |
| 0.2 | 0.062 |
| 0.4 | 0.059 |
| 0.6 | 0.049 |
| 0.8 | 0.068 |
| 1.0 | 0.056 |

Interpretation:

- `basophil` and `lymphocyte` are extremely easy to separate in DINOv2 space
- unlike the earlier incorrect single-fixed-pool null experiment, the corrected shared-pool null design keeps Type-I error near `0.05`
- this confirms that the previous inflated Type-I error was mainly an experimental-design artifact, not immediate evidence that MMMD itself was invalid here

Result files:

- `yao/results/bloodmnist224_dinov2_baso_lymph_power_sharedpool/noisy_embedding_summary.csv`
- `yao/results/bloodmnist224_dinov2_baso_type1_sharedpool/noisy_embedding_summary.csv`
- `yao/results/bloodmnist224_dinov2_lymph_type1_sharedpool/noisy_embedding_summary.csv`

### 3. BloodMNIST mixed-population experiment

Task:

- population `A = (0,1,3)` with uniform class mixture
- population `B = (0,1,6)` with uniform class mixture
- power: `A vs B`
- Type-I nulls: `A vs A`, `B vs B`

Important design detail:

- each sample is class-balanced
- `sample_size = 90`, so each class contributes exactly `30` observations

Setting:

- BloodMNIST 224 test split
- DINOv2-base, 768-d
- shared noisy embedding pool over labels `{0,1,3,6}`
- `n_rep = 10`
- `n_inner = 100`, `B_boot = 500`
- balanced sampling enabled in R

Power result for `A vs B`:

| noise | power |
| ---: | ---: |
| 0.0 | 0.922 |
| 0.2 | 0.882 |
| 0.4 | 0.942 |
| 0.6 | 0.976 |
| 0.8 | 0.830 |
| 1.0 | 0.429 |

Type-I result for `A vs A`:

| noise | type1 |
| ---: | ---: |
| 0.0 | 0.053 |
| 0.2 | 0.062 |
| 0.4 | 0.042 |
| 0.6 | 0.042 |
| 0.8 | 0.059 |
| 1.0 | 0.044 |

Type-I result for `B vs B`:

| noise | type1 |
| ---: | ---: |
| 0.0 | 0.050 |
| 0.2 | 0.038 |
| 0.4 | 0.024 |
| 0.6 | 0.025 |
| 0.8 | 0.035 |
| 1.0 | 0.028 |

Interpretation:

- this mixed-population problem is meaningfully harder than `0 vs 4`
- the two populations share labels `0` and `1`, and differ only through `3` versus `6`
- even so, power remains high through noise `0.8`, and only drops sharply at `1.0`
- Type-I error remains controlled for both null scenarios
- this is the strongest evidence so far that the DINOv2 + MMMD pipeline is useful on biologically meaningful image mixtures

Result files:

- `yao/results/bloodmnist224_dinov2_mix013_vs_016_power_sharedpool/noisy_embedding_summary.csv`
- `yao/results/bloodmnist224_dinov2_mix013_type1_sharedpool/noisy_embedding_summary.csv`
- `yao/results/bloodmnist224_dinov2_mix016_type1_sharedpool/noisy_embedding_summary.csv`

### 4. Raw pixel baseline versus DINOv2 on the mixed-population task

Comparison target:

- `A = (0,1,3)`
- `B = (0,1,6)`
- both populations sampled with uniform class balance
- test statistic fixed to `GAUSS5`

Rationale:

- `raw pixel + GAUSS5` is the cleanest representation-level baseline
- `DINOv2 + GAUSS5` isolates the effect of replacing raw pixels with a learned visual embedding while keeping the test itself unchanged

Raw-pixel power:

| noise | power |
| ---: | ---: |
| 0.0 | 0.676 |
| 0.2 | 0.765 |
| 0.4 | 0.763 |
| 0.6 | 0.683 |
| 0.8 | 0.596 |
| 1.0 | 0.515 |

Raw-pixel Type-I for `A vs A`:

| noise | type1 |
| ---: | ---: |
| 0.0 | 0.072 |
| 0.2 | 0.045 |
| 0.4 | 0.028 |
| 0.6 | 0.018 |
| 0.8 | 0.022 |
| 1.0 | 0.015 |

Raw-pixel Type-I for `B vs B`:

| noise | type1 |
| ---: | ---: |
| 0.0 | 0.049 |
| 0.2 | 0.019 |
| 0.4 | 0.017 |
| 0.6 | 0.007 |
| 0.8 | 0.010 |
| 1.0 | 0.011 |

Interpretation:

- on this mixed BloodMNIST task, DINOv2 clearly outperforms raw pixels in power for noise levels `0` through `0.8`
- at the heaviest noise level `1.0`, raw pixels are slightly stronger (`0.515` vs `0.429`)
- both approaches keep Type-I error reasonably controlled, but raw pixels tend to be slightly conservative at moderate and high noise
- overall, DINOv2 gives the better power-Type-I tradeoff on this task, especially in the low-to-moderate noise regime

Result files:

- `yao/results/bloodmnist224_rawpixel_mix013_vs_016_power_gauss5/rawpixel_gauss5_summary.csv`
- `yao/results/bloodmnist224_rawpixel_mix013_type1_gauss5/rawpixel_gauss5_summary.csv`
- `yao/results/bloodmnist224_rawpixel_mix016_type1_gauss5/rawpixel_gauss5_summary.csv`
- `yao/results/bloodmnist224_mixed_rawpixel_vs_dinov2_gauss5/bloodmnist_mixed_rawpixel_vs_dinov2_gauss5.png`

## Main Lessons

1. DINOv2 is not uniformly strong across datasets.
   On MNIST it loses power quickly as noise increases, while on BloodMNIST it remains much more useful.

2. Experimental design matters a lot for Type-I error.
   A single fixed noisy pool repeatedly resampled can produce misleadingly inflated null rejection rates.

3. Shared embedding pools are the right engineering compromise.
   They avoid repeated DINOv2 forward passes while still supporting statistically reasonable power and Type-I experiments when multiple independent noisy pools are pre-generated.

4. Mixed-population testing is feasible and informative.
   The `(0,1,3)` vs `(0,1,6)` experiment is much closer to a realistic two-population biomedical comparison than a simple one-class-vs-one-class task.

5. On the mixed BloodMNIST task, learned representation beats raw pixels.
   DINOv2 + GAUSS5 has substantially higher power than raw pixel + GAUSS5 over most noise levels while still keeping Type-I error under control.

## Recommended Next Steps

1. Plot the three mixed-population curves together:
   power, `A vs A` Type-I, `B vs B` Type-I.

2. Compare DINOv2 with CLIP on the same shared BloodMNIST pools.

3. Try a harder BloodMNIST pair or mixture, especially classes involving `immature granulocyte (3)` and `neutrophil (6)`.

4. If needed for the final report, add confidence bands or standard-error bars to all summary plots.
