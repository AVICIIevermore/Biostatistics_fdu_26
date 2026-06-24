# MMMD Experiments on MNIST, BloodMNIST, PathMNIST, and NCT-CRC

Date: 2026-05-27

## Goal

Use image representations from raw pixels, frozen foundation encoders (`DINOv2`, `CLIP`), and task-trained CNNs together with MMMD-style two-sample tests to study power, Type-I error, and sample-size efficiency on several biomedical image datasets.

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

### 5. CLIP versus DINOv2 versus raw pixel on the mixed-population task

Comparison target:

- `A = (0,1,3)`
- `B = (0,1,6)`
- same balanced sampling design as above
- same test statistic `GAUSS5`

CLIP power:

| noise | power |
| ---: | ---: |
| 0.0 | 0.980 |
| 0.2 | 0.848 |
| 0.4 | 0.993 |
| 0.6 | 0.937 |
| 0.8 | 0.637 |
| 1.0 | 0.315 |

CLIP Type-I for `A vs A`:

| noise | type1 |
| ---: | ---: |
| 0.0 | 0.046 |
| 0.2 | 0.049 |
| 0.4 | 0.044 |
| 0.6 | 0.071 |
| 0.8 | 0.069 |
| 1.0 | 0.063 |

CLIP Type-I for `B vs B`:

| noise | type1 |
| ---: | ---: |
| 0.0 | 0.034 |
| 0.2 | 0.039 |
| 0.4 | 0.028 |
| 0.6 | 0.042 |
| 0.8 | 0.042 |
| 1.0 | 0.063 |

Interpretation:

- CLIP is clearly useful on this BloodMNIST task, and it strongly outperforms raw pixels in power at low and moderate noise
- compared with DINOv2, CLIP is broadly competitive at low-to-medium noise but falls off faster at heavy noise (`0.8` and `1.0`)
- Type-I error remains close to nominal overall, but CLIP is a bit less stable than DINOv2 on the `A vs A` null at `sigma = 0.6, 0.8, 1.0`
- for this task, the ranking is roughly `DINOv2 >= CLIP >> raw pixel` over most of the practically interesting noise range

Result files:

- `yao/results/bloodmnist224_clip_mix013_vs_016_power_sharedpool/noisy_embedding_summary.csv`
- `yao/results/bloodmnist224_clip_mix013_type1_sharedpool/noisy_embedding_summary.csv`
- `yao/results/bloodmnist224_clip_mix016_type1_sharedpool/noisy_embedding_summary.csv`
- `yao/results/bloodmnist224_mixed_rawpixel_dinov2_clip_gauss5/bloodmnist_mixed_rawpixel_vs_dinov2_vs_clip_gauss5.png`

### 6. Sample-size effect at fixed noise `sigma = 0.6`

Goal:

- compare how power changes with sample size under a fixed moderate noise level
- keep the same mixed-population task `A = (0,1,3)` versus `B = (0,1,6)`
- use lighter Monte Carlo settings to shorten runtime

Setting:

- fixed noise `sigma = 0.6`
- sample sizes `30, 60, 90, 120, 150`
- `n_outer = 4`, `n_inner = 20`, `B_boot = 100`

Power summary:

| sample size | raw pixel | DINOv2 | CLIP |
| ---: | ---: | ---: | ---: |
| 30 | 0.0625 | 0.5125 | 0.5000 |
| 60 | 0.4125 | 0.8875 | 0.7750 |
| 90 | 0.7125 | 0.9625 | 0.8875 |
| 120 | 0.8875 | 1.0000 | 0.9625 |
| 150 | 0.9875 | 1.0000 | 1.0000 |

Interpretation:

- all three methods improve with sample size, which is exactly the pattern we would hope to see
- DINOv2 has the best small-sample efficiency in this experiment
- CLIP is very close to DINOv2 and is much stronger than raw pixels at `n = 30, 60, 90`
- raw pixels eventually catch up as sample size grows, but need substantially more observations to reach the same power
- this is good evidence that learned embeddings improve statistical efficiency for MMMD testing on this biological image task

Type-I summary:

- DINOv2 stays near nominal across the sweep
- raw pixels remain slightly conservative
- CLIP is acceptable overall but somewhat noisier in the smallest-sample regime, especially for `A vs A`

Result files:

- `yao/results/bloodmnist224_sample_size_sigma06/sample_size_summary.csv`
- `yao/results/bloodmnist224_sample_size_sigma06/bloodmnist_mixed_sample_size_comparison.png`

### 7. BloodMNIST CNN baseline versus frozen foundation embeddings

Goal:

- add a task-trained CNN baseline on the same BloodMNIST-224 mixed-population task
- compare `dechao`-style learned CNN features against `raw pixel`, `CLIP`, and `DINOv2`
- keep the MMMD test fixed at `GAUSS5`

CNN training setup:

- dataset: BloodMNIST-224 official `train / val / test`
- model: small CNN
  - `Conv(3,32) -> ReLU -> MaxPool`
  - `Conv(32,64) -> ReLU -> MaxPool`
  - global average pooling
  - `FC(64 -> 128) -> ReLU -> FC(128 -> 8)`
- checkpoint chosen by best validation accuracy
- training device: `MPS`
- final classifier metrics:
  - best validation accuracy: `0.7132`
  - test accuracy: `0.7053`

Testing setup:

- mixed-population task `A = (0,1,3)` vs `B = (0,1,6)`
- use the learned `final_fc128` representation (`128`-d)
- six noise levels: `0, 0.2, 0.4, 0.6, 0.8, 1.0`
- shared noisy pool with `n_rep = 10`
- `sample_size = 90`, `n_inner = 100`, `B_boot = 500`

CNN power:

| noise | power |
| ---: | ---: |
| 0.0 | 0.993 |
| 0.2 | 0.990 |
| 0.4 | 0.978 |
| 0.6 | 0.960 |
| 0.8 | 0.957 |
| 1.0 | 0.917 |

CNN Type-I for `A vs A`:

| noise | type1 |
| ---: | ---: |
| 0.0 | 0.022 |
| 0.2 | 0.035 |
| 0.4 | 0.033 |
| 0.6 | 0.038 |
| 0.8 | 0.040 |
| 1.0 | 0.055 |

CNN Type-I for `B vs B`:

| noise | type1 |
| ---: | ---: |
| 0.0 | 0.011 |
| 0.2 | 0.017 |
| 0.4 | 0.017 |
| 0.6 | 0.027 |
| 0.8 | 0.025 |
| 1.0 | 0.021 |

Interpretation:

- on this BloodMNIST mixed task, the task-trained CNN is the strongest representation among all methods tried so far
- its power stays above `0.91` even at the heaviest noise level `1.0`
- both null scenarios remain well controlled, with Type-I error mostly below or near `0.05`
- compared with the earlier frozen-embedding results, the ranking on BloodMNIST-224 becomes roughly:
  - `CNN > DINOv2 > CLIP > raw pixel`
- this is a useful counterpoint to the PathMNIST result: even on `224 x 224` biomedical images, a task-trained small CNN can still beat frozen general-purpose encoders when trained directly on the target distribution

Result files:

- `yao/models/bloodmnist_cnn/classification_metrics.csv`
- `yao/results/bloodmnist224_cnn_finalfc_mix013_vs_016_power_sharedpool/noisy_embedding_summary.csv`
- `yao/results/bloodmnist224_cnn_finalfc_mix013_type1_sharedpool/noisy_embedding_summary.csv`
- `yao/results/bloodmnist224_cnn_finalfc_mix016_type1_sharedpool/noisy_embedding_summary.csv`
- `yao/results/bloodmnist224_mixed_raw_clip_dino_cnn_gauss5/bloodmnist_mixed_raw_clip_dino_cnn_gauss5.png`

### 8. PathMNIST-28 sample-size alignment against dechao's CNN baselines

Task:

- use the same PathMNIST-28 overlapping-mixture setting as dechao
- power alternative: `mix635_vs_mix835`
- matched null: `mix635_vs_mix635_null`
- compare frozen `DINOv2` and `CLIP` embeddings with dechao's raw-pixel and CNN baselines

Setting:

- dataset: official PathMNIST-28 `test` split only for testing
- sample sizes for power: `30, 60, 90, 120, 150`
- sample sizes for Type-I: `60, 120`
- `n_outer = 10`, `n_inner = 500`, `B_boot = 500`, `alpha = 0.05`
- no additive image noise in this PathMNIST line; the comparison is purely about sample-size power

Frozen-embedding power results:

| sample size | DINOv2 | CLIP |
| ---: | ---: | ---: |
| 30 | 0.0534 | 0.1166 |
| 60 | 0.2506 | 0.1760 |
| 90 | 0.5926 | 0.3232 |
| 120 | 0.8900 | 0.5152 |
| 150 | 0.9904 | 0.7172 |

Frozen-embedding Type-I results:

| sample size | DINOv2 | CLIP |
| ---: | ---: | ---: |
| 60 | 0.0038 | 0.0542 |
| 120 | 0.0034 | 0.0288 |

Interpretation:

- on PathMNIST-28, both frozen foundation-model embeddings are weaker than dechao's best task-trained CNN baselines in power
- `DINOv2` is consistently stronger than `CLIP` on this task, especially at medium and large sample sizes
- `CLIP` is only modestly useful here and tracks closer to raw-pixel MMMD than to the stronger CNN baselines
- `DINOv2` eventually becomes very strong by `n = 150`, but still trails the best CNN methods at smaller sample sizes
- Type-I error is controlled for both methods; `DINOv2` is extremely conservative, while `CLIP` is closer to the nominal 0.05 level
- this suggests that for very small biomedical images (`28 x 28`), a task-trained small CNN remains better matched to the data than large frozen natural-image encoders

Result files:

- `yao/results/pathmnist28_dinov2_alignment/pathmnist_embedding_summary.csv`
- `yao/results/pathmnist28_clip_alignment/pathmnist_embedding_summary.csv`
- `yao/results/pathmnist28_vs_cnn_comparison/pathmnist_sample_size_vs_cnn.png`

### 9. NCT-CRC first-pass sample-size experiment without added noise

Task:

- dataset: `NCT-CRC-HE-100K`
- mixed populations:
  - `A = (NORM, STR, LYM)` i.e. labels `(6, 7, 3)`
  - `B = (TUM, STR, LYM)` i.e. labels `(8, 7, 3)`
- no additive image noise
- compare `raw pixel`, `CLIP`, and `DINOv2`

Setting:

- class-balanced sampling
- sample sizes `30, 60, 90, 120, 150`
- lighter first-pass Monte Carlo:
  - `n_outer = 4`
  - `n_inner = 20`
  - `B_boot = 100`
- for compute control, the test pool uses up to `600` test images per class

Power summary:

| sample size | raw pixel | DINOv2 | CLIP |
| ---: | ---: | ---: | ---: |
| 30 | 0.1250 | 0.0375 | 0.0125 |
| 60 | 0.1125 | 0.1875 | 0.0625 |
| 90 | 0.1750 | 0.4750 | 0.2000 |
| 120 | 0.2375 | 0.8750 | 0.5000 |
| 150 | 0.3000 | 0.9875 | 0.7500 |

Interpretation:

- on this larger pathology dataset, `DINOv2` becomes clearly stronger than both `CLIP` and raw pixels
- `CLIP` also improves with sample size, but is consistently weaker than `DINOv2`
- raw pixels remain the weakest representation throughout the sweep
- unlike BloodMNIST, here the foundation embedding looks much better matched to the visual task than raw pixels

Result files:

- `yao/results/nctcrc_sample_size_firstpass/sample_size_summary.csv`
- `yao/results/nctcrc_sample_size_firstpass/nctcrc_sample_size_comparison.png`

### 10. lrj covariance-stabilized MMMD on NCT-CRC + DINOv2

Goal:

- compare the original multi-kernel Gaussian MMMD baseline (`GEXP-5`)
- against lrj's improved `NEW-MMMD` with covariance stabilization and kernel pruning
- keep the image representation fixed to `DINOv2`

Task:

- same NCT-CRC mixed-population task as above
- sample sizes `30, 60, 90, 120, 150`
- first formal comparison:
  - `n_outer = 4`
  - `n_inner = 20`
  - `B_boot = 100`

Power summary:

| sample size | GEXP-5 | NEW-MMMD |
| ---: | ---: | ---: |
| 30 | 0.1000 | 0.0125 |
| 60 | 0.1875 | 0.1500 |
| 90 | 0.5625 | 0.5375 |
| 120 | 0.9000 | 0.9125 |
| 150 | 0.9625 | 0.9750 |

Interpretation:

- `NEW-MMMD` is more conservative at very small sample size
- by medium and large sample size, it matches or slightly exceeds `GEXP-5`
- its clearest benefit is numerical:
  - raw covariance condition numbers drop from around `10^6 - 10^7`
  - to around `10^4`
- this makes lrj's method a meaningful improvement at the testing layer, especially for high-dimensional learned embeddings

Result files:

- `yao/results/nctcrc_lrj_dinov2_formal/lrj_formal_summary.csv`
- `yao/results/nctcrc_lrj_dinov2_formal/lrj_sample_size_comparison.png`

### 11. NCT-CRC task-trained CNN baseline

Goal:

- add a task-trained CNN representation on the same NCT-CRC mixed-population task
- compare it directly with `raw pixel`, `CLIP`, and `DINOv2`

CNN training setup:

- dataset: full `NCT-CRC-HE-100K` with stratified `train / val / test` split from metadata
- model: small CNN
  - `Conv(3,32) -> ReLU -> MaxPool`
  - `Conv(32,64) -> ReLU -> MaxPool`
  - global average pooling
  - `FC(64 -> 128) -> ReLU -> FC(128 -> 9)`
- best validation accuracy: `0.8115`
- test accuracy: `0.8124`
- training device: `MPS`

Testing setup:

- same mixed populations `A = (6,7,3)` and `B = (8,7,3)`
- representation: learned `final_fc128`
- no added noise
- same first-pass sample-size settings as above
- test pool again limited to `600` images per class for compute consistency

Power summary:

| sample size | CNN |
| ---: | ---: |
| 30 | 0.1000 |
| 60 | 0.2875 |
| 90 | 0.5750 |
| 120 | 0.7375 |
| 150 | 0.8750 |

Interpretation:

- on NCT-CRC, the CNN is clearly stronger than raw pixels and CLIP
- it is also stronger than `DINOv2` at `n = 30, 60, 90`
- but at larger sample sizes `120` and `150`, `DINOv2` overtakes it
- so for this task, the picture is more nuanced than BloodMNIST:
  - `CNN` seems slightly better in the smaller-sample regime
  - `DINOv2` becomes the strongest method once sample size is large enough
- the first-pass null results are mostly acceptable, though `type1_a` around `n = 120` is somewhat high (`0.0875`) and should be rechecked in a heavier repeat setting

Result files:

- `yao/models/nctcrc_cnn/classification_metrics.csv`
- `yao/results/nctcrc_cnn_sample_size_firstpass/cnn_sample_size_summary.csv`
- `yao/results/nctcrc_sample_size_with_cnn/sample_size_summary.csv`
- `yao/results/nctcrc_sample_size_with_cnn/nctcrc_sample_size_comparison.png`

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

6. CLIP is a strong second foundation-model baseline.
   It confirms that the advantage is not unique to one encoder family, though DINOv2 appears a bit more noise-robust on this task.

7. Learned embeddings help most in the smaller-sample regime.
   The sample-size sweep suggests that raw pixels can eventually recover with enough data, but DINOv2 and CLIP reach high power with fewer samples.

8. Task-trained CNNs remain very competitive, and sometimes dominant, on biomedical image data.
   On BloodMNIST-224 the CNN baseline is currently the strongest method overall, and on PathMNIST-28 the CNN baselines also outperform the frozen foundation encoders.

9. Frozen foundation-model embeddings are not guaranteed to beat trained CNNs on tiny biomedical images.
   On PathMNIST-28, the task-trained CNN baselines remain stronger than both DINOv2 and CLIP, especially at small and medium sample sizes.

10. On larger pathology images, the story becomes dataset- and sample-size-dependent.
   On NCT-CRC, DINOv2 is much stronger than CLIP and raw pixels, but the task-trained CNN is competitive and even stronger at smaller sample sizes.

11. Method-layer improvements and representation-layer improvements can be studied separately.
   The lrj covariance-stabilized `NEW-MMMD` changes the testing layer while keeping the representation fixed, and it clearly improves numerical conditioning on high-dimensional embeddings.

## Recommended Next Steps

1. Repeat the sample-size sweep at one heavier noise level such as `sigma = 0.8`.

2. Try a harder BloodMNIST pair or mixture to see whether DINOv2's advantage over CLIP widens.

3. If runtime permits, compare `GAUSS5` with one or two additional paper baselines under the same sample-size setup.

3. Try a harder BloodMNIST pair or mixture, especially classes involving `immature granulocyte (3)` and `neutrophil (6)`.

4. If needed for the final report, add confidence bands or standard-error bars to all summary plots.
