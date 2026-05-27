# MedMNIST PathMNIST-28 MMMD Experiment Plan for Codex

## 0. Purpose of this document

This document records the final agreed plan for the next stage of the biostatistics final project.

The project is based on the paper:

> Chatterjee and Bhattacharya, **Boosting the power of kernel two-sample tests**, *Biometrika*, 2025.

The previous MNIST stage has already reproduced and extended the paper's MNIST two-sample testing setting using raw pixels and CNN-induced embeddings. The next stage is to transfer the same idea to a real biomedical image dataset.

The current task for Codex is:

> Implement the main MedMNIST PathMNIST-28 two-sample testing pipeline using an overlapping mixture distribution setting.

Do **not** implement the advanced extensions yet.

---

## 1. Current project direction

The original paper studies kernel two-sample testing:

\[
H_0: P = Q \quad \text{vs.} \quad H_1: P \neq Q.
\]

It proposes Mahalanobis MMD / MMMD, which aggregates multiple MMD estimates from multiple kernels or bandwidths using a Mahalanobis statistic.

Our extension is:

> Use CNN-induced representations as input spaces for MMMD, and compare raw-pixel MMMD with CNN embedding MMMD on biomedical images.

The previous MNIST experiments already tested:

- raw-pixel Gaussian-5 MMMD;
- CNN final FC-128 Gaussian-5 MMMD;
- CNN multilayer single-Gaussian MMMD;
- CNN multilayer Gaussian-15 MMMD;
- representative Type-I error checks;
- covariance condition-number diagnostics.

Now we move to MedMNIST PathMNIST-28.

---

## 2. Dataset decision

### 2.1 Dataset to use

Use:

> **MedMNIST v2: PathMNIST, 28×28 RGB version**

Official links:

- MedMNIST GitHub: <https://github.com/MedMNIST/MedMNIST>
- MedMNIST project page: <https://medmnist.com/>
- PathMNIST-28 direct `.npz` file: <https://zenodo.org/records/10519652/files/pathmnist.npz?download=1>
- Official dataset metadata file: <https://raw.githubusercontent.com/MedMNIST/MedMNIST/main/medmnist/info.py>

### 2.2 Dataset summary

PathMNIST is a colorectal cancer histology dataset.

Basic metadata:

| Item | Value |
|---|---:|
| Dataset | PathMNIST |
| Image size | 28×28 |
| Channels | 3, RGB |
| Number of classes | 9 |
| Train samples | 89,996 |
| Validation samples | 10,004 |
| Test samples | 7,180 |
| Recommended task | Multi-class classification |

### 2.3 Class labels

Use the official PathMNIST label convention:

| Label | Class name |
|---:|---|
| 0 | adipose |
| 1 | background |
| 2 | debris |
| 3 | lymphocytes |
| 4 | mucus |
| 5 | smooth muscle |
| 6 | normal colon mucosa |
| 7 | cancer-associated stroma |
| 8 | colorectal adenocarcinoma epithelium |

---

## 3. Important experimental decisions

### 3.1 Use PathMNIST only

Use only PathMNIST-28 for the current main line.

Do **not** use:

- BreastMNIST;
- 64×64 / 128×128 / 224×224 MedMNIST versions;
- 3D MedMNIST datasets;
- BBBC021;
- raw original pathology images.

Reason: PathMNIST-28 is already standardized, small, RGB, and close enough to the existing MNIST pipeline while still being a real biomedical image dataset.

### 3.2 Do not run simple class 6 vs class 8

The simple two-class comparison:

\[
P = \{6\}, \quad Q = \{8\}
\]

should be removed from the current plan.

Reason: although it is statistically still a two-sample test, it is too close to a standard binary classification problem and may be less convincing as the main MMMD extension.

### 3.3 Use only the overlapping mixture setting

The final agreed main alternative is:

\[
P = \{6,3,5\}, \quad Q = \{8,3,5\}.
\]

That is:

\[
P = \{\text{normal colon mucosa}, \text{lymphocytes}, \text{smooth muscle}\},
\]

\[
Q = \{\text{colorectal adenocarcinoma epithelium}, \text{lymphocytes}, \text{smooth muscle}\}.
\]

This resembles the original paper's MNIST design:

\[
P = \{1,2,3\}, \quad Q = \{1,2,8\},
\]

where most components overlap and only one class is replaced.

This is a more meaningful two-sample distribution comparison than a pure single-class comparison.

### 3.4 Use class-balanced sampling

For all mixture experiments, sample in a class-balanced way.

For example, if group size is \(n=90\), then:

- for \(P=\{6,3,5\}\), sample 30 images from class 6, 30 from class 3, and 30 from class 5;
- for \(Q=\{8,3,5\}\), sample 30 images from class 8, 30 from class 3, and 30 from class 5.

Do not use natural class frequency within the mixture, because then rejection may partly reflect uncontrolled class imbalance rather than the designed distributional shift.

---

## 4. Data usage protocol

Use official train/validation/test split.

| Split | Usage |
|---|---|
| train | Train the CNN only |
| validation | Select best CNN checkpoint |
| test | All two-sample testing and embedding extraction |

Important rule:

> The test split must not be used for CNN training or model selection.

The test split is used only after the CNN is frozen.

---

## 5. CNN model

### 5.1 Model purpose

The CNN is not the main contribution. It is used only to generate stable image representations for MMMD.

Do not spend effort on SOTA image classification. Keep the architecture close to the previous MNIST CNN.

### 5.2 Architecture

Modify the MNIST CNN only where necessary.

Main change:

```python
Conv2d(1, 32, kernel_size=3, padding=1)
```

becomes:

```python
Conv2d(3, 32, kernel_size=3, padding=1)
```

because PathMNIST is RGB.

Architecture:

```text
Input: 3 x 28 x 28

Block 1:
  Conv2d(3, 32, kernel_size=3, padding=1)
  ReLU
  MaxPool2d(2)
Output: 32 x 14 x 14

Block 2:
  Conv2d(32, 64, kernel_size=3, padding=1)
  ReLU
  MaxPool2d(2)
Output: 64 x 7 x 7

Global Average Pooling:
  64 x 7 x 7 -> 64

FC embedding:
  Linear(64, 128)
  ReLU
Output: 128-d penultimate FC embedding

Classifier:
  Linear(128, 9)
```

### 5.3 Embeddings to expose

The CNN code should expose the following representations:

| Name | Definition | Dimension |
|---|---|---:|
| `layer1_gap` | GAP over block-1 feature map | 32 |
| `layer2_gap` | GAP over block-2 feature map | 64 |
| `final_fc128` | penultimate FC embedding before classifier | 128 |

Do not use the 9-dimensional classifier logits as the embedding for the main experiments.

### 5.4 Training configuration

Suggested initial configuration:

| Parameter | Value |
|---|---|
| optimizer | Adam |
| learning rate | 1e-3 |
| batch size | 128 or 256 |
| epochs | 20–30 |
| loss | CrossEntropyLoss |
| augmentation | none for the main line |
| checkpoint selection | best validation accuracy |
| random seed | fixed, e.g. 1001 or 2026 |

Save:

- training config JSON;
- training log;
- training history CSV;
- best checkpoint;
- validation/test classification accuracy for sanity checking.

---

## 6. Main methods to compare

Run exactly these four methods for the current main line.

| Method name | Representation | Kernel count | Role |
|---|---|---:|---|
| `raw_pixel_gaussian5` | flatten RGB image, 2352-d | 5 | raw image baseline |
| `cnn_final_fc128_gaussian5` | final FC-128 embedding | 5 | single-layer CNN embedding baseline |
| `cnn_multilayer_single_gaussian` | layer1 GAP + layer2 GAP + final FC-128, one Gaussian per layer | 3 | low-complexity multilayer aggregation |
| `cnn_multilayer_gaussian15` | layer1/layer2/final × Gaussian-5 | 15 | multilayer multi-bandwidth extension |

### 6.1 Gaussian-5 definition

Use the same Gaussian-5 protocol as the MNIST experiments:

\[
\{2^{-2}, 2^{-1}, 1, 2^1, 2^2\} \times \text{median bandwidth scale}.
\]

This should remain consistent with the existing local implementation convention.

### 6.2 Ridge regularization

Use the same ridge rule as the current MNIST CNN experiments unless explicitly changed:

\[
\Sigma_{\mathrm{reg}} = \Sigma_{\hat{}} + \lambda I,
\]

with

\[
\lambda = 10^{-4} \times \mathrm{mean}(\mathrm{diag}(\Sigma_{\hat{}})).
\]

Record both unregularized and regularized condition numbers.

---

## 7. Main power experiment

### 7.1 Scenario

Only run the overlapping mixture alternative:

\[
P = \{6,3,5\}, \quad Q = \{8,3,5\}.
\]

Suggested scenario name:

```text
mix635_vs_mix835
```

### 7.2 Sampling design

Use class-balanced sampling.

| Group size \(n\) | Per-class count |
|---:|---:|
| 30 | 10 per class |
| 60 | 20 per class |
| 90 | 30 per class |
| 120 | 40 per class |
| 150 | 50 per class |

Sample sizes:

\[
n \in \{30,60,90,120,150\}.
\]

For each repetition:

- sample \(n/3\) images from class 6, \(n/3\) from class 3, and \(n/3\) from class 5 for group \(X\);
- sample \(n/3\) images from class 8, \(n/3\) from class 3, and \(n/3\) from class 5 for group \(Y\);
- run the four MMMD methods;
- record reject / not reject.

Sampling should be without replacement within each two-sample test whenever possible.

Since PathMNIST test split has finite class counts, first print class counts and confirm that all required per-class counts are feasible.

### 7.3 Calibration and repetitions

Use the same convention as current MNIST runs unless code refactoring requires clearer naming.

| Parameter | Value |
|---|---:|
| alpha | 0.05 |
| bootstrap resamples | 500 |
| outer repetitions | start with 10 for consistency with current MNIST runs |
| sampling | without replacement within a test |
| split | test only |

If the code currently uses `n.rep` and `n.iter` in the old local convention, preserve compatibility and document clearly which one means outer repetition and which one means bootstrap resampling.

### 7.4 Output

Main output:

\[
n \mapsto \widehat{\text{rejection rate}}.
\]

Plot one curve per method.

---

## 8. Type-I error check

### 8.1 Purpose

The Type-I check is not a separate scientific extension. It is a finite-sample calibration sanity check for the main methods.

Do not run Type-I for every possible H1. There is only one main H1 in the current plan.

Run one matched null scenario:

\[
P = Q = \{6,3,5\}.
\]

Suggested scenario name:

```text
mix635_vs_mix635_null
```

### 8.2 Sampling design

Use class-balanced sampling from the same mixture distribution.

Sample sizes:

\[
n \in \{60,120\}.
\]

For each repetition:

- group \(X\): sample \(n/3\) images from each of class 6, 3, and 5;
- group \(Y\): independently sample another \(n/3\) images from each of class 6, 3, and 5;
- groups should be disjoint within each class whenever possible;
- run the same four methods;
- record reject / not reject.

### 8.3 Methods

Use the same four methods:

1. `raw_pixel_gaussian5`
2. `cnn_final_fc128_gaussian5`
3. `cnn_multilayer_single_gaussian`
4. `cnn_multilayer_gaussian15`

### 8.4 Output

For each method and each null sample size, report empirical Type-I error.

Expected reference level:

\[
\alpha = 0.05.
\]

Plot empirical Type-I error against sample size with a horizontal line at 0.05.

---

## 9. Required diagnostics

Every test run should save the following quantities.

| Variable | Meaning |
|---|---|
| `method` | method name |
| `scenario` | `mix635_vs_mix835` or `mix635_vs_mix635_null` |
| `sample_size` | group size \(n\) |
| `outer_iter` | outer repetition index |
| `alpha` | significance level |
| `kernel_count` | 5 / 3 / 15 |
| `stat` | MMMD statistic |
| `cutoff` | bootstrap cutoff |
| `reject` | 0/1 |
| `lambda` | ridge regularization value |
| `cond_sigma_hat` | condition number before ridge |
| `cond_sigma_reg` | condition number after ridge |
| `runtime_sec` | optional but recommended |
| `seed` | random seed or repetition seed |

Condition-number diagnostics are important, especially for `cnn_multilayer_gaussian15`, because it uses 15 MMD components and may have worse covariance conditioning.

---

## 10. Directory structure and file management

Use a dataset-level workspace so PathMNIST data and shared CNN model artifacts are not owned by any single experiment. Each experiment still gets its own isolated folder for code, logs, and results.

Dataset root:

```text
dechao_reproduction/medmnist_pathmnist28/
```

Directory layout:

```text
dechao_reproduction/medmnist_pathmnist28/
  README.md
  data/
    pathmnist.npz
  shared_model/
    README.md
    models/
    python/
    logs/
  experiments/
    00_dataset_sanity_check/
      README.md
      python/
      Results/
      logs/
    01_cnn_training/
      README.md
      python/
      Results/
      logs/
    02_mmmd_smoke_test/
      README.md
      python/
      R/
      Results/
      logs/
    03_sample_size_power_mix635_vs_mix835/
      README.md
      python/
      R/
      Results/
      logs/
    04_type1_mix635_vs_mix635_null/
      README.md
      python/
      R/
      Results/
      logs/
```

Rules:

- `data/` is shared by all PathMNIST experiments.
- `shared_model/` holds reusable CNN code, logs, and checkpoints; no experiment directory owns the model.
- `experiments/` contains one folder per experiment, and each experiment owns its own code, logs, and `Results/`.
- Completed MNIST experiment directories must not be modified.

Phase 1 output files:

| File | Content |
|---|---|
| `experiments/00_dataset_sanity_check/Results/data_summary.csv` | split shapes and image ranges |
| `experiments/00_dataset_sanity_check/Results/pathmnist_class_counts.csv` | class counts by split |
| `experiments/00_dataset_sanity_check/Results/sampling_feasibility.csv` | disjoint sampling feasibility for planned H1/H0 sample sizes |
| `experiments/00_dataset_sanity_check/Results/pathmnist_examples.png` | example image montage |
| `experiments/00_dataset_sanity_check/logs/dataset_sanity_check.log` | human-readable Phase 1 summary |

Later shared model output files:

| File | Content |
|---|---|
| `shared_model/models/pathmnist_cnn_checkpoint.pt` | best validation checkpoint |
| `shared_model/models/train_config.json` | CNN training config |
| `shared_model/models/train_history.csv` | training/validation metrics |
| `shared_model/logs/train.log` | training log |

Later experiment output files should live under the owning experiment's `Results/` directory.

## 11. Recommended execution phases

### Phase 1: Dataset sanity check only

Codex should first implement only:

1. create the new experiment directory;
2. download/load PathMNIST-28;
3. print and save train/val/test shapes;
4. print and save class counts for each split;
5. confirm that classes 3, 5, 6, and 8 have enough test samples for the planned sampling;
6. save example image montage.

Do not train the CNN and do not run MMMD before this sanity check is reviewed.

### Phase 2: CNN training

After Phase 1 is confirmed:

1. implement RGB PathMNIST CNN;
2. train on train split;
3. select checkpoint by validation accuracy;
4. save checkpoint and training logs;
5. report sanity classification accuracy.

### Phase 3: Smoke test for MMMD pipeline

Before full experiments:

- run only `n=30` for the H1 setting;
- run only two methods first:
  - `raw_pixel_gaussian5`;
  - `cnn_final_fc128_gaussian5`;
- use very small outer repetitions if needed;
- confirm that result CSVs, diagnostics, and plots are produced correctly.

### Phase 4: Full main power experiment

Run:

- scenario: `mix635_vs_mix835`;
- sample sizes: `30, 60, 90, 120, 150`;
- methods: all four main methods;
- bootstrap: 500;
- alpha: 0.05.

### Phase 5: Representative Type-I check

Run:

- scenario: `mix635_vs_mix635_null`;
- sample sizes: `60, 120`;
- methods: all four main methods;
- bootstrap: 500;
- alpha: 0.05.

### Phase 6: Summary and plots

Produce:

- sample-size rejection-rate table;
- Type-I error table;
- condition-number summary;
- main power curve;
- Type-I error plot;
- condition-number plot.

---

## 12. Things explicitly not to do yet

Do not implement the following in the current main task:

| Deferred item | Reason |
|---|---|
| simple `6_vs_8` experiment | removed from current main plan |
| BreastMNIST | too small / not main line now |
| 64/128/224 MedMNIST | unnecessary architecture and compute complexity |
| 3D MedMNIST | out of scope |
| BBBC021 | too much preprocessing and not compatible with current pipeline |
| contamination / mixture curve | valuable advanced extension, but postponed |
| same-class source-vs-external-center domain shift | valuable advanced extension, but postponed |
| ResNet / pretrained model | may turn the project into image classification engineering |
| noise curve on PathMNIST | less natural than sample-size curve for current medical setting |
| classifier-logit embedding | not part of main comparison |
| layer1 2x2 pooled ablation | already useful on MNIST, but not necessary for current PathMNIST main line |

---

## 13. Final agreed experiment table

### 13.1 Power experiment

| Item | Configuration |
|---|---|
| Dataset | PathMNIST-28 |
| Split | test only |
| Scenario name | `mix635_vs_mix835` |
| Alternative | \(P=\{6,3,5\}, Q=\{8,3,5\}\) |
| Sampling | class-balanced |
| Group sizes | \(n=30,60,90,120,150\) |
| Per-class counts | \(n/3\) |
| Methods | 4 main methods |
| Bootstrap resamples | 500 |
| Alpha | 0.05 |
| Main output | sample size vs rejection rate |

### 13.2 Type-I check

| Item | Configuration |
|---|---|
| Dataset | PathMNIST-28 |
| Split | test only |
| Scenario name | `mix635_vs_mix635_null` |
| Null | \(P=Q=\{6,3,5\}\) |
| Sampling | class-balanced, two independent/disjoint groups |
| Group sizes | \(n=60,120\) |
| Per-class counts | \(n/3\) |
| Methods | same 4 main methods |
| Bootstrap resamples | 500 |
| Alpha | 0.05 |
| Main output | empirical Type-I error |

### 13.3 Method table

| Method | Representation | Kernel setup | Kernel count |
|---|---|---|---:|
| `raw_pixel_gaussian5` | flattened RGB pixels, 2352-d | Gaussian-5 | 5 |
| `cnn_final_fc128_gaussian5` | penultimate FC-128 | Gaussian-5 | 5 |
| `cnn_multilayer_single_gaussian` | layer1 GAP + layer2 GAP + final FC-128 | one median Gaussian per layer | 3 |
| `cnn_multilayer_gaussian15` | layer1 GAP + layer2 GAP + final FC-128 | Gaussian-5 per layer | 15 |

---

## 14. Suggested first instruction to Codex

Use this as the first command/task prompt to Codex:

```text
Please implement only Phase 1 first. Create a new experiment directory under dechao_reproduction/medmnist_pathmnist28_mmmd/. Load MedMNIST PathMNIST-28 using the official medmnist API or the official pathmnist.npz file. Save train/val/test shapes, class counts by split, and a small image montage. Confirm that test classes 3, 5, 6, and 8 have enough samples for class-balanced group sizes n=30,60,90,120,150 in the H1 setting and n=60,120 in the H0 setting. Do not train the CNN and do not run MMMD yet.
```

After Phase 1 output is reviewed, continue to CNN training and MMMD experiments.
