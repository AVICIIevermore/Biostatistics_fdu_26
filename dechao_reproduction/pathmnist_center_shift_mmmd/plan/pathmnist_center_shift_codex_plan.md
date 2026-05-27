# PathMNIST Within-Class Center Shift Detection: Codex Experiment Plan

## 0. Goal

This experiment extends the current MNIST CNN-MMMD pipeline to a real medical image dataset using **PathMNIST** from MedMNIST.

The key idea is **not** ordinary class-vs-class testing. Instead, we test whether two image samples from the **same histological class** but different clinical centers have the same distribution.

Main hypothesis design:

\[
H_0: P = Q = \text{same tissue class, same source domain}
\]

\[
H_1: P = \text{same tissue class, source domain}, \quad Q = \text{same tissue class, external domain}
\]

In words:

> Within the same tissue label, can CNN-MMMD detect source-to-external clinical-center distribution shift?

PathMNIST is suitable because its official train/validation splits come from NCT-CRC-HE-100K, while the official test split comes from CRC-VAL-HE-7K, a different clinical center. The task has 9 colorectal histology classes.

Target labels for the main experiment:

```text
label 6 = normal colon mucosa
label 8 = colorectal adenocarcinoma epithelium
```

Optional later:

```text
label 7 = cancer-associated stroma
```

---

## 1. Important conceptual framing

CNN-MMMD should **not** be described as a completely prior-free classical two-sample test.

Use this interpretation:

> CNN-MMMD is a representation-assisted kernel two-sample test, where the CNN representation is learned from independent auxiliary data and frozen before hypothesis testing.

The CNN defines a feature map:

\[
\phi(x) = \text{CNN embedding}(x)
\]

The test then evaluates distributional equality in representation space:

\[
H_0: \phi(X) \sim \phi(Y)
\]

If the raw image distributions are identical, then the embedded distributions are also identical. Therefore, Type-I error can still be valid **if the CNN is trained independently and frozen before testing**.

However, the reverse is not guaranteed:

\[
\phi(X) \sim \phi(Y) \nRightarrow X \sim Y
\]

Therefore, CNN-MMMD should be described as a medically informed, representation-space distribution test, not a fully omnibus raw-image test.

This is a strength if stated carefully:

```text
Raw-pixel MMMD: more generic, less prior information.
CNN-MMMD: uses independent labeled medical images to gain sensitivity to clinically meaningful shifts.
```

---

## 2. Data split design

Use:

```text
dataset = PathMNIST
size = 28
channels = 3
task = multi-class
num_classes = 9
```

Avoid leakage. Do **not** train the CNN on samples used for two-sample testing.

Split the official train split as follows:

```text
official train split:
    stratified split into:
        cnn_train_pool: 80%
        source_holdout_pool: 20%

official val split:
    model selection / checkpoint selection only

official test split:
    external_pool
```

Two-sample testing may only use:

```text
source_holdout_pool
external_pool
```

Do not use `cnn_train_pool` in MMMD testing.

---

## 3. Class count check

Before training or testing, generate:

```text
Results/pathmnist_class_counts.csv
```

Required columns:

```text
split
label
label_name
count
```

Required split names:

```text
cnn_train_pool
source_holdout_pool
val
external_pool
```

This file is important because the official test set is smaller than the train set, and all feasible sample sizes must respect within-label counts.

---

## 4. CNN training

Train the CNN using:

```text
training data = cnn_train_pool, all 9 labels
validation data = official val split, all 9 labels
test data = never used for CNN training or checkpoint selection
```

Important: do **not** train only on labels 6 and 8.

Reason:

```text
The goal is to learn a general pathology tissue representation from source-domain labeled auxiliary data, not a narrow binary normal-vs-tumor representation.
```

Suggested small CNN architecture:

```text
Conv2d(3, 32, kernel_size=3, padding=1) + ReLU + MaxPool
Conv2d(32, 64, kernel_size=3, padding=1) + ReLU + MaxPool
GlobalAveragePooling
Linear(64, 128) + ReLU
Linear(128, 9)
```

Save:

```text
models/pathmnist_cnn_checkpoint.pt
models/train_config.json
models/train_history.csv
logs/pathmnist_train.log
Results/pathmnist_training_curves.png
```

Also report:

```text
train accuracy
validation accuracy
best validation epoch
```

---

## 5. Embedding definitions

Use the same sampled image indices across raw-pixel and CNN-embedding methods.

Representations:

```text
raw_pixel:
    flatten 3 x 28 x 28 image = 2352-dimensional vector

layer1_gap:
    global-average-pooled output after first conv block = 32-dimensional vector

layer2_gap:
    global-average-pooled output after second conv block = 64-dimensional vector

final_fc:
    penultimate embedding before classifier = 128-dimensional vector

multilayer:
    concatenate layer1_gap + layer2_gap + final_fc
```

Freeze the CNN before extracting all embeddings.

---

## 6. Methods to compare

Run only these four methods:

```text
1. raw_pixel_gaussian5
2. final_fc_gaussian5
3. multilayer_single_gaussian
4. multilayer_gaussian15
```

Do not add extra models at this stage.

Reason:

```text
Previous MNIST results suggested that multilayer single-Gaussian can be more stable than the full Gaussian-15 version, while Gaussian-15 may suffer more covariance-conditioning issues. Therefore both should be retained.
```

MMMD statistic:

\[
T = \mathbf{m}^T (\widehat{\Sigma} + \lambda I)^{-1} \mathbf{m}
\]

For `multilayer_gaussian15`:

```text
3 layers x 5 bandwidths = 15 MMD components
```

Use one Mahalanobis aggregation over all layer-bandwidth components. Do not do layer-level second-stage aggregation.

---

## 7. Testing settings

For each target label:

```text
labels = [6, 8]
```

Run three settings.

### A. H1: source-vs-external within-class center shift

```text
X ~ source_holdout_pool[label == c]
Y ~ external_pool[label == c]
```

This is the main scientific result.

### B. H0-source: source internal null

```text
X, Y are two disjoint samples from source_holdout_pool[label == c]
```

Purpose:

```text
Check empirical Type-I error within the source domain.
```

### C. H0-external: external internal null

```text
X, Y are two disjoint samples from external_pool[label == c]
```

Purpose:

```text
Check empirical Type-I error within the external domain.
```

This is important because the external clinical-center test set may itself be heterogeneous.

---

## 8. Sample size grid

Use:

```text
n_grid = [20, 30, 50, 80, 100]
```

Each test uses:

```text
|X| = |Y| = n
sampling without replacement within each test repetition
different repetitions may resample
```

Feasibility rules:

```text
For H1:
    n <= min(source_count_c, external_count_c)

For H0-source:
    2n <= source_count_c

For H0-external:
    2n <= external_count_c
```

Automatically skip infeasible n values and record skipped cases.

Do not increase n beyond 100 unless explicitly requested later.

Reason:

```text
The official external test split is much smaller than the source split. n <= 100 should be enough for repeated subsampling while avoiding overly tight external within-class pools.
```

---

## 9. Smoke test and formal run

Because the original paper/code has confusing parameter names, use explicit variable names in the new scripts and results.

Avoid ambiguous names such as:

```text
n.rep
n.iter
resamp
```

Use explicit names:

```text
n_outer_batches
n_tests_per_cell_actual
B_boot
alpha
sample_size_per_group
```

### Smoke test

Run first:

```text
labels = [6]
n_grid = [20, 50]
methods = [
    raw_pixel_gaussian5,
    final_fc_gaussian5,
    multilayer_single_gaussian
]
settings = [
    H1_source_vs_external,
    H0_source,
    H0_external
]
n_outer_batches = 1
B_boot = 100
alpha = 0.05
```

Purpose:

```text
Confirm code correctness, data split correctness, feasible sampling, valid output files, and no obvious covariance failures.
```

### Formal run

After smoke test passes, run:

```text
labels = [6, 8]
n_grid = [20, 30, 50, 80, 100]
methods = [
    raw_pixel_gaussian5,
    final_fc_gaussian5,
    multilayer_single_gaussian,
    multilayer_gaussian15
]
settings = [
    H1_source_vs_external,
    H0_source,
    H0_external
]
n_outer_batches = 10
B_boot = 500
alpha = 0.05
```

Important:

```text
In the final summary, explicitly report the actual number of two-sample tests per cell.
```

If `n_outer_batches = 10` corresponds to only 10 actual two-sample tests per method/n/label/setting, report this clearly. Do not hide it behind ambiguous naming.

---

## 10. Required per-test output fields

Each actual two-sample test should save one row with at least:

```text
dataset
label
label_name
setting
n
method
rep_id
seed
stat
cutoff
reject
kernel_count
lambda
cond_sigma_hat
cond_sigma_reg
source_count_available
external_count_available
sample_x_indices
sample_y_indices
n_tests_per_cell_actual
B_boot_actual
alpha
```

Main raw result file:

```text
Results/pathmnist_center_shift_results.csv
```

---

## 11. Required summary files

Generate:

```text
Results/pathmnist_center_shift_summary.csv
Results/pathmnist_sigma_diagnostics.csv
```

Summary columns should include:

```text
label
label_name
setting
n
method
rejection_rate
binomial_se
ci_lower
ci_upper
mean_stat
mean_cutoff
median_cond_sigma_hat
median_cond_sigma_reg
median_lambda
n_tests_per_cell_actual
B_boot_actual
alpha
```

---

## 12. Required plots

Generate these four core plots:

```text
1. Results/pathmnist_center_shift_power_curve.png
```

Content:

```text
x = n
y = rejection_rate
facet = label
color = method
setting = H1_source_vs_external only
```

```text
2. Results/pathmnist_type1_check.png
```

Content:

```text
settings = H0_source and H0_external
include horizontal line at alpha = 0.05
x = n
y = rejection_rate
facet = label
color = method
```

```text
3. Results/pathmnist_center_shift_gap.png
```

Content:

```text
y = H1 rejection rate - matched H0 rejection rate
matched H0 can be H0_source or average of H0_source and H0_external
state clearly which version is used
```

Purpose:

```text
Show that high H1 rejection is not just Type-I inflation.
```

```text
4. Results/pathmnist_condition_number.png
```

Content:

```text
x = n
y = log10(cond_sigma_reg)
facet = label
color = method
```

Optional visualization:

```text
Results/pathmnist_embedding_pca_by_domain.png
```

Content:

```text
Use final_fc embeddings only.
Plot labels 6 and 8 separately.
Color source_holdout vs external.
Do not treat PCA as formal evidence; it is only visualization.
```

---

## 13. Result interpretation criteria

The main result is convincing only if:

```text
H0-source rejection rate is close to alpha = 0.05
H0-external rejection rate is close to alpha = 0.05
H1 source-vs-external rejection rate is clearly above alpha
H1 rejection rate increases with n
regularized covariance condition number is not pathological
```

Acceptable wording:

> Within the same histological class, CNN-MMMD detects a source-to-external distribution shift while maintaining reasonable empirical Type-I behavior under within-domain null resampling.

Avoid overclaiming:

```text
Do not say: We prove the clinical center causes the shift.
Do not say: CNN-MMMD tests all possible raw-image distribution differences.
Do not say: This is a patient-level inference.
```

Better wording:

> The detected shift is associated with the official source/external split, which corresponds to different clinical centers in PathMNIST.

Also note:

```text
This is patch-level distribution testing unless patient/slide metadata are available.
```

---

## 14. Optional sensitivity analysis: leave-target-label-out CNN

This is optional, not required for the first formal run.

Purpose:

```text
Test whether CNN-MMMD only works because the CNN saw the exact target histological class during supervised training.
```

For label 6 testing:

```text
Train CNN on source-domain official train labels except label 6.
Freeze CNN.
Test source_holdout[label 6] vs external[label 6].
```

For label 8 testing:

```text
Train CNN on source-domain official train labels except label 8.
Freeze CNN.
Test source_holdout[label 8] vs external[label 8].
```

If this still detects source-external shift, it is a strong robustness result.

Do not run this until the main experiment is complete.

---

## 15. Failure modes and fallback rules

### If H1 is too easy

If H1 rejection is almost 1.0 even at n = 20:

```text
Add smaller n values: [5, 10, 15]
Report the minimum sample size needed for reliable detection.
```

### If H1 is too weak

Do not immediately switch datasets.

First check:

```text
1. class counts
2. CNN train/validation accuracy
3. raw_pixel_gaussian5 result
4. final_fc_gaussian5 result
5. label 6 and label 8 separately
6. covariance condition numbers
```

If raw pixels detect shift but CNN embeddings do not:

```text
The shift may be color/stain/texture-based and not preserved by the CNN representation.
```

If CNN detects shift but raw pixels do not:

```text
The CNN representation may amplify medically meaningful morphology differences.
```

If all methods fail:

```text
The source/external within-class difference may be weak at 28x28 resolution, or the implementation may need checking.
```

---

## 16. Recommended directory structure

Create a new experiment folder:

```text
dechao_reproduction/pathmnist_center_shift_mmmd/
```

Suggested structure:

```text
Code/
  train_pathmnist_cnn.py
  extract_pathmnist_embeddings.py
  run_pathmnist_center_shift.R
  summarize_pathmnist_results.R
  plot_pathmnist_results.R

Results/
models/
logs/
configs/
```

If the current repo is Python-only or R-only, adapt file extensions accordingly, but keep the same logical separation:

```text
train CNN
extract embeddings
run tests
summarize results
plot results
```

---

## 17. Final deliverables from Codex

After completion, report:

```text
1. Path to all result CSV files
2. Path to all plots
3. CNN training accuracy and validation accuracy
4. Class counts for labels 6 and 8 in source_holdout_pool and external_pool
5. Actual number of tests per method/n/label/setting
6. Bootstrap draws per test
7. Any near-singular covariance warnings
8. Runtime summary by method
9. A 5-8 line experiment summary
```

The final 5-8 line summary should answer:

```text
Did H1 source-vs-external rejection exceed within-domain H0 rejection?
Which method was strongest?
Did Type-I checks look acceptable?
Did covariance diagnostics look stable?
Was label 6 or label 8 easier to detect?
```

---

## 18. Minimal first action

Start only with the smoke test:

```text
label = 6
n_grid = [20, 50]
methods = [raw_pixel_gaussian5, final_fc_gaussian5, multilayer_single_gaussian]
settings = [H1_source_vs_external, H0_source, H0_external]
n_outer_batches = 1
B_boot = 100
```

Do not run the full formal experiment until the smoke test produces all expected files and the sampling logic has been verified.
