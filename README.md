# Code for Boosting the Power of Kernel Two-Sample Test

This GitHub repository contains codes for experiments presented in the paper [Boosting the Power of Kernel Two-Sample Test](https://arxiv.org/abs/2302.10687), along with extended reproductions, embedding experiments, and numerical improvements.

## Background & Motivation

Kernel two-sample testing is fundamental in statistics for detecting differences between distributions. The Mahalanobis Aggregated MMD (MMMD) method improves upon single-kernel MMD by combining multiple kernels, enhancing statistical power while controlling Type-I error. This repository extends the original work with reproducibility studies, embedding-based experiments on real image data, and numerical improvements to handle high-dimensional and ill-conditioned settings.

## Related Works

- **MMD (Maximum Mean Discrepancy)** - Gretton et al.'s foundational kernel two-sample test
- **MMDAgg** - Kernel aggregation approach by Pfister et al.
- **Graph-based kernels** - Alternative kernel construction methods
- **Visual embeddings** - Deep learning encoders (CLIP, DINOv2) for image-based testing

## Table of Contents
1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Original Paper Results](#original-paper-results)
4. [Dechao Reproduction Experiments](#dechao-reproduction-experiments)
5. [Embedding MMMD Experiments](#embedding-mmmd-experiments)
6. [MMMD Extensions: Covariance Matrix Numerical Improvements](#mmmd-extensions-covariance-matrix-numerical-improvements)
7. [Technical Notes](#technical-notes)
8. [Team & Contributions](#team--contributions)

---

## Project Structure

This project is organized into three main experiment families:

```
MMMD-boost-kernel-two-sample/
├── Over sample size - dim2/           # Original paper: Sample size effects
├── Power over Dimensions/             # Original paper: Dimensionality study
├── Mixture Alternatives/              # Original paper: Mixture distributions
├── Local Alternatives/                # Original paper: Local power analysis
├── High Dimensional Alternatives/     # Original paper: High-dimensional regime
├── MNIST-Additive Noise/              # Original paper: MNIST experiments
├── MNIST-Reduced Contrast and AWGN/   # Original paper: Contrast degradation
├── MMDAggComparison/                  # Original paper: Comparison with MMDAgg
├── Quadratic and Linear Time Estimates/ # Original paper: Computational complexity
├── Time and Power Comparison/         # Original paper: MMMD vs MMD trade-offs
├── dechao_reproduction/               # Dechao reproductions (MNIST, PathMNIST, embeddings)
├── lrj_covariance_matrix/             # New covariance matrix methods
├── code/                              # Shared Python environment
└── README.md                          # This file
```

### Tracked Content
- Experiment code and configuration files
- README and plan files
- Summary CSVs and audit CSVs
- Figures for reporting power and Type-I error

### Not Tracked (Regenerable)
- Raw data/cache files under `data/`
- CNN model weights (`.pt`, `.pth`)
- R checkpoints (`.rds`)
- Large per-resample diagnostics
- Full raw result tables (summaries and plots are committed)

---

## Requirements

### Python
- Python 3.9+
- Dependencies managed via `uv` at `code/Biostatistics_fdu_26/pyproject.toml`

Setup:
```bash
cd code/Biostatistics_fdu_26
uv python install 3.12
uv sync
```

### R
- R 4.2.2 or later
- Required package: [`kernlab`](https://rdrr.io/cran/kernlab/)

### Note on Kernel Parameterization
The `kernlab` package uses:
- **Gaussian Kernel**: $\exp(-\sigma\|x-x'\|_2^2)$ 
- **Laplace Kernel**: $\exp(-\sigma\|x-x'\|_2)$

Our code outputs the squared median bandwidth, used directly for Gaussian kernels and square-rooted for Laplace kernels.

---

## Original Paper Results

We provide codes for generating Figures 1-9 from the paper. Complete results and figure locations:

### Figures 1-3: Sample Size and Dimensionality Effects
- **Figure 1(a-b)**: Sample size effects on Type-I error and power (Gaussian scale alternative)
  - Code: [Over sample size - dim2](https://github.com/anirbanc96/MMMD-boost-kernel-two-sample/tree/main/Over%20sample%20size%20-%20dim2)
  
- **Figures 2-3**: Power over dimensions and distribution families
  - Code: [Power over Dimensions](https://github.com/anirbanc96/MMMD-boost-kernel-two-sample/tree/main/Power%20over%20Dimensions/Code)

### Figures 4-5: Mixture and Local Alternatives
- **Figure 4**: Mixture alternatives with varying mixture parameters
  - Code: [Mixture Alternatives](https://github.com/anirbanc96/MMMD-boost-kernel-two-sample/tree/main/Mixture%20Alternatives/Code)
  
- **Figure 5**: Local alternatives and high-dimensional regime
  - Code: [Local Alternatives](https://github.com/anirbanc96/MMMD-boost-kernel-two-sample/tree/main/Local%20Alternatives/Code) and [High Dimensional Alternatives](https://github.com/anirbanc96/MMMD-boost-kernel-two-sample/tree/main/High%20Dimensional%20Alternatives/Code)

### Figures 6-8: Real Data (MNIST)
- **Figures 6-7**: Noisy MNIST with additive Gaussian noise
  - Code: [MNIST-Additive Noise](https://github.com/anirbanc96/MMMD-boost-kernel-two-sample/tree/main/MNIST-Additive%20Noise/Code)
  
- **Figure 8**: MNIST with reduced contrast and AWGN
  - Code: [MNIST-Reduced Contrast and AWGN](https://github.com/anirbanc96/MMMD-boost-kernel-two-sample/tree/main/MNIST-Reduced%20Contrast%20and%20AWGN/Code)

### Figure 9: Comparison with MMDAgg
- Code: [MMDAggComparison](https://github.com/anirbanc96/MMMD-boost-kernel-two-sample/tree/main/MMDAggComparison)

### Computational Complexity Analysis
- **Quadratic vs Linear Time**: [Quadratic and Linear Time Estimates](https://github.com/anirbanc96/MMMD-boost-kernel-two-sample/tree/main/Quadratic%20and%20Linear%20Time%20Estimates)
- **MMMD vs MMD Trade-offs**: [Time and Power Comparison](https://github.com/anirbanc96/MMMD-boost-kernel-two-sample/tree/main/Time%20and%20Power%20Comparison)

The Graph and Kernel based experiments can be implemented by calling `Body.R` in appropriate sub-folders with the appropriate parameters.

---

## Dechao Reproduction Experiments

This directory (`dechao_reproduction/`) contains three experiment families reproducing and extending the paper's results:

### Directory Organization

```
dechao_reproduction/
├── mnist/                    # MNIST additive-noise reproductions
├── medmnist_pathmnist28/     # PathMNIST 28×28 MMMD experiments
├── pathmnist_center_shift_mmmd/ # PathMNIST center-shift experiments
└── yao/                      # Embedding experiments (CLIP, DINOv2)
```

### 1. MNIST Experiments (`mnist/`)
Original MNIST additive-noise reproductions with faster reruns, CNN embedding experiments, and shared CNN model metadata.

### 2. MedMNIST PathMNIST 28×28 (`medmnist_pathmnist28/`)
PathMNIST class-mixture MMMD experiments including CNN training, smoke tests, power analysis, and Type-I error checks.

### 3. PathMNIST Center-Shift (`pathmnist_center_shift_mmmd/`)
Source-vs-external center-shift MMMD experiments with split sanity checks and comprehensive statistical validation.

---

## Embedding MMMD Experiments

The `dechao_reproduction/yao/` folder implements image-embedding two-sample tests using frozen visual embeddings (CLIP/DINOv2) with the Mahalanobis aggregated MMD test.

### Self-Contained Setup

This folder is self-contained with statistical testing code copied into `src/mmmd_functions.R`.

### Key Files

**Python Scripts:**
- `scripts/extract_medmnist_embeddings.py`: Download MedMNIST dataset and extract CLIP/DINOv2 embeddings
- `scripts/extract_mnist_embeddings.py`: Extract embeddings from local MNIST for smoke tests
- `scripts/extract_manifest_embeddings.py`: Extract embeddings from custom image manifest
- `scripts/build_imagefolder_metadata.py`: Scan class-per-folder dataset and write stratified metadata
- `scripts/extract_imagefolder_noise_embeddings.py`: Build shared embedding pools from local dataset
- `scripts/run_rawpixel_gauss5_imagefolder.py`: Raw-pixel baseline for local dataset

**R Code:**
- `src/mmmd_functions.R`: Single-kernel MMD and Mahalanobis multi-kernel MMD with Gaussian multiplier bootstrap
- `scripts/run_embedding_testing.R`: Estimate Type-I error and power from embedding CSV
- `scripts/run_noisy_embedding_testing.R`: Power/Type-I error from noisy embeddings
- `configs/example_medmnist_embedding.R`: Example configuration template

**Data/Results:**
- `data/embeddings/`: Recommended location for embedding CSV files
- `results/`: Recommended location for testing summaries

### Python Setup for Embeddings

```bash
cd code/Biostatistics_fdu_26
uv python install 3.12
uv sync
```

### Extract Embeddings

**DINOv2 on PathMNIST (224px):**
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

**CLIP embeddings:**
```bash
uv run python yao/scripts/extract_medmnist_embeddings.py \
  --dataset pathmnist \
  --size 224 \
  --split test \
  --encoder clip \
  --output yao/data/embeddings/pathmnist_clip.csv
```

**Smoke test (limited images):**
```bash
uv run python yao/scripts/extract_medmnist_embeddings.py \
  --dataset pathmnist \
  --size 224 \
  --split test \
  --encoder dinov2 \
  --max-images 500 \
  --output yao/data/embeddings/pathmnist_dinov2_smoke.csv
```

**From custom image manifest:**
```bash
uv run python yao/scripts/extract_manifest_embeddings.py \
  --manifest yao/data/bbbc021_manifest.csv \
  --image-root yao/data/bbbc021_images \
  --split test \
  --encoder dinov2 \
  --output yao/data/embeddings/bbbc021_dinov2.csv
```
Manifest format: `image_path,label` with optional columns `id,split`.

**Repeated noisy MNIST runs (directory output):**
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

Directory output contains `embeddings.npy` and `metadata.csv`; use directory path (without `.csv`) in R configs.

**MedMNIST noise with shared embedding pool:**
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

### Run Statistical Tests

1. Edit `configs/example_medmnist_embedding.R` to match your dataset
2. Run:
```bash
cd code/Biostatistics_fdu_26
Rscript yao/scripts/run_embedding_testing.R yao/configs/example_medmnist_embedding.R
```

**Output files:**
- `embedding_mmmd_replicates.csv`: One row per repetition and method
- `embedding_mmmd_summary.csv`: Type-I error and power estimates

### Supported Methods

- `GAUSS1`: Single Gaussian MMD (median bandwidth)
- `LAP1`: Single Laplace MMD (median bandwidth)
- `GAUSS5`: Mahalanobis MMD with 5 Gaussian bandwidths
- `LAP5`: Mahalanobis MMD with 5 Laplace bandwidths
- `MIXED`: Mahalanobis MMD with 3 Gaussian + 3 Laplace kernels

### Experimental Interpretation

- `setting == "type1"`: Splits one pool into two samples; reject rate estimates Type-I error
- `setting == "power"`: Samples from distinct groups; reject rate estimates power

---

## MMMD Extensions: Covariance Matrix Numerical Improvements

Located in `lrj_covariance_matrix/`, this section describes numerical improvements to covariance matrix estimation for Mahalanobis MMD.

### Problem Statement

The Mahalanobis MMD requires estimating covariance matrices of kernel statistics across multiple bandwidths. For kernels $K_1, \ldots, K_R$:

$$\widehat K_a = [K_a(X_i, X_j)]_{1 \leq i,j \leq m}$$

The centered Gram matrix is:
$$\widetilde K_a = C_m \widehat K_a C_m, \quad C_m = I - \frac{1}{m}\mathbf{1}\mathbf{1}^\top$$

Covariance estimate: $\widehat\Sigma_{ab} = \frac{2}{\widehat\rho^2(1-\widehat\rho)^2} \cdot \frac{1}{m^2} \langle \widetilde K_a, \widetilde K_b \rangle_F$

When vectors $\operatorname{vec}(\widetilde K_a)$ have high correlation, $\widehat\Sigma$ can be ill-conditioned.

### Three Numerical Strategies

#### 1. Pivoted Cholesky Kernel Selection

Use pivoted Cholesky decomposition to select numerically independent kernel subsets before computing full covariance:

- Input: $R$ candidate kernels, relative tolerance $\varepsilon$ (e.g., $10^{-3}$ or $10^{-4}$)
- Output: Subset $S = \{p_1, \ldots, p_q\}$ with $q \ll R$

**Algorithm outline:**
- Initialize residuals $r_j^{(0)} = \widehat\Sigma_{j,j}$
- At step $t$: Select kernel with largest residual $p_t = \arg\max_j r_j^{(t-1)}$
- Check: If $r_{p_t}^{(t-1)} \leq \varepsilon \max_j \widehat\Sigma_{j,j}$, stop
- Update: Compute Cholesky factor $L_{j,t}$ and update residuals $r_j^{(t)} = r_j^{(t-1)} - L_{j,t}^2$

**Complexity:** $O(RqN^2 + Rq^2)$ vs. naive $O(R^2N^2)$ for full covariance.

#### 2. Lazy Pivoted Cholesky

Defer covariance computation until needed. Only query $\widehat\Sigma_{\cdot,p}$ when $p$-th kernel is selected by pivoted Cholesky.

**Complexity:** Kernel evaluation $O(RqN^2)$, Cholesky updates $O(Rq^2)$, bootstrap $O(BqN^2)$.

#### 3. Gaussian/Laplace Kernel Fast Covariance

Exploit exponential kernel family: $K_t(x,y) = \exp(-t \cdot d(x,y))$

For Gaussian: $t = 1/\sigma^2$; for Laplace: $t = 1/\sigma$.

Key property: $K_{t_a}(x,y) \cdot K_{t_b}(x,y) = K_{t_a + t_b}(x,y)$

Pre-compute: Distance matrix $D_{ij}$, row/column sums $s_a$, $S_a$, and values $H(u) = \sum_{i,j} \exp(-u D_{ij})$ for all possible sums $u = t_a + t_b$ ($\sim 2R$ values instead of $R^2$).

**Complexity:** $O(dm^2 + Rm^2 + mR^2)$ vs. naive $O(R^2m^2)$.

### Combined Strategy

Integrate fast covariance (Method 3) with lazy pivoted Cholesky (Method 2):

**Total complexity:** $O(RN^2 + BqN^2 + B\log B)$ where $q \ll R$.

When $q \approx r$ (paper's $r$ kernel MMMD), main term remains $O(r^2N^2)$; when $q \ll R$, improvement is substantial.

### Numerical Experiments

#### Experiment 1: Reproduce Figure 1(b) with New Method

- Data: $d=2$, $X \sim N(0, I_2)$, $Y \sim N(0, 1.25 I_2)$
- Sample sizes: $n \in \{50, 100, 200, 300, 400, 500\}$
- Methods: GAUSS single-kernel, GEXP MMMD (5 kernels), NEW MMMD (13 kernels with kernel selection)

**Results:** NEW achieves GEXP-comparable power with covariance matrix $\sim 10^4$ condition number vs. $\sim 10^5$. Numerical stability improved while maintaining statistical power.

#### Experiment 2: Time vs. Power Trade-off

- Data: $d=5$, $Y \sim N(0, 1.2 I_5)$
- NEW-13: $\sim 4.8\times$ faster than GEXP-rich-13; NEW-25: $\sim 4.5\times$ faster
- Kernel selection: $q=4$–5 of 13 kernels selected via pivoted Cholesky

#### Experiment 3: PathMNIST Real Data

- Dimension: $d = 2352$ (raw pixel)
- Mix635 vs Mix835: Power improves with NEW method, condition numbers from $10^5$ to $10^4$ range
- H0 Type-I: GEXP inflates to 0.10 (vs. $\alpha=0.05$); NEW maintains ~0.05

### Implementation Files

- `fast_cov_gaussian_t()`: Fast Gaussian kernel covariance (Method 3)
- `pivoted_chol_select()`: Kernel selection via pivoted Cholesky (Methods 1–2)
- `mmmd_test()`: Updated test with optional kernel selection

---

## Extended Experiments: Running Simulations

### Quick Start (Smoke Test)
```bash
cd dechao_reproduction
Rscript run_all_tasks.R
```

### Standard Parameters (Recommended)
```bash
Rscript experiments/task1_epsilon_sensitivity.R 30 50 300 0.5
Rscript experiments/task2_variance_sensitivity.R 30 50 300 0.5
Rscript experiments/task3_typeI_and_roc.R 30 100 100 300 0.5
Rscript experiments/task4_graph_demo.R 30 50 300 0.5 "0,0.1,0.25,0.5,0.75,0.9"
Rscript experiments/fig2_mixture_alt.R 30 50 300 100 0.5
```

### Full Production (More Stable)
```bash
Rscript experiments/task1_epsilon_sensitivity.R 30 100 500 0.5
Rscript experiments/task2_variance_sensitivity.R 30 100 500 0.5
Rscript experiments/task3_typeI_and_roc.R 30 200 200 500 0.5
Rscript experiments/task4_graph_demo.R 30 100 500 0.5 "0,0.1,0.25,0.5,0.75,0.9"
Rscript experiments/fig2_mixture_alt.R 30 100 500 100 0.5
```

### Parameter Guide
- `d`: Data dimension
- `N_trials`: Monte Carlo replications
- `B`: Multiplier bootstrap repetitions
- `rho`: AR(1) correlation coefficient
- `n`: Sample size (for fig2)
- `skew_grid`: Skewness levels (for task4)

### Key Experiments

**task1_epsilon_sensitivity.R**: Contamination robustness
- Tests mixture $Q = (1-\epsilon)N(0,\Sigma) + \epsilon t_{10}(0,\Sigma)$
- Finds epsilon where power ≈ 0.80

**task2_variance_sensitivity.R**: Variance scale detection
- Tests $Q = N(0, k\Sigma_0)$ for varying $k$
- Compares GAUSS MMMD, LAP MMMD, Mixed MMMD vs. single-kernel MMD

**task3_typeI_and_roc.R**: Type-I calibration and ROC analysis
- Computes Type-I error under $H_0: X,Y \sim N(0,\Sigma_{AR1})$
- Generates ROC curves for varying decision thresholds

**task4_graph_demo.R**: Graph-MMMD Type-I validation
- Tests Graph-MMMD across multiple null distributions
- Includes skewness robustness analysis

**fig2_mixture_alt.R**: Mixture distribution power
- Reproduces Figure 2 with mixture alternatives
- Evaluates power over mixture proportions

---

## Integrating Custom Real Data

### Single-condition Type-I:
```r
source("R/load.R")
M <- as.matrix(read.csv("data/my_data.csv"))
ds <- ds_resample_from_matrix(M)
xy <- ds(100, 100)
mmmd_test(xy$X, xy$Y, family = "GEXP", B = 500)
```

### Two-condition Power:
```r
source("R/load.R")
MX <- as.matrix(read.csv("data/group_x.csv"))
MY <- as.matrix(read.csv("data/group_y.csv"))
ds <- ds_resample_two_matrices(MX, MY)
xy <- ds(100, 100)
mmmd_test(xy$X, xy$Y, family = "GEXP", B = 500)
```

### Graph-MMMD:
```r
graph_mmmd_test(xy$X, xy$Y, B = 500)
```

---

## Technical Notes

### Kernel Bandwidth Selection
Bandwidth selection uses the median heuristic. For Gaussian kernels, we output squared median bandwidth; for Laplace, we use the square root.

### Numerical Stability (Ridge Regularization)
In high-dimensional or ill-conditioned settings, ridge regularization may be applied:
$$(\widehat\Sigma + \lambda I)^{-1} \text{ instead of } \widehat\Sigma^{-1}$$

Default: $\lambda = 10^{-5} \cdot \min_i(\widehat\Sigma_{ii})$. For custom ridge, adjust this parameter.

### Kernel Selection Benefits
- Reduces effective kernel count from $R$ to $q$ (typically $q \ll R$)
- Improves numerical conditioning of covariance matrix
- Maintains statistical power while improving stability

---

## Team & Contributions

**Original Paper:**
- Anirban Chatterjee (University of Pennsylvania)

**Extensions & Reproductions:**

| Component | Contributor |
|-----------|-------------|
| `dechao_reproduction/` | Dechao Huang (黄德超) |
| `dechao_reproduction/yao/` | Yao Yu (于尧) |
| `lrj_covariance_matrix/` | Ruijie Li (李睿杰) |
| `experiments/` | Ziheng Zheng (郑子恒) |

---

## Future Directions

- Extend kernel selection strategies to adaptive bandwidth tuning for different data modalities
- Develop GPU-accelerated implementations for large-scale datasets
- Apply MMMD to temporal and sequential data with specialized kernels
- Integrate with causal inference methods for treatment effect heterogeneity
- Explore theoretical guarantees under model misspecification

---

## Citation

If you use this code, please cite:

```bibtex
@article{chatterjee2023boosting,
  title={Boosting the Power of Kernel Two-Sample Test},
  author={Chatterjee, Anirban},
  journal={arXiv preprint arXiv:2302.10687},
  year={2023}
}
```
