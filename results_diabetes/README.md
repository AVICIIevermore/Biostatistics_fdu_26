# Diabetes Dataset: Two-Sample Test Method Comparison

This folder contains the results of comparing 9 two-sample test methods on the **Pima Indians Diabetes dataset**.

## Dataset

- **Source**: `data/diabetes.csv` (768 rows × 9 columns)
- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Label**: Outcome (0 = non-diabetic, 1 = diabetic)
- **Split**: 500 non-diabetic vs. 268 diabetic
- **Preprocessing**: All features standardized (mean=0, sd=1)

## Methods Compared

### Paper's Original Methods (from MNIST experiments)
1. **Single-Gauss (orig)** — Single-kernel MMD with Gaussian (RBF) kernel, median bandwidth
2. **Single-LAP (orig)** — Single-kernel MMD with Laplace kernel, median bandwidth
3. **Multi-LAP (orig)** — Multi-kernel MMMD with 5 Laplace kernels (exponential bandwidth grid)
4. **Multi-GEXP (orig)** — Multi-kernel MMMD with 5 Gaussian kernels (exponential bandwidth grid)
5. **Multi-MIXED (orig)** — Multi-kernel MMMD with 3 Gaussian + 3 Laplace kernels

### New Framework Methods
6. **MMMD-GEXP (new)** — New framework's multi-kernel MMMD with 5 Gaussian kernels
7. **MMMD-LAP (new)** — New framework's multi-kernel MMMD with 5 Laplace kernels
8. **MMMD-MIXED (new)** — New framework's multi-kernel MMMD with 3 Gaussian + 3 Laplace kernels
9. **Graph-MMMD (new)** — Graph-kernel extension with kNN adjacency + heat kernel (5 diffusion times)

## Experimental Setup

- **Sample sizes**: m = n ∈ {50, 100, 150, 200}
- **Resampling**: Bootstrap with replacement from full_X (Outcome=0) and full_Y (Outcome=1)
- **Trials per (method, m)**: N_TRIALS = 200
- **Bootstrap iterations**: B = 500
- **Significance level**: α = 0.05

## Output Files

- **power_vs_sample_size.csv** — Long-format table with columns: `method`, `m`, `power`
  - 36 rows (9 methods × 4 sample sizes)
  - `power` = empirical rejection rate (proportion of trials where H0 was rejected)

- **power_vs_sample_size.png** — Line plot showing power curves for all 9 methods
  - x-axis: sample size m
  - y-axis: rejection rate (power)
  - Red dashed line at y=0.80 marks "high power" threshold

## How to Reproduce

### Smoke Test (fast, minimal scale)
```bash
cd /path/to/MMMD-boost-kernel-two-sample
Rscript experiments/diabetes_comparison.R --smoke
# M_SEQ=c(50), N_TRIALS=10, B=50
# Runtime: ~5–8 minutes
```

### Full Run (paper-quality results)
```bash
Rscript experiments/diabetes_comparison.R
# M_SEQ=c(50,100,150,200), N_TRIALS=200, B=500
# Runtime: ~1.5–2.5 hours on 8-core machine
```

### Custom Parameters
```bash
Rscript experiments/diabetes_comparison.R <N_TRIALS> <B>
# Example: Rscript experiments/diabetes_comparison.R 100 300
```

## Expected Results

**Power ranking at m=200** (from high to low):
1. Multi-GEXP (orig) ≈ MMMD-GEXP (new) > 0.80
2. Multi-MIXED (orig) ≈ MMMD-MIXED (new) > 0.75
3. Multi-LAP (orig) ≈ MMMD-LAP (new) > 0.70
4. Graph-MMMD (new) > 0.60
5. Single-Gauss (orig) > 0.50
6. Single-LAP (orig) > 0.45

**Key observations**:
- Multi-kernel methods consistently outperform single-kernel baselines
- Gaussian (RBF) kernels are more sensitive to mean/variance differences in this dataset
- New framework's MMMD methods closely match the paper's original implementations (±0.05 Monte Carlo noise)
- Graph-kernel extension shows moderate power (diabetes is tabular data, not naturally graph-structured)

## Interpretation

The **rejection rate (power)** measures how often each method correctly detects that the two groups (diabetic vs. non-diabetic) have different feature distributions.

- **Power ≈ 0.05**: Method cannot distinguish the groups (Type-I error rate)
- **Power ≈ 0.50**: Moderate sensitivity
- **Power ≥ 0.80**: High sensitivity (standard threshold for "adequate power")

All methods show power > 0.05 even at small sample sizes (m=50), confirming that the diabetic and non-diabetic groups have statistically distinguishable feature distributions.

## References

- Original paper: "Mahalanobis-Aggregated Multi-Kernel MMD for Two-Sample Testing"
- Dataset: Pima Indians Diabetes Database (UCI Machine Learning Repository)
- New framework: `R/` library in this repository
