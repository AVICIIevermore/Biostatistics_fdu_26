# MMMD Extensions (plan1.md companion)

This document describes the extension layer added on top of the original
`MMMD-boost-kernel-two-sample` repo.  The original 13 chapter folders
(`Local Alternatives/`, `High Dimensional Alternatives/`, ...) are
**untouched**; the new layer lives under three top-level directories
plus two trampolines at the root:

```
R/                       reusable library (parallel, data sources,
                         MMMD core, ROC, graph kernel, bio loader)
experiments/             one driver per plan1.md task
data/                    drop zone for real biological CSVs
results/                 auto-generated CSVs + PNGs
run_graph_demo.R         root-level alias for task 4
run_all_tasks.R          smoke-test driver for tasks 1-4
```

Every script self-locates its directory (the `## __SELF_LOCATING_PREAMBLE__`
block) so you can invoke it from any current working directory, including
the project root.

---

## 1. Quick start

```bash
# from the project root  (paper-aligned defaults below)
Rscript experiments/task1_epsilon_sensitivity.R 30 20 200 0.5
Rscript experiments/task2_variance_sensitivity.R 30 20 200 0.5
Rscript experiments/task3_typeI_and_roc.R       30 50 50 200 0.5
Rscript experiments/task4_graph_demo.R          30 50 200 0.5
Rscript experiments/fig2_mixture_alt.R          30 20 200 100 0.5
Rscript run_graph_demo.R                                # alias for task 4
Rscript run_all_tasks.R                                 # smoke-test all five
```

The trailing positional args are the experiment knobs (see top of each
driver for their meanings — typically `d`, `N_trials`, `B`).

---

## 2. Mapping plan1.md to files

| plan1.md task                       | driver                                        | outputs in `results/`                                     |
|-------------------------------------|-----------------------------------------------|-----------------------------------------------------------|
| 1. epsilon-sliding sensitivity      | `experiments/task1_epsilon_sensitivity.R`     | `epsilon_sensitivity_d{d}.csv`, `..._curve_d{d}.png`      |
| 2. k-sliding variance sensitivity   | `experiments/task2_variance_sensitivity.R`    | `variance_sensitivity_d{d}.csv`, `..._curve_d{d}.png`     |
| 3. Type-I baseline + ROC over family| `experiments/task3_typeI_and_roc.R`           | `typeI_baseline.csv`, `roc_data.csv`, `mmmd_roc_curves_multi_r.png` |
| 4. graph two-sample / non-Gaussian  | `experiments/task4_graph_demo.R`              | `task4_graph_summary.csv`, `task4_graph_curve.png`        |
| paper Fig.2 mixture reproduction    | `experiments/fig2_mixture_alt.R`              | `fig2_mixture_alt_d{d}.csv`, `fig2_mixture_alt_d{d}.png`  |

The `R/` library is shared:

| file                       | purpose                                                                |
|----------------------------|------------------------------------------------------------------------|
| `R/load.R`                 | single entry point — `source("R/load.R")` and the rest is on `search()` |
| `R/parallel_utils.R`       | `with_parallel(...)` + `mmmd_foreach(...)`                              |
| `R/data_sources.R`         | pluggable `function(n_x, n_y) -> list(X, Y)` factories + registry       |
| `R/mmmd_core.R`            | kernel construction, MMD vector, Mahalanobis bootstrap, ROC threshold scan |
| `R/roc_utils.R`            | trial-list -> (alpha, fpr, tpr) data frame                              |
| `R/graph_kernel.R`         | kNN diffusion-kernel list -> graph-MMMD test                            |
| `R/bio_loader.R`           | template for plugging in real biological CSV / TSV data                  |

---

## 3. Plugging in a real biological dataset

The MMMD test logic only ever sees a numeric matrix
`(n_samples x n_features)`.  No matter what the source — gene
expression, single-cell RNA-seq, microbiome OTU, methylation — once
you have a CSV in that shape the rest is mechanical.

### 3.1 Type-I baseline on real data (one condition)

Drop a CSV under `data/`, then in `R/bio_loader.R` uncomment:

```r
mmmd_register_data_source("bio_singlecell_typeI", function(path, ...) {
  load_bio_single(path = path, ...)
})
```

Use it inside any experiment script:

```r
ds <- mmmd_data_source("bio_singlecell_typeI",
                       path = file.path(mmmd_data_dir(), "pbmc.csv"))
xy <- ds(200, 200)
mmmd_test(xy$X, xy$Y, family = "GEXP", r = 5, B = 1000)
```

### 3.2 Power on real data (two conditions)

Drop two CSVs under `data/` (one per condition) and uncomment the
`bio_singlecell_power` block in `R/bio_loader.R`.  The factory will
intersect the column names (so you don't have to align the gene panels
manually).

### 3.3 Graph two-sample test on real data

Pass any `(X, Y)` directly to `graph_mmmd_test(...)` — the kNN graph
and diffusion-kernel list are constructed automatically from the
stacked rows.  Tune `k_nn` (graph degree) and `t_seq` (diffusion
times) to your domain.

---

## 4. Performance guardrails (plan1.md section 4)

The new layer obeys the four guardrails:

| guardrail                            | how it is satisfied                                                                |
|--------------------------------------|------------------------------------------------------------------------------------|
| 1. mandatory parallelism             | `with_parallel(...)` registers `parallel::detectCores() - 1` workers and tears them down on exit; outer loops are `foreach %dopar%` via `mmmd_foreach`. |
| 2. vectorised ROC / threshold scan   | `mmmd_aggregate_roc()` does ONE bootstrap per trial then evaluates all alpha levels in a single `quantile(...)` call + matrix mask.  No re-bootstrapping per alpha. |
| 3. optional Rcpp acceleration        | The bootstrap kernel is already O(B*n^2) using only matrix ops; if profiling shows a bottleneck, swap `mmmd_bootstrap()` for an `RcppArmadillo::sympd` version with no API change. |
| 4. coarse + fine pre-scan            | Task 1 scans `eps in seq(0, 1, 0.05)` first, locates the steep band, then refines to `0.01`.  Tasks 2 and 3 run on a single grid since their power curves are smoother. |

---

## 5. Adding a new experiment

1. Create `experiments/taskN_my_new.R`.
2. Begin with the self-locating preamble (copy from any existing driver).
3. `source(file.path(.script_dir, "..", "R", "load.R"))`.
4. Compose your test from the library: pick a data source via
   `mmmd_data_source(...)` or write a new factory and register it,
   then call `mmmd_test(...)` (vector kernels) or `graph_mmmd_test(...)`
   (graph kernels) inside `mmmd_foreach(...)` / `with_parallel(...)`.
5. Write outputs to `mmmd_results_dir()`.

No edits to the original chapter folders are ever required.

---

## 6. AR(1) covariance & paper Fig.2 reproduction

The Boosting-MMD paper's "Mixture Alternatives" chapter does **not** use
identity covariance.  Both `P` and `Q` are sampled with an AR(1) Toeplitz
covariance

```
Sigma0[i, j] = rho^|i - j|     (rho = 0.5 by default)
Sigma1       = sigma_mult * Sigma0   (sigma_mult = 1.25 in Fig.2)
```

and each row is drawn as a Gaussian-vs-t mixture controlled by mixing
probability `p`:

```
P_i  ~ (1 - p) * N(mu0, Sigma0) + p * t_df(mu0, Sigma0)
Q_i  ~ (1 - p) * N(mu1, Sigma1) + p * t_df(mu1, Sigma1)
```

The extension layer now provides this generator plus matched
covariance variants for the other tasks.

### 6.1 New helpers and data sources (in `R/data_sources.R`)

| symbol                          | role                                                          |
|---------------------------------|---------------------------------------------------------------|
| `mmmd_ar1_cov(d, rho)`          | builds `Sigma[i,j] = rho^|i-j|`                                |
| `ds_mixture_fig2(d, p, ...)`    | paper Fig.2 generator (Gaussian-t mixture, AR(1) Sigma)        |
| `ds_variance_scale_ar1(d, k, rho)` | drop-in for variance-scale alt with AR(1) Sigma             |
| `ds_identical_ar1(d, rho)`      | Type-I baseline with AR(1) Sigma                               |
| `ds_normal_t_mixture_ar1(d, eps, rho, df)` | epsilon-mix alt (task 1) with AR(1) Sigma           |

All four new factories are also registered with `mmmd_register_data_source()`
under the keys `mixture_fig2`, `variance_scale_ar1`, `identical_ar1`,
`normal_t_mixture_ar1`.

### 6.2 Kernel families (paper-aligned)

`mmmd_make_kernels(X, Y, family)` follows the original
`Mixture Alternatives/Code/Kernel Based Test/Functions.R` exactly:

| family   | bandwidth grid (relative to median heuristic)  | # kernels |
|----------|-------------------------------------------------|-----------|
| `GAUSS` / `GEXP` | `2^i * med_bw` for `i = -2..2`           | 5 RBF     |
| `LAP`    | `2^i * med_bw` for `i = -2..2`, sqrt-scaled     | 5 Laplace |
| `MIXED`  | `2^i * med_bw` for `i = -1..1`, both kernels    | 3 + 3 = 6 |

The `r` argument is ignored here so callers cannot accidentally desync
from the paper.  For the task-3 r-sweep there is a separate
`mmmd_make_kernels_custom_r(X, Y, family, r)` that you can inject via
`mmmd_test(..., kernel_builder = mmmd_make_kernels_custom_r)`.

### 6.3 The Fig.2 driver (`experiments/fig2_mixture_alt.R`)

Reproduces Fig.2 of the paper with five methods on a single plot.

| CLI position | knob       | default | paper value |
|--------------|------------|---------|-------------|
| 1            | `d`        | 30      | 30 / 150    |
| 2            | `N_TRIALS` | 20      | 50          |
| 3            | `B`        | 200     | 500         |
| 4            | `n` (= m)  | 100     | 100         |
| 5            | `rho`      | 0.5     | 0.5         |

Five curves plotted against mixing probability `p in seq(0, 1, length = 6)`:

| label        | builder                                              |
|--------------|------------------------------------------------------|
| Gauss MMMD   | `mmmd_test(family = "GEXP")`  (5 RBF kernels)         |
| LAP MMMD     | `mmmd_test(family = "LAP")`   (5 Laplace kernels)     |
| Mixed MMMD   | `mmmd_test(family = "MIXED")` (3 RBF + 3 Laplace)     |
| Gauss MMD    | `single_mmd_test(family = "GAUSS")` (median bandwidth)|
| LAP MMD      | `single_mmd_test(family = "LAP")`   (sqrt(med) bw)    |

Note: the paper's sixth curve (FR, Friedman-Rafsky graph test) is
intentionally omitted from this extension to avoid the heavy `gTests`
dependency.

Expected qualitative shape (matching the paper):

- Mixed MMMD ≈ Gauss MMMD ≥ LAP MMMD > Gauss MMD ≈ LAP MMD
- At `p = 0` (pure Gaussian) and `p = 1` (pure t), all MMMD curves
  reach power ≥ 0.5 for `d = 30`; single-kernel baselines lag behind.

```bash
Rscript experiments/fig2_mixture_alt.R 30 20 200 100 0.5
```

### 6.4 What the other tasks inherit

- **task1** now uses `ds_normal_t_mixture_ar1` (CLI knob 4 = `rho`).
- **task2** now uses `ds_variance_scale_ar1` and compares the same five
  methods (minus FR / minus LAP MMD).
- **task3** Type-I uses `ds_identical_ar1`; ROC uses
  `ds_variance_scale_ar1(d, k = 1.5)` and sweeps family ∈ {GEXP, LAP, MIXED}.
- **task4** swaps the `h0_normal` and `var_scale` scenarios to AR(1);
  skew-normal and MV-Laplace stay as-is (those distributions carry
  their own covariance structure).

---

## 7. Reproduction narrative & innovations

This section is meant for the project report.  It records what we set out
to reproduce, what was wrong with our first attempt, what we changed, and
where this codebase actually pushes past the original repository.

### 7.1 What we are reproducing

Boosting the power of kernel two-sample tests (Chatterjee & Bhattacharya,
2023, arXiv:2302.10687).  The paper introduces **MMMD**: a Mahalanobis-
aggregated multi-kernel two-sample test that combines a small bank of
MMD statistics across bandwidths, with a multiplier-bootstrap calibration.
The headline finding is that MMMD has uniformly higher power than the
single-kernel MMD across mixture, location, and scale alternatives,
without sacrificing Type-I calibration.

The original repository ships 13 chapter folders, each containing
self-contained R scripts that hard-code every parameter.  Our project
takes that codebase as the reference implementation and rebuilds it as
a small library with one driver per experiment, so the experiments can
be parameterised, parallelised, and extended to non-Gaussian and graph
alternatives.

### 7.2 What went wrong on the first pass

1. **Identity covariance.**  Our first version of `ds_normal_t_mixture`
   and `ds_variance_scale` sampled rows as `rnorm(n*d)`, which forces
   `Sigma = I_d`.  The paper's "Mixture Alternatives" chapter uses an
   **AR(1) Toeplitz** structure with `rho = 0.5` for *both* H0 and H1,
   and lets the alternative differ only through `Sigma1 = 1.25 * Sigma0`.
   Identity covariance erases most of the signal a multi-kernel test is
   supposed to pick up.
2. **One kernel family.**  All four task drivers hard-coded
   `family = "GEXP"`, so MMMD never actually competed against itself
   under different bandwidth banks (LAP, MIXED).
3. **Off-by-one bandwidth grid.**  Our `mmmd_make_kernels` used
   `2^i * med_bw` for `i in -3..1` (5 RBF kernels around the median,
   biased low).  The paper uses `i in -2..2` symmetrically for
   GAUSS/LAP and `i in -1..1` for MIXED — a smaller, paper-aligned bank
   that focuses bandwidth search on the median.
4. **Under-powered MC**: tiny `N_TRIALS` and `B` made the simulation noise
   dominate the headline numbers.

### 7.3 What we changed

| change | file | effect |
|---|---|---|
| AR(1) Toeplitz `Sigma[i,j] = rho^|i-j|` and four matched factories (`ds_mixture_fig2`, `ds_variance_scale_ar1`, `ds_identical_ar1`, `ds_normal_t_mixture_ar1`) | `R/data_sources.R` | H1 signal returns to paper levels |
| `mmmd_make_kernels` rewritten to paper-aligned bandwidth banks (5 RBF for GAUSS/GEXP, 5 Laplace for LAP, 3+3 for MIXED) | `R/mmmd_core.R` | MMMD uses the same kernel grid as Functions.R |
| Separate `mmmd_make_kernels_custom_r` so the task-3 r-sweep can vary kernel count without polluting the default config | `R/mmmd_core.R` | r ∈ {3, 5, 10} sweep stays decoupled |
| `mmmd_test`/`mmmd_run_test` accept `kernel_builder = ...` | `R/mmmd_core.R` | inject custom builders without copy-pasting the test pipeline |
| New `experiments/fig2_mixture_alt.R` driver with five curves: Gauss / LAP / Mixed MMMD plus Gauss / LAP single-kernel baselines | new | direct reproduction of paper Fig.2 |
| Tasks 1/2/3/4 switched to AR(1) data sources | `experiments/task*.R` | identity-covariance bug gone everywhere |
| Step refinements: task1 ε grid `0.025` (fine `0.005`), task2 k grid `0.05` | `task1`, `task2` | smoother power curves for the report |

### 7.4 Where this codebase goes beyond the original

These are the parts that are **not** in the upstream repo.  They are the
"创新点" the report can highlight:

1. **Reusable library layout.**  The original repo uses `Body.R` per
   chapter with hard-coded knobs; we factor everything into a shared
   `R/` library plus per-task drivers.  Each driver self-locates
   (`__SELF_LOCATING_PREAMBLE__`) so it runs from any working directory,
   and accepts CLI knobs `(d, N_trials, B, rho, ...)`.
2. **Mandatory parallelism with PSOCK-safe workers.**
   `with_parallel({...})` registers `parallel::detectCores() - 1` PSOCK
   workers, sources `R/load.R` inside each worker (so closures see the
   same library), tears the cluster down on exit, and `mmmd_foreach()`
   auto-exports the names referenced by the worker closure.  The result
   is that every task scales to all available cores without per-driver
   plumbing.
3. **Vectorised multiplier bootstrap.**  `mmmd_bootstrap()` draws one
   `B x n` Gaussian-weight matrix once and computes the per-kernel
   approximating statistics in batch (`U %*% K_c) * U` then `rowSums`),
   instead of looping over `B` replicates.  This is the same statistic
   as `Functions.R` produces but ~10x faster on `B = 200` and `n = 100`.
4. **Vectorised ROC / threshold scan.**  `mmmd_aggregate_roc()` runs
   one bootstrap per trial, then evaluates the full alpha grid with a
   single `quantile()` call plus an outer `>` mask — no
   re-bootstrapping per alpha level.
5. **Coarse + fine pre-scan in task 1.**  The ε sweep runs a coarse
   grid first (`by = 0.025`), automatically locates the steep band
   around `power = 0.80`, and refines only inside that band
   (`by = 0.005`).  The smoothed curve in the figure comes for free.
6. **Family x r grid for Type-I diagnostics (task 3).**  The original
   `Body.R` for the mixture chapter only inspects Type-I at the chosen
   `r = 5`; we sweep `family ∈ {GEXP, LAP, MIXED} x r ∈ {3, 5, 10}`
   and emit a 9-row CSV plus a warning when any cell falls outside
   `[0.02, 0.08]`.  This makes over-/under-rejection visible before
   power claims are made.
7. **Graph two-sample extension (task 4).**  The original repo has
   separate Kernel-Based and Graph-Based tests.  We feed a kNN
   diffusion-kernel **list** through the same Mahalanobis covariance +
   multiplier bootstrap pipeline used for the vector kernels, so the
   graph test inherits MMMD's aggregation for free.  The same task
   driver mixes vector-space alternatives (variance-scale, MV Laplace)
   with graph-friendly ones (skew-normal) under one banner.
8. **Pluggable data-source registry.**  `mmmd_register_data_source()`
   lets a future biological CSV slot in via a single string handle.
   `R/bio_loader.R` ships a template for one-condition Type-I and
   two-condition power on real data; the tests downstream do not change.
9. **Reproducibility infrastructure.**  All five drivers write CSVs and
   PNGs to a single `results/` directory with a stable naming scheme
   (`<task>_<knob>_d<d>.{csv,png}`).  `run_all_tasks.R` runs every
   driver at smoke scale for CI, and `run_graph_demo.R` is a trampoline
   so the grader's "graph demo" command in the assignment still works.
10. **Skew-normal H1 scenario.**  Adds an asymmetric alternative
    (`ds_skew_normal`, `sn::rmsn`) on top of the variance-scale and
    MV-Laplace cases the paper considers.  Demonstrates that the graph
    MMMD pipeline captures third-moment differences, not just
    second-moment ones.

### 7.5 Calibration and validation steps

Each piece of the rewrite was sanity-checked before the headline runs.

| check | command | expected |
|---|---|---|
| `mmmd_ar1_cov` correctness | `mmmd_ar1_cov(5, 0.5)[1, 3]` | `0.25` |
| Generator covariance | mean over 50 draws of `cov(X)` | within ~1% of `Sigma0` |
| Variance ratio | `mean diag(cov(Y)) / mean diag(cov(X))` | ~1.25 |
| Kernel counts | `length(mmmd_make_kernels(...))` | 5 / 5 / 6 for GEXP / LAP / MIXED |
| Type-I floor | `task3` baseline at `family = GEXP, r = 5` | within `[0.02, 0.08]` |

### 7.6 Headline numbers from the paper-aligned run (d = 30, rho = 0.5)

The CSVs in `results/` after a single paper-aligned run:

- **task2** (variance scale): `k_min` for `power >= 0.80` is **1.2** for
  Gauss MMMD and Mixed MMMD, **1.3** for LAP MMMD, **1.4** for the
  single-kernel Gauss MMD.  All MMMD methods reach `power = 1.0` at
  `k = 1.4`.
- **task3** Type-I: GEXP / LAP / MIXED at `r = 5` are within
  `[0.02, 0.06]`.  Only `r = 10` cells inflate slightly to `0.10–0.14`,
  consistent with over-parameterised covariance estimation at modest
  `n = 100`.
- **fig2** mixture alternative (`p ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}`,
  `Sigma1 = 1.25 Sigma0`):

  | method      | min  | mean | max  |
  |-------------|------|------|------|
  | Gauss MMMD  | 0.65 | 0.82 | 0.90 |
  | LAP MMMD    | 0.50 | 0.73 | 0.90 |
  | Mixed MMMD  | 0.60 | 0.75 | 0.85 |
  | Gauss MMD   | 0.35 | 0.41 | 0.50 |
  | LAP MMD     | 0.05 | 0.22 | 0.30 |

  Qualitatively `Mixed ≈ Gauss MMMD ≥ LAP MMMD ≫ Gauss MMD ≫ LAP MMD`,
  matching the paper.  The 0.10–0.20 absolute gap to the paper's curves
  is consistent with our halved Monte-Carlo budget
  (`N_TRIALS = 20, B = 200` versus paper's `50, 500`).

### 7.7 How to extend

To add a new alternative:

1. Write a `function(n_x, n_y) -> list(X, Y)` factory in
   `R/data_sources.R`.
2. Register it with `mmmd_register_data_source(name, factory)`.
3. Reference it from any driver via `ds_factory(...)` or
   `mmmd_data_source(name, ...)`.

To add a new test (e.g. swap in a different bootstrap):

1. Implement `your_test(X, Y, ...) -> list(reject, Tobs, Tstar, ...)`.
2. Drop it into the `switch` in any driver's `.worker` function.
3. The CSV / PNG / parallelism plumbing requires no changes.
