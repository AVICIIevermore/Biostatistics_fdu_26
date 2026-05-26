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
# from the project root
Rscript experiments/task1_epsilon_sensitivity.R   2 100 1000
Rscript experiments/task2_variance_sensitivity.R 10 100 1000
Rscript experiments/task3_typeI_and_roc.R        10 200 200 500
Rscript experiments/task4_graph_demo.R            5  50 500
Rscript run_graph_demo.R                                # alias for task 4
Rscript run_all_tasks.R                                 # smoke-test all four
```

The trailing positional args are the experiment knobs (see top of each
driver for their meanings — typically `d`, `N_trials`, `B`).

---

## 2. Mapping plan1.md to files

| plan1.md task                       | driver                                        | outputs in `results/`                                     |
|-------------------------------------|-----------------------------------------------|-----------------------------------------------------------|
| 1. epsilon-sliding sensitivity      | `experiments/task1_epsilon_sensitivity.R`     | `epsilon_sensitivity_d{d}.csv`, `..._curve_d{d}.png`      |
| 2. k-sliding variance sensitivity   | `experiments/task2_variance_sensitivity.R`    | `variance_sensitivity_d{d}.csv`, `..._curve_d{d}.png`     |
| 3. Type-I baseline + ROC over r     | `experiments/task3_typeI_and_roc.R`           | `typeI_baseline.csv`, `roc_data.csv`, `mmmd_roc_curves_multi_r.png` |
| 4. graph two-sample / non-Gaussian  | `experiments/task4_graph_demo.R`              | `task4_graph_summary.csv`, `task4_graph_curve.png`        |

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
