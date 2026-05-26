# `data/` — biological dataset drop zone

This folder is the canonical place to drop the real biological CSVs (or
TSVs) that you want to test with MMMD.

## Quick contract

The MMMD test logic only ever sees a numeric matrix of shape
`(n_samples x n_features)`.  Whatever bio data you have, reshape it once
into that form and the test runs unchanged.

## Two typical setups

| scenario                           | what to drop here          | how to load |
|------------------------------------|----------------------------|-------------|
| Type-I baseline (one condition)    | `condition_A.csv`          | `load_bio_single("data/condition_A.csv")` |
| Power on real data (two conds)     | `condition_A.csv`, `condition_B.csv` | `load_bio_two("data/condition_A.csv", "data/condition_B.csv")` |

CSVs must have one row per sample.  If the first column is a sample id,
pass `drop_id = 1`.

## Plug it into an experiment

After dropping the file(s), edit `R/bio_loader.R` and uncomment the two
`mmmd_register_data_source(...)` blocks at the bottom.  Then any
experiment driver can pull the data by name:

```r
ds <- mmmd_data_source("bio_singlecell_typeI",
                       path = file.path(mmmd_data_dir(), "pbmc.csv"))
xy <- ds(200, 200)
mmmd_test(xy$X, xy$Y, family = "GEXP", r = 5, B = 1000)
```

## Examples of public bio matrices that fit this shape

* GTEx / TCGA expression: rows = samples, cols = genes.
* Single-cell RNA-seq: rows = cells, cols = genes (after log-normalize).
* Microbiome OTU/ASV: rows = subjects, cols = taxa abundances.
* Methylation arrays: rows = samples, cols = CpG sites.

For very high-dimensional bio data (genes >> samples) consider PCA
projection to ~50 components before feeding MMMD; the Mahalanobis
covariance estimator is more stable that way.
