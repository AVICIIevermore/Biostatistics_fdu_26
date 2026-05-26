## ============================================================================ #
## bio_loader.R                                                                  #
##                                                                               #
##  Plug-in template for real biological data.  This file is the canonical      #
##  example of how to extend the project to a new dataset without touching the  #
##  test logic.                                                                  #
##                                                                               #
##  The contract for any data-source factory is:                                 #
##                                                                               #
##      factory <- function(...) function(n_x, n_y) list(X = ..., Y = ...)      #
##                                                                               #
##  X and Y must be numeric matrices with rows = observations.  After defining  #
##  a factory, register it once and any experiment script can target it via    #
##  its registered name.                                                        #
##                                                                               #
##  Sources of public bio data this stub is designed to ingest:                 #
##                                                                               #
##    * Gene-expression: GTEx / TCGA tab-separated samples-by-genes matrices.   #
##    * Single-cell RNA-seq: a cell x gene log-counts matrix per condition.    #
##    * Microbiome: OTU / ASV abundance (samples x taxa) per cohort.            #
##    * Imaging: pre-flattened pixel vectors per sample.                        #
##                                                                               #
##  All a user must supply is *one* CSV (or two CSVs) with rows = observations. #
## ============================================================================ #

#' Load a single CSV / TSV of observations and return a generic resampling
#' factory.  Use this for Type-I-error baselines on real data (a single
#' homogeneous condition).
#'
#' @param path       path to a CSV / TSV (rows = observations)
#' @param header     read.csv header argument
#' @param sep        field separator (",", "\t", ...)
#' @param drop_id    columns to drop (e.g. sample IDs).  Indices or names.
#' @param scale      logical, scale columns to unit variance after centring
#' @return  closure compatible with mmmd_data_source(...)
load_bio_single <- function(path, header = TRUE, sep = ",",
                            drop_id = NULL, scale = TRUE) {
  if (!file.exists(path))
    stop("File not found: ", path,
         "\n  (drop your CSV under data/ and update the path).")
  raw <- utils::read.table(path, header = header, sep = sep,
                           check.names = FALSE)
  if (!is.null(drop_id)) raw <- raw[, -drop_id, drop = FALSE]
  M <- as.matrix(raw)
  storage.mode(M) <- "numeric"
  if (scale) M <- scale(M, center = TRUE, scale = TRUE)
  M <- M[, apply(M, 2, function(z) all(is.finite(z))), drop = FALSE]
  ds_resample_from_matrix(M)
}

#' Load two CSVs (two conditions) and return a paired-resampling factory.
#' Use this for power experiments when both groups are observed.
load_bio_two <- function(path_x, path_y, header = TRUE, sep = ",",
                         drop_id = NULL, scale = TRUE) {
  fx <- load_bio_single(path_x, header, sep, drop_id, scale)
  fy <- load_bio_single(path_y, header, sep, drop_id, scale)
  MX <- environment(fx)$M
  MY <- environment(fy)$M
  common <- intersect(colnames(MX), colnames(MY))
  if (length(common) > 0) {
    MX <- MX[, common, drop = FALSE]
    MY <- MY[, common, drop = FALSE]
  }
  ds_resample_two_matrices(MX, MY)
}

## ---- registration ------------------------------------------------------------ #
##
## Edit the paths below to point at YOUR biological CSV(s), then uncomment the
## `mmmd_register_data_source(...)` lines.  After that any experiment driver
## can swap in the real data with one line, e.g.:
##
##     ds <- mmmd_data_source("bio_singlecell_typeI",
##                            path = file.path(mmmd_data_dir(), "pbmc.csv"))
##
## (Type-I baseline: same condition resampled.)

# example_path_h0 <- file.path(mmmd_data_dir(), "condition_A.csv")
# example_path_h1 <- file.path(mmmd_data_dir(), "condition_B.csv")
#
# mmmd_register_data_source("bio_singlecell_typeI", function(path = example_path_h0,
#                                                            ...) {
#   load_bio_single(path = path, ...)
# })
#
# mmmd_register_data_source("bio_singlecell_power",
#                           function(path_x = example_path_h0,
#                                    path_y = example_path_h1, ...) {
#   load_bio_two(path_x = path_x, path_y = path_y, ...)
# })

invisible(NULL)
