## ============================================================================ #
## task3_typeI_and_roc.R                                                         #
##                                                                               #
##  plan1.md TASK 3                                                              #
##  -----------------                                                            #
##  (a) Type-I error baseline at alpha = 0.05.                                   #
##      Uses AR(1) Sigma (rho = 0.5).  Sweeps both kernel family                 #
##      (GEXP/LAP/MIXED) and number-of-kernels r in {3, 5, 10}.                  #
##  (b) ROC curves for each kernel family, using a single bootstrap +            #
##      vectorised threshold scan.                                               #
##                                                                               #
##  Outputs (in `results/`):                                                     #
##    typeI_baseline.csv        (rows: family x r, cols: family, r, type1)      #
##    roc_data.csv              (rows: alpha grid x family)                     #
##    mmmd_roc_curves_multi_r.png                                                #
## ============================================================================ #

## __SELF_LOCATING_PREAMBLE__
.script_dir <- tryCatch({
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable() &&
      nzchar(rstudioapi::getActiveDocumentContext()$path)) {
    dirname(rstudioapi::getActiveDocumentContext()$path)
  } else {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
    if (length(file_arg) > 0) {
      dirname(normalizePath(file_arg[1]))
    } else if (!is.null(sys.frame(1)$ofile)) {
      dirname(normalizePath(sys.frame(1)$ofile))
    } else {
      getwd()
    }
  }
}, error = function(e) getwd())
## __END_PREAMBLE__

source(file.path(.script_dir, "..", "R", "load.R"))

.cli <- commandArgs(trailingOnly = TRUE)
.arg <- function(i, default) if (length(.cli) >= i) as.numeric(.cli[i]) else default

D            <- .arg(1, 30)
N_TRIALS_T1  <- .arg(2, 50)
N_TRIALS_ROC <- .arg(3, 100)
B            <- .arg(4, 200)
RHO          <- .arg(5, 0.5)
M <- N       <- 100
ALPHA        <- 0.05
ALT_K        <- 1.15
R_GRID       <- c(3, 5, 10)
FAM_GRID     <- c("GEXP", "LAP", "MIXED")
ALPHA_GRID   <- seq(0, 1, by = 0.01)

cat(sprintf("[task3] d = %d  N_T1 = %d  N_ROC = %d  B = %d  m = n = %d  rho = %.2f\n",
            D, N_TRIALS_T1, N_TRIALS_ROC, B, M, RHO))

## ---- (a) Type-I error baseline ---------------------------------------------- #
.t1_worker <- function(i, d, m, n, B, alpha, family, r, rho) {
  ds <- ds_identical_ar1(d, rho = rho)
  xy <- ds(m, n)
  tr <- mmmd_test(xy$X, xy$Y, family = family, r = r,
                  B = B, alpha = alpha,
                  kernel_builder = mmmd_make_kernels_custom_r)
  as.integer(tr$reject)
}

.typeI_one_cell <- function(family, r) {
  rejects <- mmmd_foreach(
    seq_len(N_TRIALS_T1), .t1_worker,
    d = D, m = M, n = N, B = B, alpha = ALPHA,
    family = family, r = r, rho = RHO,
    .packages = c("kernlab", "Rfast"),
    .combine = c
  )
  mean(as.numeric(rejects))
}

## ---- (b) ROC: collect (Tobs, Tstar) for H0 and H1 trials -------------------- #
.roc_h0_worker <- function(i, d, m, n, B, family, rho) {
  ds <- ds_identical_ar1(d, rho = rho)
  xy <- ds(m, n)
  list(mmmd_run_test(xy$X, xy$Y, family = family, B = B))
}

.roc_h1_worker <- function(i, d, m, n, B, family, rho, alt_k) {
  ds <- ds_variance_scale_ar1(d, k = alt_k, rho = rho)
  xy <- ds(m, n)
  list(mmmd_run_test(xy$X, xy$Y, family = family, B = B))
}

.roc_one_family <- function(family) {
  trials_h0 <- mmmd_foreach(
    seq_len(N_TRIALS_ROC), .roc_h0_worker,
    d = D, m = M, n = N, B = B, family = family, rho = RHO,
    .packages = c("kernlab", "Rfast"),
    .combine = c
  )
  if (!is.null(trials_h0$Tobs))
    trials_h0 <- list(trials_h0)

  trials_h1 <- mmmd_foreach(
    seq_len(N_TRIALS_ROC), .roc_h1_worker,
    d = D, m = M, n = N, B = B, family = family, rho = RHO, alt_k = ALT_K,
    .packages = c("kernlab", "Rfast"),
    .combine = c
  )
  if (!is.null(trials_h1$Tobs))
    trials_h1 <- list(trials_h1)

  trials <- c(trials_h0, trials_h1)
  is_alt <- c(rep(FALSE, length(trials_h0)), rep(TRUE, length(trials_h1)))
  roc <- mmmd_aggregate_roc(trials, is_alt, alphas = ALPHA_GRID)
  roc$family <- family
  roc
}

## ---------------------------------------------------------------------------- #
with_parallel({
  cat("[task3] (a) Type-I baseline (AR(1) identical generator)\n")
  t1_grid <- expand.grid(family = FAM_GRID, r = R_GRID,
                         KEEP.OUT.ATTRS = FALSE,
                         stringsAsFactors = FALSE)
  t1_grid$type1 <- vapply(seq_len(nrow(t1_grid)), function(j) {
    cat(sprintf("[task3]  family = %-6s r = %2d ... ",
                t1_grid$family[j], t1_grid$r[j]))
    pw <- .typeI_one_cell(t1_grid$family[j], t1_grid$r[j])
    cat(sprintf("type1 = %.3f\n", pw))
    pw
  }, numeric(1))
  print(t1_grid)
  warn <- t1_grid$type1 > 0.08 | t1_grid$type1 < 0.02
  if (any(warn)) {
    cat("[task3] WARNING: Type-I outside [0.02, 0.08] for cells:\n")
    print(t1_grid[warn, ])
  }

  cat("[task3] (b) ROC sweep over family in {",
      paste(FAM_GRID, collapse = ","), "}\n")
  roc_all <- do.call(rbind, lapply(FAM_GRID, .roc_one_family))

  out_dir <- mmmd_results_dir()
  utils::write.csv(t1_grid, file.path(out_dir, "typeI_baseline.csv"),
                   row.names = FALSE)
  utils::write.csv(roc_all, file.path(out_dir, "roc_data.csv"),
                   row.names = FALSE)
  png_path <- file.path(out_dir, "mmmd_roc_curves_multi_r.png")

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p <- ggplot2::ggplot(roc_all, ggplot2::aes(x = fpr, y = tpr,
                                               colour = family,
                                               group = family)) +
      ggplot2::geom_line(linewidth = 0.7) +
      ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                           colour = "grey50") +
      ggplot2::labs(title = sprintf("MMMD ROC by family (d = %d, k = %.2f, rho = %.2f)",
                                    D, ALT_K, RHO),
                    subtitle = sprintf("m = n = %d, B = %d, N_trials = %d",
                                       M, B, N_TRIALS_ROC),
                    x = "FPR (empirical Type-I)",
                    y = "TPR (empirical Power)",
                    colour = "family") +
      ggplot2::coord_equal() + ggplot2::theme_bw() +
      ggplot2::theme(legend.position = "bottom")
    ggplot2::ggsave(png_path, p, width = 5.5, height = 5.5, dpi = 150)
  } else {
    grDevices::png(png_path, width = 800, height = 800, res = 130)
    cols <- c("black", "red", "blue")
    plot(0, 0, xlim = c(0, 1), ylim = c(0, 1), type = "n",
         xlab = "FPR", ylab = "TPR",
         main = sprintf("ROC by family (d = %d, k = %.2f)", D, ALT_K))
    abline(0, 1, lty = 2, col = "grey50")
    for (j in seq_along(FAM_GRID)) {
      sub <- roc_all[roc_all$family == FAM_GRID[j], ]
      lines(sub$fpr, sub$tpr, col = cols[j], lwd = 1.6)
    }
    legend("bottomright", legend = FAM_GRID, col = cols, lwd = 1.6)
    grDevices::dev.off()
  }
  cat("[task3] PNG  ->", png_path, "\n")
  cat("[task3] DONE.\n")
})
