## ============================================================================ #
## task3_typeI_and_roc.R                                                         #
##                                                                               #
##  plan1.md TASK 3                                                              #
##  -----------------                                                            #
##  (a) Type-I error baseline at alpha = 0.05.                                   #
##  (b) ROC curves for varying number-of-kernels r in {3, 5, 10}                 #
##      using a single bootstrap + vectorised threshold scan.                    #
##                                                                               #
##  Outputs (in `results/`):                                                     #
##    typeI_baseline.csv                                                         #
##    roc_data.csv                                                               #
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

D            <- .arg(1, 10)
N_TRIALS_T1  <- .arg(2, 200)   # plan calls 1000; default lower for reasonable runtime
N_TRIALS_ROC <- .arg(3, 200)
B            <- .arg(4, 500)
M <- N       <- 200
ALPHA        <- 0.05
ALT_K        <- 1.5
R_GRID       <- c(3, 5, 10)
ALPHA_GRID   <- seq(0, 1, by = 0.01)

cat(sprintf("[task3] d = %d  N_T1 = %d  N_ROC = %d  B = %d  m = n = %d\n",
            D, N_TRIALS_T1, N_TRIALS_ROC, B, M))

## ---- (a) Type-I error baseline ---------------------------------------------- #
.t1_worker <- function(i, d, m, n, B, alpha, r) {
  ds <- ds_identical_normal(d)
  xy <- ds(m, n)
  tr <- mmmd_test(xy$X, xy$Y, family = "GEXP", r = r,
                  B = B, alpha = alpha)
  as.integer(tr$reject)
}

.typeI_one_r <- function(r) {
  rejects <- mmmd_foreach(
    seq_len(N_TRIALS_T1), .t1_worker,
    d = D, m = M, n = N, B = B, alpha = ALPHA, r = r,
    .packages = c("kernlab", "Rfast"),
    .combine = c
  )
  mean(as.numeric(rejects))
}

## ---- (b) ROC: collect (Tobs, Tstar) for H0 and H1 trials -------------------- #
.roc_h0_worker <- function(i, d, m, n, B, r) {
  ds <- ds_identical_normal(d)
  xy <- ds(m, n)
  list(mmmd_run_test(xy$X, xy$Y, family = "GEXP", r = r, B = B))
}

.roc_h1_worker <- function(i, d, m, n, B, r, alt_k) {
  ds <- ds_variance_scale(d, k = alt_k)
  xy <- ds(m, n)
  list(mmmd_run_test(xy$X, xy$Y, family = "GEXP", r = r, B = B))
}

.roc_one_r <- function(r) {
  trials_h0 <- mmmd_foreach(
    seq_len(N_TRIALS_ROC), .roc_h0_worker,
    d = D, m = M, n = N, B = B, r = r,
    .packages = c("kernlab", "Rfast"),
    .combine = c
  )
  if (!is.null(trials_h0$Tobs))   # single result not wrapped
    trials_h0 <- list(trials_h0)

  trials_h1 <- mmmd_foreach(
    seq_len(N_TRIALS_ROC), .roc_h1_worker,
    d = D, m = M, n = N, B = B, r = r, alt_k = ALT_K,
    .packages = c("kernlab", "Rfast"),
    .combine = c
  )
  if (!is.null(trials_h1$Tobs))
    trials_h1 <- list(trials_h1)

  trials <- c(trials_h0, trials_h1)
  is_alt <- c(rep(FALSE, length(trials_h0)), rep(TRUE, length(trials_h1)))
  roc <- mmmd_aggregate_roc(trials, is_alt, alphas = ALPHA_GRID)
  roc$r <- r
  roc
}

## ---------------------------------------------------------------------------- #
with_parallel({
  cat("[task3] (a) Type-I baseline (synthetic identical normal)\n")
  typeI_rates <- vapply(R_GRID, .typeI_one_r, numeric(1))
  typeI_df <- data.frame(r = R_GRID, type1 = typeI_rates)
  print(typeI_df)
  warn <- typeI_df$type1 > 0.08 | typeI_df$type1 < 0.02
  if (any(warn)) {
    cat("[task3] WARNING: Type-I outside [0.02, 0.08] for r = ",
        paste(R_GRID[warn], collapse = ","), "\n")
  }

  cat("[task3] (b) ROC sweep over r in {", paste(R_GRID, collapse = ","), "}\n")
  roc_all <- do.call(rbind, lapply(R_GRID, .roc_one_r))

  out_dir <- mmmd_results_dir()
  utils::write.csv(typeI_df, file.path(out_dir, "typeI_baseline.csv"),
                   row.names = FALSE)
  utils::write.csv(roc_all, file.path(out_dir, "roc_data.csv"),
                   row.names = FALSE)
  png_path <- file.path(out_dir, "mmmd_roc_curves_multi_r.png")

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p <- ggplot2::ggplot(roc_all, ggplot2::aes(x = fpr, y = tpr,
                                               colour = factor(r),
                                               group = factor(r))) +
      ggplot2::geom_line(linewidth = 0.7) +
      ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                           colour = "grey50") +
      ggplot2::labs(title = sprintf("MMMD ROC (d = %d, k = %.2f, m = n = %d)",
                                    D, ALT_K, M),
                    x = "FPR (empirical Type-I)",
                    y = "TPR (empirical Power)",
                    colour = "kernel count r") +
      ggplot2::coord_equal() + ggplot2::theme_bw() +
      ggplot2::theme(legend.position = "bottom")
    ggplot2::ggsave(png_path, p, width = 5.5, height = 5.5, dpi = 150)
  } else {
    grDevices::png(png_path, width = 800, height = 800, res = 130)
    cols <- c("black", "red", "blue")
    plot(0, 0, xlim = c(0, 1), ylim = c(0, 1), type = "n",
         xlab = "FPR", ylab = "TPR",
         main = sprintf("ROC by r (d = %d, k = %.2f)", D, ALT_K))
    abline(0, 1, lty = 2, col = "grey50")
    for (j in seq_along(R_GRID)) {
      sub <- roc_all[roc_all$r == R_GRID[j], ]
      lines(sub$fpr, sub$tpr, col = cols[j], lwd = 1.6)
    }
    legend("bottomright", legend = paste("r =", R_GRID),
           col = cols, lwd = 1.6)
    grDevices::dev.off()
  }
  cat("[task3] PNG  ->", png_path, "\n")
  cat("[task3] DONE.\n")
})
