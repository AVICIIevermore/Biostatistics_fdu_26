## ============================================================================ #
## task2_variance_sensitivity.R                                                  #
##                                                                               #
##  plan1.md TASK 2                                                              #
##  -----------------                                                            #
##  Slide k for the variance-scale alternative                                   #
##                                                                               #
##      P = N(0, Sigma0),   Q = N(0, k * Sigma0)                                 #
##      Sigma0[i,j] = rho^|i-j|       (AR(1) Toeplitz, rho = 0.5 by default)     #
##                                                                               #
##  Compare 4 methods (Gauss/LAP/Mixed MMMD + Gauss MMD baseline) and report     #
##  the smallest k achieving power >= 0.80 for each.                             #
##                                                                               #
##  Outputs (in `results/`):                                                     #
##    variance_sensitivity_d{d}.csv                                              #
##    variance_sensitivity_curve_d{d}.png                                        #
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

D        <- .arg(1, 30)
N_TRIALS <- .arg(2, 20)
B        <- .arg(3, 200)
RHO      <- .arg(4, 0.5)
M <- N   <- 100
ALPHA    <- 0.05
TARGET_POWER  <- 0.80
K_GRID        <- seq(1.00, 2.00, by = 0.05)
METHODS  <- c("Gauss MMMD", "LAP MMMD", "Mixed MMMD", "Gauss MMD")

cat(sprintf("[task2] d = %d  N_trials = %d  B = %d  m = n = %d  rho = %.2f\n",
            D, N_TRIALS, B, M, RHO))

.k_worker <- function(i, k, d, m, n, B, alpha, method, rho) {
  ds <- ds_variance_scale_ar1(d = d, k = k, rho = rho)
  xy <- ds(m, n)
  tr <- switch(method,
    "Gauss MMMD" = mmmd_test(xy$X, xy$Y, family = "GEXP",  B = B, alpha = alpha),
    "LAP MMMD"   = mmmd_test(xy$X, xy$Y, family = "LAP",   B = B, alpha = alpha),
    "Mixed MMMD" = mmmd_test(xy$X, xy$Y, family = "MIXED", B = B, alpha = alpha),
    "Gauss MMD"  = single_mmd_test(xy$X, xy$Y, family = "GAUSS",
                                   B = B, alpha = alpha),
    stop("Unknown method: ", method)
  )
  as.integer(tr$reject)
}

.power_at_k <- function(k, method) {
  rejects <- mmmd_foreach(
    seq_len(N_TRIALS), .k_worker,
    k = k, d = D, m = M, n = N, B = B, alpha = ALPHA,
    method = method, rho = RHO,
    .packages = c("kernlab", "Rfast"),
    .combine = c
  )
  mean(as.numeric(rejects))
}

with_parallel({
  cat("[task2] sweeping k = ", paste(K_GRID, collapse = ","), "\n")

  grid <- expand.grid(k = K_GRID, method = METHODS,
                      KEEP.OUT.ATTRS = FALSE,
                      stringsAsFactors = FALSE)
  grid$power <- vapply(seq_len(nrow(grid)), function(j) {
    cat(sprintf("[task2]  k = %.2f  method = %-12s ... ",
                grid$k[j], grid$method[j]))
    pw <- .power_at_k(grid$k[j], grid$method[j])
    cat(sprintf("power = %.3f\n", pw))
    pw
  }, numeric(1))

  for (m_name in METHODS) {
    sub <- grid[grid$method == m_name, ]
    sub <- sub[order(sub$k), ]
    hit <- which(sub$power >= TARGET_POWER)
    k_min <- if (length(hit) == 0) NA_real_ else sub$k[min(hit)]
    cat(sprintf("[task2] k_min  %-12s = %s\n", m_name, format(k_min)))
  }

  out_dir  <- mmmd_results_dir()
  csv_path <- file.path(out_dir, sprintf("variance_sensitivity_d%d.csv", D))
  png_path <- file.path(out_dir, sprintf("variance_sensitivity_curve_d%d.png", D))
  utils::write.csv(grid, csv_path, row.names = FALSE)
  cat("[task2] CSV  ->", csv_path, "\n")

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    grid$method <- factor(grid$method, levels = METHODS)
    p <- ggplot2::ggplot(grid, ggplot2::aes(x = k, y = power, colour = method,
                                            shape = method, group = method)) +
      ggplot2::geom_line(linewidth = 0.6) +
      ggplot2::geom_point(size = 2) +
      ggplot2::geom_hline(yintercept = TARGET_POWER, linetype = "dashed",
                          colour = "red") +
      ggplot2::ylim(0, 1) +
      ggplot2::labs(title = sprintf("Power vs Variance-Scale k (d = %d, rho = %.2f)",
                                    D, RHO),
                    x = "k  (Q has covariance k * Sigma0)",
                    y = "Empirical Power") +
      ggplot2::theme_bw() +
      ggplot2::theme(legend.position = "right",
                     legend.title = ggplot2::element_blank())
    ggplot2::ggsave(png_path, p, width = 7, height = 4.2, dpi = 150)
  } else {
    grDevices::png(png_path, width = 900, height = 540, res = 130)
    cols <- c("black", "red", "darkgreen", "blue")
    plot(0, 0, xlim = range(K_GRID), ylim = c(0, 1), type = "n",
         xlab = "k", ylab = "power",
         main = sprintf("Power vs k (d = %d, rho = %.2f)", D, RHO))
    for (j in seq_along(METHODS)) {
      sub <- grid[grid$method == METHODS[j], ]
      sub <- sub[order(sub$k), ]
      lines(sub$k, sub$power, col = cols[j], lwd = 1.6, type = "b", pch = j)
    }
    legend("bottomright", METHODS, col = cols, pch = seq_along(METHODS),
           lwd = 1.6)
    abline(h = TARGET_POWER, lty = 2, col = "grey50")
    grDevices::dev.off()
  }
  cat("[task2] PNG  ->", png_path, "\n")
  cat("[task2] DONE.\n")
})
