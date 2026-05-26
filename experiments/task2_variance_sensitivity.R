## ============================================================================ #
## task2_variance_sensitivity.R                                                  #
##                                                                               #
##  plan1.md TASK 2                                                              #
##  -----------------                                                            #
##  Fix epsilon = 0.5 (in spirit) and slide k for the variance-scale alternative #
##                                                                               #
##      P = N(0, I_d),   Q = N(0, k * I_d)                                       #
##                                                                               #
##  Compare MMMD against the median-bandwidth single-kernel MMD baseline and     #
##  report the smallest k achieving power >= 0.80 for each.                      #
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

D        <- .arg(1, 10)
N_TRIALS <- .arg(2, 100)
B        <- .arg(3, 1000)
M <- N   <- 200
ALPHA    <- 0.05
KERNEL_FAMILY <- "GEXP"
KERNEL_R      <- 5
TARGET_POWER  <- 0.80
K_GRID        <- seq(1.00, 2.00, by = 0.05)

cat(sprintf("[task2] d = %d  N_trials = %d  B = %d  m = n = %d\n",
            D, N_TRIALS, B, M))

.k_worker <- function(i, k, d, m, n, B, alpha, family, r, method) {
  ds <- ds_variance_scale(d = d, k = k)
  xy <- ds(m, n)
  if (method == "mmmd") {
    tr <- mmmd_test(xy$X, xy$Y, family = family, r = r,
                    B = B, alpha = alpha)
  } else {
    tr <- single_mmd_test(xy$X, xy$Y, family = "GAUSS",
                          B = B, alpha = alpha)
  }
  as.integer(tr$reject)
}

.power_at_k <- function(k, method = c("mmmd", "single_gauss")) {
  method <- match.arg(method)
  rejects <- mmmd_foreach(
    seq_len(N_TRIALS), .k_worker,
    k = k, d = D, m = M, n = N, B = B, alpha = ALPHA,
    family = KERNEL_FAMILY, r = KERNEL_R, method = method,
    .packages = c("kernlab", "Rfast"),
    .combine = c
  )
  mean(as.numeric(rejects))
}

with_parallel({
  cat("[task2] sweeping k = ", paste(K_GRID, collapse = ","), "\n")

  pow_mmmd   <- vapply(K_GRID, .power_at_k, numeric(1), method = "mmmd")
  pow_single <- vapply(K_GRID, .power_at_k, numeric(1), method = "single_gauss")

  res <- rbind(
    data.frame(k = K_GRID, power = pow_mmmd,   method = "MMMD"),
    data.frame(k = K_GRID, power = pow_single, method = "Single MMD (Gauss)")
  )

  k_min <- function(p) {
    hit <- which(p >= TARGET_POWER)
    if (length(hit) == 0) NA_real_ else K_GRID[min(hit)]
  }
  k_min_mmmd   <- k_min(pow_mmmd)
  k_min_single <- k_min(pow_single)
  cat(sprintf("[task2] k_min  MMMD = %s   Single = %s\n",
              format(k_min_mmmd),  format(k_min_single)))

  out_dir  <- mmmd_results_dir()
  csv_path <- file.path(out_dir, sprintf("variance_sensitivity_d%d.csv", D))
  png_path <- file.path(out_dir, sprintf("variance_sensitivity_curve_d%d.png", D))
  utils::write.csv(res, csv_path, row.names = FALSE)
  cat("[task2] CSV  ->", csv_path, "\n")

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p <- ggplot2::ggplot(res, ggplot2::aes(x = k, y = power, colour = method,
                                           shape = method)) +
      ggplot2::geom_line(linewidth = 0.6) +
      ggplot2::geom_point() +
      ggplot2::geom_hline(yintercept = TARGET_POWER, linetype = "dashed",
                          colour = "red") +
      ggplot2::labs(title = sprintf("Power vs Variance-Scale k (d = %d)", D),
                    x = "k (Q has covariance k*I)", y = "Empirical Power") +
      ggplot2::theme_bw() + ggplot2::theme(legend.position = "bottom")
    ggplot2::ggsave(png_path, p, width = 6, height = 4, dpi = 150)
  } else {
    grDevices::png(png_path, width = 800, height = 540, res = 130)
    plot(K_GRID, pow_mmmd, type = "b", pch = 19, ylim = c(0, 1),
         xlab = "k", ylab = "power",
         main = sprintf("Power vs k (d = %d)", D))
    points(K_GRID, pow_single, type = "b", pch = 17, col = "red")
    legend("bottomright", c("MMMD", "Single MMD"), pch = c(19, 17),
           col = c("black", "red"))
    abline(h = TARGET_POWER, lty = 2, col = "blue")
    grDevices::dev.off()
  }
  cat("[task2] PNG  ->", png_path, "\n")
  cat("[task2] DONE.\n")
})
