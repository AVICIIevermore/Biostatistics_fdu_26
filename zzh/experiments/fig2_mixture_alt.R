## ============================================================================ #
## fig2_mixture_alt.R                                                            #
##                                                                               #
##  Reproduction of Boosting-MMD Fig.2 ("Empirical powers as a function of the   #
##  mixing proportion").                                                         #
##                                                                               #
##  Data:                                                                        #
##    P_i.i.d. ~ (1 - p) N(0, Sigma0) + p t_df(0, Sigma0)                        #
##    Q_i.i.d. ~ (1 - p) N(0, Sigma1) + p t_df(0, Sigma1),  Sigma1 = c * Sigma0  #
##    Sigma0[i,j] = rho^|i-j|  (AR(1) Toeplitz, rho = 0.5)                       #
##                                                                               #
##  Methods (5 curves):                                                          #
##    Gauss MMMD  (GEXP, l0..l1 = -2..2,  r = 5)                                 #
##    LAP   MMMD  (LAP,  l0..l1 = -2..2,  r = 5)                                 #
##    Mixed MMMD  (MIXED, l0..l1 = -1..1, r = 6 = 3 RBF + 3 Laplace)             #
##    Gauss MMD   (single Gaussian, median bandwidth)                            #
##    LAP   MMD   (single Laplace,  sqrt(median bandwidth))                      #
##                                                                               #
##  Outputs (in `results/`):                                                     #
##    fig2_mixture_alt_d{d}.csv                                                  #
##    fig2_mixture_alt_d{d}.png                                                  #
##                                                                               #
##  Usage:                                                                       #
##    Rscript experiments/fig2_mixture_alt.R [d] [N_trials] [B] [n] [rho]        #
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

D          <- .arg(1, 30)
N_TRIALS   <- .arg(2, 20)
B          <- .arg(3, 200)
M <- N     <- .arg(4, 100)
RHO        <- .arg(5, 0.5)
SIGMA_MULT <- 1.25
ALPHA      <- 0.05
P_GRID     <- seq(0, 1, length.out = 6)
METHODS    <- c("Gauss MMMD", "LAP MMMD", "Mixed MMMD",
                "Gauss MMD",  "LAP MMD")

cat(sprintf("[fig2] d = %d  N_trials = %d  B = %d  m = n = %d  rho = %.2f\n",
            D, N_TRIALS, B, M, RHO))
cat("[fig2] mixing probability grid p = ",
    paste(sprintf("%.2f", P_GRID), collapse = ","), "\n")

.fig2_worker <- function(i, p, d, n, B, alpha, method, rho, sigma_mult) {
  ds <- ds_mixture_fig2(d = d, p = p, rho = rho, sigma.mult = sigma_mult)
  xy <- ds(n, n)
  tr <- switch(method,
    "Gauss MMMD" = mmmd_test(xy$X, xy$Y, family = "GEXP",
                             B = B, alpha = alpha),
    "LAP MMMD"   = mmmd_test(xy$X, xy$Y, family = "LAP",
                             B = B, alpha = alpha),
    "Mixed MMMD" = mmmd_test(xy$X, xy$Y, family = "MIXED",
                             B = B, alpha = alpha),
    "Gauss MMD"  = single_mmd_test(xy$X, xy$Y, family = "GAUSS",
                                   B = B, alpha = alpha),
    "LAP MMD"    = single_mmd_test(xy$X, xy$Y, family = "LAP",
                                   B = B, alpha = alpha),
    stop("Unknown method: ", method)
  )
  as.integer(tr$reject)
}

.power_cell <- function(p, method) {
  rejects <- mmmd_foreach(
    seq_len(N_TRIALS), .fig2_worker,
    p = p, d = D, n = M, B = B, alpha = ALPHA, method = method,
    rho = RHO, sigma_mult = SIGMA_MULT,
    .packages = c("kernlab", "Rfast"),
    .combine = c
  )
  mean(as.numeric(rejects))
}

with_parallel({
  grid <- expand.grid(p = P_GRID, method = METHODS,
                      KEEP.OUT.ATTRS = FALSE,
                      stringsAsFactors = FALSE)

  cat("[fig2] running", nrow(grid), "(p, method) cells x", N_TRIALS,
      "trials each\n")

  grid$power <- vapply(seq_len(nrow(grid)), function(j) {
    cat(sprintf("[fig2]  p = %.2f  method = %-12s ... ",
                grid$p[j], grid$method[j]))
    pw <- .power_cell(grid$p[j], grid$method[j])
    cat(sprintf("power = %.3f\n", pw))
    pw
  }, numeric(1))

  grid$d        <- D
  grid$n_trials <- N_TRIALS

  out_dir  <- mmmd_results_dir()
  csv_path <- file.path(out_dir, sprintf("fig2_mixture_alt_d%d.csv", D))
  png_path <- file.path(out_dir, sprintf("fig2_mixture_alt_d%d.png", D))
  utils::write.csv(grid, csv_path, row.names = FALSE)
  cat("[fig2] CSV  ->", csv_path, "\n")

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    grid$method <- factor(grid$method, levels = METHODS)
    p <- ggplot2::ggplot(grid,
                         ggplot2::aes(x = p, y = power,
                                      colour = method, shape = method,
                                      group  = method)) +
      ggplot2::geom_line(linewidth = 0.6) +
      ggplot2::geom_point(size = 2) +
      ggplot2::geom_hline(yintercept = ALPHA, linetype = "dashed",
                          colour = "grey50") +
      ggplot2::ylim(0, 1) +
      ggplot2::labs(title = sprintf("Mixture Alternatives (d = %d, n = %d, rho = %.2f)",
                                    D, M, RHO),
                    subtitle = sprintf("Sigma1 = %.2f * Sigma0   B = %d   N_trials = %d",
                                       SIGMA_MULT, B, N_TRIALS),
                    x = "Mixing probability  p",
                    y = "Empirical Power") +
      ggplot2::theme_bw() +
      ggplot2::theme(legend.position = "right",
                     legend.title = ggplot2::element_blank())
    ggplot2::ggsave(png_path, p, width = 7, height = 4.5, dpi = 150)
  } else {
    grDevices::png(png_path, width = 900, height = 540, res = 130)
    cols <- c("black", "red", "darkgreen", "blue", "purple")
    plot(0, 0, xlim = c(0, 1), ylim = c(0, 1), type = "n",
         xlab = "p", ylab = "power",
         main = sprintf("Mixture Alt (d = %d, rho = %.2f)", D, RHO))
    for (j in seq_along(METHODS)) {
      sub <- grid[grid$method == METHODS[j], ]
      sub <- sub[order(sub$p), ]
      lines(sub$p, sub$power, col = cols[j], lwd = 1.6, type = "b", pch = j)
    }
    legend("topleft", METHODS, col = cols, pch = seq_along(METHODS), lwd = 1.6)
    abline(h = ALPHA, lty = 2, col = "grey50")
    grDevices::dev.off()
  }
  cat("[fig2] PNG  ->", png_path, "\n")
  cat("[fig2] DONE.\n")
})
