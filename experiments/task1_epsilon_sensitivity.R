## ============================================================================ #
## task1_epsilon_sensitivity.R                                                   #
##                                                                               #
##  plan1.md TASK 1                                                              #
##  -----------------                                                            #
##  Slide the contamination ratio epsilon for                                    #
##                                                                               #
##      P = N(0, Sigma),   Q = (1-eps) * N(0, Sigma) + eps * t_10(0, Sigma)      #
##      Sigma[i,j] = rho^|i-j|        (AR(1) Toeplitz, rho = 0.5 by default)     #
##                                                                               #
##  Detect the smallest epsilon at which MMMD power exceeds 0.80, with auto      #
##  step-refinement in the steep band.                                           #
##                                                                               #
##  Outputs (in `results/`):                                                     #
##    epsilon_sensitivity_d{d}.csv                                               #
##    epsilon_sensitivity_curve_d{d}.png                                         #
##                                                                               #
##  Usage (any CWD):                                                             #
##    Rscript experiments/task1_epsilon_sensitivity.R [d] [N_trials] [B]         #
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

## ---- CLI args / defaults ----------------------------------------------------- #
.cli <- commandArgs(trailingOnly = TRUE)
.arg <- function(i, default) if (length(.cli) >= i) as.numeric(.cli[i]) else default

D         <- .arg(1, 30)       # dimension (paper Fig.2 uses 30 / 150)
N_TRIALS  <- .arg(2, 20)       # outer reps per epsilon
B         <- .arg(3, 200)      # bootstrap reps
RHO       <- .arg(4, 0.5)      # AR(1) covariance parameter; 0 = identity
M <- N    <- 100               # sample sizes (paper-aligned)
ALPHA     <- 0.05
KERNEL_FAMILY <- "GEXP"
KERNEL_R      <- 5

EPS_GRID_COARSE <- seq(0, 1, by = 0.025)
EPS_STEP_FINE   <- 0.005
TARGET_POWER    <- 0.80

cat(sprintf("[task1] d = %d  N_trials = %d  B = %d  m = n = %d  rho = %.2f\n",
            D, N_TRIALS, B, M, RHO))

## ---- workhorse: power at a single epsilon ------------------------------------ #
.eps_worker <- function(i, eps, d, m, n, B, alpha, family, r, rho) {
  ds <- ds_normal_t_mixture_ar1(d = d, epsilon = eps, rho = rho)
  xy <- ds(m, n)
  tr <- mmmd_test(xy$X, xy$Y, family = family, r = r,
                  B = B, alpha = alpha)
  as.integer(tr$reject)
}

.power_at_eps <- function(eps) {
  rejects <- mmmd_foreach(
    seq_len(N_TRIALS),
    .eps_worker,
    eps = eps, d = D, m = M, n = N, B = B, alpha = ALPHA,
    family = KERNEL_FAMILY, r = KERNEL_R, rho = RHO,
    .packages = c("kernlab", "Rfast"),
    .combine = c
  )
  mean(as.numeric(rejects))
}

## ---- coarse + fine sweep ----------------------------------------------------- #
with_parallel({
  cat("[task1] coarse sweep over epsilon = ", paste(EPS_GRID_COARSE, collapse = ","), "\n")
  power_coarse <- vapply(EPS_GRID_COARSE, .power_at_eps, numeric(1))
  print(data.frame(epsilon = EPS_GRID_COARSE, power = power_coarse))

  ## locate the steep band: smallest epsilon with power >= TARGET_POWER and
  ## largest epsilon with power < TARGET_POWER, then refine between them.
  hit_idx  <- which(power_coarse >= TARGET_POWER)
  if (length(hit_idx) == 0) {
    cat("[task1] coarse sweep never reached target; skipping refinement.\n")
    fine_eps <- numeric(0); fine_pow <- numeric(0)
  } else {
    hi <- min(hit_idx)
    lo <- max(c(0, which(power_coarse < TARGET_POWER & seq_along(power_coarse) < hi)))
    if (lo == 0) {
      fine_eps <- numeric(0); fine_pow <- numeric(0)
    } else {
      eps_lo <- EPS_GRID_COARSE[lo]
      eps_hi <- EPS_GRID_COARSE[hi]
      fine_eps <- setdiff(seq(eps_lo, eps_hi, by = EPS_STEP_FINE),
                          EPS_GRID_COARSE)
      cat(sprintf("[task1] refining %.3f .. %.3f at step %.3f (%d points)\n",
                  eps_lo, eps_hi, EPS_STEP_FINE, length(fine_eps)))
      fine_pow <- vapply(fine_eps, .power_at_eps, numeric(1))
    }
  }

  res <- data.frame(epsilon = EPS_GRID_COARSE, power = power_coarse,
                    stage = "coarse")
  if (length(fine_eps) > 0) {
    res <- rbind(res,
                 data.frame(epsilon = fine_eps, power = fine_pow,
                            stage = "fine"))
  }
  res <- res[order(res$epsilon), ]
  rownames(res) <- NULL

  hit <- res[res$power >= TARGET_POWER, ]
  eps_min <- if (nrow(hit) > 0) min(hit$epsilon) else NA_real_
  cat(sprintf("[task1] epsilon_min (Power >= %.2f) = %s\n",
              TARGET_POWER, format(eps_min)))

  ## ---- save -------------------------------------------------------------- #
  out_dir <- mmmd_results_dir()
  csv_path <- file.path(out_dir, sprintf("epsilon_sensitivity_d%d.csv", D))
  png_path <- file.path(out_dir, sprintf("epsilon_sensitivity_curve_d%d.png", D))
  utils::write.csv(res, csv_path, row.names = FALSE)
  cat("[task1] CSV  ->", csv_path, "\n")

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p <- ggplot2::ggplot(res, ggplot2::aes(x = epsilon, y = power)) +
      ggplot2::geom_line(linewidth = 0.6) +
      ggplot2::geom_point(ggplot2::aes(shape = stage)) +
      ggplot2::geom_hline(yintercept = TARGET_POWER, linetype = "dashed",
                          colour = "red") +
      ggplot2::geom_vline(xintercept = if (is.na(eps_min)) 0 else eps_min,
                          linetype = "dotted", colour = "blue") +
      ggplot2::labs(title = sprintf("MMMD Power vs Contamination eps (d = %d, rho = %.2f)",
                                    D, RHO),
                    subtitle = sprintf("Q = (1-eps) N(0,Sigma) + eps t_10(0,Sigma)  -  eps_min = %s",
                                       format(eps_min, digits = 3)),
                    x = expression(epsilon), y = "Empirical Power") +
      ggplot2::theme_bw()
    ggplot2::ggsave(png_path, p, width = 6, height = 4, dpi = 150)
    cat("[task1] PNG  ->", png_path, "\n")
  } else {
    grDevices::png(png_path, width = 800, height = 540, res = 130)
    plot(res$epsilon, res$power, type = "b", pch = 19,
         xlab = "epsilon", ylab = "power",
         main = sprintf("MMMD Power vs eps (d = %d)", D))
    abline(h = TARGET_POWER, lty = 2, col = "red")
    if (!is.na(eps_min)) abline(v = eps_min, lty = 3, col = "blue")
    grDevices::dev.off()
    cat("[task1] PNG  -> ", png_path, " (base graphics)\n")
  }

  cat(sprintf("[task1] DONE.  eps_min = %s\n", format(eps_min)))
})
