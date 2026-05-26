## ============================================================================ #
## task4_graph_demo.R                                                            #
##                                                                               #
##  plan1.md TASK 4                                                              #
##  -----------------                                                            #
##  Framework extension demos:                                                   #
##                                                                               #
##  (1) Non-Gaussian / asymmetric alternatives via skew-normal & MV-Laplace.    #
##  (2) Graph two-sample test using a kNN diffusion-kernel list, fed into the   #
##      same Mahalanobis covariance + multiplier bootstrap engine.              #
##                                                                               #
##  Outputs (in `results/`):                                                     #
##    task4_graph_summary.csv                                                    #
##    task4_graph_curve.png                                                      #
##                                                                               #
##  This file is also the canonical `run_graph_demo.R` requested by plan1.md   #
##  section 5.  A trampoline at <root>/run_graph_demo.R sources this file.      #
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

D        <- .arg(1, 5)
N_TRIALS <- .arg(2, 50)
B        <- .arg(3, 500)
M <- N   <- 100
ALPHA    <- 0.05
T_SEQ    <- c(0.1, 0.5, 1.0, 2.0, 5.0)
K_NN     <- 5

cat(sprintf("[task4] d = %d  N_trials = %d  B = %d  m = n = %d\n",
            D, N_TRIALS, B, M))
cat("[task4] diffusion times t = ", paste(T_SEQ, collapse = ","), "\n")

.t4_worker <- function(i, scenario, d, m, n, B, alpha, t_seq, k_nn,
                       ds_args) {
  ds <- switch(scenario,
    "h0_normal"      = ds_identical_normal(d),
    "var_scale"      = ds_variance_scale(d, k = ds_args$k),
    "skew_normal"    = ds_skew_normal(d, alpha = ds_args$alpha),
    "mv_laplace"     = ds_mv_laplace(d),
    stop("unknown scenario: ", scenario)
  )
  xy <- ds(m, n)
  tr <- graph_mmmd_test(xy$X, xy$Y, k_nn = k_nn, t_seq = t_seq,
                        B = B, alpha = alpha)
  as.integer(tr$reject)
}

.power_under <- function(label, scenario, ds_args = list()) {
  rejects <- mmmd_foreach(
    seq_len(N_TRIALS), .t4_worker,
    scenario = scenario, d = D, m = M, n = N, B = B, alpha = ALPHA,
    t_seq = T_SEQ, k_nn = K_NN, ds_args = ds_args,
    .packages = c("Rfast"),
    .combine = c
  )
  data.frame(scenario = label, power = mean(as.numeric(rejects)),
             trials = N_TRIALS, error = NA_character_,
             stringsAsFactors = FALSE)
}

with_parallel({
  scenarios <- list(
    list(label = "H0: identical N(0, I)", key = "h0_normal",   args = list()),
    list(label = "Variance-scale k=1.5",  key = "var_scale",   args = list(k = 1.5)),
    list(label = "Skew-normal alpha=5",   key = "skew_normal", args = list(alpha = 5)),
    list(label = "Multivariate Laplace",  key = "mv_laplace",  args = list())
  )

  results <- do.call(rbind, lapply(scenarios, function(s) {
    cat("[task4] scenario:", s$label, "\n")
    res <- tryCatch(.power_under(s$label, s$key, s$args),
                    error = function(e) data.frame(scenario = s$label,
                                                   power = NA_real_,
                                                   trials = N_TRIALS,
                                                   error = conditionMessage(e),
                                                   stringsAsFactors = FALSE))
    print(res)
    res
  }))

  out_dir  <- mmmd_results_dir()
  csv_path <- file.path(out_dir, "task4_graph_summary.csv")
  png_path <- file.path(out_dir, "task4_graph_curve.png")
  utils::write.csv(results, csv_path, row.names = FALSE)
  cat("[task4] CSV  ->", csv_path, "\n")

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p <- ggplot2::ggplot(results, ggplot2::aes(x = scenario, y = power,
                                               fill = scenario)) +
      ggplot2::geom_col(width = 0.6, show.legend = FALSE) +
      ggplot2::geom_hline(yintercept = ALPHA, linetype = "dashed",
                          colour = "red") +
      ggplot2::ylim(0, 1) +
      ggplot2::labs(title = "Graph-MMMD on (P, Q) scenarios",
                    subtitle = sprintf("kNN = %d, diffusion times t = {%s}",
                                       K_NN, paste(T_SEQ, collapse = ", ")),
                    y = "Empirical reject rate") +
      ggplot2::theme_bw() +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 25, hjust = 1))
    ggplot2::ggsave(png_path, p, width = 7, height = 4.2, dpi = 150)
  } else {
    grDevices::png(png_path, width = 900, height = 540, res = 130)
    par(mar = c(8, 4, 3, 1))
    barplot(results$power, names.arg = results$scenario, las = 2,
            ylim = c(0, 1), ylab = "reject rate",
            main = "Graph-MMMD reject rates")
    abline(h = ALPHA, lty = 2, col = "red")
    grDevices::dev.off()
  }
  cat("[task4] PNG  ->", png_path, "\n")
  cat("[task4] DONE.\n")
})
