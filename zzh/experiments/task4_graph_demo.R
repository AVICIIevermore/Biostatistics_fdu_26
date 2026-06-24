## ============================================================================ #
## task4_graph_demo.R                                                            #
##                                                                               #
##  plan1.md TASK 4                                                              #
##  -----------------                                                            #
##  Framework extension demos:                                                   #
##                                                                               #
##  (1) Type-I calibration under Gaussian, skew-normal, and MV-Laplace nulls.   #
##  (2) Graph two-sample test using a kNN diffusion-kernel list, fed into the   #
##      same Mahalanobis covariance + multiplier bootstrap engine.              #
##                                                                               #
##  Outputs (in `results/`):                                                     #
##    task4_graph_summary.csv                                                    #
##    task4_graph_curve.png                                                      #
##    task4_skew_type1_curve.csv                                                 #
##    task4_skew_type1_curve.png                                                 #
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
.arg_vec <- function(i, default) {
  if (length(.cli) < i) return(default)
  as.numeric(strsplit(.cli[i], ",", fixed = TRUE)[[1]])
}

D        <- .arg(1, 30)
N_TRIALS <- .arg(2, 50)
B        <- .arg(3, 200)
RHO      <- .arg(4, 0.5)
M <- N   <- 100
ALPHA    <- 0.05
T_SEQ    <- c(0.1, 0.5, 1.0, 2.0, 5.0)
K_NN     <- 5
STANDARDIZED_SKEWNESS_SEQ <- .arg_vec(5, c(0, 0.1, 0.25, 0.5, 0.75, 0.9))

cat(sprintf("[task4] d = %d  N_trials = %d  B = %d  m = n = %d  rho = %.2f\n",
            D, N_TRIALS, B, M, RHO))
cat("[task4] diffusion times t = ", paste(T_SEQ, collapse = ","), "\n")

.t4_worker <- function(i, scenario, d, m, n, B, alpha, t_seq, k_nn,
                       rho, ds_args) {
  ds <- switch(scenario,
    "ar1_normal" = ds_identical_ar1(d, rho = rho),
    "var_scale_ar1" = {
      Sigma <- ds_args$k * mmmd_ar1_cov(d, rho = rho)
      function(n_x, n_y) list(
        X = .rmvn(n_x, rep(0, d), Sigma),
        Y = .rmvn(n_y, rep(0, d), Sigma)
      )
    },
    "skew_normal" = ds_identical_skew_normal(d, alpha = ds_args$alpha),
    "mv_laplace" = {
      function(n_x, n_y) {
        if (!requireNamespace("LaplacesDemon", quietly = TRUE))
          stop("install.packages('LaplacesDemon')")
        list(
          X = LaplacesDemon::rmvl(n_x, mu = rep(0, d), Sigma = diag(1, d)),
          Y = LaplacesDemon::rmvl(n_y, mu = rep(0, d), Sigma = diag(1, d))
        )
      }
    },
    stop("unknown scenario: ", scenario)
  )
  xy <- ds(m, n)
  tr <- graph_mmmd_test(xy$X, xy$Y, k_nn = k_nn, t_seq = t_seq,
                        B = B, alpha = alpha)
  as.integer(tr$reject)
}

.type1_under_distribution <- function(label, scenario, ds_args = list()) {
  rejects <- mmmd_foreach(
    seq_len(N_TRIALS), .t4_worker,
    scenario = scenario, d = D, m = M, n = N, B = B, alpha = ALPHA,
    t_seq = T_SEQ, k_nn = K_NN, rho = RHO, ds_args = ds_args,
    .packages = c("Rfast"),
    .combine = c
  )
  data.frame(distribution = label, type1_error = mean(as.numeric(rejects)),
             trials = N_TRIALS, error = NA_character_,
             stringsAsFactors = FALSE)
}

.t4_skew_type1_worker <- function(i, shape_alpha, d, m, n, B, alpha, t_seq, k_nn) {
  ds <- ds_identical_skew_normal(d, alpha = shape_alpha)
  xy <- ds(m, n)
  tr <- graph_mmmd_test(xy$X, xy$Y, k_nn = k_nn, t_seq = t_seq,
                        B = B, alpha = alpha)
  as.integer(tr$reject)
}

.skewness_to_shape_alpha <- function(target_skewness) {
  max_skewness <- skew_normal_standard_skewness(1e6)
  if (!is.finite(target_skewness) || abs(target_skewness) >= max_skewness) {
    stop(sprintf(
      "standardized skewness must be finite and have absolute value < %.6f",
      max_skewness
    ), call. = FALSE)
  }
  if (abs(target_skewness) < .Machine$double.eps) return(0)

  sign_target <- sign(target_skewness)
  root <- stats::uniroot(
    function(a) skew_normal_standard_skewness(a) - abs(target_skewness),
    interval = c(0, 1e6),
    tol = 1e-8
  )$root
  sign_target * root
}

.type1_under_skewness <- function(standardized_skewness) {
  shape_alpha <- .skewness_to_shape_alpha(standardized_skewness)
  rejects <- mmmd_foreach(
    seq_len(N_TRIALS), .t4_skew_type1_worker,
    shape_alpha = shape_alpha, d = D, m = M, n = N, B = B, alpha = ALPHA,
    t_seq = T_SEQ, k_nn = K_NN,
    .packages = c("Rfast"),
    .combine = c
  )
  data.frame(
    shape_alpha = shape_alpha,
    standardized_skewness = standardized_skewness,
    type1_error = mean(as.numeric(rejects)),
    trials = N_TRIALS,
    alpha = ALPHA,
    error = NA_character_,
    stringsAsFactors = FALSE
  )
}

with_parallel({
  scenarios <- list(
    list(label = sprintf("AR1 normal (rho=%.2f)", RHO),
         key = "ar1_normal", args = list()),
    list(label = "AR1 normal, variance scale k=1.5",
         key = "var_scale_ar1", args = list(k = 1.5)),
    list(label = "Skew-normal (shape alpha=5)",
         key = "skew_normal", args = list(alpha = 5)),
    list(label = "Multivariate Laplace",
         key = "mv_laplace",  args = list())
  )

  results <- do.call(rbind, lapply(scenarios, function(s) {
    cat("[task4] null distribution:", s$label, "\n")
    res <- tryCatch(.type1_under_distribution(s$label, s$key, s$args),
                    error = function(e) data.frame(distribution = s$label,
                                                   type1_error = NA_real_,
                                                   trials = N_TRIALS,
                                                   error = conditionMessage(e),
                                                   stringsAsFactors = FALSE))
    print(res)
    res
  }))

  out_dir  <- mmmd_results_dir()
  csv_path <- file.path(out_dir, "task4_graph_summary.csv")
  png_path <- file.path(out_dir, "task4_graph_curve.png")
  skew_csv_path <- file.path(out_dir, "task4_skew_type1_curve.csv")
  skew_png_path <- file.path(out_dir, "task4_skew_type1_curve.png")
  utils::write.csv(results, csv_path, row.names = FALSE)
  cat("[task4] CSV  ->", csv_path, "\n")

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p <- ggplot2::ggplot(results, ggplot2::aes(x = distribution, y = type1_error,
                                               fill = distribution)) +
      ggplot2::geom_col(width = 0.6, show.legend = FALSE) +
      ggplot2::geom_hline(yintercept = ALPHA, linetype = "dashed",
                          colour = "red") +
      ggplot2::ylim(0, 1) +
      ggplot2::labs(title = "Graph-MMMD Type-I error across null distributions",
                    subtitle = sprintf("kNN = %d, diffusion times t = {%s}",
                                       K_NN, paste(T_SEQ, collapse = ", ")),
                    x = "null distribution",
                    y = "Empirical Type-I error") +
      ggplot2::theme_bw() +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 25, hjust = 1))
    ggplot2::ggsave(png_path, p, width = 7, height = 4.2, dpi = 150)
  } else {
    grDevices::png(png_path, width = 900, height = 540, res = 130)
    par(mar = c(8, 4, 3, 1))
    barplot(results$type1_error, names.arg = results$distribution, las = 2,
            ylim = c(0, 1), ylab = "Type-I error",
            main = "Graph-MMMD Type-I error across null distributions")
    abline(h = ALPHA, lty = 2, col = "red")
    grDevices::dev.off()
  }
  cat("[task4] PNG  ->", png_path, "\n")

  cat("[task4] skewness Type-I scan: standardized skewness = ",
      paste(STANDARDIZED_SKEWNESS_SEQ, collapse = ","), "\n", sep = "")
  skew_results <- do.call(rbind, lapply(STANDARDIZED_SKEWNESS_SEQ, function(gamma1) {
    cat(sprintf("[task4] skew-null standardized skewness = %.4f\n", gamma1))
    res <- tryCatch(.type1_under_skewness(gamma1),
                    error = function(e) data.frame(
                      shape_alpha = NA_real_,
                      standardized_skewness = gamma1,
                      type1_error = NA_real_,
                      trials = N_TRIALS,
                      alpha = ALPHA,
                      error = conditionMessage(e),
                      stringsAsFactors = FALSE
                    ))
    print(res)
    res
  }))
  utils::write.csv(skew_results, skew_csv_path, row.names = FALSE)
  cat("[task4] skew CSV ->", skew_csv_path, "\n")

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p_skew <- ggplot2::ggplot(
      skew_results,
      ggplot2::aes(x = standardized_skewness, y = type1_error)
    ) +
      ggplot2::geom_hline(yintercept = ALPHA, linetype = "dashed",
                          colour = "red") +
      ggplot2::geom_line(colour = "#2C7FB8", linewidth = 0.8,
                         na.rm = TRUE) +
      ggplot2::geom_point(ggplot2::aes(colour = shape_alpha), size = 2.2,
                          na.rm = TRUE) +
      ggplot2::scale_colour_gradient(low = "#7FCDBB", high = "#253494",
                                     name = "implied shape alpha") +
      ggplot2::coord_cartesian(ylim = c(0, 1)) +
      ggplot2::labs(
        title = "Graph-MMMD Type-I error under skew-normal nulls",
        subtitle = sprintf("X and Y share the same skew-normal law; nominal alpha = %.2f", ALPHA),
        x = "standardized skewness E[(X - mu)^3] / sd(X)^3",
        y = "Empirical Type-I error"
      ) +
      ggplot2::theme_bw()
    ggplot2::ggsave(skew_png_path, p_skew, width = 7, height = 4.2, dpi = 150)
  } else {
    grDevices::png(skew_png_path, width = 900, height = 540, res = 130)
    plot(skew_results$standardized_skewness, skew_results$type1_error,
         type = "b", pch = 16, ylim = c(0, 1),
         xlab = "standardized skewness E[(X - mu)^3] / sd(X)^3",
         ylab = "Empirical Type-I error",
         main = "Graph-MMMD Type-I error under skew-normal nulls")
    abline(h = ALPHA, lty = 2, col = "red")
    grDevices::dev.off()
  }
  cat("[task4] skew PNG ->", skew_png_path, "\n")
  cat("[task4] DONE.\n")
})
