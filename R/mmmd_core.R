## ============================================================================ #
## mmmd_core.R                                                                   #
##                                                                               #
##  Reusable, side-effect-free implementation of the MMMD pipeline:              #
##                                                                               #
##    1. kernel-list construction (RBF / Laplace, exponential bandwidth grid)    #
##    2. unbiased MMD_u^2 statistic per kernel                                   #
##    3. Mahalanobis-aggregated multi-kernel statistic                           #
##    4. multiplier (Gaussian-weight) bootstrap of the null distribution         #
##    5. vectorised threshold scan -> reject indicator over an alpha grid        #
##                                                                               #
##  Designed for plan1.md tasks 1-4: it accepts an arbitrary kernel-matrix list  #
##  so the same engine handles vector-space kernels (RBF / Laplace) AND graph    #
##  kernels (task 4) without modification.                                       #
##                                                                               #
##  Style note: this file intentionally mirrors the logic of the per-folder      #
##  Functions.R but factors it out so experiments can be composed.               #
## ============================================================================ #

## ---- bandwidth heuristics --------------------------------------------------- #

#' Median heuristic on squared pairwise distances of stacked (X, Y).
#' Returns 1 / median(d^2) so it is plug-compatible with kernlab::rbfdot(sigma=).
mmmd_med_bw <- function(X, Y) {
  Z <- rbind(as.matrix(X), as.matrix(Y))
  nu <- median(stats::dist(Z) ^ 2)
  if (!is.finite(nu) || nu <= 0) nu <- 1
  1 / nu
}

#' Exponential grid of bandwidths around the median heuristic.
#' Returns r = (l1 - l0 + 1) bandwidths { 2^i * med_bw : i = l0..l1 }.
mmmd_expo_band <- function(X, Y, l0 = -3, l1 = 1) {
  med <- mmmd_med_bw(X, Y)
  (2 ^ seq.int(l0, l1)) * med
}

#' Build a list of `kernlab` kernel objects following the paper's Mixture
#' Alternatives convention.
#'
#' Per family the bandwidth grid is fixed to match the original code in
#' `Mixture Alternatives/Code/Kernel Based Test/Functions.R`:
#'   GAUSS / GEXP : { 2^i * med_bw : i = -2..2 }   (5 RBF kernels)
#'   LAP          : { 2^i * med_bw : i = -2..2 }   (5 Laplace, sqrt-scaled)
#'   MIXED        : { 2^i * med_bw : i = -1..1 } x {RBF, Laplace}   (6 kernels)
#'
#' The `r` argument is *ignored* here so callers cannot accidentally desync
#' from the paper.  Use `mmmd_make_kernels_custom_r()` if you want to sweep
#' the number-of-kernels knob (e.g. plan1.md task 3).
mmmd_make_kernels <- function(X, Y, family = "GEXP", r = 5) {
  if (!requireNamespace("kernlab", quietly = TRUE)) stop("install.packages('kernlab')")
  switch(family,
    "GAUSS" = ,
    "GEXP"  = {
      bw <- mmmd_expo_band(X, Y, l0 = -2, l1 = 2)
      lapply(bw, function(s) kernlab::rbfdot(sigma = s))
    },
    "LAP"   = {
      bw <- mmmd_expo_band(X, Y, l0 = -2, l1 = 2)
      lapply(bw, function(s) kernlab::laplacedot(sigma = sqrt(s)))
    },
    "MIXED" = {
      bw <- mmmd_expo_band(X, Y, l0 = -1, l1 = 1)
      c(lapply(bw, function(s) kernlab::rbfdot(sigma = s)),
        lapply(bw, function(s) kernlab::laplacedot(sigma = sqrt(s))))
    },
    stop("Unknown kernel family: ", family)
  )
}

#' Variant of `mmmd_make_kernels()` that honours the `r` knob (used by
#' plan1.md task 3 to sweep r in {3, 5, 10}).  Builds an exponential
#' bandwidth grid centred on the median heuristic with `r` slots; for
#' MIXED, the resulting grid is duplicated across RBF + Laplace yielding
#' 2*r kernels.
mmmd_make_kernels_custom_r <- function(X, Y, family = "GEXP", r = 5) {
  if (!requireNamespace("kernlab", quietly = TRUE)) stop("install.packages('kernlab')")
  half <- (r - 1) %/% 2
  l0   <- -half
  l1   <- r - 1 + l0
  bw   <- mmmd_expo_band(X, Y, l0, l1)
  switch(family,
    "GAUSS" = ,
    "GEXP"  = lapply(bw, function(s) kernlab::rbfdot(sigma = s)),
    "LAP"   = lapply(bw, function(s) kernlab::laplacedot(sigma = sqrt(s))),
    "MIXED" = c(lapply(bw, function(s) kernlab::rbfdot(sigma = s)),
                lapply(bw, function(s) kernlab::laplacedot(sigma = sqrt(s)))),
    stop("Unknown kernel family: ", family)
  )
}

## ---- core statistic ---------------------------------------------------------- #

#' Unbiased MMD_u^2 for a single kernel matrix-triple.
.mmd2_u <- function(KX, KY, KXY) {
  m <- nrow(KX); n <- nrow(KY)
  sx  <- (sum(KX)  - sum(diag(KX))) / (m * (m - 1))
  sy  <- (sum(KY)  - sum(diag(KY))) / (n * (n - 1))
  sxy <- (sum(KXY) - sum(diag(KXY))) / (m * (n - 1))
  sx + sy - 2 * sxy
}

#' Vector of MMD_u^2 across a list of kernels for the SAME (X, Y) pair.
mmmd_mmd_vector <- function(X, Y, kernels) {
  vapply(kernels, function(k) {
    KX  <- kernlab::kernelMatrix(k, X)
    KY  <- kernlab::kernelMatrix(k, Y)
    KXY <- kernlab::kernelMatrix(k, X, Y)
    .mmd2_u(KX, KY, KXY)
  }, numeric(1))
}

#' Centered kernel matrix list given combined data Z = rbind(X, Y).
.centered_K_list <- function(Z, kernels) {
  n <- nrow(Z)
  C <- diag(1, n) - matrix(1 / n, n, n)
  lapply(kernels, function(k) C %*% kernlab::kernelMatrix(k, Z) %*% C)
}

#' Estimated asymptotic Sigma_hat (r x r) under H_0 from data X.
mmmd_est_cov <- function(X, kernels) {
  n  <- nrow(X)
  C  <- diag(1, n) - matrix(1 / n, n, n)
  Kc <- lapply(kernels, function(k) C %*% kernlab::kernelMatrix(k, X) %*% C)
  r  <- length(Kc)
  S  <- matrix(0, r, r)
  for (i in seq_len(r)) for (j in seq_len(r)) {
    S[i, j] <- (8 / n^2) * sum(Kc[[i]] * t(Kc[[j]]))
  }
  ridge <- 1e-5 * min(diag(S))
  S + ridge * diag(1, r)
}

#' Inverse of Sigma_hat with SPD correction; uses Rfast::spdinv if available.
mmmd_inv_cov <- function(S) {
  if (requireNamespace("Rfast", quietly = TRUE)) Rfast::spdinv(S) else solve(S)
}

## ---- multiplier bootstrap (vectorised) -------------------------------------- #

#' One-shot multiplier bootstrap of the multi-kernel Mahalanobis statistic.
#'
#' Returns the vector of B null-statistic samples T*_b.  Vectorised: a single
#' set of Gaussian weights is drawn per replicate and re-used across kernels,
#' and the per-kernel quadratic form is computed in batch.
#'
#' @param X       reference data (used to build centred kernel matrices)
#' @param kernels list of kernlab kernels
#' @param invSig  pre-computed Sigma_hat^{-1}
#' @param B       number of bootstrap replicates
mmmd_bootstrap <- function(X, kernels, invSig, B = 1000) {
  n <- nrow(X)
  Kc <- lapply(kernels, function(k) {
    Cn <- diag(1, n) - matrix(1 / n, n, n)
    (1 / n) * (Cn %*% kernlab::kernelMatrix(k, X) %*% Cn)
  })

  ## shared random Gaussian weights across kernels: B x n
  U  <- matrix(stats::rnorm(B * n, sd = sqrt(2)), nrow = B, ncol = n)

  ## per-kernel approximating statistic for each replicate (B x r)
  approx_mat <- vapply(Kc, function(k.mat) {
    rowSums((U %*% k.mat) * U) - 2 * sum(diag(k.mat))
  }, numeric(B))
  if (is.null(dim(approx_mat))) approx_mat <- matrix(approx_mat, ncol = 1)

  ## Mahalanobis-aggregated null statistic per replicate
  rowSums((approx_mat %*% invSig) * approx_mat)
}

## ---- vectorised ROC threshold scan ------------------------------------------ #

#' Scan rejection over an alpha grid given pre-bootstrapped null and observed
#' statistics.  Returns one logical row per (statistic, alpha) cell.
#'
#' @param Tstar   numeric vector, length B, bootstrap null statistics
#' @param Tobs    numeric vector, length n_trials, observed test statistics
#' @param alphas  numeric vector of nominal levels in (0, 1)
#' @return  matrix n_trials x length(alphas) of TRUE = reject
mmmd_reject_grid <- function(Tstar, Tobs, alphas) {
  thresh <- stats::quantile(Tstar, probs = 1 - alphas, names = FALSE)
  outer(Tobs, thresh, ">")
}

## ---- end-to-end one-trial test ---------------------------------------------- #

#' Run a single MMMD test on the given (X, Y) and return the observed statistic
#' plus the bootstrap null sample.  Use this when you want to combine many
#' trials into a single ROC sweep.
#'
#' `kernel_builder` lets task 3 swap in `mmmd_make_kernels_custom_r` for the
#' r-sweep without disturbing the default paper-aligned config.
mmmd_run_test <- function(X, Y, family = "GEXP", r = 5, B = 1000,
                          kernel_builder = mmmd_make_kernels) {
  kernels <- kernel_builder(X, Y, family = family, r = r)
  invS    <- mmmd_inv_cov(mmmd_est_cov(X, kernels))
  Tstar   <- mmmd_bootstrap(X, kernels, invS, B = B)
  mmd_vec <- nrow(X) * mmmd_mmd_vector(X, Y, kernels)
  Tobs    <- as.numeric(t(mmd_vec) %*% invS %*% mmd_vec)
  list(Tobs = Tobs, Tstar = Tstar)
}

#' Reject decision at level alpha for one trial.  Convenience wrapper.
mmmd_test <- function(X, Y, family = "GEXP", r = 5, B = 1000, alpha = 0.05,
                      kernel_builder = mmmd_make_kernels) {
  res <- mmmd_run_test(X, Y, family = family, r = r, B = B,
                       kernel_builder = kernel_builder)
  threshold <- stats::quantile(res$Tstar, 1 - alpha)
  list(reject = res$Tobs > threshold,
       Tobs   = res$Tobs,
       Tstar  = res$Tstar,
       threshold = threshold)
}

## ---- single-kernel reference test (for plan1.md task 2) --------------------- #

#' Single-kernel MMD test using the median bandwidth.  Used as a baseline for
#' the multi-kernel Mahalanobis aggregation.
single_mmd_test <- function(X, Y, family = "GAUSS", B = 1000, alpha = 0.05) {
  bw  <- mmmd_med_bw(X, Y)
  ker <- switch(family,
    "GAUSS" = kernlab::rbfdot(sigma = bw),
    "LAP"   = kernlab::laplacedot(sigma = sqrt(bw)),
    stop("Unknown family: ", family)
  )
  KX  <- kernlab::kernelMatrix(ker, X)
  KY  <- kernlab::kernelMatrix(ker, Y)
  KXY <- kernlab::kernelMatrix(ker, X, Y)
  Tobs <- nrow(X) * .mmd2_u(KX, KY, KXY)

  n  <- nrow(X)
  Cn <- diag(1, n) - matrix(1 / n, n, n)
  Kc <- (1 / n) * (Cn %*% KX %*% Cn)
  U  <- matrix(stats::rnorm(B * n, sd = sqrt(2)), nrow = B, ncol = n)
  Tstar <- rowSums((U %*% Kc) * U) - 2 * sum(diag(Kc))

  list(reject = Tobs > stats::quantile(Tstar, 1 - alpha),
       Tobs = Tobs, Tstar = Tstar)
}
