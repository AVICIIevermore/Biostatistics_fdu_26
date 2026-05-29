## ============================================================================ #
## data_sources.R                                                                #
##   Pluggable two-sample data generators.                                       #
##                                                                               #
##   A "data source" is any function `function(n_x, n_y) -> list(X = ..., Y = ...)`
##   returning two numeric matrices.  Adding a new data source (e.g. real        #
##   biological data) means writing one such factory and registering it.        #
## ============================================================================ #

## ---- Covariance helpers ----------------------------------------------------- #

#' AR(1) / Toeplitz covariance: Sigma[i,j] = rho^|i-j|.
#'
#' This matches the covariance structure used in the paper's Mixture
#' Alternatives chapter (rho = 0.5 by default; Sigma1 = sigma.mult * Sigma0
#' is then the H1 covariance).
mmmd_ar1_cov <- function(d, rho = 0.5) {
  idx <- matrix(seq_len(d), d, d)
  rho ^ abs(idx - t(idx))
}

#' Sample n rows from N(mu, Sigma) using a Cholesky factor.
.rmvn <- function(n, mu, Sigma) {
  d <- length(mu)
  L <- chol(Sigma)                 # upper-tri, t(L) %*% L = Sigma
  Z <- matrix(rnorm(n * d), nrow = n, ncol = d)
  sweep(Z %*% L, 2, mu, "+")
}

#' Sample n rows from a multivariate t with covariance Sigma and df degrees
#' of freedom.  Uses the standard scale-mixture representation:
#'   X = mu + Z * sqrt(df / W),   Z ~ N(0, Sigma),  W ~ chi^2_df.
.rmvt <- function(n, mu, Sigma, df = 10) {
  d <- length(mu)
  Z <- .rmvn(n, rep(0, d), Sigma)
  g <- sqrt(df / rchisq(n, df))
  sweep(Z * g, 2, mu, "+")
}

## ---- Vector-space generators ------------------------------------------------ #

#' (1-eps) * N(0, I_d) + eps * t_10(0, I_d) mixture data source.
#'
#' Used by plan1.md task 1 to slide the contamination ratio epsilon.
#'
#' @param d         dimension of each observation
#' @param epsilon   contamination ratio, 0 <= epsilon <= 1
#' @param df        degrees of freedom of the contaminating t distribution
#' @param mu_y      optional mean shift applied to Y component (default 0)
#' @return  closure  function(n_x, n_y) -> list(X, Y)
ds_normal_t_mixture <- function(d, epsilon, df = 10, mu_y = 0) {
  force(d); force(epsilon); force(df); force(mu_y)
  function(n_x, n_y) {
    list(
      X = matrix(rnorm(n_x * d), nrow = n_x, ncol = d),
      Y = .mix_normal_t(n_y, d, epsilon, df, mu_y)
    )
  }
}

.mix_normal_t <- function(n, d, epsilon, df, mu_y) {
  if (epsilon <= 0)  return(matrix(rnorm(n * d), nrow = n, ncol = d) + mu_y)
  if (epsilon >= 1)  return(matrix(rt(n * d, df = df), nrow = n, ncol = d) + mu_y)
  pick <- rbinom(n, size = 1, prob = epsilon)
  out <- matrix(0, nrow = n, ncol = d)
  n1 <- sum(pick)
  if (n1 > 0)
    out[pick == 1, ] <- matrix(rt(n1 * d, df = df), nrow = n1, ncol = d)
  if (n1 < n)
    out[pick == 0, ] <- matrix(rnorm((n - n1) * d), nrow = n - n1, ncol = d)
  out + mu_y
}

#' Variance-scale alternative: P = N(0, I_d), Q = N(0, k * I_d).
#' Used by plan1.md task 2 to slide k.
ds_variance_scale <- function(d, k) {
  force(d); force(k)
  s <- sqrt(k)
  function(n_x, n_y) {
    list(
      X = matrix(rnorm(n_x * d), nrow = n_x, ncol = d),
      Y = matrix(rnorm(n_y * d) * s, nrow = n_y, ncol = d)
    )
  }
}

#' Identical generators for both samples (Type-I baseline).
ds_identical_normal <- function(d) {
  force(d)
  function(n_x, n_y) {
    list(
      X = matrix(rnorm(n_x * d), nrow = n_x, ncol = d),
      Y = matrix(rnorm(n_y * d), nrow = n_y, ncol = d)
    )
  }
}

#' Multivariate skew-normal alternative (task 4 - non-Gaussian / asymmetric).
#' Requires the `sn` package.
ds_skew_normal <- function(d, alpha = 5) {
  force(d); force(alpha)
  function(n_x, n_y) {
    if (!requireNamespace("sn", quietly = TRUE))
      stop("install.packages('sn')")
    Omega <- diag(1, d, d)
    list(
      X = matrix(rnorm(n_x * d), nrow = n_x, ncol = d),
      Y = sn::rmsn(n_y, xi = rep(0, d), Omega = Omega, alpha = rep(alpha, d))
    )
  }
}

#' Independent-coordinate skew-normal sampler without external dependencies.
#'
#' This uses the standard stochastic representation:
#'   Z = delta * |U| + sqrt(1 - delta^2) * V,
#' where U,V are independent standard normal variables and
#' delta = alpha / sqrt(1 + alpha^2).
.rskew_normal_matrix <- function(n, d, alpha = 0) {
  delta <- alpha / sqrt(1 + alpha ^ 2)
  U <- matrix(rnorm(n * d), nrow = n, ncol = d)
  V <- matrix(rnorm(n * d), nrow = n, ncol = d)
  delta * abs(U) + sqrt(1 - delta ^ 2) * V
}

#' Marginal standardized skewness of the univariate skew-normal distribution.
#'
#' The standardized skewness is E[(X - mu)^3] / sd(X)^3.
skew_normal_standard_skewness <- function(alpha) {
  delta <- alpha / sqrt(1 + alpha ^ 2)
  numerator <- ((4 - pi) / 2) * (delta * sqrt(2 / pi)) ^ 3
  denominator <- (1 - 2 * delta ^ 2 / pi) ^ (3 / 2)
  numerator / denominator
}

#' Identical skew-normal null for Type-I calibration under skewness.
#'
#' Both samples come from the same skew-normal distribution, so the target
#' rejection rate remains alpha even when the null distribution is asymmetric.
ds_identical_skew_normal <- function(d, alpha = 0) {
  force(d); force(alpha)
  function(n_x, n_y) {
    list(
      X = .rskew_normal_matrix(n_x, d, alpha = alpha),
      Y = .rskew_normal_matrix(n_y, d, alpha = alpha)
    )
  }
}

#' Multivariate Laplace alternative (task 4).
ds_mv_laplace <- function(d) {
  force(d)
  function(n_x, n_y) {
    if (!requireNamespace("LaplacesDemon", quietly = TRUE))
      stop("install.packages('LaplacesDemon')")
    list(
      X = matrix(rnorm(n_x * d), nrow = n_x, ncol = d),
      Y = LaplacesDemon::rmvl(n_y, mu = rep(0, d), Sigma = diag(1, d))
    )
  }
}

## ---- Paper-aligned generators with AR(1) covariance ------------------------- #

.mix_gauss_t <- function(n, mu, Sigma, p, df) {
  d <- length(mu)
  if (p <= 0)  return(.rmvn(n, mu, Sigma))
  if (p >= 1)  return(.rmvt(n, mu, Sigma, df))
  pick <- rbinom(n, size = 1, prob = p)        # 1 = t, 0 = Gaussian
  out  <- matrix(0, nrow = n, ncol = d)
  n1 <- sum(pick == 1)
  if (n1 > 0) out[pick == 1, ] <- .rmvt(n1, mu, Sigma, df)
  if (n1 < n) out[pick == 0, ] <- .rmvn(n - n1, mu, Sigma)
  out
}

#' Paper Fig.2 mixture alternative.
#'
#' For each sample, every observation is independently drawn from
#'   (1 - p) * N(mu_g, Sigma_g) + p * t_df(mu_g, Sigma_g)
#' (element-wise mixture).  P uses (mu0, Sigma0); Q uses (mu1, Sigma1) with
#' Sigma1 = sigma.mult * Sigma0.  By default Sigma0 is the AR(1) Toeplitz
#' matrix Sigma0[i,j] = rho^|i-j|.
ds_mixture_fig2 <- function(d, p, rho = 0.5, sigma.mult = 1.25,
                            mu0 = NULL, mu1 = NULL, df = 10) {
  force(d); force(p); force(rho); force(sigma.mult); force(df)
  if (is.null(mu0)) mu0 <- rep(0, d)
  if (is.null(mu1)) mu1 <- rep(0, d)
  Sigma0 <- mmmd_ar1_cov(d, rho)
  Sigma1 <- sigma.mult * Sigma0
  function(n_x, n_y) list(
    X = .mix_gauss_t(n_x, mu0, Sigma0, p, df),
    Y = .mix_gauss_t(n_y, mu1, Sigma1, p, df)
  )
}

#' Variance-scale alternative with AR(1) covariance:
#'   P = N(0, Sigma0),   Q = N(0, k * Sigma0).
#' Use this in place of `ds_variance_scale` when the paper's covariance
#' structure (rho = 0.5) is desired.
ds_variance_scale_ar1 <- function(d, k, rho = 0.5) {
  force(d); force(k); force(rho)
  Sigma0 <- mmmd_ar1_cov(d, rho)
  Sigma1 <- k * Sigma0
  function(n_x, n_y) list(
    X = .rmvn(n_x, rep(0, d), Sigma0),
    Y = .rmvn(n_y, rep(0, d), Sigma1)
  )
}

#' AR(1) identical generator for both samples (Type-I baseline matched to
#' the paper's covariance).
ds_identical_ar1 <- function(d, rho = 0.5) {
  force(d); force(rho)
  Sigma <- mmmd_ar1_cov(d, rho)
  function(n_x, n_y) list(
    X = .rmvn(n_x, rep(0, d), Sigma),
    Y = .rmvn(n_y, rep(0, d), Sigma)
  )
}

#' AR(1)-aware variant of `ds_normal_t_mixture` for task 1.  When rho = 0
#' this falls back to the original identity-covariance behaviour.
ds_normal_t_mixture_ar1 <- function(d, epsilon, rho = 0.5, df = 10,
                                    mu_y = 0) {
  force(d); force(epsilon); force(rho); force(df); force(mu_y)
  Sigma <- mmmd_ar1_cov(d, rho)
  mu_x  <- rep(0, d)
  mu_y_vec <- rep(mu_y, d)
  function(n_x, n_y) {
    list(
      X = .rmvn(n_x, mu_x, Sigma),
      Y = .mix_gauss_t(n_y, mu_y_vec, Sigma, epsilon, df)
    )
  }
}

## ---- Real-data hooks -------------------------------------------------------- #

#' Build a data source from a single pre-loaded matrix by sampling without
#' replacement into two halves.  Use this for Type-I-error baselines on real
#' data (e.g. all observations come from the same biological condition).
ds_resample_from_matrix <- function(M) {
  M <- as.matrix(M)
  function(n_x, n_y) {
    if (n_x + n_y > nrow(M))
      stop("Requested ", n_x + n_y, " rows but matrix has only ", nrow(M))
    idx <- sample.int(nrow(M), n_x + n_y, replace = FALSE)
    list(
      X = M[idx[1:n_x], , drop = FALSE],
      Y = M[idx[(n_x + 1):(n_x + n_y)], , drop = FALSE]
    )
  }
}

#' Build a data source from two pre-loaded matrices (different conditions).
#' Each call resamples without replacement from the corresponding matrix.
ds_resample_two_matrices <- function(MX, MY) {
  MX <- as.matrix(MX); MY <- as.matrix(MY)
  function(n_x, n_y) {
    if (n_x > nrow(MX)) stop("n_x > nrow(MX)")
    if (n_y > nrow(MY)) stop("n_y > nrow(MY)")
    list(
      X = MX[sample.int(nrow(MX), n_x), , drop = FALSE],
      Y = MY[sample.int(nrow(MY), n_y), , drop = FALSE]
    )
  }
}

## ---- Registry --------------------------------------------------------------- #

#' Internal: registry of named data-source factories.  Plug-in points for
#' future biological datasets register themselves here so experiment drivers
#' can be parameterised by a single string.
.MMMD_DATA_REGISTRY <- new.env(parent = emptyenv())

mmmd_register_data_source <- function(name, factory) {
  if (!is.function(factory))
    stop("factory must be a function returning function(n_x, n_y) -> list(X, Y)")
  assign(name, factory, envir = .MMMD_DATA_REGISTRY)
  invisible(NULL)
}

mmmd_data_source <- function(name, ...) {
  if (!exists(name, envir = .MMMD_DATA_REGISTRY, inherits = FALSE))
    stop("Unknown data source: ", name,
         " (registered: ",
         paste(ls(.MMMD_DATA_REGISTRY), collapse = ", "), ")")
  get(name, envir = .MMMD_DATA_REGISTRY)(...)
}

mmmd_register_data_source("normal_t_mixture", ds_normal_t_mixture)
mmmd_register_data_source("variance_scale",   ds_variance_scale)
mmmd_register_data_source("identical_normal", ds_identical_normal)
mmmd_register_data_source("skew_normal",      ds_skew_normal)
mmmd_register_data_source("identical_skew_normal", ds_identical_skew_normal)
mmmd_register_data_source("mv_laplace",       ds_mv_laplace)
mmmd_register_data_source("mixture_fig2",       ds_mixture_fig2)
mmmd_register_data_source("variance_scale_ar1", ds_variance_scale_ar1)
mmmd_register_data_source("identical_ar1",      ds_identical_ar1)
mmmd_register_data_source("normal_t_mixture_ar1", ds_normal_t_mixture_ar1)
