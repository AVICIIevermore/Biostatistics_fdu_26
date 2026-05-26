## ============================================================================ #
## data_sources.R                                                                #
##   Pluggable two-sample data generators.                                       #
##                                                                               #
##   A "data source" is any function `function(n_x, n_y) -> list(X = ..., Y = ...)`
##   returning two numeric matrices.  Adding a new data source (e.g. real        #
##   biological data) means writing one such factory and registering it.        #
## ============================================================================ #

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
mmmd_register_data_source("mv_laplace",       ds_mv_laplace)
