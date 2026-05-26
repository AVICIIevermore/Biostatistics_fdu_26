## ============================================================================ #
## roc_utils.R                                                                   #
##   Vectorised TPR / FPR aggregation across a batch of trials, satisfying       #
##   plan1.md section 4.2: "single bootstrap + vectorised threshold scan",       #
##   no re-bootstrapping per alpha level.                                        #
## ============================================================================ #

#' Aggregate (Tobs, Tstar) from many trials into a TPR/FPR sweep.
#'
#' @param trials      list of trial results, each `list(Tobs, Tstar)`.
#'                    `Tstar` is a length-B null sample, `Tobs` is the observed
#'                    statistic.  Trials may be from H0 (FPR contribution) or
#'                    H1 (TPR contribution); the `is_alt` mask indicates which.
#' @param is_alt      logical vector, length(trials).  TRUE = H1 trial.
#' @param alphas      numeric grid of nominal levels in (0, 1).
#' @return  data.frame with columns alpha, fpr, tpr.
mmmd_aggregate_roc <- function(trials, is_alt, alphas = seq(0, 1, by = 0.01)) {
  stopifnot(length(trials) == length(is_alt))
  reject_mat <- t(vapply(trials, function(tr) {
    thr <- stats::quantile(tr$Tstar, 1 - alphas, names = FALSE)
    as.numeric(tr$Tobs > thr)
  }, numeric(length(alphas))))

  fpr <- if (any(!is_alt)) colMeans(reject_mat[!is_alt, , drop = FALSE]) else rep(NA_real_, length(alphas))
  tpr <- if (any( is_alt)) colMeans(reject_mat[ is_alt, , drop = FALSE]) else rep(NA_real_, length(alphas))

  data.frame(alpha = alphas, fpr = fpr, tpr = tpr)
}
