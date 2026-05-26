## ============================================================================ #
## graph_kernel.R                                                                #
##                                                                               #
##  Graph two-sample test extension of MMMD (plan1.md task 4).                   #
##                                                                               #
##  Strategy: build a heat / diffusion kernel on a single shared graph that      #
##  joins the two samples (via a kNN similarity graph on stacked rows), then     #
##  treat the rows-of-the-resulting-kernel-matrix as the "kernel matrix" of a    #
##  graph Hilbert space.  Sliding the diffusion time t produces a kernel list    #
##  K_list = list(K_t1, ..., K_tr) which feeds directly into the existing        #
##  Mahalanobis covariance + multiplier bootstrap modules in mmmd_core.R.        #
##                                                                               #
##  The implementation uses base R + igraph (no GraphKernel dependency required).
## ============================================================================ #

#' Build a kNN similarity matrix from rows of Z.
#'
#' Returns a symmetric (n x n) adjacency where W[i, j] = 1 iff i is among the
#' k nearest neighbours of j or vice versa.
gk_knn_adjacency <- function(Z, k = 5) {
  n <- nrow(Z)
  D <- as.matrix(stats::dist(Z))
  W <- matrix(0, n, n)
  for (i in seq_len(n)) {
    nn <- order(D[i, ])[2:(k + 1)]   # exclude self
    W[i, nn] <- 1
  }
  pmax(W, t(W))
}

#' Heat / diffusion kernel  K_t = exp(-t * L)  where L is the unnormalised
#' graph Laplacian.  Computed via eigendecomposition once and reused across
#' the diffusion-time grid.
#'
#' @param W      adjacency matrix (n x n)
#' @param t_seq  vector of diffusion times
#' @return       list of K_t matrices, one per time
gk_heat_kernel_list <- function(W, t_seq) {
  n  <- nrow(W)
  D  <- diag(rowSums(W))
  L  <- D - W
  eL <- eigen(L, symmetric = TRUE)
  V  <- eL$vectors
  ev <- eL$values

  lapply(t_seq, function(t) {
    K <- V %*% diag(exp(-t * ev)) %*% t(V)
    (K + t(K)) / 2
  })
}

#' Centred kernel matrix from a precomputed K (used in place of
#' kernlab::kernelMatrix output).  Returns (1 / n) * C K C.
gk_center_scale <- function(K) {
  n <- nrow(K)
  C <- diag(1, n) - matrix(1 / n, n, n)
  (1 / n) * (C %*% K %*% C)
}

#' Estimated Sigma_hat for a list of precomputed (n x n) kernel matrices Klist
#' restricted to the rows/columns of X (the first m rows of the stacked data).
gk_est_cov <- function(Klist, m) {
  Cn <- diag(1, m) - matrix(1 / m, m, m)
  Kc <- lapply(Klist, function(K) Cn %*% K[1:m, 1:m, drop = FALSE] %*% Cn)
  r  <- length(Kc)
  S  <- matrix(0, r, r)
  for (i in seq_len(r)) for (j in seq_len(r)) {
    S[i, j] <- (8 / m^2) * sum(Kc[[i]] * t(Kc[[j]]))
  }
  ridge <- 1e-5 * min(diag(S))
  S + ridge * diag(1, r)
}

#' Unbiased MMD_u^2 for one precomputed kernel matrix on the full stacked data.
gk_mmd2_u <- function(K, m, n) {
  KX  <- K[1:m, 1:m, drop = FALSE]
  KY  <- K[(m + 1):(m + n), (m + 1):(m + n), drop = FALSE]
  KXY <- K[1:m, (m + 1):(m + n), drop = FALSE]
  sx  <- (sum(KX)  - sum(diag(KX)))  / (m * (m - 1))
  sy  <- (sum(KY)  - sum(diag(KY)))  / (n * (n - 1))
  sxy <- (sum(KXY) - sum(diag(KXY))) / (m * (n - 1))
  sx + sy - 2 * sxy
}

#' Multiplier bootstrap of the multi-graph-kernel statistic.
gk_bootstrap <- function(Klist, m, invSig, B = 1000) {
  Kc <- lapply(Klist, function(K) {
    Cn <- diag(1, m) - matrix(1 / m, m, m)
    (1 / m) * (Cn %*% K[1:m, 1:m, drop = FALSE] %*% Cn)
  })
  U <- matrix(stats::rnorm(B * m, sd = sqrt(2)), nrow = B, ncol = m)
  approx_mat <- vapply(Kc, function(k.mat) {
    rowSums((U %*% k.mat) * U) - 2 * sum(diag(k.mat))
  }, numeric(B))
  if (is.null(dim(approx_mat))) approx_mat <- matrix(approx_mat, ncol = 1)
  rowSums((approx_mat %*% invSig) * approx_mat)
}

#' End-to-end MMMD test on graph kernels.
#'
#' @param X,Y    data matrices (rows = observations).
#' @param k_nn   k for the kNN graph.
#' @param t_seq  diffusion-time grid (length r).
#' @param B      bootstrap replicates.
#' @param alpha  level.
#' @return list(reject, Tobs, Tstar, threshold)
graph_mmmd_test <- function(X, Y, k_nn = 5,
                            t_seq = c(0.1, 0.5, 1, 2, 5),
                            B = 1000, alpha = 0.05) {
  m <- nrow(X); n <- nrow(Y)
  Z <- rbind(X, Y)
  W <- gk_knn_adjacency(Z, k = k_nn)
  Klist <- gk_heat_kernel_list(W, t_seq)

  if (requireNamespace("Rfast", quietly = TRUE)) {
    invS <- Rfast::spdinv(gk_est_cov(Klist, m))
  } else {
    invS <- solve(gk_est_cov(Klist, m))
  }

  ## stacking K matrices into a 3-d array, as required by plan1.md task 4
  K_array <- array(unlist(Klist), dim = c(nrow(Z), nrow(Z), length(t_seq)))

  mmd_vec <- vapply(seq_along(Klist), function(j) gk_mmd2_u(Klist[[j]], m, n),
                    numeric(1)) * m
  Tobs  <- as.numeric(t(mmd_vec) %*% invS %*% mmd_vec)
  Tstar <- gk_bootstrap(Klist, m, invS, B = B)

  list(reject    = Tobs > stats::quantile(Tstar, 1 - alpha),
       Tobs      = Tobs,
       Tstar     = Tstar,
       threshold = stats::quantile(Tstar, 1 - alpha),
       K_array   = K_array,
       t_seq     = t_seq)
}
