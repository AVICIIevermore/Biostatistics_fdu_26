## Graph-MMMD utilities for the MNIST-CNN final-FC supplement.
##
## The functions here operate on an already exported embedding table.  They do
## not train a CNN, download data, or call the existing Gaussian5 pipeline.

graph_mmmd_as_matrix <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

graph_mmmd_knn_adjacency <- function(Z, k_nn = 5L) {
  Z <- graph_mmmd_as_matrix(Z)
  n <- nrow(Z)
  if (n < 2L) {
    stop("Need at least two observations to build a kNN graph.", call. = FALSE)
  }
  k_nn <- max(1L, min(as.integer(k_nn), n - 1L))
  D <- as.matrix(stats::dist(Z))
  W <- matrix(0, nrow = n, ncol = n)
  for (i in seq_len(n)) {
    nn <- order(D[i, ])[seq_len(k_nn) + 1L]
    W[i, nn] <- 1
  }
  W <- pmax(W, t(W))
  diag(W) <- 0
  W
}

graph_mmmd_heat_kernels <- function(W, t_seq = c(0.1, 0.5, 1, 2, 5)) {
  W <- graph_mmmd_as_matrix(W)
  D <- diag(rowSums(W), nrow = nrow(W), ncol = ncol(W))
  L <- D - W
  eig <- eigen(L, symmetric = TRUE)
  lapply(t_seq, function(t) {
    K <- eig$vectors %*% diag(exp(-as.numeric(t) * eig$values), nrow = nrow(W)) %*% t(eig$vectors)
    (K + t(K)) / 2
  })
}

graph_mmmd_mmd2_u <- function(K, m, n) {
  idx_x <- seq_len(m)
  idx_y <- m + seq_len(n)
  KX <- K[idx_x, idx_x, drop = FALSE]
  KY <- K[idx_y, idx_y, drop = FALSE]
  KXY <- K[idx_x, idx_y, drop = FALSE]
  sx <- (sum(KX) - sum(diag(KX))) / (m * (m - 1L))
  sy <- (sum(KY) - sum(diag(KY))) / (n * (n - 1L))
  sxy <- sum(KXY) / (m * n)
  sx + sy - 2 * sxy
}

graph_mmmd_est_cov <- function(K_list, m, ridge_scale = 1e-5) {
  C <- diag(1, nrow = m, ncol = m) - matrix(1 / m, nrow = m, ncol = m)
  Kc <- lapply(K_list, function(K) C %*% K[seq_len(m), seq_len(m), drop = FALSE] %*% C)
  r <- length(Kc)
  S <- matrix(0, nrow = r, ncol = r)
  for (i in seq_len(r)) {
    for (j in seq_len(r)) {
      S[i, j] <- (8 / (m ^ 2)) * sum(Kc[[i]] * t(Kc[[j]]))
    }
  }

  diag_ref <- min(diag(S))
  if (!is.finite(diag_ref) || diag_ref <= 0) {
    diag_ref <- mean(diag(S))
  }
  if (!is.finite(diag_ref) || diag_ref <= 0) {
    diag_ref <- 1
  }
  S + ridge_scale * diag_ref * diag(1, r)
}

graph_mmmd_bootstrap <- function(K_list, m, inv_cov, B = 200L) {
  C <- diag(1, nrow = m, ncol = m) - matrix(1 / m, nrow = m, ncol = m)
  Kc <- lapply(K_list, function(K) {
    (1 / m) * (C %*% K[seq_len(m), seq_len(m), drop = FALSE] %*% C)
  })
  U <- matrix(stats::rnorm(as.integer(B) * m, sd = sqrt(2)), nrow = as.integer(B), ncol = m)
  approx_mat <- vapply(Kc, function(Kmat) {
    rowSums((U %*% Kmat) * U) - 2 * sum(diag(Kmat))
  }, numeric(as.integer(B)))
  if (is.null(dim(approx_mat))) {
    approx_mat <- matrix(approx_mat, ncol = 1L)
  }
  rowSums((approx_mat %*% inv_cov) * approx_mat)
}

graph_mmmd_test <- function(X, Y, k_nn = 5L, t_seq = c(0.1, 0.5, 1, 2, 5),
                            B = 200L, alpha = 0.05, ridge_scale = 1e-5) {
  X <- graph_mmmd_as_matrix(X)
  Y <- graph_mmmd_as_matrix(Y)
  m <- nrow(X)
  n <- nrow(Y)
  if (m < 2L || n < 2L) {
    stop("Both samples need at least two observations.", call. = FALSE)
  }

  Z <- rbind(X, Y)
  W <- graph_mmmd_knn_adjacency(Z, k_nn = k_nn)
  K_list <- graph_mmmd_heat_kernels(W, t_seq = t_seq)
  cov_hat <- graph_mmmd_est_cov(K_list, m = m, ridge_scale = ridge_scale)
  inv_cov <- solve(cov_hat)

  mmd_vec <- m * vapply(K_list, graph_mmmd_mmd2_u, numeric(1), m = m, n = n)
  stat <- as.numeric(t(mmd_vec) %*% inv_cov %*% mmd_vec)
  boot <- graph_mmmd_bootstrap(K_list, m = m, inv_cov = inv_cov, B = B)
  threshold <- stats::quantile(boot, probs = 1 - alpha, names = FALSE, type = 8)

  list(
    reject = as.integer(stat > threshold),
    stat = stat,
    threshold = as.numeric(threshold),
    kernel_count = length(K_list),
    k_nn = as.integer(k_nn),
    t_seq = t_seq
  )
}

graph_mmmd_sample_balanced <- function(labels, target_labels, sample_size, replace = FALSE) {
  target_labels <- sort(unique(as.integer(target_labels)))
  if (sample_size %% length(target_labels) != 0L) {
    stop("sample_size must be divisible by the number of target labels.", call. = FALSE)
  }
  per_label <- sample_size %/% length(target_labels)
  idx <- integer(0)
  for (label in target_labels) {
    pool <- which(as.integer(labels) == label)
    if (!replace && length(pool) < per_label) {
      stop(sprintf("Label %s has %d rows, but %d are required.", label, length(pool), per_label), call. = FALSE)
    }
    idx <- c(idx, sample(pool, size = per_label, replace = replace))
  }
  sample(idx, size = length(idx), replace = FALSE)
}

graph_mmmd_sample_balanced_null_pair <- function(labels, target_labels, sample_size, replace = FALSE) {
  target_labels <- sort(unique(as.integer(target_labels)))
  if (sample_size %% length(target_labels) != 0L) {
    stop("sample_size must be divisible by the number of target labels.", call. = FALSE)
  }
  per_label <- sample_size %/% length(target_labels)
  x_idx <- integer(0)
  y_idx <- integer(0)
  for (label in target_labels) {
    pool <- which(as.integer(labels) == label)
    needed <- if (replace) per_label else 2L * per_label
    if (length(pool) < needed) {
      stop(sprintf("Label %s has %d rows, but %d are required for null sampling.", label, length(pool), needed), call. = FALSE)
    }
    if (replace) {
      x_idx <- c(x_idx, sample(pool, size = per_label, replace = TRUE))
      y_idx <- c(y_idx, sample(pool, size = per_label, replace = TRUE))
    } else {
      chosen <- sample(pool, size = 2L * per_label, replace = FALSE)
      x_idx <- c(x_idx, chosen[seq_len(per_label)])
      y_idx <- c(y_idx, chosen[per_label + seq_len(per_label)])
    }
  }
  list(
    x = sample(x_idx, size = length(x_idx), replace = FALSE),
    y = sample(y_idx, size = length(y_idx), replace = FALSE)
  )
}
