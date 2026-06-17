ensure_packages <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop("Missing required R packages: ", paste(missing, collapse = ", "),
         ". Install them before running the experiment.", call. = FALSE)
  }
}

ensure_packages(c("kernlab", "MASS", "psych"))

as_numeric_matrix <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

compute_mmd <- function(X, Y, kernel) {
  X <- as_numeric_matrix(X)
  Y <- as_numeric_matrix(Y)
  kx <- kernlab::kernelMatrix(kernel, X)
  ky <- kernlab::kernelMatrix(kernel, Y)
  kxy <- kernlab::kernelMatrix(kernel, X, Y)

  mean(kx[row(kx) != col(kx)]) + mean(ky[row(ky) != col(ky)]) - 2 * mean(kxy)
}

compute_mmd_vec <- function(X, Y, kernels) {
  vapply(kernels, function(kernel) compute_mmd(X, Y, kernel), numeric(1))
}

median_bandwidth <- function(X, Y) {
  Z <- rbind(as_numeric_matrix(X), as_numeric_matrix(Y))
  distances_sq <- stats::dist(Z) ^ 2
  med <- stats::median(distances_sq)
  if (!is.finite(med) || med <= 0) {
    stop("Median squared distance is not positive; check embeddings.", call. = FALSE)
  }
  1 / med
}

exponential_bandwidths <- function(X, Y, l0 = -2L, l1 = 2L) {
  med <- median_bandwidth(X, Y)
  (2 ^ seq.int(l0, l1)) * med
}

build_kernels <- function(X, Y, kernel_choice = c("GAUSS5", "LAP5", "MIXED", "GAUSS1", "LAP1")) {
  kernel_choice <- match.arg(kernel_choice)

  if (kernel_choice == "GAUSS1") {
    return(list(kernlab::rbfdot(sigma = median_bandwidth(X, Y))))
  }
  if (kernel_choice == "LAP1") {
    return(list(kernlab::laplacedot(sigma = sqrt(median_bandwidth(X, Y)))))
  }
  if (kernel_choice == "GAUSS5") {
    return(lapply(exponential_bandwidths(X, Y, -2L, 2L), kernlab::rbfdot))
  }
  if (kernel_choice == "LAP5") {
    return(lapply(exponential_bandwidths(X, Y, -2L, 2L), function(s) kernlab::laplacedot(sigma = sqrt(s))))
  }

  sigma_vec <- exponential_bandwidths(X, Y, -1L, 1L)
  c(
    lapply(sigma_vec, kernlab::rbfdot),
    lapply(sigma_vec, function(s) kernlab::laplacedot(sigma = sqrt(s)))
  )
}

centering_matrix <- function(n) {
  diag(1, nrow = n, ncol = n) - matrix(1 / n, nrow = n, ncol = n)
}

estimate_null_covariance <- function(X, kernels, ridge_scale = 1e-5) {
  X <- as_numeric_matrix(X)
  n <- nrow(X)
  kernel_count <- length(kernels)
  C <- centering_matrix(n)
  centered <- lapply(kernels, function(kernel) C %*% kernlab::kernelMatrix(kernel, X) %*% C)

  sigma_hat <- matrix(0, nrow = kernel_count, ncol = kernel_count)
  for (i in seq_len(kernel_count)) {
    for (j in seq_len(kernel_count)) {
      sigma_hat[i, j] <- (8 / (n ^ 2)) * psych::tr(centered[[i]] %*% centered[[j]])
    }
  }

  diag_ref <- min(diag(sigma_hat))
  if (!is.finite(diag_ref) || diag_ref <= 0) {
    diag_ref <- mean(diag(sigma_hat))
  }
  if (!is.finite(diag_ref) || diag_ref <= 0) {
    diag_ref <- 1
  }
  ridge <- ridge_scale * diag_ref
  sigma_reg <- sigma_hat + ridge * diag(kernel_count)

  list(sigma_hat = sigma_hat, sigma_reg = sigma_reg, ridge = ridge)
}

single_bootstrap_cutoff <- function(X, kernel, B = 500L, alpha = 0.05) {
  X <- as_numeric_matrix(X)
  n <- nrow(X)
  C <- centering_matrix(n)
  k_mat <- (1 / n) * (C %*% kernlab::kernelMatrix(kernel, X) %*% C)
  u_mat <- MASS::mvrnorm(B, mu = rep(0, n), Sigma = diag(2, nrow = n, ncol = n))
  boot_stat <- colSums(t(u_mat) * (k_mat %*% t(u_mat))) - 2 * psych::tr(k_mat)
  stats::quantile(boot_stat, probs = 1 - alpha, names = FALSE, type = 8)
}

multi_bootstrap_cutoff <- function(X, kernels, inv_cov, B = 500L, alpha = 0.05) {
  X <- as_numeric_matrix(X)
  n <- nrow(X)
  kernel_count <- length(kernels)
  C <- centering_matrix(n)
  boot_kernels <- lapply(kernels, function(kernel) {
    (1 / n) * (C %*% kernlab::kernelMatrix(kernel, X) %*% C)
  })
  u_mat <- MASS::mvrnorm(B, mu = rep(0, n), Sigma = diag(2, nrow = n, ncol = n))
  kernel_stats <- matrix(0, nrow = B, ncol = kernel_count)

  for (i in seq_len(kernel_count)) {
    ku <- boot_kernels[[i]] %*% t(u_mat)
    kernel_stats[, i] <- colSums(t(u_mat) * ku) - 2 * psych::tr(boot_kernels[[i]])
  }

  boot_stat <- rowSums((kernel_stats %*% inv_cov) * kernel_stats)
  stats::quantile(boot_stat, probs = 1 - alpha, names = FALSE, type = 8)
}

run_single_mmd_test <- function(X, Y, kernel_choice = c("GAUSS1", "LAP1"), B = 500L, alpha = 0.05) {
  kernel_choice <- match.arg(kernel_choice)
  kernels <- build_kernels(X, Y, kernel_choice)
  kernel <- kernels[[1]]
  n <- nrow(X)
  stat <- n * compute_mmd(X, Y, kernel)
  cutoff <- single_bootstrap_cutoff(X, kernel, B = B, alpha = alpha)
  list(stat = as.numeric(stat), cutoff = as.numeric(cutoff), reject = as.integer(stat > cutoff))
}

run_mmmd_test <- function(X, Y, kernel_choice = c("GAUSS5", "LAP5", "MIXED"),
                          B = 500L, alpha = 0.05, ridge_scale = 1e-5) {
  kernel_choice <- match.arg(kernel_choice)
  X <- as_numeric_matrix(X)
  Y <- as_numeric_matrix(Y)
  if (nrow(X) != nrow(Y)) {
    stop("This implementation expects equal sample sizes for X and Y.", call. = FALSE)
  }

  kernels <- build_kernels(X, Y, kernel_choice)
  cov_out <- estimate_null_covariance(X, kernels, ridge_scale = ridge_scale)
  inv_cov <- solve(cov_out$sigma_reg)
  n <- nrow(X)
  mmd_vec <- n * compute_mmd_vec(X, Y, kernels)
  stat <- as.numeric(t(mmd_vec) %*% inv_cov %*% mmd_vec)
  cutoff <- multi_bootstrap_cutoff(X, kernels, inv_cov, B = B, alpha = alpha)

  list(
    stat = stat,
    cutoff = as.numeric(cutoff),
    reject = as.integer(stat > cutoff),
    mmd_vector = as.numeric(mmd_vec),
    kernel_count = length(kernels),
    ridge = cov_out$ridge,
    cond_sigma = kappa(cov_out$sigma_reg)
  )
}

run_test_by_method <- function(X, Y, method, B = 500L, alpha = 0.05, ridge_scale = 1e-5) {
  if (method %in% c("GAUSS1", "LAP1")) {
    return(run_single_mmd_test(X, Y, kernel_choice = method, B = B, alpha = alpha))
  }
  run_mmmd_test(X, Y, kernel_choice = method, B = B, alpha = alpha, ridge_scale = ridge_scale)
}
