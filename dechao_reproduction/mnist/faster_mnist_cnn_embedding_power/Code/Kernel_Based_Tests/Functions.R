select_digit_pools <- function(test.x, test.label, set.x, set.y){
  list(
    data.X = test.x[test.label %in% set.x, , drop = FALSE],
    data.Y = test.x[test.label %in% set.y, , drop = FALSE]
  )
}

med.bandwidth <- function(X, Y){
  Z <- rbind(as.matrix(X), as.matrix(Y))
  1 / median(stats::dist(Z) ^ 2)
}

expo.band <- function(X, Y, l0 = -2L, l1 = 2L){
  med.band <- med.bandwidth(X, Y)
  (2 ^ seq.int(l0, l1)) * med.band
}

centering_matrix <- function(n){
  diag(1, nrow = n, ncol = n) - (1 / n) * matrix(1, nrow = n, ncol = n)
}

compute_bootstrap_u <- function(B_boot, n){
  MASS::mvrnorm(B_boot, mu = rep(0, n), Sigma = diag(2, nrow = n, ncol = n))
}

compute_kernel_mats <- function(X, Y, kernel.vec){
  list(
    Kxx = lapply(kernel.vec, function(k) kernlab::kernelMatrix(k, X)),
    Kyy = lapply(kernel.vec, function(k) kernlab::kernelMatrix(k, Y)),
    Kxy = lapply(kernel.vec, function(k) kernlab::kernelMatrix(k, X, Y))
  )
}

compute_scaled_mmd_vec_from_mats <- function(kxx.list, kyy.list, kxy.list, n){
  vapply(
    seq_along(kxx.list),
    function(i) n * mean(
      kxx.list[[i]][row(kxx.list[[i]]) != col(kxx.list[[i]])] +
        kyy.list[[i]][row(kyy.list[[i]]) != col(kyy.list[[i]])] -
        2 * kxy.list[[i]][row(kxy.list[[i]]) != col(kxy.list[[i]])]
    ),
    numeric(1)
  )
}

center_kxx_list <- function(kxx.list, C, scale = 1){
  lapply(kxx.list, function(kxx) scale * (C %*% kxx %*% C))
}

est.cov.raw <- function(n, centered.list){
  k.len <- length(centered.list)
  cov.mat.est <- matrix(0, nrow = k.len, ncol = k.len)

  for (i in seq_len(k.len)) {
    for (j in seq_len(k.len)) {
      cov.mat.est[i, j] <- (8 / (n ^ 2)) * psych::tr(centered.list[[i]] %*% centered.list[[j]])
    }
  }

  cov.mat.est
}

regularize_sigma <- function(sigma.hat, ridge_scale = 1e-4){
  diag_mean <- mean(diag(sigma.hat))
  if (!is.finite(diag_mean) || diag_mean <= 0) {
    diag_mean <- 1
  }
  lambda <- ridge_scale * diag_mean
  sigma.reg <- sigma.hat + lambda * diag(nrow(sigma.hat))
  list(sigma_reg = sigma.reg, lambda = lambda)
}

condition_number <- function(mat){
  values <- tryCatch(
    base::svd(mat, nu = 0, nv = 0)$d,
    error = function(e) rep(NA_real_, 2)
  )
  if (length(values) == 0 || any(!is.finite(values)) || min(values) <= 0) {
    return(NA_real_)
  }
  max(values) / min(values)
}

multi.cutoff.from.centered <- function(kboot.list, invcov, u.mat, alpha = 0.05){
  kernel_count <- length(kboot.list)
  boot_count <- nrow(u.mat)
  test.kernel.mat <- matrix(0, nrow = boot_count, ncol = kernel_count)

  for (i in seq_len(kernel_count)) {
    kboot <- kboot.list[[i]]
    ku <- kboot %*% t(u.mat)
    test.kernel.mat[, i] <- colSums(t(u.mat) * ku) - 2 * psych::tr(kboot)
  }

  test.stat <- rowSums((test.kernel.mat %*% invcov) * test.kernel.mat)
  stats::quantile(test.stat, probs = 1 - alpha, names = FALSE)
}

build_gaussian_kernels <- function(X, Y, mode = c("gaussian5", "single")){
  mode <- match.arg(mode)
  sigma.vec <- if (mode == "gaussian5") expo.band(X, Y, -2L, 2L) else med.bandwidth(X, Y)
  lapply(sigma.vec, kernlab::rbfdot)
}

aggregate_kernel_lists <- function(kernel_mats_by_rep){
  list(
    Kxx = unlist(lapply(kernel_mats_by_rep, `[[`, "Kxx"), recursive = FALSE),
    Kyy = unlist(lapply(kernel_mats_by_rep, `[[`, "Kyy"), recursive = FALSE),
    Kxy = unlist(lapply(kernel_mats_by_rep, `[[`, "Kxy"), recursive = FALSE)
  )
}

run_mmmd_from_kernel_mats <- function(kernel.mats, sample_size, B_boot, alpha = 0.05, u.mat = NULL, ridge_scale = 1e-4){
  C <- centering_matrix(sample_size)
  centered.kxx <- center_kxx_list(kernel.mats$Kxx, C)
  sigma.hat <- est.cov.raw(sample_size, centered.kxx)
  sigma.reg.out <- regularize_sigma(sigma.hat, ridge_scale = ridge_scale)
  sigma.reg <- sigma.reg.out$sigma_reg
  inv.cov <- solve(sigma.reg)
  kboot.list <- center_kxx_list(kernel.mats$Kxx, C, scale = 1 / sample_size)
  if (is.null(u.mat)) {
    u.mat <- compute_bootstrap_u(B_boot, sample_size)
  }
  mmd.vector <- compute_scaled_mmd_vec_from_mats(kernel.mats$Kxx, kernel.mats$Kyy, kernel.mats$Kxy, sample_size)
  cutoff <- multi.cutoff.from.centered(kboot.list, inv.cov, u.mat, alpha)
  stat <- as.numeric(t(mmd.vector) %*% inv.cov %*% mmd.vector)

  list(
    stat = stat,
    cutoff = unname(cutoff),
    reject = unname(stat > cutoff),
    mmd_vector = unname(mmd.vector),
    sigma_hat = sigma.hat,
    sigma_reg = sigma.reg,
    lambda = sigma.reg.out$lambda,
    cond_sigma_hat = condition_number(sigma.hat),
    cond_sigma_reg = condition_number(sigma.reg),
    kernel_count = length(kernel.mats$Kxx)
  )
}

run_gaussian5_mmmd <- function(X, Y, B_boot, alpha = 0.05, u.mat = NULL, ridge_scale = 1e-4){
  kernel.vec <- build_gaussian_kernels(X, Y, mode = "gaussian5")
  kernel.mats <- compute_kernel_mats(X, Y, kernel.vec)
  run_mmmd_from_kernel_mats(kernel.mats, sample_size = nrow(X), B_boot = B_boot, alpha = alpha, u.mat = u.mat, ridge_scale = ridge_scale)
}

run_multilayer_mmmd <- function(X.list, Y.list, mode = c("gaussian5", "single"), B_boot, alpha = 0.05, u.mat = NULL, ridge_scale = 1e-4){
  mode <- match.arg(mode)
  kernel.mats.by.rep <- lapply(seq_along(X.list), function(i) {
    kernels <- build_gaussian_kernels(X.list[[i]], Y.list[[i]], mode = mode)
    compute_kernel_mats(X.list[[i]], Y.list[[i]], kernels)
  })
  kernel.mats <- aggregate_kernel_lists(kernel.mats.by.rep)
  run_mmmd_from_kernel_mats(kernel.mats, sample_size = nrow(X.list[[1]]), B_boot = B_boot, alpha = alpha, u.mat = u.mat, ridge_scale = ridge_scale)
}

sample_without_replacement_matrix <- function(pool_size, draws, sample_size){
  t(replicate(draws, sample.int(pool_size, size = sample_size, replace = FALSE)))
}

extract_cnn_embeddings <- function(py_module, checkpoint_path, flat_images, batch_size){
  py_out <- py_module$extract_embeddings(
    checkpoint_path = checkpoint_path,
    flat_images = reticulate::r_to_py(unname(flat_images)),
    batch_size = as.integer(batch_size)
  )
  list(
    layer1 = py_to_r(py_out$layer1),
    layer2 = py_to_r(py_out$layer2),
    final = py_to_r(py_out$final)
  )
}

run_methods_on_resamples <- function(data.X, data.Y, embed.X, embed.Y, resamp.x, resamp.y, B_boot, alpha, noise_sigma, outer_iter, ridge_scale = 1e-4){
  method_names <- c(
    "raw_pixel_gaussian5",
    "final_embedding_gaussian5",
    "multilayer_single_gaussian",
    "multilayer_gaussian15"
  )

  reject_counts <- setNames(integer(length(method_names)), method_names)
  diag_rows <- vector("list", length = nrow(resamp.x) * length(method_names))
  row_idx <- 1L

  for (i in seq_len(nrow(resamp.x))) {
    idx.x <- resamp.x[i, ]
    idx.y <- resamp.y[i, ]
    u.mat <- compute_bootstrap_u(B_boot, ncol(resamp.x))

    raw.out <- run_gaussian5_mmmd(data.X[idx.x, , drop = FALSE], data.Y[idx.y, , drop = FALSE], B_boot = B_boot, alpha = alpha, u.mat = u.mat, ridge_scale = ridge_scale)
    final.out <- run_gaussian5_mmmd(embed.X$final[idx.x, , drop = FALSE], embed.Y$final[idx.y, , drop = FALSE], B_boot = B_boot, alpha = alpha, u.mat = u.mat, ridge_scale = ridge_scale)
    multi.single.out <- run_multilayer_mmmd(
      X.list = list(embed.X$layer1[idx.x, , drop = FALSE], embed.X$layer2[idx.x, , drop = FALSE], embed.X$final[idx.x, , drop = FALSE]),
      Y.list = list(embed.Y$layer1[idx.y, , drop = FALSE], embed.Y$layer2[idx.y, , drop = FALSE], embed.Y$final[idx.y, , drop = FALSE]),
      mode = "single",
      B_boot = B_boot,
      alpha = alpha,
      u.mat = u.mat,
      ridge_scale = ridge_scale
    )
    multi.gauss.out <- run_multilayer_mmmd(
      X.list = list(embed.X$layer1[idx.x, , drop = FALSE], embed.X$layer2[idx.x, , drop = FALSE], embed.X$final[idx.x, , drop = FALSE]),
      Y.list = list(embed.Y$layer1[idx.y, , drop = FALSE], embed.Y$layer2[idx.y, , drop = FALSE], embed.Y$final[idx.y, , drop = FALSE]),
      mode = "gaussian5",
      B_boot = B_boot,
      alpha = alpha,
      u.mat = u.mat,
      ridge_scale = ridge_scale
    )

    outputs <- list(
      raw_pixel_gaussian5 = raw.out,
      final_embedding_gaussian5 = final.out,
      multilayer_single_gaussian = multi.single.out,
      multilayer_gaussian15 = multi.gauss.out
    )

    for (method in method_names) {
      out <- outputs[[method]]
      reject_counts[[method]] <- reject_counts[[method]] + as.integer(out$reject)
      diag_rows[[row_idx]] <- data.frame(
        outer_iter = outer_iter,
        inner_iter = i,
        noise_sigma = noise_sigma,
        method = method,
        kernel_count = out$kernel_count,
        lambda = out$lambda,
        cond_sigma_hat = out$cond_sigma_hat,
        cond_sigma_reg = out$cond_sigma_reg,
        reject = as.integer(out$reject),
        stat = out$stat,
        cutoff = out$cutoff,
        stringsAsFactors = FALSE
      )
      row_idx <- row_idx + 1L
    }
  }

  power_rows <- data.frame(
    outer_iter = outer_iter,
    noise_sigma = noise_sigma,
    method = method_names,
    reject_rate = as.numeric(reject_counts[method_names]) / nrow(resamp.x),
    stringsAsFactors = FALSE
  )

  list(
    power_rows = power_rows,
    sigma_rows = do.call(rbind, diag_rows)
  )
}
