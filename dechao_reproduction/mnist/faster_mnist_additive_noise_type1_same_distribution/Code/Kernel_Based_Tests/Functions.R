compute.MMD <- function(X, Y, k){
  k.X <- kernlab::kernelMatrix(k, X)
  k.Y <- kernlab::kernelMatrix(k, Y)
  k.XY <- kernlab::kernelMatrix(k, X, Y)

  mean(k.X[row(k.X) != col(k.X)] + k.Y[row(k.Y) != col(k.Y)] -
         2 * k.XY[row(k.XY) != col(k.XY)])
}

compute.MMD.vec <- function(X, Y, kernel.vec){
  vapply(kernel.vec, function(k) compute.MMD(X, Y, k), numeric(1))
}

select_digit_pools <- function(train.x, train.label, set.x, set.y){
  list(
    data.X = train.x[train.label %in% set.x,, drop = FALSE],
    data.Y = train.x[train.label %in% set.y,, drop = FALSE]
  )
}

est.cov <- function(n, x, k.vec){
  k.len <- length(k.vec)
  C <- diag(1, nrow = n, ncol = n) - (1 / n) * matrix(1, nrow = n, ncol = n)

  kvec.mat <- vector("list", k.len)
  for (i in seq_len(k.len)) {
    kvec.mat[[i]] <- C %*% kernlab::kernelMatrix(k.vec[[i]], x) %*% C
  }

  cov.mat.est <- matrix(0, nrow = k.len, ncol = k.len)
  for (i in seq_len(k.len)) {
    for (j in seq_len(k.len)) {
      cov.mat.est[i, j] <- (8 / (n ^ 2)) * psych::tr(kvec.mat[[i]] %*% kvec.mat[[j]])
    }
  }

  cov.mat.est + (10 ^ -5) * min(diag(cov.mat.est)) * diag(1, k.len, k.len)
}

med.bandwidth <- function(X, Y){
  Z <- rbind(as.matrix(X), as.matrix(Y))
  1 / median(stats::dist(Z) ^ 2)
}

single.H0.cutoff <- function(m, x, k, n.iter = 1000, alpha = 0.05){
  C <- diag(1, nrow = m, ncol = m) - (1 / m) * matrix(1, nrow = m, ncol = m)
  k.mat <- (1 / m) * (C %*% kernlab::kernelMatrix(k, x) %*% C)
  u.mat <- MASS::mvrnorm(n.iter, mu = rep(0, m), Sigma = diag(2, nrow = m, ncol = m))
  test.stat <- colSums(t(u.mat) * (k.mat %*% t(u.mat))) - 2 * psych::tr(k.mat)
  stats::quantile(test.stat, probs = 1 - alpha, names = FALSE)
}

min.max.band <- function(X, Y){
  Z <- rbind(as.matrix(X), as.matrix(Y))
  quantiles.norm <- stats::quantile(stats::dist(Z) ^ 2, probs = c(0.95, 0.05))
  1 / quantiles.norm
}

expo.band <- function(X, Y, l0, l1){
  med.band <- med.bandwidth(X, Y)
  exponents <- seq.int(l0, l1)
  (2 ^ exponents) * med.band
}

multi.func <- function(x, param){
  as.numeric(t(x) %*% param %*% x)
}

multi.k.approx.stat <- function(k.mat, u.mat){
  colSums(t(u.mat) * (k.mat %*% t(u.mat))) - 2 * psych::tr(k.mat)
}

multi.H0.cutoff <- function(n, x, k.vec, invcov, n.iter = 1000, alpha = 0.05){
  k.len <- length(k.vec)
  C <- diag(1, nrow = n, ncol = n) - (1 / n) * matrix(1, nrow = n, ncol = n)

  kvec.mat <- vector("list", k.len)
  for (i in seq_len(k.len)) {
    kvec.mat[[i]] <- (1 / n) * (C %*% kernlab::kernelMatrix(k.vec[[i]], x) %*% C)
  }

  u.mat <- MASS::mvrnorm(n.iter, mu = rep(0, n), Sigma = diag(2, nrow = n, ncol = n))
  test.kernel.mat <- sapply(kvec.mat, multi.k.approx.stat, u.mat = u.mat)
  test.stat <- apply(test.kernel.mat, 1, multi.func, param = invcov)

  stats::quantile(test.stat, probs = 1 - alpha, names = FALSE)
}

k.choice <- function(X, Y, kernel.choice){
  if (kernel.choice == "MINMAX") {
    sigma.vec <- seq(min.max.band(X, Y)[1], min.max.band(X, Y)[2], length.out = 5)
    return(lapply(sigma.vec, function(s) kernlab::rbfdot(sigma = s)))
  }

  if (kernel.choice == "GEXP") {
    sigma.vec <- expo.band(X, Y, -2, 2)
    return(lapply(sigma.vec, function(s) kernlab::rbfdot(sigma = s)))
  }

  if (kernel.choice == "MIXED") {
    sigma.vec <- expo.band(X, Y, -1, 1)
    kernels <- lapply(sigma.vec, function(s) kernlab::rbfdot(sigma = s))
    kernels <- c(kernels, lapply(sigma.vec, function(s) kernlab::laplacedot(sigma = sqrt(s))))
    return(kernels)
  }

  if (kernel.choice == "LAP") {
    sigma.vec <- expo.band(X, Y, -2, 2)
    return(lapply(sigma.vec, function(s) kernlab::laplacedot(sigma = sqrt(s))))
  }

  stop("Unsupported multi-kernel choice: ", kernel.choice)
}

centering_matrix <- function(n){
  diag(1, nrow = n, ncol = n) - (1 / n) * matrix(1, nrow = n, ncol = n)
}

compute_kernel_mats <- function(X, Y, kernel.vec){
  list(
    Kxx = lapply(kernel.vec, function(k) kernlab::kernelMatrix(k, X)),
    Kyy = lapply(kernel.vec, function(k) kernlab::kernelMatrix(k, Y)),
    Kxy = lapply(kernel.vec, function(k) kernlab::kernelMatrix(k, X, Y))
  )
}

compute_bootstrap_u <- function(B_boot, n){
  MASS::mvrnorm(B_boot, mu = rep(0, n), Sigma = diag(2, nrow = n, ncol = n))
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

est.cov.from.centered <- function(n, centered.list){
  k.len <- length(centered.list)
  cov.mat.est <- matrix(0, nrow = k.len, ncol = k.len)

  for (i in seq_len(k.len)) {
    for (j in seq_len(k.len)) {
      cov.mat.est[i, j] <- (8 / (n ^ 2)) * psych::tr(centered.list[[i]] %*% centered.list[[j]])
    }
  }

  cov.mat.est + (10 ^ -5) * min(diag(cov.mat.est)) * diag(1, k.len, k.len)
}

single.cutoff.from.centered <- function(kboot, u.mat, alpha = 0.05){
  test.stat <- colSums(t(u.mat) * (kboot %*% t(u.mat))) - 2 * psych::tr(kboot)
  stats::quantile(test.stat, probs = 1 - alpha, names = FALSE)
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

progress_tick <- function(i, total, label = NULL, every = 50L){
  every <- max(1L, as.integer(every))
  if (i == 1L || i == total || (i %% every) == 0L) {
    prefix <- if (is.null(label) || !nzchar(label)) "" else paste0(label, " ")
    cat(sprintf("%srep %d/%d\n", prefix, i, total))
    flush.console()
  }
}

single_mmd_diagnostics <- function(X, Y, kernel.choice, B_boot = 1000, alpha = 0.05, u.mat = NULL){
  n <- nrow(X)
  sigma.med <- med.bandwidth(X, Y)

  if (kernel.choice == "GAUSS") {
    kernel <- kernlab::rbfdot(sigma = sigma.med)
  } else if (kernel.choice == "LAP") {
    kernel <- kernlab::laplacedot(sigma = sqrt(sigma.med))
  } else {
    stop("Unsupported single-kernel choice: ", kernel.choice)
  }

  kernel.mats <- compute_kernel_mats(X, Y, list(kernel))
  C <- centering_matrix(n)
  kboot <- center_kxx_list(kernel.mats$Kxx, C, scale = 1 / n)[[1]]
  if (is.null(u.mat)) {
    u.mat <- compute_bootstrap_u(B_boot, n)
  }
  mmd.value <- compute_scaled_mmd_vec_from_mats(kernel.mats$Kxx, kernel.mats$Kyy, kernel.mats$Kxy, n)[[1]]
  threshold <- single.cutoff.from.centered(kboot, u.mat, alpha)

  list(
    kernel = kernel,
    sigma.med = sigma.med,
    mmd.value = unname(mmd.value),
    cutoff = unname(threshold),
    reject = unname(mmd.value > threshold),
    u.mat = u.mat
  )
}

multi_mmd_diagnostics <- function(X, Y, kernel.choice, B_boot = 1000, alpha = 0.05, u.mat = NULL){
  n <- nrow(X)
  kernel.vec <- k.choice(X, Y, kernel.choice)
  kernel.mats <- compute_kernel_mats(X, Y, kernel.vec)
  C <- centering_matrix(n)
  centered.kxx <- center_kxx_list(kernel.mats$Kxx, C)
  sigma.hat <- est.cov.from.centered(n, centered.kxx)
  inv.cov <- solve(sigma.hat)
  kboot.list <- center_kxx_list(kernel.mats$Kxx, C, scale = 1 / n)
  if (is.null(u.mat)) {
    u.mat <- compute_bootstrap_u(B_boot, n)
  }
  mmd.vector <- compute_scaled_mmd_vec_from_mats(kernel.mats$Kxx, kernel.mats$Kyy, kernel.mats$Kxy, n)
  cutoff <- multi.cutoff.from.centered(kboot.list, inv.cov, u.mat, alpha)
  stat <- multi.func(mmd.vector, param = inv.cov)

  list(
    kernel.vec = kernel.vec,
    sigma.hat = sigma.hat,
    inv.cov = inv.cov,
    mmd.vector = unname(mmd.vector),
    cutoff = unname(cutoff),
    stat = unname(stat),
    reject = unname(stat > cutoff),
    u.mat = u.mat
  )
}

Single.MMD <- function(data.X, data.Y, resamp.x, resamp.y, resamp.size,
                       kernel.choice, n.iter = 1000, B_boot = n.iter,
                       alpha = 0.05, bootstrap_u_list = NULL,
                       progress_label = NULL, progress_every = 50L){
  count <- 0

  for (i in seq_len(n.iter)) {
    progress_tick(i, n.iter, label = progress_label, every = progress_every)
    X <- as.matrix(data.X[resamp.x[i, ],, drop = FALSE])
    Y <- as.matrix(data.Y[resamp.y[i, ],, drop = FALSE])
    u.mat <- if (is.null(bootstrap_u_list)) NULL else bootstrap_u_list[[i]]
    diag.out <- single_mmd_diagnostics(
      X = X,
      Y = Y,
      kernel.choice = kernel.choice,
      B_boot = B_boot,
      alpha = alpha,
      u.mat = u.mat
    )
    count <- count + diag.out$reject
  }

  count / n.iter
}

Multi.MMD <- function(data.X, data.Y, resamp.x, resamp.y, resamp.size,
                      kernel.choice, n.iter = 1000, B_boot = n.iter,
                      alpha = 0.05, bootstrap_u_list = NULL,
                      progress_label = NULL, progress_every = 50L){
  count <- 0

  for (i in seq_len(n.iter)) {
    progress_tick(i, n.iter, label = progress_label, every = progress_every)
    X <- as.matrix(data.X[resamp.x[i, ],, drop = FALSE])
    Y <- as.matrix(data.Y[resamp.y[i, ],, drop = FALSE])
    u.mat <- if (is.null(bootstrap_u_list)) NULL else bootstrap_u_list[[i]]
    diag.out <- multi_mmd_diagnostics(
      X = X,
      Y = Y,
      kernel.choice = kernel.choice,
      B_boot = B_boot,
      alpha = alpha,
      u.mat = u.mat
    )
    count <- count + diag.out$reject
  }

  count / n.iter
}

power.d <- function(train.x, train.label, resamp, error.sigma, kernel.choice,
                    n.iter = 500, set.x = c(1, 2, 3), set.y = c(1, 2, 8),
                    n.cores = 1, seed = NULL, alpha = 0.05,
                    progress_prefix = NULL, progress_every = 50L){
  library(foreach)
  library(doParallel)

  n.rep.inner <- n.iter
  B_boot <- n.iter
  n.cores <- max(1L, as.integer(n.cores))
  cl <- parallel::makeCluster(n.cores, outfile = "")
  doParallel::registerDoParallel(cl)
  on.exit(parallel::stopCluster(cl), add = TRUE)

  out.compare <- foreach::foreach(
    k = seq_along(error.sigma),
    .combine = rbind,
    .packages = c("kernlab", "MASS", "LaplacesDemon", "Rfast", "SpatialPack", "psych"),
    .export = c(
      "select_digit_pools", "Single.MMD", "Multi.MMD", "single.H0.cutoff", "multi.H0.cutoff",
      "compute.MMD", "compute.MMD.vec", "est.cov", "med.bandwidth", "min.max.band", "expo.band",
      "multi.func", "multi.k.approx.stat", "k.choice", "centering_matrix", "compute_kernel_mats",
      "compute_bootstrap_u", "compute_scaled_mmd_vec_from_mats", "center_kxx_list",
      "est.cov.from.centered", "single.cutoff.from.centered", "multi.cutoff.from.centered",
      "progress_tick", "single_mmd_diagnostics", "multi_mmd_diagnostics"
    )
  ) %dopar% {
    worker.label <- sprintf(
      "%ssigma[%d/%d]=%.3f",
      if (is.null(progress_prefix)) "" else paste0(progress_prefix, " "),
      k,
      length(error.sigma),
      error.sigma[k]
    )
    cat(sprintf("%s started\n", worker.label))
    flush.console()

    if (!is.null(seed)) {
      set.seed(seed + k - 1L)
    }

    selected <- select_digit_pools(train.x, train.label, set.x, set.y)
    data.X <- selected$data.X
    data.Y <- selected$data.Y

    data.X <- SpatialPack::imnoise(data.X, type = "gaussian", mean = 0, sd = error.sigma[k])
    data.Y <- SpatialPack::imnoise(data.Y, type = "gaussian", mean = 0, sd = error.sigma[k])

    m.x <- nrow(data.X)
    n.y <- nrow(data.Y)
    resamp.x <- matrix(sample.int(m.x, size = n.rep.inner * resamp, replace = TRUE), nrow = n.rep.inner)
    resamp.y <- matrix(sample.int(n.y, size = n.rep.inner * resamp, replace = TRUE), nrow = n.rep.inner)

    out <- c(
      k,
      Single.MMD(
        data.X, data.Y, resamp.x, resamp.y, resamp, kernel.choice[1],
        n.iter = n.rep.inner, B_boot = B_boot, alpha = alpha,
        progress_label = sprintf("%s %s", worker.label, kernel.choice[1]),
        progress_every = progress_every
      ),
      Single.MMD(
        data.X, data.Y, resamp.x, resamp.y, resamp, kernel.choice[2],
        n.iter = n.rep.inner, B_boot = B_boot, alpha = alpha,
        progress_label = sprintf("%s %s", worker.label, kernel.choice[2]),
        progress_every = progress_every
      ),
      Multi.MMD(
        data.X, data.Y, resamp.x, resamp.y, resamp, kernel.choice[3],
        n.iter = n.rep.inner, B_boot = B_boot, alpha = alpha,
        progress_label = sprintf("%s %s", worker.label, kernel.choice[3]),
        progress_every = progress_every
      ),
      Multi.MMD(
        data.X, data.Y, resamp.x, resamp.y, resamp, kernel.choice[4],
        n.iter = n.rep.inner, B_boot = B_boot, alpha = alpha,
        progress_label = sprintf("%s %s", worker.label, kernel.choice[4]),
        progress_every = progress_every
      ),
      Multi.MMD(
        data.X, data.Y, resamp.x, resamp.y, resamp, kernel.choice[5],
        n.iter = n.rep.inner, B_boot = B_boot, alpha = alpha,
        progress_label = sprintf("%s %s", worker.label, kernel.choice[5]),
        progress_every = progress_every
      )
    )

    cat(sprintf("%s finished\n", worker.label))
    flush.console()
    out
  }

  out.compare <- as.data.frame(out.compare)
  colnames(out.compare) <- c(
    "Set Choice", "Single Kernel-1", "Single Kernel-2",
    "Multiple Kernel-1", "Multiple Kernel-2", "Multiple Kernel-3"
  )
  out.compare
}
