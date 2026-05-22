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

Single.MMD <- function(data.X, data.Y, resamp.x, resamp.y, resamp.size,
                       kernel.choice, n.iter = 1000, alpha = 0.05){
  count <- 0

  for (i in seq_len(n.iter)) {
    X <- as.matrix(data.X[resamp.x[i, ],, drop = FALSE])
    Y <- as.matrix(data.Y[resamp.y[i, ],, drop = FALSE])
    sigma.med <- med.bandwidth(X, Y)

    if (kernel.choice == "GAUSS") {
      kernel <- kernlab::rbfdot(sigma = sigma.med)
    } else if (kernel.choice == "LAP") {
      kernel <- kernlab::laplacedot(sigma = sqrt(sigma.med))
    } else {
      stop("Unsupported single-kernel choice: ", kernel.choice)
    }

    MMD.threshold <- single.H0.cutoff(resamp.size, X, kernel, n.iter, alpha)
    MMD.val <- resamp.size * compute.MMD(X, Y, kernel)
    count <- count + (MMD.val > MMD.threshold)
  }

  count / n.iter
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

Multi.MMD <- function(data.X, data.Y, resamp.x, resamp.y, resamp.size,
                      kernel.choice, n.iter = 1000, alpha = 0.05){
  count <- 0

  for (i in seq_len(n.iter)) {
    X <- as.matrix(data.X[resamp.x[i, ],, drop = FALSE])
    Y <- as.matrix(data.Y[resamp.y[i, ],, drop = FALSE])
    kernel.vec <- k.choice(X, Y, kernel.choice)
    inv.cov.samp <- solve(est.cov(resamp.size, X, kernel.vec))
    MMD.func.threshold <- multi.H0.cutoff(resamp.size, X, kernel.vec, inv.cov.samp, n.iter, alpha)
    MMD.samp.val <- resamp.size * compute.MMD.vec(X, Y, kernel.vec)
    MMD.samp.func <- multi.func(MMD.samp.val, param = inv.cov.samp)
    count <- count + (MMD.samp.func > MMD.func.threshold)
  }

  count / n.iter
}

power.d <- function(train.x, train.label, resamp, error.sigma, kernel.choice,
                    n.iter = 500, set.x = c(1, 2, 3), set.y = c(1, 2, 8),
                    n.cores = 1, seed = NULL, alpha = 0.05){
  library(foreach)
  library(doParallel)

  n.cores <- max(1L, as.integer(n.cores))
  cl <- parallel::makeCluster(n.cores)
  doParallel::registerDoParallel(cl)
  on.exit(parallel::stopCluster(cl), add = TRUE)

  out.compare <- foreach::foreach(
    k = seq_along(error.sigma),
    .combine = rbind,
    .packages = c("kernlab", "MASS", "LaplacesDemon", "Rfast", "SpatialPack", "psych"),
    .export = c("select_digit_pools", "Single.MMD", "Multi.MMD", "single.H0.cutoff", "multi.H0.cutoff", "compute.MMD", "compute.MMD.vec", "est.cov", "med.bandwidth", "min.max.band", "expo.band", "multi.func", "multi.k.approx.stat", "k.choice")
  ) %dopar% {
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
    resamp.x <- matrix(sample.int(m.x, size = n.iter * resamp, replace = TRUE), nrow = n.iter)
    resamp.y <- matrix(sample.int(n.y, size = n.iter * resamp, replace = TRUE), nrow = n.iter)

    c(
      k,
      Single.MMD(data.X, data.Y, resamp.x, resamp.y, resamp, kernel.choice[1], n.iter, alpha),
      Single.MMD(data.X, data.Y, resamp.x, resamp.y, resamp, kernel.choice[2], n.iter, alpha),
      Multi.MMD(data.X, data.Y, resamp.x, resamp.y, resamp, kernel.choice[3], n.iter, alpha),
      Multi.MMD(data.X, data.Y, resamp.x, resamp.y, resamp, kernel.choice[4], n.iter, alpha),
      Multi.MMD(data.X, data.Y, resamp.x, resamp.y, resamp, kernel.choice[5], n.iter, alpha)
    )
  }

  out.compare <- as.data.frame(out.compare)
  colnames(out.compare) <- c(
    "Set Choice", "Single Kernel-1", "Single Kernel-2",
    "Multiple Kernel-1", "Multiple Kernel-2", "Multiple Kernel-3"
  )
  out.compare
}
