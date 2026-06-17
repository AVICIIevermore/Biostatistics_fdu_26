################################################################################
# Functions.R for Reproduction of Figure 1(b) with the new method
#   - GAUSS single-kernel MMD                      (baseline, from paper)
#   - GEXP MMMD (5 Gaussian kernels, geometric)    (Chatterjee & Bhattacharya 2025)
#   - NEW MMMD (13 Gaussian kernels, arithmetic grid in t,
#               fast covariance via multiplicative closure
#               + pivoted Cholesky kernel pruning)  (Covariance_matrix.md)
################################################################################

################################################################################
# 1. Sample generators (Gaussian-only since Figure 1 uses p = 0)
################################################################################

# mvrnorm-compatible wrapper using LaplacesDemon::rmvn, which is what's installed
# in the conda env. (mmmd_boost has LaplacesDemon but not MASS.)
mvrnorm.compat <- function(n, mu, Sigma) {
  LaplacesDemon::rmvn(n = n, mu = mu, Sigma = Sigma)
}

X.gen <- function(n, dim, p, gen.var) {
  mu0 <- gen.var[[1]]; mu1 <- gen.var[[2]]
  Sigma0 <- gen.var[[3]]; Sigma1 <- gen.var[[4]]
  if (p == 0) {
    mvrnorm.compat(n, mu = mu0, Sigma = Sigma0)
  } else if (p == 1) {
    LaplacesDemon::rmvt(n, mu = mu0, S = Sigma0, df = 10)
  } else {
    cv <- rbinom(n, 1, p)
    g <- mvrnorm.compat(n, mu = mu0, Sigma = Sigma0)
    o <- LaplacesDemon::rmvt(n, mu = mu0, S = Sigma0, df = 10)
    (1 - cv) * g + cv * o
  }
}

Y.gen <- function(n, dim, p, gen.var) {
  mu0 <- gen.var[[1]]; mu1 <- gen.var[[2]]
  Sigma0 <- gen.var[[3]]; Sigma1 <- gen.var[[4]]
  if (p == 0) {
    mvrnorm.compat(n, mu = mu1, Sigma = Sigma1)
  } else if (p == 1) {
    LaplacesDemon::rmvt(n, mu = mu1, S = Sigma1, df = 10)
  } else {
    cv <- rbinom(n, 1, p)
    g <- mvrnorm.compat(n, mu = mu1, Sigma = Sigma1)
    o <- LaplacesDemon::rmvt(n, mu = mu1, S = Sigma1, df = 10)
    (1 - cv) * g + cv * o
  }
}

################################################################################
# 2. Median bandwidth (squared) and MMD U-statistic
################################################################################

med.bandwidth <- function(X, Y) {
  X <- as.matrix(X); Y <- as.matrix(Y)
  Z <- rbind(X, Y)
  1 / median(dist(Z)^2)            # 1/median squared distance, matches kernlab rbfdot sigma
}

compute.MMD <- function(X, Y, k) {
  k.X  <- kernlab::kernelMatrix(k, X)
  k.Y  <- kernlab::kernelMatrix(k, Y)
  k.XY <- kernlab::kernelMatrix(k, X, Y)
  mean(k.X[row(k.X) != col(k.X)] +
       k.Y[row(k.Y) != col(k.Y)] -
       2 * k.XY[row(k.XY) != col(k.XY)])
}

compute.MMD.vec <- function(X, Y, kernel.vec) {
  vapply(kernel.vec, function(k) compute.MMD(X, Y, k), numeric(1))
}

################################################################################
# 3. Multiplier-bootstrap building blocks
################################################################################

multi.func <- function(x, param) drop(t(x) %*% param %*% x)

multi.k.approx.stat <- function(k.mat, u.mat) {
  colSums(t(u.mat) * (k.mat %*% t(u.mat))) - 2 * psych::tr(k.mat)
}

################################################################################
# 4. SINGLE-KERNEL test (GAUSS baseline in the figure)
################################################################################

single.H0.cutoff <- function(n, x, k, n.iter = 1000) {
  C     <- diag(1, n) - (1 / n) * matrix(1, n, n)
  k.mat <- (1 / n) * (C %*% kernlab::kernelMatrix(k, x) %*% C)
  u.mat <- mvrnorm.compat(n.iter, mu = rep(0, n), Sigma = diag(2, n))
  ts    <- colSums(t(u.mat) * (k.mat %*% t(u.mat))) - 2 * psych::tr(k.mat)
  quantile(ts, probs = 0.95)
}

Single.MMD <- function(n, d, gen.var, p, kernel.choice, n.iter = 1000) {
  X.list <- lapply(seq_len(n.iter), function(.) X.gen(n, d, p, gen.var))
  Y.list <- lapply(seq_len(n.iter), function(.) Y.gen(n, d, p, gen.var))

  rej <- foreach(k = seq_len(n.iter), .combine = "c",
                 .export   = ls(envir = globalenv()),
                 .packages = c("LaplacesDemon", "kernlab", "psych")) %dopar% {
    X <- X.list[[k]]; Y <- Y.list[[k]]
    smed <- med.bandwidth(X, Y)
    kn <- if (kernel.choice == "GAUSS")
            kernlab::rbfdot(sigma = smed)
          else
            kernlab::laplacedot(sigma = sqrt(smed))
    thr <- single.H0.cutoff(n, X, kn, n.iter)
    n * compute.MMD(X, Y, kn) > thr
  }
  mean(rej)
}

################################################################################
# 5. ORIGINAL GEXP MMMD test (5 Gaussian kernels, geometric grid)
#    - Builds Sigma via the slow naive est.cov below
################################################################################

expo.band <- function(X, Y, l0, l1) {
  m <- med.bandwidth(X, Y)
  vapply(l0:l1, function(i) (2^i) * m, numeric(1))
}

k.choice.GEXP <- function(X, Y) {
  s <- expo.band(X, Y, -2, 2)              # 5 bandwidths: 2^{-2..2} * t_med
  lapply(s, function(si) kernlab::rbfdot(sigma = si))
}

# Naive O(R^2 m^3) covariance, kept here for direct comparison with the new method.
est.cov <- function(n, x, k.vec) {
  R <- length(k.vec)
  C <- diag(1, n) - (1 / n) * matrix(1, n, n)
  Kc <- lapply(k.vec, function(k) C %*% kernlab::kernelMatrix(k, x) %*% C)
  S  <- matrix(0, R, R)
  for (i in seq_len(R)) for (j in seq_len(R))
    S[i, j] <- (8 / n^2) * psych::tr(Kc[[i]] %*% Kc[[j]])
  S + (1e-5) * min(diag(S)) * diag(R)
}

multi.H0.cutoff <- function(n, x, k.vec, Sigma, n.iter = 1000) {
  R <- length(k.vec)
  C <- diag(1, n) - (1 / n) * matrix(1, n, n)
  Kc <- lapply(k.vec, function(k) (1 / n) * (C %*% kernlab::kernelMatrix(k, x) %*% C))
  invS <- Rfast::spdinv(Sigma)
  u.mat <- mvrnorm.compat(n.iter, mu = rep(0, n), Sigma = diag(2, n))
  T.mat <- sapply(Kc, multi.k.approx.stat, u.mat = u.mat)
  ts    <- apply(T.mat, 1, multi.func, param = invS)
  quantile(ts, probs = 0.95)
}

GEXP.Multi.MMD <- function(n, d, gen.var, p, n.iter = 1000) {
  X.list <- lapply(seq_len(n.iter), function(.) X.gen(n, d, p, gen.var))
  Y.list <- lapply(seq_len(n.iter), function(.) Y.gen(n, d, p, gen.var))

  rej <- foreach(i = seq_len(n.iter), .inorder = TRUE, .combine = "c",
                 .export   = ls(envir = globalenv()),
                 .packages = c("LaplacesDemon", "kernlab", "Rfast", "psych")) %dopar% {
    X <- X.list[[i]]; Y <- Y.list[[i]]
    kv  <- k.choice.GEXP(X, Y)
    Sg  <- est.cov(n, X, kv)
    thr <- multi.H0.cutoff(n, X, kv, Sg, n.iter)
    Tn  <- n * compute.MMD.vec(X, Y, kv)
    multi.func(Tn, Rfast::spdinv(Sg)) > thr
  }
  mean(rej)
}

################################################################################
# 6. NEW METHOD (Covariance_matrix.md)
#
#   (a) fast.cov.gaussian.t :
#         <C K_a C, C K_b C>_F = H(t_a + t_b) - (2/m) * s_a^T s_b
#                              + (1/m^2) * S_a * S_b
#       built on a single squared-distance matrix D.
#       With an arithmetic t-grid the unique t_a + t_b values number 2R-1
#       instead of R^2, giving an O(R) speed-up of the H(u) phase.
#
#   (b) pivoted.chol.select :
#         pivoted-Cholesky pruning of redundant kernels.
#         The relative residual r_j / Sigma_jj is exactly the Schur complement
#         of the already-selected block, so kernels whose centered Gram matrix
#         is nearly explained by the already-selected ones are dropped.
#
#   (c) New.Multi.MMD :
#         (a) builds Sigma over R = 13 candidate kernels,
#         (b) prunes them to a numerically independent subset of size q,
#         (c) runs the same multiplier bootstrap as GEXP, but on the q
#             selected kernels with the corresponding sub-Sigma.
################################################################################

# (a) fast covariance via multiplicative closure of the Gaussian kernel family
fast.cov.gaussian.t <- function(X, t.vec) {
  m <- nrow(X)
  R <- length(t.vec)

  D <- as.matrix(dist(X))^2          # squared Euclidean distance matrix, m x m

  s.mat <- matrix(0, m, R)           # column a: row-sum vector s_a
  S.vec <- numeric(R)                # entry  a: total sum     S_a
  for (a in seq_len(R)) {
    Ka <- exp(-t.vec[a] * D)
    s.mat[, a] <- rowSums(Ka)
    S.vec[a]   <- sum(Ka)
  }

  # H(u) = sum_{i,j} exp(-u * D_ij), one evaluation per *unique* u = t_a + t_b
  sum.flat   <- as.vector(outer(t.vec, t.vec, "+"))
  sum.round  <- round(sum.flat, 10)
  uniq.sums  <- unique(sum.round)
  H.unique   <- vapply(uniq.sums, function(u) sum(exp(-u * D)), numeric(1))
  H.mat      <- matrix(H.unique[match(sum.round, uniq.sums)], R, R)

  G.mat   <- crossprod(s.mat)        # G[a, b] = s_a^T s_b, R x R
  inner   <- H.mat - (2 / m) * G.mat + (1 / m^2) * tcrossprod(S.vec)

  Sigma   <- (8 / m^2) * inner
  Sigma   <- (Sigma + t(Sigma)) / 2  # symmetrise away the rounding asymmetry
  Sigma + (1e-5) * min(diag(Sigma)) * diag(R)
}

# (b) pivoted Cholesky kernel-subset selection
pivoted.chol.select <- function(Sigma, eps = 1e-4, q.max = NULL) {
  R <- nrow(Sigma)
  if (is.null(q.max)) q.max <- R

  diag.S    <- diag(Sigma)
  resid     <- diag.S
  L         <- matrix(0, R, q.max)
  selected  <- integer(0)
  threshold <- eps * max(diag.S)

  for (t in seq_len(q.max)) {
    cand <- setdiff(seq_len(R), selected)
    if (length(cand) == 0) break
    pivot <- cand[which.max(resid[cand])]
    if (resid[pivot] <= threshold) break

    selected <- c(selected, pivot)
    if (t == 1) {
      L[, t] <- Sigma[, pivot] / sqrt(resid[pivot])
    } else {
      L[, t] <- (Sigma[, pivot] -
                   L[, seq_len(t - 1), drop = FALSE] %*%
                   L[pivot, seq_len(t - 1)]) / sqrt(resid[pivot])
    }
    resid <- pmax(resid - L[, t]^2, 0)
  }

  q <- length(selected)
  list(selected = selected, L = L[, seq_len(q), drop = FALSE], q = q,
       resid = resid)
}

# Build the candidate t-grid: arithmetic, centered on t_med, range [t_med/4, 4*t_med]
new.t.grid <- function(X, Y, R = 13) {
  tm <- med.bandwidth(X, Y)
  seq(0.25 * tm, 4.0 * tm, length.out = R)
}

# Bootstrap cutoff for the new method, given a precomputed sub-Sigma
new.H0.cutoff <- function(n, X, k.vec.sel, Sigma.SS, n.iter = 1000) {
  q  <- length(k.vec.sel)
  C  <- diag(1, n) - (1 / n) * matrix(1, n, n)
  Kc <- lapply(k.vec.sel, function(k) (1 / n) * (C %*% kernlab::kernelMatrix(k, X) %*% C))
  invS  <- Rfast::spdinv(Sigma.SS)
  u.mat <- mvrnorm.compat(n.iter, mu = rep(0, n), Sigma = diag(2, n))
  T.mat <- sapply(Kc, multi.k.approx.stat, u.mat = u.mat)
  ts    <- apply(T.mat, 1, multi.func, param = invS)
  quantile(ts, probs = 0.95)
}

# (c) full new test: fast cov  -->  pivoted-Cholesky pruning  -->  bootstrap
New.Multi.MMD <- function(n, d, gen.var, p, n.iter = 1000,
                          R = 13, eps = 1e-4) {
  X.list <- lapply(seq_len(n.iter), function(.) X.gen(n, d, p, gen.var))
  Y.list <- lapply(seq_len(n.iter), function(.) Y.gen(n, d, p, gen.var))

  rej <- foreach(i = seq_len(n.iter), .inorder = TRUE, .combine = "c",
                 .export   = ls(envir = globalenv()),
                 .packages = c("LaplacesDemon", "kernlab", "Rfast", "psych")) %dopar% {
    X <- X.list[[i]]; Y <- Y.list[[i]]

    t.vec      <- new.t.grid(X, Y, R = R)
    Sigma.full <- fast.cov.gaussian.t(X, t.vec)

    sel        <- pivoted.chol.select(Sigma.full, eps = eps)
    idx        <- sel$selected
    Sigma.SS   <- Sigma.full[idx, idx, drop = FALSE]
    k.vec.sel  <- lapply(t.vec[idx], function(t) kernlab::rbfdot(sigma = t))

    thr <- new.H0.cutoff(n, X, k.vec.sel, Sigma.SS, n.iter)
    Tn  <- n * compute.MMD.vec(X, Y, k.vec.sel)
    multi.func(Tn, Rfast::spdinv(Sigma.SS)) > thr
  }
  mean(rej)
}

################################################################################
# 7. Driver: power across n.seq for the three methods on the figure
################################################################################

power.d <- function(n.seq, sigma.param, sigma.mult, mu.param, d, p,
                    n.iter = 500, R.new = 13, eps.new = 1e-4) {
  out <- c()
  for (k in seq_along(n.seq)) {
    library(LaplacesDemon)
    library(Rfast)

    Sigma0  <- diag(sigma.param, d, d)
    Sigma1  <- sigma.mult * Sigma0
    mu0     <- rep(0, d); mu1 <- rep(mu.param, d)
    gen.var <- list(mu0, mu1, Sigma0, Sigma1)

    cat(sprintf("[%s] n = %d ...\n", format(Sys.time(), "%H:%M:%S"), n.seq[k]))

    pg <- Single.MMD     (n.seq[k], d, gen.var, p, "GAUSS", n.iter)
    cat(sprintf("    GAUSS single  = %.3f\n", pg))

    pe <- GEXP.Multi.MMD (n.seq[k], d, gen.var, p, n.iter)
    cat(sprintf("    GEXP   MMMD   = %.3f\n", pe))

    pn <- New.Multi.MMD  (n.seq[k], d, gen.var, p, n.iter,
                          R = R.new, eps = eps.new)
    cat(sprintf("    NEW    MMMD   = %.3f\n", pn))

    out <- rbind(out, c(n.seq[k], pg, pe, pn))
  }

  out <- as.data.frame(out)
  colnames(out) <- c("n", "GAUSS_single", "GEXP_MMMD", "NEW_MMMD")
  out
}

################################################################################
# 8. Standalone correctness check (call by hand)
################################################################################
# verify.fastcov() should return a tiny number (~1e-12) on a small toy sample,
# confirming fast.cov.gaussian.t reproduces est.cov for Gaussian kernels.
verify.fastcov <- function(seed = 1L, m = 30, R = 5) {
  set.seed(seed)
  X     <- matrix(rnorm(m * 2), m, 2)
  t.vec <- 2^seq(-2, 2, length.out = R)
  k.vec <- lapply(t.vec, function(t) kernlab::rbfdot(sigma = t))
  Sa <- est.cov           (m, X, k.vec)
  Sb <- fast.cov.gaussian.t(X, t.vec)
  max(abs(Sa - Sb))
}
