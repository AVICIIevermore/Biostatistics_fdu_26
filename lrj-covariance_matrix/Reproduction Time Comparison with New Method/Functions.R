################################################################################
# Functions.R - reproduces the paper's Time-and-Power comparison experiment
# (Time and Power Comparison/MMMDvsMMD/) with the new method added.
#
# Four methods compared:
#   GEXP-5      : 5 Gaussian kernels, geometric grid sigma^2 = 2^l * t_med, l in -2..2
#                 paper baseline, naive est.cov
#   GEXP-rich-13: 13 Gaussian kernels, geometric grid 2^seq(-3, 3, length.out=13)
#                 same naive est.cov - shows what happens when you just add kernels
#   NEW-13      : 13 Gaussian kernels, arithmetic t-grid [0.25, 4]*t_med,
#                 fast covariance + pivoted Cholesky pruning
#   NEW-25      : same as NEW-13 but with 25 candidate kernels - stress-tests
#                 that the new method scales gracefully with R
#
# Each foreach iteration records per-stage wall-clock time:
#   t_kern  : building the kernel object list  (and the t-grid for NEW)
#   t_cov   : covariance estimation            (est.cov vs fast.cov.gaussian.t)
#   t_sel   : pivoted Cholesky pruning         (NEW only, 0 for GEXP)
#   t_boot  : bootstrap cutoff computation
#   t_stat  : MMD vector + Sigma^-1 + test stat
#   t_total : sum of all of the above
# Times are accumulated per iteration and averaged to mean per-iteration time.
################################################################################

############################ Sample generators #################################

mvrnorm.compat <- function(n, mu, Sigma) {
  LaplacesDemon::rmvn(n = n, mu = mu, Sigma = Sigma)
}

X.gen <- function(n, dim, p, gen.var) {
  mu0 <- gen.var[[1]]; Sigma0 <- gen.var[[3]]
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
  mu1 <- gen.var[[2]]; Sigma1 <- gen.var[[4]]
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

############################ MMD primitives ####################################

med.bandwidth <- function(X, Y) {
  X <- as.matrix(X); Y <- as.matrix(Y)
  1 / median(dist(rbind(X, Y))^2)
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

multi.func <- function(x, param) drop(t(x) %*% param %*% x)

multi.k.approx.stat <- function(k.mat, u.mat) {
  colSums(t(u.mat) * (k.mat %*% t(u.mat))) - 2 * psych::tr(k.mat)
}

##################### Naive est.cov (paper's slow path) ########################

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
  C <- diag(1, n) - (1 / n) * matrix(1, n, n)
  Kc <- lapply(k.vec, function(k) (1 / n) * (C %*% kernlab::kernelMatrix(k, x) %*% C))
  invS <- Rfast::spdinv(Sigma)
  u.mat <- mvrnorm.compat(n.iter, mu = rep(0, n), Sigma = diag(2, n))
  T.mat <- sapply(Kc, multi.k.approx.stat, u.mat = u.mat)
  ts    <- apply(T.mat, 1, multi.func, param = invS)
  quantile(ts, probs = 0.95)
}

####################### Kernel-set builders ####################################

# Geometric grid sigma^2 = 2^l * t_med, l in [l0, l1] (default GEXP = [-2, 2])
k.choice.GEXP <- function(X, Y, l0 = -2, l1 = 2, R = NULL) {
  tm <- med.bandwidth(X, Y)
  if (is.null(R)) {
    ls <- seq(l0, l1, by = 1)
  } else {
    ls <- seq(l0, l1, length.out = R)
  }
  s <- (2^ls) * tm
  lapply(s, function(si) kernlab::rbfdot(sigma = si))
}

############################ NEW method ########################################

# Fast covariance via multiplicative closure for Gaussian kernels
fast.cov.gaussian.t <- function(X, t.vec) {
  m <- nrow(X)
  R <- length(t.vec)

  D <- as.matrix(dist(X))^2

  s.mat <- matrix(0, m, R)
  S.vec <- numeric(R)
  for (a in seq_len(R)) {
    Ka <- exp(-t.vec[a] * D)
    s.mat[, a] <- rowSums(Ka)
    S.vec[a]   <- sum(Ka)
  }

  sum.flat   <- as.vector(outer(t.vec, t.vec, "+"))
  sum.round  <- round(sum.flat, 10)
  uniq.sums  <- unique(sum.round)
  H.unique   <- vapply(uniq.sums, function(u) sum(exp(-u * D)), numeric(1))
  H.mat      <- matrix(H.unique[match(sum.round, uniq.sums)], R, R)

  G.mat <- crossprod(s.mat)
  inner <- H.mat - (2 / m) * G.mat + (1 / m^2) * tcrossprod(S.vec)
  Sigma <- (8 / m^2) * inner
  Sigma <- (Sigma + t(Sigma)) / 2
  Sigma + (1e-5) * min(diag(Sigma)) * diag(R)
}

# Pivoted Cholesky kernel-subset selection
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
  list(selected = selected, q = q)
}

# Arithmetic candidate t-grid, [0.25, 4]*t_med
new.t.grid <- function(X, Y, R = 13) {
  tm <- med.bandwidth(X, Y)
  seq(0.25 * tm, 4.0 * tm, length.out = R)
}

new.H0.cutoff <- function(n, X, k.vec.sel, Sigma.SS, n.iter = 1000) {
  C  <- diag(1, n) - (1 / n) * matrix(1, n, n)
  Kc <- lapply(k.vec.sel, function(k) (1 / n) * (C %*% kernlab::kernelMatrix(k, X) %*% C))
  invS  <- Rfast::spdinv(Sigma.SS)
  u.mat <- mvrnorm.compat(n.iter, mu = rep(0, n), Sigma = diag(2, n))
  T.mat <- sapply(Kc, multi.k.approx.stat, u.mat = u.mat)
  ts    <- apply(T.mat, 1, multi.func, param = invS)
  quantile(ts, probs = 0.95)
}

############################ Timing helpers ####################################

now <- function() Sys.time()
secs <- function(t1, t0) as.numeric(difftime(t1, t0, units = "secs"))

# ------- TIMED GEXP body (one iteration) --------------------------------------
# Used for both GEXP-5 and GEXP-rich-13 by passing different (l0, l1, R).
gexp.timed.one <- function(X, Y, n, n.iter, l0, l1, R = NULL) {
  t0 <- now()
  kv <- k.choice.GEXP(X, Y, l0, l1, R = R)
  t1 <- now()
  Sg <- est.cov(n, X, kv)
  t2 <- now()
  thr <- multi.H0.cutoff(n, X, kv, Sg, n.iter)
  t3 <- now()
  Tn  <- n * compute.MMD.vec(X, Y, kv)
  inv <- Rfast::spdinv(Sg)
  stat <- multi.func(Tn, inv)
  t4 <- now()
  c(rej     = as.integer(stat > thr),
    q       = length(kv),
    t_kern  = secs(t1, t0),
    t_cov   = secs(t2, t1),
    t_sel   = 0,
    t_boot  = secs(t3, t2),
    t_stat  = secs(t4, t3),
    t_total = secs(t4, t0))
}

# ------- TIMED NEW body (one iteration) ---------------------------------------
new.timed.one <- function(X, Y, n, n.iter, R, eps = 1e-4) {
  t0 <- now()
  t.vec <- new.t.grid(X, Y, R = R)
  t1 <- now()
  Sigma.full <- fast.cov.gaussian.t(X, t.vec)
  t2 <- now()
  sel <- pivoted.chol.select(Sigma.full, eps = eps)
  idx <- sel$selected
  Sigma.SS  <- Sigma.full[idx, idx, drop = FALSE]
  k.vec.sel <- lapply(t.vec[idx], function(t) kernlab::rbfdot(sigma = t))
  t3 <- now()
  thr <- new.H0.cutoff(n, X, k.vec.sel, Sigma.SS, n.iter)
  t4 <- now()
  Tn  <- n * compute.MMD.vec(X, Y, k.vec.sel)
  inv <- Rfast::spdinv(Sigma.SS)
  stat <- multi.func(Tn, inv)
  t5 <- now()
  c(rej     = as.integer(stat > thr),
    q       = sel$q,
    t_kern  = secs(t1, t0),
    t_cov   = secs(t2, t1),
    t_sel   = secs(t3, t2),
    t_boot  = secs(t4, t3),
    t_stat  = secs(t5, t4),
    t_total = secs(t5, t0))
}

############################ Method drivers ####################################
# Each returns a list: power, mean per-iter time per stage, mean q.

GEXP.timed <- function(n, d, gen.var, p, n.iter, label, l0, l1, R = NULL) {
  X.list <- lapply(seq_len(n.iter), function(.) X.gen(n, d, p, gen.var))
  Y.list <- lapply(seq_len(n.iter), function(.) Y.gen(n, d, p, gen.var))
  M <- foreach(i = seq_len(n.iter), .combine = rbind,
               .export   = ls(envir = globalenv()),
               .packages = c("LaplacesDemon", "kernlab", "Rfast", "psych")) %dopar% {
    gexp.timed.one(X.list[[i]], Y.list[[i]], n, n.iter, l0, l1, R = R)
  }
  list(label   = label,
       power   = mean(M[, "rej"]),
       q_mean  = mean(M[, "q"]),
       t_kern  = mean(M[, "t_kern"]),
       t_cov   = mean(M[, "t_cov"]),
       t_sel   = 0,
       t_boot  = mean(M[, "t_boot"]),
       t_stat  = mean(M[, "t_stat"]),
       t_total = mean(M[, "t_total"]))
}

NEW.timed <- function(n, d, gen.var, p, n.iter, label, R, eps = 1e-4) {
  X.list <- lapply(seq_len(n.iter), function(.) X.gen(n, d, p, gen.var))
  Y.list <- lapply(seq_len(n.iter), function(.) Y.gen(n, d, p, gen.var))
  M <- foreach(i = seq_len(n.iter), .combine = rbind,
               .export   = ls(envir = globalenv()),
               .packages = c("LaplacesDemon", "kernlab", "Rfast", "psych")) %dopar% {
    new.timed.one(X.list[[i]], Y.list[[i]], n, n.iter, R, eps)
  }
  list(label   = label,
       power   = mean(M[, "rej"]),
       q_mean  = mean(M[, "q"]),
       t_kern  = mean(M[, "t_kern"]),
       t_cov   = mean(M[, "t_cov"]),
       t_sel   = mean(M[, "t_sel"]),
       t_boot  = mean(M[, "t_boot"]),
       t_stat  = mean(M[, "t_stat"]),
       t_total = mean(M[, "t_total"]))
}

############################ Top-level driver ##################################

power.time.d <- function(n.seq, sigma.param, sigma.mult, mu.param, d, p, n.iter) {
  rows <- list()
  for (n in n.seq) {
    library(LaplacesDemon)
    library(Rfast)
    Sigma0 <- diag(sigma.param, d, d)
    Sigma1 <- sigma.mult * Sigma0
    gen.var <- list(rep(0, d), rep(mu.param, d), Sigma0, Sigma1)

    cat(sprintf("\n[%s] n = %d ...\n", format(Sys.time(), "%H:%M:%S"), n))

    out_list <- list(
      GEXP.timed(n, d, gen.var, p, n.iter, "GEXP_5",       l0 = -2, l1 = 2),
      GEXP.timed(n, d, gen.var, p, n.iter, "GEXP_rich_13", l0 = -3, l1 = 3, R = 13),
      NEW.timed (n, d, gen.var, p, n.iter, "NEW_13",       R = 13),
      NEW.timed (n, d, gen.var, p, n.iter, "NEW_25",       R = 25)
    )
    for (o in out_list) {
      cat(sprintf("  %-13s power=%.3f  q=%.1f  t_total=%.3fs",
                  o$label, o$power, o$q_mean, o$t_total))
      cat(sprintf("  [kern=%.3f cov=%.3f sel=%.3f boot=%.3f stat=%.3f]\n",
                  o$t_kern, o$t_cov, o$t_sel, o$t_boot, o$t_stat))
      rows[[length(rows) + 1]] <- data.frame(
        n       = n,
        method  = o$label,
        power   = o$power,
        q_mean  = o$q_mean,
        t_kern  = o$t_kern,
        t_cov   = o$t_cov,
        t_sel   = o$t_sel,
        t_boot  = o$t_boot,
        t_stat  = o$t_stat,
        t_total = o$t_total
      )
    }
    saveRDS(rows, file = "intermediate.rds")
  }
  do.call(rbind, rows)
}
