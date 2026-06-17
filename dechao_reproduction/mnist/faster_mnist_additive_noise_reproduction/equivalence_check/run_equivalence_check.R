args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1) args[[1]] else "."
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

old_root <- normalizePath(file.path(output_dir, "..", "..", "mnist_additive_noise_reproduction"), mustWork = TRUE)
fast_root <- normalizePath(file.path(output_dir, ".."), mustWork = TRUE)
old_env <- new.env(parent = globalenv())
fast_env <- new.env(parent = globalenv())
sys.source(file.path(old_root, "Code", "Kernel_Based_Tests", "Functions.R"), envir = old_env)
sys.source(file.path(fast_root, "Code", "Kernel_Based_Tests", "Functions.R"), envir = fast_env)
sys.source(file.path(fast_root, "config.R"), envir = fast_env)
sys.source(file.path(fast_root, "mnist_loader.R"), envir = fast_env)

config <- fast_env$experiment_config
n_rep_check <- 2L
B_boot_check <- 10L
sigma_value <- config$noise_levels[[3]]
resamp <- config$resamp
alpha <- config$alpha
kernel_choice <- config$kernel_choice

append_line <- function(path, text){
  cat(text, "\n", sep = "", file = path, append = TRUE)
  cat(text, "\n", sep = "")
}

old_single_diag <- function(X, Y, kernel_choice, alpha, u.mat){
  sigma.med <- old_env$med.bandwidth(X, Y)
  kernel <- if (kernel_choice == "GAUSS") {
    kernlab::rbfdot(sigma = sigma.med)
  } else if (kernel_choice == "LAP") {
    kernlab::laplacedot(sigma = sqrt(sigma.med))
  } else {
    stop("unsupported single kernel")
  }
  n <- nrow(X)
  C <- diag(1, nrow = n, ncol = n) - (1 / n) * matrix(1, nrow = n, ncol = n)
  kxx <- kernlab::kernelMatrix(kernel, X)
  kyy <- kernlab::kernelMatrix(kernel, Y)
  kxy <- kernlab::kernelMatrix(kernel, X, Y)
  kboot <- (1 / n) * (C %*% kxx %*% C)
  test.stat <- colSums(t(u.mat) * (kboot %*% t(u.mat))) - 2 * psych::tr(kboot)
  cutoff <- stats::quantile(test.stat, probs = 1 - alpha, names = FALSE)
  mmd.value <- n * mean(kxx[row(kxx) != col(kxx)] + kyy[row(kyy) != col(kyy)] - 2 * kxy[row(kxy) != col(kxy)])
  list(mmd.value = unname(mmd.value), cutoff = unname(cutoff), reject = unname(mmd.value > cutoff))
}

old_multi_diag <- function(X, Y, kernel_choice, alpha, u.mat){
  n <- nrow(X)
  kernel.vec <- old_env$k.choice(X, Y, kernel_choice)
  C <- diag(1, nrow = n, ncol = n) - (1 / n) * matrix(1, nrow = n, ncol = n)
  kxx.list <- lapply(kernel.vec, function(k) kernlab::kernelMatrix(k, X))
  kyy.list <- lapply(kernel.vec, function(k) kernlab::kernelMatrix(k, Y))
  kxy.list <- lapply(kernel.vec, function(k) kernlab::kernelMatrix(k, X, Y))
  centered <- lapply(kxx.list, function(kxx) C %*% kxx %*% C)
  sigma.hat <- matrix(0, nrow = length(kernel.vec), ncol = length(kernel.vec))
  for (i in seq_along(centered)) {
    for (j in seq_along(centered)) {
      sigma.hat[i, j] <- (8 / (n ^ 2)) * psych::tr(centered[[i]] %*% centered[[j]])
    }
  }
  sigma.hat <- sigma.hat + (10 ^ -5) * min(diag(sigma.hat)) * diag(1, length(kernel.vec), length(kernel.vec))
  inv.cov <- solve(sigma.hat)
  kboot.list <- lapply(kxx.list, function(kxx) (1 / n) * (C %*% kxx %*% C))
  test.kernel.mat <- sapply(kboot.list, function(kboot) {
    colSums(t(u.mat) * (kboot %*% t(u.mat))) - 2 * psych::tr(kboot)
  })
  if (is.null(dim(test.kernel.mat))) {
    test.kernel.mat <- matrix(test.kernel.mat, ncol = 1)
  }
  test.stat <- apply(test.kernel.mat, 1, function(x) old_env$multi.func(x, param = inv.cov))
  cutoff <- stats::quantile(test.stat, probs = 1 - alpha, names = FALSE)
  mmd.vector <- vapply(seq_along(kernel.vec), function(i) {
    n * mean(kxx.list[[i]][row(kxx.list[[i]]) != col(kxx.list[[i]])] +
      kyy.list[[i]][row(kyy.list[[i]]) != col(kyy.list[[i]])] -
      2 * kxy.list[[i]][row(kxy.list[[i]]) != col(kxy.list[[i]])])
  }, numeric(1))
  stat <- old_env$multi.func(mmd.vector, param = inv.cov)
  list(mmd.vector = unname(mmd.vector), sigma.hat = sigma.hat, cutoff = unname(cutoff), reject = unname(stat > cutoff))
}

max_abs_diff <- function(x, y){
  if (is.null(x) && is.null(y)) return(0)
  max(abs(as.numeric(x) - as.numeric(y)))
}

compare_objects <- function(old_obj, fast_obj, type){
  if (type == "single") {
    diffs <- c(
      mmd = max_abs_diff(old_obj$mmd.value, fast_obj$mmd.value),
      cutoff = max_abs_diff(old_obj$cutoff, fast_obj$cutoff),
      reject = as.numeric(old_obj$reject != fast_obj$reject)
    )
    pass <- isTRUE(all.equal(old_obj$mmd.value, fast_obj$mmd.value, tolerance = 1e-10)) &&
      isTRUE(all.equal(old_obj$cutoff, fast_obj$cutoff, tolerance = 1e-10)) &&
      identical(old_obj$reject, fast_obj$reject)
  } else {
    diffs <- c(
      mmd = max_abs_diff(old_obj$mmd.vector, fast_obj$mmd.vector),
      sigma = max_abs_diff(old_obj$sigma.hat, fast_obj$sigma.hat),
      cutoff = max_abs_diff(old_obj$cutoff, fast_obj$cutoff),
      reject = as.numeric(old_obj$reject != fast_obj$reject)
    )
    pass <- isTRUE(all.equal(old_obj$mmd.vector, fast_obj$mmd.vector, tolerance = 1e-10)) &&
      isTRUE(all.equal(old_obj$sigma.hat, fast_obj$sigma.hat, tolerance = 1e-10)) &&
      isTRUE(all.equal(old_obj$cutoff, fast_obj$cutoff, tolerance = 1e-10)) &&
      identical(old_obj$reject, fast_obj$reject)
  }
  list(pass = pass, diffs = diffs)
}

mnist_data <- fast_env$load_mnist_train_flat(file.path(fast_root, "data"))
train.x <- mnist_data$train.x
train.label <- mnist_data$train.label
selected <- fast_env$select_digit_pools(train.x, train.label, config$set_x, config$set_y)

set.seed(31001L)
data.X <- SpatialPack::imnoise(selected$data.X, type = "gaussian", mean = 0, sd = sigma_value)
set.seed(31002L)
data.Y <- SpatialPack::imnoise(selected$data.Y, type = "gaussian", mean = 0, sd = sigma_value)

set.seed(31003L)
resamp.x <- matrix(sample.int(nrow(data.X), size = n_rep_check * resamp, replace = TRUE), nrow = n_rep_check)
set.seed(31004L)
resamp.y <- matrix(sample.int(nrow(data.Y), size = n_rep_check * resamp, replace = TRUE), nrow = n_rep_check)

bootstrap_inputs <- list(
  single = list(),
  multi = list()
)
for (kernel_name in kernel_choice[1:2]) {
  bootstrap_inputs$single[[kernel_name]] <- lapply(seq_len(n_rep_check), function(i) {
    set.seed(32000L + match(kernel_name, kernel_choice) * 100L + i)
    MASS::mvrnorm(B_boot_check, mu = rep(0, resamp), Sigma = diag(2, nrow = resamp, ncol = resamp))
  })
}
for (kernel_name in kernel_choice[3:5]) {
  bootstrap_inputs$multi[[kernel_name]] <- lapply(seq_len(n_rep_check), function(i) {
    set.seed(33000L + match(kernel_name, kernel_choice) * 100L + i)
    MASS::mvrnorm(B_boot_check, mu = rep(0, resamp), Sigma = diag(2, nrow = resamp, ncol = resamp))
  })
}

saveRDS(
  list(
    sigma_value = sigma_value,
    n_rep_check = n_rep_check,
    B_boot_check = B_boot_check,
    resamp = resamp,
    set_x = config$set_x,
    set_y = config$set_y,
    resamp.x = resamp.x,
    resamp.y = resamp.y,
    bootstrap_inputs = bootstrap_inputs,
    noise_seeds = c(data_x = 31001L, data_y = 31002L)
  ),
  file.path(output_dir, "inputs.rds")
)

comparison_path <- file.path(output_dir, "comparison.log")
if (file.exists(comparison_path)) file.remove(comparison_path)
append_line(comparison_path, sprintf("Baseline equivalence check: sigma=%.3f, n_rep=%d, B_boot=%d", sigma_value, n_rep_check, B_boot_check))

results <- list(single = list(), multi = list())
overall_pass <- TRUE

for (kernel_name in kernel_choice[1:2]) {
  results$single[[kernel_name]] <- vector("list", n_rep_check)
  for (i in seq_len(n_rep_check)) {
    X <- as.matrix(data.X[resamp.x[i, ],, drop = FALSE])
    Y <- as.matrix(data.Y[resamp.y[i, ],, drop = FALSE])
    u.mat <- bootstrap_inputs$single[[kernel_name]][[i]]
    old_obj <- old_single_diag(X, Y, kernel_name, alpha, u.mat)
    fast_obj <- fast_env$single_mmd_diagnostics(X, Y, kernel_name, B_boot = B_boot_check, alpha = alpha, u.mat = u.mat)
    cmp <- compare_objects(old_obj, fast_obj, type = "single")
    results$single[[kernel_name]][[i]] <- list(old = old_obj, fast = fast_obj, compare = cmp)
    overall_pass <- overall_pass && cmp$pass
    append_line(comparison_path, sprintf("single %s rep %d: pass=%s mmd=%.3e cutoff=%.3e reject_diff=%d",
      kernel_name, i, cmp$pass, cmp$diffs[["mmd"]], cmp$diffs[["cutoff"]], as.integer(cmp$diffs[["reject"]])))
  }
}

for (kernel_name in kernel_choice[3:5]) {
  results$multi[[kernel_name]] <- vector("list", n_rep_check)
  for (i in seq_len(n_rep_check)) {
    X <- as.matrix(data.X[resamp.x[i, ],, drop = FALSE])
    Y <- as.matrix(data.Y[resamp.y[i, ],, drop = FALSE])
    u.mat <- bootstrap_inputs$multi[[kernel_name]][[i]]
    old_obj <- old_multi_diag(X, Y, kernel_name, alpha, u.mat)
    fast_obj <- fast_env$multi_mmd_diagnostics(X, Y, kernel_name, B_boot = B_boot_check, alpha = alpha, u.mat = u.mat)
    cmp <- compare_objects(old_obj, fast_obj, type = "multi")
    results$multi[[kernel_name]][[i]] <- list(old = old_obj, fast = fast_obj, compare = cmp)
    overall_pass <- overall_pass && cmp$pass
    append_line(comparison_path, sprintf("multi %s rep %d: pass=%s mmd=%.3e sigma=%.3e cutoff=%.3e reject_diff=%d",
      kernel_name, i, cmp$pass, cmp$diffs[["mmd"]], cmp$diffs[["sigma"]], cmp$diffs[["cutoff"]], as.integer(cmp$diffs[["reject"]])))
  }
}

saveRDS(list(pass = overall_pass, results = results), file.path(output_dir, "comparison.rds"))
append_line(comparison_path, sprintf("OVERALL %s", if (overall_pass) "PASS" else "FAIL"))
if (!overall_pass) stop("equivalence check failed")
