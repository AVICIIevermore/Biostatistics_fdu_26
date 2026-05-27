## __SELF_LOCATING_PREAMBLE__
# Simple approach: assume user runs from project root
if (!file.exists("R/load.R")) {
  stop("Please run this script from the project root directory:\n  Rscript experiments/diabetes_comparison.R")
}
## __END_PREAMBLE__

################################################################################
########## Diabetes Dataset: Two-Sample Test Method Comparison ################
################################################################################
#
# This script compares 9 two-sample test methods on the Pima Indians Diabetes
# dataset (data/diabetes.csv):
#   - 5 methods from the paper's original MNIST experiments (Single-Gauss,
#     Single-LAP, Multi-LAP, Multi-GEXP, Multi-MIXED)
#   - 3 methods from the new R/ framework (MMMD-GEXP, MMMD-LAP, MMMD-MIXED)
#   - 1 graph-kernel extension (Graph-MMMD)
#
# The dataset has 768 rows × 9 columns. We split by Outcome (0=non-diabetic,
# 1=diabetic) and test whether the feature distributions differ.
#
# Output: results_diabetes/power_vs_sample_size.csv + .png
#
################################################################################

source("R/load.R")

################################################################################
########################## CLI Arguments #######################################
################################################################################

.arg <- function(i, default) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= i && !grepl("^--", args[i])) {
    as.numeric(args[i])
  } else {
    default
  }
}

SMOKE_MODE <- "--smoke" %in% commandArgs(trailingOnly = TRUE)

if (SMOKE_MODE) {
  M_SEQ <- c(50)
  N_TRIALS <- 10
  B <- 50
  cat("SMOKE MODE: M_SEQ=c(50), N_TRIALS=10, B=50\n")
} else {
  M_SEQ <- c(50, 100, 150, 200)
  N_TRIALS <- .arg(1, 200)
  B <- .arg(2, 500)
  cat(sprintf("FULL MODE: M_SEQ=c(50,100,150,200), N_TRIALS=%d, B=%d\n", N_TRIALS, B))
}

ALPHA <- 0.05

################################################################################
########################## Data Loading ########################################
################################################################################

cat("Loading data/diabetes.csv...\n")
df <- read.csv("data/diabetes.csv", header = TRUE, stringsAsFactors = FALSE)

# Split by Outcome (0=non-diabetic, 1=diabetic)
df_X <- df[df$Outcome == 0, ]
df_Y <- df[df$Outcome == 1, ]

# Drop Outcome column, keep 8 features
full_X <- as.matrix(df_X[, setdiff(names(df_X), "Outcome")])
full_Y <- as.matrix(df_Y[, setdiff(names(df_Y), "Outcome")])

# Standardize (center + scale)
full_X <- scale(full_X)
full_Y <- scale(full_Y)

cat(sprintf("  full_X: %d rows × %d cols (Outcome=0)\n", nrow(full_X), ncol(full_X)))
cat(sprintf("  full_Y: %d rows × %d cols (Outcome=1)\n", nrow(full_Y), ncol(full_Y)))

################################################################################
########################## Paper's Original Functions ##########################
################################################################################
# Extracted from MNIST-Additive Noise/Code/Kernel Based Tests/Functions.R
# These are pure functions (no global dependencies) used to wrap the paper's
# Single.MMD and Multi.MMD methods.

# compute.MMD: MMD_u^2 for a single kernel
compute.MMD <- function(X, Y, k){
  k.X <- kernlab::kernelMatrix(k, X)
  k.Y <- kernlab::kernelMatrix(k, Y)
  k.XY <- kernlab::kernelMatrix(k, X, Y)
  MMD.out <- mean(k.X[row(k.X)!=col(k.X)] + k.Y[row(k.Y)!=col(k.Y)] -
                    2*k.XY[row(k.XY)!=col(k.XY)])
  return (MMD.out)
}

# compute.MMD.vec: MMD_u^2 vector for multiple kernels
compute.MMD.vec <- function(X, Y, kernel.vec){
  MMD.vec <- rep(NA, length(kernel.vec))
  for (i in 1:length(kernel.vec)){
    MMD.vec[i] <- compute.MMD(X, Y, kernel.vec[[i]])
  }
  return (MMD.vec)
}

# cov.mat.est: Estimated covariance matrix under H0
cov.mat.est <- function(n, k.vec, x){
  require(psych)  # for tr()
  k.len <- length(k.vec)
  C <- diag(1, nrow = n, ncol = n) - (1/n)*matrix(1, nrow = n, ncol = n)
  kvec.mat <- vector("list", k.len)
  for (i in 1:k.len){
    kvec.mat[[i]] <- C%*%kernlab::kernelMatrix(k.vec[[i]], x)%*%C
  }
  cov.mat.est <- matrix(0, nrow = k.len, ncol = k.len)
  for (i in 1:k.len){
    for (j in 1:k.len){
      cov.mat.est[i,j] <- (8/(n^2))*tr(kvec.mat[[i]]%*%kvec.mat[[j]])
    }
  }
  return (cov.mat.est + (10^-5)*min(diag(cov.mat.est))*diag(1, k.len,k.len))
}

# med.bandwidth: Median bandwidth heuristic
med.bandwidth <- function(X, Y){
  X <- as.matrix(X); Y <- as.matrix(Y)
  Z <- rbind(X,Y)
  nu.med <- median(dist(Z)^2)
  sigma.hat <- 1/nu.med
  return (sigma.hat)
}

# single.H0.cutoff: H0 cutoff for single kernel test (multiplier bootstrap)
single.H0.cutoff <- function(m, x, k, n.iter = 1000){
  require(psych)
  C <- diag(1, nrow = m, ncol = m) - (1/m)*matrix(1, nrow = m, ncol = m)
  k.mat <- (1/m)*(C%*%kernlab::kernelMatrix(k, x)%*%C)
  u.mat <- MASS::mvrnorm(n.iter, mu = rep(0,m),
                         Sigma = diag(2, nrow = m, ncol = m))
  test.stat <- colSums(t(u.mat) * (k.mat %*% t(u.mat))) - 2*tr(k.mat)
  H0.thresh <- sort(test.stat)[as.integer(0.95*n.iter)]
  return (H0.thresh)
}

# min.max.band: Min-max bandwidth heuristic
min.max.band <- function(X, Y){
  X <- as.matrix(X); Y <- as.matrix(Y)
  Z <- rbind(X,Y)
  norm.vals <- dist(Z)^2
  quantiles.norm <- quantile(norm.vals, probs = c(0.95,0.05))
  return (1/quantiles.norm)
}

# expo.band: Exponential bandwidth grid (2^l rule)
expo.band <- function(X,Y, l0, l1){
  med.band <- med.bandwidth(X,Y)
  band <- c();i <- l0
  while(i<=l1){
    band <- c(band,(2^i)*med.band)
    i <- i + 1
  }
  return(band)
}

# multi.func: Mahalanobis-type quadratic form
multi.func <- function(x, param){
  out <- t(x)%*%param%*%x
  return (out)
}

# k.choice: Construct kernel list based on choice
k.choice <- function(X,Y, kernel.choice){
  if (kernel.choice == "MINMAX"){
    sigma.len <- 5
    sigma.bd <- min.max.band(X,Y)
    sigma.vec <- seq(sigma.bd[1],sigma.bd[2], length.out = sigma.len)
    kernel.vec <- c()
    for(i in 1:sigma.len){
      kernel.vec <- c(kernel.vec, kernlab::rbfdot(sigma = sigma.vec[i]))
    }
  }
  else if (kernel.choice == "GEXP"){
    l0 <- -2; l1 <- 2
    sigma.vec <- expo.band(X,Y, l0, l1)
    sigma.len <- length(sigma.vec)
    kernel.vec <- c()
    for(i in 1:sigma.len){
      kernel.vec <- c(kernel.vec, kernlab::rbfdot(sigma = sigma.vec[i]))
    }
  }
  else if (kernel.choice == "MIXED"){
    l0 <- -1; l1 <- 1
    sigma.vec <- expo.band(X,Y, l0, l1)
    sigma.len <- length(sigma.vec)
    kernel.vec <- c()
    for(i in 1:sigma.len){
      kernel.vec <- c(kernel.vec, kernlab::rbfdot(sigma = sigma.vec[i]))
    }
    for(i in 1:sigma.len){
      kernel.vec <- c(kernel.vec, kernlab::laplacedot(sigma = sqrt(sigma.vec[i])))
    }
  }
  else if (kernel.choice == "LAP"){
    l0 <- -2; l1 <- 2
    sigma.vec <- expo.band(X,Y, l0, l1)
    sigma.len <- length(sigma.vec)
    kernel.vec <- c()
    for(i in 1:sigma.len){
      kernel.vec <- c(kernel.vec, kernlab::laplacedot(sigma = sqrt(sigma.vec[i])))
    }
  }
  return (kernel.vec)
}

# multi.H0.cutoff: H0 cutoff for multiple kernel test
multi.H0.cutoff <- function(n, x, k.vec, invcov, n.iter = 1000){
  require(psych)
  k.len <- length(k.vec)
  C <- diag(1, nrow = n, ncol = n) - (1/n)*matrix(1, nrow = n, ncol = n)
  kvec.mat <- vector("list", k.len)
  for (i in 1:k.len){
    kvec.mat[[i]] <- (1/n)*(C%*%kernlab::kernelMatrix(k.vec[[i]], x)%*%C)
  }
  invcov.mat.est <- invcov
  u.mat <- MASS::mvrnorm(n.iter, mu = rep(0,n),
                         Sigma = diag(2, nrow = n, ncol = n))
  multi.k.approx.stat <- function(k.mat, u.mat){
    test.stat <- colSums(t(u.mat) * (k.mat %*% t(u.mat))) - 2*tr(k.mat)
    return (test.stat)
  }
  test.kernel.mat <- sapply(kvec.mat, multi.k.approx.stat, u.mat = u.mat)
  test.stat <- apply(test.kernel.mat, 1, multi.func, param = invcov.mat.est)
  H0.thresh <- sort(test.stat)[as.integer(0.95*n.iter)]
  return (H0.thresh)
}

################################################################################
########################## Method Definitions ##################################
################################################################################

# Wrapper for paper's single-kernel MMD
.orig_single_mmd <- function(X, Y, kernel_choice, B, alpha=0.05) {
  sigma_med <- med.bandwidth(X, Y)
  if (kernel_choice == "GAUSS") {
    kernel <- kernlab::rbfdot(sigma = sigma_med)
  } else if (kernel_choice == "LAP") {
    kernel <- kernlab::laplacedot(sigma = sqrt(sigma_med))
  }
  threshold <- single.H0.cutoff(nrow(X), X, kernel, n.iter = B)
  Tobs <- nrow(X) * compute.MMD(X, Y, kernel)
  list(reject = (Tobs > threshold), Tobs = Tobs, threshold = threshold)
}

# Wrapper for paper's multi-kernel MMMD
.orig_multi_mmd <- function(X, Y, kernel_choice, B, alpha=0.05) {
  kernel_vec <- k.choice(X, Y, kernel_choice)
  MMD_vec <- compute.MMD.vec(X, Y, kernel_vec)
  n <- nrow(X)
  cov_mat <- cov.mat.est(n, kernel_vec, X)
  invcov <- solve(cov_mat)
  Tobs <- multi.func(MMD_vec, invcov)
  threshold <- multi.H0.cutoff(n, X, kernel_vec, invcov, n.iter = B)
  list(reject = (Tobs > threshold), Tobs = Tobs, threshold = threshold)
}

# For now, only new framework methods (MMMD-GEXP, MMMD-LAP, MMMD-MIXED)
# Paper's original methods will be added later

METHODS <- c(
  "Single-Gauss (orig)",
  "Single-LAP (orig)",
  "Multi-LAP (orig)",
  "Multi-GEXP (orig)",
  "Multi-MIXED (orig)",
  "MMMD-GEXP (new)",
  "MMMD-LAP (new)",
  "MMMD-MIXED (new)",
  "Graph-MMMD (new)"
)

cat(sprintf("Methods to compare: %s\n", paste(METHODS, collapse=", ")))

################################################################################
########################## Worker Function #####################################
################################################################################

.worker <- function(i, m, method, full_X, full_Y, B, ALPHA) {
  # Define paper's functions inside worker (so they're available on all workers)
  compute.MMD <- function(X, Y, k){
    k.X <- kernlab::kernelMatrix(k, X)
    k.Y <- kernlab::kernelMatrix(k, Y)
    k.XY <- kernlab::kernelMatrix(k, X, Y)
    MMD.out <- mean(k.X[row(k.X)!=col(k.X)] + k.Y[row(k.Y)!=col(k.Y)] -
                      2*k.XY[row(k.XY)!=col(k.XY)])
    return (MMD.out)
  }
  compute.MMD.vec <- function(X, Y, kernel.vec){
    MMD.vec <- rep(NA, length(kernel.vec))
    for (i in 1:length(kernel.vec)){
      MMD.vec[i] <- compute.MMD(X, Y, kernel.vec[[i]])
    }
    return (MMD.vec)
  }
  cov.mat.est <- function(n, k.vec, x){
    require(psych)
    k.len <- length(k.vec)
    C <- diag(1, nrow = n, ncol = n) - (1/n)*matrix(1, nrow = n, ncol = n)
    kvec.mat <- vector("list", k.len)
    for (i in 1:k.len){
      kvec.mat[[i]] <- C%*%kernlab::kernelMatrix(k.vec[[i]], x)%*%C
    }
    cov.mat.est <- matrix(0, nrow = k.len, ncol = k.len)
    for (i in 1:k.len){
      for (j in 1:k.len){
        cov.mat.est[i,j] <- (8/(n^2))*tr(kvec.mat[[i]]%*%kvec.mat[[j]])
      }
    }
    # Method A: Increase ridge from 1e-5 to 1e-2 for better numerical stability
    return (cov.mat.est + (1e-2)*min(diag(cov.mat.est))*diag(1, k.len,k.len))
  }
  med.bandwidth <- function(X, Y){
    X <- as.matrix(X); Y <- as.matrix(Y)
    Z <- rbind(X,Y)
    nu.med <- median(dist(Z)^2)
    sigma.hat <- 1/nu.med
    return (sigma.hat)
  }
  single.H0.cutoff <- function(m, x, k, n.iter = 1000){
    require(psych)
    C <- diag(1, nrow = m, ncol = m) - (1/m)*matrix(1, nrow = m, ncol = m)
    k.mat <- (1/m)*(C%*%kernlab::kernelMatrix(k, x)%*%C)
    u.mat <- MASS::mvrnorm(n.iter, mu = rep(0,m),
                           Sigma = diag(2, nrow = m, ncol = m))
    test.stat <- colSums(t(u.mat) * (k.mat %*% t(u.mat))) - 2*tr(k.mat)
    H0.thresh <- sort(test.stat)[as.integer(0.95*n.iter)]
    return (H0.thresh)
  }
  min.max.band <- function(X, Y){
    X <- as.matrix(X); Y <- as.matrix(Y)
    Z <- rbind(X,Y)
    norm.vals <- dist(Z)^2
    quantiles.norm <- quantile(norm.vals, probs = c(0.95,0.05))
    return (1/quantiles.norm)
  }
  expo.band <- function(X,Y, l0, l1){
    med.band <- med.bandwidth(X,Y)
    band <- c();i <- l0
    while(i<=l1){
      band <- c(band,(2^i)*med.band)
      i <- i + 1
    }
    return(band)
  }
  multi.func <- function(x, param){
    out <- t(x)%*%param%*%x
    return (out)
  }
  k.choice <- function(X,Y, kernel.choice){
    if (kernel.choice == "MINMAX"){
      sigma.len <- 5
      sigma.bd <- min.max.band(X,Y)
      sigma.vec <- seq(sigma.bd[1],sigma.bd[2], length.out = sigma.len)
      kernel.vec <- c()
      for(i in 1:sigma.len){
        kernel.vec <- c(kernel.vec, kernlab::rbfdot(sigma = sigma.vec[i]))
      }
    }
    else if (kernel.choice == "GEXP"){
      l0 <- -2; l1 <- 2
      sigma.vec <- expo.band(X,Y, l0, l1)
      sigma.len <- length(sigma.vec)
      kernel.vec <- c()
      for(i in 1:sigma.len){
        kernel.vec <- c(kernel.vec, kernlab::rbfdot(sigma = sigma.vec[i]))
      }
    }
    else if (kernel.choice == "MIXED"){
      l0 <- -1; l1 <- 1
      sigma.vec <- expo.band(X,Y, l0, l1)
      sigma.len <- length(sigma.vec)
      kernel.vec <- c()
      for(i in 1:sigma.len){
        kernel.vec <- c(kernel.vec, kernlab::rbfdot(sigma = sigma.vec[i]))
      }
      for(i in 1:sigma.len){
        kernel.vec <- c(kernel.vec, kernlab::laplacedot(sigma = sqrt(sigma.vec[i])))
      }
    }
    else if (kernel.choice == "LAP"){
      l0 <- -2; l1 <- 2
      sigma.vec <- expo.band(X,Y, l0, l1)
      sigma.len <- length(sigma.vec)
      kernel.vec <- c()
      for(i in 1:sigma.len){
        kernel.vec <- c(kernel.vec, kernlab::laplacedot(sigma = sqrt(sigma.vec[i])))
      }
    }
    return (kernel.vec)
  }
  multi.H0.cutoff <- function(n, x, k.vec, invcov, n.iter = 1000){
    require(psych)
    k.len <- length(k.vec)
    C <- diag(1, nrow = n, ncol = n) - (1/n)*matrix(1, nrow = n, ncol = n)
    kvec.mat <- vector("list", k.len)
    for (i in 1:k.len){
      kvec.mat[[i]] <- (1/n)*(C%*%kernlab::kernelMatrix(k.vec[[i]], x)%*%C)
    }
    invcov.mat.est <- invcov
    u.mat <- MASS::mvrnorm(n.iter, mu = rep(0,n),
                           Sigma = diag(2, nrow = n, ncol = n))
    multi.k.approx.stat <- function(k.mat, u.mat){
      test.stat <- colSums(t(u.mat) * (k.mat %*% t(u.mat))) - 2*tr(k.mat)
      return (test.stat)
    }
    test.kernel.mat <- sapply(kvec.mat, multi.k.approx.stat, u.mat = u.mat)
    test.stat <- apply(test.kernel.mat, 1, multi.func, param = invcov.mat.est)
    H0.thresh <- sort(test.stat)[as.integer(0.95*n.iter)]
    return (H0.thresh)
  }
  .orig_single_mmd <- function(X, Y, kernel_choice, B, alpha=0.05) {
    sigma_med <- med.bandwidth(X, Y)
    if (kernel_choice == "GAUSS") {
      kernel <- kernlab::rbfdot(sigma = sigma_med)
    } else if (kernel_choice == "LAP") {
      kernel <- kernlab::laplacedot(sigma = sqrt(sigma_med))
    }
    threshold <- single.H0.cutoff(nrow(X), X, kernel, n.iter = B)
    Tobs <- nrow(X) * compute.MMD(X, Y, kernel)
    list(reject = (Tobs > threshold), Tobs = Tobs, threshold = threshold)
  }
  .orig_multi_mmd <- function(X, Y, kernel_choice, B, alpha=0.05) {
    kernel_vec <- k.choice(X, Y, kernel_choice)
    n <- nrow(X)
    # Method B: Scale MMD vector by n (as in paper's original Multi.MMD line 490)
    MMD_vec <- n * compute.MMD.vec(X, Y, kernel_vec)
    cov_mat <- cov.mat.est(n, kernel_vec, X)
    invcov <- solve(cov_mat)
    Tobs <- multi.func(MMD_vec, invcov)
    threshold <- multi.H0.cutoff(n, X, kernel_vec, invcov, n.iter = B)
    list(reject = (Tobs > threshold), Tobs = Tobs, threshold = threshold)
  }

  # Resample m rows from each group (with replacement)
  X <- full_X[sample(nrow(full_X), m, replace = TRUE), ]
  Y <- full_Y[sample(nrow(full_Y), m, replace = TRUE), ]

  # Call the appropriate test
  if (method == "Single-Gauss (orig)") {
    tr <- .orig_single_mmd(X, Y, "GAUSS", B, ALPHA)
    return(as.integer(tr$reject))
  } else if (method == "Single-LAP (orig)") {
    tr <- .orig_single_mmd(X, Y, "LAP", B, ALPHA)
    return(as.integer(tr$reject))
  } else if (method == "Multi-LAP (orig)") {
    tr <- .orig_multi_mmd(X, Y, "LAP", B, ALPHA)
    return(as.integer(tr$reject))
  } else if (method == "Multi-GEXP (orig)") {
    tr <- .orig_multi_mmd(X, Y, "GEXP", B, ALPHA)
    return(as.integer(tr$reject))
  } else if (method == "Multi-MIXED (orig)") {
    tr <- .orig_multi_mmd(X, Y, "MIXED", B, ALPHA)
    return(as.integer(tr$reject))
  } else if (method == "MMMD-GEXP (new)") {
    tr <- mmmd_test(X, Y, family = "GEXP", r = 5, B = B, alpha = ALPHA)
    return(as.integer(tr$reject))
  } else if (method == "MMMD-LAP (new)") {
    tr <- mmmd_test(X, Y, family = "LAP", r = 5, B = B, alpha = ALPHA)
    return(as.integer(tr$reject))
  } else if (method == "MMMD-MIXED (new)") {
    tr <- mmmd_test(X, Y, family = "MIXED", r = 6, B = B, alpha = ALPHA)
    return(as.integer(tr$reject))
  } else if (method == "Graph-MMMD (new)") {
    tr <- graph_mmmd_test(X, Y, k_nn = 5, t_seq = c(0.1, 0.5, 1, 2, 5), B = B, alpha = ALPHA)
    return(as.integer(tr$reject))
  } else {
    stop(sprintf("Unknown method: %s", method))
  }
}

################################################################################
########################## Main Loop ###########################################
################################################################################

results <- data.frame(method = character(), m = integer(), power = numeric(),
                      stringsAsFactors = FALSE)

cat("\nStarting experiments...\n")
start_time <- Sys.time()

for (m in M_SEQ) {
  for (method in METHODS) {
    cat(sprintf("\n[m=%d, method=%s] Running %d trials...\n", m, method, N_TRIALS))

    rejects <- with_parallel({
      mmmd_foreach(1:N_TRIALS, .worker,
                   m = m, method = method, full_X = full_X, full_Y = full_Y,
                   B = B, ALPHA = ALPHA)
    })

    power <- mean(unlist(rejects))
    results <- rbind(results, data.frame(method = method, m = m, power = power,
                                         stringsAsFactors = FALSE))
    cat(sprintf("  Power = %.3f\n", power))
  }
}

end_time <- Sys.time()
cat(sprintf("\nTotal time: %.1f minutes\n", as.numeric(difftime(end_time, start_time, units = "mins"))))

################################################################################
########################## Output ##############################################
################################################################################

# Write CSV
out_csv <- "results_diabetes/power_vs_sample_size.csv"
write.csv(results, out_csv, row.names = FALSE)
cat(sprintf("\nWrote %s (%d rows)\n", out_csv, nrow(results)))

# Plot
library(ggplot2)

p <- ggplot(results, aes(x = m, y = power, color = method, group = method)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "red", size = 0.5) +
  labs(
    title = "Two-Sample Test Power vs. Sample Size (Diabetes Dataset)",
    x = "Sample size (m = n)",
    y = "Rejection rate (power)",
    color = "Method"
  ) +
  theme_minimal() +
  theme(legend.position = "right")

out_png <- "results_diabetes/power_vs_sample_size.png"
ggsave(out_png, p, width = 10, height = 6, dpi = 150)
cat(sprintf("Wrote %s\n", out_png))

cat("\nDone.\n")
