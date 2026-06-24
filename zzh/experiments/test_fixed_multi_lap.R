## Test the fixed Multi-LAP implementation

source("R/load.R")

df <- read.csv("data/diabetes.csv", header = TRUE, stringsAsFactors = FALSE)
df_X <- df[df$Outcome == 0, ]
df_Y <- df[df$Outcome == 1, ]
full_X <- scale(as.matrix(df_X[, setdiff(names(df_X), "Outcome")]))
full_Y <- scale(as.matrix(df_Y[, setdiff(names(df_Y), "Outcome")]))

set.seed(123)
m <- 50
X <- full_X[sample(nrow(full_X), m, replace = TRUE), ]
Y <- full_Y[sample(nrow(full_Y), m, replace = TRUE), ]

# Paper's functions with fixes
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

med.bandwidth <- function(X, Y){
  X <- as.matrix(X); Y <- as.matrix(Y)
  Z <- rbind(X,Y)
  nu.med <- median(dist(Z)^2)
  sigma.hat <- 1/nu.med
  return (sigma.hat)
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

k.choice <- function(X,Y, kernel.choice){
  if (kernel.choice == "LAP"){
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
  # Method A: ridge = 1e-2 (was 1e-5)
  return (cov.mat.est + (1e-2)*min(diag(cov.mat.est))*diag(1, k.len,k.len))
}

multi.func <- function(x, param){
  out <- t(x)%*%param%*%x
  return (out)
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

.orig_multi_mmd <- function(X, Y, kernel_choice, B, alpha=0.05) {
  kernel_vec <- k.choice(X, Y, kernel_choice)
  n <- nrow(X)
  # Method B: Scale MMD vector by n
  MMD_vec <- n * compute.MMD.vec(X, Y, kernel_vec)
  cov_mat <- cov.mat.est(n, kernel_vec, X)
  invcov <- solve(cov_mat)
  Tobs <- multi.func(MMD_vec, invcov)
  threshold <- multi.H0.cutoff(n, X, kernel_vec, invcov, n.iter = B)
  list(reject = (Tobs > threshold), Tobs = Tobs, threshold = threshold)
}

cat("=== Fixed Multi-LAP Diagnostic ===\n\n")

# Run test
result <- .orig_multi_mmd(X, Y, "LAP", B = 50, alpha = 0.05)

cat("T_obs:", result$Tobs, "\n")
cat("Threshold (95th percentile):", result$threshold, "\n")
cat("Reject H0?", result$reject, "\n\n")

# Compare with new framework
cat("=== New Framework MMMD-LAP (for comparison) ===\n")
kernels <- mmmd_make_kernels(X, Y, family = "LAP", r = 5)
invS <- mmmd_inv_cov(mmmd_est_cov(X, kernels))
Tstar <- mmmd_bootstrap(X, kernels, invS, B = 50)
mmd_vec <- nrow(X) * mmmd_mmd_vector(X, Y, kernels)
Tobs_new <- as.numeric(t(mmd_vec) %*% invS %*% mmd_vec)
threshold_new <- quantile(Tstar, probs = 0.95)
reject_new <- Tobs_new > threshold_new

cat("T_obs:", Tobs_new, "\n")
cat("Threshold (95th percentile):", threshold_new, "\n")
cat("Reject H0?", reject_new, "\n")
