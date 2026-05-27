## Diagnostic script for Multi-LAP failure
## Run this to see why Multi-LAP (orig) has zero power

source("R/load.R")

# Load diabetes data
df <- read.csv("data/diabetes.csv", header = TRUE, stringsAsFactors = FALSE)
df_X <- df[df$Outcome == 0, ]
df_Y <- df[df$Outcome == 1, ]
full_X <- scale(as.matrix(df_X[, setdiff(names(df_X), "Outcome")]))
full_Y <- scale(as.matrix(df_Y[, setdiff(names(df_Y), "Outcome")]))

# Sample m=50 from each group
set.seed(123)
m <- 50
X <- full_X[sample(nrow(full_X), m, replace = TRUE), ]
Y <- full_Y[sample(nrow(full_Y), m, replace = TRUE), ]

# Paper's functions (copied from diabetes_comparison.R)
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
  return (cov.mat.est + (10^-5)*min(diag(cov.mat.est))*diag(1, k.len,k.len))
}

# Run Multi-LAP diagnostic
cat("=== Multi-LAP Diagnostic ===\n\n")

# 1. Construct kernel list
kernel_vec <- k.choice(X, Y, "LAP")
cat("Number of kernels:", length(kernel_vec), "\n")

# 2. Extract bandwidths
sigma.vec <- expo.band(X, Y, -2, 2)
cat("Median bandwidth (before sqrt):", sigma.vec[3], "\n")
cat("Laplace bandwidths (after sqrt):", sqrt(sigma.vec), "\n\n")

# 3. Compute MMD vector
MMD_vec <- compute.MMD.vec(X, Y, kernel_vec)
cat("MMD vector:\n")
print(MMD_vec)
cat("\n")

# 4. Compute covariance matrix
n <- nrow(X)
cov_mat <- cov.mat.est(n, kernel_vec, X)
cat("Covariance matrix Sigma_hat:\n")
print(cov_mat)
cat("\n")

# 5. Check condition number
eig <- eigen(cov_mat, symmetric = TRUE, only.values = TRUE)$values
kappa <- max(eig) / min(eig)
cat("Eigenvalues of Sigma_hat:", eig, "\n")
cat("Condition number kappa(Sigma_hat):", kappa, "\n\n")

# 6. Invert and check
invcov <- solve(cov_mat)
cat("Inverse covariance matrix Sigma_hat^{-1}:\n")
print(invcov)
cat("\n")
cat("Max element in Sigma_hat^{-1}:", max(abs(invcov)), "\n")
cat("Min element in Sigma_hat^{-1}:", min(abs(invcov[invcov != 0])), "\n\n")

# 7. Compute Mahalanobis statistic
Tobs <- t(MMD_vec) %*% invcov %*% MMD_vec
cat("Observed Mahalanobis statistic T_obs:", Tobs, "\n\n")

# 8. Compare with Single-LAP
single_kernel <- kernlab::laplacedot(sigma = sqrt(sigma.vec[3]))  # median bandwidth
single_MMD <- compute.MMD(X, Y, single_kernel)
cat("Single-LAP MMD^2 (median bandwidth):", single_MMD, "\n")
cat("Scaled Single-LAP statistic (n * MMD^2):", n * single_MMD, "\n")
