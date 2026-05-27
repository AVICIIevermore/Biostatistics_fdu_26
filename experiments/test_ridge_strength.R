## Test different ridge strengths for Multi-LAP

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

# Paper's functions
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

cov.mat.est.ridge <- function(n, k.vec, x, ridge_factor = 1e-5){
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
  return (cov.mat.est + ridge_factor*min(diag(cov.mat.est))*diag(1, k.len,k.len))
}

# Build kernels
kernel_vec <- k.choice(X, Y, "LAP")
MMD_vec <- compute.MMD.vec(X, Y, kernel_vec)
n <- nrow(X)

cat("=== Testing Different Ridge Strengths ===\n\n")

ridge_factors <- c(1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

for (rf in ridge_factors) {
  cat(sprintf("Ridge factor: %.0e\n", rf))

  cov_mat <- cov.mat.est.ridge(n, kernel_vec, X, ridge_factor = rf)
  eig <- eigen(cov_mat, symmetric = TRUE, only.values = TRUE)$values
  kappa <- max(eig) / min(eig)

  invcov <- solve(cov_mat)
  Tobs <- t(MMD_vec) %*% invcov %*% MMD_vec

  cat("  Condition number:", sprintf("%.1e", kappa), "\n")
  cat("  Max element in inv(S):", sprintf("%.1e", max(abs(invcov))), "\n")
  cat("  T_obs:", sprintf("%.3f", Tobs), "\n\n")
}
