## Diagnostic for new framework's MMMD-LAP

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

cat("=== New Framework MMMD-LAP Diagnostic ===\n\n")

# Build kernels
kernels <- mmmd_make_kernels(X, Y, family = "LAP", r = 5)
cat("Number of kernels:", length(kernels), "\n\n")

# Compute MMD vector
mmd_vec <- nrow(X) * mmmd_mmd_vector(X, Y, kernels)
cat("MMD vector (scaled by n):\n")
print(mmd_vec)
cat("\n")

# Estimate covariance
S <- mmmd_est_cov(X, kernels)
cat("Covariance matrix Sigma_hat:\n")
print(S)
cat("\n")

eig <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
kappa <- max(eig) / min(eig)
cat("Eigenvalues:", eig, "\n")
cat("Condition number:", kappa, "\n\n")

# Invert
invS <- mmmd_inv_cov(S)
cat("Inverse covariance (first 3x3 block):\n")
print(invS[1:3, 1:3])
cat("Max element in inv(S):", max(abs(invS)), "\n\n")

# Compute Mahalanobis statistic
Tobs <- as.numeric(t(mmd_vec) %*% invS %*% mmd_vec)
cat("Observed Mahalanobis statistic T_obs:", Tobs, "\n\n")

# Run bootstrap
Tstar <- mmmd_bootstrap(X, kernels, invS, B = 50)
cat("Bootstrap statistics T* (first 10):\n")
print(Tstar[1:10])
cat("\n")
cat("Bootstrap quantiles:\n")
print(quantile(Tstar, probs = c(0.05, 0.5, 0.95)))
cat("\n")

# Test decision
threshold <- quantile(Tstar, probs = 0.95)
reject <- Tobs > threshold
cat("Rejection threshold (95th percentile):", threshold, "\n")
cat("Reject H0?", reject, "\n")
