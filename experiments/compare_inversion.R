## Compare solve() vs Rfast::spdinv() on the ill-conditioned LAP covariance matrix

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

# Build LAP kernels using new framework
kernels <- mmmd_make_kernels(X, Y, family = "LAP", r = 5)
S <- mmmd_est_cov(X, kernels)

cat("=== Covariance Matrix Sigma_hat ===\n")
print(S)
cat("\n")

eig <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
kappa <- max(eig) / min(eig)
cat("Eigenvalues:", eig, "\n")
cat("Condition number:", kappa, "\n\n")

# Compare solve() vs Rfast::spdinv()
cat("=== Method 1: solve() (paper original) ===\n")
inv1 <- solve(S)
cat("Max element:", max(abs(inv1)), "\n")
cat("Frobenius norm:", norm(inv1, "F"), "\n\n")

cat("=== Method 2: Rfast::spdinv() (new framework) ===\n")
inv2 <- Rfast::spdinv(S)
cat("Max element:", max(abs(inv2)), "\n")
cat("Frobenius norm:", norm(inv2, "F"), "\n\n")

cat("=== Comparison ===\n")
cat("Difference (Frobenius norm):", norm(inv1 - inv2, "F"), "\n")
cat("Relative difference:", norm(inv1 - inv2, "F") / norm(inv1, "F"), "\n\n")

# Test: S * inv(S) should be identity
cat("=== Numerical Stability Check ===\n")
I1 <- S %*% inv1
I2 <- S %*% inv2
cat("solve():        ||S * inv(S) - I||_F =", norm(I1 - diag(5), "F"), "\n")
cat("Rfast::spdinv: ||S * inv(S) - I||_F =", norm(I2 - diag(5), "F"), "\n")
