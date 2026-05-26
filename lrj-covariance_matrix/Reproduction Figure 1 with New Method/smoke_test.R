################################################################################
# Smoke test (call by hand, NOT used by Body.R):
#   - check fast.cov.gaussian.t reproduces est.cov within 1e-8
#   - check pivoted.chol.select returns sane subsets
#   - run a tiny power.d (n.iter = 20, n.seq = c(50, 100)) to time end-to-end
################################################################################

source("Functions.R")

cat("\n--- 1. fast.cov.gaussian.t vs est.cov (Gaussian) ---\n")
err <- verify.fastcov(seed = 1, m = 30, R = 5)
cat(sprintf("max abs diff = %.3e (should be ~1e-12)\n", err))

cat("\n--- 2. pivoted.chol.select on a redundant Sigma ---\n")
set.seed(42)
m <- 50; X <- matrix(rnorm(m * 2), m, 2)
t.med <- med.bandwidth(X, X)
# 7 bandwidths very close to t.med => Sigma should be near singular
t.vec <- t.med * c(0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05)
S     <- fast.cov.gaussian.t(X, t.vec)
sel   <- pivoted.chol.select(S, eps = 1e-3)
cat(sprintf("Eigenvalues of full Sigma: %s\n",
            paste(formatC(sort(eigen(S)$values), format = "e", digits = 2), collapse = " ")))
cat(sprintf("Pivoted Cholesky kept %d of %d kernels: %s\n",
            sel$q, length(t.vec), paste(sel$selected, collapse = ", ")))

cat("\n--- 3. tiny end-to-end power.d run ---\n")
library(foreach); library(doParallel)
cores <- max(1, parallel::detectCores() - 1)
cl <- makeCluster(cores); registerDoParallel(cl)

t0 <- Sys.time()
res <- power.d(n.seq      = c(50, 100),
               sigma.param = 1, sigma.mult = 1.25,
               mu.param    = 0, d = 2, p = 0,
               n.iter      = 20)            # tiny n.iter just for smoke test
t1 <- Sys.time()

stopCluster(cl)

cat("\nResult:\n"); print(res)
cat(sprintf("\nElapsed: %.1f sec on %d cores.\n",
            as.numeric(t1 - t0, units = "secs"), cores))
