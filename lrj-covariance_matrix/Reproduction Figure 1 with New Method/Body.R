################################################################################
# Body.R - reproduces Figure 1(b) Power and adds the new method.
#
# Figure 1(b) setup: Gaussian Scale alternative, d = 2, sigma.mult = 1.25
#   X ~ N(0, I_2),    Y ~ N(0, 1.25 * I_2)
#
# Defaults below are a "quick demo" (n.rep = 10, n.iter = 100) -- finishes in
# roughly 30-60 minutes on a multi-core workstation. To match the published
# figure exactly, set n.rep = 50 and n.iter = 500 (roughly 6-12 hours).
#
# Run from this directory:
#   conda activate mmmd_boost
#   Rscript --no-save Body.R
################################################################################

source("Functions.R")

start <- Sys.time()

## ------------------------------ parameters ----------------------------------
n.rep       <- 10                         # power-curve repetitions
n.seq       <- c(50, 100, 200, 300, 400, 500)
d           <- 2                          # data dimension
p           <- 0                          # mixing prob (0 = pure Gaussian)
sigma.param <- 1                          # Sigma_0 = sigma.param * I
sigma.mult  <- 1.25                       # Sigma_1 = sigma.mult * Sigma_0  (Power)
mu.param    <- 0                          # no mean shift
n.iter      <- 100                        # bootstrap + MC repetitions per power point

# new method's hyperparameters
R.new       <- 13                         # candidate Gaussian kernels (arithmetic t-grid)
eps.new     <- 1e-4                       # pivoted Cholesky relative-residual cutoff

## ------------------------- one-time correctness check -----------------------
err <- verify.fastcov()
cat(sprintf("Sanity check: max|est.cov - fast.cov.gaussian.t| = %.3e\n", err))
if (err > 1e-6) stop("fast.cov.gaussian.t disagrees with est.cov - aborting.")

## --------------------- parallel cluster (local cores) -----------------------
library(foreach)
library(doParallel)

cores <- max(1, parallel::detectCores() - 1)
cat(sprintf("Using %d cores.\n", cores))
cl <- makeCluster(cores)
registerDoParallel(cl)

## --------------------------- run experiment ---------------------------------
out.list <- vector("list", n.rep)
for (rep in seq_len(n.rep)) {
  cat(sprintf("\n=========  repetition %d / %d  =========\n", rep, n.rep))
  out.list[[rep]] <- power.d(n.seq, sigma.param, sigma.mult, mu.param,
                             d, p, n.iter,
                             R.new = R.new, eps.new = eps.new)
  saveRDS(out.list, file = "intermediate.rds")     # crash-resume snapshot
}

stopCluster(cl)

## --------------------------- save CSV outputs -------------------------------
methods <- c("GAUSS_single", "GEXP_MMMD", "NEW_MMMD")
for (m in methods) {
  mat <- matrix(0, length(n.seq), n.rep)
  for (rep in seq_len(n.rep)) mat[, rep] <- out.list[[rep]][[m]]
  df  <- cbind(n = n.seq, as.data.frame(mat))
  colnames(df)[-1] <- paste0("rep", seq_len(n.rep))
  write.csv(df, file = sprintf("Power-%s.csv", m), row.names = FALSE)
}

end <- Sys.time()
cat(sprintf("\nTotal wall time: %.1f minutes\n",
            as.numeric(end - start, units = "mins")))
