################################################################################
# Body.R - time-and-power comparison experiment.
#
# Reproduces "Time and Power Comparison/MMMDvsMMD/" with 4 methods:
#   GEXP_5        : paper baseline, 5 Gaussian kernels, naive est.cov
#   GEXP_rich_13  : paper algorithm but with 13 Gaussian kernels - shows R^2 cost
#   NEW_13        : 13 Gaussian arithmetic grid, fast cov + pivoted Cholesky
#   NEW_25        : 25 Gaussian arithmetic grid, fast cov + pivoted Cholesky
#
# Setup matches the paper's MMMDvsMMD experiment:
#   d = 5, sigma.mult = 1.2
# Sample sizes go up to 700; we use n.iter = 200, n.rep = 1.
# Wall-clock budget on a 23-core workstation: ~40-60 minutes.
################################################################################

source("Functions.R")
start <- Sys.time()

## ------------------------------ parameters ----------------------------------
n.seq       <- c(50, 100, 200, 300, 400, 500, 600, 700)
n.iter      <- 200
n.rep       <- 5
d           <- 5
p           <- 0
sigma.param <- 1
sigma.mult  <- 1.2
mu.param    <- 0

## --------------------- parallel cluster (local cores) -----------------------
library(foreach)
library(doParallel)
cores <- max(1, parallel::detectCores() - 1)
cat(sprintf("Using %d cores.\n", cores))
cl <- makeCluster(cores)
registerDoParallel(cl)

## --------------------------- run experiment ---------------------------------
df.all <- NULL
for (rep in seq_len(n.rep)) {
  cat(sprintf("\n=========  repetition %d / %d  =========\n", rep, n.rep))
  df <- power.time.d(n.seq, sigma.param, sigma.mult, mu.param, d, p, n.iter)
  df$rep <- rep
  df.all <- rbind(df.all, df)
  write.csv(df.all, file = "TimePower-all.csv", row.names = FALSE)
}

stopCluster(cl)

end <- Sys.time()
cat(sprintf("\nTotal wall time: %.1f minutes\n",
            as.numeric(end - start, units = "mins")))
