################################################################################
# Tiny smoke test: n.iter = 20, n.seq = c(50, 100, 200) - takes ~30 seconds
################################################################################
source("Functions.R")

library(foreach); library(doParallel)
cores <- max(1, parallel::detectCores() - 1)
cl <- makeCluster(cores); registerDoParallel(cl)

t0 <- Sys.time()
df <- power.time.d(n.seq = c(50, 100, 200),
                   sigma.param = 1, sigma.mult = 1.2,
                   mu.param = 0, d = 5, p = 0, n.iter = 20)
t1 <- Sys.time()
stopCluster(cl)

cat("\n\n--- summary ---\n")
print(df[, c("n", "method", "power", "q_mean", "t_cov", "t_total")], row.names = FALSE)
cat(sprintf("\nElapsed: %.1f sec\n", as.numeric(t1 - t0, units = "secs")))
