################################################################################
# verify_anomaly.R - rerun the suspect (n, method) pairs to test if the
# non-monotone NEW-25 timing at n=700 reproduces.
#
# Plan: 5 repetitions each of the 4 corner pairs, n.iter = 200.
# Total wall time: ~15 minutes on 23 cores.
################################################################################

source("Functions.R")

library(foreach); library(doParallel)

cores <- max(1, parallel::detectCores() - 1)
cat(sprintf("Using %d cores.\n", cores))
cl <- makeCluster(cores)
registerDoParallel(cl)

n.iter <- 200
n.rep  <- 5
d <- 5
sigma.param <- 1
sigma.mult  <- 1.2

gen.var.for <- function(n, d, sigma.param, sigma.mult) {
  Sigma0 <- diag(sigma.param, d, d)
  Sigma1 <- sigma.mult * Sigma0
  list(rep(0, d), rep(0, d), Sigma0, Sigma1)
}

# Each (n, method, rep) record
rows <- list()

pairs <- list(
  list(n = 600, method = "NEW_13"),
  list(n = 600, method = "NEW_25"),
  list(n = 700, method = "NEW_13"),
  list(n = 700, method = "NEW_25")
)

for (rep in seq_len(n.rep)) {
  cat(sprintf("\n========= rep %d / %d =========\n", rep, n.rep))
  # Shuffle order each rep to decorrelate from system-load drift
  ord <- sample(seq_along(pairs))
  for (k in ord) {
    n <- pairs[[k]]$n
    method <- pairs[[k]]$method
    gen.var <- gen.var.for(n, d, sigma.param, sigma.mult)

    invisible(gc(verbose = FALSE, full = TRUE))  # control for GC drift

    if (method == "NEW_13") {
      o <- NEW.timed(n, d, gen.var, p = 0, n.iter = n.iter,
                     label = "NEW_13", R = 13)
    } else {
      o <- NEW.timed(n, d, gen.var, p = 0, n.iter = n.iter,
                     label = "NEW_25", R = 25)
    }
    cat(sprintf("  [n=%d %s] rep %d  t_total=%.3fs  t_cov=%.3fs  t_boot=%.3fs  q=%.2f\n",
                n, method, rep, o$t_total, o$t_cov, o$t_boot, o$q_mean))
    rows[[length(rows) + 1L]] <- data.frame(
      rep = rep, n = n, method = method,
      t_kern = o$t_kern, t_cov = o$t_cov, t_sel = o$t_sel,
      t_boot = o$t_boot, t_stat = o$t_stat, t_total = o$t_total,
      q_mean = o$q_mean
    )
    saveRDS(rows, file = "verify_intermediate.rds")
  }
}
stopCluster(cl)

df <- do.call(rbind, rows)
write.csv(df, "verify_anomaly.csv", row.names = FALSE)

cat("\n\n=========== summary (mean +/- sd across reps) ===========\n")
agg <- aggregate(cbind(t_cov, t_boot, t_stat, t_total) ~ n + method,
                 data = df, FUN = function(x) c(mean = mean(x), sd = sd(x)))
print(agg)

# Pretty table
cat("\n\n=========== per-rep t_total (sec) ===========\n")
wide <- reshape(df[, c("rep", "n", "method", "t_total")],
                idvar = c("n", "method"), timevar = "rep", direction = "wide")
print(wide)
