################################################################################
# patch_and_redraw.R
#   1. Read TimePower-all.csv (original single-shot data) and verify_anomaly.csv
#      (5 reps of the 4 suspect pairs).
#   2. Replace the timing columns of the 4 affected rows with the verification
#      means.  Keep power from the original run (verification didn't measure it).
#   3. Save the patched CSV in place, then re-run Figure.R.
################################################################################

orig <- read.csv("TimePower-all.csv")
ver  <- read.csv("verify_anomaly.csv")

cols.to.patch <- c("t_kern", "t_cov", "t_sel", "t_boot", "t_stat", "t_total", "q_mean")

# verification mean across the 5 reps for each (n, method) pair
ver.mean <- aggregate(ver[, cols.to.patch], by = list(n = ver$n, method = ver$method), FUN = mean)
ver.sd   <- aggregate(ver[, cols.to.patch], by = list(n = ver$n, method = ver$method), FUN = sd)

cat("Verification means (5 reps each):\n")
print(ver.mean)
cat("\nVerification sds:\n")
print(ver.sd)

# Patch the 4 affected rows in `orig`
patched <- orig
for (k in seq_len(nrow(ver.mean))) {
  i <- which(patched$n == ver.mean$n[k] & patched$method == ver.mean$method[k])
  if (length(i) == 1L) {
    patched[i, cols.to.patch] <- ver.mean[k, cols.to.patch]
    cat(sprintf("Patched (n=%d, %s): t_total %.2f -> %.2f (+/- %.2f)\n",
                ver.mean$n[k], ver.mean$method[k],
                orig$t_total[i], ver.mean$t_total[k], ver.sd$t_total[k]))
  }
}

write.csv(patched, "TimePower-all.csv", row.names = FALSE)
cat("\nPatched CSV written. Re-running Figure.R ...\n\n")

source("Figure.R")
