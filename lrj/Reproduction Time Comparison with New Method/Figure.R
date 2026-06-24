################################################################################
# Figure.R - plot wall-clock-time curves and power curves from TimePower-all.csv
################################################################################

df <- read.csv("TimePower-all.csv")
# Average over reps in case n.rep > 1
df.avg <- aggregate(cbind(power, q_mean, t_kern, t_cov, t_sel, t_boot, t_stat, t_total)
                    ~ n + method, data = df, FUN = mean)

methods <- c("GEXP_5", "GEXP_rich_13", "NEW_13", "NEW_25")
cols    <- c("black", "blue",         "red",    "darkorange")
pchs    <- c(1,        2,             3,        4)
labels  <- c("GEXP (R=5, paper baseline)",
             "GEXP-rich (R=13, naive cov)",
             "NEW (R=13, fast cov + pivoted Cholesky)",
             "NEW (R=25, fast cov + pivoted Cholesky)")

get <- function(m, col) df.avg[df.avg$method == m, col]

#############################
#   FIGURE 1: total time    #
#############################
pdf("TimeComparison_total.pdf", width = 6.5, height = 5.5)
par(mar = c(4.5, 4.5, 3, 1))

ylim <- range(df.avg$t_total) * c(0.7, 1.3)
plot(df.avg$n, df.avg$t_total, type = "n", log = "y",
     ylim = ylim,
     xlab = "Sample size n", ylab = "Mean wall-clock time per iteration (sec, log scale)",
     main = "End-to-end time per MC iteration (d = 5, n.iter = 200)")
for (i in seq_along(methods)) {
  ns <- get(methods[i], "n"); ts <- get(methods[i], "t_total")
  lines (ns, ts, col = cols[i], lwd = 2)
  points(ns, ts, col = cols[i], pch = pchs[i], lwd = 2)
}
legend("topleft", legend = labels, col = cols, pch = pchs, lwd = 2, bg = "gray95", cex = 0.8)
dev.off()

#############################
#   FIGURE 2: stacked bars  #
#############################
# For each method, a separate bar plot showing stage breakdown vs n
pdf("TimeComparison_stages.pdf", width = 8, height = 6.5)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

stage.cols <- c(t_kern = "lightgray",
                t_cov  = "#d62728",
                t_sel  = "#9467bd",
                t_boot = "#1f77b4",
                t_stat = "#2ca02c")
stages <- c("t_kern", "t_cov", "t_sel", "t_boot", "t_stat")

for (i in seq_along(methods)) {
  m  <- methods[i]
  ns <- get(m, "n")
  M  <- t(rbind(get(m, "t_kern"), get(m, "t_cov"),
                get(m, "t_sel"),  get(m, "t_boot"), get(m, "t_stat")))
  rownames(M) <- ns
  colnames(M) <- stages
  barplot(t(M), col = stage.cols[stages], names.arg = ns,
          xlab = "n", ylab = "sec / iter", border = NA,
          main = labels[i], las = 1)
  if (i == 1) legend("topleft", names(stage.cols), fill = stage.cols, bg = "white", cex = 0.85)
}
dev.off()

#############################
#   FIGURE 3: cov stage     #
#############################
pdf("TimeComparison_cov_stage.pdf", width = 6.5, height = 5.5)
par(mar = c(4.5, 4.5, 3, 1))
ylim <- range(df.avg$t_cov) * c(0.7, 1.3)
plot(df.avg$n, df.avg$t_cov, type = "n", log = "y",
     ylim = ylim,
     xlab = "Sample size n", ylab = "Mean Sigma-construction time (sec, log scale)",
     main = "Covariance estimation: naive est.cov vs fast.cov.gaussian.t")
for (i in seq_along(methods)) {
  ns <- get(methods[i], "n"); ts <- get(methods[i], "t_cov")
  lines (ns, ts, col = cols[i], lwd = 2)
  points(ns, ts, col = cols[i], pch = pchs[i], lwd = 2)
}
legend("topleft", legend = labels, col = cols, pch = pchs, lwd = 2, bg = "gray95", cex = 0.8)
dev.off()

#############################
#   FIGURE 4: power         #
#############################
pdf("TimeComparison_power.pdf", width = 6.5, height = 5.5)
par(mar = c(4.5, 4.5, 3, 1))
plot(df.avg$n, df.avg$power, type = "n", ylim = c(0, 1),
     xlab = "Sample size n", ylab = "Power (alpha = 0.05)",
     main = "Power: same alternative as MMMDvsMMD (d = 5, sigma.mult = 1.2)")
for (i in seq_along(methods)) {
  ns <- get(methods[i], "n"); ps <- get(methods[i], "power")
  lines (ns, ps, col = cols[i], lwd = 2)
  points(ns, ps, col = cols[i], pch = pchs[i], lwd = 2)
}
abline(h = 0.05, lty = 3, col = "gray60")
legend("bottomright", legend = labels, col = cols, pch = pchs, lwd = 2, bg = "gray95", cex = 0.8)
dev.off()

cat("Wrote:\n  TimeComparison_total.pdf\n  TimeComparison_stages.pdf\n  TimeComparison_cov_stage.pdf\n  TimeComparison_power.pdf\n")
