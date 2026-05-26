################################################################################
# Figure.R - reads the three Power-*.csv files and produces the comparison PDF.
# Run AFTER Body.R has finished.
################################################################################

gauss <- read.csv("Power-GAUSS_single.csv")
gexp  <- read.csv("Power-GEXP_MMMD.csv")
nu    <- read.csv("Power-NEW_MMMD.csv")

n.seq    <- gauss$n
mean.row <- function(df) rowMeans(df[, -1, drop = FALSE])
se.row   <- function(df) {
  M <- as.matrix(df[, -1, drop = FALSE])
  apply(M, 1, sd) / sqrt(ncol(M))
}

g.mean <- mean.row(gauss); g.se <- se.row(gauss)
e.mean <- mean.row(gexp);  e.se <- se.row(gexp)
n.mean <- mean.row(nu);    n.se <- se.row(nu)

pdf("Figure1b_with_NewMethod.pdf", width = 6.5, height = 6.0)
par(mar = c(4.5, 4.5, 3, 1))

plot(n.seq, e.mean, type = "b", col = 1, pch = 1, lwd = 2,
     ylim = c(0, 1), xlab = "Sample size n", ylab = "Power",
     main = "Figure 1(b) reproduction (Gaussian Scale, d = 2)\nwith new method")

points(n.seq, n.mean, type = "b", col = 2, pch = 2, lwd = 2)
points(n.seq, g.mean, type = "b", col = 3, pch = 3, lwd = 2)

# +/- 1 SE error bars
arrows(n.seq, e.mean - e.se, n.seq, e.mean + e.se, length = 0.03, angle = 90, code = 3, col = 1)
arrows(n.seq, n.mean - n.se, n.seq, n.mean + n.se, length = 0.03, angle = 90, code = 3, col = 2)
arrows(n.seq, g.mean - g.se, n.seq, g.mean + g.se, length = 0.03, angle = 90, code = 3, col = 3)

abline(h = 0.05, lty = 3, col = "gray60")        # nominal 5% line

legend("bottomright",
       legend = c("GEXP MMMD  (5 Gaussian, geometric grid)",
                  "NEW  MMMD  (13 Gaussian arithmetic + pivoted Cholesky)",
                  "GAUSS single MMD (median bandwidth)"),
       col = c(1, 2, 3), pch = c(1, 2, 3), lwd = 2, bg = "gray95", cex = 0.85)

dev.off()

cat("Wrote Figure1b_with_NewMethod.pdf\n")
