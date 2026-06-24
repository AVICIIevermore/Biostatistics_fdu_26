df <- read.csv("D:/Fudan/GitHub repositories/MMMD-boost-kernel-two-sample/Reproduction Time Comparison with New Method/TimePower-all.csv")
df.avg <- aggregate(cbind(power, q_mean, t_kern, t_cov, t_sel, t_boot, t_stat, t_total) ~ n + method, data=df, FUN=mean)
df.sd  <- aggregate(cbind(t_total, t_cov) ~ n + method, data=df, FUN=sd)

cat("=== n=700 averages ===\n")
sub700 <- df.avg[df.avg$n==700,]
print(sub700[,c("method","power","q_mean","t_cov","t_total")])

cat("\n=== n=700 SDs ===\n")
sub700sd <- df.sd[df.sd$n==700,]
print(sub700sd)

cat("\n=== all n, t_cov ===\n")
print(reshape(df.avg[,c("n","method","t_cov")], idvar="n", timevar="method", direction="wide"))

cat("\n=== all n, t_total ===\n")
print(reshape(df.avg[,c("n","method","t_total")], idvar="n", timevar="method", direction="wide"))

cat("\n=== all n, power ===\n")
print(reshape(df.avg[,c("n","method","power")], idvar="n", timevar="method", direction="wide"))
