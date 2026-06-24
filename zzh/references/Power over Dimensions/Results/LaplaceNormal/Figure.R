## __SELF_LOCATING_PREAMBLE__
.script_dir <- tryCatch({
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable() &&
      nzchar(rstudioapi::getActiveDocumentContext()$path)) {
    dirname(rstudioapi::getActiveDocumentContext()$path)
  } else {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
    if (length(file_arg) > 0) {
      dirname(normalizePath(file_arg[1]))
    } else if (!is.null(sys.frame(1)$ofile)) {
      dirname(normalizePath(sys.frame(1)$ofile))
    } else {
      getwd()
    }
  }
}, error = function(e) getwd())
setwd(.script_dir)
## __END_PREAMBLE__




GEXP<-read.csv("MultiPower-GEXP.csv")
LAPMultiple<-read.csv("MultiPower-LAP.csv")
Mixed<-read.csv("MultiPower-MIXED.csv") 
FR<-read.csv("Power-FR.csv")
GAUSSSingle<-read.csv("SinglePower-GAUSS.csv")
LAPLACESingle<-read.csv("SinglePower-LAP.csv")

d=GEXP[,2]



pdf(file="LaplaceNormal.pdf")

plot(d, apply(GEXP[,-c(1,2)], 1, mean), type='b', col=1, pch=1, lwd=1.5, ylim=c(0, 0.8), xlab="Dimension", 
ylab='Power', main="Gaussian and Laplace Mixture")
points(d, apply(LAPMultiple[,-c(1,2)], 1, mean), type='b', col=2, pch=2, lwd=1.5)
points(d, apply(Mixed[,-c(1,2)], 1, mean), type='b', col=3, pch=3, lwd=1.5)
points(d, apply(FR[,-c(1,2)], 1, mean), type='b', col=4, pch=4, lwd=1.5)
points(d, apply(GAUSSSingle[,-c(1,2)], 1, mean), type='b', col=5, pch=5, lwd=1.5)
points(d, apply(LAPLACESingle[,-c(1,2)], 1, mean), type='b', col=6, pch=6, lwd=1.5)


legend("topleft", c("Gauss MMMD", "LAP MMMD", "Mixed MMMD", "FR", "Gauss MMD", "LAP MMD"), col=c(1,2,3,4,5,6),
       pch = c(1, 2, 3, 4, 5, 6), bg = 'gray90')



dev.off()

