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

require(tidyverse)

PowerTime <- read_csv("PCombOverSample.csv")[,2:6]
colnames(PowerTime) <- c("Sample Size", "Bonferroni", "HM", "Bonferroni and GM",
                         "MMMD")
d <- (PowerTime)%>%pull(1)

pdf(file="ScaleNormalSample.pdf")

plot(d, PowerTime$Bonferroni, type='b', col=1, pch=1, lwd=1.5, ylim=c(0, 1), xlab="Dimension", 
     ylab='Power', main="Gaussian Location")
points(d, PowerTime$HM, type='b', col=2, pch=2, lwd=1.5)
points(d, PowerTime$`Bonferroni and GM`, type='b', col=3, pch=3, lwd=1.5)
points(d, PowerTime$MMMD, type='b', col=4, pch=4, lwd=1.5)

legend("topleft", c("Bonferroni","Harmonic Mean","Bonferroni and Geometric Mean",
                    "Gauss MMMD"), 
       bg='transparent', col=c(1,2,3,4), pch = c(1, 2, 3, 4))



dev.off()
