## ============================================================================ #
## run_all_tasks.R                                                               #
##   Convenience driver that runs all four plan1.md tasks at small smoke-test    #
##   scale.  Increase the N_TRIALS / B arguments when running the real          #
##   experiments.                                                                #
## ============================================================================ #

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
## __END_PREAMBLE__

run_one <- function(name, args = character()) {
  path <- file.path(.script_dir, "experiments", name)
  cat("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
  cat(">>", name, "\n")
  cat(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
  old_args <- commandArgs(trailingOnly = TRUE)
  ## R has no clean way to override commandArgs() for a sourced file, so just
  ## assign in the caller env: each driver reads from .cli at top of file.
  assign(".cli", args, envir = .GlobalEnv)
  source(path, local = FALSE)
}

## defaults are smoke-test sized; pass real settings via per-task CLI calls
run_one("task1_epsilon_sensitivity.R", c("30", "8",  "100", "0.5"))
run_one("task2_variance_sensitivity.R", c("30", "8",  "100", "0.5"))
run_one("task3_typeI_and_roc.R",        c("30", "12", "12",  "100", "0.5"))
run_one("task4_graph_demo.R",           c("5",  "12", "100", "0.5"))
run_one("fig2_mixture_alt.R",           c("30", "8",  "100", "100", "0.5"))

cat("\n[run_all_tasks] DONE.  See results/ for outputs.\n")
