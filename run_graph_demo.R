## ============================================================================ #
## run_graph_demo.R                                                              #
##                                                                               #
##  Trampoline at the project root that satisfies plan1.md section 5             #
##  ("演示 R 脚本 run_graph_demo.R").  All actual logic lives in                  #
##  experiments/task4_graph_demo.R; this file just ensures the demo can be       #
##  invoked from the repo root with no extra arguments.                          #
##                                                                               #
##    Rscript run_graph_demo.R                                                   #
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

source(file.path(.script_dir, "experiments", "task4_graph_demo.R"))
