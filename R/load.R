## ============================================================================ #
## load.R                                                                        #
##   Single source point for the extension library.  Experiment scripts          #
##   typically begin with:                                                       #
##                                                                               #
##       source("R/load.R")     # from project root                              #
##       source(file.path(MMMD_ROOT, "R", "load.R"))                             #
##                                                                               #
##   The self-locating preamble keeps each experiment runnable from any CWD.    #
## ============================================================================ #

## __SELF_LOCATING_PREAMBLE__
##
## Locate this file robustly across:
##   - direct `Rscript R/load.R`         (commandArgs --file=)
##   - `source('R/load.R')` from any other script (walk frame stack
##                                                  for the deepest ofile)
##   - RStudio active document
##   - fallback: ./R relative to current working directory.
##
## Pick the candidate whose directory actually contains "parallel_utils.R"
## so we never silently load from the wrong place.
.this_dir <- local({
  candidates <- character()

  ## (a) walk frame stack, deepest ofile wins
  if (sys.nframe() > 0) {
    for (i in seq_len(sys.nframe())) {
      fr <- tryCatch(sys.frame(i), error = function(e) NULL)
      if (!is.null(fr) &&
          exists("ofile", envir = fr, inherits = FALSE)) {
        of <- get("ofile", envir = fr, inherits = FALSE)
        if (is.character(of) && length(of) == 1L && nzchar(of)) {
          candidates <- c(candidates, dirname(normalizePath(of, mustWork = FALSE)))
        }
      }
    }
  }

  ## (b) Rscript --file=
  args <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
  if (length(fa) > 0)
    candidates <- c(candidates, dirname(normalizePath(fa[1], mustWork = FALSE)))

  ## (c) RStudio
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable() &&
      nzchar(rstudioapi::getActiveDocumentContext()$path))
    candidates <- c(candidates,
                    dirname(rstudioapi::getActiveDocumentContext()$path))

  ## (d) fallback: ./R relative to CWD
  candidates <- c(candidates, file.path(getwd(), "R"), getwd())

  ok <- vapply(candidates,
               function(d) file.exists(file.path(d, "parallel_utils.R")),
               logical(1))
  if (!any(ok))
    stop("R/load.R: cannot locate sibling library files.  Tried:\n  ",
         paste(unique(candidates), collapse = "\n  "))
  candidates[which(ok)[length(which(ok))]]   # deepest valid candidate
})
## __END_PREAMBLE__

MMMD_ROOT <- normalizePath(file.path(.this_dir, ".."), winslash = "/")
options(mmmd.root = MMMD_ROOT)

# Define helper functions before sourcing sub-modules
mmmd_results_dir <- function(create = TRUE) {
  d <- file.path(MMMD_ROOT, "results")
  if (create && !dir.exists(d)) dir.create(d, recursive = TRUE)
  d
}

mmmd_data_dir <- function() file.path(MMMD_ROOT, "data")

source(file.path(.this_dir, "parallel_utils.R"))
source(file.path(.this_dir, "data_sources.R"))
source(file.path(.this_dir, "mmmd_core.R"))
source(file.path(.this_dir, "roc_utils.R"))
source(file.path(.this_dir, "graph_kernel.R"))
source(file.path(.this_dir, "bio_loader.R"))

invisible(NULL)
