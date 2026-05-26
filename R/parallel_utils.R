## ============================================================================ #
## parallel_utils.R                                                              #
##   Shared parallel-backend setup for the MMMD extension experiments.           #
##   Conforms to the performance guardrails of plan1.md section 4.               #
## ============================================================================ #

#' Start a fork/PSOCK parallel cluster sized to (detectCores() - 1).
#'
#' Returns the cluster handle.  Caller is responsible for stopping it
#' (or using `with_parallel()` which handles cleanup automatically).
mmmd_start_cluster <- function(reserve_cores = 1L, type = NULL,
                               source_load = TRUE) {
  if (!requireNamespace("parallel", quietly = TRUE))   stop("install.packages('parallel')")
  if (!requireNamespace("doParallel", quietly = TRUE)) stop("install.packages('doParallel')")
  if (!requireNamespace("foreach", quietly = TRUE))    stop("install.packages('foreach')")

  total_cores <- parallel::detectCores(logical = TRUE)
  if (is.na(total_cores) || total_cores < 2L) total_cores <- 2L
  n_workers <- max(1L, total_cores - reserve_cores)

  if (is.null(type)) {
    type <- if (.Platform$OS.type == "windows") "PSOCK" else "FORK"
  }

  cl <- parallel::makeCluster(n_workers, type = type)
  doParallel::registerDoParallel(cl)
  attr(cl, "n_workers") <- n_workers

  ## On PSOCK workers (Windows) the library is not inherited.  Re-source
  ## R/load.R on each worker so the experiment closures can rely on every
  ## library function being present in the worker's .GlobalEnv.
  if (isTRUE(source_load) && type == "PSOCK") {
    root <- getOption("mmmd.root", default = NULL)
    if (is.null(root)) root <- getwd()
    load_path <- file.path(root, "R", "load.R")
    if (file.exists(load_path)) {
      parallel::clusterCall(cl, function(p) {
        source(p, local = FALSE)
        invisible(NULL)
      }, load_path)
    }
  }
  cl
}

mmmd_stop_cluster <- function(cl) {
  if (!is.null(cl)) try(parallel::stopCluster(cl), silent = TRUE)
  try(foreach::registerDoSEQ(), silent = TRUE)
  invisible(NULL)
}

#' Run a body with a temporary parallel cluster.  Cluster is stopped no matter
#' what (errors propagate).
with_parallel <- function(expr, reserve_cores = 1L, type = NULL) {
  cl <- mmmd_start_cluster(reserve_cores = reserve_cores, type = type)
  on.exit(mmmd_stop_cluster(cl), add = TRUE)
  force(expr)
}

#' Convenience wrapper around `foreach %dopar%` that prefers parallel
#' execution but falls back to a sequential `lapply` if no backend is
#' registered.  This keeps the experiment scripts portable.
#'
#' Note about Windows PSOCK clusters: workers do NOT inherit the parent
#' environment, so any function/object referenced by `FUN` must be either
#' (a) a top-level binding in `.GlobalEnv`, or (b) listed in `.export`.
#' This implementation snapshots the *names* of `FUN`'s closure environment
#' and forwards them on top of the user-supplied `.export`.
mmmd_foreach <- function(seq, FUN, ..., .packages = NULL, .export = NULL,
                         .combine = NULL) {
  if (!requireNamespace("foreach", quietly = TRUE)) {
    out <- lapply(seq, FUN, ...)
    if (!is.null(.combine)) {
      if (identical(.combine, "rbind")) return(do.call(rbind, out))
      if (identical(.combine, "cbind")) return(do.call(cbind, out))
      if (identical(.combine, "c"))     return(do.call(c, out))
    }
    return(out)
  }

  has_backend <- foreach::getDoParRegistered() &&
    foreach::getDoParWorkers() > 1L

  ## Auto-export FUN's enclosing environment so PSOCK workers can resolve
  ## names FUN closes over.  We attach the closure env as a `.GlobalEnv`
  ## sibling on each worker via .export of all visible names.
  fun_env <- environment(FUN)
  auto_names <- if (is.environment(fun_env) &&
                    !identical(fun_env, .GlobalEnv) &&
                    !identical(fun_env, baseenv()) &&
                    !identical(fun_env, emptyenv())) {
    ls(envir = fun_env, all.names = TRUE)
  } else character()
  exp <- unique(c(.export, auto_names))

  i <- NULL
  fe <- foreach::foreach(i = seq, .packages = .packages, .export = exp,
                         .combine = if (is.null(.combine)) "c" else .combine)
  if (has_backend) {
    foreach::`%dopar%`(fe, FUN(i, ...))
  } else {
    foreach::`%do%`(fe, FUN(i, ...))
  }
}

#' Snapshot a list of names from an environment into the global env of every
#' worker in `cl`.  Use before issuing a `%dopar%` that depends on values
#' from a non-global env (PSOCK workers can't see the parent's frame).
mmmd_export_to_workers <- function(cl, names_vec, envir) {
  if (is.null(cl)) return(invisible(NULL))
  for (nm in names_vec) {
    if (exists(nm, envir = envir, inherits = TRUE)) {
      val <- get(nm, envir = envir, inherits = TRUE)
      parallel::clusterCall(cl, function(n, v) {
        assign(n, v, envir = .GlobalEnv); invisible(NULL)
      }, nm, val)
    }
  }
  invisible(NULL)
}
