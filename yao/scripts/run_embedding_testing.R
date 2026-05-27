#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript scripts/run_embedding_testing.R configs/example_medmnist_embedding.R", call. = FALSE)
}

config_path <- normalizePath(args[[1]], mustWork = TRUE)
project_dir <- normalizePath(file.path(dirname(config_path), ".."), mustWork = TRUE)
source(file.path(project_dir, "src", "mmmd_functions.R"))
config_env <- new.env(parent = globalenv())
source(config_path, local = config_env)

required_config <- c(
  "embedding_file", "output_dir", "sample_size", "n_reps", "B_boot", "alpha",
  "methods", "alternative_x_labels", "alternative_y_labels", "null_labels"
)
missing_config <- required_config[!vapply(required_config, exists, logical(1), envir = config_env, inherits = FALSE)]
if (length(missing_config) > 0) {
  stop("Config is missing required variables: ", paste(missing_config, collapse = ", "), call. = FALSE)
}

for (name in ls(config_env, all.names = TRUE)) {
  assign(name, get(name, envir = config_env), envir = .GlobalEnv)
}

if (!file.exists(embedding_file)) {
  embedding_file <- file.path(project_dir, embedding_file)
}
if (!grepl("^/", output_dir)) {
  output_dir <- file.path(project_dir, output_dir)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

load_embedding_table <- function(path) {
  if (grepl("\\.rds$", path, ignore.case = TRUE)) {
    obj <- readRDS(path)
    embeddings <- as.matrix(obj$embeddings)
    labels <- obj$labels
    ids <- if (!is.null(obj$ids)) obj$ids else seq_len(nrow(embeddings))
  } else {
    tab <- utils::read.csv(path, check.names = FALSE)
    if (!("label" %in% names(tab))) {
      stop("CSV embedding file must contain a 'label' column.", call. = FALSE)
    }
    labels <- tab$label
    ids <- if ("id" %in% names(tab)) tab$id else seq_len(nrow(tab))
    feature_cols <- setdiff(names(tab), c("id", "label", "split"))
    embeddings <- as.matrix(tab[, feature_cols, drop = FALSE])
    storage.mode(embeddings) <- "double"
  }
  list(embeddings = embeddings, labels = labels, ids = ids)
}

sample_rows <- function(pool_idx, n) {
  if (length(pool_idx) < n) {
    stop("Pool has fewer observations than sample_size: ", length(pool_idx), " < ", n, call. = FALSE)
  }
  sample(pool_idx, size = n, replace = FALSE)
}

run_one_experiment <- function(dat, setting, rep_id) {
  if (setting == "type1") {
    pool <- which(dat$labels %in% null_labels)
    permuted <- sample(pool, size = 2 * sample_size, replace = FALSE)
    idx_x <- permuted[seq_len(sample_size)]
    idx_y <- permuted[sample_size + seq_len(sample_size)]
  } else if (setting == "power") {
    idx_x <- sample_rows(which(dat$labels %in% alternative_x_labels), sample_size)
    idx_y <- sample_rows(which(dat$labels %in% alternative_y_labels), sample_size)
  } else {
    stop("Unknown setting: ", setting, call. = FALSE)
  }

  X <- dat$embeddings[idx_x, , drop = FALSE]
  Y <- dat$embeddings[idx_y, , drop = FALSE]

  rows <- lapply(methods, function(method) {
    out <- run_test_by_method(X, Y, method = method, B = B_boot, alpha = alpha, ridge_scale = ridge_scale)
    data.frame(
      setting = setting,
      rep_id = rep_id,
      method = method,
      reject = out$reject,
      stat = out$stat,
      cutoff = out$cutoff,
      kernel_count = if (!is.null(out$kernel_count)) out$kernel_count else 1L,
      cond_sigma = if (!is.null(out$cond_sigma)) out$cond_sigma else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

ridge_scale <- if (exists("ridge_scale", inherits = FALSE)) ridge_scale else 1e-5
seed <- if (exists("seed", inherits = FALSE)) seed else 20260526L
set.seed(seed)

dat <- load_embedding_table(embedding_file)
message(sprintf("Loaded embeddings: n=%d, d=%d", nrow(dat$embeddings), ncol(dat$embeddings)))
message(sprintf("Methods: %s", paste(methods, collapse = ", ")))

all_rows <- vector("list", 2 * n_reps)
row_id <- 1L
for (rep_id in seq_len(n_reps)) {
  if (rep_id %% max(1L, floor(n_reps / 10)) == 0) {
    message(sprintf("Replication %d/%d", rep_id, n_reps))
  }
  all_rows[[row_id]] <- run_one_experiment(dat, "type1", rep_id)
  row_id <- row_id + 1L
  all_rows[[row_id]] <- run_one_experiment(dat, "power", rep_id)
  row_id <- row_id + 1L
}

results <- do.call(rbind, all_rows)
rate <- stats::aggregate(reject ~ setting + method, data = results, FUN = mean)
se <- stats::aggregate(reject ~ setting + method, data = results, FUN = function(x) {
  sqrt(stats::var(x) / length(x))
})
names(rate)[names(rate) == "reject"] <- "reject_rate"
names(se)[names(se) == "reject"] <- "se"
summary <- merge(rate, se, by = c("setting", "method"), all = TRUE)

utils::write.csv(results, file.path(output_dir, "embedding_mmmd_replicates.csv"), row.names = FALSE)
utils::write.csv(summary, file.path(output_dir, "embedding_mmmd_summary.csv"), row.names = FALSE)

print(summary)
message("Wrote results to: ", normalizePath(output_dir, mustWork = FALSE))
