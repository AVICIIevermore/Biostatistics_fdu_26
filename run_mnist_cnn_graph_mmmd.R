#!/usr/bin/env Rscript

## Run the lightweight MNIST-CNN final-FC graph-MMMD supplement.
##
## Default input:
##   results_data/mnist_final_fc128_embeddings.csv
##
## The script expects an exported embedding table with a `label` column and
## 128 numeric feature columns. Optional `id` and `split` columns are ignored.

parse_cli <- function(args) {
  out <- list()
  for (arg in args) {
    if (arg == "--smoke") {
      out$smoke <- TRUE
    } else if (grepl("^--[^=]+=", arg)) {
      key <- sub("^--([^=]+)=.*$", "\\1", arg)
      val <- sub("^--[^=]+=", "", arg)
      out[[gsub("-", "_", key)]] <- val
    }
  }
  out
}

split_numbers <- function(x, integer = FALSE) {
  vals <- strsplit(x, ",", fixed = TRUE)[[1]]
  vals <- vals[nzchar(vals)]
  if (integer) as.integer(vals) else as.numeric(vals)
}

repo_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
args <- parse_cli(commandArgs(trailingOnly = TRUE))

embedding_file <- if (!is.null(args$embedding_file)) args$embedding_file else "results_data/mnist_final_fc128_embeddings.csv"
output_dir <- if (!is.null(args$output_dir)) args$output_dir else "results_data"
sample_sizes <- if (!is.null(args$sample_sizes)) split_numbers(args$sample_sizes, integer = TRUE) else c(60L, 120L)
n_reps <- if (!is.null(args$n_reps)) as.integer(args$n_reps) else 30L
B_boot <- if (!is.null(args$B_boot)) as.integer(args$B_boot) else 200L
alpha <- if (!is.null(args$alpha)) as.numeric(args$alpha) else 0.05
seed <- if (!is.null(args$seed)) as.integer(args$seed) else 20260528L
k_nn <- if (!is.null(args$k_nn)) as.integer(args$k_nn) else 5L
t_seq <- if (!is.null(args$t_seq)) split_numbers(args$t_seq) else c(0.1, 0.5, 1, 2, 5)
ridge_scale <- if (!is.null(args$ridge_scale)) as.numeric(args$ridge_scale) else 1e-5

if (isTRUE(args$smoke)) {
  n_reps <- 3L
  B_boot <- 20L
}

if (!grepl("^[A-Za-z]:|^/", embedding_file)) {
  embedding_file <- file.path(repo_root, embedding_file)
}
if (!grepl("^[A-Za-z]:|^/", output_dir)) {
  output_dir <- file.path(repo_root, output_dir)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

source(file.path(repo_root, "R", "graph_mmmd_mnist_cnn.R"))

note_path <- file.path(output_dir, "short_method_note.md")
write_method_note <- function(status, summary_path = NULL, plot_path = NULL) {
  lines <- c(
    "# MNIST-CNN Graph-MMMD Supplement",
    "",
    paste0("- Status: ", status),
    "- Scope: graph-MMMD only; existing Gaussian5/CNN-MMMD results are referenced, not rerun.",
    "- Dataset target: MNIST-CNN reproduction task from `dechao_reproduction/`.",
    "- Feature input: final FC 128-d embeddings from `results_data/mnist_final_fc128_embeddings.csv`.",
    "- Power task: `set123_vs_set128`, labels `{1,2,3}` vs `{1,2,8}`.",
    "- Null task: `set123_vs_set123`, labels `{1,2,3}` split into two balanced samples.",
    paste0("- Parameters: sample_sizes=", paste(sample_sizes, collapse = ","),
           "; n_reps=", n_reps,
           "; B_boot=", B_boot,
           "; alpha=", alpha,
           "; k_nn=", k_nn,
           "; t_seq=", paste(t_seq, collapse = ","),
           "; seed=", seed, "."),
    "- Notes: this is a lightweight supplement and should not be treated as a full-scale estimate."
  )
  if (!is.null(summary_path)) {
    lines <- c(lines, paste0("- Graph summary: `", summary_path, "`."))
  }
  if (!is.null(plot_path)) {
    lines <- c(lines, paste0("- Comparison plot: `", plot_path, "`."))
  }
  writeLines(lines, note_path, useBytes = TRUE)
}

if (!file.exists(embedding_file)) {
  write_method_note("waiting for embedding CSV; no graph-MMMD run was executed")
  stop(
    "Missing embedding CSV: ", embedding_file, "\n",
    "Create this file with columns label + final-FC feature columns, then rerun this script.",
    call. = FALSE
  )
}

emb <- utils::read.csv(embedding_file, check.names = FALSE)
if (!("label" %in% names(emb))) {
  stop("Embedding CSV must contain a `label` column.", call. = FALSE)
}
feature_cols <- setdiff(names(emb), c("id", "label", "split"))
if (length(feature_cols) < 1L) {
  stop("Embedding CSV has no feature columns after excluding id/label/split.", call. = FALSE)
}
X_all <- as.matrix(emb[, feature_cols, drop = FALSE])
storage.mode(X_all) <- "double"
labels <- as.integer(emb$label)
if (anyNA(labels)) {
  stop("The `label` column must be integer-like.", call. = FALSE)
}

check_balanced_capacity <- function(target_labels, sample_size, null_pair = FALSE) {
  per_label <- sample_size %/% length(target_labels)
  multiplier <- if (null_pair) 2L else 1L
  for (label in target_labels) {
    available <- sum(labels == label)
    needed <- multiplier * per_label
    if (available < needed) {
      stop(sprintf(
        "Insufficient rows for label %s at sample_size=%d: need %d, found %d.",
        label, sample_size, needed, available
      ), call. = FALSE)
    }
  }
}

for (n in sample_sizes) {
  if (n %% 3L != 0L) {
    stop("All sample sizes must be divisible by 3 for balanced MNIST set sampling.", call. = FALSE)
  }
  check_balanced_capacity(c(1L, 2L, 3L), n, null_pair = TRUE)
  check_balanced_capacity(c(1L, 2L, 3L), n, null_pair = FALSE)
  check_balanced_capacity(c(1L, 2L, 8L), n, null_pair = FALSE)
}

set.seed(seed)
rows <- list()
row_id <- 1L

run_one <- function(task, sample_size) {
  if (task == "set123_vs_set123") {
    idx <- graph_mmmd_sample_balanced_null_pair(labels, c(1L, 2L, 3L), sample_size)
    X <- X_all[idx$x, , drop = FALSE]
    Y <- X_all[idx$y, , drop = FALSE]
  } else if (task == "set123_vs_set128") {
    idx_x <- graph_mmmd_sample_balanced(labels, c(1L, 2L, 3L), sample_size)
    idx_y <- graph_mmmd_sample_balanced(labels, c(1L, 2L, 8L), sample_size)
    X <- X_all[idx_x, , drop = FALSE]
    Y <- X_all[idx_y, , drop = FALSE]
  } else {
    stop("Unknown task: ", task, call. = FALSE)
  }
  graph_mmmd_test(X, Y, k_nn = k_nn, t_seq = t_seq, B = B_boot, alpha = alpha, ridge_scale = ridge_scale)
}

tasks <- c("set123_vs_set123", "set123_vs_set128")
for (sample_size in sample_sizes) {
  for (task in tasks) {
    rejects <- integer(n_reps)
    for (rep_id in seq_len(n_reps)) {
      out <- run_one(task, sample_size)
      rejects[[rep_id]] <- out$reject
      message(sprintf("task=%s sample_size=%d rep=%d/%d reject=%d", task, sample_size, rep_id, n_reps, out$reject))
    }
    rows[[row_id]] <- data.frame(
      task = task,
      method = "graph_mmmd",
      feature = "final_fc128",
      sample_size = as.integer(sample_size),
      n_reps = as.integer(n_reps),
      B_boot = as.integer(B_boot),
      alpha = alpha,
      reject_rate = mean(rejects),
      se = if (n_reps > 1L) sqrt(stats::var(rejects) / n_reps) else NA_real_,
      k_nn = as.integer(k_nn),
      t_seq = paste(t_seq, collapse = ","),
      seed = as.integer(seed),
      stringsAsFactors = FALSE
    )
    row_id <- row_id + 1L
  }
}

summary_df <- do.call(rbind, rows)
summary_path <- file.path(output_dir, "graph_mmmd_mnist_cnn_summary.csv")
utils::write.csv(summary_df, summary_path, row.names = FALSE)
write_method_note(
  "graph-MMMD summary generated",
  summary_path = "results_data/graph_mmmd_mnist_cnn_summary.csv",
  plot_path = "results_data/graph_mmmd_vs_existing_mmmd.png"
)

print(summary_df)
message("Wrote graph-MMMD summary to: ", normalizePath(summary_path, winslash = "/", mustWork = FALSE))
message("Wrote method note to: ", normalizePath(note_path, winslash = "/", mustWork = FALSE))
