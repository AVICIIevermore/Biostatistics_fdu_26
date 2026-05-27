#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript yao/scripts/run_noisy_embedding_testing.R yao/configs/xxx.R", call. = FALSE)
}

config_path <- normalizePath(args[[1]], mustWork = TRUE)
project_dir <- normalizePath(file.path(dirname(config_path), ".."), mustWork = TRUE)
source(file.path(project_dir, "src", "mmmd_functions.R"))

config_env <- new.env(parent = globalenv())
source(config_path, local = config_env)
required_config <- c(
  "embedding_file", "output_dir", "sample_size", "n_inner", "B_boot", "alpha",
  "methods", "alternative_x_labels", "alternative_y_labels", "seed"
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

ridge_scale <- if (exists("ridge_scale", inherits = FALSE)) ridge_scale else 1e-5
scenario <- if (exists("scenario", inherits = FALSE)) scenario else "power"
null_x_labels <- if (exists("null_x_labels", inherits = FALSE)) null_x_labels else alternative_x_labels
null_y_labels <- if (exists("null_y_labels", inherits = FALSE)) null_y_labels else alternative_y_labels
n_cores <- if (exists("n_cores", inherits = FALSE)) max(1L, as.integer(n_cores)) else 1L
sample_replace <- if (exists("sample_replace", inherits = FALSE)) isTRUE(sample_replace) else FALSE
n_outer <- if (exists("n_outer", inherits = FALSE)) as.integer(n_outer) else NA_integer_
balanced_sampling <- if (exists("balanced_sampling", inherits = FALSE)) isTRUE(balanced_sampling) else FALSE
noise_subset <- if (exists("noise_subset", inherits = FALSE)) as.numeric(noise_subset) else NULL

read_npy_matrix <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con), add = TRUE)

  magic <- readBin(con, what = "raw", n = 6)
  if (!identical(magic, charToRaw("\x93NUMPY"))) {
    stop("Not a .npy file: ", path, call. = FALSE)
  }
  version <- readBin(con, what = "raw", n = 2)
  major <- as.integer(version[1])
  header_len <- if (major == 1L) {
    readBin(con, what = "integer", n = 1, size = 2, endian = "little", signed = FALSE)
  } else {
    readBin(con, what = "integer", n = 1, size = 4, endian = "little", signed = FALSE)
  }
  header <- rawToChar(readBin(con, what = "raw", n = header_len))

  descr <- sub(".*'descr': '([^']+)'.*", "\\1", header)
  fortran_order <- grepl("'fortran_order': True", header, fixed = TRUE)
  shape_text <- sub(".*'shape': \\(([^\\)]*)\\).*", "\\1", header)
  dims <- as.integer(strsplit(gsub("L| ", "", shape_text), ",")[[1]])
  dims <- dims[!is.na(dims)]
  if (length(dims) != 2L) {
    stop("Expected a 2D .npy array, got shape: ", shape_text, call. = FALSE)
  }
  if (descr %in% c("<f4", "|f4")) {
    values <- readBin(con, what = "numeric", n = prod(dims), size = 4, endian = "little")
  } else if (descr %in% c("<f8", "|f8")) {
    values <- readBin(con, what = "numeric", n = prod(dims), size = 8, endian = "little")
  } else {
    stop("Unsupported .npy dtype: ", descr, call. = FALSE)
  }
  matrix(values, nrow = dims[1], ncol = dims[2], byrow = !fortran_order)
}

load_noisy_embedding_table <- function(path) {
  if (dir.exists(path)) {
    npy_path <- file.path(path, "embeddings.npy")
    meta_path <- file.path(path, "metadata.csv")
    if (!file.exists(npy_path) || !file.exists(meta_path)) {
      stop("Embedding directory must contain embeddings.npy and metadata.csv.", call. = FALSE)
    }
    return(list(
      meta = utils::read.csv(meta_path, check.names = FALSE),
      embeddings = read_npy_matrix(npy_path)
    ))
  }

  tab <- utils::read.csv(path, check.names = FALSE)
  required_cols <- c("id", "label", "outer_iter", "noise_sigma")
  missing_cols <- setdiff(required_cols, names(tab))
  if (length(missing_cols) > 0) {
    stop("Embedding CSV missing required columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }
  feature_cols <- grep("^feat_", names(tab), value = TRUE)
  list(
    meta = tab[, setdiff(names(tab), feature_cols), drop = FALSE],
    embeddings = as.matrix(tab[, feature_cols, drop = FALSE])
  )
}

sample_matrix <- function(pool_idx, draws, sample_size, seed_offset) {
  set.seed(seed + seed_offset)
  t(replicate(draws, sample(pool_idx, size = sample_size, replace = sample_replace)))
}

sample_balanced_matrix <- function(meta_labels, target_labels, draws, sample_size, seed_offset) {
  labels_unique <- sort(unique(as.integer(target_labels)))
  n_labels <- length(labels_unique)
  if (sample_size %% n_labels != 0L) {
    stop("sample_size must be divisible by the number of target labels for balanced sampling.", call. = FALSE)
  }
  per_label <- sample_size %/% n_labels
  pools <- lapply(labels_unique, function(lbl) which(meta_labels == lbl))
  for (i in seq_along(pools)) {
    if (!sample_replace && length(pools[[i]]) < per_label) {
      stop(sprintf(
        "Label %s has only %d observations, fewer than per-label sample size %d.",
        labels_unique[[i]], length(pools[[i]]), per_label
      ), call. = FALSE)
    }
  }

  set.seed(seed + seed_offset)
  out <- matrix(NA_integer_, nrow = draws, ncol = sample_size)
  for (i in seq_len(draws)) {
    draw_idx <- integer(0)
    for (j in seq_along(labels_unique)) {
      draw_idx <- c(draw_idx, sample(pools[[j]], size = per_label, replace = sample_replace))
    }
    out[i, ] <- sample(draw_idx, size = length(draw_idx), replace = FALSE)
  }
  out
}

sample_type1_pairs <- function(pool_size, draws, sample_size, seed_offset) {
  set.seed(seed + seed_offset)
  out <- vector("list", draws)
  for (i in seq_len(draws)) {
    if (sample_replace) {
      x_idx <- sample(seq_len(pool_size), size = sample_size, replace = TRUE)
      y_idx <- sample(seq_len(pool_size), size = sample_size, replace = TRUE)
    } else {
      chosen <- sample(seq_len(pool_size), size = 2L * sample_size, replace = FALSE)
      x_idx <- chosen[seq_len(sample_size)]
      y_idx <- chosen[sample_size + seq_len(sample_size)]
    }
    out[[i]] <- list(x = x_idx, y = y_idx)
  }
  out
}

sample_type1_pairs_balanced <- function(pool_labels, target_labels, draws, sample_size, seed_offset) {
  labels_unique <- sort(unique(as.integer(target_labels)))
  n_labels <- length(labels_unique)
  if (sample_size %% n_labels != 0L) {
    stop("sample_size must be divisible by the number of target labels for balanced type-I sampling.", call. = FALSE)
  }
  per_label <- sample_size %/% n_labels
  pools <- lapply(labels_unique, function(lbl) which(pool_labels == lbl))
  for (i in seq_along(pools)) {
    if (!sample_replace && length(pools[[i]]) < 2L * per_label) {
      stop(sprintf(
        "Label %s has only %d observations, fewer than required 2*per-label sample size %d.",
        labels_unique[[i]], length(pools[[i]]), 2L * per_label
      ), call. = FALSE)
    }
  }

  set.seed(seed + seed_offset)
  out <- vector("list", draws)
  for (i in seq_len(draws)) {
    x_idx <- integer(0)
    y_idx <- integer(0)
    for (j in seq_along(labels_unique)) {
      if (sample_replace) {
        x_idx <- c(x_idx, sample(pools[[j]], size = per_label, replace = TRUE))
        y_idx <- c(y_idx, sample(pools[[j]], size = per_label, replace = TRUE))
      } else {
        chosen <- sample(pools[[j]], size = 2L * per_label, replace = FALSE)
        x_idx <- c(x_idx, chosen[seq_len(per_label)])
        y_idx <- c(y_idx, chosen[per_label + seq_len(per_label)])
      }
    }
    out[[i]] <- list(
      x = sample(x_idx, size = length(x_idx), replace = FALSE),
      y = sample(y_idx, size = length(y_idx), replace = FALSE)
    )
  }
  out
}

run_method_for_layer <- function(
  X_pool, Y_pool, method, outer_iter, noise_sigma, shared_null_pool = FALSE,
  balanced_x_labels = NULL, balanced_y_labels = NULL, X_meta_labels = NULL, Y_meta_labels = NULL
) {
  if (!is.null(balanced_x_labels)) {
    resamp_x <- sample_balanced_matrix(
      meta_labels = X_meta_labels,
      target_labels = balanced_x_labels,
      draws = n_inner,
      sample_size = sample_size,
      seed_offset = outer_iter * 100000 + round(noise_sigma * 1000) + 11
    )
  } else {
    resamp_x <- sample_matrix(seq_len(nrow(X_pool)), n_inner, sample_size, outer_iter * 100000 + round(noise_sigma * 1000) + 11)
  }
  if (!is.null(balanced_y_labels)) {
    resamp_y <- sample_balanced_matrix(
      meta_labels = Y_meta_labels,
      target_labels = balanced_y_labels,
      draws = n_inner,
      sample_size = sample_size,
      seed_offset = outer_iter * 100000 + round(noise_sigma * 1000) + 29
    )
  } else {
    resamp_y <- sample_matrix(seq_len(nrow(Y_pool)), n_inner, sample_size, outer_iter * 100000 + round(noise_sigma * 1000) + 29)
  }
  if (shared_null_pool) {
    if (!is.null(balanced_x_labels)) {
      null_pairs <- sample_type1_pairs_balanced(
        pool_labels = X_meta_labels,
        target_labels = balanced_x_labels,
        draws = n_inner,
        sample_size = sample_size,
        seed_offset = outer_iter * 100000 + round(noise_sigma * 1000) + 71
      )
    } else {
      null_pairs <- sample_type1_pairs(
        pool_size = nrow(X_pool),
        draws = n_inner,
        sample_size = sample_size,
        seed_offset = outer_iter * 100000 + round(noise_sigma * 1000) + 71
      )
    }
  } else {
    null_pairs <- NULL
  }

  process_one <- function(i) {
    if (shared_null_pool) {
      X <- X_pool[null_pairs[[i]]$x, , drop = FALSE]
      Y <- X_pool[null_pairs[[i]]$y, , drop = FALSE]
    } else {
      X <- X_pool[resamp_x[i, ], , drop = FALSE]
      Y <- Y_pool[resamp_y[i, ], , drop = FALSE]
    }
    out <- run_test_by_method(X, Y, method = method, B = B_boot, alpha = alpha, ridge_scale = ridge_scale)
    as.integer(out$reject)
  }

  rejects <- if (n_cores > 1L) {
    unlist(parallel::mclapply(seq_len(n_inner), process_one, mc.cores = n_cores), use.names = FALSE)
  } else {
    vapply(seq_len(n_inner), process_one, integer(1))
  }

  data.frame(
    outer_iter = outer_iter,
    noise_sigma = noise_sigma,
    method = method,
    reject_rate = mean(rejects),
    scenario = scenario,
    stringsAsFactors = FALSE
  )
}

dat <- load_noisy_embedding_table(embedding_file)
message(sprintf("Loaded noisy embeddings: n=%d, d=%d", nrow(dat$embeddings), ncol(dat$embeddings)))

outer_pool_values <- sort(unique(dat$meta$outer_iter))
if (length(outer_pool_values) == 0L) {
  outer_pool_values <- 1L
}
if (is.na(n_outer)) {
  outer_values <- outer_pool_values
} else {
  outer_values <- seq_len(n_outer)
}
sigma_values <- sort(unique(dat$meta$noise_sigma))
if (!is.null(noise_subset)) {
  sigma_values <- sigma_values[sigma_values %in% noise_subset]
}
if (length(sigma_values) == 0L) {
  stop("No noise levels left after applying noise_subset.", call. = FALSE)
}
rows <- list()

for (outer_iter in outer_values) {
  for (noise_sigma in sigma_values) {
    outer_pool <- if (length(outer_pool_values) == 1L) outer_pool_values[[1]] else outer_iter
    idx_layer <- which(dat$meta$outer_iter == outer_pool & dat$meta$noise_sigma == noise_sigma)
    meta <- dat$meta[idx_layer, , drop = FALSE]
    emb <- dat$embeddings[idx_layer, , drop = FALSE]

    if ("group" %in% names(meta)) {
      idx_x <- which(meta$group == "X")
      idx_y <- which(meta$group == "Y")
      shared_null_pool <- FALSE
      balanced_x_labels <- NULL
      balanced_y_labels <- NULL
    } else if (scenario == "type1") {
      idx_x <- which(meta$label %in% null_x_labels)
      idx_y <- which(meta$label %in% null_y_labels)
      shared_null_pool <- identical(sort(unique(null_x_labels)), sort(unique(null_y_labels)))
      balanced_x_labels <- if (balanced_sampling) null_x_labels else NULL
      balanced_y_labels <- if (balanced_sampling) null_y_labels else NULL
    } else {
      idx_x <- which(meta$label %in% alternative_x_labels)
      idx_y <- which(meta$label %in% alternative_y_labels)
      shared_null_pool <- FALSE
      balanced_x_labels <- if (balanced_sampling) alternative_x_labels else NULL
      balanced_y_labels <- if (balanced_sampling) alternative_y_labels else NULL
    }
    X_pool <- emb[idx_x, , drop = FALSE]
    Y_pool <- emb[idx_y, , drop = FALSE]
    X_meta_labels <- meta$label[idx_x]
    Y_meta_labels <- meta$label[idx_y]
    message(sprintf(
      "outer=%s pool_outer=%s sigma=%s |X|=%d |Y|=%d replace=%s shared_null=%s balanced=%s",
      outer_iter, outer_pool, noise_sigma, nrow(X_pool), nrow(Y_pool), sample_replace, shared_null_pool, balanced_sampling
    ))

    for (method in methods) {
      rows[[length(rows) + 1L]] <- run_method_for_layer(
        X_pool, Y_pool, method, outer_iter, noise_sigma,
        shared_null_pool = shared_null_pool,
        balanced_x_labels = balanced_x_labels,
        balanced_y_labels = balanced_y_labels,
        X_meta_labels = X_meta_labels,
        Y_meta_labels = Y_meta_labels
      )
    }
  }
}

results <- do.call(rbind, rows)
summary <- stats::aggregate(reject_rate ~ noise_sigma + method + scenario, data = results, FUN = mean)
names(summary)[names(summary) == "reject_rate"] <- if (scenario == "type1") "type1_mean" else "power_mean"
se <- stats::aggregate(reject_rate ~ noise_sigma + method + scenario, data = results, FUN = function(x) sqrt(stats::var(x) / length(x)))
names(se)[names(se) == "reject_rate"] <- if (scenario == "type1") "type1_se" else "power_se"
summary <- merge(summary, se, by = c("noise_sigma", "method", "scenario"), all = TRUE)
summary$n_rep <- length(outer_values)

utils::write.csv(results, file.path(output_dir, "noisy_embedding_results.csv"), row.names = FALSE)
utils::write.csv(summary, file.path(output_dir, "noisy_embedding_summary.csv"), row.names = FALSE)

print(summary)
message("Wrote results to: ", normalizePath(output_dir, mustWork = FALSE))
