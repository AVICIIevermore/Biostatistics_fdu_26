#!/usr/bin/env Rscript

## Plot graph-MMMD supplement results against existing final-FC Gaussian5
## summaries when those reference CSVs are available locally.

parse_cli <- function(args) {
  out <- list()
  for (arg in args) {
    if (grepl("^--[^=]+=", arg)) {
      key <- sub("^--([^=]+)=.*$", "\\1", arg)
      val <- sub("^--[^=]+=", "", arg)
      out[[gsub("-", "_", key)]] <- val
    }
  }
  out
}

to_abs <- function(path, root) {
  if (grepl("^[A-Za-z]:|^/", path)) path else file.path(root, path)
}

repo_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
args <- parse_cli(commandArgs(trailingOnly = TRUE))

graph_summary <- to_abs(
  if (!is.null(args$graph_summary)) args$graph_summary else "results_data/graph_mmmd_mnist_cnn_summary.csv",
  repo_root
)
output_dir <- to_abs(if (!is.null(args$output_dir)) args$output_dir else "results_data", repo_root)
existing_power <- to_abs(
  if (!is.null(args$existing_power)) {
    args$existing_power
  } else {
    "dechao_reproduction/faster_mnist_testset_paired_raw_vs_final_gaussian5_power/Results/mnist_cnn_power_summary.csv"
  },
  repo_root
)
existing_type1 <- to_abs(
  if (!is.null(args$existing_type1)) {
    args$existing_type1
  } else {
    "dechao_reproduction/faster_mnist_cnn_type1_123_vs_123_gaussian/Results/mnist_cnn_type1_summary.csv"
  },
  repo_root
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(graph_summary)) {
  stop("Missing graph-MMMD summary: ", graph_summary, call. = FALSE)
}

graph <- utils::read.csv(graph_summary, check.names = FALSE)
required_graph_cols <- c("task", "method", "sample_size", "reject_rate", "se")
missing_graph_cols <- setdiff(required_graph_cols, names(graph))
if (length(missing_graph_cols) > 0L) {
  stop("Graph summary missing columns: ", paste(missing_graph_cols, collapse = ", "), call. = FALSE)
}
graph_plot <- data.frame(
  task = graph$task,
  method_label = paste0("graph-MMMD n=", graph$sample_size),
  sample_size = graph$sample_size,
  reject_rate = graph$reject_rate,
  se = graph$se,
  source = "graph_mmmd",
  stringsAsFactors = FALSE
)

reference_rows <- list()

if (file.exists(existing_power)) {
  power <- utils::read.csv(existing_power, check.names = FALSE)
  if (all(c("noise_sigma", "method", "reject_rate") %in% names(power))) {
    row <- power[power$method == "final_embedding_gaussian5" & power$noise_sigma == 0, , drop = FALSE]
    if (nrow(row) > 0L) {
      reference_rows[[length(reference_rows) + 1L]] <- data.frame(
        task = "set123_vs_set128",
        method_label = "existing final-FC Gaussian5 n=100",
        sample_size = 100L,
        reject_rate = row$reject_rate[[1]],
        se = NA_real_,
        source = "existing_gaussian5",
        stringsAsFactors = FALSE
      )
    }
  }
} else {
  message("Reference power CSV not found, skipping: ", existing_power)
}

if (file.exists(existing_type1)) {
  type1 <- utils::read.csv(existing_type1, check.names = FALSE)
  if (all(c("noise_sigma", "method", "reject_rate") %in% names(type1))) {
    row <- type1[type1$method == "final_embedding_gaussian5" & type1$noise_sigma == 0, , drop = FALSE]
    if (nrow(row) > 0L) {
      reference_rows[[length(reference_rows) + 1L]] <- data.frame(
        task = "set123_vs_set123",
        method_label = "existing final-FC Gaussian5 n=100",
        sample_size = 100L,
        reject_rate = row$reject_rate[[1]],
        se = NA_real_,
        source = "existing_gaussian5",
        stringsAsFactors = FALSE
      )
    }
  }
} else {
  message("Reference Type-I CSV not found, skipping: ", existing_type1)
}

plot_dat <- if (length(reference_rows) > 0L) {
  rbind(graph_plot, do.call(rbind, reference_rows))
} else {
  graph_plot
}
plot_dat$task <- factor(plot_dat$task, levels = c("set123_vs_set123", "set123_vs_set128"))
plot_dat <- plot_dat[order(plot_dat$task, plot_dat$method_label), ]

png_path <- file.path(output_dir, "graph_mmmd_vs_existing_mmmd.png")
grDevices::png(png_path, width = 1800, height = 1050, res = 180)
old_par <- graphics::par(mar = c(6.5, 5.2, 4.2, 1.2))

tasks <- levels(plot_dat$task)
x_base <- seq_along(tasks)
labels <- unique(plot_dat$method_label)
offsets <- seq(-0.24, 0.24, length.out = length(labels))
names(offsets) <- labels
cols <- setNames(c("#1F77B4", "#2CA02C", "#D62728", "#9467BD", "#8C564B")[seq_along(labels)], labels)
pchs <- setNames(c(16, 17, 15, 18, 8)[seq_along(labels)], labels)

graphics::plot(
  NA,
  xlim = c(0.5, length(tasks) + 0.5),
  ylim = c(0, 1),
  xaxt = "n",
  xlab = "",
  ylab = "Rejection rate",
  main = "MNIST-CNN final-FC: graph-MMMD supplement vs existing Gaussian5 reference"
)
graphics::axis(1, at = x_base, labels = c("Null: {1,2,3} vs {1,2,3}", "Power: {1,2,3} vs {1,2,8}"), las = 2)
graphics::abline(h = 0.05, lty = 2, col = "gray55")
graphics::grid(nx = NA, ny = NULL, col = "gray88")

for (label in labels) {
  dd <- plot_dat[plot_dat$method_label == label, , drop = FALSE]
  x <- match(as.character(dd$task), tasks) + offsets[[label]]
  graphics::points(x, dd$reject_rate, pch = pchs[[label]], col = cols[[label]], cex = 1.4)
  yy0 <- pmax(0, dd$reject_rate - 1.96 * dd$se)
  yy1 <- pmin(1, dd$reject_rate + 1.96 * dd$se)
  has_se <- is.finite(dd$se) & yy1 > yy0
  if (any(has_se)) {
    graphics::arrows(x[has_se], yy0[has_se], x[has_se], yy1[has_se], angle = 90, code = 3, length = 0.04, col = cols[[label]])
  }
}
graphics::legend("topright", legend = labels, col = cols[labels], pch = pchs[labels], bty = "n", cex = 0.9)
graphics::mtext("Existing Gaussian5 points use committed n=100, noise_sigma=0 summaries when available.", side = 3, line = 0.3, cex = 0.82)
graphics::par(old_par)
grDevices::dev.off()

message("Wrote comparison plot to: ", normalizePath(png_path, winslash = "/", mustWork = FALSE))
