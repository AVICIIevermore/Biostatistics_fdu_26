
root <- "/home/dechao/kernel_two-sample"
out_dir <- file.path(root, "dechao_reproduction", "faster_mnist_cnn_main_power_summary", "Results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sources <- data.frame(
  source_dir = c(
    "faster_mnist_testset_paired_raw_vs_final_gaussian5_power",
    "faster_mnist_cnn_layer1_gaussian5_power",
    "faster_mnist_cnn_layer1_pool2x2_gaussian5_power",
    "faster_mnist_cnn_layer2_gaussian5_power",
    "faster_mnist_cnn_multilayer_single_gaussian_power",
    "faster_mnist_cnn_multilayer_gaussian15_power"
  ),
  stringsAsFactors = FALSE
)

read_result <- function(source_dir, filename) {
  path <- file.path(root, "dechao_reproduction", source_dir, "Results", filename)
  if (!file.exists(path)) stop("missing file: ", path)
  df <- read.csv(path, stringsAsFactors = FALSE)
  df$source_dir <- source_dir
  df
}

power <- do.call(rbind, lapply(sources$source_dir, read_result, filename = "mnist_cnn_power_results.csv"))
sigma <- do.call(rbind, lapply(sources$source_dir, read_result, filename = "mnist_cnn_sigma_diagnostics.csv"))

method_order <- c(
  "raw_pixel_gaussian5",
  "layer1_gaussian5",
  "layer1_pool2x2_gaussian5",
  "layer2_gaussian5",
  "final_embedding_gaussian5",
  "multilayer_single_gaussian",
  "multilayer_gaussian15"
)
method_labels <- c(
  raw_pixel_gaussian5 = "Raw Pixel Gaussian-5 (784-d)",
  layer1_gaussian5 = "Layer1 GAP Gaussian-5 (32-d)",
  layer1_pool2x2_gaussian5 = "Layer1 2x2 Gaussian-5 (128-d)",
  layer2_gaussian5 = "Layer2 GAP Gaussian-5 (64-d)",
  final_embedding_gaussian5 = "Final FC Gaussian-5 (128-d)",
  multilayer_single_gaussian = "Multilayer Single Gaussian (3 comps)",
  multilayer_gaussian15 = "Multilayer Gaussian-15 (15 comps)"
)
method_cols <- c(
  raw_pixel_gaussian5 = "#000000",
  layer1_gaussian5 = "#0072B2",
  layer1_pool2x2_gaussian5 = "#D55E00",
  layer2_gaussian5 = "#009E73",
  final_embedding_gaussian5 = "#CC79A7",
  multilayer_single_gaussian = "#E69F00",
  multilayer_gaussian15 = "#56B4E9"
)
method_pch <- c(
  raw_pixel_gaussian5 = 16,
  layer1_gaussian5 = 17,
  layer1_pool2x2_gaussian5 = 15,
  layer2_gaussian5 = 18,
  final_embedding_gaussian5 = 19,
  multilayer_single_gaussian = 8,
  multilayer_gaussian15 = 3
)

power$method <- factor(power$method, levels = method_order)
power <- power[!is.na(power$method), ]

summary_list <- aggregate(reject_rate ~ method + noise_sigma, power, function(x) c(mean = mean(x), se = sd(x) / sqrt(length(x)), n_rep = length(x)))
power_summary <- data.frame(
  method = as.character(summary_list$method),
  method_label = unname(method_labels[as.character(summary_list$method)]),
  noise_sigma = summary_list$noise_sigma,
  power_mean = summary_list$reject_rate[, "mean"],
  power_se = summary_list$reject_rate[, "se"],
  n_rep = summary_list$reject_rate[, "n_rep"],
  stringsAsFactors = FALSE
)
power_summary <- power_summary[order(match(power_summary$method, method_order), power_summary$noise_sigma), ]
write.csv(power_summary, file.path(out_dir, "mnist_cnn_main_power_summary.csv"), row.names = FALSE)
write.csv(power, file.path(out_dir, "mnist_cnn_main_power_all_outer_iters.csv"), row.names = FALSE)

sigma$method <- factor(sigma$method, levels = method_order)
sigma <- sigma[!is.na(sigma$method), ]
sigma_summary <- aggregate(cbind(cond_sigma_hat, cond_sigma_reg, lambda) ~ method + noise_sigma, sigma, median, na.rm = TRUE)
sigma_summary$method_label <- unname(method_labels[as.character(sigma_summary$method)])
sigma_summary <- sigma_summary[order(match(as.character(sigma_summary$method), method_order), sigma_summary$noise_sigma), ]
write.csv(sigma_summary, file.path(out_dir, "mnist_cnn_main_sigma_summary.csv"), row.names = FALSE)
write.csv(sigma, file.path(out_dir, "mnist_cnn_main_sigma_all_inner_iters.csv"), row.names = FALSE)

png(file.path(out_dir, "mnist_cnn_main_power_comparison.png"), width = 1500, height = 950, res = 150)
par(mar = c(5, 5, 4, 16), xpd = TRUE)
plot(NA, xlim = range(power_summary$noise_sigma), ylim = c(0, 1), xlab = "Noise sigma", ylab = "Power / rejection rate", main = "MNIST CNN Embedding MMMD Power Comparison")
grid(col = "gray85")
for (m in method_order) {
  dd <- power_summary[power_summary$method == m, ]
  if (nrow(dd) == 0) next
  lines(dd$noise_sigma, dd$power_mean, col = method_cols[m], lwd = 2.3)
  points(dd$noise_sigma, dd$power_mean, col = method_cols[m], pch = method_pch[m], cex = 1.1)
}
legend("right", inset = c(-0.48, 0), legend = method_labels[method_order], col = method_cols[method_order], pch = method_pch[method_order], lwd = 2.3, bty = "n", cex = 0.85)
dev.off()

plot_condition <- function(column, filename, title, ylab) {
  png(file.path(out_dir, filename), width = 1500, height = 950, res = 150)
  par(mar = c(5, 5, 4, 16), xpd = TRUE)
  vals <- sigma_summary[[column]]
  plot(NA, xlim = range(sigma_summary$noise_sigma), ylim = range(log10(vals[is.finite(vals) & vals > 0]), na.rm = TRUE), xlab = "Noise sigma", ylab = ylab, main = title)
  grid(col = "gray85")
  for (m in method_order) {
    dd <- sigma_summary[sigma_summary$method == m, ]
    if (nrow(dd) == 0) next
    yy <- log10(dd[[column]])
    lines(dd$noise_sigma, yy, col = method_cols[m], lwd = 2.3)
    points(dd$noise_sigma, yy, col = method_cols[m], pch = method_pch[m], cex = 1.1)
  }
  legend("right", inset = c(-0.48, 0), legend = method_labels[method_order], col = method_cols[method_order], pch = method_pch[method_order], lwd = 2.3, bty = "n", cex = 0.85)
  dev.off()
}
plot_condition("cond_sigma_hat", "mnist_cnn_main_cond_sigma_hat_log10.png", "MNIST CNN MMMD Sigma_hat Condition Number", "log10 median cond(Sigma_hat)")
plot_condition("cond_sigma_reg", "mnist_cnn_main_cond_sigma_reg_log10.png", "MNIST CNN MMMD Regularized Sigma Condition Number", "log10 median cond(Sigma_reg)")

print(power_summary)
print(sigma_summary)
