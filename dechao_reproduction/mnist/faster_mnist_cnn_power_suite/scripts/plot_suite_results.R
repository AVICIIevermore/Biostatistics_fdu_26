get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(path) > 0) {
    return(dirname(normalizePath(path[1])))
  }
  normalizePath(getwd())
}

method_labels <- c(
  raw_pixel_gaussian5 = "Raw Pixel Gaussian-5",
  layer1_gaussian5 = "Layer1 Gaussian-5",
  layer2_gaussian5 = "Layer2 Gaussian-5",
  final_embedding_gaussian5 = "Final Embedding Gaussian-5",
  multilayer_single_gaussian = "Multilayer Single Gaussian",
  multilayer_gaussian15 = "Multilayer Gaussian-15"
)
method_cols <- c(
  raw_pixel_gaussian5 = "#1b9e77",
  layer1_gaussian5 = "#66a61e",
  layer2_gaussian5 = "#e6ab02",
  final_embedding_gaussian5 = "#d95f02",
  multilayer_single_gaussian = "#7570b3",
  multilayer_gaussian15 = "#e7298a"
)
method_pchs <- c(
  raw_pixel_gaussian5 = 16,
  layer1_gaussian5 = 15,
  layer2_gaussian5 = 18,
  final_embedding_gaussian5 = 17,
  multilayer_single_gaussian = 8,
  multilayer_gaussian15 = 3
)

read_power <- function(exp_dir) {
  path <- file.path(exp_dir, "Results", "mnist_cnn_power_results.csv")
  dat <- utils::read.csv(path, check.names = FALSE)
  aggregate(reject_rate ~ noise_sigma + method, data = dat, FUN = mean)
}

read_sigma <- function(exp_dir) {
  path <- file.path(exp_dir, "Results", "mnist_cnn_sigma_diagnostics.csv")
  dat <- utils::read.csv(path, check.names = FALSE)
  aggregate(cbind(cond_sigma_hat, cond_sigma_reg, lambda) ~ noise_sigma + method, data = dat, FUN = mean)
}

plot_power <- function(power_summary, output_path) {
  method_order <- c("raw_pixel_gaussian5", "layer1_gaussian5", "layer2_gaussian5", "final_embedding_gaussian5", "multilayer_single_gaussian", "multilayer_gaussian15")
  method_order <- method_order[method_order %in% unique(as.character(power_summary$method))]
  first <- subset(power_summary, method == method_order[1])

  grDevices::png(output_path, width = 1200, height = 900, res = 150)
  graphics::plot(first$noise_sigma, first$reject_rate, type = "b", pch = method_pchs[method_order[1]], col = method_cols[method_order[1]], lwd = 2, ylim = c(0, 1), xlab = "Noise Sigma", ylab = "Estimated Power", main = "MNIST CNN Power Suite: Test-Set Aligned Comparison")
  if (length(method_order) > 1) {
    for (method in method_order[-1]) {
      dat <- subset(power_summary, method == method)
      graphics::lines(dat$noise_sigma, dat$reject_rate, type = "b", pch = method_pchs[method], col = method_cols[method], lwd = 2)
    }
  }
  graphics::legend("topright", legend = method_labels[method_order], col = method_cols[method_order], pch = method_pchs[method_order], lty = 1, lwd = 2, cex = 0.85, bg = "white")
  grDevices::dev.off()
}

plot_sigma <- function(sigma_summary, output_path) {
  method_order <- c("raw_pixel_gaussian5", "layer1_gaussian5", "layer2_gaussian5", "final_embedding_gaussian5", "multilayer_single_gaussian", "multilayer_gaussian15")
  method_order <- method_order[method_order %in% unique(as.character(sigma_summary$method))]
  first <- subset(sigma_summary, method == method_order[1])

  grDevices::png(output_path, width = 1400, height = 900, res = 150)
  graphics::par(mfrow = c(1, 2))
  graphics::plot(first$noise_sigma, first$cond_sigma_hat, type = "b", col = method_cols[method_order[1]], pch = 16, log = "y", xlab = "Noise Sigma", ylab = "Mean Condition Number", main = "Suite Sigma_hat vs Sigma_reg")
  graphics::lines(first$noise_sigma, first$cond_sigma_reg, type = "b", col = method_cols[method_order[1]], pch = 1, lty = 2)
  if (length(method_order) > 1) {
    for (method in method_order[-1]) {
      dat <- subset(sigma_summary, method == method)
      graphics::lines(dat$noise_sigma, dat$cond_sigma_hat, type = "b", col = method_cols[method], pch = 16)
      graphics::lines(dat$noise_sigma, dat$cond_sigma_reg, type = "b", col = method_cols[method], pch = 1, lty = 2)
    }
  }
  graphics::legend("topleft", legend = c(paste(method_labels[method_order], "Sigma_hat"), paste(method_labels[method_order], "Sigma_reg")), col = c(method_cols[method_order], method_cols[method_order]), pch = c(rep(16, length(method_order)), rep(1, length(method_order))), lty = c(rep(1, length(method_order)), rep(2, length(method_order))), cex = 0.6, bg = "white")

  graphics::plot(first$noise_sigma, first$lambda, type = "b", col = method_cols[method_order[1]], pch = 16, log = "y", xlab = "Noise Sigma", ylab = "Mean Ridge Lambda", main = "Suite Ridge Regularization")
  if (length(method_order) > 1) {
    for (method in method_order[-1]) {
      dat <- subset(sigma_summary, method == method)
      graphics::lines(dat$noise_sigma, dat$lambda, type = "b", col = method_cols[method], pch = 16)
    }
  }
  graphics::legend("topleft", legend = method_labels[method_order], col = method_cols[method_order], pch = 16, lty = 1, cex = 0.75, bg = "white")
  grDevices::dev.off()
}

script_dir <- get_script_dir()
suite_dir <- dirname(script_dir)
repo_root <- normalizePath(file.path(suite_dir, "..", ".."))
exp_dirs <- c(
  file.path(repo_root, "dechao_reproduction", "faster_mnist_testset_paired_raw_vs_final_gaussian5_power"),
  file.path(repo_root, "dechao_reproduction", "faster_mnist_cnn_layer1_gaussian5_power"),
  file.path(repo_root, "dechao_reproduction", "faster_mnist_cnn_layer2_gaussian5_power"),
  file.path(repo_root, "dechao_reproduction", "faster_mnist_cnn_multilayer_single_gaussian_power"),
  file.path(repo_root, "dechao_reproduction", "faster_mnist_cnn_multilayer_gaussian15_power")
)
power_summary <- do.call(rbind, lapply(exp_dirs, read_power))
power_summary <- aggregate(reject_rate ~ noise_sigma + method, data = power_summary, FUN = mean)
sigma_summary <- do.call(rbind, lapply(exp_dirs, read_sigma))
sigma_summary <- aggregate(cbind(cond_sigma_hat, cond_sigma_reg, lambda) ~ noise_sigma + method, data = sigma_summary, FUN = mean)
results_dir <- file.path(suite_dir, "Results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(power_summary, file.path(results_dir, "mnist_cnn_power_suite_summary.csv"), row.names = FALSE)
utils::write.csv(sigma_summary, file.path(results_dir, "mnist_cnn_sigma_suite_summary.csv"), row.names = FALSE)
plot_power(power_summary, file.path(results_dir, "mnist_cnn_power_suite_comparison.png"))
plot_sigma(sigma_summary, file.path(results_dir, "mnist_cnn_sigma_suite_comparison.png"))
