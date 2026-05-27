get_script_dir <- function() {
  env_dir <- Sys.getenv("PLOT_SCRIPT_DIR", unset = "")
  if (nzchar(env_dir)) {
    return(normalizePath(env_dir))
  }
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

build_power_table <- function(results_dir) {
  dat <- utils::read.csv(file.path(results_dir, "mnist_cnn_type1_results.csv"), check.names = FALSE)
  out <- aggregate(reject_rate ~ noise_sigma + method, data = dat, FUN = mean)
  method_order <- c(
    "raw_pixel_gaussian5",
    "layer1_gaussian5",
    "layer2_gaussian5",
    "final_embedding_gaussian5",
    "multilayer_single_gaussian",
    "multilayer_gaussian15"
  )
  out$method <- factor(as.character(out$method), levels = method_order)
  out[order(out$method, out$noise_sigma), ]
}

plot_power <- function(power_summary, output_path) {
  method_order <- levels(power_summary$method)[levels(power_summary$method) %in% as.character(power_summary$method)]
  power_summary <- power_summary[order(power_summary$method, power_summary$noise_sigma), ]
  first <- subset(power_summary, method == method_order[1])

  grDevices::png(output_path, width = 1200, height = 900, res = 150)
  graphics::plot(first$noise_sigma, first$reject_rate, type = "b", pch = method_pchs[method_order[1]], col = method_cols[method_order[1]], lwd = 2, ylim = c(0, 1), xlab = "Noise Sigma", ylab = "Empirical Type-I Error", main = "MNIST Type-I Error: {1,2,3} vs {1,2,3}")
  if (length(method_order) > 1) {
    for (method in method_order[-1]) {
      dat <- subset(power_summary, method == method)
      graphics::lines(dat$noise_sigma, dat$reject_rate, type = "b", pch = method_pchs[method], col = method_cols[method], lwd = 2)
    }
  }
  graphics::legend("topright", legend = method_labels[method_order], col = method_cols[method_order], pch = method_pchs[method_order], lty = 1, lwd = 2, cex = 0.9, bg = "white")
  grDevices::dev.off()
}

plot_paired_power <- function(power_summary, output_path) {
  if (!all(c("raw_pixel_gaussian5", "final_embedding_gaussian5") %in% as.character(power_summary$method))) {
    return(invisible(NULL))
  }
  raw <- subset(power_summary, method == "raw_pixel_gaussian5")
  final <- subset(power_summary, method == "final_embedding_gaussian5")

  grDevices::png(output_path, width = 1400, height = 900, res = 150)
  graphics::par(mar = c(4.5, 4.8, 3.2, 1.2))
  graphics::plot(
    raw$noise_sigma,
    raw$reject_rate,
    type = "b",
    pch = 16,
    col = method_cols["raw_pixel_gaussian5"],
    lwd = 3,
    lty = 1,
    ylim = c(0, 1),
    xlab = "Noise Sigma",
    ylab = "Empirical Type-I Error",
    main = "Type-I Error: Raw Pixel vs Final CNN Embedding"
  )
  graphics::abline(h = 0.05, col = "gray35", lty = 3, lwd = 2)
  graphics::grid(col = "gray85")
  graphics::lines(
    raw$noise_sigma,
    raw$reject_rate,
    type = "b",
    pch = 16,
    col = method_cols["raw_pixel_gaussian5"],
    lwd = 3,
    lty = 1
  )
  graphics::lines(
    final$noise_sigma,
    final$reject_rate,
    type = "b",
    pch = 17,
    col = method_cols["final_embedding_gaussian5"],
    lwd = 3,
    lty = 2
  )
  graphics::legend(
    "bottomleft",
    legend = c("Raw Pixel Gaussian-5", "Final Embedding Gaussian-5"),
    col = c(method_cols["raw_pixel_gaussian5"], method_cols["final_embedding_gaussian5"]),
    pch = c(16, 17),
    lty = c(1, 2),
    lwd = 3,
    cex = 0.95,
    bg = "white"
  )
  grDevices::dev.off()
}

plot_sigma_diag <- function(results_dir, output_path) {
  dat <- utils::read.csv(file.path(results_dir, "mnist_cnn_type1_sigma_diagnostics.csv"), check.names = FALSE)
  agg <- aggregate(cbind(cond_sigma_hat, cond_sigma_reg, lambda) ~ noise_sigma + method, data = dat, FUN = mean)
  methods <- unique(as.character(agg$method))
  first <- subset(agg, method == methods[1])

  grDevices::png(output_path, width = 1400, height = 900, res = 150)
  graphics::par(mfrow = c(1, 2))
  graphics::plot(first$noise_sigma, first$cond_sigma_hat, type = "b", col = method_cols[methods[1]], pch = 16, log = "y", xlab = "Noise Sigma", ylab = "Mean Condition Number", main = "Sigma_hat vs Sigma_reg")
  graphics::lines(first$noise_sigma, first$cond_sigma_reg, type = "b", col = method_cols[methods[1]], pch = 1, lty = 2)
  if (length(methods) > 1) {
    for (method in methods[-1]) {
      dat <- subset(agg, method == method)
      graphics::lines(dat$noise_sigma, dat$cond_sigma_hat, type = "b", col = method_cols[method], pch = 16)
      graphics::lines(dat$noise_sigma, dat$cond_sigma_reg, type = "b", col = method_cols[method], pch = 1, lty = 2)
    }
  }
  graphics::legend("topleft", legend = c(paste(method_labels[methods], "Sigma_hat"), paste(method_labels[methods], "Sigma_reg")), col = c(method_cols[methods], method_cols[methods]), pch = c(rep(16, length(methods)), rep(1, length(methods))), lty = c(rep(1, length(methods)), rep(2, length(methods))), cex = 0.7, bg = "white")

  graphics::plot(first$noise_sigma, first$lambda, type = "b", col = method_cols[methods[1]], pch = 16, log = "y", xlab = "Noise Sigma", ylab = "Mean Ridge Lambda", main = "Ridge Regularization")
  if (length(methods) > 1) {
    for (method in methods[-1]) {
      dat <- subset(agg, method == method)
      graphics::lines(dat$noise_sigma, dat$lambda, type = "b", col = method_cols[method], pch = 16)
    }
  }
  graphics::legend("topleft", legend = method_labels[methods], col = method_cols[methods], pch = 16, lty = 1, cex = 0.8, bg = "white")
  grDevices::dev.off()
}

script_dir <- get_script_dir()
results_dir <- file.path(script_dir, "Results")
power_summary <- build_power_table(results_dir)
utils::write.csv(power_summary, file.path(results_dir, "mnist_cnn_type1_summary.csv"), row.names = FALSE)
plot_power(power_summary, file.path(results_dir, "mnist_cnn_type1_error.png"))
plot_paired_power(power_summary, file.path(results_dir, "mnist_cnn_type1_raw_vs_final.png"))
plot_sigma_diag(results_dir, file.path(results_dir, "mnist_type1_sigma_condition_number.png"))
