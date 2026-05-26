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

method_order_all <- c(
  "raw_pixel_gaussian5",
  "layer1_gaussian5",
  "layer2_gaussian5",
  "final_embedding_gaussian5",
  "multilayer_single_gaussian",
  "multilayer_gaussian15"
)
method_labels <- c(
  raw_pixel_gaussian5 = "Raw Pixel Gaussian-5",
  layer1_gaussian5 = "Layer1 Gaussian-5",
  layer2_gaussian5 = "Layer2 Gaussian-5",
  final_embedding_gaussian5 = "Penultimate FC-128 Gaussian-5",
  multilayer_single_gaussian = "Multilayer Single Gaussian",
  multilayer_gaussian15 = "Multilayer Gaussian-15"
)
method_cols <- c(
  raw_pixel_gaussian5 = "#0072B2",
  layer1_gaussian5 = "#009E73",
  layer2_gaussian5 = "#CC79A7",
  final_embedding_gaussian5 = "#D55E00",
  multilayer_single_gaussian = "#56B4E9",
  multilayer_gaussian15 = "#E69F00"
)
method_pchs <- c(
  raw_pixel_gaussian5 = 16,
  layer1_gaussian5 = 15,
  layer2_gaussian5 = 18,
  final_embedding_gaussian5 = 17,
  multilayer_single_gaussian = 8,
  multilayer_gaussian15 = 3
)
method_ltys <- c(
  raw_pixel_gaussian5 = 1,
  layer1_gaussian5 = 2,
  layer2_gaussian5 = 3,
  final_embedding_gaussian5 = 4,
  multilayer_single_gaussian = 5,
  multilayer_gaussian15 = 6
)

present_methods <- function(methods) {
  method_order_all[method_order_all %in% unique(as.character(methods))]
}

build_power_table <- function(results_dir) {
  dat <- utils::read.csv(file.path(results_dir, "mnist_cnn_power_results.csv"), check.names = FALSE)
  out <- aggregate(reject_rate ~ noise_sigma + method, data = dat, FUN = mean)
  out$method <- as.character(out$method)
  methods <- present_methods(out$method)
  out$method <- factor(out$method, levels = methods)
  out[order(out$method, out$noise_sigma), ]
}

plot_power <- function(power_summary, output_path) {
  methods <- present_methods(power_summary$method)
  noise <- sort(unique(power_summary$noise_sigma))

  grDevices::png(output_path, width = 1400, height = 900, res = 150)
  graphics::par(mar = c(4.8, 5.0, 3.2, 1.2))
  graphics::plot(
    range(noise),
    c(0, 1),
    type = "n",
    xlab = "Noise Sigma",
    ylab = "Estimated Power",
    main = "MNIST Power: Test-Set Methods"
  )
  graphics::grid(col = "gray88")

  for (method in methods) {
    dat <- power_summary[as.character(power_summary$method) == method, ]
    dat <- dat[order(dat$noise_sigma), ]
    graphics::lines(
      dat$noise_sigma,
      dat$reject_rate,
      type = "b",
      pch = method_pchs[method],
      col = method_cols[method],
      lwd = 3,
      lty = method_ltys[method]
    )
  }

  graphics::legend(
    "topright",
    legend = method_labels[methods],
    col = method_cols[methods],
    pch = method_pchs[methods],
    lty = method_ltys[methods],
    lwd = 3,
    cex = 0.82,
    bg = "white"
  )
  grDevices::dev.off()
}

plot_paired_power <- function(power_summary, output_path) {
  methods <- c("raw_pixel_gaussian5", "final_embedding_gaussian5")
  if (!all(methods %in% as.character(power_summary$method))) {
    return(invisible(NULL))
  }
  grDevices::png(output_path, width = 1400, height = 900, res = 150)
  graphics::par(mar = c(4.8, 5.0, 3.2, 1.2))
  noise <- sort(unique(power_summary$noise_sigma))
  graphics::plot(range(noise), c(0, 1), type = "n", xlab = "Noise Sigma", ylab = "Estimated Power", main = "Paired Test-Set Power: Raw Pixel vs Penultimate FC-128")
  graphics::grid(col = "gray88")
  for (method in methods) {
    dat <- power_summary[as.character(power_summary$method) == method, ]
    dat <- dat[order(dat$noise_sigma), ]
    graphics::lines(dat$noise_sigma, dat$reject_rate, type = "b", pch = method_pchs[method], col = method_cols[method], lwd = 3, lty = method_ltys[method])
  }
  graphics::legend("bottomleft", legend = method_labels[methods], col = method_cols[methods], pch = method_pchs[methods], lty = method_ltys[methods], lwd = 3, cex = 0.95, bg = "white")
  grDevices::dev.off()
}

plot_sigma_diag <- function(results_dir, output_path) {
  dat <- utils::read.csv(file.path(results_dir, "mnist_cnn_sigma_diagnostics.csv"), check.names = FALSE)
  agg <- aggregate(cbind(cond_sigma_hat, cond_sigma_reg, lambda) ~ noise_sigma + method, data = dat, FUN = mean)
  agg$method <- as.character(agg$method)
  methods <- present_methods(agg$method)
  noise <- sort(unique(agg$noise_sigma))

  grDevices::png(output_path, width = 1800, height = 900, res = 150)
  graphics::par(mfrow = c(1, 3), mar = c(4.8, 4.8, 3.0, 1.0))

  panels <- list(
    list(col = "cond_sigma_hat", ylab = "Mean Cond(Sigma_hat)", main = "Sigma_hat Condition Number"),
    list(col = "cond_sigma_reg", ylab = "Mean Cond(Sigma_reg)", main = "Sigma_reg Condition Number"),
    list(col = "lambda", ylab = "Mean Ridge Lambda", main = "Ridge Lambda")
  )

  for (panel in panels) {
    yvals <- agg[[panel$col]]
    yvals <- yvals[is.finite(yvals) & yvals > 0]
    yrange <- if (length(yvals) > 0) range(yvals) else c(1, 10)
    graphics::plot(range(noise), yrange, type = "n", log = "y", xlab = "Noise Sigma", ylab = panel$ylab, main = panel$main)
    graphics::grid(col = "gray88")
    for (method in methods) {
      method_dat <- agg[agg$method == method, ]
      method_dat <- method_dat[order(method_dat$noise_sigma), ]
      graphics::lines(method_dat$noise_sigma, method_dat[[panel$col]], type = "b", pch = method_pchs[method], col = method_cols[method], lwd = 2.5, lty = method_ltys[method])
    }
    graphics::legend("topright", legend = method_labels[methods], col = method_cols[methods], pch = method_pchs[methods], lty = method_ltys[methods], lwd = 2.5, cex = 0.65, bg = "white")
  }
  grDevices::dev.off()
}

script_dir <- get_script_dir()
results_dir <- file.path(script_dir, "Results")
power_summary <- build_power_table(results_dir)
utils::write.csv(power_summary, file.path(results_dir, "mnist_cnn_power_summary.csv"), row.names = FALSE)
plot_power(power_summary, file.path(results_dir, "mnist_cnn_power_comparison.png"))
plot_paired_power(power_summary, file.path(results_dir, "mnist_cnn_paired_raw_vs_final_power.png"))
plot_sigma_diag(results_dir, file.path(results_dir, "mnist_sigma_condition_number.png"))
