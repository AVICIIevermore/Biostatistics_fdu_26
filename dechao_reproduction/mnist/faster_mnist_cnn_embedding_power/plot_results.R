get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(path) > 0) {
    return(dirname(normalizePath(path[1])))
  }
  normalizePath(getwd())
}

build_power_table <- function(results_dir) {
  dat <- utils::read.csv(file.path(results_dir, "mnist_cnn_power_results.csv"), check.names = FALSE)
  aggregate(reject_rate ~ noise_sigma + method, data = dat, FUN = mean)
}

plot_power <- function(power_summary, output_path) {
  method_order <- c(
    "raw_pixel_gaussian5",
    "final_embedding_gaussian5",
    "multilayer_single_gaussian",
    "multilayer_gaussian15"
  )
  labels <- c(
    raw_pixel_gaussian5 = "Raw Pixel Gaussian-5",
    final_embedding_gaussian5 = "Final Embedding Gaussian-5",
    multilayer_single_gaussian = "Multilayer Single Gaussian",
    multilayer_gaussian15 = "Multilayer Gaussian-15"
  )
  cols <- c(
    raw_pixel_gaussian5 = "#1b9e77",
    final_embedding_gaussian5 = "#d95f02",
    multilayer_single_gaussian = "#7570b3",
    multilayer_gaussian15 = "#e7298a"
  )
  pchs <- c(
    raw_pixel_gaussian5 = 16,
    final_embedding_gaussian5 = 17,
    multilayer_single_gaussian = 15,
    multilayer_gaussian15 = 18
  )

  power_summary$method <- factor(power_summary$method, levels = method_order)
  power_summary <- power_summary[order(power_summary$method, power_summary$noise_sigma), ]
  first <- subset(power_summary, method == method_order[1])

  grDevices::png(output_path, width = 1200, height = 900, res = 150)
  graphics::plot(
    first$noise_sigma,
    first$reject_rate,
    type = "b",
    pch = pchs[method_order[1]],
    col = cols[method_order[1]],
    lwd = 2,
    ylim = c(0, 1),
    xlab = "Noise Sigma",
    ylab = "Estimated Power",
    main = "MNIST CNN Embedding Power: {1,2,3} vs {1,2,8}"
  )

  for (method in method_order[-1]) {
    dat <- subset(power_summary, method == method)
    graphics::lines(
      dat$noise_sigma,
      dat$reject_rate,
      type = "b",
      pch = pchs[method],
      col = cols[method],
      lwd = 2
    )
  }

  graphics::legend(
    "topright",
    legend = labels[method_order],
    col = cols[method_order],
    pch = pchs[method_order],
    lty = 1,
    lwd = 2,
    cex = 0.9,
    bg = "white"
  )
  grDevices::dev.off()
}

plot_sigma_diag <- function(results_dir, output_path) {
  dat <- utils::read.csv(file.path(results_dir, "mnist_cnn_sigma_diagnostics.csv"), check.names = FALSE)
  agg <- aggregate(
    cbind(cond_sigma_hat, cond_sigma_reg, lambda) ~ noise_sigma + method,
    data = dat,
    FUN = mean
  )
  methods <- unique(agg$method)
  cols <- c(
    raw_pixel_gaussian5 = "#1b9e77",
    final_embedding_gaussian5 = "#d95f02",
    multilayer_single_gaussian = "#7570b3",
    multilayer_gaussian15 = "#e7298a"
  )

  grDevices::png(output_path, width = 1400, height = 900, res = 150)
  graphics::par(mfrow = c(1, 2))

  first <- subset(agg, method == methods[1])
  graphics::plot(
    first$noise_sigma,
    first$cond_sigma_hat,
    type = "b",
    col = cols[methods[1]],
    pch = 16,
    log = "y",
    xlab = "Noise Sigma",
    ylab = "Mean Condition Number",
    main = "Sigma_hat vs Sigma_reg"
  )
  graphics::lines(first$noise_sigma, first$cond_sigma_reg, type = "b", col = cols[methods[1]], pch = 1, lty = 2)
  for (method in methods[-1]) {
    dat <- subset(agg, method == method)
    graphics::lines(dat$noise_sigma, dat$cond_sigma_hat, type = "b", col = cols[method], pch = 16)
    graphics::lines(dat$noise_sigma, dat$cond_sigma_reg, type = "b", col = cols[method], pch = 1, lty = 2)
  }
  graphics::legend(
    "topleft",
    legend = c(
      paste(names(cols), "Sigma_hat"),
      paste(names(cols), "Sigma_reg")
    ),
    col = c(cols, cols),
    pch = c(rep(16, length(cols)), rep(1, length(cols))),
    lty = c(rep(1, length(cols)), rep(2, length(cols))),
    cex = 0.7,
    bg = "white"
  )

  graphics::plot(
    first$noise_sigma,
    first$lambda,
    type = "b",
    col = cols[methods[1]],
    pch = 16,
    log = "y",
    xlab = "Noise Sigma",
    ylab = "Mean Ridge Lambda",
    main = "Ridge Regularization"
  )
  for (method in methods[-1]) {
    dat <- subset(agg, method == method)
    graphics::lines(dat$noise_sigma, dat$lambda, type = "b", col = cols[method], pch = 16)
  }
  graphics::legend("topleft", legend = methods, col = cols[methods], pch = 16, lty = 1, cex = 0.8, bg = "white")
  grDevices::dev.off()
}

script_dir <- get_script_dir()
results_dir <- file.path(script_dir, "Results")
power_summary <- build_power_table(results_dir)
plot_power(power_summary, file.path(results_dir, "mnist_cnn_power_comparison.png"))
plot_sigma_diag(results_dir, file.path(results_dir, "mnist_sigma_condition_number.png"))
