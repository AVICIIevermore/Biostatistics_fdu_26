script_dir <- Sys.getenv("PLOT_SCRIPT_DIR", unset = getwd())
if (!dir.exists(script_dir)) script_dir <- getwd()
results_dir <- file.path(script_dir, "Results")
alpha <- 0.05

type1_path <- file.path(results_dir, "mnist_cnn_type1_results.csv")
sigma_path <- file.path(results_dir, "mnist_cnn_type1_sigma_diagnostics.csv")
if (!file.exists(type1_path)) stop("Missing type-I result file: ", type1_path)
if (!file.exists(sigma_path)) stop("Missing sigma diagnostics file: ", sigma_path)

type1 <- read.csv(type1_path, stringsAsFactors = FALSE)
sigma <- read.csv(sigma_path, stringsAsFactors = FALSE)

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
  raw_pixel_gaussian5 = "Raw Pixel Gaussian-5",
  layer1_gaussian5 = "Layer1 GAP Gaussian-5",
  layer1_pool2x2_gaussian5 = "Layer1 2x2 Gaussian-5",
  layer2_gaussian5 = "Layer2 GAP Gaussian-5",
  final_embedding_gaussian5 = "Final FC Gaussian-5",
  multilayer_single_gaussian = "Multilayer Single Gaussian",
  multilayer_gaussian15 = "Multilayer Gaussian-15"
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
method_order <- intersect(method_order, unique(type1$method))

summary_list <- aggregate(reject_rate ~ scenario + method + noise_sigma, type1, function(x) c(mean = mean(x), se = sd(x) / sqrt(length(x)), n_rep = length(x)))
type1_summary <- data.frame(
  scenario = summary_list$scenario,
  method = summary_list$method,
  method_label = unname(method_labels[summary_list$method]),
  noise_sigma = summary_list$noise_sigma,
  type1_mean = summary_list$reject_rate[, "mean"],
  type1_se = summary_list$reject_rate[, "se"],
  n_rep = summary_list$reject_rate[, "n_rep"],
  stringsAsFactors = FALSE
)
type1_summary <- type1_summary[order(type1_summary$scenario, match(type1_summary$method, method_order), type1_summary$noise_sigma), ]
write.csv(type1_summary, file.path(results_dir, "mnist_cnn_type1_summary.csv"), row.names = FALSE)

sigma_summary <- aggregate(cbind(cond_sigma_hat, cond_sigma_reg, lambda) ~ scenario + method + noise_sigma + kernel_count, sigma, median, na.rm = TRUE)
sigma_summary$method_label <- unname(method_labels[sigma_summary$method])
sigma_summary <- sigma_summary[order(sigma_summary$scenario, match(sigma_summary$method, method_order), sigma_summary$noise_sigma), ]
write.csv(sigma_summary, file.path(results_dir, "mnist_cnn_type1_sigma_summary.csv"), row.names = FALSE)

plot_type1 <- function(filename, title, ylim = c(0, 0.16)) {
  scenarios <- unique(type1_summary$scenario)
  png(file.path(results_dir, filename), width = 1600, height = 900, res = 150)
  par(mfrow = c(1, length(scenarios)), mar = c(5, 5, 4, 1), oma = c(0, 0, 0, 13), xpd = FALSE)
  for (sc in scenarios) {
    dd0 <- type1_summary[type1_summary$scenario == sc, ]
    plot(NA, xlim = range(dd0$noise_sigma), ylim = ylim, xlab = "Noise sigma", ylab = "Empirical Type-I error", main = sc)
    grid(col = "gray85")
    abline(h = alpha, col = "red", lty = 2, lwd = 2)
    for (m in method_order) {
      dd <- dd0[dd0$method == m, ]
      if (nrow(dd) == 0) next
      lines(dd$noise_sigma, dd$type1_mean, col = method_cols[m], lwd = 2.2)
      points(dd$noise_sigma, dd$type1_mean, col = method_cols[m], pch = method_pch[m], cex = 1)
    }
  }
  par(xpd = TRUE)
  legend("right", inset = c(-0.72, 0), legend = c(method_labels[method_order], "alpha=0.05"), col = c(method_cols[method_order], "red"), pch = c(method_pch[method_order], NA), lty = c(rep(1, length(method_order)), 2), lwd = c(rep(2.2, length(method_order)), 2), bty = "n", cex = 0.8)
  mtext(title, outer = TRUE, line = -1.5, cex = 1.2)
  dev.off()
}
plot_type1("mnist_cnn_type1_error_two_nulls.png", "MNIST CNN MMMD Type-I Error: Two Null Scenarios")
plot_type1("mnist_cnn_type1_error_two_nulls_zoom.png", "MNIST CNN MMMD Type-I Error: Zoomed", ylim = c(0, max(0.1, min(0.25, max(type1_summary$type1_mean + 0.02, na.rm = TRUE)))))

plot_condition <- function(column, filename, title, ylab) {
  scenarios <- unique(sigma_summary$scenario)
  vals <- sigma_summary[[column]]
  y_range <- range(log10(vals[is.finite(vals) & vals > 0]), na.rm = TRUE)
  png(file.path(results_dir, filename), width = 1600, height = 900, res = 150)
  par(mfrow = c(1, length(scenarios)), mar = c(5, 5, 4, 1), oma = c(0, 0, 0, 13), xpd = FALSE)
  for (sc in scenarios) {
    dd0 <- sigma_summary[sigma_summary$scenario == sc, ]
    plot(NA, xlim = range(dd0$noise_sigma), ylim = y_range, xlab = "Noise sigma", ylab = ylab, main = sc)
    grid(col = "gray85")
    for (m in method_order) {
      dd <- dd0[dd0$method == m, ]
      if (nrow(dd) == 0) next
      lines(dd$noise_sigma, log10(dd[[column]]), col = method_cols[m], lwd = 2.2)
      points(dd$noise_sigma, log10(dd[[column]]), col = method_cols[m], pch = method_pch[m], cex = 1)
    }
  }
  par(xpd = TRUE)
  legend("right", inset = c(-0.72, 0), legend = method_labels[method_order], col = method_cols[method_order], pch = method_pch[method_order], lty = 1, lwd = 2.2, bty = "n", cex = 0.8)
  mtext(title, outer = TRUE, line = -1.5, cex = 1.2)
  dev.off()
}
plot_condition("cond_sigma_hat", "mnist_cnn_type1_cond_sigma_hat_log10.png", "Type-I Sigma_hat Condition Number", "log10 median cond(Sigma_hat)")
plot_condition("cond_sigma_reg", "mnist_cnn_type1_cond_sigma_reg_log10.png", "Type-I Regularized Sigma Condition Number", "log10 median cond(Sigma_reg)")

