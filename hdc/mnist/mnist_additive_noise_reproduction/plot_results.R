get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(path) > 0) {
    return(dirname(normalizePath(path[1])))
  }
  normalizePath(getwd())
}

read_metric <- function(results_dir, filename) {
  dat <- utils::read.csv(file.path(results_dir, filename), check.names = FALSE)
  data.frame(
    noise = as.numeric(dat[[1]]),
    value = rowMeans(dat[, -1, drop = FALSE]),
    stringsAsFactors = FALSE
  )
}

build_metrics <- function(results_dir) {
  metrics <- list(
    "Gauss MMMD" = read_metric(results_dir, "MultiPower-GEXP.csv"),
    "LAP MMMD" = read_metric(results_dir, "MultiPower-LAP.csv"),
    "Mixed MMMD" = read_metric(results_dir, "MultiPower-MIXED.csv"),
    "Gauss MMD" = read_metric(results_dir, "SinglePower-GAUSS.csv"),
    "LAP MMD" = read_metric(results_dir, "SinglePower-LAP.csv")
  )

  fr_path <- file.path(results_dir, "Power-FR.csv")
  if (file.exists(fr_path)) {
    fr_metric <- read_metric(results_dir, "Power-FR.csv")
    if (identical(fr_metric$noise, metrics[[1]]$noise)) {
      metrics <- c(metrics[1:3], list("FR" = fr_metric), metrics[4:5])
    } else {
      warning("Skipping FR: stale noise grid does not match current kernel results.")
    }
  }

  metrics
}

plot_metric_list <- function(metrics, output_path) {
  cols <- c(
    "Gauss MMMD" = "#1b9e77",
    "LAP MMMD" = "#d95f02",
    "Mixed MMMD" = "#7570b3",
    "FR" = "#4d4d4d",
    "Gauss MMD" = "#e7298a",
    "LAP MMD" = "#66a61e"
  )
  pchs <- c(
    "Gauss MMMD" = 16,
    "LAP MMMD" = 17,
    "Mixed MMMD" = 15,
    "FR" = 18,
    "Gauss MMD" = 8,
    "LAP MMD" = 3
  )

  first_name <- names(metrics)[1]
  first_metric <- metrics[[1]]

  grDevices::pdf(output_path, width = 8, height = 6)
  graphics::plot(
    first_metric$noise,
    first_metric$value,
    type = "b",
    pch = pchs[first_name],
    col = cols[first_name],
    lwd = 2,
    ylim = c(0, 1),
    xlab = "Noise Strength",
    ylab = "Estimated Power",
    main = "MNIST Additive Noise: {1,2,3} vs {1,2,8}"
  )

  if (length(metrics) > 1) {
    for (name in names(metrics)[-1]) {
      graphics::lines(
        metrics[[name]]$noise,
        metrics[[name]]$value,
        type = "b",
        pch = pchs[name],
        col = cols[name],
        lwd = 2
      )
    }
  }

  graphics::legend(
    "topright",
    legend = names(metrics),
    col = cols[names(metrics)],
    pch = pchs[names(metrics)],
    lty = 1,
    lwd = 2,
    cex = 0.85,
    bg = "white"
  )
  grDevices::dev.off()
}

script_dir <- get_script_dir()
results_dir <- file.path(script_dir, "Results")
metrics <- build_metrics(results_dir)
plot_metric_list(metrics, file.path(results_dir, "mnist_additive_noise_power.pdf"))
