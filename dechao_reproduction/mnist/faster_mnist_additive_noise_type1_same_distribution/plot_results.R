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
    "Gauss MMMD" = read_metric(results_dir, "TypeI-Multi-GEXP.csv"),
    "LAP MMMD" = read_metric(results_dir, "TypeI-Multi-LAP.csv"),
    "Mixed MMMD" = read_metric(results_dir, "TypeI-Multi-MIXED.csv"),
    "Gauss MMD" = read_metric(results_dir, "TypeI-Single-GAUSS.csv"),
    "LAP MMD" = read_metric(results_dir, "TypeI-Single-LAP.csv")
  )

  fr_path <- file.path(results_dir, "TypeI-FR.csv")
  if (file.exists(fr_path)) {
    fr_metric <- read_metric(results_dir, "TypeI-FR.csv")
    if (identical(fr_metric$noise, metrics[[1]]$noise)) {
      metrics <- c(metrics[1:3], list("FR" = fr_metric), metrics[4:5])
    } else {
      warning(paste("Skipping FR for", basename(results_dir), ": stale noise grid does not match current kernel results."))
    }
  }

  metrics
}

plot_scenario <- function(metrics, scenario_name) {
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

  graphics::plot(
    first_metric$noise,
    first_metric$value,
    type = "b",
    pch = pchs[first_name],
    col = cols[first_name],
    lwd = 2,
    ylim = c(0, 0.08),
    xlab = "Noise Strength",
    ylab = "Estimated Type-I Error",
    main = scenario_name
  )
  graphics::abline(h = 0.05, lty = 2, col = "gray50")

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
    cex = 0.8,
    bg = "white"
  )
}

script_dir <- get_script_dir()
results_root <- file.path(script_dir, "Results")
scenario_labels <- c(
  set123_vs_set123 = "Type-I Error: {1,2,3} vs {1,2,3}",
  set128_vs_set128 = "Type-I Error: {1,2,8} vs {1,2,8}"
)
scenario_names <- names(scenario_labels)
scenario_metrics <- lapply(
  scenario_names,
  function(name) build_metrics(file.path(results_root, name))
)
names(scenario_metrics) <- scenario_names

for (name in scenario_names) {
  grDevices::pdf(file.path(results_root, name, "mnist_additive_noise_type1.pdf"), width = 8, height = 6)
  plot_scenario(scenario_metrics[[name]], scenario_labels[[name]])
  grDevices::dev.off()
}

grDevices::pdf(file.path(results_root, "mnist_additive_noise_type1_combined.pdf"), width = 12, height = 5.5)
old_par <- graphics::par(mfrow = c(1, 2), mar = c(4.2, 4.2, 3.2, 1.2))
on.exit(graphics::par(old_par), add = TRUE)
for (name in scenario_names) {
  plot_scenario(scenario_metrics[[name]], scenario_labels[[name]])
}
grDevices::dev.off()
