experiment_config <- list(
  experiment_name = "mnist_additive_noise_reproduction",
  noise_levels = seq(0, 1, length.out = 6),
  n.rep = 50,
  resamp = 100,
  n.iter = 500,
  n_cores = 16,
  alpha = 0.05,
  seed = 20260521L,
  set_x = c(1, 2, 3),
  set_y = c(1, 2, 8),
  kernel_choice = c("LAP", "GAUSS", "LAP", "GEXP", "MIXED"),
  results_dir = file.path(normalizePath(".", winslash = "/", mustWork = FALSE), "Results")
)
