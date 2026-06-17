experiment_config <- list(
  experiment_name = "mnist_additive_noise_type1_same_distribution",
  noise_levels = seq(0, 1, length.out = 6),
  n.rep = 10,
  resamp = 100,
  n.iter = 500,
  n_cores = 8,
  alpha = 0.05,
  kernel_choice = c("LAP", "GAUSS", "LAP", "GEXP", "MIXED"),
  scenarios = list(
    list(name = "set123_vs_set123", set_x = c(1, 2, 3), set_y = c(1, 2, 3), seed = 20260522L),
    list(name = "set128_vs_set128", set_x = c(1, 2, 8), set_y = c(1, 2, 8), seed = 20260523L)
  )
)
