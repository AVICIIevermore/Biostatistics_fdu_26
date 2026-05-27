embedding_file <- "data/embeddings/mnist_dinov2_dechao_power_smoke"
output_dir <- "results/mnist_dinov2_dechao_power_paperparams_2sigma"

sample_size <- 100L
n_inner <- 500L
B_boot <- 500L
alpha <- 0.05
seed <- 20260525L
ridge_scale <- 1e-5
n_cores <- 8L

scenario <- "power"
methods <- c("GAUSS5")

alternative_x_labels <- c(1, 2, 3)
alternative_y_labels <- c(1, 2, 8)
