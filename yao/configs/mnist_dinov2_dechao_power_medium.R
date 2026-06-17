embedding_file <- "data/embeddings/mnist_dinov2_dechao_power_medium"
output_dir <- "results/mnist_dinov2_dechao_power_medium"

sample_size <- 100L
n_inner <- 100L
n_outer <- 2L
B_boot <- 100L
alpha <- 0.05
seed <- 20260525L
ridge_scale <- 1e-5
n_cores <- 8L
sample_replace <- TRUE

scenario <- "power"
methods <- c("GAUSS5")

alternative_x_labels <- c(1, 2, 3)
alternative_y_labels <- c(1, 2, 8)
