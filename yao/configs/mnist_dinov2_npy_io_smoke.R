embedding_file <- "data/embeddings/mnist_dinov2_npy_io_smoke"
output_dir <- "results/mnist_dinov2_npy_io_smoke"

sample_size <- 30L
n_inner <- 3L
B_boot <- 5L
alpha <- 0.05
seed <- 20260525L
ridge_scale <- 1e-5
n_cores <- 1L

scenario <- "power"
methods <- c("GAUSS5")

alternative_x_labels <- c(1, 2, 3)
alternative_y_labels <- c(1, 2, 8)
