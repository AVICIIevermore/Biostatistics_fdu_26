embedding_file <- "data/embeddings/mnist_dinov2_smoke.csv"
output_dir <- "results/mnist_dinov2_smoke"

sample_size <- 30L
n_reps <- 5L
B_boot <- 50L
alpha <- 0.05
seed <- 20260526L
ridge_scale <- 1e-5

methods <- c("GAUSS1", "GAUSS5")

# Null experiment: split digit 1 embeddings into two same-distribution samples.
null_labels <- c(1)

# Power smoke experiment: reproduce the MNIST paper-style set comparison.
alternative_x_labels <- c(1, 2, 3)
alternative_y_labels <- c(1, 2, 8)
