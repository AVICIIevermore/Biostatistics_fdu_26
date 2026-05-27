embedding_file <- "data/embeddings/bloodmnist224_dinov2_mix013_016_sharedpool"
output_dir <- "yao/results/bloodmnist224_sample_size_sigma06/dinov2_power_n30"

sample_size <- 30L
n_inner <- 20L
n_outer <- 4L
B_boot <- 100L
alpha <- 0.05
seed <- 20260527L
ridge_scale <- 1e-5
n_cores <- 8L
sample_replace <- FALSE
balanced_sampling <- TRUE
noise_subset <- c(0.6)

methods <- c("GAUSS5")

scenario <- "power"
alternative_x_labels <- c(0, 1, 3)
alternative_y_labels <- c(0, 1, 6)

