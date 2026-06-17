embedding_file <- "data/embeddings/bloodmnist224_dinov2_mix013_016_sharedpool"
output_dir <- "results/bloodmnist224_dinov2_mix013_vs_016_power_baselines"

sample_size <- 90L
n_inner <- 100L
B_boot <- 500L
alpha <- 0.05
seed <- 20260527L
ridge_scale <- 1e-5
n_cores <- 8L
sample_replace <- FALSE
balanced_sampling <- TRUE

scenario <- "power"
methods <- c("GAUSS1", "LAP1", "GAUSS5", "LAP5", "MIXED")

alternative_x_labels <- c(0, 1, 3)
alternative_y_labels <- c(0, 1, 6)
