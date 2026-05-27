embedding_file <- "data/embeddings/bloodmnist224_dinov2_baso_lymph_sharedpool"
output_dir <- "results/bloodmnist224_dinov2_baso_type1_sharedpool"

sample_size <- 100L
n_inner <- 100L
B_boot <- 500L
alpha <- 0.05
seed <- 20260527L
ridge_scale <- 1e-5
n_cores <- 8L
sample_replace <- FALSE

scenario <- "type1"
methods <- c("GAUSS5")

alternative_x_labels <- c(0)
alternative_y_labels <- c(0)
null_x_labels <- c(0)
null_y_labels <- c(0)
