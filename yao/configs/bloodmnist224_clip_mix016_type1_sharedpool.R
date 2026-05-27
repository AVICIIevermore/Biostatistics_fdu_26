embedding_file <- "data/embeddings/bloodmnist224_clip_mix013_016_sharedpool"
output_dir <- "results/bloodmnist224_clip_mix016_type1_sharedpool"

sample_size <- 90L
n_inner <- 100L
B_boot <- 500L
alpha <- 0.05
seed <- 20260527L
ridge_scale <- 1e-5
n_cores <- 8L
sample_replace <- FALSE
balanced_sampling <- TRUE

scenario <- "type1"
methods <- c("GAUSS5")

alternative_x_labels <- c(0, 1, 6)
alternative_y_labels <- c(0, 1, 6)
null_x_labels <- c(0, 1, 6)
null_y_labels <- c(0, 1, 6)
