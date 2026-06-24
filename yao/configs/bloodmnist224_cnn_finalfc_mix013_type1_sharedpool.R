embedding_file <- "data/embeddings/bloodmnist224_cnn_finalfc_mix013_016_sharedpool"
output_dir <- "results/bloodmnist224_cnn_finalfc_mix013_type1_sharedpool"

sample_size <- 90L
n_inner <- 100L
B_boot <- 500L
alpha <- 0.05
seed <- 20260604L
ridge_scale <- 1e-5
n_cores <- 3L
sample_replace <- FALSE
balanced_sampling <- TRUE

scenario <- "type1"
methods <- c("GAUSS5")

alternative_x_labels <- c(0, 1, 3)
alternative_y_labels <- c(0, 1, 3)
null_x_labels <- c(0, 1, 3)
null_y_labels <- c(0, 1, 3)
