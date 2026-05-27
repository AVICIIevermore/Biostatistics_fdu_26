embedding_file <- "data/embeddings/pathmnist_dinov2.csv"
output_dir <- "results/pathmnist_dinov2"

sample_size <- 100L
n_reps <- 100L
B_boot <- 500L
alpha <- 0.05
seed <- 20260526L
ridge_scale <- 1e-5

methods <- c("GAUSS1", "GAUSS5", "LAP5", "MIXED")

# For PathMNIST labels, adjust these after inspecting the generated label map.
# Type-I error: split one label pool into two independent samples.
null_labels <- c(0)

# Power: compare two different label groups.
alternative_x_labels <- c(0)
alternative_y_labels <- c(1)
