get_script_dir <- function(){
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(path) > 0) {
    return(dirname(normalizePath(path[1])))
  }
  normalizePath(getwd())
}

script_dir <- get_script_dir()
experiment_root <- dirname(dirname(script_dir))
source(file.path(script_dir, "Functions.R"))
source(file.path(experiment_root, "config.R"))
source(file.path(experiment_root, "mnist_loader.R"))
config <- experiment_config
config$results_dir <- file.path(experiment_root, "Results")

dir.create(config$results_dir, recursive = TRUE, showWarnings = FALSE)
mnist_data <- load_mnist_train_flat(file.path(experiment_root, "data"))
train.x <- mnist_data$train.x
train.label <- mnist_data$train.label

start <- Sys.time()
out.d <- c()
for (iter in seq_len(config$n.rep)) {
  out.d.iter <- power.d(
    train.x = train.x,
    train.label = train.label,
    resamp = config$resamp,
    error.sigma = config$noise_levels,
    n.iter = config$n.iter,
    set.x = config$set_x,
    set.y = config$set_y,
    n.cores = config$n_cores,
    seed = config$seed + 2000L * iter,
    alpha = config$alpha
  )
  out.d <- c(out.d, out.d.iter)
  print(iter)
}
out.d <- as.matrix(as.data.frame(out.d))
end <- Sys.time()
print(end - start)

single.FR.d1 <- 2 * (1:config$n.rep)
power.FR.mat1 <- matrix(0, nrow = length(config$noise_levels), ncol = config$n.rep)
for (k in seq_along(config$noise_levels)) {
  power.FR.mat1[k, ] <- out.d[k, single.FR.d1]
}
power.FR.mat1 <- cbind(config$noise_levels, power.FR.mat1)
write.csv(power.FR.mat1, file = file.path(config$results_dir, "Power-FR.csv"), row.names = FALSE)
