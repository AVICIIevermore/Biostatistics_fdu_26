get_script_dir <- function(){
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(path) > 0) {
    return(dirname(normalizePath(path[1])))
  }
  normalizePath(getwd())
}

write_graph_outputs <- function(out.d, noise_levels, n.rep, results_dir){
  fr.idx <- 2 * (1:n.rep)
  mat.fr <- matrix(0, nrow = length(noise_levels), ncol = n.rep)

  for (k in seq_along(noise_levels)) {
    mat.fr[k, ] <- out.d[k, fr.idx]
  }

  write.csv(cbind(noise_levels, mat.fr), file = file.path(results_dir, "TypeI-FR.csv"), row.names = FALSE)
}

script_dir <- get_script_dir()
experiment_root <- dirname(dirname(script_dir))
source(file.path(script_dir, "Functions.R"))
source(file.path(experiment_root, "config.R"))
source(file.path(experiment_root, "mnist_loader.R"))
config <- experiment_config
mnist_data <- load_mnist_train_flat(file.path(experiment_root, "data"))
train.x <- mnist_data$train.x
train.label <- mnist_data$train.label

for (scenario in config$scenarios) {
  results_dir <- file.path(experiment_root, "Results", scenario$name)
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  out.d <- c()
  for (iter in seq_len(config$n.rep)) {
    out.d.iter <- power.d(
      train.x = train.x,
      train.label = train.label,
      resamp = config$resamp,
      error.sigma = config$noise_levels,
      n.iter = config$n.iter,
      set.x = scenario$set_x,
      set.y = scenario$set_y,
      n.cores = config$n_cores,
      seed = scenario$seed + 2000L * iter,
      alpha = config$alpha
    )
    out.d <- c(out.d, out.d.iter)
    print(c(scenario$name, iter))
  }

  out.d <- as.matrix(as.data.frame(out.d))
  write_graph_outputs(out.d, config$noise_levels, config$n.rep, results_dir)
}
