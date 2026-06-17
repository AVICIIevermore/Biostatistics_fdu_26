get_script_dir <- function(){
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(path) > 0) {
    return(dirname(normalizePath(path[1])))
  }
  normalizePath(getwd())
}

append_log <- function(log_file, ..., .timestamp = TRUE, .console = TRUE){
  line <- paste0(..., collapse = "")
  prefix <- if (.timestamp) sprintf("[%s] ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")) else ""
  message <- paste0(prefix, line)
  cat(message, "\n", sep = "", file = log_file, append = TRUE)
  if (.console) {
    cat(message, "\n", sep = "")
    flush.console()
  }
}

script_dir <- get_script_dir()
experiment_root <- dirname(dirname(script_dir))
source(file.path(script_dir, "Functions.R"))
source(file.path(experiment_root, "config.R"))
source(file.path(experiment_root, "mnist_loader.R"))
config <- experiment_config
config$results_dir <- file.path(experiment_root, "Results")
log_file <- file.path(config$results_dir, "run.log")
checkpoint_file <- file.path(config$results_dir, "checkpoint.rds")
checkpoint_path <- config$checkpoint_path

suppressPackageStartupMessages({
  library(reticulate)
})

dir.create(config$results_dir, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(log_file)) file.create(log_file)
append_log(log_file, "cnn embedding type-I run invoked")
append_log(log_file, sprintf(
  "target config: noise_levels=%s; n.rep=%d; resamp=%d; n.iter=%d; alpha=%.3f",
  paste(config$noise_levels, collapse = ","), config$n.rep, config$resamp, config$n.iter, config$alpha
))

use_python(config$python_executable, required = TRUE)
py_module <- import_from_path("cnn_pipeline", path = config$python_module_dir)
append_log(log_file, sprintf("reticulate python=%s", py_config()$python))

train_bundle <- load_mnist_train_flat(config$shared_data_dir)
test_bundle <- load_mnist_test_flat(config$shared_data_dir)
append_log(log_file, sprintf("loaded MNIST train set: n=%d, p=%d", nrow(train_bundle$train.x), ncol(train_bundle$train.x)))
append_log(log_file, sprintf("loaded MNIST test set: n=%d, p=%d", nrow(test_bundle$test.x), ncol(test_bundle$test.x)))

append_log(log_file, "training or refreshing CNN checkpoint")
train_info <- py_module$train_cnn_model(
  data_dir = config$shared_data_dir,
  checkpoint_path = checkpoint_path,
  batch_size = as.integer(config$train_batch_size),
  epochs = as.integer(config$train_epochs),
  lr = config$train_lr,
  seed = as.integer(config$seed),
  val_fraction = config$val_fraction,
  target_val_accuracy = config$target_val_accuracy,
  history_path = file.path(config$shared_model_dir, "train_history.csv"),
  train_log_path = file.path(config$shared_model_dir, "train.log"),
  train_plot_path = file.path(config$shared_model_dir, "training_curves.png"),
  config_dump_path = file.path(config$shared_model_dir, "train_config.json"),
  force_retrain = FALSE
)
append_log(log_file, sprintf(
  "cnn checkpoint ready: path=%s; best_val_acc=%.4f; device=%s",
  checkpoint_path,
  as.numeric(train_info$best_val_acc),
  as.character(train_info$device)
))

selected <- select_digit_pools(test_bundle$test.x, test_bundle$test.label, config$set_x, config$set_y)
pool.X <- selected$data.X
pool.Y <- selected$data.Y
append_log(log_file, sprintf("selected test pools: |X|=%d, |Y|=%d", nrow(pool.X), nrow(pool.Y)))

required_layers <- unique(unlist(lapply(config$method_order, function(method) {
  switch(
    method,
    layer1_gaussian5 = "layer1",
    layer2_gaussian5 = "layer2",
    final_embedding_gaussian5 = "final",
    multilayer_single_gaussian = c("layer1", "layer2", "final"),
    multilayer_gaussian15 = c("layer1", "layer2", "final"),
    raw_pixel_gaussian5 = character(0),
    character(0)
  )
})))
if (length(required_layers) == 0) {
  required_layers <- character(0)
}
append_log(log_file, sprintf("required embedding layers: %s", if (length(required_layers) == 0) "<none>" else paste(required_layers, collapse = ",")))

start <- Sys.time()
power_rows <- list()
sigma_rows <- list()
start_iter <- 1L
if (file.exists(checkpoint_file)) {
  ckpt <- readRDS(checkpoint_file)
  power_rows <- ckpt$power_rows
  sigma_rows <- ckpt$sigma_rows
  start_iter <- ckpt$iter_completed + 1L
  append_log(log_file, sprintf("resuming from checkpoint: iter_completed=%d; target_n.rep=%d", ckpt$iter_completed, config$n.rep))
}

if (start_iter <= config$n.rep) {
  for (iter in start_iter:config$n.rep) {
    iter_start <- Sys.time()
    append_log(log_file, sprintf("iter %d/%d started", iter, config$n.rep))
    set.seed(config$seed + 1000L * iter)

    iter_power_rows <- list()
    iter_sigma_rows <- list()

    for (k in seq_along(config$noise_levels)) {
      sigma <- config$noise_levels[[k]]
      append_log(log_file, sprintf("iter %d/%d sigma %d/%d = %.3f started", iter, config$n.rep, k, length(config$noise_levels), sigma))

      noisy.X <- pmin(pmax(pool.X + matrix(rnorm(length(pool.X), mean = 0, sd = sigma), nrow = nrow(pool.X)), 0), 1)
      noisy.Y <- pmin(pmax(pool.Y + matrix(rnorm(length(pool.Y), mean = 0, sd = sigma), nrow = nrow(pool.Y)), 0), 1)

      embed.X <- if (length(required_layers) > 0) extract_cnn_embeddings(py_module, checkpoint_path, noisy.X, config$eval_batch_size, layers = required_layers) else list()
      embed.Y <- if (length(required_layers) > 0) extract_cnn_embeddings(py_module, checkpoint_path, noisy.Y, config$eval_batch_size, layers = required_layers) else list()

      resamp.x <- sample_without_replacement_matrix(nrow(noisy.X), config$n.iter, config$resamp)
      resamp.y <- sample_without_replacement_matrix(nrow(noisy.Y), config$n.iter, config$resamp)

      out <- run_methods_on_resamples(
        data.X = noisy.X,
        data.Y = noisy.Y,
        embed.X = embed.X,
        embed.Y = embed.Y,
        resamp.x = resamp.x,
        resamp.y = resamp.y,
        B_boot = config$n.iter,
        alpha = config$alpha,
        noise_sigma = sigma,
        outer_iter = iter,
        method_names = config$method_order,
        n_cores = config$n_cores
      )

      iter_power_rows[[length(iter_power_rows) + 1L]] <- out$power_rows
      iter_sigma_rows[[length(iter_sigma_rows) + 1L]] <- out$sigma_rows
      append_log(log_file, sprintf("iter %d/%d sigma %.3f finished", iter, config$n.rep, sigma))
    }

    power_rows[[length(power_rows) + 1L]] <- do.call(rbind, iter_power_rows)
    sigma_rows[[length(sigma_rows) + 1L]] <- do.call(rbind, iter_sigma_rows)
    saveRDS(
      list(
        iter_completed = iter,
        power_rows = power_rows,
        sigma_rows = sigma_rows,
        timestamp = Sys.time(),
        config = config
      ),
      checkpoint_file
    )
    append_log(log_file, sprintf("iter %d/%d finished in %s; checkpoint updated", iter, config$n.rep, format(Sys.time() - iter_start)))
  }
} else {
  append_log(log_file, sprintf("no new iterations needed: checkpoint already at iter %d", start_iter - 1L))
}

power_df <- do.call(rbind, power_rows)
sigma_df <- do.call(rbind, sigma_rows)
utils::write.csv(power_df, file.path(config$results_dir, "mnist_cnn_type1_results.csv"), row.names = FALSE)
utils::write.csv(sigma_df, file.path(config$results_dir, "mnist_cnn_type1_sigma_diagnostics.csv"), row.names = FALSE)
append_log(log_file, "cnn type-I CSV outputs written")

Sys.setenv(PLOT_SCRIPT_DIR = experiment_root)
on.exit(Sys.unsetenv("PLOT_SCRIPT_DIR"), add = TRUE)
source(file.path(experiment_root, "plot_results.R"))
append_log(log_file, "automatic plotting finished")
append_log(log_file, sprintf("all requested iterations available in %s", format(Sys.time() - start)))
