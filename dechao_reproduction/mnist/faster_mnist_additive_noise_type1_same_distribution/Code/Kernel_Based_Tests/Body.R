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

write_kernel_outputs <- function(out.d, noise_levels, n.rep, results_dir){
  single.idx1 <- 2 + 6 * (0:(n.rep - 1))
  single.idx2 <- 3 + 6 * (0:(n.rep - 1))
  multi.idx1 <- 4 + 6 * (0:(n.rep - 1))
  multi.idx2 <- 5 + 6 * (0:(n.rep - 1))
  multi.idx3 <- 6 * (1:n.rep)

  mat.single1 <- matrix(0, nrow = length(noise_levels), ncol = n.rep)
  mat.single2 <- matrix(0, nrow = length(noise_levels), ncol = n.rep)
  mat.multi1 <- matrix(0, nrow = length(noise_levels), ncol = n.rep)
  mat.multi2 <- matrix(0, nrow = length(noise_levels), ncol = n.rep)
  mat.multi3 <- matrix(0, nrow = length(noise_levels), ncol = n.rep)

  for (k in seq_along(noise_levels)) {
    mat.single1[k, ] <- out.d[k, single.idx1]
    mat.single2[k, ] <- out.d[k, single.idx2]
    mat.multi1[k, ] <- out.d[k, multi.idx1]
    mat.multi2[k, ] <- out.d[k, multi.idx2]
    mat.multi3[k, ] <- out.d[k, multi.idx3]
  }

  write.csv(cbind(noise_levels, mat.single1), file = file.path(results_dir, "TypeI-Single-LAP.csv"), row.names = FALSE)
  write.csv(cbind(noise_levels, mat.single2), file = file.path(results_dir, "TypeI-Single-GAUSS.csv"), row.names = FALSE)
  write.csv(cbind(noise_levels, mat.multi1), file = file.path(results_dir, "TypeI-Multi-LAP.csv"), row.names = FALSE)
  write.csv(cbind(noise_levels, mat.multi2), file = file.path(results_dir, "TypeI-Multi-GEXP.csv"), row.names = FALSE)
  write.csv(cbind(noise_levels, mat.multi3), file = file.path(results_dir, "TypeI-Multi-MIXED.csv"), row.names = FALSE)
}

script_dir <- get_script_dir()
experiment_root <- dirname(dirname(script_dir))
source(file.path(script_dir, "Functions.R"))
source(file.path(experiment_root, "config.R"))
source(file.path(experiment_root, "mnist_loader.R"))
config <- experiment_config
results_root <- file.path(experiment_root, "Results")
log_file <- file.path(results_root, "run.log")
dir.create(results_root, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(log_file)) file.create(log_file)
append_log(log_file, "type-I kernel fast run invoked")
append_log(log_file, sprintf("target config: noise_levels=%s; n.rep=%d; resamp=%d; n.iter=%d; n_cores=%d; alpha=%.3f",
  paste(config$noise_levels, collapse = ","), config$n.rep, config$resamp, config$n.iter, config$n_cores, config$alpha))

mnist_data <- load_mnist_train_flat(file.path(experiment_root, "data"))
train.x <- mnist_data$train.x
train.label <- mnist_data$train.label
append_log(log_file, sprintf("loaded MNIST train set: n=%d, p=%d", nrow(train.x), ncol(train.x)))

for (scenario in config$scenarios) {
  results_dir <- file.path(results_root, scenario$name)
  checkpoint_file <- file.path(results_dir, "checkpoint.rds")
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  append_log(log_file, sprintf("scenario %s invoked", scenario$name))

  scenario_start <- Sys.time()
  out.d <- c()
  start_iter <- 1L
  if (file.exists(checkpoint_file)) {
    ckpt <- readRDS(checkpoint_file)
    out.d <- ckpt$out_d
    start_iter <- ckpt$iter_completed + 1L
    append_log(log_file, sprintf("scenario %s resuming from checkpoint: iter_completed=%d; target_n.rep=%d", scenario$name, ckpt$iter_completed, config$n.rep))
  }

  if (start_iter <= config$n.rep) {
    for (iter in start_iter:config$n.rep) {
      iter_start <- Sys.time()
      append_log(log_file, sprintf("scenario %s iter %d/%d started", scenario$name, iter, config$n.rep))
      out.d.iter <- power.d(
        train.x = train.x,
        train.label = train.label,
        resamp = config$resamp,
        error.sigma = config$noise_levels,
        kernel.choice = config$kernel_choice,
        n.iter = config$n.iter,
        set.x = scenario$set_x,
        set.y = scenario$set_y,
        n.cores = config$n_cores,
        seed = scenario$seed + 1000L * iter,
        alpha = config$alpha,
        progress_prefix = sprintf("%s iter %d/%d", scenario$name, iter, config$n.rep),
        progress_every = 50L
      )
      out.d <- c(out.d, out.d.iter)
      saveRDS(list(scenario = scenario$name, iter_completed = iter, out_d = out.d, timestamp = Sys.time(), config = config), checkpoint_file)
      iter_elapsed <- Sys.time() - iter_start
      append_log(log_file, sprintf("scenario %s iter %d/%d finished in %s; checkpoint updated", scenario$name, iter, config$n.rep, format(iter_elapsed)))
      print(c(scenario$name, iter))
    }
  } else {
    append_log(log_file, sprintf("scenario %s needs no new iterations: checkpoint already at iter %d", scenario$name, start_iter - 1L))
  }

  out.d <- as.matrix(as.data.frame(out.d))
  write_kernel_outputs(out.d, config$noise_levels, config$n.rep, results_dir)
  append_log(log_file, sprintf("scenario %s CSV outputs written for n.rep=%d in %s", scenario$name, config$n.rep, format(Sys.time() - scenario_start)))
}
append_log(log_file, "type-I kernel fast run finished")
