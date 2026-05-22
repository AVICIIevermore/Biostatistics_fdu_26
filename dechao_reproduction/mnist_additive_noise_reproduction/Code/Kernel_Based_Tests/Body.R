get_script_dir <- function(){
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(path) > 0) {
    return(dirname(normalizePath(path[1])))
  }
  normalizePath(getwd())
}

append_log <- function(log_file, ..., .timestamp = TRUE){
  prefix <- if (.timestamp) sprintf("[%s] ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")) else ""
  cat(prefix, ..., "\n", sep = "", file = log_file, append = TRUE)
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

dir.create(config$results_dir, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(log_file)) file.create(log_file)
append_log(log_file, "baseline kernel run invoked")
append_log(log_file, sprintf("target config: noise_levels=%s; n.rep=%d; resamp=%d; n.iter=%d; n_cores=%d; alpha=%.3f",
  paste(config$noise_levels, collapse = ","), config$n.rep, config$resamp, config$n.iter, config$n_cores, config$alpha))

mnist_data <- load_mnist_train_flat(file.path(experiment_root, "data"))
train.x <- mnist_data$train.x
train.label <- mnist_data$train.label
append_log(log_file, sprintf("loaded MNIST train set: n=%d, p=%d", nrow(train.x), ncol(train.x)))

start <- Sys.time()
out.d <- c()
start_iter <- 1L
if (file.exists(checkpoint_file)) {
  ckpt <- readRDS(checkpoint_file)
  out.d <- ckpt$out_d
  start_iter <- ckpt$iter_completed + 1L
  append_log(log_file, sprintf("resuming from checkpoint: iter_completed=%d; target_n.rep=%d", ckpt$iter_completed, config$n.rep))
}

if (start_iter <= config$n.rep) {
  for (iter in start_iter:config$n.rep) {
    iter_start <- Sys.time()
    append_log(log_file, sprintf("iter %d/%d started", iter, config$n.rep))
    out.d.iter <- power.d(
      train.x = train.x,
      train.label = train.label,
      resamp = config$resamp,
      error.sigma = config$noise_levels,
      kernel.choice = config$kernel_choice,
      n.iter = config$n.iter,
      set.x = config$set_x,
      set.y = config$set_y,
      n.cores = config$n_cores,
      seed = config$seed + 1000L * iter,
      alpha = config$alpha
    )
    out.d <- c(out.d, out.d.iter)
    saveRDS(list(iter_completed = iter, out_d = out.d, timestamp = Sys.time(), config = config), checkpoint_file)
    iter_elapsed <- Sys.time() - iter_start
    append_log(log_file, sprintf("iter %d/%d finished in %s; checkpoint updated", iter, config$n.rep, format(iter_elapsed)))
    print(iter)
  }
} else {
  append_log(log_file, sprintf("no new iterations needed: checkpoint already at iter %d", start_iter - 1L))
}

out.d <- as.matrix(as.data.frame(out.d))
end <- Sys.time()
append_log(log_file, sprintf("all requested iterations available in %s", format(end - start)))
print(end - start)

single.power.d1 <- 2 + 6 * (0:(config$n.rep - 1))
single.power.d2 <- 3 + 6 * (0:(config$n.rep - 1))
multi.power.d1 <- 4 + 6 * (0:(config$n.rep - 1))
multi.power.d2 <- 5 + 6 * (0:(config$n.rep - 1))
multi.power.d3 <- 6 * (1:config$n.rep)

power.single.mat1 <- matrix(0, nrow = length(config$noise_levels), ncol = config$n.rep)
power.single.mat2 <- matrix(0, nrow = length(config$noise_levels), ncol = config$n.rep)
power.multi.mat1 <- matrix(0, nrow = length(config$noise_levels), ncol = config$n.rep)
power.multi.mat2 <- matrix(0, nrow = length(config$noise_levels), ncol = config$n.rep)
power.multi.mat3 <- matrix(0, nrow = length(config$noise_levels), ncol = config$n.rep)

for (k in seq_along(config$noise_levels)) {
  power.single.mat1[k, ] <- out.d[k, single.power.d1]
  power.single.mat2[k, ] <- out.d[k, single.power.d2]
  power.multi.mat1[k, ] <- out.d[k, multi.power.d1]
  power.multi.mat2[k, ] <- out.d[k, multi.power.d2]
  power.multi.mat3[k, ] <- out.d[k, multi.power.d3]
}

power.single.mat1 <- cbind(config$noise_levels, power.single.mat1)
power.single.mat2 <- cbind(config$noise_levels, power.single.mat2)
power.multi.mat1 <- cbind(config$noise_levels, power.multi.mat1)
power.multi.mat2 <- cbind(config$noise_levels, power.multi.mat2)
power.multi.mat3 <- cbind(config$noise_levels, power.multi.mat3)

write.csv(power.single.mat1, file = file.path(config$results_dir, "SinglePower-LAP.csv"), row.names = FALSE)
write.csv(power.single.mat2, file = file.path(config$results_dir, "SinglePower-GAUSS.csv"), row.names = FALSE)
write.csv(power.multi.mat1, file = file.path(config$results_dir, "MultiPower-LAP.csv"), row.names = FALSE)
write.csv(power.multi.mat2, file = file.path(config$results_dir, "MultiPower-GEXP.csv"), row.names = FALSE)
write.csv(power.multi.mat3, file = file.path(config$results_dir, "MultiPower-MIXED.csv"), row.names = FALSE)
append_log(log_file, "baseline kernel CSV outputs written")
