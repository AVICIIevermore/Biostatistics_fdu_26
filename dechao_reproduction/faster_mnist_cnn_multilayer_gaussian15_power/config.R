experiment_config <- list(
  experiment_name = "faster_mnist_cnn_multilayer_gaussian15_power",
  noise_levels = seq(0, 1, length.out = 6),
  n.rep = 10,
  resamp = 100,
  n.iter = 500,
  n_cores = 8L,
  alpha = 0.05,
  seed = 20260525L,
  set_x = c(1, 2, 3),
  set_y = c(1, 2, 8),
  results_dir = file.path(normalizePath(".", winslash = "/", mustWork = FALSE), "Results"),
  checkpoint_path = "/home/dechao/kernel_two-sample/dechao_reproduction/faster_mnist_cnn_shared_model/models/mnist_cnn_checkpoint.pt",
  train_batch_size = 128L,
  train_epochs = 30L,
  train_lr = 1e-3,
  val_fraction = 5000 / 60000,
  eval_batch_size = 1024L,
  target_val_accuracy = 0.98,
  python_executable = "/home/dechao/.conda/envs/cv-hw2/bin/python",
  python_module_dir = "/home/dechao/kernel_two-sample/dechao_reproduction/faster_mnist_cnn_shared_model/python",
  shared_data_dir = "/home/dechao/kernel_two-sample/dechao_reproduction/faster_mnist_cnn_shared_model/data",
  shared_model_dir = "/home/dechao/kernel_two-sample/dechao_reproduction/faster_mnist_cnn_shared_model/models",
  method_order = c(
    "multilayer_gaussian15"
  )
)
