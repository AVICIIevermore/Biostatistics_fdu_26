ensure_mnist_file <- function(url, dest){
  if (!file.exists(dest)) {
    dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
    utils::download.file(url, destfile = dest, mode = "wb", quiet = FALSE)
  }
}

read_mnist_images <- function(path){
  con <- gzfile(path, open = "rb")
  on.exit(close(con), add = TRUE)
  invisible(readBin(con, integer(), n = 1, size = 4, endian = "big"))
  n <- readBin(con, integer(), n = 1, size = 4, endian = "big")
  nrow_img <- readBin(con, integer(), n = 1, size = 4, endian = "big")
  ncol_img <- readBin(con, integer(), n = 1, size = 4, endian = "big")
  x <- readBin(con, integer(), n = n * nrow_img * ncol_img, size = 1, signed = FALSE)
  matrix(x, nrow = n, ncol = nrow_img * ncol_img, byrow = TRUE) / 255
}

read_mnist_labels <- function(path){
  con <- gzfile(path, open = "rb")
  on.exit(close(con), add = TRUE)
  invisible(readBin(con, integer(), n = 1, size = 4, endian = "big"))
  n <- readBin(con, integer(), n = 1, size = 4, endian = "big")
  readBin(con, integer(), n = n, size = 1, signed = FALSE)
}

ensure_mnist_data <- function(data_dir){
  base_url <- "https://storage.googleapis.com/cvdf-datasets/mnist"
  files <- c(
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
  )
  for (filename in files) {
    ensure_mnist_file(paste0(base_url, "/", filename), file.path(data_dir, filename))
  }
}

load_mnist_train_flat <- function(data_dir){
  ensure_mnist_data(data_dir)
  list(
    train.x = read_mnist_images(file.path(data_dir, "train-images-idx3-ubyte.gz")),
    train.label = read_mnist_labels(file.path(data_dir, "train-labels-idx1-ubyte.gz"))
  )
}

load_mnist_test_flat <- function(data_dir){
  ensure_mnist_data(data_dir)
  list(
    test.x = read_mnist_images(file.path(data_dir, "t10k-images-idx3-ubyte.gz")),
    test.label = read_mnist_labels(file.path(data_dir, "t10k-labels-idx1-ubyte.gz"))
  )
}
