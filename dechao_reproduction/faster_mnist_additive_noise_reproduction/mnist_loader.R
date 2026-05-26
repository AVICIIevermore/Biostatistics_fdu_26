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

load_mnist_train_flat <- function(data_dir){
  base_url <- "https://storage.googleapis.com/cvdf-datasets/mnist"
  image_path <- file.path(data_dir, "train-images-idx3-ubyte.gz")
  label_path <- file.path(data_dir, "train-labels-idx1-ubyte.gz")
  ensure_mnist_file(paste0(base_url, "/train-images-idx3-ubyte.gz"), image_path)
  ensure_mnist_file(paste0(base_url, "/train-labels-idx1-ubyte.gz"), label_path)

  list(
    train.x = read_mnist_images(image_path),
    train.label = read_mnist_labels(label_path)
  )
}
