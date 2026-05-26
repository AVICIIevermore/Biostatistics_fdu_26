FR.dist <- function(z){
  as.matrix(stats::dist(z))
}

count.dups <- function(DF){
  DT <- data.table::data.table(DF)
  DT[, list(N = .N, Index = .I[1L]), by = names(DT)]
}

select_digit_pools <- function(train.x, train.label, set.x, set.y){
  list(
    data.X = train.x[train.label %in% set.x,, drop = FALSE],
    data.Y = train.x[train.label %in% set.y,, drop = FALSE]
  )
}

FR.test <- function(data.X, data.Y, resamp.x, resamp.y,
                    resamp.size, test.type, n.iter = 1000,
                    alpha = 0.05){
  count <- 0

  for (i in seq_len(n.iter)) {
    Z.dups <- count.dups(rbind(
      data.X[resamp.x[i, ],, drop = FALSE],
      data.Y[resamp.y[i, ],, drop = FALSE]
    ))

    Z <- as.matrix(Z.dups[, -c("N", "Index")])
    counts.mat <- cbind(seq_len(nrow(Z.dups)), Z.dups$N)
    X.ID <- which(Z.dups$Index <= resamp.size)
    Y.ID <- which(Z.dups$Index > resamp.size)
    dist.mat <- FR.dist(Z)
    similarity.mat <- gTests::getGraph(counts.mat, dist.mat, 5)
    FRtest <- gTests::g.tests(similarity.mat, X.ID, Y.ID, test.type = test.type)
    count <- count + (FRtest[1][[1]][2] < alpha)
  }

  count / n.iter
}

power.d <- function(train.x, train.label, resamp, error.sigma, n.iter = 500,
                    set.x = c(1, 2, 3), set.y = c(1, 2, 8), n.cores = 1,
                    seed = NULL, alpha = 0.05){
  library(foreach)
  library(doParallel)

  n.cores <- max(1L, as.integer(n.cores))
  cl <- parallel::makeCluster(n.cores)
  doParallel::registerDoParallel(cl)
  on.exit(parallel::stopCluster(cl), add = TRUE)

  out.compare <- foreach::foreach(
    k = seq_along(error.sigma),
    .combine = rbind,
    .packages = c("LaplacesDemon", "Rfast", "SpatialPack", "data.table", "gTests"),
    .export = c("select_digit_pools", "FR.test", "FR.dist", "count.dups")
  ) %dopar% {
    if (!is.null(seed)) {
      set.seed(seed + k - 1L)
    }

    selected <- select_digit_pools(train.x, train.label, set.x, set.y)
    data.X <- selected$data.X
    data.Y <- selected$data.Y

    data.X <- SpatialPack::imnoise(data.X, type = "gaussian", mean = 0, sd = error.sigma[k])
    data.Y <- SpatialPack::imnoise(data.Y, type = "gaussian", mean = 0, sd = error.sigma[k])

    m.x <- nrow(data.X)
    n.y <- nrow(data.Y)
    resamp.x <- matrix(sample.int(m.x, size = n.iter * resamp, replace = TRUE), nrow = n.iter)
    resamp.y <- matrix(sample.int(n.y, size = n.iter * resamp, replace = TRUE), nrow = n.iter)

    c(
      k,
      FR.test(data.X, data.Y, resamp.x, resamp.y, resamp, test.type = "o", n.iter, alpha)
    )
  }

  out.compare <- as.data.frame(out.compare)
  colnames(out.compare) <- c("Set Choice", "FR test")
  out.compare
}
