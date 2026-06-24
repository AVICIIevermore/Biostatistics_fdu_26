# MMMD MNIST 复现加速方案

## 目标

优化当前 `MNIST-Additive Noise` 复现代码的运行速度。

严格要求：

- 不改变实验配置；
- 不改变统计方法；
- 不改变 `alpha`；
- 不改变外层重复次数 `n_rep = 500`；
- 不改变内层 bootstrap 次数 `B_boot = 500`；
- 不改变 resampling with replacement；
- 不改变 multiplier bootstrap cutoff 的定义；
- 不改变 MMD / MMMD 的计算公式；
- 在相同随机输入下，优化前后结果应逐项一致，最多允许浮点误差级别差异。

---

## 1. 拆分 `n.iter`，但数值仍然都是 500

当前代码里 `n.iter` 同时表示两件事：

```r
n_rep = 500      # 外层 Monte Carlo repetition，用于估计 power / Type-I error
B_boot = 500     # 内层 multiplier bootstrap 次数，用于估计 cutoff
```

请不要继续用同一个 `n.iter` 传来传去。改成：

```r
n_rep <- 500
B_boot <- 500
```

这不是减少计算量，而是让代码语义清楚，避免误改实验配置。

---

## 2. 同一次 repetition 内复用 kernel matrices

当前每次外层 repetition 中，kernel matrix 会被重复计算：

- `compute.MMD` 计算一次；
- `est.cov` 再计算一次；
- `single.H0.cutoff` / `multi.H0.cutoff` 再计算一次。

请重构为：

```text
for each repetition:
    抽 X, Y
    根据 X, Y 计算 bandwidths
    对当前 kernel set 一次性计算所有 kernel matrices
    后续 MMD、Sigma_hat、bootstrap cutoff 全部复用这些 matrices
```

不要在 `compute.MMD`、`est.cov`、`multi.H0.cutoff` 里重复调用 `kernlab::kernelMatrix()`。

---

## 3. 单次 repetition 的推荐计算流程

当前一次抽到：

```text
X: n × d
Y: n × d
n = 100
r = kernel 数量
B_boot = 500
```

先一次性得到：

```text
Kxx_list: length r, each n × n
Kyy_list: length r, each n × n
Kxy_list: length r, each n × n
```

然后复用到以下三个步骤。

---

### 3.1 计算 MMD vector

保持当前代码公式，不要擅自改成别的 MMD 版本。

输出：

```text
mmd_vec_scaled = n * mmd_vec
```

single-kernel 时是一个数；multi-kernel 时是 r 维向量。

---

### 3.2 计算 `Sigma_hat`

对每个 kernel 的 X-side Gram matrix：

```r
C <- I - 1/n * 11^T
Kc_a <- C %*% Kxx_a %*% C
```

然后：

```r
Sigma_hat[a,b] <- (8 / n^2) * tr(Kc_a %*% Kc_b)
```

保持现有 `est.cov` 的公式。

加原有 regularization：

```r
Sigma_reg <- Sigma_hat + lambda * I
lambda <- 1e-5 * min(diag(Sigma_hat))
```

如果数值不稳，可以额外记录，但不要默认改变方法。

---

### 3.3 计算 multiplier bootstrap cutoff

对每个 repetition，只生成一次：

```r
U <- MASS::mvrnorm(B_boot, mu = rep(0, n), Sigma = diag(2, n))
```

形状：

```text
U: B_boot × n
```

对每个 kernel 的 centered matrix：

```r
Kboot_a <- (1/n) * C %*% Kxx_a %*% C
```

一次性算出所有 bootstrap MMD 分量：

```text
E: B_boot × r
```

其中：

```r
E[b,a] = U_b^T Kboot_a U_b - 2 * tr(Kboot_a)
```

不要写 500 次 for-loop 来逐个 bootstrap 计算。

可以用矩阵运算：

```r
KU <- Kboot_a %*% t(U)
E_a <- colSums(t(U) * KU) - 2 * tr(Kboot_a)
```

对 r 个 kernel 组成：

```r
E <- cbind(E_1, E_2, ..., E_r)
```

然后：

```r
T_boot <- rowSums((E %*% Sigma_inv) * E)
threshold <- stats::quantile(T_boot, probs = 1 - alpha, names = FALSE)
```

必须保持：

```r
stats::quantile(..., probs = 1-alpha, names = FALSE)
```

的行为。

---

## 4. multi-kernel bootstrap 的矩阵化写法

当前 multi-kernel cutoff 可以优化成：

```r
# Kboot_list: length r
# U: B_boot × n

E <- matrix(0, nrow = B_boot, ncol = r)

for (a in seq_len(r)) {
    Kboot <- Kboot_list[[a]]
    KU <- Kboot %*% t(U)
    E[, a] <- colSums(t(U) * KU) - 2 * psych::tr(Kboot)
}

T_boot <- rowSums((E %*% Sigma_inv) * E)
threshold <- stats::quantile(T_boot, probs = 1 - alpha, names = FALSE)
```

这仍然有一个 kernel-level loop，但没有 bootstrap-level loop。
因为 r 很小，B_boot=500，所以这是合理的优化。

---

## 5. 不改变随机机制：严格对比时预生成随机输入

为了验证优化前后结果一致，建议加入 debug mode。

预先生成并保存：

```text
resamp.x
resamp.y
bootstrap U for each repetition and method
noise realization or seed
```

然后旧代码和新代码都使用同一批随机输入。

需要验证：

```text
MMD value 一致
Sigma_hat 一致
bootstrap threshold 一致
reject / not reject 一致
最终 rejection rate 一致
```

如果不预生成 `U`，只是改变代码结构，R 的随机数调用顺序可能变化，结果不会 bitwise identical，但统计意义应一致。

如果要求“计算结果一模一样”，必须使用同一批 resampling indices 和 same bootstrap multiplier matrices。

---

## 6. 并行方式：从 noise-level 并行改为 repetition-block 并行

当前代码主要按 noise sigma 并行。MNIST 只有 6 个 sigma，所以 100 核利用率很差。

建议并行粒度改成：

```text
(noise sigma, method, repetition block)
```

例如每个 worker 处理一批 repetitions。

但必须保证随机数可复现：

```r
seed_rep <- base_seed + 100000 * sigma_id + 1000 * method_id + rep_id
```

每个 repetition 使用独立 seed。
这样并行顺序改变也不会改变结果。

---

## 7. 设置 BLAS 线程，避免线程嵌套

运行前设置：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

原因：

R 外层已经开 worker，如果每个 worker 内部 BLAS 再开多线程，会造成线程嵌套，100 核反而更慢。

建议先用：

```text
n_cores = 16 或 32
```

不要默认 100 核。

---

## 8. 可选：用 pairwise distance 替代重复 `kernlab::kernelMatrix`

如果要进一步优化，可以先计算距离矩阵：

```text
Dxx = pairwise squared Euclidean distance of X
Dyy = pairwise squared Euclidean distance of Y
Dxy = pairwise squared Euclidean distance between X and Y
```

然后对 Gaussian kernels：

```r
Kxx_a = exp(-sigma_a * Dxx)
Kyy_a = exp(-sigma_a * Dyy)
Kxy_a = exp(-sigma_a * Dxy)
```

但只有在确认它和 `kernlab::rbfdot(sigma=...)` 完全一致后才能替换。

必须加单元测试：

```text
max(abs(K_from_distance - kernlab::kernelMatrix(k, X))) < 1e-12
```

Laplace kernel 同理，必须先确认公式和 `kernlab::laplacedot` 一致。

如果不能确认，就先保留 `kernlab::kernelMatrix`，只做复用，不要重写 kernel 公式。

---

## 9. 必须保留的现有公式

不要“顺手修正”以下内容，否则结果就不再是原论文复现：

1. `compute.MMD` 的公式保持不变；
2. `est.cov` 的 `8 / n^2` 系数保持不变；
3. `single.H0.cutoff` 的公式保持不变：

```r
k.mat <- (1 / m) * C K C
U ~ N(0, 2I)
stat <- U^T k.mat U - 2 tr(k.mat)
```

4. `multi.H0.cutoff` 的公式保持不变：

```r
Kboot_a <- (1 / n) * C K_a C
E_a <- U^T Kboot_a U - 2 tr(Kboot_a)
T_boot <- E^T Sigma_inv E
```

5. `quantile(..., probs = 1-alpha)` 保持不变。

---

## 10. 建议新增 fast 函数，不要直接覆盖旧函数

建议新增：

```r
Single.MMD.fast(...)
Multi.MMD.fast(...)
single.H0.cutoff.fast(...)
multi.H0.cutoff.fast(...)
precompute_kernel_mats(...)
compute_mmd_from_mats(...)
est_cov_from_mats(...)
```

然后写验证脚本：

```r
compare_old_new_one_repetition.R
```

验证一组固定 `X, Y, kernels, U` 下：

```text
old MMD == new MMD
old Sigma_hat == new Sigma_hat
old threshold == new threshold
old reject == new reject
```

通过后再大规模替换。

---

## 11. 加速优先级

按这个顺序做：

1. 拆分 `n_rep` 和 `B_boot`，但数值都设为 500；
2. 预计算并复用 kernel matrices；
3. 向量化 `B_boot` 次 multiplier bootstrap；
4. 用固定随机输入验证 old vs new 一致；
5. 改并行粒度到 repetition block；
6. 设置 BLAS 线程为 1；
7. 可选：用距离矩阵手写 kernel，但必须先通过与 `kernlab` 的一致性测试。

---

## 12. 最终正式实验配置

优化后的正式实验仍然使用：

```text
n_rep = 500
B_boot = 500
alpha = 0.05
sample size per group = 100
noise sigma = {0,0.2,0.4,0.6,0.8,1.0}
```

优化目标：

```text
same statistical method
same random inputs -> same test statistics / thresholds / decisions
faster runtime
```
