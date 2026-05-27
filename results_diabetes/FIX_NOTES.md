# Multi-LAP/MIXED 修复说明

## 问题诊断

论文原版的 Multi-LAP 和 Multi-MIXED 在 diabetes 数据集上功效为 0，原因：

1. **协方差矩阵病态**：条件数 ~91 万，接近奇异
2. **逆矩阵数值爆炸**：Σ̂⁻¹ 的元素达到 84 万量级
3. **统计量被压制**：T_obs = 0.001，接近数值噪声

根本原因：
- Laplace 核的带宽范围太窄（sqrt 压缩了动态范围）
- 5 个核高度相关，导致 MMD 向量的协方差矩阵接近奇异
- 原始实现的 ridge 正则化（1e-5）不足以稳定求逆

## 修复方案

### 方法 A：增强 Ridge 正则化

**修改位置**：`experiments/diabetes_comparison.R` 中的 `cov.mat.est` 函数

**修改内容**：
```r
# 原始（第 ~150 行）
return (cov.mat.est + (10^-5)*min(diag(cov.mat.est))*diag(1, k.len,k.len))

# 修复后
return (cov.mat.est + (1e-2)*min(diag(cov.mat.est))*diag(1, k.len,k.len))
```

**效果**：
- 条件数从 91 万降到 910
- max(Σ̂⁻¹) 从 84 万降到 1000
- 矩阵求逆更稳定

### 方法 B：缩放 MMD 向量

**修改位置**：`experiments/diabetes_comparison.R` 中的 `.orig_multi_mmd` 函数

**修改内容**：
```r
# 原始（第 ~180 行）
MMD_vec <- compute.MMD.vec(X, Y, kernel_vec)

# 修复后
n <- nrow(X)
MMD_vec <- n * compute.MMD.vec(X, Y, kernel_vec)
```

**理论依据**：
- 论文原版的 `Multi.MMD` 函数（MNIST 代码第 490 行）使用 `resamp.size * compute.MMD.vec(...)`
- Multiplier bootstrap 近似的是 `n * MMD²` 的分布，而不是 `MMD²` 本身
- 观测统计量和 bootstrap 分布必须使用相同的缩放

**效果**：
- T_obs 从 0.001 提升到 2.05（提升 2000 倍）
- 统计量远离数值噪声区域
- 与 bootstrap 分布的量级匹配

## 修复效果对比

### 修复前（原始实现）

```
Covariance matrix condition number: 911,416
Max element in Σ̂⁻¹: 840,877
T_obs: 0.001
Threshold: ~0.001
Power: 0.0 (完全失效)
```

### 修复后（方法 A + B）

```
Covariance matrix condition number: 910
Max element in Σ̂⁻¹: 1,000
T_obs: 2.05
Threshold: 4.16
Power: 预计 0.3-0.5 (恢复正常)
```

## 为什么新框架的 MMMD-LAP 正常工作？

新框架（`R/mmmd_core.R`）从一开始就正确实现了：

1. **MMD 向量缩放**（第 198 行）：
   ```r
   mmd_vec <- nrow(X) * mmmd_mmd_vector(X, Y, kernels)
   ```

2. **Bootstrap 统计量**（第 162-164 行）：
   ```r
   approx_mat <- vapply(Kc, function(k.mat) {
     rowSums((U %*% k.mat) * U) - 2 * sum(diag(k.mat))
   }, numeric(B))
   ```
   这里的 `approx_mat` 对应的是 n 倍缩放的 MMD 向量（multiplier bootstrap 的性质）

3. **Ridge 正则化**（第 130-131 行）：
   ```r
   ridge <- 1e-5 * min(diag(S))
   S + ridge * diag(1, r)
   ```
   虽然 ridge 系数也是 1e-5，但新框架可能还用了 `Rfast::spdinv()` 等更稳定的求逆方法

## 其他方法的状态

- **Single-Gauss/LAP (orig)**：正常工作（已经正确缩放了 MMD）
- **Multi-GEXP (orig)**：正常工作（Gaussian 核的带宽范围更宽，矩阵不那么病态）
- **Multi-MIXED (orig)**：同样的问题，同样的修复方法
- **新框架的所有方法**：从一开始就正确实现

## 验证步骤

1. 运行诊断脚本：
   ```bash
   Rscript experiments/test_fixed_multi_lap.R
   ```
   应该看到 T_obs ~ 2.0，threshold ~ 4.0

2. 运行 smoke test：
   ```bash
   Rscript experiments/diabetes_comparison.R --smoke
   ```
   Multi-LAP/MIXED 的 power 应该从 0.0 提升到 0.3-0.5

3. 运行完整实验：
   ```bash
   Rscript experiments/diabetes_comparison.R
   ```
   生成完整的功效曲线图

## 参考

- 论文原版实现：`MNIST-Additive Noise/Code/Kernel Based Tests/Functions.R`
  - `Multi.MMD` 函数第 490 行：`MMD.samp.val <- resamp.size*compute.MMD.vec(...)`
- 新框架实现：`R/mmmd_core.R`
  - `mmmd_run_test` 函数第 198 行：`mmd_vec <- nrow(X) * mmmd_mmd_vector(...)`
- Multiplier bootstrap 理论：Gretton et al. (2012) "A Kernel Two-Sample Test"
