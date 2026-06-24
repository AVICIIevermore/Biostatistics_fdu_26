# MMMD 扩展实验说明

本文档说明本仓库新增的 R 扩展层如何运行、每个 `experiments/*.R` 脚本在做什么，以及如何按机器算力自适应调整参数。

原始论文复现代码所在的章节目录基本不需要改动；新增实验主要由下面几部分组成：

```text
R/                 共享函数库：数据生成、MMMD 核心、并行、ROC、图核等
experiments/       每个实验一个 R driver
data/              真实数据或示例数据输入
results/           自动生成的 CSV 和 PNG
results_data/      MNIST-CNN graph-MMMD 补充实验的轻量结果
run_all_tasks.R    一站式 smoke test
run_graph_demo.R   task4_graph_demo.R 的根目录快捷入口
```

## 1. 推荐正常运行参数

如果只是想得到一套可解释、但不至于跑太久的结果，建议从下面这一组开始：

```powershell
Rscript experiments/task1_epsilon_sensitivity.R 30 50 300 0.5
Rscript experiments/task2_variance_sensitivity.R 30 50 300 0.5
Rscript experiments/task3_typeI_and_roc.R 30 100 100 300 0.5
Rscript experiments/task4_graph_demo.R 30 50 300 0.5 "0,0.1,0.25,0.5,0.75,0.9"
Rscript experiments/fig2_mixture_alt.R 30 50 300 100 0.5
```

参数含义：

```text
d          数据维度
N_trials   Monte Carlo 重复次数；越大曲线越稳，越慢
B          multiplier bootstrap 次数；越大阈值估计越稳，越慢
rho        AR(1) 协方差参数，Sigma[i,j] = rho^|i-j|
n          单组样本量，fig2 脚本中第 4 个参数
skew grid  task4 第 5 个参数，直接输入标准化偏斜度网格
```

更快的 smoke test：

```powershell
Rscript run_all_tasks.R
```

更接近正式报告的参数：

```powershell
Rscript experiments/task1_epsilon_sensitivity.R 30 100 500 0.5
Rscript experiments/task2_variance_sensitivity.R 30 100 500 0.5
Rscript experiments/task3_typeI_and_roc.R 30 200 200 500 0.5
Rscript experiments/task4_graph_demo.R 30 100 500 0.5 "0,0.1,0.25,0.5,0.75,0.9"
Rscript experiments/fig2_mixture_alt.R 30 100 500 100 0.5
```

## 2. 一站式跑通与自适应调参

推荐顺序：

1. 先跑 `Rscript run_all_tasks.R`，确认包、路径、并行后端、输出目录都正常。
2. 再用第 1 节“推荐正常运行参数”跑单个脚本。
3. 如果时间太长，优先降低 `N_trials`，其次降低 `B`；不要一开始就改 `d` 和 `n`，否则实验含义会变。
4. 如果曲线抖动明显，提高 `N_trials`；如果 Type-I error 明显不稳定，提高 `B`。
5. 每次运行都会覆盖 `results/` 中同名 CSV/PNG，正式图建议最后统一重跑一次。

粗略调参建议：

| 场景 | 建议 |
|---|---|
| 只检查代码能不能跑 | `N_trials=2~10`, `B=20~100` |
| 课程项目图表 | `N_trials=50~100`, `B=300~500` |
| 更稳健的报告数值 | `N_trials>=200`, `B>=500` |
| 机器较慢 | 固定 `d=30,n=100`，先把 `N_trials` 降到 30 |

## 3. experiments 中每个 R 程序

### `task1_epsilon_sensitivity.R`

用途：探索污染比例 epsilon 的最小可检测边界。

数据设定：

```text
P = N(0, Sigma)
Q = (1 - epsilon) N(0, Sigma) + epsilon t_10(0, Sigma)
Sigma[i,j] = rho^|i-j|
```

运行：

```powershell
Rscript experiments/task1_epsilon_sensitivity.R 30 50 300 0.5
```

参数：

```text
1: d
2: N_trials，每个 epsilon 下重复多少次
3: B，bootstrap 次数
4: rho，AR(1) 相关系数
```

自适应机制：脚本先用 `epsilon = 0,0.025,...,1` 粗扫，找到 power 接近 0.80 的区间后自动加密到 `0.005` 步长。

输出：

```text
results/epsilon_sensitivity_d{d}.csv
results/epsilon_sensitivity_curve_d{d}.png
```

### `task2_variance_sensitivity.R`

用途：测试方差尺度扰动 `k` 多大时能被 MMMD 检出，并与单核 MMD 比较。

数据设定：

```text
P = N(0, Sigma0)
Q = N(0, k Sigma0)
Sigma0[i,j] = rho^|i-j|
```

运行：

```powershell
Rscript experiments/task2_variance_sensitivity.R 30 50 300 0.5
```

比较方法：

```text
Gauss MMMD
LAP MMMD
Mixed MMMD
Gauss MMD
```

输出：

```text
results/variance_sensitivity_d{d}.csv
results/variance_sensitivity_curve_d{d}.png
```

看图方式：红线是 power = 0.80；哪条方法更早达到 0.80，说明对方差扰动更敏感。

### `task3_typeI_and_roc.R`

用途：同时做 Type-I error 校准和 ROC 曲线。

运行：

```powershell
Rscript experiments/task3_typeI_and_roc.R 30 100 100 300 0.5
```

参数：

```text
1: d
2: N_trials_T1，Type-I error 重复次数
3: N_trials_ROC，ROC 中 H0/H1 各自重复次数
4: B
5: rho
```

内容：

```text
Type-I: X,Y iid N(0, Sigma_AR1)，扫 family x r
ROC: H0 为同分布，H1 为 variance-scale，阈值 alpha 从 0 到 1 扫描
```

输出：

```text
results/typeI_baseline.csv
results/roc_data.csv
results/mmmd_roc_curves_multi_r.png
```

自适应说明：ROC 已经采用“一次 bootstrap + 多阈值向量化扫描”，不会为每个 alpha 重新算核矩阵。

### `task4_graph_demo.R`

用途：Graph-MMMD 的 Type-I error 可扩展性检验。

现在主图的四根柱子都表示 Type-I error，不再把 power 和 Type-I 混在一起。每个柱子对应一个 null 分布：从同一个分布中随机抽两组 `X,Y`，在显著性水平 0.05 下重复检验。

四个 null 分布：

```text
1. X,Y iid N(0, Sigma_AR1)
2. X,Y iid N(0, 1.5 Sigma_AR1)
3. X,Y iid SkewNormal(shape alpha = 5)
4. X,Y iid Multivariate Laplace(0, I)
```

运行：

```powershell
Rscript experiments/task4_graph_demo.R 30 50 300 0.5 "0,0.1,0.25,0.5,0.75,0.9"
```

参数：

```text
1: d
2: N_trials
3: B
4: rho
5: 标准化偏斜度网格，直接输入 E[(X-mu)^3]/sd(X)^3
```

Graph-MMMD 机制：

```text
stack(X,Y) -> kNN 图 -> graph Laplacian -> heat kernels K_t
t = {0.1, 0.5, 1, 2, 5}
多个 graph kernels -> Mahalanobis 聚合 -> multiplier bootstrap
```

输出：

```text
results/task4_graph_summary.csv
results/task4_graph_curve.png
results/task4_skew_type1_curve.csv
results/task4_skew_type1_curve.png
```

解释：

```text
task4_graph_curve.png
  比较不同 null 分布下 Type-I error 是否仍接近 0.05

task4_skew_type1_curve.png
  横轴是标准化偏斜度，纵轴是 Type-I error
  用于说明偏态增强时检验是否仍保持校准
```

### `fig2_mixture_alt.R`

用途：复现论文 Fig.2 的混合分布 power 曲线。

数据设定：

```text
P_i ~ (1-p) N(0, Sigma0) + p t_df(0, Sigma0)
Q_i ~ (1-p) N(0, Sigma1) + p t_df(0, Sigma1)
Sigma1 = 1.25 Sigma0
Sigma0[i,j] = rho^|i-j|
```

运行：

```powershell
Rscript experiments/fig2_mixture_alt.R 30 50 300 100 0.5
```

参数：

```text
1: d
2: N_trials
3: B
4: n，单组样本量
5: rho
```

输出：

```text
results/fig2_mixture_alt_d{d}.csv
results/fig2_mixture_alt_d{d}.png
```

### `diabetes_comparison.R`

用途：在 `data/diabetes.csv` 上比较原始方法、新框架 MMMD 和 Graph-MMMD。

运行：

```powershell
Rscript experiments/diabetes_comparison.R --smoke
Rscript experiments/diabetes_comparison.R 200 500
```

参数：

```text
--smoke: 只跑 m=50, N_trials=10, B=50
1: N_trials，正式模式下重复次数
2: B
```

输出：

```text
results_diabetes/power_vs_sample_size.csv
results_diabetes/power_vs_sample_size.png
```

### 诊断和测试脚本

这些脚本不是主实验图，主要用于排查数值问题：

| 脚本 | 作用 |
|---|---|
| `compare_inversion.R` | 比较 `solve()` 和 `Rfast::spdinv()` 对病态协方差矩阵求逆的差异 |
| `diagnose_mmmd_lap.R` | 诊断新框架 LAP-MMMD 的协方差、特征值、bootstrap 分位数 |
| `diagnose_multi_lap.R` | 诊断原始 Multi-LAP 在 diabetes 数据上 power 异常的原因 |
| `test_fixed_multi_lap.R` | 测试加大 ridge 后 Multi-LAP 是否改善 |
| `test_ridge_strength.R` | 扫不同 ridge 强度，观察统计量和阈值稳定性 |

这些脚本通常从项目根目录运行：

```powershell
Rscript experiments/diagnose_mmmd_lap.R
```

## 4. R/ 共享库说明

| 文件 | 作用 |
|---|---|
| `R/load.R` | 统一入口，自动定位项目根目录并加载全部共享模块 |
| `R/parallel_utils.R` | 并行封装：`with_parallel()`、`mmmd_foreach()` |
| `R/data_sources.R` | 合成数据和真实数据抽样工厂 |
| `R/mmmd_core.R` | 多核 MMD、Mahalanobis 聚合、bootstrap、单核 MMD |
| `R/roc_utils.R` | ROC 阈值扫描和 FPR/TPR 汇总 |
| `R/graph_kernel.R` | kNN heat-kernel Graph-MMMD |
| `R/bio_loader.R` | 真实生物数据 CSV/TSV 读取模板 |

## 5. MNIST-CNN Graph-MMMD 补充实验

这部分和 `dechao_reproduction` 的 MNIST-CNN final FC 128-d embedding 对接，但不训练 CNN、不重跑已有 Gaussian5。

需要输入：

```text
results_data/mnist_final_fc128_embeddings.csv
```

CSV 结构：

```text
id,label,split,feat_001,...,feat_128
```

运行：

```powershell
Rscript run_mnist_cnn_graph_mmmd.R --smoke
Rscript run_mnist_cnn_graph_mmmd.R
Rscript plot_graph_mmmd_vs_existing.R
```

输出：

```text
results_data/graph_mmmd_mnist_cnn_summary.csv
results_data/graph_mmmd_vs_existing_mmmd.png
results_data/short_method_note.md
```

## 6. 如何接入新的真实数据

所有检验最终只需要两个数值矩阵：

```text
X: n_x x p
Y: n_y x p
```

如果是一个条件下做 Type-I error：

```r
source("R/load.R")
M <- as.matrix(read.csv("data/my_condition.csv"))
ds <- ds_resample_from_matrix(M)
xy <- ds(100, 100)
mmmd_test(xy$X, xy$Y, family = "GEXP", B = 500)
```

如果是两个条件下做 power：

```r
source("R/load.R")
MX <- as.matrix(read.csv("data/group_x.csv"))
MY <- as.matrix(read.csv("data/group_y.csv"))
ds <- ds_resample_two_matrices(MX, MY)
xy <- ds(100, 100)
mmmd_test(xy$X, xy$Y, family = "GEXP", B = 500)
```

如果要走图核：

```r
graph_mmmd_test(xy$X, xy$Y, B = 500)
```

## 7. 报告里可以怎么表述

可以把本扩展层概括为：

```text
我们没有改动原始章节代码，而是在其上新增了一个参数化、并行化、可复用的 MMMD 实验层。
该层将数据源、核构造、Mahalanobis 聚合、multiplier bootstrap 和图核扩展拆开，
因此同一套引擎既能复现论文中的向量空间实验，也能扩展到偏态分布、重尾分布和 graph-kernel two-sample test。
```

关于 task4：

```text
task4_graph_curve.png 现在只用于比较不同 null 分布下的 Type-I error。
task4_skew_type1_curve.png 进一步把横轴改成标准化偏斜度，用来观察分布偏态增强时检验是否仍保持 0.05 水平附近的校准性。
```

## 8. 演讲稿（个人部分，控制在 3 分钟以内）

> 以下为 Andrew Zheng在 `representation_final.tex` 中 Part 0（MMMD Kernel Extensions）的讲稿，
> 对应幻灯片第 3–11 页。按正常语速约 **2 分 50 秒**，建议配合激光笔指示图上关键区域。

---

**Slide 1 — MMMD Framework（约 30 秒）**

> 我负责的部分是 MMMD 的多核框架扩展和数值实验。
> 首先简单回顾一下 MMMD 的流程：
> 给定两组样本 X 和 Y，我们构造一组多尺度核函数，计算每个核下的 MMD 统计量，
> 得到一个 r 维向量。然后通过 Mahalanobis 距离对核间相关性做去相关，
> 最后用 multiplier bootstrap 做假设检验。
> 我们的框架支持三种核族——GEXP、LAP 和混合核，
> 并且扩展了 Graph-MMMD、偏态分布和并行计算后端。

**Slide 2 — Task 1: ε-Contamination Sensitivity（约 20 秒）**

> 第一个实验是污染敏感性。
> 我们在 Q 分布中混入 t₁₀ 的污染分量，扫描污染比例 ε。
> 左图横轴是 ε，纵轴是 power。
> 可以看到在 ε 约 0.625 时 power 达到 0.80，
> 并且 power 在 ε ∈ [0.45, 0.65] 区间内从 0.57 快速上升到 0.80，过渡非常陡峭。

**Slide 3 — Task 2: Variance-Scale k Sensitivity（约 20 秒）**

> 第二个实验测试对方差尺度变化的检测能力。
> Q 分布的协方差乘以 k，H₀ 对应 k=1。
> 表格对比了三种 MMMD 和单核 Gauss MMD 的最小可检测 k 值。
> 可以看到 Mixed MMMD 在 k=1.25 就达到了 0.80 power，
> 而单核 Gauss MMD 需要 k=1.40——MMMD 提前了约 15%。
> 在 k=1.20 时，MMMD 的 power 是单核方法的 4 倍。

**Slide 4 — Task 3a: Type-I Error Calibration（约 15 秒）**

> 第三个实验先看 Type-I error 的校准。
> 在 H₀ 下检验三种核族、三种 r 值。
> r=3 和 r=5 时所有方法都在 0.02 到 0.055 之间，校准良好。
> 但 r=10 的 GEXP 膨胀到 0.12——说明核太多会恶化协方差估计。
> LAP 核族在所有 r 下都保持稳定。

**Slide 5 — Task 3b: ROC Analysis（约 15 秒）**

> ROC 分析部分，我们用了向量化扫描：
> 一次 bootstrap 就可以算出所有 α 阈值下的决策，
> 不需要为每个 α 重新算核矩阵。
> 右图 ROC 曲线——三条曲线都在对角线上方，
> 且 FPR 都接近名义水平 0.05，说明检验是有效的。

**Slide 6 — Task 4a: Graph-MMMD Extension（约 20 秒）**

> 第四个实验是 Graph-MMMD 扩展。
> 我们把两组样本合并，构建 kNN 图，取图 Laplacian，
> 然后用 heat kernel 生成一系列图核，再接入 MMMD 流程。
> 左图的四根柱子是在四种不同零分布下的 Type-I error——
> 包括正态、方差放大、偏正态和多元 Laplace。
> 所有值都 ≤ 0.053，说明图核方法在不同分布下都保持良好的校准。

**Slide 7 — Task 4b: Skewness Robustness（约 15 秒）**

> 进一步地，我们扫描了偏正态分布从 0 到 0.90 的标准化偏斜度。
> 可以看到即使在极端偏斜度 0.90 下，
> Type-I error 仍然只有 0.020，始终控制在 0.053 以下。
> 这说明 Graph-MMMD 对分布的非对称性非常稳健。

**Slide 8 — Fig. 2 Reproduction: Mixture Distribution Power（约 20 秒）**

> 最后我们复现了论文 Fig.2 的混合分布实验。
> P 和 Q 都是正态-t 混合分布，区别在于 Q 的协方差放大了 1.25 倍。
> 表格对比了不同混合比例 p 下的 power。
> 在 p=0——也就是纯正态时——Mixed MMMD 达到 0.96，
> 而单核 Gauss MMD 只有 0.46，MMMD 的优势是 2 到 5 倍。

**Slide 9 — Summary（约 15 秒）**

> 总结一下：
> MMMD 在污染检测上比单核方法早约 15%，
> 校准良好（只要核数不过多），
> Graph-MMMD 在不同分布和偏斜度下都保持稳健，
> 在混合分布场景下 power 是单核方法的 2–5 倍。
> 整个实验框架是模块化的 R 库，支持并行计算，可以方便地接入新数据和新的核函数。
> 谢谢！

---

**时间分配总览：**

| 幻灯片 | 内容 | 建议时长 |
|---|---|---|
| 1 | MMMD Framework | ~30s |
| 2 | Task 1: ε-Contamination | ~20s |
| 3 | Task 2: Variance k | ~20s |
| 4 | Task 3a: Type-I Error | ~15s |
| 5 | Task 3b: ROC Analysis | ~15s |
| 6 | Task 4a: Graph-MMMD | ~20s |
| 7 | Task 4b: Skewness Robustness | ~15s |
| 8 | Fig.2: Mixture Power | ~20s |
| 9 | Summary | ~15s |
| **合计** | | **~2′50″** |

> 提示：如果时间紧张，Task 3b（ROC）可以一句话带过（"ROC 曲线验证了检验的有效性"），
> 把时间压缩到约 2′35″；如果时间充裕，可以在 Framework 和 Summary 页各多停留 5 秒。
