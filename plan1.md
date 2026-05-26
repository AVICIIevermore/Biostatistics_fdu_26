# AI Agent 自动执行指令：MMMD 算法边界探索与泛化创新实验计划书 (R 语言版)

## 1. 任务背景与目标 (Context & Objective)
当前任务基于 GitHub 仓库 `anirbanc96/MMMD-boost-kernel-two-sample`。该仓库基于 **R 语言** 实现了马氏距离聚合多核最大均值差异（MMMD）的两样本检验框架。
你（AI Agent）的任务是读取该 R 仓库的现有核心代码（特别是涉及数据生成、MMMD 统计量计算、Multiplier Bootstrap 的 R 脚本），并在其基础上完成 4 个进阶实验与框架创新。

---

## 2. 核心代码切入点定位 (Code Analysis & Entry Points)
在克隆 R 仓库后，请首先定位并解析以下核心 R 脚本与函数逻辑（通过模糊匹配或静态分析查找）：
1. **统计量计算模块**：寻找计算多核 MMD 向量及估计其马氏协方差矩阵 $\hat{\Sigma}$ 的 R 函数（通常包含 `MMMD`、`mahalanobis.mmd` 关键字，或使用了 `kernlab` 包计算基础核矩阵）。
2. **重采样模块**：寻找执行高斯乘子自举（Multiplier Bootstrap）获取 Null Distribution 分位数的函数（通常涉及 `rnorm` 生成权重向量，通过矩阵乘法加速）。
3. **实验脚本**：定位论文第 6.3 节对应的模拟脚本（通常在 `experiments/` 或 `simulations/` 目录下，包含 `rnorm` 和 `rt` 分布生成的模拟循环）。

---

## 3. 实验任务详细指令（面向 Agent 自动化编写）

### 任务 1：自动滑动 $\epsilon$ 探索正态与 $t_{10}$ 混淆的最小检出边界
* **输入参数**：
    * 固定显著性水平 `alpha <- 0.05`，样本量 `m <- n <- 500`，维度 `d` 为 2 或 10。
    * 自举重采样次数 `B <- 1000`。
* **Agent 逻辑实现步骤**：
    1.  修改现有的 $H_1$ 混合数据生成逻辑：使用 `matrix(rnorm(...))` 生成 $P = \mathcal{N}(0, I_d)$；利用 `matrix(rt(...))` 生成 $t_{10}$ 分布，按比例组合成 $Q = (1-\epsilon)\mathcal{N}(0, I_d) + \epsilon \cdot t_{10}$。
    2.  编写外层循环，令 $\epsilon$ 在 `seq(0.0, 1.0, by = 0.05)` 区间内滑动。
    3.  对每个 $\epsilon$ 运行检验流程 `N_trials <- 100`（或 500）次，计算拒绝率（Power）。
    4.  **智能收敛条件**：如果发现 Power 在某个区间急剧上升，请在 R 中自动调用 `seq()` 或二分查找将局部步长加密至 `0.01`。
* **定义输出**：当 Power 首次 $\ge 0.80$ 时的最小 $\epsilon$ 值为 `epsilon_min`。将数据输出为 CSV 并自动调用 `ggplot2` 绘制 $\epsilon \text{ vs. Power}$ 折线图。

### 任务 2：固定混合比例，自动滑动 $k$ 测试方差敏感性边界
* **输入参数**：
    * 固定混合比例 `epsilon <- 0.5`。
    * 备择假设 $Q$ 内部的分量协方差矩阵调整为 $k \cdot I_d$（在 R 中通过乘以系数 `sqrt(k)` 控制标准差实现）。
* **Agent 逻辑实现步骤**：
    1.  定位原 R 代码中 $k=1.25$ 的硬编码位置，将其抽象为可传参变量 `k`。
    2.  编写外层循环，令 `k` 在 `seq(1.0, 2.0, by = 0.05)` 递增滑动。
    3.  在每个 `k` 值下，运行双样本检验并统计拒绝 $H_0$ 的比例。
* **定义输出**：找出使 MMMD 检验功效达到 $\ge 0.80$ 的最小 $k$ 值，记为 `k_min`。对比退化的单核 MMD，量化多核聚合框架对尺度扰动的灵敏度提升幅度。

### 任务 3：补充 Type-I 错误实测与自动化 ROC 曲线 Trade-off 分析
* **Agent 逻辑实现步骤**：
    1.  **第一类错误（Type-I Error）自动化基准测试**：
        * **合成数据**：令 $P$ 和 $Q$ 均自相同正态分布生成。
        * **MNIST 数据**：修改原 R 评估脚本（若涉及数据导入，检查是否使用 `readBin` 或 `mnist` 现成包），确保两组样本均从同一种数字类别（如 `label == 3`）中无放回随机抽样划分为 $\mathbb{X}_m$ 和 $\mathbb{Y}_n$。
        * 在 `alpha <- 0.05` 下运行 `N_trials <- 1000` 次，记录实际拒绝率。如果实际值 $> 0.08$ 或 $< 0.02$，触发警告并检查自举分位数计算。
    2.  **两类错误 ROC 曲线自动生成**：
        * 在备择假设 $H_1$ 下，保持数据不变。通过调整自举检验的拒绝阈值分位数（名义 `alpha <- seq(0.0, 1.0, by = 0.01)`）。
        * 收集每一个阈值下的实际一类错误率（FPR）和实际功效（TPR）。
    3.  **多核数量 $r$ 的调控超参实验**：
        * 改变 R 脚本中的带宽列表长度（例如修改包含多个带宽的 `sigma_list` 或 `bandwidths` 向量，分别试验 $r=3, 5, 10$）。
        * 为不同的 $r$ 分别绘制 ROC 曲线，定量分析核过多时对一类错误膨胀和二类错误压缩的影响。

### 任务 4：框架理论泛化与图结构（Graph Two-Sample Test）扩展创新
* **说明**：这是本项目的核心架构创新。MMMD 的数学本质是通过马氏距离去相关聚合核矩阵向量。这在 R 的多核框架下天然支持推广到非欧氏空间。
* **Agent 拓展步骤**：
    1.  **引入非高斯/偏态分布数据源**：
        * 利用 R 的 `sn` 包或 `LaplacesDemon` 包，编写多元拉普拉斯分布（Multivariate Laplace）或多元偏正态分布（Skew-Normal）的数据生成函数。
        * 测试算法对偏态非对称差异的捕获能力。
    2.  **实现图结构双样本检验（Graph MMD）扩展**：
        * 在项目中新建 `graph_kernel_mmmd.R` 脚本。
        * **定义图核函数**：引入 R 的 `igraph` 包或 `GraphKernel` 包，实现图指数核（Graph Exponential Kernel）或随机游走核（Random Walk Kernel）。
        * **多核构建**：通过改变图核的超参数（如滑动扩散时间 $t$），构建一个包含多个图核矩阵的 `list` 集合：`K_list <- list(K_t1, K_t2, ..., K_tr)`。
        * **矩阵对接**：将生成的图核矩阵 `list` 组合成一个三维的 R 数组（`array`），直接无缝喂入原有的马氏协方差计算与 Multiplier Bootstrap 模块。

---

## 4. R 语言规范下的性能加速与执行要求 (R Performance Guardrails)

鉴于原框架主要基于 R 语言构建，为防止因高维高频循环导致算力阻塞，AI Agent 必须强制遵守以下 **R 性能优化规范**：

1.  **强制并行化改造 (Mandatory Parallelization)**:
    - 严禁使用原生的单线程 `for` 循环处理外层的参数滑动（$\epsilon$ 和 $k$ 的滑动）及多次重复实验。
    - 必须在脚本顶部引入 `library(parallel)`、`library(foreach)` 及 `library(doParallel)`。
    - 自动检测本地 CPU 核心数 `cores <- parallel::detectCores() - 1`，通过 `registerDoParallel(cores)` 注册后端，并将循环改写为 `foreach(i = 1:N_trials) %dopar% { ... }` 结构。

2.  **R 向量化与矩阵缓存机制 (Vectorization & Storage)**:
    - 严禁在改变名义 $\alpha$ 计算 ROC 曲线时重复生成数据和重计算核矩阵。
    - Agent 必须采用“单次自举计算 + 向量化阈值扫描”的逻辑：利用 R 的 `quantile()` 函数一次性批量计算多个 `alpha` 水平下的分位数阈值，再用矩阵掩码（Matrix Masking）进行统计。

3.  **底层 C++ 加速通道 (Rcpp Optional Acceleration)**:
    - 如果 Agent 在 Profiling 中发现内层的 Multiplier Bootstrap 矩阵相乘和求和是严重瓶颈，允许引入 `library(Rcpp)` 或 `library(RcppArmadillo)`。
    - 将自举高斯权重乘法逻辑外包给 10-20 行的 C++ 函数，以获得 50 倍以上的编译级提速。

4.  **粗细结合扫描机制 (Pre-scanning)**:
    - 针对任务 1 和任务 2，先用低精度（如 `B = 200`, `N_trials = 20`）进行全范围粗筛，快速定位 Power 从低于 0.2 飙升到 0.8 以上的敏感区间。
    - 仅在敏感区间内开启高精度（`B = 1000`, `N_trials = 200`）并行细扫，从而在整体上节约 70% 以上的 R 运行时长。

## 5. Agent 最终产出物 (Outputs)
- 一个自动生成的 `results/` 文件夹。
- 由 R 的 `ggplot2` 自动渲染并保存的三张高清图表：`epsilon_sensitivity_curve.png` (任务1), `variance_sensitivity_curve.png` (任务2), `mmmd_roc_curves_multi_r.png` (任务3)。
- 包含图结构扩展的演示 R 脚本 `run_graph_demo.R` (任务4)。